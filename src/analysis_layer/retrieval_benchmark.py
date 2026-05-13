"""
Offline retrieval comparison: keyword TF-IDF vs embedding kNN vs community-neighborhood bundle.

Writes JSONL under ``out/comparison_runs/<run_tag>/metrics.jsonl`` (one object per seed × resolution × catalog run).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from app_layer.demo_search import (
    DemoAssets,
    build_demo_assets_and_graph,
    load_models_from_project,
    lookup_paper,
    search_keyword,
)
from data_layer.breakpoint_schedule import (
    default_breakpoints_csv,
    default_topic_k_dir_for_run,
    load_breakpoint_resolutions_for_run,
    resolve_topic_communities_csv,
)
from data_layer.experiment_contracts import ExperimentRunManifest
from data_layer.experiment_fallback_cli import register_experiment_catalog_fallback_args
from data_layer.experiment_registry import available_resolutions_for_manifest, build_run_catalog, default_paths_manifest
from foundation_layer.diagram2d import l2_normalize_rows
from foundation_layer.project_paths import REPO_ROOT, out_dir


def _default_topic_root_for_manifest(base_dir: Path, manifest: ExperimentRunManifest, *, k_topics: int = 10) -> Optional[Path]:
    """``out/topic_runs/<sweep>/K{k}`` when it exists (matches topic batch layout)."""
    tags = manifest.tags if isinstance(manifest.tags, dict) else None
    return default_topic_k_dir_for_run(
        Path(base_dir),
        str(manifest.run_id),
        manifest_tags=tags,
        k_topics_default=int(k_topics),
    )


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return float(len(a & b) / u) if u else 0.0


def _paper_query_text(papers: Sequence[Any], pid: int, *, max_abs_chars: int = 800) -> str:
    p = papers[int(pid)]
    title = str(getattr(p, "name", "") or "").strip()
    abstract = str(getattr(p, "abstract", "") or "").strip()
    chunk = abstract[: int(max_abs_chars)] if max_abs_chars > 0 else ""
    return (title + "\n" + chunk).strip()


def _hits_to_pids(result: Mapping[str, Any]) -> List[int]:
    out: List[int] = []
    for h in result.get("hits") or []:
        pl = h.get("payload") or {}
        pid = pl.get("pid")
        if pid is not None:
            out.append(int(pid))
    return out


def _mean_tfidf_query_doc_scores(assets: DemoAssets, qtext: str, pids: Sequence[int]) -> Optional[float]:
    """
    Mean TF-IDF dot product ``x_doc^T q_vec`` for returned papers (same scoring axis as ``search_keyword``).

    Used to report **keyword-style relevance** of the top-k set for any retrieval channel (keyword / vector / bundle).
    Returns ``None`` if the keyword index is missing or the query is empty.
    """
    if not pids:
        return None
    q = str(qtext or "").strip()
    if not q:
        return None
    try:
        vectorizer, X = assets.load_keyword_index()
        qv = vectorizer.transform([q])
        sims = (X @ qv.T).toarray().ravel().astype(np.float64)
    except Exception:
        return None
    n = int(sims.size)
    vals: List[float] = []
    for pid in pids:
        i0 = int(pid) - 1
        if i0 < 0 or i0 >= n:
            continue
        vals.append(max(0.0, float(sims[i0])))
    if not vals:
        return None
    return float(sum(vals) / len(vals))
def _community_retrieval_pids(
    *,
    assets: DemoAssets,
    papers: Sequence[Any],
    graph: Any,
    pid: int,
    k_neighbors: int,
    k_neighbors_in_comm: int,
    k_extra_from_comm: int,
    top_k: int,
) -> Tuple[List[int], float]:
    t0 = time.perf_counter()
    lp = lookup_paper(
        assets=assets,
        papers=papers,
        graph=graph,
        pid=int(pid),
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        k_neighbor_comms=8,
        authors=None,
    )
    seen: Set[int] = set()
    ordered: List[int] = []

    def _take(rows: Any, key: str = "pid") -> None:
        nonlocal ordered
        if not rows:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            qpid = row.get(key)
            if qpid is None:
                continue
            qi = int(qpid)
            if qi == int(pid):
                continue
            if qi not in seen:
                seen.add(qi)
                ordered.append(qi)
                if len(ordered) >= int(top_k):
                    return

    if lp.get("hits"):
        pay = lp["hits"][0].get("payload") or {}
        _take(pay.get("neighbors"))
        _take(pay.get("neighbors_in_community"))
        cs = pay.get("community_summary") or {}
        _take(cs.get("center_papers"))
        _take(cs.get("bridge_papers"))
        # fill from raw community membership head if still short
        cid = pay.get("community")
        if cid is not None and len(ordered) < int(top_k):
            comm = graph.communities.get(int(cid))
            if comm is not None:
                head = list(comm.paper_ids)[: max(1, int(k_extra_from_comm))]
                for qpid in head:
                    if int(qpid) == int(pid):
                        continue
                    if int(qpid) not in seen:
                        seen.add(int(qpid))
                        ordered.append(int(qpid))
                        if len(ordered) >= int(top_k):
                            break
    dt = time.perf_counter() - t0
    return ordered[: int(top_k)], float(dt)


def _vector_top_k(
    emb_norm: np.ndarray,
    seed_idx0: int,
    *,
    top_k: int,
    exclude: Set[int],
) -> Tuple[List[int], List[float], float]:
    t0 = time.perf_counter()
    v = emb_norm[int(seed_idx0)]
    sims = emb_norm @ v
    sims[int(seed_idx0)] = -1e9
    # mask excluded in pid space -> idx0
    for pid in exclude:
        i0 = int(pid) - 1
        if 0 <= i0 < sims.shape[0]:
            sims[i0] = -1e9
    k = int(top_k)
    if k <= 0:
        return [], [], time.perf_counter() - t0
    if sims.size <= k:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    pids = [int(i) + 1 for i in idx.tolist() if sims[int(i)] > -1e8][:k]
    scores = [float(sims[int(p) - 1]) for p in pids]
    return pids, scores, time.perf_counter() - t0


def _csr_from_assets(assets: DemoAssets) -> csr_matrix:
    indptr, indices, data = assets.ensure_csr()
    n = int(indptr.shape[0] - 1)
    return csr_matrix((data, indices, indptr), shape=(n, n))


def _mean_shortest_hops(
    csr: csr_matrix,
    seed_idx0: int,
    pids: Sequence[int],
) -> Tuple[Optional[float], int]:
    dist_row = shortest_path(csr, directed=False, indices=int(seed_idx0), unweighted=True)
    vals: List[float] = []
    unreachable = 0
    for pid in pids:
        i0 = int(pid) - 1
        if i0 < 0 or i0 >= dist_row.shape[0]:
            unreachable += 1
            continue
        d = float(dist_row[i0])
        if not np.isfinite(d):
            unreachable += 1
            continue
        vals.append(d)
    if not vals:
        return None, unreachable
    return float(np.mean(vals)), unreachable


def _mean_cosine_to_seed(emb_norm: np.ndarray, seed_idx0: int, pids: Sequence[int]) -> Optional[float]:
    if not pids:
        return None
    v = emb_norm[int(seed_idx0)]
    acc = 0.0
    n = 0
    for pid in pids:
        i0 = int(pid) - 1
        if i0 < 0 or i0 >= emb_norm.shape[0]:
            continue
        acc += float(np.dot(v, emb_norm[i0]))
        n += 1
    return float(acc / n) if n else None


def _load_community_top1_map(csv_path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(float(row.get("community_id", "nan")))
                t1 = row.get("top1_topic")
                if t1 is None or str(t1).strip() == "":
                    continue
                out[cid] = int(float(t1))
            except Exception:
                continue
    return out


def topic_match_rate(
    *,
    graph: Any,
    top1_map: Mapping[int, int],
    seed_pid: int,
    pids: Sequence[int],
) -> Optional[float]:
    if not top1_map:
        return None
    seed_cid = graph.paper_to_community.get(int(seed_pid))
    if seed_cid is None:
        return None
    ref = top1_map.get(int(seed_cid))
    if ref is None:
        return None
    hits = 0
    tot = 0
    for pid in pids:
        cid = graph.paper_to_community.get(int(pid))
        if cid is None:
            continue
        t1 = top1_map.get(int(cid))
        if t1 is None:
            continue
        tot += 1
        if int(t1) == int(ref):
            hits += 1
    return float(hits / tot) if tot else None


def run_benchmark_row(
    *,
    base_dir: Path,
    manifest: ExperimentRunManifest,
    resolution: float,
    seed_pid: int,
    emb_path: Path,
    keyword_index_dir: Path,
    topic_root: Optional[Path],
    top_k: int,
    k_neighbors: int,
    k_neighbors_in_comm: int,
    k_extra_from_comm: int,
) -> Dict[str, Any]:
    authors, papers = load_models_from_project(base_dir, exclude_selfcite=False, force_reingest=False)
    n_expect = len(papers) - 1
    emb = np.load(emb_path).astype(np.float32, copy=False)
    n_emb = int(emb.shape[0])
    # Align with 1-based paper ids: some pipelines save a leading dummy row at index 0.
    if n_emb == int(n_expect) + 1:
        emb = emb[1:]
    elif n_emb != int(n_expect):
        raise ValueError(f"embedding rows {n_emb} != n_papers {n_expect} (or {n_expect + 1} with dummy row)")
    emb_norm = l2_normalize_rows(emb)

    assets, graph = build_demo_assets_and_graph(
        base_dir=base_dir,
        papers=papers,
        leiden_dir=Path(manifest.leiden_dir),
        resolution=float(resolution),
        graph_npz=Path(manifest.graph_npz),
        keyword_index_dir=Path(keyword_index_dir),
        coords_2d_path=Path(manifest.coords_2d_path) if manifest.coords_2d_path else None,
        topic_communities_csv=Path(manifest.topic_communities_csv) if manifest.topic_communities_csv else None,
    )
    r_eff = float(assets.resolution)
    seed_idx0 = int(seed_pid) - 1
    if seed_idx0 < 0 or seed_idx0 >= n_expect:
        raise ValueError(f"seed_pid out of range: {seed_pid}")

    csr = _csr_from_assets(assets)
    qtext = _paper_query_text(papers, int(seed_pid))

    topic_csv = resolve_topic_communities_csv(topic_root, r_eff)
    top1_map = _load_community_top1_map(topic_csv) if topic_csv else {}

    # keyword
    kw_debug: Dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        kw_res = search_keyword(
            assets=assets,
            papers=papers,
            graph=graph,
            query=qtext,
            top_k=int(top_k),
            offset=0,
            authors=authors,
        )
        kw_dt = time.perf_counter() - t0
        kw_pids = _hits_to_pids(kw_res)
        kw_debug = dict(kw_res.get("debug") or {})
    except Exception as e:
        kw_res = {"hits": []}
        kw_pids = []
        kw_dt = time.perf_counter() - t0
        kw_debug = {"error": f"{type(e).__name__}: {e}"}

    vec_pids, vec_scores, vec_dt = _vector_top_k(
        emb_norm,
        seed_idx0,
        top_k=int(top_k),
        exclude={int(seed_pid)},
    )

    comm_pids, comm_dt = _community_retrieval_pids(
        assets=assets,
        papers=papers,
        graph=graph,
        pid=int(seed_pid),
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        k_extra_from_comm=int(k_extra_from_comm),
        top_k=int(top_k),
    )

    sets = {
        "keyword": set(kw_pids),
        "vector_nn": set(vec_pids),
        "community_bundle": set(comm_pids),
    }
    pairwise = {
        "keyword__vector_nn": _jaccard(sets["keyword"], sets["vector_nn"]),
        "keyword__community_bundle": _jaccard(sets["keyword"], sets["community_bundle"]),
        "vector_nn__community_bundle": _jaccard(sets["vector_nn"], sets["community_bundle"]),
    }

    mk_kw = _mean_tfidf_query_doc_scores(assets, qtext, kw_pids)
    mk_vec = _mean_tfidf_query_doc_scores(assets, qtext, vec_pids)
    mk_comm = _mean_tfidf_query_doc_scores(assets, qtext, comm_pids)

    def _pack(
        method: str,
        pids: Sequence[int],
        timing: float,
        extra: Optional[Dict[str, Any]] = None,
        *,
        mean_keyword_tfidf: Optional[float] = None,
    ) -> Dict[str, Any]:
        mean_hops, n_inf = _mean_shortest_hops(csr, seed_idx0, pids)
        row: Dict[str, Any] = {
            "method": method,
            "pids": [int(x) for x in pids],
            "time_sec": float(timing),
            "n_results": len(pids),
            "mean_cosine_to_seed": _mean_cosine_to_seed(emb_norm, seed_idx0, pids),
            "mean_shortest_path_hops": mean_hops,
            "n_unreachable_in_graph": int(n_inf),
            "topic_top1_match_rate_vs_seed_community": topic_match_rate(
                graph=graph, top1_map=top1_map, seed_pid=int(seed_pid), pids=pids
            ),
        }
        if mean_keyword_tfidf is not None:
            row["mean_keyword_tfidf"] = float(mean_keyword_tfidf)
        if extra:
            row["extra"] = extra
        return row

    return {
        "run_id": str(manifest.run_id),
        "algorithm": str(manifest.algorithm),
        "time_window": str(manifest.time_window),
        "leiden_dir": str(Path(manifest.leiden_dir)),
        "graph_npz": str(Path(manifest.graph_npz)),
        "resolution_requested": float(resolution),
        "resolution_effective": float(r_eff),
        "seed_pid": int(seed_pid),
        "keyword_query_chars": int(len(qtext)),
        "topic_communities_csv": None if topic_csv is None else str(topic_csv.resolve()),
        "methods": {
            "keyword": _pack("keyword", kw_pids, kw_dt, {"search_debug": kw_debug}, mean_keyword_tfidf=mk_kw),
            "vector_nn": _pack(
                "vector_nn", vec_pids, vec_dt, {"similarities": vec_scores}, mean_keyword_tfidf=mk_vec
            ),
            "community_bundle": _pack("community_bundle", comm_pids, comm_dt, mean_keyword_tfidf=mk_comm),
        },
        "pairwise_jaccard": pairwise,
    }


def iter_manifests_for_cli(
    *,
    base_dir: Path,
    manifest_paths: Optional[Sequence[Path]],
    run_ids: Optional[Sequence[str]],
    single_leiden: Optional[Path],
    single_graph: Optional[Path],
    fallback: ExperimentRunManifest,
) -> List[ExperimentRunManifest]:
    if manifest_paths:
        out: List[ExperimentRunManifest] = []
        for mp in manifest_paths:
            d = json.loads(Path(mp).read_text(encoding="utf-8"))
            out.append(ExperimentRunManifest.from_json_dict(d))
        return out
    if single_leiden is not None:
        m = default_paths_manifest(
            run_id="cli_single",
            base_dir=base_dir,
            leiden_dir=Path(single_leiden),
            graph_npz=Path(single_graph or fallback.graph_npz),
            keyword_index_dir=Path(fallback.keyword_index_dir) if fallback.keyword_index_dir else None,
            coords_2d_path=Path(fallback.coords_2d_path) if fallback.coords_2d_path else None,
            algorithm=str(fallback.algorithm),
            time_window=str(fallback.time_window),
            default_resolution=float(fallback.default_resolution),
        )
        return [m]
    cat = build_run_catalog(base_dir=base_dir, fallback=fallback)
    if run_ids:
        return [cat[r] for r in run_ids if r in cat]
    return list(cat.values())


def select_resolutions(manifest: ExperimentRunManifest, args: argparse.Namespace) -> List[float]:
    if args.resolutions:
        return [float(x) for x in args.resolutions]
    src = str(getattr(args, "resolution_source", "breakpoints"))
    if src == "breakpoints":
        csv_path = Path(args.breakpoints_csv)
        if csv_path.is_file():
            rs_bp = load_breakpoint_resolutions_for_run(
                csv_path,
                run_id=str(manifest.run_id),
                time_window=str(getattr(args, "breakpoint_time_window", "all")),
            )
            if rs_bp:
                return rs_bp
        print(
            "[retrieval-benchmark] warn: breakpoints CSV missing or empty for "
            f"run_id={manifest.run_id}; falling back to summary grid"
        )
    rs = available_resolutions_for_manifest(manifest)
    rmin = float(args.r_min)
    rmax = float(args.r_max)
    rs = [r for r in rs if rmin <= r <= rmax]
    step = int(args.resolution_stride)
    if step > 1:
        rs = rs[::step]
    cap = int(args.max_resolutions)
    if cap > 0 and len(rs) > cap:
        rs = rs[:cap]
    return rs


def register_retrieval_benchmark_args(
    p: argparse.ArgumentParser,
    *,
    base_dir: Optional[Path] = None,
    default_out_dir: Optional[Path] = None,
) -> None:
    root = REPO_ROOT if base_dir is None else Path(base_dir)
    od = default_out_dir if default_out_dir is not None else out_dir(root)
    p.add_argument("--run-tag", type=str, default="default", help="output folder name under out/comparison_runs/")
    p.add_argument("--emb-path", type=str, default=str(root / "data" / "paper_embeddings_specter2.npy"))
    p.add_argument("--keyword-index-dir", type=str, default=str(od / "keyword_index"))
    p.add_argument(
        "--topic-root",
        type=str,
        default=None,
        help=(
            "e.g. out/topic_runs/leiden_sweep_cpm/K10 for topic_top1_match_rate. "
            "If omitted, uses out/topic_runs/<sweep>/K10 per manifest run_id when that directory exists."
        ),
    )
    p.add_argument("--manifest", type=str, nargs="*", default=None, help="explicit manifest.json paths")
    p.add_argument("--run-id", type=str, nargs="*", default=None, help="catalog run_id filter")
    p.add_argument("--leiden-dir", type=str, default=None, help="single-run mode: one sweep directory")
    p.add_argument("--graph-npz", type=str, default=None, help="single-run mode graph path")
    register_experiment_catalog_fallback_args(p, out_dir=od)

    p.add_argument(
        "--resolution-source",
        type=str,
        choices=["breakpoints", "summary"],
        default="breakpoints",
        help="breakpoints: use comparison_breakpoints.csv per manifest run_id (~10 r each); summary: slice summary.npy",
    )
    p.add_argument(
        "--breakpoints-csv",
        type=str,
        default=str(default_breakpoints_csv(root)),
        help="experiment-comparison-breakpoints output; used when --resolution-source breakpoints",
    )
    p.add_argument(
        "--breakpoint-time-window",
        type=str,
        default="all",
        help="filter CSV rows (manifest time_window is usually all)",
    )

    p.add_argument("--seed-pid", type=int, nargs="*", default=None)
    p.add_argument("--n-random-seeds", type=int, default=0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k-neighbors", type=int, default=30)
    p.add_argument("--k-neighbors-in-comm", type=int, default=15)
    p.add_argument("--k-extra-from-comm", type=int, default=40)
    p.add_argument("--resolutions", type=float, nargs="*", default=None)
    p.add_argument("--r-min", type=float, default=0.001)
    p.add_argument("--r-max", type=float, default=2.0)
    p.add_argument("--resolution-stride", type=int, default=1, help="take every Nth resolution from summary order")
    p.add_argument("--max-resolutions", type=int, default=0, help="0 = no cap")
    p.add_argument("--append", action="store_true", help="append to metrics.jsonl instead of truncate")


def run_retrieval_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    base_dir = REPO_ROOT

    fallback = default_paths_manifest(
        run_id="cli_default",
        base_dir=base_dir,
        leiden_dir=Path(args.fallback_leiden_dir),
        graph_npz=Path(args.fallback_graph_npz),
        keyword_index_dir=Path(args.fallback_keyword_index_dir),
        coords_2d_path=Path(args.fallback_coords_2d_path),
        algorithm=str(args.fallback_algorithm),
        time_window=str(args.fallback_time_window),
        default_resolution=float(args.fallback_resolution),
    )
    manifests = iter_manifests_for_cli(
        base_dir=base_dir,
        manifest_paths=[Path(x) for x in args.manifest] if args.manifest else None,
        run_ids=list(args.run_id) if args.run_id else None,
        single_leiden=Path(args.leiden_dir) if args.leiden_dir else None,
        single_graph=Path(args.graph_npz) if args.graph_npz else None,
        fallback=fallback,
    )
    if not manifests:
        raise SystemExit("no manifests to run")

    seeds: List[int] = []
    if args.seed_pid:
        seeds.extend(int(x) for x in args.seed_pid)
    if int(args.n_random_seeds) > 0:
        rng = random.Random(int(args.random_seed))
        # sample from valid pid range after loading once
        _authors, papers = load_models_from_project(base_dir, exclude_selfcite=False, force_reingest=False)
        n = len(papers) - 1
        pool = list(range(1, n + 1))
        rng.shuffle(pool)
        take = min(int(args.n_random_seeds), len(pool))
        seeds.extend(pool[:take])
    if not seeds:
        raise SystemExit("pass --seed-pid and/or --n-random-seeds")
    seeds = list(dict.fromkeys(int(x) for x in seeds))

    out_dir = base_dir / "out" / "comparison_runs" / str(args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.jsonl"
    mode = "a" if args.append else "w"
    topic_root_cli = Path(args.topic_root).resolve() if args.topic_root else None
    kw_dir = Path(args.keyword_index_dir)

    emb_path = Path(args.emb_path)
    with out_path.open(mode, encoding="utf-8") as sink:
        for m in manifests:
            topic_root = topic_root_cli or _default_topic_root_for_manifest(base_dir, m)
            resolutions = select_resolutions(m, args)
            for r in resolutions:
                for seed_pid in seeds:
                    row = run_benchmark_row(
                        base_dir=base_dir,
                        manifest=m,
                        resolution=float(r),
                        seed_pid=int(seed_pid),
                        emb_path=emb_path,
                        keyword_index_dir=kw_dir,
                        topic_root=topic_root,
                        top_k=int(args.top_k),
                        k_neighbors=int(args.k_neighbors),
                        k_neighbors_in_comm=int(args.k_neighbors_in_comm),
                        k_extra_from_comm=int(args.k_extra_from_comm),
                    )
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[retrieval-benchmark] wrote {out_path}")
    return {"metrics_jsonl": str(out_path.resolve()), "n_manifests": len(manifests), "n_seed_draws": len(seeds)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline retrieval benchmark (keyword vs embedding vs community bundle).")
    register_retrieval_benchmark_args(p, base_dir=REPO_ROOT, default_out_dir=out_dir(REPO_ROOT))
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_retrieval_benchmark(args)


if __name__ == "__main__":
    main()
