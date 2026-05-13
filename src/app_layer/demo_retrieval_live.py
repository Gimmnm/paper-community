"""
Online retrieval helpers for demo-api: vector NN, community bundle, live metrics, six-way CSV.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from analysis_layer.retrieval_benchmark import (
    _community_retrieval_pids,
    _csr_from_assets,
    _hits_to_pids,
    _load_community_top1_map,
    _mean_cosine_to_seed,
    _mean_shortest_hops,
    _mean_tfidf_query_doc_scores,
    _paper_query_text,
    _vector_top_k,
    topic_match_rate,
)
from app_layer.demo_search import (
    DemoAssets,
    SearchHit,
    _paper_abs,
    _paper_author_payload,
    _paper_title,
    _paper_year,
    _result,
    build_demo_assets_and_graph,
    resolve_topic_weights_csv_for_web,
    search_keyword,
)
from data_layer.experiment_contracts import ExperimentRunManifest
from foundation_layer.diagram2d import l2_normalize_rows
from foundation_layer.project_paths import REPO_ROOT, embedding_path_specter2, out_dir


def load_emb_norm_for_papers(*, emb_path: Path, n_papers_expect: int) -> np.ndarray:
    emb = np.load(emb_path).astype(np.float32, copy=False)
    n_emb = int(emb.shape[0])
    if n_emb == int(n_papers_expect) + 1:
        emb = emb[1:]
    elif n_emb != int(n_papers_expect):
        raise ValueError(f"embedding rows {n_emb} != n_papers {n_papers_expect} (or +1 dummy row)")
    return l2_normalize_rows(emb)


def _allowed_idx0_from_domain(base_dir: Path, domain_run_id: str, domain: int) -> Optional[Set[int]]:
    p = Path(base_dir) / "out" / str(domain_run_id) / f"domain_{int(domain)}_vertex_indices.npy"
    if not p.is_file():
        return None
    verts = np.load(p).astype(np.int64)
    verts = np.unique(verts)
    return {int(i) for i in verts.tolist() if int(i) >= 0}


def search_vector_nn(
    *,
    papers: Sequence[Any],
    graph: Any,
    emb_norm: np.ndarray,
    pid: int,
    top_k: int = 20,
    authors: Optional[Sequence[Any]] = None,
    domain_run_id: Optional[str] = None,
    domain: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    pid = int(pid)
    n_expect = int(len(papers)) - 1
    seed_idx0 = pid - 1
    if seed_idx0 < 0 or seed_idx0 >= n_expect:
        return _result("vector_nn", {"pid": pid, "top_k": int(top_k)}, [], debug={"error": "paper_id_out_of_range"})

    allowed_idx0: Optional[Set[int]] = None
    if domain_run_id and domain is not None and base_dir is not None:
        allowed_idx0 = _allowed_idx0_from_domain(Path(base_dir), str(domain_run_id), int(domain))

    t0 = time.perf_counter()
    v = emb_norm[int(seed_idx0)]
    sims = emb_norm @ v
    sims[int(seed_idx0)] = -1e9
    for j in range(int(sims.shape[0])):
        if allowed_idx0 is not None and j not in allowed_idx0:
            sims[j] = -1e9
    k = int(top_k)
    hits: List[SearchHit] = []
    if k <= 0:
        order_list: List[int] = []
    elif sims.size <= k:
        order_list = np.argsort(-sims).tolist()[:k]
    else:
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        order_list = idx[np.argsort(-sims[idx])].tolist()
    order = order_list
    for i0 in order:
        if len(hits) >= k:
            break
        s = float(sims[int(i0)])
        if s <= -1e8:
            continue
        qpid = int(i0) + 1
        qp = papers[qpid]
        cid = graph.paper_to_community.get(qpid)
        pl = {
            "pid": qpid,
            "title": _paper_title(qp),
            "year": _paper_year(qp),
            "community": None if cid is None else int(cid),
            "cosine_to_seed": s,
        }
        pl.update(_paper_author_payload(authors, qp, limit=8))
        hits.append(SearchHit(kind="paper", id=str(qpid), score=float(s), payload=pl))
    dt = time.perf_counter() - t0
    dbg: Dict[str, Any] = {"time_sec": float(dt), "domain_run_id": domain_run_id, "domain": domain}
    return _result("vector_nn", {"pid": pid, "top_k": int(top_k)}, hits, debug=dbg)


def search_community_bundle(
    *,
    base_dir: Path,
    manifest: ExperimentRunManifest,
    papers: Sequence[Any],
    authors: Optional[Sequence[Any]],
    pid: int,
    resolution: float,
    keyword_index_dir: Path,
    top_k: int = 20,
    k_neighbors: int = 50,
    k_neighbors_in_comm: int = 30,
    k_extra_from_comm: int = 200,
) -> Dict[str, Any]:
    pid = int(pid)
    topic_csv_path = resolve_topic_weights_csv_for_web(
        base_dir=Path(base_dir),
        run_id=str(manifest.run_id),
        resolution=float(resolution),
        manifest_topic_csv=str(manifest.topic_communities_csv).strip() if manifest.topic_communities_csv else None,
        env_topic_csv=None,
        manifest_tags=manifest.tags if isinstance(manifest.tags, dict) else None,
    )
    assets, graph = build_demo_assets_and_graph(
        base_dir=Path(base_dir),
        papers=papers,
        leiden_dir=Path(manifest.leiden_dir),
        resolution=float(resolution),
        graph_npz=Path(manifest.graph_npz),
        keyword_index_dir=Path(keyword_index_dir),
        coords_2d_path=Path(manifest.coords_2d_path) if manifest.coords_2d_path else None,
        topic_communities_csv=topic_csv_path,
    )
    t0 = time.perf_counter()
    ordered, _dt_inner = _community_retrieval_pids(
        assets=assets,
        papers=papers,
        graph=graph,
        pid=pid,
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        k_extra_from_comm=int(k_extra_from_comm),
        top_k=int(top_k),
    )
    dt = time.perf_counter() - t0
    hits = []
    for qpid in ordered:
        qp = papers[int(qpid)]
        cid = graph.paper_to_community.get(int(qpid))
        pl = {
            "pid": int(qpid),
            "title": _paper_title(qp),
            "year": _paper_year(qp),
            "community": None if cid is None else int(cid),
        }
        pl.update(_paper_author_payload(authors, qp, limit=8))
        hits.append(SearchHit(kind="paper", id=str(qpid), score=1.0, payload=pl))
    return _result(
        "community_bundle",
        {
            "pid": pid,
            "partition_run_id": str(manifest.run_id),
            "top_k": int(top_k),
            "resolution": float(assets.resolution),
        },
        hits,
        debug={"time_sec": float(dt), "partition": str(manifest.run_id)},
    )


def _pack_row(
    *,
    method: str,
    pids: Sequence[int],
    timing: float,
    emb_norm: np.ndarray,
    csr: csr_matrix,
    seed_idx0: int,
    assets: DemoAssets,
    qtext: str,
    graph: Any,
    top1_map: Mapping[int, int],
    seed_pid: int,
    extras: Optional[Dict[str, Any]] = None,
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
    mk = _mean_tfidf_query_doc_scores(assets, qtext, pids)
    if mk is not None:
        row["mean_keyword_tfidf"] = float(mk)
    if extras:
        row["extra"] = extras
    return row


def run_live_retrieval_for_seed(
    *,
    base_dir: Path,
    partition_manifest: ExperimentRunManifest,
    papers: Sequence[Any],
    authors: Optional[Sequence[Any]],
    emb_norm: np.ndarray,
    seed_pid: int,
    method: str,
    resolution: float,
    keyword_index_dir: Path,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    ``method`` ∈ {keyword, vector_nn, community_bundle} using ``partition_manifest`` graph
    (matches offline retrieval benchmark row layout).
    """
    seed_pid = int(seed_pid)
    n_expect = len(papers) - 1
    seed_idx0 = seed_pid - 1
    if seed_idx0 < 0 or seed_idx0 >= n_expect:
        return {"error": "paper_id_out_of_range", "seed_pid": seed_pid}

    topic_csv_path = resolve_topic_weights_csv_for_web(
        base_dir=Path(base_dir),
        run_id=str(partition_manifest.run_id),
        resolution=float(resolution),
        manifest_topic_csv=str(partition_manifest.topic_communities_csv).strip()
        if partition_manifest.topic_communities_csv
        else None,
        env_topic_csv=None,
        manifest_tags=partition_manifest.tags if isinstance(partition_manifest.tags, dict) else None,
    )
    assets, graph = build_demo_assets_and_graph(
        base_dir=Path(base_dir),
        papers=papers,
        leiden_dir=Path(partition_manifest.leiden_dir),
        resolution=float(resolution),
        graph_npz=Path(partition_manifest.graph_npz),
        keyword_index_dir=Path(keyword_index_dir),
        coords_2d_path=Path(partition_manifest.coords_2d_path) if partition_manifest.coords_2d_path else None,
        topic_communities_csv=topic_csv_path,
    )
    r_eff = float(assets.resolution)
    csr = _csr_from_assets(assets)
    qtext = _paper_query_text(papers, seed_pid)
    top1_map = _load_community_top1_map(topic_csv_path) if topic_csv_path else {}

    method_l = str(method).strip().lower()
    if method_l == "keyword":
        t0 = time.perf_counter()
        kw_res = search_keyword(
            assets=assets,
            papers=papers,
            graph=graph,
            query=qtext,
            top_k=int(top_k),
            offset=0,
            authors=authors,
        )
        dt = time.perf_counter() - t0
        kw_pids = _hits_to_pids(kw_res)
        pack = _pack_row(
            method="keyword",
            pids=kw_pids,
            timing=dt,
            emb_norm=emb_norm,
            csr=csr,
            seed_idx0=seed_idx0,
            assets=assets,
            qtext=qtext,
            graph=graph,
            top1_map=top1_map,
            seed_pid=seed_pid,
            extras={"search_debug": dict(kw_res.get("debug") or {})},
        )
        return {"method": method_l, "summary": pack, "hits": kw_res.get("hits") or [], "partition_run_id": str(partition_manifest.run_id)}

    if method_l == "vector_nn":
        t0 = time.perf_counter()
        vec_pids, vec_scores, vec_dt = _vector_top_k(emb_norm, seed_idx0, top_k=int(top_k), exclude={seed_pid})
        pack = _pack_row(
            method="vector_nn",
            pids=vec_pids,
            timing=vec_dt,
            emb_norm=emb_norm,
            csr=csr,
            seed_idx0=seed_idx0,
            assets=assets,
            qtext=qtext,
            graph=graph,
            top1_map=top1_map,
            seed_pid=seed_pid,
            extras={"similarities": vec_scores},
        )
        hits = []
        for qpid, sc in zip(vec_pids, vec_scores):
            qp = papers[int(qpid)]
            cid = graph.paper_to_community.get(int(qpid))
            pl = {
                "pid": int(qpid),
                "title": _paper_title(qp),
                "year": _paper_year(qp),
                "abstract": _paper_abs(qp)[:800],
                "community": None if cid is None else int(cid),
                "cosine_to_seed": float(sc),
            }
            pl.update(_paper_author_payload(authors, qp, limit=10))
            hits.append({"kind": "paper", "id": str(qpid), "score": float(sc), "payload": pl})
        return {"method": method_l, "summary": pack, "hits": hits, "partition_run_id": str(partition_manifest.run_id)}

    if method_l == "community_bundle":
        t0 = time.perf_counter()
        comm_pids, comm_dt = _community_retrieval_pids(
            assets=assets,
            papers=papers,
            graph=graph,
            pid=seed_pid,
            k_neighbors=50,
            k_neighbors_in_comm=30,
            k_extra_from_comm=200,
            top_k=int(top_k),
        )
        dt = time.perf_counter() - t0
        pack = _pack_row(
            method="community_bundle",
            pids=comm_pids,
            timing=dt,
            emb_norm=emb_norm,
            csr=csr,
            seed_idx0=seed_idx0,
            assets=assets,
            qtext=qtext,
            graph=graph,
            top1_map=top1_map,
            seed_pid=seed_pid,
        )
        hits = []
        for qpid in comm_pids:
            qp = papers[int(qpid)]
            cid = graph.paper_to_community.get(int(qpid))
            pl = {
                "pid": int(qpid),
                "title": _paper_title(qp),
                "year": _paper_year(qp),
                "abstract": _paper_abs(qp)[:800],
                "community": None if cid is None else int(cid),
            }
            pl.update(_paper_author_payload(authors, qp, limit=10))
            hits.append({"kind": "paper", "id": str(qpid), "score": 1.0, "payload": pl})
        return {"method": method_l, "summary": pack, "hits": hits, "partition_run_id": str(partition_manifest.run_id)}

    return {"error": "unknown_method", "method": method}


def read_retrieval_sixway_rows(repo_root: Optional[Path] = None, *, comparison_run_tag: str = "master_breakpoints") -> Dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    csv_path = out_dir(root) / "experiment_eval" / "comparison_retrieval_sixway_long.csv"
    if not csv_path.is_file():
        return {"error": "file_missing", "path": str(csv_path), "rows": []}
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("comparison_run_tag", "")).strip() != str(comparison_run_tag).strip():
                continue
            rows.append(dict(row))
    metrics = [
        ("mean_cosine_mean", False),
        ("mean_keyword_tfidf_mean", False),
        ("mean_hops_mean", True),
        ("time_sec_mean", True),
        ("topic_top1_match_mean", False),
    ]
    averages: Dict[str, float] = {}
    for key, _ in metrics:
        vals = []
        for r in rows:
            try:
                v = float(r.get(key, "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        averages[key] = float(np.mean(vals)) if vals else float("nan")

    return {
        "comparison_run_tag": comparison_run_tag,
        "csv_path": str(csv_path),
        "rows": rows,
        "averages": averages,
        "metric_specs": [{"key": k, "lower_is_better": lb} for k, lb in metrics],
    }


def live_vs_sixway_positions(
    *,
    sixway_rows: Sequence[Mapping[str, Any]],
    live_summary: Mapping[str, Any],
    method_row_method: str,
) -> Dict[str, Any]:
    """
    Map one live ``summary`` (single seed) onto the five offline aggregate metrics by
    comparing the live scalar to the **six** per-method offline means (rank / z vs mean).
    ``method_row_method`` is the ``method`` column in six-way CSV, e.g. ``vector_nn``.
    """
    row_by_method: Dict[str, Mapping[str, Any]] = {}
    for r in sixway_rows:
        m = str(r.get("method", "")).strip()
        if m:
            row_by_method[m] = r

    pairs = [
        ("mean_cosine_to_seed", "mean_cosine_mean", False),
        ("mean_keyword_tfidf", "mean_keyword_tfidf_mean", False),
        ("mean_shortest_path_hops", "mean_hops_mean", True),
        ("time_sec", "time_sec_mean", True),
        ("topic_top1_match_rate_vs_seed_community", "topic_top1_match_mean", False),
    ]
    out: Dict[str, Any] = {}
    for live_key, csv_key, lower_better in pairs:
        try:
            cur = float(live_summary.get(live_key, "nan"))
        except (TypeError, ValueError):
            cur = float("nan")
        refs: List[float] = []
        for r in sixway_rows:
            try:
                v = float(r.get(csv_key, "nan"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                refs.append(v)
        if not refs or not np.isfinite(cur):
            out[live_key] = {
                "offline_csv_key": csv_key,
                "live": cur,
                "mean_of_six": None,
                "std_of_six": None,
                "z_vs_mean_of_six": None,
                "beat_count_vs_six_offline": None,
                "tie_count": None,
                "beat_fraction_vs_six_offline": None,
                "count": len(refs),
            }
            continue
        arr = np.array(refs, dtype=np.float64)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=0)) if arr.size > 1 else 0.0
        z = float((cur - mu) / sd) if sd > 1e-12 else 0.0
        if lower_better:
            beat = int(np.sum(cur < arr))
            tie = int(np.sum(np.isclose(cur, arr)))
        else:
            beat = int(np.sum(cur > arr))
            tie = int(np.sum(np.isclose(cur, arr)))
        beat_frac = float(beat / max(1, arr.size))
        out[live_key] = {
            "offline_csv_key": csv_key,
            "live": cur,
            "mean_of_six": mu,
            "std_of_six": sd,
            "z_vs_mean_of_six": z,
            "beat_count_vs_six_offline": beat,
            "tie_count": tie,
            "beat_fraction_vs_six_offline": beat_frac,
            "count": len(refs),
            "higher_is_better": not lower_better,
        }
    out["method_row"] = method_row_method
    out["offline_row"] = dict(row_by_method.get(method_row_method, {}))
    return out
