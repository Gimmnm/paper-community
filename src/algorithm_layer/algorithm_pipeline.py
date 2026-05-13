from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithm_layer.community import (
    DEFAULT_RESOLUTION_NDIGITS,
    induced_subgraph_edge_list,
    make_resolutions,
    membership_filename,
    summarize_sweep_results,
)
from algorithm_layer.time_window import window_indices_from_years
from foundation_layer.network import build_or_load_mutual_knn_graph, save_edges_npz

try:
    import igraph as ig
except Exception:  # pragma: no cover
    ig = None

try:
    import leidenalg
except Exception:  # pragma: no cover
    leidenalg = None


def build_igraph_from_edges(n_nodes: int, u: np.ndarray, v: np.ndarray, w: np.ndarray):
    if ig is None:  # pragma: no cover
        raise ImportError("python-igraph is required for community detection")
    g = ig.Graph(n=int(n_nodes), edges=list(zip(np.asarray(u).tolist(), np.asarray(v).tolist())), directed=False)
    g.es["weight"] = np.asarray(w, dtype=np.float32).astype(float).tolist()
    return g


def _n_comm_from_membership(mem: np.ndarray) -> int:
    mem = np.asarray(mem, dtype=np.int32)
    if mem.size == 0:
        return 0
    if bool(np.any(mem < 0)):
        vv = mem[mem >= 0]
        return int(np.unique(vv).size) if vv.size else 0
    return int(np.unique(mem).size)


_RESOLUTION_PARTITION_TYPES = frozenset(
    {
        "RBConfigurationVertexPartition",
        "CPMVertexPartition",
        "RBERVertexPartition",
    }
)


def _run_leiden(
    g,
    *,
    resolution: float,
    seed: int = 42,
    partition_type: str = "RBConfigurationVertexPartition",
    weights_attr: str = "weight",
) -> Tuple[np.ndarray, float, float]:
    if leidenalg is None:  # pragma: no cover
        raise ImportError("leidenalg is required for Leiden/CPM runs")
    if partition_type == "RBConfigurationVertexPartition":
        cls = leidenalg.RBConfigurationVertexPartition
    elif partition_type == "CPMVertexPartition":
        cls = leidenalg.CPMVertexPartition
    elif partition_type == "RBERVertexPartition":
        cls = leidenalg.RBERVertexPartition
    elif partition_type == "ModularityVertexPartition":
        cls = leidenalg.ModularityVertexPartition
    else:
        raise ValueError(
            "partition_type must be RBConfigurationVertexPartition, CPMVertexPartition, "
            "RBERVertexPartition, or ModularityVertexPartition"
        )
    t0 = time.time()
    fp_kw: Dict[str, Any] = {"seed": int(seed)}
    if weights_attr in g.es.attributes():
        fp_kw["weights"] = g.es[weights_attr]
    if partition_type in _RESOLUTION_PARTITION_TYPES:
        fp_kw["resolution_parameter"] = float(resolution)
    part = leidenalg.find_partition(
        g,
        cls,
        **fp_kw,
    )
    membership = np.asarray(part.membership, dtype=np.int32)
    return membership, float(part.quality()), float(time.time() - t0)


def run_community_sweep(
    *,
    g,
    out_dir: Path,
    algorithm: str,
    r_min: float,
    r_max: float,
    step: float,
    include: Optional[List[float]] = None,
    seed: int = 42,
    weights_attr: str = "weight",
    partition_type: Optional[str] = None,
    resolution_mode: str = "linear",
    reuse_existing: bool = True,
    save_each_membership: bool = True,
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """
    Standardized sweep entry for multiple algorithms (all support a resolution grid).

    - ``leiden``: RBConfigurationVertexPartition (RB-style resolution γ).
    - ``leiden_cpm``: CPMVertexPartition (CPM γ).
    - ``louvain``: multiresolution objective via ``RBERVertexPartition`` (Reichardt–Bornholdt Potts / ER null;
      supports ``resolution_parameter``). Older ``leidenalg`` builds used ``ModularityVertexPartition``, which
      **cannot** take a resolution grid in current releases (single-resolution modularity only).
      Override with ``--partition-type`` only if you know the class exists in ``leidenalg``.
    """
    algo = str(algorithm).lower().strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include = include or []
    results: Dict[float, Dict[str, Any]] = {}

    if algo == "louvain":
        ptype = partition_type or "RBERVertexPartition"
    elif algo == "leiden_cpm":
        ptype = partition_type or "CPMVertexPartition"
    else:
        ptype = partition_type or "RBConfigurationVertexPartition"

    resolutions = make_resolutions(
        float(r_min),
        float(r_max),
        float(step),
        include=include,
        ndigits=DEFAULT_RESOLUTION_NDIGITS,
        mode=str(resolution_mode),
    )
    for r in resolutions:
        rr = round(float(r), DEFAULT_RESOLUTION_NDIGITS)
        p = out_dir / membership_filename(rr, ndigits=DEFAULT_RESOLUTION_NDIGITS)
        if reuse_existing and p.exists():
            t0 = time.perf_counter()
            mem = np.load(p).astype(np.int32)
            q = float("nan")
            elapsed = max(float(time.perf_counter() - t0), 1e-9)
        else:
            mem, q, elapsed = _run_leiden(
                g,
                resolution=rr,
                seed=seed,
                partition_type=str(ptype),
                weights_attr=weights_attr,
            )
            if save_each_membership:
                np.save(p, mem)
        results[rr] = {
            "resolution": float(rr),
            "membership": mem,
            "n_comm": _n_comm_from_membership(mem),
            "quality": float(q),
            "time": float(elapsed),
        }
        if verbose:
            print(f"[pipeline] {algo} r={rr:.4f} n_comm={results[rr]['n_comm']} time={results[rr]['time']:.2f}s")

    summary = summarize_sweep_results(results)
    summary["algorithm"] = str(algo)
    np.save(out_dir / "summary.npy", summary, allow_pickle=True)
    return results


def _expand_membership_file(path: Path, window_idx0: np.ndarray, n_full: int) -> None:
    path = Path(path)
    loc = np.load(path).astype(np.int32)
    if int(loc.shape[0]) == int(n_full):
        return
    if int(loc.shape[0]) != int(window_idx0.shape[0]):
        raise ValueError(
            f"{path.name}: expected membership length {window_idx0.shape[0]} (window) or {n_full} (global), got {loc.shape[0]}"
        )
    full = np.full(int(n_full), -1, dtype=np.int32)
    full[window_idx0] = loc
    np.save(path, full)


def run_community_sweep_time_window(
    *,
    papers: Sequence[Optional[object]],
    embs: np.ndarray,
    years_by_idx0: np.ndarray,
    start_year: int,
    end_year: int,
    out_dir: Path,
    algorithm: str,
    r_min: float,
    r_max: float,
    step: float,
    include: Optional[List[float]] = None,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    normalize: bool = True,
    seed: int = 42,
    partition_type: Optional[str] = None,
    resolution_mode: str = "linear",
    include_unknown_year: bool = False,
    reuse_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calendar window → papers subset → mutual-kNN **refit** on their embeddings → Leiden/Louvain sweep.

    Writes the same artifacts as ``run_community_sweep``, plus:
      - ``window_vertex_indices.npy`` (global idx0 included in the window)
      - ``mutual_knn_k{k}.npz`` edge list using **global** idx0 (``n_nodes`` = full corpus size)
      - ``window_sweep_meta.json`` describing the window + refit graph

    Membership arrays are expanded to full-corpus length with ``-1`` outside the window so the web demo
    can reuse ``build_demo_assets_and_graph`` with this ``leiden_dir`` and graph npz.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_full = int(len(papers) - 1)
    if n_full <= 0:
        raise ValueError("empty papers")

    years_by_idx0 = np.asarray(years_by_idx0, dtype=np.int32)
    if years_by_idx0.shape[0] != n_full:
        raise ValueError(f"years_by_idx0 length {years_by_idx0.shape[0]} != n_full {n_full}")

    window_idx0 = window_indices_from_years(
        years_by_idx0,
        int(start_year),
        int(end_year),
        include_unknown=bool(include_unknown_year),
    )
    if window_idx0.size == 0:
        raise ValueError(f"empty year window: [{int(start_year)}, {int(end_year)}]")

    X_full = np.asarray(embs[1:], dtype=np.float32)
    X_window = X_full[window_idx0]

    local_cache = out_dir / f"_window_local_mutual_knn_k{int(k)}.npz"
    _, (loc_u, loc_v, loc_w) = build_or_load_mutual_knn_graph(
        X_window,
        k=int(k),
        cache_npz=local_cache,
        knn_backend=str(knn_backend),  # type: ignore[arg-type]
        knn_batch_size=int(knn_batch_size),
        normalize=bool(normalize),
        verbose=bool(verbose),
    )

    g_u = window_idx0[np.asarray(loc_u, dtype=np.int64)]
    g_v = window_idx0[np.asarray(loc_v, dtype=np.int64)]
    graph_npz = out_dir / f"mutual_knn_k{int(k)}.npz"
    save_edges_npz(
        graph_npz,
        u=g_u.astype(np.int32),
        v=g_v.astype(np.int32),
        w=np.asarray(loc_w, dtype=np.float32),
        n_nodes=int(n_full),
        k=int(k),
        normalized=bool(normalize),
        note=f"window refit mutual-kNN [{int(start_year)},{int(end_year)}] mapped to global idx0",
    )

    g = build_igraph_from_edges(int(window_idx0.size), loc_u, loc_v, loc_w)
    sweep_res = run_community_sweep(
        g=g,
        out_dir=out_dir,
        algorithm=str(algorithm),
        r_min=float(r_min),
        r_max=float(r_max),
        step=float(step),
        include=include,
        seed=int(seed),
        partition_type=partition_type,
        resolution_mode=str(resolution_mode),
        reuse_existing=bool(reuse_existing),
        verbose=bool(verbose),
    )

    for p in sorted(out_dir.glob("membership_r*.npy")):
        _expand_membership_file(p, window_idx0, n_full)

    np.save(out_dir / "window_vertex_indices.npy", window_idx0.astype(np.int64))

    meta = {
        "kind": "window_refit_sweep",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "include_unknown_year": bool(include_unknown_year),
        "n_full": int(n_full),
        "n_window": int(window_idx0.size),
        "algorithm": str(algorithm),
        "k": int(k),
        "knn_backend": str(knn_backend),
        "graph_npz": str(graph_npz.name),
        "partition_type": partition_type,
    }
    (out_dir / "window_sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out_dir.resolve()),
        "window_idx0_path": str((out_dir / "window_vertex_indices.npy").resolve()),
        "graph_npz": str(graph_npz.resolve()),
        "n_window": int(window_idx0.size),
        "n_full": int(n_full),
        "sweep": sweep_res,
    }


def _resolutions_from_domain_sweep_dir(domain_dir: Path) -> List[float]:
    sp = Path(domain_dir) / "summary.npy"
    if not sp.exists():
        return []
    d = np.load(sp, allow_pickle=True).item()
    return [float(x) for x in np.asarray(d.get("resolutions", []), dtype=np.float64).tolist()]


def run_coarse_kmeans_then_community_sweep(
    *,
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    domain_vertex_indices: Sequence[np.ndarray],
    out_dir: Path,
    algorithm: str,
    r_min: float,
    r_max: float,
    step: float,
    include: Optional[List[float]] = None,
    seed: int = 42,
    partition_type: Optional[str] = None,
    resolution_mode: str = "linear",
    reuse_existing: bool = True,
    verbose: bool = True,
    kmeans_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    K-means coarse domains (caller supplies ``domain_vertex_indices`` per domain) → per-domain
    **induced subgraph** on the global mutual-kNN edge list → same ``run_community_sweep`` as whole-graph.

    Writes merged ``membership_r*.npy`` (length ``n_nodes``, global idx0) and ``summary.npy`` under ``out_dir``.
    Per-domain sweeps are kept under ``out_dir / "_domain_{i}_sweep"`` for debugging.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    algo = str(algorithm).lower().strip()
    domains: List[np.ndarray] = [np.asarray(x, dtype=np.int64).reshape(-1) for x in domain_vertex_indices]
    domains = [d for d in domains if d.size > 0]
    if not domains:
        raise ValueError("coarse_kmeans sweep: no non-empty domain vertex sets")

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    n_nodes = int(n_nodes)

    domain_dirs: List[Path] = []
    aligned_domains: List[np.ndarray] = []
    for i, verts in enumerate(domains):
        sub = out_dir / f"_domain_{i}_sweep"
        sub.mkdir(parents=True, exist_ok=True)
        ul, vl, wl, global_sorted = induced_subgraph_edge_list(n_nodes, u, v, w, verts)
        aligned_domains.append(np.asarray(global_sorted, dtype=np.int64).reshape(-1))
        if ul.size == 0 and verbose:
            print(f"[coarse_kmeans] domain {i}: induced subgraph has no edges (n={global_sorted.size})")
        g = build_igraph_from_edges(int(global_sorted.size), ul, vl, wl)
        run_community_sweep(
            g=g,
            out_dir=sub,
            algorithm=algo,
            r_min=float(r_min),
            r_max=float(r_max),
            step=float(step),
            include=include,
            seed=int(seed),
            partition_type=partition_type,
            resolution_mode=str(resolution_mode),
            reuse_existing=bool(reuse_existing),
            verbose=bool(verbose),
        )
        domain_dirs.append(sub)

    res_lists = [_resolutions_from_domain_sweep_dir(d) for d in domain_dirs]
    common = sorted(set(res_lists[0]).intersection(*(set(x) for x in res_lists[1:])))
    if not common:
        raise RuntimeError(
            "coarse_kmeans sweep: empty resolution intersection across domains — "
            f"per-domain resolutions: {res_lists[:3]}{'...' if len(res_lists) > 3 else ''}"
        )

    merged_results: Dict[float, Dict[str, Any]] = {}
    for rr in common:
        global_mem = np.zeros(int(n_nodes), dtype=np.int32)
        offset = 0
        total_time = 0.0
        total_quality = 0.0
        q_count = 0
        for di, verts in enumerate(aligned_domains):
            dom_dir = domain_dirs[di]
            mp = dom_dir / membership_filename(rr, ndigits=DEFAULT_RESOLUTION_NDIGITS)
            if not mp.exists():
                raise FileNotFoundError(f"missing membership for domain {di}: {mp}")
            loc = np.load(mp).astype(np.int32).reshape(-1)
            if int(loc.shape[0]) != int(verts.shape[0]):
                raise ValueError(f"{mp.name}: length {loc.shape[0]} != induced subgraph verts {verts.shape[0]}")
            if loc.size:
                mx = int(loc.max())
                global_mem[verts] = loc + int(offset)
                offset += mx + 1
            dom_summary = np.load(dom_dir / "summary.npy", allow_pickle=True).item()
            rs = np.asarray(dom_summary["resolutions"], dtype=np.float64)
            ts = np.asarray(dom_summary["time"], dtype=np.float64)
            qs = np.asarray(dom_summary["quality"], dtype=np.float64)
            idx = int(np.where(np.isclose(rs, float(rr)))[0][0])
            total_time += float(ts[idx])
            qv = float(qs[idx])
            if np.isfinite(qv):
                total_quality += qv
                q_count += 1
        n_comm = int(np.unique(global_mem).size)
        merged_results[float(rr)] = {
            "resolution": float(rr),
            "membership": global_mem,
            "n_comm": n_comm,
            "quality": float(total_quality / max(q_count, 1)),
            "time": float(total_time),
        }
        outp = out_dir / membership_filename(rr, ndigits=DEFAULT_RESOLUTION_NDIGITS)
        np.save(outp, global_mem)

    summary = summarize_sweep_results(merged_results)
    summary["algorithm"] = "coarse_kmeans"
    summary["inner_algorithm"] = str(algo)
    np.save(out_dir / "summary.npy", summary, allow_pickle=True)

    meta = {
        "kind": "coarse_kmeans_global_merge",
        "n_nodes": int(n_nodes),
        "n_domains": int(len(domains)),
        "domain_dir_names": [d.name for d in domain_dirs],
        "inner_algorithm": str(algo),
        "partition_type": partition_type,
        "resolutions_merged": [float(x) for x in common],
        "kmeans": kmeans_meta or {},
    }
    (out_dir / "coarse_kmeans_sweep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out_dir.resolve()),
        "n_domains": int(len(domains)),
        "resolutions": [float(x) for x in common],
        "merged_results": merged_results,
    }
