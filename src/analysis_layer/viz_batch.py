"""
Batch offline figures: UMAP colored by membership, coarse community graph, top-community induced subgraphs on UMAP.

Output root: ``out/viz_batch/<run_tag>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from app_layer.demo_graph import DemoCommunityGraph

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app_layer.demo_graph import build_demo_graph_from_membership, load_membership_for_resolution_light, resolve_membership_for_resolution_light
from app_layer.demo_search import load_papers_from_project
from data_layer.breakpoint_schedule import default_breakpoints_csv, infer_breakpoint_run_id_from_leiden_dir, load_breakpoint_resolutions_for_run
from foundation_layer.diagram2d import plot_scatter
from foundation_layer.network import load_edges_npz
from foundation_layer.project_paths import REPO_ROOT, out_dir


def resolve_coarse_kmeans_domain_labels(leiden_dir: Path, *, override: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load per-paper domain ids (0..K-1) for coarse_kmeans pipelines; optional explicit npy path."""
    if override is not None and str(override).strip():
        p = Path(override)
        if p.is_file():
            return np.load(p).astype(np.int32).reshape(-1)
        return None
    ld = Path(leiden_dir)
    cand = ld / "labels.npy"
    if cand.is_file():
        return np.load(cand).astype(np.int32).reshape(-1)
    meta_p = ld / "coarse_kmeans_sweep_meta.json"
    if meta_p.is_file():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            km = meta.get("kmeans") or {}
            raw_ln = km.get("labels_npy")
            if raw_ln:
                lp = Path(str(raw_ln))
                if lp.is_file():
                    return np.load(lp).astype(np.int32).reshape(-1)
            dd = km.get("domains_dir")
            if dd:
                lp = Path(str(dd)) / "labels.npy"
                if lp.is_file():
                    return np.load(lp).astype(np.int32).reshape(-1)
        except Exception:
            pass
    return None


def _summary_resolutions(leiden_dir: Path) -> List[float]:
    summary = leiden_dir / "summary.npy"
    if not summary.is_file():
        return []
    try:
        d = np.load(summary, allow_pickle=True).item()
        rs = np.asarray(d.get("resolutions", []), dtype=np.float64).tolist()
        return sorted(set(round(float(x), 4) for x in rs))
    except Exception:
        return []


def select_resolutions(leiden_dir: Path, args: argparse.Namespace) -> List[float]:
    if args.resolutions:
        return [float(x) for x in args.resolutions]
    rs: List[float] = []
    src = str(getattr(args, "resolution_source", "breakpoints"))
    if src == "breakpoints":
        csv_p = Path(args.breakpoints_csv)
        rid = getattr(args, "breakpoint_run_id", None) or infer_breakpoint_run_id_from_leiden_dir(leiden_dir)
        if rid and csv_p.is_file():
            rs_bp = load_breakpoint_resolutions_for_run(
                csv_p,
                run_id=str(rid),
                time_window=str(getattr(args, "breakpoint_time_window", "all")),
            )
            if rs_bp:
                rs = [float(x) for x in rs_bp]
        if not rs:
            print("[viz-batch] warn: breakpoints unavailable; falling back to summary slice")
    if not rs:
        rs = _summary_resolutions(leiden_dir)
        if not rs:
            rs = [float(args.default_resolution)]
    rmin, rmax = float(args.r_min), float(args.r_max)
    rs = [r for r in rs if rmin <= r <= rmax]
    step = int(args.resolution_stride)
    if step > 1:
        rs = rs[::step]
    cap = int(args.max_resolutions)
    if cap > 0 and len(rs) > cap:
        rs = rs[:cap]
    return rs


def plot_umap_membership_resolution(
    *,
    umap_xy: np.ndarray,
    leiden_dir: Path,
    resolution: float,
    out_png: Path,
    max_points: int,
) -> Tuple[str, float]:
    path_used, r_eff, _exact = resolve_membership_for_resolution_light(leiden_dir, float(resolution))
    labels = load_membership_for_resolution_light(leiden_dir, float(resolution))
    if int(labels.shape[0]) != int(umap_xy.shape[0]):
        raise ValueError(f"UMAP rows {umap_xy.shape[0]} != membership {labels.shape[0]}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = f"UMAP × communities (r≈{r_eff:.4f}, {leiden_dir.name})"
    mp = int(max_points) if max_points and max_points > 0 else None
    plot_scatter(
        umap_xy.astype(np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        title=title,
        out_png=out_png,
        point_size=1.2,
        alpha=0.72,
        max_points=mp,
        verbose=False,
    )
    return str(path_used), float(r_eff)


def plot_community_meta_graph(
    *,
    graph: "DemoCommunityGraph",
    out_png: Path,
    max_nodes: int,
) -> None:
    try:
        import igraph as ig  # type: ignore
    except Exception:
        plt.figure(figsize=(8, 6), dpi=140)
        plt.text(0.5, 0.5, "igraph not installed; skip community graph", ha="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    comms = sorted(graph.communities.values(), key=lambda c: int(c.size), reverse=True)
    take = min(int(max_nodes), len(comms))
    nodes = comms[:take]
    id_set = {int(c.cid) for c in nodes}
    edges: List[Tuple[int, int, float]] = []
    for (a, b), w in graph.community_edges.items():
        if int(a) in id_set and int(b) in id_set:
            edges.append((int(a), int(b), float(w)))

    g = ig.Graph()
    g.add_vertices([str(int(c.cid)) for c in nodes])
    name_to_idx = {str(int(c.cid)): i for i, c in enumerate(nodes)}
    elist = []
    weights = []
    for a, b, w in edges:
        sa, sb = str(int(a)), str(int(b))
        if sa in name_to_idx and sb in name_to_idx:
            elist.append((name_to_idx[sa], name_to_idx[sb]))
            weights.append(float(w))
    if elist:
        g.add_edges(elist)
        try:
            layout = g.layout_fruchterman_reingold(weights=weights)
        except Exception:
            layout = g.layout_fruchterman_reingold()
    else:
        layout = g.layout_circle()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7), dpi=160)
    xs = [layout[i][0] for i in range(g.vcount())]
    ys = [layout[i][1] for i in range(g.vcount())]
    plt.scatter(xs, ys, s=[max(18.0, min(220.0, float(c.size) * 0.25)) for c in nodes], alpha=0.78, c="#3468a5")
    for i, c in enumerate(nodes):
        plt.text(xs[i], ys[i], str(int(c.cid)), fontsize=6, ha="center", va="center", color="white")
    if elist:
        for (a, b), w in zip(elist, weights):
            x1, y1 = layout[a]
            x2, y2 = layout[b]
            plt.plot([x1, x2], [y1, y2], color="#888888", linewidth=max(0.2, min(2.5, float(w) * 0.02)), alpha=0.35)
    plt.axis("off")
    plt.title(f"Community meta-graph (top {take} by size)")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_induced_subgraph_umap(
    *,
    umap_xy: np.ndarray,
    membership: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cid: int,
    out_png: Path,
    max_nodes: int,
    max_edges_draw: int,
    seed: int,
) -> None:
    membership = np.asarray(membership, dtype=np.int32)
    idx0 = np.flatnonzero(membership == int(cid)).astype(np.int64, copy=False)
    if idx0.size == 0:
        return
    if idx0.size > int(max_nodes):
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(idx0, size=int(max_nodes), replace=False)
        member_idx = np.sort(pick)
    else:
        member_idx = idx0
    mset = set(int(x) for x in member_idx.tolist())

    eu = []
    ev = []
    ew = []
    for a, b, ww in zip(u.tolist(), v.tolist(), w.tolist()):
        if int(a) in mset and int(b) in mset:
            eu.append(int(a))
            ev.append(int(b))
            ew.append(float(ww))
    edges_uv = list(zip(eu, ev, ew))
    edges_uv.sort(key=lambda t: -t[2])
    edges_uv = edges_uv[: int(max_edges_draw)]

    xy = umap_xy.astype(np.float32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7), dpi=160)
    sub = xy[member_idx]
    plt.scatter(sub[:, 0], sub[:, 1], s=4.0, alpha=0.75, c="#c44e52", linewidths=0)
    for a, b, _ww in edges_uv:
        plt.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color="#333333", alpha=0.12, linewidth=0.35)
    plt.title(f"Induced subgraph on UMAP (community {int(cid)}, n={member_idx.size})")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def run_batch_for_leiden_dir(
    *,
    base_dir: Path,
    leiden_dir: Path,
    graph_npz: Path,
    umap_npy: Path,
    out_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not umap_npy.is_file():
        raise FileNotFoundError(f"missing UMAP: {umap_npy}")
    Y = np.load(umap_npy).astype(np.float32)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"expected (N,2) in {umap_npy}")

    papers = load_papers_from_project(base_dir, exclude_selfcite=False, force_reingest=False)
    n_expect = len(papers) - 1
    if int(Y.shape[0]) != int(n_expect):
        raise ValueError(f"umap rows {Y.shape[0]} != papers-1 {n_expect}")

    u, v, w, n_nodes, _k0, _norm = load_edges_npz(graph_npz)
    if int(n_nodes) != int(n_expect):
        raise ValueError(f"graph n_nodes={n_nodes} != {n_expect}")

    dom_override = getattr(args, "domain_labels_npy", None)
    dom_path = Path(str(dom_override)) if dom_override and str(dom_override).strip() else None
    domain_labels = resolve_coarse_kmeans_domain_labels(leiden_dir, override=dom_path)
    resolutions = select_resolutions(leiden_dir, args)
    written: List[str] = []
    if domain_labels is not None:
        if int(domain_labels.shape[0]) != int(n_expect):
            print(
                f"[viz-batch] warn: domain labels length {domain_labels.shape[0]} != {n_expect}; "
                "skip umap_kmeans_domains.png",
            )
        else:
            k_dom = int(np.unique(domain_labels).size)
            dom_png = out_root / "umap_kmeans_domains.png"
            mp = int(args.max_points_umap) if args.max_points_umap and args.max_points_umap > 0 else None
            plot_scatter(
                Y.astype(np.float32),
                labels=np.asarray(domain_labels, dtype=np.int32),
                title=f"UMAP × k-means domains ({leiden_dir.name}, K={k_dom})",
                out_png=dom_png,
                point_size=1.2,
                alpha=0.72,
                max_points=mp,
                verbose=False,
            )
            written.append(str(dom_png))

    for r in resolutions:
        r_tag = f"r{r:.4f}"
        sub = out_root / r_tag
        sub.mkdir(parents=True, exist_ok=True)

        path_used, r_eff = plot_umap_membership_resolution(
            umap_xy=Y,
            leiden_dir=leiden_dir,
            resolution=float(r),
            out_png=sub / "umap_membership.png",
            max_points=int(args.max_points_umap),
        )
        meta = {"membership_path": path_used, "resolution_effective": float(r_eff)}
        (sub / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(sub / "umap_membership.png"))

        membership = load_membership_for_resolution_light(leiden_dir, float(r_eff))
        g = build_demo_graph_from_membership(
            list(papers),
            membership,
            resolution=float(r_eff),
            u=u,
            v=v,
            w=w,
            top_center=int(args.top_center),
            top_bridge=int(args.top_bridge),
            top_neighbor_comms=int(args.top_neighbor_comms),
        )
        plot_community_meta_graph(
            graph=g,
            out_png=sub / "community_meta_graph.png",
            max_nodes=int(args.meta_graph_max_nodes),
        )
        written.append(str(sub / "community_meta_graph.png"))

        comms = sorted(g.communities.values(), key=lambda c: int(c.size), reverse=True)
        topn = min(int(args.top_communities_subgraph), len(comms))
        sg_root = sub / f"subgraphs_top{topn}"
        sg_root.mkdir(parents=True, exist_ok=True)
        for c in comms[:topn]:
            plot_induced_subgraph_umap(
                umap_xy=Y,
                membership=membership,
                u=u,
                v=v,
                w=w,
                cid=int(c.cid),
                out_png=sg_root / f"c{int(c.cid)}.png",
                max_nodes=int(args.subgraph_max_nodes),
                max_edges_draw=int(args.subgraph_max_edges),
                seed=int(args.layout_seed) + int(c.cid),
            )
            written.append(str(sg_root / f"c{int(c.cid)}.png"))

    return {"n_resolutions": len(resolutions), "artifacts": written}


def register_viz_batch_args(
    p: argparse.ArgumentParser,
    *,
    base_dir: Optional[Path] = None,
    default_out_dir: Optional[Path] = None,
) -> None:
    root = REPO_ROOT if base_dir is None else Path(base_dir)
    od = default_out_dir if default_out_dir is not None else out_dir(root)
    p.add_argument("--run-tag", type=str, required=True, help="folder under out/viz_batch/")
    p.add_argument("--leiden-dir", type=str, required=True)
    p.add_argument("--graph-npz", type=str, default=str(od / "mutual_knn_k50.npz"))
    p.add_argument("--umap-npy", type=str, default=str(od / "umap2d.npy"))
    p.add_argument(
        "--resolution-source",
        type=str,
        choices=["breakpoints", "summary"],
        default="breakpoints",
        help="breakpoints: ~10 r per algorithm from comparison_breakpoints.csv; summary: slice membership grid",
    )
    p.add_argument("--breakpoints-csv", type=str, default=str(default_breakpoints_csv(root)))
    p.add_argument(
        "--breakpoint-run-id",
        type=str,
        default=None,
        help="CSV run_id; default: infer from leiden-dir folder name (e.g. leiden_sweep_cpm → leiden_cpm)",
    )
    p.add_argument("--breakpoint-time-window", type=str, default="all")
    p.add_argument("--resolutions", type=float, nargs="*", default=None)
    p.add_argument("--r-min", type=float, default=0.001)
    p.add_argument("--r-max", type=float, default=2.0)
    p.add_argument("--resolution-stride", type=int, default=1)
    p.add_argument("--max-resolutions", type=int, default=12)
    p.add_argument("--default-resolution", type=float, default=0.2, help="if summary missing")
    p.add_argument("--max-points-umap", type=int, default=35_000)
    p.add_argument("--meta-graph-max-nodes", type=int, default=80)
    p.add_argument("--top-communities-subgraph", type=int, default=50)
    p.add_argument("--subgraph-max-nodes", type=int, default=2500)
    p.add_argument("--subgraph-max-edges", type=int, default=12_000)
    p.add_argument("--top-center", type=int, default=8)
    p.add_argument("--top-bridge", type=int, default=8)
    p.add_argument("--top-neighbor-comms", type=int, default=12)
    p.add_argument("--layout-seed", type=int, default=42)
    p.add_argument(
        "--domain-labels-npy",
        type=str,
        default="",
        help="可选：coarse k-means 的 labels.npy（每篇论文 domain id）；用于额外导出 umap_kmeans_domains.png",
    )


def run_viz_batch(args: argparse.Namespace) -> Dict[str, Any]:
    base_dir = REPO_ROOT
    out_root = base_dir / "out" / "viz_batch" / str(args.run_tag)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = run_batch_for_leiden_dir(
        base_dir=base_dir,
        leiden_dir=Path(args.leiden_dir),
        graph_npz=Path(args.graph_npz),
        umap_npy=Path(args.umap_npy),
        out_root=out_root,
        args=args,
    )
    (out_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[viz-batch] wrote under {out_root}")
    summary["out_root"] = str(out_root.resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch UMAP / community-graph / subgraph viz.")
    register_viz_batch_args(p, base_dir=REPO_ROOT, default_out_dir=out_dir(REPO_ROOT))
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_viz_batch(args)


if __name__ == "__main__":
    main()
