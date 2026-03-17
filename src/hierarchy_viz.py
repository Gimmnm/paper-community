from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Robust loaders for current community.py outputs
# community.py writes CSV with encoding='utf-8-sig', so we must read with
# utf-8-sig or strip BOM from field names. Otherwise 'resolution' may become
# '\ufeffresolution' and all nodes collapse to y=0.
# -----------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, str]] = []
        for r in reader:
            clean = {}
            for k, v in r.items():
                kk = (k or "").lstrip("\ufeff").strip()
                clean[kk] = v
            rows.append(clean)
        return rows



def _try_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default



def _try_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default



def _pick_first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


# -----------------------------------------------------------------------------
# Data normalization for current hierarchy_nodes.csv / hierarchy_edges.csv
# produced by community.py::build_hierarchy_from_sweep
# -----------------------------------------------------------------------------


def _mk_node_id(resolution: float, community: Any) -> str:
    return f"r={float(resolution):.4f}|c={int(float(community)) if str(community).replace('.', '', 1).isdigit() else community}"



def load_hierarchy_nodes(nodes_path: Path) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(nodes_path)
    out: List[Dict[str, Any]] = []
    for r in rows:
        resolution = _try_float(_pick_first(r, ["resolution", "r", "level", "res"]), None)
        community = _pick_first(r, ["community", "community_id", "cluster", "cid", "label"], None)
        size = _try_float(_pick_first(r, ["size", "n", "count", "num_papers"]), 1.0)
        node_id = _pick_first(r, ["node_id", "id", "node", "key"], None)
        if resolution is None:
            raise ValueError(
                f"{nodes_path} missing a readable resolution column. Found columns: {list(r.keys())}"
            )
        if community is None:
            raise ValueError(
                f"{nodes_path} missing a readable community column. Found columns: {list(r.keys())}"
            )
        if node_id is None:
            node_id = _mk_node_id(float(resolution), community)
        out.append(
            {
                "node_id": str(node_id),
                "resolution": float(resolution),
                "community": int(float(community)) if _try_int(community) is not None else str(community),
                "size": float(size if size is not None else 1.0),
                "raw": r,
            }
        )
    if not out:
        raise ValueError(f"no nodes found in {nodes_path}")
    return out



def load_hierarchy_edges(edges_path: Path, nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(edges_path)
    node_ids = {n["node_id"] for n in nodes}
    out: List[Dict[str, Any]] = []
    dropped = 0
    for r in rows:
        src = _pick_first(r, ["source", "src", "parent", "parent_id", "from"], None)
        dst = _pick_first(r, ["target", "dst", "child", "child_id", "to"], None)
        if src is None:
            pr = _try_float(_pick_first(r, ["r_parent", "parent_resolution", "source_resolution", "parent_r"]), None)
            pc = _pick_first(r, ["community_parent", "parent_community", "source_community", "c_parent", "parent_c"], None)
            if pr is not None and pc is not None:
                src = _mk_node_id(pr, pc)
        if dst is None:
            cr = _try_float(_pick_first(r, ["r_child", "child_resolution", "target_resolution", "child_r"]), None)
            cc = _pick_first(r, ["community_child", "child_community", "target_community", "c_child", "child_c"], None)
            if cr is not None and cc is not None:
                dst = _mk_node_id(cr, cc)

        if src not in node_ids or dst not in node_ids:
            dropped += 1
            continue

        out.append(
            {
                "source": str(src),
                "target": str(dst),
                "intersection": float(_try_float(_pick_first(r, ["intersection", "overlap", "count", "weight"]), 1.0) or 1.0),
                "child_share": float(_try_float(_pick_first(r, ["child_share", "flow_share", "share"]), 0.0) or 0.0),
                "parent_share": float(_try_float(_pick_first(r, ["parent_share"]), 0.0) or 0.0),
                "jaccard": float(_try_float(_pick_first(r, ["jaccard"]), 0.0) or 0.0),
                "raw": r,
            }
        )
    if dropped:
        print(f"[hierarchy_viz] dropped {dropped} edges because source/target nodes were not found")
    return out



def load_breakpoints(hierarchy_dir: Path) -> List[Dict[str, Any]]:
    p = hierarchy_dir / "breakpoints.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []



def load_summary(hierarchy_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    p_csv = hierarchy_dir / "summary.csv"
    p_npy = hierarchy_dir / "summary.npy"
    if p_csv.exists():
        rows = _read_csv_rows(p_csv)
        if not rows:
            return None
        cols: Dict[str, List[Any]] = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                cols[k].append(v)
        out: Dict[str, np.ndarray] = {}
        for k, vals in cols.items():
            parsed = [_try_float(v, np.nan) for v in vals]
            out[k] = np.asarray(parsed, dtype=np.float64)
        return out
    if p_npy.exists():
        obj = np.load(p_npy, allow_pickle=True)
        if hasattr(obj, "item"):
            maybe = obj.item()
            if isinstance(maybe, dict):
                return {str(k): np.asarray(v) for k, v in maybe.items()}
    return None


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------


def _group_nodes_by_resolution(nodes: Sequence[Dict[str, Any]]) -> List[Tuple[float, List[Dict[str, Any]]]]:
    by_r: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for n in nodes:
        by_r[float(n["resolution"])].append(n)
    return sorted(by_r.items(), key=lambda kv: kv[0])



def _compute_layer_orders(nodes: Sequence[Dict[str, Any]], edges: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    layers = _group_nodes_by_resolution(nodes)
    pos_x: Dict[str, float] = {}

    for _, layer_nodes in layers:
        layer_nodes.sort(key=lambda n: (-float(n["size"]), str(n["community"])))
        for i, n in enumerate(layer_nodes):
            pos_x[n["node_id"]] = float(i)

    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for e in edges:
        w = max(float(e.get("intersection", 1.0)), 1e-9)
        incoming[e["target"]].append((e["source"], w))
        outgoing[e["source"]].append((e["target"], w))

    for _, layer_nodes in layers[1:]:
        scored = []
        for n in layer_nodes:
            inc = incoming.get(n["node_id"], [])
            bary = sum(pos_x[src] * w for src, w in inc) / sum(w for _, w in inc) if inc else pos_x[n["node_id"]]
            scored.append((bary, -float(n["size"]), str(n["community"]), n))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        for i, (_, _, _, n) in enumerate(scored):
            pos_x[n["node_id"]] = float(i)

    for _, layer_nodes in reversed(layers[:-1]):
        scored = []
        for n in layer_nodes:
            out = outgoing.get(n["node_id"], [])
            bary = sum(pos_x[dst] * w for dst, w in out) / sum(w for _, w in out) if out else pos_x[n["node_id"]]
            scored.append((bary, -float(n["size"]), str(n["community"]), n))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        for i, (_, _, _, n) in enumerate(scored):
            pos_x[n["node_id"]] = float(i)

    for _, layer_nodes in layers:
        xs = [pos_x[n["node_id"]] for n in layer_nodes]
        mean_x = float(np.mean(xs)) if xs else 0.0
        for n in layer_nodes:
            pos_x[n["node_id"]] -= mean_x
    return pos_x



def _edge_weight_value(e: Dict[str, Any], weight_by: str) -> float:
    if weight_by == "child_share":
        return float(e.get("child_share", 0.0))
    if weight_by == "jaccard":
        return float(e.get("jaccard", 0.0))
    return float(e.get("intersection", 1.0))


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_hierarchy_layered(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    out_svg: Optional[Path] = None,
    weight_by: str = "intersection",
    min_edge_value: float = 0.0,
    max_nodes_per_layer: Optional[int] = None,
    annotate_top_n_per_layer: int = 8,
    y_mode: str = "resolution",
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 180,
) -> Dict[str, Any]:
    node_by_id = {n["node_id"]: n for n in nodes}
    layers = _group_nodes_by_resolution(nodes)
    resolutions = [r for r, _ in layers]
    if len(resolutions) <= 1:
        print(
            "[hierarchy_viz] warning: only one unique resolution found in hierarchy_nodes.csv. "
            "If this is unexpected, check CSV columns and whether hierarchy really used multiple sweep levels."
        )
    pos_x = _compute_layer_orders(nodes, edges)

    y_by_resolution: Dict[float, float]
    if y_mode == "layer_index":
        y_by_resolution = {r: float(i) for i, r in enumerate(resolutions)}
    else:
        y_by_resolution = {r: float(r) for r in resolutions}

    keep_ids = set(node_by_id)
    if max_nodes_per_layer is not None and max_nodes_per_layer > 0:
        keep_ids = set()
        for _, layer_nodes in layers:
            ranked = sorted(layer_nodes, key=lambda n: -float(n["size"]))[: int(max_nodes_per_layer)]
            keep_ids.update(n["node_id"] for n in ranked)
        expanded = set(keep_ids)
        for e in edges:
            if e["source"] in keep_ids or e["target"] in keep_ids:
                expanded.add(e["source"])
                expanded.add(e["target"])
        keep_ids = expanded

    plot_nodes = [n for n in nodes if n["node_id"] in keep_ids]
    plot_edges = [
        e for e in edges
        if e["source"] in keep_ids and e["target"] in keep_ids and _edge_weight_value(e, weight_by) >= float(min_edge_value)
    ]
    if not plot_nodes:
        raise ValueError("no nodes remain after filtering; relax max_nodes_per_layer/min_edge_value")

    sizes = np.asarray([max(float(n["size"]), 1.0) for n in plot_nodes], dtype=np.float64)
    sqrt_sizes = np.sqrt(sizes)
    s_scaled = 35.0 + 420.0 * (sqrt_sizes - sqrt_sizes.min()) / max(sqrt_sizes.max() - sqrt_sizes.min(), 1e-9)

    cmap = plt.get_cmap("viridis")
    r_min, r_max = float(min(resolutions)), float(max(resolutions))

    def color_for_r(r: float):
        if math.isclose(r_min, r_max):
            return cmap(0.6)
        t = (float(r) - r_min) / max(r_max - r_min, 1e-12)
        return cmap(t)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if plot_edges:
        vals = np.asarray([max(_edge_weight_value(e, weight_by), 1e-9) for e in plot_edges], dtype=np.float64)
        sqrt_vals = np.sqrt(vals)
        lw = 0.4 + 4.2 * (sqrt_vals - sqrt_vals.min()) / max(sqrt_vals.max() - sqrt_vals.min(), 1e-9)
        for e, line_w in zip(plot_edges, lw):
            s = node_by_id[e["source"]]
            t = node_by_id[e["target"]]
            xs = pos_x[s["node_id"]]
            xt = pos_x[t["node_id"]]
            ys = y_by_resolution[float(s["resolution"])]
            yt = y_by_resolution[float(t["resolution"])]
            ax.plot([xs, xt], [ys, yt], color=(0.2, 0.2, 0.2, 0.16), lw=float(line_w), solid_capstyle="round", zorder=1)

    xs = np.asarray([pos_x[n["node_id"]] for n in plot_nodes], dtype=np.float64)
    ys = np.asarray([y_by_resolution[float(n["resolution"])] for n in plot_nodes], dtype=np.float64)
    cs = [color_for_r(float(n["resolution"])) for n in plot_nodes]
    ax.scatter(xs, ys, s=s_scaled, c=cs, edgecolors="black", linewidths=0.35, alpha=0.95, zorder=3)

    by_layer: Dict[float, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for n in plot_nodes:
        by_layer[float(n["resolution"])].append((float(n["size"]), n))
    for r, items in by_layer.items():
        items.sort(key=lambda t: -t[0])
        for _, n in items[: max(0, int(annotate_top_n_per_layer))]:
            x = pos_x[n["node_id"]]
            y = y_by_resolution[float(r)]
            ax.text(x + 0.08, y, f"{n['community']} ({int(round(float(n['size'])))})", fontsize=7, alpha=0.9, va="center")

    for r in resolutions:
        ax.axhline(y_by_resolution[float(r)], color=(0.4, 0.4, 0.4, 0.10), lw=0.8, zorder=0)

    ax.set_title(f"Hierarchy layered view  |  weight={weight_by}  |  y={y_mode}")
    ax.set_xlabel("community order within resolution")
    ax.set_ylabel("resolution" if y_mode == "resolution" else "layer index")

    if y_mode == "layer_index":
        tick_r = resolutions[:: max(1, len(resolutions) // 12)]
        ax.set_yticks([y_by_resolution[float(r)] for r in tick_r])
        ax.set_yticklabels([f"{float(r):.4f}" for r in tick_r])
    else:
        tick_r = resolutions[:: max(1, len(resolutions) // 12)]
        ax.set_yticks(tick_r)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)

    return {
        "n_layers": int(len(resolutions)),
        "n_nodes_plotted": int(len(plot_nodes)),
        "n_edges_plotted": int(len(plot_edges)),
        "resolutions": [float(r) for r in resolutions],
        "y_mode": y_mode,
    }



def plot_breakpoint_diagnostics(
    *,
    hierarchy_dir: Path,
    out_png: Path,
    out_svg: Optional[Path] = None,
    top_k: int = 12,
    dpi: int = 180,
) -> Optional[Dict[str, Any]]:
    summary = load_summary(hierarchy_dir)
    breakpoints = load_breakpoints(hierarchy_dir)
    if summary is None:
        return None

    r = None
    for key in ["resolution", "resolutions", "r"]:
        if key in summary:
            r = np.asarray(summary[key], dtype=np.float64)
            break
    if r is None:
        return None

    n_comm = summary.get("n_comm")
    vi = summary.get("vi_adjacent")
    quality = summary.get("quality")

    nrows = 2 if quality is None else 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 8 if nrows == 2 else 10), dpi=dpi, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    if n_comm is not None:
        axes[0].plot(r, np.asarray(n_comm, dtype=np.float64), lw=1.5)
        axes[0].set_ylabel("# communities")
        axes[0].set_title("Sweep diagnostics")
    if vi is not None:
        axes[1].plot(r, np.asarray(vi, dtype=np.float64), lw=1.5)
        axes[1].set_ylabel("VI(adjacent)")
    if quality is not None:
        axes[2].plot(r, np.asarray(quality, dtype=np.float64), lw=1.5)
        axes[2].set_ylabel("quality")
        axes[2].set_xlabel("resolution")
    else:
        axes[-1].set_xlabel("resolution")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    if breakpoints:
        bp_sorted = sorted(breakpoints, key=lambda x: float(_pick_first(x, ["score"], 0.0) or 0.0), reverse=True)
        bp_top = bp_sorted[: int(top_k)]
        for ax in axes:
            for bp in bp_top:
                rr = _try_float(_pick_first(bp, ["resolution", "r", "r_child"]), None)
                if rr is not None:
                    ax.axvline(rr, color="crimson", lw=1.0, alpha=0.28)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)
    return {"n_breakpoints": int(len(breakpoints))}



def write_overview_json(
    *,
    hierarchy_dir: Path,
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    layered_info: Dict[str, Any],
    breakpoint_info: Optional[Dict[str, Any]],
) -> Path:
    layers = _group_nodes_by_resolution(nodes)
    out = {
        "hierarchy_dir": str(hierarchy_dir),
        "n_layers": int(len(layers)),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "nodes_per_layer": {f"{float(r):.4f}": int(len(ns)) for r, ns in layers},
        "layered_plot": layered_info,
        "breakpoint_plot": breakpoint_info,
    }
    path = hierarchy_dir / "hierarchy_viz_overview.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize hierarchy outputs from community.py::build_hierarchy_from_sweep")
    p.add_argument("--hierarchy-dir", required=True, help="Directory containing hierarchy_nodes.csv / hierarchy_edges.csv")
    p.add_argument("--nodes-file", default="hierarchy_nodes.csv")
    p.add_argument("--edges-file", default="hierarchy_edges.csv")
    p.add_argument("--weight-by", choices=["intersection", "child_share", "jaccard"], default="intersection")
    p.add_argument("--min-edge-value", type=float, default=0.0)
    p.add_argument("--max-nodes-per-layer", type=int, default=40)
    p.add_argument("--annotate-top-n-per-layer", type=int, default=8)
    p.add_argument("--y-mode", choices=["resolution", "layer_index"], default="resolution")
    p.add_argument("--out-prefix", default="hierarchy")
    p.add_argument("--no-breakpoints", action="store_true")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    hierarchy_dir = Path(args.hierarchy_dir)
    nodes_path = hierarchy_dir / args.nodes_file
    edges_path = hierarchy_dir / args.edges_file

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"edges file not found: {edges_path}")

    nodes = load_hierarchy_nodes(nodes_path)
    edges = load_hierarchy_edges(edges_path, nodes)

    unique_res = sorted({float(n['resolution']) for n in nodes})
    print(f"[hierarchy_viz] loaded nodes={len(nodes)} edges={len(edges)} unique_resolutions={len(unique_res)}")
    if unique_res:
        print(f"[hierarchy_viz] resolution range: {unique_res[0]:.4f} -> {unique_res[-1]:.4f}")

    layered_png = hierarchy_dir / f"{args.out_prefix}_layered.png"
    layered_svg = hierarchy_dir / f"{args.out_prefix}_layered.svg"
    layered_info = plot_hierarchy_layered(
        nodes,
        edges,
        out_png=layered_png,
        out_svg=layered_svg,
        weight_by=args.weight_by,
        min_edge_value=float(args.min_edge_value),
        max_nodes_per_layer=int(args.max_nodes_per_layer) if args.max_nodes_per_layer and args.max_nodes_per_layer > 0 else None,
        annotate_top_n_per_layer=int(args.annotate_top_n_per_layer),
        y_mode=args.y_mode,
    )

    breakpoint_info = None
    if not args.no_breakpoints:
        bp_png = hierarchy_dir / f"{args.out_prefix}_breakpoints.png"
        bp_svg = hierarchy_dir / f"{args.out_prefix}_breakpoints.svg"
        breakpoint_info = plot_breakpoint_diagnostics(
            hierarchy_dir=hierarchy_dir,
            out_png=bp_png,
            out_svg=bp_svg,
        )

    overview = write_overview_json(
        hierarchy_dir=hierarchy_dir,
        nodes=nodes,
        edges=edges,
        layered_info=layered_info,
        breakpoint_info=breakpoint_info,
    )
    print(f"[hierarchy_viz] layered plot -> {layered_png}")
    if breakpoint_info is not None:
        print(f"[hierarchy_viz] breakpoint plot -> {hierarchy_dir / f'{args.out_prefix}_breakpoints.png'}")
    print(f"[hierarchy_viz] overview -> {overview}")


if __name__ == "__main__":
    main()
