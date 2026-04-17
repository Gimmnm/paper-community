from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class NodeKey:
    r: float
    c: int

    @property
    def id(self) -> str:
        return f"r={self.r:.4f}|c={self.c}"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        out: List[Dict[str, str]] = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                kk = (k or "").lstrip("\ufeff").strip()
                clean[kk] = v
            out.append(clean)
        return out


def _try_float(x: Optional[str], default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _try_int(x: Optional[str], default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def load_nodes(nodes_csv: Path) -> Dict[NodeKey, float]:
    """
    Returns mapping NodeKey -> size.
    """
    rows = _read_csv_rows(nodes_csv)
    out: Dict[NodeKey, float] = {}
    for r in rows:
        rr = round(_try_float(r.get("resolution") or r.get("r")), 4)
        cc = _try_int(r.get("community") or r.get("community_id") or r.get("cid"))
        if not np.isfinite(rr) or cc < 0:
            continue
        size = _try_float(r.get("size"), 1.0)
        out[NodeKey(rr, cc)] = float(size if np.isfinite(size) else 1.0)
    return out


def load_edges(edges_csv: Path) -> Tuple[Dict[NodeKey, List[NodeKey]], Dict[NodeKey, NodeKey], Dict[Tuple[NodeKey, NodeKey], float]]:
    """
    Returns:
      - children_by_parent: parent -> [child...]
      - parent_by_child: child -> parent
      - child_share: (parent, child) -> child_share
    """
    rows = _read_csv_rows(edges_csv)
    children_by_parent: Dict[NodeKey, List[NodeKey]] = defaultdict(list)
    parent_by_child: Dict[NodeKey, NodeKey] = {}
    child_share: Dict[Tuple[NodeKey, NodeKey], float] = {}
    for r in rows:
        rp = round(_try_float(r.get("r_parent")), 4)
        rc = round(_try_float(r.get("r_child")), 4)
        cp = _try_int(r.get("community_parent"))
        cc = _try_int(r.get("community_child"))
        if not np.isfinite(rp) or not np.isfinite(rc) or cp < 0 or cc < 0:
            continue
        p = NodeKey(rp, cp)
        c = NodeKey(rc, cc)
        children_by_parent[p].append(c)
        parent_by_child[c] = p
        child_share[(p, c)] = float(_try_float(r.get("child_share"), 0.0))
    return children_by_parent, parent_by_child, child_share


def roots_at_resolution(node_sizes: Dict[NodeKey, float], r0: float) -> List[NodeKey]:
    r0 = round(float(r0), 4)
    roots = [k for k in node_sizes.keys() if math_isclose(k.r, r0)]
    roots.sort(key=lambda k: (-node_sizes.get(k, 1.0), k.c))
    return roots


def math_isclose(a: float, b: float, tol: float = 5e-5) -> bool:
    return abs(float(a) - float(b)) <= float(tol)


def subtree_nodes(
    root: NodeKey,
    *,
    children_by_parent: Dict[NodeKey, List[NodeKey]],
    r_max: Optional[float] = None,
) -> Set[NodeKey]:
    out: Set[NodeKey] = set()
    q = deque([root])
    while q:
        cur = q.popleft()
        if cur in out:
            continue
        if r_max is not None and float(cur.r) > float(r_max) + 1e-12:
            continue
        out.add(cur)
        for ch in children_by_parent.get(cur, []):
            q.append(ch)
    return out


def layered_tree_layout(
    nodes: Sequence[NodeKey],
    *,
    node_sizes: Dict[NodeKey, float],
    children_by_parent: Dict[NodeKey, List[NodeKey]],
) -> Tuple[Dict[NodeKey, float], Dict[NodeKey, float], List[float]]:
    """
    y is resolution (sorted), x is computed by a simple DFS order per layer.
    Returns x_map, y_map, sorted unique resolutions.
    """
    by_r: Dict[float, List[NodeKey]] = defaultdict(list)
    for n in nodes:
        by_r[float(n.r)].append(n)
    resolutions = sorted(by_r.keys())

    # default ordering within a layer: by size desc, then id
    x_map: Dict[NodeKey, float] = {}
    for rr in resolutions:
        layer = by_r[rr]
        layer.sort(key=lambda k: (-node_sizes.get(k, 1.0), k.c))
        for i, k in enumerate(layer):
            x_map[k] = float(i)

    # refine ordering by barycenter from parents (one-pass)
    for rr in resolutions[1:]:
        layer = by_r[rr]
        scored = []
        for k in layer:
            # pick its (unique) parent at prev layers if present
            # since we are already on a subtree, use any parent that points to this k
            # (edges are only between adjacent resolutions)
            parents = [p for p, chs in children_by_parent.items() if k in chs]
            if parents:
                px = float(np.mean([x_map[p] for p in parents]))
            else:
                px = x_map[k]
            scored.append((px, -node_sizes.get(k, 1.0), k.c, k))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        for i, (_, _, _, k) in enumerate(scored):
            x_map[k] = float(i)

    # center per layer
    for rr in resolutions:
        layer = by_r[rr]
        xs = [x_map[k] for k in layer]
        mean_x = float(np.mean(xs)) if xs else 0.0
        for k in layer:
            x_map[k] -= mean_x

    y_map = {k: float(k.r) for k in nodes}
    return x_map, y_map, resolutions


def plot_subtree(
    root: NodeKey,
    *,
    out_path: Path,
    nodes: Sequence[NodeKey],
    node_sizes: Dict[NodeKey, float],
    children_by_parent: Dict[NodeKey, List[NodeKey]],
    child_share: Dict[Tuple[NodeKey, NodeKey], float],
    weight_min: float = 0.0,
    dpi: int = 220,
) -> Dict[str, int]:
    x_map, y_map, resolutions = layered_tree_layout(nodes, node_sizes=node_sizes, children_by_parent=children_by_parent)

    # edges within the subtree
    plot_edges: List[Tuple[NodeKey, NodeKey, float]] = []
    node_set = set(nodes)
    for p in node_set:
        for c in children_by_parent.get(p, []):
            if c not in node_set:
                continue
            w = float(child_share.get((p, c), 0.0))
            if w >= float(weight_min):
                plot_edges.append((p, c, w))

    # node marker sizes
    sizes = np.asarray([max(node_sizes.get(k, 1.0), 1.0) for k in nodes], dtype=float)
    s = np.sqrt(sizes)
    s_scaled = 10.0 + 140.0 * (s - s.min()) / max(s.max() - s.min(), 1e-9)

    fig_h = max(3.5, 0.55 * len(resolutions))
    fig_w = 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if plot_edges:
        ws = np.asarray([max(w, 1e-9) for _, _, w in plot_edges], dtype=float)
        lw = 0.35 + 2.2 * (ws - ws.min()) / max(ws.max() - ws.min(), 1e-9)
        for (p, c, w), line_w in zip(plot_edges, lw):
            ax.plot([x_map[p], x_map[c]], [y_map[p], y_map[c]], color=(0.2, 0.2, 0.2, 0.25), lw=float(line_w))

    xs = np.asarray([x_map[k] for k in nodes], dtype=float)
    ys = np.asarray([y_map[k] for k in nodes], dtype=float)
    ax.scatter(xs, ys, s=s_scaled, color=(0.25, 0.55, 0.85, 0.95), edgecolors="black", linewidths=0.25, zorder=3)

    ax.set_title(f"Root tree from {root.id}  (nodes={len(nodes)})", fontsize=10)
    ax.set_xlabel("within-layer order")
    ax.set_ylabel("resolution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.12, linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {"n_nodes": int(len(nodes)), "n_edges": int(len(plot_edges)), "n_layers": int(len(resolutions))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render one tree per root community at an initial resolution")
    p.add_argument("--hierarchy-dir", required=True, help="Directory containing hierarchy_nodes.csv / hierarchy_edges.csv")
    p.add_argument("--root-resolution", type=float, default=0.001, help="Root layer resolution (e.g. 0.001 for CPM)")
    p.add_argument("--r-max", type=float, default=0.01, help="Only trace descendants up to this resolution")
    p.add_argument("--weight-min", type=float, default=0.10, help="Min child_share to draw an edge")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <hierarchy-dir>/root_trees_*)")
    p.add_argument("--format", type=str, default="png", choices=["png", "svg"])
    p.add_argument("--only-top", type=int, default=None, help="Only render top-N largest roots (debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy_dir = Path(args.hierarchy_dir)
    nodes_csv = hierarchy_dir / "hierarchy_nodes.csv"
    edges_csv = hierarchy_dir / "hierarchy_edges.csv"
    if not nodes_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError("hierarchy_nodes.csv / hierarchy_edges.csv not found under hierarchy-dir")

    node_sizes = load_nodes(nodes_csv)
    children_by_parent, _parent_by_child, child_share = load_edges(edges_csv)

    r0 = float(args.root_resolution)
    roots = [k for k in node_sizes.keys() if math_isclose(k.r, round(r0, 4))]
    roots.sort(key=lambda k: (-node_sizes.get(k, 1.0), k.c))
    if args.only_top is not None and int(args.only_top) > 0:
        roots = roots[: int(args.only_top)]

    out_dir = Path(args.out_dir) if args.out_dir else hierarchy_dir / f"root_trees_r{round(r0,4):.4f}_to_r{float(args.r_max):.4f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "hierarchy_dir": str(hierarchy_dir),
        "root_resolution": round(r0, 4),
        "r_max": float(args.r_max),
        "weight_min": float(args.weight_min),
        "n_roots": int(len(roots)),
        "roots": [],
    }

    for i, root in enumerate(roots, start=1):
        sub = subtree_nodes(root, children_by_parent=children_by_parent, r_max=float(args.r_max))
        nodes = sorted(sub, key=lambda k: (k.r, k.c))
        fname = f"root_c{root.c:05d}_n{int(node_sizes.get(root, 1.0)):04d}.{args.format}"
        out_path = out_dir / fname
        stats = plot_subtree(
            root,
            out_path=out_path,
            nodes=nodes,
            node_sizes=node_sizes,
            children_by_parent=children_by_parent,
            child_share=child_share,
            weight_min=float(args.weight_min),
        )
        index["roots"].append(
            {
                "root": root.id,
                "root_size": float(node_sizes.get(root, 1.0)),
                "file": str(out_path),
                **stats,
            }
        )
        if i % 50 == 0 or i == len(roots):
            print(f"[root-trees] rendered {i}/{len(roots)}")

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[root-trees] done ->", out_dir)


if __name__ == "__main__":
    main()

