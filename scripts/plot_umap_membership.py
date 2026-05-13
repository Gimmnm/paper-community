#!/usr/bin/env python3
"""
离线可视化：在已有 out/umap2d.npy 上按某分辨率的 membership 着色导出 PNG。

不重新跑 sweep；membership 缺失时会按 summary.npy 取最近分辨率（与 demo 一致）。

用法（仓库根目录）：
  PYTHONPATH=src python scripts/plot_umap_membership.py --help

或直接（脚本会自动把 src/ 加入 sys.path）：
  python scripts/plot_umap_membership.py \\
    --umap-npy out/umap2d.npy \\
    --leiden-dir out/leiden_sweep_cpm \\
    --resolution 0.2 \\
    --out-png out/viz/umap_communities_cpm.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    repo = _bootstrap_path()

    p = argparse.ArgumentParser(description="Plot UMAP colored by community membership.")
    p.add_argument("--umap-npy", type=str, default=str(repo / "out" / "umap2d.npy"))
    p.add_argument("--leiden-dir", type=str, default=str(repo / "out" / "leiden_sweep_cpm"))
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--out-png", type=str, default=str(repo / "out" / "viz" / "umap_communities.png"))
    p.add_argument("--max-points", type=int, default=35_000, help="subsample for faster PNG; omit heavy overlap")
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    import numpy as np

    from app_layer.demo_graph import load_membership_for_resolution_light, resolve_membership_for_resolution_light
    from foundation_layer.diagram2d import plot_scatter

    umap_path = Path(args.umap_npy)
    leiden_dir = Path(args.leiden_dir)
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not umap_path.exists():
        raise SystemExit(f"missing UMAP coords: {umap_path}")

    Y = np.load(umap_path).astype(np.float32)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise SystemExit(f"expected (N,2) array in {umap_path}, got {Y.shape}")

    path_used, r_eff, _exact = resolve_membership_for_resolution_light(leiden_dir, float(args.resolution))
    labels = load_membership_for_resolution_light(leiden_dir, float(args.resolution))

    if int(labels.shape[0]) != int(Y.shape[0]):
        raise SystemExit(
            f"length mismatch: umap rows={Y.shape[0]} membership rows={labels.shape[0]} "
            f"(from {path_used}, r≈{r_eff:.4f})"
        )

    title = args.title or f"UMAP colored by communities (r={r_eff:.4f}, dir={leiden_dir.name})"
    mp = int(args.max_points) if args.max_points and args.max_points > 0 else None

    plot_scatter(
        Y,
        labels=labels,
        title=title,
        out_png=out_png,
        point_size=1.2,
        alpha=0.72,
        max_points=mp,
        verbose=True,
    )
    print(f"[plot_umap_membership] wrote {out_png}")


if __name__ == "__main__":
    main()
