from __future__ import annotations

"""
基于 core.py 的 2D 坐标 + 多分辨率主题建模结果的可视化脚本
=====================================================

你已有：
- out/umap2d.npy               （core.py 保存）
- out/graph_drl2d.npy          （core.py 保存，可选）
- out/leiden_sweep/membership_r*.npy
- out/topic_modeling_multi/K{K}/r{res}/communities_topic_weights.csv

本脚本会生成：
- 逐分辨率 UMAP 上按 Top1 Topic 染色的论文散点帧
- 逐分辨率 graph-layout 上按 Top1 Topic 染色的论文散点帧（若 graph 坐标存在）
- 逐分辨率“社区质心图”（质心位置=社区在2D中的中心，颜色=Top1 topic，大小=社区规模）
- 跨分辨率统计曲线（社区数、主题纯度、主题占比）
"""

from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Iterable

# 尽量避免 GUI backend / OpenMP 混用导致的原生崩溃（Windows 上更常见）
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# 不再依赖 topic_modeling / diagram2d，避免在可视化阶段引入重型依赖与 OpenMP 冲突
tm = None


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
MEM_RE = re.compile(r"membership_r([0-9]+(?:\.[0-9]+)?)\.npy$")


def parse_r_from_path(p: Path) -> float:
    m = MEM_RE.search(p.name)
    if not m:
        raise ValueError(f"not a membership file: {p}")
    return float(m.group(1))


def discover_memberships(leiden_dir: Path) -> list[tuple[float, Path]]:
    items = []
    for p in sorted(leiden_dir.glob("membership_r*.npy")):
        try:
            items.append((parse_r_from_path(p), p))
        except Exception:
            pass
    items.sort(key=lambda x: x[0])
    return items


def choose_memberships_exact(all_items: list[tuple[float, Path]], resolutions: Iterable[float] | None) -> list[tuple[float, Path]]:
    """显式指定分辨率时的精确匹配（容差匹配）。"""
    if resolutions is None:
        return all_items
    req = [float(x) for x in resolutions]
    out = []
    for rr in req:
        for r, p in all_items:
            if abs(r - rr) <= 1e-8:
                out.append((r, p))
                break
    # 去重排序
    uniq = {round(r, 10): (r, p) for r, p in out}
    return [uniq[k] for k in sorted(uniq)]


def choose_memberships_by_interval(
    all_items: list[tuple[float, Path]],
    r_min: float | None = 0.0001,
    r_max: float | None = 5.0,
    include: Iterable[float] | None = None,
    tol: float = 1e-8,
) -> list[tuple[float, Path]]:
    """按区间筛选目录里已有 membership 文件。

    与 topic_modeling_multi.py 保持一致：
    - 不按 step 重建理论分辨率网格
    - 直接扫描目录并筛选区间内已有文件
    """
    include = [float(x) for x in (include or [])]
    out = []
    for r, p in all_items:
        in_range = True
        if r_min is not None:
            in_range = in_range and (float(r) >= float(r_min) - tol)
        if r_max is not None:
            in_range = in_range and (float(r) <= float(r_max) + tol)
        in_include = any(abs(float(r) - x) <= tol for x in include)
        if in_range or in_include:
            out.append((r, p))
    uniq = {round(r, 10): (r, p) for r, p in out}
    return [uniq[k] for k in sorted(uniq)]


def get_topic_result_dir(topic_root: Path, r: float) -> Path:
    # 优先匹配 r1.0000 目录
    cands = [topic_root / f"r{r:.4f}", topic_root / f"r{r}"]
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def load_comm_topic_table(topic_dir: Path) -> pd.DataFrame:
    csv_path = topic_dir / "communities_topic_weights.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少文件: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def load_membership_local(mem_path: Path, n_papers_expected: int | None = None) -> np.ndarray:
    """轻量级 membership 读取（避免 topic_modeling 导入失败时无法运行）。"""
    arr = np.load(mem_path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    # Leiden membership 常见为整数；这里统一转 int32 供后续索引使用
    if arr.dtype.kind not in ("i", "u"):
        arr = arr.astype(np.int64)
    arr = arr.astype(np.int32, copy=False)
    if n_papers_expected is not None and arr.shape[0] != int(n_papers_expected):
        raise ValueError(f"membership 长度 {arr.shape[0]} != 期望 {int(n_papers_expected)}: {mem_path}")
    return arr


def load_membership_safe(mem_path: Path, n_papers_expected: int | None = None) -> np.ndarray:
    """优先使用 topic_modeling.load_membership；失败则回退本地实现。"""
    if tm is not None and hasattr(tm, "load_membership"):
        try:
            return tm.load_membership(mem_path, n_papers_expected=n_papers_expected)
        except Exception:
            # 回退轻量实现，增强健壮性
            pass
    return load_membership_local(mem_path, n_papers_expected=n_papers_expected)


def plot_scatter_safe(
    Y: np.ndarray,
    colors: np.ndarray,
    title: str,
    out_png: Path,
    point_size: float = 1.0,
    alpha: float = 0.8,
    max_points: int | None = None,
    verbose: bool = False,
) -> None:
    """纯 matplotlib 散点图，替代 diagram2d.plot_scatter，避免某些环境下的原生崩溃。"""
    Y = np.asarray(Y, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Y 形状应为 (N,2)，实际 {Y.shape}")
    if colors.ndim != 2 or colors.shape[1] != 4 or colors.shape[0] != Y.shape[0]:
        raise ValueError(f"colors 形状应为 (N,4) 且与 Y 对齐，实际 {colors.shape} vs {Y.shape}")

    n = Y.shape[0]
    if max_points is not None and max_points > 0 and n > int(max_points):
        rng = np.random.default_rng(42)  # 固定随机种子，保证可复现
        idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
        Yp = Y[idx]
        Cp = colors[idx]
        if verbose:
            print(f"[viz] sampled points: {len(idx)}/{n}")
    else:
        Yp = Y
        Cp = colors

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.gca()
    ax.scatter(
        Yp[:, 0],
        Yp[:, 1],
        s=float(point_size),
        c=Cp,
        linewidths=0,
        alpha=float(alpha),
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def build_discrete_topic_palette(k: int, cmap_name: str = "tab20") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    if k <= 20 and cmap_name == "tab20":
        colors = [cmap(i) for i in range(k)]
    else:
        colors = [cmap(0.0 if k <= 1 else i / (k - 1)) for i in range(k)]
    return np.asarray(colors, dtype=np.float32)


def colors_from_topics(topic_ids: np.ndarray, k: int, unknown_color=(0.7, 0.7, 0.7, 0.35)) -> np.ndarray:
    topic_ids = np.asarray(topic_ids, dtype=int)
    pal = build_discrete_topic_palette(k)
    rgba = np.zeros((topic_ids.shape[0], 4), dtype=np.float32)
    for i, t in enumerate(topic_ids.tolist()):
        if 0 <= t < k:
            rgba[i] = pal[t]
        else:
            rgba[i] = np.array(unknown_color, dtype=np.float32)
    return rgba


def map_papers_to_top1_topic(membership: np.ndarray, df_comm: pd.DataFrame, n_papers: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 paper_top1_topic（长度N）与 paper_top1_weight（长度N）。"""
    mem = tm.load_membership(Path("dummy"), n_papers_expected=n_papers) if False else membership.astype(int)
    if membership.shape[0] != n_papers:
        raise ValueError(f"membership 长度 {membership.shape[0]} != n_papers {n_papers}")

    dft = df_comm.copy()
    if "community_id" not in dft.columns or "top1_topic" not in dft.columns:
        raise KeyError("communities_topic_weights.csv 至少需要 community_id, top1_topic 两列")
    dft["community_id"] = pd.to_numeric(dft["community_id"], errors="coerce").astype("Int64")
    dft["top1_topic"] = pd.to_numeric(dft["top1_topic"], errors="coerce").astype("Int64")
    if "top1_weight" in dft.columns:
        dft["top1_weight"] = pd.to_numeric(dft["top1_weight"], errors="coerce")
    else:
        dft["top1_weight"] = np.nan

    c2t = {int(c): int(t) for c, t in zip(dft["community_id"], dft["top1_topic"]) if pd.notna(c) and pd.notna(t)}
    c2w = {int(c): float(w) for c, w in zip(dft["community_id"], dft["top1_weight"]) if pd.notna(c)}

    paper_topic = np.full(n_papers, -1, dtype=np.int32)
    paper_w1 = np.full(n_papers, np.nan, dtype=np.float32)
    for i in range(n_papers):
        c = int(mem[i])
        if c in c2t:
            paper_topic[i] = c2t[c]
        if c in c2w:
            paper_w1[i] = c2w[c]
    return paper_topic, paper_w1


def plot_topic_legend(out_png: Path, k: int, title: str = "Top1 Topic Legend") -> None:
    pal = build_discrete_topic_palette(k)
    ncols = 2 if k > 12 else 1
    nrows = int(np.ceil(k / ncols))
    fig_h = max(2.0, 0.35 * nrows + 0.8)
    plt.figure(figsize=(4.8 * ncols, fig_h), dpi=160)
    ax = plt.gca()
    ax.set_title(title)
    ax.axis("off")
    for t in range(k):
        col = t // nrows
        row = t % nrows
        x0 = col * 4.2
        y0 = nrows - row - 1
        ax.scatter([x0], [y0], s=60, c=[pal[t]], linewidths=0)
        ax.text(x0 + 0.25, y0, f"Topic {t}", va="center", fontsize=9)
    ax.set_xlim(-0.6, ncols * 4.2)
    ax.set_ylim(-0.8, nrows + 0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def community_centroid_plot(
    Y: np.ndarray,
    membership: np.ndarray,
    df_comm: pd.DataFrame,
    k: int,
    out_png: Path,
    title: str,
    alpha: float = 0.9,
    min_size: float = 6.0,
    size_scale: float = 1.4,
    annotate_top_n: int = 0,
) -> None:
    Y = np.asarray(Y, dtype=np.float32)
    n = Y.shape[0]
    if membership.shape[0] != n:
        raise ValueError("membership 长度与坐标点数不一致")

    # 统计社区质心和规模
    mem = membership.astype(int)
    uniq, inv, counts = np.unique(mem, return_inverse=True, return_counts=True)

    sum_xy = np.zeros((uniq.shape[0], 2), dtype=np.float64)
    np.add.at(sum_xy, inv, Y)
    cent = (sum_xy / counts[:, None]).astype(np.float32)

    # 社区 -> top1 topic / top1 weight
    dft = df_comm.copy()
    dft["community_id"] = pd.to_numeric(dft["community_id"], errors="coerce").astype("Int64")
    dft["top1_topic"] = pd.to_numeric(dft["top1_topic"], errors="coerce").astype("Int64")
    if "top1_weight" in dft.columns:
        dft["top1_weight"] = pd.to_numeric(dft["top1_weight"], errors="coerce")
    else:
        dft["top1_weight"] = np.nan
    c2t = {int(c): int(t) for c, t in zip(dft["community_id"], dft["top1_topic"]) if pd.notna(c) and pd.notna(t)}
    c2w = {int(c): float(w) for c, w in zip(dft["community_id"], dft["top1_weight"]) if pd.notna(c)}

    comm_topic = np.array([c2t.get(int(c), -1) for c in uniq], dtype=int)
    comm_w1 = np.array([c2w.get(int(c), np.nan) for c in uniq], dtype=float)
    colors = colors_from_topics(comm_topic, k)

    # 让“越纯”的社区越不透明（若 top1_weight 有值）
    if np.isfinite(comm_w1).any():
        a = np.nan_to_num(comm_w1, nan=0.35)
        a = np.clip(0.2 + 0.8 * a, 0.15, 1.0)
        colors[:, 3] = a.astype(np.float32)

    sizes = min_size + size_scale * np.sqrt(np.maximum(counts, 1)).astype(float)

    plt.figure(figsize=(10, 8), dpi=160)
    plt.scatter(cent[:, 0], cent[:, 1], s=sizes, c=colors, linewidths=0.2, edgecolors="black", alpha=alpha)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    if annotate_top_n > 0:
        idx_sort = np.argsort(counts)[::-1][:annotate_top_n]
        for ii in idx_sort:
            plt.text(cent[ii, 0], cent[ii, 1], str(int(uniq[ii])), fontsize=6)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_summary_curves(summary_df: pd.DataFrame, k: int, out_dir: Path) -> None:
    if summary_df.empty:
        return
    d = summary_df.sort_values("resolution").reset_index(drop=True)

    # 1) 社区数与纯度随分辨率变化
    if {"resolution", "n_communities", "mean_top1_weight"}.issubset(d.columns):
        plt.figure(figsize=(10, 6), dpi=160)
        ax1 = plt.gca()
        ax1.plot(d["resolution"], d["n_communities"], marker="o", linewidth=1)
        ax1.set_xlabel("Leiden resolution")
        ax1.set_ylabel("#communities")
        ax2 = ax1.twinx()
        ax2.plot(d["resolution"], d["mean_top1_weight"], marker="s", linewidth=1)
        ax2.set_ylabel("mean top1 weight")
        plt.title("Community count and topic purity vs resolution")
        plt.tight_layout()
        plt.savefig(out_dir / "curve_ncomm_purity_vs_resolution.png", bbox_inches="tight")
        plt.close()

    # 2) 主题占比曲线（按社区规模加权平均主题权重）
    topic_cols = [f"topic_weighted_mean_{i}" for i in range(k) if f"topic_weighted_mean_{i}" in d.columns]
    if topic_cols:
        plt.figure(figsize=(10, 6), dpi=160)
        x = d["resolution"].to_numpy(dtype=float)
        for col in topic_cols:
            plt.plot(x, d[col].to_numpy(dtype=float), linewidth=1, label=col.replace("topic_weighted_mean_", "T"))
        plt.xlabel("Leiden resolution")
        plt.ylabel("Weighted mean topic weight")
        plt.title("Topic prevalence across resolutions")
        if len(topic_cols) <= 15:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "curve_topic_prevalence_vs_resolution.png", bbox_inches="tight")
        plt.close()

    # 3) Top1 mass 曲线（硬划分视角）
    top1_cols = [f"topic_top1_mass_{i}" for i in range(k) if f"topic_top1_mass_{i}" in d.columns]
    if top1_cols:
        plt.figure(figsize=(10, 6), dpi=160)
        x = d["resolution"].to_numpy(dtype=float)
        for col in top1_cols:
            plt.plot(x, d[col].to_numpy(dtype=float), linewidth=1, label=col.replace("topic_top1_mass_", "T"))
        plt.xlabel("Leiden resolution")
        plt.ylabel("Share of papers (by top1 topic)")
        plt.title("Top1 topic mass across resolutions")
        if len(top1_cols) <= 15:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "curve_top1_mass_vs_resolution.png", bbox_inches="tight")
        plt.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多分辨率主题建模结果可视化（基于 core.py 的 2D 坐标）")
    p.add_argument("--k", type=int, required=True, help="全局主题数 K，与主题建模输出一致")
    p.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep"))
    p.add_argument("--topic-root", type=str, default=None,
                   help="主题建模多分辨率输出根目录；默认 out/topic_modeling_multi/K{K}")
    p.add_argument("--out-dir", type=str, default=None,
                   help="可视化输出目录；默认 out/topic_viz_multi/K{K}")
    p.add_argument("--umap", type=str, default=str(OUT_DIR / "umap2d.npy"))
    p.add_argument("--graph-layout", type=str, default=str(OUT_DIR / "graph_drl2d.npy"))
    p.add_argument("--resolutions", type=float, nargs="*", default=None, help="只画指定分辨率列表；提供后优先于范围模式")
    p.add_argument("--r-min", type=float, default=0.0001, help="范围模式下最小分辨率（默认 0.0001）")
    p.add_argument("--r-max", type=float, default=5.0, help="范围模式下最大分辨率（默认 5.0）")
    p.add_argument("--include", type=float, nargs="*", default=None, help="范围模式下额外强制包含的分辨率")
    p.add_argument("--step", type=float, default=None, help="兼容参数：范围模式不再用于匹配文件，仅用于记录")
    p.add_argument("--max-points", type=int, default=None, help="论文点过多时可随机采样")
    p.add_argument("--point-size", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--skip-graph", action="store_true", help="不画 graph layout 帧")
    p.add_argument("--community-centroid", action="store_true", help="同时输出社区质心图")
    p.add_argument("--annotate-top-n-communities", type=int, default=0)
    # segmented batch mode（读取 aligned_segmented/segments.csv，逐段调用本脚本生成帧图）
    p.add_argument("--batch-segments", action="store_true", help="按 segments.csv 批量生成所有 segment 的帧图（自动使用 segment_name / out_dir，避免 segment_id 跳号问题）")
    p.add_argument("--segments-csv", type=str, default=None, help="分段对齐输出的 segments.csv 路径；batch 模式下默认 <topic_root>/aligned_segmented/segments.csv")
    p.add_argument("--segmented-out-root", type=str, default=None, help="batch 模式下所有 segment 可视化输出根目录；默认 out/topic_viz_multi/K{K}_segmented")
    p.add_argument("--clean-segment-out", action="store_true", help="batch 模式下若 segment 可视化目录已存在则先删除")
    p.add_argument("--quiet", action="store_true")
    return p


def _run_batch_segments(args) -> None:
    """读取 segments.csv，逐段调用本脚本（单段模式）生成帧图。
    使用 segment_name + out_dir 列，避免 segment_id 跳号导致路径拼接错误。
    """
    verbose = not args.quiet
    base_topic_root = Path(args.topic_root) if args.topic_root else (OUT_DIR / "topic_modeling_multi" / f"K{args.k}")
    segments_csv = Path(args.segments_csv) if args.segments_csv else (base_topic_root / "aligned_segmented" / "segments.csv")
    if not segments_csv.exists():
        raise FileNotFoundError(f"segments.csv 不存在: {segments_csv}")

    seg_out_root = Path(args.segmented_out_root) if args.segmented_out_root else (OUT_DIR / "topic_viz_multi" / f"K{args.k}_segmented")
    seg_out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(segments_csv, encoding="utf-8-sig")
    need_cols = {"segment_name", "out_dir", "r_start", "r_end"}
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise KeyError(f"segments.csv 缺少列: {miss}（需要 {sorted(need_cols)}）")

    # 按 r_start 排序，避免依赖 segment_id 连续性
    df = df.sort_values(["r_start", "r_end", "segment_name"]).reset_index(drop=True)

    if verbose:
        print(f"[viz-batch] segments_csv={segments_csv}")
        print(f"[viz-batch] n_segments={len(df)}")
        print(f"[viz-batch] out_root={seg_out_root}")

    for _, row in df.iterrows():
        seg_name = str(row["segment_name"])
        seg_topic_root = Path(str(row["out_dir"]))
        r_min = float(row["r_start"])
        r_max = float(row["r_end"])
        out_dir = seg_out_root / seg_name

        if args.clean_segment_out and out_dir.exists():
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--k", str(args.k),
            "--leiden-dir", str(args.leiden_dir),
            "--topic-root", str(seg_topic_root),
            "--out-dir", str(out_dir),
            "--umap", str(args.umap),
            "--graph-layout", str(args.graph_layout),
            "--r-min", str(r_min),
            "--r-max", str(r_max),
        ]
        if args.include:
            cmd += ["--include", *[str(x) for x in args.include]]
        if args.step is not None:
            cmd += ["--step", str(args.step)]
        if args.max_points is not None:
            cmd += ["--max-points", str(args.max_points)]
        if args.point_size is not None:
            cmd += ["--point-size", str(args.point_size)]
        if args.alpha is not None:
            cmd += ["--alpha", str(args.alpha)]
        if args.skip_graph:
            cmd += ["--skip-graph"]
        if args.community_centroid:
            cmd += ["--community-centroid"]
        if int(args.annotate_top_n_communities or 0) > 0:
            cmd += ["--annotate-top-n-communities", str(int(args.annotate_top_n_communities))]
        if args.quiet:
            cmd += ["--quiet"]

        if verbose:
            print(f"[viz-batch] segment={seg_name} r=[{r_min:.4f}, {r_max:.4f}] topic_root={seg_topic_root}")

        subprocess.run(cmd, check=True)

    if verbose:
        print(f"[viz-batch] all segments done -> {seg_out_root}")



def main() -> None:
    args = build_argparser().parse_args()
    if args.batch_segments:
        _run_batch_segments(args)
        return
    verbose = not args.quiet

    leiden_dir = Path(args.leiden_dir)
    topic_root = Path(args.topic_root) if args.topic_root else (OUT_DIR / "topic_modeling_multi" / f"K{args.k}")
    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / "topic_viz_multi" / f"K{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not leiden_dir.exists():
        raise FileNotFoundError(f"leiden_dir 不存在: {leiden_dir}")
    if not topic_root.exists():
        raise FileNotFoundError(f"topic_root 不存在: {topic_root}")

    Y_umap = np.load(Path(args.umap)).astype(np.float32)
    Y_graph = None
    graph_path = Path(args.graph_layout)
    if (not args.skip_graph) and graph_path.exists():
        Y_graph = np.load(graph_path).astype(np.float32)

    all_items = discover_memberships(leiden_dir)
    if args.resolutions is not None and len(args.resolutions) > 0:
        items = choose_memberships_exact(all_items, args.resolutions)
    else:
        items = choose_memberships_by_interval(
            all_items,
            r_min=args.r_min,
            r_max=args.r_max,
            include=args.include,
        )
    if not items:
        raise RuntimeError("未找到可用 membership 文件")

    if verbose:
        print(f"[viz] discovered memberships={len(all_items)}")
        print(f"[viz] selected memberships ={len(items)}")
        print(f"[viz] topic_root={topic_root}")
        print(f"[viz] out_dir={out_dir}")
        print(f"[viz] n_points={Y_umap.shape[0]}")
        print(f"[viz] resolutions={', '.join(f'{r:.4f}' for r, _ in items)}")

    frames_umap = out_dir / "frames_topic_umap"
    frames_umap.mkdir(parents=True, exist_ok=True)
    frames_graph = out_dir / "frames_topic_graph"
    if Y_graph is not None:
        frames_graph.mkdir(parents=True, exist_ok=True)
    frames_comm_umap = out_dir / "frames_comm_centroid_umap"
    frames_comm_graph = out_dir / "frames_comm_centroid_graph"
    if args.community_centroid:
        frames_comm_umap.mkdir(parents=True, exist_ok=True)
        if Y_graph is not None:
            frames_comm_graph.mkdir(parents=True, exist_ok=True)

    plot_topic_legend(out_dir / "topic_legend.png", args.k)

    summary_rows = []
    for i, (r, mem_path) in enumerate(items):
        topic_dir = get_topic_result_dir(topic_root, r)
        if not topic_dir.exists():
            if verbose:
                print(f"[viz] skip r={r:.4f}: topic dir not found -> {topic_dir}")
            continue

        try:
            df_comm = load_comm_topic_table(topic_dir)
        except Exception as e:
            if verbose:
                print(f"[viz] skip r={r:.4f}: {e}")
            continue

        membership = load_membership_safe(mem_path, n_papers_expected=Y_umap.shape[0])
        if membership.shape[0] != Y_umap.shape[0]:
            raise ValueError(f"r={r:.4f}: membership 长度 {membership.shape[0]} != UMAP 点数 {Y_umap.shape[0]}")
        if Y_graph is not None and Y_graph.shape[0] != membership.shape[0]:
            raise ValueError(f"graph_layout 点数 {Y_graph.shape[0]} 与 membership 长度 {membership.shape[0]} 不一致")

        paper_topic, paper_w1 = map_papers_to_top1_topic(membership, df_comm, n_papers=membership.shape[0])
        colors = colors_from_topics(paper_topic, args.k)
        # 用社区的 top1_weight 调整透明度（“纯度”高更实）
        if np.isfinite(paper_w1).any():
            a = np.nan_to_num(paper_w1, nan=0.2)
            a = np.clip(0.15 + 0.85 * a, 0.1, 1.0)
            colors[:, 3] = a.astype(np.float32)

        n_comm = int(df_comm.shape[0])
        mean_top1 = float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").mean()) if "top1_weight" in df_comm.columns else float("nan")
        title_suffix = f"r={r:.4f} | communities={n_comm} | mean_top1={mean_top1:.3f}" if np.isfinite(mean_top1) else f"r={r:.4f} | communities={n_comm}"

        # --- 论文级散点（UMAP）---
        plot_scatter_safe(
            Y_umap,
            colors=colors,
            title=f"UMAP colored by community Top1 topic ({title_suffix})",
            out_png=frames_umap / f"frame_{i:04d}_r{r:.4f}.png",
            point_size=args.point_size,
            alpha=args.alpha,
            max_points=args.max_points,
            verbose=verbose,
        )

        # --- 论文级散点（Graph layout）---
        if Y_graph is not None:
            plot_scatter_safe(
                Y_graph,
                colors=colors,
                title=f"Graph layout colored by community Top1 topic ({title_suffix})",
                out_png=frames_graph / f"frame_{i:04d}_r{r:.4f}.png",
                point_size=args.point_size,
                alpha=args.alpha,
                max_points=args.max_points,
                verbose=verbose,
            )

        # --- 社区级质心图（可选）---
        if args.community_centroid:
            community_centroid_plot(
                Y_umap,
                membership,
                df_comm,
                args.k,
                out_png=frames_comm_umap / f"frame_{i:04d}_r{r:.4f}.png",
                title=f"UMAP community centroids by Top1 topic ({title_suffix})",
                annotate_top_n=args.annotate_top_n_communities,
            )
            if Y_graph is not None:
                community_centroid_plot(
                    Y_graph,
                    membership,
                    df_comm,
                    args.k,
                    out_png=frames_comm_graph / f"frame_{i:04d}_r{r:.4f}.png",
                    title=f"Graph community centroids by Top1 topic ({title_suffix})",
                    annotate_top_n=args.annotate_top_n_communities,
                )

        # 跨分辨率统计行
        row = {
            "resolution": float(r),
            "n_communities": int(n_comm),
            "mean_top1_weight": mean_top1,
            "median_top1_weight": float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").median()) if "top1_weight" in df_comm.columns else float("nan"),
            "mean_top1_minus_top2": float((pd.to_numeric(df_comm.get("top1_weight"), errors="coerce") - pd.to_numeric(df_comm.get("top2_weight"), errors="coerce")).mean()) if {"top1_weight", "top2_weight"}.issubset(df_comm.columns) else float("nan"),
        }
        # 主题占比（两种口径）
        n_papers_per_comm = pd.to_numeric(df_comm.get("n_papers", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
        n_papers_per_comm = np.maximum(n_papers_per_comm, 1.0)
        for t in range(args.k):
            c = f"topic_{t}"
            if c in df_comm.columns:
                vals = pd.to_numeric(df_comm[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                row[f"topic_weighted_mean_{t}"] = float(np.average(vals, weights=n_papers_per_comm))
            if "top1_topic" in df_comm.columns:
                t1 = pd.to_numeric(df_comm["top1_topic"], errors="coerce").fillna(-1).astype(int).to_numpy()
                row[f"topic_top1_mass_{t}"] = float(n_papers_per_comm[t1 == t].sum() / n_papers_per_comm.sum())

        summary_rows.append(row)

        if verbose:
            print(f"[viz] done r={r:.4f}")

    # summary & curves
    df_summary = pd.DataFrame(summary_rows).sort_values("resolution").reset_index(drop=True) if summary_rows else pd.DataFrame()
    df_summary.to_csv(out_dir / "summary_visualization_metrics.csv", index=False, encoding="utf-8-sig")
    plot_summary_curves(df_summary, args.k, out_dir)

    # 保存运行信息
    meta = {
        "k": int(args.k),
        "leiden_dir": str(leiden_dir),
        "topic_root": str(topic_root),
        "out_dir": str(out_dir),
        "n_resolutions_processed": int(len(df_summary)),
        "resolutions_processed": df_summary["resolution"].tolist() if not df_summary.empty else [],
        "umap": str(args.umap),
        "graph_layout": str(args.graph_layout) if Y_graph is not None else None,
        "community_centroid": bool(args.community_centroid),
        "range_mode": None if (args.resolutions is not None and len(args.resolutions) > 0) else {"r_min": args.r_min, "r_max": args.r_max, "include": args.include},
        "resolutions_arg": args.resolutions,
        "step_arg_compat": args.step,
        "max_points": args.max_points,
        "point_size": float(args.point_size),
        "alpha": float(args.alpha),
    }
    (out_dir / "viz_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[viz] outputs saved to:", out_dir)


if __name__ == "__main__":
    main()
