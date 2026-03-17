# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

from checklist import compare_clusterings
from community import load_or_run_leiden_partition
from diagram2d import compute_xy_limits, labels_to_colors_by_centroid, plot_scatter
from network import build_or_load_mutual_knn_graph


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

def _to_builtin(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x



def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")



def _res_tag(resolution: float) -> str:
    s = f"{float(resolution):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def compact_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将任意整数标签压缩到 0..C-1，同时返回原始 unique label 映射。
    """
    uniq, inv = np.unique(np.asarray(labels, dtype=np.int32), return_inverse=True)
    return inv.astype(np.int32), uniq.astype(np.int32)



def build_igraph_from_edges(n_nodes: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> ig.Graph:
    g = ig.Graph(n=int(n_nodes), edges=list(zip(np.asarray(u).tolist(), np.asarray(v).tolist())), directed=False)
    g.es["weight"] = np.asarray(w, dtype=np.float32).astype(float).tolist()
    return g



def collect_time_info(
    papers: Sequence[object],
    *,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    预处理时间信息。

    返回：
      - years_by_pid: 1-based，长度=len(papers)
      - years_by_idx0: 0-based，长度=N
      - valid_mask_idx0: year>0
      - counts_by_year / min_year / max_year
    """
    n_total = len(papers)
    years_by_pid = np.zeros(n_total, dtype=np.int32)
    start_idx = 1 if (n_total > 0 and papers[0] is None) else 0

    for pid in range(start_idx, n_total):
        p = papers[pid]
        years_by_pid[pid] = int(getattr(p, "year", 0) or 0)

    years_by_idx0 = years_by_pid[start_idx:].copy()
    valid_mask_idx0 = years_by_idx0 > 0
    valid_years = years_by_idx0[valid_mask_idx0]

    if valid_years.size == 0:
        raise ValueError("no valid years found in papers")

    uniq, cnt = np.unique(valid_years, return_counts=True)
    counts_by_year = {int(y): int(c) for y, c in zip(uniq, cnt)}

    result = {
        "indexing_start": start_idx,
        "years_by_pid": years_by_pid,
        "years_by_idx0": years_by_idx0,
        "valid_mask_idx0": valid_mask_idx0,
        "min_year": int(valid_years.min()),
        "max_year": int(valid_years.max()),
        "n_valid": int(valid_years.size),
        "n_unknown": int(np.sum(~valid_mask_idx0)),
        "counts_by_year": counts_by_year,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "time_info.npz",
            years_by_pid=years_by_pid,
            years_by_idx0=years_by_idx0,
            valid_mask_idx0=valid_mask_idx0.astype(np.int8),
            min_year=np.int32(result["min_year"]),
            max_year=np.int32(result["max_year"]),
        )
        _save_json(
            out_dir / "time_info_summary.json",
            {
                "min_year": result["min_year"],
                "max_year": result["max_year"],
                "n_valid": result["n_valid"],
                "n_unknown": result["n_unknown"],
                "counts_by_year": counts_by_year,
            },
        )
        if verbose:
            print(f"[time] saved time info -> {out_dir}")

    return result



def list_sliding_windows(
    min_year: int,
    max_year: int,
    *,
    window_size: int = 5,
    step: int = 1,
) -> List[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    windows: List[Tuple[int, int]] = []
    for start in range(int(min_year), int(max_year) - int(window_size) + 2, int(step)):
        end = start + int(window_size) - 1
        windows.append((start, end))
    return windows



def window_indices_from_years(
    years_by_idx0: np.ndarray,
    start_year: int,
    end_year: int,
    *,
    include_unknown: bool = False,
) -> np.ndarray:
    years = np.asarray(years_by_idx0, dtype=np.int32)
    if include_unknown:
        mask = ((years >= int(start_year)) & (years <= int(end_year))) | (years <= 0)
    else:
        mask = (years >= int(start_year)) & (years <= int(end_year))
    return np.where(mask)[0].astype(np.int32)



def induced_subgraph_from_full_edges(
    n_nodes_full: int,
    selected_idx0: np.ndarray,
    full_u: np.ndarray,
    full_v: np.ndarray,
    full_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从全图边表中抽出选定窗口的 induced subgraph。

    返回：
      - local_u, local_v, local_w: 局部 0-based 边表
      - old_to_local: 长度 n_nodes_full 的映射，未选中位置为 -1
    """
    selected_idx0 = np.asarray(selected_idx0, dtype=np.int32)
    full_u = np.asarray(full_u, dtype=np.int32)
    full_v = np.asarray(full_v, dtype=np.int32)
    full_w = np.asarray(full_w, dtype=np.float32)

    node_mask = np.zeros(int(n_nodes_full), dtype=np.bool_)
    node_mask[selected_idx0] = True

    keep = node_mask[full_u] & node_mask[full_v]

    old_to_local = -np.ones(int(n_nodes_full), dtype=np.int32)
    old_to_local[selected_idx0] = np.arange(selected_idx0.shape[0], dtype=np.int32)

    local_u = old_to_local[full_u[keep]]
    local_v = old_to_local[full_v[keep]]
    local_w = full_w[keep].astype(np.float32)
    return local_u, local_v, local_w, old_to_local



def community_structure_from_edges(
    labels: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
    local_w: np.ndarray,
) -> Dict[str, Any]:
    """
    根据窗口内的边和标签，聚合出“社区结构”：
      - 每个社区的点数
      - 社区间边权总和（无向）
      - 社区内部边权总和
    """
    labels = np.asarray(labels, dtype=np.int32)
    local_u = np.asarray(local_u, dtype=np.int32)
    local_v = np.asarray(local_v, dtype=np.int32)
    local_w = np.asarray(local_w, dtype=np.float32)

    uniq, counts = np.unique(labels, return_counts=True)
    label_to_pos = {int(c): i for i, c in enumerate(uniq.tolist())}

    agg: Dict[Tuple[int, int], float] = {}
    for a, b, w in zip(labels[local_u], labels[local_v], local_w):
        ia = label_to_pos[int(a)]
        ib = label_to_pos[int(b)]
        if ia <= ib:
            key = (ia, ib)
        else:
            key = (ib, ia)
        agg[key] = agg.get(key, 0.0) + float(w)

    if agg:
        cu = np.array([k[0] for k in agg.keys()], dtype=np.int32)
        cv = np.array([k[1] for k in agg.keys()], dtype=np.int32)
        cw = np.array([agg[k] for k in agg.keys()], dtype=np.float32)
    else:
        cu = np.zeros(0, dtype=np.int32)
        cv = np.zeros(0, dtype=np.int32)
        cw = np.zeros(0, dtype=np.float32)

    internal_weight = float(sum(w for (a, b), w in agg.items() if a == b))
    external_weight = float(sum(w for (a, b), w in agg.items() if a != b))

    return {
        "community_ids": uniq.astype(np.int32),
        "community_sizes": counts.astype(np.int32),
        "comm_u": cu,
        "comm_v": cv,
        "comm_w": cw,
        "n_comm": int(len(uniq)),
        "internal_weight": internal_weight,
        "external_weight": external_weight,
    }



def _save_window_result_npz(
    path: Path,
    *,
    pids: np.ndarray,
    idx0: np.ndarray,
    years: np.ndarray,
    labels_original: np.ndarray,
    labels_compact_arr: np.ndarray,
    label_vocab: np.ndarray,
    local_u: np.ndarray,
    local_v: np.ndarray,
    local_w: np.ndarray,
    structure: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pids=np.asarray(pids, dtype=np.int32),
        idx0=np.asarray(idx0, dtype=np.int32),
        years=np.asarray(years, dtype=np.int32),
        labels_original=np.asarray(labels_original, dtype=np.int32),
        labels_compact=np.asarray(labels_compact_arr, dtype=np.int32),
        label_vocab=np.asarray(label_vocab, dtype=np.int32),
        local_u=np.asarray(local_u, dtype=np.int32),
        local_v=np.asarray(local_v, dtype=np.int32),
        local_w=np.asarray(local_w, dtype=np.float32),
        community_ids=np.asarray(structure["community_ids"], dtype=np.int32),
        community_sizes=np.asarray(structure["community_sizes"], dtype=np.int32),
        comm_u=np.asarray(structure["comm_u"], dtype=np.int32),
        comm_v=np.asarray(structure["comm_v"], dtype=np.int32),
        comm_w=np.asarray(structure["comm_w"], dtype=np.float32),
        n_comm=np.int32(structure["n_comm"]),
        internal_weight=np.float32(structure["internal_weight"]),
        external_weight=np.float32(structure["external_weight"]),
    )



def _plot_dual_window_compare(
    Y_window: np.ndarray,
    labels_left: np.ndarray,
    labels_right: np.ndarray,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    left_title: str,
    right_title: str,
    suptitle: str,
    out_png: Path,
    point_size: float = 3.0,
    alpha: float = 0.85,
    verbose: bool = True,
) -> None:
    colors_left = labels_to_colors_by_centroid(Y_window, labels_left)
    colors_right = labels_to_colors_by_centroid(Y_window, labels_right)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    panels = [
        (axes[0], colors_left, left_title),
        (axes[1], colors_right, right_title),
    ]

    for ax, colors, title in panels:
        ax.scatter(
            Y_window[:, 0],
            Y_window[:, 1],
            s=point_size,
            alpha=alpha,
            c=colors,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(suptitle)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    if verbose:
        print(f"[time] saved comparison figure -> {out_png}")



def _write_video_from_frames(
    frame_paths: Sequence[Path],
    out_video: Path,
    *,
    fps: int = 2,
    verbose: bool = True,
) -> Path:
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if len(frame_paths) == 0:
        raise ValueError("frame_paths is empty")

    import imageio.v2 as imageio

    frames = [imageio.imread(p) for p in frame_paths]
    max_h = max(int(im.shape[0]) for im in frames)
    max_w = max(int(im.shape[1]) for im in frames)
    # H.264 + yuv420p 通常要求偶数尺寸；这里统一补白并对齐到偶数
    target_h = (max_h + 1) // 2 * 2
    target_w = (max_w + 1) // 2 * 2

    norm_dir = out_video.parent / "_video_frames"
    norm_dir.mkdir(parents=True, exist_ok=True)

    norm_paths: List[Path] = []
    for i, im in enumerate(frames):
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        if im.shape[2] == 4:
            im = im[:, :, :3]

        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        h, w = int(im.shape[0]), int(im.shape[1])
        y0 = (target_h - h) // 2
        x0 = (target_w - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = im[:, :, :3]

        fp = norm_dir / f"f_{i:04d}.png"
        imageio.imwrite(fp, canvas)
        norm_paths.append(fp)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(norm_dir / "f_%04d.png"),
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_video),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if verbose:
                print(f"[time] saved video -> {out_video}")
            return out_video
        except subprocess.CalledProcessError as e:
            if verbose:
                err = e.stderr.decode("utf-8", errors="ignore")[:500]
                print(f"[time] ffmpeg encode failed, fallback to gif: {err}")

    gif_path = out_video.with_suffix(".gif")
    imageio.mimsave(gif_path, [imageio.imread(p) for p in norm_paths], fps=fps)
    if verbose:
        print(f"[time] saved gif -> {gif_path}")
    return gif_path


# -----------------------------------------------------------------------------
# 主流程：单个时间窗
# -----------------------------------------------------------------------------

def analyze_time_window(
    papers: Sequence[object],
    embs: np.ndarray,
    Y: np.ndarray,
    full_u: np.ndarray,
    full_v: np.ndarray,
    full_w: np.ndarray,
    *,
    start_year: int,
    end_year: int,
    resolution: float,
    out_dir: Path,
    global_membership: Optional[np.ndarray] = None,
    global_graph: Optional[ig.Graph] = None,
    global_leiden_dir: Optional[Path] = None,
    time_info: Optional[Dict[str, Any]] = None,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    normalize: bool = True,
    seed: int = 42,
    include_unknown_year: bool = False,
    point_size: float = 3.0,
    alpha: float = 0.85,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    单个时间窗的完整分析：
      0) 时间预处理与索引
      1) 继承全图社区，抽 induced subgraph，存储并可视化
      2) 仅保留窗口内论文，重建图、重跑 Leiden，存储并可视化
      3) 对比两套分类结果
      4) 额外输出 side-by-side 对比图（供后续动画复用）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(embs[1:], dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    n_full = int(X.shape[0])
    if Y.shape[0] != n_full:
        raise ValueError(f"Y rows {Y.shape[0]} != X rows {n_full}")

    if time_info is None:
        time_info = collect_time_info(papers, out_dir=out_dir / "time_preprocess", verbose=verbose)
    years_by_idx0 = np.asarray(time_info["years_by_idx0"], dtype=np.int32)

    window_idx0 = window_indices_from_years(
        years_by_idx0,
        start_year,
        end_year,
        include_unknown=include_unknown_year,
    )
    if window_idx0.size == 0:
        raise ValueError(f"empty time window: [{start_year}, {end_year}]")

    window_pids = window_idx0 + 1
    window_years = years_by_idx0[window_idx0]
    Y_window = Y[window_idx0]
    X_window = X[window_idx0]

    xlim, ylim = compute_xy_limits(Y)

    window_dir = out_dir / f"w_{int(start_year)}_{int(end_year)}_r{_res_tag(resolution)}"
    inherited_dir = window_dir / "inherited"
    refit_dir = window_dir / "refit"
    compare_dir = window_dir / "compare"
    inherited_dir.mkdir(parents=True, exist_ok=True)
    refit_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 路线 1：继承全图社区
    # ------------------------------------------------------------------
    if global_membership is None:
        if global_graph is None or global_leiden_dir is None:
            raise ValueError(
                "global_membership is None, so both global_graph and global_leiden_dir are required"
            )
        global_result = load_or_run_leiden_partition(
            global_graph,
            out_dir=global_leiden_dir,
            resolution=resolution,
            seed=seed,
            verbose=verbose,
        )
        global_membership = np.asarray(global_result["membership"], dtype=np.int32)
    else:
        global_membership = np.asarray(global_membership, dtype=np.int32)

    if global_membership.shape[0] != n_full:
        raise ValueError(f"global_membership length {global_membership.shape[0]} != {n_full}")

    inherited_labels_original = global_membership[window_idx0]
    inherited_labels_compact, inherited_vocab = compact_labels(inherited_labels_original)

    inh_u, inh_v, inh_w, old_to_local = induced_subgraph_from_full_edges(
        n_nodes_full=n_full,
        selected_idx0=window_idx0,
        full_u=full_u,
        full_v=full_v,
        full_w=full_w,
    )
    inherited_structure = community_structure_from_edges(
        inherited_labels_original,
        inh_u,
        inh_v,
        inh_w,
    )

    _save_window_result_npz(
        inherited_dir / "window_inherited_structure.npz",
        pids=window_pids,
        idx0=window_idx0,
        years=window_years,
        labels_original=inherited_labels_original,
        labels_compact_arr=inherited_labels_compact,
        label_vocab=inherited_vocab,
        local_u=inh_u,
        local_v=inh_v,
        local_w=inh_w,
        structure=inherited_structure,
    )
    inherited_summary = {
        "mode": "inherited",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "resolution": float(resolution),
        "n_papers": int(window_idx0.size),
        "n_edges": int(inh_u.shape[0]),
        "n_comm": int(inherited_structure["n_comm"]),
        "internal_weight": float(inherited_structure["internal_weight"]),
        "external_weight": float(inherited_structure["external_weight"]),
        "label_vocab": inherited_vocab.tolist(),
    }
    _save_json(inherited_dir / "summary.json", inherited_summary)

    plot_scatter(
        Y_window,
        labels=inherited_labels_compact,
        title=f"Inherited communities | years=[{start_year}, {end_year}] | r={resolution:.4f}",
        out_png=inherited_dir / "scatter_inherited.png",
        point_size=point_size,
        alpha=alpha,
        xlim=xlim,
        ylim=ylim,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 路线 2：窗口内重建图、重跑 Leiden
    # ------------------------------------------------------------------
    refit_graph_cache = refit_dir / f"mutual_knn_k{k}.npz"
    A_win, (refu, refv, refw) = build_or_load_mutual_knn_graph(
        X_window,
        k=int(k),
        cache_npz=refit_graph_cache,
        knn_backend=knn_backend,
        knn_batch_size=int(knn_batch_size),
        normalize=normalize,
        verbose=verbose,
    )
    _ = A_win  # 这里只保留变量，方便你后续如果想扩展别的统计
    G_window = build_igraph_from_edges(len(window_idx0), refu, refv, refw)
    refit_result = load_or_run_leiden_partition(
        G_window,
        out_dir=refit_dir,
        resolution=resolution,
        seed=seed,
        verbose=verbose,
    )
    refit_labels_original = np.asarray(refit_result["membership"], dtype=np.int32)
    refit_labels_compact, refit_vocab = compact_labels(refit_labels_original)
    refit_structure = community_structure_from_edges(
        refit_labels_original,
        refu,
        refv,
        refw,
    )

    _save_window_result_npz(
        refit_dir / "window_refit_structure.npz",
        pids=window_pids,
        idx0=window_idx0,
        years=window_years,
        labels_original=refit_labels_original,
        labels_compact_arr=refit_labels_compact,
        label_vocab=refit_vocab,
        local_u=refu,
        local_v=refv,
        local_w=refw,
        structure=refit_structure,
    )
    refit_summary = {
        "mode": "refit",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "resolution": float(resolution),
        "n_papers": int(window_idx0.size),
        "n_edges": int(refu.shape[0]),
        "n_comm": int(refit_structure["n_comm"]),
        "internal_weight": float(refit_structure["internal_weight"]),
        "external_weight": float(refit_structure["external_weight"]),
        "label_vocab": refit_vocab.tolist(),
    }
    _save_json(refit_dir / "summary.json", refit_summary)

    plot_scatter(
        Y_window,
        labels=refit_labels_compact,
        title=f"Refit within window | years=[{start_year}, {end_year}] | r={resolution:.4f}",
        out_png=refit_dir / "scatter_refit.png",
        point_size=point_size,
        alpha=alpha,
        xlim=xlim,
        ylim=ylim,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 路线 3：对比两种分类
    # ------------------------------------------------------------------
    compare_result = compare_clusterings(
        inherited_labels_original,
        refit_labels_original,
        name_a="inherited",
        name_b="refit",
        pids=window_pids,
        write_report_path=str(compare_dir / "cluster_compare.txt"),
    )
    compare_summary = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "resolution": float(resolution),
        "n_papers": int(window_idx0.size),
        "ari": compare_result["ari"],
        "nmi": compare_result["nmi"],
        "direct_equal_rate": compare_result["direct_equal_rate"],
        "best_alignment_acc": compare_result["best_alignment_acc"],
        "purity_inherited_to_refit": compare_result["purity_a_to_b"],
        "purity_refit_to_inherited": compare_result["purity_b_to_a"],
        "top_matches": compare_result["top_matches"],
    }
    _save_json(compare_dir / "summary.json", compare_summary)

    _plot_dual_window_compare(
        Y_window,
        inherited_labels_compact,
        refit_labels_compact,
        xlim=xlim,
        ylim=ylim,
        left_title=f"Inherited | n_comm={inherited_summary['n_comm']}",
        right_title=f"Refit | n_comm={refit_summary['n_comm']}",
        suptitle=f"Window [{start_year}, {end_year}] | resolution={resolution:.4f}",
        out_png=compare_dir / "compare_scatter.png",
        point_size=point_size,
        alpha=alpha,
        verbose=verbose,
    )

    final_summary = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "resolution": float(resolution),
        "n_papers": int(window_idx0.size),
        "window_dir": str(window_dir),
        "inherited": inherited_summary,
        "refit": refit_summary,
        "compare": compare_summary,
    }
    _save_json(window_dir / "summary.json", final_summary)

    return {
        "window_idx0": window_idx0,
        "window_pids": window_pids,
        "window_years": window_years,
        "Y_window": Y_window,
        "xlim": xlim,
        "ylim": ylim,
        "inherited_labels": inherited_labels_original,
        "refit_labels": refit_labels_original,
        "compare": compare_result,
        "window_dir": window_dir,
        "compare_png": compare_dir / "compare_scatter.png",
        "summary": final_summary,
        "old_to_local": old_to_local,
    }


# -----------------------------------------------------------------------------
# 动态可视化：5 年窗口、1 年步长
# -----------------------------------------------------------------------------

def make_sliding_window_video(
    papers: Sequence[object],
    embs: np.ndarray,
    Y: np.ndarray,
    full_u: np.ndarray,
    full_v: np.ndarray,
    full_w: np.ndarray,
    *,
    resolution: float,
    out_dir: Path,
    global_membership: Optional[np.ndarray] = None,
    global_graph: Optional[ig.Graph] = None,
    global_leiden_dir: Optional[Path] = None,
    time_info: Optional[Dict[str, Any]] = None,
    window_size: int = 5,
    step: int = 1,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    normalize: bool = True,
    seed: int = 42,
    fps: int = 2,
    point_size: float = 3.0,
    alpha: float = 0.85,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    生成滑动窗口动态可视化：
      - 默认 5 年窗口、1 年步长
      - 每一帧都是 inherited / refit 的并排对比图
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if time_info is None:
        time_info = collect_time_info(papers, out_dir=out_dir / "time_preprocess", verbose=verbose)

    windows = list_sliding_windows(
        time_info["min_year"],
        time_info["max_year"],
        window_size=window_size,
        step=step,
    )
    if not windows:
        raise ValueError("no sliding windows generated")

    windows_dir = out_dir / "windows"
    frames_dir = out_dir / "frames"
    windows_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: List[Path] = []
    manifest: List[Dict[str, Any]] = []

    for i, (start_year, end_year) in enumerate(windows):
        result = analyze_time_window(
            papers,
            embs,
            Y,
            full_u,
            full_v,
            full_w,
            start_year=start_year,
            end_year=end_year,
            resolution=resolution,
            out_dir=windows_dir,
            global_membership=global_membership,
            global_graph=global_graph,
            global_leiden_dir=global_leiden_dir,
            time_info=time_info,
            k=k,
            knn_backend=knn_backend,
            knn_batch_size=knn_batch_size,
            normalize=normalize,
            seed=seed,
            point_size=point_size,
            alpha=alpha,
            verbose=verbose,
        )

        src = Path(result["compare_png"])
        dst = frames_dir / f"frame_{i:04d}.png"
        shutil.copyfile(src, dst)
        frame_paths.append(dst)
        manifest.append(
            {
                "frame_index": i,
                "start_year": start_year,
                "end_year": end_year,
                "window_dir": str(result["window_dir"]),
                "frame_png": str(dst),
            }
        )

    _save_json(frames_dir / "manifest.json", {"frames": manifest})
    step_tag = "" if int(step) == 1 else f"_s{int(step)}"
    out_video = out_dir / f"tw_{int(window_size)}y_r{_res_tag(resolution)}{step_tag}.mp4"
    actual_video = _write_video_from_frames(frame_paths, out_video, fps=fps, verbose=verbose)

    summary = {
        "window_size": int(window_size),
        "step": int(step),
        "resolution": float(resolution),
        "n_frames": len(frame_paths),
        "video_path": str(actual_video),
        "frames_dir": str(frames_dir),
    }
    _save_json(out_dir / "video_summary.json", summary)
    return summary
