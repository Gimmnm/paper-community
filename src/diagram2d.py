# 2ddiagram.py
# -*- coding: utf-8 -*-

from __future__ import annotations  # 允许前向引用类型标注

# pathlib.Path：缓存坐标与图片输出路径
from pathlib import Path

# time：计时、进度
import time

# typing：类型标注
from typing import Literal, Optional, Tuple

# numpy：矩阵运算、保存/读取 .npy
import numpy as np

# matplotlib：绘制散点图与保存 PNG
import matplotlib.pyplot as plt


def _ensure_float32_contiguous(X: np.ndarray) -> np.ndarray:
    """保证输入是 float32 且连续内存（很多库要求）。"""
    return np.asarray(X, dtype=np.float32, order="C")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """行 L2 归一化，用于 cosine 几何。"""
    X = _ensure_float32_contiguous(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def embed_2d(
    X: np.ndarray,
    *,
    method: Literal["pca", "umap", "pacmap"] = "umap",
    normalize: bool = True,
    pca_dim: int = 50,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    random_state: int = 42,
    cache_npy: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    将 (N, D) embedding 降到 (N, 2)：
      - method='pca'：纯 PCA（最快、最稳的全局线性基线）
      - method='umap'：PCA(pca_dim) -> UMAP（局部结构强，工程常用）
      - method='pacmap'：PaCMAP（更偏全局+局部折中，需 pacmap 库）

    cache：
      - 如果 cache_npy 存在则直接加载并返回
      - 否则计算并保存到 cache_npy
    """
    if cache_npy is not None and Path(cache_npy).exists():
        if verbose:
            print(f"[2d] loading cached embedding -> {cache_npy}")
        return np.load(cache_npy).astype(np.float32)

    t0 = time.time()

    X = _ensure_float32_contiguous(X)
    if normalize:
        X = l2_normalize_rows(X)

    if method == "pca":
        try:
            from sklearn.decomposition import PCA  # sklearn PCA：基线降维
        except Exception as e:
            raise ImportError("sklearn 不可用：pip install scikit-learn") from e

        pca = PCA(n_components=2, random_state=random_state)
        Y = pca.fit_transform(X).astype(np.float32)

    elif method == "umap":
        try:
            from sklearn.decomposition import PCA  # PCA 预处理：更快更稳
        except Exception as e:
            raise ImportError("sklearn 不可用：pip install scikit-learn") from e

        try:
            import umap  # umap-learn：UMAP 降维
        except Exception as e:
            raise ImportError("umap 不可用：pip install umap-learn；或 method='pca'") from e

        # PCA 到 50 维（推荐）
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X50 = pca.fit_transform(X).astype(np.float32)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(umap_neighbors),
            min_dist=float(umap_min_dist),
            metric=str(umap_metric),
            random_state=int(random_state),
            n_jobs=-1,
        )
        Y = reducer.fit_transform(X50).astype(np.float32)

    elif method == "pacmap":
        try:
            import pacmap  # PaCMAP：更强调全局结构
        except Exception as e:
            raise ImportError("pacmap 不可用：pip install pacmap；或 method='umap'/'pca'") from e

        # PaCMAP 通常直接吃原始向量即可（也可以 PCA 预处理）
        reducer = pacmap.PaCMAP(n_components=2, random_state=random_state)
        Y = reducer.fit_transform(X).astype(np.float32)

    else:
        raise ValueError(f"unknown method={method}")

    if verbose:
        print(f"[2d] method={method} done, Y shape={Y.shape}, time={time.time()-t0:.1f}s")

    if cache_npy is not None:
        Path(cache_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_npy, Y)
        if verbose:
            print(f"[2d] saved -> {cache_npy}")

    return Y


def labels_to_colors_by_centroid(
    Y: np.ndarray,
    labels: np.ndarray,
    *,
    cmap_name: str = "viridis",
) -> np.ndarray:
    """
    把社区 label 映射为“空间相近 -> 颜色相近”的渐变色：
      1) 计算每个社区在 2D 坐标 Y 中的 centroid
      2) 按 centroid 的 x 坐标排序社区
      3) 按排序用连续 colormap 分配颜色

    返回：
      - colors: (N,4) RGBA float
    """
    Y = np.asarray(Y, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    N = Y.shape[0]
    assert labels.shape[0] == N

    uniq = np.unique(labels)
    # 计算 centroid
    centroids = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        centroids[int(c)] = Y[idx].mean(axis=0)

    # 按 centroid.x 排序
    ordered = sorted(uniq.tolist(), key=lambda c: float(centroids[int(c)][0]))

    # colormap
    cmap = plt.get_cmap(cmap_name)
    m = len(ordered)
    c2color = {}
    for i, c in enumerate(ordered):
        t = 0.0 if m <= 1 else (i / (m - 1))
        c2color[int(c)] = cmap(t)

    colors = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        colors[i] = np.array(c2color[int(labels[i])], dtype=np.float32)

    return colors


def plot_scatter(
    Y: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: str = "",
    out_png: Optional[Path] = None,
    point_size: float = 1.0,
    alpha: float = 0.7,
    max_points: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> None:
    """
    画 2D 散点图：
      - 如果提供 labels：自动用 centroid 渐变色上色
      - 如果提供 colors：直接用 colors（RGBA）画
      - max_points：如果你担心画 83k 太慢/太大，可以随机采样一部分点
    """
    Y = np.asarray(Y, dtype=np.float32)
    N = Y.shape[0]

    if labels is not None and colors is None:
        colors = labels_to_colors_by_centroid(Y, np.asarray(labels, dtype=np.int32))

    if max_points is not None and max_points < N:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=int(max_points), replace=False)
        Yp = Y[idx]
        Cp = colors[idx] if colors is not None else None
    else:
        Yp = Y
        Cp = colors

    t0 = time.time()
    plt.figure(figsize=(10, 8), dpi=160)

    if Cp is not None:
        plt.scatter(Yp[:, 0], Yp[:, 1], s=point_size, alpha=alpha, c=Cp, linewidths=0, rasterized=True)
    else:
        plt.scatter(Yp[:, 0], Yp[:, 1], s=point_size, alpha=alpha, linewidths=0, rasterized=True)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")
        if verbose:
            print(f"[plot] saved -> {out_png}")

    plt.close()

    if verbose:
        print(f"[plot] done in {time.time()-t0:.1f}s (N={Yp.shape[0]})")


def graph_layout_2d(
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    method: Literal["drl", "fr"] = "drl",
    random_state: int = 42,
    init_xy: Optional[np.ndarray] = None,      # <<< 新增：初始坐标 (n,2)，推荐传 UMAP
    cache_npy: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    从“网络图”本身生成 2D layout。

    重要：
      igraph 的 seed 参数是“初始坐标矩阵”，不是随机种子整数。
      所以这里我们：
        - 如果 init_xy 给了：用它做 seed（强烈推荐：传 UMAP 2D）
        - 否则：自己生成一个随机 (n,2) 当 seed

    cache：
      - cache_npy 存在就直接加载
      - 否则计算并保存
    """
    if cache_npy is not None and Path(cache_npy).exists():
        if verbose:
            print(f"[layout] loading cached layout -> {cache_npy}")
        return np.load(cache_npy).astype(np.float32)

    try:
        import igraph as ig
    except Exception as e:
        raise ImportError("python-igraph 不可用：pip install python-igraph") from e

    t0 = time.time()

    # edges：确保是 0-based 节点编号；如果你内部是 1-based，需要先 -1
    edges = list(zip(u.tolist(), v.tolist()))
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es["weight"] = w.astype(np.float32).tolist()

    # ----- 构造 seed：必须是 (n,2) -----
    if init_xy is None:
        rng = np.random.default_rng(int(random_state))
        seed_xy = rng.normal(size=(n_nodes, 2)).astype(np.float32)
    else:
        seed_xy = np.asarray(init_xy, dtype=np.float32)
        if seed_xy.shape != (n_nodes, 2):
            raise ValueError(f"init_xy shape {seed_xy.shape} != ({n_nodes}, 2)")

    seed_list = seed_xy.tolist()  # igraph 更稳的输入形式

    if method == "drl":
        layout = g.layout_drl(weights="weight", seed=seed_list)
    elif method == "fr":
        layout = g.layout_fruchterman_reingold(weights="weight", seed=seed_list)
    else:
        raise ValueError(f"unknown method={method}")

    Y = np.asarray(layout.coords, dtype=np.float32)

    if verbose:
        print(f"[layout] method={method} done, Y shape={Y.shape}, time={time.time()-t0:.1f}s")

    if cache_npy is not None:
        Path(cache_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_npy, Y)
        if verbose:
            print(f"[layout] saved -> {cache_npy}")

    return Y