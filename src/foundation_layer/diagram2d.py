# 2ddiagram.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import time
from typing import Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ensure_float32_contiguous(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32, order="C")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
            from sklearn.decomposition import PCA
        except Exception as e:
            raise ImportError("sklearn 不可用：pip install scikit-learn") from e
        pca = PCA(n_components=2, random_state=random_state)
        Y = pca.fit_transform(X).astype(np.float32)

    elif method == "umap":
        try:
            from sklearn.decomposition import PCA
        except Exception as e:
            raise ImportError("sklearn 不可用：pip install scikit-learn") from e
        try:
            import umap
        except Exception as e:
            raise ImportError("umap 不可用：pip install umap-learn；或 method='pca'") from e
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
            import pacmap
        except Exception as e:
            raise ImportError("pacmap 不可用：pip install pacmap；或 method='umap'/'pca'") from e
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
    Y = np.asarray(Y, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    N = Y.shape[0]
    assert labels.shape[0] == N

    uniq = np.unique(labels)
    centroids = {}
    for c in uniq:
        idx = np.where(labels == c)[0]
        centroids[int(c)] = Y[idx].mean(axis=0)
    ordered = sorted(uniq.tolist(), key=lambda c: float(centroids[int(c)][0]))
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


def compute_xy_limits(Y: np.ndarray, pad_ratio: float = 0.03) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    Y = np.asarray(Y, dtype=np.float32)
    xmin = float(np.min(Y[:, 0]))
    xmax = float(np.max(Y[:, 0]))
    ymin = float(np.min(Y[:, 1]))
    ymax = float(np.max(Y[:, 1]))
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    padx = dx * float(pad_ratio)
    pady = dy * float(pad_ratio)
    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)


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
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 160,
    bbox_tight: bool = True,
    verbose: bool = True,
) -> None:
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
    plt.figure(figsize=figsize, dpi=dpi)
    if Cp is not None:
        plt.scatter(Yp[:, 0], Yp[:, 1], s=point_size, alpha=alpha, c=Cp, linewidths=0, rasterized=True)
    else:
        plt.scatter(Yp[:, 0], Yp[:, 1], s=point_size, alpha=alpha, linewidths=0, rasterized=True)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"} if bbox_tight else {}
        plt.savefig(out_png, **save_kwargs)
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
    init_xy: Optional[np.ndarray] = None,
    cache_npy: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    if cache_npy is not None and Path(cache_npy).exists():
        if verbose:
            print(f"[layout] loading cached layout -> {cache_npy}")
        return np.load(cache_npy).astype(np.float32)

    try:
        import igraph as ig
    except Exception as e:
        raise ImportError("python-igraph 不可用：pip install python-igraph") from e

    t0 = time.time()
    edges = list(zip(u.tolist(), v.tolist()))
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.es["weight"] = w.astype(np.float32).tolist()

    if init_xy is None:
        rng = np.random.default_rng(int(random_state))
        seed_xy = rng.normal(size=(n_nodes, 2)).astype(np.float32)
    else:
        seed_xy = np.asarray(init_xy, dtype=np.float32)
        if seed_xy.shape != (n_nodes, 2):
            raise ValueError(f"init_xy shape {seed_xy.shape} != ({n_nodes}, 2)")

    seed_list = seed_xy.tolist()

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
