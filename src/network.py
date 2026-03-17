# network.py
# -*- coding: utf-8 -*-

from __future__ import annotations  # 允许前向引用类型标注（更灵活）

# pathlib.Path：处理路径/缓存文件
from pathlib import Path

# time：打印耗时与进度
import time

# typing：类型标注（让函数接口更清晰）
from typing import Literal, Optional, Tuple

# numpy：向量、KNN 结果与 edges 的基础数据结构
import numpy as np

# scipy.sparse：用 CSR/COO 表示大图（内存与速度都更稳）
import scipy.sparse as sp


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对矩阵 X 的每一行做 L2 归一化，返回 float32 连续数组。
    - 对 cosine 相似度非常重要（cosine = normalize 后的 inner product）
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def knn_search_cosine(
    X: np.ndarray,
    k: int,
    *,
    backend: Literal["faiss", "sklearn", "hnswlib"] = "hnswlib",
    batch_size: int = 4096,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在归一化后的向量上做 KNN（cosine 相似度 = inner product）：
    返回：
      - knn_idx: (N, k)  每行是邻居索引（0-based）
      - knn_sim: (N, k)  每行是相似度（float32）

    注意：
      - 你必须保证 X 已经行归一化，否则相似度不是 cosine
      - 这里会自动排除 self（自己作为最近邻），确保输出邻居都是 “他人”
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    N, D = X.shape
    assert k >= 1

    # 为了去掉 self，我们先查 k+1，再把 self 过滤掉
    k_search = k + 1

    if backend == "faiss":
        try:
            import faiss  # faiss：CPU 下的大规模 ANN/精确检索都很快
        except Exception as e:
            raise ImportError(
                "faiss 不可用。请 pip install faiss-cpu；或把 backend 改成 'sklearn'（会更慢）。"
            ) from e

        # IndexFlatIP：精确 inner product（对 normalize 后向量就是 cosine）
        index = faiss.IndexFlatIP(D)
        index.add(X)

        knn_idx = np.empty((N, k), dtype=np.int32)
        knn_sim = np.empty((N, k), dtype=np.float32)

        t0 = time.time()
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            sims, ids = index.search(X[s:e], k_search)  # sims, ids: (B, k+1)
            # 过滤 self
            for i in range(e - s):
                row_ids = ids[i]
                row_sims = sims[i]
                keep_ids = []
                keep_sims = []
                for j, nb in enumerate(row_ids):
                    if nb == (s + i):
                        continue
                    keep_ids.append(nb)
                    keep_sims.append(row_sims[j])
                    if len(keep_ids) == k:
                        break
                knn_idx[s + i] = np.array(keep_ids, dtype=np.int32)
                knn_sim[s + i] = np.array(keep_sims, dtype=np.float32)

            if verbose and (e % 20000 == 0 or e == N):
                dt = time.time() - t0
                print(f"[knn/faiss] queried {e}/{N} rows ... ({dt:.1f}s)")

        return knn_idx, knn_sim

    # sklearn fallback：慢但依赖少
    if backend == "sklearn":
        try:
            from sklearn.neighbors import NearestNeighbors  # sklearn：通用 KNN（大规模会慢）
        except Exception as e:
            raise ImportError("sklearn 不可用。请 pip install scikit-learn") from e

        # metric='cosine'：返回的是 cosine distance = 1 - cosine similarity
        nn = NearestNeighbors(n_neighbors=k_search, metric="cosine", algorithm="brute", n_jobs=-1)
        t0 = time.time()
        nn.fit(X)
        dist, idx = nn.kneighbors(X, return_distance=True)
        if verbose:
            print(f"[knn/sklearn] done in {time.time()-t0:.1f}s")

        # dist = 1 - sim
        sim = (1.0 - dist).astype(np.float32)
        idx = idx.astype(np.int32)

        # 过滤 self（和 faiss 同逻辑）
        knn_idx = np.empty((N, k), dtype=np.int32)
        knn_sim = np.empty((N, k), dtype=np.float32)
        for i in range(N):
            keep_ids = []
            keep_sims = []
            for j, nb in enumerate(idx[i]):
                if nb == i:
                    continue
                keep_ids.append(nb)
                keep_sims.append(sim[i, j])
                if len(keep_ids) == k:
                    break
            knn_idx[i] = np.array(keep_ids, dtype=np.int32)
            knn_sim[i] = np.array(keep_sims, dtype=np.float32)

        return knn_idx, knn_sim
    
    if backend == "hnswlib":
        try:
            import hnswlib
        except Exception as e:
            raise ImportError("hnswlib 不可用：pip install hnswlib") from e

        # hnswlib 的 cosine 距离是 (1 - cosine_sim)
        # X 必须 float32 contiguous
        X = np.asarray(X, dtype=np.float32, order="C")
        N, D = X.shape
        k_search = k + 1

        index = hnswlib.Index(space="cosine", dim=D)
        # M / ef_construction：越大越准但越慢；下面这组对 83k 很常用
        index.init_index(max_elements=N, ef_construction=200, M=32)
        index.add_items(X, np.arange(N, dtype=np.int32))
        index.set_ef(max(50, k_search * 2))

        knn_idx = np.empty((N, k), dtype=np.int32)
        knn_sim = np.empty((N, k), dtype=np.float32)

        t0 = time.time()
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)

            ids, dist = index.knn_query(X[s:e], k=k_search)   # dist: cosine distance
            sims = (1.0 - dist).astype(np.float32)            # -> cosine similarity
            ids = ids.astype(np.int32)

            # 过滤 self
            for i in range(e - s):
                row_ids = ids[i]
                row_sims = sims[i]
                keep_ids = []
                keep_sims = []
                self_id = s + i
                for j, nb in enumerate(row_ids):
                    if nb == self_id:
                        continue
                    keep_ids.append(nb)
                    keep_sims.append(row_sims[j])
                    if len(keep_ids) == k:
                        break
                knn_idx[self_id] = np.array(keep_ids, dtype=np.int32)
                knn_sim[self_id] = np.array(keep_sims, dtype=np.float32)

            if verbose and (e % 20000 == 0 or e == N):
                dt = time.time() - t0
                print(f"[knn/hnswlib] queried {e}/{N} rows ... ({dt:.1f}s)")

        return knn_idx, knn_sim


    raise ValueError(f"unknown backend={backend}")


def build_directed_knn_csr(knn_idx: np.ndarray, knn_sim: np.ndarray, n_nodes: int) -> sp.csr_matrix:
    """
    把 KNN 结果（每个点指向 k 个邻居）构造成“有向”稀疏矩阵 A：
      - A[i, j] = sim(i, j)
      - shape=(n_nodes, n_nodes)
      - CSR 适合后续 mutual 操作与图统计
    """
    knn_idx = np.asarray(knn_idx, dtype=np.int32)
    knn_sim = np.asarray(knn_sim, dtype=np.float32)

    N, k = knn_idx.shape
    assert N == n_nodes
    assert knn_sim.shape == (N, k)

    rows = np.repeat(np.arange(N, dtype=np.int32), k)
    cols = knn_idx.reshape(-1)
    data = knn_sim.reshape(-1)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    A.eliminate_zeros()
    return A


def mutualize_directed_knn(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    把有向 KNN 图 A 变成 mutual-kNN 的“无向对称”图：
    逻辑：
      - 保留 i->j 与 j->i 都存在的边
      - 权重用 (w_ij + w_ji)/2
    实现：
      - M1 = A 在 mutual 位置的权重
      - 对称平均：M = (M1 + M1.T)/2
    """
    A = A.tocsr()
    AT = A.transpose().tocsr()

    # reciprocal mask：AT > 0 表示 j->i 存在
    mask = (AT > 0).astype(np.bool_)
    M1 = A.multiply(mask)  # 保留 i->j 的 w_ij（仅 mutual）
    M = (M1 + M1.transpose()) * 0.5
    M.eliminate_zeros()
    return M.tocsr()


def csr_to_undirected_edge_list(A_sym: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    把对称 CSR（无向图）转为上三角边表：
      - u, v, w，满足 u < v
    便于：
      - igraph 构图
      - 落盘缓存
    """
    A_sym = A_sym.tocsr()
    tri = sp.triu(A_sym, k=1).tocoo()
    u = tri.row.astype(np.int32)
    v = tri.col.astype(np.int32)
    w = tri.data.astype(np.float32)
    return u, v, w


def save_edges_npz(
    path: Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_nodes: int,
    k: int,
    normalized: bool,
    note: str = "",
) -> None:
    """
    保存边表到 .npz：
      - u, v, w
      - n_nodes / k / normalized 等元信息
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=np.asarray(u, dtype=np.int32),
        v=np.asarray(v, dtype=np.int32),
        w=np.asarray(w, dtype=np.float32),
        n_nodes=np.int32(n_nodes),
        k=np.int32(k),
        normalized=np.int8(1 if normalized else 0),
        note=np.array([note], dtype=object),
    )


def load_edges_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, bool]:
    """
    读取 .npz 边表缓存，返回：
      - u, v, w
      - n_nodes, k, normalized
    """
    z = np.load(Path(path), allow_pickle=True)
    u = z["u"].astype(np.int32)
    v = z["v"].astype(np.int32)
    w = z["w"].astype(np.float32)
    n_nodes = int(z["n_nodes"])
    k = int(z["k"])
    normalized = bool(int(z["normalized"]))
    return u, v, w, n_nodes, k, normalized


def build_or_load_mutual_knn_graph(
    X: np.ndarray,
    *,
    k: int,
    cache_npz: Optional[Path],
    knn_backend: Literal["faiss", "sklearn"] = "faiss",
    knn_batch_size: int = 4096,
    normalize: bool = True,
    verbose: bool = True,
) -> Tuple[sp.csr_matrix, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    一站式：mutual-kNN 建图（并支持缓存）
    输入：
      - X: (N, D) float32
    输出：
      - A_sym: (N, N) CSR（对称、无向、权重=cosine 相似）
      - (u, v, w): 上三角边表

    缓存：
      - 若 cache_npz 存在则直接读边表并重建 CSR
      - 否则建图并保存
    """
    N = int(X.shape[0])

    if cache_npz is not None and Path(cache_npz).exists():
        if verbose:
            print(f"[network] loading cached edges: {cache_npz}")
        u, v, w, n_nodes, k0, normalized = load_edges_npz(cache_npz)
        assert n_nodes == N
        assert k0 == k
        # 重建 CSR（对称）
        rows = np.concatenate([u, v])
        cols = np.concatenate([v, u])
        data = np.concatenate([w, w]).astype(np.float32)
        A_sym = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
        A_sym.eliminate_zeros()
        return A_sym, (u, v, w)

    # 建图
    t0 = time.time()

    Xn = l2_normalize_rows(X) if normalize else np.asarray(X, dtype=np.float32, order="C")
    if verbose:
        print(f"[network] X shape={Xn.shape}, normalized={normalize}")

    knn_idx, knn_sim = knn_search_cosine(
        Xn, k=k, backend=knn_backend, batch_size=knn_batch_size, verbose=verbose
    )

    A = build_directed_knn_csr(knn_idx, knn_sim, n_nodes=N)
    A_sym = mutualize_directed_knn(A)
    u, v, w = csr_to_undirected_edge_list(A_sym)

    if verbose:
        print(f"[network] directed edges = {A.nnz}")
        print(f"[network] mutual undirected edges = {len(u)}")
        print(f"[network] build time = {time.time()-t0:.1f}s")

    if cache_npz is not None:
        note = f"mutual-kNN graph (cosine via IP), backend={knn_backend}"
        save_edges_npz(
            cache_npz,
            u=u, v=v, w=w,
            n_nodes=N, k=k,
            normalized=normalize,
            note=note,
        )
        if verbose:
            print(f"[network] cached edges -> {cache_npz}")

    return A_sym, (u, v, w)
