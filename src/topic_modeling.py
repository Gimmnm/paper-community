from __future__ import annotations

"""
基于 Leiden 社区结果的全局主题建模（Topic-SCORE / SCORE 风格）
=====================================================

目标：
- 从 data_store.pkl 读取论文标题/作者/摘要
- 按社区（如 membership_r1.0000.npy）聚合文本，构造 社区-词 矩阵 D（词 x 社区）
- 参考 preprocessing.R 做停用词与频次过滤
- 参考 score_functions.R / 《Using SVD for Topic Modeling》做谱方法主题建模
- 在全局设定 K 个主题下，为每个社区估计 K 维主题权重（Top1 / Top2）
- 输出主题词、社区主题权重、社区标签等结果文件

默认适配你的项目目录结构：
project/
├── data/
│   ├── stopwords.txt
│   └── data_store.pkl
├── src/
│   └── topic_modeling.py   <-- 本文件放这里
└── out/
    └── leiden_sweep/
        └── membership_r1.0000.npy

运行示例（在 project 根目录）：
python src/topic_modeling.py --k 10 --resolution 1.0

也可显式指定 membership 文件：
python src/topic_modeling.py --k 10 --membership out/leiden_sweep/membership_r1.0000.npy
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
from typing import Iterable, Sequence
import argparse
import json
import math
import pickle
import re
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.optimize import nnls, minimize
from sklearn.cluster import KMeans


# -----------------------------
# 路径默认值（与 core.py 保持一致风格）
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"


# -----------------------------
# 文本处理
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


def load_stopwords(path: Path | None, add_sklearn_english: bool = True) -> set[str]:
    stopwords: set[str] = set()
    if path is not None and path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.strip().lower()
            if w:
                stopwords.add(w)
    if add_sklearn_english:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stopwords |= set(ENGLISH_STOP_WORDS)
        except Exception:
            pass
    # 一些常见噪声词（可按需要增删）
    stopwords |= {
        "et", "al", "using", "used", "use", "based", "via", "new",
        "results", "result", "method", "methods", "paper", "study"
    }
    return stopwords


def tokenize_text(text: str | None, stopwords: set[str], min_len: int = 2) -> list[str]:
    if text is None:
        return []
    s = str(text)
    if not s or s.strip().lower() in {"", "nan", "na", "none"}:
        return []
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(s)]
    return [t for t in toks if len(t) >= min_len and t not in stopwords]


def repeat_extend(counter: Counter, tokens: Sequence[str], w: int = 1) -> None:
    if w <= 0 or not tokens:
        return
    if w == 1:
        counter.update(tokens)
    else:
        counter.update(t for t in tokens for _ in range(w))


# -----------------------------
# 数据读取
# -----------------------------

def load_data_store(pkl_path: Path) -> dict:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    required = {
        "n_papers", "author_names", "paper_authors", "paper_title",
        "paper_abstract", "paper_abstract_clean"
    }
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"data_store.pkl 缺少字段: {sorted(missing)}")
    return data


def load_membership(path: Path, n_papers_expected: int) -> np.ndarray:
    mem = np.load(path)
    mem = np.asarray(mem)
    if mem.ndim != 1:
        raise ValueError(f"membership 必须是一维数组，实际 shape={mem.shape}")

    # 两种常见情况：长度 = N（0-based 对应 pid=1..N），或长度 = N+1（1号位开始）
    if mem.size == n_papers_expected:
        return mem.astype(int)
    if mem.size == n_papers_expected + 1:
        return mem[1:].astype(int)
    raise ValueError(
        f"membership 长度={mem.size} 与 n_papers={n_papers_expected} 不匹配（期望 {n_papers_expected} 或 {n_papers_expected+1}）"
    )


def membership_path_from_resolution(base_out_dir: Path, resolution: float) -> Path:
    return base_out_dir / "leiden_sweep" / f"membership_r{resolution:.4f}.npy"


# -----------------------------
# 构造社区-词矩阵 D（词 x 社区）
# -----------------------------

@dataclass
class BuildMatrixConfig:
    include_title: bool = True
    include_abstract: bool = True
    include_authors: bool = True
    use_clean_abstract: bool = False
    title_weight: int = 3
    abstract_weight: int = 1
    author_weight: int = 1
    min_token_len: int = 2
    words_percent: float = 0.2       # 参考 preprocessing.R：保留高频词比例（按总频次阈值）
    docs_percent: float = 1.0        # 默认保留全部社区参与拟合；若<1 则仅对大社区拟合，但会对全部社区推断权重
    min_community_size: int = 1
    max_papers: int | None = None    # 调试用，默认全量


@dataclass
class MatrixBuildResult:
    D_all: sparse.csr_matrix              # 过滤词后，词 x 全部社区
    D_fit: sparse.csr_matrix              # 用于拟合 A 的词 x 拟合社区（可能与 D_all 相同）
    vocab: list[str]
    community_ids_all: np.ndarray         # D_all 列对应社区ID
    community_ids_fit: np.ndarray         # D_fit 列对应社区ID
    community_sizes: dict[int, int]       # 社区内论文数
    community_token_counts: dict[int, int]# 社区聚合文本token总数（过滤前近似）
    paper_to_comm: np.ndarray             # 长度 N，paper pid=idx+1 -> community id
    paper_ids_by_comm: dict[int, list[int]]


def _quantile_threshold(arr: np.ndarray, keep_percent: float) -> float:
    if arr.size == 0:
        return 0.0
    keep_percent = float(keep_percent)
    keep_percent = max(0.0, min(1.0, keep_percent))
    if keep_percent >= 1.0:
        return float(np.min(arr))
    if keep_percent <= 0.0:
        return float(np.max(arr))
    q = 1.0 - keep_percent
    return float(np.quantile(arr, q))


def build_community_term_matrix(
    data: dict,
    membership: np.ndarray,
    stopwords: set[str],
    cfg: BuildMatrixConfig,
) -> MatrixBuildResult:
    n_papers = int(data["n_papers"])
    if membership.size != n_papers:
        raise ValueError(f"membership 长度 {membership.size} != n_papers {n_papers}")

    author_names = data["author_names"]
    paper_authors = data["paper_authors"]
    paper_title = data["paper_title"]
    paper_abs = data["paper_abstract_clean"] if cfg.use_clean_abstract else data["paper_abstract"]

    # 社区统计
    comm_sizes = Counter(int(c) for c in membership.tolist())
    valid_comms = sorted([c for c, sz in comm_sizes.items() if sz >= cfg.min_community_size])
    valid_set = set(valid_comms)

    paper_ids_by_comm: dict[int, list[int]] = defaultdict(list)
    for pid in range(1, n_papers + 1):
        c = int(membership[pid - 1])
        if c in valid_set:
            paper_ids_by_comm[c].append(pid)

    # author token cache（加速）
    author_token_cache: dict[int, list[str]] = {}

    # 聚合到每个社区 Counter
    comm_counters: dict[int, Counter] = {c: Counter() for c in valid_comms}
    comm_token_counts: dict[int, int] = {c: 0 for c in valid_comms}
    global_token_sum = Counter()

    max_pid = n_papers if cfg.max_papers is None else min(n_papers, int(cfg.max_papers))

    for pid in range(1, max_pid + 1):
        c = int(membership[pid - 1])
        if c not in valid_set:
            continue
        cc = comm_counters[c]

        if cfg.include_title:
            tt = tokenize_text(paper_title[pid], stopwords, cfg.min_token_len)
            repeat_extend(cc, tt, cfg.title_weight)
            comm_token_counts[c] += len(tt) * max(cfg.title_weight, 0)

        if cfg.include_abstract:
            aa = tokenize_text(paper_abs[pid], stopwords, cfg.min_token_len)
            repeat_extend(cc, aa, cfg.abstract_weight)
            comm_token_counts[c] += len(aa) * max(cfg.abstract_weight, 0)

        if cfg.include_authors:
            author_toks: list[str] = []
            for aid in paper_authors[pid]:
                if aid not in author_token_cache:
                    author_token_cache[aid] = tokenize_text(author_names[aid], stopwords, cfg.min_token_len)
                author_toks.extend(author_token_cache[aid])
            repeat_extend(cc, author_toks, cfg.author_weight)
            comm_token_counts[c] += len(author_toks) * max(cfg.author_weight, 0)

    # 全局词频（在社区聚合后）
    for c in valid_comms:
        global_token_sum.update(comm_counters[c])

    if not global_token_sum:
        raise RuntimeError("构造的词表为空。请检查 stopwords、文本字段和 token 规则。")

    # 参考 preprocessing.R: 保留 top words_percent 高频词（按总词频）
    all_words = list(global_token_sum.keys())
    all_freq = np.array([global_token_sum[w] for w in all_words], dtype=float)
    word_thr = _quantile_threshold(all_freq, cfg.words_percent)
    kept_words = [w for w in all_words if global_token_sum[w] >= word_thr]
    # 排序：先按频次降序，再按词典序，保证稳定
    kept_words.sort(key=lambda w: (-global_token_sum[w], w))

    vocab = kept_words
    word2idx = {w: i for i, w in enumerate(vocab)}

    # 构造 D_all（词 x 全部有效社区）
    community_ids_all = np.array(valid_comms, dtype=int)
    comm2col_all = {c: j for j, c in enumerate(community_ids_all.tolist())}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for c in valid_comms:
        col = comm2col_all[c]
        for w, cnt in comm_counters[c].items():
            i = word2idx.get(w)
            if i is not None and cnt > 0:
                rows.append(i)
                cols.append(col)
                vals.append(float(cnt))

    D_all = sparse.csr_matrix((vals, (rows, cols)), shape=(len(vocab), len(community_ids_all)), dtype=np.float64)

    # 参考 preprocessing.R: 保留 top docs_percent “大文档”（这里文档=社区）用于拟合，但对全部社区做推断
    doc_lens = np.asarray(D_all.sum(axis=0)).ravel()
    doc_thr = _quantile_threshold(doc_lens, cfg.docs_percent)
    fit_mask = doc_lens >= doc_thr
    if fit_mask.sum() < 2:
        # 至少保留2个社区用于SVD；兜底用全量
        fit_mask = np.ones_like(doc_lens, dtype=bool)
    community_ids_fit = community_ids_all[fit_mask]
    D_fit = D_all[:, fit_mask].tocsr()

    return MatrixBuildResult(
        D_all=D_all,
        D_fit=D_fit,
        vocab=vocab,
        community_ids_all=community_ids_all,
        community_ids_fit=community_ids_fit,
        community_sizes={int(k): int(v) for k, v in comm_sizes.items()},
        community_token_counts={int(k): int(v) for k, v in comm_token_counts.items()},
        paper_to_comm=membership.astype(int),
        paper_ids_by_comm={int(k): v for k, v in paper_ids_by_comm.items()},
    )


# -----------------------------
# Topic-SCORE / SCORE 风格实现（参考 score_functions.R）
# -----------------------------

@dataclass
class TopicScoreConfig:
    k: int
    vh_method: str = "svs-sp"     # ['svs', 'sp', 'svs-sp']
    m: int | None = None           # kmeans 聚类中心数（SVS去噪）
    k0: int | None = None          # 候选顶点数（SVS）
    mquantile: float = 0.0         # 论文公式中的 tau（用于 M 截断）
    m_trunc_mode: str = "floor"   # 'floor' (理论公式 max), 'cap' (R脚本 pmin 兼容)
    seed: int = 42
    max_svs_combinations: int = 20000
    weighted_nnls: bool = True     # 参考论文 Eq.(8)
    eps: float = 1e-12


@dataclass
class TopicScoreResult:
    A_hat: np.ndarray              # 词 x K（每列和为1）
    W_hat_fit: np.ndarray          # K x n_fit
    W_hat_all: np.ndarray          # K x n_all
    Xi: np.ndarray                 # 词 x K
    R: np.ndarray                  # 词 x (K-1)
    V: np.ndarray                  # K x (K-1)
    Pi: np.ndarray                 # 词 x K
    M_trunk: np.ndarray            # 长度 p
    meta: dict


def _sorted_svds_u(X: sparse.spmatrix | np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # svds 返回奇异值升序，需要倒序重排
    u, s, vt = svds(X, k=k, return_singular_vectors=True)
    order = np.argsort(s)[::-1]
    return u[:, order], s[order], vt[order, :]


def successive_proj(R: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """参考 score_functions.R::successiveProj，在 R 的行上做 successive projection。"""
    n = R.shape[0]
    Y = np.hstack([np.ones((n, 1)), R]).astype(float)
    idxs: list[int] = []
    for _ in range(K):
        l2 = np.linalg.norm(Y, axis=1)
        idx = int(np.argmax(l2))
        idxs.append(idx)
        u = Y[idx] / (np.linalg.norm(Y[idx]) + 1e-15)
        # 投影剔除
        Y = Y - np.outer(Y @ u, u)
    V = R[np.array(idxs), :]
    return V, np.array(idxs, dtype=int)


def _pairwise_sqdist(X: np.ndarray) -> np.ndarray:
    g = X @ X.T
    d = np.diag(g)
    return d[:, None] + d[None, :] - 2.0 * g


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """欧氏投影到概率单纯形 {x>=0, sum x =1}。"""
    if v.ndim != 1:
        v = v.ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones(n) / n
    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    w = np.maximum(v - theta, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def simplex_distance(theta: np.ndarray, V: np.ndarray, eps: float = 1e-12) -> float:
    """
    点 theta 到由 V 的行向量构成的 simplex 的欧式距离平方。
    通过在概率单纯形上优化 b：min || theta - sum_k b_k V_k ||^2。
    """
    K = V.shape[0]

    def fun(b: np.ndarray) -> float:
        x = b @ V
        r = theta - x
        return float(r @ r)

    x0 = np.ones(K, dtype=float) / K
    cons = [{"type": "eq", "fun": lambda b: float(np.sum(b) - 1.0)}]
    bounds = [(0.0, None)] * K
    res = minimize(fun, x0=x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9, "disp": False})
    if res.success and np.isfinite(res.fun):
        return float(max(res.fun, 0.0))
    # 失败兜底：无约束最小二乘后投影到单纯形
    # solve min ||V^T b - theta||^2
    try:
        b_ls, *_ = np.linalg.lstsq(V.T, theta, rcond=None)
        b = _project_to_simplex(np.asarray(b_ls).ravel())
        return float(np.sum((theta - b @ V) ** 2))
    except Exception:
        return float(fun(x0))


def vertices_est_svs(R: np.ndarray, K: int, m: int, K0: int, seed: int = 42,
                     max_svs_combinations: int = 20000) -> tuple[np.ndarray, dict]:
    """
    参考 score_functions.R::vertices_est 的 Python 实现（SVS 风格）。
    - Step 2a: kmeans 得 m 个中心
    - Step 2b': 先选 K0 个“代表性中心”（远点 + 贪心）
    - Step 2b : 在其中枚举 K 个点作为 simplex 顶点，最小化 max simplex distance
    """
    p, d = R.shape
    m = int(max(K, min(m, p)))
    K0 = int(max(K, min(K0, m)))

    km = KMeans(n_clusters=m, n_init=10, random_state=seed)
    km.fit(R)
    theta = km.cluster_centers_.astype(float)      # (m, K-1)
    theta_original = theta.copy()

    # 2b'：从 m 个中心中挑 K0 个候选中心（仿 R 代码）
    if K0 < m:
        dist = _pairwise_sqdist(theta)
        ii, jj = np.unravel_index(np.argmax(dist), dist.shape)
        selected = [int(ii), int(jj)] if K0 >= 2 else [int(ii)]
        remaining = [idx for idx in range(m) if idx not in selected]
        while len(selected) < K0 and remaining:
            theta0 = theta[np.array(selected)]
            rem = theta[np.array(remaining)]
            # R里 distance <- rep(1,k0-1)%*%t(diag(inner))-2*theta0%*%t(theta)
            # 等价于“到已选点的平均距离（去掉常数项）”最大
            # 用平方距离均值更直观
            d2 = ((rem[None, :, :] - theta0[:, None, :]) ** 2).sum(axis=2)  # (len(selected), len(remaining))
            ave_dist = d2.mean(axis=0)
            add_pos = int(np.argmax(ave_dist))
            selected.append(remaining[add_pos])
            remaining.pop(add_pos)
        theta = theta[np.array(selected)]

    # 如果候选点数已经等于 K，直接返回
    if theta.shape[0] == K:
        return theta, {"theta_all": theta_original, "theta_used": theta, "n_combinations": 1}

    # 2b：枚举组合，找最小 max simplex distance 的 simplex
    n_centers = theta.shape[0]
    combs = list(combinations(range(n_centers), K))
    if len(combs) > max_svs_combinations:
        # 组合过多时，退化为 SP（更稳妥于直接爆炸）
        V_sp, idx_sp = successive_proj(theta, K)
        return V_sp, {
            "theta_all": theta_original,
            "theta_used": theta,
            "fallback": "sp_on_kmeans_centers",
            "n_combinations": len(combs),
        }

    best_val = math.inf
    best_comb = None
    for comb in combs:
        V = theta[np.array(comb)]
        max_val = 0.0
        for j in range(n_centers):
            d2 = simplex_distance(theta[j], V)
            if d2 > max_val:
                max_val = d2
                if max_val >= best_val:
                    break
        if max_val < best_val:
            best_val = max_val
            best_comb = comb

    assert best_comb is not None
    V = theta[np.array(best_comb)]
    return V, {
        "theta_all": theta_original,
        "theta_used": theta,
        "best_max_simplex_dist": float(best_val),
        "best_comb": [int(x) for x in best_comb],
        "n_combinations": len(combs),
    }


def estimate_A_topic_score(D_fit: sparse.csr_matrix, cfg: TopicScoreConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    参考 score_functions.R::norm_score 的核心流程，输出 A_hat 与中间量。

    返回：A_hat, Xi, R, V, Pi, M_trunk, vh_meta
    - A_hat: p x K（每列和为1）
    """
    if not sparse.isspmatrix_csr(D_fit):
        D_fit = D_fit.tocsr()
    p, n = D_fit.shape
    K = int(cfg.k)
    if K < 2:
        raise ValueError("K 必须 >= 2")
    if K >= min(p, n):
        raise ValueError(f"K={K} 必须小于 min(p,n)={min(p,n)}；当前 p={p}, n={n}")

    # Pre-SVD normalization (theory paper Eq.(6) + R脚本变体)
    M = np.asarray(D_fit.mean(axis=1)).ravel().astype(float)  # rowMeans(D)
    qv = float(np.quantile(M, cfg.mquantile))
    if cfg.m_trunc_mode.lower() in {"floor", "pmax", "theory"}:
        M_trunk = np.maximum(M, qv)   # 论文 Eq.(6): max(eta_j, quantile)
    elif cfg.m_trunc_mode.lower() in {"cap", "pmin", "r"}:
        M_trunk = np.minimum(M, qv)   # 与 score_functions.R 的 pmin 行为兼容
    else:
        raise ValueError("m_trunc_mode 只能是 floor/theory 或 cap/r")
    M_trunk = np.maximum(M_trunk, cfg.eps)

    # X = diag(1/sqrt(M_trunk)) * D_fit
    row_scale = 1.0 / np.sqrt(M_trunk)
    X = D_fit.multiply(row_scale[:, None]).astype(np.float64)

    # SVD
    Xi, svals, _ = _sorted_svds_u(X, k=K, seed=cfg.seed)

    # Step 1: SCORE ratio normalization
    xi1 = np.abs(Xi[:, 0].copy())
    xi1 = np.maximum(xi1, cfg.eps)
    Xi[:, 0] = xi1
    R = Xi[:, 1:K] / xi1[:, None]

    # Step 2: Vertex hunting
    vh = cfg.vh_method.lower()
    m = cfg.m if cfg.m is not None else min(max(10 * K, 50), p)
    K0 = cfg.k0 if cfg.k0 is not None else min(K + 2, m)

    if vh == "sp":
        V, idxs = successive_proj(R, K)
        vh_meta = {"method": "sp", "index_set": idxs.tolist()}
    elif vh == "svs":
        V, meta = vertices_est_svs(R, K=K, m=m, K0=K0, seed=cfg.seed,
                                   max_svs_combinations=cfg.max_svs_combinations)
        vh_meta = {"method": "svs", **{k: v for k, v in meta.items() if k not in {"theta_all", "theta_used"}}}
    elif vh in {"svs-sp", "svssp", "svs_sp"}:
        # 参考 R 的 vertices_est_SP：先 kmeans 再在中心上做 SP
        m_eff = int(max(K, min(m, p)))
        km = KMeans(n_clusters=m_eff, n_init=10, random_state=cfg.seed)
        km.fit(R)
        theta = km.cluster_centers_.astype(float)
        V, idxs = successive_proj(theta, K)
        vh_meta = {"method": "svs-sp", "m": m_eff, "index_set_on_centers": idxs.tolist()}
    else:
        raise ValueError("vh_method 必须是 svs / sp / svs-sp")

    # Step 3: 求 Pi（先线性解，再截断负值，再行归一化）
    R1 = np.hstack([R, np.ones((p, 1), dtype=float)])          # p x K
    V1 = np.hstack([V, np.ones((K, 1), dtype=float)])          # K x K
    try:
        V1_inv = np.linalg.inv(V1)
    except np.linalg.LinAlgError:
        V1_inv = np.linalg.pinv(V1)
    Pi = R1 @ V1_inv
    Pi = np.maximum(Pi, 0.0)
    row_sum = Pi.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= cfg.eps, 1.0, row_sum)
    Pi = Pi / row_sum

    # Step 4 + Step 5: A_hat = sqrt(M_trunk) * xi1 * Pi，然后列归一化
    A_tilde = (np.sqrt(M_trunk) * xi1)[:, None] * Pi
    col_sum = A_tilde.sum(axis=0, keepdims=True)
    col_sum = np.where(col_sum <= cfg.eps, 1.0, col_sum)
    A_hat = A_tilde / col_sum

    meta = {
        "svals": svals.tolist(),
        "vh": vh_meta,
        "m": int(m),
        "k0": int(K0),
        "p": int(p),
        "n_fit": int(n),
        "mquantile": float(cfg.mquantile),
        "m_trunc_mode": cfg.m_trunc_mode,
    }
    return A_hat, Xi, R, V, Pi, M_trunk, meta


def infer_W_from_A(
    A_hat: np.ndarray,
    D: sparse.csr_matrix,
    M_trunk_for_weight: np.ndarray | None = None,
    weighted_nnls: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    根据 A_hat 推断每个“文档”（这里是社区）的主题权重 W_hat。
    参考论文 Eq.(8) 的 weighted least squares；这里用 NNLS 近似 + simplex 归一化。

    返回 W_hat: K x n
    """
    if not sparse.isspmatrix_csr(D):
        D = D.tocsr()
    p, n = D.shape
    p2, K = A_hat.shape
    if p != p2:
        raise ValueError(f"A_hat 行数 {p2} != D 行数 {p}")

    W = np.zeros((K, n), dtype=float)

    if weighted_nnls and M_trunk_for_weight is not None:
        ww = 1.0 / np.sqrt(np.maximum(M_trunk_for_weight, eps))
        A_reg = A_hat * ww[:, None]
    else:
        ww = None
        A_reg = A_hat

    for i in range(n):
        x = D.getcol(i).toarray().ravel().astype(float)
        if ww is not None:
            x_reg = x * ww
        else:
            x_reg = x
        coef, _ = nnls(A_reg, x_reg)
        s = coef.sum()
        if s <= eps:
            coef = np.ones(K) / K
        else:
            coef = coef / s
        W[:, i] = coef
    return W


def fit_topic_score_on_communities(
    matrix_res: MatrixBuildResult,
    cfg: TopicScoreConfig,
) -> TopicScoreResult:
    # 在 D_fit 上拟合 A_hat
    A_hat, Xi, R, V, Pi, M_trunk_fit, meta = estimate_A_topic_score(matrix_res.D_fit, cfg)

    # 如果 D_all 和 D_fit 列集合不同，需要对 D_all 使用同一词表（已经保证相同词行）
    # 用 A_hat 在 D_fit / D_all 上分别推断 W_hat
    W_hat_fit = infer_W_from_A(
        A_hat=A_hat,
        D=matrix_res.D_fit,
        M_trunk_for_weight=M_trunk_fit,
        weighted_nnls=cfg.weighted_nnls,
        eps=cfg.eps,
    )

    # D_all 的加权项需要基于 D_fit 的 M_trunk（同一词空间）。若 D_all 词行和 D_fit 一样，直接复用。
    W_hat_all = infer_W_from_A(
        A_hat=A_hat,
        D=matrix_res.D_all,
        M_trunk_for_weight=M_trunk_fit if matrix_res.D_all.shape[0] == matrix_res.D_fit.shape[0] else None,
        weighted_nnls=cfg.weighted_nnls and (matrix_res.D_all.shape[0] == matrix_res.D_fit.shape[0]),
        eps=cfg.eps,
    )

    return TopicScoreResult(
        A_hat=A_hat,
        W_hat_fit=W_hat_fit,
        W_hat_all=W_hat_all,
        Xi=Xi,
        R=R,
        V=V,
        Pi=Pi,
        M_trunk=M_trunk_fit,
        meta=meta,
    )


# -----------------------------
# 输出结果（主题词、社区标签、代表论文）
# -----------------------------

def top_words_per_topic(A_hat: np.ndarray, vocab: Sequence[str], topn: int = 15) -> dict[int, list[tuple[str, float]]]:
    p, K = A_hat.shape
    out: dict[int, list[tuple[str, float]]] = {}
    for k in range(K):
        idx = np.argsort(A_hat[:, k])[::-1][:topn]
        out[k] = [(vocab[i], float(A_hat[i, k])) for i in idx]
    return out


def build_community_summary_tables(
    data: dict,
    matrix_res: MatrixBuildResult,
    topic_res: TopicScoreResult,
    topn_words: int = 10,
    topn_papers: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vocab = matrix_res.vocab
    community_ids = matrix_res.community_ids_all
    W = topic_res.W_hat_all  # K x n_all
    K = W.shape[0]
    top_words = top_words_per_topic(topic_res.A_hat, vocab, topn=max(topn_words, 20))

    # 主题表
    topic_rows = []
    for k in range(K):
        words = top_words[k][:topn_words]
        topic_rows.append({
            "topic_id": k,
            "top_words": " ".join([w for w, _ in words]),
            **{f"word_{i+1}": words[i][0] if i < len(words) else "" for i in range(topn_words)},
            **{f"weight_{i+1}": words[i][1] if i < len(words) else np.nan for i in range(topn_words)},
        })
    df_topics = pd.DataFrame(topic_rows)

    # 社区表（全局 K 维主题权重 + Top1/Top2）
    comm_rows = []
    paper_title = data["paper_title"]
    paper_year = data.get("paper_year", None)

    for j, c in enumerate(community_ids.tolist()):
        w = W[:, j]
        order = np.argsort(w)[::-1]
        t1 = int(order[0])
        t2 = int(order[1]) if K >= 2 else int(order[0])

        # 取若干代表论文（这里先按年份新->旧，再标题；你后续可换成中心性）
        pids = matrix_res.paper_ids_by_comm.get(int(c), [])
        if paper_year is not None:
            pids_sorted = sorted(pids, key=lambda pid: (int(paper_year[pid]), pid), reverse=True)
        else:
            pids_sorted = pids[:]
        reps = pids_sorted[:topn_papers]
        rep_titles = [str(paper_title[pid]).strip() for pid in reps]

        row = {
            "community_id": int(c),
            "n_papers": int(matrix_res.community_sizes.get(int(c), len(pids))),
            "token_count": int(matrix_res.community_token_counts.get(int(c), 0)),
            "top1_topic": t1,
            "top1_weight": float(w[t1]),
            "top1_keywords": " ".join([x[0] for x in top_words[t1][:6]]),
            "top2_topic": t2,
            "top2_weight": float(w[t2]),
            "top2_keywords": " ".join([x[0] for x in top_words[t2][:6]]),
            "community_label": f"T{t1}:{' '.join([x[0] for x in top_words[t1][:4]])} | T{t2}:{' '.join([x[0] for x in top_words[t2][:4]])}",
            "rep_papers": " || ".join(rep_titles),
        }
        for k in range(K):
            row[f"topic_{k}"] = float(w[k])
        comm_rows.append(row)

    df_comm = pd.DataFrame(comm_rows).sort_values(["n_papers", "community_id"], ascending=[False, True]).reset_index(drop=True)

    # 主题 -> 代表社区表
    topic_comm_rows = []
    for k in range(K):
        scores = W[k, :]
        top_idx = np.argsort(scores)[::-1][:10]
        for rank, j in enumerate(top_idx, start=1):
            c = int(community_ids[j])
            topic_comm_rows.append({
                "topic_id": k,
                "rank_in_topic": rank,
                "community_id": c,
                "topic_weight": float(scores[j]),
                "n_papers": int(matrix_res.community_sizes.get(c, 0)),
                "keywords": " ".join([x[0] for x in top_words[k][:8]]),
            })
    df_topic_communities = pd.DataFrame(topic_comm_rows)

    return df_topics, df_comm, df_topic_communities


def save_outputs(
    out_dir: Path,
    data: dict,
    matrix_res: MatrixBuildResult,
    topic_res: TopicScoreResult,
    build_cfg: BuildMatrixConfig,
    score_cfg: TopicScoreConfig,
    runtime_sec: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数组与词表
    np.save(out_dir / "A_hat.npy", topic_res.A_hat)
    np.save(out_dir / "W_hat_all.npy", topic_res.W_hat_all)
    np.save(out_dir / "W_hat_fit.npy", topic_res.W_hat_fit)
    np.save(out_dir / "community_ids_all.npy", matrix_res.community_ids_all)
    np.save(out_dir / "community_ids_fit.npy", matrix_res.community_ids_fit)
    np.save(out_dir / "M_trunk.npy", topic_res.M_trunk)
    (out_dir / "vocab.txt").write_text("\n".join(matrix_res.vocab), encoding="utf-8")

    # 稀疏矩阵（可选保存，方便复用）
    sparse.save_npz(out_dir / "D_all.npz", matrix_res.D_all)
    sparse.save_npz(out_dir / "D_fit.npz", matrix_res.D_fit)

    # 表格
    df_topics, df_comm, df_topic_comms = build_community_summary_tables(data, matrix_res, topic_res)
    df_topics.to_csv(out_dir / "topics_top_words.csv", index=False, encoding="utf-8-sig")
    df_comm.to_csv(out_dir / "communities_topic_weights.csv", index=False, encoding="utf-8-sig")
    df_topic_comms.to_csv(out_dir / "topic_representative_communities.csv", index=False, encoding="utf-8-sig")

    # 配置与元信息
    meta = {
        "runtime_sec": runtime_sec,
        "n_topics": int(score_cfg.k),
        "n_vocab": int(len(matrix_res.vocab)),
        "n_communities_all": int(matrix_res.D_all.shape[1]),
        "n_communities_fit": int(matrix_res.D_fit.shape[1]),
        "build_config": asdict(build_cfg),
        "score_config": asdict(score_cfg),
        "topic_score_meta": topic_res.meta,
        "notes": [
            "A_hat: word-topic matrix, columns sum to 1",
            "W_hat_all: topic-community weights, columns sum to 1",
            "community_ids_all[j] 对应 W_hat_all[:, j]",
            "论文ID pid 与 membership 的关系为 membership[pid-1]（默认假设）",
        ],
    }
    (out_dir / "topic_model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# -----------------------------
# 命令行入口
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 Leiden 社区的 Topic-SCORE 主题建模")
    p.add_argument("--k", type=int, required=True, help="全局主题数 K（例如 10）")
    p.add_argument("--resolution", type=float, default=1.0, help="Leiden 分辨率（默认 1.0）")
    p.add_argument("--membership", type=str, default=None, help="membership .npy 路径；不填则按 resolution 推断")

    # 数据路径
    p.add_argument("--data-store", type=str, default=str(DATA_DIR / "data_store.pkl"))
    p.add_argument("--stopwords", type=str, default=str(DATA_DIR / "stopwords.txt"))
    p.add_argument("--out-dir", type=str, default=None, help="输出目录；默认 out/topic_modeling/r{res}_K{k}")

    # 文本字段与权重
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--no-abstract", action="store_true")
    p.add_argument("--no-authors", action="store_true")
    p.add_argument("--use-clean-abstract", action="store_true", help="用 CleanAbstracts 代替 RawAbstract")
    p.add_argument("--title-weight", type=int, default=3)
    p.add_argument("--abstract-weight", type=int, default=1)
    p.add_argument("--author-weight", type=int, default=1)
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--no-sklearn-stopwords", action="store_true")

    # preprocessing.R 风格过滤
    p.add_argument("--words-percent", type=float, default=0.2,
                   help="保留高频词比例（按聚合词频 quantile 阈值），默认 0.2")
    p.add_argument("--docs-percent", type=float, default=1.0,
                   help="参与拟合的大社区比例（按社区文档长度 quantile 阈值），默认 1.0=全量")
    p.add_argument("--min-community-size", type=int, default=1,
                   help="社区最少论文数；小于该值的社区直接跳过")

    # SCORE 参数（参考 score_functions.R）
    p.add_argument("--vh-method", type=str, default="svs-sp", choices=["svs", "sp", "svs-sp"],
                   help="顶点搜索方法：svs / sp / svs-sp（默认 svs-sp，较稳且可扩展）")
    p.add_argument("--m", type=int, default=None, help="SVS 的 kmeans 中心数 m；默认自动")
    p.add_argument("--k0", type=int, default=None, help="SVS 候选顶点数 K0；默认自动（K+2）")
    p.add_argument("--mquantile", type=float, default=0.0,
                   help="预归一化 M 的截断分位数 tau（论文公式中的 tau）")
    p.add_argument("--m-trunc-mode", type=str, default="floor", choices=["floor", "cap"],
                   help="M截断方式：floor=理论公式max；cap=兼容R脚本pmin")
    p.add_argument("--max-svs-combinations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-weighted-nnls", action="store_true", help="关闭加权NNLS（论文Eq.8近似）")

    # 调试
    p.add_argument("--max-papers", type=int, default=None, help="仅处理前若干篇论文（调试）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    data_store_path = Path(args.data_store)
    stopwords_path = Path(args.stopwords) if args.stopwords else None
    if args.membership is None:
        membership_path = membership_path_from_resolution(OUT_DIR, args.resolution)
    else:
        membership_path = Path(args.membership)

    if args.out_dir is None:
        out_dir = OUT_DIR / "topic_modeling" / f"r{args.resolution:.4f}_K{args.k}"
    else:
        out_dir = Path(args.out_dir)

    print(f"[topic_model] data_store={data_store_path}")
    print(f"[topic_model] membership={membership_path}")
    print(f"[topic_model] output={out_dir}")

    data = load_data_store(data_store_path)
    membership = load_membership(membership_path, n_papers_expected=int(data["n_papers"]))

    stopwords = load_stopwords(stopwords_path, add_sklearn_english=(not args.no_sklearn_stopwords))
    print(f"[topic_model] stopwords loaded: {len(stopwords)}")

    build_cfg = BuildMatrixConfig(
        include_title=not args.no_title,
        include_abstract=not args.no_abstract,
        include_authors=not args.no_authors,
        use_clean_abstract=args.use_clean_abstract,
        title_weight=args.title_weight,
        abstract_weight=args.abstract_weight,
        author_weight=args.author_weight,
        min_token_len=args.min_token_len,
        words_percent=args.words_percent,
        docs_percent=args.docs_percent,
        min_community_size=args.min_community_size,
        max_papers=args.max_papers,
    )

    matrix_res = build_community_term_matrix(data, membership, stopwords, build_cfg)
    p_all, n_all = matrix_res.D_all.shape
    p_fit, n_fit = matrix_res.D_fit.shape
    print(f"[topic_model] D_all shape = (vocab={p_all}, communities={n_all})")
    print(f"[topic_model] D_fit shape = (vocab={p_fit}, communities={n_fit})")

    score_cfg = TopicScoreConfig(
        k=args.k,
        vh_method=args.vh_method,
        m=args.m,
        k0=args.k0,
        mquantile=args.mquantile,
        m_trunc_mode=args.m_trunc_mode,
        seed=args.seed,
        max_svs_combinations=args.max_svs_combinations,
        weighted_nnls=(not args.no_weighted_nnls),
    )

    topic_res = fit_topic_score_on_communities(matrix_res, score_cfg)
    print(f"[topic_model] fitted Topic-SCORE: A_hat={topic_res.A_hat.shape}, W_hat_all={topic_res.W_hat_all.shape}")

    save_outputs(out_dir, data, matrix_res, topic_res, build_cfg, score_cfg, runtime_sec=time.time() - t0)

    # 控制台打印简要主题结果
    top_words = top_words_per_topic(topic_res.A_hat, matrix_res.vocab, topn=10)
    print("\n[topic_model] Top words per topic:")
    for k in range(args.k):
        ws = " ".join([w for w, _ in top_words[k][:10]])
        print(f"  Topic {k:02d}: {ws}")

    print(f"\n[topic_model] done in {time.time() - t0:.1f}s")
    print(f"[topic_model] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
