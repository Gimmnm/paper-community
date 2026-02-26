from __future__ import annotations

"""
多分辨率 Topic-SCORE 结果的 topic 对齐脚本（以某个参考分辨率为基准）
=====================================================

目标：
- 解决不同分辨率下 Topic ID 不稳定（Topic 3 在相邻分辨率不一定语义相同）的问题
- 以参考分辨率（默认 r=1.0）的 topics 作为“颜色/编号基准”
- 为每个分辨率求出 target_topic -> ref_topic 的映射（Hungarian matching）
- 生成“对齐后的结果目录”，供可视化脚本直接使用

输入（来自 topic_modeling_multi.py 的输出）:
- out/topic_modeling_multi/K{K}/r{res}/A_hat.npy
- out/topic_modeling_multi/K{K}/r{res}/vocab.txt
- out/topic_modeling_multi/K{K}/r{res}/topics_top_words.csv
- out/topic_modeling_multi/K{K}/r{res}/communities_topic_weights.csv
- out/topic_modeling_multi/K{K}/r{res}/topic_representative_communities.csv

输出（默认）:
- out/topic_modeling_multi/K{K}/aligned_to_r{ref}/
  - topic_alignment.csv                   # 每个分辨率、每个 topic 的映射与相似度
  - topic_alignment_summary.csv           # 每个分辨率的平均相似度/最小相似度等
  - reference_topics.csv                  # 参考分辨率主题词（便于查看）
  - r{res}/communities_topic_weights.csv  # 已按参考 topic 编号重排（同 schema）
  - r{res}/topics_top_words.csv           # 已按参考 topic 编号重排（同 schema）
  - r{res}/topic_representative_communities.csv  # topic_id 已对齐
  - r{res}/topic_alignment_meta.json      # 该分辨率的映射详情
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
RDIR_RE = re.compile(r"^r([0-9]+(?:\.[0-9]+)?)$")


# -----------------------------
# Utilities
# -----------------------------

def _parse_r_from_dirname(name: str) -> float | None:
    m = RDIR_RE.match(name)
    return float(m.group(1)) if m else None


def _rdirname(r: float) -> str:
    return f"r{float(r):.4f}"


def discover_topic_result_dirs(topic_root: Path) -> list[tuple[float, Path]]:
    items: list[tuple[float, Path]] = []
    for p in sorted(topic_root.iterdir()):
        if not p.is_dir():
            continue
        r = _parse_r_from_dirname(p.name)
        if r is None:
            continue
        # 至少要有这几个文件才算一个可对齐结果
        if (p / "A_hat.npy").exists() and (p / "vocab.txt").exists() and (p / "communities_topic_weights.csv").exists():
            items.append((r, p))
    items.sort(key=lambda x: x[0])
    return items


def select_by_interval_or_list(
    items: list[tuple[float, Path]],
    *,
    resolutions: Iterable[float] | None = None,
    r_min: float | None = 0.0001,
    r_max: float | None = 5.0,
    include: Iterable[float] | None = None,
    tol: float = 1e-8,
) -> list[tuple[float, Path]]:
    include = [float(x) for x in (include or [])]
    if resolutions is not None:
        req = [float(x) for x in resolutions]
        out: list[tuple[float, Path]] = []
        for rr in req + include:
            hit = None
            for r, p in items:
                if abs(float(r) - rr) <= tol:
                    hit = (r, p)
                    break
            if hit is not None:
                out.append(hit)
        uniq = {round(r, 10): (r, p) for r, p in out}
        return [uniq[k] for k in sorted(uniq.keys())]

    out = []
    for r, p in items:
        in_range = True
        if r_min is not None:
            in_range = in_range and (float(r) >= float(r_min) - tol)
        if r_max is not None:
            in_range = in_range and (float(r) <= float(r_max) + tol)
        in_include = any(abs(float(r) - x) <= tol for x in include)
        if in_range or in_include:
            out.append((r, p))
    uniq = {round(r, 10): (r, p) for r, p in out}
    return [uniq[k] for k in sorted(uniq.keys())]


def load_vocab(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _col_prob(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = np.maximum(X, 0.0)
    s = X.sum(axis=0, keepdims=True)
    s[s <= eps] = 1.0
    return X / s


def _cosine_sim_matrix(A_ref: np.ndarray, A_tgt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # 输入 shape: (V_common, K)
    X = np.asarray(A_ref, dtype=float)
    Y = np.asarray(A_tgt, dtype=float)
    xn = np.linalg.norm(X, axis=0, keepdims=True)
    yn = np.linalg.norm(Y, axis=0, keepdims=True)
    xn[xn <= eps] = 1.0
    yn[yn <= eps] = 1.0
    return (X.T @ Y) / (xn.T @ yn)


def _js_similarity_matrix(A_ref: np.ndarray, A_tgt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # 1 - JS divergence, 值域近似 [0,1]
    P = _col_prob(A_ref, eps=eps)
    Q = _col_prob(A_tgt, eps=eps)
    K = P.shape[1]
    L = Q.shape[1]
    sim = np.zeros((K, L), dtype=float)
    for i in range(K):
        p = P[:, i].copy()
        p = np.clip(p, eps, None)
        p /= p.sum()
        for j in range(L):
            q = Q[:, j].copy()
            q = np.clip(q, eps, None)
            q /= q.sum()
            m = 0.5 * (p + q)
            js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
            sim[i, j] = float(max(0.0, 1.0 - js))
    return sim


def build_common_vocab_topic_mats(
    ref_A: np.ndarray,
    ref_vocab: list[str],
    tgt_A: np.ndarray,
    tgt_vocab: list[str],
    *,
    topn_per_topic: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """将两个 A_hat 投影到共同词表。

    返回 (A_ref_common, A_tgt_common, common_vocab)，形状均为 (V_common, K).
    若 topn_per_topic 不为空，则只保留双方 topic topn 词并取并集后再交集，可减少噪声。
    """
    ref_A = np.asarray(ref_A)
    tgt_A = np.asarray(tgt_A)
    ref_map = {w: i for i, w in enumerate(ref_vocab)}
    tgt_map = {w: i for i, w in enumerate(tgt_vocab)}

    if topn_per_topic is not None and topn_per_topic > 0:
        keep_ref = set()
        for k in range(ref_A.shape[1]):
            idx = np.argsort(ref_A[:, k])[::-1][:topn_per_topic]
            keep_ref.update(ref_vocab[i] for i in idx)
        keep_tgt = set()
        for k in range(tgt_A.shape[1]):
            idx = np.argsort(tgt_A[:, k])[::-1][:topn_per_topic]
            keep_tgt.update(tgt_vocab[i] for i in idx)
        common = sorted((set(ref_map) & set(tgt_map)) & (keep_ref | keep_tgt))
    else:
        common = sorted(set(ref_map) & set(tgt_map))

    if not common:
        raise RuntimeError("参考分辨率与目标分辨率没有共同词表，无法进行 topic 对齐。")

    ref_idx = np.array([ref_map[w] for w in common], dtype=int)
    tgt_idx = np.array([tgt_map[w] for w in common], dtype=int)
    return ref_A[ref_idx, :], tgt_A[tgt_idx, :], common


def hungarian_match(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给定相似度矩阵 sim[ref_topic, tgt_topic]，返回匹配 (ref_ids, tgt_ids)。"""
    sim = np.asarray(sim, dtype=float)
    if linear_sum_assignment is not None:
        try:
            r_idx, c_idx = linear_sum_assignment(-sim)
            return r_idx.astype(int), c_idx.astype(int)
        except Exception:
            pass

    # greedy fallback
    K, L = sim.shape
    used_r, used_c = set(), set()
    pairs = []
    flat = [(-float(sim[i, j]), i, j) for i in range(K) for j in range(L)]
    flat.sort()
    for _neg, i, j in flat:
        if i in used_r or j in used_c:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
        if len(pairs) >= min(K, L):
            break
    r_idx = np.array([i for i, _ in pairs], dtype=int)
    c_idx = np.array([j for _, j in pairs], dtype=int)
    return r_idx, c_idx


def invert_mapping_tgt_to_ref(t2r: dict[int, int], K: int) -> dict[int, int]:
    # ref -> tgt；若有缺失则留空
    r2t = {}
    for t, r in t2r.items():
        if 0 <= int(r) < K:
            r2t[int(r)] = int(t)
    return r2t


def remap_topics_table(df_topics: pd.DataFrame, t2r: dict[int, int], K: int) -> pd.DataFrame:
    d = df_topics.copy()
    if "topic_id" not in d.columns:
        return d
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce").astype("Int64")
    d["topic_id_original"] = d["topic_id"]
    d["topic_id"] = d["topic_id"].map(lambda x: t2r.get(int(x), -1) if pd.notna(x) else -1).astype(int)
    d = d.sort_values("topic_id").reset_index(drop=True)

    # 保证 0..K-1 都有一行（缺失时补空行，理论上不会发生）
    present = set(pd.to_numeric(d["topic_id"], errors="coerce").astype(int).tolist())
    rows = []
    for rr in range(K):
        if rr not in present:
            rows.append({"topic_id": rr, "topic_id_original": np.nan, "top_words": ""})
    if rows:
        d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True).sort_values("topic_id").reset_index(drop=True)
    return d


def remap_topic_representative_communities(df_tc: pd.DataFrame, t2r: dict[int, int]) -> pd.DataFrame:
    d = df_tc.copy()
    if "topic_id" not in d.columns:
        return d
    d["topic_id_original"] = d["topic_id"]
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce").map(lambda x: t2r.get(int(x), -1) if pd.notna(x) else -1)
    d["topic_id"] = d["topic_id"].astype(int)
    sort_cols = [c for c in ["topic_id", "rank_in_topic", "community_id"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols).reset_index(drop=True)
    return d


def remap_communities_topic_weights(df_comm: pd.DataFrame, t2r: dict[int, int], K: int) -> pd.DataFrame:
    d = df_comm.copy()
    # 1) 复制原信息
    if "top1_topic" in d.columns:
        d["top1_topic_original"] = d["top1_topic"]
    if "top2_topic" in d.columns:
        d["top2_topic_original"] = d["top2_topic"]

    # 2) 重排 topic_0..topic_{K-1} 到参考编号顺序（保持列名不变）
    orig_cols = [f"topic_{i}" for i in range(K) if f"topic_{i}" in d.columns]
    if len(orig_cols) >= 1:
        tmp = {}
        for tgt_t in range(K):
            col = f"topic_{tgt_t}"
            if col not in d.columns:
                continue
            ref_t = t2r.get(tgt_t, None)
            if ref_t is None or not (0 <= int(ref_t) < K):
                continue
            tmp[int(ref_t)] = pd.to_numeric(d[col], errors="coerce")
        for ref_t in range(K):
            d[f"topic_{ref_t}"] = tmp.get(ref_t, pd.Series(np.nan, index=d.index))

    # 3) 重算 top1/top2（按对齐后的 topic_* 列）
    topic_mat = np.column_stack([
        pd.to_numeric(d.get(f"topic_{i}"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for i in range(K)
    ]) if K > 0 else np.zeros((len(d), 0), dtype=float)

    if K > 0 and topic_mat.size > 0:
        order = np.argsort(topic_mat, axis=1)[:, ::-1]
        t1 = order[:, 0]
        t2 = order[:, 1] if K >= 2 else order[:, 0]
        d["top1_topic"] = t1.astype(int)
        d["top1_weight"] = topic_mat[np.arange(len(d)), t1].astype(float)
        d["top2_topic"] = t2.astype(int)
        d["top2_weight"] = topic_mat[np.arange(len(d)), t2].astype(float)

    # 4) 兼容原 schema：更新关键词与 label（若存在）
    # 注意：这里先不重写 top1_keywords/top2_keywords/community_label，因为需要 topics_top_words。
    return d


def refresh_keyword_columns_from_topics(df_comm: pd.DataFrame, df_topics: pd.DataFrame) -> pd.DataFrame:
    d = df_comm.copy()
    topic_words = {}
    if "topic_id" in df_topics.columns:
        for _, row in df_topics.iterrows():
            try:
                tid = int(row["topic_id"])
            except Exception:
                continue
            tw = str(row.get("top_words", "")).strip()
            topic_words[tid] = tw

    def _kw(t: int, n=6) -> str:
        ws = topic_words.get(int(t), "").split()
        return " ".join(ws[:n])

    if "top1_topic" in d.columns:
        d["top1_keywords"] = [ _kw(int(t), 6) if pd.notna(t) else "" for t in pd.to_numeric(d["top1_topic"], errors="coerce") ]
    if "top2_topic" in d.columns:
        d["top2_keywords"] = [ _kw(int(t), 6) if pd.notna(t) else "" for t in pd.to_numeric(d["top2_topic"], errors="coerce") ]
    if {"top1_topic", "top2_topic"}.issubset(d.columns):
        labels = []
        t1s = pd.to_numeric(d["top1_topic"], errors="coerce").fillna(-1).astype(int).tolist()
        t2s = pd.to_numeric(d["top2_topic"], errors="coerce").fillna(-1).astype(int).tolist()
        for t1, t2 in zip(t1s, t2s):
            k1 = _kw(t1, 4)
            k2 = _kw(t2, 4)
            labels.append(f"T{t1}:{k1} | T{t2}:{k2}")
        d["community_label"] = labels
    return d


# -----------------------------
# Main alignment logic
# -----------------------------

@dataclass
class AlignConfig:
    metric: str = "cosine"          # cosine | js
    topn_common_vocab: int | None = None  # 若设为正数，则仅基于各topic前N词的并集∩共同词表
    min_common_vocab: int = 50
    ref_resolution: float = 1.0
    r_min: float = 0.0001
    r_max: float = 5.0


def align_one_target_to_ref(
    ref_dir: Path,
    tgt_dir: Path,
    cfg: AlignConfig,
) -> dict:
    ref_A = np.load(ref_dir / "A_hat.npy")
    tgt_A = np.load(tgt_dir / "A_hat.npy")
    ref_vocab = load_vocab(ref_dir / "vocab.txt")
    tgt_vocab = load_vocab(tgt_dir / "vocab.txt")

    if ref_A.ndim != 2 or tgt_A.ndim != 2:
        raise ValueError("A_hat.npy 维度错误，应为 2D")
    K_ref = int(ref_A.shape[1])
    K_tgt = int(tgt_A.shape[1])
    if K_ref != K_tgt:
        raise ValueError(f"K 不一致：ref={K_ref}, tgt={K_tgt}。请确保都来自同一个 K 目录。")
    K = K_ref

    Aref_c, Atgt_c, common_vocab = build_common_vocab_topic_mats(
        ref_A, ref_vocab, tgt_A, tgt_vocab,
        topn_per_topic=cfg.topn_common_vocab,
    )
    n_common = len(common_vocab)
    if n_common < int(cfg.min_common_vocab):
        raise RuntimeError(f"共同词表过少：{n_common} < min_common_vocab={cfg.min_common_vocab}")

    if cfg.metric == "cosine":
        sim = _cosine_sim_matrix(Aref_c, Atgt_c)
    elif cfg.metric == "js":
        sim = _js_similarity_matrix(Aref_c, Atgt_c)
    else:
        raise ValueError(f"未知 metric: {cfg.metric}")

    ref_ids, tgt_ids = hungarian_match(sim)
    t2r = {int(t): int(r) for r, t in zip(ref_ids.tolist(), tgt_ids.tolist())}  # target_topic -> ref_topic

    # 汇总指标
    matched_sims = [float(sim[r, t]) for r, t in zip(ref_ids.tolist(), tgt_ids.tolist())]
    avg_sim = float(np.mean(matched_sims)) if matched_sims else float("nan")
    min_sim = float(np.min(matched_sims)) if matched_sims else float("nan")

    return {
        "K": K,
        "n_common_vocab": int(n_common),
        "common_vocab": common_vocab,
        "similarity_matrix": sim,
        "ref_ids": ref_ids,
        "tgt_ids": tgt_ids,
        "t2r": t2r,
        "avg_similarity": avg_sim,
        "min_similarity": min_sim,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多分辨率 topic 对齐（固定参考分辨率）")
    p.add_argument("--k", type=int, required=True, help="全局主题数 K")
    p.add_argument("--topic-root", type=str, default=None,
                   help="topic_modeling_multi 的 K 目录；默认 out/topic_modeling_multi/K{K}")
    p.add_argument("--out-root", type=str, default=None,
                   help="输出根目录；默认 <topic-root>/aligned_to_r{ref}")

    p.add_argument("--ref-resolution", type=float, default=1.0, help="参考分辨率（默认 1.0）")
    p.add_argument("--metric", type=str, default="cosine", choices=["cosine", "js"], help="topic 相似度度量")
    p.add_argument("--topn-common-vocab", type=int, default=None,
                   help="仅用各 topic 前N词构建对齐共同词表（可降噪，如 200）")
    p.add_argument("--min-common-vocab", type=int, default=50, help="共同词表最小要求")

    p.add_argument("--r-min", type=float, default=0.0001)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--include", type=float, nargs="*", default=None)
    p.add_argument("--resolutions", type=float, nargs="*", default=None)

    p.add_argument("--copy-meta-json", action="store_true", help="拷贝 topic_model_meta.json 到对齐输出目录")
    p.add_argument("--save-sim-matrix", action="store_true", help="保存每个分辨率的 topic 相似度矩阵 .npy")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    verbose = not args.quiet

    topic_root = Path(args.topic_root) if args.topic_root else (OUT_DIR / "topic_modeling_multi" / f"K{args.k}")
    if not topic_root.exists():
        raise FileNotFoundError(f"topic_root 不存在: {topic_root}")

    discovered = discover_topic_result_dirs(topic_root)
    if not discovered:
        raise FileNotFoundError(f"在 {topic_root} 中未发现 r*/A_hat.npy 等主题结果目录")

    selected = select_by_interval_or_list(
        discovered,
        resolutions=args.resolutions,
        r_min=args.r_min,
        r_max=args.r_max,
        include=args.include,
    )
    if not selected:
        raise RuntimeError("筛选后没有可用分辨率结果")

    # 参考分辨率：优先精确命中，否则选最近的
    r_vals = np.array([r for r, _ in selected], dtype=float)
    idx_ref = int(np.argmin(np.abs(r_vals - float(args.ref_resolution))))
    ref_r, ref_dir = selected[idx_ref]

    out_root = Path(args.out_root) if args.out_root else (topic_root / f"aligned_to_r{ref_r:.4f}")
    out_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[align] topic_root={topic_root}")
        print(f"[align] discovered={len(discovered)} selected={len(selected)}")
        print(f"[align] reference: r={ref_r:.4f} -> {ref_dir}")
        print(f"[align] out_root={out_root}")

    if args.dry_run:
        print("[align] resolutions:", ", ".join(f"{r:.4f}" for r, _ in selected))
        return

    cfg = AlignConfig(
        metric=args.metric,
        topn_common_vocab=args.topn_common_vocab,
        min_common_vocab=args.min_common_vocab,
        ref_resolution=float(ref_r),
        r_min=float(args.r_min),
        r_max=float(args.r_max),
    )

    # 保存参考主题表（原始）
    ref_topics_path = ref_dir / "topics_top_words.csv"
    if ref_topics_path.exists():
        df_ref_topics = pd.read_csv(ref_topics_path, encoding="utf-8-sig")
        df_ref_topics.to_csv(out_root / "reference_topics.csv", index=False, encoding="utf-8-sig")
    else:
        df_ref_topics = pd.DataFrame()

    rows_alignment = []
    rows_summary = []

    for r, src_dir in selected:
        dst_dir = out_root / _rdirname(r)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # 读取原表
        df_topics = pd.read_csv(src_dir / "topics_top_words.csv", encoding="utf-8-sig") if (src_dir / "topics_top_words.csv").exists() else pd.DataFrame()
        df_comm = pd.read_csv(src_dir / "communities_topic_weights.csv", encoding="utf-8-sig") if (src_dir / "communities_topic_weights.csv").exists() else pd.DataFrame()
        df_tcomm = pd.read_csv(src_dir / "topic_representative_communities.csv", encoding="utf-8-sig") if (src_dir / "topic_representative_communities.csv").exists() else pd.DataFrame()

        if abs(float(r) - float(ref_r)) <= 1e-10:
            # 参考分辨率本身：恒等映射
            A = np.load(src_dir / "A_hat.npy")
            K = int(A.shape[1])
            t2r = {t: t for t in range(K)}
            sim_mat = np.eye(K, dtype=float)
            align_info = {
                "K": K,
                "n_common_vocab": int(A.shape[0]),
                "t2r": t2r,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "similarity_matrix": sim_mat,
                "ref_ids": np.arange(K),
                "tgt_ids": np.arange(K),
            }
        else:
            align_info = align_one_target_to_ref(ref_dir, src_dir, cfg)
            K = int(align_info["K"])
            t2r = {int(k): int(v) for k, v in align_info["t2r"].items()}
            sim_mat = np.asarray(align_info["similarity_matrix"], dtype=float)

        # 写逐topic映射表行
        for tgt_t in range(K):
            ref_t = t2r.get(tgt_t, -1)
            sim_val = float(sim_mat[ref_t, tgt_t]) if (0 <= ref_t < sim_mat.shape[0] and 0 <= tgt_t < sim_mat.shape[1]) else np.nan
            rows_alignment.append({
                "resolution": float(r),
                "is_reference": bool(abs(float(r) - float(ref_r)) <= 1e-10),
                "target_topic": int(tgt_t),
                "ref_topic": int(ref_t),
                "similarity": sim_val,
                "n_common_vocab": int(align_info.get("n_common_vocab", np.nan)),
                "metric": cfg.metric,
            })

        rows_summary.append({
            "resolution": float(r),
            "is_reference": bool(abs(float(r) - float(ref_r)) <= 1e-10),
            "avg_similarity": float(align_info.get("avg_similarity", np.nan)),
            "min_similarity": float(align_info.get("min_similarity", np.nan)),
            "n_common_vocab": int(align_info.get("n_common_vocab", np.nan)),
            "metric": cfg.metric,
            "src_dir": str(src_dir),
            "dst_dir": str(dst_dir),
        })

        # 生成对齐后的三张表（保持 schema 尽量不变）
        df_topics_aligned = remap_topics_table(df_topics, t2r, K) if not df_topics.empty else df_topics
        df_comm_aligned = remap_communities_topic_weights(df_comm, t2r, K) if not df_comm.empty else df_comm
        df_tcomm_aligned = remap_topic_representative_communities(df_tcomm, t2r) if not df_tcomm.empty else df_tcomm

        if not df_comm_aligned.empty and not df_topics_aligned.empty:
            df_comm_aligned = refresh_keyword_columns_from_topics(df_comm_aligned, df_topics_aligned)

        # 输出对齐结果：使用与原脚本一致的文件名，方便 topic_visualization_multires.py 直接指向该目录使用
        if not df_topics_aligned.empty:
            df_topics_aligned.to_csv(dst_dir / "topics_top_words.csv", index=False, encoding="utf-8-sig")
        if not df_comm_aligned.empty:
            df_comm_aligned.to_csv(dst_dir / "communities_topic_weights.csv", index=False, encoding="utf-8-sig")
        if not df_tcomm_aligned.empty:
            df_tcomm_aligned.to_csv(dst_dir / "topic_representative_communities.csv", index=False, encoding="utf-8-sig")

        # 额外写一份保留原文件名的说明元信息
        meta = {
            "resolution": float(r),
            "reference_resolution": float(ref_r),
            "metric": cfg.metric,
            "n_common_vocab": int(align_info.get("n_common_vocab", 0)),
            "avg_similarity": float(align_info.get("avg_similarity", np.nan)),
            "min_similarity": float(align_info.get("min_similarity", np.nan)),
            "target_to_ref": {str(k): int(v) for k, v in sorted(t2r.items())},
            "note": "本目录中的 topics/communities 表已按参考分辨率 topic 编号重排；topic_0..topic_{K-1}、top1_topic/top2_topic 均为对齐后的参考编号。",
            "source_dir": str(src_dir),
        }
        (dst_dir / "topic_alignment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.copy_meta_json and (src_dir / "topic_model_meta.json").exists():
            try:
                (dst_dir / "topic_model_meta.json").write_text((src_dir / "topic_model_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
        if args.save_sim_matrix:
            np.save(dst_dir / "topic_similarity_to_ref.npy", sim_mat)

        if verbose:
            print(f"[align] r={r:.4f} -> avg_sim={meta['avg_similarity']:.4f}, min_sim={meta['min_similarity']:.4f}, common_vocab={meta['n_common_vocab']}")

    # 汇总输出
    df_align = pd.DataFrame(rows_alignment).sort_values(["resolution", "target_topic"]).reset_index(drop=True)
    df_sum = pd.DataFrame(rows_summary).sort_values("resolution").reset_index(drop=True)
    df_align.to_csv(out_root / "topic_alignment.csv", index=False, encoding="utf-8-sig")
    df_sum.to_csv(out_root / "topic_alignment_summary.csv", index=False, encoding="utf-8-sig")

    run_meta = {
        "topic_root": str(topic_root),
        "out_root": str(out_root),
        "selected_resolutions": [float(r) for r, _ in selected],
        "reference_resolution_requested": float(args.ref_resolution),
        "reference_resolution_actual": float(ref_r),
        "config": asdict(cfg),
        "args": vars(args),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[align] =============================")
    print(f"[align] done. reference r={ref_r:.4f}")
    print(f"[align] outputs -> {out_root}")
    print(f"[align] summary -> {out_root / 'topic_alignment_summary.csv'}")


if __name__ == "__main__":
    main()
