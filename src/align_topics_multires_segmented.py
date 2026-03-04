from __future__ import annotations

"""
分段（segmented）Topic 对齐脚本（多分辨率 Topic-SCORE）
=====================================================

为什么需要分段：
- 固定单一参考分辨率（例如 r=1.0）对齐全区间时，低分辨率段常出现很低的 avg/min similarity：
  这往往不是“参考选错”，而是跨尺度确实发生了 merge/split/recomposition。
- 但相邻分辨率之间通常更连续（adjacent similarity 高），因此更适合用“分段锚点”来做稳定可视化：
  每一段内部用同一个 anchor 分辨率做 Hungarian 一对一对齐；段与段之间不强求语义身份一致。

输入（来自 topic_modeling_multi.py 的输出）:
- out/topic_modeling_multi/K{K}/r{res}/A_hat.npy
- out/topic_modeling_multi/K{K}/r{res}/vocab.txt
- out/topic_modeling_multi/K{K}/r{res}/topics_top_words.csv
- out/topic_modeling_multi/K{K}/r{res}/communities_topic_weights.csv
- out/topic_modeling_multi/K{K}/r{res}/topic_representative_communities.csv

输出（默认）:
- out/topic_modeling_multi/K{K}/aligned_segmented/
  - segments.csv                             # 每段的范围与 anchor
  - topic_alignment_adjacent_summary.csv      # 计算断点用：相邻分辨率对齐质量
  - topic_alignment_adjacent.csv              # 可选：逐 topic 的相邻对齐记录（用于诊断）
  - segment_{id}_anchor_r{anchor}/
      - topic_alignment.csv                   # 段内：每个分辨率、每个 topic 的映射与相似度
      - topic_alignment_summary.csv           # 段内：每个分辨率的平均/最小相似度等
      - reference_topics.csv                  # 该段 anchor 主题词（便于查看）
      - r{res}/communities_topic_weights.csv  # 已按 anchor topic 编号重排（同 schema）
      - r{res}/topics_top_words.csv           # 已按 anchor topic 编号重排（同 schema）
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
    X = np.asarray(A_ref, dtype=float)
    Y = np.asarray(A_tgt, dtype=float)
    xn = np.linalg.norm(X, axis=0, keepdims=True)
    yn = np.linalg.norm(Y, axis=0, keepdims=True)
    xn[xn <= eps] = 1.0
    yn[yn <= eps] = 1.0
    return (X.T @ Y) / (xn.T @ yn)


def _js_similarity_matrix(A_ref: np.ndarray, A_tgt: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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


def remap_topics_table(df_topics: pd.DataFrame, t2r: dict[int, int], K: int) -> pd.DataFrame:
    d = df_topics.copy()
    if "topic_id" not in d.columns:
        return d
    d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce").astype("Int64")
    d["topic_id_original"] = d["topic_id"]
    d["topic_id"] = d["topic_id"].map(lambda x: t2r.get(int(x), -1) if pd.notna(x) else -1).astype(int)
    d = d.sort_values("topic_id").reset_index(drop=True)

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
    if "top1_topic" in d.columns:
        d["top1_topic_original"] = d["top1_topic"]
    if "top2_topic" in d.columns:
        d["top2_topic_original"] = d["top2_topic"]

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
# Alignment logic
# -----------------------------

@dataclass
class AlignConfig:
    metric: str = "cosine"          # cosine | js
    topn_common_vocab: int | None = None
    min_common_vocab: int = 50


def align_dir_to_dir(
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
        raise ValueError(f"K 不一致：ref={K_ref}, tgt={K_tgt}")
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

    matched_sims = [float(sim[r, t]) for r, t in zip(ref_ids.tolist(), tgt_ids.tolist())]
    avg_sim = float(np.mean(matched_sims)) if matched_sims else float("nan")
    min_sim = float(np.min(matched_sims)) if matched_sims else float("nan")

    return {
        "K": K,
        "n_common_vocab": int(n_common),
        "similarity_matrix": sim,
        "ref_ids": ref_ids,
        "tgt_ids": tgt_ids,
        "t2r": t2r,
        "avg_similarity": avg_sim,
        "min_similarity": min_sim,
    }


def compute_adjacent_alignment(
    selected: list[tuple[float, Path]],
    cfg: AlignConfig,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对每一对相邻分辨率做一次 Hungarian 对齐（prev 作为 ref）"""
    rows_sum = []
    rows_topic = []
    for i in range(len(selected) - 1):
        r0, d0 = selected[i]
        r1, d1 = selected[i + 1]
        info = align_dir_to_dir(d0, d1, cfg)
        K = int(info["K"])
        sim = np.asarray(info["similarity_matrix"], dtype=float)
        t2r = info["t2r"]

        # 逐 topic 行（以 next 的 target_topic 为主）
        for tgt_t in range(K):
            ref_t = t2r.get(tgt_t, -1)
            sim_val = float(sim[ref_t, tgt_t]) if (0 <= ref_t < sim.shape[0] and 0 <= tgt_t < sim.shape[1]) else np.nan
            rows_topic.append({
                "resolution_prev": float(r0),
                "resolution_next": float(r1),
                "target_topic_next": int(tgt_t),
                "ref_topic_prev": int(ref_t),
                "similarity": sim_val,
                "metric": cfg.metric,
            })

        rows_sum.append({
            "resolution_prev": float(r0),
            "resolution_next": float(r1),
            "avg_similarity": float(info["avg_similarity"]),
            "min_similarity": float(info["min_similarity"]),
            "n_common_vocab": int(info["n_common_vocab"]),
            "metric": cfg.metric,
        })
        if verbose:
            print(f"[seg] adjacent {r0:.4f}->{r1:.4f} avg={info['avg_similarity']:.4f} min={info['min_similarity']:.4f}")

    df_sum = pd.DataFrame(rows_sum)
    df_topic = pd.DataFrame(rows_topic)
    return df_sum, df_topic


def segment_resolutions(
    selected: list[tuple[float, Path]],
    df_adj_sum: pd.DataFrame,
    *,
    break_avg_thresh: float = 0.85,
    break_min_thresh: float = 0.10,
    min_segment_size: int = 2,
) -> list[dict]:
    """基于相邻对齐质量切段：若相邻 avg 或 min 低于阈值，则在该边之后切断。"""
    if len(selected) == 0:
        return []
    # map edge -> (avg,min)
    edge = {}
    for _, row in df_adj_sum.iterrows():
        edge[(float(row["resolution_prev"]), float(row["resolution_next"]))] = (float(row["avg_similarity"]), float(row["min_similarity"]))

    r_list = [float(r) for r, _ in selected]
    break_after_idx = set()
    for i in range(len(r_list) - 1):
        key = (r_list[i], r_list[i + 1])
        avg, mn = edge.get(key, (1.0, 1.0))
        if (avg < float(break_avg_thresh)) or (mn < float(break_min_thresh)):
            break_after_idx.add(i)

    # build segments
    segs = []
    start = 0
    seg_id = 0
    for i in range(len(r_list) - 1):
        if i in break_after_idx:
            end = i
            segs.append({"segment_id": seg_id, "start_idx": start, "end_idx": end})
            seg_id += 1
            start = i + 1
    segs.append({"segment_id": seg_id, "start_idx": start, "end_idx": len(r_list) - 1})

    # merge tiny segments (simple)
    def seg_len(s): return int(s["end_idx"] - s["start_idx"] + 1)
    merged = []
    for s in segs:
        if not merged:
            merged.append(s)
            continue
        if seg_len(s) < int(min_segment_size):
            # merge into previous
            merged[-1]["end_idx"] = s["end_idx"]
        else:
            merged.append(s)

    # finalize with r range（此处重新连续编号，确保后续目录名 segment_00/01/... 不会出现跳号）
    out = []
    for new_sid, s in enumerate(merged):
        si, ei = int(s["start_idx"]), int(s["end_idx"])
        out.append({
            "segment_id": int(new_sid),
            "start_idx": si,
            "end_idx": ei,
            "r_start": float(r_list[si]),
            "r_end": float(r_list[ei]),
        })
    return out


def choose_anchor_for_segment(
    seg: dict,
    selected: list[tuple[float, Path]],
    df_adj_sum: pd.DataFrame,
    *,
    strategy: str = "best-adjacent",
) -> tuple[float, Path]:
    """在段内选一个 anchor 分辨率（默认：与相邻最相似的那个，避免选到段边界）。"""
    si = int(seg["start_idx"])
    ei = int(seg["end_idx"])
    r_list = [float(r) for r, _ in selected]
    d_list = [p for _, p in selected]

    if strategy == "middle":
        mid = (si + ei) // 2
        return r_list[mid], d_list[mid]

    # best-adjacent
    # edge avg map
    edge_avg = {}
    for _, row in df_adj_sum.iterrows():
        edge_avg[(float(row["resolution_prev"]), float(row["resolution_next"]))] = float(row["avg_similarity"])

    best_i = None
    best_score = -1e9
    mid = (si + ei) / 2.0
    for i in range(si, ei + 1):
        score = 0.0
        if i - 1 >= si:
            score += edge_avg.get((r_list[i - 1], r_list[i]), 0.0)
        if i + 1 <= ei:
            score += edge_avg.get((r_list[i], r_list[i + 1]), 0.0)
        # mild preference for being near middle
        score -= 1e-3 * abs(i - mid)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        best_i = (si + ei) // 2
    return r_list[best_i], d_list[best_i]


def align_segment_and_write(
    seg: dict,
    selected: list[tuple[float, Path]],
    cfg: AlignConfig,
    out_root: Path,
    *,
    anchor_strategy: str = "best-adjacent",
    df_adj_sum: pd.DataFrame,
    verbose: bool = True,
    save_sim_matrix: bool = False,
) -> dict:
    sid = int(seg["segment_id"])
    si, ei = int(seg["start_idx"]), int(seg["end_idx"])
    segment_items = selected[si:ei + 1]

    anchor_r, anchor_dir = choose_anchor_for_segment(seg, selected, df_adj_sum, strategy=anchor_strategy)
    seg_name = f"segment_{sid:02d}_anchor_r{anchor_r:.4f}"
    seg_out = out_root / seg_name
    seg_out.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[seg] === segment {sid} [{seg['r_start']:.4f}, {seg['r_end']:.4f}] n={len(segment_items)} ===")
        print(f"[seg] anchor: r={anchor_r:.4f} -> {anchor_dir}")
        print(f"[seg] out   : {seg_out}")

    # save anchor topics table
    if (anchor_dir / "topics_top_words.csv").exists():
        df_anchor_topics = pd.read_csv(anchor_dir / "topics_top_words.csv", encoding="utf-8-sig")
        df_anchor_topics.to_csv(seg_out / "reference_topics.csv", index=False, encoding="utf-8-sig")
    else:
        df_anchor_topics = pd.DataFrame()

    rows_alignment = []
    rows_summary = []

    # Align each resolution in segment to anchor
    for r, src_dir in segment_items:
        dst_dir = seg_out / _rdirname(r)
        dst_dir.mkdir(parents=True, exist_ok=True)

        df_topics = pd.read_csv(src_dir / "topics_top_words.csv", encoding="utf-8-sig") if (src_dir / "topics_top_words.csv").exists() else pd.DataFrame()
        df_comm = pd.read_csv(src_dir / "communities_topic_weights.csv", encoding="utf-8-sig") if (src_dir / "communities_topic_weights.csv").exists() else pd.DataFrame()
        df_tcomm = pd.read_csv(src_dir / "topic_representative_communities.csv", encoding="utf-8-sig") if (src_dir / "topic_representative_communities.csv").exists() else pd.DataFrame()

        if abs(float(r) - float(anchor_r)) <= 1e-10:
            A = np.load(src_dir / "A_hat.npy")
            K = int(A.shape[1])
            t2r = {t: t for t in range(K)}
            sim_mat = np.eye(K, dtype=float)
            info = {
                "K": K,
                "n_common_vocab": int(A.shape[0]),
                "t2r": t2r,
                "avg_similarity": 1.0,
                "min_similarity": 1.0,
                "similarity_matrix": sim_mat,
            }
            is_anchor = True
        else:
            info = align_dir_to_dir(anchor_dir, src_dir, cfg)
            K = int(info["K"])
            t2r = {int(k): int(v) for k, v in info["t2r"].items()}
            sim_mat = np.asarray(info["similarity_matrix"], dtype=float)
            is_anchor = False

        # topic-level rows
        for tgt_t in range(K):
            ref_t = t2r.get(tgt_t, -1)
            sim_val = float(sim_mat[ref_t, tgt_t]) if (0 <= ref_t < sim_mat.shape[0] and 0 <= tgt_t < sim_mat.shape[1]) else np.nan
            rows_alignment.append({
                "segment_id": sid,
                "anchor_resolution": float(anchor_r),
                "resolution": float(r),
                "is_anchor": bool(is_anchor),
                "target_topic": int(tgt_t),
                "anchor_topic": int(ref_t),
                "similarity": sim_val,
                "n_common_vocab": int(info.get("n_common_vocab", np.nan)),
                "metric": cfg.metric,
            })

        rows_summary.append({
            "segment_id": sid,
            "anchor_resolution": float(anchor_r),
            "resolution": float(r),
            "is_anchor": bool(is_anchor),
            "avg_similarity": float(info.get("avg_similarity", np.nan)),
            "min_similarity": float(info.get("min_similarity", np.nan)),
            "n_common_vocab": int(info.get("n_common_vocab", np.nan)),
            "metric": cfg.metric,
            "src_dir": str(src_dir),
            "dst_dir": str(dst_dir),
        })

        # remap tables
        df_topics_aligned = remap_topics_table(df_topics, t2r, K) if not df_topics.empty else df_topics
        df_comm_aligned = remap_communities_topic_weights(df_comm, t2r, K) if not df_comm.empty else df_comm
        df_tcomm_aligned = remap_topic_representative_communities(df_tcomm, t2r) if not df_tcomm.empty else df_tcomm
        if not df_comm_aligned.empty and not df_topics_aligned.empty:
            df_comm_aligned = refresh_keyword_columns_from_topics(df_comm_aligned, df_topics_aligned)

        if not df_topics_aligned.empty:
            df_topics_aligned.to_csv(dst_dir / "topics_top_words.csv", index=False, encoding="utf-8-sig")
        if not df_comm_aligned.empty:
            df_comm_aligned.to_csv(dst_dir / "communities_topic_weights.csv", index=False, encoding="utf-8-sig")
        if not df_tcomm_aligned.empty:
            df_tcomm_aligned.to_csv(dst_dir / "topic_representative_communities.csv", index=False, encoding="utf-8-sig")

        meta = {
            "segment_id": sid,
            "resolution": float(r),
            "anchor_resolution": float(anchor_r),
            "metric": cfg.metric,
            "n_common_vocab": int(info.get("n_common_vocab", 0)),
            "avg_similarity": float(info.get("avg_similarity", np.nan)),
            "min_similarity": float(info.get("min_similarity", np.nan)),
            "target_to_anchor": {str(k): int(v) for k, v in sorted(t2r.items())},
            "note": "本目录中的 topics/communities 表已按本段 anchor 的 topic 编号重排；跨段不保证同一 topic_id 语义一致。",
            "source_dir": str(src_dir),
        }
        (dst_dir / "topic_alignment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if save_sim_matrix:
            np.save(dst_dir / "topic_similarity_to_anchor.npy", sim_mat)

        if verbose:
            print(f"[seg] r={r:.4f} -> avg_sim={meta['avg_similarity']:.4f}, min_sim={meta['min_similarity']:.4f}")

    # write segment alignment tables
    df_align = pd.DataFrame(rows_alignment).sort_values(["resolution", "target_topic"]).reset_index(drop=True)
    df_sum = pd.DataFrame(rows_summary).sort_values("resolution").reset_index(drop=True)
    df_align.to_csv(seg_out / "topic_alignment.csv", index=False, encoding="utf-8-sig")
    df_sum.to_csv(seg_out / "topic_alignment_summary.csv", index=False, encoding="utf-8-sig")

    run_meta = {
        "segment_id": sid,
        "segment_name": seg_name,
        "anchor_resolution": float(anchor_r),
        "selected_resolutions": [float(r) for r, _ in segment_items],
        "config": asdict(cfg),
    }
    (seg_out / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "segment_id": sid,
        "segment_name": seg_name,
        "r_start": float(seg["r_start"]),
        "r_end": float(seg["r_end"]),
        "n_resolutions": int(len(segment_items)),
        "anchor_resolution": float(anchor_r),
        "anchor_dir": str(anchor_dir),
        "out_dir": str(seg_out),
    }


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多分辨率 topic 分段对齐（每段一个 anchor）")
    p.add_argument("--k", type=int, required=True, help="全局主题数 K")
    p.add_argument("--topic-root", type=str, default=None,
                   help="topic_modeling_multi 的 K 目录；默认 out/topic_modeling_multi/K{K}")
    p.add_argument("--out-root", type=str, default=None,
                   help="输出根目录；默认 <topic-root>/aligned_segmented")

    p.add_argument("--metric", type=str, default="cosine", choices=["cosine", "js"], help="topic 相似度度量")
    p.add_argument("--topn-common-vocab", type=int, default=None, help="仅用各 topic 前N词构建共同词表（如 200）")
    p.add_argument("--min-common-vocab", type=int, default=50, help="共同词表最小要求")

    p.add_argument("--r-min", type=float, default=0.0001)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--include", type=float, nargs="*", default=None)
    p.add_argument("--resolutions", type=float, nargs="*", default=None)

    # segmentation params
    p.add_argument("--break-avg-thresh", type=float, default=0.85, help="相邻 avg_similarity 低于该值则切段")
    p.add_argument("--break-min-thresh", type=float, default=0.10, help="相邻 min_similarity 低于该值则切段")
    p.add_argument("--min-segment-size", type=int, default=2, help="过短段会与前一段合并（简单合并策略）")
    p.add_argument("--anchor-strategy", type=str, default="best-adjacent", choices=["best-adjacent", "middle"],
                   help="段内 anchor 选择：best-adjacent（默认）或 middle")

    p.add_argument("--save-sim-matrix", action="store_true", help="保存每个分辨率到 anchor 的 topic 相似度矩阵 .npy")
    p.add_argument("--save-adjacent-topic-rows", action="store_true", help="保存逐 topic 的相邻对齐记录 topic_alignment_adjacent.csv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--clean-segment-dirs", action="store_true", help="写入前删除 out_root 下已有 segment_* 目录（避免历史跳号目录残留）")
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

    out_root = Path(args.out_root) if args.out_root else (topic_root / "aligned_segmented")
    out_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[seg] topic_root={topic_root}")
        print(f"[seg] discovered={len(discovered)} selected={len(selected)}")
        print(f"[seg] out_root={out_root}")
        print("[seg] resolutions:", ", ".join(f"{r:.4f}" for r, _ in selected))

    if args.clean_segment_dirs:
        import shutil
        old_seg_dirs = [p for p in out_root.glob("segment_*_anchor_r*") if p.is_dir()]
        for p in old_seg_dirs:
            shutil.rmtree(p, ignore_errors=True)
        if verbose and old_seg_dirs:
            print(f"[seg] removed stale segment dirs: {len(old_seg_dirs)}")

    if args.dry_run:
        return

    cfg = AlignConfig(
        metric=args.metric,
        topn_common_vocab=args.topn_common_vocab,
        min_common_vocab=args.min_common_vocab,
    )

    # 1) compute adjacent alignment quality (for segmentation)
    df_adj_sum, df_adj_topic = compute_adjacent_alignment(selected, cfg, verbose=verbose)
    df_adj_sum = df_adj_sum.sort_values(["resolution_prev", "resolution_next"]).reset_index(drop=True)
    df_adj_sum.to_csv(out_root / "topic_alignment_adjacent_summary.csv", index=False, encoding="utf-8-sig")
    if args.save_adjacent_topic_rows:
        df_adj_topic = df_adj_topic.sort_values(["resolution_prev", "resolution_next", "target_topic_next"]).reset_index(drop=True)
        df_adj_topic.to_csv(out_root / "topic_alignment_adjacent.csv", index=False, encoding="utf-8-sig")

    # 2) segment
    segs = segment_resolutions(
        selected,
        df_adj_sum,
        break_avg_thresh=float(args.break_avg_thresh),
        break_min_thresh=float(args.break_min_thresh),
        min_segment_size=int(args.min_segment_size),
    )
    if verbose:
        print(f"\n[seg] segments={len(segs)} (break_avg<{args.break_avg_thresh}, break_min<{args.break_min_thresh})")

    # 3) align each segment and write outputs
    seg_rows = []
    for seg in segs:
        seg_info = align_segment_and_write(
            seg, selected, cfg, out_root,
            anchor_strategy=args.anchor_strategy,
            df_adj_sum=df_adj_sum,
            verbose=verbose,
            save_sim_matrix=bool(args.save_sim_matrix),
        )
        seg_rows.append(seg_info)

    df_segs = pd.DataFrame(seg_rows).sort_values(["segment_id", "r_start", "r_end"]).reset_index(drop=True)
    # segment_id / segment_name / out_dir 在写文件阶段已经确定，此处不再二次改号，避免与目录名不一致
    df_segs.to_csv(out_root / "segments.csv", index=False, encoding="utf-8-sig")

    # 4) write top-level run meta
    run_meta = {
        "topic_root": str(topic_root),
        "out_root": str(out_root),
        "selected_resolutions": [float(r) for r, _ in selected],
        "segmentation": {
            "break_avg_thresh": float(args.break_avg_thresh),
            "break_min_thresh": float(args.break_min_thresh),
            "min_segment_size": int(args.min_segment_size),
            "anchor_strategy": str(args.anchor_strategy),
        },
        "config": asdict(cfg),
        "args": vars(args),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[seg] =============================")
    print("[seg] done.")
    print(f"[seg] outputs -> {out_root}")
    print(f"[seg] segments -> {out_root / 'segments.csv'}")


if __name__ == "__main__":
    main()
