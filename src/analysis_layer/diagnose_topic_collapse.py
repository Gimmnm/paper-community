#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose topic-collapse / effective-topic behavior from communities_topic_weights.csv.

Supports:
1) Single file analysis
2) Batch analysis by recursively scanning a root directory (e.g., aligned_segmented)

Outputs CSV summaries and a few simple plots for downstream analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOPIC_COL_RE = re.compile(r"^topic_(\d+)$")
R_DIR_RE = re.compile(r"^r([0-9]+(?:\.[0-9]+)?)$")
SEG_DIR_RE = re.compile(r"^(segment_\d+_anchor_r.+)$")


def _find_topic_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if TOPIC_COL_RE.match(str(c)):
            cols.append(str(c))
    cols.sort(key=lambda x: int(TOPIC_COL_RE.match(x).group(1)))
    return cols


def _safe_entropy_rowwise(P: np.ndarray) -> np.ndarray:
    # P is row-normalized
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H = -(P * logP).sum(axis=1)
    return H


def _effective_from_prevalence(prev: np.ndarray) -> Tuple[float, float]:
    prev = np.asarray(prev, dtype=float)
    s = prev.sum()
    if s <= 0:
        return float("nan"), float("nan")
    prev = prev / s
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(np.where(prev > 0, prev * np.log(prev), 0.0)).sum()
    shannon_eff = float(np.exp(H))
    simpson_denom = float((prev ** 2).sum())
    simpson_eff = float(1.0 / simpson_denom) if simpson_denom > 0 else float("nan")
    return shannon_eff, simpson_eff


def _row_normalize(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    row_sums = P.sum(axis=1)
    Pn = np.zeros_like(P, dtype=float)
    nz = row_sums > 0
    if np.any(nz):
        Pn[nz] = P[nz] / row_sums[nz, None]
    return Pn, row_sums


def _infer_resolution_and_segment(csv_path: Path) -> Tuple[Optional[float], Optional[str]]:
    resolution = None
    segment_name = None
    for part in csv_path.parts:
        m = R_DIR_RE.match(part)
        if m:
            try:
                resolution = float(m.group(1))
            except ValueError:
                pass
        sm = SEG_DIR_RE.match(part)
        if sm:
            segment_name = sm.group(1)
    return resolution, segment_name


def _load_topwords_redundancy(topics_csv_path: Path) -> Dict[str, float]:
    out = {
        "topic_words_available": 0,
        "topic_pairs": np.nan,
        "topic_words_mean_pair_jaccard": np.nan,
        "topic_words_max_pair_jaccard": np.nan,
        "topic_words_pairs_jaccard_ge_0_3": np.nan,
        "topic_words_pairs_jaccard_ge_0_5": np.nan,
    }
    if not topics_csv_path.exists():
        return out

    try:
        tdf = pd.read_csv(topics_csv_path)
    except Exception:
        return out

    word_cols = [c for c in tdf.columns if re.fullmatch(r"word_\d+", str(c))]
    if word_cols:
        word_cols.sort(key=lambda c: int(str(c).split("_")[1]))
        sets = []
        for _, row in tdf.iterrows():
            ws = [str(row[c]).strip().lower() for c in word_cols if pd.notna(row[c]) and str(row[c]).strip()]
            sets.append(set(ws))
    elif "top_words" in tdf.columns:
        sets = []
        for _, row in tdf.iterrows():
            s = str(row["top_words"]) if pd.notna(row["top_words"]) else ""
            ws = [w.strip().lower() for w in s.split() if w.strip()]
            sets.append(set(ws[:10]))
    else:
        return out

    if len(sets) < 2:
        out["topic_words_available"] = int(len(sets))
        out["topic_pairs"] = 0
        return out

    jaccs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = len(a | b)
            jac = (len(a & b) / union) if union > 0 else 0.0
            jaccs.append(jac)
    jaccs_arr = np.array(jaccs, dtype=float)
    out.update(
        {
            "topic_words_available": int(len(sets)),
            "topic_pairs": int(len(jaccs)),
            "topic_words_mean_pair_jaccard": float(np.mean(jaccs_arr)) if len(jaccs_arr) else np.nan,
            "topic_words_max_pair_jaccard": float(np.max(jaccs_arr)) if len(jaccs_arr) else np.nan,
            "topic_words_pairs_jaccard_ge_0_3": int(np.sum(jaccs_arr >= 0.3)),
            "topic_words_pairs_jaccard_ge_0_5": int(np.sum(jaccs_arr >= 0.5)),
        }
    )
    return out


def analyze_communities_csv(csv_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(csv_path)
    topic_cols = _find_topic_cols(df)
    if not topic_cols:
        raise ValueError(f"No topic_* columns found in {csv_path}")

    topic_ids = [int(TOPIC_COL_RE.match(c).group(1)) for c in topic_cols]
    P = df[topic_cols].fillna(0.0).to_numpy(dtype=float)
    Pn, row_sums = _row_normalize(P)
    n_rows, K = P.shape

    # Row weights for prevalence / weighted shares (prefer n_papers if present)
    if "n_papers" in df.columns:
        row_weights = df["n_papers"].fillna(0).to_numpy(dtype=float)
        row_weights_name = "n_papers"
    else:
        row_weights = np.ones(n_rows, dtype=float)
        row_weights_name = "uniform"
    if row_weights.sum() <= 0:
        row_weights = np.ones(n_rows, dtype=float)
        row_weights_name = "uniform_fallback"

    # Top1 assignments from the topic columns themselves (robust to stale columns)
    top1_idx = np.argmax(Pn, axis=1)
    top1_topic_from_cols = np.array([topic_ids[i] for i in top1_idx], dtype=int)
    top1_weight_from_cols = Pn[np.arange(n_rows), top1_idx]

    # Optional consistency checks vs stored columns
    top1_topic_mismatch_rate = np.nan
    top1_weight_abs_diff_mean = np.nan
    if "top1_topic" in df.columns:
        try:
            stored = df["top1_topic"].to_numpy(dtype=int)
            top1_topic_mismatch_rate = float(np.mean(stored != top1_topic_from_cols))
        except Exception:
            pass
    if "top1_weight" in df.columns:
        try:
            storedw = df["top1_weight"].to_numpy(dtype=float)
            top1_weight_abs_diff_mean = float(np.mean(np.abs(storedw - top1_weight_from_cols)))
        except Exception:
            pass

    # Prevalence = weighted average over row-normalized topic vectors
    prevalence = np.average(Pn, axis=0, weights=row_weights)
    prevalence = prevalence / prevalence.sum() if prevalence.sum() > 0 else prevalence

    # Top1 shares (community-count and weighted)
    counts = pd.Series(top1_topic_from_cols).value_counts().to_dict()
    weighted_counts: Dict[int, float] = {}
    for tid in topic_ids:
        weighted_counts[tid] = float(row_weights[top1_topic_from_cols == tid].sum())
    total_weight = float(row_weights.sum())

    top1_share_by_count = np.array([counts.get(tid, 0) / n_rows for tid in topic_ids], dtype=float)
    top1_share_by_weight = np.array([
        (weighted_counts.get(tid, 0.0) / total_weight) if total_weight > 0 else np.nan for tid in topic_ids
    ], dtype=float)

    dominant_prevalence_idx = int(np.nanargmax(prevalence))
    dominant_count_idx = int(np.nanargmax(top1_share_by_count))
    dominant_weight_idx = int(np.nanargmax(top1_share_by_weight))

    # Effective topics (global prevalence) and per-community effective topics
    global_eff_shannon, global_eff_simpson = _effective_from_prevalence(prevalence)
    H_rows = _safe_entropy_rowwise(Pn)
    row_eff_shannon = np.exp(H_rows)
    row_eff_simpson = 1.0 / np.clip((Pn ** 2).sum(axis=1), 1e-15, None)

    # Row-sum diagnostics on original (possibly unnormalized) topic columns
    row_sum_err = np.abs(row_sums - 1.0)

    # Topic-word redundancy from sibling topics_top_words.csv
    sibling_topics = csv_path.with_name("topics_top_words.csv")
    tw_metrics = _load_topwords_redundancy(sibling_topics)

    resolution, segment_name = _infer_resolution_and_segment(csv_path)

    # Main one-row summary
    summary: Dict[str, object] = {
        "path": str(csv_path),
        "filename": csv_path.name,
        "resolution": resolution,
        "segment_name": segment_name,
        "n_communities": int(n_rows),
        "n_topics": int(K),
        "row_weight_field": row_weights_name,
        "row_weight_sum": total_weight,
        "mean_top1_weight": float(np.mean(top1_weight_from_cols)),
        "median_top1_weight": float(np.median(top1_weight_from_cols)),
        "q90_top1_weight": float(np.quantile(top1_weight_from_cols, 0.90)),
        "q99_top1_weight": float(np.quantile(top1_weight_from_cols, 0.99)),
        "global_effective_topics_shannon": global_eff_shannon,
        "global_effective_topics_simpson": global_eff_simpson,
        "community_effective_topics_shannon_mean": float(np.mean(row_eff_shannon)),
        "community_effective_topics_shannon_median": float(np.median(row_eff_shannon)),
        "community_effective_topics_simpson_mean": float(np.mean(row_eff_simpson)),
        "community_effective_topics_simpson_median": float(np.median(row_eff_simpson)),
        "rowsum_mean": float(np.mean(row_sums)),
        "rowsum_min": float(np.min(row_sums)),
        "rowsum_max": float(np.max(row_sums)),
        "rowsum_abs_err_mean": float(np.mean(row_sum_err)),
        "rowsum_abs_err_max": float(np.max(row_sum_err)),
        "top1_topic_mismatch_rate_vs_stored": top1_topic_mismatch_rate,
        "top1_weight_abs_diff_mean_vs_stored": top1_weight_abs_diff_mean,
        "dominant_topic_by_prevalence": int(topic_ids[dominant_prevalence_idx]),
        "dominant_topic_prevalence": float(prevalence[dominant_prevalence_idx]),
        "dominant_topic_by_top1_share_count": int(topic_ids[dominant_count_idx]),
        "dominant_topic_top1_share_count": float(top1_share_by_count[dominant_count_idx]),
        "dominant_topic_by_top1_share_weight": int(topic_ids[dominant_weight_idx]),
        "dominant_topic_top1_share_weight": float(top1_share_by_weight[dominant_weight_idx]),
    }
    summary.update(tw_metrics)

    # Long-form per-topic metrics (for plots / deeper analysis)
    rows = []
    for local_i, tid in enumerate(topic_ids):
        rows.append(
            {
                "path": str(csv_path),
                "resolution": resolution,
                "segment_name": segment_name,
                "topic_id": int(tid),
                "prevalence": float(prevalence[local_i]),
                "top1_share_count": float(top1_share_by_count[local_i]),
                "top1_share_weight": float(top1_share_by_weight[local_i]),
            }
        )
    per_topic_df = pd.DataFrame(rows)

    return per_topic_df, summary


def _scan_communities_csvs(root: Path) -> List[Path]:
    files = [p for p in root.rglob("communities_topic_weights.csv") if p.is_file()]
    # Prefer paths under r* folders, but keep others too
    return sorted(files, key=lambda p: str(p))


def _sort_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    # segment_name then numeric resolution if available
    out = df.copy()
    out["_seg"] = out["segment_name"].fillna("")
    out["_r"] = pd.to_numeric(out["resolution"], errors="coerce")
    out = out.sort_values(["_seg", "_r", "path"], na_position="last").drop(columns=["_seg", "_r"])
    return out.reset_index(drop=True)


def _plot_series(df_summary: pd.DataFrame, out_dir: Path, per_topic: Optional[pd.DataFrame] = None) -> None:
    if df_summary.empty:
        return
    sdf = df_summary.copy()
    sdf = sdf[pd.notna(sdf["resolution"])].copy()
    if sdf.empty:
        return
    sdf = sdf.sort_values(["segment_name", "resolution"], na_position="last")

    # Plot 1: effective topics vs resolution
    plt.figure(figsize=(8, 4.5))
    for seg, g in sdf.groupby("segment_name", dropna=False):
        x = g["resolution"].to_numpy(dtype=float)
        y1 = g["global_effective_topics_shannon"].to_numpy(dtype=float)
        y2 = g["global_effective_topics_simpson"].to_numpy(dtype=float)
        label_base = str(seg) if pd.notna(seg) else "(no segment)"
        plt.plot(x, y1, marker="o", markersize=3, linewidth=1, label=f"{label_base} | Shannon")
        plt.plot(x, y2, marker="s", markersize=3, linewidth=1, label=f"{label_base} | Simpson")
    plt.xlabel("resolution")
    plt.ylabel("effective topics")
    plt.title("Effective topics vs resolution")
    plt.grid(True, alpha=0.3)
    if len(sdf["segment_name"].dropna().unique()) <= 6:
        plt.legend(fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(out_dir / "effective_topics_vs_resolution.png", dpi=180)
    plt.close()

    # Plot 2: top1 dominance and sharpness vs resolution
    plt.figure(figsize=(8, 4.5))
    for seg, g in sdf.groupby("segment_name", dropna=False):
        x = g["resolution"].to_numpy(dtype=float)
        label_base = str(seg) if pd.notna(seg) else "(no segment)"
        plt.plot(x, g["dominant_topic_top1_share_count"].to_numpy(dtype=float), marker="o", markersize=3, linewidth=1,
                 label=f"{label_base} | dominant top1 share (count)")
        plt.plot(x, g["mean_top1_weight"].to_numpy(dtype=float), marker="s", markersize=3, linewidth=1,
                 label=f"{label_base} | mean top1 weight")
    plt.xlabel("resolution")
    plt.ylabel("share / weight")
    plt.title("Top1 dominance and sharpness vs resolution")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    if len(sdf["segment_name"].dropna().unique()) <= 6:
        plt.legend(fontsize=7, ncol=1)
    plt.tight_layout()
    plt.savefig(out_dir / "top1_dominance_vs_resolution.png", dpi=180)
    plt.close()

    # Plot 3: row-sum check (sanity)
    plt.figure(figsize=(8, 4.5))
    for seg, g in sdf.groupby("segment_name", dropna=False):
        x = g["resolution"].to_numpy(dtype=float)
        y = g["rowsum_abs_err_max"].to_numpy(dtype=float)
        label_base = str(seg) if pd.notna(seg) else "(no segment)"
        plt.plot(x, y, marker="o", markersize=3, linewidth=1, label=f"{label_base}")
    plt.xlabel("resolution")
    plt.ylabel("max |row_sum - 1|")
    plt.title("Topic-row normalization check vs resolution")
    plt.grid(True, alpha=0.3)
    if len(sdf["segment_name"].dropna().unique()) <= 6:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "rowsum_abs_err_max_vs_resolution.png", dpi=180)
    plt.close()

    # Optional Plot 4: per-topic prevalence curves (if provided)
    if per_topic is not None and not per_topic.empty:
        p = per_topic.copy()
        p = p[pd.notna(p["resolution"])].copy()
        if not p.empty:
            p = p.sort_values(["segment_name", "topic_id", "resolution"], na_position="last")
            # If segmented, draw one plot per segment to avoid clutter
            seg_values = list(p["segment_name"].dropna().unique())
            if seg_values:
                for seg in seg_values:
                    gseg = p[p["segment_name"] == seg]
                    plt.figure(figsize=(8, 4.5))
                    for tid, gt in gseg.groupby("topic_id"):
                        plt.plot(gt["resolution"].to_numpy(dtype=float), gt["prevalence"].to_numpy(dtype=float),
                                 marker="o", markersize=2.5, linewidth=1, label=f"T{int(tid)}")
                    plt.xlabel("resolution")
                    plt.ylabel("prevalence")
                    plt.title(f"Topic prevalence vs resolution | {seg}")
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=7, ncol=2)
                    plt.tight_layout()
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(seg))
                    plt.savefig(out_dir / f"topic_prevalence_vs_resolution_{safe_name}.png", dpi=180)
                    plt.close()
            else:
                plt.figure(figsize=(8, 4.5))
                for tid, gt in p.groupby("topic_id"):
                    plt.plot(gt["resolution"].to_numpy(dtype=float), gt["prevalence"].to_numpy(dtype=float),
                             marker="o", markersize=2.5, linewidth=1, label=f"T{int(tid)}")
                plt.xlabel("resolution")
                plt.ylabel("prevalence")
                plt.title("Topic prevalence vs resolution")
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=7, ncol=2)
                plt.tight_layout()
                plt.savefig(out_dir / "topic_prevalence_vs_resolution.png", dpi=180)
                plt.close()


def _print_single_summary(summary: Dict[str, object], per_topic_df: pd.DataFrame) -> None:
    print("=== Topic collapse diagnostics (single file) ===")
    keys = [
        "path", "n_communities", "n_topics", "row_weight_field",
        "mean_top1_weight", "median_top1_weight", "q90_top1_weight", "q99_top1_weight",
        "global_effective_topics_shannon", "global_effective_topics_simpson",
        "community_effective_topics_simpson_mean", "community_effective_topics_simpson_median",
        "dominant_topic_by_prevalence", "dominant_topic_prevalence",
        "dominant_topic_by_top1_share_count", "dominant_topic_top1_share_count",
        "dominant_topic_by_top1_share_weight", "dominant_topic_top1_share_weight",
        "rowsum_abs_err_mean", "rowsum_abs_err_max",
        "topic_words_mean_pair_jaccard", "topic_words_max_pair_jaccard",
        "topic_words_pairs_jaccard_ge_0_3", "topic_words_pairs_jaccard_ge_0_5",
    ]
    for k in keys:
        if k in summary:
            print(f"{k}: {summary[k]}")
    print("\nPer-topic (prevalence / top1 shares):")
    show = per_topic_df[["topic_id", "prevalence", "top1_share_count", "top1_share_weight"]].copy()
    print(show.sort_values("topic_id").to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose topic collapse/effective topics from communities_topic_weights.csv")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="Path to a single communities_topic_weights.csv")
    g.add_argument("--root", type=str, help="Root directory to recursively scan for communities_topic_weights.csv")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (default: beside input or under root/_diagnostics_topic)")
    ap.add_argument("--save-per-topic", action="store_true", help="Save long-form per-topic metrics CSV")
    ap.add_argument("--save-plots", action="store_true", help="Save diagnostic plots (batch mode particularly useful)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        out_dir = Path(args.out_dir) if args.out_dir else in_path.parent / "_diagnostics_topic"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_topic_df, summary = analyze_communities_csv(in_path)
        _print_single_summary(summary, per_topic_df)

        pd.DataFrame([summary]).to_csv(out_dir / "topic_collapse_summary.csv", index=False, encoding="utf-8-sig")
        if args.save_per_topic:
            per_topic_df.to_csv(out_dir / "topic_collapse_per_topic.csv", index=False, encoding="utf-8-sig")
        # Single-file plots don't need x=resolution; skip unless resolution exists and user asks
        if args.save_plots:
            _plot_series(pd.DataFrame([summary]), out_dir, per_topic_df if args.save_per_topic else None)
        print(f"[done] outputs saved to: {out_dir}")
        return

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "_diagnostics_topic"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _scan_communities_csvs(root)
    if not files:
        raise FileNotFoundError(f"No communities_topic_weights.csv found under {root}")
    print(f"[diag] found communities csv files: {len(files)}")

    summaries: List[Dict[str, object]] = []
    per_topic_frames: List[pd.DataFrame] = []
    failed: List[Tuple[str, str]] = []

    for p in files:
        try:
            if args.verbose:
                print(f"[diag] analyzing {p}")
            per_topic_df, summary = analyze_communities_csv(p)
            summaries.append(summary)
            per_topic_frames.append(per_topic_df)
        except Exception as e:
            failed.append((str(p), repr(e)))
            print(f"[diag] skip {p}: {e}")

    df_summary = _sort_summary_df(pd.DataFrame(summaries)) if summaries else pd.DataFrame()
    df_summary.to_csv(out_dir / "topic_collapse_diagnostics.csv", index=False, encoding="utf-8-sig")
    print(f"[diag] wrote: {out_dir / 'topic_collapse_diagnostics.csv'}")

    df_per_topic = pd.concat(per_topic_frames, ignore_index=True) if per_topic_frames else pd.DataFrame()
    if args.save_per_topic and not df_per_topic.empty:
        df_per_topic.to_csv(out_dir / "topic_collapse_per_topic_long.csv", index=False, encoding="utf-8-sig")
        print(f"[diag] wrote: {out_dir / 'topic_collapse_per_topic_long.csv'}")

    if args.save_plots and not df_summary.empty:
        _plot_series(df_summary, out_dir, df_per_topic if args.save_per_topic else None)
        print(f"[diag] plots saved under: {out_dir}")

    if failed:
        fail_df = pd.DataFrame(failed, columns=["path", "error"])
        fail_df.to_csv(out_dir / "topic_collapse_diagnostics_failed.csv", index=False, encoding="utf-8-sig")
        print(f"[diag] failed files: {len(failed)} (see topic_collapse_diagnostics_failed.csv)")

    # Helpful text summary
    if not df_summary.empty:
        msg_cols = [
            "global_effective_topics_shannon",
            "dominant_topic_top1_share_count",
            "mean_top1_weight",
        ]
        med = df_summary[msg_cols].median(numeric_only=True).to_dict()
        print("[diag] median summary:")
        for k, v in med.items():
            print(f"  - {k}: {v:.4f}")

    print(f"[done] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
