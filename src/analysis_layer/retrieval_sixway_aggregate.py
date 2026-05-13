"""
Six-way retrieval comparison: keyword + vector NN + four community partitions.

The offline benchmark (``experiment-retrieval-benchmark``) writes one JSON object per
line with ``methods.keyword``, ``methods.vector_nn``, and ``methods.community_bundle``.
Keyword and vector scores do not depend on which community algorithm produced the
labels; we report them once using rows from a **canonical** ``run_id`` (first match
among ``leiden_cpm``, ``leiden``, ``louvain``, ``coarse_kmeans``). Community-bundle
metrics are aggregated **per** ``run_id`` so the four partition strategies are compared.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from foundation_layer.project_paths import REPO_ROOT

# Order for picking baseline rows (keyword / vector — partition-agnostic).
_BASELINE_RUN_ID_ORDER = ("leiden_cpm", "leiden", "louvain", "coarse_kmeans")

_COMMUNITY_RUN_IDS = ("leiden_cpm", "leiden", "louvain", "coarse_kmeans")


def _comparison_metrics_dirs(cr: Path, comparison_run_tags: Optional[Sequence[str]]) -> List[Path]:
    """Same semantics as ``evaluation_metrics._comparison_metrics_dirs`` (local copy)."""
    tags = comparison_run_tags
    if tags is None:
        raw = os.environ.get("PC_EVAL_COMPARISON_RUN_TAGS", "").strip()
        if raw:
            tags = [t.strip() for t in raw.split(",") if t.strip()]
    if tags:
        out: List[Path] = []
        for t in tags:
            sub = cr / str(t).strip()
            if sub.is_dir() and (sub / "metrics.jsonl").is_file():
                out.append(sub)
        return out
    return [p for p in sorted(cr.iterdir()) if p.is_dir() and (p / "metrics.jsonl").is_file()]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _method_block(row: Dict[str, Any], name: str) -> Dict[str, Any]:
    methods = row.get("methods") or {}
    m = methods.get(name)
    return m if isinstance(m, dict) else {}


def _stats(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    vv = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vv:
        return None, None, 0
    mu = float(sum(vv) / len(vv))
    sd = float(statistics.pstdev(vv)) if len(vv) > 1 else 0.0
    return mu, sd, len(vv)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _pick_baseline_run_id(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    found = {str(r.get("run_id", "")).strip() for r in rows if str(r.get("run_id", "")).strip()}
    for rid in _BASELINE_RUN_ID_ORDER:
        if rid in found:
            return rid
    if found:
        return sorted(found)[0]
    return None


def _aggregate_method(rows: Sequence[Dict[str, Any]], method_key: str) -> Dict[str, Any]:
    cos = [_safe_float(_method_block(r, method_key).get("mean_cosine_to_seed")) for r in rows]
    hops = [_safe_float(_method_block(r, method_key).get("mean_shortest_path_hops")) for r in rows]
    tsec = [_safe_float(_method_block(r, method_key).get("time_sec")) for r in rows]
    top1 = [_safe_float(_method_block(r, method_key).get("topic_top1_match_rate_vs_seed_community")) for r in rows]
    kw_tf = [_safe_float(_method_block(r, method_key).get("mean_keyword_tfidf")) for r in rows]
    c_mu, c_sd, c_n = _stats(cos)
    h_mu, h_sd, h_n = _stats(hops)
    t_mu, t_sd, t_n = _stats(tsec)
    p_mu, p_sd, p_n = _stats(top1)
    k_mu, k_sd, k_n = _stats(kw_tf)
    return {
        "n_lines": len(rows),
        "mean_cosine_mean": c_mu,
        "mean_cosine_std": c_sd,
        "mean_cosine_n": c_n,
        "mean_hops_mean": h_mu,
        "mean_hops_std": h_sd,
        "mean_hops_n": h_n,
        "time_sec_mean": t_mu,
        "time_sec_std": t_sd,
        "time_sec_n": t_n,
        "mean_keyword_tfidf_mean": k_mu,
        "mean_keyword_tfidf_std": k_sd,
        "mean_keyword_tfidf_n": k_n,
        "topic_top1_match_mean": p_mu,
        "topic_top1_match_std": p_sd,
        "topic_top1_match_n": p_n,
    }


def _pairwise_stats(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[float], int]:
    vals: List[Optional[float]] = []
    for r in rows:
        pj = r.get("pairwise_jaccard") or {}
        if isinstance(pj, dict):
            vals.append(_safe_float(pj.get(key)))
    return _stats(vals)


def aggregate_sixway_for_tag_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    comparison_run_tag: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (long_rows, meta) where long_rows are CSV-ready dicts with columns:
    comparison_run_tag, method, baseline_run_id_used (nullable), partition_run_id (nullable), ...
    """
    baseline_rid = _pick_baseline_run_id(rows)
    baseline_rows = [r for r in rows if baseline_rid and str(r.get("run_id", "")).strip() == baseline_rid]
    if not baseline_rows and rows:
        baseline_rows = list(rows)
        baseline_rid = str(rows[0].get("run_id", "")).strip() or None

    long_rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "comparison_run_tag": comparison_run_tag,
        "baseline_run_id_for_keyword_vector": baseline_rid,
        "n_jsonl_lines": len(rows),
    }

    for label, key in (("keyword", "keyword"), ("vector_nn", "vector_nn")):
        agg = _aggregate_method(baseline_rows, key)
        long_rows.append(
            {
                "comparison_run_tag": comparison_run_tag,
                "method": label,
                "partition_run_id": "",
                "baseline_run_id_used": baseline_rid or "",
                **agg,
            }
        )

    for rid in _COMMUNITY_RUN_IDS:
        sub = [r for r in rows if str(r.get("run_id", "")).strip() == rid]
        agg = _aggregate_method(sub, "community_bundle")
        long_rows.append(
            {
                "comparison_run_tag": comparison_run_tag,
                "method": f"community_bundle:{rid}",
                "partition_run_id": rid,
                "baseline_run_id_used": "",
                **agg,
            }
        )

    pj_keys = ("keyword__vector_nn", "keyword__community_bundle", "vector_nn__community_bundle")
    pj_summary: Dict[str, Any] = {}
    for pk in pj_keys:
        mu, sd, n = _pairwise_stats(baseline_rows, pk)
        pj_summary[f"baseline_{pk}_mean"] = mu
        pj_summary[f"baseline_{pk}_std"] = sd
        pj_summary[f"baseline_{pk}_n"] = n
    meta["pairwise_jaccard_on_baseline_rows"] = pj_summary

    return long_rows, meta


def _flatten_wide(tag: str, long_rows: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "comparison_run_tag": tag,
        "baseline_run_id_for_keyword_vector": meta.get("baseline_run_id_for_keyword_vector"),
        "n_jsonl_lines": meta.get("n_jsonl_lines"),
    }
    pj = meta.get("pairwise_jaccard_on_baseline_rows") or {}
    # Keys in ``pj`` are already ``baseline_<pair>_{mean,std,n}``; do not prefix again.
    for k, v in pj.items():
        row[k] = v

    metric_keys = (
        "n_lines",
        "mean_cosine_mean",
        "mean_cosine_std",
        "mean_cosine_n",
        "mean_hops_mean",
        "mean_hops_std",
        "mean_hops_n",
        "time_sec_mean",
        "time_sec_std",
        "time_sec_n",
        "mean_keyword_tfidf_mean",
        "mean_keyword_tfidf_std",
        "mean_keyword_tfidf_n",
        "topic_top1_match_mean",
        "topic_top1_match_std",
        "topic_top1_match_n",
    )
    for lr in long_rows:
        method = str(lr.get("method", ""))
        prefix = method.replace(":", "_").replace(".", "_")
        for mk in metric_keys:
            if mk in lr:
                row[f"{prefix}__{mk}"] = lr.get(mk)
    return row


def write_retrieval_sixway_for_repo(
    repo_root: Path,
    out_dir: Path,
    comparison_run_tags: Optional[Sequence[str]] = None,
    *,
    plot_rankings: bool = True,
    plot_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Read ``out/comparison_runs/<tag>/metrics.jsonl`` and write
    ``comparison_retrieval_sixway_long.csv``, ``comparison_retrieval_sixway_wide.csv``,
    ``comparison_retrieval_sixway_meta.json`` under ``out_dir``.

    When ``plot_rankings`` is True, also writes ranking bar charts under
    ``plot_dir`` (default ``out_dir / "retrieval_sixway_plots"``).

    Always (when six-way CSVs are written successfully) also writes per-resolution
    retrieval tables and ``comparison_retrieval_resolution_meta.json``; when
    ``plot_rankings`` is True, adds resolution curve and best-pick bar PNGs under
    ``out_dir / "retrieval_resolution_plots"``.
    """
    repo_root = Path(repo_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cr = repo_root / "out" / "comparison_runs"
    dirs = _comparison_metrics_dirs(cr, comparison_run_tags)
    if not dirs:
        return {
            "ok": False,
            "reason": "no comparison_runs/*/metrics.jsonl (run experiment-retrieval-benchmark first)",
        }

    all_long: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for d in dirs:
        tag = d.name
        rows = _load_jsonl(d / "metrics.jsonl")
        if not rows:
            continue
        long_part, meta = aggregate_sixway_for_tag_rows(rows, comparison_run_tag=tag)
        all_long.extend(long_part)
        wide_rows.append(_flatten_wide(tag, long_part, meta))
        meta["comparison_run_tag"] = tag
        metas.append(meta)

    if not all_long:
        return {"ok": False, "reason": "metrics.jsonl files were empty"}

    long_path = out_dir / "comparison_retrieval_sixway_long.csv"
    fieldnames_long = sorted({k for r in all_long for k in r.keys()})
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_long)
        w.writeheader()
        for r in all_long:
            w.writerow(r)

    wide_path = out_dir / "comparison_retrieval_sixway_wide.csv"
    fieldnames_wide = sorted({k for r in wide_rows for k in r.keys()})
    with wide_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_wide)
        w.writeheader()
        for r in wide_rows:
            w.writerow(r)

    meta_path = out_dir / "comparison_retrieval_sixway_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "description": (
                    "Six-way retrieval: keyword + vector_nn from baseline run_id rows only; "
                    "four community_bundle series keyed by partition run_id. "
                    f"Baseline preference order: {_BASELINE_RUN_ID_ORDER}."
                ),
                "runs": metas,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result: Dict[str, Any] = {
        "ok": True,
        "long_csv": str(long_path),
        "wide_csv": str(wide_path),
        "meta_json": str(meta_path),
        "n_tags": len(wide_rows),
    }
    try:
        from analysis_layer.retrieval_resolution_analysis import (
            plot_resolution_curves_and_best_bars,
            write_resolution_tables_for_tags,
        )

        res_br = write_resolution_tables_for_tags(repo_root, out_dir, comparison_run_tags)
        result["resolution_breakdown"] = res_br
        if plot_rankings and res_br.get("ok"):
            try:
                rp_dir = out_dir / "retrieval_resolution_plots"
                rp = plot_resolution_curves_and_best_bars(
                    Path(res_br["resolution_long_csv"]),
                    Path(res_br["best_resolution_long_csv"]),
                    rp_dir,
                )
                result["resolution_curve_plots"] = rp.get("resolution_curve_plots", [])
                result["best_pick_bar_plots"] = rp.get("best_pick_bar_plots", [])
            except Exception as exc:  # pragma: no cover - optional viz deps
                result["resolution_curve_plots"] = []
                result["best_pick_bar_plots"] = []
                result["resolution_plots_error"] = str(exc)
        elif plot_rankings and not res_br.get("ok"):
            result["resolution_curve_plots"] = []
            result["best_pick_bar_plots"] = []
    except Exception as exc:  # pragma: no cover - optional analysis
        result["resolution_breakdown"] = {"ok": False, "error": str(exc)}
        if plot_rankings:
            result["resolution_curve_plots"] = []
            result["best_pick_bar_plots"] = []

    if plot_rankings:
        try:
            from analysis_layer.retrieval_sixway_plots import (
                plot_retrieval_sixway_metric_absolute_banded_lines_csv,
                plot_retrieval_sixway_metric_lines_csv,
                plot_retrieval_sixway_rankings_csv,
            )

            pd = Path(plot_dir) if plot_dir is not None else (out_dir / "retrieval_sixway_plots")
            plots = plot_retrieval_sixway_rankings_csv(long_path, pd)
            line_plots = plot_retrieval_sixway_metric_lines_csv(long_path, pd)
            abs_plots = plot_retrieval_sixway_metric_absolute_banded_lines_csv(long_path, pd)
            result["ranking_plots"] = [str(p) for p in plots]
            result["metric_line_plots"] = [str(p) for p in line_plots]
            result["metric_absolute_banded_plots"] = [str(p) for p in abs_plots]
            result["metric_absolute_facets_plots"] = result["metric_absolute_banded_plots"]
        except Exception as exc:  # pragma: no cover - optional viz deps
            result["ranking_plots"] = []
            result["ranking_plots_error"] = str(exc)
    return result


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build 6-way retrieval comparison CSVs from comparison_runs metrics.jsonl")
    p.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "out" / "experiment_eval"))
    p.add_argument(
        "--comparison-run-tag",
        action="append",
        default=None,
        metavar="TAG",
        help="Limit to these out/comparison_runs/<TAG>/ (repeatable); default all subdirs with metrics.jsonl",
    )
    p.add_argument("--no-sixway-plots", action="store_true", help="Do not write ranking PNGs")
    args = p.parse_args(argv)
    tags = [str(t).strip() for t in (args.comparison_run_tag or []) if str(t).strip()] or None
    res = write_retrieval_sixway_for_repo(
        Path(args.repo_root),
        Path(args.out_dir),
        tags,
        plot_rankings=not bool(args.no_sixway_plots),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_cli()
