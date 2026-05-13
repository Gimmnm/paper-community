"""
Per-resolution retrieval summaries and ``best r'' picks from ``metrics.jsonl``.

The six-way aggregate pools all breakpoint resolutions (and seeds) into one mean per method.
This module keeps **resolution-wise** means (averaged over seeds only) and picks, per community
partition ``run_id``, the resolution that maximizes a small composite score, then emits a
``best-of'' comparison row for each algorithm at its own ``r*``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from analysis_layer.retrieval_sixway_aggregate import (
    _COMMUNITY_RUN_IDS,
    _aggregate_method,
    _pick_baseline_run_id,
    _safe_float,
)

_RKEY_DIGITS = 6


def _r_key(res: Any) -> float:
    return float(round(float(res), _RKEY_DIGITS))


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


def _rows_for_run_resolution(rows: Sequence[Dict[str, Any]], run_id: str, r_key: float) -> List[Dict[str, Any]]:
    rid = str(run_id).strip()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if str(r.get("run_id", "")).strip() != rid:
            continue
        if _r_key(r.get("resolution_effective")) != float(r_key):
            continue
        out.append(r)
    return out


def build_resolution_long_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    comparison_run_tag: str,
) -> List[Dict[str, Any]]:
    """
    One CSV row per (``comparison_run_tag``, ``run_id``, ``resolution_effective``, ``method``)
    after averaging over seeds in ``metrics.jsonl``.
    """
    keys: set[Tuple[str, float]] = set()
    for r in rows:
        rid = str(r.get("run_id", "")).strip()
        if not rid:
            continue
        keys.add((rid, _r_key(r.get("resolution_effective"))))

    long_rows: List[Dict[str, Any]] = []
    for rid, rk in sorted(keys, key=lambda t: (t[0], t[1])):
        sub = _rows_for_run_resolution(rows, rid, rk)
        if not sub:
            continue
        r0 = sub[0]
        r_req = r0.get("resolution_requested")
        for method_key, method_label in (
            ("keyword", "keyword"),
            ("vector_nn", "vector_nn"),
            ("community_bundle", "community_bundle"),
        ):
            agg = _aggregate_method(sub, method_key)
            row: Dict[str, Any] = {
                "comparison_run_tag": comparison_run_tag,
                "run_id": rid,
                "resolution_effective": rk,
                "resolution_requested": float(r_req) if r_req is not None else None,
                "method": method_label,
                **agg,
            }
            long_rows.append(row)
    return long_rows


def _finite_series(xs: List[Optional[float]]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def _min_max_norm(vals: Sequence[Optional[float]], *, lower_is_better: bool) -> List[float]:
    """Per-index normalized to [0,1] across finite entries; NaN if no span."""
    finite = _finite_series(list(vals))
    if not finite:
        return [float("nan")] * len(vals)
    lo, hi = min(finite), max(finite)
    span = hi - lo if hi > lo else 1.0
    out: List[float] = []
    for v in vals:
        fv = _safe_float(v)
        if fv is None:
            out.append(float("nan"))
            continue
        t = (float(fv) - lo) / span
        if lower_is_better:
            t = 1.0 - t
        out.append(float(t))
    return out


def _composite_scores_for_community_curve(
    res_rows: Sequence[Dict[str, Any]],
    *,
    run_id: str,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """
    For one ``run_id``, collect sorted ``resolution_effective`` and composite score
    (mean of per-metric min–max scores across resolutions; hops & time lower-is-better).
    Returns (r_keys_sorted, scores, aggs_per_r) where ``aggs_per_r`` aligns with r_keys_sorted.
    """
    rids = {(_r_key(r.get("resolution_effective"))) for r in res_rows if str(r.get("run_id", "")).strip() == run_id}
    r_sorted = sorted(rids)
    aggs: List[Dict[str, Any]] = []
    for rk in r_sorted:
        sub = _rows_for_run_resolution(res_rows, run_id, rk)
        aggs.append(_aggregate_method(sub, "community_bundle"))

    def col(key: str) -> List[Optional[float]]:
        return [a.get(key) for a in aggs]

    z_cos = _min_max_norm(col("mean_cosine_mean"), lower_is_better=False)
    z_kw = _min_max_norm(col("mean_keyword_tfidf_mean"), lower_is_better=False)
    z_top = _min_max_norm(col("topic_top1_match_mean"), lower_is_better=False)
    z_hop = _min_max_norm(col("mean_hops_mean"), lower_is_better=True)
    z_time = _min_max_norm(col("time_sec_mean"), lower_is_better=True)

    scores: List[float] = []
    for i in range(len(r_sorted)):
        parts = [z_cos[i], z_hop[i], z_time[i]]
        if not (z_kw[i] != z_kw[i]):  # not nan
            parts.append(z_kw[i])
        if not (z_top[i] != z_top[i]):
            parts.append(z_top[i])
        finite = [p for p in parts if p == p]
        scores.append(float(sum(finite) / len(finite)) if finite else float("nan"))
    return r_sorted, scores, aggs


def build_best_of_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    comparison_run_tag: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Pick best ``resolution_effective`` per community ``run_id`` via composite on ``community_bundle``.
    Keyword / vector rows use the baseline partition's best ``r*`` (same composite on baseline
    community curve). Emits six rows compatible with six-way ``method`` naming where applicable.
    """
    baseline_rid = _pick_baseline_run_id(rows) or ""
    r_bl, sc_bl, _agg_bl = _composite_scores_for_community_curve(rows, run_id=baseline_rid) if baseline_rid else ([], [], [])
    best_r_baseline: Optional[float] = None
    best_sc_baseline: Optional[float] = None
    if r_bl and sc_bl:
        j = max(range(len(sc_bl)), key=lambda i: sc_bl[i] if sc_bl[i] == sc_bl[i] else -1.0)
        if sc_bl[j] == sc_bl[j]:
            best_r_baseline = float(r_bl[j])
            best_sc_baseline = float(sc_bl[j])

    meta: Dict[str, Any] = {
        "comparison_run_tag": comparison_run_tag,
        "baseline_run_id_for_keyword_vector": baseline_rid,
        "baseline_best_resolution_effective": best_r_baseline,
        "baseline_best_composite": best_sc_baseline,
        "composite_note": (
            "Per run_id, composite = mean of per-metric scores; each metric min–max normalized "
            "across resolutions for that run_id (hops & time_sec: smaller raw is better → higher score)."
        ),
    }

    best_by_rid: Dict[str, float] = {}
    best_sc_by_rid: Dict[str, float] = {}
    for rid in _COMMUNITY_RUN_IDS:
        if rid not in {str(r.get("run_id", "")).strip() for r in rows}:
            continue
        rs, sc, _ = _composite_scores_for_community_curve(rows, run_id=rid)
        if not rs or not sc:
            continue
        j = max(range(len(sc)), key=lambda i: sc[i] if sc[i] == sc[i] else -1.0)
        if sc[j] == sc[j]:
            best_by_rid[rid] = float(rs[j])
            best_sc_by_rid[rid] = float(sc[j])

    meta["best_resolution_effective_by_run_id"] = best_by_rid
    meta["best_composite_by_run_id"] = best_sc_by_rid

    def _agg_at(rid: str, rk: float, method_key: str) -> Dict[str, Any]:
        sub = _rows_for_run_resolution(rows, rid, rk)
        return _aggregate_method(sub, method_key) if sub else {}

    long_rows: List[Dict[str, Any]] = []
    r_kw_vec = float(best_r_baseline) if best_r_baseline is not None else None

    for label, key in (("keyword", "keyword"), ("vector_nn", "vector_nn")):
        agg: Dict[str, Any] = {}
        if baseline_rid and r_kw_vec is not None:
            agg = _agg_at(baseline_rid, r_kw_vec, key)
        long_rows.append(
            {
                "comparison_run_tag": comparison_run_tag,
                "method": label,
                "partition_run_id": "",
                "baseline_run_id_used": baseline_rid,
                "resolution_effective_used": r_kw_vec,
                "resolution_pick_policy": "baseline_community_bundle_composite_max",
                "composite_score_at_pick": best_sc_baseline,
                **{f"at_pick__{k}": v for k, v in agg.items()},
            }
        )

    for rid in _COMMUNITY_RUN_IDS:
        rk = best_by_rid.get(rid)
        if rk is None:
            continue
        sc = best_sc_by_rid.get(rid)
        agg = _agg_at(rid, float(rk), "community_bundle")
        long_rows.append(
            {
                "comparison_run_tag": comparison_run_tag,
                "method": f"community_bundle:{rid}",
                "partition_run_id": rid,
                "baseline_run_id_used": "",
                "resolution_effective_used": float(rk),
                "resolution_pick_policy": "partition_community_bundle_composite_max",
                "composite_score_at_pick": sc,
                **{f"at_pick__{k}": v for k, v in agg.items()},
            }
        )

    return long_rows, meta


def write_resolution_tables_for_tags(
    repo_root: Path,
    out_dir: Path,
    comparison_run_tags: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Write ``comparison_retrieval_resolution_long.csv`` and ``comparison_retrieval_best_resolution_long.csv``."""
    from analysis_layer.retrieval_sixway_aggregate import _comparison_metrics_dirs

    cr = Path(repo_root) / "out" / "comparison_runs"
    dirs = _comparison_metrics_dirs(cr, comparison_run_tags)
    if not dirs:
        return {"ok": False, "reason": "no comparison_runs metrics"}

    all_res: List[Dict[str, Any]] = []
    all_best: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for d in dirs:
        tag = d.name
        rows = _load_jsonl(d / "metrics.jsonl")
        if not rows:
            continue
        res_long = build_resolution_long_rows(rows, comparison_run_tag=tag)
        all_res.extend(res_long)
        best_rows, meta = build_best_of_rows(rows, comparison_run_tag=tag)
        all_best.extend(best_rows)
        meta["comparison_run_tag"] = tag
        metas.append(meta)

    if not all_res:
        return {"ok": False, "reason": "empty resolution breakdown"}

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "comparison_retrieval_resolution_long.csv"
    fn_res = sorted({k for r in all_res for k in r.keys()})
    with res_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn_res)
        w.writeheader()
        for r in all_res:
            w.writerow(r)

    best_path = out_dir / "comparison_retrieval_best_resolution_long.csv"
    fn_best = sorted({k for r in all_best for k in r.keys()})
    if not fn_best:
        fn_best = [
            "comparison_run_tag",
            "method",
            "partition_run_id",
            "baseline_run_id_used",
            "resolution_effective_used",
            "resolution_pick_policy",
            "composite_score_at_pick",
        ]
    with best_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn_best)
        w.writeheader()
        for r in all_best:
            w.writerow(r)

    meta_path = out_dir / "comparison_retrieval_resolution_meta.json"
    meta_path.write_text(
        json.dumps({"runs": metas}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "resolution_long_csv": str(res_path),
        "best_resolution_long_csv": str(best_path),
        "resolution_meta_json": str(meta_path),
    }


def _bundle_resolution_series(
    trows: Sequence[Dict[str, str]],
    *,
    partition_run_id: str,
) -> Optional[Dict[str, Any]]:
    """
    From resolution-long CSV rows for one tag, extract ``community_bundle`` points for
    ``partition_run_id``, sorted by ``resolution_effective``.

    Returns normalized series (same construction as composite ``r*``) plus raw values for tooltips.
    """
    rid = str(partition_run_id).strip()
    sub = [
        dict(r)
        for r in trows
        if str(r.get("run_id", "")).strip() == rid and str(r.get("method", "")).strip() == "community_bundle"
    ]
    if not sub:
        return None
    sub.sort(key=lambda r: float(_safe_float(r.get("resolution_effective")) or 0.0))
    xs: List[float] = []
    cos: List[Optional[float]] = []
    hop: List[Optional[float]] = []
    tsec: List[Optional[float]] = []
    kw: List[Optional[float]] = []
    top: List[Optional[float]] = []
    kw_n: List[int] = []
    top_n: List[int] = []
    for r in sub:
        rv = _safe_float(r.get("resolution_effective"))
        if rv is None:
            continue
        xs.append(float(rv))
        cos.append(_safe_float(r.get("mean_cosine_mean")))
        hop.append(_safe_float(r.get("mean_hops_mean")))
        tsec.append(_safe_float(r.get("time_sec_mean")))
        kw.append(_safe_float(r.get("mean_keyword_tfidf_mean")))
        top.append(_safe_float(r.get("topic_top1_match_mean")))
        try:
            kw_n.append(int(float(str(r.get("mean_keyword_tfidf_n", "0") or "0"))))
        except (TypeError, ValueError):
            kw_n.append(0)
        try:
            top_n.append(int(float(str(r.get("topic_top1_match_n", "0") or "0"))))
        except (TypeError, ValueError):
            top_n.append(0)
    if not xs:
        return None

    z_cos = _min_max_norm(cos, lower_is_better=False)
    z_hop = _min_max_norm(hop, lower_is_better=True)
    z_time = _min_max_norm(tsec, lower_is_better=True)
    z_kw = _min_max_norm(kw, lower_is_better=False)
    z_top = _min_max_norm(top, lower_is_better=False)

    has_kw = any(n > 0 for n in kw_n) and any(k is not None and k == k for k in kw)
    has_top = any(n > 0 for n in top_n) and any(t is not None and t == t for t in top)

    composite: List[float] = []
    for i in range(len(xs)):
        parts = [z_cos[i], z_hop[i], z_time[i]]
        if has_kw and not (z_kw[i] != z_kw[i]):
            parts.append(z_kw[i])
        if has_top and not (z_top[i] != z_top[i]):
            parts.append(z_top[i])
        finite = [p for p in parts if p == p]
        composite.append(float(sum(finite) / len(finite)) if finite else float("nan"))

    return {
        "xs": xs,
        "z_cos": z_cos,
        "z_hop": z_hop,
        "z_time": z_time,
        "z_kw": z_kw,
        "z_top": z_top,
        "composite": composite,
        "has_kw": bool(has_kw),
        "has_top": bool(has_top),
        "raw_cos": cos,
        "raw_hop": hop,
        "raw_time": tsec,
    }


def plot_resolution_curves_and_best_bars(
    resolution_long_csv: Path,
    best_long_csv: Path,
    plot_dir: Path,
) -> Dict[str, List[str]]:
    """
    Per ``comparison_run_tag``:

    1. **Bundle vs resolution** — four panels (community algorithms). In each panel, lines are
       **min–max normalized across resolutions** for ``community_bundle`` (cosine, hops, time;
       optional TF–IDF / topic when n>0), matching the composite used for ``r*``. A thick black
       line shows the **composite**; a red vertical line marks picked ``r*``.

    2. **Best pick vs metrics** — same style as six-way ``retrieval_metric_lines``: x = metrics,
       y = **column-normalized** ``at_pick__*`` means across the six methods (hops/time flipped).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    from analysis_layer.retrieval_sixway_plots import (
        _METHOD_ORDER,
        _col_normalize,
        _method_display,
    )

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _read_csv(p: Path) -> List[Dict[str, str]]:
        if not p.is_file():
            return []
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    res_rows = _read_csv(resolution_long_csv)
    best_rows = _read_csv(best_long_csv)
    if not res_rows:
        return {"resolution_curve_plots": [], "best_pick_bar_plots": []}

    by_tag_res: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in res_rows:
        by_tag_res[str(r.get("comparison_run_tag", "")).strip()].append(dict(r))
    by_tag_best: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in best_rows:
        by_tag_best[str(r.get("comparison_run_tag", "")).strip()].append(dict(r))

    curve_paths: List[str] = []
    bar_paths: List[str] = []

    _rc_style = {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#bbbbbb",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 10,
        "legend.fontsize": 8,
    }

    for tag in sorted(by_tag_res.keys()):
        trows = by_tag_res[tag]
        present = {rid for rid in _COMMUNITY_RUN_IDS if _bundle_resolution_series(trows, partition_run_id=rid)}
        if not present:
            continue
        ncols = 2
        nrows = int(math.ceil(len(present) / ncols))
        with plt.rc_context(_rc_style):
            # Wider/taller figure + room outside each panel for legend (avoids label overlap).
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(8.8 * ncols, 6.8 * nrows),
                squeeze=False,
                constrained_layout=False,
            )
            ax_flat = axes.ravel()
            cmap = plt.get_cmap("tab10")
            metric_styles = [
                ("z_cos", "cosine (↑)", cmap(0)),
                ("z_hop", "hops (↑ z)", cmap(1)),
                ("z_time", "time (↑ z)", cmap(2)),
            ]

            pi = 0
            for rid in _COMMUNITY_RUN_IDS:
                if rid not in present:
                    continue
                ax = ax_flat[pi]
                pack = _bundle_resolution_series(trows, partition_run_id=rid)
                assert pack is not None
                xs = pack["xs"]
                x_arr = np.asarray(xs, dtype=np.float64)
                use_log = bool(np.all(x_arr > 0) and float(np.nanmax(x_arr)) / float(np.nanmin(x_arr)) >= 25.0)
                if use_log:
                    ax.set_xscale("log")

                for zkey, lab, col in metric_styles:
                    ys = pack[zkey]  # type: ignore[index]
                    ax.plot(xs, ys, "-o", color=col, linewidth=1.45, markersize=4, label=lab, alpha=0.9, zorder=3)

                if pack["has_kw"] and any(v == v for v in pack["z_kw"]):  # type: ignore[operator]
                    ax.plot(
                        xs,
                        pack["z_kw"],
                        "-o",
                        color=cmap(4),
                        linewidth=1.25,
                        markersize=3,
                        label="kw TF-IDF (↑)",
                        alpha=0.92,
                        zorder=3,
                    )
                if pack["has_top"] and any(v == v for v in pack["z_top"]):  # type: ignore[operator]
                    ax.plot(
                        xs,
                        pack["z_top"],
                        "-o",
                        color=cmap(5),
                        linewidth=1.25,
                        markersize=3,
                        label="topic top1 (↑)",
                        alpha=0.92,
                        zorder=3,
                    )

                z_comp = pack["composite"]
                ax.plot(
                    xs,
                    z_comp,
                    color="#111111",
                    linewidth=2.75,
                    marker="s",
                    markersize=5,
                    label="composite",
                    zorder=5,
                )

                ax.set_ylim(-0.06, 1.06)
                ax.axhline(0.0, color="#cccccc", linewidth=0.7, zorder=1)
                ax.grid(True, axis="y", alpha=0.38)
                ax.grid(True, axis="x", alpha=0.22)
                ax.set_ylabel("norm. 0–1 (within panel)", fontsize=9)
                ax.set_xlabel("resolution" + (" (log)" if use_log else ""), fontsize=9)
                ax.set_title(_method_display(f"community_bundle:{rid}"), fontsize=11, fontweight="600")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
                for lab in ax.get_xticklabels():
                    lab.set_rotation(28)
                    lab.set_ha("right")
                    lab.set_fontsize(8)
                ax.tick_params(axis="y", labelsize=9)

                rk_star: Optional[float] = None
                for br in by_tag_best.get(tag, []):
                    if str(br.get("method", "")).strip() == f"community_bundle:{rid}":
                        rk_star = _safe_float(br.get("resolution_effective_used"))
                        break
                if rk_star is not None:
                    ax.axvline(
                        float(rk_star),
                        color="#c44e52",
                        linestyle="--",
                        linewidth=1.45,
                        alpha=0.92,
                        label="picked r*",
                        zorder=4,
                    )
                    best_i = min(range(len(xs)), key=lambda i: abs(float(xs[i]) - float(rk_star)))
                    y_mark = float("nan")
                    if best_i < len(z_comp) and z_comp[best_i] == z_comp[best_i]:
                        y_mark = float(z_comp[best_i])
                    if y_mark == y_mark:
                        ax.scatter(
                            [float(rk_star)],
                            [y_mark],
                            color="#c44e52",
                            s=78,
                            zorder=7,
                            edgecolors="white",
                            linewidths=1.1,
                        )

                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=8,
                    frameon=True,
                    framealpha=0.97,
                    fancybox=True,
                    borderaxespad=0.35,
                    labelspacing=0.65,
                    handlelength=2.5,
                    handletextpad=0.55,
                    borderpad=0.45,
                )
                pi += 1

            for j in range(pi, len(ax_flat)):
                ax_flat[j].set_visible(False)

            fig.suptitle(
                f"community_bundle vs resolution — {tag}\n"
                "Traces: per-metric min–max along r (hops/time: lower raw is better). "
                "Black composite = mean of available traces (same objective as r*).",
                fontsize=10,
                fontweight="600",
                y=0.995,
            )
            # Reserve right margin for out-of-axes legends; extra hspace for titles/x labels.
            fig.subplots_adjust(left=0.07, right=0.66, top=0.90, bottom=0.12, hspace=0.52, wspace=0.48)
            cp = plot_dir / f"retrieval_resolution_bundle_vs_r__{tag.replace('/', '_')}.png"
            fig.savefig(cp, dpi=175, bbox_inches="tight", pad_inches=0.18)
            plt.close(fig)
        curve_paths.append(str(cp))

        # --- best pick: same spirit as six-way metric line chart (column-normalize across methods)
        by_m = {str(r.get("method", "")).strip(): r for r in by_tag_best.get(tag, []) if str(r.get("method", "")).strip()}
        ordered: List[Dict[str, str]] = []
        for m in _METHOD_ORDER:
            if m in by_m:
                ordered.append(by_m[m])
        if len(ordered) >= 2:
            def ap(row: Dict[str, str], k: str) -> Optional[float]:
                return _safe_float(row.get(f"at_pick__{k}"))

            cos_raw = [ap(r, "mean_cosine_mean") for r in ordered]
            hops_raw = [ap(r, "mean_hops_mean") for r in ordered]
            t_raw = [ap(r, "time_sec_mean") for r in ordered]
            top_raw = [ap(r, "topic_top1_match_mean") for r in ordered]
            topic_n = []
            for r in ordered:
                try:
                    topic_n.append(int(float(str(r.get("at_pick__topic_top1_match_n", "0") or "0"))))
                except (TypeError, ValueError):
                    topic_n.append(0)
            kw_raw = [ap(r, "mean_keyword_tfidf_mean") for r in ordered]
            kw_n = []
            for r in ordered:
                try:
                    kw_n.append(int(float(str(r.get("at_pick__mean_keyword_tfidf_n", "0") or "0"))))
                except (TypeError, ValueError):
                    kw_n.append(0)

            has_topic = any(n > 0 for n in topic_n)
            has_kw_tfidf = any(n > 0 for n in kw_n)
            metric_keys = ["cosine"]
            lower_flags = [False]
            raw_cols: List[List[Optional[float]]] = [cos_raw]
            if has_kw_tfidf:
                metric_keys.append("kw_tfidf")
                lower_flags.append(False)
                raw_cols.append(kw_raw)
            metric_keys.extend(["hops", "time_s"])
            lower_flags.extend([True, True])
            raw_cols.extend([hops_raw, t_raw])
            if has_topic:
                metric_keys.append("topic_top1")
                lower_flags.append(False)
                raw_cols.append(top_raw)

            norm_cols: List[List[float]] = []
            for raw, low in zip(raw_cols, lower_flags):
                norm_cols.append(_col_normalize(raw, lower_is_better=low))

            labels = [_method_display(str(r.get("method", ""))) for r in ordered]
            x = np.arange(len(metric_keys))
            with plt.rc_context(_rc_style):
                cmap2 = plt.get_cmap("tab10")
                fig2, ax2 = plt.subplots(figsize=(11.0, 6.2), constrained_layout=False)
                for i, lab in enumerate(labels):
                    y = [norm_cols[j][i] for j in range(len(metric_keys))]
                    ax2.plot(x, y, marker="o", linewidth=2.0, markersize=6, label=lab, color=cmap2(i % 10))

                ax2.set_xticks(x)
                ax2.set_xticklabels(metric_keys, fontsize=10, rotation=22, ha="right")
                ax2.set_ylabel("normalized score (within metric, best=1)", fontsize=10)
                ax2.set_ylim(-0.05, 1.05)
                ax2.axhline(0.0, color="#cccccc", linewidth=0.8)
                ax2.grid(True, axis="y", alpha=0.35)
                ax2.tick_params(axis="y", labelsize=9)
                ax2.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=9,
                    frameon=True,
                    framealpha=0.96,
                    labelspacing=0.7,
                    handlelength=2.2,
                    borderpad=0.45,
                )
                fig2.suptitle(f"At picked r* — multi-metric (column-normalized) — {tag}", fontsize=11, fontweight="600", y=0.98)
                ax2.set_title(
                    "Each metric column min–max across the six methods (hops & time: lower raw is better → higher score).",
                    fontsize=9,
                    color="#444444",
                    pad=12,
                )
                fig2.subplots_adjust(left=0.10, right=0.68, top=0.84, bottom=0.18)
                bp = plot_dir / f"retrieval_best_pick_metric_lines__{tag.replace('/', '_')}.png"
                fig2.savefig(bp, dpi=175, bbox_inches="tight", pad_inches=0.2)
                plt.close(fig2)
            bar_paths.append(str(bp))

    return {"resolution_curve_plots": curve_paths, "best_pick_bar_plots": bar_paths}
