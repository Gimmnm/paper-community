"""
Ranking bar charts and retrieval metric plots from ``comparison_retrieval_sixway_long.csv``.

Includes: ranked bar panels, **normalized** multi-metric line chart (shared 0–1 y), and **absolute**
multi-metric **Cartesian** line chart: one axes, x = metrics, y = stacked bands (each band its own
vertical scale mapping raw means; ranges annotated).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

# Canonical row order in ``comparison_retrieval_sixway_long.csv`` (matches aggregate output).
_METHOD_ORDER = (
    "keyword",
    "vector_nn",
    "community_bundle:leiden_cpm",
    "community_bundle:leiden",
    "community_bundle:louvain",
    "community_bundle:coarse_kmeans",
)


def _method_display(method: str) -> str:
    m = str(method).strip()
    if m.startswith("community_bundle:"):
        rid = m.split(":", 1)[1]
        short = {"leiden_cpm": "CPM", "leiden": "RB", "louvain": "Louvain", "coarse_kmeans": "coarseKM"}.get(
            rid, rid
        )
        return f"bundle_{short}"
    return m


def _finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    import math

    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _ranked_bars(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    higher_better: bool,
) -> None:
    import numpy as np

    pairs = list(zip(labels, values))
    pairs.sort(key=lambda t: t[1], reverse=higher_better)
    labs = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    y = np.arange(len(labs))
    ax.barh(y, vals, color="#4C72B0", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=8, color="#333333")


def plot_retrieval_sixway_rankings_csv(
    long_csv: Path,
    plot_dir: Path,
) -> List[Path]:
    """
    For each distinct ``comparison_run_tag``, write one PNG with ranked horizontal bars
    for mean cosine (higher better), mean hops (lower), wall time (lower), mean TF--IDF query--doc
    relevance when present, and topic match if present.
    """
    long_csv = Path(long_csv)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_tag: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    with long_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tag = str(row.get("comparison_run_tag", "")).strip()
            if not tag:
                continue
            by_tag[tag].append(dict(row))

    written: List[Path] = []
    for tag, rows in sorted(by_tag.items()):
        labels: List[str] = []
        cos: List[float] = []
        hops: List[float] = []
        tsec: List[float] = []
        top1: List[float] = []
        topic_n: List[int] = []
        kw_tf: List[float] = []
        kw_n: List[int] = []
        for row in rows:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            labels.append(_method_display(method))
            cos.append(float(_finite_float(row.get("mean_cosine_mean")) or 0.0))
            hops.append(float(_finite_float(row.get("mean_hops_mean")) or 0.0))
            tsec.append(float(_finite_float(row.get("time_sec_mean")) or 0.0))
            tv = _finite_float(row.get("topic_top1_match_mean"))
            top1.append(float(tv) if tv is not None else float("nan"))
            try:
                topic_n.append(int(float(str(row.get("topic_top1_match_n", "0") or "0"))))
            except (TypeError, ValueError):
                topic_n.append(0)
            kv = _finite_float(row.get("mean_keyword_tfidf_mean"))
            kw_tf.append(float(kv) if kv is not None else float("nan"))
            try:
                kw_n.append(int(float(str(row.get("mean_keyword_tfidf_n", "0") or "0"))))
            except (TypeError, ValueError):
                kw_n.append(0)

        has_topic = any(n > 0 for n in topic_n)
        has_kw_tfidf = any(n > 0 for n in kw_n)
        ncols = 3 + (1 if has_kw_tfidf else 0) + (1 if has_topic else 0)
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), constrained_layout=True)
        ax_list: List[Any] = list(axes) if ncols > 1 else [axes]

        ax_i = 0
        _ranked_bars(
            ax_list[ax_i],
            labels,
            cos,
            title="mean_cosine_mean (↑ better)",
            xlabel="cosine",
            higher_better=True,
        )
        ax_i += 1
        _ranked_bars(
            ax_list[ax_i],
            labels,
            hops,
            title="mean_hops_mean (↓ closer in graph)",
            xlabel="hops",
            higher_better=False,
        )
        ax_i += 1
        _ranked_bars(
            ax_list[ax_i],
            labels,
            tsec,
            title="time_sec_mean (↓ faster)",
            xlabel="seconds",
            higher_better=False,
        )
        ax_i += 1
        if has_kw_tfidf:
            _ranked_bars(
                ax_list[ax_i],
                labels,
                kw_tf,
                title="mean_keyword_tfidf_mean (↑ better)",
                xlabel="TF-IDF dot",
                higher_better=True,
            )
            ax_i += 1
        if has_topic:
            _ranked_bars(
                ax_list[ax_i],
                labels,
                top1,
                title="topic_top1_match_mean (↑ better)",
                xlabel="rate",
                higher_better=True,
            )

        fig.suptitle(f"Retrieval six-way rankings — {tag}", fontsize=11)
        out = plot_dir / f"retrieval_rankings__{tag.replace('/', '_')}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)
    return written


def _col_normalize(vals: List[Optional[float]], *, lower_is_better: bool) -> List[float]:
    """Map finite values to [0,1] within-column; NaN for missing. higher score = better outcome."""
    xs: List[float] = []
    for v in vals:
        fv = _finite_float(v)
        xs.append(float(fv) if fv is not None else float("nan"))
    finite = [x for x in xs if x == x]
    if not finite:
        return [float("nan")] * len(vals)
    lo, hi = min(finite), max(finite)
    span = hi - lo if hi > lo else 1.0
    out: List[float] = []
    for x in xs:
        if x != x:
            out.append(float("nan"))
            continue
        if lower_is_better:
            out.append((hi - x) / span)
        else:
            out.append((x - lo) / span)
    return out


def plot_retrieval_sixway_metric_lines_csv(long_csv: Path, plot_dir: Path) -> List[Path]:
    """
    Line chart: x-axis = scalar metrics (cosine, TF--IDF relevance when present, hops, time, topic when present),
    one colored line per retrieval method (six lines). Values are **column-normalized to 0..1**
    within each metric so different units share one y-axis (subtitle explains).
    """
    long_csv = Path(long_csv)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_tag: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    with long_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tag = str(row.get("comparison_run_tag", "")).strip()
            if not tag:
                continue
            by_tag[tag].append(dict(row))

    written: List[Path] = []
    for tag, rows in sorted(by_tag.items()):
        labels: List[str] = []
        cos_raw: List[Optional[float]] = []
        hops_raw: List[Optional[float]] = []
        t_raw: List[Optional[float]] = []
        top_raw: List[Optional[float]] = []
        topic_n: List[int] = []
        kw_raw: List[Optional[float]] = []
        kw_n: List[int] = []
        for row in rows:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            labels.append(_method_display(method))
            cos_raw.append(_finite_float(row.get("mean_cosine_mean")))
            hops_raw.append(_finite_float(row.get("mean_hops_mean")))
            t_raw.append(_finite_float(row.get("time_sec_mean")))
            top_raw.append(_finite_float(row.get("topic_top1_match_mean")))
            try:
                topic_n.append(int(float(str(row.get("topic_top1_match_n", "0") or "0"))))
            except (TypeError, ValueError):
                topic_n.append(0)
            kw_raw.append(_finite_float(row.get("mean_keyword_tfidf_mean")))
            try:
                kw_n.append(int(float(str(row.get("mean_keyword_tfidf_n", "0") or "0"))))
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

        x = np.arange(len(metric_keys))
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        cmap = plt.get_cmap("tab10")
        for i, lab in enumerate(labels):
            y = [norm_cols[j][i] for j in range(len(metric_keys))]
            ax.plot(x, y, marker="o", linewidth=2.0, label=lab, color=cmap(i % 10))

        ax.set_xticks(x, metric_keys, fontsize=10)
        ax.set_ylabel("normalized score (within metric, best=1)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="#cccccc", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        fig.suptitle(f"Retrieval six-way — {tag}", fontsize=11)
        ax.set_title(
            "Each metric column min–max normalized separately (hops & time: lower raw is better → higher score).",
            fontsize=8,
            color="#444444",
        )

        out = plot_dir / f"retrieval_metric_lines__{tag.replace('/', '_')}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def plot_retrieval_sixway_metric_absolute_banded_lines_csv(long_csv: Path, plot_dir: Path) -> List[Path]:
    """
    One **Cartesian** figure per ``comparison_run_tag``: same layout idea as the normalized metric
    line chart (x = metrics, one colored polyline per retrieval method), but **y is not unified**
    across metrics.

    The y-axis is split into horizontal **bands** (one per metric). Within band ``j``, the vertical
    position is linear in the **raw** CSV mean between that metric's min and max across methods.
    For metrics where **smaller raw is better** (hops, wall time), the linear map is **flipped** inside
    the band so that better outcomes sit higher, while the band still spans the same absolute
    ``[min, max]`` range (annotated on the right). Lines connect consecutive metrics; **do not compare
    vertical position across bands**.

    A short footnote clarifies band-wise scaling and flip semantics.
    """
    long_csv = Path(long_csv)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_tag: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    with long_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tag = str(row.get("comparison_run_tag", "")).strip()
            if not tag:
                continue
            by_tag[tag].append(dict(row))

    written: List[Path] = []
    for tag, rows in sorted(by_tag.items()):
        row_by_method: Dict[str, Dict[str, str]] = {}
        for row in rows:
            m = str(row.get("method", "")).strip()
            if m:
                row_by_method[m] = row
        present = [m for m in _METHOD_ORDER if m in row_by_method]
        if not present:
            continue

        has_topic = any(
            int(float(str(row_by_method[m].get("topic_top1_match_n", "0") or "0"))) > 0 for m in present
        )
        has_kw_tfidf = any(
            int(float(str(row_by_method[m].get("mean_keyword_tfidf_n", "0") or "0"))) > 0 for m in present
        )

        specs: List[Tuple[str, str, str, bool]] = [
            ("mean_cosine_mean", "mean cosine → seed", "cosine", False),
        ]
        if has_kw_tfidf:
            specs.append(("mean_keyword_tfidf_mean", "mean TF–IDF (query vs doc)", "TF-IDF", False))
        specs.extend(
            [
                ("mean_hops_mean", "mean graph hops", "hops", True),
                ("time_sec_mean", "wall time", "time s", True),
            ]
        )
        if has_topic:
            specs.append(("topic_top1_match_mean", "topic top1 vs seed comm.", "topic", False))

        active: List[Tuple[str, str, str, bool]] = []
        for csv_key, title, ylab, lower in specs:
            vals = [_finite_float(row_by_method[m].get(csv_key)) for m in present]
            if any(v is not None for v in vals):
                active.append((csv_key, title, ylab, lower))
        if not active:
            continue

        n_m = len(active)
        vmin: List[float] = []
        vmax: List[float] = []
        for csv_key, _, _, _ in active:
            col = [_finite_float(row_by_method[m].get(csv_key)) for m in present]
            finite = [float(c) for c in col if c is not None]
            lo, hi = min(finite), max(finite)
            if hi <= lo:
                hi = lo + 1e-15
            vmin.append(lo)
            vmax.append(hi)

        x = np.arange(n_m, dtype=float)
        fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.55 * n_m + 2.8)))
        cmap = plt.get_cmap("tab10")
        for mi, m in enumerate(present):
            ys: List[float] = []
            skip = False
            for j, (csv_key, _, _, lower_is_better) in enumerate(active):
                v = _finite_float(row_by_method[m].get(csv_key))
                if v is None:
                    skip = True
                    break
                t = (float(v) - vmin[j]) / (vmax[j] - vmin[j])
                if lower_is_better:
                    t = 1.0 - t
                ys.append(float(j) + 0.06 + 0.88 * t)
            if skip:
                continue
            ax.plot(x, ys, "o-", color=cmap(mi % 10), linewidth=1.75, markersize=7, label=_method_display(m))

        for j in range(n_m):
            ax.axhline(float(j), color="#e8e8e8", linewidth=1.0, zorder=0)
            ax.axhline(float(j) + 1.0, color="#e8e8e8", linewidth=1.0, zorder=0)
            _, _, ylab, lower = active[j]
            flip = "\n(↓ better→↑)" if lower else ""
            ax.text(
                float(n_m - 1) + 0.28,
                float(j) + 0.5,
                f"{ylab}{flip}\n[{vmin[j]:.4g},{vmax[j]:.4g}]",
                fontsize=7,
                va="center",
                ha="left",
                color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([t[2] for t in active], fontsize=10)
        ax.set_xlim(-0.35, float(n_m - 1) + 0.25)
        ax.set_ylim(-0.08, float(n_m - 1) + 1.08)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.35)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
        ax.set_xlabel("metric", fontsize=9)
        ax.set_ylabel("band = one metric (↑ better within band; hops/time: smaller raw → higher)", fontsize=8, color="#555555")

        foot = (
            "Each band spans raw [min,max] across methods (right). Within a band, y is linear in raw value; "
            "for hops and time_s, the map is flipped so smaller (better) raw sits higher. Do not compare height across bands."
        )
        fig.text(0.5, 0.02, foot, ha="center", fontsize=7, color="#333333", wrap=True)
        fig.suptitle(f"Retrieval six-way — absolute means (banded lines) — {tag}", fontsize=11, y=0.98)
        fig.subplots_adjust(bottom=0.12, right=0.74, top=0.88)

        out = plot_dir / f"retrieval_metric_absolute_lines__{tag.replace('/', '_')}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written
