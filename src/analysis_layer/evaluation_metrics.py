from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithm_layer.community import DEFAULT_RESOLUTION_NDIGITS, membership_filename
from data_layer.breakpoint_schedule import topic_run_sweep_folder_for_run_id
from data_layer.experiment_contracts import ExperimentEvaluationBundle, ExperimentMetricRow

_RT_EPS = 1e-12


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _summary_arrays(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True).item()
    return d


def _load_optional_scorecard(leiden_dir: Path) -> Dict[str, Optional[float]]:
    candidates = [
        Path(leiden_dir) / "eval" / "scorecard.json",
        Path(leiden_dir).parent / "eval" / "scorecard.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        return {
            "retrieval_score": _safe_float(d.get("retrieval_score")),
            "topic_score": _safe_float(d.get("topic_score")),
            "practical_score": _safe_float(d.get("practical_score")),
        }
    return {"retrieval_score": None, "topic_score": None, "practical_score": None}


def _rel_under_leiden_dir(leiden_dir: Path, file_path: Path) -> Optional[str]:
    try:
        ld = leiden_dir.resolve()
        fp = file_path.resolve()
        rel = fp.relative_to(ld)
        return rel.as_posix()
    except (ValueError, OSError):
        return None


def _load_eval_artifacts_json(leiden_dir: Path) -> Dict[str, str]:
    p = leiden_dir / "eval" / "artifacts.json"
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for key in ("sweep_diagnostics", "breakpoints", "layered"):
        v = raw.get(key)
        if v is None or str(v).strip() == "":
            continue
        out[key] = str(v).strip().replace("\\", "/")
    return out


def _resolve_artifact(leiden_dir: Path, rel: str) -> Optional[Path]:
    rel = rel.strip().replace("\\", "/")
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        return None
    cand = (leiden_dir / rel).resolve()
    try:
        cand.relative_to(leiden_dir.resolve())
    except ValueError:
        return None
    return cand if cand.is_file() else None


def discover_eval_plot_paths(leiden_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (sweep, breakpoints, layered) paths relative to leiden_dir."""
    ld = Path(leiden_dir)
    declared = _load_eval_artifacts_json(ld)
    sweep: Optional[str] = None
    bp: Optional[str] = None
    layered: Optional[str] = None

    if "sweep_diagnostics" in declared:
        p = _resolve_artifact(ld, declared["sweep_diagnostics"])
        if p is not None:
            sweep = _rel_under_leiden_dir(ld, p)
    if "breakpoints" in declared:
        p = _resolve_artifact(ld, declared["breakpoints"])
        if p is not None:
            bp = _rel_under_leiden_dir(ld, p)
    if "layered" in declared:
        p = _resolve_artifact(ld, declared["layered"])
        if p is not None:
            layered = _rel_under_leiden_dir(ld, p)

    if sweep is None:
        for rel in ("eval/sweep_diagnostics.png", "eval/sweep_diagnostics.pdf"):
            p = ld / rel
            if p.is_file():
                sweep = _rel_under_leiden_dir(ld, p)
                break
        if sweep is None:
            for name in ("sweep_diagnostics.png", "sweep_diagnostics.pdf"):
                p = ld / name
                if p.is_file():
                    sweep = _rel_under_leiden_dir(ld, p)
                    break

    if bp is None:
        eval_dir = ld / "eval"
        for rel in ("eval/breakpoints_overview.png",):
            p = ld / rel
            if p.is_file():
                bp = _rel_under_leiden_dir(ld, p)
                break
        if bp is None and eval_dir.is_dir():
            for cand in sorted(eval_dir.glob("*breakpoints*.png")) + sorted(eval_dir.glob("*breakpoint*.png")):
                if cand.is_file():
                    bp = _rel_under_leiden_dir(ld, cand)
                    break
        if bp is None:
            for cand in sorted(ld.glob("*breakpoints*.png")) + sorted(ld.glob("*breakpoint*.png")):
                if cand.is_file():
                    bp = _rel_under_leiden_dir(ld, cand)
                    break

    if layered is None:
        p_layer = ld / "eval" / "layer_link_counts.png"
        if p_layer.is_file():
            layered = _rel_under_leiden_dir(ld, p_layer)
        if layered is None:
            found: Optional[Path] = None
            for pattern in ("*layered*.png", "*_layered.svg", "*cpm_layered*.png"):
                for cand in sorted(ld.glob(pattern)):
                    if cand.is_file():
                        found = cand
                        break
                if found is not None:
                    break
            if found is not None:
                layered = _rel_under_leiden_dir(ld, found)

    return sweep, bp, layered


def _mean_pairwise_jaccard(record: Dict[str, Any]) -> Optional[float]:
    pj = record.get("pairwise_jaccard")
    if not isinstance(pj, dict):
        return None
    vals: List[float] = []
    for v in pj.values():
        fv = _safe_float(v)
        if fv is not None:
            vals.append(fv)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _comparison_metrics_dirs(cr: Path, comparison_run_tags: Optional[Sequence[str]]) -> List[Path]:
    """Ordered list of directories under comparison_runs containing metrics.jsonl."""
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


def aggregate_retrieval_score_from_comparison_runs(
    repo_root: Path,
    *,
    run_id: str,
    time_window: str = "all",
    comparison_run_tags: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """
    Mean over JSONL rows: average of the three pairwise Jaccard similarities between
    keyword / vector_nn / community_bundle top-k sets.

    Directories: ``comparison_run_tags`` if set, else env ``PC_EVAL_COMPARISON_RUN_TAGS``
    (comma-separated), else every ``out/comparison_runs/*/metrics.jsonl``.
    """
    cr = Path(repo_root) / "out" / "comparison_runs"
    if not cr.is_dir():
        return None
    rid = str(run_id).strip()
    tw = str(time_window).strip()
    line_scores: List[float] = []
    for sub in _comparison_metrics_dirs(cr, comparison_run_tags):
        mp = sub / "metrics.jsonl"
        try:
            text = mp.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(obj.get("run_id", "")).strip() != rid:
                continue
            if str(obj.get("time_window", "all")).strip() != tw:
                continue
            m = _mean_pairwise_jaccard(obj)
            if m is not None:
                line_scores.append(m)
    if not line_scores:
        return None
    return float(sum(line_scores) / len(line_scores))


def _topic_summary_ok_mean_top1_weights(summary_csv: Path) -> List[float]:
    """Collect ``mean_top1_weight`` from rows that are not hard failures.

    ``topic_modeling_multi`` writes ``skipped_existing`` when ``--skip-existing`` skips a fit but
    still records metrics from disk; those rows must count toward overview ``topic_score``.
    """
    out: List[float] = []
    bad_status = frozenset({"error", "failed"})
    try:
        with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                st = str(row.get("status", "ok")).strip().lower()
                if st in bad_status:
                    continue
                v = _safe_float(row.get("mean_top1_weight"))
                if v is not None:
                    out.append(v)
    except OSError:
        pass
    return out


def aggregate_topic_score_from_topic_runs(repo_root: Path, *, run_id: str) -> Optional[float]:
    """Mean ``mean_top1_weight`` over usable rows in ``out/topic_runs/<sweep>/K*/summary_multires.csv``."""
    sweep = topic_run_sweep_folder_for_run_id(str(run_id))
    if not sweep:
        return None
    base = Path(repo_root) / "out" / "topic_runs" / sweep
    if not base.is_dir():
        return None
    vals: List[float] = []
    for summary_path in sorted(base.glob("K*/summary_multires.csv")):
        vals.extend(_topic_summary_ok_mean_top1_weights(summary_path))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _estimate_membership_load_times(leiden_dir: Path, resolutions: np.ndarray) -> np.ndarray:
    """Wall times for ``np.load(membership_r*.npy)`` per resolution (for cache-hit sweeps with stored time=0)."""
    ld = Path(leiden_dir)
    out = np.full(int(resolutions.size), np.nan, dtype=np.float64)
    for i in range(int(resolutions.size)):
        r = float(resolutions[i])
        p = ld / membership_filename(r, ndigits=DEFAULT_RESOLUTION_NDIGITS)
        if not p.is_file():
            continue
        t0 = time.perf_counter()
        _ = np.load(p, allow_pickle=False)
        out[i] = max(float(time.perf_counter() - t0), 1e-9)
    return out


def _runtime_partition_stats(runtimes: np.ndarray) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    if runtimes.size == 0:
        return None, None, None
    cached = int(np.sum(runtimes <= _RT_EPS))
    computed = int(np.sum(runtimes > _RT_EPS))
    active_mean: Optional[float] = None
    if computed > 0:
        sel = runtimes[runtimes > _RT_EPS]
        active_mean = _safe_float(float(np.nanmean(sel)))
    return active_mean, cached, computed


def build_evaluation_overview(
    manifests: List[ExperimentRunManifest],
    *,
    repo_root: Optional[Path] = None,
    comparison_run_tags: Optional[Sequence[str]] = None,
) -> ExperimentEvaluationBundle:
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT.resolve()
    rows: List[ExperimentMetricRow] = []
    notes: List[str] = []
    for m in manifests:
        summary = _summary_arrays(Path(m.leiden_dir) / "summary.npy")
        if summary is None:
            notes.append(f"{m.run_id}: summary.npy missing under {m.leiden_dir}")
            rows.append(
                ExperimentMetricRow(
                    run_id=m.run_id,
                    algorithm=m.algorithm,
                    time_window=m.time_window,
                    resolution_min=None,
                    resolution_max=None,
                    n_resolution_points=0,
                    mean_runtime_sec=None,
                    min_runtime_sec=None,
                    max_runtime_sec=None,
                    mean_n_communities=None,
                    max_n_communities=None,
                    min_n_communities=None,
                )
            )
            continue

        resolutions = np.asarray(summary.get("resolutions", []), dtype=np.float64)
        runtimes_raw = np.asarray(summary.get("time", []), dtype=np.float64)
        n_comm = np.asarray(summary.get("n_comm", []), dtype=np.int64)
        ld_path = Path(m.leiden_dir)
        scorecard = _load_optional_scorecard(ld_path)
        retrieval_score = scorecard.get("retrieval_score")
        if retrieval_score is None:
            retrieval_score = aggregate_retrieval_score_from_comparison_runs(
                root,
                run_id=str(m.run_id),
                time_window=str(m.time_window),
                comparison_run_tags=comparison_run_tags,
            )
        topic_score = scorecard.get("topic_score")
        if topic_score is None:
            topic_score = aggregate_topic_score_from_topic_runs(root, run_id=str(m.run_id))

        coarse_all_zero = (
            str(m.algorithm).strip() == "coarse_kmeans"
            and runtimes_raw.size > 0
            and bool(np.all(np.asarray(runtimes_raw, dtype=np.float64) <= _RT_EPS))
        )
        used_load_time_fallback = False
        runtimes_display = runtimes_raw.astype(np.float64).copy()

        if coarse_all_zero:
            est = _estimate_membership_load_times(ld_path, resolutions)
            if np.isfinite(est).any():
                runtimes_display = est
                used_load_time_fallback = True
                notes.append(
                    f"{m.run_id}: sweep times in summary.npy were all zero (cache hits); "
                    "overview runtime columns use timed np.load of each membership_r*.npy (not Leiden compute)."
                )
            else:
                notes.append(
                    f"{m.run_id}: sweep runtime all zero and membership files missing for load-time fallback."
                )
        elif (
            runtimes_raw.size > 0
            and bool(np.any(runtimes_raw <= _RT_EPS))
            and bool(np.any(runtimes_raw > _RT_EPS))
        ):
            est = _estimate_membership_load_times(ld_path, resolutions)
            runtimes_display = np.where(runtimes_raw <= _RT_EPS, est, runtimes_raw)
            notes.append(
                f"{m.run_id}: {int(np.sum(runtimes_raw <= _RT_EPS))} resolutions had time≈0 in summary.npy "
                "(membership files reused without a fresh Leiden timing); mean/min/max runtime use timed np.load for those slots."
            )

        mean_act, n_cached, n_comp = _runtime_partition_stats(runtimes_raw)
        mean_rt = _safe_float(np.nanmean(runtimes_display)) if runtimes_display.size else None
        min_rt = _safe_float(np.nanmin(runtimes_display)) if runtimes_display.size else None
        max_rt = _safe_float(np.nanmax(runtimes_display)) if runtimes_display.size else None

        if coarse_all_zero and not used_load_time_fallback:
            mean_rt = min_rt = max_rt = None
            mean_act = None
            n_cached = n_comp = None

        suspicious_load_only_summary = (
            runtimes_raw.size > 0
            and str(m.algorithm).strip() in ("leiden_cpm", "leiden", "louvain")
            and float(np.nanmax(runtimes_raw)) < 0.05
            and n_comm.size > 0
            and float(np.nanmean(n_comm)) > 3000.0
        )
        if suspicious_load_only_summary:
            notes.append(
                f"{m.run_id}: summary.npy ``time`` looks like np.load-only (max={float(np.nanmax(runtimes_raw)):.4g}s over "
                f"{int(runtimes_raw.size)} resolutions, mean_n_comm≈{float(np.nanmean(n_comm)):.0f}); "
                "runtime columns cleared — not comparable to a fresh Leiden sweep. "
                "Restore summary.npy from backup, or run ``experiment-sweep --no-reuse-existing`` (slow; overwrites membership)."
            )
            mean_rt = min_rt = max_rt = None
            mean_act = None
            n_cached = n_comp = None

        ev_sw, ev_bp, ev_lay = discover_eval_plot_paths(ld_path)

        rows.append(
            ExperimentMetricRow(
                run_id=m.run_id,
                algorithm=m.algorithm,
                time_window=m.time_window,
                resolution_min=_safe_float(resolutions.min()) if resolutions.size else None,
                resolution_max=_safe_float(resolutions.max()) if resolutions.size else None,
                n_resolution_points=int(resolutions.size),
                mean_runtime_sec=mean_rt,
                min_runtime_sec=min_rt,
                max_runtime_sec=max_rt,
                mean_n_communities=_safe_float(np.nanmean(n_comm)) if n_comm.size else None,
                max_n_communities=_safe_int(np.nanmax(n_comm)) if n_comm.size else None,
                min_n_communities=_safe_int(np.nanmin(n_comm)) if n_comm.size else None,
                retrieval_score=retrieval_score,
                topic_score=topic_score,
                practical_score=scorecard.get("practical_score"),
                mean_runtime_active_sec=mean_act,
                n_partitions_cached=n_cached,
                n_partitions_computed=n_comp,
                eval_sweep_plot=ev_sw,
                eval_breakpoints_plot=ev_bp,
                eval_layered_plot=ev_lay,
            )
        )
    return ExperimentEvaluationBundle(generated_at_unix=float(time.time()), rows=rows, notes=notes)


def save_evaluation_overview(
    *,
    out_dir: Path,
    bundle: ExperimentEvaluationBundle,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_overview.json"
    csv_path = out_dir / "evaluation_overview.csv"

    json_path.write_text(json.dumps(bundle.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(ExperimentMetricRow.__dataclass_fields__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in bundle.rows:
            w.writerow(row.to_json_dict())

    return {"json": str(json_path), "csv": str(csv_path)}
