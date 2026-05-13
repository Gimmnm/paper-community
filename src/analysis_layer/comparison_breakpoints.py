"""
Per-run comparison breakpoints for multi-algorithm evaluation (see docs/experiment-comparison-pipeline.md).

Selects up to K resolutions per sweep (default 10): prefer detect_breakpoints scores, then pad with
uniformly spaced indices along the sorted resolution grid.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from algorithm_layer.community import detect_breakpoints

BREAKPOINT_POLICY_V1 = "breakpoint_policy_v1"


def infer_sweep_meta(summary: Dict[str, Any]) -> Dict[str, Any]:
    rs = np.asarray(summary.get("resolutions", []), dtype=float)
    if rs.size == 0:
        return {}
    rs_sorted = np.sort(rs)
    if rs_sorted.size >= 2:
        step = float(np.median(np.diff(rs_sorted)))
    else:
        step = None
    return {
        "r_min": float(rs_sorted[0]),
        "r_max": float(rs_sorted[-1]),
        "step": step,
        "n_points": int(rs.size),
    }


def select_comparison_breakpoints(
    summary: Dict[str, Any],
    *,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Return up to min(k, n_resolutions) rows, each aligned to one index in summary['resolutions'].

    Each dict: resolution, n_communities, resolution_index, source ('breakpoint'|'uniform_pad'),
    breakpoint_score (float or ''), delta_n_comm (int).
    """
    resolutions = np.asarray(summary.get("resolutions", []), dtype=float)
    n_comm = np.asarray(summary.get("n_comm", []), dtype=np.int64)
    n = int(resolutions.size)
    if n == 0:
        return []
    eff_k = min(int(k), n)
    dn = np.abs(np.diff(n_comm.astype(float), prepend=float(n_comm[0]))).astype(int)

    chosen_idx: List[int] = []
    chosen_set: set[int] = set()
    rows: List[Dict[str, Any]] = []

    bps = detect_breakpoints(summary, top_k=max(50, eff_k * 10))
    for bp in bps:
        if len(chosen_idx) >= eff_k:
            break
        r_tgt = float(bp["resolution"])
        i = int(np.argmin(np.abs(resolutions - r_tgt)))
        if i in chosen_set:
            continue
        chosen_set.add(i)
        chosen_idx.append(i)
        rows.append(
            {
                "resolution": float(resolutions[i]),
                "n_communities": int(n_comm[i]),
                "resolution_index": i,
                "source": "breakpoint",
                "breakpoint_score": float(bp.get("score", 0.0)),
                "delta_n_comm": int(dn[i]),
            }
        )

    if len(rows) < eff_k:
        order = np.argsort(resolutions)
        ordered_indices = [int(order[j]) for j in range(n)]
        remaining = [i for i in ordered_indices if i not in chosen_set]
        need = eff_k - len(rows)
        if remaining:
            if len(remaining) <= need:
                extra_list = remaining
            else:
                pos = np.linspace(0, len(remaining) - 1, need)
                extra_list = []
                seen_e: set[int] = set()
                for x in pos:
                    j = int(round(float(x)))
                    j = max(0, min(j, len(remaining) - 1))
                    idx_r = remaining[j]
                    if idx_r not in seen_e:
                        seen_e.add(idx_r)
                        extra_list.append(idx_r)
                for idx_r in remaining:
                    if len(extra_list) >= need:
                        break
                    if idx_r not in seen_e:
                        seen_e.add(idx_r)
                        extra_list.append(idx_r)

            for i in extra_list:
                if len(rows) >= eff_k:
                    break
                if i in chosen_set:
                    continue
                chosen_set.add(i)
                chosen_idx.append(i)
                rows.append(
                    {
                        "resolution": float(resolutions[i]),
                        "n_communities": int(n_comm[i]),
                        "resolution_index": i,
                        "source": "uniform_pad",
                        "breakpoint_score": "",
                        "delta_n_comm": int(dn[i]),
                    }
                )

    out: List[Dict[str, Any]] = []
    for bi, rdict in enumerate(rows):
        item = dict(rdict)
        item["breakpoint_index"] = bi
        item["breakpoint_policy_version"] = BREAKPOINT_POLICY_V1
        out.append(item)
    return out


def write_comparison_breakpoints_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = (
            "run_id",
            "algorithm",
            "time_window",
            "breakpoint_index",
            "resolution_index",
            "resolution",
            "n_communities",
            "delta_n_comm",
            "breakpoint_score",
            "source",
            "breakpoint_policy_version",
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def write_breakpoints_run_meta(path: Path, *, n_breakpoints: int, n_runs: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "breakpoint_policy_version": BREAKPOINT_POLICY_V1,
        "n_breakpoints_requested": int(n_breakpoints),
        "n_manifests_with_summary": int(n_runs),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
