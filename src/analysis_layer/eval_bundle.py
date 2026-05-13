"""
Offline per-run evaluation assets under <leiden_dir>/eval/.

Writes PNGs + breakpoints.json + artifacts.json so the web Eval tab and
evaluation_overview can resolve paths consistently (see evaluation-metrics-spec).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from algorithm_layer.community import (
    build_parent_child_links,
    detect_breakpoints,
    membership_filename,
    plot_sweep_diagnostics,
)
from data_layer.experiment_contracts import ExperimentRunManifest


def _summary_from_disk(leiden_dir: Path) -> Optional[Dict[str, Any]]:
    p = leiden_dir / "summary.npy"
    if not p.is_file():
        return None
    try:
        return np.load(p, allow_pickle=True).item()
    except Exception:
        return None


def _load_memberships_for_resolutions(leiden_dir: Path, resolutions: List[float]) -> Dict[float, np.ndarray]:
    out: Dict[float, np.ndarray] = {}
    for r in resolutions:
        fn = membership_filename(float(r))
        path = leiden_dir / fn
        if not path.is_file():
            continue
        try:
            out[float(r)] = np.load(path).astype(np.int32)
        except Exception:
            continue
    return out


def subsample_resolutions(resolutions: List[float], max_n: int) -> List[float]:
    rs = sorted({float(x) for x in resolutions})
    if len(rs) <= max_n:
        return rs
    idx = np.linspace(0, len(rs) - 1, max_n).astype(int)
    return [rs[int(i)] for i in np.unique(idx)]


def plot_breakpoints_delta_bar(
    summary: Dict[str, Any],
    breakpoints: List[Dict[str, Any]],
    *,
    out_png: Path,
) -> None:
    resolutions = np.asarray(summary["resolutions"], dtype=float)
    n_comm = np.asarray(summary["n_comm"], dtype=float)
    if resolutions.size == 0:
        return
    dn = np.abs(np.diff(n_comm, prepend=n_comm[0]))
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ax.bar(np.arange(resolutions.size), dn, width=1.0, alpha=0.72, color="#5b8bd9")
    ax.set_xlabel("resolution index (sorted)")
    ax.set_ylabel("|Δ #communities|")
    ax.set_title("Community-count jumps across sweep (breakpoints marked)")
    bp_idx = set()
    bp_res = {float(b["resolution"]) for b in breakpoints}
    for i, r in enumerate(resolutions.tolist()):
        if float(r) in bp_res:
            bp_idx.add(i)
            ax.axvline(i, color="#ff6a8b", linewidth=1.1, alpha=0.55)
    plt.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


def plot_layer_link_counts(
    leiden_dir: Path,
    resolutions_sub: List[float],
    *,
    min_child_share: float,
    out_png: Path,
) -> bool:
    mem = _load_memberships_for_resolutions(leiden_dir, resolutions_sub)
    keys = sorted(mem.keys())
    if len(keys) < 2:
        return False
    xs: List[float] = []
    ys: List[int] = []
    for r_parent, r_child in zip(keys[:-1], keys[1:]):
        links = build_parent_child_links(
            mem[r_parent], mem[r_child], r_parent=float(r_parent), r_child=float(r_child)
        )
        kept = [row for row in links if float(row["child_share"]) >= float(min_child_share)]
        xs.append(float(r_child))
        ys.append(len(kept))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
    ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=4)
    ax.set_xlabel("resolution (child layer)")
    ax.set_ylabel(f"# links with child_share ≥ {min_child_share:g}")
    ax.set_title("Hierarchy connectivity across sweep (subsampled resolutions)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    return True


def generate_eval_bundle_for_manifest(
    m: ExperimentRunManifest,
    *,
    force: bool = False,
    skip_existing: bool = False,
    with_layered: bool = True,
    max_layered_resolutions: int = 36,
    min_child_share_layer_plot: float = 0.10,
) -> Dict[str, Any]:
    """
    Write eval/sweep_diagnostics.png, eval/breakpoints_overview.png,
    eval/breakpoints.json, optional eval/layer_link_counts.png, eval/artifacts.json.
    """
    ld = Path(m.leiden_dir).resolve()
    eval_dir = ld / "eval"
    summary = _summary_from_disk(ld)
    run_id = str(m.run_id)

    if summary is None:
        return {"run_id": run_id, "skipped": True, "reason": "summary.npy missing"}

    artifacts_path = eval_dir / "artifacts.json"
    if skip_existing and not force and artifacts_path.is_file():
        return {"run_id": run_id, "skipped": True, "reason": "eval/artifacts.json exists (--skip-existing)"}

    eval_dir.mkdir(parents=True, exist_ok=True)
    breakpoints = detect_breakpoints(summary)

    sweep_png = eval_dir / "sweep_diagnostics.png"
    plot_sweep_diagnostics(summary, out_png=sweep_png, breakpoints=breakpoints)

    bp_png = eval_dir / "breakpoints_overview.png"
    plot_breakpoints_delta_bar(summary, breakpoints, out_png=bp_png)

    (eval_dir / "breakpoints.json").write_text(json.dumps(breakpoints, ensure_ascii=False, indent=2), encoding="utf-8")

    layered_rel: Optional[str] = None
    if with_layered:
        rs = np.asarray(summary.get("resolutions", []), dtype=float).tolist()
        sub = subsample_resolutions(rs, int(max_layered_resolutions))
        layered_png = eval_dir / "layer_link_counts.png"
        if plot_layer_link_counts(ld, sub, min_child_share=min_child_share_layer_plot, out_png=layered_png):
            layered_rel = "eval/layer_link_counts.png"

    mapping = {
        "sweep_diagnostics": "eval/sweep_diagnostics.png",
        "breakpoints": "eval/breakpoints_overview.png",
        "layered": layered_rel,
    }
    payload = {
        "run_id": run_id,
        "algorithm": str(m.algorithm),
        "time_window": str(m.time_window),
        "generated_at_unix": float(time.time()),
        **{k: v for k, v in mapping.items()},
    }
    artifacts_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "skipped": False,
        "eval_dir": str(eval_dir),
        "artifacts": mapping,
        "n_breakpoints": len(breakpoints),
    }


def generate_eval_bundles_all(
    manifests: List[ExperimentRunManifest],
    *,
    force: bool = False,
    skip_existing: bool = False,
    with_layered: bool = True,
    max_layered_resolutions: int = 36,
    run_id_filter: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    filt = None if not run_id_filter else {str(x) for x in run_id_filter}
    rows = []
    for m in manifests:
        if filt is not None and str(m.run_id) not in filt:
            continue
        rows.append(
            generate_eval_bundle_for_manifest(
                m,
                force=force,
                skip_existing=skip_existing,
                with_layered=with_layered,
                max_layered_resolutions=max_layered_resolutions,
            )
        )
    return {"n_manifests": len(manifests), "n_processed": len(rows), "results": rows}
