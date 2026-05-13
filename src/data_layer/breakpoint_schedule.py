"""Load per-run comparison breakpoint resolutions from experiment_eval CSV (no cross-algorithm alignment)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Master script / sweep folder tag → manifest ``run_id`` (matches comparison_breakpoints.csv)
SWEEP_FOLDER_TAG_TO_RUN_ID: Dict[str, str] = {
    "leiden_sweep_cpm": "leiden_cpm",
    "leiden_sweep_rb": "leiden",
    "leiden_sweep_louvain": "louvain",
    "coarse_kmeans_then_cpm_k3_seed42": "coarse_kmeans",
}

# Inverse: catalog ``run_id`` → ``out/topic_runs/<folder>/`` basename (Topic-SCORE batch layout)
RUN_ID_TO_TOPIC_RUN_FOLDER: Dict[str, str] = {v: k for k, v in SWEEP_FOLDER_TAG_TO_RUN_ID.items()}


def topic_run_sweep_folder_for_run_id(run_id: str) -> Optional[str]:
    """Map manifest ``run_id`` (including ``leiden_cpm_5y``) to ``out/topic_runs/<folder>`` basename."""
    rid = str(run_id).strip()
    if rid in RUN_ID_TO_TOPIC_RUN_FOLDER:
        return RUN_ID_TO_TOPIC_RUN_FOLDER[rid]
    for suf in ("_5y", "_1y"):
        if rid.endswith(suf):
            base = rid[: -len(suf)]
            if base in RUN_ID_TO_TOPIC_RUN_FOLDER:
                return RUN_ID_TO_TOPIC_RUN_FOLDER[base]
    return None


def default_topic_k_dir_for_run(
    base_dir: Path,
    run_id: str,
    *,
    manifest_tags: Optional[Dict[str, Any]] = None,
    k_topics_default: int = 10,
) -> Optional[Path]:
    """
    ``out/topic_runs/<sweep>/K{k}`` when that directory exists (Topic-SCORE batch layout).
    ``k`` prefers ``manifest_tags["topic_k"]`` when present.
    """
    folder = topic_run_sweep_folder_for_run_id(run_id)
    if not folder:
        return None
    k = int(k_topics_default)
    if isinstance(manifest_tags, dict):
        tk = manifest_tags.get("topic_k")
        try:
            if tk is not None:
                k = int(tk)
        except (TypeError, ValueError):
            k = int(k_topics_default)
    p = Path(base_dir) / "out" / "topic_runs" / folder / f"K{k}"
    return p if p.is_dir() else None


def resolve_topic_communities_csv(topic_root: Optional[Path], r_eff: float) -> Optional[Path]:
    """
    Pick ``communities_topic_weights.csv`` under ``topic_root/r*/`` for resolution nearest to ``r_eff``.
    ``topic_root`` is typically ``.../topic_runs/<sweep>/K10``.
    """
    if topic_root is None:
        return None
    root = Path(topic_root)
    if not root.is_dir():
        return None
    exact = root / f"r{float(r_eff):.4f}" / "communities_topic_weights.csv"
    if exact.is_file():
        return exact
    best: Optional[Tuple[float, Path]] = None
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("r"):
            continue
        tail = child.name[1:]
        try:
            rr = float(tail)
        except ValueError:
            continue
        p = child / "communities_topic_weights.csv"
        if not p.is_file():
            continue
        dist = abs(rr - float(r_eff))
        if best is None or dist < best[0]:
            best = (dist, p)
    return None if best is None else best[1]


def load_breakpoint_resolutions_for_run(
    csv_path: Path,
    *,
    run_id: str,
    time_window: str = "all",
) -> List[float]:
    """
    Return sorted unique resolutions for ``run_id`` (and matching ``time_window``).
    CSV columns: run_id, time_window, resolution, ...
    """
    p = Path(csv_path)
    if not p.is_file():
        return []
    rid = str(run_id).strip()
    tw = str(time_window).strip()
    out: List[float] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("run_id", "")).strip() != rid:
                continue
            if str(row.get("time_window", "all")).strip() != tw:
                continue
            try:
                out.append(float(row["resolution"]))
            except (KeyError, ValueError, TypeError):
                continue
    return sorted(set(round(float(x), 10) for x in out))


def default_breakpoints_csv(repo_root: Path) -> Path:
    return Path(repo_root) / "out" / "experiment_eval" / "comparison_breakpoints.csv"


def infer_breakpoint_run_id_from_leiden_dir(leiden_dir: Path) -> Optional[str]:
    """Best-effort: map ``.../leiden_sweep_cpm`` folder name to catalog run_id."""
    name = Path(leiden_dir).name
    return SWEEP_FOLDER_TAG_TO_RUN_ID.get(name)
