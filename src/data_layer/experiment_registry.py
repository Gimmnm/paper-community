from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data_layer.experiment_contracts import ExperimentRunManifest


def _norm_run_id(s: str) -> str:
    return str(s or "").strip().replace(" ", "_")


def experiments_root(base_dir: Path) -> Path:
    return Path(base_dir) / "out" / "experiments"


def discover_experiment_manifests(base_dir: Path) -> List[ExperimentRunManifest]:
    """
    Discover runs from out/experiments/**/manifest.json.
    """
    root = experiments_root(base_dir)
    out: List[ExperimentRunManifest] = []
    if not root.exists():
        return out
    for p in sorted(root.glob("*/*/manifest.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = ExperimentRunManifest.from_json_dict(d)
            if m.run_id:
                out.append(m)
        except Exception:
            continue
    return out


def default_paths_manifest(
    *,
    run_id: str,
    base_dir: Path,
    leiden_dir: Path,
    graph_npz: Path,
    keyword_index_dir: Optional[Path],
    coords_2d_path: Optional[Path],
    algorithm: str = "leiden_cpm",
    time_window: str = "all",
    default_resolution: float = 0.2,
    topic_communities_csv: Optional[Path] = None,
) -> ExperimentRunManifest:
    """Synthetic manifest when out/experiments/ has no manifest.json yet (CLI defaults only)."""
    rid = _norm_run_id(run_id) or "cli_default"
    return ExperimentRunManifest(
        run_id=rid,
        algorithm=str(algorithm),  # type: ignore[arg-type]
        time_window=str(time_window),  # type: ignore[arg-type]
        title=f"CLI defaults ({algorithm}, {time_window})",
        leiden_dir=str(Path(leiden_dir)),
        graph_npz=str(Path(graph_npz)),
        keyword_index_dir=None if keyword_index_dir is None else str(Path(keyword_index_dir)),
        coords_2d_path=None if coords_2d_path is None else str(Path(coords_2d_path)),
        topic_communities_csv=None if topic_communities_csv is None else str(Path(topic_communities_csv)),
        default_resolution=float(default_resolution),
        tags={"source": "cli_default_paths", "base_dir": str(Path(base_dir))},
    )


def build_run_catalog(
    *,
    base_dir: Path,
    fallback: ExperimentRunManifest,
) -> Dict[str, ExperimentRunManifest]:
    """
    Merge disk manifests under out/experiments/ into a run_id → manifest map.

    If **no** manifest.json exists yet, inject ``fallback`` so demo-api / eval CLI
    still point at CLI-default sweep paths. Once any manifest is registered, omit
    fallback to avoid duplicate rows (e.g. cli_default alongside real runs).
    """
    runs = discover_experiment_manifests(base_dir)
    by_id: Dict[str, ExperimentRunManifest] = {}
    for r in runs:
        by_id[_norm_run_id(r.run_id)] = r
    if not runs and _norm_run_id(fallback.run_id) not in by_id:
        by_id[_norm_run_id(fallback.run_id)] = fallback
    return by_id


def save_manifest(base_dir: Path, manifest: ExperimentRunManifest) -> Path:
    run_id = _norm_run_id(manifest.run_id)
    algo = str(manifest.algorithm)
    t = str(manifest.time_window)
    out_dir = experiments_root(base_dir) / algo / f"{t}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _resolutions_from_membership_files(leiden_dir: Path) -> List[float]:
    """Parse ``membership_r{float}.npy`` in ``leiden_dir`` (disk truth; may differ from summary.npy)."""
    ld = Path(leiden_dir)
    if not ld.is_dir():
        return []
    pat = re.compile(r"^membership_r([0-9]+(?:\.[0-9]+)?)\.npy$", re.IGNORECASE)
    out: List[float] = []
    for p in ld.glob("membership_r*.npy"):
        m = pat.match(p.name)
        if not m:
            continue
        try:
            out.append(round(float(m.group(1)), 4))
        except ValueError:
            continue
    return sorted(set(out))


def available_resolutions_for_manifest(manifest: ExperimentRunManifest) -> List[float]:
    """
    Resolutions for which ``membership_r*.npy`` exists under ``manifest.leiden_dir`` (per-run, per-sweep).

    Falls back to ``summary.npy`` only if no membership files are found, then to ``default_resolution``.
    """
    ld = Path(manifest.leiden_dir)
    file_rs = _resolutions_from_membership_files(ld)
    if file_rs:
        return file_rs

    out: List[float] = []
    summary = ld / "summary.npy"
    if summary.exists():
        try:
            d = np.load(summary, allow_pickle=True).item()
            rs = np.asarray(d.get("resolutions", []), dtype=np.float64).tolist()
            out.extend(float(x) for x in rs)
        except Exception:
            pass
    if not out:
        out.append(float(manifest.default_resolution))
    out = sorted(set(round(float(x), 4) for x in out))
    return out
