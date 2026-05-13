"""Shared argparse bits for experiment catalog / eval CLIs (avoids duplicating fallback flags)."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_layer.experiment_contracts import ExperimentRunManifest
from data_layer.experiment_registry import default_paths_manifest


def register_experiment_catalog_fallback_args(p: argparse.ArgumentParser, *, out_dir: Path) -> None:
    """CLI flags shared by experiment-catalog / experiment-eval / eval-bundle / comparison-breakpoints."""
    od = Path(out_dir)
    p.add_argument("--fallback-leiden-dir", type=str, default=str(od / "leiden_sweep_cpm"))
    p.add_argument("--fallback-graph-npz", type=str, default=str(od / "mutual_knn_k50.npz"))
    p.add_argument("--fallback-keyword-index-dir", type=str, default=str(od / "keyword_index"))
    p.add_argument("--fallback-coords-2d-path", type=str, default=str(od / "umap2d.npy"))
    p.add_argument(
        "--fallback-algorithm",
        type=str,
        choices=["leiden", "leiden_cpm", "louvain", "coarse_kmeans"],
        default="leiden_cpm",
    )
    p.add_argument("--fallback-time-window", type=str, choices=["1y", "5y", "all"], default="all")
    p.add_argument("--fallback-resolution", type=float, default=0.2)


def experiment_catalog_fallback_manifest(base_dir: Path, args: argparse.Namespace) -> ExperimentRunManifest:
    return default_paths_manifest(
        run_id="cli_default",
        base_dir=Path(base_dir),
        leiden_dir=Path(args.fallback_leiden_dir),
        graph_npz=Path(args.fallback_graph_npz),
        keyword_index_dir=Path(args.fallback_keyword_index_dir) if args.fallback_keyword_index_dir else None,
        coords_2d_path=Path(args.fallback_coords_2d_path) if args.fallback_coords_2d_path else None,
        algorithm=str(args.fallback_algorithm),
        time_window=str(args.fallback_time_window),
        default_resolution=float(args.fallback_resolution),
    )
