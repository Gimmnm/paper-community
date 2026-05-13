#!/usr/bin/env python3
"""
从已有的 ``comparison_retrieval_sixway_long.csv`` 重画六路检索排名条形图（不读 metrics.jsonl）。

可选：若已生成 ``comparison_retrieval_resolution_long.csv`` 与
``comparison_retrieval_best_resolution_long.csv``，加 ``--also-resolution-plots``
可重画分辨率曲线与 best-pick 条形图。

用法（仓库根目录）：

  PYTHONPATH=src python scripts/plot_retrieval_sixway_rankings.py --help

或直接：

  python scripts/plot_retrieval_sixway_rankings.py \\
    --long-csv out/experiment_eval/comparison_retrieval_sixway_long.csv \\
    --plot-dir out/experiment_eval/retrieval_sixway_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> None:
    repo = _bootstrap_path()
    p = argparse.ArgumentParser(description="Plot retrieval six-way ranking bar charts from long CSV")
    p.add_argument(
        "--long-csv",
        type=str,
        default=str(repo / "out" / "experiment_eval" / "comparison_retrieval_sixway_long.csv"),
    )
    p.add_argument(
        "--plot-dir",
        type=str,
        default=str(repo / "out" / "experiment_eval" / "retrieval_sixway_plots"),
    )
    p.add_argument(
        "--also-resolution-plots",
        action="store_true",
        help="同时从 resolution / best CSV 画 retrieval_resolution_plots（需两文件存在）",
    )
    p.add_argument(
        "--resolution-long-csv",
        type=str,
        default=None,
        help="默认与 --long-csv 同目录下的 comparison_retrieval_resolution_long.csv",
    )
    p.add_argument(
        "--best-resolution-long-csv",
        type=str,
        default=None,
        help="默认与 --long-csv 同目录下的 comparison_retrieval_best_resolution_long.csv",
    )
    p.add_argument(
        "--resolution-plot-dir",
        type=str,
        default=None,
        help="分辨率图输出目录（默认与 --long-csv 同目录下的 retrieval_resolution_plots）",
    )
    args = p.parse_args()

    from analysis_layer.retrieval_sixway_plots import (
        plot_retrieval_sixway_metric_absolute_banded_lines_csv,
        plot_retrieval_sixway_metric_lines_csv,
        plot_retrieval_sixway_rankings_csv,
    )

    long_p = Path(args.long_csv)
    plot_p = Path(args.plot_dir)
    paths = plot_retrieval_sixway_rankings_csv(long_p, plot_p)
    for path in paths:
        print(path)
    for path in plot_retrieval_sixway_metric_lines_csv(long_p, plot_p):
        print(path)
    for path in plot_retrieval_sixway_metric_absolute_banded_lines_csv(long_p, plot_p):
        print(path)

    if args.also_resolution_plots:
        base = long_p.parent
        res_csv = Path(args.resolution_long_csv) if args.resolution_long_csv else base / "comparison_retrieval_resolution_long.csv"
        best_csv = (
            Path(args.best_resolution_long_csv)
            if args.best_resolution_long_csv
            else base / "comparison_retrieval_best_resolution_long.csv"
        )
        res_plot_dir = Path(args.resolution_plot_dir) if args.resolution_plot_dir else base / "retrieval_resolution_plots"
        from analysis_layer.retrieval_resolution_analysis import plot_resolution_curves_and_best_bars

        for k, plist in plot_resolution_curves_and_best_bars(res_csv, best_csv, res_plot_dir).items():
            for path in plist:
                print(path)


if __name__ == "__main__":
    main()
