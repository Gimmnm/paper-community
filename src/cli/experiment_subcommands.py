"""Register experiment-* argparse subcommands (logic stays in ``core.task_*``)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from data_layer.experiment_fallback_cli import register_experiment_catalog_fallback_args


@dataclass(frozen=True)
class ExperimentSubcommandHandlers:
    experiment_manifest: Callable[[argparse.Namespace], Any]
    experiment_catalog: Callable[[argparse.Namespace], Any]
    experiment_sweep: Callable[[argparse.Namespace], Any]
    experiment_coarse_kmeans_sweep: Callable[[argparse.Namespace], Any]
    experiment_sweep_time_window: Callable[[argparse.Namespace], Any]
    experiment_batch_time_windows: Callable[[argparse.Namespace], Any]
    experiment_eval: Callable[[argparse.Namespace], Any]
    experiment_eval_bundle: Callable[[argparse.Namespace], Any]
    experiment_init_minimal: Callable[[argparse.Namespace], Any]
    experiment_comparison_breakpoints: Callable[[argparse.Namespace], Any]
    experiment_retrieval_benchmark: Callable[[argparse.Namespace], Any]
    experiment_retrieval_sixway: Callable[[argparse.Namespace], Any]
    experiment_viz_batch: Callable[[argparse.Namespace], Any]


def register_experiment_subparsers(
    sub: Any,
    *,
    ctx: ExperimentSubcommandHandlers,
    base_dir: Path,
    out_dir: Path,
    emb_path: Path,
    add_common_runtime_flags: Callable[[argparse.ArgumentParser], None],
) -> None:
    """Attach all ``experiment-*`` parsers to ``sub``."""
    EMB_PATH = emb_path
    OUT_DIR = out_dir
    tasks = ctx

    sp = sub.add_parser("experiment-manifest", help="保存实验运行清单（manifest）")
    sp.add_argument("--run-id", type=str, required=True)
    sp.add_argument(
        "--algorithm",
        type=str,
        choices=["leiden", "leiden_cpm", "louvain", "coarse_kmeans"],
        required=True,
    )
    sp.add_argument("--time-window", type=str, choices=["1y", "5y", "all"], required=True)
    sp.add_argument("--title", type=str, default=None)
    sp.add_argument("--leiden-dir", type=str, required=True)
    sp.add_argument("--graph-npz", type=str, required=True)
    sp.add_argument("--keyword-index-dir", type=str, default=None)
    sp.add_argument("--coords-2d-path", type=str, default=None)
    sp.add_argument(
        "--topic-communities-csv",
        type=str,
        default=None,
        help="可选：Topic-SCORE 的 communities_topic_weights.csv，供 demo 右侧展示主题",
    )
    sp.add_argument("--default-resolution", type=float, default=0.2)
    sp.add_argument("--partition-type", type=str, default=None)
    sp.set_defaults(func=tasks.experiment_manifest)

    sp = sub.add_parser("experiment-catalog", help="查看实验运行目录清单")
    register_experiment_catalog_fallback_args(sp, out_dir=OUT_DIR)
    sp.add_argument("--pretty", action="store_true")
    sp.set_defaults(func=tasks.experiment_catalog)

    sp = sub.add_parser(
        "experiment-sweep",
        help="统一分辨率扫参：Leiden RB、Leiden CPM、Louvain=多分辨率 RBERVertexPartition（Potts/ER null，含 resolution_parameter）",
    )
    add_common_runtime_flags(sp)
    sp.add_argument("--algorithm", type=str, choices=["leiden", "leiden_cpm", "louvain"], required=True)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--prefer-gpu", action="store_true")
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--cache-name", type=str, default=None)
    sp.add_argument("--out-dir", type=str, required=True)
    sp.add_argument("--r-min", type=float, default=0.001)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.02)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default=None)
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="忽略已有 membership_r*.npy，每个分辨率重跑 Leiden（summary 写入真实计算耗时；默认复用磁盘结果并只记加载/跳过时间）",
    )
    sp.set_defaults(func=tasks.experiment_sweep)

    sp = sub.add_parser(
        "experiment-coarse-kmeans-sweep",
        help="k-means 粗分 domain → 各 domain 诱导子图上跑 Louvain/Leiden/CPM sweep → 合并为全图 membership",
    )
    add_common_runtime_flags(sp)
    sp.add_argument(
        "--domains-dir",
        type=str,
        default=None,
        help="已跑好的 coarse domains 目录（含 labels.npy；可选 meta.json）。不设则从 --emb-path 现场 k-means",
    )
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--k", type=int, default=3, help="现场 k-means 的簇数（--domains-dir 未设时有效）")
    sp.add_argument(
        "--graph-npz",
        type=str,
        default=str(OUT_DIR / "mutual_knn_k50.npz"),
        help="全局 mutual-kNN（与全库论文数一致），用于取诱导子图",
    )
    sp.add_argument("--out-dir", type=str, required=True)
    sp.add_argument("--algorithm", type=str, choices=["leiden", "leiden_cpm", "louvain"], default="leiden_cpm")
    sp.add_argument("--r-min", type=float, default=0.001)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.02)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default=None)
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--write-manifest", action="store_true")
    sp.add_argument("--run-id", type=str, default="", help="配合 --write-manifest；默认 coarse_kmeans")
    sp.add_argument("--keyword-index-dir", type=str, default=str(OUT_DIR / "keyword_index"))
    sp.add_argument("--coords-2d-path", type=str, default=str(OUT_DIR / "umap2d.npy"))
    sp.add_argument("--default-resolution", type=float, default=0.2)
    sp.add_argument(
        "--topic-communities-csv",
        type=str,
        default=None,
        help="写入 manifest 的可选 topic CSV 路径",
    )
    sp.set_defaults(func=tasks.experiment_coarse_kmeans_sweep)

    sp = sub.add_parser(
        "experiment-sweep-time-window",
        help="按发表年截取子集 → 子集上重建 mutual-kNN → 多分辨率社区 sweep（refit，非仅视图过滤）",
    )
    add_common_runtime_flags(sp)
    sp.add_argument("--start-year", type=int, required=True)
    sp.add_argument("--end-year", type=int, required=True)
    sp.add_argument("--include-unknown-year", action="store_true", help="窗口内包含年份未知论文（与 time-window 分析一致）")
    sp.add_argument("--algorithm", type=str, choices=["leiden", "leiden_cpm", "louvain"], required=True)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--prefer-gpu", action="store_true")
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--out-dir", type=str, default=None, help="默认 out/leiden_sweep_<algorithm>_y<start>_<end>")
    sp.add_argument("--r-min", type=float, default=0.001)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.02)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default=None)
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--write-manifest", action="store_true", help="写入 out/experiments/.../manifest.json 供 demo 目录发现")
    sp.add_argument("--run-id", type=str, default="", help="配合 --write-manifest；默认 <algorithm>_y<start>_<end>")
    sp.add_argument("--keyword-index-dir", type=str, default=str(OUT_DIR / "keyword_index"))
    sp.add_argument("--coords-2d-path", type=str, default=str(OUT_DIR / "umap2d.npy"))
    sp.add_argument("--default-resolution", type=float, default=0.2)
    sp.set_defaults(func=tasks.experiment_sweep_time_window)

    sp = sub.add_parser(
        "experiment-batch-time-windows",
        help="多个发表年窗口分别 refit sweep；每个窗口单独 leiden_dir/graph_npz + manifest（网页可选 run）",
    )
    add_common_runtime_flags(sp)
    sp.add_argument(
        "--window",
        action="append",
        dest="window",
        metavar="Y0:Y1",
        required=True,
        help="发表年闭区间，可重复；格式 Y0:Y1 或 Y0-Y1（含端点）",
    )
    sp.add_argument("--include-unknown-year", action="store_true")
    sp.add_argument("--algorithm", type=str, choices=["leiden", "leiden_cpm", "louvain"], required=True)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--prefer-gpu", action="store_true")
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="若指定，则各窗口输出到 <out-root>/yY0_Y1/（否则沿用单窗口默认 out/leiden_sweep_<algorithm>_y...）",
    )
    sp.add_argument("--r-min", type=float, default=0.001)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.02)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default=None)
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument(
        "--run-id-prefix",
        type=str,
        default="",
        help="写入 manifest 的 run_id 前缀；实际 run_id=<prefix>_<algorithm>_yY0_Y1（空则与单命令默认一致）",
    )
    sp.add_argument("--continue-on-error", action="store_true", help="某一窗口失败则记录并继续其余窗口")
    sp.add_argument("--keyword-index-dir", type=str, default=str(OUT_DIR / "keyword_index"))
    sp.add_argument("--coords-2d-path", type=str, default=str(OUT_DIR / "umap2d.npy"))
    sp.add_argument("--default-resolution", type=float, default=0.2)
    sp.set_defaults(func=tasks.experiment_batch_time_windows)

    sp = sub.add_parser("experiment-eval", help="汇总实验指标并导出 JSON/CSV")
    sp.add_argument("--out-dir", type=str, default=str(OUT_DIR / "experiment_eval"))
    sp.add_argument(
        "--comparison-run-tag",
        action="append",
        default=None,
        metavar="TAG",
        help=(
            "retrieval_score 仅从 out/comparison_runs/<TAG>/metrics.jsonl 汇总（可重复指定多个 TAG）；"
            "默认读环境变量 PC_EVAL_COMPARISON_RUN_TAGS（逗号分隔），否则合并所有子目录"
        ),
    )
    sp.add_argument(
        "--no-retrieval-sixway",
        action="store_true",
        help="不生成 comparison_retrieval_sixway_*.csv 及按分辨率的 comparison_retrieval_resolution_*.csv / best_*（关键词/向量 + 四种 community_bundle 汇总）",
    )
    sp.add_argument(
        "--no-sixway-plots",
        action="store_true",
        help="不写 six-way 与按分辨率的 PNG（排名条、归一化折线、绝对量分段折线 → --out-dir/retrieval_sixway_plots/；分辨率 cosine 曲线与 best-pick 条形图 → retrieval_resolution_plots/）",
    )
    register_experiment_catalog_fallback_args(sp, out_dir=OUT_DIR)
    sp.set_defaults(func=tasks.experiment_eval)

    sp = sub.add_parser(
        "experiment-retrieval-sixway",
        help="仅从 out/comparison_runs/*/metrics.jsonl 生成六种检索对比表（不写 evaluation_overview）",
    )
    sp.add_argument("--out-dir", type=str, default=str(OUT_DIR / "experiment_eval"))
    sp.add_argument(
        "--comparison-run-tag",
        action="append",
        default=None,
        metavar="TAG",
        help="限定读取 out/comparison_runs/<TAG>/（可重复）；默认同 experiment-eval",
    )
    sp.add_argument(
        "--no-sixway-plots",
        action="store_true",
        help="不写 six-way 与按分辨率的 PNG（排名条、归一化折线、绝对量分段折线 → --out-dir/retrieval_sixway_plots/；分辨率 cosine 曲线与 best-pick 条形图 → retrieval_resolution_plots/）",
    )
    register_experiment_catalog_fallback_args(sp, out_dir=OUT_DIR)
    sp.set_defaults(func=tasks.experiment_retrieval_sixway)

    sp = sub.add_parser(
        "experiment-eval-bundle",
        help="为每个 manifest 的 leiden_dir 生成 eval/（sweep + breakpoints + 可选分层链接计数）",
    )
    register_experiment_catalog_fallback_args(sp, out_dir=OUT_DIR)
    sp.add_argument("--force", action="store_true", help="覆盖已有 eval/artifacts.json 对应的产物")
    sp.add_argument("--skip-existing", action="store_true", help="若 eval/artifacts.json 已存在则跳过该 run")
    sp.add_argument("--no-layered", action="store_true", help="不生成 layer_link_counts.png")
    sp.add_argument("--max-layered-resolutions", type=int, default=36, help="分层链接计数最多抽样多少个分辨率")
    sp.add_argument("--run-id", nargs="*", default=None, help="只处理这些 run_id（默认全部）")
    sp.add_argument(
        "--also-overview",
        action="store_true",
        help="完成后重写 evaluation_overview（与 experiment-eval 同目录）",
    )
    sp.add_argument("--overview-out-dir", type=str, default=str(OUT_DIR / "experiment_eval"))
    sp.add_argument(
        "--comparison-run-tag",
        action="append",
        default=None,
        metavar="TAG",
        help="与 experiment-eval --comparison-run-tag 相同（仅 --also-overview 时生效）",
    )
    sp.add_argument(
        "--no-retrieval-sixway",
        action="store_true",
        help="与 experiment-eval 相同：--also-overview 时不写 six-way 检索对比表",
    )
    sp.add_argument(
        "--no-sixway-plots",
        action="store_true",
        help="与 experiment-eval 相同：不写 six-way 与按分辨率的图（排名条、折线类 → retrieval_sixway_plots/；分辨率曲线与 best-pick 条图 → retrieval_resolution_plots/；仅当写了 six-way 表时生效）",
    )
    sp.set_defaults(func=tasks.experiment_eval_bundle)

    sp = sub.add_parser(
        "experiment-init-minimal",
        help="注册实验 manifest（默认四算法 × all-time）；--include-placeholder-windows 可增加 1y/5y 占位",
    )
    sp.add_argument("--graph-npz", type=str, default=str(OUT_DIR / "mutual_knn_k50.npz"))
    sp.add_argument("--keyword-index-dir", type=str, default=str(OUT_DIR / "keyword_index"))
    sp.add_argument("--coords-2d-path", type=str, default=str(OUT_DIR / "umap2d.npy"))
    sp.add_argument(
        "--topic-communities-csv",
        type=str,
        default=None,
        help="写入每个 manifest 的可选 communities_topic_weights.csv（绝对路径或相对仓库根）",
    )
    sp.add_argument("--default-resolution", type=float, default=0.2)
    sp.add_argument("--leiden_cpm_all_dir", type=str, default=str(OUT_DIR / "leiden_sweep_cpm"))
    sp.add_argument("--leiden_cpm_5y_dir", type=str, default=None)
    sp.add_argument("--leiden_cpm_1y_dir", type=str, default=None)
    sp.add_argument("--leiden_all_dir", type=str, default=str(OUT_DIR / "leiden_sweep_rb"))
    sp.add_argument("--leiden_5y_dir", type=str, default=None)
    sp.add_argument("--leiden_1y_dir", type=str, default=None)
    sp.add_argument(
        "--louvain-all-dir",
        type=str,
        default=str(OUT_DIR / "leiden_sweep_louvain"),
        help="Louvain sweep 输出目录（需含 summary.npy）；不存在则跳过注册",
    )
    sp.add_argument("--louvain-5y-dir", type=str, default=None)
    sp.add_argument("--louvain-1y-dir", type=str, default=None)
    sp.add_argument(
        "--coarse-kmeans-all-dir",
        type=str,
        default=str(OUT_DIR / "coarse_kmeans_then_cpm_k3_seed42"),
        help="experiment-coarse-kmeans-sweep 合并产物目录；不存在则跳过",
    )
    sp.add_argument("--coarse-kmeans-5y-dir", type=str, default=None)
    sp.add_argument("--coarse-kmeans-1y-dir", type=str, default=None)
    sp.add_argument(
        "--include-placeholder-windows",
        action="store_true",
        help="同时注册 1y/5y（可与 all 共用同一 leiden_dir）；默认仅注册 all-time（对齐 experiment-comparison-pipeline）",
    )
    sp.set_defaults(func=tasks.experiment_init_minimal)

    sp = sub.add_parser(
        "experiment-comparison-breakpoints",
        help="从各 manifest 的 summary.npy 选取断点分辨率，写入 comparison_breakpoints.csv",
    )
    register_experiment_catalog_fallback_args(sp, out_dir=OUT_DIR)
    sp.add_argument(
        "--out-csv",
        type=str,
        default=str(OUT_DIR / "experiment_eval" / "comparison_breakpoints.csv"),
    )
    sp.add_argument(
        "--meta-json",
        type=str,
        default=str(OUT_DIR / "experiment_eval" / "comparison_breakpoints_meta.json"),
    )
    sp.add_argument("--n-breakpoints", type=int, default=10)
    sp.add_argument("--run-id", nargs="*", default=None, help="只处理这些 run_id（默认全部 catalog）")
    sp.set_defaults(func=tasks.experiment_comparison_breakpoints)

    sp = sub.add_parser(
        "experiment-retrieval-benchmark",
        help="离线检索对比：关键词 TF-IDF / embedding 近邻 / 社区邻域 bundle（out/comparison_runs/）",
    )
    from analysis_layer.retrieval_benchmark import register_retrieval_benchmark_args

    register_retrieval_benchmark_args(sp, base_dir=base_dir, default_out_dir=OUT_DIR)
    sp.set_defaults(func=tasks.experiment_retrieval_benchmark)

    sp = sub.add_parser(
        "experiment-viz-batch",
        help="离线批量可视化：UMAP 着色、社区元图、Top 社区诱导子图（out/viz_batch/）",
    )
    from analysis_layer.viz_batch import register_viz_batch_args

    register_viz_batch_args(sp, base_dir=base_dir, default_out_dir=OUT_DIR)
    sp.set_defaults(func=tasks.experiment_viz_batch)
