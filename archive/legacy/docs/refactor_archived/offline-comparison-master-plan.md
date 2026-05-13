# 离线多算法对比：你要跑的整条线（all‑time 先行）

本文档对齐当前代码里的 CLI / 目录约定，用来回答：**先做什么、再做什么、产物放哪**。时间窗（按年切片）**暂不纳入主流程**；等 all‑time 跑通后再加 `experiment-sweep-time-window`。

---

## 目标（五步分析）

对 **四种划分**（Louvain 语义见下、Leiden RB、Leiden CPM、k‑means 粗分再社区发现合并）在 **同一套 mutual‑kNN 图**上：

1. **结构诊断**：分辨率 sweep 曲线、分层边表与分层示意图、突变点 / 断点图（可用现有 `hierarchy`、`experiment-eval-bundle`；大图可归档）。
2. **主题建模（Topic‑SCORE）**：**每一个已有 `membership_r*.npy` 的分辨率**都跑一遍；用 `summary_multires.csv` 里的有效主题相关指标（及 `diagnose_topic_collapse` 一类扩展）做横向比较。
3. **算法速度**：以 `summary.npy` 里记录的每次分区耗时 + 主题建模 `runtime_sec` 为主（全量重跑即可得到干净计时）。
4. **社区可视化（离线）**：UMAP 着色、社区级网络图、规模前 50 社区的子图等——**不进 web 也行**，但目录要统一落在 `out/viz_batch/…`（脚本待补全）。
5. **检索对比**：固定种子论文，对比 **关键词检索 / 向量近邻 / 社区扩散（中心·桥接·同社区）**；在每个 **算法 × 分辨率** 上输出指标。CLI：`experiment-retrieval-benchmark`，默认按 `comparison_breakpoints.csv` 的断点分辨率；产物 `out/comparison_runs/<run_tag>/metrics.jsonl`。总览里的 `retrieval_score` 由 `experiment-eval` 汇总（可用 `--comparison-run-tag` 或环境变量 `PC_EVAL_COMPARISON_RUN_TAGS` 限定只用某次跑的 tag，避免合并历史目录）。

---

## 四种算法在代码里的含义

| 名称 | sweep 目录（默认） | 分区目标 |
|------|-------------------|----------|
| `leiden_cpm` | `out/leiden_sweep_cpm/` | `CPMVertexPartition`，γ 扫描 |
| `leiden`（RB） | `out/leiden_sweep_rb/` | `RBConfigurationVertexPartition` |
| `louvain` | `out/leiden_sweep_louvain/` | **多分辨率**：`RBERVertexPartition`（Reichardt–Bornholdt Potts / ER null，带 `resolution_parameter`）+ Leiden 优化器；与 igraph 单次 `multilevel` 不同 |
| `coarse_kmeans` | `out/coarse_kmeans_then_cpm_k3_seed42/`（示例） | 粗分域内再扫 CPM（或改 `--algorithm`）再合并 |

---

## 分辨率 sweep（默认参数）

`experiment-sweep` / `experiment-coarse-kmeans-sweep` / 按窗口的 sweep **默认**：

- `r_min = 0.001`
- `r_max = 2.0`（覆盖你关心的 **0.2→2.0** 高段，同时保留低段）
- `step = 0.02`（略粗；要更密可改小 `--step`）

Louvain 与 Leiden 共用同一数值网格；**RB / CPM / 模块度的 γ 刻度不可横向硬比**，比较时以「社区数曲线、主题指标、检索指标」为准。

---

## 推荐执行顺序（与你说的 1→2→3 一致）

### 阶段 1 — 四算法 sweep（建议全新跑一遍）

一键编排（只跑 sweep，可再加 `--force` 覆盖已有 membership）：

```bash
bash scripts/offline_comparison_master.sh sweep
```

或手写四条 `experiment-sweep` + 一条 `experiment-coarse-kmeans-sweep`（见脚本内注释）。

### 阶段 2 — 登记表 + eval 包 + 断点总表（现有工具）

```bash
bash scripts/daily_workflow.sh
```

（内含 `experiment-init-minimal`、`experiment-eval-bundle`、`experiment-comparison-breakpoints`、`experiment-eval`。）

### 阶段 3 — 主题建模（默认：每算法约 10 个断点分辨率）

先保证已有断点表（阶段 2 的 `daily_workflow.sh` 会跑 `experiment-comparison-breakpoints`，写出 `out/experiment_eval/comparison_breakpoints.csv`）。**不同算法的 10 个 \(r\) 不必对齐**，CSV 已按各 sweep 单独选好。

```bash
bash scripts/offline_comparison_master.sh topics   # 默认：仅 CSV 中的分辨率
TOPIC_GRID=full bash scripts/offline_comparison_master.sh topics   # 旧行为：r∈[0.0001,5] 筛目录内全部 membership
```

对每个 sweep 目录写 `out/topic_runs/<标签>/K10/`，内含各 `r*/` 与 `summary_multires.csv`。

### 阶段 4 — 五步里的剩余项

- 分层大图：`python src/core.py hierarchy …`（按需；产物大可再归档）。
- **可视化批处理**：默认与主题/检索一致按 **comparison_breakpoints.csv** 的该算法分辨率子集出图（`experiment-viz-batch` 的 `--resolution-source breakpoints`，或由 `offline_comparison_master.sh viz` 传入）。
- **主题可视化（图）**：在已有 `out/umap2d.npy` 与各 sweep 的 `out/topic_runs/...` 前提下：`bash scripts/offline_comparison_master.sh topic-viz` → `out/topic_viz_batch/`（UMAP/曲线/可选质心图；详见 [`offline-outputs-catalog.md`](../../../docs/offline-outputs-catalog.md)）。与 `topics` 共用 **`TOPIC_GRID`**：`TOPIC_GRID=full bash scripts/offline_comparison_master.sh topic-viz` 时对已建模的全分辨率批量出图（子目录 `..._full/`）。
- **检索对比**：默认 **`--resolution-source breakpoints`**（每个 manifest 的 `run_id` 对应 CSV 中约 10 个 \(r\)）；必要时 `--resolution-source summary` 回到扫 summary 网格。

---

## 与「旧文档」的关系

- `experiment-comparison-pipeline.md`：保留 manifest / 路径契约；**当前阶段的对比目标是「全分辨率主题 + 全分辨率检索」**，断点 CSV 仍可作为抽样摘要，不再是唯一评测网格。
- 重新全跑后，旧的 `out/leiden_sweep_*` 与归档里的备份如无对照需要，可只保留 archive。
