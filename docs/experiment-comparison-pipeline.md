# 多算法对比实验管线（已定案）

本文档是 **all-time、四算法、宽分辨率 sweep** 对比流程的**单一事实来源**：目录约定、断点用法、与分阶段交付（历史归档）的衔接。实现时以 manifest + 本文件为准，避免「同一 sweep 多套路径」。

**产物路径与各 CSV/JSONL 列释义**（含 `evaluation_overview`、`comparison_breakpoints`、六路检索表等）见 [`offline-outputs-catalog.md`](offline-outputs-catalog.md)，该文件随实现同步维护。

---

## 1. 已定案的设计选择

| 项目 | 决定 |
|------|------|
| 时间窗 | 第一阶段只做 **all-time**；`1y` / `5y` 后续再加。 |
| 算法 | **四种**均可直接对比：**Louvain**、**Leiden(RB)**、**Leiden(CPM)**、**k-means 粗分 → 各域内社区发现再合并**（划分策略不同，但作为第四种分区参与对比）。 |
| 分辨率 | CLI 默认 **`r_min=0.001` → `r_max=2.0`，`step=0.02`**（可在命令行收紧）；RB / CPM / 模块度 γ **数值不可横向强行对齐**。 |
| 跨算法对齐 | **不要求相同数值分辨率**；可用 §4 断点表做**摘要**；全网格对比以主题指标、检索指标等曲线为准（见归档 [`offline-comparison-master-plan.md`](../archive/legacy/docs/refactor_archived/offline-comparison-master-plan.md)）。 |
| 检索对比 | **目标**：每个算法在 **每一个 sweep 分辨率** 上做检索对比（关键词 / 向量 NN / 社区扩散）；实现进行中，占位 `out/comparison_runs/`。 |
| 主题建模 | 全局主题数 **K = 10**；**目标**：对目录内 **每一个 `membership_r*.npy`** 跑 Topic-SCORE（`topic-model-multi` / `offline_comparison_master.sh topics`）。 |
| 分层结构 | **全图** `hierarchy`：**hierarchy 边表 + layered 类图**在宽范围 sweep 上也要做；接受磁盘与时间成本，**先做再评估**是否需抽稀。 |
| 社区可视化 | **按规模前 50 社区**出子图/UMAP 着色等，属定性浏览，**不作为严谨指标**。 |
| 旧数据 | **归档**到 `archive/`（或约定子目录），不强制物理删除。 |

---

## 2. 实验因子与 manifest

- **因子**：`algorithm` × `time_window=all` × `leiden_dir`（每算法一条主 run，对应一份完整 sweep 目录）。
- **`run_id` 约定（all-time 主 run）**：使用短名 **`leiden_cpm`**、**`leiden`**（RB）、**`louvain`**、**`coarse_kmeans`**，不再带 `_all` 后缀；时间窗占位 run 仍为 **`{algorithm}_{1y|5y}`**（由 `experiment-init-minimal --include-placeholder-windows` 注册）。
- **注册路径**：`out/experiments/<algorithm>/<time_window>__<run_id>/manifest.json`（`experiment-init-minimal` / `experiment-manifest` 写入），`tags` 中至少包含：
  - `sweep`: `{ "r_min", "r_max", "step", "partition_type" }`
  - `graph_npz`, `k`（mutual-kNN）
  - `topic_k`: `10`
  - `breakpoint_policy`: `"n_breakpoints": 10` 及选用规则版本号（见 §4）
- **禁止**：多条 manifest 指向同一 `leiden_dir` 却标不同 `algorithm`（除非刻意复现）；粗分第四条算法使用**独立**输出目录名（例如 `out/sweep_coarse_kmeans_cpm_merged/`）。

---

## 3. 推荐目录布局（条理优先）

以下路径均在仓库 `out/` 下，可按需加日期后缀，但**同一轮实验内命名保持一致**。

```
out/
  experiments/<run_id>/manifest.json

  # 每条算法一条 sweep 根目录（示例名）
  sweep_louvain/
  sweep_leiden_rb/
  sweep_leiden_cpm/
  sweep_coarse_kmeans_cpm_merged/

  # 每条 sweep 内（现有约定）
  <sweep_dir>/summary.npy
  <sweep_dir>/membership_r*.npy

  # 全图 hierarchy（与 sweep 同根或子目录，二选一写死一种）
  <sweep_dir>/hierarchy/   # hierarchy_nodes.csv, hierarchy_edges.csv, breakpoints.json, …
  或
  out/hierarchy_<run_id>/  # 若需与 sweep 根分离，需在 manifest.tags 写明

  # 轻量 eval 包（现有 experiment-eval-bundle）
  <sweep_dir>/eval/
    artifacts.json
    sweep_diagnostics.png
    breakpoints_overview.png
    breakpoints.json
    layer_link_counts.png   # 可选

  # 主题建模（建议按 run + 分辨率组织，便于只跑断点）
  topic_runs/<run_id>/r_<fmt>/
    topic_model_meta.json
    communities_topic_weights.csv
    topics_top_words.csv
    …

  # 检索评测（每 run、每断点分辨率可分子目录或单行 JSONL）
  retrieval_runs/<run_id>/res_<fmt>/
    queries.jsonl           # 或统一 queries + 按列 resolution
    metrics.json

  # 定性可视化
  viz_communities/<run_id>/res_<fmt>/top50_comm_*.png

  # 本轮汇总表（跨算法）
  experiment_eval/
    evaluation_overview.csv
    comparison_breakpoints.csv    # 四算法 × 10 断点的 r 与 n_comm 等
    comparison_retrieval.csv      # 检索指标宽表（可选）
    comparison_topics.csv         # 主题质量宽表（可选）
```

实现阶段若调整子路径，**只改一处 manifest / 本文件**，并保留符号链接或 `artifacts.json` 中的相对路径。

---

## 4. 断点分辨率（每算法 10 个）

**输入**：该算法 `summary.npy` 中的 `resolutions`, `n_comm`（及可选已有 `eval/breakpoints.json`）。

**默认策略**（`breakpoint_policy_v1`，已实现）：

1. 用 `algorithm_layer.community.detect_breakpoints` 按综合分数取候选断点（社区数跳变、相邻层 VI 等）。
2. 若候选 **多于 K**（默认 10）：按分数顺序取满 K；若 **少于 K**：在尚未选中的分辨率上按 **排序后的分辨率轴均匀取点** 补足，直至 `min(K, len(resolutions))`。
3. CLI：`PYTHONPATH=src python src/core.py experiment-comparison-breakpoints` → **`out/experiment_eval/comparison_breakpoints.csv`**（及 `comparison_breakpoints_meta.json`）。列含 `run_id`, `algorithm`, `time_window`, `breakpoint_index`, `resolution`, `n_communities`, `source`（`breakpoint` / `uniform_pad`）等。

实现模块：`src/analysis_layer/comparison_breakpoints.py`。

**用途**：

- **检索对比**：仅对这 10 个 `r` 跑 §5 中的五种打分维度。
- **主题建模（减负模式）**：仅对这 10 个 `r` 调用 `topic_modeling` / 批处理；若日后算力允许再改为「全 sweep 每点」。

---

## 5. 管线顺序（实现与运维）

1. **数据与图**：`data_store.pkl`、embedding、`mutual_knn_k*.npz`、`umap2d.npy`（可视化用）；旧产物移入 `archive/`。
2. **四算法分辨率 sweep**：各自 `r_min/r_max/step`，产出 `membership_r*` + `summary.npy`；记录 wall-clock 与 `summary["time"]` 作为 **算法速度**（§6）。
3. **全图 hierarchy + layered**：对每个 sweep 目录跑 `core.py hierarchy`（或等价 API），生成边表与 layered 图；与现有 eval 轻量图并存。
4. **experiment-eval-bundle**：生成 `eval/` 与 `artifacts.json`。
5. **断点表**：生成 `comparison_breakpoints.csv`（每算法 10 点）。
6. **主题建模**：`K=10`，对每个 run 在 10 个断点 `r` 上产出 §3 中 `topic_runs/...`；汇总 **`comparison_topics.csv`**（有效主题数、mean_top1、collapse 诊断等来自现有 `summary_multires.csv` / `topic_model_meta.json`）。
7. **检索对比**：固定 **query 集**（论文 id 列表 + 随机种子写进 `run_meta.json`）；对每个 run × 10 个 `r` 跑 keyword / embedding NN / community-expansion，算 §6 指标，写入 `retrieval_runs/...` 与 **`comparison_retrieval.csv`**。
8. **定性可视化**：UMAP 着色、社区图、前 50 社区子图 → `viz_communities/...`。
9. **experiment-eval**：刷新 `evaluation_overview`，manifest 中可填 `topic_communities_csv` 指向默认分辨率下的 topic 表（用于 Demo）；完整对比以 `comparison_*.csv` 为准。

---

## 6. 检索对比指标（与已定 scope 对齐）

对每个 query、每个算法、每个**断点分辨率**：

1. **关键词匹配程度**（与 query 论文导出词或 TF-IDF 一致）。
2. **向量相似度**（与当前 embedding 一致）。
3. **主题词匹配**（使用该 `r` 下 Topic-SCORE 的 top words / 主题归属）。
4. **mutual-kNN 图上的最短路距离**（或无权 hop；需在 schema 中固定）。
5. **检索耗时**。

三种检索通道：**关键词**、**向量 NN**、**社区检索**（center / bridge / 社区内扩展等与现有 demo 逻辑对齐）。

---

## 7. 风险与备注（已知接受）

- **全图 hierarchy** 在「大范围分辨率 × 大规模图」下，`hierarchy_edges.csv` 可能极大；你已接受先做；若单机不可行，再在实现层加「按 run 分卷」或「仅保留 layered 渲染所需摘要」而不改本文件的算法对比语义。
- **RB vs CPM** 的 resolution 数值不可横向解释；所有表格必须带 **`algorithm` + `resolution` + `n_communities`**，禁止单独用 `r` 跨算法比较粗细。

---

## 8. 与现有文档的关系

- **历史分阶段路线图（归档）**：[`full-delivery-plan.md`](../archive/legacy/docs/refactor_archived/full-delivery-plan.md) 中的阶段 B（eval 包）、C（检索 scorecard）、D（topic 摘要）由本管线**实例化**；本文件补充 **四算法 + 10 断点 + 全图 hierarchy** 的约定。
- **维护**：路线图正文已冻结在归档；本文件为当前契约。
