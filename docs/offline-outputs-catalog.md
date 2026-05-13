# 离线产物目录说明（数据在哪、是什么、怎么看）

仓库 **三本核心说明** 之一（**输出与数据目录专篇**）：`out/`、`data/` 下默认路径、文件格式、主要列含义与打开方式；**与实现漂移时以本文为路径/列名真理来源**。  
**总流程、代码与命令**见 **`developer_manual_zh.md`**；**网站怎么用**见 **`user_manual_zh.md`**。若你改动了 CLI 的 `--out-dir`，以 manifest 与实际目录为准。仓库根目录下称 `$ROOT`。

**维护约定**：新增或调整离线表（列名、路径、生成命令）时，应同步更新本文档对应小节，避免与 `experiment-eval`、`experiment-retrieval-benchmark`、`retrieval_sixway_aggregate` 实现漂移。

---

## 1. 推荐生成顺序（缺什么就跑到哪一步）

| 顺序 | 命令 / 动作 | 主要产物 |
|------|-------------|----------|
| 1 | `bash scripts/offline_comparison_master.sh sweep` | 四个 sweep 目录、`coarse_domains_*` |
| 2 | `bash scripts/daily_workflow.sh` | `out/experiments/`、`out/experiment_eval/`、`comparison_breakpoints.csv`、各 sweep 下 `eval/`；若存在 `out/comparison_runs/*/metrics.jsonl`，`experiment-eval` 还会写 **六路检索汇总表**（见 §8.2）；若已存在 `out/topic_runs/**/communities_topic_weights.csv`，同一步会汇总 **Topic 有效主题诊断**（见 §6.4） |
| 3 | `bash scripts/offline_comparison_master.sh topics` | `out/topic_runs/.../K10/r*/`；结束时若扫描到 `communities_topic_weights.csv`，会写入 **`out/experiment_eval/topic_collapse_diagnostics/`**（见 §6.4） |
| 4 | `bash scripts/offline_comparison_master.sh viz` | `out/viz_batch/...` |
| 5 | `bash scripts/offline_comparison_master.sh topic-viz` | `out/topic_viz_batch/...`（需先有 `umap2d.npy` 与 topic 结果） |
| 6 | `bash scripts/offline_comparison_master.sh retrieval` | `out/comparison_runs/<tag>/metrics.jsonl` |
| 7 | （可选）仅汇总六路表 | `PYTHONPATH=src python src/core.py experiment-retrieval-sixway` |

主题建模默认只跑 **断点表**里的分辨率（约每算法 10 个 \(r\)）；全网格：`TOPIC_GRID=full bash scripts/offline_comparison_master.sh topics`。更长的离线编排说明见归档 [`offline-comparison-master-plan.md`](../archive/legacy/docs/refactor_archived/offline-comparison-master-plan.md)。

### 1.1 完整性自检（哪些步骤常「看起来没跑完」）

以下 **不** 表示 sweep 失败，只表示下游尚未生成或与上游时间不一致：

| 现象 | 常见原因 | 建议动作 |
|------|----------|----------|
| 无 `out/comparison_runs/**/metrics.jsonl` | 未跑步骤 6 `retrieval` | `bash scripts/offline_comparison_master.sh retrieval`（需先有断点表与 manifest） |
| 无 `comparison_retrieval_sixway_*.csv` | 无可用 `metrics.jsonl`，或用了 `--no-retrieval-sixway` | 先有 `comparison_runs`，再 `python src/core.py experiment-retrieval-sixway` 或 `experiment-eval` |
| `topic_runs/` 或 `viz_batch/` 早于 `membership_r*.npy` | 曾对某 sweep 做 `--no-reuse-existing` 等重算 | 对该 sweep 重跑步骤 3–5（必要时 `TOPIC_GRID=full`） |
| `evaluation_overview` 里 `retrieval_score` 为空 | 无匹配 `run_id` 的 JSONL 行，或未指定 `--comparison-run-tag` / `PC_EVAL_COMPARISON_RUN_TAGS` | 跑 retrieval 后 `experiment-eval --comparison-run-tag <与 metrics 目录名一致>` |

---

## 2. `data/`（输入与中间缓存）

| 路径 | 内容 | 如何查看 |
|------|------|----------|
| `data/data_store.pkl` | 由 RData/txt 解析后的字典缓存 | Python `pickle.load`；一般用 `core.py` 任务间接读 |
| `data/paper_embeddings_specter2.npy` | 论文向量（若 pipeline 使用 SPECTER2） | `numpy.load`；行数应对齐论文数（部分管线保留 1 行占位需与下游约定一致） |
| `data/stopwords.txt` | 主题建模停用词 | 文本编辑器 |

---

## 3. `out/` 图与索引（全局共用）

| 路径 | 内容 | 如何查看 |
|------|------|----------|
| `out/mutual_knn_k*.npz` | mutual-kNN 边与权重 | 由 `network` / `experiment-sweep` 写入；下游 `igraph`/评测脚本读取 |
| `out/umap2d.npy` | 论文 2D 坐标（通常为 `(n_papers, 2)` float32） | `numpy.load`；供网页 demo、`experiment-viz-batch`、`topic-viz` |
| `out/graph_drl2d.npy` | 可选：图布局 2D 坐标 | 存在时 `topic-viz` 会额外输出 `frames_topic_graph/`、`frames_comm_centroid_graph/` |
| `out/keyword_index/` | TF-IDF 类关键词索引 | 检索与 `demo-api`；缺失时可降级子串检索 |

---

## 4. 多分辨率社区 sweep（四种算法）

目录标签与网页 `run_id` 对应关系：

| 文件夹名（`out/` 下） | `comparison_breakpoints.csv` 中的 `run_id` |
|----------------------|---------------------------------------------|
| `leiden_sweep_cpm/` | `leiden_cpm` |
| `leiden_sweep_rb/` | `leiden` |
| `leiden_sweep_louvain/` | `louvain` |
| `coarse_kmeans_then_cpm_k3_seed42/` | `coarse_kmeans` |

### 4.1 每个 sweep 根目录常见文件

| 文件 | 含义 |
|------|------|
| `membership_r{r}.npy` | 长度为论文数的社区标签整数向量 |
| `summary.npy` | 结构化 sweep 摘要（社区数、分辨率列表、耗时等） |
| `eval/` | `experiment-eval-bundle` 生成：`sweep_diagnostics.png`、`breakpoints_overview.png`、`breakpoints.json`、`artifacts.json` 等 |

**怎么看**：诊断图用图片浏览器或 IDE 预览；`summary.npy` 用短 Python 脚本 `numpy.load(..., allow_pickle=True).item()` 查看键名。

### 4.2 粗分领域（k-means）

| 路径 | 含义 |
|------|------|
| `out/coarse_domains_kmeans_k3_seed42/labels.npy` | 每篇论文的领域 id |
| `out/coarse_domains_kmeans_k3_seed42/domain_*_vertex_indices.npy` | 各域顶点全局下标 |
| `out/coarse_domains_kmeans_k3_seed42/meta.json` | k、seed、计数等 |

合并 sweep 在 `out/coarse_kmeans_then_cpm_k3_seed42/`，格式与全图 sweep 相同。

### 4.3 各 sweep 下 `eval/` 三张诊断图（算法无关；粗分读 **合并目录** 的 `summary`）

均由 `experiment-eval-bundle` 写入，数据来自该 sweep 根目录的 `summary.npy`（粗分算法下为 **三 domain 合并后** 的全局社区数与相邻层 VI）。

| 文件 | 横轴 | 纵轴 | 指标怎么算 |
|------|------|------|------------|
| `eval/sweep_diagnostics.png` | **resolution** \(r\) | **上图**：`# communities`（`n_comm`）；**下图**：`VI(adjacent)` | `n_comm`：当前 \(r\) 上不同社区标签个数。`VI(adjacent)`：相邻两个分辨率上两套 membership 的 **variation of information**（越小表示相邻 \(r\) 分区越接近）。断点竖线来自 `detect_breakpoints`（综合「社区数跳变」与 VI 的鲁棒 \(z\) 分数）。 |
| `eval/breakpoints_overview.png` | **resolution index**（`summary` 中 \(r\) 排序后的下标 0…\(T-1\)） | **\|Δ #communities\|** | 对每个下标 \(i\)，纵轴为相邻分辨率上社区数变化的绝对量（实现为对 `n_comm` 做 `diff` 再取绝对值）；粉红竖线标出被选为「对比断点」的索引。注意横轴 **不是** \(r\) 本身，需回到 `summary` 或 `breakpoints.json` 查对应 \(r\)。 |
| `eval/layer_link_counts.png`（可选；需至少两层 membership） | **resolution**（子层 / child 层的 \(r\)） | **\# links**（满足 `child_share` 阈值的父–子链接条数） | 在子采样的相邻 \((r_{\mathrm{parent}}, r_{\mathrm{child}})\) 上，将每个子社区唯一匹配到父社区中交集最大的一个，得到链接；只统计 **子社区内点数占比** `child_share ≥` 默认 **0.1** 的链接。曲线反映「随 \(r\) 变化，分层结构是否仍保持多数可追踪的父子对应」。 |

### 4.4 粗分（`coarse_kmeans`）与分辨率 \(r\) 的约定（主线：**域内 CPM**）

- 先在 embedding 上 **k-means** 得到若干 domain（默认 `k=3`），再在 **全局 mutual-kNN** 上取各 domain 的 **诱导子图**，在每个子图上跑与 `--algorithm` 一致的社区 sweep。默认编排（`scripts/offline_comparison_master.sh sweep`）为 **`leiden_cpm`**（`CPMVertexPartition`），与全图 `leiden_cpm` 可比。
- 各 domain 子目录 `_domain_*_sweep/` 内各有完整 `membership_r*`；**合并目录**根下只保留各 domain **分辨率集合交集**上的结果，并对每个 \(r\) **同时**在三个子图上各跑一次，再把社区 id **按 domain 偏移拼接** 为全局标签。
- 因此：**合并后的 `summary.npy` / `eval/` 曲线描述的是全局合并分区**，不是三条「各 domain 独立」的曲线；`evaluation_overview` 里 `mean_runtime_sec` 等表示 **同一 \(r\) 上三段子图耗时之和**（外加合并开销），与单层全图 sweep **不宜直接按秒横向对比**，除非正文注明。

---

## 5. 实验登记与总览（网页下拉、评测汇总）

| 路径 | 含义 | 如何查看 |
|------|------|----------|
| `out/experiments/<algorithm>/all__<run_id>/manifest.json` | 单条实验的路径与 tags | JSON；`demo-api` 启动时读取 |
| `out/experiment_eval/evaluation_overview.csv` | 多算法多分辨率指标汇总表 | Excel / pandas |
| `out/experiment_eval/evaluation_overview.json` | 同上 + 结构化字段 | JSON |
| `out/experiment_eval/comparison_breakpoints.csv` | 各 `run_id` 选出的代表性分辨率（约 10 个） | CSV；`topics` / `viz` / `topic-viz` / `retrieval` 默认抽样网格 |
| `out/experiment_eval/comparison_breakpoints_meta.json` | 断点策略元信息（请求数、有 summary 的 manifest 数） | JSON；`experiment-comparison-breakpoints` 写入 |
| `out/experiment_eval/comparison_retrieval_sixway_long.csv` | 六路检索对比（长表） | 见 §8.2 |
| `out/experiment_eval/comparison_retrieval_sixway_wide.csv` | 六路检索对比（宽表） | 见 §8.2 |
| `out/experiment_eval/comparison_retrieval_sixway_meta.json` | 六路汇总元信息（baseline `run_id`、Jaccard 等） | JSON |
| `out/experiment_eval/retrieval_sixway_plots/*.png` | 六路检索：横向排名条形图 + **归一化**多指标折线图（`retrieval_metric_lines__*.png`）+ **绝对量**分段折线图（`retrieval_metric_absolute_lines__*.png`；横轴=指标，纵轴按指标分带、带内为绝对均值线性映射；**hops / time_s 带内翻转**使更小（更好）的绝对值在带内更高，区间仍为右侧所标 `[min,max]`） | 见 §8.2；`--no-sixway-plots` 可跳过 |
| `out/experiment_eval/comparison_retrieval_resolution_long.csv` | 按 `comparison_run_tag` × `run_id` × `resolution_effective` × 通道（keyword / vector_nn / community_bundle）聚合后的检索指标（**先对 seed 平均**，不同分辨率分行） | 见 §8.2；与 six-way 同一次写入；`--no-retrieval-sixway` 时跳过 |
| `out/experiment_eval/comparison_retrieval_best_resolution_long.csv` | 每个分区在 **community_bundle** 曲线上用复合分数自选的 **最佳 `r*`** 处的指标；keyword / vector 使用 **baseline 分区** 的 **`r*`** 上的值 | 见 §8.2 |
| `out/experiment_eval/comparison_retrieval_resolution_meta.json` | 各 tag 的 baseline、`r*`、复合分说明等 | JSON |
| `out/experiment_eval/retrieval_resolution_plots/*.png` | 各 tag：**`retrieval_resolution_bundle_vs_r__<tag>.png`** — 四分区子图，每图内为 **community_bundle** 多指标沿 \(r\) 的 **面板内 min–max 归一化**曲线 + **复合分数**粗线 + **\(r*\)** 竖线；**`retrieval_best_pick_metric_lines__<tag>.png`** — 与 six-way **`retrieval_metric_lines`** 相同规则：横轴为 cosine /（可选）kw_tfidf / hops / time_s /（可选）topic，纵轴为各指标在 **六路方法间** 列归一化后的得分，数据为各自 **`r*`** 上的 `at_pick__*`。 | 见 §8.2；`--no-sixway-plots` 可跳过 |

**读 `evaluation_overview.csv` 时的粗分算法**：`coarse_kmeans` 行的 `mean_runtime_sec` / `min_runtime_sec` / `max_runtime_sec` 来自合并目录 `summary.npy` 的 `time` 字段，语义是 **同一分辨率 \(r\) 下三个 domain 诱导子图各自跑社区发现（主线为域内 `leiden_cpm` / CPM）的耗时之和**（外加合成 membership 的少量开销），与只在整张图上跑一次的 `leiden_cpm` / `leiden` / `louvain` **不宜直接按秒数横向对比**，除非在文档或图表里标明这一差异。

### 5.1 `evaluation_overview.csv` / `evaluation_overview.json` 列说明

两文件一行（JSON 里一行对象）对应 **登记表里的一条 manifest**（通常四种社区算法各一行）。列名与 `ExperimentMetricRow` 一致：

| 列名 | 含义 |
|------|------|
| `run_id` | 实验短名，如 `leiden_cpm`、`leiden`（RB）、`louvain`、`coarse_kmeans` |
| `algorithm` | 算法类型字符串（与 manifest 一致） |
| `time_window` | 时间窗，主线多为 `all` |
| `resolution_min` / `resolution_max` | 该 sweep `summary.npy` 中分辨率轴两端 |
| `n_resolution_points` | 分辨率个数（如 101） |
| `mean_runtime_sec` / `min_runtime_sec` / `max_runtime_sec` | 展示用耗时（粗分语义见上文）；若触发「仅 load」类启发式可能被清空并写入 `notes` |
| `mean_n_communities` / `max_n_communities` / `min_n_communities` | 各 \(r\) 上社区数的统计 |
| `retrieval_score` | 由 `out/comparison_runs/.../metrics.jsonl` 汇总的标量（三通道 top‑k 两两 Jaccard 均值）；无数据时为空白 |
| `topic_score` | 由 `out/topic_runs/<sweep>/K*/summary_multires.csv` 中可用行的 `mean_top1_weight` 均值等汇总 |
| `practical_score` | 预留，常为空 |
| `mean_runtime_active_sec` | 来自 `summary.npy` 的「活跃」分区耗时均值（与 cached/computed 统计配套） |
| `n_partitions_cached` / `n_partitions_computed` | 基于 `summary.npy` 中 `time` 是否近似零的粗分桶计数 |
| `eval_sweep_plot` / `eval_breakpoints_plot` / `eval_layered_plot` | 相对各 `leiden_dir/eval/` 的图文件名 |

`evaluation_overview.json` 额外包含：`generated_at_unix`、`notes`（字符串数组，评测脚本写入告警或说明）。

### 5.2 `comparison_breakpoints.csv` 列说明

每行：某 `run_id` 上选中的一个「断点」分辨率（与 `breakpoint_index` 对应）。

| 列名 | 含义 |
|------|------|
| `run_id` / `algorithm` / `time_window` | 与 manifest 一致 |
| `breakpoint_index` | 0…K‑1，本策略下 K 常为 10 |
| `resolution_index` | 该点在 sweep `summary.npy` 分辨率数组中的下标 |
| `resolution` | 断点分辨率 \(r\)（**不可跨 `run_id` 数值对齐**） |
| `n_communities` | 该 \(r\) 上社区数 |
| `delta_n_comm` | 与邻层社区数差等（供排序/诊断） |
| `breakpoint_score` | 综合打分（越大越像「关键断点」）；`source=uniform_pad` 时可能为空字符串 |
| `source` | `breakpoint` 或 `uniform_pad`（候选不足时均匀补足） |
| `breakpoint_policy_version` | 策略版本，如 `breakpoint_policy_v1` |

### 5.3 `comparison_breakpoints_meta.json`

由 `experiment-comparison-breakpoints` 写入，字段包括：`breakpoint_policy_version`、`n_breakpoints_requested`、`n_manifests_with_summary`。

---

## 6. Topic-SCORE：`out/topic_runs/<sweep标签>/K10/`

`sweep标签` 与上表文件夹名一致（如 `leiden_sweep_cpm`）。

### 6.1 树结构

```
out/topic_runs/leiden_sweep_cpm/K10/
  summary_multires.csv          # 多分辨率汇总（状态、耗时、主题质量摘要等）
  r0.0410/                      # 单个分辨率目录（示例）
    topic_model_meta.json
    topics_top_words.csv
    communities_topic_weights.csv
    topic_representative_communities.csv
    A_hat.npy, W_hat_all.npy, … # 矩阵类（建模核心）
```

### 6.2 表格文件怎么读

| 文件 | 列大意 |
|------|--------|
| `topics_top_words.csv` | 每个 topic 的 top 词与权重 |
| `communities_topic_weights.csv` | 每个社区的 topic 权重、top1/top2、规模等 |
| `topic_representative_communities.csv` | 每个 topic 最能代表的社区 |

### 6.3 主题可视化图：`out/topic_viz_batch/<prefix>__<sweep标签>/K10/`

由 `bash scripts/offline_comparison_master.sh topic-viz` 生成（默认前缀 `master`，可用 `TOPIC_VIZ_TAG_PREFIX` 覆盖）。默认与 `topics` 一致只处理 **断点表**中的分辨率；若先已用 `TOPIC_GRID=full` 跑完主题建模，可用 `TOPIC_GRID=full bash scripts/offline_comparison_master.sh topic-viz` 对目录内 **\(r\in[0.0001,5]\) 且已有 topic 子目录** 的全部分辨率出图，输出落在 **`...__<sweep标签>_full/K10/`**，避免覆盖断点版图。

| 路径 | 含义 |
|------|------|
| `topic_legend.png` | topic 编号与颜色图例 |
| `frames_topic_umap/frame_*_r*.png` | 按社区 **Top1 topic** 染色的 UMAP 散点（逐分辨率；默认仅为断点 \(r\)） |
| `frames_topic_graph/…` | 同上，布局为 `graph_drl2d.npy`（若存在且未 `TOPIC_VIZ_SKIP_GRAPH=1`） |
| `frames_comm_centroid_umap/…` | 社区质心散点，颜色=Top1 topic，大小≈规模 |
| `frames_comm_centroid_graph/…` | 质心 + 图布局（条件同上） |
| `curve_ncomm_purity_vs_resolution.png` | 社区数与平均 top1 权重随 \(r\) |
| `curve_topic_prevalence_vs_resolution.png` | 各 topic 加权占比随 \(r\) |
| `curve_top1_mass_vs_resolution.png` | 各 topic 作为 top1 的论文质量占比随 \(r\) |
| `summary_visualization_metrics.csv` | 曲线所用数值表 |
| `viz_meta.json` | 本次可视化参数与路径 |

**环境变量（可选）**：`TOPIC_VIZ_UMAP`、`TOPIC_VIZ_SKIP_GRAPH=1`、`TOPIC_VIZ_MAX_POINTS`、`TOPIC_VIZ_COMMUNITY_CENTROID=0`、`TOPIC_VIZ_ANNOTATE_TOP_N`（质心图上标注规模前 N 个社区 id）。

#### 曲线类 PNG（`topic_visualization_multires.py`）

| 文件 | 横轴 | 纵轴 | 含义（读图要点） |
|------|------|------|------------------|
| `curve_ncomm_purity_vs_resolution.png` | resolution \(r\) | 左轴常为多系列：社区数 `n_comm`、平均 top1 权重等（见实现与 `summary_visualization_metrics.csv`） | 随 \(r\) 看「分区变碎」与「社区内主题是否更尖锐」是否同步。 |
| `curve_topic_prevalence_vs_resolution.png` | \(r\) | 各 **训练主题 id** 的 **prevalence**（跨社区加权平均后的全局占比曲线） | 固定 **K** 个主题槽位下，各主题在语料加权意义下的强度随 \(r\) 的变化；若单线长期主导，提示 **主题塌缩** 风险。 |
| `curve_top1_mass_vs_resolution.png` | \(r\) | 各主题作为 **社区 top1** 的 **论文质量占比**（或计数占比，见脚本） | 与 prevalence 互补：刻画「多少论文质量（或社区）把该主题当作首要解释」。 |

#### 帧图（`frames_*`）

| 目录 | 着色对象 | 颜色含义 |
|------|----------|----------|
| `frames_topic_umap/` | 每篇论文 | 取该论文所在社区的 **Top1 topic** 映射到离散调色板（见 `topic_legend.png`）。**不是**「同一社区渐变」，而是 **topic 类别色**。 |
| `frames_topic_graph/` | 同上 | 若存在 `out/graph_drl2d.npy`：2D 坐标来自 **图布局** 而非 UMAP。 |
| `frames_comm_centroid_umap/` | **社区**（一个点一个社区） | 位置 = 该社区论文在 UMAP 上的 **质心**；颜色 = 该社区 Top1 topic；点大小 ∝ 社区规模。 |

### 6.4 Topic 有效主题诊断（`diagnose_topic_collapse`）

**命令**：`PYTHONPATH=src python src/core.py diagnose-topic-collapse --root out/topic_runs --out-dir out/experiment_eval/topic_collapse_diagnostics --save-per-topic --save-plots`  
**触发**：`bash scripts/daily_workflow.sh` 在 **`out/topic_runs` 下已存在** `communities_topic_weights.csv` 时自动执行；`bash scripts/offline_comparison_master.sh topics` 结束后亦会执行。

**输入**：递归扫描 `out/topic_runs/**/communities_topic_weights.csv`（通常位于各 `r*/` 子目录）。

**输出目录**：`out/experiment_eval/topic_collapse_diagnostics/`

| 文件 | 内容 |
|------|------|
| `topic_collapse_diagnostics.csv` | 每个 CSV 一行：分辨率（从路径 `.../r0.xxxx/` 推断）、`global_effective_topics_shannon` / `global_effective_topics_simpson`（由 **全局 prevalence** 熵得到的「有效主题数」估计）、`mean_top1_weight`、主导主题的 top1 占比、`topic_words_*`（`topics_top_words.csv` 词集合两两 Jaccard）等 |
| `topic_collapse_per_topic_long.csv` | （`--save-per-topic`）长表：每个 \(r\) × 每个 topic 的 `prevalence`、`top1_share_count`、`top1_share_weight` |
| `topic_collapse_diagnostics_failed.csv` | 解析失败的输入路径与错误信息 |
| `effective_topics_vs_resolution.png` | 横轴：\(r\)；纵轴：**有效主题数**（Shannon 与 Simpson 两种估计）；多 segment 时按组曲线 |
| `top1_dominance_vs_resolution.png` | 横轴：\(r\)；纵轴：主导 topic 的 top1 占比、平均 top1 权重 |
| `rowsum_abs_err_max_vs_resolution.png` | 行归一化 sanity：\(\max |\mathrm{rowsum}-1|\) 随 \(r\) |
| `topic_prevalence_vs_resolution*.png` | （可选）各 topic 的 `prevalence` 随 \(r\) 的细线 |

**指标释义**：`global_effective_topics_shannon` = \(\exp(H)\)，\(H\) 为全局 prevalence 分布的 Shannon 熵；若接近 1 表示几乎只有一个主题占主导。`community_effective_topics_*` 为对每个社区的主题权重向量再算有效主题数后的统计量。

---

## 7. 社区结构可视化：`out/viz_batch/<prefix>__<sweep标签>/`

由 `offline_comparison_master.sh viz` 调用 `experiment-viz-batch` 生成；默认按 **断点表** 中的分辨率（每算法约 10 个 \(r\)）出图，目录常为 `out/viz_batch/<prefix>__<sweep>/r0.xxxx/`。

### 7.1 UMAP 按 **社区** 着色（`umap_membership.png`）

- **横 / 纵轴**：`out/umap2d.npy` 中的 2D 坐标（无物理单位，仅相对布局）。
- **颜色**：每个 **社区 id** 映射到 `viridis` 上的一点；映射规则为 `labels_to_colors_by_centroid`：先在 UMAP 上算各社区质心，再按质心 **x 坐标排序** 后均匀取色。效果是 **一社区一色**，不是同一社区内的渐变。
- **标题**：含有效 membership 文件路径与最近邻 \(r\)。

### 7.2 粗分域图（`umap_kmeans_domains.png`，可选）

- 当 `leiden_dir` 能解析到 k-means 的 `labels.npy`（或 `coarse_kmeans_sweep_meta.json` 指向的域标签）时生成。
- **颜色**：domain id（0…\(K-1\)），与 **合并后社区着色** 不同；用于区分「embedding 粗分域」与「域内 CPM 细分社区」。

### 7.3 社区元图（`community_meta_graph.png`）

- **点**：规模前 `meta_graph_max_nodes` 个社区；**大小** ∝ 社区规模；**标签**为社区 id。
- **边**：社区间在原始 mutual-kNN 上的聚合边权（粗细 ∝ 权重）；布局为 Force-directed（igraph）。
- **用途**：粗看「大社区之间是否强连接」；与 UMAP 点图互补。

### 7.4 大社区诱导子图（`subgraphs_top*/c*.png`）

- 在 UMAP 上只绘制该社区顶点；边为子图内边（灰色半透明）。
- **用途**：检查单个巨型社区内部是否仍呈流形/多峰结构。

---

## 8. 检索对比：`out/comparison_runs/<run_tag>/`

### 8.1 `metrics.jsonl`（原始行）

**路径**：`out/comparison_runs/<run_tag>/metrics.jsonl`（`<run_tag>` 由 `experiment-retrieval-benchmark --run-tag` 或 `offline_comparison_master.sh` 里 `RETRIEVAL_TAG` 决定）。

**生成**：`PYTHONPATH=src python src/core.py experiment-retrieval-benchmark ...` 或 `bash scripts/offline_comparison_master.sh retrieval`。

每行一个 JSON 对象，一次评测 =（一条 manifest 的 `run_id`）×（一个分辨率）×（一个 `seed_pid`）。主要键：

| 键 | 含义 |
|----|------|
| `run_id` / `algorithm` / `time_window` | 与实验登记一致 |
| `leiden_dir` / `graph_npz` | 使用的 sweep 目录与图 |
| `resolution_requested` / `resolution_effective` | 请求的 \(r\) 与实际加载 membership 的 \(r\)（最近邻） |
| `seed_pid` | 作为 query 的论文 id（1-based，与 `Paper` 下标一致） |
| `keyword_query_chars` | 构造关键词 query 的字符数 |
| `topic_communities_csv` | 写入时选用的 `communities_topic_weights.csv` 路径；若未显式传 `--topic-root`，**当** `out/topic_runs/<与 run_id 对应的 sweep 目录名>/K{topic_k}/` 存在时会自动按分辨率就近匹配，否则 `null` |
| `methods` | 三种检索通道的结果，见下表 |
| `pairwise_jaccard` | 三种 top‑k 集合两两 Jaccard：`keyword__vector_nn`、`keyword__community_bundle`、`vector_nn__community_bundle` |

`methods` 下各子键（`keyword`、`vector_nn`、`community_bundle`）对象常见字段：

| 字段 | 含义 |
|------|------|
| `method` | 通道名（与键相同） |
| `pids` | top‑k 论文 id 列表 |
| `time_sec` | 该通道耗时（秒） |
| `n_results` | 返回条数 |
| `mean_cosine_to_seed` | 结果相对 query 论文向量的平均余弦相似度 |
| `mean_keyword_tfidf` | （可选）用与关键词检索相同的 TF‑IDF 向量，对返回的每篇论文算查询–文档点积，再对 top‑k 取平均；**三种通道都会写**（便于比较「向量/社区 bundle 选出的论文」与 seed 文本的字面匹配度）。无 `keyword_index` 或加载失败时该字段省略 |
| `mean_shortest_path_hops` | 在 mutual‑kNN 图上的平均最短路 hop（不可达计入统计见实现） |
| `n_unreachable_in_graph` | 最短路为无穷的点数 |
| `topic_top1_match_rate_vs_seed_community` | 若加载了 topic 表：结果论文与 seed 所在社区 top1 topic 一致的比例 |
| `extra` | 如关键词 `search_debug`、向量的 `similarities` 等 |

### 8.2 六路检索汇总表（`experiment-eval` / `experiment-retrieval-sixway`）

在存在至少一个非空 `metrics.jsonl` 时，默认会写入 **`out/experiment_eval/`** 下三个文件（可用 `experiment-eval --no-retrieval-sixway` 关闭；亦可单独运行 `python src/core.py experiment-retrieval-sixway`）。**同一条件下**还会写入按分辨率的 **`comparison_retrieval_resolution_long.csv`**、**`comparison_retrieval_best_resolution_long.csv`**、**`comparison_retrieval_resolution_meta.json`**（与 six-way 一并关闭）。

**语义（6 路）**

| 汇总行 `method` | 含义 |
|-----------------|------|
| `keyword` | TF‑IDF 关键词检索；**与分区无关**，只统计一次，取自 **canonical** `run_id` 的行（优先顺序：`leiden_cpm` → `leiden` → `louvain` → `coarse_kmeans`） |
| `vector_nn` | 向量近邻；同上，与 canonical 行的 keyword 同源 |
| `community_bundle:<run_id>` | 社区 bundle 检索，**四种分区各一行**（`leiden_cpm`、`leiden`、`louvain`、`coarse_kmeans`） |

**`comparison_retrieval_sixway_long.csv`（长表）** — 每个 `comparison_run_tag` 固定 **6 行**（若某 `run_id` 在 JSONL 中完全缺失则该行 `n_lines=0` 等）。列：

| 列名 | 含义 |
|------|------|
| `comparison_run_tag` | 对应 `out/comparison_runs/<tag>/` 目录名 |
| `method` | `keyword` / `vector_nn` / `community_bundle:<run_id>` |
| `partition_run_id` | 社区行填 `run_id`；keyword/vector 为空 |
| `baseline_run_id_used` | keyword/vector 行填实际选用的 canonical `run_id`；社区行为空 |
| `n_lines` | 参与聚合的 JSONL 行数 |
| `mean_cosine_mean` / `mean_cosine_std` / `mean_cosine_n` | 对 `methods.*.mean_cosine_to_seed` 的均值、总体标准差、有效个数 |
| `mean_hops_mean` / `mean_hops_std` / `mean_hops_n` | 对 `mean_shortest_path_hops` 同上 |
| `time_sec_mean` / `time_sec_std` / `time_sec_n` | 对 `time_sec` 同上 |
| `mean_keyword_tfidf_mean` / `mean_keyword_tfidf_std` / `mean_keyword_tfidf_n` | 对 `mean_keyword_tfidf` 同上（无索引或 JSONL 中无该字段时常为 `n=0`） |
| `topic_top1_match_mean` / `topic_top1_match_std` / `topic_top1_match_n` | 对 `topic_top1_match_rate_vs_seed_community` 同上（未传 topic 时常为空） |

**不要把 `*_n` 理解成「检索返回了多少篇论文」**：

- 原始 `metrics.jsonl` 里，每个通道的 **`n_results`** 才是该次查询返回的篇数（默认 `--top-k 20`，社区 bundle 可能更少）。
- 汇总表里的 **`mean_cosine_n` / `mean_hops_n` 等**：表示在聚合「跨多少条 JSONL 评测记录」时，该标量有 **多少个有限值**。例如 `mean_cosine_n=30` 通常对应 **30 条**（`seed_pid` × `resolution_effective` × 当前 `run_id`）评测行，每条内部再用最多 **top_k** 篇结果去算 `mean_cosine_to_seed`。

**`topic_top1_match_*` 全空**：若 JSONL 里 `topic_communities_csv` 为 `null`，说明该 `run_id` 下没有可用的 `out/topic_runs/<sweep>/K*/` 目录（或断点分辨率在 topic 结果中找不到邻近 `r*` 子目录）。先跑 `offline_comparison_master.sh topics`；仍可为单次评测显式传 **`--topic-root out/topic_runs/leiden_sweep_cpm/K10`** 等路径覆盖自动推断。

**排名条形图（默认生成）**：与长表同一次写入时，在 **`out/experiment_eval/retrieval_sixway_plots/`**（与 `--out-dir` 并列子目录）下为每个 `comparison_run_tag` 输出 `retrieval_rankings__<tag>.png`（余弦、hops、耗时为横向排名；仅当 `mean_keyword_tfidf_n>0` 时增加 TF‑IDF 匹配度子图；仅当 `topic_top1_match_n>0` 时画 topic 子图）。另输出 **`retrieval_metric_lines__<tag>.png`**：横轴为 `cosine` /（有 TF‑IDF 时）`kw_tfidf` / `hops` / `time_s` /（有 topic 时）`topic_top1`，**六根线**对应六路方法；纵轴为「按列 min–max 归一化到 0–1」后的得分（hops 与 time 按「越小越好」翻转后再归一化），便于不同量纲画在同一张折线图里。再输出 **`retrieval_metric_absolute_lines__<tag>.png`**：**单张直角坐标折线图**，与归一化折线图同样「横轴 = 各指标、每路方法一条折线」；**纵轴不按统一物理刻度**，而是每个指标占一条水平**带**，带内纵坐标在该指标六路方法的 **``[min,max]`` 绝对均值**之间线性映射（右侧文字标出各带区间；**hops、time_s** 在带内**翻转**：绝对值越小在带内越高，与「越小越好」一致）。**不要跨带比较纵坐标高低**；带内可比较六路相对位置。`experiment-eval` 返回路径键 **`metric_absolute_banded_plots`**（别名 **`metric_absolute_facets_plots`** 同路径）。可用 `experiment-eval` / `experiment-retrieval-sixway` / `experiment-eval-bundle` 的 **`--no-sixway-plots`** 关闭。若只需从已有 CSV 补图：`python scripts/plot_retrieval_sixway_rankings.py`（脚本自举 `src/` 到 `sys.path`，见脚本内说明）。

**`comparison_retrieval_sixway_wide.csv`（宽表）** — 每个 `comparison_run_tag` **一行**：固定列 `comparison_run_tag`、`baseline_run_id_for_keyword_vector`、`n_jsonl_lines`；接着是在 canonical baseline 子集上统计的三条 pairwise Jaccard，列名为 `baseline_<pair>_mean` / `baseline_<pair>_std` / `baseline_<pair>_n`，其中 `<pair>` 为 `keyword__vector_nn`、`keyword__community_bundle`、`vector_nn__community_bundle`；再将长表六路指标展开为列，命名规则为 `{method键}__{指标名}`，`method` 中的 `:` 变为 `_`（例如 `keyword__mean_cosine_mean`、`community_bundle_leiden_cpm__time_sec_mean`）。

**`comparison_retrieval_sixway_meta.json`** — `description` 为固定说明文字；`runs` 数组每项对应一个 tag，含 `baseline_run_id_for_keyword_vector`、`n_jsonl_lines`、`pairwise_jaccard_on_baseline_rows`（在 canonical 子集上统计的三条 Jaccard）。

**按分辨率的长表与「各算法最佳 \(r\)」**（`comparison_retrieval_resolution_long.csv` / `comparison_retrieval_best_resolution_long.csv` / `comparison_retrieval_resolution_meta.json`）

- **长表**：对每个 `comparison_run_tag`，按 **`run_id`** 与 **`resolution_effective`**（保留 6 位小数对齐）分组，在组内对 **seed** 聚合得到 keyword、vector_nn、community_bundle 三路指标（列名与 six-way 中 `mean_*` 族一致）。因此一行 = 一个分区在一个分辨率上的三路表现，而不是把全部分辨率混成 six-way 的一行均值。
- **最佳 \(r*\)**：对每个社区分区 `run_id`，在 **该分区** 的 `community_bundle` 指标随 \(r\) 变化的序列上，计算 **复合分数**（对每个指标在 **该 `run_id` 的所有分辨率** 上做 min–max 归一化；`mean_hops_mean` 与 `time_sec_mean` 按「越小越好」翻转后再归一化；对 cosine、（若有）TF‑IDF、（若有）topic、hops、time 取 **可用项的平均**），取复合分 **最大** 的 \(r\) 为 **`r*`**。**keyword / vector_nn** 的两行使用 **canonical baseline 分区**（与 six-way 相同优先级）在 **其** `community_bundle` 曲线上选出的 **同一个 \(r*\)**，并在该 \(r*\) 上读取 baseline 行上的 keyword / vector 聚合指标（列前缀 `at_pick__...`）。
- **图**：`retrieval_resolution_plots/retrieval_resolution_bundle_vs_r__<tag>.png`（四分区：多指标 + 复合分数 vs \(r\)，红线标 `r*`）；`retrieval_best_pick_metric_lines__<tag>.png`（与 six-way 折线图同构的 **at_pick** 多指标对比）。`experiment-eval` 等返回体中的 **`resolution_breakdown`** 含 CSV 路径；若开启画图，另有 **`resolution_curve_plots`**、**`best_pick_bar_plots`**（失败时可能含 **`resolution_plots_error`**）。仅从已有 CSV 补分辨率图：`python scripts/plot_retrieval_sixway_rankings.py --also-resolution-plots`。

**与 `experiment-eval` 的衔接**：`--comparison-run-tag TAG` 可重复；环境变量 `PC_EVAL_COMPARISON_RUN_TAGS`（逗号分隔）等价。未指定时读取 `out/comparison_runs` 下**所有**含 `metrics.jsonl` 的子目录并各写一组六路行（宽表多行）。

### 8.3 检索 PNG 速查（写报告时对齐 §8.2）

以下文件均在 `out/experiment_eval/retrieval_sixway_plots/` 或 `out/experiment_eval/retrieval_resolution_plots/`；`<tag>` 为 `comparison_run_tag`（如 `master_breakpoints`）。

| 文件（模式） | 横轴 | 纵轴 | 读图注意 |
|--------------|------|------|----------|
| `retrieval_rankings__<tag>.png` | 子面板按 **指标** 分栏；栏内为六路方法的 **横向条形** | 条形长度 = 该指标下经 **排名/得分规则** 后的相对优劣 | 不同子图量纲不同，**勿跨子图数值相加**。 |
| `retrieval_metric_lines__<tag>.png` | 离散指标位点：`cosine` /（可选）`kw_tfidf` / `hops` / `time_s` /（可选）`topic_top1` | **0–1**：对每个指标列在六路间 **min–max 归一化**；`hops`、`time_s` 已按「越小越好」翻转 | 同一指标位点内比较六路折线 **点的高低** 有意义。 |
| `retrieval_metric_absolute_lines__<tag>.png` | 同上 | 每个指标一条水平 **带**，带内纵坐标映射该指标六路 **绝对均值** 到 `[min,max]`（带旁标注区间）；`hops`、`time_s` 在带内翻转 | **禁止跨带比较纵坐标**；仅带内比较。 |
| `retrieval_resolution_bundle_vs_r__<tag>.png` | resolution \(r\) | **四宫格**：每格为 `community_bundle` 若干指标沿 \(r\) 的曲线（**格内** min–max 归一化）+ 复合分数粗线 + **`r*`** 竖线 | 归一化仅在 **单格内**；四算法的 \(r\) 轴数值 **不对齐** 属正常。 |
| `retrieval_best_pick_metric_lines__<tag>.png` | 指标位点 | 各方法在各自 **`r*`** 上的 `at_pick` 指标，经与 six-way 相同规则归一化 | 用于「在自选的 \(r^*\) 上看多指标轮廓」。 |

---

## 9. 归档与大体积历史产物

[`out/README.txt`](../out/README.txt) 说明当前仓库选择提交/忽略的 `out/` 内容；旧的大目录可能迁至 `archive/legacy/out/...`，以其中 `README.txt` 为准。

---

## 10. 相关文档

- 编排总览（归档）：[`archive/legacy/docs/refactor_archived/offline-comparison-master-plan.md`](../archive/legacy/docs/refactor_archived/offline-comparison-master-plan.md)
- Manifest 与 run_id：[`experiment-comparison-pipeline.md`](experiment-comparison-pipeline.md)
- 网页 demo（早期基线，已归档）：[`web-version-1.md`](../archive/legacy/docs/web_archived/web-version-1.md)
- LaTeX 实验报告（编译说明见其中 `README.md`）：[`../experiment_report/README.md`](../experiment_report/README.md)

实现侧若调整列名或路径，请同时改：**本文档**、`src/analysis_layer/retrieval_sixway_aggregate.py`、`src/analysis_layer/retrieval_sixway_plots.py`、`src/analysis_layer/retrieval_resolution_analysis.py`、`src/analysis_layer/retrieval_benchmark.py`、`src/analysis_layer/evaluation_metrics.py`、**`src/analysis_layer/diagnose_topic_collapse.py`** 中与表头相关的导出逻辑。
