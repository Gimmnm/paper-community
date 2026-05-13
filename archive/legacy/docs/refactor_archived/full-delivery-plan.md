# 完整交付计划：实验库 · 算法流 · 前端（分阶段）

本文档在 [`requirements-breakdown.md`](../../docs/requirements-breakdown.md) 的目标陈述之上，给出**可验收的分阶段路线图**与**依赖关系**。细化契约仍以 [`algorithm-pipeline-spec.md`](../../docs/algorithm-pipeline-spec.md)、[`evaluation-metrics-spec.md`](../../docs/evaluation-metrics-spec.md)、`frontend-refactor-spec.md` 为准；本文件负责「何时做哪一块、做到什么算完成」。

---

## 总原则

1. **三块分离**：产物与 manifest（实验库）→ 离线脚本与 `core.py` 任务（算法流）→ `demo-api` + `web/`（前端）。前端只通过 API 读摘要与静态资源，不直接依赖算法模块。
2. **先契约后功能**：每一阶段先约定落盘文件名与 JSON shape，再实现生成与消费，避免 Eval 面板与 CSV 再次脱节。
3. **与当前主线一致**：默认不做交互式「分层钻取」网页；分辨率 sweep / 断点 / 分层图作为**离线分析资产**，在 Eval 与文档中引用。

**多算法对比（all-time、四算法、宽 sweep）** 的流程与「全分辨率主题 / 检索」目标见 **[`offline-comparison-master-plan.md`](offline-comparison-master-plan.md)**；manifest 与路径契约仍以 **[`experiment-comparison-pipeline.md`](../../docs/experiment-comparison-pipeline.md)** 为准（断点 CSV 可作摘要，不等价于唯一评测网格）。

---

## 阶段 A — 评估摘要可信 + 离线图表可发现（当前迭代）

**目标**

- `evaluation_overview` / `/api/v3/evaluations/overview` 除 `summary.npy` 外，能反映：
  - **真实运行耗时**（区分「缓存命中记 0」与「实际计算」）。
  - **discovered 或可声明**的 sweep / breakpoints / layered 图像路径（相对各 run 的 `leiden_dir`）。
- 浏览器可通过 **受限制路径校验** 的 API 拉取上述 PNG/SVG，Eval 卡片展示缩略或链接。

**验收**

- CSV/JSON 中出现 `mean_runtime_active_sec`、`n_partitions_cached`、`n_partitions_computed` 及 `eval_*_plot` 字段（见 `evaluation-metrics-spec.md`）。
- `GET /api/v3/runs/{run_id}/eval-artifact/{rel_path}` 仅允许读取该 manifest `leiden_dir` 下的文件。
- Eval Tab 加载 overview 后能看到耗时说明与已有图表（若目录中存在或通过 `eval/artifacts.json` 声明）。

**依赖**：无；算法侧可选提供 `eval/artifacts.json`。

**状态（2026-05-09）**：已实现代码与文档：`evaluation_metrics.discover_eval_plot_paths`、`ExperimentMetricRow` 扩展字段、`GET /api/v3/runs/{run_id}/eval-artifact/{rel_path}`、Eval 面板展示图与「computed-only」耗时；`experiment-eval` 可重生成 CSV/JSON。

---

## 阶段 B — 每条算法 run 的稳定离线「分析包」

**目标**

- 对每条 manifest（Louvain / Leiden / Leiden CPM / coarse_kmeans × 时间窗），批量产出：
  - 分辨率 sweep 诊断图（对标历史 `sweep_diagnostics` / `cpm_breakpoints` 类产物）。
  - 可选：分层边表上的静态摘要图（不必恢复旧 `hierarchy_viz.py`，可用轻量 matplotlib/`community.plot_sweep_diagnostics` 现有能力扩展）。
- 后置任务在每条 run 的 **`leiden_dir/eval/`** 下写入 PNG、`breakpoints.json` 与 **`eval/artifacts.json`**（路径相对于 `leiden_dir`）。

**验收**

- 文档中列出单一 CLI 入口与最少命令示例；跑完后 `eval/artifacts.json` 存在且 Eval API 能显示链接。

**状态（2026-05-09）**：已实现 `analysis_layer/eval_bundle.py` 与 CLI `experiment-eval-bundle`。对 catalog 中每条 manifest：若有 `summary.npy` 则写出 `eval/sweep_diagnostics.png`、`eval/breakpoints_overview.png`、`eval/breakpoints.json`、可选 `eval/layer_link_counts.png`（需至少两层 membership）、`eval/artifacts.json`。缺 `summary.npy` 的 run 会跳过并记录在 JSON 摘要中。

最少命令（与 `experiment-eval` 共用 fallback / catalog 约定）：

```bash
PYTHONPATH=src python src/core.py experiment-eval-bundle --also-overview
```

仅处理部分 run：`--run-id <id1> <id2>`。已有产物时用 `--skip-existing`；覆盖用 `--force`。禁用分层链接图：`--no-layered`。

---

## 阶段 C — 检索与向量对比 scorecard

**目标**

- 离线脚本：关键词 TF-IDF、向量最近邻（与 manifest 图一致）、以及「按社区扩展」等策略，在固定查询集上产出 **`eval/scorecard.json`**（含 `retrieval_score` 等），并可扩展多列指标表。
- `build_evaluation_overview` 已支持读取 scorecard；本阶段补齐**生成器**与指标定义（表格 schema）。

**验收**

- `evaluation_overview.csv` 中 `retrieval_score` 等列对至少一条 run 非空；附 `eval/retrieval_table.csv`（或 JSON）供前端后续表格化。

---

## 阶段 D — 主题建模效果摘要

**目标**

- 将 Topic-SCORE 管线输出的既有 CSV/JSON（或 `topic_model_meta.json`）汇总为 **`eval/topic_summary.json`**（每 run 可选），并入 overview 的一列或 Eval 详情引用。

**验收**

- 至少一条带 `topic_communities_csv` 的 run 能在 Eval 或 Details 侧链到 topic 摘要路径。

---

## 阶段 E — 抽样 / 论文对齐分析（离线）

**目标**

- 固定论文样本（或随机分层样本），输出「算法 A vs B 是否在同类社区」的混淆式指标（ARI/NMI 或对同一 query 集合的邻域重合度）；落盘 **`eval/pairwise_agreement.json`**（可多对算法）。

**验收**

- 文档说明输入（PID 列表或抽样种子）与输出字段；不要求首页默认展示。

---

## 阶段 F — 「论文检索场景」实用性（核心维度）

**目标**

- 定义 `practical_score` 的人类可解释构造（例如：案例查询集 + 人工/弱监督标签 + 与社区检索命中率结合）；写入 scorecard 扩展字段。
- 与阶段 C 区分：C 偏自动离线指标，F 偏产品与场景对齐。

**验收**

- `docs/evaluation-metrics-spec.md` 中增加 `practical_score` 可操作定义；示例 scorecard 一份。

---

## 阶段 G — 前端（对齐 requirements §4）

按 [`frontend-refactor-spec.md`](frontend-refactor-spec.md) 与 [`requirements-breakdown.md`](../../docs/requirements-breakdown.md) §4 执行，建议顺序：

1. **中间栏下方悬浮论文信息条**（全局节点 hover）。
2. **右侧信息密度**：作者 / IF / 关键词 / topic 卡片样式统一（已有数据通路基础上纯 UI）。
3. **向量最近邻**：右侧展示 neighbors（API 已有 mutual-kNN；若需 true embedding NN 则新增轻量端点）。
4. **算法切换**：已由 Experiment selector + session 覆盖；后续可加「快捷跳转当前 run 默认分辨率」。

**验收**

- 每项有可截图或可描述的交互完成定义；不阻塞离线阶段 B–F。

---

## 维护

- 完成某一阶段后，在本文件对应小节末补「完成日期 / PR / 备注」。
- 若阶段范围变化，先改 `../../docs/requirements-breakdown.md` 再同步本文件，避免双份冲突。
- 多算法对比的断点数、主题数 K、hierarchy 范围等若变更，先改 **`../../docs/experiment-comparison-pipeline.md`**，再在此文件引用处核对一遍。
