# 目标架构与整理说明（实验库 · 算法流 · 前端）

本文档是重构目标的**单一事实来源**：三层分工、算法与分析指标、前端交互。细化契约见同目录 `algorithm-pipeline-spec.md`、`evaluation-metrics-spec.md`；模块边界见 `architecture-boundaries.md`。前端与 API 的当前说明见 [`user_manual_zh.md`](user_manual_zh.md)、[`developer_manual_zh.md`](developer_manual_zh.md)。**历史**分阶段路线图与早期前端契约已归档：[`full-delivery-plan.md`](../archive/legacy/docs/refactor_archived/full-delivery-plan.md)、[`frontend-refactor-spec.md`](../archive/legacy/docs/refactor_archived/frontend-refactor-spec.md)。

---

## 1. 三大板块分离、互相配合

| 板块 | 职责 | 代码主位置 |
|------|------|------------|
| **实验数据库（artifact）** | Run manifest、目录约定、`out/experiments/` 等产物的登记与发现；前端只通过 API 消费摘要 | `src/data_layer/`、`src/foundation_layer/project_paths.py` |
| **算法流** | 向量化 → 建图 → 多算法社区发现 → 可视化产物 → 主题建模 → 评估指标落盘 | `src/algorithm_layer/`、`src/analysis_layer/`、`src/foundation_layer/` |
| **前端可视化** | 三栏布局、选择器、图与详情；不直接依赖脚本内部实现 | `web/`、`src/app_layer/demo_api_app.py` |

CLI 总入口 **`src/core.py`** 负责编排各层任务（见 `src/README.md`）；**`src/` 根目录除 `core.py`（及 `tools/`）外不再放实现模块**，算法与 Demo 代码均在各 `*_layer/`。

---

## 2. 算法流

**最小算法集（当前基线）**

- Louvain、Leiden（RB）、Leiden CPM；以及 **`coarse_kmeans`**（先 k-means 粗分 domain，再在各 domain 诱导子图上跑与上游一致的 sweep，合并标签）。
- 可扩展：再挂其它社区发现方法时，仍走统一 sweep/manifest 契约。

**「先聚类再社区发现」**

- 粗粒度领域（如 embedding 上 k-means）→ 子图内 Leiden/CPM 等；实现入口：`src/algorithm_layer/coarse_domains_kmeans.py`，与 `core.py` 的 `vertexset-hierarchy` 等配合。

**与现有流程一致的全链路**

文本向量化 → 建图 → 社区发现 →（2D/社区图等）可视化产物 → 主题建模 → 算法结果分析；全部产物按 manifest 约定放入可被 API 扫描的路径，供前端选用。

**时间窗实验**

- 离线侧：`1y` / `5y` / `all` 三档（见 `algorithm-pipeline-spec.md`）。
- 前端：可在 selector 中切换时间窗对应的 run。

---

## 3. 算法结果分析（七类）

1. **分辨率 sweep**：曲线/图像；不同算法下分辨率对结构的影响不同，用于对比算法在该数据集上的表现。
2. **算法速度**：各阶段耗时统计。
3. **社区结构可视化**：结构图与导出帧等。
4. **抽样分析**：对比不同算法划分下，同一批论文是否应属同一社区，做论文级对照。
5. **检索能力**：多算法社区检索、关键词、向量最近邻等并列对比；指标与表格化展示可扩展。
6. **主题建模效果**：与 Topic-SCORE 等产出对齐的质量摘要。
7. **面向论文检索场景的实用性**：理论指标好未必最适合本产品的检索体验；该维度作为核心评价角度之一。

落地格式与 API 消费方式见 `evaluation-metrics-spec.md` 与 `experiment-eval` / `/api/v3/evaluations/overview` 等。

---

## 4. 前端可视化（六点）

1. **左侧栏下方**：算法（及关联 run）选择；切换后加载对应社区结构与结果。
2. **中间图**：节点保持简洁；**鼠标悬浮**时在**中间栏下方**展示该论文节点摘要信息。
3. **右侧栏**：主题建模摘要、主要作者、影响因子（若有）、关键词等；论文与社区信息卡片化，优先美观、清晰。
4. **不做默认分层结构可视化**；主线为**算法 → 论文检索**。**时间窗**必选：`1y` / `5y` / `all` 与对应实验联动展示。
5. **左侧关键词搜索**保留；社区相关检索由中间图 + 右侧详情配合；**向量最近邻**信息主要在右侧体现。
6. **中间社区结构支持分辨率调节**（与当前 run 的 sweep 分辨率一致）。

---

## 5. 活跃文档与源码地图（整理后）

| 用途 | 路径 |
|------|------|
| 源码分层说明 | `src/README.md` |
| 模块边界 | `architecture-boundaries.md`（同目录） |
| 管线契约 | `algorithm-pipeline-spec.md`（同目录） |
| 评估指标 | `evaluation-metrics-spec.md`（同目录） |
| 分阶段交付计划（历史） | `../archive/legacy/docs/refactor_archived/full-delivery-plan.md` |
| 用户 / 开发者手册（当前） | `user_manual_zh.md`、`developer_manual_zh.md`（同目录） |
| 早期前端契约（历史） | `../archive/legacy/docs/refactor_archived/frontend-refactor-spec.md` |
| `out/` 归档策略 | `out-archive-policy.md`（同目录） |
| 文档总索引 | `README.md`（同目录） |
| Web 基线（已迁出活跃树） | `../archive/legacy/docs/web_archived/web-version-1.md` |

---

## 6. 归档范围（与「迁移」的关系）

- **放入 `archive/legacy/` 的**：早期 hierarchy-first 方案、已删除的分散规划文档、旧 pipeline 长文、以及按需迁入的过期 `out/` 超大产物（见 `archive/legacy/ARCHIVE_INDEX.md` 与 **`out-archive-policy.md`**）。
- **不归档、保留在仓库活跃路径的**：`src/core.py`（唯一 `src/` 根模块）、各 `*_layer/` 实现、`src/tools/`、`web/`、**`docs/`** 下平铺的说明与规格（见 `README.md`）。

若将来某规格全文被新文档替代，可将旧版迁入 `archive/legacy/docs/` 并在 `README.md` 中更新链接，避免双份「当前真理」。
