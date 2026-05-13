# Web 前端数据就绪清单（算法产物 → 展示）

本文把「算法与离线流水线已经产出什么」与「当前 `demo-api` + `web/` 能否直接消费」对齐，便于下一版把 **四种分区算法、社区结构、社区信息、六种检索（含分辨率维度）** 接到页面上。时间窗相关能力标为可选（你方暂不做）。

---

## 1. 总览矩阵

| 能力 | 主要数据源（仓库内） | 当前 Web 侧 | 前端就绪 |
|------|----------------------|-------------|----------|
| 多算法切换（四种 `run_id`） | `out/experiments/<algorithm>/all__<run_id>/manifest.json` | `GET /api/v3/catalog`、`/api/v3/session/switch` | **就绪**：须保证四条 manifest 存在且 `leiden_dir` 各指向对应 sweep |
| 分辨率列表与切换 | 各 manifest 的 `leiden_dir/summary.npy` → `resolutions` | `GET /api/v3/runs/{run_id}/resolutions`、`/api/v3/session/switch?resolution=`；前端 slider | **就绪** |
| 全局社区图 / 社区内子图 | `membership_r*.npy`、`mutual_knn_k*.npz`、`umap2d.npy` | `GET /api/graph/communities`、`/api/graph/community/{cid}` 等 | **就绪**（随当前 session 的 `run_id`+`resolution`） |
| 论文与社区摘要信息 | `Paper` 内存、`DemoCommunityGraph`、可选 `topic_communities_csv` | `GET /api/papers/{id}`、`/api/communities/{cid}`、`/api/expand/paper/...` | **部分就绪**：无「社区级 CSV 全文」直出；topic 依赖 manifest 或 `PC_TOPIC_COMMUNITIES_CSV` |
| 关键词检索 | `out/keyword_index/` | `GET /api/search/keyword` | **就绪** |
| 向量近邻 / 社区 bundle **在线**检索 | 与 `retrieval_benchmark` 相同逻辑（图 + membership + 向量） | **无** 独立 HTTP 路由 | **未就绪**：需新增 API 或沿用离线指标-only |
| 六种检索 **离线**对比（keyword / vector_nn / 四路 community_bundle） | `out/experiment_eval/comparison_retrieval_sixway_*.csv/json`、`retrieval_sixway_plots/*.png` | **无** 专用路由；文件在 `out/` 下 | **数据就绪、接入未就绪**：需静态挂载或 `GET` 返回 JSON/图 URL |
| 按分辨率的检索曲线与 best-\(r*\) | `comparison_retrieval_resolution_long.csv`、`comparison_retrieval_best_resolution_long.csv`、`retrieval_resolution_plots/`（`retrieval_resolution_bundle_vs_r__*.png`、`retrieval_best_pick_metric_lines__*.png`） | 同上 | **同上**（跑过带分辨率汇总的 `experiment-eval` / `experiment-retrieval-sixway` 后才有文件） |
| 登记表级评测总览 | `out/experiment_eval/evaluation_overview.json` | `GET /api/v3/evaluations/overview`（内存拼 `ExperimentEvaluationBundle`） | **就绪**：与落盘 JSON 同源信息；**不含** six-way 明细列 |
| 断点分辨率表 | `comparison_breakpoints.csv` | 无 | **数据就绪、接入未就绪** |
| 多 `comparison_run_tag` 的检索跑批 | `out/comparison_runs/<tag>/metrics.jsonl` | 仅用于服务端算 `retrieval_score`（及离线聚合） | **间接就绪**：前端若要比多个 tag，需扩展 env/API |

---

## 2. 四种算法的社区结构

| 项目 | 说明 |
|------|------|
| **数据** | 每个算法一条 manifest：`run_id` ∈ `leiden_cpm` / `leiden` / `louvain` / `coarse_kmeans`，`leiden_dir` 指向该算法 sweep（含 `summary.npy`、`membership_r*.npy`）。 |
| **前端** | `catalog.runs[]` 每条含 `algorithm`、`leiden_dir`、`default_resolution` 等；切换 run 后 `health` / `graph` 使用对应目录加载 membership。 |
| **检查** | 确认 `out/experiments/*/*/manifest.json` 共 **4** 条（或你实际要上的数量），且路径在部署机器上可读。缺 manifest 时仅有 `cli_default` 单一路径，**无法**在页面上对比四算法。 |

---

## 3. 社区「信息」指什么、数据在哪

| 信息类型 | 来源 | Web 现状 |
|----------|------|----------|
| 社区规模、中心论文、邻接社区、子图截断 | 运行时由 `DemoCommunityGraph` 从 membership + kNN 图计算 | 已在 `lookup_community` / `community_graph_payload` 等返回的 JSON 中 |
| 论文级摘要、作者、年份、邻居 | `data_store` → `Paper` | `GET /api/papers/{pid}` |
| Topic 权重、top 词（「富」社区信息） | `out/topic_runs/.../communities_topic_weights.csv`（或 manifest `topic_communities_csv`） | 若 manifest 配了且文件存在，图与 paper payload 可带 `topic_info`；否则无 |

**未直接暴露给前端的**：原始 `communities_topic_weights.csv` 全表下载、按社区 id 批量导出。若产品要在右栏展示「该社区 top 词表」，可在现有 paper/community 载荷上扩展字段，或新增只读 `GET /api/communities/{cid}/topics`。

---

## 4. 六种检索：拆成「在线」与「离线」

离线评测里的「六路」是：**keyword**、**vector_nn**、**community_bundle** × 四种 `run_id`（命名如 `community_bundle:leiden_cpm`）。

| 模式 | 离线指标与图 | 在线交互（浏览器发请求） |
|------|----------------|---------------------------|
| keyword | `comparison_retrieval_*` 中聚合 + 条形图/折线图 | **有**：`/api/search/keyword` |
| vector_nn | 同上 | **无**：需把 `retrieval_benchmark` 中向量通道封装成 `GET`（参数：`pid` 或 `q`、top_k、run_id、resolution） |
| community_bundle（四算法） | 同上 | **无**：需封装「按 seed/community 规则」的 bundle 检索 API，或与离线一致仅展示指标 |

**结论**：六种检索的 **对比数字与图** 已可由 CSV/PNG 提供；**在页面上点论文做六种检索并看列表** 目前只实现了 keyword 一路。要「六种都可点」，需要后端新路由 + 前端新 Tab/模式。

---

## 5. 分辨率：可选网格

| 项目 | 说明 |
|------|------|
| **交互图 / 社区数** | 前端已有 resolution slider：`/api/v3/runs/{run_id}/resolutions` 提供有序列表，`session/switch` 切换后重载 graph。 |
| **检索评测分辨率** | `comparison_retrieval_resolution_long.csv` 按 `(run_id, resolution_effective)` 展开；`comparison_retrieval_best_resolution_long.csv` 给出各算法自选的 \(r*\) 与 `at_pick__*` 指标。 |
| **断点表** | `comparison_breakpoints.csv`：可与 UI 上「只显示断点 \(r\)」下拉联动，减少与全量 101 点混用时的认知负担。 |

前端若要「检索曲线随 slider 动」，需要：要么把 resolution 长表 **预聚合为 JSON** 由新 API 返回，要么构建时拷到 `web/static` 由前端 `fetch` 静态文件。

---

## 6. 时间窗（暂不做）

| 项目 | 说明 |
|------|------|
| **数据** | manifest 的 `time_window`；多窗可在 `out/experiments/.../y2010_2014__.../` 等目录各有一份 manifest。 |
| **前端** | 已有算法 / 时间窗下拉占位；若多 manifest 共享同一 `leiden_dir`，文档已说明「划分仍是全时段」——与产品文案一致即可。 |
| **建议** | 暂不在本清单强制验收；上线全算法 + 全分辨率后再接日历重扫结果。 |

---

## 7. 文件级核对（部署前自检）

在仓库根执行（路径按默认 `out/` 约定）：

1. **登记表**：`glob out/experiments/*/*/manifest.json` → 期望 ≥4 且 `run_id` 互不重复（或你定义的集合）。
2. **每个 manifest**：`leiden_dir/summary.npy` 存在；`default_resolution` 落在 `resolutions` 中或邻近可加载层。
3. **评测总览**：`out/experiment_eval/evaluation_overview.json` 与 `GET /api/v3/evaluations/overview` 行数一致（同一批 manifest）。
4. **六路检索表**（若已跑 eval）：`comparison_retrieval_sixway_long.csv` 每 tag 6 行；宽表、meta 同名存在。
5. **分辨率检索表**（若已跑新版聚合）：`comparison_retrieval_resolution_long.csv`、`comparison_retrieval_best_resolution_long.csv`、`comparison_retrieval_resolution_meta.json`；图目录 `retrieval_resolution_plots/`。
6. **对比跑批**：`out/comparison_runs/<tag>/metrics.jsonl` 非空，且与 `PC_EVAL_COMPARISON_RUN_TAGS`（若设置）一致。

---

## 8. 建议的下一版 API（不写实现，仅列接口契约）

便于前端一次对接、避免直接读仓库路径：

1. `GET /api/v3/evaluations/retrieval-sixway` → 返回已解析的 `comparison_retrieval_sixway_long.csv`（或等价 JSON 数组），可选 query：`comparison_run_tag`。
2. `GET /api/v3/evaluations/retrieval-by-resolution` → 返回 resolution 长表 + best 表摘要（或合并对象）。
3. `GET /api/v3/evaluations/breakpoints` → 返回 `comparison_breakpoints.csv` 解析结果（供分辨率下拉）。
4. `GET /api/v3/static-eval/{rel}` **或** 构建时 `cp out/experiment_eval/*.png web/static/eval/` → 仅暴露只读子路径。

静态方案（无后端改动）：在 CI/本地用脚本把 `out/experiment_eval/*.csv`、`*.png` 拷到 `web/static/eval/`，前端用相对路径 `fetch('/static/eval/comparison_retrieval_sixway_long.json')`（需先把 CSV 转成 JSON 或由构建脚本生成 `*.json`）。

---

## 9. 与现有文档的关系

- 交互与已实现路由：[`web-version-1.md`](../web_archived/web-version-1.md)。
- 离线路径与列语义：[`offline-outputs-catalog.md`](../../docs/offline-outputs-catalog.md)（§5、§8.2）。

---

**一句话**：四算法 + 分辨率 + 社区图/详情 **已经可以靠现有 v3 API 与图 API 接上**；六种检索里 **只有 keyword 能在线点**；六路与分辨率检索的 **表格与图已产在 `out/experiment_eval/`，但 Web 尚未挂载 URL**，需要静态资源或少量只读 API 即可完成「衔接」。
