# 数据流与核心逻辑：单一来源说明

本文档回答三件事：**数据从哪来**、**核心算法在哪一层实现**、**是否存在重复实现及为何**。

---

## 1. 原始数据与唯一入口

| 环节 | 唯一实现位置 | 产物 / 说明 |
|------|----------------|-------------|
| RData / txt → Python 字典 | [`src/getdata.py`](../../src/getdata.py) `ingest()` | `data/data_store.pkl` |
| pickle → `Author` / `Paper` | [`src/model.py`](../../src/model.py) `build_models()` | 内存对象数组（1-based 下标） |
| 论文列表加载（CLI / API 共用路径） | [`src/project_paths.py`](../../src/project_paths.py) `data_source_paths()` + [`src/core.py`](../../src/core.py) `build_or_load()` 或 [`src/demo_search.py`](../../src/demo_search.py) `load_papers_from_project()` | 二者调用同一套 `ingest` 参数，路径由 **`project_paths` 唯一维护** |

**结论**：原始文献与关系数据的**解析与缓存**只在 `getdata.ingest`；对象建模只在 `model.build_models`。文件名与相对目录约定集中在 **`src/project_paths.py`**，避免在 `core` 与 `demo_search` 各写一份路径。

---

## 2. 下游产物（out/）与依赖关系

以下为常见流水线（与 [`README.md`](../../README.md) 一致），每一列对应**主要实现文件**：

| 步骤 | 输入 | 输出 | 实现 |
|------|------|------|------|
| Embedding | `Paper` + 摘要 | `data/paper_embeddings_specter2.npy` | [`src/embedding.py`](../../src/embedding.py) |
| 2D 可视化 | embedding | `out/umap2d.npy` 等 | [`src/diagram2d.py`](../../src/diagram2d.py) |
| mutual-kNN 图 | embedding | `out/mutual_knn_k*.npz` | [`src/network.py`](../../src/network.py) |
| Leiden / sweep | igraph 图 | `out/leiden_*/membership_r*.npy`, `summary.npy` | [`src/community.py`](../../src/community.py) |
| 关键词索引 | 论文文本 | `out/keyword_index/*` | [`src/retrieval.py`](../../src/retrieval.py) |
| Topic-SCORE（单分辨率） | `data_store.pkl` + membership | `out/topic_modeling/...` | [`src/topic_modeling.py`](../../src/topic_modeling.py) |
| Topic-SCORE（多分辨率批处理） | 多个 `membership_r*.npy` | `out/topic_modeling_multi/K*/r*/...` | [`src/topic_modeling_multi.py`](../../src/topic_modeling_multi.py)（内部调用 `topic_modeling` 管线） |
| Topic 对齐 | 多分辨率 topic 输出 | `out/.../aligned*` | [`src/align_topics_multires.py`](../../src/align_topics_multires.py) 等 |

**结论**：**主题建模核心算法只在一处**——`topic_modeling.py`（Topic-SCORE / 矩阵与 SVD 等）。`topic_modeling_multi.py` 是批处理与目录编排，不复制数学核心。

---

## 3. 社区标签（membership）读取：两处实现的原因

| 函数 | 文件 | 依赖 | 用途 |
|------|------|------|------|
| `load_membership_for_resolution` | [`src/community.py`](../../src/community.py) | `python-igraph`（模块顶层 import） | 全量流水线、`retrieval` 附社区标签 |
| `load_membership_for_resolution_light` | [`src/demo_graph.py`](../../src/demo_graph.py) | 仅 `numpy` | Demo / API：无 igraph 环境可读 sweep 结果 |
| `load_membership(path, n_papers_expected)` | [`src/topic_modeling.py`](../../src/topic_modeling.py) | 无 igraph | **显式路径**的 `.npy`，并校验长度与 `n_papers` |
| `load_membership_safe` / `load_membership_local` | [`src/topic_visualization_multires.py`](../../src/topic_visualization_multires.py) | 可选回退 | 可视化脚本在重型依赖失败时的容错 |

**结论**：**按分辨率从目录里“找文件 + 最近邻 r”** 的逻辑在 `community` 与 `demo_graph` 中语义重复，是**刻意**的：`demo_*` 路径避免强依赖 `igraph`。若未来希望严格单一实现，可将 `community.load_membership_for_resolution` 改为延迟 import `igraph`，或把“仅 numpy 的解析”抽到 `membership_io.py` 供两处调用。

---

## 4. Demo 层（网页 API / CLI）

| 模块 | 作用 | 与主线的关系 |
|------|------|----------------|
| [`src/demo_graph.py`](../../src/demo_graph.py) | `DemoCommunityGraph`、中心/桥接/社区边聚合 | 读 **同一批** `membership` + `mutual_knn_k*.npz`，不替代 `community.py` 的 Leiden |
| [`src/demo_search.py`](../../src/demo_search.py) | 关键词 / paper / community / expand 查询与 graph payload 构建 | 复用 `retrieval` 的 TF-IDF 文件格式；查询逻辑在 demo 内，**不**复制 Topic-SCORE |
| [`src/demo_api_app.py`](../../src/demo_api_app.py) | FastAPI 暴露 JSON | 启动时 `load_papers_from_project` + `build_demo_assets_and_graph` |

**结论**：Demo **不实现第二套主题建模**；若要在 UI 展示 topic 词，应读 `out/topic_modeling(_multi)/...` 的导出文件，或在后续里程碑把路径接入 `DemoCommunity.topic_info`。

---

## 5. 可选依赖导致的“降级路径”

- **`core.py`**：对 `igraph`、`torch`、`embedding`、`time_window` 等采用 try/except，缺失时部分子命令不可用，但 **`demo-graph` / `demo-api`** 等仍可运行。
- **`retrieval.py`**：对 `community.load_membership_for_resolution` 可选；无 igraph 时关键词检索仍可工作，只是结果里可能无社区列。

---

## 6. 维护建议

1. **新增任何读取 `data/*.RData` 或 `data_store.pkl` 的代码**：请先扩展或复用 [`src/project_paths.py`](../../src/project_paths.py) 中的 `DataSourcePaths`。
2. **新增主题相关功能**：优先在 `topic_modeling.py` 扩展；批处理/扫描仍放 `topic_modeling_multi.py`。
3. **合并 membership 读取**（可选重构）：抽 `src/membership_io.py`，`community` 与 `demo_graph` 共用纯 numpy 部分。

