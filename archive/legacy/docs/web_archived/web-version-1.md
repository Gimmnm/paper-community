# Web Version 1（web-version-1）

本文件是“当前网页 demo 展示部分”的**实现对照文档**：把已经实现的能力、数据依赖、启动方式、接口与前端交互、以及测试方法集中整理，作为后续迭代的稳定基线。

> 说明：这里的 “community subgraph” 是**同一层级**的“社区内部论文子图显示”，不是分层社区（hierarchical communities）的父子钻取。

---

## 1. 目标与范围

### 1.1 目标

- 在浏览器里提供一个可用的 demo：**查询 → 图展示 → 右侧详情**闭环。
- 图的位置分布以核心流程预计算的二维可视化坐标为基础（稳定、不抖动）。
- 详情面板统一展示最详细信息（paper/expand/由 community 触发的 center paper）。

### 1.2 范围（已实现）

- **FastAPI 服务**
  - 暴露一组 JSON API（keyword/paper/community/expand + graph payloads）
  - 同时挂载静态前端（`web/`）作为 `/` 与 `/static/*`
- **前端（静态，无需打包）**
  - 左侧：Keyword / Paper / Community / Eval
  - 中间：Cytoscape 图（community 全局图 / community 内部子图 / keyword paper map）
  - 右侧：Details 卡片化展示 + Raw JSON 开关

### 1.3 明确不做（web-version-1 不包含）

- 分层社区（父子社区）钻取与检索
- 复杂布局（力导稳定化、群集边 bundling 等）
- 登录/权限/多用户

> 注：重构阶段已把 hierarchy-first 的页面交互从主流程中移出，相关历史方案见 `archive/legacy/docs/hierarchy_notes.md`。

---

## 2. 代码位置（实现一览）

- **前端**
  - `web/index.html`
  - `web/style.css`
  - `web/app.js`
- **后端**
  - `src/demo_api_app.py`：FastAPI app + lifespan 加载资产 + endpoints + 挂载静态前端
  - `src/demo_search.py`：keyword/paper/community/expand 查询与 graph payload 构建
  - `src/demo_graph.py`：membership loader + DemoCommunityGraph 构建（中心/桥接/邻接社区等）
  - `src/core.py`：CLI 入口（`demo-api`, `demo-regress`, `demo-*`）

---

## 3. 数据依赖与默认路径

### 3.1 必需产物（默认）

- **CPM membership sweep**（社区标签）
  - 默认目录：`out/leiden_sweep_cpm/`
  - 文件：`membership_r*.npy` + `summary.npy`
- **mutual-kNN 图**
  - 默认：`out/mutual_knn_k50.npz`（可通过 `--graph-npz` 或 `PC_GRAPH_NPZ` 覆盖）
- **二维坐标（稳定布局基础）**
  - 默认：`out/umap2d.npy`（可通过 `PC_2D_PATH` 覆盖）
- **keyword TF‑IDF 索引（可选但推荐）**
  - 默认：`out/keyword_index/`
  - 若缺失或版本不兼容，keyword 会回退 substring（可用但慢）
- **实验登记表（网页里切换「第几种算法」）**
  - 默认：`out/experiments/<算法名>/all__<run_id>/manifest.json`（由 `bash scripts/daily_workflow.sh` 里的 `experiment-init-minimal` 写入）
  - 环境变量 `PC_ACTIVE_RUN_ID`：指定启动后默认选中的 `run_id`；未设置时在登记表中优先选 `leiden_cpm`
  - 若尚未生成任何 manifest，后端会临时使用仅含 CLI 默认路径的一条记录 `cli_default`（仍写 `legacy_default` 会被当作 `cli_default`）

### 3.2 坐标如何用于图

- **全局 community 图**：每个 community 的点位 = 该社区内部论文的 \(x,y\) 均值（从 `umap2d.npy` 聚合）
- **community subgraph（社区内部论文子图）**：每篇论文节点直接使用它自己的 \(x,y\)
- **keyword paper map**：把命中论文绘到同一张 UMAP 坐标系中

---

## 4. 启动与使用

### 4.1 安装依赖

```bash
pip install -r requirements-api.txt
```

### 4.2 启动（API + 前端）

```bash
python src/core.py demo-api \
  --resolution 0.2 \
  --leiden-dir out/leiden_sweep_cpm \
  --graph-npz out/mutual_knn_k50.npz \
  --keyword-index-dir out/keyword_index \
  --host 127.0.0.1 --port 8000
```

- 页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

### 4.3 前端交互要点（当前行为）

- **Keyword**
  - 请求：`GET /api/search/keyword?q=...&top_k=...&offset=...`
  - 显示：左侧 results + 右侧 details（keyword 卡片）+ 中间 paper map（调用 `/api/coords/papers`）
- **Paper**
  - 请求：`GET /api/papers/{pid}?k_neighbors=...`
  - 显示：右侧为最详细 paper detail（abstract + community summary + neighbors + neighbors in community）
  - 图：优先加载所属社区的 subgraph，并将当前 paper 标为 **红色 focus**
- **Community**
  - 请求：`GET /api/communities/{cid}`
  - 详情策略：把它当作“查询该社区的 center paper”
    - 先取 `center_papers[0].pid`
    - 再请求 `GET /api/papers/{center_pid}`
    - 右侧以 `type="expand"` 组合展示：paper detail + community card
  - 图：加载该 community 的 paper subgraph，并将 center paper 标为 **红色 focus**
- **红色 focus 节点规则**
  - 永远使用红色高亮（`role=focus`）
  - 图中不显示 focus 节点标题（Details 已包含信息）
  - 若 focus paper 不在 subgraph 的截断节点集合中，会作为 overlay node 被补进图中，确保“总能看到红点”

---

## 5. API 列表（web-version-1）

- `GET /api/health`
- `GET /api/search/keyword?q=...&top_k=...&offset=...`
- `GET /api/papers/{paper_id}?k_neighbors=...&k_neighbors_in_comm=...&k_neighbor_comms=...`
- `GET /api/communities/{community_id}?top_papers=...&top_neighbors=...`
- `GET /api/expand/paper/{paper_id}?k_papers=...&k_comms=...`
- `GET /api/coords/papers?ids=1,2,3&include_title=true`
- `GET /api/graph/communities?max_nodes=...&min_weight=...`
- `GET /api/graph/community/{community_id}?max_nodes=...&max_edges=...`
- `GET /`（页面）
- `GET /static/*`（静态资源）

---

## 6. 测试方法（建议按顺序）

### 6.1 CLI 最小回归（推荐）

```bash
python src/core.py demo-regress \
  --resolution 1.0 \
  --leiden-dir out/leiden_sweep_cpm \
  --graph-npz out/mutual_knn_k50.npz \
  --keyword-index-dir out/keyword_index
```

预期：输出 JSON，`keyword/paper/community/expand` 四个 check 都为 `ok: true`。

### 6.2 curl 冒烟（服务启动后）

```bash
curl -s "http://127.0.0.1:8000/api/health" | head
curl -s "http://127.0.0.1:8000/api/search/keyword?q=bayesian&top_k=3" | head
curl -s "http://127.0.0.1:8000/api/papers/83327?k_neighbors=5" | head
curl -s "http://127.0.0.1:8000/api/communities/10?top_papers=5" | head
curl -s "http://127.0.0.1:8000/api/graph/communities?max_nodes=50" | head
curl -s "http://127.0.0.1:8000/api/graph/community/10?max_nodes=30&max_edges=120" | head
```

### 6.3 手动验收（浏览器）

- 页面加载
  - 顶部出现 `papers=... communities=... r=...`
  - 默认加载 community 全局图
- Keyword
  - 输入 `bayesian`，结果列表出现
  - 翻页 Prev/Next 正常
- Paper
  - 输入一个 pid（如 `83327`）
  - 右侧显示：abstract + community summary + neighbors + neighbors in community
  - 图中能看到红色 focus 节点
- Community
  - 输入一个 cid
  - 右侧显示：center paper 的完整 detail + community 卡片
  - 图中红色 focus 节点可见（即使不在截断子图节点中，也会被补进）

---

## 7. 常见问题

### 7.1 keyword 回退 substring

当 `out/keyword_index` 缺失或 TF‑IDF 资产反序列化失败，会回退到 substring。此时仍可用，但更慢，且排序更粗糙。

### 7.2 community subgraph 的两个参数

- `max_nodes`：子图最多显示多少篇论文节点
- `max_edges`：子图最多保留多少条社区内部边（按权重截断）

### 7.3 Web 顶部显示的 `r=...` 到底是什么？（RB vs CPM）

页面顶栏的 `r=...` 来自后端 `/api/health`，它打印的是启动 `demo-api` 时的 `--resolution` 参数（环境变量 `PC_RESOLUTION` 同名）。

- **CPM（`CPMVertexPartition`）**：这个参数对应 CPM 的 \(\gamma\)（有时也直接叫 resolution）。
- **RB（`RBConfigurationVertexPartition`）**：这个参数对应 RB 的 resolution。

两者都是“Leiden 分区目标函数的参数”，但**目标函数不同**，因此：

- **没有一个全局固定的 RB↔CPM 换算公式**（不只取决于参数，还取决于图的边权尺度/权重归一化/网络密度）。
- 如果你看到“网页分辨率像是 0.1（RB 的常见量级）”，但你实际加载的是 `out/leiden_sweep_cpm/`，那通常表示你把 `--resolution` 设成了 0.1 去跑 CPM 的 sweep/加载 CPM membership —— 这在数学上是允许的，但量级是否合适取决于你的图权重尺度。

**建议做法（避免混乱）**：

- **明确目录即 partition**：
  - CPM sweep：`out/leiden_sweep_cpm/`（Web 默认）
  - RB sweep：`out/leiden_sweep_rb/`
- **用“产物的目录名 + hierarchy 数据集标签”作为真相来源**，不要仅凭 `r=...` 的数值量级去猜 partition。
- 如果想在 RB 与 CPM 间找“差不多粗细”的层，可用 sweep 的 `summary.npy` 以 **社区数/模块度曲线** 对齐，而不是靠硬换算。

### 7.4 分层可视化说明（已移出主流程）

分层（lineage）相关交互在当前重构主线中不再作为默认功能；若需查看历史思路，请参考归档文档：

- `archive/legacy/docs/hierarchy_notes.md`

### 7.5 与离线算法产物的衔接（数据就绪）

四算法 manifest、分辨率、社区图/详情与 `evaluation_overview` 等与前端消费的对应关系、缺口（例如六路检索 CSV/PNG 尚未挂 URL、在线仅 keyword 等）见专门清单：

- 历史「数据就绪矩阵」快照（可能过时）：[`../refactor_archived/web-frontend-data-readiness.md`](../refactor_archived/web-frontend-data-readiness.md)

