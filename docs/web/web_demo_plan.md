# 网页 Demo 计划：查询 + 社区画像 + 基础可视化（不含分层钻取）

本文档用于把当前阶段的网页 demo 目标、范围、里程碑、数据契约与风险排查明确下来，作为后续逐步实现的工作底稿。

> 本阶段暂不实现「分层社区 → 子社区 → 子子社区」的交互钻取与检索，但会在数据结构与 API 上预留扩展点。

---

## 0. 快速入口（按“现在代码行为”对齐）

### 0.1 一键启动（API + 前端）

```bash
pip install -r requirements-api.txt
python src/core.py demo-api \
  --resolution 1.0 \
  --leiden-dir out/leiden_sweep_cpm \
  --graph-npz out/mutual_knn_k50.npz \
  --keyword-index-dir out/keyword_index \
  --host 127.0.0.1 --port 8000
```

- 页面入口：`http://127.0.0.1:8000/`
- 交互式 API 文档：`http://127.0.0.1:8000/docs`

### 0.2 前端 demo（Milestone D/E）当前约定

- **三种查询入口**：Keyword / Paper / Community
- **详情面板（Details）统一为最详细版**
  - 论文摘要（abstract）
  - community summary
  - Neighbors（mutual-kNN）
  - Neighbors in community
- **Community 查询的详情展示策略**
  - 把它当作“查询该社区的 center paper”，详情面板展示 center paper 的完整 paper detail
  - 同时附带 community 卡片（center/bridge/example/neighbor communities 列表都可点击跳转）

### 0.3 产物路径（Demo 默认）

- **membership（社区标签）**：默认使用 `out/leiden_sweep_cpm/membership_r*.npy`（可通过 `--leiden-dir` 或环境变量覆盖）
- **mutual-kNN 图**：`out/mutual_knn_k50.npz`（或改 `k`）
- **二维坐标**：`out/umap2d.npy`（用于稳定的 preset 布局）
- **keyword TF-IDF 索引**：`out/keyword_index/`

---

## 1. 目标与范围（MVP）

### 1.1 本阶段要交付的能力

- **关键词匹配检索**：按标题/摘要/关键词/作者等字段做字符串匹配（先简单可用，后续可替换为 BM25/倒排索引）。
- **论文 → 社区 → 社区画像**：输入一篇论文（`paper_id` 或精确标题），返回：
  - 论文基本信息（标题、年份、作者、venue…）
  - 所在社区（`community_id`、规模、概述）
  - 社区主题信息（topic top words / label / representative text）
  - 社区内关键论文（中心/代表论文）
  - 桥接论文（跨社区连接强/介数高的候选）
  - 相邻论文（同社区内相似/邻接 topK）
  - 相邻社区（社区间边权 topK）
- **基础图形可视化（固定分辨率）**
  - 展示「社区级别」图：节点=社区、边=社区间关系（跨社区引用/相似）
  - 或展示「社区内部」子图：节点=论文、边=引用/相似（可选，先做一层即可）
  - 交互：点击节点在侧边栏显示详情（论文/社区画像）
- **单点发散（从论文出发）**
  - 输入论文 → 返回邻近论文、所在社区画像、相邻社区概要等（不涉及层级子社区）。

### 1.2 明确不做/先不做

- **分层社区结构检索与交互**：社区区域点击放大进入子社区并继续钻取，这一套先不实现。
  - 但会预留：`Community`/`CommunityGraph` 可附加 `parent/children` 字段；API 预留 `resolution`/`level` 参数。

---

## 2. 总体架构（建议三层拆分）

### 2.1 核心数据建模层（Domain / Model）

目标：把“社区”作为核心对象，把离线产物（社区发现、主题建模、embedding、图边等）整合成统一内存结构，并提供只读访问能力。

建议核心类（命名可按仓库风格微调）：

- `Paper`
  - `id, title, abstract, year, authors, venue`
  - 可选：`embedding`、`neighbors`（缓存 topK）、`community_id`
- `Community`
  - `id`
  - `paper_ids`
  - `topic_info`（top words/label/vector 等）
  - `center_paper_ids`（中心/代表）
  - `bridge_paper_ids`（桥接）
  - `neighbor_community_ids`（相邻社区 + 权重）
  - `stats`（size、year histogram 等）
- `CommunityGraph`（全局容器）
  - `papers: Dict[str, Paper]`
  - `communities: Dict[int, Community]`
  - 社区间边：`community_edges`（可稀疏存储）
  - （可选）论文间边：`paper_edges`

核心原则：
- 核心层只保证 **数据一致性、可索引、可聚合**；不关心 UI/路由/展示细节。

### 2.2 查询逻辑层（Search / Service）

目标：把检索需求落成可复用函数，并返回统一 JSON 结构，便于前端渲染。

建议的查询入口（MVP）：

- `search_keyword(q, filters=None, topk=50)`
- `lookup_paper(paper_id | title_exact)`
- `lookup_community(community_id)`
- `expand_from_paper(paper_id, k_papers=20, k_comms=10)`

统一返回结构建议：

```json
{
  "type": "keyword|paper|community|expand",
  "query": { "...": "..." },
  "hits": [ { "kind": "paper|community", "id": "...", "score": 0.0, "payload": { } } ],
  "graph_snippet": { "nodes": [], "edges": [] },
  "debug": { }
}
```

性能策略（demo 友好）：
- 小数据：启动时一次性加载进内存；关键词检索先 substring/简单 TF-IDF（仓库现有 `retrieval.py` 可复用/替换）。
- embedding 相似：若已有向量，可先用 `numpy` cosine + topK；量大再接 ANN（FAISS/HNSW）。

### 2.3 Web API + 可视化层（Demo App）

目标：提供 HTTP API 并做基础图展示。

后端建议：`FastAPI`（启动快、调试方便）。

建议 API（最少够用）：
- `GET /api/health`
- `GET /api/search/keyword?q=...&topk=...`
- `GET /api/papers/{paper_id}`
- `GET /api/communities/{community_id}`
- `GET /api/expand/paper/{paper_id}?k_papers=...&k_comms=...`
- `GET /api/graph/communities`：社区图（nodes/edges + 布局坐标）
- `GET /api/graph/community/{community_id}`：（可选）社区内部论文子图

前端路线（二选一）：
- **快做 demo**：静态 HTML + JS + Cytoscape.js/D3.js（最少工程成本）。
- **更工程化**：React + Vite（UI 更舒服，但搭建与打包成本更高）。

可视化交互（MVP）：
- 左侧：搜索框 + 模式切换（关键词 / 论文发散 / 社区查询）
- 中间：图（社区图为主，必要时显示社区内部子图）
- 右侧：详情面板（点击节点更新）

---

## 3. 数据契约（离线产物 → Demo 读取）

为了让 demo 稳定，建议把离线产物“收敛”为一套可读的规范文件（哪怕内部生成过程分散，最后由导出脚本统一格式）。

建议最小数据集合：

- `papers.jsonl`
  - 每行一篇论文：`id,title,abstract,year,authors,venue,(optional)keywords`
- `paper_to_community.json`
  - `paper_id -> community_id`
- `communities.json`
  - `community_id -> { paper_ids: [...], topic_info: {...}, (optional)center_paper_ids, bridge_paper_ids }`
- `community_edges.csv` 或 `community_edges.jsonl`
  - `community_a, community_b, weight, type`

可选增强：
- `paper_edges.csv/jsonl`：论文边（引用/相似）
- `paper_embeddings.npy` + `paper_index.json`：向量检索
- `community_layout.json`：社区图节点坐标（固定分辨率布局）

说明：
- 固定分辨率需求下，建议社区图布局 **离线一次算好**，写入 `community_layout.json`，前端直接渲染，避免每次力导布局带来的抖动与不可复现。

---

## 4. 里程碑（按最短路径实现）

### Milestone A：数据结构跑通（无网页）
（以下内容与原 `docs/web_demo_plan.md` 一致，已整体迁移到本文件。）

