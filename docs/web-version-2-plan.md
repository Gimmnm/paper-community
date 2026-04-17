# Web Version 2 计划（web-version-2）

本文件定义 web demo 的下一阶段（Version‑2）目标、数据契约、API/前端交互设计、里程碑与测试方法。

Version‑2 的核心增量：
- **分层结构展示**（多方案接入，但统一成同一套 web 数据契约与交互）
- **主题建模结果接入**（提升可解释性与导航能力）

本计划以你当前 `out/` 中**已经落盘的产物**为第一优先数据源，避免把重计算搬到在线 API。

---

## 1. 目标与总体交互

### 1.1 目标（你当前的主想法）

1) **KMeans Domain 作为顶层入口**
- 先在 UMAP 论文地图上展示 domain（kmeans labels）
- 用户选择 domain（0/1/2…）后，进入该 domain 的层级结构查看与钻取

2) **对每个“最初社区”（root resolution = 0.001）的分化追溯**
- 能从某个 root community（\(r=0.001\)）开始，沿着更高分辨率看到它如何分裂成多个子社区
- “追溯到最初社区 → 看分化”的视角要可交互（列表/树/断点）
- 你已经做过 1700+ 初始社区的离线分析：v2 应优先复用/接入这类预计算摘要（若现成文件缺失，则补一条可复现的生成管线）

3) **Details 仍以 paper 为核心**
- 图中 focus 用红色标识（v1 已实现）
- 分层节点/社区节点点击后，右侧 Details 展示：
  - 代表 paper（center paper）的最详细 paper detail（abstract + community summary + neighbors + neighbors in community）
  - + 分层节点的“分裂摘要/主题摘要”

### 1.2 v2 新增的三个视图（建议）

- **Domain Map（UMAP）**
  - 节点：论文（或按性能做采样/聚合）
  - 颜色：domain label
  - 交互：选中 domain → 切换到该 domain 的 Hierarchy Explorer

- **Hierarchy Explorer（层级结构视图）**
  - 数据集：global / local community / domain hierarchy（统一契约）
  - 展示：
    - Breakpoints（断点列表）
    - Root community 分裂树（从 \(r=0.001\) 开始的 lineage）
  - 交互：点击节点 → 右侧展示代表 paper + node 摘要；图上高亮对应集合

- **Topic Panel（主题解释/导航）**
  - 展示：top words、topic 权重、跨层稳定性（可选）
  - 交互：按 topic 高亮/过滤层级节点或社区

---

## 2. 现有可复用离线产物（已确认存在）

### 2.1 KMeans domain 产物

目录：`out/coarse_domains_kmeans_k3_seed42/`
- `meta.json`：k、seed、counts 等
- `labels.npy`：每篇论文的 domain label（需要后端 `np.load`）
- `domain_{i}_vertex_indices.npy`：domain 对应的全局 vertex 索引（0-based）
- 每个 domain 的层级目录：`domain_{i}_hierarchy_cpm/`
  - `hierarchy_nodes.csv`
  - `hierarchy_edges.csv`
  - `breakpoints.json`
  - `hierarchy_viz_overview.json`
  - `global_vertex_indices.npy`（用于 local→global 映射）

### 2.2 全局层级（CPM）

目录：`out/leiden_hierarchy_cpm/`
- `hierarchy_nodes.csv`（列：`resolution,community,size`）
- `hierarchy_edges.csv`（列：`r_parent,community_parent,...,child_share,jaccard`）
- `breakpoints.json`
- `hierarchy_viz_overview.json`

说明：该目录非常大（节点/边近百万级），v2 的在线接口必须支持**采样/过滤**或依赖预计算索引。

### 2.3 单社区局部层级（subgraph-hierarchy）

目录：`out/subgraph_hierarchy/k50_r0.0510_c*_cpm/`
- 同样包含 `hierarchy_nodes.csv / hierarchy_edges.csv / breakpoints.json / overview`
- `global_vertex_indices.npy`：local→global 映射

### 2.4 “root community 分裂树”脚手架

你已有脚本：`src/hierarchy_root_trees.py`
- 支持参数 `--root-resolution 0.001`、`--r-max ...`
- 目标是“每个 root community 画一棵树”
- v2 会把它升级为：**生成可被 web 直接读取的 JSON 摘要**（而不只是 png）

---

## 3. v2 数据契约（统一层级接口）

目标：不论层级来源是 global / domain / local community，前端都只依赖一套 JSON shape。

### 3.1 概念与标识

- **HierarchyDataset**
  - `hid`：层级数据集 id（稳定字符串）
  - `kind`：`global | domain | local_community | vertexset`
  - `hierarchy_dir`：对应 out 目录
  - `tags`：如 `{partition:"cpm", root_resolution:0.001, domain:0, parent_r:0.0510, community:0, ...}`

- **HierarchyNodeKey**
  - 用二元组标识：`(resolution, community_id)`
  - 传输时用稳定字符串：`node_id = "r=0.0010|c=37"`（与 `src/hierarchy_root_trees.py` 一致）

### 3.2 API（建议新增 /api/v2/*）

#### A) 数据集列表
`GET /api/v2/hierarchies`
- 返回所有可用层级数据集（扫描 `out/leiden_hierarchy_*`, `out/subgraph_hierarchy/*`, `out/coarse_domains_kmeans_*/*_hierarchy_*`）

#### B) 读取层级结构（支持过滤，避免爆内存）
`GET /api/v2/hierarchies/{hid}`
查询参数建议：
- `r_min, r_max`：分辨率区间
- `min_size`：过滤太小节点
- `top_per_layer`：每层最多取 N 个最大节点
- `max_edges`：最多返回多少条边
- `min_child_share` / `min_jaccard`：边过滤阈值

返回：
- `nodes`: [{node_id, r, c, size}]
- `edges`: [{source, target, child_share, jaccard, intersection, parent_share, child_share}]
- `breakpoints`: 从 `breakpoints.json` 读取
- `overview`: `hierarchy_viz_overview.json`

#### C) Root community 的 lineage（你要的“看分化”核心）
`GET /api/v2/hierarchies/{hid}/roots`
参数：
- `root_resolution`（默认 0.001）
- `top_k`（默认 50，按 size）
返回：root 列表（node_id + size）

`GET /api/v2/hierarchies/{hid}/lineage/{root_node_id}`
参数：
- `r_max`：追溯到的最高分辨率（例如 0.2）
- `min_child_share`：过滤弱分裂
- `max_nodes`：上限
返回：
- `subtree_nodes` / `subtree_edges`
- `splits_summary`：按层列出主要分裂（例如每个 r 下 children 的 size/child_share）

#### D) 层级节点 → 代表 paper → 复用 v1 的详情
`GET /api/v2/hierarchies/{hid}/node/{node_id}`
返回：
- `member_pids`（可分页：`offset/limit`）
- `center_pid`（一个，用于直接调用 v1 的 `/api/papers/{pid}`）
- `member_stats`（size、年份分布等可选）
- `topic_summary`（若 topic 数据存在）

实现说明（mapping）：
- global hierarchy：加载对应 `membership_r{r}.npy`，取 `membership==c`
- domain/local hierarchy：用 `global_vertex_indices.npy` 还原到全局 pid 空间

---

## 4. Topic 接入计划（信息增量）

### 4.1 v2 的最小 topic 目标
- community / hierarchy node 在 Details 中展示：
  - top topic / top words
  - topic weights（top-N）

### 4.2 数据源约定（优先离线落盘）
支持其中任一种存在即可启用：
- `out/topic_modeling/.../`（单分辨率）
- `out/topic_modeling_multi/K{K}/r{res}/...`（多分辨率）
- 可选：`aligned_*`（跨分辨率对齐）

### 4.3 API（建议）
`GET /api/v2/topics/status`
- 返回可用 topic 根目录与覆盖的 resolutions

`GET /api/v2/topics/community?resolution=...&community=...`
- 返回该 community 的 topic 摘要（top words、weights）

`GET /api/v2/topics/search?query=...`
- 按 top words/label 做简单检索（可选）

---

## 5. 前端改造范围（web v2）

### 5.1 UI 结构建议（在 v1 的基础上）
- 新增 Tab：`Hierarchy`
- 新增 Tab：`Domains`（或将 Domains 融入 Hierarchy 的 dataset 选择器）

### 5.2 Hierarchy Tab 的最小交互
- 选择数据集（global / domain / local）
- 选择 root（\(r=0.001\) 的 community）→ 展示 lineage 子树
- 点击 lineage 的某个节点：
  - 右侧展示 node 摘要 + 代表 paper（center_pid）的 v1 paper detail
  - 中间图可切换为：
    - lineage 子树图（层级节点图）
    - 或回投影到 UMAP（高亮 member_pids）

---

## 6. 里程碑（按最短路径交付）

### V2-A：只读层级数据集接入（不含 topic）
- 后端：实现 `/api/v2/hierarchies` + `/api/v2/hierarchies/{hid}`（支持过滤参数）
- 前端：新增 Hierarchy Tab，能展示 breakpoints 列表 + 简单层级图（采样）
- 测试：curl/浏览器验证能加载 global + domain + local 列表

### V2-B：root‑lineage 交互（你要的“0.001 起始分化追溯”）
- 后端：实现 `/roots` 与 `/lineage/{root_node_id}`（基于 hierarchy_edges.csv 的索引/过滤）
- 前端：root 列表（按 size 排序）+ 点击后加载 lineage 子树
- 测试：抽 5 个 root（大/中/小）确认节点数、分裂层次、边阈值效果

### V2-C：层级节点 → center_pid → 复用 v1 detail
- 后端：实现 `/node/{node_id}`（member_pids + center_pid）
- 前端：点击层级节点 → 右侧显示 v1 paper detail + node 摘要

### V2-D：Domains 首屏（KMeans domain）
- 后端：把 `labels.npy` / `domain_*_vertex_indices.npy` 暴露成可用 API（采样/聚合）
- 前端：Domain Map（UMAP）+ domain 选择器

### V2-E：Topic 最小接入
- 后端：topic status + community/node topic summary
- 前端：Details 增加 topic card；可选 topic 高亮/过滤

---

## 7. 性能与工程风险（必须提前处理）

### 7.1 hierarchy_edges.csv 过大
`out/leiden_hierarchy_cpm/hierarchy_edges.csv` 行数很大，不能每次请求全量扫描。

建议两条策略（二选一或组合）：
1) **预计算索引文件（推荐）**
   - 在 `out/leiden_hierarchy_cpm/` 旁生成一个轻量索引（例如 `.npz`/`.jsonl`/sqlite）：
     - root（r=0.001）→ 子树边列表（按 r_max/阈值可筛）
     - (r,c) → children 列表
2) **在线只做小窗口查询**
   - 强制要求 `r_min/r_max/top_per_layer`，并在后端做 streaming 读取 + 截断

### 7.2 lineage 的“阈值解释”
`min_child_share` 等阈值会显著改变树形结构，v2 UI 应展示当前阈值，并支持快速调整。

---

## 8. 测试方法（v2）

### 8.1 CLI / 脚本冒烟
- 针对某个 hierarchy_dir：
  - 能读取 `nodes/edges/breakpoints/overview`
  - 抽样 3 个 root 生成 lineage 并检查节点数不为 0

### 8.2 API 冒烟（curl）
- `/api/v2/hierarchies`
- `/api/v2/hierarchies/{hid}?r_min=0.001&r_max=0.02&top_per_layer=50`
- `/api/v2/hierarchies/{hid}/roots?top_k=20`
- `/api/v2/hierarchies/{hid}/lineage/r=0.0010|c=0?r_max=0.05&min_child_share=0.2`

### 8.3 浏览器验收
- domain → hierarchy → root lineage → 点击节点 → 右侧 paper detail 更新、图上红色 focus 可见

---

## 9. v2 准备工作（落盘索引）

### 9.1 为什么需要索引

全局层级（例如 `out/leiden_hierarchy_cpm/hierarchy_edges.csv`）规模很大，在线每次扫描 CSV 会非常慢。
因此 v2 计划在层级目录下落盘一个 SQLite 索引：
- `hierarchy_index.sqlite`
- 提供 `parent_id -> children` 的快速查询，支撑 root lineage 的 BFS。

### 9.2 如何生成索引（一次性）

对任意一个 hierarchy 目录执行：

```bash
python src/hierarchy_index.py --hierarchy-dir out/leiden_hierarchy_cpm
python src/hierarchy_index.py --hierarchy-dir out/subgraph_hierarchy/k50_r0.0510_c0_cpm
python src/hierarchy_index.py --hierarchy-dir out/coarse_domains_kmeans_k3_seed42/domain_0_hierarchy_cpm
```

生成后目录中会出现：`hierarchy_index.sqlite`。

### 9.3 如何验证索引可用

启动 API 后：
- `GET /api/v2/hierarchies` 里 `has_index=true`
- `GET /api/v2/hierarchies/{hid}/roots`
- `GET /api/v2/hierarchies/{hid}/lineage/{root_node_id}?r_max=...`

