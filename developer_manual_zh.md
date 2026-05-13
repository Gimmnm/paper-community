# 论文社区项目 — 开发者技术手册

仓库 **三本核心说明** 之一（面向开发与运维）：**总流程**、**模块与代码路径**、**数据流与实现要点**、**命令 / 脚本 / API / 前端**、缓存与典型字段。  
另两本：**[`user_manual_zh.md`](user_manual_zh.md)**（网站怎么用）、**[`offline-outputs-catalog.md`](offline-outputs-catalog.md)**（`out/` / `data/` 每个目录/文件是什么、怎么生成——**路径级单一事实来源**）。  

本文与目录说明的分工：**流程与「为什么」以本文为主**；**「哪个文件在哪、列名与打开方式」以目录说明为准**，避免双份维护。细节命令以 `src/core.py` 与各 task 的 `--help` 为准。**原根目录 `README.md` 中的环境、脚本流水线、子命令索引与缓存说明已并入本文 §7、§11–§19。** 同目录下的 **`requirements-breakdown.md`** 等补充规格见 **`README.md`** 索引。

---

## 1. 仓库与模块边界

| 目录 | 职责 |
|------|------|
| `src/foundation_layer/` | 数据读取、`Paper`/`Author`、embedding、mutual‑kNN、2D 投影 |
| `src/algorithm_layer/` | Leiden 扫描、时间窗、粗聚类+域内社区等 |
| `src/analysis_layer/` | 评测汇总、检索离线对比、topic、层级索引、作图 |
| `src/data_layer/` | 实验 manifest、断点表、topic 路径解析等 |
| `src/app_layer/` | `demo_search` / `demo_graph` / `demo_api_app` / `demo_retrieval_live` |
| `web/` | 静态前端（`index.html`、`app.js`、`style.css`） |
| `out/` | 默认产物根（图、索引、sweep、评测、topic_runs 等） |

架构与目标陈述（按需）：[`architecture-boundaries.md`](architecture-boundaries.md)、[`requirements-breakdown.md`](requirements-breakdown.md)。

---

## 2. 数据与对象模型

### 2.1 原始数据 → `data_store.pkl`

- **入口**：`foundation_layer/getdata.py`（由 `core.py` 任务封装）。  
- **输出**：`data/data_store.pkl`（作者、论文字段、摘要等）。

### 2.2 `Paper` / `Author`

- **入口**：`foundation_layer/model.py`。  
- **约定**：论文 **pid 从 1 起**；列表索引 0 常为占位，与 embedding 行对齐时注意「`n` vs `n+1`」分支（`demo_retrieval_live`、`retrieval_benchmark` 等处有显式处理）。

---

## 3. 文本向量与 2D 坐标

### 3.1 SPECTER2 嵌入

- **实现**：`foundation_layer/embedding.py`。  
- **默认产物**：`data/paper_embeddings_specter2.npy`（或项目约定的 `embedding_path_specter2`，见 `foundation_layer/project_paths.py`）。  
- **demo-api**：启动时在 `lifespan` 中尝试加载 L2 归一化矩阵到 `app.state.emb_norm`（失败则向量检索与部分后台指标不可用）。

### 3.2 全局 2D（UMAP 等）

- **实现**：`foundation_layer/diagram2d.py` 等。  
- **典型产物**：`out/umap2d.npy`，供社区节点质心、论文散点、Web 子图坐标共用。

---

## 4. 图构建

### 4.1 mutual‑kNN

- **实现**：`foundation_layer/network.py`（`load_edges_npz` 等）。  
- **产物**：`out/mutual_knn_k{K}.npz`（边 `u,v,w`、`n_nodes`、`k` 等元数据）。  
- **igraph**：`algorithm_layer` / `app_layer` 中通过三元组构建 `igraph.Graph`。

### 4.2 关键词索引（TF‑IDF）

- **产物目录**：`out/keyword_index/`（`vectorizer.pkl`、`tfidf_docs.npz`）。  
- **Web**：`DemoAssets.load_keyword_index()`；缺失时关键词检索可能降级或报错（以路由实现为准）。

---

## 5. 社区发现与多分辨率扫描

### 5.1 Leiden / Louvain / CPM

- **核心**：`algorithm_layer/community.py`（`leiden_partition`、`leiden_sweep`、`rebuild_sweep_summary_from_membership_dir` 等）。  
- **分区类型**：`RBConfigurationVertexPartition`（RB）、`CPMVertexPartition`（CPM）。  
- **产物目录**（示例）：`out/leiden_sweep_cpm/`、`out/leiden_sweep_rb/`、`out/leiden_sweep_louvain/`。  
- **文件**：`membership_r{r:.4f}.npy`、`summary.npy`、`summary.csv`；可选 `eval/`（由 `experiment-eval-bundle` 生成）。

### 5.2 summary 与磁盘 membership 不一致时

- **`available_resolutions_for_manifest`**（`data_layer/experiment_registry.py`）：优先枚举 **`membership_r*.npy`**，再回退 `summary.npy`。  
- **`rebuild_sweep_summary_from_membership_dir`**：仅根据磁盘 membership 重建 `summary.npy`（不重算分区）。

### 5.3 粗聚类 + 域内 CPM

- **入口**：`core.py` 中 `experiment-coarse-kmeans-sweep` 等任务。  
- **产物**：如 `out/coarse_kmeans_then_cpm_k3_seed42/`，合并域 membership；`infer_breakpoint_run_id_from_leiden_dir` 等用于目录名与 `run_id` 映射。

---

## 6. 实验登记与评测

### 6.1 Manifest

- **路径**：`out/experiments/<algorithm>/<time_window>__<run_id>/manifest.json`。  
- **结构**：`data_layer/experiment_contracts.py` → `ExperimentRunManifest`（`leiden_dir`、`graph_npz`、`keyword_index_dir`、`coords_2d_path`、`topic_communities_csv`、`tags` 等）。  
- **生成**：`core.py experiment-init-minimal`（常用）或手写。

### 6.2 评测 bundle 与断点

- **`experiment-eval-bundle`**：`analysis_layer/eval_bundle.py`，写各 sweep 下 `eval/`。  
- **`experiment-comparison-breakpoints`**：读各 manifest 的 `summary.npy`，写 `out/experiment_eval/comparison_breakpoints.csv`。  
- **`experiment-eval`**：`evaluation_metrics.py` + `retrieval_sixway_aggregate.py`，写 `evaluation_overview.*`、六路表与图等。  
- **流水线串联**：`scripts/daily_workflow.sh`。

### 6.3 Topic 建模产物

- **典型布局**：`out/topic_runs/<sweep_folder>/K10/r0.xxxx/communities_topic_weights.csv`。  
- **路径解析**：`data_layer/breakpoint_schedule.py` 中 `topic_run_sweep_folder_for_run_id`、`default_topic_k_dir_for_run`、`resolve_topic_communities_csv`（按 r 最近目录匹配）。  
- **Web 挂载**：`app_layer/demo_search.resolve_topic_weights_csv_for_web` 在 manifest / `PC_TOPIC_COMMUNITIES_CSV` / 自动 `topic_runs` 间解析；`build_demo_assets_and_graph` 调用 `enrich_demo_graph_topic_info`（`demo_graph.py`）。

---

## 7. Web 后端（demo-api）

### 7.1 启动与配置

- **入口**：`core.py demo-api` → `uvicorn` 加载 `app_layer.demo_api_app:app`。  
- **环境变量**（常见）：`PC_BASE_DIR`、`PC_LEIDEN_DIR`、`PC_GRAPH_NPZ`、`PC_KEYWORD_INDEX_DIR`、`PC_RESOLUTION`、`PC_TOPIC_COMMUNITIES_CSV`、`PC_ACTIVE_RUN_ID` 等（见 `demo_api_app._settings` / `core.task_demo_api`）。

**推荐本地启动前**（默认路径示例；缺一项时按报错与 `/api/health` 补全）：

- `out/leiden_sweep_cpm/`（或你在 manifest 里登记的任意 sweep：`membership_r*.npy`、`summary.npy`）
- `out/mutual_knn_k50.npz`
- `out/keyword_index/`（可选；缺失时关键词检索可能降级，见路由实现）

```bash
PYTHONPATH=src python src/core.py demo-api \
  --resolution 0.2 \
  --leiden-dir out/leiden_sweep_cpm \
  --graph-npz out/mutual_knn_k50.npz \
  --keyword-index-dir out/keyword_index \
  --host 127.0.0.1 --port 8000
```

- 页面：`http://127.0.0.1:8000/`；OpenAPI：`http://127.0.0.1:8000/docs`  
- **关闭**：前台 `Ctrl+C`；端口占用（macOS 示例）：`lsof -nP -iTCP:8000 -sTCP:LISTEN` → `kill -TERM <PID>`  

若已用 `out/experiments/**/manifest.json` 登记多 run，通常可仅用较短参数启动（见 `python src/core.py demo-api --help`）；用户向说明见 `docs/user_manual_zh.md` §6。

### 7.2 生命周期与运行时缓存

- **`lifespan`**：加载论文、构建 `run_catalog`、预建默认 `active_run_id` 的 `DemoAssets` + `DemoCommunityGraph`，放入 `app.state.runtime_cache`。  
- **`_runtime(request, run_id, resolution)`**：按 `(run_id, resolution)` 缓存；未命中则 `_build_runtime` 再跑 `build_demo_assets_and_graph`（含 topic CSV 解析）。

### 7.3 主要 HTTP 路由（节选）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 论文数、社区数、当前 r、主题表解析路径、`topic_communities_csv_loaded` 等 |
| GET | `/api/v3/catalog` | 实验目录、runs、算法列表 |
| GET | `/api/v3/session` / `POST /api/v3/session/switch` | 当前 / 切换 active run 与 resolution |
| GET | `/api/v3/runs/{run_id}/resolutions` | 分辨率列表 |
| GET | `/api/search/keyword` | TF‑IDF 检索 |
| GET | `/api/search/vector_nn` | 嵌入近邻（需 `emb_norm`） |
| GET | `/api/v3/search/community-bundle` | 社区 bundle（显式 `partition_run_id`） |
| GET | `/api/graph/communities` | 全局社区图 JSON |
| GET | `/api/graph/community/{id}` | 社区子图 |
| GET | `/api/v3/retrieval/live` | 后台单种子 live + six-way 对照 |
| GET | `/api/v3/evaluations/retrieval-sixway` | 离线 six-way 行（供后台表） |
| GET | `/static/*` | 挂载 `web/` |

### 7.4 图与检索核心逻辑

- **`build_demo_assets_and_graph`**（`demo_search.py`）：membership + kNN → `DemoCommunityGraph`；可选 topic CSV enrich。  
- **`lookup_paper` / `lookup_community`**：组装右侧详情 payload；**`structure_influence_index`** 与 **`impact_factor`** 字段当前同源（`compute_structure_influence_index`）。  
- **`search_community_bundle`**（`demo_retrieval_live.py`）：与 live 评测共用构建路径；topic CSV 通过 `resolve_topic_weights_csv_for_web` 对齐 Web。

---

## 8. Web 前端

- **入口**：`web/index.html` 引用 `app.js`、`style.css`；静态资源版本 query 用于缓存破坏。  
- **API 前缀**：`app.js` 中 `apiUrl` 将非 `/api` 路径自动加上 `/api`。  
- **模式**：`appMode` `user` | `admin`；`switchMode` 切换 `#secUser` / `#secAdmin` 与中间 `#centerUser` / `#centerAdmin`。  
- **全局图**：`loadCommunityGraph` → `/api/graph/communities`，`max_nodes` 来自可编辑输入 + `localStorage`。  
- **子图**：`loadCommunitySubgraph`；`SUBGRAPH_MAX_NODES` / `SUBGRAPH_MAX_EDGES` 常量控制规模；`SPREAD` 控制坐标缩放。  
- **详情渲染**：`renderDetailsPanel` → `renderPaperCard` / `renderCommunityCardShort` / `renderAdminLive` 等。

---

## 9. 与旧文档的关系

| 文档 | 用途 |
|------|------|
| [`user_manual_zh.md`](user_manual_zh.md) | **三本核心**之一：网站用户说明 |
| [`offline-outputs-catalog.md`](offline-outputs-catalog.md) | **三本核心**之一：`out/` / `data/` 路径与文件级说明 |
| [`README.md`](README.md) | 文档索引：三本核心 + 同目录补充规格 |
| [`../archive/legacy/docs/web_archived/web-version-1.md`](../archive/legacy/docs/web_archived/web-version-1.md) | 早期 Web/API 基线；若与本文冲突，**以代码与本文为准** |
| [`experiment-comparison-pipeline.md`](experiment-comparison-pipeline.md) | 多算法离线对比与 manifest 契约（按需） |
| [`../README.md`](../README.md)（根） | 仓库一页式入口（链到 `docs/README.md`） |

---

## 10. 扩展与排错

- **422 / `max_nodes` 上限**：以 `demo_api_app.py` 中 `Query(..., le=...)` 为准；修改后需重启 uvicorn。  
- **主题不显示**：检查 `resolve_topic_weights_csv_for_web` 是否解析到文件；`/api/health` 悬停或返回体中的路径；社区 id 是否与 CSV 一致。  
- **高 r 全局图异常**：CPM 高 γ 下单点社区极多，总图仅采样前 **N** 个大社区 + 边过滤，属预期现象；见用户手册说明。

---

## 11. 环境与依赖安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-api.txt   # demo-api / FastAPI
```

与上述等价的一组核心包（无 lockfile 时便于排查）：`numpy scipy pandas matplotlib scikit-learn pyreadr python-igraph leidenalg pillow imageio imageio-ffmpeg`。

**可选**（按任务开启）：`umap-learn pacmap hnswlib faiss-cpu transformers adapters torch`。

- **SPECTER2**（`embed`）：`transformers`、`adapters`、`torch`  
- **Leiden**：`python-igraph`、`leidenalg`  
- **KNN 构图**：`faiss` / `sklearn` / `hnswlib`（`build-graph --knn-backend`）  
- **时间窗视频**：优先系统 `ffmpeg`；缺失时可回退 GIF（`imageio-ffmpeg`）

---

## 12. 端到端数据流（总览）

与根目录历史说明一致，总控入口为 **`PYTHONPATH=src python src/core.py <task> [args...]`**（`src/core.py` 为唯一任务编排 CLI，分层实现见 `src/README.md`）。

1. 自 `RData/`、`author_name.txt` 等读取原始数据 → `data/data_store.pkl`  
2. 构建 `Author` / `Paper`  
3. 论文向量（默认 SPECTER2）→ `data/paper_embeddings_specter2.npy`  
4. 全局 2D 嵌入（UMAP / PCA / PaCMAP）→ `out/umap2d.npy` 等  
5. mutual‑kNN 图 → `out/mutual_knn_k*.npz`  
6. 图上 Leiden（及 Louvain、粗分域内 CPM 等）多分辨率扫描 → 各 `out/leiden_sweep_*` 等  
7. 层级诊断、时间窗、关键词索引、主题建模、对齐与可视化、实验登记与评测等 → 见下文子命令与 `offline-outputs-catalog.md`

---

## 13. 核心源码文件速览（`src/`）

| 模块 | 职责摘要 |
|------|----------|
| `core.py` | 唯一 CLI 总入口；各 task 委托到各 layer（勿删作「减负」——只会打散入口） |
| `foundation_layer/getdata.py` | 读 `.RData`/txt → `data_store.pkl` |
| `foundation_layer/model.py` | `data_store.pkl` → `Author`/`Paper` |
| `foundation_layer/embedding.py` | SPECTER2 等 → `paper_embeddings_specter2.npy` |
| `foundation_layer/network.py` | embedding 上 KNN → mutual‑kNN `npz` |
| `foundation_layer/diagram2d.py` | 2D 投影与散点/布局相关作图 |
| `algorithm_layer/community.py` | Leiden 扫描、summary、层级连边、断点诊断；历史分层长文见 `archive/legacy/docs/hierarchy_notes.md` |
| `app_layer/retrieval.py` | TF‑IDF 关键词检索 CLI；可与某分辨率 membership 组合 |
| `algorithm_layer/time_window.py` | 时间窗 inherited/refit、滑动窗动画等 |
| `analysis_layer/checklist.py` | 数据/embedding/聚类对比检查 |

更细的分层说明见 **`src/README.md`**；模块边界长文见 **`architecture-boundaries.md`**（同目录）。

---

## 14. 日常流水线与 `out/` 目录角色

只需记住三类目录（白话）：

| 位置 | 含义 |
|------|------|
| `out/leiden_sweep_*`、粗分合并目录等 | **社区划分结果**（`membership_r*.npy`、`summary.npy`） |
| `out/experiments/` | **登记表**：`manifest.json` 指向各 sweep；Web 下拉选 run 依赖此树 |
| `out/experiment_eval/` | **汇总评测**：如 `evaluation_overview.csv/json`、`comparison_breakpoints.csv`，Web Eval 与断点表 |

在仓库根执行（顺序：**自检数据 → 写 manifest（目录存在才登记）→ 各 sweep 轻量 `eval/` → 断点表 → 更新汇总**）：

```bash
bash scripts/daily_workflow.sh              # 完整日常流水线
bash scripts/daily_workflow.sh --force-data # 强制重写 data_check.txt 后再跑
bash scripts/daily_workflow.sh --viz        # 同上 + build-2d + 导出 `out/viz/umap_communities_cpm_r*.png`
```

本地 `out/` 中哪些提交入库、哪些大包在 `archive/`，见仓库内 **`out/README.txt`**（该文件可提交；其余 `out/**` 默认 `.gitignore`）。

**仅导出 UMAP + 当前分辨率社区着色 PNG**（不跑 sweep）：

```bash
PYTHONPATH=src python scripts/plot_umap_membership.py \
  --umap-npy out/umap2d.npy \
  --leiden-dir out/leiden_sweep_cpm \
  --resolution 0.2 \
  --out-png out/viz/umap_communities_cpm.png
```

---

## 15. 扩展实验与大规模离线对比

在已有 **`data/paper_embeddings_specter2.npy`**（或 `--emb-path`）与 **`out/mutual_knn_k*.npz`** 时，可跑 **Louvain 全图 sweep**、**k‑means 粗分 → 各 domain 诱导子图 CPM** 等扩展产物；此路径**不必安装 PyTorch**（`experiment-sweep` 从 npy 读向量）：

```bash
bash scripts/run_extended_experiments.sh
```

粗分 sweep 细节：`python src/core.py experiment-coarse-kmeans-sweep --help`。若 manifest 登记 **`--topic-communities-csv`**（Topic‑SCORE 的 `communities_topic_weights.csv`），Web 右侧详情可展示主题词与权重。

**四算法 × 多分辨率 × 主题/检索等** 的编排与目录约定以 **[`experiment-comparison-pipeline.md`](experiment-comparison-pipeline.md)**（契约长文）、**[`offline-outputs-catalog.md`](offline-outputs-catalog.md)**（路径与产物）为准；早期「推荐执行顺序」长文见 **[`offline-comparison-master-plan.md`](../archive/legacy/docs/refactor_archived/offline-comparison-master-plan.md)**。

```text
bash scripts/offline_comparison_master.sh sweep
  → bash scripts/daily_workflow.sh
  → bash scripts/offline_comparison_master.sh topics
     （默认断点表分辨率；全网格：`TOPIC_GRID=full`）
  →（可选）offline_comparison_master 的 viz / topic-viz / retrieval 等子命令
```

---

## 16. 命令行：最小主流程与研究流程

**最小主流程：**

```bash
python src/core.py check-data
python src/core.py embed
python src/core.py check-embed
python src/core.py build-2d
python src/core.py build-graph --k 50
python src/core.py sweep --r-min 0.2 --r-max 2.0 --step 0.05 --k 50
```

**研究流程（示例链）：**

```bash
python src/core.py check-data
python src/core.py embed
python src/core.py check-embed
python src/core.py build-2d --method umap
python src/core.py build-graph --k 50 --knn-backend hnswlib
python src/core.py sweep --r-min 0.2 --r-max 2.0 --step 0.02 --k 50
python src/core.py hierarchy --r-min 0.2 --r-max 2.0 --step 0.02 --k 50
python src/core.py time-window --start-year 1995 --end-year 1999 --resolution 1.0
python src/core.py time-video --resolution 1.0 --window-size 5 --step 1 --fps 2
python src/core.py keyword-index --use-bigrams
python src/core.py keyword-search --query "bayesian nonparametrics" --top-k 15 --resolution 1.0
```

（以上默认需在已 `activate` 的 venv 中且 **`PYTHONPATH=src`** 与仓库根工作目录一致；单条命令参数以 **`python src/core.py <task> --help`** 为准。）

---

## 17. `core.py` 子命令与帮助

```bash
python src/core.py --help
python src/core.py <task> --help
```

**按主题分组（完整参数与示例见各 task 的 `--help`）**

| 主题 | 子命令（节选） |
|------|----------------|
| 数据与嵌入 | `check-data`、`embed`、`check-embed` |
| 2D 与图 | `build-2d`、`build-graph`、`graph-layout` |
| 社区与层级 | `sweep`、`hierarchy`、`subgraph-hierarchy` |
| 时间 | `time-window`、`time-video` |
| 检索 | `keyword-index`、`keyword-search` |
| 主题 | `topic-model`、`topic-model-multi`、`align-topics`、`align-topics-segmented`、`topic-viz`、`diagnose-topic-collapse`、`frames-to-mp4` |
| 实验 / 评测 | `experiment-*` 系列（见 `cli/experiment_subcommands.py` 注册项） |
| Demo | `demo-api`、`demo-graph`、`demo-keyword` 等 |
| Web 生产路径 | 以 **`demo-api`** 为主（§7） |

**`hierarchy` / `subgraph-hierarchy` 示例与输出文件名** 较长，日常以 `--help` 与 `offline-outputs-catalog.md` 为准；历史将 `hierarchy_viz` 单独脚本化的说明见 **`archive/legacy/src/hierarchy_tools_notes.md`**。

---

## 18. 缓存复用与 `--force`

常见「有输出则跳过」行为（除非对应 `--force` / `--force-reingest`）：

- `check-data`、`check-embed`：报告存在则跳过  
- `embed`、`build-2d`、`build-graph`、`graph-layout`：缓存存在则加载  
- `sweep` / `hierarchy`：已有 `membership_r*.npy` 等可复用（视 task 而定）  
- `keyword-index`：索引目录存在则复用  
- `topic-model`：`topic_model_meta.json` 存在则跳过；`topic-model-multi` 见 `--skip-existing`

若更换 **数据源**、**exclude_selfcite** 或 **ingest 清洗规则**，应对数据缓存执行：

```bash
python src/core.py check-data --force-reingest --force
```

或删除 `data/data_store.pkl` 后重跑 `check-data`。

---

## 19. 典型产物字段速览

- **`out/leiden_sweep*/summary.csv`**（每行一个分辨率）：`resolution`、`n_comm`、`quality`、`time`、`vi_adjacent`、`delta_n_comm`、`ratio_n_comm` 等  
- **`out/leiden_hierarchy*/hierarchy_edges.csv`**：`r_parent`、`community_parent`、`r_child`、`community_child`、`intersection`、`jaccard` 等相邻分辨率连边字段  
- **`out/time_windows/.../compare/summary.json`**：`ari`、`nmi`、`direct_equal_rate`、`best_alignment_acc`、`purity_inherited_to_refit` 等 inherited vs refit 指标  

主题与关键词目录内文件名见 **`offline-outputs-catalog.md`**。

---

## 20. 版本说明

本手册随当前主分支实现整理；若你升级依赖或改动 manifest 字段，请以 **`src/` 源码与 `out/experiments` 样例 manifest** 为最终依据。根目录 **`README.md`** 中的长篇任务说明已收束到本文件 §11–§19 与 **`python src/core.py --help`**，避免双份维护。
