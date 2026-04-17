# Paper Community Analysis Pipeline

这个项目围绕一套统一的数据流展开：

1. 从 `RData/txt` 读取原始数据并落盘成 `data/data_store.pkl`
2. 构建 `Author` / `Paper` 对象
3. 计算论文文本向量（SPECTER2）
4. 做全局 2D 嵌入（UMAP / PCA / PaCMAP）
5. 基于 embedding 构建 mutual-kNN 图
6. 在图上做 Leiden 社区发现与多分辨率扫描
7. 进一步做层级结构诊断、时间窗口分析、关键词检索、主题建模、topic 对齐和可视化

```bash
python src/core.py <task> [args...]
```

`core.py` 已经被整理成统一的命令行总控；常见任务都可以从这里触发，而不需要手动分别运行多个脚本。

---

## 每个源码文件的职责

### `core.py`

统一命令行入口。负责把所有模块包装成黑盒任务。

### `getdata.py`

读取 `.RData` / `author_name.txt`，整理出可序列化的数据字典，落盘为 `data_store.pkl`。

### `model.py`

把 `data_store.pkl` 转成 `Author` / `Paper` 对象列表。

### `embedding.py`

用 SPECTER2 计算论文 embedding，输出 `paper_embeddings_specter2.npy`。

### `network.py`

在 embedding 上做 KNN，构建 mutual-kNN 无向加权图，并缓存成 `npz`。

### `diagram2d.py`

把 embedding 降到 2D，并支持散点图绘制与 graph-layout 2D 布局。

### `community.py`

Leiden 社区发现、多分辨率扫描、summary 输出、层级连接、突变点诊断。分层思路与常见范式对照见 [docs/hierarchy/hierarchical_communities.md](docs/hierarchy/hierarchical_communities.md)。

### `retrieval.py`

最简单的关键词检索（TF-IDF）。可选读取某个 resolution 的社区标签并附到检索结果上。

### `time_window.py`

时间窗分析：

- 路线 1：继承全图社区，再在窗口内抽 induced subgraph
- 路线 2：窗口内重建图并重跑 Leiden
- 路线 3：对比两种分类
- 路线 4：滑动窗口动画

### `checklist.py`

数据结构检查、embedding 检查、聚类结果比较。

---

## 环境依赖

```bash
pip install numpy scipy pandas matplotlib scikit-learn pyreadr python-igraph leidenalg pillow imageio imageio-ffmpeg
```

可选：

```bash
pip install umap-learn pacmap hnswlib faiss-cpu transformers adapters torch
```

说明：

- `embedding.py` 使用 SPECTER2，需要 `transformers + adapters + torch`
- `community.py` 需要 `python-igraph + leidenalg`
- `network.py` 的 KNN 后端可以选 `faiss / sklearn / hnswlib`
- `time_window.py` 导出 mp4 时优先调用系统 `ffmpeg`；没有时会回退到 GIF

---

## 运行流程

### 最小主流程

```bash
python src/core.py check-data
python src/core.py embed
python src/core.py check-embed
python src/core.py build-2d
python src/core.py build-graph --k 50
python src/core.py sweep --r-min 0.2 --r-max 2.0 --step 0.05 --k 50
```

### 研究流程

```bash
# 1) 数据 sanity check
python src/core.py check-data

# 2) 文本向量化
python src/core.py embed
python src/core.py check-embed

# 3) 全局 2D 图
python src/core.py build-2d --method umap

# 4) 构图
python src/core.py build-graph --k 50 --knn-backend hnswlib

# 5) 社区多分辨率扫描
python src/core.py sweep --r-min 0.2 --r-max 2.0 --step 0.02 --k 50

# 6) 层级诊断
python src/core.py hierarchy --r-min 0.2 --r-max 2.0 --step 0.02 --k 50

# 7) 时间窗口
python src/core.py time-window --start-year 1995 --end-year 1999 --resolution 1.0

# 8) 滑动窗口动画
python src/core.py time-video --resolution 1.0 --window-size 5 --step 1 --fps 2

# 9) 关键词检索
python src/core.py keyword-index --use-bigrams
python src/core.py keyword-search --query "bayesian nonparametrics" --top-k 15 --resolution 1.0


---

## `core.py` 支持的所有任务

可以先看帮助：

```bash
python src/core.py --help
python src/core.py <task> --help
```

下面是每个任务的作用和典型命令。

### 1. `check-data`

读取数据并做模型结构检查。

```bash
python src/core.py check-data
python src/core.py check-data --out-path data/data_check.txt
python src/core.py check-data --force-reingest --force
```

输出：

- 默认报告：`data/data_check.txt`

### 2. `embed`

计算或加载论文 embedding。

```bash
python src/core.py embed
python src/core.py embed --batch-size 8
python src/core.py embed --prefer-gpu
python src/core.py embed --emb-path data/paper_embeddings_specter2.npy --force
```

输出：

- `data/paper_embeddings_specter2.npy`

### 3. `check-embed`

检查 embedding 质量。

```bash
python src/core.py check-embed
python src/core.py check-embed --out-path data/embedding_check.txt
```

输出：

- 默认报告：`data/embedding_check.txt`

### 4. `build-2d`

生成全局 2D 坐标。

```bash
python src/core.py build-2d
python src/core.py build-2d --method pca
python src/core.py build-2d --method pacmap
python src/core.py build-2d --cache-name umap2d.npy --out-png fig_umap2d.png
```

输出：

- `out/umap2d.npy`
- `out/fig_umap2d.png`

### 5. `build-graph`

生成 mutual-kNN 图。

```bash
python src/core.py build-graph --k 50
python src/core.py build-graph --k 30 --knn-backend hnswlib
python src/core.py build-graph --k 50 --knn-backend faiss
python src/core.py build-graph --k 50 --knn-backend sklearn
```

输出：

- `out/mutual_knn_k50.npz` 或你指定的缓存名

### 6. `graph-layout`

对图本身做 2D layout。

```bash
python src/core.py graph-layout --k 50 --init-from-umap
python src/core.py graph-layout --method fr --cache-name graph_fr2d.npy
```

输出：

- `out/graph_drl2d.npy`
- `out/fig_graph_layout.png`

### 7. `sweep`

Leiden 多分辨率扫描。

```bash

python src/core.py sweep \
  --out-dir out/leiden_sweep_rb \
  --r-min 0.2 --r-max 5.0 --step 0.01 \
  --k 50 \
  --partition-type RBConfigurationVertexPartition \
  --plot-reference 5.0

python src/core.py sweep \
  --r-min 0.001 --r-max 0.2 --step 0.002 \
  --partition-type CPMVertexPartition \
  --out-dir out/leiden_sweep_cpm \
  --plot-reference 0.2

python src/core.py sweep --plot-reference 1.0 # 画一张 r≈1.0 的社区着色图
```

输出目录默认：

- `out/leiden_sweep/`

主要产物：

- `summary.npy`
- `summary.csv`
- `membership_r*.npy`
- 可选 `umap_r*.png`

### 8. `hierarchy`

在 sweep 基础上生成层级连接与断点诊断。方法说明与层间对齐语义见 [docs/hierarchy/hierarchical_communities.md](docs/hierarchy/hierarchical_communities.md)。

```bash
python src/core.py hierarchy \
  --out-dir out/leiden_hierarchy_rb \
  --r-min 0.2 --r-max 5.0 --step 0.01 \
  --k 50 \
  --partition-type RBConfigurationVertexPartition \
  --min-child-share 0.25

python src/core.py hierarchy \
  --out-dir out/leiden_hierarchy_cpm \
  --r-min 0.001 --r-max 0.2 --step 0.002 \
  --k 50 \
  --partition-type CPMVertexPartition \
  --min-child-share 0.25

  python src/core.py hierarchy \
  --out-dir out/leiden_hierarchy_rb_cs020 \
  --r-min 0.2 --r-max 5.0 --step 0.01 \
  --k 50 \
  --partition-type RBConfigurationVertexPartition \
  --min-child-share 0.20

  python src/core.py hierarchy \
  --out-dir out/leiden_hierarchy_cpm_cs010 \
  --r-min 0.001 --r-max 0.2 --step 0.002 \
  --k 50 \
  --partition-type CPMVertexPartition \
  --min-child-share 0.10

  python src/hierarchy_viz.py \
  --hierarchy-dir out/leiden_hierarchy_cpm_cs010 \
  --out-prefix cpm \
  --weight-by child_share \
  --min-edge-value 0.10 \
  --max-nodes-per-layer 25 \
  --y-mode layer_index

```

输出目录默认：

- `out/leiden_hierarchy/`

主要产物：

- `summary.npy`
- `summary.csv`
- `hierarchy_nodes.csv`
- `hierarchy_edges.csv`
- `breakpoints.json`
- `sweep_diagnostics.png`

#### 诱导子图上的局部层级（`subgraph-hierarchy`）

在全局某分辨率下选定一个 **社区 id**，取该顶点集在 mutual-kNN 上的 **诱导子图**，在子图上重新做多分辨率 Leiden + 层级连边，便于观察大块内部是否出现「一分为几」而非仅全局剥落小社区。

```bash
# 先看该分辨率下最大的社区及规模（无需图文件）
python src/core.py subgraph-hierarchy \
  --leiden-dir out/leiden_sweep \
  --parent-resolution 1.0 \
  --list-top 25

# 对社区 id=12 做局部 sweep + hierarchy（默认读 out/mutual_knn_k50.npz）
python src/core.py subgraph-hierarchy \
  --leiden-dir out/leiden_sweep \
  --parent-resolution 1.0 \
  --community 12 \
  --k 50 \
  --r-min 0.2 --r-max 2.0 --step 0.03 \
  --partition-type RBConfigurationVertexPartition

# CPM 全局 sweep 在另一目录时
python src/core.py subgraph-hierarchy \
  --leiden-dir out/leiden_sweep_cpm \
  --parent-resolution 0.05 \
  --community 3 \
  --partition-type CPMVertexPartition \
  --r-min 0.001 --r-max 0.15 --step 0.002
```

输出目录默认：`out/subgraph_hierarchy/k{K}_r{parent}_c{id}_{rb|cpm}/`，内含与全图 `hierarchy` 相同的 `summary.*`、`membership_r*.npy`、`hierarchy_*.csv` 等，并额外写入：

- `meta.json`：父分辨率、社区 id、子图规模、所用图路径
- `global_vertex_indices.npy`：子图顶点对应的全局论文下标（与 `embs[1:]` 行对齐）

对多个大社区重复实验时：先用 `--list-top N` 选 id，再对每个 id 跑一遍；局部层级图可用 `python src/hierarchy_viz.py --hierarchy-dir <子目录> ...`。批量跑完后可用小脚本汇总各子目录 `summary.npy`（示例汇总：`out/subgraph_hierarchy/cpm_parent_r051_top5_summary.json`，含每个 parent 社区首次出现 \(n_{\mathrm{comm}}\ge 2\)、\(\ge 4\) 的 \(r\) 与最大社区数）。

### 9. `time-window`

对单个时间窗做 inherited/refit 对比。

```bash
python src/core.py time-window --start-year 1995 --end-year 1999 --resolution 1.0
python src/core.py time-window --start-year 2000 --end-year 2004 --resolution 0.8 --k 50
```

输出目录默认：

- `out/time_windows/`

窗口目录命名：

- `w_1995_1999_r1`

内部包含：

- `inherited/`
- `refit/`
- `compare/`
- `summary.json`

### 10. `time-video`

做滑动时间窗动画。

```bash
python src/core.py time-video --resolution 1.0 --window-size 5 --step 1 --fps 2
python src/core.py time-video --resolution 0.8 --window-size 10 --step 2 --fps 4
```

输出目录默认：

- `out/time_windows_animation/`

视频命名：

- `tw_5y_r1.mp4`
- `tw_5y_r1_s2.mp4`

如果 mp4 编码失败，会自动回退成 GIF。

<!-- ### 11. `keyword-index`
建立 TF-IDF 关键词索引。

```bash
python src/core.py keyword-index
python src/core.py keyword-index --use-bigrams
python src/core.py keyword-index --min-df 3 --max-df 0.15 --max-features 300000
```

输出：
- `out/keyword_index/`
  - `tfidf_docs.npz`
  - `vectorizer.pkl`
  - `meta.json`

### 12. `keyword-search`
做最简单的关键词检索。

```bash
python src/core.py keyword-search --query "high-dimensional inference"
python src/core.py keyword-search --query "bayesian nonparametrics" --top-k 20 --use-bigrams
python src/core.py keyword-search --query "empirical bayes" --resolution 1.0 --save-json out/search_empirical_bayes.json
```

如果带 `--resolution`，会尝试读取对应 Leiden community 标签并把社区号附在结果上。

### 13. `topic-model`
单个分辨率的 Topic-SCORE 主题建模。

```bash
python src/core.py topic-model --k-topics 10 --resolution 1.0
python src/core.py topic-model --k-topics 12 --membership out/leiden_sweep/membership_r0.8000.npy
python src/core.py topic-model --k-topics 10 --resolution 1.0 --use-clean-abstract --rep-papers-mode exact
```

默认输出目录：
- `out/topic_modeling/K10/r1.0000/`

### 14. `topic-model-multi`
多分辨率主题建模批处理。

```bash
python src/core.py topic-model-multi --k-topics 10 --skip-existing
python src/core.py topic-model-multi --k-topics 10 --r-min 0.2 --r-max 2.0
python src/core.py topic-model-multi --k-topics 10 --resolutions 0.5 0.8 1.0 1.2
```

默认输出目录：
- `out/topic_modeling_multi/K10/`

### 15. `align-topics`
固定参考分辨率的 topic 对齐。

```bash
python src/core.py align-topics --k-topics 10 --ref-resolution 1.0
python src/core.py align-topics --k-topics 10 --metric js --save-sim-matrix
```

默认输出目录：
- `out/topic_modeling_multi/K10/aligned_to_r1.0000/`

### 16. `align-topics-segmented`
分段 topic 对齐。

```bash
python src/core.py align-topics-segmented --k-topics 10
python src/core.py align-topics-segmented --k-topics 10 --break-avg-thresh 0.80 --break-min-thresh 0.05
python src/core.py align-topics-segmented --k-topics 10 --save-adjacent-topic-rows --clean-segment-dirs
```

默认输出目录：
- `out/topic_modeling_multi/K10/aligned_segmented/`

### 17. `topic-viz`
画 topic 在 UMAP / graph-layout 上的多分辨率可视化。

```bash
python src/core.py topic-viz --k-topics 10
python src/core.py topic-viz --k-topics 10 --community-centroid
python src/core.py topic-viz --k-topics 10 --batch-segments
```

默认输出目录：
- `out/topic_viz_multi/K10/`

### 18. `diagnose-topic-collapse`
诊断 topic collapse。

```bash
python src/core.py diagnose-topic-collapse --root out/topic_modeling_multi/K10 --save-plots
python src/core.py diagnose-topic-collapse --input out/topic_modeling_multi/K10/r1.0000/communities_topic_weights.csv --save-per-topic
```

### 19. `frames-to-mp4`
将帧图目录合成视频。

```bash
python src/core.py frames-to-mp4 --frame-dir out/topic_viz_multi/K10/frames_topic_umap --fps 8
python src/core.py frames-to-mp4 --batch-subdirs --root out/topic_viz_multi/K10 --subdir-glob "frames*"
``` -->

---

## 缓存与跳过逻辑

大部分任务都遵循“有输出就复用”的逻辑。

### 会自动复用已有文件的模块

- `check-data`：报告已存在则跳过，除非 `--force`
- `check-embed`：报告已存在则跳过，除非 `--force`
- `embed`：embedding 文件存在则直接加载，除非 `--force`
- `build-2d`：2D 坐标缓存存在则直接加载，除非 `--force`
- `build-graph`：图缓存存在则直接加载，除非 `--force`
- `graph-layout`：layout 缓存存在则直接加载，除非 `--force`
- `sweep`：如果 `membership_r*.npy` 已存在，会复用，除非 `--force`
- `hierarchy`：底层 sweep 结果可复用，除非 `--force`
- `keyword-index`：索引存在则复用，除非 `--force`
- `topic-model`：如果 `topic_model_meta.json` 已存在则跳过，除非 `--force`
- `topic-model-multi`：依赖其自身的 `--skip-existing`

### 关于 `--force-reingest`

`data_store.pkl` 是 ingest 的核心缓存。如果你切换了：

- 数据源文件
- `exclude_selfcite`
- 原始数据清洗规则

建议加：

```bash
python src/core.py check-data --force-reingest --force
```

或者手动删除 `data/data_store.pkl` 后重跑。

---

## 典型输出说明

### `out/leiden_sweep/summary.csv`

每个 resolution 一行，包含：

- `resolution`
- `n_comm`
- `quality`
- `time`
- `vi_adjacent`
- `delta_n_comm`
- `ratio_n_comm`

### `out/leiden_hierarchy/hierarchy_edges.csv`

相邻分辨率间 parent-community / child-community 的连接关系，包含：

- `r_parent`
- `community_parent`
- `size_parent`
- `r_child`
- `community_child`
- `size_child`
- `intersection`
- `parent_share`
- `child_share`
- `jaccard`

### `out/time_windows/.../compare/summary.json`

单个时间窗中 inherited 和 refit 的对比指标，包含：

- `ari`
- `nmi`
- `direct_equal_rate`
- `best_alignment_acc`
- `purity_inherited_to_refit`
- `purity_refit_to_inherited`

<!-- ### `out/keyword_index/meta.json`
关键词索引参数与规模。

### `out/topic_modeling...`
主题建模常见文件：
- `topics_top_words.csv`
- `communities_topic_weights.csv`
- `topic_representative_communities.csv`
- `topic_model_meta.json`
 -->
---
<!-- 
## 你当前这套代码里已经修过的关键一致性问题

这次整理时，已经统一了以下接口：

1. `core.py` 现在是任务式 CLI，而不是旧的线性实验脚本。
2. `community.py` 补齐了：
   - `load_membership_for_resolution`
   - `load_or_run_leiden_partition`
   - `run_hierarchy_sweep`
   - 层级连接和断点诊断
3. `diagram2d.py` 补齐了：
   - `compute_xy_limits`
   - `plot_scatter(..., xlim=..., ylim=...)`
4. `checklist.py` 补齐了 `compare_clusterings()`，供时间窗 inherited/refit 对比。
5. `time_window.py` 的视频输出改成更稳的统一帧尺寸流程，并把命名简化为：
   - 窗口目录：`w_1995_1999_r1`
   - 视频：`tw_5y_r1.mp4`

--- -->

<!-- 
- 如果你后面继续做网页，比较自然的后端接口就是：
  - keyword search
  - paper neighborhood lookup
  - community summary lookup
  - hierarchy navigation
  - time-window compare lookup
 -->
