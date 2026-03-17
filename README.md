# Paper Community Analysis 项目 README

---

## 1. 项目目标

这个项目的核心目标是：

1. 从原始统计学论文数据中构建 `Author / Paper` 对象；
2. 对论文文本做 embedding（当前是 SPECTER2）；
3. 在 embedding 空间上构建 mutual-kNN 图；
4. 在图上做 Leiden 社区发现，并扫描不同 resolution；
5. 将社区结果投影到统一的 2D 嵌入图上做可视化；
6. 在社区之上进一步做主题建模、主题对齐、多分辨率比较；
7. 加入时间窗口分析；
8. 后续扩展到检索、交互式浏览和网页展示。

---

## 2. 当前源码结构总览

当前源码中主要文件及职责如下。

### 2.1 数据与对象层

#### `getdata.py`

负责从原始数据文件中读取内容，并落盘为统一缓存 `data_store.pkl`。

主要来源：

- `author_name.txt`
- `AuthorPaperInfo_py.RData`
- `TextCorpusFinal_py.RData`
- `TopicResults_py.RData`
- `RawPaper_py.RData`

主要输出：

- `data/data_store.pkl`

缓存内容包括：

- 作者名
- 作者-论文关系
- 论文-作者关系
- 引用关系 `paper_refs`
- 年份 `paper_year`
- 标题 `paper_title`
- 摘要 `paper_abstract`
- clean 摘要 `paper_abstract_clean`

#### `model.py`

负责把 `data_store.pkl` 中的原始字典，构造成程序运行时使用的对象：

- `Author`
- `Paper`

注意：

- 对象采用 **1-based** 设计；
- 即 `authors[aid].id == aid`，`papers[pid].id == pid`；
- `authors[0]` 和 `papers[0]` 是 `None` 占位。

---

### 2.2 检查与诊断层

#### `checklist.py`

负责做数据与 embedding 的 sanity check。

主要函数：

- `run_model_checks(...)`
- `run_embedding_checks(...)`
- `compare_clusterings(...)`（供时间窗口比较继承社区 vs 重跑社区）

主要输出：

- `data/data_check.txt`
- `data/embedding_check.txt`
- 时间窗口比较报告 `cluster_compare.txt`

---

### 2.3 向量化与图构建层

#### `embedding.py`

负责对每篇论文做文本 embedding。

当前方法：

- SPECTER2 base + proximity adapter
- 文本拼接方式：`title + [SEP] + abstract`

主要函数：

- `embed_all_papers(...)`

主要输出：

- `data/paper_embeddings_specter2.npy`

- 当前代码中的 `pick_device()` 固定返回 CPU；

#### `network.py`

负责在 embedding 上构建邻近图。

包含：

- 行归一化
- cosine KNN 搜索
- `faiss / sklearn / hnswlib` 三种 backend
- mutual-kNN 图构建
- 稀疏图缓存

主要函数：

- `build_or_load_mutual_knn_graph(...)`

主要输出：

- `out/mutual_knn_k50.npz`（默认命名）

---

### 2.4 2D 可视化层

#### `diagram2d.py`

负责：

- embedding 到 2D 的降维
- 社区着色散点图
- graph layout

主要函数：

- `embed_2d(...)`
- `plot_scatter(...)`
- `graph_layout_2d(...)`
- `labels_to_colors_by_centroid(...)`

主要输出：

- `out/umap2d.npy`
- `out/fig_umap2d.png`
- 可选的 graph layout 坐标与图片

当前支持的 2D 方法：

- `pca`
- `umap`
- `pacmap`

---

### 2.5 社区发现层

#### `community.py`

负责 Leiden 社区发现与分辨率扫描。

主要函数：

- `make_resolutions(...)`
- `pick_nearest_resolution(...)`
- `leiden_sweep(...)`

主要输出：

- `out/leiden_sweep/summary.npy`
- `out/leiden_sweep/membership_rXXXX.npy`

**当前实际实现需要特别注意：**

`make_resolutions(...)` 虽然接口里有 `r_min / r_max / step`，但当前实现内部仍然是：

- 使用固定的 `np.geomspace(1e-4, 5.0, 60)`
- 再 round

这意味着：

- 当前代码**不是严格按 step 线性扫描**；
- `r_min / r_max / step` 对结果的控制并不像函数签名看上去那么直接。

这一点会直接影响你对“分辨率 sweep 是否足够细”的判断。

---

### 2.6 时间窗口分析层

#### `time_window.py`

负责时间窗分析与动态可视化。

核心功能：

1. 统计年份信息；
2. 选取一个时间窗口内的论文；
3. 路线 A：继承全图社区，再抽出该窗口 induced subgraph；
4. 路线 B：只保留该窗口论文，重建图并重跑 Leiden；
5. 对比两种分类；
6. 做滑动窗口动画。

主要函数：

- `collect_time_info(...)`
- `analyze_time_window(...)`
- `make_sliding_window_video(...)`
- `list_sliding_windows(...)`
- `window_indices_from_years(...)`

主要输出：

- `out/time_info/time_info.npz`
- `out/time_info/time_info_summary.json`
- `out/time_windows/window_YYYY_YYYY_rX.XXXX/...`
- `out/time_windows_animation/...`

---

### 2.7 检索层

#### `retrieval.py`

当前已经实现的是**最基础的关键词检索**。

主要能力：

- 用标题和摘要构建 TF-IDF 索引；
- 支持 title boost；
- 支持 unigram / bigram；
- 支持在命中结果里附带某个 resolution 下的社区编号。

主要函数：

- `build_or_load_keyword_index(...)`
- `search_keywords(...)`

主要输出：

- `out/keyword_index/tfidf_docs.npz`
- `out/keyword_index/vectorizer.pkl`
- `out/keyword_index/meta.json`

**注意：当前 `retrieval.py` 没有命令行入口。**
所以目前只能在 Python 中 import 调用。

---

<!-- ### 2.8 主题建模与多分辨率分析层

#### `topic_modeling.py`
对**单个分辨率**的社区结果做 Topic-SCORE 主题建模。

输入：
- `data_store.pkl`
- 某个分辨率对应的 `membership_r....npy`

输出（默认到 `out/topic_modeling/r{resolution}_K{k}`）：
- `A_hat.npy`
- `W_hat_all.npy`
- `W_hat_fit.npy`
- `community_ids_all.npy`
- `community_ids_fit.npy`
- `M_trunk.npy`
- `vocab.txt`
- `D_all.npz`
- `D_fit.npz`
- `topics_top_words.csv`
- `communities_topic_weights.csv`
- `topic_representative_communities.csv`
- `topic_model_meta.json`

#### `topic_modeling_multi.py`
对多个 resolution 的 community 结果批量做 Topic-SCORE。

#### `align_topics_multires.py`
将多个 resolution 的主题结果，用固定参考分辨率做对齐。

#### `align_topics_multires_segmented.py`
做分段 topic 对齐；适合 resolution 很多、主题在不同区间变化明显时使用。

#### `topic_visualization_multires.py`
将多分辨率 topic 结果映射回全局 2D 图做可视化。

#### `diagnose_topic_collapse.py`
诊断 topic 是否塌缩、有效 topic 数是否过少、top1 占比是否异常等。

#### `frames_to_mp4.py`
把一组帧图合成为 MP4，既可单目录，也可批量处理多个目录。
 -->
---

## 4. `core.py`

### `build_or_load(exclude_selfcite=False)`

作用：

- 数据 ingest + data_store 加载 + Author/Paper 对象构建。

### `build_or_load_embeddings(...)`

作用：

- 读取或计算论文 embedding。

### `build_or_load_global_2d(...)`

作用：

- 读取或计算全局 2D 坐标。

### `build_or_load_global_graph(...)`

作用：

- 读取或构建全局 mutual-kNN 图。

### `build_igraph_from_edge_triplets(...)`

作用：

- 用边表 `(u, v, w)` 构建 igraph 图对象。

### `prepare_global_pipeline(...)`

作用：

- 一次性准备全局上下文：
  - authors / papers / data
  - embeddings
  - 全局 2D
  - 全局 mutual-kNN 图
  - igraph 图
  - time_info

如果你后面要频繁做时间窗口或检索，这个函数很适合当统一入口。

### `run_single_time_window(...)`

作用：

- 跑一个固定年份窗口；
- 同时得到 inherited / refit 两套结果和对比报告。

### `run_time_window_animation(...)`

作用：

- 跑滑动时间窗口；
- 复制各窗口对比图为帧；
- 合成 MP4/GIF。

---

## 5. 运行流程

### Step1

```bash
python src/core.py
```

这一步会尽量复用已有缓存，完成：

- ingest / load
- data check
- embedding
- embedding check
- UMAP 2D
- mutual-kNN 图

### 第二步：在 Python 中做 Leiden sweep

#### 主要参数说明

- `--k`：主题数
- `--resolution`：Leiden 分辨率
- `--membership`：直接指定 membership 文件
- `--data-store`：数据缓存路径
- `--stopwords`：停用词路径
- `--out-dir`：输出目录
- `--no-title / --no-abstract / --no-authors`：关闭对应文本来源
- `--use-clean-abstract`：使用 clean 摘要
- `--title-weight / --abstract-weight / --author-weight`：字段权重
- `--words-percent / --docs-percent / --min-community-size`：过滤参数
- `--vh-method`：`svs / sp / svs-sp`
- `--m / --k0 / --mquantile / --m-trunc-mode`：SCORE/vertex hunting 参数
- `--no-weighted-nnls`：关闭加权 NNLS
- `--max-papers`：只处理前若干篇（调试）

---

### 6.3 `topic_modeling_multi.py`

#### 批量处理所有 membership

```bash
python src/topic_modeling_multi.py --k 10
```

#### 跳过已完成分辨率

```bash
python src/topic_modeling_multi.py --k 10 --skip-existing
```

#### 只处理部分分辨率区间

```bash
python src/topic_modeling_multi.py \
  --k 10 \
  --r-min 0.5 \
  --r-max 1.5 \
  --include 1.0
```

#### 显式指定分辨率列表

```bash
python src/topic_modeling_multi.py \
  --k 10 \
  --resolutions 0.5 1.0 1.5
```

#### 只预览，不执行

```bash
python src/topic_modeling_multi.py --k 10 --dry-run
```

#### 主要参数说明

- `--k`：主题数
- `--leiden-dir`：membership 文件目录
- `--membership-glob`：membership 匹配模式
- `--out-root`：批量输出目录根
- `--summary-name`：总表名
- `--r-min / --r-max / --include / --resolutions`：分辨率筛选
- `--skip-existing`：已有结果则跳过
- `--continue-on-error`：单个失败继续
- 其余 topic 参数与 `topic_modeling.py` 基本一致

---

### 6.4 `align_topics_multires.py`

#### 用固定参考分辨率对齐

```bash
python src/align_topics_multires.py \
  --k 10 \
  --ref-resolution 1.0
```

#### 指定分辨率范围

```bash
python src/align_topics_multires.py \
  --k 10 \
  --r-min 0.2 \
  --r-max 2.0 \
  --include 1.0
```

#### 保存相似度矩阵

```bash
python src/align_topics_multires.py \
  --k 10 \
  --ref-resolution 1.0 \
  --save-sim-matrix
```

#### 主要参数说明

- `--k`：主题数
- `--topic-root`：topic 结果根目录
- `--out-root`：对齐结果输出根目录
- `--ref-resolution`：参考分辨率
- `--metric`：`cosine` 或 `js`
- `--topn-common-vocab / --min-common-vocab`：共同词表设置
- `--r-min / --r-max / --include / --resolutions`：挑选分辨率
- `--copy-meta-json`：复制 topic_model_meta.json
- `--save-sim-matrix`：保存相似度矩阵
- `--dry-run`：预览模式

---

### 6.5 `align_topics_multires_segmented.py`

#### 最基本用法

```bash
python src/align_topics_multires_segmented.py --k 10
```

#### 调整切段阈值

```bash
python src/align_topics_multires_segmented.py \
  --k 10 \
  --break-avg-thresh 0.85 \
  --break-min-thresh 0.10 \
  --min-segment-size 2
```

#### 保存额外对齐信息

```bash
python src/align_topics_multires_segmented.py \
  --k 10 \
  --save-sim-matrix \
  --save-adjacent-topic-rows
```

#### 主要参数说明

- `--k`：主题数
- `--metric`：topic 相似度度量
- `--break-avg-thresh`：相邻平均相似度低于该值则切段
- `--break-min-thresh`：相邻最小相似度低于该值则切段
- `--min-segment-size`：最短段长度
- `--anchor-strategy`：`best-adjacent` 或 `middle`
- `--clean-segment-dirs`：清理旧的 segment 目录

---

### 6.6 `topic_visualization_multires.py`

#### 最基本用法

```bash
python src/topic_visualization_multires.py --k 10
```

#### 只可视化部分分辨率

```bash
python src/topic_visualization_multires.py \
  --k 10 \
  --resolutions 0.5 1.0 1.5
```

#### 画社区质心图并标注大社区

```bash
python src/topic_visualization_multires.py \
  --k 10 \
  --community-centroid \
  --annotate-top-n-communities 15
```

#### 批量生成 segmented 对齐结果的可视化

```bash
python src/topic_visualization_multires.py \
  --k 10 \
  --batch-segments
```

#### 主要参数说明

- `--k`：主题数
- `--leiden-dir`：membership 目录
- `--topic-root`：topic 结果根目录
- `--out-dir`：输出目录
- `--umap`：全局 2D 坐标路径
- `--graph-layout`：graph layout 路径
- `--resolutions / --r-min / --r-max / --include`：选择分辨率
- `--max-points`：采样点数
- `--point-size / --alpha`：绘图参数
- `--skip-graph`：不画 graph layout 帧
- `--community-centroid`：额外输出社区质心图
- `--batch-segments`：处理 segmented 结果

---

### 6.7 `diagnose_topic_collapse.py`

#### 分析单个 `communities_topic_weights.csv`

```bash
python src/diagnose_topic_collapse.py \
  --input out/topic_modeling/r1.0000_K10/communities_topic_weights.csv \
  --save-per-topic \
  --save-plots
```

#### 递归分析一个根目录

```bash
python src/diagnose_topic_collapse.py \
  --root out/topic_modeling_multi/K10 \
  --save-per-topic \
  --save-plots \
  --verbose
```

主要输出：
- `topic_collapse_diagnostics.csv`
- `topic_collapse_per_topic_long.csv`
- 若开启绘图，还会输出一系列诊断图

---

### 6.8 `frames_to_mp4.py`

#### 单个帧目录合成 MP4

```bash
python src/frames_to_mp4.py \
  --frame-dir out/topic_viz_multi/K10/frames_topic_umap \
  --out out/topic_viz_multi/K10/topic_umap.mp4 \
  --fps 8
```

#### 批量处理多个 frames 子目录

```bash
python src/frames_to_mp4.py \
  --batch-subdirs \
  --root out/topic_viz_multi/K10 \
  --subdir-glob 'frames*' \
  --fps 8
```

#### 主要参数说明

- `--frame-dir`：单目录模式帧路径
- `--out`：输出 mp4 文件
- `--fps`：帧率
- `--pattern`：文件名正则过滤
- `--resize-mode`：`first` 或 `even`
- `--codec`：默认 `libx264`
- `--quality`：质量参数
- `--batch-subdirs`：批量模式开关
- `--root / --subdir-glob / --out-root`：批量模式相关参数

---

## 7. 当前关键词检索怎么运行

因为 `retrieval.py` 目前没有命令行入口，推荐这样在 Python 中使用。

### 7.1 构建或加载关键词索引

```python
from pathlib import Path
from core import build_or_load
from retrieval import build_or_load_keyword_index, KeywordIndexConfig

authors, papers, data = build_or_load(exclude_selfcite=False)

cfg = KeywordIndexConfig(
    ngram_min=1,
    ngram_max=2,
    min_df=2,
    max_df=0.2,
    max_features=250000,
    use_title=True,
    use_abstract=True,
    title_boost=3,
)

bundle = build_or_load_keyword_index(
    papers,
    out_dir=Path("out"),
    cfg=cfg,
    force=False,
    verbose=True,
)
```

### 7.2 执行查询

```python
from pathlib import Path
from core import build_or_load
from retrieval import search_keywords, KeywordIndexConfig

authors, papers, data = build_or_load(exclude_selfcite=False)

res = search_keywords(
    papers,
    query="bayesian nonparametrics",
    out_dir=Path("out"),
    top_k=20,
    resolution=1.0,
    leiden_dir=Path("out/leiden_sweep"),
    cfg=KeywordIndexConfig(ngram_min=1, ngram_max=2),
    force_reindex=False,
    verbose=True,
)

for hit in res["hits"][:10]:
    print(hit["pid"], hit["score"], hit["year"], hit["community"], hit["title"])
```

返回结果中每条命中通常包括：
- `pid`
- `score`
- `year`
- `title`
- `community`（如果你提供了 resolution + leiden_dir）
- `snippet`

---

## 8. 你当前最常用的几个目录

### `data/`
一般放：
- 原始输入数据
- 中间缓存
- 检查报告

常见文件：
- `author_name.txt`
- `AuthorPaperInfo_py.RData`
- `TextCorpusFinal_py.RData`
- `TopicResults_py.RData`
- `RawPaper_py.RData`
- `data_store.pkl`
- `paper_embeddings_specter2.npy`
- `data_check.txt`
- `embedding_check.txt`

### `out/`
一般放：
- 图结构缓存
- 2D 坐标
- 可视化图片
- 社区结果
- 主题建模结果
- 时间窗口输出

常见文件 / 子目录：
- `umap2d.npy`
- `fig_umap2d.png`
- `mutual_knn_k50.npz`
- `leiden_sweep/`
- `topic_modeling/`
- `topic_modeling_multi/`
- `time_info/`
- `time_windows/`
- `time_windows_animation/`

---

## 9. 推荐的目录结构

建议项目根目录结构类似：

```text
project_root/
├── data/
│   ├── author_name.txt
│   ├── AuthorPaperInfo_py.RData
│   ├── TextCorpusFinal_py.RData
│   ├── TopicResults_py.RData
│   ├── RawPaper_py.RData
│   ├── data_store.pkl
│   ├── paper_embeddings_specter2.npy
│   ├── data_check.txt
│   └── embedding_check.txt
├── out/
│   ├── umap2d.npy
│   ├── fig_umap2d.png
│   ├── mutual_knn_k50.npz
│   ├── leiden_sweep/
│   ├── topic_modeling/
│   ├── topic_modeling_multi/
│   ├── time_info/
│   ├── time_windows/
│   └── time_windows_animation/
└── src/
    ├── core.py
    ├── getdata.py
    ├── model.py
    ├── embedding.py
    ├── network.py
    ├── diagram2d.py
    ├── community.py
    ├── checklist.py
    ├── time_window.py
    ├── retrieval.py
    ├── topic_modeling.py
    ├── topic_modeling_multi.py
    ├── align_topics_multires.py
    ├── align_topics_multires_segmented.py
    ├── topic_visualization_multires.py
    ├── diagnose_topic_collapse.py
    └── frames_to_mp4.py
```

---

## 10. 环境依赖建议

根据当前代码，建议至少准备这些包：

```bash
pip install numpy scipy pandas matplotlib scikit-learn
pip install python-igraph leidenalg
pip install umap-learn
pip install hnswlib
pip install transformers adapters torch
pip install pyreadr
pip install imageio imageio-ffmpeg
```

如果你想测试其他 KNN backend：

```bash
pip install faiss-cpu
```

如果想用 PaCMAP：

```bash
pip install pacmap
```

---

## 11. 当前代码里的几个重要注意事项

这一节不是泛泛而谈，而是基于你当前上传代码的**真实状态**。

### 11.1 `core.py` 目前还不是“黑盒任务终端”

如果你希望未来像下面这样用：

```bash
python src/core.py embed
python src/core.py sweep
python src/core.py topic-model --k 10 --resolution 1.0
```

那还需要继续把 `core.py` 改造成 argparse 子命令分发器。

当前版本还没有做到这一步。

### 11.2 `community.py` 的 resolution sweep 目前不是严格 step 扫描

虽然函数签名里有：
- `r_min`
- `r_max`
- `step`

但内部实际是固定 `geomspace(1e-4, 5.0, 60)`。

这意味着：
- “我把 step 调成 0.01 就会更细”在当前版本里并不完全成立；
- 如果你要认真做 breakpoint / hierarchy 分析，这里建议优先修。

### 11.3 `time_window.py` 与 `diagram2d.py` 当前存在接口不一致风险

`time_window.py` 里使用了：
- `compute_xy_limits(...)`
- `plot_scatter(..., xlim=..., ylim=...)`

但当前 `diagram2d.py` 里：
- 没有 `compute_xy_limits` 定义；
- `plot_scatter` 也没有 `xlim / ylim` 参数。

这意味着当前这两个文件如果按你上传的这版直接运行，**时间窗口模块很可能会报错**。

如果你之前有一版修过的 `diagram2d.py`，需要把它和这份 `time_window.py` 对齐。

### 11.4 当前时间窗口视频导出逻辑也比较脆弱

`time_window.py` 里视频合成优先走 `imageio`，其次走 `ffmpeg`，再退回 GIF。

如果帧尺寸不一致，MP4 有可能打不开。你之前已经碰到过这一类问题，所以这块后续最好单独稳定一下。

### 11.5 embedding 当前固定 CPU

`embedding.py` 中 `pick_device()` 现在直接返回 CPU，这对大规模数据会比较慢。

---

## 12. 我建议你下一步怎么用这份代码

如果你想在不继续改代码的前提下，尽量稳定地推进，我建议顺序如下：

### 路线 A：先做一个稳定的全局结果

1. 跑：

```bash
python src/core.py
```

2. 手动跑 Leiden sweep。

3. 选择几个代表分辨率做：

```bash
python src/topic_modeling.py --k 10 --resolution 1.0
```

4. 再用：

```bash
python src/topic_modeling_multi.py --k 10 --skip-existing
```

5. 然后：

```bash
python src/align_topics_multires_segmented.py --k 10
python src/topic_visualization_multires.py --k 10 --batch-segments
```

### 路线 B：先做最简单检索

1. 先跑 `core.py` 生成基础缓存；
2. 用 `retrieval.py` 建关键词索引；
3. 先在 Python 中做查询；
4. 等你把 `core.py` 改造成子命令之后，再补命令行或网页接口。

### 路线 C：等接口对齐后再回到时间窗口

由于当前 `time_window.py` / `diagram2d.py` 接口存在不一致，建议先不要把时间窗口当最优先稳定模块，除非你先把这两个文件重新对齐。

---

## 13. 一个常见工作流示例

假设你从零开始：

### 13.1 先跑基础主流程

```bash
python src/core.py
```

### 13.2 再进入 Python 做 Leiden sweep

```python
from pathlib import Path
from core import build_or_load, build_or_load_embeddings, build_or_load_global_graph, build_igraph_from_edge_triplets
from community import leiden_sweep
import numpy as np

authors, papers, data = build_or_load()
embs = build_or_load_embeddings(papers)
X = np.asarray(embs[1:], dtype=np.float32)
_, (u, v, w) = build_or_load_global_graph(embs, k=50)
G = build_igraph_from_edge_triplets(X.shape[0], u, v, w)

leiden_sweep(
    G,
    out_dir=Path("out/leiden_sweep"),
    r_min=0.2,
    r_max=2.0,
    step=0.05,
    include=[1.0],
)
```

### 13.3 单分辨率 topic

```bash
python src/topic_modeling.py --k 10 --resolution 1.0
```

### 13.4 多分辨率 topic

```bash
python src/topic_modeling_multi.py --k 10 --skip-existing
```

### 13.5 对齐与可视化

```bash
python src/align_topics_multires_segmented.py --k 10
python src/topic_visualization_multires.py --k 10 --batch-segments
```

### 13.6 诊断 topic collapse

```bash
python src/diagnose_topic_collapse.py --root out/topic_modeling_multi/K10 --save-plots
```

### 13.7 关键词检索

```python
from pathlib import Path
from core import build_or_load
from retrieval import search_keywords

authors, papers, data = build_or_load()
res = search_keywords(
    papers,
    query="nonparametric Bayes",
    out_dir=Path("out"),
    top_k=10,
    resolution=1.0,
    leiden_dir=Path("out/leiden_sweep"),
)
print(res["hits"][:5])
```

---

## 14. 总结

当前这套代码已经具备了以下核心能力：

- 原始数据 ingest 和对象化
- 论文 embedding
- 全局 2D 嵌入
- mutual-kNN 图构建
- Leiden 社区扫描
- 基于社区的 Topic-SCORE 主题建模
- 多分辨率 topic 对齐与可视化
- 时间窗口分析框架
- 基础关键词检索

但从“工程入口”的角度看，还没有完全统一成一个总控终端。当前最适合的理解方式是：

- `core.py`：基础全局构建与若干 import 接口
- `topic_*.py / align_*.py / diagnose_*.py / frames_to_mp4.py`：已经可以单独命令行运行的分析脚本
- `retrieval.py / time_window.py`：目前更适合作为 Python 模块使用

如果你下一步要继续推进，我最推荐优先做的是两件事：

1. 把 `core.py` 改成真正的 argparse 子命令入口；
2. 统一 `time_window.py` 与 `diagram2d.py` 的接口。

这样整个项目会从“研究代码集合”进一步走向“可操作的分析系统”。
