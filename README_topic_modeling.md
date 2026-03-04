# 多分辨率主题建模、Topic 对齐、自动命名、可视化

本文档基于当前项目流程，说明如何完成：

1. 多分辨率社区结果的主题建模（Topic-SCORE）
2. 多分辨率 topic 对齐（保证颜色/编号跨分辨率尽量稳定）
3. 主题自动命名（中文候选，减少人工拼词）
4. 2D 可视化与动画（MP4）

最后有命令行，可以复制粘贴直接运行。

---

## 一、整体流程（建议顺序）

### A. 多分辨率主题建模（按目录里已有 membership 文件扫描）
输入：
- `out/leiden_sweep/membership_r*.npy`
- `data/data_store.pkl`
- `data/stopwords.txt`

输出（每个分辨率一个目录）：

#### 单分辨率子目录输出（每个分辨率一份）
每个分辨率目录下的文件由 `topic_modeling.py` 的 `save_outputs()` 函数统一生成，包含建模核心结果、词表、矩阵、元信息等：
1. 数组类文件（.npy）
- `A_hat.npy`：词-主题矩阵（p×K），每列对应一个主题的词分布（列和为1）
- `W_hat_all.npy`：所有社区的主题权重矩阵（K×n_all），每列对应一个社区的主题分布
- `W_hat_fit.npy`：仅拟合用社区的主题权重矩阵（K×n_fit）
- `community_ids_all.npy`：所有有效社区的ID数组（对应 D_all 列）
- `community_ids_fit.npy`：仅拟合用社区的ID数组（对应 D_fit 列）
- `M_trunk.npy`：截断后的词平均频次向量（Topic-SCORE 算法输入归一化用）

2. 词表文件
- `vocab.txt`：最终保留的词表（过滤低频词后），每行一个词，顺序对应 A_hat 行索引

3. 稀疏矩阵文件（.npz）
- `D_all.npz`：词×所有有效社区的稀疏矩阵（csr格式），存储词在各社区的加权频次
- `D_fit.npz`：词×仅拟合用社区的稀疏矩阵（csr格式），用于 Topic-SCORE 模型拟合

4. 分析用 CSV 文件
- `topics_top_words.csv`：各主题的核心词表，包含主题ID、Top N核心词、词权重
- `communities_topic_weights.csv`：各社区的主题权重表，包含社区ID、各主题权重、Top1主题
- `topic_representative_communities.csv`：各主题的代表性社区表，按主题权重排序

5. 元信息文件
- `topic_model_meta.json`：单分辨率建模元信息，包含：
  - 输入参数（如 title_weight、vh_method、words_percent）
  - 数据规模（n_communities_fit、n_vocab、runtime_sec）
  - 算法细节（vh_method_actual、svs_m、svs_k0 等顶点搜索参数）

#### 多分辨率根目录汇总输出
由 `topic_modeling_multi.py` 生成，用于批量管理和对比所有分辨率结果：
- `summary_multires.csv`：多分辨率汇总表，一行对应一个分辨率，包含：
  - 基础信息：分辨率值、输出目录、运行状态（success/skipped_existing/error）
  - 主题质量指标：mean_top1_weight（Top1主题权重均值）、mean_top1_minus_top2（Top1-Top2权重差）
  - 建模元数据：n_communities_fit、n_vocab、各主题核心词（仅 status=ok 行包含）
  - 算法参数：vh_method_actual、vh_m、vh_k0（仅 status=ok 行包含）
- `run_meta.json`：批处理全局元信息，包含：
  - 输入参数（k、r_min/r_max、skip_existing 等）
  - 运行统计（总耗时、成功分辨率数、失败分辨率数）
  - 输出路径、随机种子等环境信息
- `errors.json`：错误日志（如有），记录失败分辨率、错误类型、堆栈信息，便于排查问题


脚本：
- `src/topic_modeling.py`
- `src/topic_modeling_multi.py`

---

### B. Topic 对齐
输入：
- `out/topic_modeling_multi/K{K}/r*/...`

输出：
- `out/topic_modeling_multi/K{K}/aligned_segmented/segment_00_anchor_r0.0057/...`（topic 编号已对齐）
- `topic_alignment.csv`
- `topic_alignment_summary.csv`

脚本：
- `src/align_topics_multires_segmented.py`

---


### C. 可视化（UMAP / graph + 质心图）
输入：
- `out/umap2d.npy`
- `out/graph_drl2d.npy`（可选）
- `out/leiden_sweep/membership_r*.npy`

输出：
- `frames_topic_umap/`
- `frames_topic_graph/`
- `frames_comm_centroid_umap/`
- 曲线图与图例

脚本：
- `src/topic_visualization_multires.py`

---

### D. 合成 MP4（动画）
输入：
- `frames_*` 图片帧目录

输出：
- `*.mp4`

脚本：
- `src/frames_to_mp4.py`

---


### E. 数据分析
输入：
- `out/topic_modeling_multi/K10/aligned_segmented `

输出：
- `_diagnostics_topic/`

脚本：
- `diagnose_topic_collapse.py`

---


## 二、A中的主题建模步骤概括（与你项目对齐）

1. **固定全局主题数 K**
   - 所有社区共享这 `K` 个全局主题（例如 `K=10`）

2. **按社区聚合文本**
   - 将同一社区内论文的标题、摘要、作者聚合成“社区文档”

3. **构造词 × 社区矩阵**
   - 清洗、停用词过滤、词频/文档覆盖筛选
   - 得到稀疏矩阵 `D`

4. **谱方法 Topic-SCORE**
   - 归一化、SVD、SCORE 比值变换、顶点搜索、恢复 `A_hat`

5. **估计社区主题权重**
   - 对每个社区求 `K` 维主题权重向量（`W_hat_all`）
   - 给社区打 Top1/Top2 主题

6. **解释与展示**
   - 每个 topic 的关键词（`topics_top_words.csv`）
   - 社区主题权重表（`communities_topic_weights.csv`）
   - 代表社区与代表论文

---

## 三、B. Topic 对齐输出topic_alignment_summary.csv中的重要指标`avg_similarity` / `min_similarity` 解读

在 `align_topics_multires.py` 中，对每个目标分辨率 `r`：

1. 用参考分辨率（如 `r=1.0`）和目标分辨率的 `A_hat` 构造 topic-topic 相似度矩阵
2. 用 Hungarian matching 做一对一最优匹配
3. 对 **K 对匹配 topic** 的相似度统计：
   - `avg_similarity`：这 K 对的平均相似度
   - `min_similarity`：这 K 对里最小的相似度（最不稳定的一对）


### 如何理解“分辨率 < 0.2213 时 avg_similarity 很低”
这通常说明：**低分辨率下的社区划分过粗，导致主题结构与参考分辨率（如 1.0）差异明显**。常见原因：

- 社区数量太少，多个语义子领域被合并到一个大社区中
- 聚合文本过于混杂，Topic-SCORE 学到的主题更“粗粒度”
- 与参考分辨率相比，主题发生了显著合并/重组

这不一定是“坏结果”，而是说明：

- **结构层面**：低分辨率更适合看大块结构
- **语义层面**：若要做“稳定可比”的主题展示，建议重点看 `avg_similarity` 较高的分辨率段

### 经验阈值（以 cosine 相似度为例）
- `>= 0.80`：很高，topic 语义较稳定
- `0.65 ~ 0.80`：可用，有一定漂移
- `0.50 ~ 0.65`：偏低，解释演化需谨慎
- `< 0.50`：较低，说明主题体系与参考差异明显

> 实际使用建议：**同时看 `avg_similarity` 和 `min_similarity`**，并抽查 top words。

保留了单一参考分辨率的部分，`avg_similarity` 和 `min_similarity`很低，目的是作为多参考分辨率的参考。
---


## 四、主要产出文件与作用（简表）

### 主题建模（每个分辨率目录）
- `A_hat.npy`：词-主题矩阵（解释 topic 关键词）
- `W_hat_all.npy`：主题-社区矩阵（社区主题权重）
- `topics_top_words.csv`：每个 topic 的关键词
- `communities_topic_weights.csv`：每个社区的 topic 权重、Top1/Top2
- `topic_representative_communities.csv`：每个 topic 的代表社区

### 对齐
- `topic_alignment.csv`：每个分辨率下 `target_topic -> ref_topic` 映射 + 相似度
- `topic_alignment_summary.csv`：每个分辨率的 `avg_similarity` / `min_similarity`

### 可视化
- `frames_topic_umap/`：论文点图（按 topic 染色）
- `frames_comm_centroid_umap/`：社区质心图（颜色=topic，大小=社区规模）
- 曲线图：社区数、纯度、topic 占比随分辨率变化

### 数据分析
- `topic_collapse_diagnostics.csv`：有效 topic 数、top1 主导程度、权重归一化检查、topic 词表重叠等指标
- `Effective topics vs resolution/`：分别给出 Shannon 与 Simpson 有效主题数
- `Topic-row normalization check vs resolution/`：给出社区级 topic 权重行归一化误差 `max |row_sum-1|`
- `Top1 dominance and sharpness vs resolution/`：同时给出“主导 topic 的 top1 占比（按社区数）”与“社区级 `mean_top1_weight`”
- 还有一些其他图表
---

## 五、运行指令（Windows PowerShell）

> 下列命令假设你在项目根目录执行。根据`topic_collapse_diagnostics.csv`中的有效topic数，最终稳定在维持在较高水平（约 7.48-9.41），所以保留了K=10的选择。

### 1）多分辨率主题建模（扫描 `0.0001 <= r <= 5` 内已有 membership 文件）
近似版
```powershell
python src/topic_modeling_multi.py `
  --k 10 `
  --r-min 0.0001 `
  --r-max 5 `
  --rep-papers-mode approx `
  --continue-on-error
```
精确版
```powershell
python src/topic_modeling_multi.py `
  --k 10 `
  --r-min 0.1 `
  --r-max 1.5 `
  --rep-papers-mode exact `
  --continue-on-error
```
临时关闭中心性
```powershell
python src/topic_modeling_multi.py `
  --k 10 `
  --r-min 0.0001 `
  --r-max 5 `
  --rep-papers-mode off `
  --continue-on-error
```
单个分辨率
```powershell
python src/topic_modeling.py `
  --k 10 `
  --resolution 1.0 `
  --rep-papers-mode approx #可根据需要修改成--rep-papers-mode exact或--rep-papers-mode off
```

### 2）分段 Topic 对齐
```powershell
python src/align_topics_multires_segmented.py `
  --k 10 `
  --r-min 0.0001 `
  --r-max 5 `
  --metric cosine `
  --break-avg-thresh 0.8 `
  --break-min-thresh 0.05 `
  --anchor-strategy best-adjacent `
  --save-adjacent-topic-rows
```

### 3）主题自动命名（基于单一分辨率对齐，暂时跳过这一步，采用人工命名）
```powershell
python src/topic_labeling.py `
  --k 10 `
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 `
  --alignment-dir out/topic_modeling_multi/K10/aligned_to_r1.0000 `
  --ref-resolution 1.0 `
  --r-min 0.0001 `
  --r-max 5 `
  --write-labeled-community-tables
```

### 4）可视化
```powershell
python src/topic_visualization_multires.py `
  --k 10 `
  --topic-root out/topic_modeling_multi/K10 `
  --batch-segments `
  --segments-csv out/topic_modeling_multi/K10/aligned_segmented/segments.csv `
  --segmented-out-root out/topic_viz_multi/K10_segmented `
  --community-centroid `
  --skip-graph ` #完整版去掉这一行
  --max-points 60000 ` #完整版去掉这一行
  --clean-segment-out
```


### 5）合成 MP4
```powershell
Get-ChildItem out/topic_viz_multi/K10_segmented -Directory | ForEach-Object {
  $fps = 8
  if ($_.Name -match "segment_02") { $fps = 2 }  # 可选：短段放慢
  python src/frames_to_mp4.py `
    --batch-subdirs `
    --root $_.FullName `
    --subdir-glob "frames*" `
    --fps $fps
}
```

### 6）数据分析
分析单个分辨率
```powershell
python src/diagnose_topic_collapse.py `
  --input out/topic_modeling_multi/K10/aligned_segmented/segment_00_anchor_r0.0057/r0.0057/communities_topic_weights.csv `
  --save-per-topic
```

批量分析整个分段对齐目录（推荐）
```powershell
python src/diagnose_topic_collapse.py `
  --root out/topic_modeling_multi/K10/aligned_segmented `
  --save-per-topic `
  --save-plots `
  --verbose
```
