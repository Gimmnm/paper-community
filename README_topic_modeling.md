# 多分辨率主题建模、Topic 对齐、自动命名、可视化与前端打包（中文说明）

本文档基于当前项目流程，说明如何完成：

1. 多分辨率社区结果的主题建模（Topic-SCORE）
2. 多分辨率 topic 对齐（保证颜色/编号跨分辨率尽量稳定）
3. 主题自动命名（中文候选，减少人工拼词）
4. 2D 可视化与动画（MP4）
5. 打包为前端可直接使用的 JSON

---

## 一、整体流程（建议顺序）

### A. 多分辨率主题建模（按目录里已有 membership 文件扫描）
输入：
- `out/leiden_sweep/membership_r*.npy`
- `data/data_store.pkl`
- `data/stopwords.txt`

输出（每个分辨率一个目录）：
- `A_hat.npy` / `W_hat_all.npy`
- `topics_top_words.csv`
- `communities_topic_weights.csv`
- `topic_representative_communities.csv`
- `topic_model_meta.json`

脚本：
- `src/topic_modeling_multi.py`

---

### B. Topic 对齐（以参考分辨率，如 `r=1.0`）
输入：
- `out/topic_modeling_multi/K{K}/r*/...`

输出：
- `out/topic_modeling_multi/K{K}/aligned_to_r1.0000/r*/...`（topic 编号已对齐）
- `topic_alignment.csv`
- `topic_alignment_summary.csv`

脚本：
- `src/align_topics_multires.py`

---

### C. 主题自动命名（中文候选）
输入：
- 对齐后的 `topics_top_words.csv`
- `communities_topic_weights.csv`
- `topic_representative_communities.csv`

输出：
- `topic_labels_reference.csv`
- `topic_labels_reference_candidates.csv`
- `topic_labels_by_resolution.csv/json`
- （可选）`communities_topic_weights_labeled.csv`

脚本：
- `src/topic_labeling.py`

---

### D. 可视化（UMAP / graph + 质心图）
输入：
- `out/umap2d.npy`
- `out/graph_drl2d.npy`（可选）
- `out/leiden_sweep/membership_r*.npy`
- 主题结果目录（建议用对齐后的目录）

输出：
- `frames_topic_umap/`
- `frames_topic_graph/`
- `frames_comm_centroid_umap/`
- 曲线图与图例

脚本：
- `src/topic_visualization_multires.py`

---

### E. 合成 MP4（动画）
输入：
- `frames_*` 图片帧目录

输出：
- `*.mp4`

脚本：
- `src/frames_to_mp4.py`

---

### F. 打包前端 JSON（推荐）
输入：
- `membership_r*.npy`
- 对齐后的主题结果目录（建议）
- 命名结果目录（labels）
- `umap2d.npy` / `graph_drl2d.npy`

输出：
- `out/web_payload/K{K}_ref{ref}/index.json`
- `out/web_payload/K{K}_ref{ref}/resolutions/r{res}.json`

脚本：
- `src/export_web_payload.py`

---

## 二、主题建模步骤概括（与你项目对齐）

你们的主题建模是建立在 Leiden 社区结果之上的：

1. **固定全局主题数 K**
   - 所有社区共享这 `K` 个全局主题（例如 `K=10`）

2. **按社区聚合文本**
   - 将同一社区内论文的标题、摘要（可含作者）聚合成“社区文档”

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

## 三、`avg_similarity` / `min_similarity` 是什么？怎么解读？

在 `align_topics_multires.py` 中，对每个目标分辨率 `r`：

1. 用参考分辨率（如 `r=1.0`）和目标分辨率的 `A_hat` 构造 topic-topic 相似度矩阵
2. 用 Hungarian matching 做一对一最优匹配
3. 对 **K 对匹配 topic** 的相似度统计：
   - `avg_similarity`：这 K 对的平均相似度
   - `min_similarity`：这 K 对里最小的相似度（最不稳定的一对）

> 注意：`avg_similarity` 不是所有 topic 两两组合的平均，而是**最优匹配后的 K 对**平均。

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

---

## 四、“规则命中”是什么意思？（topic 自动命名）

`topic_labeling.py` 使用“规则 + 短语候选 + 回退”的半自动命名方案：

### 1）规则命中（rule hit）
脚本内置了一组统计学领域规则（例如）：

- `{survival} + {censored/hazard/failure}` → 生存分析与删失数据
- `{clinical} + {trial/trials}` → 临床试验设计与剂量反应
- `{time} + {series/autoregressive}` → 时间序列与自回归建模

当某个 topic 的 `top_words` / 代表社区标题短语满足规则条件时，就认为“规则命中”。

### 2）为什么以前首候选置信度常常都是 0.92？
旧版本脚本对规则命中使用了固定置信度 `0.92`（用于排序），这会导致看起来所有 rule 命中的 topic 置信度都一样。

### 3）当前版本已改为动态置信度（启发式）
新版 `topic_labeling.py` 已将规则命中置信度改成动态分数，会融合：

- 规则组覆盖强度（命中组内词汇丰富度）
- `top_words` 对规则词汇覆盖率
- 标题短语证据覆盖率
- 规则词在 top words 中的位置（越靠前加分越高）
- 是否出现典型短语词典证据（如 `time series`, `survival analysis`）

因此：
- 不同 topic 的 rule 命中置信度不再都相同
- 分数更适合排序与人工审核优先级

> 置信度是启发式“可用度分数”，**不是概率**。

### 4）置信度如何看（经验）
- `>= 0.85`：高（通常可直接用，人工快速确认）
- `0.70 ~ 0.85`：中高（建议看一眼证据短语）
- `0.55 ~ 0.70`：一般（建议人工确认）
- `< 0.55`：偏低（多为回退候选）

---

## 五、为什么某些 topic 只有一个候选名？

常见原因（正常现象）：

1. 规则命中后已得到高质量标签
2. 代表社区标题中抽不出稳定短语（或短语过泛）
3. 候选去重后只剩一个
4. 该 topic 本身术语较泛（例如偏“检验/分布”）

新版 `topic_labeling.py` 已增加一个改进：
- 在仅有单一 rule 候选时，会尽量补一个 `top_words_fallback` 作为备选（如果不重复）

这样更方便人工比较选择。

---

## 六、主要产出文件与作用（简表）

### 主题建模（每个分辨率目录）
- `A_hat.npy`：词-主题矩阵（解释 topic 关键词）
- `W_hat_all.npy`：主题-社区矩阵（社区主题权重）
- `topics_top_words.csv`：每个 topic 的关键词
- `communities_topic_weights.csv`：每个社区的 topic 权重、Top1/Top2
- `topic_representative_communities.csv`：每个 topic 的代表社区

### 对齐
- `topic_alignment.csv`：每个分辨率下 `target_topic -> ref_topic` 映射 + 相似度
- `topic_alignment_summary.csv`：每个分辨率的 `avg_similarity` / `min_similarity`

### 自动命名
- `topic_labels_reference.csv`：参考分辨率最终主题名（主标签）
- `topic_labels_reference_candidates.csv`：候选主题名（含 method/confidence）
- `topic_labels_by_resolution.csv/json`：各分辨率 topic 标签映射
- `communities_topic_weights_labeled.csv`（可选）：社区表增加 `top1_label_zh` 等字段

### 可视化
- `frames_topic_umap/`：论文点图（按 topic 染色）
- `frames_comm_centroid_umap/`：社区质心图（颜色=topic，大小=社区规模）
- 曲线图：社区数、纯度、topic 占比随分辨率变化

### 前端打包 JSON（新增）
- `index.json`：分辨率清单、路径、基础元信息
- `resolutions/r*.json`：某分辨率下的 topics / communities / （可选）paper_points

---

## 七、运行指令（Windows PowerShell）

> 下列命令假设你在项目根目录执行。

### 1）多分辨率主题建模（扫描 `0.0001 <= r <= 5` 内已有 membership 文件）
```powershell
python src/topic_modeling_multi.py `
  --k 10 `
  --r-min 0.0001 `
  --r-max 5 `
  --skip-existing `
  --continue-on-error
```

### 2）Topic 对齐（以 `r=1.0` 为参考）
```powershell
python src/align_topics_multires.py `
  --k 10 `
  --ref-resolution 1.0 `
  --r-min 0.0001 `
  --r-max 5 `
  --metric cosine `
  --save-sim-matrix
```

### 3）主题自动命名（基于对齐结果）
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

### 4）可视化（建议读取对齐后的目录）
```powershell
python src/topic_visualization_multires.py `
  --k 10 `
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 `
  --r-min 0.0001 `
  --r-max 5 `
  --community-centroid
```

### 5）合成 MP4（单个帧目录）
```powershell
python src/frames_to_mp4.py `
  --frame-dir out/topic_viz_multi/K10/frames_topic_umap `
  --out out/topic_viz_multi/K10/topic_umap_evolution.mp4 `
  --fps 8
```

### 6）合成 MP4（批量 `frames*` 子目录）
```powershell
python src/frames_to_mp4.py `
  --batch-subdirs `
  --root out/topic_viz_multi/K10 `
  --subdir-glob "frames*" `
  --fps 8
```

### 7）导出前端 JSON（新增）
```powershell
python src/export_web_payload.py `
  --k 10 `
  --ref-resolution 1.0 `
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 `
  --labels-dir out/topic_modeling_multi/K10/aligned_to_r1.0000/labels_ref_r1.0000 `
  --r-min 0.0001 `
  --r-max 5
```

> 若前端需要论文点（会显著增大 JSON 体积）：
```powershell
python src/export_web_payload.py `
  --k 10 `
  --ref-resolution 1.0 `
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 `
  --labels-dir out/topic_modeling_multi/K10/aligned_to_r1.0000/labels_ref_r1.0000 `
  --r-min 0.0001 `
  --r-max 5 `
  --include-paper-points `
  --paper-sample 20000
```

---

## 八、运行指令（Linux / macOS Bash）

### 1）多分辨率主题建模
```bash
python src/topic_modeling_multi.py \
  --k 10 \
  --r-min 0.0001 \
  --r-max 5 \
  --skip-existing \
  --continue-on-error
```

### 2）Topic 对齐
```bash
python src/align_topics_multires.py \
  --k 10 \
  --ref-resolution 1.0 \
  --r-min 0.0001 \
  --r-max 5 \
  --metric cosine \
  --save-sim-matrix
```

### 3）主题自动命名
```bash
python src/topic_labeling.py \
  --k 10 \
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 \
  --alignment-dir out/topic_modeling_multi/K10/aligned_to_r1.0000 \
  --ref-resolution 1.0 \
  --r-min 0.0001 \
  --r-max 5 \
  --write-labeled-community-tables
```

### 4）可视化
```bash
python src/topic_visualization_multires.py \
  --k 10 \
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 \
  --r-min 0.0001 \
  --r-max 5 \
  --community-centroid
```

### 5）合成 MP4
```bash
python src/frames_to_mp4.py \
  --frame-dir out/topic_viz_multi/K10/frames_topic_umap \
  --out out/topic_viz_multi/K10/topic_umap_evolution.mp4 \
  --fps 8
```

### 6）导出前端 JSON（新增）
```bash
python src/export_web_payload.py \
  --k 10 \
  --ref-resolution 1.0 \
  --topic-root out/topic_modeling_multi/K10/aligned_to_r1.0000 \
  --labels-dir out/topic_modeling_multi/K10/aligned_to_r1.0000/labels_ref_r1.0000 \
  --r-min 0.0001 \
  --r-max 5
```

---

## 九、前端展示建议（当前数据结构即可支持）

建议网页交互：

- **分辨率选择器**（slider / dropdown）
- **Topic 筛选**
- **显示模式**：
  - 社区质心图（推荐默认）
  - 论文点图（可选 / 抽样）
- **点击社区详情**：
  - `community_id`
  - 社区规模
  - `Top1/Top2` topic（中文名 + 权重）
  - 关键词 / 代表论文标题

`export_web_payload.py` 导出的 JSON 已包含这些字段的基础版本。

---

## 十、注意事项与排查建议

1. **未对齐的 topic 颜色不代表跨分辨率语义一致**
   - 做动画时若想讲“语义演化”，请使用对齐后的目录

2. **低分辨率 `avg_similarity` 低是常见现象**
   - 通常意味着主题更粗、与参考分辨率差异大
   - 不代表结果错误，但语义对齐解释要谨慎

3. **命名置信度不是概率**
   - 主要用于排序（先看高分项）
   - 建议人工确认参考分辨率的 `K` 个主题标签

4. **JSON 体积控制**
   - 默认只导出社区级信息（轻量）
   - 论文点建议用 `--paper-sample` 抽样；需要全量时再导出

5. **OpenMP warning（Intel/LLVM OpenMP 同时加载）**
   - 若能跑完可暂时忽略
   - 若出现卡死/崩溃，再考虑统一环境或限制线程数

---

## 十一、建议下一步（可选增强）

- `topic_labeling.py` 增加领域词典与规则（提升中文命名质量）
- `export_web_payload.py` 增加社区“跨分辨率 lineage”映射（展示分裂/合并）
- 增加 `sankey/alluvial` 数据导出，展示 topic 演化路径
