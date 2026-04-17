# 粗分领域（聚类）→ 社区发现（分层）：实验记录

本文记录一个“论文地图”导向的折中方案：**先把全体论文粗分成少数几个大领域**（便于地图叙事与落地展示），再在每个领域内部用 mutual‑kNN + Leiden/CPM 做社区发现与分层诊断。

这不替代全图社区发现；它是一个**展示/解释友好**的两阶段结构：

- 第 0 层（粗）：固定为 \(K=3\)（或其它小 K）的“领域”
- 第 1+ 层（细）：每个领域内部的多分辨率社区与层级连边

---

## 1. 为什么这样做（与你的初衷如何兼容）

你最初用社区发现希望“结构来自图”，而不是人为聚类。两阶段法的关键是：

- **粗层**只负责给地图一个“大陆划分”（3 个大区域、好解释、好导航）
- **细层**仍然由图上的 Leiden/CPM 社区发现产生（领域内部的分裂/层级仍是 network-driven）

当全图 CPM 在很低 \(r\) 依然产生大量社区（例如 \(r=0.001\) 时仍有上千个社区）时，直接从社区发现拿到“3 大领域”会很困难；此时两阶段法能把叙事层与结构层分开。

---

## 2. 本仓库里我跑过的实验（不会覆盖已有输出）

### 2.1 粗分：embedding 上 k-means=3

脚本：`src/coarse_domains_kmeans.py`

命令（示例）：

```bash
python src/coarse_domains_kmeans.py \
  --k 3 --seed 42 \
  --out-dir out/coarse_domains_kmeans_k3_seed42 \
  --plot
```

产物：

- `out/coarse_domains_kmeans_k3_seed42/umap_kmeans_k3.png`：UMAP 上按领域上色
- `labels.npy`：每篇论文的领域标签（0/1/2）
- `domain_{i}_vertex_indices.npy`：每个领域的全局顶点下标（与 `embs[1:]` 行一致）

### 2.2 细分：每个领域内做 induced subgraph + CPM hierarchy

新增 CLI：`python src/core.py vertexset-hierarchy ...`

对领域 i（0/1/2）：

```bash
python src/core.py vertexset-hierarchy \
  --vertex-indices out/coarse_domains_kmeans_k3_seed42/domain_i_vertex_indices.npy \
  --out-dir out/coarse_domains_kmeans_k3_seed42/domain_i_hierarchy_cpm \
  --partition-type CPMVertexPartition \
  --r-min 0.001 --r-max 0.15 --step 0.002 \
  --min-child-share 0.10 \
  --k 50
```

然后可视化（早期分化聚焦）：

```bash
python src/hierarchy_viz.py \
  --hierarchy-dir out/coarse_domains_kmeans_k3_seed42/domain_i_hierarchy_cpm \
  --out-prefix domain_i_cpm \
  --weight-by child_share --min-edge-value 0.10 \
  --max-nodes-per-layer 35 --y-mode layer_index \
  --r-max 0.03 --breakpoints-top-k 10
```

---

## 3. 输出在哪里

所有输出都在一个新目录下（不覆盖 `out/leiden_sweep_cpm/` 等已有结果）：

- `out/coarse_domains_kmeans_k3_seed42/`
  - `umap_kmeans_k3.png`
  - `domain_0_hierarchy_cpm/`
  - `domain_1_hierarchy_cpm/`
  - `domain_2_hierarchy_cpm/`

每个 `domain_*_hierarchy_cpm/` 内部包含：`hierarchy_nodes.csv`、`hierarchy_edges.csv`、`breakpoints.json`、`membership_r*.npy`、`meta.json`、以及 `domain_*_cpm_layered.png` 等图。

---

## 4. 注意事项

- 这是“论文地图”导向的折中：**粗层的 3 大领域不是社区发现直接给的**，而是聚类给的。
- 若你希望粗层也来自网络结构，可以尝试：
  - 更小的 CPM 分辨率区间（比如扫到 \(1e{-5}\)）看看能否降到极少社区
  - 或更全局的图构造（更大 k / 非 mutual / 加弱边）

