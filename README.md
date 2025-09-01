# paper-community

<p align="center">
  <a href="https://github.com/Gimmnm/paper-community">
    <img src="web/public/pic/logo.svg" alt="Paper Community" height="96">
  </a>
</p>

<h1 align="center">Paper Community</h1>
<p align="center">
  基于开放数据的论文社区可视化
</p>

## 项目介绍

面对日益增长的学术论文数据，科研人员亟需高效、智能的工具以快速定位核心文献并捕捉前沿研究动态。此项目在已有论文的文本深度表示[paper2vector](https://github.com/Gimmnm/paper2vector)后，利用社区发现等算法，提供论文的深层主题分类实现，并且实现页面应用，提升科研信息检索效率与准确性。

## 项目流程

1. 建图
2. 应用算法；我们有三个方案，分别是louvain、infomap、图神经网络+聚类算法
3. 初步可视化和结果处理分析
4. 前端可视化页面 整个思路是做一个网络 用颜色区分社区 做一些动态效果 点开每个节点进入论文具体信息页面 具体信息页面大概是一个表格 提供信息 并且给一个arxiv的连接 当然 网站有基本的总页面、论文列表页面、搜索页面（通过标签、论文名、社区名（暂时只能给个编号，或许可以内接ai的api来自动生成社区名称）、作者等等来进行搜索）等页面
5. 配合前端的后端数据库 存储所有需要的数据和社区等等信息
6. 测试方案
7. 后续可能实现 动态在线加入论文、分层社区（论文分类逐层递进）的功能

## 项目规划结构

```bash
paper-community/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ configs/
│  └─ default.yaml             # 运行参数
├─ data/                       # 输入/输出数据(建议 .gitignore)
│  ├─ papers.csv               # 论文元数据(id,title,authors,abstract,year,arxiv_id,categories,...)
│  ├─ embeddings.npy           # 论文向量 (N, D)
│  └─ graph/                   # 中间产物: nodes.csv, edges.csv, graph.gexf, graph.json
├─ scripts/
│  ├─ build_graph.py           # (1) 建图：mutual kNN / 阈值
│  ├─ run_community.py         # (2) Louvain / Infomap / Leiden
│  ├─ gnn_train.py             # (2) GNN + 聚类（可选）
│  ├─ analyze_results.py       # (3) 评估+命名（TF-IDF）+导出 JSON
│  ├─ export_layout.py         # (3) 预计算力导向布局/UMAP
│  ├─ load_db.py               # (5) 把 nodes/edges/communities 导入数据库
│  └─ demo_all.sh              # 一键串起来
├─ pcore/                      # Python 核心库（graph/社区/IO/评估）
│  ├─ __init__.py
│  ├─ graph.py
│  ├─ community.py
│  ├─ layout.py
│  ├─ metrics.py
├─ backend/
│  ├─ app/
│  │  ├─ main.py                # FastAPI 入口，读CSV并暴露API
│  │  ├─ loaders.py             # 读 nodes/edges/layout 并缓存
│  │  └─ schemas.py             # Pydantic 模型（可选）
│  └─ requirements.txt          # 后端依赖
├─ web/                         # React 前端（Vite）
│  ├─ index.html
│  ├─ package.json
│  ├─ vite.config.js
│  └─ src/
│     ├─ main.jsx
│     ├─ App.jsx
│     └─ components/
│        └─ GraphView.jsx 
```

### 数据准备

- data/papers.csv
至少包含列：id,title,authors,abstract
可选：year, arxiv_id, is_AP, is_NA, categories
- 如果有 is_AP/is_NA，评估时会自动用它们派生四分类（AP-only/NA-only/AP+NA/Other）。
- data/embeddings.npy
形状 (N, D) 的 NumPy 数组，N 必须与 CSV 行数一致，顺序需一一对应（通常 title + abstract 编码的向量）。

## 项目运行

```bash
# 建环境（首次）
conda env create -f environment.yml

# 激活
conda activate paper-community

conda activate paper-community
```

### 建图

```bash
# 互为 k 近邻（推荐）
python scripts/build_graph.py --csv data/papers.csv --emb data/embeddings.npy --mode mutual-knn --k 10 --tau 0.30 --outdir data/graph
```

### 社区发现算法（三选一）

```bash
# Louvain（快、稳）
python scripts/run_community.py --graph data/graph/graph.gexf --algo louvain --resolution 2.0

# Leiden（更稳定、更细）
python scripts/run_community.py --graph data/graph/graph.gexf --algo leiden  --resolution 2.0cd backend

# Infomap（基于信息流，常更细）
python scripts/run_community.py --graph data/graph/graph.gexf --algo infomap
```

### 计算布局坐标（可视化的前置）

```bash
# 在 paper-community-fa2 环境里执行
python scripts/export_layout.py --graph data/graph/graph.gexf \
  --layout fa2 --iters 300 --step 25 --init spectral --progress
  
```

### 分析与导出结果（评估、命名、JSON）

```bash
python scripts/analyze_results.py \
  --papers data/papers.csv \
  --communities data/graph/communities.csv \
  --graph data/graph/graph.gexf \
  --outdir outputs/analysis

```

### 前端运行

```bash

conda install -c conda-forge nodejs=20

cd web
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --registry=https://registry.npmjs.org

npm run dev

```

## 项目结果

![Network Graph](/web/public/pic/network.png)

![Network Graph colored by four catogories](/web/public/pic/network_four.png)

![List Page](/web/public/pic/list_page.png)

![Detailed Page](/web/public/pic/detail_page.png)
