# paper-community

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
├─ pyproject.toml              # 或 requirements.txt（二选一）
├─ requirements.txt
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
│  └─ utils.py
├─ backend/                    # (5) FastAPI + SQLite/Postgres
│  ├─ app.py
│  ├─ db.py
│  ├─ models.py
│  ├─ schemas.py
│  ├─ crud.py
│  ├─ api.py                   # 路由
│  └─ tests/
│     └─ test_api.py
└─ web/                        # (4) React + Vite + TS + Tailwind + Sigma.js(Graphology)
   ├─ index.html
   ├─ package.json
   ├─ vite.config.ts
   └─ src/
      ├─ main.tsx
      ├─ App.tsx
      ├─ api.ts
      ├─ components/
      │  └─ GraphView.tsx
      ├─ pages/
      │  ├─ Home.tsx
      │  ├─ Graph.tsx
      │  ├─ Papers.tsx
      │  ├─ Search.tsx
      │  └─ PaperDetail.tsx
      └─ styles.css
```
