Active outputs under out/ (trimmed for demo + daily_workflow.sh)

Louvain sweep：多分辨率模块度（ModularityVertexPartition），与旧版「单层 igraph multilevel」不同；重跑见 scripts/offline_comparison_master.sh。

  mutual_knn_k*.npz     — 论文相似图缓存
  umap2d.npy            — 2D 坐标（网页布局）
  keyword_index/        — 关键词检索索引
  leiden_sweep_*/       — 各算法的多分辨率社区结果：membership_r*.npy + summary.npy + eval/
  coarse_kmeans_then_cpm_k3_seed42/ — 粗分合并 sweep（同上）
  experiments/          — 登记表 manifest.json（网页切换算法）
  experiment_eval/      — 总览 CSV/JSON、对比断点表
  topic_runs/           — Topic-SCORE 按 sweep 存放（见 docs/offline-outputs-catalog.md）
  topic_viz_batch/      — 主题 UMAP/曲线等图（offline_comparison_master.sh topic-viz）

大块 hierarchy 图、topic_modeling、viz 快照、domain 子 sweep 等已迁至：
  archive/legacy/out/workspace_trim_20260510/
详见该目录 README.txt。
