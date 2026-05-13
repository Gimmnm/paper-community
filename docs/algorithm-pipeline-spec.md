# Algorithm Pipeline Spec

## Supported algorithms (phase-1 baseline)

- `leiden` (RB partition)
- `leiden_cpm` (CPM partition)
- `louvain`
- `coarse_kmeans`（k-means 粗分 domain → 各 domain 在全局 mutual-kNN 上的诱导子图跑 `run_community_sweep` → 合并 `membership_r*.npy`；`tags.inner_algorithm` 记录域内所用算法）

## Supported time windows (phase-1 baseline)

- `1y`
- `5y`
- `all`

## Unified run manifest

Each run is registered by `out/experiments/<algorithm>/<time_window>__<run_id>/manifest.json`.

Core fields:

- `run_id`
- `algorithm`
- `time_window`
- `leiden_dir`
- `graph_npz`
- `keyword_index_dir`
- `coords_2d_path`
- `topic_communities_csv`（可选：Topic-SCORE `communities_topic_weights.csv`，Demo 按 manifest 加载）
- `default_resolution`

## Unified sweep entry

Use `src/algorithm_pipeline.py::run_community_sweep`.

- Leiden/RB, Leiden-CPM: sweep over configured resolution range.
- Louvain: single effective run stored as `membership_r1.0000.npy` for contract compatibility.

Two-stage coarse pipeline: `algorithm_pipeline.py::run_coarse_kmeans_then_community_sweep`（CLI：`experiment-coarse-kmeans-sweep`）。

## CLI support

New `core.py` commands:

- `experiment-sweep`
- `experiment-coarse-kmeans-sweep`
- `experiment-manifest`
- `experiment-catalog`
- `experiment-init-minimal`

This allows running algorithms with one interface and publishing the result for API consumption.
