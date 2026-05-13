# Source Layout (Refactor Stage)

Code under `src/` is organized by layer. **`src/core.py` is the only Python module at `src/` root** (plus `src/tools/` utilities). 产品目标与分层陈述见 [`../docs/requirements-breakdown.md`](../docs/requirements-breakdown.md)；日常以 [`../docs/developer_manual_zh.md`](../docs/developer_manual_zh.md) 为准。

## Layers

- `data_layer/` — experiment manifests/contracts and run catalog discovery.
- `foundation_layer/` — ingest (`getdata`), models, paths, network/graph build, embeddings, 2D layouts.
- `algorithm_layer/` — community detection sweeps (`community`), unified pipeline (`algorithm_pipeline`), time windows (`time_window`), optional `coarse_domains_kmeans`.
- `analysis_layer/` — evaluation metrics, hierarchy index, checklists, Topic-SCORE pipeline and related scripts.
- `app_layer/` — demo FastAPI app (`demo_api_app`), search/graph helpers (`demo_search`, `demo_graph`), keyword retrieval (`retrieval`).

## Entry points

- **CLI (recommended):** from repo root, `PYTHONPATH=src python src/core.py <task> …`
- **Direct module runs** (same env): `PYTHONPATH=src python -m analysis_layer.topic_modeling`, `-m analysis_layer.topic_modeling_multi`, etc.
- **Coarse domains only:** `PYTHONPATH=src python -m algorithm_layer.coarse_domains_kmeans --help`

`core.py` delegates some topic jobs to subprocess calls against `analysis_layer/*.py` to keep heavy optional deps isolated.

## Maintainer tools

- `tools/archive_out_candidates.py` — legacy `out/` hygiene helpers.

## Scripts（仓库根目录 `scripts/`）

- `daily_workflow.sh` — 固定日报式流水线（manifest + eval；`--viz` 刷新示意图）。
- `plot_umap_membership.py` — 仅依赖已有 `umap2d.npy` 与 `membership_r*.npy` 导出着色散点图。
