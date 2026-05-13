# Architecture Boundaries

Product-level goals (three parts, seven metrics, six UI rules) are summarized in [`requirements-breakdown.md`](requirements-breakdown.md). This file maps **Python modules** to those parts.

## Target layers

1. Data/artifact layer
2. Algorithm pipeline layer
3. API/Web layer

Layers hold **libraries and scripts with clear boundaries**. **`src/core.py` stays on purpose** as the single CLI orchestrator (`python src/core.py <task>`): it composes those layers, registers argparse subcommands, runs long pipelines, and delegates to standalone scripts (for example topic batch jobs) via subprocess when that keeps dependencies isolated. Splitting every task into a separate top-level binary would duplicate glue without shrinking real logic.

## Current implementation mapping

### CLI orchestration

- `src/core.py` — primary entrypoint; imports from layer packages only; must remain thin relative to algorithm/analysis implementations (no second copy of math-heavy code).

### Data/Artifact layer

- `src/foundation_layer/project_paths.py`
- `src/foundation_layer/getdata.py`
- `src/foundation_layer/model.py`
- `src/foundation_layer/network.py`
- `src/foundation_layer/embedding.py`
- `src/foundation_layer/diagram2d.py`
- `src/data_layer/experiment_contracts.py`
- `src/data_layer/experiment_registry.py`

Responsibilities:

- Path conventions and run manifests.
- Canonical references to graph/sweep/2D/index artifacts.
- Run catalog discovery from `out/experiments/**/manifest.json`.

### Algorithm pipeline layer

- `src/algorithm_layer/algorithm_pipeline.py`
- `src/algorithm_layer/community.py`
- `src/algorithm_layer/time_window.py`
- `src/algorithm_layer/coarse_domains_kmeans.py`

Responsibilities:

- Build graph object from edge triplets.
- Run standardized community detection sweeps for:
  - Leiden (RB)
  - Leiden CPM
  - Louvain
- Produce comparable summaries (`summary.npy`, evaluation overview rows).
- Optional embedding-space k-means to define coarse “domains” before per-domain sweeps (`vertexset-hierarchy` workflows).

### Analysis layer (offline metrics / topics)

- `src/analysis_layer/evaluation_metrics.py`
- `src/analysis_layer/hierarchy_index.py`
- `src/analysis_layer/checklist.py`
- `src/analysis_layer/topic_modeling.py`
- `src/analysis_layer/topic_modeling_multi.py`
- `src/analysis_layer/align_topics_multires.py`
- `src/analysis_layer/align_topics_multires_segmented.py`
- `src/analysis_layer/topic_visualization_multires.py`
- `src/analysis_layer/diagnose_topic_collapse.py`
- `src/analysis_layer/frames_to_mp4.py`

### API/Web layer

- `src/app_layer/demo_api_app.py`
- `src/app_layer/demo_search.py`
- `src/app_layer/demo_graph.py`
- `src/app_layer/retrieval.py`
- `web/index.html`, `web/app.js`, `web/style.css`

Responsibilities:

- Resolve active run context (`run_id`, `resolution`) at request time.
- Expose stable contracts for frontend selectors and graph/detail views.
- Keep frontend independent from algorithm internals.

Minimal `src/` root:

- Besides `src/tools/` utilities, **`src/core.py` is the only Python module at `src/` root.** Implementations live under `src/*_layer/`.
- Topic batch jobs: `python src/core.py …` subcommands, or `PYTHONPATH=src python -m analysis_layer.<module>` (see each module’s docstring).
- Coarse domains CLI: `PYTHONPATH=src python -m algorithm_layer.coarse_domains_kmeans`.

## Runtime dependency rules

- API/Web may read manifest metadata and produced artifacts.
- Algorithm layer must not depend on `web/`.
- Data layer must not import FastAPI or Cytoscape-facing code.

## Duplicate logic reductions applied

- Shared igraph builder is now centralized via `src/algorithm_layer/algorithm_pipeline.py`.
- Run metadata is standardized via `ExperimentRunManifest`.
