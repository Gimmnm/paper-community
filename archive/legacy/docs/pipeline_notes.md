# Pipeline Notes (Archived)

This file preserves the old `docs/pipeline` notes so active docs can stay focused on the current refactor implementation.

## What was archived

- `docs/pipeline/README.md`
- `docs/pipeline/data_pipeline.md`

## Kept context (short version)

- Raw data parsing and cache path were centered on:
  - `src/getdata.py`
  - `src/model.py`
  - `src/project_paths.py`
- Main offline outputs in `out/` were produced by:
  - embeddings: `src/embedding.py`
  - 2D coords: `src/diagram2d.py`
  - graph: `src/network.py`
  - community sweep: `src/community.py`
  - topic modeling: `src/topic_modeling.py`, `src/topic_modeling_multi.py`
- API/demo layer depended on:
  - `src/demo_graph.py`
  - `src/demo_search.py`
  - `src/demo_api_app.py`

## Why archived

During the current refactor, these notes overlap with active contracts in:

- `docs/architecture-boundaries.md`
- `docs/algorithm-pipeline-spec.md`
- `docs/evaluation-metrics-spec.md`

Active docs now keep only implementation-facing and currently maintained content.
