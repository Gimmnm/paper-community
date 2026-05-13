# Legacy Archive Index

This folder stores documents and modules that are not in the active refactor path.

## Current product focus (what stayed outside archive)

The active system is an **algorithm → paper retrieval** loop with three cooperating parts:

1. **Experiment / artifact layer** — manifests, `out/experiments/**`, catalog APIs  
2. **Algorithm pipeline** — vectorize, graph, Louvain / Leiden / Leiden CPM (+ optional coarse-domain pre-cluster), topics, eval  
3. **Web visualization** — three-column UI, algorithm & time-window & resolution selectors, keyword + community context, vector-neighbor hints in details  

Hierarchy-first UI and long exploratory write-ups live here or under `archive/legacy/docs/`, not in the default docs surface.

## Archived document groups

- Hierarchy-focused notes from old exploration phase
- Early web v2 planning notes
- Pipeline notes moved from `docs/pipeline/`
- Root-level legacy notes:
  - `root/README_topic_modeling.md`
  - `root/文献支架_topic_labeling_与科学计量对照.md`
  - `docs/pipeline_notes.md`

## Archived out artifacts

- Legacy hierarchy/domain outputs moved under `archive/legacy/out/` with date suffix:
  - `coarse_domains_kmeans_k3_seed42__20260505`
  - `leiden_hierarchy_cpm__20260505`
  - `leiden_hierarchy_cpm_cs010__20260505`
  - `leiden_hierarchy_cpm_cs020__20260505`
  - `leiden_hierarchy_rb__20260505`
  - `subgraph_hierarchy__20260505`

## Active docs entry points

- `docs/README.md` — 文档索引（三本核心 + 同目录补充规格）
- `docs/*.md` — 用户手册、开发者手册、目录说明及 `requirements-breakdown` 等规格（与 `README` 同目录）
- `archive/legacy/docs/web_archived/` — 原 `docs/web/` 早期基线英文说明

## Why archived

The refactor target is centered on:

1. Experiment database / artifact layer  
2. Algorithm pipeline layer (multi-algorithm + time windows `1y` / `5y` / `all`)  
3. API + web visualization layer (retrieval-centric, not hierarchy-first)

Authoritative active wording: `docs/README.md` + `docs/developer_manual_zh.md` + `docs/offline-outputs-catalog.md`.  
补充规格（架构、指标、对比流水线等）与上述文件同目录。`archive/legacy/docs/specs/` 仅保留占位说明（规格已迁回 `docs/`）。  
Hierarchy-first exploration documents are intentionally moved out of the active docs surface
to keep the repository focused and easier to navigate.

## Note on code migration

Implementations live under `src/*_layer/`. The `src/` root keeps **`core.py` only** (plus `tools/`); there are no re-export shims. Older notebooks or scripts that did `import community` / `python src/topic_modeling.py` must switch to layer imports or `python -m …` / `python src/core.py …`.
