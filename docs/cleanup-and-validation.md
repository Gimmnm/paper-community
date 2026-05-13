# Cleanup and Validation Guide

This document defines what can be safely removed, what should be archived first,
and what must be kept for reproducibility.

## Scope

- Goal: reduce noise and keep the repository focused on the refactored
  `data -> algorithms -> api/web` flow.
- Constraint: do not delete high-cost artifacts or raw source data without backup.

## Keep / Archive / Remove

### Keep (critical)

- `src/` runtime code (`core.py`, `demo_api_app.py`, `demo_search.py`, `community.py`, `time_window.py`)
- `web/` frontend runtime files
- Primary docs:
  - `README.md`
  - `../web_archived/web-version-1.md`
  - `docs/*.md` 中除三本核心外的补充规格（见 `docs/README.md`）
- Raw inputs and expensive artifacts:
  - `data/*.RData`
  - `data/data_store.pkl`
  - `data/paper_embeddings_specter2.npy`
  - canonical demo artifacts under `out/` used by `demo-api`

### Archive First (experiment history)

- Older or non-canonical experiment outputs under `out/`, especially multiple
  sweep/hierarchy variants that are no longer active.
- Planning documents that overlap current source-of-truth docs (move to
  archive folder before deletion).

### Remove (safe/regenerable noise)

- LaTeX temp files under `docs/slides/`:
  - `*.aux`, `*.log`, `*.nav`, `*.out`, `*.toc`, `*.snm`
- System junk:
  - `.DS_Store`

## Deletion Safety Checks

Run these checks before deleting anything beyond obvious temp files:

1. **Reference check**
   - Confirm the target path is not referenced by docs or scripts.
2. **Runtime check**
   - Verify `demo-api` still starts and key endpoints work (`/api/health`,
     keyword search, paper lookup, community graph).
3. **Rebuild check**
   - Ensure there is a known command to regenerate the artifact.
4. **Cost check**
   - If regeneration requires high compute/time, archive before deletion.
5. **Data safety check**
   - Never delete raw datasets or expensive embeddings without backup.

## Immediate Cleanup Applied

- Removed LaTeX intermediate files from `docs/slides/`:
  - `paper-community-report.aux`
  - `paper-community-report.log`
  - `paper-community-report.nav`
  - `paper-community-report.out`
  - `paper-community-report.toc`
  - `paper-community-report.snm`
