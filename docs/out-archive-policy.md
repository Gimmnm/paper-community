# Out Directory Cleanup Policy

Use this policy during the refactor phase to keep `out/` focused on active runs.

## Keep in active `out/`

- Artifacts pointed to by manifests under `out/experiments/**/manifest.json`
- Current demo/API dependencies for selected run(s)
- Fresh evaluation exports used by frontend/API

## Move to archive

Move to `archive/legacy/out/` when all are true:

1. not referenced by any active experiment manifest
2. not used by current `demo-api` session
3. replaced by newer output or no longer in current scope

## Minimal archival note

For each moved folder, include:

- original path
- move date
- reason (`obsolete_hierarchy`, `superseded_run`, `temporary_debug_output`, etc.)

## Safety checks

- verify active run catalog still resolves
- verify `demo-api` health/session endpoints still load
- verify web graph + keyword/community lookup still work

## Helper command

Dry-run deprecated folder candidates:

```bash
PYTHONPATH=src python src/tools/archive_out_candidates.py
```

Execute moves:

```bash
PYTHONPATH=src python src/tools/archive_out_candidates.py --execute
```
