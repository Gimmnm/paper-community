# Frontend Refactor Spec

## Selector model

Left panel now includes:

- Algorithm selector
- Time-window selector
- Run selector
- Resolution selector
- Evaluation loader (overview table-like cards)

These selectors map to API runtime params:

- `run_id`
- `resolution`

## API contract additions

- `GET /api/v3/catalog`
- `GET /api/v3/session`
- `GET /api/v3/session/switch?run_id=...`
- `GET /api/v3/evaluations/overview`

Core existing query APIs accept optional:

- `run_id`
- `resolution`

## Interaction updates

- Community graph/search/details are scoped by active run/resolution.
- Graph center panel adds a hover info strip for node quick preview.
- Right panel enriches paper/community cards with:
  - keywords
  - author ids / author count
  - impact factor (if available)
  - topic terms (if available)
  - center/bridge papers (existing)

## Non-goals in this phase

- Full componentized frontend build tooling.
- Time-window-specific subgraph logic by date filtering in browser.
- Rich tooltip portal system (current hover strip is intentionally lightweight).
