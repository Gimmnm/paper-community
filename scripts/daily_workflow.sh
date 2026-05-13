#!/usr/bin/env bash
# 日常固定流程（一条命令跑完）：数据自检 → 登记各算法产物路径 → 每个 sweep 目录补 eval/ 图表
# → 写对比断点 CSV → 更新 experiment_eval 总览。可选：UMAP 着色 PNG。
#
# Usage (repo root):
#   bash scripts/daily_workflow.sh
#   bash scripts/daily_workflow.sh --force-data   # 强制重写 data_check.txt
#   bash scripts/daily_workflow.sh --viz          # 额外跑 build-2d + plot_umap_membership
#   bash scripts/daily_workflow.sh --with-retrieval  # 断点表生成后跑一轮检索对比（较慢），再写总览
#
# Env:
#   PC_RESOLUTION   可视化所用的分辨率（默认 0.2），与 demo-api 常用值对齐
#   RETRIEVAL_TAG / RETRIEVAL_SEEDS  与 scripts/offline_comparison_master.sh retrieval 相同语义（可选）
#   PC_EVAL_COMPARISON_RUN_TAGS  未使用 --with-retrieval 时，限定 experiment-eval 从哪些
#                                out/comparison_runs/<tag>/ 读 retrieval_score（逗号分隔，与 master retrieval 的 RETRIEVAL_TAG 对齐）
#
# 若已存在 out/topic_runs/**/communities_topic_weights.csv，则在 experiment-eval 之后会写出
# out/experiment_eval/topic_collapse_diagnostics/（见 docs/offline-outputs-catalog.md §6.4）。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

FORCE_DATA=0
DO_VIZ=0
WITH_RETRIEVAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-data) FORCE_DATA=1 ;;
    --viz) DO_VIZ=1 ;;
    --with-retrieval) WITH_RETRIEVAL=1 ;;
    -h|--help)
      sed -n '1,28p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
  shift
done

echo "[daily] repo: ${ROOT}"
echo "[daily] python: ${PY}"

if [[ "${FORCE_DATA}" -eq 1 ]]; then
  "${PY}" src/core.py check-data --force
else
  "${PY}" src/core.py check-data
fi

"${PY}" src/core.py experiment-init-minimal
"${PY}" src/core.py experiment-eval-bundle --skip-existing
"${PY}" src/core.py experiment-comparison-breakpoints

if [[ "${WITH_RETRIEVAL}" -eq 1 ]]; then
  BP_CSV="${ROOT}/out/experiment_eval/comparison_breakpoints.csv"
  if [[ ! -f "$BP_CSV" ]]; then
    echo "[daily] --with-retrieval requires ${BP_CSV}" >&2
    exit 1
  fi
  TAG="${RETRIEVAL_TAG:-daily_smoke}"
  SEEDS="${RETRIEVAL_SEEDS:-42}"
  echo "[daily] retrieval benchmark -> out/comparison_runs/${TAG}/ (seeds: ${SEEDS})"
  # shellcheck disable=SC2086
  "${PY}" src/core.py experiment-retrieval-benchmark \
    --run-tag "${TAG}" \
    --emb-path "${ROOT}/data/paper_embeddings_specter2.npy" \
    --keyword-index-dir "${ROOT}/out/keyword_index" \
    --resolution-source breakpoints \
    --breakpoints-csv "${BP_CSV}" \
    --seed-pid ${SEEDS}
fi

if [[ "${WITH_RETRIEVAL}" -eq 1 ]]; then
  "${PY}" src/core.py experiment-eval --comparison-run-tag "${RETRIEVAL_TAG:-daily_smoke}"
else
  "${PY}" src/core.py experiment-eval
fi

TOPIC_ROOT="${ROOT}/out/topic_runs"
TOPIC_DIAG_OUT="${ROOT}/out/experiment_eval/topic_collapse_diagnostics"
if [[ -d "${TOPIC_ROOT}" ]] && find "${TOPIC_ROOT}" -name 'communities_topic_weights.csv' -print -quit 2>/dev/null | grep -q .; then
  echo "[daily] topic collapse diagnostics -> ${TOPIC_DIAG_OUT}"
  "${PY}" src/core.py diagnose-topic-collapse \
    --root "${TOPIC_ROOT}" \
    --out-dir "${TOPIC_DIAG_OUT}" \
    --save-per-topic \
    --save-plots || echo "[daily] warn: diagnose-topic-collapse failed (see stderr)" >&2
else
  echo "[daily] skip topic collapse diagnostics (no communities_topic_weights.csv under ${TOPIC_ROOT})"
fi

if [[ "${DO_VIZ}" -eq 1 ]]; then
  echo "[daily] viz: build-2d (跳过已有缓存时很快)"
  "${PY}" src/core.py build-2d
  R="${PC_RESOLUTION:-0.2}"
  echo "[daily] viz: UMAP + membership @ r=${R}"
  "${PY}" scripts/plot_umap_membership.py \
    --umap-npy out/umap2d.npy \
    --leiden-dir out/leiden_sweep_cpm \
    --resolution "${R}" \
    --out-png out/viz/umap_communities_cpm_r${R}.png \
    --max-points 30000
fi

echo "[daily] done."
