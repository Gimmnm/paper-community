#!/usr/bin/env bash
# 离线多算法对比 — 阶段编排（all-time；时间窗勿用本脚本）
#
#   bash scripts/offline_comparison_master.sh sweep   # 四算法分辨率 sweep（默认 r: 0.001→2.0 step 0.02）
#   bash scripts/offline_comparison_master.sh topics    # Topic-SCORE：默认仅跑各算法 comparison_breakpoints.csv 里 ~10 个 r（需先有断点表）
#   bash scripts/offline_comparison_master.sh viz       # 同上：默认按断点分辨率出图
#   bash scripts/offline_comparison_master.sh topic-viz # Topic 可视化：默认与 topics 一致按断点 r（需断点表 + topics + umap2d）
#   TOPIC_GRID=full bash ... topic-viz   # 与 topics 相同：按目录内 [0.0001,5] 凡有 membership+topic 的全部分辨率出图（落到 ...__<sweep>_full/）
#   bash scripts/offline_comparison_master.sh retrieval # 检索对比：默认按断点分辨率 × catalog（可调 SEEDS）
#   TOPIC_GRID=full bash ... topics   # 恢复「按 r-min/r-max 扫目录内全部 membership」旧行为
#   TOPIC_REP_PAPERS_MODE=approx bash ... topics   # 默认 off（跳过代表论文中心性，大批量快得多）
#   bash scripts/offline_comparison_master.sh help
#
# 阶段 2（登记表 + eval + 断点 + 总览）请用:
#   bash scripts/daily_workflow.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
# So progress lines reach `tee`/log files promptly (Python buffers stdout when not a TTY).
export PYTHONUNBUFFERED=1
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

EMB="${ROOT}/data/paper_embeddings_specter2.npy"
GRAPH="${ROOT}/out/mutual_knn_k50.npz"
K_TOPICS=10
BP_CSV="${ROOT}/out/experiment_eval/comparison_breakpoints.csv"

run_id_for_sweep_name() {
  case "$1" in
    leiden_sweep_cpm) echo leiden_cpm ;;
    leiden_sweep_rb) echo leiden ;;
    leiden_sweep_louvain) echo louvain ;;
    coarse_kmeans_then_cpm_k3_seed42) echo coarse_kmeans ;;
    *) echo "" ;;
  esac
}

run_sweep() {
  echo "[master/sweep] Leiden CPM -> out/leiden_sweep_cpm"
  "${PY}" src/core.py experiment-sweep --algorithm leiden_cpm --out-dir "${ROOT}/out/leiden_sweep_cpm" --emb-path "${EMB}"
  echo "[master/sweep] Leiden RB -> out/leiden_sweep_rb"
  "${PY}" src/core.py experiment-sweep --algorithm leiden --out-dir "${ROOT}/out/leiden_sweep_rb" --emb-path "${EMB}"
  echo "[master/sweep] Louvain (multiresolution / RBERVertexPartition) -> out/leiden_sweep_louvain"
  "${PY}" src/core.py experiment-sweep --algorithm louvain --out-dir "${ROOT}/out/leiden_sweep_louvain" --emb-path "${EMB}"
  COARSE="${ROOT}/out/coarse_domains_kmeans_k3_seed42"
  MERGED="${ROOT}/out/coarse_kmeans_then_cpm_k3_seed42"
  echo "[master/sweep] coarse k-means domains -> ${COARSE}"
  "${PY}" src/algorithm_layer/coarse_domains_kmeans.py --k 3 --seed 42 --emb-npy "${EMB}" --out-dir "${COARSE}"
  echo "[master/sweep] coarse + per-domain CPM merge -> ${MERGED}"
  "${PY}" src/core.py experiment-coarse-kmeans-sweep \
    --domains-dir "${COARSE}" \
    --graph-npz "${GRAPH}" \
    --out-dir "${MERGED}" \
    --algorithm leiden_cpm \
    --write-manifest
  echo "[master/sweep] done. Next: bash scripts/daily_workflow.sh"
}

run_topics() {
  local rmin=0.0001
  local rmax=5.0
  local rep_mode="${TOPIC_REP_PAPERS_MODE:-off}"
  local use_bp="${TOPIC_GRID:-breakpoints}"
  if [[ "$use_bp" == "breakpoints" ]] && [[ ! -f "$BP_CSV" ]]; then
    echo "[master/topics] missing ${BP_CSV} — run first: bash scripts/daily_workflow.sh"
    exit 1
  fi
  for tag_dir in \
    "leiden_sweep_cpm:${ROOT}/out/leiden_sweep_cpm" \
    "leiden_sweep_rb:${ROOT}/out/leiden_sweep_rb" \
    "leiden_sweep_louvain:${ROOT}/out/leiden_sweep_louvain" \
    "coarse_kmeans_then_cpm_k3_seed42:${ROOT}/out/coarse_kmeans_then_cpm_k3_seed42"
  do
    local name="${tag_dir%%:*}"
    local ld="${tag_dir#*:}"
    if [[ ! -d "$ld" ]]; then
      echo "[master/topics] skip missing $ld"
      continue
    fi
    local out="${ROOT}/out/topic_runs/${name}/K${K_TOPICS}"
    echo "[master/topics] $name -> $out"
    if [[ "$use_bp" == "breakpoints" ]]; then
      local rid
      rid="$(run_id_for_sweep_name "$name")"
      if [[ -z "$rid" ]]; then
        echo "[master/topics] skip unknown sweep tag $name"
        continue
      fi
      "${PY}" src/core.py topic-model-multi \
        --k-topics "${K_TOPICS}" \
        --leiden-dir "${ld}" \
        --out-root "${out}" \
        --breakpoints-csv "${BP_CSV}" \
        --breakpoint-run-id "${rid}" \
        --rep-papers-mode "${rep_mode}" \
        --skip-existing \
        --continue-on-error
    else
      "${PY}" src/core.py topic-model-multi \
        --k-topics "${K_TOPICS}" \
        --leiden-dir "${ld}" \
        --out-root "${out}" \
        --r-min "${rmin}" \
        --r-max "${rmax}" \
        --rep-papers-mode "${rep_mode}" \
        --skip-existing \
        --continue-on-error
    fi
  done
  TOPIC_DIAG_OUT="${ROOT}/out/experiment_eval/topic_collapse_diagnostics"
  if find "${ROOT}/out/topic_runs" -name 'communities_topic_weights.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[master/topics] topic collapse diagnostics -> ${TOPIC_DIAG_OUT}"
    "${PY}" src/core.py diagnose-topic-collapse \
      --root "${ROOT}/out/topic_runs" \
      --out-dir "${TOPIC_DIAG_OUT}" \
      --save-per-topic \
      --save-plots || echo "[master/topics] warn: diagnose-topic-collapse failed" >&2
  else
    echo "[master/topics] skip topic collapse diagnostics (no communities_topic_weights.csv under out/topic_runs)"
  fi
  echo "[master/topics] done."
}

run_viz() {
  if [[ ! -f "$BP_CSV" ]]; then
    echo "[master/viz] missing ${BP_CSV} — run first: bash scripts/daily_workflow.sh"
    exit 1
  fi
  local TAG="${VIZ_TAG_PREFIX:-master}"
  for tag_dir in \
    "leiden_sweep_cpm:${ROOT}/out/leiden_sweep_cpm" \
    "leiden_sweep_rb:${ROOT}/out/leiden_sweep_rb" \
    "leiden_sweep_louvain:${ROOT}/out/leiden_sweep_louvain" \
    "coarse_kmeans_then_cpm_k3_seed42:${ROOT}/out/coarse_kmeans_then_cpm_k3_seed42"
  do
    local name="${tag_dir%%:*}"
    local ld="${tag_dir#*:}"
    if [[ ! -d "$ld" ]]; then
      echo "[master/viz] skip missing $ld"
      continue
    fi
    local rid
    rid="$(run_id_for_sweep_name "$name")"
    if [[ -z "$rid" ]]; then
      echo "[master/viz] skip unknown sweep tag $name"
      continue
    fi
    echo "[master/viz] ${TAG}__${name} (breakpoints run_id=${rid})"
    local -a extra_viz=()
    if [[ "$name" == "coarse_kmeans_then_cpm_k3_seed42" ]]; then
      lbl="${ROOT}/out/coarse_domains_kmeans_k3_seed42/labels.npy"
      if [[ -f "$lbl" ]]; then
        extra_viz+=(--domain-labels-npy "$lbl")
      fi
    fi
    # shellcheck disable=SC2198  # empty-array safe under `set -u` across bash versions
    if [[ ${#extra_viz[@]} -eq 0 ]]; then
      "${PY}" src/core.py experiment-viz-batch \
        --run-tag "${TAG}__${name}" \
        --leiden-dir "${ld}" \
        --graph-npz "${GRAPH}" \
        --umap-npy "${ROOT}/out/umap2d.npy" \
        --resolution-source breakpoints \
        --breakpoints-csv "${BP_CSV}" \
        --breakpoint-run-id "${rid}"
    else
      "${PY}" src/core.py experiment-viz-batch \
        --run-tag "${TAG}__${name}" \
        --leiden-dir "${ld}" \
        --graph-npz "${GRAPH}" \
        --umap-npy "${ROOT}/out/umap2d.npy" \
        --resolution-source breakpoints \
        --breakpoints-csv "${BP_CSV}" \
        --breakpoint-run-id "${rid}" \
        "${extra_viz[@]}"
    fi
  done
  echo "[master/viz] done -> ${ROOT}/out/viz_batch/"
}

run_topic_viz() {
  local rmin=0.0001
  local rmax=5.0
  local use_bp="${TOPIC_GRID:-breakpoints}"
  local out_suffix=""
  if [[ "$use_bp" != "breakpoints" ]]; then
    out_suffix="_full"
  fi
  local UMAP_NPY="${TOPIC_VIZ_UMAP:-${ROOT}/out/umap2d.npy}"
  if [[ ! -f "$UMAP_NPY" ]]; then
    echo "[master/topic-viz] missing ${UMAP_NPY} — build 2D coords first (e.g. core.py build-2d)"
    exit 1
  fi
  if [[ "$use_bp" == "breakpoints" ]] && [[ ! -f "$BP_CSV" ]]; then
    echo "[master/topic-viz] missing ${BP_CSV} — run first: bash scripts/daily_workflow.sh"
    exit 1
  fi
  local TAG="${TOPIC_VIZ_TAG_PREFIX:-master}"
  local annotate="${TOPIC_VIZ_ANNOTATE_TOP_N:-8}"
  local -a skip_graph_flag=()
  if [[ "${TOPIC_VIZ_SKIP_GRAPH:-0}" == "1" ]]; then
    skip_graph_flag+=(--skip-graph)
  fi
  local -a max_pts_flag=()
  if [[ -n "${TOPIC_VIZ_MAX_POINTS:-}" ]]; then
    max_pts_flag+=(--max-points "${TOPIC_VIZ_MAX_POINTS}")
  fi
  local -a centroid_flag=(--community-centroid)
  if [[ "${TOPIC_VIZ_COMMUNITY_CENTROID:-1}" == "0" ]]; then
    centroid_flag=()
  fi
  local -a annotate_flag=()
  if [[ "${annotate}" != "0" ]] && [[ ${#centroid_flag[@]} -gt 0 ]]; then
    annotate_flag+=(--annotate-top-n-communities "${annotate}")
  fi
  for tag_dir in \
    "leiden_sweep_cpm:${ROOT}/out/leiden_sweep_cpm" \
    "leiden_sweep_rb:${ROOT}/out/leiden_sweep_rb" \
    "leiden_sweep_louvain:${ROOT}/out/leiden_sweep_louvain" \
    "coarse_kmeans_then_cpm_k3_seed42:${ROOT}/out/coarse_kmeans_then_cpm_k3_seed42"
  do
    local name="${tag_dir%%:*}"
    local ld="${tag_dir#*:}"
    if [[ ! -d "$ld" ]]; then
      echo "[master/topic-viz] skip missing $ld"
      continue
    fi
    local rid
    rid="$(run_id_for_sweep_name "$name")"
    if [[ -z "$rid" ]]; then
      echo "[master/topic-viz] skip unknown sweep tag $name"
      continue
    fi
    local topic_root="${ROOT}/out/topic_runs/${name}/K${K_TOPICS}"
    if [[ ! -d "$topic_root" ]]; then
      echo "[master/topic-viz] skip missing topic_root $topic_root (run: bash scripts/offline_comparison_master.sh topics)"
      continue
    fi
    local out="${ROOT}/out/topic_viz_batch/${TAG}__${name}${out_suffix}/K${K_TOPICS}"
    if [[ "$use_bp" == "breakpoints" ]]; then
      local rs
      rs="$("${PY}" -c "
import sys
from pathlib import Path
sys.path.insert(0, '${ROOT}/src')
from data_layer.breakpoint_schedule import load_breakpoint_resolutions_for_run
csv_path = Path('${BP_CSV}')
rs = load_breakpoint_resolutions_for_run(csv_path, run_id='${rid}', time_window='all')
if not rs:
    sys.exit(2)
print(' '.join(repr(float(x)) for x in rs))
")" || {
        echo "[master/topic-viz] skip $name: no breakpoints for run_id=${rid} in ${BP_CSV}"
        continue
      }
      echo "[master/topic-viz] ${TAG}__${name} (breakpoints) -> ${out}"
      local -a tv_cmd=(
        "${PY}" src/core.py topic-viz
        --k-topics "${K_TOPICS}"
        --leiden-dir "${ld}"
        --topic-root "${topic_root}"
        --out-dir "${out}"
        --umap "${UMAP_NPY}"
      )
      # shellcheck disable=SC2206
      tv_cmd+=( --resolutions ${rs} )
      [[ ${#skip_graph_flag[@]} -gt 0 ]] && tv_cmd+=( "${skip_graph_flag[@]}" )
      [[ ${#max_pts_flag[@]} -gt 0 ]] && tv_cmd+=( "${max_pts_flag[@]}" )
      [[ ${#centroid_flag[@]} -gt 0 ]] && tv_cmd+=( "${centroid_flag[@]}" )
      [[ ${#annotate_flag[@]} -gt 0 ]] && tv_cmd+=( "${annotate_flag[@]}" )
      "${tv_cmd[@]}"
    else
      echo "[master/topic-viz] ${TAG}__${name}${out_suffix} TOPIC_GRID=${use_bp} r∈[${rmin},${rmax}] -> ${out}"
      local -a tv_cmd=(
        "${PY}" src/core.py topic-viz
        --k-topics "${K_TOPICS}"
        --leiden-dir "${ld}"
        --topic-root "${topic_root}"
        --out-dir "${out}"
        --umap "${UMAP_NPY}"
        --r-min "${rmin}"
        --r-max "${rmax}"
      )
      [[ ${#skip_graph_flag[@]} -gt 0 ]] && tv_cmd+=( "${skip_graph_flag[@]}" )
      [[ ${#max_pts_flag[@]} -gt 0 ]] && tv_cmd+=( "${max_pts_flag[@]}" )
      [[ ${#centroid_flag[@]} -gt 0 ]] && tv_cmd+=( "${centroid_flag[@]}" )
      [[ ${#annotate_flag[@]} -gt 0 ]] && tv_cmd+=( "${annotate_flag[@]}" )
      "${tv_cmd[@]}"
    fi
  done
  echo "[master/topic-viz] done -> ${ROOT}/out/topic_viz_batch/"
}

run_retrieval() {
  local TAG="${RETRIEVAL_TAG:-master_smoke}"
  local SEEDS="${RETRIEVAL_SEEDS:-42 43 44}"
  local RSRC="${RETRIEVAL_RESOLUTION_SOURCE:-breakpoints}"
  local extra=()
  if [[ "$RSRC" == "breakpoints" ]]; then
    if [[ ! -f "$BP_CSV" ]]; then
      echo "[master/retrieval] missing ${BP_CSV} — run first: bash scripts/daily_workflow.sh"
      exit 1
    fi
    extra+=(--resolution-source breakpoints --breakpoints-csv "${BP_CSV}")
  else
    extra+=(--resolution-source summary --max-resolutions "${RETRIEVAL_MAX_RESOLUTIONS:-24}" --resolution-stride "${RETRIEVAL_RESOLUTION_STRIDE:-5}" --r-min 0.001 --r-max 2.0)
  fi
  "${PY}" src/core.py experiment-retrieval-benchmark \
    --run-tag "${TAG}" \
    --emb-path "${EMB}" \
    --keyword-index-dir "${ROOT}/out/keyword_index" \
    --seed-pid ${SEEDS} \
    "${extra[@]}"
  echo "[master/retrieval] done -> ${ROOT}/out/comparison_runs/${TAG}/"
}

case "${1:-help}" in
  sweep) run_sweep ;;
  topics) run_topics ;;
  viz) run_viz ;;
  topic-viz) run_topic_viz ;;
  retrieval) run_retrieval ;;
  help|*)
    sed -n '1,26p' "$0"
    ;;
esac
