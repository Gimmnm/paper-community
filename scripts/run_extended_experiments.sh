#!/usr/bin/env bash
# 扩展算法离线产物：Louvain 全图 sweep、k-means 粗分 + 各 domain 诱导子图 CPM sweep（合并标签）。
# 依赖已有：data/paper_embeddings_specter2.npy、全局 mutual-kNN（默认 out/mutual_knn_k50.npz）。
# 无需安装 torch/transformers：core 会从 npy 直读 embedding；只有缺失 npy 且需现算时才要重型依赖。
#
# Usage（仓库根目录）:
#   bash scripts/run_extended_experiments.sh
#
# 完成后可执行:
#   PYTHONPATH=src python src/core.py experiment-init-minimal --topic-communities-csv path/to/communities_topic_weights.csv
#   PYTHONPATH=src python src/core.py experiment-eval

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

COARSE_ROOT="${ROOT}/out/coarse_domains_kmeans_k3_seed42"
MERGED="${ROOT}/out/coarse_kmeans_then_cpm_k3_seed42"
EMB="${ROOT}/data/paper_embeddings_specter2.npy"
GRAPH="${ROOT}/out/mutual_knn_k50.npz"

echo "[ext] Louvain full-graph sweep -> out/leiden_sweep_louvain"
"${PY}" src/core.py experiment-sweep \
  --algorithm louvain \
  --out-dir "${ROOT}/out/leiden_sweep_louvain"

echo "[ext] Coarse k-means domains -> ${COARSE_ROOT}"
"${PY}" src/algorithm_layer/coarse_domains_kmeans.py \
  --k 3 --seed 42 \
  --emb-npy "${EMB}" \
  --out-dir "${COARSE_ROOT}"

echo "[ext] Per-domain induced subgraph CPM sweeps + merged memberships -> ${MERGED}"
"${PY}" src/core.py experiment-coarse-kmeans-sweep \
  --domains-dir "${COARSE_ROOT}" \
  --graph-npz "${GRAPH}" \
  --out-dir "${MERGED}" \
  --algorithm leiden_cpm \
  --r-min 0.001 \
  --r-max 2.0 \
  --step 0.02 \
  --seed 42 \
  --write-manifest

echo "[ext] Register catalogs (skips missing dirs) + evaluation overview"
"${PY}" src/core.py experiment-init-minimal
"${PY}" src/core.py experiment-eval

echo "[ext] done."
