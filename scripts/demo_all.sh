#!/usr/bin/env bash
set -e
python scripts/build_graph.py --csv data/papers.csv --emb data/embeddings.npy --mode mutual-knn --k 10 --tau 0.3
python scripts/run_community.py --graph data/graph/graph.gexf --algo louvain --resolution 1.0
python scripts/export_layout.py --graph data/graph/graph.gexf --layout fa2
python scripts/analyze_results.py --csv data/papers.csv --graph data/graph/graph.gexf \
  --communities data/graph/communities.csv --layout data/graph/layout.csv --outjson data/graph/graph.json
python scripts/load_db.py --nodes data/graph/nodes.csv --edges data/graph/edges.csv \
  --communities data/graph/communities.csv --layout data/graph/layout.csv --names data/graph/graph.json
echo "[done] offline pipeline finished."