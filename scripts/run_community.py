#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行社区发现（louvain/leiden/infomap），把 community 写回 GEXF，并导出 CSV。
- 读取 graph.gexf（节点键可能是字符串或整数）
- 优先从节点属性中读取 index（构图时已固化）
- 导出 communities.csv: node_key, index, community, paper_id, title
"""
from __future__ import annotations
import os, argparse, sys, numpy as np, pandas as pd, networkx as nx
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pcore.community import run_louvain, run_leiden, run_infomap  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--algo", default="leiden", choices=["louvain","leiden","infomap"])
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight-attr", type=str, default="weight")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.graph)) or "."
    os.makedirs(outdir, exist_ok=True)

    G = nx.read_gexf(args.graph)
    nodes = list(G.nodes())
    print(f"[read] graph nodes={len(nodes)} edges={G.number_of_edges()} file={args.graph}")

    if args.algo == "louvain":
        part, comms = run_louvain(G, weight_attr=args.weight_attr, resolution=args.resolution, seed=args.seed)
    elif args.algo == "leiden":
        part, comms = run_leiden(G, weight_attr=args.weight_attr, resolution=args.resolution, seed=args.seed)
    else:
        part, comms = run_infomap(G, weight_attr=args.weight_attr, seed=args.seed)

    print(f"[community] algo={args.algo} communities={len(comms)}")

    missing = [n for n in nodes if n not in part]
    if missing:
        raise KeyError(f"{len(missing)} nodes not in partition, e.g. {missing[:5]}")

    labels = np.array([int(part[n]) for n in nodes], dtype=int)

    # 写回社区标签
    for n in nodes:
        G.nodes[n]["community"] = int(part[n])
    out_gexf = os.path.join(outdir, "graph__with_community.gexf")
    nx.write_gexf(G, out_gexf)
    print(f"[ok] write gexf -> {out_gexf}")

    # 导出 communities.csv
    def _as_int_or_nan(x):
        try: return int(x)
        except Exception: return np.nan

    rows = []
    for n in nodes:
        data = G.nodes[n]
        # 优先使用节点属性里的 index；没有就尝试把 node 键转 int
        idx = data.get("index", _as_int_or_nan(n))
        rows.append({
            "node": n,
            "index": idx,
            "community": int(part[n]),
            "paper_id": data.get("paper_id"),
            "title": data.get("title"),
        })
    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, "communities.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"[ok] write csv  -> {out_csv}")

    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"[stats] #communities={len(sizes)} | size head:\n{sizes.head()}")

if __name__ == "__main__":
    main()