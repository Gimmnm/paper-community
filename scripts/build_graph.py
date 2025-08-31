#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构图脚本（在建图时将论文元数据写入节点属性）：
- 输入：papers.csv（含 id/title/authors/...），embeddings.npy
- 支持：mutual-kNN / threshold
- 节点属性：index（原行号）、paper_id（原 id 列）、title、authors、可选标签
- 导出：nodes.csv / edges.csv / graph.gexf
"""
from __future__ import annotations
import os, math, argparse
import numpy as np, pandas as pd, networkx as nx
from typing import Dict, Any, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def l2norm(E: np.ndarray) -> np.ndarray:
    return normalize(E, norm="l2", axis=1, copy=False)

def build_mutual_knn_graph(E: np.ndarray, k: int, tau: float = 0.0) -> nx.Graph:
    n = E.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine").fit(E)
    dist, idx = nn.kneighbors(E)
    nbrs = [set(row[1:]) for row in idx]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for jpos, j in enumerate(idx[i,1:]):
            j = int(j)
            if i in nbrs[j]:
                sim = 1.0 - float(dist[i, jpos+1])
                if sim >= tau:
                    u, v = (i, j) if i < j else (j, i)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=sim)
    return G

def build_threshold_graph(E: np.ndarray, tau: float) -> nx.Graph:
    n = E.shape[0]
    S = E @ E.T
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.triu_indices(n, k=1)
    sims = S[rows, cols]
    mask = sims >= tau
    for u, v, w in zip(rows[mask], cols[mask], sims[mask]):
        G.add_edge(int(u), int(v), weight=float(w))
    return G

def _maybe_set(node: Dict[str, Any], key: str, val: Any) -> None:
    if val is None: return
    if isinstance(val, float) and math.isnan(val): return
    node[key] = val

def attach_node_attrs(G: nx.Graph, df: pd.DataFrame,
                      id_col="id", title_col="title", authors_col="authors") -> None:
    """写节点属性：index（行号）、paper_id、title、authors、可选标签"""
    n = len(df)
    opt_cols = [c for c in ["field","field_multi","is_AP","is_NA","categories","year","venue"] if c in df.columns]
    for i in range(n):
        if i not in G: continue
        # 固化原行号
        G.nodes[i]["index"] = int(i)
        # 用 paper_id（避免与 GEXF 的保留 id 冲突）
        if id_col in df.columns:
            v = df.at[i, id_col]; _maybe_set(G.nodes[i], "paper_id", None if pd.isna(v) else str(v))
        if title_col in df.columns:
            v = df.at[i, title_col]; _maybe_set(G.nodes[i], "title", None if pd.isna(v) else str(v))
        if authors_col in df.columns:
            v = df.at[i, authors_col]; _maybe_set(G.nodes[i], "authors", None if pd.isna(v) else str(v))
        for c in opt_cols:
            v = df.at[i, c]
            if pd.isna(v): continue
            if isinstance(v, (np.integer, int)): _maybe_set(G.nodes[i], c, int(v))
            elif isinstance(v, (np.floating, float)):
                if not math.isnan(float(v)): _maybe_set(G.nodes[i], c, float(v))
            else: _maybe_set(G.nodes[i], c, str(v))

def export_nodes_edges_csv(G: nx.Graph, df: pd.DataFrame, outdir: str) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    n = len(df)
    deg = dict(G.degree(weight=None))
    strength = {u: 0.0 for u in G.nodes()}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        strength[u] = strength.get(u, 0.0) + w
        strength[v] = strength.get(v, 0.0) + w

    rows = []
    for i in range(n):
        nd = G.nodes[i]
        row = {
            "index": i,
            "paper_id": nd.get("paper_id"),
            "title": nd.get("title"),
            "authors": nd.get("authors"),
            "degree": int(deg.get(i, 0)),
            "strength": float(strength.get(i, 0.0)),
        }
        for c in ["field","field_multi","is_AP","is_NA","categories","year","venue"]:
            if c in nd: row[c] = nd[c]
        rows.append(row)
    nodes_csv = os.path.join(outdir, "nodes.csv")
    pd.DataFrame(rows).to_csv(nodes_csv, index=False)

    erows = []
    for u, v, data in G.edges(data=True):
        a, b = (u, v) if u < v else (v, u)
        erows.append({"source": int(a), "target": int(b), "weight": float(data.get("weight", 1.0))})
    edges_csv = os.path.join(outdir, "edges.csv")
    pd.DataFrame(erows).to_csv(edges_csv, index=False)
    return nodes_csv, edges_csv

def sanitize_for_gexf(G: nx.Graph) -> None:
    def fix(v):
        if v is None: return None, False
        if isinstance(v, (np.integer,)): return int(v), True
        if isinstance(v, (np.floating,)):
            fv = float(v); 
            if math.isnan(fv): return None, False
            return fv, True
        if isinstance(v, (np.bool_,)): return bool(v), True
        if isinstance(v, (int,float,bool,str)): 
            if isinstance(v, float) and math.isnan(v): return None, False
            return v, True
        try: return str(v), True
        except: return None, False
    for _, data in G.nodes(data=True):
        for k in list(data.keys()):
            vv, ok = fix(data[k])
            if ok: data[k] = vv
            else: del data[k]
    for _, _, data in G.edges(data=True):
        for k in list(data.keys()):
            vv, ok = fix(data[k])
            if ok: data[k] = vv
            else: del data[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--mode", default="mutual-knn", choices=["mutual-knn","threshold"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--outdir", default="data/graph")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--title-col", default="title")
    ap.add_argument("--authors-col", default="authors")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    E = np.load(args.emb); assert E.shape[0] == len(df), "emb 行数与 CSV 不一致"
    E = l2norm(E)

    G = build_mutual_knn_graph(E, k=args.k, tau=args.tau) if args.mode=="mutual-knn" else build_threshold_graph(E, tau=args.tau)
    print(f"[graph] mode={args.mode}  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  (k={args.k}, tau={args.tau})")

    attach_node_attrs(G, df, id_col=args.id_col, title_col=args.title_col, authors_col=args.authors_col)
    nodes_csv, edges_csv = export_nodes_edges_csv(G, df, args.outdir)
    print(f"[ok] write -> {nodes_csv}")
    print(f"[ok] write -> {edges_csv}")

    sanitize_for_gexf(G)
    gexf_path = os.path.join(args.outdir, "graph.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"[ok] write -> {gexf_path}")

if __name__ == "__main__":
    main()