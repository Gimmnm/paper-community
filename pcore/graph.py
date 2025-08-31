# pcore/graph.py
from __future__ import annotations
import numpy as np, networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def l2norm(E: np.ndarray) -> np.ndarray:
    return normalize(E, norm="l2", axis=1, copy=False)

def mutual_knn_graph(E: np.ndarray, k: int = 10, tau: float = 0.0) -> nx.Graph:
    """互为近邻 + 相似度阈值过滤；边权=cosine sim."""
    n = E.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k+1, n), metric="cosine").fit(E)
    dist, idx = nn.kneighbors(E)
    nbrs = [set(row[1:]) for row in idx]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for jpos, j in enumerate(idx[i,1:]):
            if i in nbrs[j]:
                sim = 1.0 - dist[i, jpos+1]
                if sim >= tau:
                    G.add_edge(int(i), int(j), weight=float(sim))
    return G

def threshold_graph(E: np.ndarray, tau: float) -> nx.Graph:
    """全对相似度阈值（N~2000 可承受），边权=cosine sim."""
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(E, E)
    np.fill_diagonal(S, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(E)))
    i_idx, j_idx = np.where(S >= tau)
    for i, j in zip(i_idx, j_idx):
        if i < j:
            G.add_edge(int(i), int(j), weight=float(S[i, j]))
    return G