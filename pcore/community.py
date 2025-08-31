# pcore/community.py
from __future__ import annotations
from typing import Dict, List, Hashable, Tuple
import networkx as nx

def _to_int_graph(G: nx.Graph, weight_attr: str = "weight") -> Tuple[Dict[Hashable, int], Dict[int, Hashable], list[tuple[int,int]], list[float]]:
    nodes = list(G.nodes())
    n2i = {n: i for i, n in enumerate(nodes)}
    i2n = {i: n for n, i in n2i.items()}
    edges_i = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges_i.append((n2i[u], n2i[v]))
        weights.append(float(data.get(weight_attr, 1.0)))
    return n2i, i2n, edges_i, weights

def run_louvain(G: nx.Graph, weight_attr: str = "weight", resolution: float = 1.0, seed: int = 42):
    """
    返回: (part, comms)
      - part: {node_key -> community_id}
      - comms: List[List[node_key]]
    """
    try:
        import community as community_louvain  # python-louvain
    except Exception as e:
        raise RuntimeError("需要安装 python-louvain： pip install python-louvain") from e

    part = community_louvain.best_partition(G, weight=weight_attr, random_state=seed, resolution=resolution)
    # 规整为 int 社区 id（best_partition 已经是 int）
    comms: Dict[int, List] = {}
    for n, c in part.items():
        comms.setdefault(int(c), []).append(n)
    # 排序并重映射社区 id 连续化（0..C-1）
    remap = {c: i for i, c in enumerate(sorted(comms))}
    part2 = {n: remap[int(c)] for n, c in part.items()}
    comms2 = [[] for _ in range(len(remap))]
    for n, c in part2.items():
        comms2[c].append(n)
    return part2, comms2

def run_leiden(G: nx.Graph, weight_attr: str = "weight", resolution: float = 1.0, seed: int = 42):
    """Leiden（基于 igraph + leidenalg）"""
    try:
        import igraph as ig
        import leidenalg
    except Exception as e:
        raise RuntimeError("需要安装 igraph 与 leidenalg： conda install python-igraph leidenalg") from e

    n2i, i2n, edges_i, weights = _to_int_graph(G, weight_attr=weight_attr)
    g = ig.Graph(n=len(n2i), edges=edges_i, directed=False)
    g.es["weight"] = weights
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    # part 是一个包含若干社区（每个是顶点 id 列表）的对象
    comms = [[i2n[int(v)] for v in comm] for comm in part]
    # 展平成 {node -> cid}
    part_map = {}
    for cid, members in enumerate(comms):
        for n in members:
            part_map[n] = cid
    return part_map, comms

def run_infomap(G: nx.Graph, weight_attr: str = "weight", seed: int = 42):
    """Infomap（基于 infomap Python 包）"""
    try:
        from infomap import Infomap
    except Exception as e:
        raise RuntimeError("需要安装 infomap： pip install infomap") from e

    n2i, i2n, edges_i, weights = _to_int_graph(G, weight_attr=weight_attr)
    im = Infomap(f"--two-level --undirected --silent --seed {seed}")
    for (u, v), w in zip(edges_i, weights):
        im.add_link(u, v, float(w))
    im.run()

    modules = im.get_modules()  # dict: int_node_id -> module_id
    # 规范化 module id 连续化（0..C-1）
    uniq = sorted(set(int(m) for m in modules.values()))
    remap = {m: i for i, m in enumerate(uniq)}
    part_map = {i2n[i]: remap[int(m)] for i, m in modules.items()}

    # 反推 comms
    comms: Dict[int, List] = {}
    for n, c in part_map.items():
        comms.setdefault(c, []).append(n)
    comms2 = [comms[i] for i in range(len(comms))]
    return part_map, comms2