#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 2D 布局坐标（CSV），支持：
- fa2         : ForceAtlas2（优先 fa2l；失败尝试 fa2；最后回退 nx-spring），支持分步+进度条
- nx-spring   : NetworkX spring_layout（分步+进度条）
- kk          : NetworkX kamada_kawai_layout（一次性）
- spectral    : NetworkX spectral_layout（一次性，很快）
- igraph-fr   : igraph 的 Fruchterman-Reingold（一次性，C 实现，通常更快）

输入：graph.gexf（节点键可能是字符串或整数）
输出：layout.csv（列：node, index, x, y, [community], [paper_id], [title]）

示例：
  python scripts/export_layout.py --graph data/graph/graph.gexf \
    --layout fa2 --iters 300 --step 25 --init spectral --progress
"""

from __future__ import annotations
import os, argparse, inspect, time
import numpy as np
import pandas as pd
import networkx as nx

# 可选进度条
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def get_node_data(G, n):
    try:
        return G.nodes[n]   # nx >= 2
    except Exception:
        return G.node[n]    # nx 1.x

def build_integer_graph(G: nx.Graph, weight_attr: str = "weight"):
    nodes = list(G.nodes())
    node2int = {n: i for i, n in enumerate(nodes)}
    int2node = {i: n for n, i in node2int.items()}
    H = nx.Graph()
    H.add_nodes_from(range(len(nodes)))
    for u, v, data in G.edges(data=True):
        try:
            w = float(data.get(weight_attr, 1.0))
        except Exception:
            w = 1.0
        H.add_edge(node2int[u], node2int[v], **{weight_attr: w})
    return H, node2int, int2node

def _call_with_accepted_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return func(*args, **filtered)

def make_bar(total, enable: bool, desc: str):
    if enable and tqdm is not None:
        return tqdm(total=total, desc=desc)
    # dummy
    class _Dummy:
        def update(self, n): pass
        def close(self): pass
    return _Dummy()

def layout_init(H: nx.Graph, kind: str, weight_attr="weight"):
    if kind == "none":
        return None
    if kind == "spectral":
        print("[init] spectral_layout")
        return nx.spectral_layout(H, dim=2, weight=weight_attr)
    if kind == "kk":
        print("[init] kamada_kawai_layout (init)")
        try:
            return nx.kamada_kawai_layout(H, weight=weight_attr)
        except TypeError:
            return nx.kamada_kawai_layout(H)
    raise ValueError("init must be one of: none, spectral, kk")

# ---------------- FA2（分步）----------------
def _pick_fa2l_func():
    """选择可用的 fa2l 函数（不同版本函数名不同）。返回 (callable, name) 或 (None, reason)。"""
    try:
        import fa2l
    except Exception as e:
        return None, f"import fa2l failed: {e}"
    candidates = []
    for name in ("forceatlas2_networkx_layout", "force_atlas2_layout"):
        f = getattr(fa2l, name, None)
        if callable(f):
            candidates.append(("fa2l."+name, f))
    try:
        import fa2l.fa2l as fa2l_mod
        f2 = getattr(fa2l_mod, "force_atlas2_layout", None)
        if callable(f2):
            candidates.append(("fa2l.fa2l.force_atlas2_layout", f2))
    except Exception:
        pass
    if not candidates:
        return None, f"fa2l loaded ({getattr(fa2l,'__file__','?')}), but no FA2 function. attrs={dir(fa2l)}"
    # 选第一个命中的
    return candidates[0]  # (name, func)

def layout_fa2_stepwise(H: nx.Graph, weight_attr: str, total_iters: int, step: int,
                        pos_init=None, show_progress=False):
    """fa2l/fa2 分步调用，实现进度条。"""
    pos = pos_init
    # 先尝试 fa2l
    name_func = _pick_fa2l_func()
    if isinstance(name_func, tuple) and callable(name_func[1]):
        fname, func = name_func
        print(f"[fa2] using {fname} (stepwise, step={step})")
        bar = make_bar(total_iters, show_progress, "FA2 (fa2l)")
        remaining = total_iters
        while remaining > 0:
            it = min(step, remaining)
            kw = dict(iterations=int(it), pos=pos, weight_attr=weight_attr, weight=weight_attr)
            try:
                pos = _call_with_accepted_kwargs(func, H, **kw)
            except TypeError:
                # 签名不支持 pos/weight_attr，就去掉再试
                pos = _call_with_accepted_kwargs(func, H, iterations=int(it))
            remaining -= it
            bar.update(it)
        bar.close()
        return pos

    # fa2l 不行，尝试 python-fa2
    try:
        from fa2 import ForceAtlas2
        print(f"[fa2] using python-fa2 ForceAtlas2 (stepwise, step={step})")
        fa2 = ForceAtlas2(
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=1.0,
            jitterTolerance=1.0,
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            verbose=False,
        )
        bar = make_bar(total_iters, show_progress, "FA2 (fa2)")
        remaining = total_iters
        while remaining > 0:
            it = min(step, remaining)
            pos = fa2.forceatlas2_networkx_layout(H, pos=pos, iterations=int(it), weight_attr=weight_attr)
            remaining -= it
            bar.update(it)
        bar.close()
        return pos
    except Exception as e2:
        raise RuntimeError(f"FA2 不可用：fa2l={name_func[1] if isinstance(name_func, tuple) else name_func}；fa2失败=<{e2}>")

# ---------------- 其它布局 ----------------
def layout_nx_spring_stepwise(H: nx.Graph, weight_attr: str, total_iters: int, step: int,
                              pos_init=None, show_progress=False):
    print(f"[nx] spring_layout (stepwise, step={step})")
    pos = pos_init
    bar = make_bar(total_iters, show_progress, "spring")
    remaining = total_iters
    while remaining > 0:
        it = min(step, remaining)
        try:
            pos = nx.spring_layout(H, iterations=int(it), weight=weight_attr, seed=42, pos=pos)
        except TypeError:
            pos = nx.spring_layout(H, iterations=int(it), weight=weight_attr)
        remaining -= it
        bar.update(it)
    bar.close()
    return pos

def layout_kk(H: nx.Graph, weight_attr="weight"):
    print("[nx] kamada_kawai_layout")
    try:
        return nx.kamada_kawai_layout(H, weight=weight_attr)
    except TypeError:
        return nx.kamada_kawai_layout(H)

def layout_spectral(H: nx.Graph, weight_attr="weight"):
    print("[nx] spectral_layout")
    return nx.spectral_layout(H, dim=2, weight=weight_attr)

def layout_igraph_fr(H: nx.Graph, weight_attr="weight", iters=300, show_progress=False):
    print("[igraph] fruchterman_reingold (one-shot)")
    try:
        import igraph as ig
    except Exception as e:
        raise RuntimeError(f"python-igraph 未安装：{e}")
    n = H.number_of_nodes()
    edges = list(H.edges())
    weights = [float(H[u][v].get(weight_attr, 1.0)) for u, v in edges]
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    # igraph 没有迭代回调，这里只能一次性运行；如果需要“假进度条”，可以在外层包一圈。
    coords = np.asarray(g.layout_fruchterman_reingold(niter=int(iters), weights="weight").coords, dtype=np.float64)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-9)
    return {i: (float(coords[i,0]), float(coords[i,1])) for i in range(n)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="输入 GEXF 图路径")
    ap.add_argument("--layout", default="fa2",
                    choices=["fa2", "nx-spring", "kk", "spectral", "igraph-fr"],
                    help="布局算法")
    ap.add_argument("--init", default="spectral", choices=["none", "spectral", "kk"],
                    help="初始坐标（可显著加速收敛）")
    ap.add_argument("--iters", type=int, default=300, help="总迭代步数（fa2/spring/igraph-fr）")
    ap.add_argument("--step", type=int, default=25, help="每次调用的步数（进度条一格）")
    ap.add_argument("--progress", action="store_true", help="显示进度条（需要 tqdm）")
    ap.add_argument("--weight-attr", default="weight", help="边权属性名")
    ap.add_argument("--out", default=None, help="输出 CSV（默认与 graph 同目录 layout.csv）")
    args = ap.parse_args()

    t0 = time.perf_counter()
    G = nx.read_gexf(args.graph)
    print(f"[read] graph nodes={G.number_of_nodes()} edges={G.number_of_edges()} file={args.graph}")

    H, node2int, int2node = build_integer_graph(G, weight_attr=args.weight_attr)
    pos0 = layout_init(H, args.init, weight_attr=args.weight_attr)  # 可能为 None

    if args.layout == "fa2":
        try:
            pos_int = layout_fa2_stepwise(H, weight_attr=args.weight_attr,
                                          total_iters=args.iters, step=args.step,
                                          pos_init=pos0, show_progress=args.progress)
        except Exception as e:
            print(f"[warn] fa2 失败，回退 nx-spring：{e}")
            pos_int = layout_nx_spring_stepwise(H, weight_attr=args.weight_attr,
                                                total_iters=args.iters, step=args.step,
                                                pos_init=pos0, show_progress=args.progress)
    elif args.layout == "nx-spring":
        pos_int = layout_nx_spring_stepwise(H, weight_attr=args.weight_attr,
                                            total_iters=args.iters, step=args.step,
                                            pos_init=pos0, show_progress=args.progress)
    elif args.layout == "kk":
        pos_int = layout_kk(H, weight_attr=args.weight_attr)
    elif args.layout == "spectral":
        pos_int = layout_spectral(H, weight_attr=args.weight_attr)
    elif args.layout == "igraph-fr":
        # 一次性；可在外层给个“假进度条”但没太大意义
        pos_int = layout_igraph_fr(H, weight_attr=args.weight_attr, iters=args.iters, show_progress=args.progress)
    else:
        raise ValueError("unknown layout")

    rows = []
    for i in range(len(int2node)):
        n = int2node[i]
        x, y = float(pos_int[i][0]), float(pos_int[i][1])
        data = get_node_data(G, n)
        idx = data.get("index")
        if idx is None:
            try:
                idx = int(n)
            except Exception:
                idx = np.nan
        row = {"node": n, "index": idx, "x": x, "y": y}
        for extra in ("community", "paper_id", "title"):
            if extra in data:
                row[extra] = data[extra]
        rows.append(row)

    out_csv = args.out or os.path.join(os.path.dirname(os.path.abspath(args.graph)) or ".", "layout.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    dt = time.perf_counter() - t0
    print(f"[ok] layout -> {out_csv}  (elapsed {dt:.1f}s)")

if __name__ == "__main__":
    main()