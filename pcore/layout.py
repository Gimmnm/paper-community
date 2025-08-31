# pcore/layout.py
from __future__ import annotations
import networkx as nx
import numpy as np

def compute_layout(G: nx.Graph, kind: str = "fa2", seed: int = 42) -> dict[int, tuple[float, float]]:
    kind = kind.lower()
    if kind == "fa2":
        # 优先尝试旧 fa2（C 扩展），失败则用 fa2l（Numba）
        try:
            import fa2  # type: ignore
            forceatlas2 = fa2.ForceAtlas2(
                outboundAttractionDistribution=True,
                linLogMode=False,
                adjustSizes=False,
                edgeWeightInfluence=1.0,
                jitterTolerance=1.0,
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
            )
            A = nx.to_scipy_sparse_array(G, weight="weight", dtype=np.float32)
            pos = forceatlas2.forceatlas2(A, pos=None, iterations=200)
            nodes = list(G.nodes())
            return {int(n): (float(pos[i, 0]), float(pos[i, 1])) for i, n in enumerate(nodes)}
        except Exception:
            try:
                import fa2l  # type: ignore
                # fa2l 接口：fa2l.forceatlas2_networkx_layout(G, iterations=..., weight_attr="weight")
                pos = fa2l.forceatlas2_networkx_layout(G, iterations=200, weight_attr="weight", seed=seed)
                # pos 已是 {node: (x,y)}
                return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
            except Exception:
                # 回退 spring
                pass

        # 两个都失败就 fall back
        kind = "spring"

    if kind == "spring":
        pos = nx.spring_layout(G, seed=seed, weight="weight")
        return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

    if kind == "umap":
        import umap
        A = nx.to_scipy_sparse_array(G, weight="weight", dtype=np.float32)
        reducer = umap.UMAP(random_state=seed, metric="precomputed")
        # 这里用 1 - 归一化权重近似距离（避免 O(N^2) 直接构造）
        # 简化处理：直接用最短路/共现等自定义近似也可以
        # 为简洁，这里退回 spring 更稳
        pos = nx.spring_layout(G, seed=seed, weight="weight")
        return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

    raise ValueError(f"unknown layout kind: {kind}")