#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, numpy as np, torch, networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph/graph.gexf")
    ap.add_argument("--emb", default="data/embeddings.npy")
    ap.add_argument("--outdir", default="data/graph")
    ap.add_argument("--k", type=int, default=10, help="KMeans 簇数")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    G = nx.read_gexf(args.graph)
    X = np.load(args.emb); X = normalize(X)

    # 依赖 torch-geometric（按需安装）
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    data.x = torch.tensor(X, dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)

    # 简单两层 GCN 自监督重构
    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F
    class GCN(torch.nn.Module):
        def __init__(self, in_dim, hid=256, out_dim=128):
            super().__init__()
            self.g1 = GCNConv(in_dim, hid)
            self.g2 = GCNConv(hid, out_dim)
        def forward(self, data):
            x = self.g1(data.x, data.edge_index); x = F.relu(x)
            x = self.g2(x, data.edge_index)
            return x

    model = GCN(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        z = model(data)
        loss = torch.mean((z - data.x)**2)  # 重构损失
        loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"[ep {ep+1}] loss={loss.item():.4f}")

    Z = model(data).detach().cpu().numpy()
    km = KMeans(n_clusters=args.k, n_init=10, random_state=42).fit(Z)
    labels = km.labels_
    import pandas as pd
    pd.DataFrame({"node": list(G.nodes()), "community": labels}).to_csv(f"{args.outdir}/communities_gnn.csv", index=False)
    print(f"[ok] -> communities_gnn.csv")

if __name__ == "__main__":
    main()