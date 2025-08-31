# pcore/metrics.py
from __future__ import annotations
import numpy as np, pandas as pd, networkx as nx
from typing import Iterable, List, Dict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from networkx.algorithms.community.quality import modularity

def derive_fourclass(df: pd.DataFrame) -> np.ndarray:
    """根据 is_AP/is_NA 生成四分类；若无则兼容 field/field_multi -> 四类。"""
    if "is_AP" in df.columns or "is_NA" in df.columns:
        a = df.get("is_AP", 0).astype(int).values
        n = df.get("is_NA", 0).astype(int).values
        out = np.array(["Other"] * len(df), dtype=object)
        out[(a==1) & (n==0)] = "AP-only"
        out[(a==0) & (n==1)] = "NA-only"
        out[(a==1) & (n==1)] = "AP+NA"
        return out
    if "field" in df.columns:
        f = df["field"].astype(str)
        out = np.array(["Other"] * len(df), dtype=object)
        out[f=="AP"] = "AP-only"
        out[f=="NA"] = "NA-only"
        out[f=="AP_NA"] = "AP+NA"
        return out
    return np.array(["Other"] * len(df), dtype=object)

def edge_homophily(G: nx.Graph, labels: Iterable[str], weighted: bool = False) -> float:
    """边同质性：端点标签相同的边比例；weighted=True 按边权加权。"""
    lab = list(labels)
    same = 0.0; tot = 0.0
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0)) if weighted else 1.0
        tot += w
        if lab[int(u)] == lab[int(v)]:
            same += w
    return float(same / tot) if tot > 0 else 0.0

def modularity_weighted(G: nx.Graph, communities: List[List[int]]) -> float:
    """加权模块度（包装 networkx.modularity）。"""
    comm_sets = [set(map(int, c)) for c in communities]
    return float(modularity(G, comm_sets, weight="weight"))

def majority_label_f1(df: pd.DataFrame, cluster_labels: np.ndarray, label_col: str = "four") -> float:
    """用每个社区多数标签作为预测，计算 Macro-F1。"""
    cats = df[label_col].astype(str).values
    uniq = sorted(pd.unique(cats))
    lut = {c:i for i,c in enumerate(uniq)}
    y_true = np.array([lut[c] for c in cats], dtype=int)
    y_pred = np.zeros_like(y_true)
    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) == 0: continue
        vals, cnts = np.unique(y_true[idx], return_counts=True)
        y_pred[idx] = vals[np.argmax(cnts)]
    return float(f1_score(y_true, y_pred, average="macro"))

def nmi_ari_fourclass(df: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[str, float]:
    """以四分类为真值，计算 NMI / ARI。"""
    y_true = derive_fourclass(df)
    cats = {c:i for i,c in enumerate(sorted(pd.unique(y_true)))}
    yi = np.array([cats[c] for c in y_true])
    yj = np.asarray(cluster_labels, dtype=int)
    return {
        "NMI": float(normalized_mutual_info_score(yi, yj)),
        "ARI": float(adjusted_rand_score(yi, yj)),
    }