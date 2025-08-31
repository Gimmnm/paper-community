from __future__ import annotations
import os, pandas as pd
from functools import lru_cache

DATA_DIR = os.environ.get("PC_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "graph"))

@lru_cache(maxsize=1)
def load_nodes() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "nodes.csv")
    df = pd.read_csv(p, dtype={"index": str, "paper_id": str}, keep_default_na=False)
    # 保证有 index（字符串）和 community
    if "index" not in df.columns: df["index"] = df.index.astype(str)
    if "community" not in df.columns: df["community"] = -1
    return df

@lru_cache(maxsize=1)
def load_edges() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "edges.csv")
    df = pd.read_csv(p, dtype={"src": str, "dst": str})
    if "weight" not in df.columns: df["weight"] = 1.0
    return df

@lru_cache(maxsize=1)
def load_layout() -> pd.DataFrame:
    p = os.path.join(DATA_DIR, "layout.csv")
    df = pd.read_csv(p, dtype={"index": str})
    # 坐标列名固定为 x,y
    if "x" not in df.columns or "y" not in df.columns:
        raise RuntimeError("layout.csv 需包含列 x,y")
    return df

def build_joined_nodes() -> pd.DataFrame:
    nodes = load_nodes()
    layout = load_layout()
    # 以 index 连接
    merged = nodes.merge(layout[["index","x","y"]], on="index", how="left")
    # 缺失坐标的先置 0
    merged["x"] = merged["x"].fillna(0.0)
    merged["y"] = merged["y"].fillna(0.0)
    return merged

def graph_summary() -> dict:
    nodes = load_nodes(); edges = load_edges()
    n = len(nodes); m = len(edges)
    degree_avg = 0 if n==0 else (2*m)/n
    communities = nodes["community"].nunique() if "community" in nodes.columns else 0
    return {
        "nodes": n,
        "edges": m,
        "avg_degree": degree_avg,
        "communities": int(communities),
    }