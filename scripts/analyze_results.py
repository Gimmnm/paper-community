#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析社区发现结果：
- 读 papers.csv（包含 is_AP/is_NA / field / field_multi）
- 读 communities.csv（node,index,community,...）
- 可选读 graph.gexf 计算 modularity / 边同质性
- 计算：NMI、ARI、macro-F1(多数投票)、社区规模分布、标签覆盖率等
- 导出：report.json、crosstab.csv、community_sizes.csv、nodes_labeled.csv

用法：
  python scripts/analyze_results.py \
    --papers data/papers.csv \
    --communities data/graph/communities.csv \
    --graph data/graph/graph.gexf \
    --outdir outputs/analysis
"""
from __future__ import annotations

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    f1_score,
)


# ----------------------- utils -----------------------
def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def derive_fourclass(df: pd.DataFrame) -> pd.Series:
    """
    用 is_AP/is_NA 导出四类：AP-only / NA-only / AP+NA / Other
    若无 is_AP/is_NA，则尝试 field / field_multi 回退。
    返回：字符串标签的 Series（名称 'four'）
    """
    if "is_AP" in df.columns or "is_NA" in df.columns:
        a = df.get("is_AP", 0)
        n = df.get("is_NA", 0)
        a = (a.fillna(0).astype(int) > 0).values
        n = (n.fillna(0).astype(int) > 0).values
        out = np.array(["Other"] * len(df), dtype=object)
        out[(a) & (~n)] = "AP-only"
        out[(~a) & (n)] = "NA-only"
        out[(a) & (n)] = "AP+NA"
        return pd.Series(out, name="four")

    if "field" in df.columns:
        f = df["field"].fillna("").astype(str)
        out = f.copy()
        out[(f == "AP")] = "AP-only"
        out[(f == "NA")] = "NA-only"
        out[(f == "AP_NA")] = "AP+NA"
        out[~out.isin(["AP-only", "NA-only", "AP+NA"])] = "Other"
        out.name = "four"
        return out

    if "field_multi" in df.columns:
        vals = df["field_multi"].fillna("").astype(str).tolist()
        out = []
        for s in vals:
            sset = {x.strip() for x in s.split(";") if x.strip()}
            ap = "AP" in sset
            na = "NA" in sset
            if ap and na:
                out.append("AP+NA")
            elif ap:
                out.append("AP-only")
            elif na:
                out.append("NA-only")
            else:
                out.append("Other")
        return pd.Series(out, name="four")

    return pd.Series(["Other"] * len(df), name="four")


def majority_label_f1(y_true: np.ndarray, comm: np.ndarray) -> float:
    """
    按社区多数标签作为预测，计算 Macro-F1。
    y_true 为字符串标签（AP-only / NA-only / AP+NA / Other），comm 为整数社区编号。
    """
    y_pred = np.empty_like(y_true, dtype=object)
    for c in np.unique(comm):
        idx = np.where(comm == c)[0]
        if len(idx) == 0:
            continue
        vals, cnts = np.unique(y_true[idx], return_counts=True)
        y_pred[idx] = vals[np.argmax(cnts)]
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def community_stats(comm: np.ndarray) -> pd.DataFrame:
    cnt = Counter(comm)
    df = pd.DataFrame({"community": list(cnt.keys()), "size": list(cnt.values())})
    df = df.sort_values("size", ascending=False).reset_index(drop=True)
    return df


def edge_homophily(G, node_label: Dict, weighted: bool = False, weight_key: str = "weight") -> float:
    """
    边同质性：端点标签一致的边（权重和）/ 全部边（权重和）
    node_label: {node: "AP-only"/"NA-only"/"AP+NA"/"Other"}
    """
    same, tot = 0.0, 0.0
    for u, v, data in G.edges(data=True):
        if (u in node_label) and (v in node_label):
            w = float(data.get(weight_key, 1.0)) if weighted else 1.0
            tot += w
            if node_label[u] == node_label[v]:
                same += w
    return float(same / tot) if tot > 0 else 0.0


# ----------------------- main -----------------------
def analyze(
    papers_csv: str,
    communities_csv: str,
    graph_gexf: Optional[str],
    outdir: str,
    debug: bool = False,
) -> Dict:
    ensure_outdir(outdir)

    # 1) 读数据
    papers = pd.read_csv(papers_csv)
    if "index" not in papers.columns:
        papers = papers.copy()
        papers["index"] = np.arange(len(papers))

    comm = pd.read_csv(communities_csv)

    # 列名兼容：paper_id / id
    if "paper_id" not in comm.columns and "id" in comm.columns:
        comm = comm.rename(columns={"id": "paper_id"})

    # 2) 对齐键：index
    if "index" not in comm.columns:
        if "node" in comm.columns:
            try:
                comm["index"] = comm["node"].astype(int)
            except Exception as e:
                raise ValueError(
                    "communities.csv 缺少 index 列，且 node 无法转换为整数。"
                    "请确保 run_community.py 写入了 index。"
                ) from e
        else:
            raise ValueError("communities.csv 缺少 index 列。")

    # 基本清洗
    n = len(papers)
    comm = comm.copy()
    comm["index"] = comm["index"].astype(int)
    comm = comm[(comm["index"] >= 0) & (comm["index"] < n)]
    # 有些工具会把社区编号存成 float
    comm["community"] = comm["community"].astype(float).astype(int)

    # 3) 四类标签（字符串）
    four = derive_fourclass(papers).astype(str)
    y_true = four.values  # 字符串数组
    four_names = ["AP-only", "NA-only", "AP+NA", "Other"]
    label_coverage = float((four != "Other").mean())

    # 4) y_pred（社区编号），未覆盖置 -1
    y_pred = np.full(n, -1, dtype=int)
    y_pred[comm["index"].values] = comm["community"].values

    # 覆盖率（被分到社区的比例）
    mask = (y_pred != -1)
    coverage = float(mask.mean())

    if debug:
        print("[debug] papers len:", n)
        print("[debug] unique four:", dict(Counter(y_true)))
        print("[debug] y_pred==-1:", int((y_pred == -1).sum()))

    # 5) 评估（覆盖子集）
    if mask.sum() > 0:
        nmi = float(normalized_mutual_info_score(y_true[mask], y_pred[mask]))
        ari = float(adjusted_rand_score(y_true[mask], y_pred[mask]))
        macro_f1 = float(majority_label_f1(y_true[mask], y_pred[mask]))
    else:
        nmi = ari = macro_f1 = float("nan")

    # 6) 社区规模
    sizes_df = community_stats(y_pred[mask])
    sizes_path = os.path.join(outdir, "community_sizes.csv")
    sizes_df.to_csv(sizes_path, index=False)

    # 7) 交叉表（四类 vs 社区）—— 直接用字符串四类
    ct = pd.crosstab(
        pd.Series(y_true[mask], name="four"),
        pd.Series(y_pred[mask], name="community"),
        normalize=False,
    )
    # 固定行顺序、列排序
    ct = ct.reindex(index=[x for x in four_names if x in ct.index], fill_value=0)
    ct = ct.reindex(columns=sorted(ct.columns), fill_value=0)
    ct_path = os.path.join(outdir, "crosstab.csv")
    ct.to_csv(ct_path)

    # 8) 可选：modularity / 边同质性
    modularity = None
    homophily_unweighted = None
    homophily_weighted = None
    if graph_gexf is not None and os.path.exists(graph_gexf):
        import networkx as nx
        from networkx.algorithms.community.quality import modularity as nx_modularity

        G = nx.read_gexf(graph_gexf)

        # 聚合社区集合（只加入覆盖子集中的节点）
        comm_groups = defaultdict(set)
        node_label_for_hom = {}
        for n_node, data in G.nodes(data=True):
            # index 属性（build_graph / run_community 写入）
            idx = data.get("index")
            if idx is None:
                try:
                    idx = int(n_node)
                except Exception:
                    continue
            idx = int(idx)
            if idx < 0 or idx >= n:
                continue
            c = y_pred[idx]
            if c == -1:
                continue
            comm_groups[int(c)].add(n_node)
            node_label_for_hom[n_node] = y_true[idx]

        comm_list = list(comm_groups.values())
        if len(comm_list) >= 2:
            modularity = float(nx_modularity(G, comm_list, weight="weight"))
        else:
            modularity = float("nan")

        homophily_unweighted = float(edge_homophily(G, node_label_for_hom, weighted=False))
        homophily_weighted = float(edge_homophily(G, node_label_for_hom, weighted=True, weight_key="weight"))

    # 9) 标注后的节点（便于前端/排查）
    nodes_labeled_path = os.path.join(outdir, "nodes_labeled.csv")
    keep_cols = ["index"]
    for c in ["id", "paper_id", "title", "authors", "field", "field_multi", "is_AP", "is_NA"]:
        if c in papers.columns:
            keep_cols.append(c)
    out_df = papers[keep_cols].copy()
    out_df["community"] = y_pred
    out_df["four"] = y_true
    out_df.to_csv(nodes_labeled_path, index=False)

    # 10) 汇总报告
    report = {
        "coverage": coverage,                       # 样本被社区覆盖的比例
        "label_coverage": label_coverage,           # 四类标签非 Other 的比例
        "nmi": nmi,                                 # 四类 vs 社区
        "ari": ari,
        "macro_f1_majority": macro_f1,
        "modularity": modularity,                   # 需 --graph
        "edge_homophily_unweighted": homophily_unweighted,  # 需 --graph
        "edge_homophily_weighted": homophily_weighted,      # 需 --graph
        "num_communities": int(sizes_df.shape[0]),
        "top_sizes": sizes_df.head(10).to_dict(orient="records"),
        "paths": {
            "community_sizes_csv": sizes_path,
            "crosstab_csv": ct_path,
            "nodes_labeled_csv": nodes_labeled_path,
        },
    }

    report_path = os.path.join(outdir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[ok] report -> {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=str, default="data/papers.csv")
    ap.add_argument("--communities", type=str, default="data/graph/communities.csv")
    ap.add_argument("--graph", type=str, default=None, help="可选：data/graph/graph.gexf，用于 modularity/同质性")
    ap.add_argument("--outdir", type=str, default="outputs/analysis")
    ap.add_argument("--debug", action="store_true", help="打印额外调试信息")
    args = ap.parse_args()

    analyze(args.papers, args.communities, args.graph, args.outdir, debug=args.debug)


if __name__ == "__main__":
    main()