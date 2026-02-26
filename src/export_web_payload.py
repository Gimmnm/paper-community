from __future__ import annotations

"""
将多分辨率主题建模/对齐/命名结果打包为前端可用 JSON
================================================

默认输入（推荐）：
- out/leiden_sweep/membership_r*.npy
- out/umap2d.npy（可选）
- out/graph_drl2d.npy（可选）
- out/topic_modeling_multi/K{K}/aligned_to_r{ref}/r*/communities_topic_weights.csv
- out/topic_modeling_multi/K{K}/aligned_to_r{ref}/labels_ref_r{ref}/  (topic_labeling.py 输出，可选)

输出：
- out/web_payload/K{K}_ref{ref}/index.json
- out/web_payload/K{K}_ref{ref}/resolutions/r{res}.json
"""

from pathlib import Path
import argparse
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
MEM_RE = re.compile(r"membership_r([0-9]+(?:\.[0-9]+)?)\.npy$")


def parse_r_from_mem(p: Path) -> float | None:
    m = MEM_RE.match(p.name)
    return float(m.group(1)) if m else None


def discover_memberships(leiden_dir: Path) -> list[tuple[float, Path]]:
    items = []
    if not leiden_dir.exists():
        return items
    for p in sorted(leiden_dir.glob("membership_r*.npy")):
        r = parse_r_from_mem(p)
        if r is not None:
            items.append((r, p))
    items.sort(key=lambda x: x[0])
    return items


def select_items(items: list[tuple[float, Path]], resolutions: Iterable[float] | None, r_min: float, r_max: float, include: Iterable[float] | None = None):
    include = [float(x) for x in (include or [])]
    if resolutions is not None:
        req = [float(x) for x in resolutions] + include
        out = []
        for rr in req:
            for r, p in items:
                if abs(r - rr) <= 1e-8:
                    out.append((r, p))
                    break
        uniq = {round(r, 10): (r, p) for r, p in out}
        return [uniq[k] for k in sorted(uniq)]
    out = [(r, p) for r, p in items if (r_min - 1e-8) <= r <= (r_max + 1e-8)]
    for x in include:
        for r, p in items:
            if abs(r - x) <= 1e-8:
                out.append((r, p))
                break
    uniq = {round(r, 10): (r, p) for r, p in out}
    return [uniq[k] for k in sorted(uniq)]


def _find_rdir(root: Path, r: float) -> Path | None:
    cands = [root / f"r{r:.4f}", root / f"r{r}"]
    for c in cands:
        if c.exists():
            return c
    # 模糊找一下（目录命名有时保留更多小数）
    if root.exists():
        target = f"{r:.4f}"
        for p in root.iterdir():
            if p.is_dir() and p.name.startswith("r"):
                try:
                    rr = float(p.name[1:])
                    if abs(rr - r) <= 1e-8:
                        return p
                except Exception:
                    pass
    return None


def _to_records_float32(arr: np.ndarray, digits: int = 6):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return [None if not np.isfinite(x) else round(float(x), digits) for x in a.tolist()]
    return [[None if not np.isfinite(x) else round(float(x), digits) for x in row] for row in a.tolist()]


def _safe_num(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        xf = float(x)
        if np.isfinite(xf):
            if abs(xf - round(xf)) < 1e-12:
                return int(round(xf))
            return xf
        return None
    except Exception:
        return x


def _community_centroids(XY: np.ndarray, membership: np.ndarray):
    XY = np.asarray(XY, dtype=float)
    mem = np.asarray(membership, dtype=int)
    uniq, inv, counts = np.unique(mem, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 2), dtype=float)
    np.add.at(sums, inv, XY)
    cent = sums / counts[:, None]
    return uniq.astype(int), cent.astype(float), counts.astype(int)


def _load_label_maps(labels_dir: Path | None):
    by_res = {}
    ref_labels = {}
    if labels_dir is None or not labels_dir.exists():
        return by_res, ref_labels
    p_ref = labels_dir / "topic_labels_reference.json"
    if p_ref.exists():
        try:
            j = json.loads(p_ref.read_text(encoding="utf-8"))
            ref_labels = {int(k): v for k, v in j.get("labels", {}).items()}
        except Exception:
            pass
    p_by = labels_dir / "topic_labels_by_resolution.json"
    if p_by.exists():
        try:
            j = json.loads(p_by.read_text(encoding="utf-8"))
            for rk, m in j.get("topics", {}).items():
                by_res[str(rk)] = {int(k): v for k, v in m.items()}
        except Exception:
            pass
    return by_res, ref_labels


def _read_topics_table_labeled(topic_dir: Path, labels_rdir: Path | None):
    cands = []
    if labels_rdir is not None:
        cands.extend([labels_rdir / "topics_top_words_labeled.csv"])
    cands.extend([topic_dir / "topics_top_words_labeled.csv", topic_dir / "topics_top_words.csv"])
    for p in cands:
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()


def _read_comm_table_labeled(topic_dir: Path, labels_rdir: Path | None):
    cands = []
    if labels_rdir is not None:
        cands.extend([labels_rdir / "communities_topic_weights_labeled.csv"])
    cands.extend([topic_dir / "communities_topic_weights_labeled.csv", topic_dir / "communities_topic_weights.csv"])
    for p in cands:
        if p.exists():
            return pd.read_csv(p, encoding="utf-8-sig")
    return pd.DataFrame()


def build_argparser():
    p = argparse.ArgumentParser(description="打包前端 JSON（多分辨率主题/社区）")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--ref-resolution", type=float, default=1.0)
    p.add_argument("--leiden-dir", type=str, default=None)
    p.add_argument("--topic-root", type=str, default=None, help="建议指向 aligned_to_rXXXX 目录")
    p.add_argument("--labels-dir", type=str, default=None, help="topic_labeling.py 输出 labels_ref_rXXXX 目录（可选）")
    p.add_argument("--out-root", type=str, default=None)

    p.add_argument("--r-min", type=float, default=0.0001)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--include", type=float, nargs="*", default=None)
    p.add_argument("--resolutions", type=float, nargs="*", default=None)

    p.add_argument("--umap", type=str, default=None, help="默认 out/umap2d.npy")
    p.add_argument("--graph", type=str, default=None, help="默认 out/graph_drl2d.npy")
    p.add_argument("--include-paper-points", action="store_true", help="在每个分辨率 JSON 中写出论文点（可能很大）")
    p.add_argument("--paper-sample", type=int, default=0, help="若>0，随机采样论文点数量（与 --include-paper-points 配合）")
    p.add_argument("--round-digits", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    verbose = not args.quiet

    leiden_dir = Path(args.leiden_dir) if args.leiden_dir else (OUT_DIR / "leiden_sweep")
    topic_root_default = OUT_DIR / "topic_modeling_multi" / f"K{args.k}" / f"aligned_to_r{args.ref_resolution:.4f}"
    topic_root = Path(args.topic_root) if args.topic_root else topic_root_default
    labels_dir = Path(args.labels_dir) if args.labels_dir else (topic_root / f"labels_ref_r{args.ref_resolution:.4f}")
    out_root = Path(args.out_root) if args.out_root else (OUT_DIR / "web_payload" / f"K{args.k}_ref{args.ref_resolution:.4f}")

    umap_path = Path(args.umap) if args.umap else (OUT_DIR / "umap2d.npy")
    graph_path = Path(args.graph) if args.graph else (OUT_DIR / "graph_drl2d.npy")

    out_res_dir = out_root / "resolutions"
    out_res_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_memberships(leiden_dir)
    if not discovered:
        raise FileNotFoundError(f"未发现 membership 文件: {leiden_dir}")
    selected = select_items(discovered, args.resolutions, args.r_min, args.r_max, args.include)
    if not selected:
        raise RuntimeError("筛选后无可用分辨率")

    XY_umap = np.load(umap_path) if umap_path.exists() else None
    XY_graph = np.load(graph_path) if graph_path.exists() else None
    if XY_umap is None and XY_graph is None:
        raise FileNotFoundError("至少需要 out/umap2d.npy 或 out/graph_drl2d.npy")

    n_papers = int((XY_umap if XY_umap is not None else XY_graph).shape[0])

    labels_by_res, ref_labels = _load_label_maps(labels_dir if labels_dir.exists() else None)

    if verbose:
        print(f"[web] leiden_dir={leiden_dir}")
        print(f"[web] topic_root={topic_root}")
        print(f"[web] labels_dir={'None' if not labels_dir.exists() else labels_dir}")
        print(f"[web] selected={len(selected)}")
        print(f"[web] out_root={out_root}")

    index_rows = []
    topic_color_palette = [  # 与 tab20 前20色近似一致的离散色（便于前端直接用）
        "#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896",
        "#9467bd","#c5b0d5","#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7",
        "#bcbd22","#dbdb8d","#17becf","#9edae5"
    ]
    rng = np.random.default_rng(42)

    for r, mem_path in selected:
        mem = np.load(mem_path)
        if mem.shape[0] == n_papers + 1:
            mem = mem[1:]
        if mem.shape[0] != n_papers:
            if verbose:
                print(f"[web] skip r={r:.4f}: membership len {mem.shape[0]} != n_papers {n_papers}")
            continue

        rdir = _find_rdir(topic_root, r)
        if rdir is None:
            if verbose:
                print(f"[web] skip r={r:.4f}: no topic result dir in {topic_root}")
            continue

        labels_rdir = _find_rdir(labels_dir, r) if labels_dir.exists() else None
        dfc = _read_comm_table_labeled(rdir, labels_rdir)
        dft = _read_topics_table_labeled(rdir, labels_rdir)

        if dfc.empty or "community_id" not in dfc.columns:
            if verbose:
                print(f"[web] skip r={r:.4f}: missing community table")
            continue

        # 标准化字段
        d = dfc.copy()
        for col in ["community_id", "top1_topic", "top2_topic"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        for col in ["top1_weight", "top2_weight"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        # community_id -> 行
        d["_cid_int"] = pd.to_numeric(d["community_id"], errors="coerce").astype("Int64")
        d = d[d["_cid_int"].notna()].copy()
        d["_cid_int"] = d["_cid_int"].astype(int)
        crows = {int(row["_cid_int"]): row for _, row in d.iterrows()}

        uniq, cent_umap, counts = _community_centroids(XY_umap if XY_umap is not None else XY_graph, mem)
        cent_graph = None
        if XY_graph is not None:
            _, cent_graph, _ = _community_centroids(XY_graph, mem)

        # 主题表
        topics = []
        topic_cols = [c for c in d.columns if re.match(r"^topic[_\-]?\d+$", str(c))]
        for _, row in dft.iterrows():
            tid = row.get("topic_id", None)
            try:
                tid = int(pd.to_numeric(tid, errors="coerce"))
            except Exception:
                continue
            label_obj = None
            key_res = f"{r:.4f}"
            if key_res in labels_by_res and tid in labels_by_res[key_res]:
                label_obj = labels_by_res[key_res][tid]
            elif tid in ref_labels:
                label_obj = ref_labels[tid]
            topics.append({
                "topic_id": tid,
                "ref_topic": None if label_obj is None else _safe_num(label_obj.get("ref_topic")),
                "label_zh": "" if label_obj is None else str(label_obj.get("label_zh", "")),
                "label_en": "" if label_obj is None else str(label_obj.get("label_en", "")),
                "top_words": str(row.get("top_words", "")) if "top_words" in dft.columns else "",
                "color": topic_color_palette[tid % len(topic_color_palette)],
            })

        # 社区记录
        comm_records = []
        c2top1 = {}
        c2w1 = {}
        for i, cid in enumerate(uniq.tolist()):
            row = crows.get(int(cid))
            top1 = _safe_num(row.get("top1_topic")) if row is not None and "top1_topic" in d.columns else None
            top2 = _safe_num(row.get("top2_topic")) if row is not None and "top2_topic" in d.columns else None
            top1_w = _safe_num(row.get("top1_weight")) if row is not None and "top1_weight" in d.columns else None
            top2_w = _safe_num(row.get("top2_weight")) if row is not None and "top2_weight" in d.columns else None
            c2top1[int(cid)] = int(top1) if top1 is not None else -1
            c2w1[int(cid)] = float(top1_w) if top1_w is not None else None

            topic_weights = None
            if topic_cols and row is not None:
                # 统一按 topic_id 顺序输出
                kv = []
                for col in sorted(topic_cols, key=lambda x: int(re.findall(r"(\d+)$", str(x))[0])):
                    kv.append(_safe_num(row.get(col)))
                topic_weights = kv

            comm_records.append({
                "community_id": int(cid),
                "size": int(counts[i]),
                "centroid_umap": None if XY_umap is None else [round(float(cent_umap[i,0]), args.round_digits), round(float(cent_umap[i,1]), args.round_digits)],
                "centroid_graph": None if cent_graph is None else [round(float(cent_graph[i,0]), args.round_digits), round(float(cent_graph[i,1]), args.round_digits)],
                "top1_topic": top1,
                "top1_weight": top1_w,
                "top2_topic": top2,
                "top2_weight": top2_w,
                "top1_label_zh": ("" if row is None else str(row.get("top1_label_zh", ""))),
                "top1_label_en": ("" if row is None else str(row.get("top1_label_en", ""))),
                "top2_label_zh": ("" if row is None else str(row.get("top2_label_zh", ""))),
                "top2_label_en": ("" if row is None else str(row.get("top2_label_en", ""))),
                "community_label": ("" if row is None else str(row.get("community_label", ""))),
                "top1_keywords": ("" if row is None else str(row.get("top1_keywords", ""))),
                "top2_keywords": ("" if row is None else str(row.get("top2_keywords", ""))),
                "rep_papers": ("" if row is None else str(row.get("rep_papers", ""))),
                "topic_weights": topic_weights,
            })

        # 汇总指标
        w1_vals = [x for x in c2w1.values() if x is not None and np.isfinite(x)]
        mean_top1 = float(np.mean(w1_vals)) if w1_vals else None
        n_comm = len(comm_records)

        # 可选论文点
        paper_points = None
        if args.include_paper_points:
            idx = np.arange(n_papers)
            if args.paper_sample and args.paper_sample > 0 and args.paper_sample < n_papers:
                idx = np.sort(rng.choice(idx, size=int(args.paper_sample), replace=False))
            XY = XY_umap if XY_umap is not None else XY_graph
            paper_points = []
            for i in idx.tolist():
                cid = int(mem[i])
                t1 = c2top1.get(cid, -1)
                w1 = c2w1.get(cid, None)
                rec = {
                    "paper_index": int(i),
                    "community_id": cid,
                    "top1_topic": (None if t1 < 0 else int(t1)),
                    "top1_weight": (None if w1 is None else round(float(w1), args.round_digits)),
                    "xy": [round(float(XY[i,0]), args.round_digits), round(float(XY[i,1]), args.round_digits)],
                }
                if XY_umap is not None and XY_graph is not None:
                    rec["xy_umap"] = [round(float(XY_umap[i,0]), args.round_digits), round(float(XY_umap[i,1]), args.round_digits)]
                    rec["xy_graph"] = [round(float(XY_graph[i,0]), args.round_digits), round(float(XY_graph[i,1]), args.round_digits)]
                paper_points.append(rec)

        # 写分辨率 JSON
        payload = {
            "resolution": round(float(r), 6),
            "k": int(args.k),
            "n_papers": int(n_papers),
            "n_communities": int(n_comm),
            "mean_top1_weight": None if mean_top1 is None else round(mean_top1, args.round_digits),
            "coords_available": {
                "umap": bool(XY_umap is not None),
                "graph": bool(XY_graph is not None),
            },
            "topics": topics,
            "communities": comm_records,
            "paper_points": paper_points,
            "meta": {
                "membership_file": str(mem_path),
                "topic_dir": str(rdir),
                "labels_dir_r": (None if labels_rdir is None else str(labels_rdir)),
            }
        }
        out_json = out_res_dir / f"r{r:.4f}.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

        index_rows.append({
            "resolution": round(float(r), 6),
            "json": f"resolutions/{out_json.name}",
            "n_communities": int(n_comm),
            "mean_top1_weight": (None if mean_top1 is None else round(mean_top1, args.round_digits)),
        })

        if verbose:
            print(f"[web] r={r:.4f} -> {out_json.name} | C={n_comm}")

    index_payload = {
        "k": int(args.k),
        "reference_resolution": float(args.ref_resolution),
        "resolutions": sorted(index_rows, key=lambda x: x["resolution"]),
        "palette": topic_color_palette,
        "paths": {
            "topic_root": str(topic_root),
            "labels_dir": str(labels_dir) if labels_dir.exists() else None,
            "leiden_dir": str(leiden_dir),
            "umap": str(umap_path) if umap_path.exists() else None,
            "graph": str(graph_path) if graph_path.exists() else None,
        }
    }
    (out_root / "index.json").write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[web] =============================")
    print(f"[web] outputs -> {out_root}")
    print(f"[web] index   -> {out_root / 'index.json'}")
    print(f"[web] n_res   -> {len(index_rows)}")


if __name__ == "__main__":
    main()
