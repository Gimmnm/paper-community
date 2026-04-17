from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class DemoPaper:
    pid: int
    title: str
    year: int = 0


@dataclass
class DemoCommunity:
    cid: int
    paper_ids: List[int]
    size: int
    center_paper_ids: List[int] = field(default_factory=list)
    bridge_paper_ids: List[int] = field(default_factory=list)
    neighbor_communities: List[Tuple[int, float]] = field(default_factory=list)  # (cid, weight)
    topic_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoCommunityGraph:
    """
    面向网页 demo 的“扁平社区图”数据结构（Milestone A）。

    - 只建模：论文、社区、社区间连接（从 mutual-kNN 边聚合得到）
    - 分层社区相关字段暂不包含（后续 Milestone 再扩展）
    """

    n_papers: int
    n_communities: int
    resolution: float
    papers: Dict[int, DemoPaper]
    communities: Dict[int, DemoCommunity]

    # 论文 -> 社区（pid -> cid）
    paper_to_community: Dict[int, int]

    # 社区图边（无向，key=(min_cid,max_cid)）
    community_edges: Dict[Tuple[int, int], float]


DEFAULT_RESOLUTION_NDIGITS = 4


def _round_resolution(r: float, ndigits: int = DEFAULT_RESOLUTION_NDIGITS) -> float:
    return round(float(r), ndigits)


def membership_filename(resolution: float, ndigits: int = DEFAULT_RESOLUTION_NDIGITS) -> str:
    r = _round_resolution(resolution, ndigits=ndigits)
    return f"membership_r{r:.{ndigits}f}.npy"


def load_membership_for_resolution_light(
    out_dir: Path,
    r_target: float,
    *,
    allow_nearest: bool = True,
    ndigits: int = DEFAULT_RESOLUTION_NDIGITS,
) -> np.ndarray:
    """
    轻量 membership loader：不依赖 igraph/leidenalg，仅从 sweep 输出目录读取 .npy。
    """
    out_dir = Path(out_dir)
    r_target = _round_resolution(r_target, ndigits=ndigits)
    exact_path = out_dir / membership_filename(r_target, ndigits=ndigits)
    if exact_path.exists():
        return np.load(exact_path).astype(np.int32)
    if not allow_nearest:
        raise FileNotFoundError(f"membership not found: {exact_path}")

    summary_path = out_dir / "summary.npy"
    if not summary_path.exists():
        raise FileNotFoundError(f"membership not found: {exact_path}; summary.npy missing under {out_dir}")
    summary = np.load(summary_path, allow_pickle=True).item()
    resolutions = [float(x) for x in np.asarray(summary.get("resolutions", [])).tolist()]
    if not resolutions:
        raise ValueError(f"no resolutions recorded in {summary_path}")
    nearest = min(resolutions, key=lambda x: abs(x - r_target))
    nearest_path = out_dir / membership_filename(nearest, ndigits=ndigits)
    if not nearest_path.exists():
        raise FileNotFoundError(f"nearest membership file missing: {nearest_path}")
    return np.load(nearest_path).astype(np.int32)


def _paper_title(p: object) -> str:
    return str(getattr(p, "name", "") or "").strip()


def _paper_year(p: object) -> int:
    try:
        return int(getattr(p, "year", 0) or 0)
    except Exception:
        return 0


def build_demo_graph_from_membership(
    papers: List[object],
    membership: np.ndarray,
    *,
    resolution: float,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    top_center: int = 8,
    top_bridge: int = 8,
    top_neighbor_comms: int = 12,
    neighbor_weight: str = "sum_w",
) -> DemoCommunityGraph:
    """
    从某个 resolution 的 Leiden membership + mutual-kNN 边表，构建 demo 用的社区图对象。

    参数约定：
    - papers: 1-based 列表（papers[0] is None）
    - membership: shape=(n_papers,), 0-based，对应 papers[1:] 的社区标签
    - u,v,w: mutual-kNN 上三角边表（0-based 节点索引，对应 papers[1:]）
    """
    membership = np.asarray(membership, dtype=np.int32)
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    w = np.asarray(w, dtype=np.float32)

    n_papers = int(membership.size)
    if len(papers) != n_papers + 1:
        raise ValueError(f"papers length mismatch: len(papers)={len(papers)} vs membership={n_papers}")

    # 1) 论文基础信息
    paper_infos: Dict[int, DemoPaper] = {}
    paper_to_comm: Dict[int, int] = {}
    for idx0 in range(n_papers):
        pid = idx0 + 1
        p = papers[pid]
        cid = int(membership[idx0])
        paper_to_comm[pid] = cid
        paper_infos[pid] = DemoPaper(pid=pid, title=_paper_title(p), year=_paper_year(p))

    # 2) 社区 -> 论文列表
    comm_to_papers: Dict[int, List[int]] = {}
    for pid, cid in paper_to_comm.items():
        comm_to_papers.setdefault(cid, []).append(pid)

    # 3) 计算 “中心/桥接” 论文（基于 mutual-kNN 边）
    # within_degree[idx0]：同社区内边数（无向）
    # cross_degree[idx0]：跨社区边数（无向）
    cu = membership[u]
    cv = membership[v]
    same = cu == cv
    cross = ~same

    within_degree = np.zeros(n_papers, dtype=np.int32)
    cross_degree = np.zeros(n_papers, dtype=np.int32)

    if same.any():
        uu = u[same]
        vv = v[same]
        within_degree += np.bincount(uu, minlength=n_papers).astype(np.int32)
        within_degree += np.bincount(vv, minlength=n_papers).astype(np.int32)

    if cross.any():
        uu = u[cross]
        vv = v[cross]
        cross_degree += np.bincount(uu, minlength=n_papers).astype(np.int32)
        cross_degree += np.bincount(vv, minlength=n_papers).astype(np.int32)

    # 4) 聚合社区间边
    # key=(min_cid,max_cid) -> weight
    comm_edges: Dict[Tuple[int, int], float] = {}
    if cross.any():
        a = cu[cross].astype(np.int32)
        b = cv[cross].astype(np.int32)
        ww = w[cross].astype(np.float32)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs = np.stack([lo, hi], axis=1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        if neighbor_weight == "count":
            agg = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
        else:  # sum_w
            agg = np.bincount(inv, weights=ww.astype(np.float64), minlength=uniq.shape[0]).astype(np.float64)
        for (c1, c2), val in zip(uniq.tolist(), agg.tolist()):
            if int(c1) == int(c2):
                continue
            comm_edges[(int(c1), int(c2))] = float(val)

    # 5) 构建社区对象（center/bridge/neighbor）
    communities: Dict[int, DemoCommunity] = {}
    for cid, pids in comm_to_papers.items():
        # 选中心：within_degree 高
        idx0s = np.asarray([pid - 1 for pid in pids], dtype=np.int32)
        wd = within_degree[idx0s]
        bd = cross_degree[idx0s]

        # top center
        k1 = min(int(top_center), int(idx0s.size))
        if k1 > 0:
            top_local = np.argpartition(wd, -k1)[-k1:]
            center_idx0 = idx0s[top_local][np.argsort(wd[top_local])[::-1]]
            center_pids = (center_idx0 + 1).astype(int).tolist()
        else:
            center_pids = []

        # top bridge
        k2 = min(int(top_bridge), int(idx0s.size))
        if k2 > 0:
            top_local = np.argpartition(bd, -k2)[-k2:]
            bridge_idx0 = idx0s[top_local][np.argsort(bd[top_local])[::-1]]
            bridge_pids = (bridge_idx0 + 1).astype(int).tolist()
        else:
            bridge_pids = []

        communities[int(cid)] = DemoCommunity(
            cid=int(cid),
            paper_ids=[int(x) for x in pids],
            size=int(len(pids)),
            center_paper_ids=center_pids,
            bridge_paper_ids=bridge_pids,
        )

    # 6) 给每个社区补邻居社区 topK（从 comm_edges 反向索引）
    nbr: Dict[int, List[Tuple[int, float]]] = {int(cid): [] for cid in communities.keys()}
    for (c1, c2), val in comm_edges.items():
        if c1 in nbr:
            nbr[c1].append((int(c2), float(val)))
        if c2 in nbr:
            nbr[c2].append((int(c1), float(val)))

    for cid, lst in nbr.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        communities[cid].neighbor_communities = lst[: max(int(top_neighbor_comms), 0)]

    return DemoCommunityGraph(
        n_papers=n_papers,
        n_communities=int(len(communities)),
        resolution=float(resolution),
        papers=paper_infos,
        communities=communities,
        paper_to_community=paper_to_comm,
        community_edges=comm_edges,
    )


def summarize_demo_graph(
    g: DemoCommunityGraph,
    *,
    top_n: int = 8,
    papers_lookup: Optional[Dict[int, DemoPaper]] = None,
) -> Dict[str, Any]:
    papers_lookup = papers_lookup or g.papers
    comms = list(g.communities.values())
    comms.sort(key=lambda c: c.size, reverse=True)
    top = comms[: max(int(top_n), 0)]

    def _p_title(pid: int) -> str:
        p = papers_lookup.get(int(pid))
        return "" if p is None else str(p.title)

    return {
        "n_papers": int(g.n_papers),
        "n_communities": int(g.n_communities),
        "resolution": float(g.resolution),
        "community_edges": int(len(g.community_edges)),
        "top_communities_by_size": [
            {
                "cid": int(c.cid),
                "size": int(c.size),
                "center_papers": [{"pid": int(pid), "title": _p_title(pid)} for pid in c.center_paper_ids[:5]],
                "bridge_papers": [{"pid": int(pid), "title": _p_title(pid)} for pid in c.bridge_paper_ids[:5]],
                "neighbor_communities": [{"cid": int(x), "weight": float(w)} for x, w in c.neighbor_communities[:5]],
            }
            for c in top
        ],
    }


def save_demo_graph_json(g: DemoCommunityGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"n_papers": g.n_papers, "n_communities": g.n_communities, "resolution": g.resolution},
        "papers": {str(pid): asdict(p) for pid, p in g.papers.items()},
        "paper_to_community": {str(pid): int(cid) for pid, cid in g.paper_to_community.items()},
        "communities": {str(cid): asdict(c) for cid, c in g.communities.items()},
        "community_edges": [{"a": int(a), "b": int(b), "w": float(w)} for (a, b), w in g.community_edges.items()],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _cli_print_summary(summary: Dict[str, Any]) -> None:
    print(
        f"[demo-graph] papers={summary['n_papers']}  communities={summary['n_communities']}  "
        f"r={summary['resolution']:.4f}  comm_edges={summary['community_edges']}"
    )
    for row in summary.get("top_communities_by_size", []):
        print(f"  - cid={row['cid']:<5d} size={row['size']:<6d}")
        if row["center_papers"]:
            t = row["center_papers"][0]
            print(f"      center:  pid={t['pid']:<6d}  {t['title'][:90]}")
        if row["bridge_papers"]:
            t = row["bridge_papers"][0]
            print(f"      bridge:  pid={t['pid']:<6d}  {t['title'][:90]}")
        if row["neighbor_communities"]:
            t = row["neighbor_communities"][0]
            print(f"      neighbor: cid={t['cid']:<6d} weight={t['weight']:.3f}")


if __name__ == "__main__":  # pragma: no cover
    # 轻量 CLI：用于快速验证 Milestone A 的建模/加载逻辑。
    import argparse

    from community import load_membership_for_resolution
    from core import build_or_load, OUT_DIR
    from network import load_edges_npz

    ap = argparse.ArgumentParser()
    ap.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep_rb"))
    ap.add_argument("--resolution", type=float, required=True)
    ap.add_argument("--graph-npz", type=str, default=str(OUT_DIR / "mutual_knn_k50.npz"))
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--save-json", type=str, default=None)
    args = ap.parse_args()

    _, papers, _ = build_or_load()
    membership = load_membership_for_resolution(Path(args.leiden_dir), float(args.resolution), allow_nearest=True)
    u, v, w, _, _, _ = load_edges_npz(Path(args.graph_npz))
    g = build_demo_graph_from_membership(papers, membership, resolution=float(args.resolution), u=u, v=v, w=w)
    summary = summarize_demo_graph(g, top_n=int(args.top_n))
    _cli_print_summary(summary)
    if args.save_json:
        save_demo_graph_json(g, Path(args.save_json))

