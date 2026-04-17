from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from demo_graph import (
    DemoCommunityGraph,
    load_membership_for_resolution_light,
)
from model import Paper
from network import load_edges_npz
from project_paths import data_source_paths


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _paper_title(p: object) -> str:
    return str(getattr(p, "name", "") or "").strip()


def _paper_abs(p: object) -> str:
    return str(getattr(p, "abstract", "") or "").strip()


def _paper_year(p: object) -> int:
    return _safe_int(getattr(p, "year", 0) or 0)


def _make_snippet(text: str, query: str, max_chars: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    q = str(query or "").strip().lower()
    if not q:
        return text[:max_chars]
    pos = text.lower().find(q)
    if pos < 0:
        return text[:max_chars]
    start = max(0, pos - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]


@dataclass(frozen=True)
class SearchHit:
    kind: str  # "paper" | "community"
    id: str
    score: float
    payload: Dict[str, Any]


def _result(
    typ: str,
    query: Dict[str, Any],
    hits: List[SearchHit],
    *,
    graph_snippet: Optional[Dict[str, Any]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "type": str(typ),
        "query": dict(query),
        "hits": [
            {
                "kind": h.kind,
                "id": h.id,
                "score": float(h.score),
                "payload": h.payload,
            }
            for h in hits
        ],
        "graph_snippet": graph_snippet or {"nodes": [], "edges": []},
        "debug": debug or {},
    }


class DemoAssets:
    """
    Milestone B 的“查询资产”聚合器：加载并缓存 demo 需要的索引/边表。

    设计原则：
    - 不依赖 igraph/leidenalg/torch
    - 尽量复用现有 out/ 产物
    """

    def __init__(
        self,
        *,
        base_dir: Path,
        leiden_dir: Path,
        resolution: float,
        graph_npz: Path,
        keyword_index_dir: Path,
        coords_2d_path: Optional[Path] = None,
    ):
        self.base_dir = Path(base_dir)
        self.leiden_dir = Path(leiden_dir)
        self.resolution = float(resolution)
        self.graph_npz = Path(graph_npz)
        self.keyword_index_dir = Path(keyword_index_dir)
        self.coords_2d_path = None if coords_2d_path is None else Path(coords_2d_path)

        # lazy-loaded
        self._u: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._w: Optional[np.ndarray] = None
        self._n_nodes: Optional[int] = None

        self._csr_indptr: Optional[np.ndarray] = None
        self._csr_indices: Optional[np.ndarray] = None
        self._csr_data: Optional[np.ndarray] = None

        self._vectorizer: Any = None
        self._X_tfidf: Any = None

        self._coords2d: Optional[np.ndarray] = None

    def load_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if self._u is not None:
            assert self._v is not None and self._w is not None and self._n_nodes is not None
            return self._u, self._v, self._w, self._n_nodes
        u, v, w, n_nodes, _, _ = load_edges_npz(self.graph_npz)
        self._u = u
        self._v = v
        self._w = w
        self._n_nodes = int(n_nodes)
        return u, v, w, int(n_nodes)

    def load_membership(self) -> np.ndarray:
        return load_membership_for_resolution_light(self.leiden_dir, self.resolution, allow_nearest=True)

    def ensure_csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        把无向边表（上三角 u,v,w）构建成 CSR（对称），便于按论文取邻居。
        返回 (indptr, indices, data)。
        """
        if self._csr_indptr is not None:
            assert self._csr_indices is not None and self._csr_data is not None
            return self._csr_indptr, self._csr_indices, self._csr_data

        u, v, w, n = self.load_edges()
        rows = np.concatenate([u, v]).astype(np.int32, copy=False)
        cols = np.concatenate([v, u]).astype(np.int32, copy=False)
        data = np.concatenate([w, w]).astype(np.float32, copy=False)

        # 手写 CSR（避免强依赖 scipy）：先按 row 排序，再建 indptr
        order = np.lexsort((cols, rows))
        rows = rows[order]
        cols = cols[order]
        data = data[order]

        counts = np.bincount(rows, minlength=int(n)).astype(np.int64)
        indptr = np.zeros(int(n) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        self._csr_indptr = indptr
        self._csr_indices = cols.astype(np.int32, copy=False)
        self._csr_data = data.astype(np.float32, copy=False)
        return self._csr_indptr, self._csr_indices, self._csr_data

    def neighbors_of_paper0(self, idx0: int, *, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        输入 0-based paper index，返回邻居 0-based index + weight（相似度）topK。
        """
        indptr, indices, data = self.ensure_csr()
        idx0 = int(idx0)
        start = int(indptr[idx0])
        end = int(indptr[idx0 + 1])
        if end <= start:
            return []
        nb = indices[start:end]
        ww = data[start:end]
        if nb.size <= top_k:
            order = np.argsort(ww)[::-1]
            return [(int(nb[i]), float(ww[i])) for i in order.tolist()]
        k = int(top_k)
        top = np.argpartition(ww, -k)[-k:]
        top = top[np.argsort(ww[top])[::-1]]
        return [(int(nb[i]), float(ww[i])) for i in top.tolist()]

    def load_keyword_index(self) -> Tuple[Any, Any]:
        """
        读取 out/keyword_index 下的 TF-IDF 资产：
        - vectorizer.pkl
        - tfidf_docs.npz
        """
        if self._vectorizer is not None and self._X_tfidf is not None:
            return self._vectorizer, self._X_tfidf

        vec_path = self.keyword_index_dir / "vectorizer.pkl"
        mat_path = self.keyword_index_dir / "tfidf_docs.npz"

        if not vec_path.exists() or not mat_path.exists():
            raise FileNotFoundError(f"keyword index missing under {self.keyword_index_dir} (need vectorizer.pkl & tfidf_docs.npz)")

        try:
            from scipy import sparse  # type: ignore
        except Exception as e:
            raise ImportError("需要 scipy 才能读取 tfidf_docs.npz（pip install scipy）") from e

        with vec_path.open("rb") as f:
            vectorizer = pickle.load(f)
        X = sparse.load_npz(mat_path).tocsr()
        self._vectorizer = vectorizer
        self._X_tfidf = X
        return vectorizer, X

    def load_coords2d(self) -> Optional[np.ndarray]:
        """
        读取二维坐标（通常是 out/umap2d.npy），shape=(n_papers, 2)，对应 pid=1..N 的 0-based 索引。
        若未配置或文件不存在，则返回 None。
        """
        if self._coords2d is not None:
            return self._coords2d
        if self.coords_2d_path is None:
            return None
        p = Path(self.coords_2d_path)
        if not p.exists():
            return None
        arr = np.load(p)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"coords2d must have shape (N,2), got {arr.shape} at {p}")
        self._coords2d = arr
        return self._coords2d


def build_demo_assets_and_graph(
    *,
    base_dir: Path,
    papers: Sequence[Optional[Paper]],
    leiden_dir: Path,
    resolution: float,
    graph_npz: Path,
    keyword_index_dir: Path,
    coords_2d_path: Optional[Path] = None,
    top_center: int = 8,
    top_bridge: int = 8,
    top_neighbor_comms: int = 12,
) -> Tuple[DemoAssets, DemoCommunityGraph]:
    from demo_graph import build_demo_graph_from_membership

    assets = DemoAssets(
        base_dir=base_dir,
        leiden_dir=leiden_dir,
        resolution=float(resolution),
        graph_npz=graph_npz,
        keyword_index_dir=keyword_index_dir,
        coords_2d_path=coords_2d_path,
    )
    membership = assets.load_membership()
    u, v, w, _ = assets.load_edges()
    g = build_demo_graph_from_membership(
        list(papers),
        membership,
        resolution=float(resolution),
        u=u,
        v=v,
        w=w,
        top_center=top_center,
        top_bridge=top_bridge,
        top_neighbor_comms=top_neighbor_comms,
    )
    return assets, g


def search_keyword(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    query: str,
    top_k: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    关键词检索：
    - 优先使用 out/keyword_index 的 TF-IDF
    - 若缺依赖/索引，则回退到 title/abstract substring（速度会慢，但可用）
    """
    q = str(query or "").strip()
    offset = max(int(offset), 0)
    if not q:
        return _result("keyword", {"q": q, "top_k": int(top_k), "offset": int(offset)}, [], debug={"mode": "empty"})

    # TF-IDF 模式
    tfidf_error: Optional[str] = None
    try:
        vectorizer, X = assets.load_keyword_index()
        qv = vectorizer.transform([q])
        scores = (X @ qv.T).toarray().ravel().astype(np.float32)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return _result(
                "keyword",
                {"q": q, "top_k": int(top_k), "offset": int(offset)},
                [],
                debug={"mode": "tfidf", "hits": 0},
            )
        k_take = min(int(top_k) + int(offset), int(nz.size))
        top_local = np.argpartition(scores[nz], -k_take)[-k_take:]
        idx0_all = nz[top_local]
        idx0_all = idx0_all[np.argsort(scores[idx0_all])[::-1]].astype(np.int32, copy=False)
        idx0 = idx0_all[int(offset) : int(offset) + int(top_k)]

        hits: List[SearchHit] = []
        for doc_idx0 in idx0.tolist():
            pid = int(doc_idx0 + 1)
            p = papers[pid]
            title = _paper_title(p)
            abstract = _paper_abs(p)
            cid = graph.paper_to_community.get(pid)
            bonus = 0.0
            if q.lower() in title.lower():
                bonus += 0.15
            hits.append(
                SearchHit(
                    kind="paper",
                    id=str(pid),
                    score=float(scores[doc_idx0] + bonus),
                    payload={
                        "pid": pid,
                        "title": title,
                        "year": _paper_year(p),
                        "community": None if cid is None else int(cid),
                        "snippet": _make_snippet(abstract, q),
                    },
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return _result("keyword", {"q": q, "top_k": int(top_k), "offset": int(offset)}, hits, debug={"mode": "tfidf"})
    except Exception as e:
        tfidf_error = f"{type(e).__name__}: {e}"

    # substring fallback
    q_low = q.lower()
    scored: List[Tuple[float, int]] = []
    start = 1 if papers and papers[0] is None else 0
    for pid in range(start, len(papers)):
        p = papers[pid]
        title = _paper_title(p)
        abstract = _paper_abs(p)
        hay = (title + "\n" + abstract).lower()
        if q_low not in hay:
            continue
        score = 2.0 if q_low in title.lower() else 1.0
        scored.append((score, pid))
    scored.sort(reverse=True)
    scored = scored[int(offset) : int(offset) + int(top_k)]

    hits = []
    for score, pid in scored:
        p = papers[pid]
        cid = graph.paper_to_community.get(int(pid))
        hits.append(
            SearchHit(
                kind="paper",
                id=str(pid),
                score=float(score),
                payload={
                    "pid": int(pid),
                    "title": _paper_title(p),
                    "year": _paper_year(p),
                    "community": None if cid is None else int(cid),
                    "snippet": _make_snippet(_paper_abs(p), q),
                },
            )
        )
    dbg: Dict[str, Any] = {"mode": "substring"}
    if tfidf_error:
        dbg["tfidf_error"] = tfidf_error
    return _result("keyword", {"q": q, "top_k": int(top_k), "offset": int(offset)}, hits, debug=dbg)


def lookup_community(
    *,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    cid: int,
    top_papers: int = 20,
    top_neighbors: int = 12,
) -> Dict[str, Any]:
    cid = int(cid)
    comm = graph.communities.get(cid)
    if comm is None:
        return _result("community", {"cid": cid}, [], debug={"error": "community_not_found"})

    def _p_payload(pid: int) -> Dict[str, Any]:
        p = papers[int(pid)]
        return {"pid": int(pid), "title": _paper_title(p), "year": _paper_year(p)}

    # top papers：优先用中心论文，否则取前 top_papers 个
    pids = comm.center_paper_ids[:] if comm.center_paper_ids else comm.paper_ids[:]
    pids = pids[: int(top_papers)]

    payload = {
        "cid": int(comm.cid),
        "size": int(comm.size),
        "topic_info": dict(comm.topic_info or {}),
        "center_papers": [_p_payload(pid) for pid in comm.center_paper_ids[: int(top_papers)]],
        "bridge_papers": [_p_payload(pid) for pid in comm.bridge_paper_ids[: int(top_papers)]],
        "example_papers": [_p_payload(pid) for pid in pids],
        "neighbor_communities": [
            {"cid": int(x), "weight": float(w), "size": int(graph.communities.get(int(x)).size) if int(x) in graph.communities else None}
            for x, w in (comm.neighbor_communities[: int(top_neighbors)] if comm.neighbor_communities else [])
        ],
    }

    hits = [SearchHit(kind="community", id=str(cid), score=1.0, payload=payload)]
    return _result("community", {"cid": cid}, hits)


def lookup_paper(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    pid: int,
    k_neighbors: int = 20,
    k_neighbors_in_comm: int = 10,
    k_neighbor_comms: int = 8,
) -> Dict[str, Any]:
    pid = int(pid)
    if pid <= 0 or pid >= len(papers):
        return _result("paper", {"pid": pid}, [], debug={"error": "paper_id_out_of_range"})
    p = papers[pid]
    if p is None:
        return _result("paper", {"pid": pid}, [], debug={"error": "paper_not_found"})

    cid = graph.paper_to_community.get(pid)
    comm = None if cid is None else graph.communities.get(int(cid))

    # neighbors in graph
    nb = assets.neighbors_of_paper0(pid - 1, top_k=int(k_neighbors))

    # neighbors in same community
    nb_in_comm: List[Tuple[int, float]] = []
    if comm is not None:
        cset = set(int(x) for x in comm.paper_ids)
        for idx0, sim in nb:
            nb_pid = int(idx0 + 1)
            if nb_pid in cset:
                nb_in_comm.append((nb_pid, float(sim)))
            if len(nb_in_comm) >= int(k_neighbors_in_comm):
                break

    neighbor_papers = [
        {
            "pid": int(idx0 + 1),
            "score": float(sim),
            "title": _paper_title(papers[int(idx0 + 1)]),
            "year": _paper_year(papers[int(idx0 + 1)]),
            "community": int(graph.paper_to_community.get(int(idx0 + 1), -1)),
        }
        for idx0, sim in nb[: int(k_neighbors)]
    ]

    payload = {
        "pid": int(pid),
        "title": _paper_title(p),
        "year": _paper_year(p),
        "abstract": _paper_abs(p),
        "community": None if cid is None else int(cid),
        "community_summary": None
        if comm is None
        else {
            "cid": int(comm.cid),
            "size": int(comm.size),
            "center_papers": [
                {"pid": int(x), "title": _paper_title(papers[int(x)]), "year": _paper_year(papers[int(x)])}
                for x in comm.center_paper_ids[:6]
            ],
            "bridge_papers": [
                {"pid": int(x), "title": _paper_title(papers[int(x)]), "year": _paper_year(papers[int(x)])}
                for x in comm.bridge_paper_ids[:6]
            ],
            "neighbor_communities": [
                {"cid": int(x), "weight": float(w), "size": int(graph.communities.get(int(x)).size) if int(x) in graph.communities else None}
                for x, w in (comm.neighbor_communities[: int(k_neighbor_comms)] if comm.neighbor_communities else [])
            ],
        },
        "neighbors": neighbor_papers,
        "neighbors_in_community": [
            {
                "pid": int(nb_pid),
                "score": float(sim),
                "title": _paper_title(papers[int(nb_pid)]),
                "year": _paper_year(papers[int(nb_pid)]),
            }
            for nb_pid, sim in nb_in_comm
        ],
    }

    hits = [SearchHit(kind="paper", id=str(pid), score=1.0, payload=payload)]
    return _result(
        "paper",
        {"pid": int(pid), "k_neighbors": int(k_neighbors)},
        hits,
        graph_snippet={
            "nodes": [{"kind": "paper", "id": str(pid)}] + [{"kind": "paper", "id": str(int(idx0 + 1))} for idx0, _ in nb[: min(30, len(nb))]],
            "edges": [{"source": str(pid), "target": str(int(idx0 + 1)), "weight": float(sim)} for idx0, sim in nb[: min(30, len(nb))]],
        },
    )


def expand_from_paper(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    pid: int,
    k_papers: int = 20,
    k_comms: int = 10,
) -> Dict[str, Any]:
    base = lookup_paper(
        assets=assets,
        papers=papers,
        graph=graph,
        pid=int(pid),
        k_neighbors=int(k_papers),
        k_neighbors_in_comm=min(10, int(k_papers)),
        k_neighbor_comms=min(12, int(k_comms)),
    )
    if not base["hits"]:
        base["type"] = "expand"
        base["query"] = {"pid": int(pid), "k_papers": int(k_papers), "k_comms": int(k_comms)}
        return base

    paper_payload = base["hits"][0]["payload"]
    cid = paper_payload.get("community", None)
    comm_result = lookup_community(papers=papers, graph=graph, cid=int(cid), top_neighbors=int(k_comms)) if cid is not None else None

    hits: List[SearchHit] = []
    hits.append(SearchHit(kind="paper", id=str(pid), score=1.0, payload=paper_payload))
    if comm_result and comm_result.get("hits"):
        hits.append(SearchHit(kind="community", id=str(cid), score=1.0, payload=comm_result["hits"][0]["payload"]))

    return _result(
        "expand",
        {"pid": int(pid), "k_papers": int(k_papers), "k_comms": int(k_comms)},
        hits,
        graph_snippet=base.get("graph_snippet"),
        debug={"composed_from": ["lookup_paper", "lookup_community"]},
    )


def save_result_json(result: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def load_papers_from_project(
    base_dir: Path,
    *,
    exclude_selfcite: bool = False,
    force_reingest: bool = False,
) -> List[Optional[Paper]]:
    """
    从 `data/data_store.pkl` 加载论文列表（与 `core.build_or_load` 使用同一套数据源路径）。
    供 Web API 等场景使用，避免 import 整个 `core.py`。
    """
    from getdata import ingest, load_data
    from model import build_models

    p = data_source_paths(Path(base_dir))

    if force_reingest and p.cache_path.exists():
        p.cache_path.unlink()
    if not p.cache_path.exists():
        ingest(
            authorpaper_rdata=p.authorpaper_rdata,
            author_name_txt=p.author_name_txt,
            textcorpus_rdata=p.textcorpus_rdata,
            topicresults_rdata=p.topicresults_rdata,
            rawpaper_rdata=p.rawpaper_rdata,
            out_path=p.cache_path,
            exclude_selfcite=exclude_selfcite,
        )
    data = load_data(p.cache_path)
    _, papers = build_models(data)
    return list(papers)


def community_graph_payload(
    assets: DemoAssets,
    graph: DemoCommunityGraph,
    *,
    max_nodes: Optional[int] = 400,
    min_weight: float = 0.0,
    sample_papers_per_comm: int = 400,
) -> Dict[str, Any]:
    """
    社区级图：节点=社区，边=跨社区边聚合权重（来自 Milestone A 的 `community_edges`）。
    """
    comms = list(graph.communities.values())
    comms.sort(key=lambda c: c.size, reverse=True)
    if max_nodes is not None:
        comms = comms[: int(max_nodes)]
    id_set = {int(c.cid) for c in comms}
    coords = assets.load_coords2d()
    nodes = []
    for c in comms:
        node = {"id": int(c.cid), "size": int(c.size), "label": f"C{c.cid}"}
        if coords is not None:
            pids = c.paper_ids
            if sample_papers_per_comm > 0 and len(pids) > int(sample_papers_per_comm):
                # deterministic sampling: take every k-th
                step = max(1, len(pids) // int(sample_papers_per_comm))
                pids = pids[::step][: int(sample_papers_per_comm)]
            idx0 = np.asarray([int(pid) - 1 for pid in pids if int(pid) > 0], dtype=np.int64)
            idx0 = idx0[(idx0 >= 0) & (idx0 < coords.shape[0])]
            if idx0.size > 0:
                xy = coords[idx0]
                node["x"] = float(np.mean(xy[:, 0]))
                node["y"] = float(np.mean(xy[:, 1]))
        nodes.append(node)
    edges: List[Dict[str, Any]] = []
    for (a, b), w in graph.community_edges.items():
        if float(w) < float(min_weight):
            continue
        ai, bi = int(a), int(b)
        if ai in id_set and bi in id_set:
            edges.append({"source": ai, "target": bi, "weight": float(w)})
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "max_nodes": max_nodes,
            "min_weight": float(min_weight),
        },
    }


def community_paper_subgraph_payload(
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    cid: int,
    *,
    max_nodes: int = 60,
    max_edges: int = 200,
) -> Dict[str, Any]:
    """
    单个社区内的论文子图（边来自 mutual-kNN，且两端都在该社区内）。
    节点按社区内度数截断到 max_nodes，边截断到 max_edges。
    """
    cid = int(cid)
    comm = graph.communities.get(cid)
    if comm is None:
        return {"nodes": [], "edges": [], "meta": {"error": "community_not_found", "cid": cid}}

    cset = {int(x) for x in comm.paper_ids}
    u, v, w, _ = assets.load_edges()
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    w = np.asarray(w, dtype=np.float32)

    deg: Dict[int, int] = {}
    triples: List[Tuple[int, int, float]] = []
    for i in range(int(u.size)):
        a0, b0 = int(u[i]), int(v[i])
        pa, pb = a0 + 1, b0 + 1
        if pa in cset and pb in cset:
            deg[pa] = deg.get(pa, 0) + 1
            deg[pb] = deg.get(pb, 0) + 1
            triples.append((pa, pb, float(w[i])))

    ranked = sorted(cset, key=lambda pid: deg.get(pid, 0), reverse=True)[: int(max_nodes)]
    rset = set(ranked)
    triples_f = [(a, b, ww) for a, b, ww in triples if a in rset and b in rset]
    triples_f.sort(key=lambda t: t[2], reverse=True)
    triples_f = triples_f[: int(max_edges)]

    coords = assets.load_coords2d()
    nodes = []
    for pid in ranked:
        node = {
            "id": int(pid),
            "label": (_paper_title(papers[int(pid)]) or "")[:120],
            "year": _paper_year(papers[int(pid)]),
        }
        if coords is not None:
            i0 = int(pid) - 1
            if 0 <= i0 < coords.shape[0]:
                node["x"] = float(coords[i0, 0])
                node["y"] = float(coords[i0, 1])
        nodes.append(node)
    edges = [{"source": int(a), "target": int(b), "weight": float(ww)} for a, b, ww in triples_f]
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "cid": cid,
            "community_size": int(comm.size),
            "n_nodes": len(nodes),
            "n_edges": len(edges),
        },
    }

