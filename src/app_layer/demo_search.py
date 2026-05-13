from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app_layer.demo_graph import (
    DemoCommunityGraph,
    enrich_demo_graph_topic_info,
    load_community_topic_info_from_csv,
    load_membership_for_resolution_light,
    resolve_membership_for_resolution_light,
)
from foundation_layer.model import Author, Paper
from foundation_layer.network import load_edges_npz
from foundation_layer.project_paths import data_source_paths


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


def year_in_publication_window(y: int, year_min: Optional[int], year_max: Optional[int]) -> bool:
    """Strict calendar filter: unknown years (<=0) never match when a bound is set."""
    if year_min is None and year_max is None:
        return True
    yi = int(y)
    if yi <= 0:
        return False
    if year_min is not None and yi < int(year_min):
        return False
    if year_max is not None and yi > int(year_max):
        return False
    return True


def _filter_pids_ordered(
    pids: Iterable[int],
    papers: Sequence[Optional[Paper]],
    year_min: Optional[int],
    year_max: Optional[int],
    limit: int,
) -> List[int]:
    lim = int(limit)
    if lim <= 0:
        return []
    if year_min is None and year_max is None:
        out_head: List[int] = []
        for pid in pids:
            out_head.append(int(pid))
            if len(out_head) >= lim:
                break
        return out_head
    out_y: List[int] = []
    for pid in pids:
        if len(out_y) >= lim:
            break
        pid_i = int(pid)
        if pid_i <= 0 or pid_i >= len(papers):
            continue
        p = papers[pid_i]
        if p is None:
            continue
        if year_in_publication_window(_paper_year(p), year_min, year_max):
            out_y.append(pid_i)
    return out_y


def _community_member_count_in_year(
    paper_ids: Iterable[int],
    papers: Sequence[Optional[Paper]],
    year_min: Optional[int],
    year_max: Optional[int],
) -> Optional[int]:
    if year_min is None and year_max is None:
        return None
    n = 0
    for pid in paper_ids:
        pid_i = int(pid)
        if pid_i <= 0 or pid_i >= len(papers):
            continue
        p = papers[pid_i]
        if p is None:
            continue
        if year_in_publication_window(_paper_year(p), year_min, year_max):
            n += 1
    return int(n)


def _year_pid_mask(
    papers: Sequence[Optional[Paper]],
    year_min: Optional[int],
    year_max: Optional[int],
) -> Optional[np.ndarray]:
    """Boolean mask indexed by pid (same length as papers). None = no filter."""
    if year_min is None and year_max is None:
        return None
    n = len(papers)
    ok = np.zeros(n, dtype=np.bool_)
    start = 1 if n > 0 and papers[0] is None else 0
    for pid in range(start, n):
        if year_in_publication_window(_paper_year(papers[pid]), year_min, year_max):
            ok[pid] = True
    return ok


def _cross_community_edges_year_filtered(
    membership: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    ok_pid: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    """
    Sum mutual-kNN edge weights between distinct communities, counting only edges
    whose both endpoints fall in ok_pid (1-based paper ids index into ok_pid).
    """
    mem = np.asarray(membership, dtype=np.int32)
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    if u.size == 0:
        return {}
    cu = mem[u]
    cv = mem[v]
    pa = u + 1
    pb = v + 1
    bounds_ok = (pa > 0) & (pb > 0) & (pa < ok_pid.shape[0]) & (pb < ok_pid.shape[0])
    inside = (cu >= 0) & (cv >= 0) & (cu != cv) & bounds_ok
    ok_a = np.zeros(pa.shape[0], dtype=np.bool_)
    ok_b = np.zeros(pb.shape[0], dtype=np.bool_)
    idx = np.flatnonzero(bounds_ok)
    if idx.size > 0:
        ok_a[idx] = ok_pid[pa[idx]]
        ok_b[idx] = ok_pid[pb[idx]]
    m = inside & ok_a & ok_b
    if not np.any(m):
        return {}
    lo = np.minimum(cu[m], cv[m]).astype(np.int32, copy=False)
    hi = np.maximum(cu[m], cv[m]).astype(np.int32, copy=False)
    ww = w[m].astype(np.float64, copy=False)
    pairs = np.stack([lo, hi], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    agg = np.bincount(inv, weights=ww, minlength=int(uniq.shape[0]))
    out: Dict[Tuple[int, int], float] = {}
    for (c1, c2), val in zip(uniq.tolist(), agg.tolist()):
        c1i, c2i = int(c1), int(c2)
        if c1i == c2i:
            continue
        out[(c1i, c2i)] = float(val)
    return out


def _author_names_inline(authors: Optional[Sequence[Optional[Author]]], p: object, *, limit: int = 5) -> str:
    pl = _paper_author_payload(authors, p, limit=int(limit))
    names = [str(a.get("name") or "").strip() for a in pl.get("authors") or []]
    names = [x for x in names if x]
    return "; ".join(names)


def _coerce_author_id(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if hasattr(x, "item") and callable(getattr(x, "item")):
            return int(x.item())
        return int(x)
    except (TypeError, ValueError):
        return None


def _paper_author_payload(
    authors: Optional[Sequence[Optional[Author]]],
    p: object,
    *,
    limit: int = 20,
) -> Dict[str, Any]:
    raw = list(getattr(p, "author", []) or [])
    ids: List[int] = []
    for x in raw[: int(limit)]:
        aid = _coerce_author_id(x)
        if aid is None:
            continue
        ids.append(int(aid))
    n_all = int(len(raw))
    if not authors:
        return {"author_ids": ids, "n_authors": n_all, "authors": []}
    named: List[Dict[str, Any]] = []
    for aid in ids:
        a = authors[aid] if 0 <= aid < len(authors) else None
        name = str(getattr(a, "name", "") or "").strip() if a is not None else ""
        named.append({"id": int(aid), "name": name})
    return {"author_ids": ids, "n_authors": n_all, "authors": named}


def _paper_keywords(p: object, max_terms: int = 10) -> List[str]:
    txt = f"{_paper_title(p)} {_paper_abs(p)}".lower()
    toks = [t.strip(".,:;()[]{}!?\"'") for t in txt.split()]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "using",
        "based",
        "study",
        "paper",
        "approach",
        "method",
        "results",
    }
    out: List[str] = []
    seen = set()
    for t in toks:
        if len(t) < 3 or t in stop:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= int(max_terms):
            break
    return out


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
    kind: str
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
    pagination: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
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
    if pagination is not None:
        out["pagination"] = dict(pagination)
    return out


def _node_id_for_comm(resolution: float, cid: int) -> str:
    return f"r={float(resolution):.4f}|c={int(cid)}"


def _parse_comm_node_id(node_id: str) -> Optional[Tuple[float, int]]:
    try:
        s = str(node_id)
        a, b = s.split("|", 1)
        r = float(a.split("=", 1)[1])
        c = int(b.split("=", 1)[1])
        return (round(float(r), 4), int(c))
    except Exception:
        return None


class DemoAssets:
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
        p, r_eff, _is_exact = resolve_membership_for_resolution_light(self.leiden_dir, self.resolution, allow_nearest=True)
        self.resolution = float(r_eff)
        return np.load(p).astype(np.int32)

    def ensure_csr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._csr_indptr is not None:
            assert self._csr_indices is not None and self._csr_data is not None
            return self._csr_indptr, self._csr_indices, self._csr_data

        u, v, w, n = self.load_edges()
        rows = np.concatenate([u, v]).astype(np.int32, copy=False)
        cols = np.concatenate([v, u]).astype(np.int32, copy=False)
        data = np.concatenate([w, w]).astype(np.float32, copy=False)
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


def _paper_neighbor_rows_for_detail(
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    pid: int,
    *,
    k_neighbors: int,
    k_neighbors_in_comm: int,
    year_min: Optional[int],
    year_max: Optional[int],
    comm: Optional[Any],
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Ordered mutual-kNN neighbors; optional publication-year filter applied before truncation."""
    idx0 = int(pid) - 1
    k_neighbors = int(k_neighbors)
    k_neighbors_in_comm = int(k_neighbors_in_comm)
    y_use = year_min is not None or year_max is not None
    fetch_k = int(k_neighbors)
    if y_use:
        fetch_k = min(2000, max(k_neighbors * 50, k_neighbors_in_comm * 50, k_neighbors + 120))
    nb_all = assets.neighbors_of_paper0(idx0, top_k=fetch_k)

    nb_rows: List[Tuple[int, float]] = []
    if not y_use:
        nb_rows = list(nb_all[:k_neighbors])
    else:
        for nb_idx0, sim in nb_all:
            nb_pid = int(nb_idx0 + 1)
            if nb_pid <= 0 or nb_pid >= len(papers):
                continue
            p = papers[nb_pid]
            if p is None:
                continue
            if not year_in_publication_window(_paper_year(p), year_min, year_max):
                continue
            nb_rows.append((nb_idx0, float(sim)))
            if len(nb_rows) >= k_neighbors:
                break

    nb_in_comm: List[Tuple[int, float]] = []
    if comm is not None and k_neighbors_in_comm > 0:
        cset = set(int(x) for x in comm.paper_ids)
        head = nb_all[:k_neighbors] if not y_use else nb_all
        for nb_idx0, sim in head:
            nb_pid = int(nb_idx0 + 1)
            if y_use:
                if nb_pid <= 0 or nb_pid >= len(papers):
                    continue
                qp = papers[nb_pid]
                if qp is None or not year_in_publication_window(_paper_year(qp), year_min, year_max):
                    continue
            if nb_pid in cset:
                nb_in_comm.append((nb_pid, float(sim)))
                if len(nb_in_comm) >= k_neighbors_in_comm:
                    break
        if y_use and len(nb_in_comm) < k_neighbors_in_comm:
            for nb_idx0, sim in nb_all[k_neighbors:]:
                nb_pid = int(nb_idx0 + 1)
                if nb_pid <= 0 or nb_pid >= len(papers):
                    continue
                qp = papers[nb_pid]
                if qp is None or not year_in_publication_window(_paper_year(qp), year_min, year_max):
                    continue
                if nb_pid in cset:
                    nb_in_comm.append((nb_pid, float(sim)))
                    if len(nb_in_comm) >= k_neighbors_in_comm:
                        break

    return nb_rows, nb_in_comm


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
    topic_communities_csv: Optional[Path] = None,
) -> Tuple[DemoAssets, DemoCommunityGraph]:
    from app_layer.demo_graph import build_demo_graph_from_membership

    c2d: Optional[Path] = None if coords_2d_path is None else Path(coords_2d_path)
    if c2d is None or not c2d.is_file():
        fb = Path(base_dir) / "out" / "umap2d.npy"
        if fb.is_file():
            c2d = fb

    assets = DemoAssets(
        base_dir=base_dir,
        leiden_dir=leiden_dir,
        resolution=float(resolution),
        graph_npz=graph_npz,
        keyword_index_dir=keyword_index_dir,
        coords_2d_path=c2d,
    )
    membership = assets.load_membership()
    u, v, w, _ = assets.load_edges()
    g = build_demo_graph_from_membership(
        list(papers),
        membership,
        resolution=float(assets.resolution),
        u=u,
        v=v,
        w=w,
        top_center=top_center,
        top_bridge=top_bridge,
        top_neighbor_comms=top_neighbor_comms,
    )
    tcp = Path(topic_communities_csv).resolve() if topic_communities_csv else None
    if tcp is not None and tcp.is_file():
        enrich_demo_graph_topic_info(g, load_community_topic_info_from_csv(tcp))
    return assets, g


def search_keyword(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    query: str,
    top_k: int = 20,
    offset: int = 0,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    authors: Optional[Sequence[Optional[Author]]] = None,
) -> Dict[str, Any]:
    q = str(query or "").strip()
    offset = max(int(offset), 0)
    top_k = int(top_k)
    y_filter = year_min is not None or year_max is not None

    if not q:
        return _result(
            "keyword",
            {"q": q, "top_k": top_k, "offset": offset},
            [],
            debug={"mode": "empty"},
            pagination={"offset": offset, "limit": top_k, "returned": 0, "total_available": 0},
        )

    tfidf_error: Optional[str] = None
    try:
        vectorizer, X = assets.load_keyword_index()
        qv = vectorizer.transform([q])
        scores = (X @ qv.T).toarray().ravel().astype(np.float32)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            total_avail = 0
            return _result(
                "keyword",
                {"q": q, "top_k": top_k, "offset": offset},
                [],
                debug={"mode": "tfidf", "hits": 0},
                pagination={"offset": offset, "limit": top_k, "returned": 0, "total_available": 0},
            )

        def _hits_from_idx0(idx0_arr: np.ndarray) -> List[SearchHit]:
            out_h: List[SearchHit] = []
            for doc_idx0 in idx0_arr.tolist():
                pid = int(doc_idx0 + 1)
                p = papers[pid]
                title = _paper_title(p)
                abstract = _paper_abs(p)
                cid = graph.paper_to_community.get(pid)
                bonus = 0.0
                if q.lower() in title.lower():
                    bonus += 0.15
                pl: Dict[str, Any] = {
                    "pid": pid,
                    "title": title,
                    "year": _paper_year(p),
                    "community": None if cid is None else int(cid),
                    "snippet": _make_snippet(abstract, q),
                }
                pl.update(_paper_author_payload(authors, p, limit=8))
                out_h.append(
                    SearchHit(
                        kind="paper",
                        id=str(pid),
                        score=float(scores[doc_idx0] + bonus),
                        payload=pl,
                    )
                )
            out_h.sort(key=lambda h: h.score, reverse=True)
            return out_h

        if not y_filter:
            k_take = min(top_k + offset, int(nz.size))
            top_local = np.argpartition(scores[nz], -k_take)[-k_take:]
            idx0_all = nz[top_local]
            idx0_all = idx0_all[np.argsort(scores[idx0_all])[::-1]].astype(np.int32, copy=False)
            total_avail = int(nz.size)
            idx0 = idx0_all[offset : offset + top_k]
            hits = _hits_from_idx0(idx0)
            return _result(
                "keyword",
                {"q": q, "top_k": top_k, "offset": offset},
                hits,
                debug={"mode": "tfidf"},
                pagination={"offset": offset, "limit": top_k, "returned": len(hits), "total_available": total_avail},
            )

        idx_sorted = nz[np.argsort(scores[nz])[::-1]].astype(np.int32, copy=False)
        filtered_idx: List[int] = []
        for i0 in idx_sorted.tolist():
            pid = int(i0 + 1)
            py = _paper_year(papers[pid])
            if year_in_publication_window(py, year_min, year_max):
                filtered_idx.append(int(i0))
        total_avail = len(filtered_idx)
        sel = filtered_idx[offset : offset + top_k]
        hits = _hits_from_idx0(np.asarray(sel, dtype=np.int32))
        return _result(
            "keyword",
            {"q": q, "top_k": top_k, "offset": offset},
            hits,
            debug={"mode": "tfidf"},
            pagination={"offset": offset, "limit": top_k, "returned": len(hits), "total_available": total_avail},
        )
    except Exception as e:
        tfidf_error = f"{type(e).__name__}: {e}"

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
        py = _paper_year(p)
        if y_filter and not year_in_publication_window(py, year_min, year_max):
            continue
        score = 2.0 if q_low in title.lower() else 1.0
        scored.append((score, pid))
    scored.sort(reverse=True)
    total_avail = len(scored)
    scored_page = scored[offset : offset + top_k]

    hits = []
    for score, pid in scored_page:
        p = papers[pid]
        cid = graph.paper_to_community.get(int(pid))
        pl2: Dict[str, Any] = {
            "pid": int(pid),
            "title": _paper_title(p),
            "year": _paper_year(p),
            "community": None if cid is None else int(cid),
            "snippet": _make_snippet(_paper_abs(p), q),
        }
        pl2.update(_paper_author_payload(authors, p, limit=8))
        hits.append(
            SearchHit(
                kind="paper",
                id=str(pid),
                score=float(score),
                payload=pl2,
            )
        )
    dbg: Dict[str, Any] = {"mode": "substring"}
    if tfidf_error:
        dbg["tfidf_error"] = tfidf_error
    return _result(
        "keyword",
        {"q": q, "top_k": top_k, "offset": offset},
        hits,
        debug=dbg,
        pagination={"offset": offset, "limit": top_k, "returned": len(hits), "total_available": total_avail},
    )


def lookup_community(
    *,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    cid: int,
    top_papers: int = 20,
    top_neighbors: int = 12,
    authors: Optional[Sequence[Optional[Author]]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    cid = int(cid)
    comm = graph.communities.get(cid)
    if comm is None:
        return _result("community", {"cid": cid}, [], debug={"error": "community_not_found"})

    def _p_payload(pid: int) -> Dict[str, Any]:
        p = papers[int(pid)]
        row: Dict[str, Any] = {"pid": int(pid), "title": _paper_title(p), "year": _paper_year(p)}
        row.update(_paper_author_payload(authors, p, limit=8))
        return row

    tp = int(top_papers)
    centers = _filter_pids_ordered(comm.center_paper_ids or [], papers, year_min, year_max, tp)
    bridges = _filter_pids_ordered(comm.bridge_paper_ids or [], papers, year_min, year_max, tp)
    base_example = comm.center_paper_ids[:] if comm.center_paper_ids else list(comm.paper_ids)
    example_pids = _filter_pids_ordered(base_example, papers, year_min, year_max, tp)
    if len(example_pids) < tp:
        seen = set(example_pids)
        for pid in _filter_pids_ordered(comm.paper_ids, papers, year_min, year_max, tp):
            if pid not in seen:
                example_pids.append(pid)
                seen.add(pid)
            if len(example_pids) >= tp:
                break

    size_in_win = _community_member_count_in_year(comm.paper_ids, papers, year_min, year_max)
    payload = {
        "resolution": float(graph.resolution),
        "cid": int(comm.cid),
        "size": int(comm.size),
        "topic_info": dict(comm.topic_info or {}),
        "center_papers": [_p_payload(pid) for pid in centers],
        "bridge_papers": [_p_payload(pid) for pid in bridges],
        "example_papers": [_p_payload(pid) for pid in example_pids],
        "neighbor_communities": [
            {
                "resolution": float(graph.resolution),
                "cid": int(x),
                "weight": float(w),
                "size": int(graph.communities.get(int(x)).size) if int(x) in graph.communities else None,
            }
            for x, w in (comm.neighbor_communities[: int(top_neighbors)] if comm.neighbor_communities else [])
        ],
    }
    if size_in_win is not None:
        payload["size_in_window"] = int(size_in_win)

    cq: Dict[str, Any] = {"cid": cid}
    if year_min is not None or year_max is not None:
        cq["year_min"] = year_min
        cq["year_max"] = year_max
    hits = [SearchHit(kind="community", id=str(cid), score=1.0, payload=payload)]
    return _result("community", cq, hits)


def lookup_community_at_resolution(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    resolution: float,
    cid: int,
    top_papers: int = 20,
    top_neighbors: int = 12,
    authors: Optional[Sequence[Optional[Author]]] = None,
    graph: Optional[DemoCommunityGraph] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    r = round(float(resolution), 4)
    cid = int(cid)
    membership = load_membership_for_resolution_light(assets.leiden_dir, r, allow_nearest=True)
    try:
        _, r_eff, _ = resolve_membership_for_resolution_light(assets.leiden_dir, r, allow_nearest=True)
        r = float(r_eff)
    except Exception:
        r = float(r)

    idx0 = np.flatnonzero(np.asarray(membership, dtype=np.int32) == int(cid)).astype(np.int64, copy=False)
    pids = (idx0 + 1).astype(np.int64, copy=False).tolist()
    if len(pids) == 0:
        return _result("community", {"cid": cid, "resolution": float(r)}, [], debug={"error": "community_not_found"})

    u, v, w, _ = assets.load_edges()
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    w = np.asarray(w, dtype=np.float32)
    mem = np.asarray(membership, dtype=np.int32)
    cu = mem[u]
    cv = mem[v]
    same = cu == cv
    cross = ~same

    n_papers = int(mem.size)
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

    member_idx0 = np.asarray(idx0, dtype=np.int64)
    wd = within_degree[member_idx0]
    bd = cross_degree[member_idx0]

    def _p_payload(pid: int) -> Dict[str, Any]:
        p = papers[int(pid)]
        row: Dict[str, Any] = {"pid": int(pid), "title": _paper_title(p), "year": _paper_year(p)}
        row.update(_paper_author_payload(authors, p, limit=8))
        return row

    k1 = min(int(top_papers), int(member_idx0.size))
    if k1 > 0:
        top_local = np.argpartition(wd, -k1)[-k1:]
        center_idx0 = member_idx0[top_local][np.argsort(wd[top_local])[::-1]]
        center_pids = (center_idx0 + 1).astype(int).tolist()
    else:
        center_pids = []

    k2 = min(int(top_papers), int(member_idx0.size))
    if k2 > 0:
        top_local = np.argpartition(bd, -k2)[-k2:]
        bridge_idx0 = member_idx0[top_local][np.argsort(bd[top_local])[::-1]]
        bridge_pids = (bridge_idx0 + 1).astype(int).tolist()
    else:
        bridge_pids = []

    nb_w: Dict[int, float] = {}
    if cross.any():
        a = cu[cross].astype(np.int32, copy=False)
        b = cv[cross].astype(np.int32, copy=False)
        ww = w[cross].astype(np.float32, copy=False)
        mask = (a == cid) | (b == cid)
        aa = a[mask]
        bb = b[mask]
        ww = ww[mask]
        for x, y, val in zip(aa.tolist(), bb.tolist(), ww.tolist()):
            other = int(y) if int(x) == cid else int(x)
            if other == cid:
                continue
            nb_w[other] = nb_w.get(other, 0.0) + float(val)
    nb_sorted = sorted(nb_w.items(), key=lambda t: t[1], reverse=True)[: int(top_neighbors)]

    tp = int(top_papers)
    center_show = _filter_pids_ordered(center_pids, papers, year_min, year_max, tp)
    bridge_show = _filter_pids_ordered(bridge_pids, papers, year_min, year_max, tp)
    ex_base = center_pids if center_pids else pids
    example_show = _filter_pids_ordered(ex_base, papers, year_min, year_max, tp)
    if len(example_show) < tp:
        seen = set(example_show)
        for pid in _filter_pids_ordered(pids, papers, year_min, year_max, tp):
            if pid not in seen:
                example_show.append(pid)
                seen.add(pid)
            if len(example_show) >= tp:
                break

    size_in_win = _community_member_count_in_year(pids, papers, year_min, year_max)
    payload = {
        "node_id": _node_id_for_comm(r, cid),
        "resolution": float(r),
        "cid": int(cid),
        "size": int(len(pids)),
        "center_papers": [_p_payload(pid) for pid in center_show],
        "bridge_papers": [_p_payload(pid) for pid in bridge_show],
        "example_papers": [_p_payload(pid) for pid in example_show],
        "neighbor_communities": [
            {"node_id": _node_id_for_comm(r, int(x)), "resolution": float(r), "cid": int(x), "weight": float(val), "size": None}
            for x, val in nb_sorted
        ],
    }
    if size_in_win is not None:
        payload["size_in_window"] = int(size_in_win)
    if graph is not None and abs(float(r) - float(graph.resolution)) < 1e-5:
        oc = graph.communities.get(int(cid))
        if oc is not None and oc.topic_info:
            payload["topic_info"] = dict(oc.topic_info)
    hits = [SearchHit(kind="community", id=str(payload["node_id"]), score=1.0, payload=payload)]
    cq2: Dict[str, Any] = {"node_id": payload["node_id"], "resolution": float(r), "cid": int(cid)}
    if year_min is not None or year_max is not None:
        cq2["year_min"] = year_min
        cq2["year_max"] = year_max
    return _result("community", cq2, hits)


def compute_structure_influence_index(
    assets: DemoAssets,
    graph: DemoCommunityGraph,
    pid: int,
    comm: Any,
) -> float:
    """
    Heuristic "influence-like" score from mutual-kNN weights and community placement
    (not journal impact factor). Bounded roughly to [0, 15] for UI display.
    """
    if comm is None:
        return float("nan")
    idx0 = int(pid) - 1
    nb = assets.neighbors_of_paper0(idx0, top_k=100)
    cset = {int(x) for x in comm.paper_ids}
    w_in = 0.0
    w_out = 0.0
    for i0, ww in nb:
        p2 = int(i0) + 1
        if p2 in cset:
            w_in += float(ww)
        else:
            w_out += float(ww)
    sz = max(1, int(comm.size))
    core = float(np.log1p(w_in) / np.log1p(float(sz)))
    bridge = float(w_out / (1e-6 + w_in + 1.0))
    score = 2.0 + 8.0 * core + 2.0 * float(np.exp(-bridge))
    return float(max(0.0, min(15.0, score)))


def lookup_paper(
    *,
    assets: DemoAssets,
    papers: Sequence[Optional[Paper]],
    graph: DemoCommunityGraph,
    pid: int,
    k_neighbors: int = 20,
    k_neighbors_in_comm: int = 10,
    k_neighbor_comms: int = 8,
    authors: Optional[Sequence[Optional[Author]]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    pid = int(pid)
    if pid <= 0 or pid >= len(papers):
        return _result("paper", {"pid": pid}, [], debug={"error": "paper_id_out_of_range"})
    p = papers[pid]
    if p is None:
        return _result("paper", {"pid": pid}, [], debug={"error": "paper_not_found"})

    cid = graph.paper_to_community.get(pid)
    comm = None if cid is None else graph.communities.get(int(cid))
    nb, nb_in_comm = _paper_neighbor_rows_for_detail(
        assets,
        papers,
        pid,
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        year_min=year_min,
        year_max=year_max,
        comm=comm,
    )

    def _paper_row_brief(npid: int, **extra: Any) -> Dict[str, Any]:
        npid = int(npid)
        q = papers[npid]
        row: Dict[str, Any] = {
            "pid": npid,
            "title": _paper_title(q),
            "year": _paper_year(q),
            "authors_line": _author_names_inline(authors, q, limit=6),
        }
        row.update(_paper_author_payload(authors, q, limit=8))
        row.update(extra)
        return row

    neighbor_papers = [
        _paper_row_brief(
            int(idx0 + 1),
            score=float(sim),
            community=int(graph.paper_to_community.get(int(idx0 + 1), -1)),
        )
        for idx0, sim in nb[: int(k_neighbors)]
    ]

    size_in_win = (
        _community_member_count_in_year(comm.paper_ids, papers, year_min, year_max) if comm is not None else None
    )
    community_summary: Optional[Dict[str, Any]] = None
    if comm is not None:
        community_summary = {
            "cid": int(comm.cid),
            "size": int(comm.size),
            "topic_info": dict(comm.topic_info or {}),
            "center_papers": [
                _paper_row_brief(int(x))
                for x in _filter_pids_ordered(comm.center_paper_ids or [], papers, year_min, year_max, 6)
            ],
            "bridge_papers": [
                _paper_row_brief(int(x))
                for x in _filter_pids_ordered(comm.bridge_paper_ids or [], papers, year_min, year_max, 6)
            ],
            "neighbor_communities": [
                {"cid": int(x), "weight": float(w), "size": int(graph.communities.get(int(x)).size) if int(x) in graph.communities else None}
                for x, w in (comm.neighbor_communities[: int(k_neighbor_comms)] if comm.neighbor_communities else [])
            ],
        }
        if size_in_win is not None:
            community_summary["size_in_window"] = int(size_in_win)

    sinf: Optional[float] = None
    if comm is not None:
        try:
            sinf = round(float(compute_structure_influence_index(assets, graph, pid, comm)), 4)
        except Exception:
            sinf = None

    payload = {
        "pid": int(pid),
        "title": _paper_title(p),
        "year": _paper_year(p),
        "abstract": _paper_abs(p),
        **_paper_author_payload(authors, p, limit=20),
        "authors_line": _author_names_inline(authors, p, limit=12),
        "keywords": _paper_keywords(p, max_terms=12),
        "structure_influence_index": sinf,
        "impact_factor": sinf,
        "community": None if cid is None else int(cid),
        "community_summary": community_summary,
        "neighbors": neighbor_papers,
        "neighbors_in_community": [
            _paper_row_brief(int(nb_pid), score=float(sim))
            for nb_pid, sim in nb_in_comm
        ],
    }

    hits = [SearchHit(kind="paper", id=str(pid), score=1.0, payload=payload)]
    paper_query: Dict[str, Any] = {"pid": int(pid), "k_neighbors": int(k_neighbors)}
    if year_min is not None or year_max is not None:
        paper_query["year_min"] = year_min
        paper_query["year_max"] = year_max
    return _result(
        "paper",
        paper_query,
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
    authors: Optional[Sequence[Optional[Author]]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    base = lookup_paper(
        assets=assets,
        papers=papers,
        graph=graph,
        pid=int(pid),
        k_neighbors=int(k_papers),
        k_neighbors_in_comm=min(10, int(k_papers)),
        k_neighbor_comms=min(12, int(k_comms)),
        authors=authors,
        year_min=year_min,
        year_max=year_max,
    )
    if not base["hits"]:
        base["type"] = "expand"
        base["query"] = {"pid": int(pid), "k_papers": int(k_papers), "k_comms": int(k_comms)}
        return base

    paper_payload = base["hits"][0]["payload"]
    cid = paper_payload.get("community", None)
    comm_result = (
        lookup_community(
            papers=papers,
            graph=graph,
            cid=int(cid),
            top_neighbors=int(k_comms),
            authors=authors,
            year_min=year_min,
            year_max=year_max,
        )
        if cid is not None
        else None
    )
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


def load_models_from_project(
    base_dir: Path,
    *,
    exclude_selfcite: bool = False,
    force_reingest: bool = False,
) -> Tuple[List[Optional[Author]], List[Optional[Paper]]]:
    from foundation_layer.getdata import ingest, load_data
    from foundation_layer.model import build_models

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
    authors, papers = build_models(data)
    return list(authors), list(papers)


def load_papers_from_project(
    base_dir: Path,
    *,
    exclude_selfcite: bool = False,
    force_reingest: bool = False,
) -> List[Optional[Paper]]:
    _, papers = load_models_from_project(
        base_dir,
        exclude_selfcite=exclude_selfcite,
        force_reingest=force_reingest,
    )
    return papers


def community_graph_payload(
    assets: DemoAssets,
    graph: DemoCommunityGraph,
    *,
    max_nodes: Optional[int] = 1500,
    min_weight: float = 0.0,
    sample_papers_per_comm: int = 400,
    include_positions: bool = False,
    papers: Optional[Sequence[Optional[Paper]]] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    y_use = year_min is not None or year_max is not None
    ok_pid = _year_pid_mask(papers, year_min, year_max) if y_use and papers is not None else None

    def _eff_size(c: Any) -> int:
        if ok_pid is None:
            return int(c.size)
        return int(sum(1 for pid in c.paper_ids if bool(ok_pid[int(pid)])))

    comms = list(graph.communities.values())
    if ok_pid is not None:
        comms = [c for c in comms if _eff_size(c) > 0]
    comms.sort(key=lambda c: _eff_size(c), reverse=True)
    if max_nodes is not None:
        comms = comms[: int(max_nodes)]
    id_set = {int(c.cid) for c in comms}

    comm_edges_use: Dict[Tuple[int, int], float]
    if ok_pid is None:
        comm_edges_use = dict(graph.community_edges)
    else:
        mem = assets.load_membership()
        u, v, w, _ = assets.load_edges()
        comm_edges_use = _cross_community_edges_year_filtered(mem, u, v, w, ok_pid)

    coords = assets.load_coords2d() if bool(include_positions) else None
    nodes = []
    for c in comms:
        node_id = _node_id_for_comm(assets.resolution, int(c.cid))
        eff = _eff_size(c)
        label = f"C{c.cid}"
        if ok_pid is not None:
            label = f"C{c.cid} · {eff}/{int(c.size)}"
        node = {
            "id": node_id,
            "cid": int(c.cid),
            "resolution": float(assets.resolution),
            "size": eff,
            "n_members": eff,
            "label": label,
        }
        if ok_pid is not None:
            node["size_full"] = int(c.size)
        if coords is not None:
            pids = list(c.paper_ids)
            if ok_pid is not None:
                pids = [pid for pid in pids if bool(ok_pid[int(pid)])]
            if sample_papers_per_comm > 0 and len(pids) > int(sample_papers_per_comm):
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
    for (a, b), ew in comm_edges_use.items():
        if float(ew) < float(min_weight):
            continue
        ai, bi = int(a), int(b)
        if ai in id_set and bi in id_set:
            edges.append(
                {
                    "source": _node_id_for_comm(assets.resolution, ai),
                    "target": _node_id_for_comm(assets.resolution, bi),
                    "weight": float(ew),
                }
            )
    effs_shown = [_eff_size(c) for c in comms]
    meta_singleton: Optional[str] = None
    if effs_shown and max(effs_shown) <= 1:
        meta_singleton = (
            "当前分辨率 r 在扫参上端（γ 很大）：该 run 下分区几乎全是「一篇论文 = 一个社区」，"
            "不是界面算错。请将 r 滑块向左调小，才能看到更大的社区。"
        )
    meta_note = None
    if ok_pid is not None:
        meta_note = (
            "Publication-year filter: node size and edges use only papers and mutual-kNN edges inside "
            "[year_min, year_max]. Community ids still come from the offline partition (not refit per calendar window)."
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "max_nodes": max_nodes,
            "min_weight": float(min_weight),
            "resolution": float(assets.resolution),
            "year_min": year_min,
            "year_max": year_max,
            "year_filter_active": bool(y_use),
            "note": meta_note,
            "community_size_max_shown": int(max(effs_shown)) if effs_shown else 0,
            **({"singleton_partition_hint": meta_singleton} if meta_singleton else {}),
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
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Dict[str, Any]:
    cid = int(cid)
    comm = graph.communities.get(cid)
    if comm is None:
        return {"nodes": [], "edges": [], "meta": {"error": "community_not_found", "cid": cid}}

    y_use = year_min is not None or year_max is not None
    cset_full = {int(x) for x in comm.paper_ids}
    if y_use:
        cset = {
            pid
            for pid in cset_full
            if year_in_publication_window(_paper_year(papers[int(pid)]), year_min, year_max)
        }
        if not cset:
            return {
                "nodes": [],
                "edges": [],
                "meta": {
                    "error": "no_papers_in_year_window",
                    "cid": cid,
                    "year_min": year_min,
                    "year_max": year_max,
                    "community_size_full": int(comm.size),
                },
            }
    else:
        cset = cset_full
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

    # Prefer showing **all** community members as nodes when under max_nodes cap;
    # otherwise keep high-degree nodes (legacy behavior for very large communities).
    if len(cset) <= int(max_nodes):
        ranked = sorted(cset, key=lambda pid: int(pid))
    else:
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
    meta: Dict[str, Any] = {
        "cid": cid,
        "community_size": len(cset),
        "community_size_full": int(comm.size),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "year_min": year_min,
        "year_max": year_max,
        "year_filter_active": bool(y_use),
    }
    if y_use:
        meta["note"] = (
            "Only papers whose publication year falls in [year_min, year_max] are shown; "
            "edges are mutual-kNN restricted to this vertex subset."
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
    }


def resolve_topic_weights_csv_for_web(
    *,
    base_dir: Path,
    run_id: str,
    resolution: float,
    manifest_topic_csv: Optional[str],
    env_topic_csv: Optional[Path],
    manifest_tags: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Resolve ``communities_topic_weights.csv`` for the demo web API:
    manifest path → ``PC_TOPIC_COMMUNITIES_CSV`` (cfg) →
    ``out/topic_runs/<sweep>/K*/r*/`` nearest to ``resolution``.
    """
    from data_layer.breakpoint_schedule import default_topic_k_dir_for_run, resolve_topic_communities_csv

    if manifest_topic_csv:
        tp = Path(str(manifest_topic_csv).strip())
        if str(tp) != "" and str(tp) != ".":
            tc_path = tp if tp.is_absolute() else (Path(base_dir) / tp).resolve()
            if tc_path.is_file():
                return tc_path
    if env_topic_csv is not None and env_topic_csv.is_file():
        return env_topic_csv
    root = default_topic_k_dir_for_run(Path(base_dir), str(run_id), manifest_tags=manifest_tags)
    return resolve_topic_communities_csv(root, float(resolution))
