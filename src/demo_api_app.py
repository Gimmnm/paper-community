"""
Milestone C：FastAPI 服务，启动时加载论文 + DemoCommunityGraph，暴露与 CLI demo-* 一致的 JSON 查询能力。

运行（需在仓库根目录或设置 PYTHONPATH=src）：
  uvicorn demo_api_app:app --host 0.0.0.0 --port 8000

或通过：
  python src/core.py demo-api --resolution 1.0
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from demo_search import (
    DemoAssets,
    DemoCommunityGraph,
    build_demo_assets_and_graph,
    community_graph_payload,
    community_paper_subgraph_payload,
    expand_from_paper,
    load_papers_from_project,
    lookup_community,
    lookup_paper,
    search_keyword,
)
from hierarchy_index import bfs_lineage, build_sqlite_index, query_roots
import sqlite3
import json


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_path(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    if v:
        p = Path(v)
        return p if p.is_absolute() else (_repo_root() / p).resolve()
    return default


def _settings() -> Dict[str, Any]:
    root = _repo_root()
    resolution = float(os.environ.get("PC_RESOLUTION", "1.0"))
    # Default to CPM partition outputs for better connectivity; can be overridden via PC_LEIDEN_DIR.
    leiden_dir = _env_path("PC_LEIDEN_DIR", root / "out" / "leiden_sweep_cpm")
    k = int(os.environ.get("PC_K", "50"))
    graph_npz = _env_path("PC_GRAPH_NPZ", root / "out" / f"mutual_knn_k{k}.npz")
    keyword_index_dir = _env_path("PC_KEYWORD_INDEX_DIR", root / "out" / "keyword_index")
    coords_2d_path = _env_path("PC_2D_PATH", root / "out" / "umap2d.npy")
    exclude_selfcite = os.environ.get("PC_EXCLUDE_SELFCITE", "0").lower() in ("1", "true", "yes")
    return {
        "base_dir": root,
        "resolution": resolution,
        "leiden_dir": leiden_dir,
        "graph_npz": graph_npz,
        "keyword_index_dir": keyword_index_dir,
        "coords_2d_path": coords_2d_path,
        "exclude_selfcite": exclude_selfcite,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _settings()
    papers = load_papers_from_project(
        cfg["base_dir"],
        exclude_selfcite=bool(cfg["exclude_selfcite"]),
        force_reingest=False,
    )
    assets, graph = build_demo_assets_and_graph(
        base_dir=cfg["base_dir"],
        papers=papers,
        leiden_dir=Path(cfg["leiden_dir"]),
        resolution=float(cfg["resolution"]),
        graph_npz=Path(cfg["graph_npz"]),
        keyword_index_dir=Path(cfg["keyword_index_dir"]),
        coords_2d_path=Path(cfg["coords_2d_path"]) if cfg.get("coords_2d_path") is not None else None,
    )
    app.state.papers = papers
    app.state.assets = assets  # type: ignore[attr-defined]
    app.state.graph = graph  # type: ignore[attr-defined]
    app.state.config = cfg  # type: ignore[attr-defined]
    yield


app = FastAPI(
    title="Paper Community Demo API",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve Milestone D static frontend
_WEB_DIR = _repo_root() / "web"
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR), html=False), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("PC_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _papers(request: Request) -> List[Any]:
    return request.app.state.papers


def _assets(request: Request) -> DemoAssets:
    return request.app.state.assets


def _graph(request: Request) -> DemoCommunityGraph:
    return request.app.state.graph


def _list_hierarchy_datasets(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover hierarchy directories under out/ and return minimal dataset descriptors.
    """
    out_dir = Path(base_dir) / "out"
    items: List[Tuple[str, Path, Dict[str, Any]]] = []

    def add(hid: str, p: Path, tags: Dict[str, Any]) -> None:
        if not (p / "hierarchy_nodes.csv").exists() or not (p / "hierarchy_edges.csv").exists():
            return
        items.append((hid, p, tags))

    # global
    add("global_cpm", out_dir / "leiden_hierarchy_cpm", {"kind": "global", "partition": "cpm"})
    add("global_rb", out_dir / "leiden_hierarchy_rb", {"kind": "global", "partition": "rb"})

    # local community runs
    sg = out_dir / "subgraph_hierarchy"
    if sg.exists():
        for d in sorted(sg.iterdir()):
            if not d.is_dir():
                continue
            add(f"subgraph_{d.name}", d, {"kind": "local_community"})

    # coarse domains
    for d in sorted(out_dir.glob("coarse_domains_kmeans_*")):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        for hd in sorted(d.glob("domain_*_hierarchy_*")):
            if not hd.is_dir():
                continue
            add(f"{d.name}__{hd.name}", hd, {"kind": "domain", "kmeans": d.name, **({"k": meta.get("k"), "seed": meta.get("seed")} if meta else {})})

    out: List[Dict[str, Any]] = []
    for hid, p, tags in items:
        overview_path = p / "hierarchy_viz_overview.json"
        overview = None
        if overview_path.exists():
            try:
                overview = json.loads(overview_path.read_text(encoding="utf-8"))
            except Exception:
                overview = None
        out.append(
            {
                "hid": hid,
                "hierarchy_dir": str(p),
                "tags": tags,
                "overview": overview,
                "has_index": (p / "hierarchy_index.sqlite").exists(),
            }
        )
    return out


def _open_hierarchy_db(hierarchy_dir: Path) -> sqlite3.Connection:
    db_path = Path(hierarchy_dir) / "hierarchy_index.sqlite"
    if not db_path.exists():
        # Build on-demand for small hierarchies; for global this may take a while.
        build_sqlite_index(hierarchy_dir=Path(hierarchy_dir), overwrite=False)
    con = sqlite3.connect(str(db_path))
    return con


@app.get("/api/coords/papers")
async def api_coords_papers(
    request: Request,
    ids: str = Query(..., description="Comma-separated paper ids, e.g. 1,2,3"),
    include_title: bool = Query(False),
    max_ids: int = Query(300, ge=1, le=2000),
) -> Dict[str, Any]:
    """
    轻量坐标查询：把给定 pid 列表映射到固定 2D 坐标（umap2d.npy）。
    用于 Keyword 命中论文的 “paper map” 展示。
    """
    assets = _assets(request)
    coords = assets.load_coords2d()
    if coords is None:
        return {"nodes": [], "meta": {"error": "coords2d_missing"}}

    raw = [x.strip() for x in str(ids).split(",") if x.strip()]
    if not raw:
        return {"nodes": [], "meta": {"n_nodes": 0}}
    raw = raw[: int(max_ids)]
    pids: List[int] = []
    for s in raw:
        try:
            pids.append(int(s))
        except Exception:
            continue
    pids = [pid for pid in pids if 1 <= pid <= coords.shape[0]]
    pids = list(dict.fromkeys(pids))  # stable unique

    papers = _papers(request)
    g = _graph(request)
    nodes: List[Dict[str, Any]] = []
    for pid in pids:
        i0 = pid - 1
        node = {
            "id": int(pid),
            "x": float(coords[i0, 0]),
            "y": float(coords[i0, 1]),
            "community": int(g.paper_to_community.get(int(pid), -1)),
        }
        if include_title:
            try:
                p = papers[int(pid)]
                node["label"] = str(getattr(p, "name", "") or "")[:120]
                node["year"] = int(getattr(p, "year", 0) or 0)
            except Exception:
                pass
        nodes.append(node)

    return {"nodes": nodes, "meta": {"n_nodes": len(nodes), "max_ids": int(max_ids), "include_title": bool(include_title)}}


@app.get("/api/v2/hierarchies")
async def api_v2_hierarchies(request: Request) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    return {"datasets": _list_hierarchy_datasets(base_dir)}


@app.get("/api/v2/hierarchies/{hid}/roots")
async def api_v2_hierarchy_roots(
    request: Request,
    hid: str,
    root_resolution: float = Query(0.001, ge=0.0),
    top_k: int = Query(50, ge=1, le=5000),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    datasets = _list_hierarchy_datasets(base_dir)
    ds = next((x for x in datasets if x["hid"] == hid), None)
    if ds is None:
        return {"error": "hierarchy_not_found", "hid": hid}
    hdir = Path(ds["hierarchy_dir"])
    con = _open_hierarchy_db(hdir)
    try:
        roots = query_roots(con, root_resolution=float(root_resolution), top_k=int(top_k))
        return {"hid": hid, "roots": roots, "meta": {"root_resolution": float(root_resolution), "top_k": int(top_k)}}
    finally:
        con.close()


@app.get("/api/v2/hierarchies/{hid}/lineage/{root_node_id:path}")
async def api_v2_hierarchy_lineage(
    request: Request,
    hid: str,
    root_node_id: str,
    r_max: float = Query(..., gt=0.0),
    min_child_share: float = Query(0.15, ge=0.0, le=1.0),
    max_nodes: int = Query(5000, ge=10, le=200000),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    datasets = _list_hierarchy_datasets(base_dir)
    ds = next((x for x in datasets if x["hid"] == hid), None)
    if ds is None:
        return {"error": "hierarchy_not_found", "hid": hid}
    hdir = Path(ds["hierarchy_dir"])
    con = _open_hierarchy_db(hdir)
    try:
        out = bfs_lineage(
            con,
            root_node_id=str(root_node_id),
            r_max=float(r_max),
            min_child_share=float(min_child_share),
            max_nodes=int(max_nodes),
        )
        out["hid"] = hid
        return out
    finally:
        con.close()

@app.get("/api/health")
async def api_health(request: Request) -> Dict[str, Any]:
    g: DemoCommunityGraph = _graph(request)
    cfg = request.app.state.config
    return {
        "status": "ok",
        "n_papers": int(g.n_papers),
        "n_communities": int(g.n_communities),
        "resolution": float(g.resolution),
        "leiden_dir": str(cfg["leiden_dir"]),
        "graph_npz": str(cfg["graph_npz"]),
        "keyword_index_dir": str(cfg["keyword_index_dir"]),
    }


@app.get("/api/search/keyword")
async def api_search_keyword(
    request: Request,
    q: str = Query(..., description="检索词"),
    top_k: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0, le=20000),
) -> Dict[str, Any]:
    return search_keyword(
        assets=_assets(request),
        papers=_papers(request),
        graph=_graph(request),
        query=q,
        top_k=int(top_k),
        offset=int(offset),
    )


@app.get("/api/papers/{paper_id}")
async def api_paper(
    request: Request,
    paper_id: int,
    k_neighbors: int = Query(20, ge=1, le=200),
    k_neighbors_in_comm: int = Query(10, ge=0, le=200),
    k_neighbor_comms: int = Query(8, ge=0, le=100),
) -> Dict[str, Any]:
    return lookup_paper(
        assets=_assets(request),
        papers=_papers(request),
        graph=_graph(request),
        pid=int(paper_id),
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        k_neighbor_comms=int(k_neighbor_comms),
    )


@app.get("/api/communities/{community_id}")
async def api_community(
    request: Request,
    community_id: int,
    top_papers: int = Query(20, ge=1, le=500),
    top_neighbors: int = Query(12, ge=0, le=200),
) -> Dict[str, Any]:
    return lookup_community(
        papers=_papers(request),
        graph=_graph(request),
        cid=int(community_id),
        top_papers=int(top_papers),
        top_neighbors=int(top_neighbors),
    )


@app.get("/api/expand/paper/{paper_id}")
async def api_expand(
    request: Request,
    paper_id: int,
    k_papers: int = Query(20, ge=1, le=200),
    k_comms: int = Query(10, ge=0, le=200),
) -> Dict[str, Any]:
    return expand_from_paper(
        assets=_assets(request),
        papers=_papers(request),
        graph=_graph(request),
        pid=int(paper_id),
        k_papers=int(k_papers),
        k_comms=int(k_comms),
    )


@app.get("/api/graph/communities")
async def api_graph_communities(
    request: Request,
    max_nodes: Optional[int] = Query(400, ge=10, le=5000),
    min_weight: float = Query(0.0, ge=0.0),
    sample_papers_per_comm: int = Query(400, ge=10, le=5000),
) -> Dict[str, Any]:
    return community_graph_payload(
        _assets(request),
        _graph(request),
        max_nodes=max_nodes,
        min_weight=min_weight,
        sample_papers_per_comm=int(sample_papers_per_comm),
    )


@app.get("/api/graph/community/{community_id}")
async def api_graph_community_papers(
    request: Request,
    community_id: int,
    max_nodes: int = Query(60, ge=5, le=500),
    max_edges: int = Query(200, ge=10, le=5000),
) -> Dict[str, Any]:
    return community_paper_subgraph_payload(
        _assets(request),
        _papers(request),
        _graph(request),
        int(community_id),
        max_nodes=int(max_nodes),
        max_edges=int(max_edges),
    )


@app.get("/", response_model=None)
async def root():
    index_path = _WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"docs": "/docs", "health": "/api/health"}
