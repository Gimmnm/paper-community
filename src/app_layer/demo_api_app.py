"""
Milestone C：FastAPI 服务，启动时加载论文 + DemoCommunityGraph，暴露与 CLI demo-* 一致的 JSON 查询能力。

运行（需在仓库根目录或设置 PYTHONPATH=src）：
  uvicorn app_layer.demo_api_app:app --host 0.0.0.0 --port 8000

或通过：
  python src/core.py demo-api --resolution 1.0
"""

from __future__ import annotations

import json
import mimetypes
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import random

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from analysis_layer.evaluation_metrics import build_evaluation_overview
from foundation_layer.project_paths import REPO_ROOT, embedding_path_specter2
from analysis_layer.hierarchy_index import bfs_lineage, build_sqlite_index, query_roots
from app_layer.demo_retrieval_live import (
    live_vs_sixway_positions,
    load_emb_norm_for_papers,
    read_retrieval_sixway_rows,
    run_live_retrieval_for_seed,
    search_community_bundle,
    search_vector_nn,
)
from app_layer.demo_search import (
    DemoAssets,
    DemoCommunityGraph,
    _paper_year,
    build_demo_assets_and_graph,
    community_graph_payload,
    community_paper_subgraph_payload,
    expand_from_paper,
    load_models_from_project,
    lookup_community,
    lookup_community_at_resolution,
    lookup_paper,
    resolve_topic_weights_csv_for_web,
    search_keyword,
    year_in_publication_window,
)
from data_layer.experiment_registry import (
    available_resolutions_for_manifest,
    build_run_catalog,
    default_paths_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _env_path(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    if v:
        p = Path(v)
        return p if p.is_absolute() else (_repo_root() / p).resolve()
    return default


def _settings() -> Dict[str, Any]:
    root = _repo_root()
    leiden_dir = _env_path("PC_LEIDEN_DIR", root / "out" / "leiden_sweep_cpm")
    if os.environ.get("PC_RESOLUTION") is not None:
        resolution = float(os.environ.get("PC_RESOLUTION", "1.0"))
    else:
        rs: List[float] = []
        try:
            for p in Path(leiden_dir).glob("membership_r*.npy"):
                s = p.stem
                if not s.startswith("membership_r"):
                    continue
                try:
                    rs.append(float(s.split("membership_r", 1)[1]))
                except Exception:
                    continue
        except Exception:
            rs = []
        resolution = max(rs) if rs else 1.0
    k = int(os.environ.get("PC_K", "50"))
    graph_npz = _env_path("PC_GRAPH_NPZ", root / "out" / f"mutual_knn_k{k}.npz")
    keyword_index_dir = _env_path("PC_KEYWORD_INDEX_DIR", root / "out" / "keyword_index")
    coords_2d_path = _env_path("PC_2D_PATH", root / "out" / "umap2d.npy")
    exclude_selfcite = os.environ.get("PC_EXCLUDE_SELFCITE", "0").lower() in ("1", "true", "yes")
    active_algorithm = str(os.environ.get("PC_ALGORITHM", "leiden_cpm")).strip().lower()
    active_time_window = str(os.environ.get("PC_TIME_WINDOW", "all")).strip().lower()
    topic_csv_env = os.environ.get("PC_TOPIC_COMMUNITIES_CSV")
    topic_communities_csv = None
    if topic_csv_env:
        p = Path(topic_csv_env)
        topic_communities_csv = p if p.is_absolute() else (root / p).resolve()
    return {
        "base_dir": root,
        "resolution": resolution,
        "leiden_dir": leiden_dir,
        "graph_npz": graph_npz,
        "keyword_index_dir": keyword_index_dir,
        "coords_2d_path": coords_2d_path,
        "exclude_selfcite": exclude_selfcite,
        "active_algorithm": active_algorithm,
        "active_time_window": active_time_window,
        "topic_communities_csv": topic_communities_csv,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _settings()
    authors, papers = load_models_from_project(
        cfg["base_dir"],
        exclude_selfcite=bool(cfg["exclude_selfcite"]),
        force_reingest=False,
    )
    fallback = default_paths_manifest(
        run_id="cli_default",
        base_dir=Path(cfg["base_dir"]),
        leiden_dir=Path(cfg["leiden_dir"]),
        graph_npz=Path(cfg["graph_npz"]),
        keyword_index_dir=Path(cfg["keyword_index_dir"]) if cfg.get("keyword_index_dir") is not None else None,
        coords_2d_path=Path(cfg["coords_2d_path"]) if cfg.get("coords_2d_path") is not None else None,
        algorithm=str(cfg.get("active_algorithm", "leiden_cpm")),
        time_window=str(cfg.get("active_time_window", "all")),
        default_resolution=float(cfg["resolution"]),
    )
    run_catalog = build_run_catalog(base_dir=Path(cfg["base_dir"]), fallback=fallback)
    env_rid = os.environ.get("PC_ACTIVE_RUN_ID")
    env_rid = str(env_rid).strip() if env_rid else ""
    if env_rid == "legacy_default":
        env_rid = "cli_default"
    if env_rid and env_rid in run_catalog:
        active_run_id = env_rid
    elif "leiden_cpm" in run_catalog:
        active_run_id = "leiden_cpm"
    else:
        active_run_id = fallback.run_id
    runtime_cache: Dict[str, Dict[str, Any]] = {}

    def _runtime_key(run_id: str, resolution: Optional[float]) -> str:
        if resolution is None:
            return str(run_id)
        return f"{run_id}@{float(resolution):.4f}"

    def _build_runtime(run_id: str, resolution_override: Optional[float] = None) -> Dict[str, Any]:
        m = run_catalog[run_id]
        rr = float(m.default_resolution if resolution_override is None else resolution_override)
        tc_manifest = getattr(m, "topic_communities_csv", None)
        tc_manifest_str = str(tc_manifest).strip() if tc_manifest else None
        if tc_manifest_str == "":
            tc_manifest_str = None
        tc_env = cfg.get("topic_communities_csv")
        env_p = tc_env if isinstance(tc_env, Path) else None
        tc_eff = resolve_topic_weights_csv_for_web(
            base_dir=Path(cfg["base_dir"]),
            run_id=str(m.run_id),
            resolution=float(rr),
            manifest_topic_csv=tc_manifest_str,
            env_topic_csv=env_p,
            manifest_tags=m.tags if isinstance(m.tags, dict) else None,
        )
        assets, graph = build_demo_assets_and_graph(
            base_dir=cfg["base_dir"],
            papers=papers,
            leiden_dir=Path(m.leiden_dir),
            resolution=rr,
            graph_npz=Path(m.graph_npz),
            keyword_index_dir=Path(m.keyword_index_dir) if m.keyword_index_dir else Path(cfg["keyword_index_dir"]),
            coords_2d_path=Path(m.coords_2d_path) if m.coords_2d_path else (Path(cfg["coords_2d_path"]) if cfg.get("coords_2d_path") is not None else None),
            topic_communities_csv=tc_eff,
        )
        return {"manifest": m, "assets": assets, "graph": graph}

    _rt0 = _build_runtime(active_run_id, None)
    runtime_cache[_runtime_key(active_run_id, None)] = _rt0
    _init_res = float(_rt0["graph"].resolution)
    runtime_cache[_runtime_key(active_run_id, _init_res)] = _rt0
    paper_year_min: Optional[int] = None
    paper_year_max: Optional[int] = None
    for i in range(1, len(papers)):
        p = papers[i]
        if p is None:
            continue
        y = int(getattr(p, "year", 0) or 0)
        if y <= 0:
            continue
        paper_year_min = y if paper_year_min is None else min(paper_year_min, y)
        paper_year_max = y if paper_year_max is None else max(paper_year_max, y)
    app.state.papers = papers
    app.state.authors = authors  # type: ignore[attr-defined]
    app.state.paper_year_min = paper_year_min  # type: ignore[attr-defined]
    app.state.paper_year_max = paper_year_max  # type: ignore[attr-defined]
    app.state.run_catalog = run_catalog  # type: ignore[attr-defined]
    app.state.active_run_id = active_run_id  # type: ignore[attr-defined]
    app.state.active_resolution = _init_res  # type: ignore[attr-defined]
    app.state.runtime_cache = runtime_cache  # type: ignore[attr-defined]
    app.state.runtime_key = _runtime_key  # type: ignore[attr-defined]
    app.state.build_runtime = _build_runtime  # type: ignore[attr-defined]
    app.state.config = cfg  # type: ignore[attr-defined]
    app.state.emb_norm = None  # type: ignore[attr-defined]
    app.state.emb_load_error = None  # type: ignore[attr-defined]
    try:
        n_expect = max(0, len(papers) - 1)
        ep = embedding_path_specter2(Path(cfg["base_dir"]))
        app.state.emb_norm = load_emb_norm_for_papers(emb_path=ep, n_papers_expect=n_expect)  # type: ignore[attr-defined]
    except Exception as e:
        app.state.emb_load_error = f"{type(e).__name__}: {e}"  # type: ignore[attr-defined]
    yield


app = FastAPI(
    title="Paper Community Demo API",
    version="0.1.0",
    lifespan=lifespan,
)

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


def _authors(request: Request) -> List[Any]:
    return getattr(request.app.state, "authors", None)


def _run_catalog(request: Request) -> Dict[str, Any]:
    return request.app.state.run_catalog


def _active_run_id(request: Request) -> str:
    return str(request.app.state.active_run_id)


def _runtime(request: Request, *, run_id: Optional[str] = None, resolution: Optional[float] = None) -> Dict[str, Any]:
    rid = str(run_id or _active_run_id(request))
    cat = _run_catalog(request)
    if rid not in cat:
        rid = _active_run_id(request)
    res_eff = resolution
    if res_eff is None:
        ar = getattr(request.app.state, "active_resolution", None)
        if ar is not None:
            res_eff = float(ar)
    key_fn = request.app.state.runtime_key
    key = key_fn(rid, res_eff)
    cache = request.app.state.runtime_cache
    if key in cache:
        return cache[key]
    build_fn = request.app.state.build_runtime
    rt = build_fn(rid, res_eff)
    cache[key] = rt
    return rt


def _assets(request: Request, *, run_id: Optional[str] = None, resolution: Optional[float] = None) -> DemoAssets:
    return _runtime(request, run_id=run_id, resolution=resolution)["assets"]


def _graph(request: Request, *, run_id: Optional[str] = None, resolution: Optional[float] = None) -> DemoCommunityGraph:
    return _runtime(request, run_id=run_id, resolution=resolution)["graph"]


def _list_hierarchy_datasets(base_dir: Path) -> List[Dict[str, Any]]:
    out_dir = Path(base_dir) / "out"
    items: List[tuple[str, Path, Dict[str, Any]]] = []

    def add(hid: str, p: Path, tags: Dict[str, Any]) -> None:
        if not (p / "hierarchy_nodes.csv").exists() or not (p / "hierarchy_edges.csv").exists():
            return
        items.append((hid, p, tags))

    add("global_cpm", out_dir / "leiden_hierarchy_cpm", {"kind": "global", "partition": "cpm"})
    add("global_rb", out_dir / "leiden_hierarchy_rb", {"kind": "global", "partition": "rb"})

    sg = out_dir / "subgraph_hierarchy"
    if sg.exists():
        for d in sorted(sg.iterdir()):
            if not d.is_dir():
                continue
            add(f"subgraph_{d.name}", d, {"kind": "local_community"})

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
        build_sqlite_index(hierarchy_dir=Path(hierarchy_dir), overwrite=False)
    con = sqlite3.connect(str(db_path))
    return con


def _parse_node_id(node_id: str) -> Optional[tuple[float, int]]:
    try:
        s = str(node_id)
        a, b = s.split("|", 1)
        r = float(a.split("=", 1)[1])
        c = int(b.split("=", 1)[1])
        return (round(float(r), 4), int(c))
    except Exception:
        return None


def _membership_path(hierarchy_dir: Path, r: float) -> Path:
    return Path(hierarchy_dir) / f"membership_r{float(r):.4f}.npy"


def _load_global_vertex_indices(hierarchy_dir: Path) -> Optional[np.ndarray]:
    p = Path(hierarchy_dir) / "global_vertex_indices.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    arr = np.asarray(arr, dtype=np.int64)
    return arr


def _node_members_pids(*, hierarchy_dir: Path, r: float, c: int) -> np.ndarray:
    mem_path = _membership_path(hierarchy_dir, r)
    if not mem_path.exists():
        raise FileNotFoundError(f"membership not found: {mem_path}")
    mem = np.load(mem_path, mmap_mode="r")
    mem = np.asarray(mem, dtype=np.int32)
    idx0 = np.where(mem == int(c))[0].astype(np.int64)
    gidx = _load_global_vertex_indices(hierarchy_dir)
    if gidx is None:
        return (idx0 + 1).astype(np.int64)
    idx0 = idx0[(idx0 >= 0) & (idx0 < gidx.shape[0])]
    global_idx0 = gidx[idx0].astype(np.int64, copy=False)
    return (global_idx0 + 1).astype(np.int64)


def _pick_center_pid(*, assets: DemoAssets, member_pids: np.ndarray, sample_k: int = 200) -> Optional[int]:
    if member_pids.size == 0:
        return None
    pids = np.asarray(member_pids, dtype=np.int64)
    pids = pids[(pids > 0)]
    if pids.size == 0:
        return None
    if pids.size > int(sample_k):
        step = max(1, int(pids.size) // int(sample_k))
        pids = pids[::step][: int(sample_k)]
    member_set = set(int(x) for x in member_pids[: min(member_pids.size, 5000)].tolist())

    best_pid = int(pids[0])
    best_score = -1.0
    for pid in pids.tolist():
        nb = assets.neighbors_of_paper0(int(pid) - 1, top_k=50)
        s = 0.0
        for idx0, w in nb:
            nb_pid = int(idx0 + 1)
            if nb_pid in member_set:
                s += float(w)
        if s > best_score:
            best_score = s
            best_pid = int(pid)
    return int(best_pid)


def _discover_kmeans_runs(base_dir: Path) -> List[Dict[str, Any]]:
    out_dir = Path(base_dir) / "out"
    runs: List[Dict[str, Any]] = []
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
        labels_path = d / "labels.npy"
        runs.append(
            {
                "run_id": d.name,
                "dir": str(d),
                "meta": meta,
                "has_labels": labels_path.exists(),
                "domains": sorted([int(p.name.split("_")[1]) for p in d.glob("domain_*_vertex_indices.npy") if p.is_file() and "_" in p.name]),
            }
        )
    return runs


@app.get("/api/v2/domains")
async def api_v2_domains(request: Request) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    return {"runs": _discover_kmeans_runs(base_dir)}


@app.get("/api/v2/domains/{run_id}/points")
async def api_v2_domain_points(
    request: Request,
    run_id: str,
    domain: int = Query(..., ge=0),
    max_points: int = Query(4000, ge=50, le=50000),
    include_title: bool = Query(False),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    out_dir = base_dir / "out"
    run_dir = out_dir / str(run_id)
    if not run_dir.exists():
        return {"error": "run_not_found", "run_id": run_id}
    vertex_path = run_dir / f"domain_{int(domain)}_vertex_indices.npy"
    if not vertex_path.exists():
        return {"error": "domain_not_found", "run_id": run_id, "domain": int(domain)}

    verts = np.load(vertex_path).astype(np.int64)
    verts = np.unique(verts)
    if verts.size == 0:
        return {"nodes": [], "meta": {"n_nodes": 0}}
    if verts.size > int(max_points):
        step = max(1, int(verts.size) // int(max_points))
        verts = verts[::step][: int(max_points)]

    assets = _assets(request)
    coords = assets.load_coords2d()
    if coords is None:
        return {"nodes": [], "meta": {"error": "coords2d_missing"}}

    papers = _papers(request)
    nodes: List[Dict[str, Any]] = []
    for idx0 in verts.tolist():
        if idx0 < 0 or idx0 >= coords.shape[0]:
            continue
        pid = int(idx0 + 1)
        node = {"id": pid, "x": float(coords[idx0, 0]), "y": float(coords[idx0, 1]), "domain": int(domain)}
        if include_title:
            try:
                p = papers[pid]
                node["label"] = str(getattr(p, "name", "") or "")[:120]
                node["year"] = int(getattr(p, "year", 0) or 0)
            except Exception:
                pass
        nodes.append(node)
    return {"nodes": nodes, "meta": {"run_id": run_id, "domain": int(domain), "n_nodes": len(nodes), "max_points": int(max_points)}}


@app.get("/api/coords/papers")
async def api_coords_papers(
    request: Request,
    ids: str = Query(..., description="Comma-separated paper ids, e.g. 1,2,3"),
    include_title: bool = Query(False),
    max_ids: int = Query(300, ge=1, le=2000),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    assets = _assets(request, run_id=run_id, resolution=resolution)
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
    pids = list(dict.fromkeys(pids))

    papers = _papers(request)
    y_use = year_min is not None or year_max is not None
    if y_use:
        pids = [
            pid
            for pid in pids
            if year_in_publication_window(_paper_year(papers[int(pid)]), year_min, year_max)
        ]
    g = _graph(request, run_id=run_id, resolution=resolution)
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


@app.get("/api/v2/hierarchies/{hid}/node/{node_id:path}")
async def api_v2_hierarchy_node(
    request: Request,
    hid: str,
    node_id: str,
    member_offset: int = Query(0, ge=0, le=10_000_000),
    member_limit: int = Query(200, ge=1, le=5000),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    base_dir = Path(cfg["base_dir"])
    datasets = _list_hierarchy_datasets(base_dir)
    ds = next((x for x in datasets if x["hid"] == hid), None)
    if ds is None:
        return {"error": "hierarchy_not_found", "hid": hid}

    rc = _parse_node_id(node_id)
    if rc is None:
        return {"error": "bad_node_id", "node_id": node_id}
    r, c = rc
    hdir = Path(ds["hierarchy_dir"])
    try:
        member_pids = _node_members_pids(hierarchy_dir=hdir, r=float(r), c=int(c))
    except Exception as e:
        return {"error": "member_lookup_failed", "node_id": node_id, "detail": f"{type(e).__name__}: {e}"}

    assets = _assets(request)
    center_pid = _pick_center_pid(assets=assets, member_pids=member_pids)
    total = int(member_pids.size)
    off = int(member_offset)
    lim = int(member_limit)
    chunk = member_pids[off : off + lim].astype(np.int64, copy=False).tolist()

    return {
        "hid": hid,
        "node_id": str(node_id),
        "resolution": float(r),
        "community": int(c),
        "size": total,
        "center_pid": None if center_pid is None else int(center_pid),
        "member_pids": chunk,
        "meta": {"offset": off, "limit": lim, "total": total},
    }


@app.get("/api/v3/catalog")
async def api_v3_catalog(request: Request) -> Dict[str, Any]:
    cat = _run_catalog(request)
    rows = [cat[k].to_json_dict() for k in sorted(cat.keys())]
    tw = sorted({str(m.time_window) for m in cat.values()})
    algos = sorted({str(m.algorithm) for m in cat.values()})
    return {
        "active_run_id": _active_run_id(request),
        "algorithms": algos if algos else ["leiden", "leiden_cpm", "louvain", "coarse_kmeans"],
        "time_windows": tw if tw else ["1y", "5y", "all"],
        "runs": rows,
    }


@app.get("/api/v3/session")
async def api_v3_session(request: Request) -> Dict[str, Any]:
    rid = _active_run_id(request)
    rt = _runtime(request, run_id=rid, resolution=None)
    m = rt["manifest"]
    g: DemoCommunityGraph = rt["graph"]
    ar = getattr(request.app.state, "active_resolution", None)
    return {
        "active_run_id": rid,
        "algorithm": m.algorithm,
        "time_window": m.time_window,
        "resolution": float(ar) if ar is not None else float(g.resolution),
        "default_resolution": float(m.default_resolution),
    }


@app.get("/api/v3/session/switch")
async def api_v3_session_switch(
    request: Request,
    run_id: str = Query(...),
    resolution: Optional[float] = Query(None),
) -> Dict[str, Any]:
    cat = _run_catalog(request)
    if run_id not in cat:
        return {"error": "run_not_found", "run_id": run_id}
    request.app.state.active_run_id = str(run_id)
    m = cat[run_id]
    if resolution is not None:
        request.app.state.active_resolution = float(resolution)  # type: ignore[attr-defined]
    else:
        request.app.state.active_resolution = float(m.default_resolution)  # type: ignore[attr-defined]
    rt = _runtime(request, run_id=run_id, resolution=None)
    g: DemoCommunityGraph = rt["graph"]
    return {
        "ok": True,
        "active_run_id": str(run_id),
        "resolution": float(getattr(request.app.state, "active_resolution", g.resolution)),
    }


@app.get("/api/v3/runs/{run_id}/resolutions")
async def api_v3_run_resolutions(
    request: Request,
    run_id: str,
) -> Dict[str, Any]:
    cat = _run_catalog(request)
    if run_id not in cat:
        return {"error": "run_not_found", "run_id": run_id, "resolutions": []}
    m = cat[run_id]
    rs = available_resolutions_for_manifest(m)
    return {"run_id": run_id, "resolutions": rs, "default_resolution": float(m.default_resolution)}


def _comparison_tags_from_env() -> Optional[List[str]]:
    raw = os.environ.get("PC_EVAL_COMPARISON_RUN_TAGS", "").strip()
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


@app.get("/api/v3/evaluations/overview")
async def api_v3_evaluations_overview(request: Request) -> Dict[str, Any]:
    cat = _run_catalog(request)
    bundle = build_evaluation_overview(
        list(cat.values()),
        repo_root=REPO_ROOT,
        comparison_run_tags=_comparison_tags_from_env(),
    )
    return bundle.to_json_dict()


@app.get("/api/v3/evaluations/retrieval-sixway")
async def api_v3_evaluations_retrieval_sixway(
    request: Request,
    comparison_run_tag: str = Query("master_breakpoints"),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    return read_retrieval_sixway_rows(Path(cfg["base_dir"]), comparison_run_tag=comparison_run_tag)


@app.get("/api/tools/random-paper")
async def api_tools_random_paper(request: Request) -> Dict[str, Any]:
    papers = _papers(request)
    cands = [i for i in range(1, len(papers)) if papers[i] is not None]
    if not cands:
        raise HTTPException(status_code=404, detail="no_papers")
    return {"pid": int(random.choice(cands))}


@app.get("/api/search/vector_nn")
async def api_search_vector_nn(
    request: Request,
    paper_id: int = Query(..., ge=1),
    top_k: int = Query(20, ge=1, le=200),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    domain_run_id: Optional[str] = Query(None),
    domain: Optional[int] = Query(None, ge=0),
) -> Dict[str, Any]:
    emb = getattr(request.app.state, "emb_norm", None)
    if emb is None:
        return {
            "type": "vector_nn",
            "query": {"pid": paper_id},
            "hits": [],
            "error": "embeddings_unavailable",
            "detail": getattr(request.app.state, "emb_load_error", None),
        }
    cfg = request.app.state.config
    graph = _graph(request, run_id=run_id, resolution=resolution)
    return search_vector_nn(
        papers=_papers(request),
        graph=graph,
        emb_norm=emb,
        pid=paper_id,
        top_k=top_k,
        authors=_authors(request),
        domain_run_id=domain_run_id,
        domain=domain,
        base_dir=Path(cfg["base_dir"]),
    )


@app.get("/api/search/community_bundle")
async def api_search_community_bundle(
    request: Request,
    paper_id: int = Query(..., ge=1),
    partition_run_id: str = Query(..., description="Catalog run_id for partition (e.g. leiden_cpm)"),
    resolution: Optional[float] = Query(None),
    top_k: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    cat = _run_catalog(request)
    if partition_run_id not in cat:
        raise HTTPException(status_code=400, detail="partition_run_id not_in_catalog")
    m = cat[partition_run_id]
    res_eff = float(resolution if resolution is not None else m.default_resolution)
    kw_dir = Path(m.keyword_index_dir) if m.keyword_index_dir else Path(cfg["keyword_index_dir"])
    return search_community_bundle(
        base_dir=Path(cfg["base_dir"]),
        manifest=m,
        papers=_papers(request),
        authors=_authors(request),
        pid=paper_id,
        resolution=res_eff,
        keyword_index_dir=kw_dir,
        top_k=top_k,
    )


@app.get("/api/v3/retrieval/live")
async def api_v3_retrieval_live(
    request: Request,
    seed_pid: int = Query(..., ge=1),
    method: str = Query(..., description="keyword | vector_nn | community_bundle"),
    partition_run_id: str = Query(...),
    resolution: Optional[float] = Query(None),
    top_k: int = Query(20, ge=1, le=200),
    comparison_run_tag: str = Query("master_breakpoints"),
) -> Dict[str, Any]:
    emb = getattr(request.app.state, "emb_norm", None)
    cfg = request.app.state.config
    cat = _run_catalog(request)
    if partition_run_id not in cat:
        raise HTTPException(status_code=400, detail="partition_run_id not_in_catalog")
    m = cat[partition_run_id]
    res_eff = float(resolution if resolution is not None else m.default_resolution)
    kw_dir = Path(m.keyword_index_dir) if m.keyword_index_dir else Path(cfg["keyword_index_dir"])
    if emb is None:
        return {
            "error": "embeddings_unavailable",
            "detail": getattr(request.app.state, "emb_load_error", None),
        }
    out = run_live_retrieval_for_seed(
        base_dir=Path(cfg["base_dir"]),
        partition_manifest=m,
        papers=_papers(request),
        authors=_authors(request),
        emb_norm=emb,
        seed_pid=seed_pid,
        method=method,
        resolution=res_eff,
        keyword_index_dir=kw_dir,
        top_k=top_k,
    )
    if out.get("error"):
        return out
    six = read_retrieval_sixway_rows(Path(cfg["base_dir"]), comparison_run_tag=comparison_run_tag)
    rows = six.get("rows") or []
    ml = str(method).strip().lower()
    if ml == "community_bundle":
        method_row = f"community_bundle:{partition_run_id}"
    else:
        method_row = ml
    out["sixway_tag"] = comparison_run_tag
    out["sixway_comparison"] = live_vs_sixway_positions(
        sixway_rows=rows,
        live_summary=out.get("summary") or {},
        method_row_method=method_row,
    )
    return out


def _safe_eval_artifact_file(request: Request, run_id: str, rel_path: str) -> Path:
    cat = _run_catalog(request)
    if run_id not in cat:
        raise HTTPException(status_code=404, detail="run_not_found")
    base = Path(cat[run_id].leiden_dir).resolve()
    rel_path = str(rel_path or "").strip().lstrip("/")
    if not rel_path or ".." in Path(rel_path).parts:
        raise HTTPException(status_code=400, detail="bad_rel_path")
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="path_not_under_leiden_dir")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file_not_found")
    return target


@app.get("/api/v3/runs/{run_id}/eval-artifact/{rel_path:path}")
async def api_v3_run_eval_artifact(request: Request, run_id: str, rel_path: str):
    target = _safe_eval_artifact_file(request, run_id, rel_path)
    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream")


@app.get("/api/health")
async def api_health(
    request: Request,
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
) -> Dict[str, Any]:
    cfg = request.app.state.config
    cat = _run_catalog(request)
    rid = str(run_id).strip() if run_id else _active_run_id(request)
    if rid not in cat:
        rid = _active_run_id(request)
    rt = _runtime(request, run_id=rid, resolution=resolution)
    m = rt["manifest"]
    g: DemoCommunityGraph = rt["graph"]
    root = Path(cfg["base_dir"])
    tc_attr = getattr(m, "topic_communities_csv", None)
    tc_manifest_str = str(tc_attr).strip() if tc_attr else None
    if tc_manifest_str == "":
        tc_manifest_str = None
    env_tc = cfg.get("topic_communities_csv")
    env_p = env_tc if isinstance(env_tc, Path) else None
    tc_path_obj = resolve_topic_weights_csv_for_web(
        base_dir=root,
        run_id=str(m.run_id),
        resolution=float(g.resolution),
        manifest_topic_csv=tc_manifest_str,
        env_topic_csv=env_p,
        manifest_tags=m.tags if isinstance(m.tags, dict) else None,
    )
    tc_resolved = str(tc_path_obj) if tc_path_obj is not None else None
    sample_topic = False
    for _cid, comm in list(g.communities.items())[:40]:
        if getattr(comm, "topic_info", None):
            sample_topic = True
            break
    return {
        "status": "ok",
        "n_papers": int(g.n_papers),
        "n_communities": int(g.n_communities),
        "resolution": float(g.resolution),
        "active_run_id": rid,
        "algorithm": m.algorithm,
        "time_window": m.time_window,
        "tags": getattr(m, "tags", None) or {},
        "leiden_dir": str(m.leiden_dir),
        "graph_npz": str(m.graph_npz),
        "keyword_index_dir": str(m.keyword_index_dir or cfg["keyword_index_dir"]),
        "paper_year_min": getattr(request.app.state, "paper_year_min", None),
        "paper_year_max": getattr(request.app.state, "paper_year_max", None),
        "topic_communities_csv": tc_resolved,
        "topic_communities_csv_loaded": bool(tc_resolved and sample_topic),
        "embeddings_loaded": getattr(request.app.state, "emb_norm", None) is not None,
        "embeddings_error": getattr(request.app.state, "emb_load_error", None),
    }


@app.get("/api/search/keyword")
async def api_search_keyword(
    request: Request,
    q: str = Query(..., description="检索词"),
    top_k: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0, le=20000),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    return search_keyword(
        assets=_assets(request, run_id=run_id, resolution=resolution),
        papers=_papers(request),
        graph=_graph(request, run_id=run_id, resolution=resolution),
        query=q,
        top_k=int(top_k),
        offset=int(offset),
        year_min=year_min,
        year_max=year_max,
        authors=_authors(request),
    )


@app.get("/api/papers/{paper_id}")
async def api_paper(
    request: Request,
    paper_id: int,
    k_neighbors: int = Query(20, ge=1, le=200),
    k_neighbors_in_comm: int = Query(10, ge=0, le=200),
    k_neighbor_comms: int = Query(8, ge=0, le=100),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    return lookup_paper(
        assets=_assets(request, run_id=run_id, resolution=resolution),
        papers=_papers(request),
        graph=_graph(request, run_id=run_id, resolution=resolution),
        pid=int(paper_id),
        k_neighbors=int(k_neighbors),
        k_neighbors_in_comm=int(k_neighbors_in_comm),
        k_neighbor_comms=int(k_neighbor_comms),
        authors=_authors(request),
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/api/communities/{community_id}")
async def api_community(
    request: Request,
    community_id: int,
    top_papers: int = Query(20, ge=1, le=500),
    top_neighbors: int = Query(12, ge=0, le=200),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    return lookup_community(
        papers=_papers(request),
        graph=_graph(request, run_id=run_id, resolution=resolution),
        cid=int(community_id),
        top_papers=int(top_papers),
        top_neighbors=int(top_neighbors),
        authors=_authors(request),
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/api/v2/communities/{node_id:path}")
async def api_v2_community(
    request: Request,
    node_id: str,
    top_papers: int = Query(20, ge=1, le=500),
    top_neighbors: int = Query(12, ge=0, le=200),
    run_id: Optional[str] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    try:
        s = str(node_id)
        a, b = s.split("|", 1)
        r = float(a.split("=", 1)[1])
        c = int(b.split("=", 1)[1])
    except Exception:
        return {"error": "bad_node_id", "node_id": str(node_id)}
    g = _graph(request, run_id=run_id, resolution=float(r))
    return lookup_community_at_resolution(
        assets=_assets(request, run_id=run_id, resolution=float(r)),
        papers=_papers(request),
        resolution=float(r),
        cid=int(c),
        top_papers=int(top_papers),
        top_neighbors=int(top_neighbors),
        authors=_authors(request),
        graph=g,
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/api/expand/paper/{paper_id}")
async def api_expand(
    request: Request,
    paper_id: int,
    k_papers: int = Query(20, ge=1, le=200),
    k_comms: int = Query(10, ge=0, le=200),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    return expand_from_paper(
        assets=_assets(request, run_id=run_id, resolution=resolution),
        papers=_papers(request),
        graph=_graph(request, run_id=run_id, resolution=resolution),
        pid=int(paper_id),
        k_papers=int(k_papers),
        k_comms=int(k_comms),
        authors=_authors(request),
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/api/graph/communities")
async def api_graph_communities(
    request: Request,
    # 上限与 /api/v2/hierarchies/.../lineage 等路由一致；若仍见 le=5000 的 422，说明进程未加载本文件（需重启 demo-api）。
    max_nodes: int = Query(1500, ge=10, le=200_000),
    min_weight: float = Query(0.0, ge=0.0),
    sample_papers_per_comm: int = Query(400, ge=10, le=5000),
    include_positions: bool = Query(False),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    assets = _assets(request, run_id=run_id, resolution=resolution)
    graph = _graph(request, run_id=run_id, resolution=resolution)
    y_use = year_min is not None or year_max is not None
    return community_graph_payload(
        assets,
        graph,
        max_nodes=max_nodes,
        min_weight=min_weight,
        sample_papers_per_comm=int(sample_papers_per_comm),
        include_positions=bool(include_positions),
        papers=_papers(request) if y_use else None,
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/api/graph/community/{community_id}")
async def api_graph_community_papers(
    request: Request,
    community_id: int,
    max_nodes: int = Query(60, ge=5, le=5000),
    max_edges: int = Query(200, ge=10, le=8000),
    run_id: Optional[str] = Query(None),
    resolution: Optional[float] = Query(None),
    year_min: Optional[int] = Query(None),
    year_max: Optional[int] = Query(None),
) -> Dict[str, Any]:
    assets = _assets(request, run_id=run_id, resolution=resolution)
    graph = _graph(request, run_id=run_id, resolution=resolution)
    return community_paper_subgraph_payload(
        assets,
        _papers(request),
        graph,
        int(community_id),
        max_nodes=int(max_nodes),
        max_edges=int(max_edges),
        year_min=year_min,
        year_max=year_max,
    )


@app.get("/", response_model=None)
async def root():
    index_path = _WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"docs": "/docs", "health": "/api/health"}
