from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import igraph as ig
import numpy as np

from checklist import run_embedding_checks, run_model_checks
from community import (
    leiden_sweep,
    load_membership_for_resolution,
    pick_nearest_resolution,
    run_hierarchy_sweep,
)
from diagram2d import embed_2d, graph_layout_2d, plot_scatter
from embedding import embed_all_papers
from getdata import ingest, load_data
from model import build_models
from network import build_or_load_mutual_knn_graph
from time_window import analyze_time_window, collect_time_info, make_sliding_window_video


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

AUTHOR_NAME_TXT = DATA_DIR / "author_name.txt"
AUTHORPAPER_RDATA = DATA_DIR / "AuthorPaperInfo_py.RData"
TEXTCORPUS_RDATA = DATA_DIR / "TextCorpusFinal_py.RData"
TOPICRESULTS_RDATA = DATA_DIR / "TopicResults_py.RData"
RAWPAPER_RDATA = DATA_DIR / "RawPaper_py.RData"

EMB_PATH = DATA_DIR / "paper_embeddings_specter2.npy"
CACHE_PATH = DATA_DIR / "data_store.pkl"


# -----------------------------------------------------------------------------
# 基础加载
# -----------------------------------------------------------------------------

def build_or_load(*, exclude_selfcite: bool = False, force_reingest: bool = False):
    if force_reingest and CACHE_PATH.exists():
        CACHE_PATH.unlink()
    if not CACHE_PATH.exists():
        ingest(
            authorpaper_rdata=AUTHORPAPER_RDATA,
            author_name_txt=AUTHOR_NAME_TXT,
            textcorpus_rdata=TEXTCORPUS_RDATA,
            topicresults_rdata=TOPICRESULTS_RDATA,
            rawpaper_rdata=RAWPAPER_RDATA,
            out_path=CACHE_PATH,
            exclude_selfcite=exclude_selfcite,
        )
    data = load_data(CACHE_PATH)
    authors, papers = build_models(data)
    return authors, papers, data


def build_or_load_embeddings(
    papers,
    *,
    emb_path: Path = EMB_PATH,
    batch_size: int = 16,
    prefer_gpu: bool = False,
    force: bool = False,
) -> np.ndarray:
    emb_path = Path(emb_path)
    if emb_path.exists() and not force:
        embs = np.load(emb_path, mmap_mode="r")
        print("[emb] loaded from disk:", embs.shape, embs.dtype, "example:", embs[1, :5])
        return embs
    embs = embed_all_papers(
        papers=papers,
        out_npy_path=str(emb_path),
        batch_size=batch_size,
        prefer_gpu=prefer_gpu,
        attach_to_papers=False,
    )
    print("[emb] computed and saved:", embs.shape, embs.dtype)
    return embs


def build_or_load_global_2d(
    embs: np.ndarray,
    *,
    out_dir: Path = OUT_DIR,
    cache_name: str = "umap2d.npy",
    method: str = "umap",
    random_state: int = 42,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    force: bool = False,
) -> np.ndarray:
    X = np.asarray(embs[1:], dtype=np.float32)
    cache_path = Path(out_dir) / cache_name
    if force and cache_path.exists():
        cache_path.unlink()
    return embed_2d(
        X,
        method=method,
        normalize=True,
        pca_dim=50,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric="cosine",
        random_state=random_state,
        cache_npy=cache_path,
        verbose=True,
    )


def build_or_load_global_graph(
    embs: np.ndarray,
    *,
    out_dir: Path = OUT_DIR,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    cache_name: Optional[str] = None,
    force: bool = False,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    X = np.asarray(embs[1:], dtype=np.float32)
    if cache_name is None:
        cache_name = f"mutual_knn_k{k}.npz"
    edges_cache = Path(out_dir) / cache_name
    if force and edges_cache.exists():
        edges_cache.unlink()
    return build_or_load_mutual_knn_graph(
        X,
        k=k,
        cache_npz=edges_cache,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
        normalize=True,
        verbose=True,
    )


def build_igraph_from_edge_triplets(n_nodes: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> ig.Graph:
    G = ig.Graph(n=int(n_nodes), edges=list(zip(u.tolist(), v.tolist())), directed=False)
    G.es["weight"] = np.asarray(w, dtype=np.float32).astype(float).tolist()
    return G


def prepare_global_pipeline(
    *,
    exclude_selfcite: bool = False,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    emb_path: Path = EMB_PATH,
) -> Dict[str, Any]:
    authors, papers, data = build_or_load(exclude_selfcite=exclude_selfcite)
    embs = build_or_load_embeddings(papers, emb_path=emb_path)
    X = np.asarray(embs[1:], dtype=np.float32)
    print("[core] X:", X.shape, X.dtype)
    Y = build_or_load_global_2d(embs, out_dir=OUT_DIR)
    plot_scatter(
        Y,
        title="UMAP(2D) of paper embeddings",
        out_png=OUT_DIR / "fig_umap2d.png",
        point_size=1.0,
        alpha=0.6,
        max_points=None,
    )
    A_sym, (u, v, w) = build_or_load_global_graph(
        embs,
        out_dir=OUT_DIR,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
    )
    G = build_igraph_from_edge_triplets(X.shape[0], u, v, w)
    time_info = collect_time_info(papers, out_dir=OUT_DIR / "time_info", verbose=True)
    return {
        "authors": authors,
        "papers": papers,
        "data": data,
        "embs": embs,
        "X": X,
        "Y": Y,
        "A_sym": A_sym,
        "u": u,
        "v": v,
        "w": w,
        "G": G,
        "time_info": time_info,
    }


# -----------------------------------------------------------------------------
# 时间窗口
# -----------------------------------------------------------------------------

def run_single_time_window(
    *,
    start_year: int,
    end_year: int,
    resolution: float,
    exclude_selfcite: bool = False,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
) -> Dict[str, Any]:
    ctx = prepare_global_pipeline(
        exclude_selfcite=exclude_selfcite,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
    )
    return analyze_time_window(
        ctx["papers"],
        ctx["embs"],
        ctx["Y"],
        ctx["u"],
        ctx["v"],
        ctx["w"],
        start_year=start_year,
        end_year=end_year,
        resolution=resolution,
        out_dir=OUT_DIR / "time_windows",
        global_graph=ctx["G"],
        global_leiden_dir=OUT_DIR / "leiden_global_single",
        time_info=ctx["time_info"],
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
        normalize=True,
        seed=42,
        point_size=3.0,
        alpha=0.85,
        verbose=True,
    )


def run_time_window_animation(
    *,
    resolution: float,
    window_size: int = 5,
    step: int = 1,
    fps: int = 2,
    exclude_selfcite: bool = False,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
) -> Dict[str, Any]:
    ctx = prepare_global_pipeline(
        exclude_selfcite=exclude_selfcite,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
    )
    return make_sliding_window_video(
        ctx["papers"],
        ctx["embs"],
        ctx["Y"],
        ctx["u"],
        ctx["v"],
        ctx["w"],
        resolution=resolution,
        out_dir=OUT_DIR / "time_windows_animation",
        global_graph=ctx["G"],
        global_leiden_dir=OUT_DIR / "leiden_global_single",
        time_info=ctx["time_info"],
        window_size=window_size,
        step=step,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
        normalize=True,
        seed=42,
        fps=fps,
        point_size=3.0,
        alpha=0.85,
        verbose=True,
    )


# -----------------------------------------------------------------------------
# 实用工具
# -----------------------------------------------------------------------------

def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _subprocess_run(cmd: list[str]) -> Dict[str, Any]:
    subprocess.run(cmd, check=True)
    return {"cmd": cmd}


def _topic_root_for_k(k_topics: int) -> Path:
    return OUT_DIR / "topic_modeling_multi" / f"K{k_topics}"


# -----------------------------------------------------------------------------
# 任务
# -----------------------------------------------------------------------------

def task_check_data(args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.out_path) if args.out_path else (DATA_DIR / "data_check.txt")
    if out_path.exists() and not args.force:
        print(f"[skip] data check already exists -> {out_path}")
        return {"report": str(out_path), "skipped": True}
    authors, papers, data = build_or_load(
        exclude_selfcite=args.exclude_selfcite,
        force_reingest=args.force_reingest,
    )
    summary = run_model_checks(
        authors,
        papers,
        data,
        seed=42,
        sample_authors=80,
        sample_papers=120,
        max_show_examples=5,
        write_report_path=str(out_path),
    )
    return {"report": str(out_path), "summary": summary}


def task_embed(args: argparse.Namespace) -> Dict[str, Any]:
    _, papers, _ = build_or_load(
        exclude_selfcite=args.exclude_selfcite,
        force_reingest=args.force_reingest,
    )
    embs = build_or_load_embeddings(
        papers,
        emb_path=Path(args.emb_path),
        batch_size=args.batch_size,
        prefer_gpu=args.prefer_gpu,
        force=args.force,
    )
    return {"emb_path": str(args.emb_path), "shape": tuple(int(x) for x in embs.shape)}


def task_check_embed(args: argparse.Namespace) -> Dict[str, Any]:
    out_path = Path(args.out_path) if args.out_path else (DATA_DIR / "embedding_check.txt")
    if out_path.exists() and not args.force:
        print(f"[skip] embedding check already exists -> {out_path}")
        return {"report": str(out_path), "skipped": True}
    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    embs = build_or_load_embeddings(papers, emb_path=Path(args.emb_path))
    summary = run_embedding_checks(
        papers=papers,
        embs=embs,
        expected_dim=768,
        seed=42,
        sample_papers=8,
        sample_pairs=12,
        write_report_path=str(out_path),
    )
    return {"report": str(out_path), "summary": summary}


def task_build_2d(args: argparse.Namespace) -> Dict[str, Any]:
    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    embs = build_or_load_embeddings(papers, emb_path=Path(args.emb_path))
    Y = build_or_load_global_2d(
        embs,
        out_dir=OUT_DIR,
        cache_name=args.cache_name,
        method=args.method,
        random_state=args.seed,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        force=args.force,
    )
    out_png = OUT_DIR / args.out_png
    if not out_png.exists() or args.force:
        plot_scatter(
            Y,
            title=f"{args.method.upper()}(2D) of paper embeddings",
            out_png=out_png,
            point_size=args.point_size,
            alpha=args.alpha,
            max_points=None,
        )
    else:
        print(f"[skip] 2D plot already exists -> {out_png}")
    return {"xy_path": str(OUT_DIR / args.cache_name), "plot": str(out_png), "shape": tuple(int(x) for x in Y.shape)}


def task_build_graph(args: argparse.Namespace) -> Dict[str, Any]:
    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    embs = build_or_load_embeddings(papers, emb_path=Path(args.emb_path))
    A_sym, (u, v, w) = build_or_load_global_graph(
        embs,
        out_dir=OUT_DIR,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
        cache_name=args.cache_name,
        force=args.force,
    )
    return {
        "graph_cache": str(OUT_DIR / (args.cache_name or f"mutual_knn_k{args.k}.npz")),
        "n_nodes": int(A_sym.shape[0]),
        "n_edges": int(len(u)),
        "k": int(args.k),
    }


def task_graph_layout(args: argparse.Namespace) -> Dict[str, Any]:
    ctx = prepare_global_pipeline(
        exclude_selfcite=args.exclude_selfcite,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
    )
    out_npy = OUT_DIR / args.cache_name
    if args.force and out_npy.exists():
        out_npy.unlink()
    Yg = graph_layout_2d(
        n_nodes=ctx["X"].shape[0],
        u=ctx["u"],
        v=ctx["v"],
        w=ctx["w"],
        method=args.method,
        init_xy=ctx["Y"] if args.init_from_umap else None,
        cache_npy=out_npy,
        verbose=True,
    )
    out_png = OUT_DIR / args.out_png
    if not out_png.exists() or args.force:
        plot_scatter(
            Yg,
            title=f"Graph layout ({args.method})",
            out_png=out_png,
            point_size=args.point_size,
            alpha=args.alpha,
            max_points=None,
        )
    else:
        print(f"[skip] graph layout plot already exists -> {out_png}")
    return {"xy_path": str(out_npy), "plot": str(out_png), "shape": tuple(int(x) for x in Yg.shape)}


def task_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    embs = build_or_load_embeddings(papers, emb_path=Path(args.emb_path))
    X = np.asarray(embs[1:], dtype=np.float32)
    _, (u, v, w) = build_or_load_global_graph(
        embs,
        out_dir=OUT_DIR,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
    )
    G = build_igraph_from_edge_triplets(X.shape[0], u, v, w)
    results = leiden_sweep(
        G,
        out_dir=Path(args.out_dir),
        r_min=args.r_min,
        r_max=args.r_max,
        step=args.step,
        include=args.include,
        seed=args.seed,
        save_each_membership=True,
        reuse_existing=(not args.force),
        resolution_mode=args.resolution_mode,
        partition_type=args.partition_type,
        verbose=True,
    )
    if args.plot_reference is not None:
        Y = build_or_load_global_2d(embs, out_dir=OUT_DIR)
        r0 = pick_nearest_resolution(results, args.plot_reference)
        labels = results[r0]["membership"]
        plot_scatter(
            Y,
            labels=labels,
            title=f"UMAP(2D) colored by Leiden (r={r0:.4f})",
            out_png=Path(args.out_dir) / f"umap_r{r0:.4f}.png",
            point_size=args.point_size,
            alpha=args.alpha,
            max_points=None,
        )
    summary = np.load(Path(args.out_dir) / "summary.npy", allow_pickle=True).item()
    return {
        "out_dir": str(args.out_dir),
        "n_resolutions": int(len(results)),
        "resolution_min": float(summary["resolutions"][0]),
        "resolution_max": float(summary["resolutions"][-1]),
    }


def task_hierarchy(args: argparse.Namespace) -> Dict[str, Any]:
    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    embs = build_or_load_embeddings(papers, emb_path=Path(args.emb_path))
    X = np.asarray(embs[1:], dtype=np.float32)
    _, (u, v, w) = build_or_load_global_graph(
        embs,
        out_dir=OUT_DIR,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
    )
    G = build_igraph_from_edge_triplets(X.shape[0], u, v, w)
    res = run_hierarchy_sweep(
        G,
        out_dir=Path(args.out_dir),
        r_min=args.r_min,
        r_max=args.r_max,
        step=args.step,
        include=args.include,
        seed=args.seed,
        save_each_membership=True,
        reuse_existing=(not args.force),
        resolution_mode=args.resolution_mode,
        partition_type=args.partition_type,
        min_child_share=args.min_child_share,
        verbose=True,
    )
    bps = res["hierarchy"]["breakpoints"]
    print("[hierarchy] top breakpoints:")
    for bp in bps[:10]:
        print(f"  r={bp['resolution']:.4f}  score={bp['score']:.3f}  ΔC={bp['delta_n_comm']}  VI={bp['vi_adjacent']}")
    return {
        "out_dir": str(args.out_dir),
        "n_nodes": int(len(res["hierarchy"]["nodes"])),
        "n_edges": int(len(res["hierarchy"]["edges"])),
        "n_breakpoints": int(len(bps)),
    }


def task_time_window(args: argparse.Namespace) -> Dict[str, Any]:
    return run_single_time_window(
        start_year=args.start_year,
        end_year=args.end_year,
        resolution=args.resolution,
        exclude_selfcite=args.exclude_selfcite,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
    )


def task_time_video(args: argparse.Namespace) -> Dict[str, Any]:
    return run_time_window_animation(
        resolution=args.resolution,
        window_size=args.window_size,
        step=args.step,
        fps=args.fps,
        exclude_selfcite=args.exclude_selfcite,
        k=args.k,
        knn_backend=args.knn_backend,
        knn_batch_size=args.knn_batch_size,
    )


def task_keyword_index(args: argparse.Namespace) -> Dict[str, Any]:
    from retrieval import KeywordIndexConfig, build_or_load_keyword_index

    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    cfg = KeywordIndexConfig(
        ngram_min=1,
        ngram_max=2 if args.use_bigrams else 1,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        use_title=(not args.no_title),
        use_abstract=(not args.no_abstract),
        title_boost=args.title_boost,
    )
    res = build_or_load_keyword_index(papers, out_dir=OUT_DIR, cfg=cfg, force=args.force, verbose=True)
    return {"index_dir": str(res["index_dir"]), "meta": res["meta"]}


def task_keyword_search(args: argparse.Namespace) -> Dict[str, Any]:
    from retrieval import KeywordIndexConfig, search_keywords

    _, papers, _ = build_or_load(exclude_selfcite=args.exclude_selfcite, force_reingest=args.force_reingest)
    cfg = KeywordIndexConfig(
        ngram_min=1,
        ngram_max=2 if args.use_bigrams else 1,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        use_title=(not args.no_title),
        use_abstract=(not args.no_abstract),
        title_boost=args.title_boost,
    )
    res = search_keywords(
        papers,
        query=args.query,
        out_dir=OUT_DIR,
        top_k=args.top_k,
        resolution=args.resolution,
        leiden_dir=Path(args.leiden_dir) if args.resolution is not None else None,
        cfg=cfg,
        force_reindex=args.force,
        verbose=True,
    )
    print(f"\n[keyword-search] query={args.query!r}\n")
    for i, hit in enumerate(res["hits"], start=1):
        comm = "-" if hit["community"] is None else str(hit["community"])
        print(f"{i:02d}. PID={hit['pid']:5d}  score={hit['score']:.4f}  year={hit['year']}  comm={comm}")
        print(f"    {hit['title']}")
        if hit["snippet"]:
            print(f"    {hit['snippet']}")
    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def task_topic_model(args: argparse.Namespace) -> Dict[str, Any]:
    import topic_modeling as tm

    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / "topic_modeling" / f"K{args.k_topics}" / f"r{float(args.resolution):.4f}")
    meta_path = out_dir / "topic_model_meta.json"
    if meta_path.exists() and not args.force:
        print(f"[skip] topic model already exists -> {out_dir}")
        return {"out_dir": str(out_dir), "skipped": True}

    t0 = time.time()
    data = tm.load_data_store(Path(args.data_store))
    membership_path = Path(args.membership) if args.membership else tm.membership_path_from_resolution(OUT_DIR, args.resolution)
    membership = tm.load_membership(membership_path, n_papers_expected=int(data["n_papers"]))
    stopwords = tm.load_stopwords(Path(args.stopwords) if args.stopwords else None, add_sklearn_english=(not args.no_sklearn_stopwords))

    build_cfg = tm.BuildMatrixConfig(
        include_title=not args.no_title,
        include_abstract=not args.no_abstract,
        include_authors=not args.no_authors,
        use_clean_abstract=args.use_clean_abstract,
        title_weight=args.title_weight,
        abstract_weight=args.abstract_weight,
        author_weight=args.author_weight,
        min_token_len=args.min_token_len,
        words_percent=args.words_percent,
        docs_percent=args.docs_percent,
        min_community_size=args.min_community_size,
        max_papers=args.max_papers,
    )
    matrix_res = tm.build_community_term_matrix(data, membership, stopwords, build_cfg)
    score_cfg = tm.TopicScoreConfig(
        k=args.k_topics,
        vh_method=args.vh_method,
        m=args.m,
        k0=args.k0,
        mquantile=args.mquantile,
        m_trunc_mode=args.m_trunc_mode,
        seed=args.seed,
        max_svs_combinations=args.max_svs_combinations,
        weighted_nnls=(not args.no_weighted_nnls),
    )
    topic_res = tm.fit_topic_score_on_communities(matrix_res, score_cfg)
    tm.save_outputs(
        out_dir,
        data,
        matrix_res,
        topic_res,
        build_cfg,
        score_cfg,
        runtime_sec=time.time() - t0,
        rep_papers_mode=args.rep_papers_mode,
    )
    print(f"[done] topic model -> {out_dir}")
    return {"out_dir": str(out_dir), "membership": str(membership_path), "n_topics": int(args.k_topics)}


def task_topic_model_multi(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "topic_modeling_multi.py"), "--k", str(args.k_topics)]
    cmd += ["--leiden-dir", str(args.leiden_dir)]
    if args.out_root:
        cmd += ["--out-root", str(args.out_root)]
    if args.r_min is not None:
        cmd += ["--r-min", str(args.r_min)]
    if args.r_max is not None:
        cmd += ["--r-max", str(args.r_max)]
    if args.step is not None:
        cmd += ["--step", str(args.step)]
    if args.include:
        cmd += ["--include", *[str(x) for x in args.include]]
    if args.resolutions:
        cmd += ["--resolutions", *[str(x) for x in args.resolutions]]
    if args.skip_existing:
        cmd += ["--skip-existing"]
    if args.continue_on_error:
        cmd += ["--continue-on-error"]
    if args.quiet:
        cmd += ["--quiet"]
    return _subprocess_run(cmd)


def task_align_topics(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "align_topics_multires.py"), "--k", str(args.k_topics)]
    if args.topic_root:
        cmd += ["--topic-root", str(args.topic_root)]
    if args.out_root:
        cmd += ["--out-root", str(args.out_root)]
    cmd += ["--ref-resolution", str(args.ref_resolution)]
    cmd += ["--metric", str(args.metric)]
    cmd += ["--min-common-vocab", str(args.min_common_vocab)]
    if args.topn_common_vocab is not None:
        cmd += ["--topn-common-vocab", str(args.topn_common_vocab)]
    if args.r_min is not None:
        cmd += ["--r-min", str(args.r_min)]
    if args.r_max is not None:
        cmd += ["--r-max", str(args.r_max)]
    if args.include:
        cmd += ["--include", *[str(x) for x in args.include]]
    if args.resolutions:
        cmd += ["--resolutions", *[str(x) for x in args.resolutions]]
    if args.copy_meta_json:
        cmd += ["--copy-meta-json"]
    if args.save_sim_matrix:
        cmd += ["--save-sim-matrix"]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.quiet:
        cmd += ["--quiet"]
    return _subprocess_run(cmd)


def task_align_topics_segmented(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "align_topics_multires_segmented.py"), "--k", str(args.k_topics)]
    if args.topic_root:
        cmd += ["--topic-root", str(args.topic_root)]
    if args.out_root:
        cmd += ["--out-root", str(args.out_root)]
    cmd += ["--metric", str(args.metric)]
    cmd += ["--min-common-vocab", str(args.min_common_vocab)]
    if args.topn_common_vocab is not None:
        cmd += ["--topn-common-vocab", str(args.topn_common_vocab)]
    if args.r_min is not None:
        cmd += ["--r-min", str(args.r_min)]
    if args.r_max is not None:
        cmd += ["--r-max", str(args.r_max)]
    if args.include:
        cmd += ["--include", *[str(x) for x in args.include]]
    if args.resolutions:
        cmd += ["--resolutions", *[str(x) for x in args.resolutions]]
    cmd += ["--break-avg-thresh", str(args.break_avg_thresh)]
    cmd += ["--break-min-thresh", str(args.break_min_thresh)]
    cmd += ["--min-segment-size", str(args.min_segment_size)]
    cmd += ["--anchor-strategy", str(args.anchor_strategy)]
    if args.save_sim_matrix:
        cmd += ["--save-sim-matrix"]
    if args.save_adjacent_topic_rows:
        cmd += ["--save-adjacent-topic-rows"]
    if args.clean_segment_dirs:
        cmd += ["--clean-segment-dirs"]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.quiet:
        cmd += ["--quiet"]
    return _subprocess_run(cmd)


def task_topic_viz(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "topic_visualization_multires.py"), "--k", str(args.k_topics)]
    if args.leiden_dir:
        cmd += ["--leiden-dir", str(args.leiden_dir)]
    if args.topic_root:
        cmd += ["--topic-root", str(args.topic_root)]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    cmd += ["--umap", str(args.umap)]
    cmd += ["--graph-layout", str(args.graph_layout)]
    if args.resolutions:
        cmd += ["--resolutions", *[str(x) for x in args.resolutions]]
    if args.r_min is not None:
        cmd += ["--r-min", str(args.r_min)]
    if args.r_max is not None:
        cmd += ["--r-max", str(args.r_max)]
    if args.include:
        cmd += ["--include", *[str(x) for x in args.include]]
    if args.step is not None:
        cmd += ["--step", str(args.step)]
    if args.max_points is not None:
        cmd += ["--max-points", str(args.max_points)]
    cmd += ["--point-size", str(args.point_size), "--alpha", str(args.alpha)]
    if args.skip_graph:
        cmd += ["--skip-graph"]
    if args.community_centroid:
        cmd += ["--community-centroid"]
    if args.annotate_top_n_communities:
        cmd += ["--annotate-top-n-communities", str(args.annotate_top_n_communities)]
    if args.batch_segments:
        cmd += ["--batch-segments"]
    if args.segments_csv:
        cmd += ["--segments-csv", str(args.segments_csv)]
    if args.segmented_out_root:
        cmd += ["--segmented-out-root", str(args.segmented_out_root)]
    if args.clean_segment_out:
        cmd += ["--clean-segment-out"]
    if args.quiet:
        cmd += ["--quiet"]
    return _subprocess_run(cmd)


def task_diagnose_topic_collapse(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "diagnose_topic_collapse.py")]
    if args.input:
        cmd += ["--input", str(args.input)]
    if args.root:
        cmd += ["--root", str(args.root)]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.save_per_topic:
        cmd += ["--save-per-topic"]
    if args.save_plots:
        cmd += ["--save-plots"]
    if args.verbose:
        cmd += ["--verbose"]
    return _subprocess_run(cmd)


def task_frames_to_mp4(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, str(SRC_DIR / "frames_to_mp4.py")]
    if args.batch_subdirs:
        cmd += ["--batch-subdirs", "--root", str(args.root)]
        if args.subdir_glob:
            cmd += ["--subdir-glob", str(args.subdir_glob)]
        if args.out_root:
            cmd += ["--out-root", str(args.out_root)]
    else:
        cmd += ["--frame-dir", str(args.frame_dir)]
        if args.out:
            cmd += ["--out", str(args.out)]
    cmd += ["--fps", str(args.fps), "--resize-mode", str(args.resize_mode), "--codec", str(args.codec)]
    if args.pattern:
        cmd += ["--pattern", str(args.pattern)]
    if args.quality is not None:
        cmd += ["--quality", str(args.quality)]
    if args.quiet:
        cmd += ["--quiet"]
    return _subprocess_run(cmd)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="core.py: 统一运行数据、建图、社区、时间窗、检索、主题建模任务")
    sub = p.add_subparsers(dest="task", required=True)

    def add_common_runtime_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--exclude-selfcite", action="store_true")
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--force-reingest", action="store_true")

    # check-data
    sp = sub.add_parser("check-data", help="读取数据并做模型结构检查")
    add_common_runtime_flags(sp)
    sp.add_argument("--out-path", type=str, default=None)
    sp.set_defaults(func=task_check_data)

    # embed
    sp = sub.add_parser("embed", help="计算或加载论文 embedding")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--prefer-gpu", action="store_true")
    sp.set_defaults(func=task_embed)

    # check-embed
    sp = sub.add_parser("check-embed", help="embedding 质量检查")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--out-path", type=str, default=None)
    sp.set_defaults(func=task_check_embed)

    # build-2d
    sp = sub.add_parser("build-2d", help="生成或加载全局 2D 嵌入")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--method", type=str, default="umap", choices=["pca", "umap", "pacmap"])
    sp.add_argument("--cache-name", type=str, default="umap2d.npy")
    sp.add_argument("--out-png", type=str, default="fig_umap2d.png")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--umap-neighbors", type=int, default=30)
    sp.add_argument("--umap-min-dist", type=float, default=0.1)
    sp.add_argument("--point-size", type=float, default=1.0)
    sp.add_argument("--alpha", type=float, default=0.6)
    sp.set_defaults(func=task_build_2d)

    # build-graph
    sp = sub.add_parser("build-graph", help="生成或加载 mutual-kNN 图")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--cache-name", type=str, default=None)
    sp.set_defaults(func=task_build_graph)

    # graph-layout
    sp = sub.add_parser("graph-layout", help="对图本身做 2D layout")
    add_common_runtime_flags(sp)
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--method", type=str, default="drl", choices=["drl", "fr"])
    sp.add_argument("--cache-name", type=str, default="graph_drl2d.npy")
    sp.add_argument("--out-png", type=str, default="fig_graph_layout.png")
    sp.add_argument("--init-from-umap", action="store_true")
    sp.add_argument("--point-size", type=float, default=1.0)
    sp.add_argument("--alpha", type=float, default=0.6)
    sp.set_defaults(func=task_graph_layout)

    # sweep
    sp = sub.add_parser("sweep", help="Leiden 分辨率扫描")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--out-dir", type=str, default=str(OUT_DIR / "leiden_sweep"))
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--r-min", type=float, default=0.2)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.05)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default="RBConfigurationVertexPartition", choices=["RBConfigurationVertexPartition", "CPMVertexPartition"])
    sp.add_argument("--plot-reference", type=float, default=None)
    sp.add_argument("--point-size", type=float, default=1.0)
    sp.add_argument("--alpha", type=float, default=0.7)
    sp.set_defaults(func=task_sweep)

    # hierarchy
    sp = sub.add_parser("hierarchy", help="Leiden sweep + 层级连接诊断")
    add_common_runtime_flags(sp)
    sp.add_argument("--emb-path", type=str, default=str(EMB_PATH))
    sp.add_argument("--out-dir", type=str, default=str(OUT_DIR / "leiden_hierarchy"))
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.add_argument("--r-min", type=float, default=0.2)
    sp.add_argument("--r-max", type=float, default=2.0)
    sp.add_argument("--step", type=float, default=0.05)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--resolution-mode", type=str, default="linear", choices=["linear", "log"])
    sp.add_argument("--partition-type", type=str, default="RBConfigurationVertexPartition", choices=["RBConfigurationVertexPartition", "CPMVertexPartition"])
    sp.add_argument("--min-child-share", type=float, default=0.25)
    sp.set_defaults(func=task_hierarchy)

    # time-window
    sp = sub.add_parser("time-window", help="单个时间窗分析")
    add_common_runtime_flags(sp)
    sp.add_argument("--start-year", type=int, required=True)
    sp.add_argument("--end-year", type=int, required=True)
    sp.add_argument("--resolution", type=float, required=True)
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.set_defaults(func=task_time_window)

    # time-video
    sp = sub.add_parser("time-video", help="滑动时间窗动画")
    add_common_runtime_flags(sp)
    sp.add_argument("--resolution", type=float, required=True)
    sp.add_argument("--window-size", type=int, default=5)
    sp.add_argument("--step", type=int, default=1)
    sp.add_argument("--fps", type=int, default=2)
    sp.add_argument("--k", type=int, default=50)
    sp.add_argument("--knn-backend", type=str, default="hnswlib", choices=["faiss", "sklearn", "hnswlib"])
    sp.add_argument("--knn-batch-size", type=int, default=4096)
    sp.set_defaults(func=task_time_video)

    # keyword-index
    sp = sub.add_parser("keyword-index", help="构建关键词检索索引")
    add_common_runtime_flags(sp)
    sp.add_argument("--min-df", type=int, default=2)
    sp.add_argument("--max-df", type=float, default=0.2)
    sp.add_argument("--max-features", type=int, default=250000)
    sp.add_argument("--use-bigrams", action="store_true")
    sp.add_argument("--no-title", action="store_true")
    sp.add_argument("--no-abstract", action="store_true")
    sp.add_argument("--title-boost", type=int, default=3)
    sp.set_defaults(func=task_keyword_index)

    # keyword-search
    sp = sub.add_parser("keyword-search", help="最简单的关键词检索")
    add_common_runtime_flags(sp)
    sp.add_argument("--query", type=str, required=True)
    sp.add_argument("--top-k", type=int, default=20)
    sp.add_argument("--resolution", type=float, default=None)
    sp.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep"))
    sp.add_argument("--save-json", type=str, default=None)
    sp.add_argument("--min-df", type=int, default=2)
    sp.add_argument("--max-df", type=float, default=0.2)
    sp.add_argument("--max-features", type=int, default=250000)
    sp.add_argument("--use-bigrams", action="store_true")
    sp.add_argument("--no-title", action="store_true")
    sp.add_argument("--no-abstract", action="store_true")
    sp.add_argument("--title-boost", type=int, default=3)
    sp.set_defaults(func=task_keyword_search)

    # topic-model
    sp = sub.add_parser("topic-model", help="单个分辨率的 Topic-SCORE 主题建模")
    add_common_runtime_flags(sp)
    sp.add_argument("--k-topics", type=int, required=True)
    sp.add_argument("--resolution", type=float, default=1.0)
    sp.add_argument("--membership", type=str, default=None)
    sp.add_argument("--data-store", type=str, default=str(DATA_DIR / "data_store.pkl"))
    sp.add_argument("--stopwords", type=str, default=str(DATA_DIR / "stopwords.txt"))
    sp.add_argument("--out-dir", type=str, default=None)
    sp.add_argument("--no-title", action="store_true")
    sp.add_argument("--no-abstract", action="store_true")
    sp.add_argument("--no-authors", action="store_true")
    sp.add_argument("--use-clean-abstract", action="store_true")
    sp.add_argument("--title-weight", type=int, default=3)
    sp.add_argument("--abstract-weight", type=int, default=1)
    sp.add_argument("--author-weight", type=int, default=1)
    sp.add_argument("--min-token-len", type=int, default=2)
    sp.add_argument("--no-sklearn-stopwords", action="store_true")
    sp.add_argument("--words-percent", type=float, default=0.2)
    sp.add_argument("--docs-percent", type=float, default=1.0)
    sp.add_argument("--min-community-size", type=int, default=1)
    sp.add_argument("--vh-method", type=str, default="svs-sp", choices=["svs", "sp", "svs-sp"])
    sp.add_argument("--m", type=int, default=None)
    sp.add_argument("--k0", type=int, default=None)
    sp.add_argument("--mquantile", type=float, default=0.0)
    sp.add_argument("--m-trunc-mode", type=str, default="floor", choices=["floor", "cap"])
    sp.add_argument("--max-svs-combinations", type=int, default=20000)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-weighted-nnls", action="store_true")
    sp.add_argument("--max-papers", type=int, default=None)
    sp.add_argument("--rep-papers-mode", type=str, default="approx", choices=["exact", "approx", "off"])
    sp.set_defaults(func=task_topic_model)

    # topic-model-multi
    sp = sub.add_parser("topic-model-multi", help="多分辨率 Topic-SCORE 批处理")
    add_common_runtime_flags(sp)
    sp.add_argument("--k-topics", type=int, required=True)
    sp.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep"))
    sp.add_argument("--out-root", type=str, default=None)
    sp.add_argument("--r-min", type=float, default=0.0001)
    sp.add_argument("--r-max", type=float, default=5.0)
    sp.add_argument("--step", type=float, default=None)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--resolutions", type=float, nargs="*", default=None)
    sp.add_argument("--skip-existing", action="store_true")
    sp.add_argument("--continue-on-error", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=task_topic_model_multi)

    # align-topics
    sp = sub.add_parser("align-topics", help="多分辨率 topic 对齐（固定参考分辨率）")
    sp.add_argument("--k-topics", type=int, required=True)
    sp.add_argument("--topic-root", type=str, default=None)
    sp.add_argument("--out-root", type=str, default=None)
    sp.add_argument("--ref-resolution", type=float, default=1.0)
    sp.add_argument("--metric", type=str, default="cosine", choices=["cosine", "js"])
    sp.add_argument("--topn-common-vocab", type=int, default=None)
    sp.add_argument("--min-common-vocab", type=int, default=50)
    sp.add_argument("--r-min", type=float, default=0.0001)
    sp.add_argument("--r-max", type=float, default=5.0)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--resolutions", type=float, nargs="*", default=None)
    sp.add_argument("--copy-meta-json", action="store_true")
    sp.add_argument("--save-sim-matrix", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=task_align_topics)

    # align-topics-segmented
    sp = sub.add_parser("align-topics-segmented", help="多分辨率 topic 分段对齐")
    sp.add_argument("--k-topics", type=int, required=True)
    sp.add_argument("--topic-root", type=str, default=None)
    sp.add_argument("--out-root", type=str, default=None)
    sp.add_argument("--metric", type=str, default="cosine", choices=["cosine", "js"])
    sp.add_argument("--topn-common-vocab", type=int, default=None)
    sp.add_argument("--min-common-vocab", type=int, default=50)
    sp.add_argument("--r-min", type=float, default=0.0001)
    sp.add_argument("--r-max", type=float, default=5.0)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--resolutions", type=float, nargs="*", default=None)
    sp.add_argument("--break-avg-thresh", type=float, default=0.85)
    sp.add_argument("--break-min-thresh", type=float, default=0.10)
    sp.add_argument("--min-segment-size", type=int, default=2)
    sp.add_argument("--anchor-strategy", type=str, default="best-adjacent", choices=["best-adjacent", "middle"])
    sp.add_argument("--save-sim-matrix", action="store_true")
    sp.add_argument("--save-adjacent-topic-rows", action="store_true")
    sp.add_argument("--clean-segment-dirs", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=task_align_topics_segmented)

    # topic-viz
    sp = sub.add_parser("topic-viz", help="多分辨率 topic 可视化")
    sp.add_argument("--k-topics", type=int, required=True)
    sp.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep"))
    sp.add_argument("--topic-root", type=str, default=None)
    sp.add_argument("--out-dir", type=str, default=None)
    sp.add_argument("--umap", type=str, default=str(OUT_DIR / "umap2d.npy"))
    sp.add_argument("--graph-layout", type=str, default=str(OUT_DIR / "graph_drl2d.npy"))
    sp.add_argument("--resolutions", type=float, nargs="*", default=None)
    sp.add_argument("--r-min", type=float, default=0.0001)
    sp.add_argument("--r-max", type=float, default=5.0)
    sp.add_argument("--include", type=float, nargs="*", default=None)
    sp.add_argument("--step", type=float, default=None)
    sp.add_argument("--max-points", type=int, default=None)
    sp.add_argument("--point-size", type=float, default=1.0)
    sp.add_argument("--alpha", type=float, default=0.8)
    sp.add_argument("--skip-graph", action="store_true")
    sp.add_argument("--community-centroid", action="store_true")
    sp.add_argument("--annotate-top-n-communities", type=int, default=0)
    sp.add_argument("--batch-segments", action="store_true")
    sp.add_argument("--segments-csv", type=str, default=None)
    sp.add_argument("--segmented-out-root", type=str, default=None)
    sp.add_argument("--clean-segment-out", action="store_true")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=task_topic_viz)

    # diagnose-topic-collapse
    sp = sub.add_parser("diagnose-topic-collapse", help="诊断 topic collapse / 有效主题数")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, default=None)
    g.add_argument("--root", type=str, default=None)
    sp.add_argument("--out-dir", type=str, default=None)
    sp.add_argument("--save-per-topic", action="store_true")
    sp.add_argument("--save-plots", action="store_true")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=task_diagnose_topic_collapse)

    # frames-to-mp4
    sp = sub.add_parser("frames-to-mp4", help="把帧图合成为 MP4")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-dir", type=str, default=None)
    g.add_argument("--batch-subdirs", action="store_true")
    sp.add_argument("--root", type=str, default=None)
    sp.add_argument("--out", type=str, default=None)
    sp.add_argument("--out-root", type=str, default=None)
    sp.add_argument("--subdir-glob", type=str, default="frames*")
    sp.add_argument("--fps", type=int, default=8)
    sp.add_argument("--pattern", type=str, default=None)
    sp.add_argument("--resize-mode", type=str, choices=["first", "even"], default="first")
    sp.add_argument("--codec", type=str, default="libx264")
    sp.add_argument("--quality", type=int, default=None)
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=task_frames_to_mp4)

    return p


def main() -> None:
    args = build_parser().parse_args()
    res = args.func(args)
    if isinstance(res, dict):
        print("\n[summary]")
        _print_json(res)


if __name__ == "__main__":
    main()
