from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import igraph as ig
import numpy as np

from checklist import run_embedding_checks, run_model_checks
from community import leiden_sweep, pick_nearest_resolution
from diagram2d import embed_2d, graph_layout_2d, plot_scatter
from embedding import embed_all_papers
from getdata import ingest, load_data
from model import build_models
from network import build_or_load_mutual_knn_graph
from time_window import analyze_time_window, collect_time_info, make_sliding_window_video


# 你给定的目录（绝对路径）
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

# 四个数据源路径
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

def build_or_load(exclude_selfcite: bool = False):
    """
    读取原始数据并构建 Author / Paper 对象。
    """
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
) -> np.ndarray:
    """
    加载或计算 paper embedding。
    约定：返回矩阵 embs，且 embs[pid] 对应 papers[pid]（1-based）。
    """
    emb_path = Path(emb_path)
    if emb_path.exists():
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
) -> np.ndarray:
    X = np.asarray(embs[1:], dtype=np.float32)
    Y_cache = Path(out_dir) / cache_name
    Y = embed_2d(
        X,
        method=method,
        normalize=True,
        pca_dim=50,
        umap_neighbors=30,
        umap_min_dist=0.1,
        umap_metric="cosine",
        random_state=random_state,
        cache_npy=Y_cache,
        verbose=True,
    )
    return Y



def build_or_load_global_graph(
    embs: np.ndarray,
    *,
    out_dir: Path = OUT_DIR,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
    cache_name: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    X = np.asarray(embs[1:], dtype=np.float32)
    if cache_name is None:
        cache_name = f"mutual_knn_k{k}.npz"
    edges_cache = Path(out_dir) / cache_name
    return build_or_load_mutual_knn_graph(
        X,
        k=k,
        cache_npz=edges_cache,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
        normalize=True,
        verbose=True,
    )



def build_igraph_from_edge_triplets(
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> ig.Graph:
    G = ig.Graph(n=int(n_nodes), edges=list(zip(u.tolist(), v.tolist())), directed=False)
    G.es["weight"] = np.asarray(w, dtype=np.float32).astype(float)
    return G



def prepare_global_pipeline(
    *,
    exclude_selfcite: bool = False,
    k: int = 50,
    knn_backend: str = "hnswlib",
    knn_batch_size: int = 4096,
) -> Dict[str, Any]:
    """
    一次性准备全局分析所需的全部对象：
      - authors / papers / data
      - embs / X
      - 全局 2D 坐标 Y
      - 全图 mutual-kNN 图与 igraph
      - 时间信息 time_info

    这是后续时间窗口模块最直接的入口。
    """
    authors, papers, data = build_or_load(exclude_selfcite=exclude_selfcite)
    embs = build_or_load_embeddings(papers)

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
# 直接给时间窗口模块调用的包装入口
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
    """
    运行单个时间窗：
      1) 继承全图社区
      2) 仅在窗口内部重建图并重跑社区
      3) 输出对比图与对比报告
    """
    ctx = prepare_global_pipeline(
        exclude_selfcite=exclude_selfcite,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
    )
    result = analyze_time_window(
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
    return result



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
    """
    运行滑动窗口动态可视化：
      - 默认 5 年窗口
      - 默认 1 年一滑
      - 每帧都是 inherited / refit 并排对比图
    """
    ctx = prepare_global_pipeline(
        exclude_selfcite=exclude_selfcite,
        k=k,
        knn_backend=knn_backend,
        knn_batch_size=knn_batch_size,
    )
    result = make_sliding_window_video(
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
    return result


if __name__ == "__main__":
    """
    保留你原来的主流程，并额外把时间窗口入口函数挂在 core.py 里，方便直接 import 调用。

    用法示例（建议在交互环境里调用，而不是默认一次性全跑）：

        from core import run_single_time_window, run_time_window_animation

        # 单个窗口
        run_single_time_window(start_year=1995, end_year=1999, resolution=1.0)

        # 动画：5 年窗口，1 年步长
        run_time_window_animation(resolution=1.0, window_size=5, step=1, fps=2)
    """
    authors, papers, data = build_or_load(exclude_selfcite=False)

    run_model_checks(
        authors,
        papers,
        data,
        seed=42,
        sample_authors=80,
        sample_papers=120,
        max_show_examples=5,
        write_report_path=str(DATA_DIR / "data_check.txt"),
    )

    embs = build_or_load_embeddings(papers)

    run_embedding_checks(
        papers=papers,
        embs=embs,
        expected_dim=768,
        seed=42,
        sample_papers=8,
        sample_pairs=12,
        write_report_path=str(DATA_DIR / "embedding_check.txt"),
    )

    X = np.asarray(embs[1:], dtype=np.float32)
    N = X.shape[0]
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
        k=50,
        knn_backend="hnswlib",
        knn_batch_size=4096,
    )

    G = build_igraph_from_edge_triplets(N, u, v, w)

    

    # 下面这段 sweep 保留为注释，方便你继续做全图分辨率扫描
    # leiden_out_dir = OUT_DIR / "leiden_sweep"
    # results = leiden_sweep(
    #     G,
    #     out_dir=leiden_out_dir,
    #     r_min=0.2,
    #     r_max=2.0,
    #     step=0.05,
    #     include=[1.0],
    #     seed=42,
    #     save_each_membership=True,
    #     verbose=True,
    # )
    # r0 = pick_nearest_resolution(results, 1.0)
    # labels = results[r0]["membership"]
    # plot_scatter(
    #     Y,
    #     labels=labels,
    #     title=f"UMAP(2D) colored by Leiden (r={r0})",
    #     out_png=OUT_DIR / "fig_umap_leiden.png",
    #     point_size=1.0,
    #     alpha=0.7,
    # )

    # 可选：网络布局视图
    # Yg = graph_layout_2d(
    #     n_nodes=N,
    #     u=u,
    #     v=v,
    #     w=w,
    #     method="drl",
    #     init_xy=Y,
    #     cache_npy=OUT_DIR / "graph_drl2d.npy",
    # )
    
        # 单个时间窗
    run_single_time_window(
        start_year=1995,
        end_year=1999,
        resolution=1.0,
    )

    # 动画：5 年窗口，一年一年滑
    run_time_window_animation(
        resolution=1.0,
        window_size=5,
        step=1,
        fps=2,
    )
