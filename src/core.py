from __future__ import annotations  # 允许更灵活的类型注解（未来你加模块时更方便）

# pathlib.Path：处理你提供的绝对路径，拼接/判断文件存在更稳
from pathlib import Path

# getdata：负责“读取 RData/txt 并落盘成 pickle”
from getdata import ingest, load_data

# model：负责“创建 Author/Paper 对象并填充字段”
from model import build_models

from checklist import run_model_checks

from embedding import embed_all_papers

import numpy as np  # 预定义的 embedding 字段类型

from checklist import run_embedding_checks

from network import build_or_load_mutual_knn_graph
import igraph as ig
from community import pick_nearest_resolution, leiden_sweep
from diagram2d import embed_2d, plot_scatter, graph_layout_2d


# 你给定的目录（绝对路径）
DATA_DIR = Path("/Users/gzh/Documents/paper-community/data")
OUT_DIR = Path("/Users/gzh/Documents/paper-community/out")

# 四个数据源路径
AUTHOR_NAME_TXT = DATA_DIR / "author_name.txt"
AUTHORPAPER_RDATA = DATA_DIR / "AuthorPaperInfo_py.RData"
TEXTCORPUS_RDATA = DATA_DIR / "TextCorpusFinal_py.RData"
TOPICRESULTS_RDATA = DATA_DIR / "TopicResults_py.RData"
RAWPAPER_RDATA = DATA_DIR / "RawPaper_py.RData"

EMB_PATH = DATA_DIR / "paper_embeddings_specter2.npy"

# 缓存文件：第一次 ingest 后生成；后续直接 load
CACHE_PATH = DATA_DIR / "data_store.pkl"


def build_or_load(exclude_selfcite: bool = False):
    """
    作用：
      - 这是 core 层的入口函数
      - 如果缓存不存在：
          调用 getdata.ingest() 从 RData/txt 读原始数据并生成 data_store.pkl
      - 然后：
          调用 getdata.load_data() 读回 payload
          调用 model.build_models() 创建并填充 Author/Paper 对象数组

    参数：
      - exclude_selfcite：
          是否过滤自引引用边（PapPapMat 的 SelfCite 字段）

    返回：
      - authors: authors[aid] 是 Author（1..2831），authors[0]=None
      - papers: papers[pid] 是 Paper（1..83331），papers[0]=None
      - data: 原始 payload dict（你若需要直接访问 list-of-lists 也可以用它）
    """
    # 1) 缓存不存在就 ingest 一次
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

    # 2) 读取缓存并构建对象
    data = load_data(CACHE_PATH)
    authors, papers = build_models(data)
    return authors, papers, data




if __name__ == "__main__":
    """
    作用：
      - 让 core.py 可以直接运行做一次全量构建
      - 并输出一些基本 sanity check，确认数据和对象结构一致
    """
    authors, papers, data = build_or_load(exclude_selfcite=False)

    # 生成一份更详细的建模检查报告
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
    
    if EMB_PATH.exists():
        # 已经算过：直接用 memmap 方式加载（不占用大量内存）
        embs = np.load(EMB_PATH, mmap_mode="r")
        print("[emb] loaded from disk:", embs.shape, embs.dtype, "example:", embs[1, :5])
    else:
        # 没算过：先计算，再保存
        embs = embed_all_papers(
            papers=papers,
            out_npy_path=EMB_PATH,
            batch_size=16,
            prefer_gpu=False,
            attach_to_papers=False,
        )
        print("[emb] computed and saved:", embs.shape, embs.dtype)
    
    # （可选）如果你后续模块想统一从 data 里取
    # data["paper_embeddings"] = embs
    
    
    run_embedding_checks(
        papers=papers,
        embs=embs,
        expected_dim=768,
        seed=42,
        sample_papers=8,
        sample_pairs=12,
        write_report_path=str(DATA_DIR / "embedding_check.txt"),
    )

    
    # ------------------------------------------------------------
    # 0) 准备 0-based 视角的数据：X shape=(N,768), idx=0..N-1 对应 pid=idx+1
    # ------------------------------------------------------------
    X = np.asarray(embs[1:], dtype=np.float32)  # (83331, 768)
    N = X.shape[0]
    print("[core] X:", X.shape, X.dtype)

    # ------------------------------------------------------------
    # 1) 2D 嵌入（embedding->2D）
    # ------------------------------------------------------------
    Y_cache = OUT_DIR / "umap2d.npy"
    Y = embed_2d(
        X,
        method="umap",
        normalize=True,
        pca_dim=50,
        umap_neighbors=30,
        umap_min_dist=0.1,
        umap_metric="cosine",
        random_state=42,
        cache_npy=Y_cache,
        verbose=True,
    )

    # 保存一个“纯 2D 不上色”的图
    plot_scatter(
        Y,
        title="UMAP(2D) of paper embeddings",
        out_png=OUT_DIR / "fig_umap2d.png",
        point_size=1.0,
        alpha=0.6,
        max_points=None,
    )

    # ------------------------------------------------------------
    # 2) mutual-kNN 建图（无向权重图）
    # ------------------------------------------------------------
    edges_cache = OUT_DIR / "mutual_knn_k50.npz"
    A_sym, (u, v, w) = build_or_load_mutual_knn_graph(
        X,
        k=50,
        cache_npz=edges_cache,
        knn_backend="hnswlib",     # hnswlib
        knn_batch_size=4096,
        normalize=True,          # cosine 几何必须 normalize
        verbose=True,
    )

    # ------------------------------------------------------------
    # 3) Leiden 多分辨率扫描（分层结果）
    # ------------------------------------------------------------
    
    # u, v, w 是 mutual-kNN 得到的边（0-based）
    # N = 83331
    G = ig.Graph(n=N, edges=list(zip(u, v)), directed=False)
    G.es["weight"] = w.astype(float)

    leiden_out_dir = OUT_DIR / "leiden_sweep"
    results = leiden_sweep(
        G,
        out_dir=leiden_out_dir,
        r_min=0.2,
        r_max=2.0,
        step=0.05,           # <<<< 扫细；想更细就 0.02
        include=[1.0],       # 强制包含 1.0（双保险）
        seed=42,
        save_each_membership=True,
        verbose=True,
    )

    resolutions = sorted(results.keys())
    # A) UMAP 2D 上染色（输出成帧）
    frames_umap = OUT_DIR / "frames_umap"
    frames_umap.mkdir(parents=True, exist_ok=True)

    for i, r0 in enumerate(resolutions):
        labels = results[r0]["membership"]
        print(f"[core] using resolution r={r0:.4f}, n_comm={results[r0]['n_comm']}")

        plot_scatter(
            Y,
            labels=labels,
            title=f"UMAP(2D) colored by Leiden (r={r0})",
            out_png=frames_umap / f"frame_{i:04d}.png",   # 用序号保证 ffmpeg 顺序正确
            point_size=1.0,
            alpha=0.7,
            max_points=None,
        )



    # ------------------------------------------------------------
    # 5) 可视化 B：先对网络做 layout->2D，再按社区染色
    # ------------------------------------------------------------
    Yg = graph_layout_2d(
        n_nodes=N,
        u=u,
        v=v,
        w=w,
        method="drl",
        init_xy=Y,
        cache_npy=OUT_DIR / "graph_drl2d.npy",
    )

    frames_graph = OUT_DIR / "frames_graph"
    frames_graph.mkdir(parents=True, exist_ok=True)

    for i, r0 in enumerate(resolutions):
        labels = results[r0]["membership"]   # ✅ 这里必须重新取
        plot_scatter(
            Yg,
            labels=labels,
            title=f"Graph layout (DRL) colored by Leiden (r={r0})",
            out_png=frames_graph / f"frame_{i:04d}.png",
            point_size=1.0,
            alpha=0.7,
            max_points=None,
        )


