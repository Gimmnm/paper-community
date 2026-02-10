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


# 你给定的目录（绝对路径）
DATA_DIR = Path("/Users/gzh/Documents/paper-community/data")

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
        write_report_path=str(DATA_DIR / "check_report.txt"),
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
