"""
项目内「原始数据 → 缓存 → 产物」路径的单一事实来源（Single Source of Truth）。

- 所有指向 `data/*.RData`、`data/data_store.pkl` 的 ingest 入口应通过
  `data_source_paths(repo_root)` 取路径，避免在多处硬编码文件名导致漂移。
- `BASE_DIR` / `OUT_DIR` 等与仓库根目录相关的常量也可从这里导入。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# 仓库根目录（本文件位于 `src/project_paths.py`）
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataSourcePaths:
    """与 `getdata.ingest` / `getdata.load_data` 配套的一组路径。"""

    root: Path
    data_dir: Path
    cache_path: Path
    author_name_txt: Path
    authorpaper_rdata: Path
    textcorpus_rdata: Path
    topicresults_rdata: Path
    rawpaper_rdata: Path


def data_source_paths(repo_root: Optional[Path] = None) -> DataSourcePaths:
    """
    默认 `repo_root` 为当前仓库根目录；也可传入其它根目录（例如测试夹具目录），
    只要其下保持 `data/AuthorPaperInfo_py.RData` 等相对结构即可。
    """
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    dd = root / "data"
    return DataSourcePaths(
        root=root,
        data_dir=dd,
        cache_path=dd / "data_store.pkl",
        author_name_txt=dd / "author_name.txt",
        authorpaper_rdata=dd / "AuthorPaperInfo_py.RData",
        textcorpus_rdata=dd / "TextCorpusFinal_py.RData",
        topicresults_rdata=dd / "TopicResults_py.RData",
        rawpaper_rdata=dd / "RawPaper_py.RData",
    )


# 常用产物目录（与 `core.py` / 各脚本默认约定一致）
def out_dir(repo_root: Optional[Path] = None) -> Path:
    r = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    return r / "out"


def embedding_path_specter2(repo_root: Optional[Path] = None) -> Path:
    return data_source_paths(repo_root).data_dir / "paper_embeddings_specter2.npy"
