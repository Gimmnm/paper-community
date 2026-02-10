from __future__ import annotations  # 允许更灵活的类型注解（forward reference 等）

# pathlib.Path：跨平台路径/文件操作（拼接路径、创建目录、读写文件）
from pathlib import Path

# typing：类型注解（提高可读性，IDE 可提示）
from typing import Any, Dict, List, Union

# pickle：把 Python dict/list 等对象序列化为缓存文件，加速二次运行
import pickle

# numpy：把各种输入（如 R 向量/矩阵）统一为 ndarray，便于做形状处理
import numpy as np

# pandas：pyreadr 常把 R 对象读成 DataFrame/Series，这里用来统一处理与遍历
import pandas as pd


# 你已经确认的全量规模
N_AUTHORS = 47311
N_PAPERS = 83331


def read_rdata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    作用：
      - 读取 .RData 文件，返回 dict：{对象名 -> 对象内容}
      - 对象内容可能是 DataFrame / ndarray / list / vector 等

    说明：
      - 使用 pyreadr（底层 librdata），可直接读取 R 保存的数据对象
    """
    import pyreadr
    res = pyreadr.read_r(str(path))
    return dict(res)


def read_author_names_txt(path: Union[str, Path], encoding: str = "utf-8") -> List[str]:
    """
    作用：
      - 读取 author_name.txt（UTF-8），返回作者名列表
      - 数据集约定：第 i 行对应 Author ID = i（1-based）
    """
    p = Path(path)
    lines = p.read_text(encoding=encoding, errors="strict").splitlines()
    return [x.strip() for x in lines]


def _as_df(x: Any, cols: List[str]) -> pd.DataFrame:
    """
    作用：
      - 把 pyreadr 读出来的“表/矩阵”统一转成 pandas.DataFrame
      - 若本来就是 DataFrame，则复制并强制列名
      - 若是 ndarray，则按 cols 赋列名
    """
    if isinstance(x, pd.DataFrame):
        df = x.copy()
        if len(df.columns) == len(cols):
            df.columns = cols
        return df

    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[1] != len(cols):
        raise ValueError(f"Unexpected shape {arr.shape}, expect (*,{len(cols)})")
    return pd.DataFrame(arr, columns=cols)


def _to_int(s: pd.Series) -> pd.Series:
    """
    作用：
      - 把 ID/year 等列转为 int
      - 防止从 R 读出来变成 float（比如 2014.0）
    """
    return pd.to_numeric(s, errors="raise").astype(int)


def _dedupe_keep_order(lst: List[int]) -> List[int]:
    """
    作用：
      - 去重但保留顺序（第一次出现为准）
      - 数据一般干净，但做一层保险
    """
    return list(dict.fromkeys(lst))


def _get_obj(d: Dict[Any, Any], name: str) -> Any:
    """
    作用：
      - 稳健地从 pyreadr 返回的 dict 中取对象
      - 兼容：
        1) key 是 bytes（b"xxx"）
        2) 大小写不一致
        3) 文件里只有一个对象（直接取唯一对象）
    """
    if name in d:
        return d[name]

    for k in d.keys():
        if isinstance(k, (bytes, bytearray)):
            ks = k.decode("utf-8", errors="ignore")
            if ks == name:
                return d[k]

    name_low = name.lower()
    for k in d.keys():
        ks = k.decode("utf-8", errors="ignore") if isinstance(k, (bytes, bytearray)) else str(k)
        if ks.lower() == name_low:
            return d[k]

    if len(d) == 1:
        return next(iter(d.values()))

    raise KeyError(f"Key '{name}' not found. Available keys={list(d.keys())}")


def _as_1d_str_list(x: Any, name: str, expected_len: int) -> List[str]:
    """
    作用：
      - 把“R 的字符向量/一列df/series/(N,1)数组”等各种形式统一成 List[str]

    为什么需要：
      - pyreadr 读取 R 的 vector 时，有时会变成：
          * pandas.DataFrame (N行1列)
          * pandas.Series
          * numpy.ndarray (N,) 或 (N,1)
      - 直接 list(df) 会得到列名，长度=1（你之前遇到的坑）
    """
    if isinstance(x, pd.Series):
        out = x.astype(str).tolist()

    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            out = x.iloc[:, 0].astype(str).tolist()
        elif x.shape[0] == 1:
            out = x.iloc[0, :].astype(str).tolist()
        else:
            raise ValueError(f"{name} DataFrame shape={x.shape} not 1D-like")

    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            out = [("" if v is None else str(v)) for v in arr.tolist()]
        elif arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            out = [("" if v is None else str(v)) for v in arr.reshape(-1).tolist()]
        elif isinstance(x, list):
            out = [("" if v is None else str(v)) for v in x]
        else:
            raise ValueError(f"{name} cannot be converted to 1D list, type={type(x)}, ndim={arr.ndim}")

    if len(out) != expected_len:
        raise ValueError(f"{name} size={len(out)} != {expected_len}")

    return out


def ingest(
    authorpaper_rdata: Union[str, Path],
    author_name_txt: Union[str, Path],
    textcorpus_rdata: Union[str, Path],
    topicresults_rdata: Union[str, Path],
    rawpaper_rdata: Union[str, Path],   # ✅ 新增：RawPaper_py.RData（RawAbstract/RawTitle/RawYear）
    out_path: Union[str, Path],
    exclude_selfcite: bool = False,
) -> Path:
    """
    作用（数据搬运 + 缓存）：
      - 从数据源构建最简单可用的 Python 结构（1-based）并 pickle 保存
      - 关键点：Paper.abstract 使用“原始摘要 RawAbstract”（适合 specter2 embedding）

    输入数据源：
      1) AuthorPaperInfo(.RData)：AuPapMat / PapPapMat -> 构建作者-论文关系 + 引用关系 + 年份
      2) author_name.txt：作者显示名
      3) TextCorpusFinal(.RData)：CleanAbstracts（可选保留，用于传统 topic/对比）
      4) TopicResults(.RData)：paperTitle -> Paper.name
      5) RawPaper_py.RData：RawAbstract（原始摘要） -> Paper.abstract
    """
    authorpaper = read_rdata(authorpaper_rdata)
    textcorpus = read_rdata(textcorpus_rdata)
    topicres = read_rdata(topicresults_rdata)
    rawpaper = read_rdata(rawpaper_rdata)

    # ---- 1) full data：AuPapMat / PapPapMat ----
    AuPapMat = _as_df(_get_obj(authorpaper, "AuPapMat"), ["idxAu", "idxPap", "year", "journal"])
    PapPapMat = _as_df(_get_obj(authorpaper, "PapPapMat"), ["FromPap", "ToPap", "FromYear", "ToYear", "SelfCite"])

    AuPapMat["idxAu"] = _to_int(AuPapMat["idxAu"])
    AuPapMat["idxPap"] = _to_int(AuPapMat["idxPap"])
    AuPapMat["year"] = _to_int(AuPapMat["year"])

    PapPapMat["FromPap"] = _to_int(PapPapMat["FromPap"])
    PapPapMat["ToPap"] = _to_int(PapPapMat["ToPap"])
    PapPapMat["SelfCite"] = _to_int(PapPapMat["SelfCite"])

    if exclude_selfcite:
        PapPapMat = PapPapMat[PapPapMat["SelfCite"] == 0]

    # ---- 2) author names（47311行）----
    names_raw = read_author_names_txt(author_name_txt, encoding="utf-8")
    if len(names_raw) != N_AUTHORS:
        raise ValueError(f"author_name lines={len(names_raw)} != {N_AUTHORS}")
    author_names = [""] + names_raw  # 1-based

    # ---- 3) Paper.name（标题）来自 TopicResults 的 paperTitle ----
    paper_title_raw = _as_1d_str_list(_get_obj(topicres, "paperTitle"), "paperTitle", N_PAPERS)
    paper_title = [""] + paper_title_raw  # 1-based

    # ---- 4) Paper.abstract：优先用 RawPaper_py 的 RawAbstract（原始英文摘要）----
    raw_abs = _as_1d_str_list(_get_obj(rawpaper, "RawAbstract"), "RawAbstract", N_PAPERS)
    paper_abstract = [None] + raw_abs  # 1-based

    # （可选保留）CleanAbstracts：将来做传统 topic/对比可能用得到
    clean_abs = _as_1d_str_list(_get_obj(textcorpus, "CleanAbstracts"), "CleanAbstracts", N_PAPERS)
    clean_abs = [("" if (x is None or str(x).strip()=="" or str(x).strip().lower() in {"nan","na"}) else str(x)) for x in clean_abs]
    paper_abstract_clean = [None] + clean_abs

    # ---- 5) 预分配关系结构（0号位空）----
    author_papers: List[List[int]] = [[] for _ in range(N_AUTHORS + 1)]
    paper_authors: List[List[int]] = [[] for _ in range(N_PAPERS + 1)]
    paper_refs: List[List[int]] = [[] for _ in range(N_PAPERS + 1)]
    paper_year: List[int] = [0] * (N_PAPERS + 1)

    # ---- 6) 从 AuPapMat 填 A.paper / P.author / P.year ----
    for row in AuPapMat.itertuples(index=False):
        aid = int(row.idxAu)
        pid = int(row.idxPap)
        y = int(row.year)

        author_papers[aid].append(pid)
        paper_authors[pid].append(aid)
        if paper_year[pid] == 0:
            paper_year[pid] = y

    for aid in range(1, N_AUTHORS + 1):
        author_papers[aid] = _dedupe_keep_order(author_papers[aid])
    for pid in range(1, N_PAPERS + 1):
        paper_authors[pid] = _dedupe_keep_order(paper_authors[pid])

    # ---- 7) 从 PapPapMat 填 P.ref（引用边）----
    for row in PapPapMat.itertuples(index=False):
        frm = int(row.FromPap)
        to = int(row.ToPap)
        paper_refs[frm].append(to)

    for pid in range(1, N_PAPERS + 1):
        refs = _dedupe_keep_order(paper_refs[pid])
        paper_refs[pid] = [to for to in refs if to != pid]

    # ---- 8) 落盘缓存（pickle）----
    payload = {
        "n_authors": N_AUTHORS,
        "n_papers": N_PAPERS,
        "author_names": author_names,
        "author_papers": author_papers,
        "paper_authors": paper_authors,
        "paper_refs": paper_refs,
        "paper_year": paper_year,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,                # ✅ 原始摘要（给 specter2）
        "paper_abstract_clean": paper_abstract_clean,    # 可选：clean 摘要（给传统 topic）
        "meta": {
            "exclude_selfcite": exclude_selfcite,
            "abstract_source": "RawPaper_py.RData::RawAbstract",
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


def load_data(path: Union[str, Path]) -> Dict[str, Any]:
    """
    作用：
      - 读取 ingest() 生成的 pickle 缓存
      - core.py 通过它快速恢复关系结构
    """
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)
