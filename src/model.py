from __future__ import annotations  # 允许更灵活的类型注解（尤其是 forward reference）

# typing：用于给 build_models 的入参/返回值注解，便于阅读与IDE提示
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # 预定义的 embedding 字段类型


class Author:
    """
    Author 类：对应一个作者节点（Author）。

    字段定义（与你的设计一致）：
      - id: 作者 ID（1-based）
      - name: 作者姓名（来自 author_name.txt）
      - paper: 该作者参与的论文 ID 列表（来自 AuPapMat）
    """
    __slots__ = ("id", "name", "paper")  # 限制属性集合，减少内存占用（适合大量对象）

    def __init__(self, id: int):
        self.id: int = id
        self.name: str = ""
        self.paper: List[int] = []  # paper ids


class Paper:
    """
    Paper 类：对应一篇论文节点（Paper）。

    字段定义（与你的设计一致）：
      - id: 论文 ID（1-based）
      - name: 论文标题（来自 TopicResults.RData 的 paperTitle）
      - ref: 该论文引用的论文 ID 列表（来自 PapPapMat：FromPap -> ToPap）
      - author: 该论文作者 ID 列表（来自 AuPapMat：idxPap -> idxAu）
      - year: 论文年份（来自 AuPapMat 的 year 聚合；0 表示未知）
      - abstract: 摘要文本（来自 TextCorpusFinal.RData 的 CleanAbstracts，已转成字符串）
    """
    __slots__ = ("id", "name", "ref", "author", "year", "abstract", "embedding")  # 限制属性集合，减少内存占用（适合大量对象）

    def __init__(self, id: int):
        self.id: int = id
        self.name: str = ""
        self.ref: List[int] = []
        self.author: List[int] = []
        self.year: int = 0
        self.abstract: Optional[str] = None
        # 约定：np.ndarray shape=(dim,), dtype=float32；未向量化时为 None
        self.embedding: Optional[np.ndarray] = None


def build_models(data: Dict[str, Any]) -> Tuple[List[Optional[Author]], List[Optional[Paper]]]:
    """
    作用：
      - 根据 getdata.ingest() 输出的数据结构，构建程序中可直接使用的 Author/Paper 对象数组
      - 你希望 “对象数组下标就是 id”，这里严格按 1-based 构建：
          authors[aid].id == aid
          papers[pid].id == pid
        且 authors[0] / papers[0] 置为 None 作为占位

    输入 data 的关键字段：
      - n_authors, n_papers
      - author_names, author_papers
      - paper_authors, paper_refs, paper_year
      - paper_title, paper_abstract

    返回：
      - authors: List[Optional[Author]]，长度 n_authors+1
      - papers: List[Optional[Paper]]，长度 n_papers+1
    """
    n_authors = int(data["n_authors"])
    n_papers = int(data["n_papers"])

    # 1) 一次性创建所有对象（固定数量；满足你“直接创建对应个数对象”的要求）
    authors: List[Optional[Author]] = [None] + [Author(i) for i in range(1, n_authors + 1)]
    papers: List[Optional[Paper]] = [None] + [Paper(i) for i in range(1, n_papers + 1)]

    # 2) 取出关系数据
    author_names = data["author_names"]
    author_papers = data["author_papers"]

    paper_authors = data["paper_authors"]
    paper_refs = data["paper_refs"]
    paper_year = data["paper_year"]
    paper_title = data["paper_title"]
    paper_abstract = data["paper_abstract"]

    # 3) 填 Author：name + paper list
    for aid in range(1, n_authors + 1):
        a = authors[aid]
        a.name = author_names[aid]
        a.paper = author_papers[aid]

    # 4) 填 Paper：author list + ref list + year + title + abstract
    for pid in range(1, n_papers + 1):
        p = papers[pid]
        p.author = paper_authors[pid]
        p.ref = paper_refs[pid]
        p.year = paper_year[pid]
        p.name = paper_title[pid]
        p.abstract = paper_abstract[pid]

    return authors, papers
