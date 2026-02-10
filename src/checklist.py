from __future__ import annotations

# 标准库：路径处理（写报告文件可用）
from pathlib import Path

# 标准库：随机抽样做“深度一致性检查”
import random

# 标准库：统计信息（均值、分位数等）
import statistics

# 标准库：更方便的计数/TopK
from collections import Counter

# typing：类型注解，便于阅读和IDE提示
from typing import Any, Dict, List, Optional, Tuple

from typing import Sequence
# numpy：检查 embedding 的 shape/dtype、统计 norm、cosine 等
import numpy as np



def _is_empty_text(x: Any) -> bool:
    """
    判定文本是否“等价为空”：
      - None / "" / 全空白
      - "nan"/"na"/"none"/"null" 等（有时从 dataframe/astype(str) 产生）
    """
    if x is None:
        return True
    t = str(x).strip()
    if t == "":
        return True
    if t.lower() in {"nan", "na", "none", "null"}:
        return True
    return False


def _describe_int_list(values: List[int], name: str, topk: int = 5) -> str:
    """
    对整数列表做一个简洁统计描述：count/min/max/mean/median + TopK
    """
    if not values:
        return f"{name}: (empty)\n"
    values_sorted = sorted(values)
    n = len(values_sorted)
    mn = values_sorted[0]
    mx = values_sorted[-1]
    mean = statistics.mean(values_sorted)
    med = statistics.median(values_sorted)

    # 简单分位数（不依赖 numpy）
    def q(p: float) -> int:
        idx = int(round((n - 1) * p))
        return values_sorted[idx]

    ctr = Counter(values_sorted)
    top = ctr.most_common(topk)
    top_str = ", ".join([f"{k}x{v}" for k, v in top])

    return (
        f"{name}: n={n}, min={mn}, max={mx}, mean={mean:.3f}, median={med}\n"
        f"  p10={q(0.10)}, p25={q(0.25)}, p75={q(0.75)}, p90={q(0.90)}\n"
        f"  top{topk} (value x freq): {top_str}\n"
    )


def run_model_checks(
    authors: List[Any],
    papers: List[Any],
    data: Dict[str, Any],
    *,
    seed: int = 42,
    sample_authors: int = 80,
    sample_papers: int = 120,
    max_show_examples: int = 5,
    write_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    建模正确性检查主函数（建议在 core.py 中调用）。

    输入：
      - authors: authors[aid] 是 Author，authors[0] 为 None（1-based）
      - papers: papers[pid] 是 Paper，papers[0] 为 None（1-based）
      - data: getdata.ingest() 输出的 payload（pickle load 后）

    检查内容（全量 + 抽样）：
      A) 基本结构一致性
         - n_authors / n_papers 是否匹配
         - 1-based 索引规则是否成立（id == index）
      B) 文本字段质量
         - title/abstract 空值比例
         - 示例输出
      C) 年份字段质量
         - year 为 0 的数量（未知）
         - min/max 范围
      D) 关系一致性与合法性（全量）
         - author->paper 边数量 与 paper->author 边数量一致
         - id 越界统计（作者ID、论文ID）
         - 引用边 ref 的越界统计
         - 重复 id（同一个列表里重复）
      E) 抽样深度一致性
         - 随机抽 author/paper 做双向包含关系验证
         - 随机抽 ref 验证范围/自环等

    输出：
      - 返回一个 summary dict（你也可以后续写成 JSON）
      - 同时会 print 一份较长的报告（方便你肉眼确认）
      - 可选写入文件 write_report_path
    """
    random.seed(seed)

    # -------------------------
    # 0) 基本规模与结构检查
    # -------------------------
    n_authors_list = len(authors) - 1
    n_papers_list = len(papers) - 1

    n_authors_data = int(data.get("n_authors", -1))
    n_papers_data = int(data.get("n_papers", -1))

    issues: List[str] = []
    notes: List[str] = []

    if authors[0] is not None:
        issues.append("authors[0] should be None (1-based convention broken).")
    if papers[0] is not None:
        issues.append("papers[0] should be None (1-based convention broken).")

    if n_authors_data != -1 and n_authors_data != n_authors_list:
        issues.append(f"n_authors mismatch: data={n_authors_data}, list={n_authors_list}")
    if n_papers_data != -1 and n_papers_data != n_papers_list:
        issues.append(f"n_papers mismatch: data={n_papers_data}, list={n_papers_list}")

    # id == index 检查（抽样 + 关键点）
    # 全量检查也不重（47k+83k），但这里做抽样+头尾检查，避免打印太多
    def check_id_index(samples: List[int], obj_list: List[Any], label: str):
        for idx in samples:
            obj = obj_list[idx]
            if obj is None:
                issues.append(f"{label}[{idx}] is None (unexpected).")
                continue
            if getattr(obj, "id", None) != idx:
                issues.append(f"{label}[{idx}].id != index: {getattr(obj, 'id', None)} vs {idx}")

    author_samples = list({1, n_authors_list, random.randint(1, n_authors_list)})
    paper_samples = list({1, n_papers_list, random.randint(1, n_papers_list)})
    check_id_index(author_samples, authors, "authors")
    check_id_index(paper_samples, papers, "papers")

    # -------------------------
    # 1) 文本字段与空值统计（全量）
    # -------------------------
    empty_title = 0
    empty_abs = 0

    # 额外：统计一些“可能编码/格式问题”的作者名（例如 `t）
    backtick_t_names = 0

    for aid in range(1, n_authors_list + 1):
        name = getattr(authors[aid], "name", "")
        if "`t" in str(name):
            backtick_t_names += 1

    for pid in range(1, n_papers_list + 1):
        p = papers[pid]
        if _is_empty_text(getattr(p, "name", "")):
            empty_title += 1
        if _is_empty_text(getattr(p, "abstract", None)):
            empty_abs += 1

    # -------------------------
    # 2) 年份统计（全量）
    # -------------------------
    years = []
    year_zero = 0
    for pid in range(1, n_papers_list + 1):
        y = int(getattr(papers[pid], "year", 0) or 0)
        if y <= 0:
            year_zero += 1
        else:
            years.append(y)

    # -------------------------
    # 3) 边与一致性（全量）
    # -------------------------
    # A.paper 总边数 & P.author 总边数
    total_ap_edges = sum(len(getattr(authors[aid], "paper", [])) for aid in range(1, n_authors_list + 1))
    total_pa_edges = sum(len(getattr(papers[pid], "author", [])) for pid in range(1, n_papers_list + 1))

    if total_ap_edges != total_pa_edges:
        issues.append(f"edge count mismatch: sum(A.paper)={total_ap_edges} vs sum(P.author)={total_pa_edges}")

    # 越界检查 + 重复检查
    invalid_author_ids_in_papers = 0
    invalid_paper_ids_in_authors = 0
    dup_in_paper_authors = 0
    dup_in_author_papers = 0

    # 引用边检查
    total_refs = 0
    invalid_ref_ids = 0
    self_ref_edges = 0
    dup_in_refs = 0

    # papers -> authors：检查 author id 范围 + 去重情况
    for pid in range(1, n_papers_list + 1):
        a_list = getattr(papers[pid], "author", [])
        if len(a_list) != len(set(a_list)):
            dup_in_paper_authors += 1
        for aid in a_list:
            if not (1 <= int(aid) <= n_authors_list):
                invalid_author_ids_in_papers += 1

        r_list = getattr(papers[pid], "ref", [])
        total_refs += len(r_list)
        if len(r_list) != len(set(r_list)):
            dup_in_refs += 1
        for to in r_list:
            to = int(to)
            if to == pid:
                self_ref_edges += 1
            if not (1 <= to <= n_papers_list):
                invalid_ref_ids += 1

    # authors -> papers：检查 paper id 范围 + 去重情况
    for aid in range(1, n_authors_list + 1):
        p_list = getattr(authors[aid], "paper", [])
        if len(p_list) != len(set(p_list)):
            dup_in_author_papers += 1
        for pid in p_list:
            if not (1 <= int(pid) <= n_papers_list):
                invalid_paper_ids_in_authors += 1

    # -------------------------
    # 4) 抽样深度一致性检查（双向包含）
    # -------------------------
    # 这里做“随机抽样”，避免对每条边都做 membership（虽然也能做，但你说想输出多，我们就重点抽样+打印）
    sampled_inconsistencies = 0
    sampled_checks = 0

    sample_authors = min(sample_authors, n_authors_list)
    sample_papers = min(sample_papers, n_papers_list)

    sampled_aids = random.sample(range(1, n_authors_list + 1), sample_authors)
    sampled_pids = random.sample(range(1, n_papers_list + 1), sample_papers)

    # author -> paper：检查 paper.author 是否包含该 author
    for aid in sampled_aids:
        a = authors[aid]
        pids = getattr(a, "paper", [])
        # 每个作者只抽最多 30 个 paper 来验证（避免极大作者导致输出/耗时过大）
        if len(pids) > 30:
            pids = random.sample(pids, 30)
        for pid in pids:
            sampled_checks += 1
            if aid not in getattr(papers[pid], "author", []):
                sampled_inconsistencies += 1

    # paper -> author：检查 author.paper 是否包含该 paper
    for pid in sampled_pids:
        p = papers[pid]
        aids = getattr(p, "author", [])
        for aid in aids:
            sampled_checks += 1
            if pid not in getattr(authors[aid], "paper", []):
                sampled_inconsistencies += 1

    if sampled_inconsistencies > 0:
        issues.append(f"sampled bidirectional inconsistencies: {sampled_inconsistencies}/{sampled_checks}")

    # -------------------------
    # 5) 分布统计（输出更丰富）
    # -------------------------
    papers_per_author = [len(getattr(authors[aid], "paper", [])) for aid in range(1, n_authors_list + 1)]
    authors_per_paper = [len(getattr(papers[pid], "author", [])) for pid in range(1, n_papers_list + 1)]
    refs_per_paper = [len(getattr(papers[pid], "ref", [])) for pid in range(1, n_papers_list + 1)]

    # Top 作者（按论文数）
    top_authors = sorted(range(1, n_authors_list + 1), key=lambda aid: len(authors[aid].paper), reverse=True)[:10]
    top_authors_info = [(aid, authors[aid].name, len(authors[aid].paper)) for aid in top_authors]

    # Top 论文（按引用数 out-degree）
    top_ref_out = sorted(range(1, n_papers_list + 1), key=lambda pid: len(papers[pid].ref), reverse=True)[:10]
    top_ref_out_info = [(pid, papers[pid].name[:80], len(papers[pid].ref)) for pid in top_ref_out]

    # -------------------------
    # 6) 组织输出报告
    # -------------------------
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("MODEL CHECK REPORT")
    lines.append("=" * 80)
    lines.append(f"n_authors(list) = {n_authors_list}")
    lines.append(f"n_papers(list)  = {n_papers_list}")
    lines.append(f"n_authors(data) = {n_authors_data}")
    lines.append(f"n_papers(data)  = {n_papers_data}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("A) Basic text fields quality")
    lines.append("-" * 80)
    lines.append(f"empty titles    : {empty_title} / {n_papers_list} ({empty_title / n_papers_list:.4%})")
    lines.append(f"empty abstracts : {empty_abs} / {n_papers_list} ({empty_abs / n_papers_list:.4%})")
    lines.append(f"author names containing '`t' : {backtick_t_names} / {n_authors_list} ({backtick_t_names / n_authors_list:.4%})")
    lines.append("")

    lines.append("-" * 80)
    lines.append("B) Year field quality")
    lines.append("-" * 80)
    if years:
        lines.append(f"year=0 (unknown): {year_zero} / {n_papers_list} ({year_zero / n_papers_list:.4%})")
        lines.append(f"year min/max    : {min(years)} / {max(years)}")
    else:
        lines.append("No positive years found (unexpected).")
    lines.append("")

    lines.append("-" * 80)
    lines.append("C) Edge counts & validity")
    lines.append("-" * 80)
    lines.append(f"sum(A.paper) edges  = {total_ap_edges}")
    lines.append(f"sum(P.author) edges = {total_pa_edges}")
    lines.append(f"invalid paper ids in authors[*].paper : {invalid_paper_ids_in_authors}")
    lines.append(f"invalid author ids in papers[*].author: {invalid_author_ids_in_papers}")
    lines.append(f"duplicate entries in authors[*].paper lists: {dup_in_author_papers} authors")
    lines.append(f"duplicate entries in papers[*].author lists : {dup_in_paper_authors} papers")
    lines.append("")
    lines.append(f"total ref edges      : {total_refs}")
    lines.append(f"invalid ref ids       : {invalid_ref_ids}")
    lines.append(f"self-ref edges (pid->pid): {self_ref_edges}")
    lines.append(f"duplicate entries in ref lists: {dup_in_refs} papers")
    lines.append("")

    lines.append("-" * 80)
    lines.append("D) Degree distributions")
    lines.append("-" * 80)
    lines.append(_describe_int_list(papers_per_author, "papers_per_author"))
    lines.append(_describe_int_list(authors_per_paper, "authors_per_paper"))
    lines.append(_describe_int_list(refs_per_paper, "refs_per_paper"))
    lines.append("")

    lines.append("-" * 80)
    lines.append("E) Sampled deep consistency checks")
    lines.append("-" * 80)
    lines.append(f"seed={seed}, sampled_authors={sample_authors}, sampled_papers={sample_papers}")
    lines.append(f"sampled membership checks: {sampled_checks}")
    lines.append(f"sampled inconsistencies  : {sampled_inconsistencies}")
    lines.append("")

    lines.append("-" * 80)
    lines.append("F) Top examples")
    lines.append("-" * 80)
    lines.append("Top authors by paper_count:")
    for aid, name, cnt in top_authors_info:
        lines.append(f"  aid={aid:5d}  papers={cnt:5d}  name={name}")

    lines.append("")
    lines.append("Top papers by out-ref count:")
    for pid, title80, cnt in top_ref_out_info:
        lines.append(f"  pid={pid:5d}  out_refs={cnt:4d}  title={title80}")

    # 随机展示几篇论文的 title/abstract 开头
    lines.append("")
    lines.append("Random paper samples (title + first 200 chars of abstract):")
    show_pids = random.sample(range(1, n_papers_list + 1), min(max_show_examples, n_papers_list))
    for pid in show_pids:
        p = papers[pid]
        title = str(getattr(p, "name", ""))[:120]
        abs_ = str(getattr(p, "abstract", "") or "")
        abs_ = abs_.replace("\n", " ").strip()
        lines.append(f"  pid={pid} | year={getattr(p, 'year', 0)} | title={title}")
        lines.append(f"       abstract[:200]={abs_[:200]}")
    lines.append("")

    # 抽取一些 “空摘要但有标题” 的例子（给你后续 embedding 兜底用）
    if empty_abs > 0:
        lines.append("Examples: empty abstract but non-empty title (for title-fallback embedding):")
        found = 0
        for pid in range(1, n_papers_list + 1):
            p = papers[pid]
            if _is_empty_text(p.abstract) and (not _is_empty_text(p.name)):
                lines.append(f"  pid={pid} year={p.year} title={str(p.name)[:140]}")
                found += 1
                if found >= max_show_examples:
                    break
        lines.append("")

    # issues
    lines.append("-" * 80)
    lines.append("G) Issues / Warnings")
    lines.append("-" * 80)
    if issues:
        for it in issues:
            lines.append(f"[ISSUE] {it}")
    else:
        lines.append("No blocking issues found.")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if write_report_path:
        rp = Path(write_report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding="utf-8")

    summary = {
        "n_authors": n_authors_list,
        "n_papers": n_papers_list,
        "empty_title": empty_title,
        "empty_abstract": empty_abs,
        "year_zero": year_zero,
        "year_min": (min(years) if years else None),
        "year_max": (max(years) if years else None),
        "edge_ap": total_ap_edges,
        "edge_pa": total_pa_edges,
        "invalid_paper_ids_in_authors": invalid_paper_ids_in_authors,
        "invalid_author_ids_in_papers": invalid_author_ids_in_papers,
        "invalid_ref_ids": invalid_ref_ids,
        "self_ref_edges": self_ref_edges,
        "dup_in_author_papers_lists": dup_in_author_papers,
        "dup_in_paper_authors_lists": dup_in_paper_authors,
        "dup_in_ref_lists": dup_in_refs,
        "sampled_checks": sampled_checks,
        "sampled_inconsistencies": sampled_inconsistencies,
        "issues": issues,
        "notes": notes,
    }
    return summary



## embedding check

def run_embedding_checks(
    papers: Sequence[object],
    embs: np.ndarray,
    *,
    expected_dim: int = 768,
    seed: int = 42,
    sample_papers: int = 8,
    sample_pairs: int = 12,
    chunk_rows: int = 4096,
    write_report_path: Optional[str] = None,
) -> None:
    """
    检查 paper embedding 是否“基本合理”。

    你当前的工程是 1-based：
      - papers[0] = None
      - embs.shape[0] = len(papers) = N_PAPERS + 1
      - embs[0] 一般是 0 向量占位

    本函数做的检查：
      A) 基本信息：shape / dtype / 是否符合预期 dim
      B) 数值健康：NaN / Inf 数量
      C) 零向量行：一般应只有 embs[0]，以及少量“空文本”行（你这里基本不会）
      D) norm 分布：min / max / mean / 分位数
      E) 随机样本：打印 pid、norm、title、abstract 是否为空、embedding 前 5 维
      F) 随机 pair cosine：检查范围、以及是否出现“异常接近 1”的重复向量

    参数：
      - papers：Paper 列表（1-based），用于抽样打印 title/abstract
      - embs：np.load(..., mmap_mode="r") 的结果，或 embed_all_papers 返回的矩阵
      - expected_dim：默认 768（SPECTER2 base hidden size）
      - chunk_rows：分块扫描，避免一次性把整个矩阵复制到内存
      - write_report_path：写到文件（例如 data/embedding_check.txt），不写就只打印到控制台
    """

    # -----------------------------
    # 1) 输出函数（同时支持写文件）
    # -----------------------------
    out_lines = []

    def _w(line: str = "") -> None:
        print(line)
        out_lines.append(line)

    # -----------------------------
    # 2) 基本信息检查
    # -----------------------------
    n_rows, dim = embs.shape
    n_papers_list = len(papers)

    _w("=" * 80)
    _w("EMBEDDING CHECK REPORT")
    _w("=" * 80)
    _w(f"embs.shape      = {embs.shape}")
    _w(f"embs.dtype      = {embs.dtype}")
    _w(f"len(papers)     = {n_papers_list}")
    _w(f"expected_dim    = {expected_dim}")
    _w("")

    if n_rows != n_papers_list:
        _w("[WARN] embs.shape[0] != len(papers)  (1-based 项目一般应该相等)")
    if dim != expected_dim:
        _w("[WARN] embs.shape[1] != expected_dim  (维度不一致，后续算法会出问题)")
    if embs.dtype != np.float32:
        _w("[WARN] embs.dtype is not float32 (建议 float32：省空间，且足够用)")

    # 1-based：papers[0] 通常是 None
    start_idx = 1 if (n_papers_list > 0 and papers[0] is None) else 0
    real_n = n_rows - start_idx
    _w(f"indexing mode   = {'1-based' if start_idx == 1 else '0-based'}")
    _w(f"real paper rows = {real_n}")
    _w("")

    # embs[0] 是否为 0 向量（占位）
    if start_idx == 1:
        z0 = float(np.linalg.norm(embs[0]))
        _w(f"norm(embs[0])   = {z0:.6f}  (1-based 占位行通常应为 0)")
        _w("")

    # -----------------------------
    # 3) 分块扫描：NaN/Inf、零向量、norm 分布
    # -----------------------------
    nan_cnt = 0
    inf_cnt = 0
    zero_cnt = 0

    # norm 统计：我们不一次性存所有 norm（也可以存；83331 其实不大）
    # 这里存下来，方便分位数统计（83331 个 float，不到 1MB）
    norms = np.empty((real_n,), dtype=np.float32)

    pos = 0
    for s in range(start_idx, n_rows, chunk_rows):
        e = min(s + chunk_rows, n_rows)

        block = np.asarray(embs[s:e])  # 确保是 ndarray（memmap 切片也可以）
        nan_cnt += int(np.isnan(block).sum())
        inf_cnt += int(np.isinf(block).sum())

        block_norm = np.linalg.norm(block, axis=1)
        norms[pos : pos + (e - s)] = block_norm.astype(np.float32)

        zero_cnt += int((block_norm == 0.0).sum())
        pos += (e - s)

    _w("-" * 80)
    _w("A) Numeric health")
    _w("-" * 80)
    _w(f"NaN count       : {nan_cnt}")
    _w(f"Inf count       : {inf_cnt}")
    _w(f"zero-vector rows: {zero_cnt} / {real_n}")
    if nan_cnt > 0 or inf_cnt > 0:
        _w("[WARN] Found NaN/Inf in embeddings (这会导致相似度/聚类出错)")
    _w("")

    _w("-" * 80)
    _w("B) Norm distribution (L2)")
    _w("-" * 80)
    _w(f"norm min/max    : {float(norms.min()):.4f} / {float(norms.max()):.4f}")
    _w(f"norm mean/std   : {float(norms.mean()):.4f} / {float(norms.std()):.4f}")
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        _w(f"p{q:02d}           : {float(np.percentile(norms, q)):.4f}")
    _w("")

    # -----------------------------
    # 4) 随机样本检查：对应 paper 的 title/abstract + embedding 前几维
    # -----------------------------
    _w("-" * 80)
    _w("C) Random samples (paper fields + embedding preview)")
    _w("-" * 80)

    rnd = random.Random(seed)
    candidates = list(range(start_idx, n_rows))
    sample_pids = rnd.sample(candidates, k=min(sample_papers, len(candidates)))

    for pid in sample_pids:
        p = papers[pid]
        title = getattr(p, "name", "")
        abstract = getattr(p, "abstract", "")

        v = np.asarray(embs[pid], dtype=np.float32)
        nrm = float(np.linalg.norm(v))
        has_abs = bool(str(abstract).strip())
        _w(f"pid={pid} | norm={nrm:.4f} | title_len={len(str(title))} | has_abs={has_abs}")
        _w(f"  title   : {str(title)[:120]}")
        _w(f"  abs[:120]: {str(abstract)[:120]}")
        _w(f"  vec[:5]  : {v[:5].tolist()}")
        _w("")

    # -----------------------------
    # 5) Cosine sanity：随机 pairs，检查是否异常重复
    # -----------------------------
    _w("-" * 80)
    _w("D) Random pair cosine similarity sanity check")
    _w("-" * 80)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    pair_pids = rnd.sample(candidates, k=min(max(sample_pairs, 2), len(candidates)))
    # 两两配对： (0,1), (2,3), ...
    suspicious = 0
    for i in range(0, len(pair_pids) - 1, 2):
        a_id = pair_pids[i]
        b_id = pair_pids[i + 1]
        a = np.asarray(embs[a_id], dtype=np.float32)
        b = np.asarray(embs[b_id], dtype=np.float32)
        c = _cos(a, b)
        _w(f"cos(pid {a_id}, {b_id}) = {c:.6f}")
        if c > 0.999:
            suspicious += 1

    if suspicious > 0:
        _w(f"[WARN] Found {suspicious} pairs with cosine > 0.999 (可能存在重复向量或输入文本重复)")
    _w("")

    _w("E) Summary")
    _w("-" * 80)
    _w("No blocking issues if: NaN/Inf=0, dim correct, zero rows minimal, norms in a reasonable range.")
    _w("")

    # -----------------------------
    # 6) 写报告文件（可选）
    # -----------------------------
    if write_report_path:
        Path(write_report_path).write_text("\n".join(out_lines), encoding="utf-8")
        _w(f"[saved] report -> {write_report_path}")
