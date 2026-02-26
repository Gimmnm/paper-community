from __future__ import annotations

"""
主题自动命名脚本（中文候选）
========================

用途：
- 针对某个参考分辨率（建议 r=1.0）自动生成每个 topic 的中文主题名候选
- 支持读取对齐后的多分辨率结果目录（aligned_to_rXXXX），把参考主题名传播到各分辨率
- 输出网页友好的标签文件（CSV/JSON）与带标签的社区表

输入（至少满足其一）：
1) 未对齐：out/topic_modeling_multi/K{K}/r{ref}/...（只生成参考分辨率主题名）
2) 已对齐：out/topic_modeling_multi/K{K}/aligned_to_r{ref}/r*/...（推荐；可生成各分辨率继承标签）

说明：
- 本脚本采用“规则 + 短语候选”的半自动方案，生成可读的中文候选名；
- 最终建议人工确认一次（只需看 K 个主题，不必逐社区人工拼词）。
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
RDIR_RE = re.compile(r"^r([0-9]+(?:\.[0-9]+)?)$")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{1,}")

# -----------------------------
# 基础词典（可按你们项目继续扩充）
# -----------------------------

EN_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "under", "using", "use",
    "based", "via", "new", "study", "studies", "analysis", "method", "methods", "model",
    "models", "approach", "approaches", "paper", "data", "problem", "results", "theory",
    "estimation", "estimators", "estimator", "distribution", "distributions", "test", "tests",
    "sample", "samples", "statistical", "statistics", "linear", "nonlinear", "bayesian",
    "maximum", "likelihood", "asymptotic", "asymptotics", "application", "applications",
    "random", "effects", "effect", "design", "designs", "regression", "time", "process", "processes",
}

# 主题关键词到中文词汇（回退拼接用）
WORD_ZH = {
    "gene": "基因", "genetic": "遗传", "genes": "基因", "expression": "表达", "genome": "基因组",
    "clinical": "临床", "trial": "试验", "trials": "试验", "dose": "剂量", "treatment": "治疗", "sequential": "序贯",
    "survival": "生存", "censored": "删失", "censoring": "删失", "hazard": "风险", "failure": "失效",
    "spatial": "空间", "disease": "疾病", "hiv": "HIV", "point": "点", "process": "过程", "processes": "过程",
    "time": "时间", "series": "序列", "autoregressive": "自回归",
    "stochastic": "随机", "brownian": "布朗", "variables": "变量",
    "longitudinal": "纵向", "missing": "缺失", "mixed": "混合", "effects": "效应",
    "multivariate": "多元", "normal": "正态", "testing": "检验",
    "optimal": "最优", "factorial": "析因", "block": "区组", "orthogonal": "正交", "experiment": "实验", "experiments": "实验",
    "nonparametric": "非参数", "density": "密度", "selection": "选择", "classification": "分类",
}

# 常见短语词典（优先）
PHRASE_ZH = {
    "time series": "时间序列",
    "survival analysis": "生存分析",
    "clinical trial": "临床试验",
    "clinical trials": "临床试验",
    "point process": "点过程",
    "point processes": "点过程",
    "random effects": "随机效应",
    "mixed effects": "混合效应",
    "factorial design": "析因设计",
    "factorial designs": "析因设计",
    "optimal design": "最优设计",
    "optimal designs": "最优设计",
    "experimental design": "实验设计",
    "autoregressive process": "自回归过程",
    "autoregressive processes": "自回归过程",
    "brownian motion": "布朗运动",
    "gene expression": "基因表达",
    "spatial statistics": "空间统计",
    "spatial process": "空间过程",
    "stochastic process": "随机过程",
    "stochastic processes": "随机过程",
    "longitudinal data": "纵向数据",
    "hazard function": "风险函数",
}

# 规则模板（按你目前数据分布偏统计学优化）
RULES = [
    # name, any/all keyword conditions, zh, en
    {
        "name": "genetics",
        "all": [{"gene", "genetic"}],
        "zh": "统计遗传与基因表达分析",
        "en": "Statistical Genetics & Gene Expression",
    },
    {
        "name": "clinical_trials",
        "all": [{"clinical"}, {"trial", "trials"}],
        "zh": "临床试验设计与剂量反应",
        "en": "Clinical Trials & Dose-Response",
    },
    {
        "name": "survival",
        "all": [{"survival"}, {"censored", "censoring", "hazard", "failure"}],
        "zh": "生存分析与删失数据",
        "en": "Survival Analysis & Censoring",
    },
    {
        "name": "time_series",
        "all": [{"time"}, {"series", "autoregressive"}],
        "zh": "时间序列与自回归建模",
        "en": "Time Series & Autoregressive Modeling",
    },
    {
        "name": "spatial",
        "all": [{"spatial"}, {"point", "disease", "process", "hiv"}],
        "zh": "空间统计与点过程建模",
        "en": "Spatial Statistics & Point Processes",
    },
    {
        "name": "stochastic",
        "all": [{"stochastic", "random"}, {"process", "processes", "brownian"}],
        "zh": "随机过程与布朗运动",
        "en": "Stochastic Processes & Brownian Motion",
    },
    {
        "name": "experimental_design",
        "all": [{"design", "designs"}, {"factorial", "block", "optimal", "orthogonal"}],
        "zh": "最优实验设计与析因设计",
        "en": "Optimal & Factorial Experimental Design",
    },
    {
        "name": "longitudinal",
        "all": [{"longitudinal", "missing", "random"}, {"regression", "treatment", "response"}],
        "zh": "纵向数据与混合效应回归",
        "en": "Longitudinal Data & Mixed-Effects Regression",
    },
    {
        "name": "hypothesis_testing",
        "all": [{"test", "tests", "testing"}, {"multivariate", "normal", "distribution", "sample"}],
        "zh": "假设检验与多元分布",
        "en": "Hypothesis Testing & Multivariate Distributions",
    },
    {
        "name": "nonparametric_regression",
        "all": [{"regression", "estimation"}, {"nonparametric", "density", "estimators", "selection"}],
        "zh": "非参数估计与回归方法",
        "en": "Nonparametric Estimation & Regression",
    },
]


# -----------------------------
# Helpers
# -----------------------------

def parse_rdir_name(name: str) -> float | None:
    m = RDIR_RE.match(name)
    return float(m.group(1)) if m else None


def discover_result_dirs(root: Path) -> list[tuple[float, Path]]:
    items = []
    if not root.exists():
        return items
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        r = parse_rdir_name(p.name)
        if r is None:
            continue
        if (p / "topics_top_words.csv").exists() and (p / "communities_topic_weights.csv").exists():
            items.append((r, p))
    items.sort(key=lambda x: x[0])
    return items


def select_dirs(items: list[tuple[float, Path]], resolutions: Iterable[float] | None, r_min: float, r_max: float, include: Iterable[float] | None = None) -> list[tuple[float, Path]]:
    include = [float(x) for x in (include or [])]
    if resolutions is not None and len(list(resolutions)) > 0:
        req = [float(x) for x in resolutions] + include
        out = []
        for rr in req:
            hit = None
            for r, p in items:
                if abs(r - rr) <= 1e-8:
                    hit = (r, p)
                    break
            if hit is not None:
                out.append(hit)
        uniq = {round(r, 10): (r, p) for r, p in out}
        return [uniq[k] for k in sorted(uniq.keys())]
    out = []
    for r, p in items:
        if (r_min - 1e-8) <= r <= (r_max + 1e-8):
            out.append((r, p))
    for x in include:
        for r, p in items:
            if abs(r - x) <= 1e-8:
                out.append((r, p))
                break
    uniq = {round(r, 10): (r, p) for r, p in out}
    return [uniq[k] for k in sorted(uniq.keys())]


def normalize_tokens(words: list[str]) -> list[str]:
    return [w.lower().strip() for w in words if str(w).strip()]


def tokenize_en(text: str) -> list[str]:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]
    return toks


def extract_title_ngrams(texts: list[str], top_words: set[str], ngram_min=2, ngram_max=3) -> Counter:
    c = Counter()
    for txt in texts:
        toks = [t for t in tokenize_en(txt) if t not in EN_STOPWORDS]
        if len(toks) < ngram_min:
            continue
        for n in range(ngram_min, ngram_max + 1):
            for i in range(len(toks) - n + 1):
                ng = toks[i:i+n]
                # 至少一个 token 出现在 top_words，减少噪声
                if top_words and not any(t in top_words for t in ng):
                    continue
                # 避免全是太泛的词
                if all(t in EN_STOPWORDS for t in ng):
                    continue
                phrase = " ".join(ng)
                c[phrase] += 1
    return c


def phrase_score(phrase: str, cnt: int, top_words: set[str]) -> float:
    toks = phrase.split()
    overlap = sum(1 for t in toks if t in top_words)
    bonus = 0.0
    if phrase in PHRASE_ZH:
        bonus += 3.0
    # 含领域词加分
    if any(t in {"survival", "censored", "hazard", "clinical", "trial", "trials", "time", "series", "spatial", "stochastic", "brownian", "factorial", "optimal", "longitudinal", "gene", "genetic"} for t in toks):
        bonus += 1.2
    # 稍微偏好 2-gram
    len_pen = 0.3 if len(toks) == 2 else 0.0
    return float(cnt * (1.0 + 0.35 * overlap) + bonus + len_pen)


def join_top_words_fallback_zh(top_words: list[str], max_terms: int = 3) -> str:
    chosen = []
    for w in top_words:
        zh = PHRASE_ZH.get(w) or WORD_ZH.get(w)
        if zh and zh not in chosen:
            chosen.append(zh)
        if len(chosen) >= max_terms:
            break
    if not chosen:
        chosen = top_words[:max_terms]
    if len(chosen) >= 2:
        return "与".join(chosen[:2]) + ("相关方法" if len(chosen) <= 2 else "建模")
    return (chosen[0] if chosen else "统计主题") + "相关方法"


def title_case_phrase(phrase: str) -> str:
    return " ".join([w.capitalize() if w.lower() not in {"and", "of", "for", "to", "in", "on", "with"} else w.lower() for w in phrase.split()])



def _rule_confidence_dynamic(
    *,
    rule: dict,
    top_words: list[str],
    phrase_candidates: list[str],
) -> tuple[float, dict]:
    """规则命中后的动态置信度（启发式分数，不是概率）。

    目标：
    - 避免所有 rule 命中都固定为 0.92
    - 融合 top_words 覆盖、短语证据、词位置等因素
    """
    tw = [w for w in normalize_tokens(top_words) if w]
    tw_set = set(tw)

    # phrase token bag
    phrase_tokens = []
    for ph in phrase_candidates[:8]:
        phrase_tokens.extend(normalize_tokens(ph.split()))
    pc_set = set(phrase_tokens)

    all_groups = [set(g) for g in rule.get("all", [])]
    any_groups = [set(g) for g in rule.get("any", [])]
    rule_vocab = set().union(*all_groups) if all_groups else set()
    for g in any_groups:
        rule_vocab |= set(g)

    # 1) 规则组覆盖强度（不仅看是否命中，还看每组命中多少）
    group_strengths = []
    for g in all_groups:
        hits = len(g & (tw_set | pc_set))
        denom = max(1, len(g))
        group_strengths.append(min(1.0, hits / denom))
    group_support = float(np.mean(group_strengths)) if group_strengths else 1.0

    # 2) top_words 对规则词汇覆盖
    rule_top_hits = len(rule_vocab & tw_set) if rule_vocab else 0
    top_coverage = min(1.0, rule_top_hits / max(2, min(6, len(rule_vocab) if rule_vocab else 2)))

    # 3) 短语证据覆盖（代表社区标题短语里是否支持）
    rule_phrase_hits = len(rule_vocab & pc_set) if rule_vocab else 0
    phrase_coverage = min(1.0, rule_phrase_hits / 3.0)

    # 4) 位置加权：规则词若出现在 top_words 更靠前位置，加分更高
    pos_scores = []
    for i, w in enumerate(tw[:10], start=1):
        if w in rule_vocab:
            pos_scores.append(1.0 / np.sqrt(i))
    pos_signal = float(np.mean(pos_scores)) if pos_scores else 0.0  # [0,1]

    # 5) 是否出现典型短语词典证据
    lex_phrase_hit = 0.0
    for ph in phrase_candidates[:5]:
        if ph in PHRASE_ZH:
            ph_tokens = set(ph.split())
            if not rule_vocab or (len(ph_tokens & rule_vocab) >= 1):
                lex_phrase_hit = 1.0
                break

    # 组合分数（经验启发式）
    conf = (
        0.58
        + 0.14 * group_support
        + 0.12 * top_coverage
        + 0.08 * phrase_coverage
        + 0.06 * pos_signal
        + 0.04 * lex_phrase_hit
    )
    conf = float(np.clip(conf, 0.50, 0.97))

    detail = {
        "group_support": round(group_support, 4),
        "top_coverage": round(top_coverage, 4),
        "phrase_coverage": round(phrase_coverage, 4),
        "pos_signal": round(pos_signal, 4),
        "lex_phrase_hit": round(lex_phrase_hit, 4),
        "rule_top_hits": int(rule_top_hits),
        "rule_phrase_hits": int(rule_phrase_hits),
        "rule_vocab_size": int(len(rule_vocab)),
    }
    return conf, detail


def match_rule(top_words: list[str], phrase_candidates: list[str]) -> dict | None:
    tw = set(normalize_tokens(top_words))
    pc = set(normalize_tokens(" ".join(phrase_candidates).split()))
    bag = tw | pc
    for rule in RULES:
        all_ok = True
        for group in rule.get("all", []):
            if not (set(group) & bag):
                all_ok = False
                break
        any_groups = rule.get("any", [])
        any_ok = True if not any_groups else any((set(g) & bag) for g in any_groups)
        if all_ok and any_ok:
            conf, detail = _rule_confidence_dynamic(
                rule=rule,
                top_words=top_words,
                phrase_candidates=phrase_candidates,
            )
            return {
                "label_zh": rule["zh"],
                "label_en": rule["en"],
                "method": f"rule:{rule['name']}",
                "confidence": conf,
                "confidence_detail": json.dumps(detail, ensure_ascii=False),
            }
    return None
def generate_topic_label_candidates(
    *,
    topic_id: int,
    top_words: list[str],
    rep_titles: list[str],
    n_candidates: int = 3,
) -> list[dict]:
    tw_norm = normalize_tokens(top_words)
    tw_set = set(tw_norm)
    ngrams = extract_title_ngrams(rep_titles, tw_set, ngram_min=2, ngram_max=3)
    ranked_phrases = sorted(ngrams.items(), key=lambda kv: phrase_score(kv[0], kv[1], tw_set), reverse=True)
    phrase_list = [p for p, _ in ranked_phrases[:10]]

    cands: list[dict] = []

    # 1) 规则命中优先
    hit = match_rule(tw_norm, phrase_list)
    if hit is not None:
        hit["topic_id"] = int(topic_id)
        hit["phrase_evidence"] = "; ".join(phrase_list[:5])
        hit["top_words"] = " ".join(top_words[:10])
        cands.append(hit)

    # 2) 基于短语词典/标题n-gram 的候选
    for rank, (phrase, cnt) in enumerate(ranked_phrases[:8], start=1):
        zh = PHRASE_ZH.get(phrase)
        if zh:
            cands.append({
                "topic_id": int(topic_id),
                "label_zh": zh if "与" in zh or "分析" in zh or "建模" in zh else zh + "相关方法",
                "label_en": title_case_phrase(phrase),
                "method": "phrase_dict",
                "confidence": min(0.88, 0.62 + 0.06 * cnt + (0.04 if rank == 1 else 0.0)),
                "phrase_evidence": phrase,
                "top_words": " ".join(top_words[:10]),
            })
        else:
            # 词级翻译组合（若覆盖较好）
            toks = phrase.split()
            zh_parts = [WORD_ZH.get(t) for t in toks if t in WORD_ZH]
            if len(zh_parts) >= max(1, len(toks) - 1):
                zh_label = "".join(zh_parts[:3])
                if zh_label and not zh_label.endswith(("分析", "建模", "设计", "过程")):
                    zh_label += "相关方法"
                cands.append({
                    "topic_id": int(topic_id),
                    "label_zh": zh_label or f"主题{topic_id}相关方法",
                    "label_en": title_case_phrase(phrase),
                    "method": "phrase_translate",
                    "confidence": min(0.75, 0.50 + 0.04 * cnt),
                    "phrase_evidence": phrase,
                    "top_words": " ".join(top_words[:10]),
                })

    # 3) top words 回退（保证一定有结果；同时在仅有单一规则命中时提供备选）
    need_fallback = (len(cands) == 0) or (len(cands) < max(2, n_candidates))
    if need_fallback:
        zh = join_top_words_fallback_zh(tw_norm, max_terms=3)
        en = title_case_phrase(" ".join(tw_norm[:3])) if tw_norm else f"Topic {topic_id}"
        cands.append({
            "topic_id": int(topic_id),
            "label_zh": zh,
            "label_en": en,
            "method": "top_words_fallback",
            "confidence": (0.45 if not cands else 0.38),
            "phrase_evidence": "; ".join(phrase_list[:5]),
            "top_words": " ".join(top_words[:10]),
        })

    # 去重（按中文名）并按置信度排序
    uniq = {}
    for c in cands:
        key = (str(c.get("label_zh", "")).strip(), str(c.get("label_en", "")).strip())
        if key not in uniq or float(c.get("confidence", 0)) > float(uniq[key].get("confidence", 0)):
            uniq[key] = c
    out = sorted(uniq.values(), key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return out[:max(1, n_candidates)]


def collect_rep_titles_for_topic(df_comm: pd.DataFrame, df_tcomm: pd.DataFrame, topic_id: int, top_n_communities: int = 8) -> list[str]:
    # 从 topic_representative_communities 取该 topic 的代表社区，再去社区表读取 rep_papers 标题串
    comm_ids = []
    if not df_tcomm.empty and {"topic_id", "community_id"}.issubset(df_tcomm.columns):
        d = df_tcomm.copy()
        d["topic_id"] = pd.to_numeric(d["topic_id"], errors="coerce").astype("Int64")
        d["community_id"] = pd.to_numeric(d["community_id"], errors="coerce").astype("Int64")
        if "rank_in_topic" in d.columns:
            d = d.sort_values(["topic_id", "rank_in_topic"]) 
        sub = d[d["topic_id"] == int(topic_id)].head(top_n_communities)
        comm_ids = [int(x) for x in sub["community_id"].dropna().tolist()]

    titles: list[str] = []
    if not df_comm.empty and {"community_id", "rep_papers"}.issubset(df_comm.columns):
        d2 = df_comm.copy()
        d2["community_id"] = pd.to_numeric(d2["community_id"], errors="coerce").astype("Int64")
        if comm_ids:
            d2 = d2[d2["community_id"].isin(comm_ids)]
        else:
            # 回退：用 top1_topic=topic_id 的社区前若干个
            if "top1_topic" in d2.columns:
                d2["top1_topic"] = pd.to_numeric(d2["top1_topic"], errors="coerce").astype("Int64")
                d2 = d2[d2["top1_topic"] == int(topic_id)].head(top_n_communities)
            else:
                d2 = d2.head(top_n_communities)
        for s in d2["rep_papers"].fillna("").astype(str).tolist():
            for t in s.split("||"):
                tt = t.strip()
                if tt:
                    titles.append(tt)
    return titles


def read_topics_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 统一 topic_id 类型
    if "topic_id" in df.columns:
        df["topic_id"] = pd.to_numeric(df["topic_id"], errors="coerce").astype(int)
    return df


def get_topic_top_words(df_topics: pd.DataFrame, topic_id: int, topn: int = 10) -> list[str]:
    if df_topics.empty:
        return []
    sub = df_topics[df_topics["topic_id"] == int(topic_id)]
    if sub.empty:
        return []
    row = sub.iloc[0]
    # 优先 word_1... 列
    words = []
    for i in range(1, topn + 1):
        col = f"word_{i}"
        if col in sub.columns:
            w = str(row.get(col, "")).strip()
            if w:
                words.append(w)
    if words:
        return words[:topn]
    # 回退 top_words 字符串
    tw = str(row.get("top_words", "")).strip()
    return tw.split()[:topn]


@dataclass
class LabelConfig:
    ref_resolution: float = 1.0
    r_min: float = 0.0001
    r_max: float = 5.0
    n_candidates: int = 3
    top_n_communities: int = 8
    overwrite_label_in_comm_table: bool = False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="主题自动命名（中文候选）")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--topic-root", type=str, default=None,
                   help="主题结果根目录。可指向 K{K}（未对齐）或 aligned_to_rXXXX（已对齐）")
    p.add_argument("--alignment-dir", type=str, default=None,
                   help="对齐目录（可选）。提供后将读取 topic_alignment.csv 并为各分辨率写标签传播表")
    p.add_argument("--out-root", type=str, default=None,
                   help="输出目录；默认 <topic-root>/labels_ref_r{ref}")
    p.add_argument("--ref-resolution", type=float, default=1.0)
    p.add_argument("--r-min", type=float, default=0.0001)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--include", type=float, nargs="*", default=None)
    p.add_argument("--resolutions", type=float, nargs="*", default=None)
    p.add_argument("--n-candidates", type=int, default=3)
    p.add_argument("--top-n-communities", type=int, default=8)
    p.add_argument("--write-labeled-community-tables", action="store_true",
                   help="在输出目录下为各分辨率写带主题名的社区表副本")
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    verbose = not args.quiet

    default_topic_root = OUT_DIR / "topic_modeling_multi" / f"K{args.k}"
    topic_root = Path(args.topic_root) if args.topic_root else default_topic_root
    if not topic_root.exists():
        raise FileNotFoundError(f"topic_root 不存在: {topic_root}")

    items = discover_result_dirs(topic_root)
    if not items:
        raise FileNotFoundError(f"在 {topic_root} 未发现 r*/topics_top_words.csv 与 communities_topic_weights.csv")
    selected = select_dirs(items, args.resolutions, args.r_min, args.r_max, args.include)
    if not selected:
        raise RuntimeError("筛选后没有可用分辨率目录")

    r_vals = np.array([r for r, _ in selected], dtype=float)
    idx_ref = int(np.argmin(np.abs(r_vals - float(args.ref_resolution))))
    ref_r, ref_dir = selected[idx_ref]

    out_root = Path(args.out_root) if args.out_root else (topic_root / f"labels_ref_r{ref_r:.4f}")
    out_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[label] topic_root={topic_root}")
        print(f"[label] selected={len(selected)} resolutions")
        print(f"[label] reference r={ref_r:.4f} -> {ref_dir}")
        print(f"[label] out_root={out_root}")

    # 读取参考分辨率表
    ref_topics = read_topics_table(ref_dir / "topics_top_words.csv")
    ref_comm = pd.read_csv(ref_dir / "communities_topic_weights.csv", encoding="utf-8-sig") if (ref_dir / "communities_topic_weights.csv").exists() else pd.DataFrame()
    ref_tcomm = pd.read_csv(ref_dir / "topic_representative_communities.csv", encoding="utf-8-sig") if (ref_dir / "topic_representative_communities.csv").exists() else pd.DataFrame()

    if "topic_id" not in ref_topics.columns:
        raise KeyError(f"{ref_dir / 'topics_top_words.csv'} 缺少 topic_id 列")

    topic_ids = sorted(pd.to_numeric(ref_topics["topic_id"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(topic_ids) != int(args.k):
        if verbose:
            print(f"[label] warning: 参考分辨率主题数={len(topic_ids)}，与 --k={args.k} 不一致（将按实际主题行处理）")

    # 为参考分辨率每个 topic 生成候选
    ref_rows = []
    cand_rows = []
    for t in topic_ids:
        top_words = get_topic_top_words(ref_topics, t, topn=10)
        rep_titles = collect_rep_titles_for_topic(ref_comm, ref_tcomm, t, top_n_communities=args.top_n_communities)
        cands = generate_topic_label_candidates(topic_id=t, top_words=top_words, rep_titles=rep_titles, n_candidates=args.n_candidates)
        if not cands:
            cands = [{
                "topic_id": t,
                "label_zh": f"主题{t}",
                "label_en": f"Topic {t}",
                "method": "fallback_empty",
                "confidence": 0.1,
                "phrase_evidence": "",
                "top_words": " ".join(top_words),
            }]

        # 主标签 = 第1候选
        best = dict(cands[0])
        best["ref_resolution"] = float(ref_r)
        best["topic_id_ref"] = int(t)
        best["candidate_count"] = len(cands)
        ref_rows.append(best)

        for rank, c in enumerate(cands, start=1):
            rr = dict(c)
            rr["ref_resolution"] = float(ref_r)
            rr["topic_id_ref"] = int(t)
            rr["candidate_rank"] = int(rank)
            cand_rows.append(rr)

    df_ref_labels = pd.DataFrame(ref_rows).sort_values("topic_id_ref").reset_index(drop=True)
    df_ref_cands = pd.DataFrame(cand_rows).sort_values(["topic_id_ref", "candidate_rank"]).reset_index(drop=True)

    df_ref_labels.to_csv(out_root / "topic_labels_reference.csv", index=False, encoding="utf-8-sig")
    df_ref_cands.to_csv(out_root / "topic_labels_reference_candidates.csv", index=False, encoding="utf-8-sig")

    labels_json = {
        "reference_resolution": float(ref_r),
        "labels": {
            str(int(row["topic_id_ref"])): {
                "label_zh": str(row.get("label_zh", "")),
                "label_en": str(row.get("label_en", "")),
                "method": str(row.get("method", "")),
                "confidence": float(row.get("confidence", np.nan)) if pd.notna(row.get("confidence", np.nan)) else None,
                "top_words": str(row.get("top_words", "")),
            }
            for _, row in df_ref_labels.iterrows()
        }
    }
    (out_root / "topic_labels_reference.json").write_text(json.dumps(labels_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 若提供了对齐目录，则生成“各分辨率标签传播表”
    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else None
    df_align = None
    if alignment_dir is not None:
        p_align = alignment_dir / "topic_alignment.csv"
        if not p_align.exists():
            raise FileNotFoundError(f"alignment_dir 中缺少 {p_align.name}: {p_align}")
        df_align = pd.read_csv(p_align, encoding="utf-8-sig")
        if verbose:
            print(f"[label] using alignment: {p_align}")

    # 构建 ref topic -> label 映射
    ref_label_map = {int(r["topic_id_ref"]): str(r["label_zh"]) for _, r in df_ref_labels.iterrows()}
    ref_label_en_map = {int(r["topic_id_ref"]): str(r.get("label_en", "")) for _, r in df_ref_labels.iterrows()}

    rows_by_res = []
    for r, rdir in selected:
        # 若有对齐目录，则优先取对齐后的 rdir（它的 topic_id 就已经是 ref 编号）
        actual_rdir = rdir
        if alignment_dir is not None:
            cand = alignment_dir / f"r{r:.4f}"
            if cand.exists() and (cand / "communities_topic_weights.csv").exists():
                actual_rdir = cand

        # topics 表（可能是对齐后的 topic_id）
        topics_path = actual_rdir / "topics_top_words.csv"
        comm_path = actual_rdir / "communities_topic_weights.csv"
        if not topics_path.exists() or not comm_path.exists():
            continue
        dft = read_topics_table(topics_path)
        dfc = pd.read_csv(comm_path, encoding="utf-8-sig")

        # 构建 target_topic -> ref_topic 映射
        t2r_map: dict[int, int]
        if df_align is not None:
            sub = df_align[np.isclose(pd.to_numeric(df_align["resolution"], errors="coerce"), float(r))].copy()
            if not sub.empty and {"target_topic", "ref_topic"}.issubset(sub.columns):
                t2r_map = {int(a): int(b) for a, b in zip(pd.to_numeric(sub["target_topic"], errors="coerce").fillna(-1), pd.to_numeric(sub["ref_topic"], errors="coerce").fillna(-1))}
            else:
                # 如果 alignment_dir 已经重排 topics，则恒等映射即可
                t2r_map = {int(t): int(t) for t in pd.to_numeric(dft["topic_id"], errors="coerce").dropna().astype(int).tolist()}
        else:
            # 未对齐模式：只有参考分辨率能稳定命名，其他分辨率不做传播
            if abs(float(r) - float(ref_r)) <= 1e-8:
                t2r_map = {int(t): int(t) for t in pd.to_numeric(dft["topic_id"], errors="coerce").dropna().astype(int).tolist()}
            else:
                t2r_map = {}

        # 汇总 topic 标签映射（当前分辨率）
        topic_ids_cur = sorted(pd.to_numeric(dft["topic_id"], errors="coerce").dropna().astype(int).unique().tolist())
        for t in topic_ids_cur:
            ref_t = t2r_map.get(int(t), int(t) if alignment_dir is not None else -1)
            rows_by_res.append({
                "resolution": float(r),
                "topic_id": int(t),
                "ref_topic": int(ref_t),
                "label_zh": ref_label_map.get(int(ref_t), ""),
                "label_en": ref_label_en_map.get(int(ref_t), ""),
                "has_alignment": bool(alignment_dir is not None),
                "is_reference": bool(abs(float(r) - float(ref_r)) <= 1e-8),
                "topic_dir": str(actual_rdir),
            })

        # 可选：写“带标签”的社区表副本（非常适合网页/可视化直接读取）
        if args.write_labeled_community_tables and not dfc.empty:
            out_rdir = out_root / f"r{r:.4f}"
            out_rdir.mkdir(parents=True, exist_ok=True)
            d = dfc.copy()
            if "top1_topic" in d.columns:
                d["top1_topic"] = pd.to_numeric(d["top1_topic"], errors="coerce").astype("Int64")
                d["top1_label_zh"] = d["top1_topic"].map(lambda x: ref_label_map.get(int(x), "") if pd.notna(x) else "")
                d["top1_label_en"] = d["top1_topic"].map(lambda x: ref_label_en_map.get(int(x), "") if pd.notna(x) else "")
            if "top2_topic" in d.columns:
                d["top2_topic"] = pd.to_numeric(d["top2_topic"], errors="coerce").astype("Int64")
                d["top2_label_zh"] = d["top2_topic"].map(lambda x: ref_label_map.get(int(x), "") if pd.notna(x) else "")
                d["top2_label_en"] = d["top2_topic"].map(lambda x: ref_label_en_map.get(int(x), "") if pd.notna(x) else "")
            d.to_csv(out_rdir / "communities_topic_weights_labeled.csv", index=False, encoding="utf-8-sig")

            # 也写一份 topic 表带名称
            dt = dft.copy()
            if "topic_id" in dt.columns:
                dt["topic_id"] = pd.to_numeric(dt["topic_id"], errors="coerce").astype("Int64")
                dt["topic_label_zh"] = dt["topic_id"].map(lambda x: ref_label_map.get(int(x), "") if pd.notna(x) else "")
                dt["topic_label_en"] = dt["topic_id"].map(lambda x: ref_label_en_map.get(int(x), "") if pd.notna(x) else "")
            dt.to_csv(out_rdir / "topics_top_words_labeled.csv", index=False, encoding="utf-8-sig")

    df_by_res = pd.DataFrame(rows_by_res)
    if not df_by_res.empty:
        df_by_res = df_by_res.sort_values(["resolution", "topic_id"]).reset_index(drop=True)
        df_by_res.to_csv(out_root / "topic_labels_by_resolution.csv", index=False, encoding="utf-8-sig")
        # JSON（按分辨率分组）
        payload = {
            "reference_resolution": float(ref_r),
            "topics": {
                f"{float(r):.4f}": {
                    str(int(row["topic_id"])): {
                        "ref_topic": int(row["ref_topic"]),
                        "label_zh": str(row.get("label_zh", "")),
                        "label_en": str(row.get("label_en", "")),
                    }
                    for _, row in grp.iterrows()
                }
                for r, grp in df_by_res.groupby("resolution", sort=True)
            }
        }
        (out_root / "topic_labels_by_resolution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    run_meta = {
        "topic_root": str(topic_root),
        "alignment_dir": str(alignment_dir) if alignment_dir is not None else None,
        "out_root": str(out_root),
        "reference_resolution_requested": float(args.ref_resolution),
        "reference_resolution_actual": float(ref_r),
        "selected_resolutions": [float(r) for r, _ in selected],
        "config": asdict(LabelConfig(
            ref_resolution=float(ref_r),
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            n_candidates=int(args.n_candidates),
            top_n_communities=int(args.top_n_communities),
            overwrite_label_in_comm_table=False,
        )),
        "args": vars(args),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[label] =============================")
    print(f"[label] reference r={ref_r:.4f}")
    print(f"[label] outputs -> {out_root}")
    print(f"[label] main labels -> {out_root / 'topic_labels_reference.csv'}")
    if (out_root / "topic_labels_by_resolution.csv").exists():
        print(f"[label] by-res labels -> {out_root / 'topic_labels_by_resolution.csv'}")


if __name__ == "__main__":
    main()
