from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse

from community import load_membership_for_resolution


@dataclass
class KeywordIndexConfig:
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.2
    max_features: Optional[int] = 250000
    use_title: bool = True
    use_abstract: bool = True
    title_boost: int = 3
    sublinear_tf: bool = True


@dataclass
class SearchHit:
    pid: int
    score: float
    year: int
    title: str
    community: Optional[int] = None
    snippet: str = ""


class KeywordSearcher:
    def __init__(self, vectorizer: Any, X_tfidf: sparse.csr_matrix):
        self.vectorizer = vectorizer
        self.X_tfidf = X_tfidf.tocsr()

    def search(self, query: str, *, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        q = str(query or "").strip()
        if not q:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        qv = self.vectorizer.transform([q])
        scores = (self.X_tfidf @ qv.T).toarray().ravel().astype(np.float32)
        nz = np.flatnonzero(scores > 0)
        if nz.size == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        k = min(int(top_k), int(nz.size))
        top_local = np.argpartition(scores[nz], -k)[-k:]
        idx = nz[top_local]
        order = np.argsort(scores[idx])[::-1]
        idx = idx[order].astype(np.int32)
        return idx, scores[idx]


def _paper_title(p: object) -> str:
    s = str(getattr(p, "name", "") or "").strip()
    return s


def _paper_abs(p: object) -> str:
    s = str(getattr(p, "abstract", "") or "").strip()
    return s


def build_corpus_texts(
    papers: Sequence[object],
    *,
    use_title: bool = True,
    use_abstract: bool = True,
    title_boost: int = 3,
) -> List[str]:
    texts: List[str] = []
    start = 1 if papers and papers[0] is None else 0
    for pid in range(start, len(papers)):
        p = papers[pid]
        parts: List[str] = []
        if use_title:
            t = _paper_title(p)
            if t:
                parts.extend([t] * max(int(title_boost), 1))
        if use_abstract:
            a = _paper_abs(p)
            if a:
                parts.append(a)
        texts.append("\n".join(parts).strip())
    return texts


def _index_dir(base_dir: Path) -> Path:
    return Path(base_dir) / "keyword_index"


def build_or_load_keyword_index(
    papers: Sequence[object],
    *,
    out_dir: Path,
    cfg: Optional[KeywordIndexConfig] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cfg = cfg or KeywordIndexConfig()
    idx_dir = _index_dir(out_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = idx_dir / "tfidf_docs.npz"
    vec_path = idx_dir / "vectorizer.pkl"
    meta_path = idx_dir / "meta.json"

    if (not force) and matrix_path.exists() and vec_path.exists() and meta_path.exists():
        if verbose:
            print(f"[retrieval] loading cached keyword index -> {idx_dir}")
        X = sparse.load_npz(matrix_path).tocsr()
        with vec_path.open("rb") as f:
            vectorizer = pickle.load(f)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"index_dir": idx_dir, "X": X, "vectorizer": vectorizer, "meta": meta}

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:  # pragma: no cover
        raise ImportError("需要 scikit-learn 来构建关键词检索索引：pip install scikit-learn") from e

    texts = build_corpus_texts(
        papers,
        use_title=cfg.use_title,
        use_abstract=cfg.use_abstract,
        title_boost=cfg.title_boost,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
        ngram_range=(int(cfg.ngram_min), int(cfg.ngram_max)),
        min_df=int(cfg.min_df),
        max_df=float(cfg.max_df),
        max_features=cfg.max_features,
        sublinear_tf=bool(cfg.sublinear_tf),
        norm="l2",
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(texts).tocsr().astype(np.float32)

    sparse.save_npz(matrix_path, X)
    with vec_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    meta = {
        "n_docs": int(X.shape[0]),
        "n_terms": int(X.shape[1]),
        "config": {
            "ngram_min": int(cfg.ngram_min),
            "ngram_max": int(cfg.ngram_max),
            "min_df": int(cfg.min_df),
            "max_df": float(cfg.max_df),
            "max_features": None if cfg.max_features is None else int(cfg.max_features),
            "use_title": bool(cfg.use_title),
            "use_abstract": bool(cfg.use_abstract),
            "title_boost": int(cfg.title_boost),
            "sublinear_tf": bool(cfg.sublinear_tf),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"[retrieval] built keyword index -> {idx_dir}  shape={X.shape}")
    return {"index_dir": idx_dir, "X": X, "vectorizer": vectorizer, "meta": meta}



def _make_snippet(text: str, query: str, max_chars: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    q = str(query or "").strip().lower()
    if not q:
        return text[:max_chars]
    pos = text.lower().find(q)
    if pos < 0:
        return text[:max_chars]
    start = max(0, pos - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]



def search_keywords(
    papers: Sequence[object],
    *,
    query: str,
    out_dir: Path,
    top_k: int = 20,
    resolution: Optional[float] = None,
    leiden_dir: Optional[Path] = None,
    cfg: Optional[KeywordIndexConfig] = None,
    force_reindex: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    bundle = build_or_load_keyword_index(
        papers,
        out_dir=out_dir,
        cfg=cfg,
        force=force_reindex,
        verbose=verbose,
    )
    searcher = KeywordSearcher(bundle["vectorizer"], bundle["X"])
    idx0, scores = searcher.search(query, top_k=top_k)

    membership = None
    if resolution is not None and leiden_dir is not None:
        try:
            membership = load_membership_for_resolution(Path(leiden_dir), float(resolution), allow_nearest=True)
        except Exception:
            membership = None

    hits: List[SearchHit] = []
    for doc_idx0, score in zip(idx0.tolist(), scores.tolist()):
        pid = int(doc_idx0 + 1)
        p = papers[pid]
        title = _paper_title(p)
        abstract = _paper_abs(p)
        year = int(getattr(p, "year", 0) or 0)
        bonus = 0.0
        q_low = query.strip().lower()
        if q_low and q_low in title.lower():
            bonus += 0.15
        comm = int(membership[doc_idx0]) if membership is not None else None
        hits.append(
            SearchHit(
                pid=pid,
                score=float(score + bonus),
                year=year,
                title=title,
                community=comm,
                snippet=_make_snippet(abstract, query),
            )
        )

    hits.sort(key=lambda h: h.score, reverse=True)
    return {
        "query": str(query),
        "top_k": int(top_k),
        "resolution": None if resolution is None else float(resolution),
        "index_dir": str(bundle["index_dir"]),
        "hits": [
            {
                "pid": int(h.pid),
                "score": float(h.score),
                "year": int(h.year),
                "title": h.title,
                "community": None if h.community is None else int(h.community),
                "snippet": h.snippet,
            }
            for h in hits
        ],
    }
