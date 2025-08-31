# pcore/naming.py
from __future__ import annotations
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def majority_name(texts: Iterable[str], topk: int = 3,
                  max_features: int = 5000, ngram_range=(1,2), stop_words="english") -> str:
    texts = list(texts)
    if len(texts) == 0:
        return ""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    X = vec.fit_transform(texts)
    idx = np.asarray(X.sum(axis=0)).ravel().argsort()[::-1][:topk]
    vocab = np.array(vec.get_feature_names_out())
    return ", ".join(vocab[i] for i in idx if i < len(vocab))