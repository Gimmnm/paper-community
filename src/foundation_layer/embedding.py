# embedding.py
# -*- coding: utf-8 -*-

from __future__ import annotations  # 允许在类型标注中使用 Paper 等尚未加载的类型

import os  # 读取/设置环境变量（如 HF 缓存、离线模式）
import logging  # 静音 adapters 的 warning、控制日志等级
from typing import List, Optional, Tuple  # 类型标注：列表、可选值、元组

import numpy as np  # 保存 embedding 矩阵（float32），以及落盘 npy
import torch  # 推理加速（GPU/CPU）、no_grad、张量运算
from transformers import AutoTokenizer  # HuggingFace tokenizer：把文本变成 token id
from adapters import AutoAdapterModel  # adapters 版本的 AutoModel：支持 load_adapter / set_active

from foundation_layer.model import Paper

# -----------------------------------------------------------------------------
# 全局：静音一个常见但不影响推理的 adapters 告警
# -----------------------------------------------------------------------------
logging.getLogger("adapters.model_mixin").setLevel(logging.ERROR)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    return torch.device("cpu")  # ✅ 永远 CPU


def load_specter2_proximity(device: torch.device) -> Tuple[AutoTokenizer, AutoAdapterModel]:
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2",
        source="hf",
        load_as="proximity",
        set_active=True,
    )
    model.to(device)
    model.eval()
    try:
        print(f"[specter2] active_adapters = {model.active_adapters}")
    except Exception:
        pass
    return tokenizer, model


def build_text(title: str, abstract: str, sep_token: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if title and abstract:
        return title + sep_token + abstract
    if title:
        return title
    if abstract:
        return abstract
    return ""


@torch.no_grad()
def encode_batch(
    tokenizer: AutoTokenizer,
    model: AutoAdapterModel,
    texts: List[str],
    device: torch.device,
    max_length: int = 512,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    emb = out.last_hidden_state[:, 0, :]
    return emb.detach().cpu().to(torch.float32).numpy()


def sanity_check_adapter_effect(
    tokenizer: AutoTokenizer,
    model: AutoAdapterModel,
    device: torch.device,
) -> None:
    sample = "BERT" + tokenizer.sep_token + "We introduce a new language representation model called BERT."
    e1 = encode_batch(tokenizer, model, [sample], device=device)[0]
    old_active = getattr(model, "active_adapters", None)
    try:
        model.set_active_adapters(None)
    except Exception:
        try:
            model.active_adapters = None
        except Exception:
            pass
    e2 = encode_batch(tokenizer, model, [sample], device=device)[0]
    try:
        model.set_active_adapters(old_active)
    except Exception:
        try:
            model.active_adapters = old_active
        except Exception:
            pass
    num = float(np.dot(e1, e2))
    den = float(np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
    cos = num / den
    print(f"[sanity] cosine(with_adapter, without_adapter) = {cos:.6f} (越小差异越明显)")


def embed_all_papers(
    papers: List[Optional[Paper]],  # ✅ 支持 papers[0]=None 的 1-based
    out_npy_path: str,
    batch_size: int = 32,
    prefer_gpu: bool = True,
    attach_to_papers: bool = False,
    max_length: int = 512,
    print_every: int = 2000,
) -> np.ndarray:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    device = pick_device(prefer_gpu=prefer_gpu)
    tokenizer, model = load_specter2_proximity(device=device)
    sanity_check_adapter_effect(tokenizer, model, device=device)

    n_total = len(papers)
    dim = 768
    embs = np.zeros((n_total, dim), dtype=np.float32)
    sep = tokenizer.sep_token
    start_idx = 1 if (n_total > 0 and papers[0] is None) else 0

    idxs: List[int] = []
    texts: List[str] = []
    empty_cnt = 0

    for i in range(start_idx, n_total):
        p = papers[i]
        if p is None:
            empty_cnt += 1
            continue
        text = build_text(p.name, p.abstract, sep_token=sep)
        if not text.strip():
            empty_cnt += 1
            continue
        idxs.append(i)
        texts.append(text)

    if empty_cnt:
        print(f"[embedding] skipped empty rows: {empty_cnt}/{n_total}")

    processed = 0
    for s in range(0, len(texts), batch_size):
        e = min(s + batch_size, len(texts))
        batch_texts = texts[s:e]
        batch_idxs = idxs[s:e]
        batch_emb = encode_batch(
            tokenizer=tokenizer,
            model=model,
            texts=batch_texts,
            device=device,
            max_length=max_length,
        )
        for j, idx in enumerate(batch_idxs):
            embs[idx] = batch_emb[j]
        processed = e
        if (processed % print_every) == 0 or processed == len(texts):
            print(f"[embedding] encoded {processed}/{len(texts)} non-empty papers ...")

    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, embs)
    print(f"[embedding] saved: {out_npy_path}")
    print(f"[embedding] done: embedded={len(texts)} total_rows={n_total} dim={dim}")

    if attach_to_papers:
        for i in range(start_idx, n_total):
            p = papers[i]
            if p is not None:
                p.embedding = embs[i]
        print("[embedding] attached embeddings to papers[*].embedding")

    return embs
