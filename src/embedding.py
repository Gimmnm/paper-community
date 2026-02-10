# embedding.py
# -*- coding: utf-8 -*-

from __future__ import annotations  # 允许在类型标注中使用 Paper 等尚未加载的类型

import os  # 读取/设置环境变量（如 HF 缓存、离线模式）
import time  # 打印耗时、进度统计
import logging  # 静音 adapters 的 warning、控制日志等级
from typing import List, Optional, Tuple  # 类型标注：列表、可选值、元组

import numpy as np  # 保存 embedding 矩阵（float32），以及落盘 npy
import torch  # 推理加速（GPU/CPU）、no_grad、张量运算
from transformers import AutoTokenizer  # HuggingFace tokenizer：把文本变成 token id
from adapters import AutoAdapterModel  # adapters 版本的 AutoModel：支持 load_adapter / set_active

from model import Paper

# -----------------------------------------------------------------------------
# 全局：静音一个常见但不影响推理的 adapters 告警
# 说明：该告警在 adapters 社区里有人提过（即便 adapter 已激活也会出现）。
# -----------------------------------------------------------------------------
logging.getLogger("adapters.model_mixin").setLevel(logging.ERROR)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    return torch.device("cpu")  # ✅ 永远 CPU



def load_specter2_proximity(device: torch.device) -> Tuple[AutoTokenizer, AutoAdapterModel]:
    """
    加载 SPECTER2（proximity/retrieval adapter）：
    - base: allenai/specter2_base
    - adapter: allenai/specter2  （proximity adapter，官方示例）
    返回：tokenizer, model（已经 set_active=True）
    """
    # 1) tokenizer：负责把 title/abstract 文本转成 input_ids / attention_mask
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")

    # 2) base model：必须用 AutoAdapterModel 才能 load_adapter
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

    # 3) 加载并激活 proximity adapter（官方 HuggingFace 卡片示例）
    #    load_as 给 adapter 一个本地名字，便于打印和后续组合
    model.load_adapter(
        "allenai/specter2",
        source="hf",
        load_as="proximity",
        set_active=True,
    )

    model.to(device)
    model.eval()  # 推理模式：关闭 dropout 等

    # 打印一下，方便你确认（即便仍有 warning，也以这个为准）
    try:
        print(f"[specter2] active_adapters = {model.active_adapters}")
    except Exception:
        # 某些版本属性名略有差异，失败也不影响后续 forward
        pass

    return tokenizer, model


def build_text(title: str, abstract: str, sep_token: str) -> str:
    """
    按 SPECTER2 推荐方式拼接输入：
    text = title + [SEP] + abstract

    - 如果 abstract 为空，就只用 title（用于 title-fallback）
    - 如果 title 也为空，就返回空串（这种样本通常应跳过或置零向量）
    """
    title = (title or "").strip()
    abstract = (abstract or "").strip()

    if title and abstract:
        return title + sep_token + abstract
    if title:
        return title
    if abstract:
        # 理论上很少出现“无 title 有 abstract”，但也兼容一下
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
    """
    把一批文本编码成 embedding：
    - tokenize（padding/truncation）
    - forward
    - 取 CLS token：output.last_hidden_state[:, 0, :]
    返回：numpy float32, shape=(batch, 768)
    """
    # return_token_type_ids=False：BERT/RoBERTa 在很多任务里不需要 token_type_ids
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

    # 官方示例：取第一个 token（CLS）作为 embedding :contentReference[oaicite:3]{index=3}
    emb = out.last_hidden_state[:, 0, :]  # (B, 768)

    # 转回 CPU + numpy
    return emb.detach().cpu().to(torch.float32).numpy()


def sanity_check_adapter_effect(
    tokenizer: AutoTokenizer,
    model: AutoAdapterModel,
    device: torch.device,
) -> None:
    """
    用一个小样本证明“adapter 确实参与了 forward”：
    - 先用当前 active_adapters 计算 embedding
    - 再临时关闭 adapter（active_adapters=None）计算 embedding
    - 比较 cosine 相似度：如果不是接近 1.0，说明 adapter 的确改变了输出
    """
    sample = "BERT" + tokenizer.sep_token + "We introduce a new language representation model called BERT."

    # 1) with adapter
    e1 = encode_batch(tokenizer, model, [sample], device=device)[0]

    # 2) without adapter（临时关闭）
    old_active = getattr(model, "active_adapters", None)
    try:
        model.set_active_adapters(None)  # adapters 官方支持的关闭方式之一 :contentReference[oaicite:4]{index=4}
    except Exception:
        # 不同版本可能没有 set_active_adapters(None)，那就尽量退化处理
        try:
            model.active_adapters = None
        except Exception:
            pass

    e2 = encode_batch(tokenizer, model, [sample], device=device)[0]

    # 3) restore
    try:
        model.set_active_adapters(old_active)
    except Exception:
        try:
            model.active_adapters = old_active
        except Exception:
            pass

    # cosine
    num = float(np.dot(e1, e2))
    den = float(np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
    cos = num / den
    print(f"[sanity] cosine(with_adapter, without_adapter) = {cos:.6f} (越小差异越明显)")


from typing import List, Optional

def embed_all_papers(
    papers: List[Optional[Paper]],   # ✅ 支持 papers[0]=None 的 1-based
    out_npy_path: str,
    batch_size: int = 32,
    prefer_gpu: bool = True,
    attach_to_papers: bool = False,
    max_length: int = 512,
    print_every: int = 2000,
) -> np.ndarray:
    """
    1-based 兼容版本：
      - papers[0] 可以是 None
      - 输出 embs 与 papers 等长：embs[pid] 对应 papers[pid]
      - embs[0] 永远是 0 向量（占位）
    """

    torch.set_num_threads(4)         # 你可以试 4 / 6 / 8
    torch.set_num_interop_threads(2)


    device = pick_device(prefer_gpu=prefer_gpu)
    tokenizer, model = load_specter2_proximity(device=device)

    sanity_check_adapter_effect(tokenizer, model, device=device)

    n_total = len(papers)
    dim = 768
    embs = np.zeros((n_total, dim), dtype=np.float32)

    sep = tokenizer.sep_token

    # ✅ 只处理真实 paper 的下标（跳过 0）
    start_idx = 1 if (n_total > 0 and papers[0] is None) else 0

    # 先收集需要编码的 (idx, text)
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

    # 分 batch 编码并写回 embs[idx]
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

    # 保存
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

