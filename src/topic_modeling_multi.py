from __future__ import annotations

"""
多分辨率 Leiden 社区结果的 Topic-SCORE 主题建模批处理脚本
===================================================

用途：
- 读取 core.py 生成的 out/leiden_sweep/membership_r*.npy
- 对每个分辨率运行 topic_modeling.py 的同一套主题建模流程
- 输出到 out/topic_modeling_multi/K{K}/r{resolution}/ 或你指定目录
- 生成一份总表 summary.csv，便于比较不同分辨率结果

推荐用法（在项目根目录）：
    python src/run_topic_modeling_multi.py --k 10

只跑部分分辨率：
    python src/run_topic_modeling_multi.py --k 10 --r-min 0.5 --r-max 1.5 --step 0.1 --include 1.0

显式指定 membership 文件：
    python src/run_topic_modeling_multi.py --k 10 \
      --membership-glob "out/leiden_sweep/membership_r*.npy"
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import math
import re
import sys
import time
import traceback
from typing import Iterable

import numpy as np
import pandas as pd

# 直接复用你已有的单分辨率主题建模实现
try:
    import topic_modeling as tm
except Exception as e:  # pragma: no cover
    raise ImportError(
        "无法导入 topic_modeling.py。请确保本脚本与 topic_modeling.py 同在 src/ 目录，"
        "并从项目根目录运行。"
    ) from e


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"

MEM_RE = re.compile(r"membership_r([0-9]+(?:\.[0-9]+)?)\.npy$")


def _parse_resolution_from_membership_path(path: Path) -> float:
    m = MEM_RE.search(path.name)
    if not m:
        raise ValueError(f"无法从文件名解析分辨率: {path.name}（期望 membership_r1.0000.npy）")
    return float(m.group(1))


def _frange_inclusive(r_min: float, r_max: float, step: float) -> list[float]:
    vals = []
    x = float(r_min)
    # 给浮点误差留余地
    while x <= float(r_max) + 1e-12:
        vals.append(round(x, 10))
        x += float(step)
    return vals


def _normalize_requested_resolutions(
    *,
    r_min: float | None,
    r_max: float | None,
    step: float | None,
    include: Iterable[float] | None,
) -> list[float] | None:
    """返回用户期望的分辨率列表；若为 None 表示不过滤（全扫目录）。"""
    include_set = set(float(x) for x in (include or []))
    if r_min is None or r_max is None or step is None:
        return sorted(include_set) if include_set else None
    vals = set(_frange_inclusive(r_min, r_max, step))
    vals |= include_set
    return sorted(vals)


def discover_memberships(leiden_dir: Path, membership_glob: str) -> list[tuple[float, Path]]:
    paths = sorted(leiden_dir.glob(Path(membership_glob).name if '*' in membership_glob else membership_glob))
    # 若传的是相对路径模式（含目录），直接对 BASE_DIR 展开
    if not paths and ("*" in membership_glob or "?" in membership_glob or "[" in membership_glob):
        paths = sorted(BASE_DIR.glob(membership_glob))
    out: list[tuple[float, Path]] = []
    for p in paths:
        if p.is_file() and p.name.startswith("membership_r") and p.suffix == ".npy":
            try:
                r = _parse_resolution_from_membership_path(p)
                out.append((r, p))
            except Exception:
                continue
    out.sort(key=lambda x: x[0])
    return out


def select_memberships(
    discovered: list[tuple[float, Path]],
    requested_resolutions: list[float] | None,
    tol: float = 1e-8,
) -> list[tuple[float, Path]]:
    if requested_resolutions is None:
        return discovered
    selected: list[tuple[float, Path]] = []
    for r_req in requested_resolutions:
        hit = None
        for r, p in discovered:
            if abs(float(r) - float(r_req)) <= tol:
                hit = (r, p)
                break
        if hit is not None:
            selected.append(hit)
    # 去重并按 r 排序
    uniq = {}
    for r, p in selected:
        uniq[round(float(r), 10)] = (r, p)
    return [uniq[k] for k in sorted(uniq.keys())]




def select_memberships_by_interval(
    discovered: list[tuple[float, Path]],
    r_min: float | None = 0.0001,
    r_max: float | None = 5.0,
    include: list[float] | None = None,
    tol: float = 1e-8,
) -> list[tuple[float, Path]]:
    """按区间筛选目录里已有的 membership 文件。

    适用于 out/leiden_sweep 中混有多轮 sweep 结果时：
    不再按 (r_min, r_max, step) 生成理论网格做“精确匹配”，
    而是直接扫描目录并筛选落在区间内的文件。
    """
    include = [float(x) for x in (include or [])]

    selected: list[tuple[float, Path]] = []
    for r, p in discovered:
        in_range = True
        if r_min is not None:
            in_range = in_range and (float(r) >= float(r_min) - tol)
        if r_max is not None:
            in_range = in_range and (float(r) <= float(r_max) + tol)
        in_include = any(abs(float(r) - x) <= tol for x in include)
        if in_range or in_include:
            selected.append((r, p))

    uniq = {}
    for r, p in selected:
        uniq[round(float(r), 10)] = (r, p)
    return [uniq[k] for k in sorted(uniq.keys())]
def _weighted_topic_prevalence(df_comm: pd.DataFrame, k: int) -> list[float]:
    """按社区论文数加权，计算每个主题平均权重（比只看 top1 更稳）。"""
    if df_comm.empty:
        return [float("nan")] * k
    w = df_comm.get("n_papers", pd.Series(np.ones(len(df_comm)))).to_numpy(dtype=float)
    w = np.maximum(w, 1.0)
    out = []
    for t in range(k):
        col = f"topic_{t}"
        if col not in df_comm.columns:
            out.append(float("nan"))
        else:
            x = pd.to_numeric(df_comm[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            out.append(float(np.average(x, weights=w)))
    return out


def _top1_mass(df_comm: pd.DataFrame, k: int) -> list[float]:
    """按社区规模统计 top1 topic 的占比（硬划分视角）。"""
    if df_comm.empty or "top1_topic" not in df_comm.columns:
        return [float("nan")] * k
    size = pd.to_numeric(df_comm.get("n_papers", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
    size = np.maximum(size, 1.0)
    t1 = pd.to_numeric(df_comm["top1_topic"], errors="coerce").fillna(-1).astype(int).to_numpy()
    total = float(size.sum()) if size.size else 1.0
    vals = []
    for t in range(k):
        vals.append(float(size[t1 == t].sum() / total))
    return vals


def run_one_resolution(
    *,
    resolution: float,
    membership_path: Path,
    out_dir: Path,
    data: dict,
    stopwords: set[str],
    build_cfg: tm.BuildMatrixConfig,
    score_cfg: tm.TopicScoreConfig,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    n_papers = int(data["n_papers"])
    membership = tm.load_membership(membership_path, n_papers_expected=n_papers)

    if verbose:
        print(f"\n[multi] === r={resolution:.4f} ===")
        print(f"[multi] membership = {membership_path}")
        print(f"[multi] output     = {out_dir}")

    matrix_res = tm.build_community_term_matrix(data, membership, stopwords, build_cfg)
    topic_res = tm.fit_topic_score_on_communities(matrix_res, score_cfg)
    runtime = time.time() - t0
    tm.save_outputs(out_dir, data, matrix_res, topic_res, build_cfg, score_cfg, runtime_sec=runtime)

    # 读回摘要表以生成跨分辨率对比指标（避免重复写逻辑）
    df_comm = pd.read_csv(out_dir / "communities_topic_weights.csv", encoding="utf-8-sig")

    top1_weight_mean = float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").mean()) if not df_comm.empty else float("nan")
    top1_weight_median = float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").median()) if not df_comm.empty else float("nan")
    purity_gap = None
    if {"top1_weight", "top2_weight"}.issubset(df_comm.columns):
        gap = pd.to_numeric(df_comm["top1_weight"], errors="coerce") - pd.to_numeric(df_comm["top2_weight"], errors="coerce")
        purity_gap = float(gap.mean())
    else:
        purity_gap = float("nan")

    topic_weighted = _weighted_topic_prevalence(df_comm, score_cfg.k)
    topic_top1_mass = _top1_mass(df_comm, score_cfg.k)

    # 解析 topic_model_meta 以补充 SVD / 顶点搜索信息
    meta_json = out_dir / "topic_model_meta.json"
    topic_meta = {}
    if meta_json.exists():
        try:
            topic_meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            topic_meta = {}

    row = {
        "resolution": float(resolution),
        "membership_path": str(membership_path),
        "out_dir": str(out_dir),
        "runtime_sec": float(runtime),
        "n_communities_all": int(matrix_res.D_all.shape[1]),
        "n_communities_fit": int(matrix_res.D_fit.shape[1]),
        "n_vocab": int(matrix_res.D_all.shape[0]),
        "mean_top1_weight": top1_weight_mean,
        "median_top1_weight": top1_weight_median,
        "mean_top1_minus_top2": purity_gap,
        "status": "ok",
    }
    for t, v in enumerate(topic_weighted):
        row[f"topic_weighted_mean_{t}"] = float(v)
    for t, v in enumerate(topic_top1_mass):
        row[f"topic_top1_mass_{t}"] = float(v)

    # 摘几个主题词用于快速浏览
    try:
        df_topics = pd.read_csv(out_dir / "topics_top_words.csv", encoding="utf-8-sig")
        for t in range(min(score_cfg.k, len(df_topics))):
            row[f"topic_{t}_top_words"] = str(df_topics.loc[t, "top_words"])
    except Exception:
        pass

    vh_meta = ((topic_meta or {}).get("topic_score_meta") or {}).get("vh") or {}
    row["vh_method_actual"] = vh_meta.get("method", score_cfg.vh_method)
    if "m" in ((topic_meta or {}).get("topic_score_meta") or {}):
        row["vh_m"] = ((topic_meta or {}).get("topic_score_meta") or {}).get("m")
    if "k0" in ((topic_meta or {}).get("topic_score_meta") or {}):
        row["vh_k0"] = ((topic_meta or {}).get("topic_score_meta") or {}).get("k0")

    if verbose:
        print(
            f"[multi] done r={resolution:.4f} in {runtime:.1f}s | "
            f"C={matrix_res.D_all.shape[1]} V={matrix_res.D_all.shape[0]} "
            f"mean_top1={top1_weight_mean:.3f}"
        )

    return row


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="多分辨率 Leiden 结果的 Topic-SCORE 批处理")

    # ---- 关键参数 ----
    p.add_argument("--k", type=int, required=True, help="全局主题数 K（例如 10）")
    p.add_argument("--leiden-dir", type=str, default=str(OUT_DIR / "leiden_sweep"), help="membership 文件目录")
    p.add_argument("--membership-glob", type=str, default="membership_r*.npy", help="membership 文件 glob 模式")
    p.add_argument("--out-root", type=str, default=None,
                   help="输出根目录；默认 out/topic_modeling_multi/K{K}")
    p.add_argument("--summary-name", type=str, default="summary_multires.csv")

    # ---- 分辨率选择（与 core.py 对齐风格）----
    p.add_argument("--r-min", type=float, default=0.0001, help="范围模式下最小分辨率（默认 0.0001）")
    p.add_argument("--r-max", type=float, default=5.0, help="范围模式下最大分辨率（默认 5.0）")
    p.add_argument("--step", type=float, default=None, help="仅用于记录/兼容；范围模式不再按 step 精确匹配")
    p.add_argument("--include", type=float, nargs="*", default=None, help="额外强制包含的分辨率，如 1.0")
    p.add_argument("--resolutions", type=float, nargs="*", default=None,
                   help="显式指定分辨率列表；提供后优先于 r-min/r-max/step")

    # ---- 行为控制 ----
    p.add_argument("--skip-existing", action="store_true", help="若输出目录已有 topic_model_meta.json，则跳过")
    p.add_argument("--continue-on-error", action="store_true", help="单个分辨率失败后继续跑其他分辨率")
    p.add_argument("--dry-run", action="store_true", help="只列出将要处理的分辨率，不执行")
    p.add_argument("--quiet", action="store_true")

    # ---- 复用 topic_modeling.py 的参数 ----
    p.add_argument("--data-store", type=str, default=str(DATA_DIR / "data_store.pkl"))
    p.add_argument("--stopwords", type=str, default=str(DATA_DIR / "stopwords.txt"))

    p.add_argument("--no-title", action="store_true")
    p.add_argument("--no-abstract", action="store_true")
    p.add_argument("--no-authors", action="store_true")
    p.add_argument("--use-clean-abstract", action="store_true")
    p.add_argument("--title-weight", type=int, default=3)
    p.add_argument("--abstract-weight", type=int, default=1)
    p.add_argument("--author-weight", type=int, default=1)
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--no-sklearn-stopwords", action="store_true")

    p.add_argument("--words-percent", type=float, default=0.2)
    p.add_argument("--docs-percent", type=float, default=1.0)
    p.add_argument("--min-community-size", type=int, default=1)

    p.add_argument("--vh-method", type=str, default="svs-sp", choices=["svs", "sp", "svs-sp"])
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--k0", type=int, default=None)
    p.add_argument("--mquantile", type=float, default=0.0)
    p.add_argument("--m-trunc-mode", type=str, default="floor", choices=["floor", "cap"])
    p.add_argument("--max-svs-combinations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-weighted-nnls", action="store_true")
    p.add_argument("--max-papers", type=int, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    verbose = not args.quiet
    t_all = time.time()

    leiden_dir = Path(args.leiden_dir)
    if not leiden_dir.exists():
        raise FileNotFoundError(f"leiden_dir 不存在: {leiden_dir}")

    discovered = discover_memberships(leiden_dir, args.membership_glob)
    if not discovered:
        raise FileNotFoundError(
            f"在 {leiden_dir} 中未找到 membership 文件（模式={args.membership_glob}）。"
            "请确认 core.py 的 leiden_sweep 已运行并保存 membership_r*.npy。"
        )

    if args.resolutions is not None and len(args.resolutions) > 0:
        # 显式指定分辨率：保持精确匹配行为
        requested = sorted(set(float(x) for x in args.resolutions) | set(float(x) for x in (args.include or [])))
        selected = select_memberships(discovered, requested)
    else:
        # 范围模式：按区间筛选目录里已有文件（不按 step 重建理论网格）
        selected = select_memberships_by_interval(
            discovered,
            r_min=args.r_min,
            r_max=args.r_max,
            include=args.include,
        )
    if not selected:
        req_desc = args.resolutions if args.resolutions else {"r_min": args.r_min, "r_max": args.r_max, "include": args.include}
        raise RuntimeError(f"没有匹配到任何 membership 文件。请求条件={req_desc}")

    # 输出根目录
    out_root = Path(args.out_root) if args.out_root else (OUT_DIR / "topic_modeling_multi" / f"K{args.k}")
    out_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[multi] discovered memberships: {len(discovered)}")
        print(f"[multi] selected memberships : {len(selected)}")
        if args.resolutions is None:
            print(f"[multi] selection mode       : interval scan in [{args.r_min}, {args.r_max}] (step ignored for matching)")
        print("[multi] resolutions:", ", ".join(f"{r:.4f}" for r, _ in selected))
        print(f"[multi] out_root={out_root}")

    if args.dry_run:
        return

    # 一次性加载 data / stopwords（节省时间）
    data_store_path = Path(args.data_store)
    stopwords_path = Path(args.stopwords) if args.stopwords else None
    data = tm.load_data_store(data_store_path)
    stopwords = tm.load_stopwords(stopwords_path, add_sklearn_english=(not args.no_sklearn_stopwords))
    if verbose:
        print(f"[multi] data_store={data_store_path}")
        print(f"[multi] stopwords={stopwords_path} loaded={len(stopwords)}")

    build_cfg = tm.BuildMatrixConfig(
        include_title=not args.no_title,
        include_abstract=not args.no_abstract,
        include_authors=not args.no_authors,
        use_clean_abstract=args.use_clean_abstract,
        title_weight=args.title_weight,
        abstract_weight=args.abstract_weight,
        author_weight=args.author_weight,
        min_token_len=args.min_token_len,
        words_percent=args.words_percent,
        docs_percent=args.docs_percent,
        min_community_size=args.min_community_size,
        max_papers=args.max_papers,
    )
    score_cfg = tm.TopicScoreConfig(
        k=args.k,
        vh_method=args.vh_method,
        m=args.m,
        k0=args.k0,
        mquantile=args.mquantile,
        m_trunc_mode=args.m_trunc_mode,
        seed=args.seed,
        max_svs_combinations=args.max_svs_combinations,
        weighted_nnls=(not args.no_weighted_nls) if hasattr(args, 'no_weighted_nls') else (not args.no_weighted_nnls),
    )

    # 兼容上面 typo 防护（正常走 no_weighted_nnls）
    score_cfg.weighted_nnls = not args.no_weighted_nnls

    rows: list[dict] = []
    errors: list[dict] = []

    for r, mem_path in selected:
        out_dir = out_root / f"r{r:.4f}"
        meta_path = out_dir / "topic_model_meta.json"

        if args.skip_existing and meta_path.exists():
            if verbose:
                print(f"[multi] skip existing r={r:.4f} -> {out_dir}")
            # 尽量仍写入 summary 行（从已有文件读取）
            try:
                df_comm = pd.read_csv(out_dir / "communities_topic_weights.csv", encoding="utf-8-sig")
                row = {
                    "resolution": float(r),
                    "membership_path": str(mem_path),
                    "out_dir": str(out_dir),
                    "status": "skipped_existing",
                    "n_communities_all": int(df_comm.shape[0]),
                    "mean_top1_weight": float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").mean()),
                    "median_top1_weight": float(pd.to_numeric(df_comm.get("top1_weight"), errors="coerce").median()),
                    "mean_top1_minus_top2": float((pd.to_numeric(df_comm.get("top1_weight"), errors="coerce") - pd.to_numeric(df_comm.get("top2_weight"), errors="coerce")).mean()) if {"top1_weight", "top2_weight"}.issubset(df_comm.columns) else float("nan"),
                }
                for t, v in enumerate(_weighted_topic_prevalence(df_comm, args.k)):
                    row[f"topic_weighted_mean_{t}"] = float(v)
                for t, v in enumerate(_top1_mass(df_comm, args.k)):
                    row[f"topic_top1_mass_{t}"] = float(v)
                rows.append(row)
            except Exception:
                rows.append({"resolution": float(r), "membership_path": str(mem_path), "out_dir": str(out_dir), "status": "skipped_existing"})
            continue

        try:
            row = run_one_resolution(
                resolution=r,
                membership_path=mem_path,
                out_dir=out_dir,
                data=data,
                stopwords=stopwords,
                build_cfg=build_cfg,
                score_cfg=score_cfg,
                verbose=verbose,
            )
            rows.append(row)
        except Exception as e:
            err = {
                "resolution": float(r),
                "membership_path": str(mem_path),
                "out_dir": str(out_dir),
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            rows.append({k: v for k, v in err.items() if k != "traceback"})
            print(f"[multi][ERROR] r={r:.4f}: {e}", file=sys.stderr)
            if not args.continue_on_error:
                break

    # 写总表
    df_summary = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True) if rows else pd.DataFrame()
    summary_csv = out_root / args.summary_name
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    run_meta = {
        "runtime_total_sec": time.time() - t_all,
        "k": int(args.k),
        "leiden_dir": str(leiden_dir),
        "selected_resolutions": [float(r) for r, _ in selected],
        "n_selected": len(selected),
        "n_ok": int((df_summary.get("status") == "ok").sum()) if not df_summary.empty and "status" in df_summary.columns else 0,
        "n_error": len(errors),
        "build_cfg": asdict(build_cfg),
        "score_cfg": asdict(score_cfg),
        "args": vars(args),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if errors:
        (out_root / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[multi] =============================")
    print(f"[multi] finished in {time.time() - t_all:.1f}s")
    print(f"[multi] summary -> {summary_csv}")
    print(f"[multi] ok={run_meta['n_ok']} error={run_meta['n_error']}")
    if errors:
        print("[multi] errors saved ->", out_root / "errors.json")


if __name__ == "__main__":
    main()
