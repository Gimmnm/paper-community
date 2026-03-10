from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import igraph as ig
import leidenalg
import numpy as np


DEFAULT_RESOLUTION_NDIGITS = 4


def _round_resolution(r: float, ndigits: int = DEFAULT_RESOLUTION_NDIGITS) -> float:
    return round(float(r), ndigits)


def membership_filename(resolution: float, ndigits: int = DEFAULT_RESOLUTION_NDIGITS) -> str:
    r = _round_resolution(resolution, ndigits=ndigits)
    return f"membership_r{r:.{ndigits}f}.npy"



def make_resolutions(
    r_min: float,
    r_max: float,
    step: float,
    include: Optional[List[float]] = None,
    ndigits: int = DEFAULT_RESOLUTION_NDIGITS,
) -> List[float]:
    """
    生成 resolution 列表，按 step 扫描，并额外强制包含 include 中的值（比如 1.0）。
    同时统一 round，避免 float key 精度坑。

    说明：
      - 这里保留你原先“更偏 dense sampling”的思路，默认还是用 geomspace；
      - 但会按照 [r_min, r_max] 过滤，避免扫出无关范围。
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if r_min <= 0 or r_max <= 0 or r_max < r_min:
        raise ValueError("require 0 < r_min <= r_max")

    arr = np.geomspace(max(r_min, 1e-6), r_max, 60).round(6).tolist()
    res = [_round_resolution(float(x), ndigits=ndigits) for x in arr if (r_min <= float(x) <= r_max)]

    if include:
        for x in include:
            x = _round_resolution(float(x), ndigits=ndigits)
            if r_min <= x <= r_max and x not in res:
                res.append(x)

    res = sorted(set(res))
    return res



def pick_nearest_resolution(results: Dict[float, Any], r_target: float) -> float:
    """
    给定 results（key 为 resolution float），返回最接近 r_target 的那个 key。
    """
    keys = sorted(results.keys())
    if not keys:
        raise ValueError("empty results")
    r_target = float(r_target)
    return min(keys, key=lambda r: abs(r - r_target))



def leiden_partition(
    G: ig.Graph,
    resolution: float,
    *,
    seed: int = 42,
    weights_attr: str = "weight",
    membership_out_npy: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    在单个 resolution 上跑一次 Leiden。

    返回格式与 leiden_sweep 的单次结果保持一致，便于后续统一处理。
    """
    resolution = _round_resolution(resolution)
    t0 = time.time()
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=G.es[weights_attr] if weights_attr in G.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    membership = np.asarray(part.membership, dtype=np.int32)
    result = {
        "resolution": resolution,
        "membership": membership,
        "n_comm": int(len(set(membership.tolist()))),
        "quality": float(part.quality()),
        "time": float(time.time() - t0),
    }

    if membership_out_npy is not None:
        membership_out_npy = Path(membership_out_npy)
        membership_out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(membership_out_npy, membership)
        if verbose:
            print(f"[leiden] saved membership -> {membership_out_npy}")

    if verbose:
        print(
            f"[leiden] single r={resolution:.4f}  "
            f"n_comm={result['n_comm']:5d}  quality={result['quality']:.6f}  time={result['time']:.1f}s"
        )

    return result



def load_membership_for_resolution(
    out_dir: Path,
    r_target: float,
    *,
    allow_nearest: bool = True,
    ndigits: int = DEFAULT_RESOLUTION_NDIGITS,
) -> np.ndarray:
    """
    从 out_dir 中读取给定 resolution 的 membership。

    读取策略：
      1) 先尝试精确文件名 membership_r{r}.npy
      2) 若找不到且 allow_nearest=True，则读取 summary.npy，选最近的 resolution
    """
    out_dir = Path(out_dir)
    r_target = _round_resolution(r_target, ndigits=ndigits)
    exact_path = out_dir / membership_filename(r_target, ndigits=ndigits)
    if exact_path.exists():
        return np.load(exact_path).astype(np.int32)

    if not allow_nearest:
        raise FileNotFoundError(f"membership not found: {exact_path}")

    summary_path = out_dir / "summary.npy"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"membership not found: {exact_path}; summary.npy also missing under {out_dir}"
        )

    summary = np.load(summary_path, allow_pickle=True).item()
    resolutions = [float(x) for x in np.asarray(summary["resolutions"]).tolist()]
    if not resolutions:
        raise ValueError(f"no resolutions recorded in {summary_path}")

    nearest = min(resolutions, key=lambda x: abs(x - r_target))
    nearest_path = out_dir / membership_filename(nearest, ndigits=ndigits)
    if not nearest_path.exists():
        raise FileNotFoundError(f"nearest membership file missing: {nearest_path}")
    return np.load(nearest_path).astype(np.int32)



def load_or_run_leiden_partition(
    G: ig.Graph,
    out_dir: Path,
    resolution: float,
    *,
    seed: int = 42,
    weights_attr: str = "weight",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    优先从缓存加载单个 resolution 的 membership；没有则现算并保存。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_path = out_dir / membership_filename(resolution)

    if membership_path.exists():
        membership = np.load(membership_path).astype(np.int32)
        result = {
            "resolution": _round_resolution(resolution),
            "membership": membership,
            "n_comm": int(len(set(membership.tolist()))),
            "quality": float("nan"),
            "time": 0.0,
        }
        if verbose:
            print(f"[leiden] loading cached membership -> {membership_path}")
        return result

    return leiden_partition(
        G,
        resolution,
        seed=seed,
        weights_attr=weights_attr,
        membership_out_npy=membership_path,
        verbose=verbose,
    )



def leiden_sweep(
    G: ig.Graph,
    out_dir: Path,
    *,
    r_min: float = 0.2,
    r_max: float = 2.0,
    step: float = 0.05,
    include: Optional[List[float]] = None,
    seed: int = 42,
    weights_attr: str = "weight",
    save_each_membership: bool = True,
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """
    对图 G 做 Leiden resolution sweep。

    返回：
      results[r] = {
        "resolution": r,
        "membership": np.ndarray shape (N,),
        "n_comm": int,
        "quality": float,
        "time": float
      }

    落盘：
      - out_dir/summary.npy
      - 如果 save_each_membership=True：out_dir/membership_r{r:.4f}.npy
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if include is None:
        include = [1.0]

    resolutions = make_resolutions(r_min, r_max, step, include=include, ndigits=DEFAULT_RESOLUTION_NDIGITS)
    if verbose:
        print(f"[leiden] sweep resolutions: {resolutions[:8]} ... {resolutions[-5:]}  (n={len(resolutions)})")

    results: Dict[float, Dict[str, Any]] = {}
    t_all = time.time()
    for r in resolutions:
        membership_out_npy = out_dir / membership_filename(r) if save_each_membership else None
        result = leiden_partition(
            G,
            r,
            seed=seed,
            weights_attr=weights_attr,
            membership_out_npy=membership_out_npy,
            verbose=verbose,
        )
        results[r] = result

    summary = {
        "resolutions": np.array(sorted(results.keys()), dtype=np.float32),
        "n_comm": np.array([results[r]["n_comm"] for r in sorted(results.keys())], dtype=np.int32),
        "quality": np.array([results[r]["quality"] for r in sorted(results.keys())], dtype=np.float64),
        "time": np.array([results[r]["time"] for r in sorted(results.keys())], dtype=np.float64),
    }
    np.save(out_dir / "summary.npy", summary, allow_pickle=True)

    if verbose:
        print(f"[leiden] saved summary -> {out_dir / 'summary.npy'}")
        print(f"[leiden] sweep done, runs={len(resolutions)}, total_time={time.time()-t_all:.1f}s")

    return results
