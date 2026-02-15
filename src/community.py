# community.py

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

import igraph as ig
import leidenalg


def make_resolutions(
    r_min: float,
    r_max: float,
    step: float,
    include: Optional[List[float]] = None,
    ndigits: int = 4,
) -> List[float]:
    """
    生成 resolution 列表，按 step 扫描，并额外强制包含 include 中的值（比如 1.0）。
    同时统一 round，避免 float key 精度坑。
    """
    if step <= 0:
        raise ValueError("step must be > 0")

    # np.arange 可能末尾差一点点，所以给一个小 epsilon
    eps = step * 0.5
    # arr = np.arange(r_min, r_max + eps, step, dtype=np.float64) # 可能有精度问题，改用 geomspace + round
    arr = np.geomspace(1e-4, 5.0, 60).round(6).tolist()
    res = [round(float(x), ndigits) for x in arr]

    if include:
        for x in include:
            x = round(float(x), ndigits)
            if x not in res:
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


def leiden_sweep(
    G: ig.Graph,
    out_dir: Path,
    *,
    r_min: float = 0.2,
    r_max: float = 2.0,
    step: float = 0.05,            # <<<<<< 扫细：你要更细就改 0.02
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
        include = [1.0]  # 默认强制包含 1.0，避免你 core 里取不到

    resolutions = make_resolutions(r_min, r_max, step, include=include, ndigits=4)
    if verbose:
        print(f"[leiden] sweep resolutions: {resolutions[:8]} ... {resolutions[-5:]}  (n={len(resolutions)})")

    results: Dict[float, Dict[str, Any]] = {}

    # sweep
    t_all = time.time()
    for r in resolutions:
        t0 = time.time()
        part = leidenalg.find_partition(
            G,
            leidenalg.RBConfigurationVertexPartition,
            weights=G.es[weights_attr] if weights_attr in G.es.attributes() else None,
            resolution_parameter=r,
            seed=seed,
        )

        membership = np.asarray(part.membership, dtype=np.int32)
        n_comm = len(set(membership.tolist()))
        quality = float(part.quality())
        dt = time.time() - t0

        results[r] = {
            "resolution": r,
            "membership": membership,
            "n_comm": int(n_comm),
            "quality": quality,
            "time": float(dt),
        }

        if verbose:
            print(f"[leiden] r={r:.4f}  n_comm={n_comm:5d}  quality={quality:.6f}  time={dt:.1f}s")

        if save_each_membership:
            np.save(out_dir / f"membership_r{r:.4f}.npy", membership)

    # summary
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
