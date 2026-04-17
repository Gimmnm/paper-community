from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


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
    mode: str = "linear",
    num_log: int = 60,
) -> List[float]:
    """
    生成 resolution 列表。

    mode='linear' 时，真正按 step 做线性扫描；
    mode='log' 时，按对数刻度生成 num_log 个点。
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if r_max < r_min:
        raise ValueError("require r_max >= r_min")

    include = include or []
    vals: List[float] = []
    mode = str(mode).lower()

    if mode == "linear":
        cur = float(r_min)
        eps = max(1e-12, float(step) * 0.25)
        while cur <= float(r_max) + eps:
            vals.append(_round_resolution(cur, ndigits=ndigits))
            cur += float(step)
        if not vals:
            vals = [_round_resolution(float(r_min), ndigits=ndigits)]
        if abs(vals[-1] - float(r_max)) > eps:
            vals.append(_round_resolution(float(r_max), ndigits=ndigits))
    elif mode == "log":
        if r_min <= 0 or r_max <= 0:
            raise ValueError("log mode requires 0 < r_min <= r_max")
        arr = np.geomspace(float(r_min), float(r_max), int(num_log))
        vals.extend(_round_resolution(x, ndigits=ndigits) for x in arr.tolist())
    else:
        raise ValueError("mode must be 'linear' or 'log'")

    for x in include:
        x = _round_resolution(float(x), ndigits=ndigits)
        if r_min <= x <= r_max:
            vals.append(x)

    return sorted(set(vals))


def pick_nearest_resolution(results: Dict[float, Any], r_target: float) -> float:
    keys = sorted(results.keys())
    if not keys:
        raise ValueError("empty results")
    return min(keys, key=lambda r: abs(float(r) - float(r_target)))


def _partition_class(partition_type: str):
    if partition_type == "RBConfigurationVertexPartition":
        return leidenalg.RBConfigurationVertexPartition
    if partition_type == "CPMVertexPartition":
        return leidenalg.CPMVertexPartition
    raise ValueError("partition_type must be RBConfigurationVertexPartition or CPMVertexPartition")


def leiden_partition(
    G: ig.Graph,
    resolution: float,
    *,
    seed: int = 42,
    weights_attr: str = "weight",
    membership_out_npy: Optional[Path] = None,
    partition_type: str = "RBConfigurationVertexPartition",
    verbose: bool = True,
) -> Dict[str, Any]:
    resolution = _round_resolution(resolution)
    t0 = time.time()
    part_cls = _partition_class(partition_type)
    part = leidenalg.find_partition(
        G,
        part_cls,
        weights=G.es[weights_attr] if weights_attr in G.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = np.asarray(part.membership, dtype=np.int32)
    result = {
        "resolution": resolution,
        "membership": membership,
        "n_comm": int(np.unique(membership).size),
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
            f"[leiden] r={resolution:.4f}  n_comm={result['n_comm']:5d}  "
            f"quality={result['quality']:.6f}  time={result['time']:.1f}s"
        )
    return result


def load_membership_for_resolution(
    out_dir: Path,
    r_target: float,
    *,
    allow_nearest: bool = True,
    ndigits: int = DEFAULT_RESOLUTION_NDIGITS,
) -> np.ndarray:
    out_dir = Path(out_dir)
    r_target = _round_resolution(r_target, ndigits=ndigits)
    exact_path = out_dir / membership_filename(r_target, ndigits=ndigits)
    if exact_path.exists():
        return np.load(exact_path).astype(np.int32)
    if not allow_nearest:
        raise FileNotFoundError(f"membership not found: {exact_path}")
    summary_path = out_dir / "summary.npy"
    if not summary_path.exists():
        raise FileNotFoundError(f"membership not found: {exact_path}; summary.npy missing under {out_dir}")
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
    partition_type: str = "RBConfigurationVertexPartition",
    verbose: bool = True,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_path = out_dir / membership_filename(resolution)
    if membership_path.exists():
        membership = np.load(membership_path).astype(np.int32)
        if verbose:
            print(f"[leiden] loading cached membership -> {membership_path}")
        return {
            "resolution": _round_resolution(resolution),
            "membership": membership,
            "n_comm": int(np.unique(membership).size),
            "quality": float("nan"),
            "time": 0.0,
        }
    return leiden_partition(
        G,
        resolution,
        seed=seed,
        weights_attr=weights_attr,
        membership_out_npy=membership_path,
        partition_type=partition_type,
        verbose=verbose,
    )


def _label_contingency(labels_a: np.ndarray, labels_b: np.ndarray):
    a = np.asarray(labels_a, dtype=np.int32)
    b = np.asarray(labels_b, dtype=np.int32)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    data = np.ones_like(ia, dtype=np.int64)
    M = sparse.coo_matrix((data, (ia, ib)), shape=(ua.size, ub.size), dtype=np.int64).tocsr()
    size_a = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    size_b = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    return ua, ub, M, size_a, size_b


def variation_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    ua, ub, M, size_a, size_b = _label_contingency(labels_a, labels_b)
    n = float(M.sum())
    if n <= 0:
        return float("nan")
    pa = size_a.astype(np.float64) / n
    pb = size_b.astype(np.float64) / n
    rows, cols = M.nonzero()
    vals = np.asarray(M[rows, cols]).ravel().astype(np.float64) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pa = np.where(pa > 0, np.log(pa), 0.0)
        log_pb = np.where(pb > 0, np.log(pb), 0.0)
        log_pij = np.where(vals > 0, np.log(vals), 0.0)
    H_a = float(-(pa[pa > 0] * log_pa[pa > 0]).sum())
    H_b = float(-(pb[pb > 0] * log_pb[pb > 0]).sum())
    MI = float((vals * (log_pij - log_pa[rows] - log_pb[cols])).sum())
    return float(H_a + H_b - 2.0 * MI)


def summarize_sweep_results(results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(results.keys())
    if not keys:
        raise ValueError("empty results")
    resolutions = np.asarray(keys, dtype=np.float32)
    n_comm = np.asarray([results[r]["n_comm"] for r in keys], dtype=np.int32)
    quality = np.asarray([results[r]["quality"] for r in keys], dtype=np.float64)
    run_time = np.asarray([results[r]["time"] for r in keys], dtype=np.float64)
    vi = np.full(len(keys), np.nan, dtype=np.float64)
    dn = np.zeros(len(keys), dtype=np.int32)
    ratio = np.full(len(keys), np.nan, dtype=np.float64)
    for i in range(1, len(keys)):
        vi[i] = variation_information(results[keys[i - 1]]["membership"], results[keys[i]]["membership"])
        dn[i] = int(n_comm[i] - n_comm[i - 1])
        ratio[i] = float(n_comm[i] / max(int(n_comm[i - 1]), 1))
    return {
        "resolutions": resolutions,
        "n_comm": n_comm,
        "quality": quality,
        "time": run_time,
        "vi_adjacent": vi,
        "delta_n_comm": dn,
        "ratio_n_comm": ratio,
    }


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
    reuse_existing: bool = True,
    resolution_mode: str = "linear",
    partition_type: str = "RBConfigurationVertexPartition",
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if include is None:
        include = [1.0]
    resolutions = make_resolutions(
        r_min,
        r_max,
        step,
        include=include,
        ndigits=DEFAULT_RESOLUTION_NDIGITS,
        mode=resolution_mode,
    )
    if verbose:
        print(f"[leiden] sweep resolutions: {resolutions[:8]} ... {resolutions[-5:]}  (n={len(resolutions)})")
    results: Dict[float, Dict[str, Any]] = {}
    t_all = time.time()
    for r in resolutions:
        membership_path = out_dir / membership_filename(r)
        if reuse_existing and membership_path.exists():
            membership = np.load(membership_path).astype(np.int32)
            result = {
                "resolution": _round_resolution(r),
                "membership": membership,
                "n_comm": int(np.unique(membership).size),
                "quality": float("nan"),
                "time": 0.0,
            }
            if verbose:
                print(f"[leiden] reuse cached membership -> {membership_path}")
        else:
            result = leiden_partition(
                G,
                r,
                seed=seed,
                weights_attr=weights_attr,
                membership_out_npy=membership_path if save_each_membership else None,
                partition_type=partition_type,
                verbose=verbose,
            )
        results[_round_resolution(r)] = result
    summary = summarize_sweep_results(results)
    np.save(out_dir / "summary.npy", summary, allow_pickle=True)
    _write_summary_csv(out_dir / "summary.csv", summary)
    if verbose:
        print(f"[leiden] saved summary -> {out_dir / 'summary.npy'}")
        print(f"[leiden] sweep done, runs={len(resolutions)}, total_time={time.time() - t_all:.1f}s")
    return results


def detect_breakpoints(
    summary: Dict[str, Any],
    *,
    top_k: int = 8,
    weight_n_comm: float = 1.0,
    weight_vi: float = 1.0,
) -> List[Dict[str, Any]]:
    resolutions = np.asarray(summary["resolutions"], dtype=float)
    n_comm = np.asarray(summary["n_comm"], dtype=float)
    vi = np.asarray(summary.get("vi_adjacent", np.full_like(n_comm, np.nan)), dtype=float)
    if resolutions.size <= 1:
        return []
    dn = np.abs(np.diff(n_comm, prepend=n_comm[0]))

    def _robust_z(x: np.ndarray) -> np.ndarray:
        finite = np.isfinite(x)
        if int(finite.sum()) <= 1:
            return np.zeros_like(x, dtype=float)
        med = float(np.median(x[finite]))
        mad = float(np.median(np.abs(x[finite] - med)))
        scale = max(mad * 1.4826, 1e-12)
        z = np.zeros_like(x, dtype=float)
        z[finite] = (x[finite] - med) / scale
        return z

    score = weight_n_comm * np.maximum(_robust_z(dn), 0.0) + weight_vi * np.maximum(
        _robust_z(np.nan_to_num(vi, nan=0.0)), 0.0
    )
    order = np.argsort(score)[::-1]
    out: List[Dict[str, Any]] = []
    seen = set()
    for idx in order:
        if idx == 0:
            continue
        r = float(resolutions[idx])
        if r in seen or score[idx] <= 0:
            continue
        seen.add(r)
        out.append(
            {
                "resolution": r,
                "score": float(score[idx]),
                "n_comm": int(n_comm[idx]),
                "delta_n_comm": int(dn[idx]),
                "vi_adjacent": None if not np.isfinite(vi[idx]) else float(vi[idx]),
            }
        )
        if len(out) >= int(top_k):
            break
    return out


def build_parent_child_links(
    parent_labels: np.ndarray,
    child_labels: np.ndarray,
    *,
    r_parent: float,
    r_child: float,
) -> List[Dict[str, Any]]:
    up, uc, M, size_p, size_c = _label_contingency(parent_labels, child_labels)
    Mcsc = M.tocsc()
    links: List[Dict[str, Any]] = []
    for j in range(Mcsc.shape[1]):
        start, end = Mcsc.indptr[j], Mcsc.indptr[j + 1]
        rows = Mcsc.indices[start:end]
        vals = Mcsc.data[start:end]
        if vals.size == 0:
            continue
        best = int(np.argmax(vals))
        i = int(rows[best])
        inter = int(vals[best])
        parent_size = int(size_p[i])
        child_size = int(size_c[j])
        links.append(
            {
                "r_parent": float(r_parent),
                "community_parent": int(up[i]),
                "size_parent": parent_size,
                "r_child": float(r_child),
                "community_child": int(uc[j]),
                "size_child": child_size,
                "intersection": inter,
                "parent_share": float(inter / max(parent_size, 1)),
                "child_share": float(inter / max(child_size, 1)),
                "jaccard": float(inter / max(parent_size + child_size - inter, 1)),
            }
        )
    return links


def _write_dict_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(str(k))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    resolutions = np.asarray(summary["resolutions"]).tolist()
    n_comm = np.asarray(summary["n_comm"]).tolist()
    quality = np.asarray(summary["quality"]).tolist()
    run_time = np.asarray(summary["time"]).tolist()
    vi = np.asarray(summary.get("vi_adjacent", [])).tolist()
    dn = np.asarray(summary.get("delta_n_comm", [])).tolist()
    ratio = np.asarray(summary.get("ratio_n_comm", [])).tolist()
    for i, r in enumerate(resolutions):
        rows.append(
            {
                "resolution": float(r),
                "n_comm": int(n_comm[i]),
                "quality": None if not np.isfinite(quality[i]) else float(quality[i]),
                "time": None if not np.isfinite(run_time[i]) else float(run_time[i]),
                "vi_adjacent": None if i >= len(vi) or not np.isfinite(vi[i]) else float(vi[i]),
                "delta_n_comm": None if i >= len(dn) else int(dn[i]),
                "ratio_n_comm": None if i >= len(ratio) or not np.isfinite(ratio[i]) else float(ratio[i]),
            }
        )
    _write_dict_rows_csv(path, rows)


def plot_sweep_diagnostics(
    summary: Dict[str, Any],
    *,
    out_png: Path,
    breakpoints: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    resolutions = np.asarray(summary["resolutions"], dtype=float)
    n_comm = np.asarray(summary["n_comm"], dtype=float)
    vi = np.asarray(summary.get("vi_adjacent", np.full_like(n_comm, np.nan)), dtype=float)
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(resolutions, n_comm, marker="o", linewidth=1.2, markersize=3)
    ax1.set_title("Leiden sweep diagnostics")
    ax1.set_ylabel("# communities")
    ax1.grid(alpha=0.25)
    if breakpoints:
        for bp in breakpoints:
            x = float(bp["resolution"])
            ax1.axvline(x, linewidth=1.0, alpha=0.45)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(resolutions, vi, marker="o", linewidth=1.2, markersize=3)
    ax2.set_xlabel("resolution")
    ax2.set_ylabel("VI(adjacent)")
    ax2.grid(alpha=0.25)
    if breakpoints:
        for bp in breakpoints:
            x = float(bp["resolution"])
            ax2.axvline(x, linewidth=1.0, alpha=0.45)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def build_hierarchy_from_sweep(
    results: Dict[float, Dict[str, Any]],
    *,
    out_dir: Optional[Path] = None,
    min_child_share: float = 0.25,
    verbose: bool = True,
) -> Dict[str, Any]:
    keys = sorted(results.keys())
    summary = summarize_sweep_results(results)
    breakpoints = detect_breakpoints(summary)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for r in keys:
        labels = np.asarray(results[r]["membership"], dtype=np.int32)
        uniq, cnt = np.unique(labels, return_counts=True)
        for c, n in zip(uniq.tolist(), cnt.tolist()):
            nodes.append({"resolution": float(r), "community": int(c), "size": int(n)})

    for r_parent, r_child in zip(keys[:-1], keys[1:]):
        parent_labels = np.asarray(results[r_parent]["membership"], dtype=np.int32)
        child_labels = np.asarray(results[r_child]["membership"], dtype=np.int32)
        links = build_parent_child_links(parent_labels, child_labels, r_parent=r_parent, r_child=r_child)
        for row in links:
            if row["child_share"] >= float(min_child_share):
                edges.append(row)

    out = {"summary": summary, "breakpoints": breakpoints, "nodes": nodes, "edges": edges}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_dict_rows_csv(out_dir / "hierarchy_nodes.csv", nodes)
        _write_dict_rows_csv(out_dir / "hierarchy_edges.csv", edges)
        (out_dir / "breakpoints.json").write_text(json.dumps(breakpoints, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_sweep_diagnostics(summary, out_png=out_dir / "sweep_diagnostics.png", breakpoints=breakpoints)
        if verbose:
            print(f"[hierarchy] saved hierarchy -> {out_dir}")
    return out


def run_hierarchy_sweep(
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
    reuse_existing: bool = True,
    resolution_mode: str = "linear",
    partition_type: str = "RBConfigurationVertexPartition",
    min_child_share: float = 0.25,
    verbose: bool = True,
) -> Dict[str, Any]:
    results = leiden_sweep(
        G,
        out_dir=out_dir,
        r_min=r_min,
        r_max=r_max,
        step=step,
        include=include,
        seed=seed,
        weights_attr=weights_attr,
        save_each_membership=save_each_membership,
        reuse_existing=reuse_existing,
        resolution_mode=resolution_mode,
        partition_type=partition_type,
        verbose=verbose,
    )
    hierarchy = build_hierarchy_from_sweep(
        results,
        out_dir=Path(out_dir),
        min_child_share=min_child_share,
        verbose=verbose,
    )
    return {"results": results, "hierarchy": hierarchy}


def induced_subgraph_edge_list(
    n_nodes: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    vertex_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    由全局 mutual-kNN 边表 (u,v,w) 取出 **顶点集上的诱导子图** 边表。

    - vertex_indices: 全局顶点编号（与 embedding 行 0..n_nodes-1 一致）
    - 返回的 u_loc,v_loc 为子图内 0..n_sub-1 编号；global_sorted 为排序后的全局下标，
      满足 global_sorted[local_id] == 全局编号。
    """
    verts = np.unique(np.asarray(vertex_indices, dtype=np.int64))
    if verts.size == 0:
        raise ValueError("vertex_indices is empty")
    if int(verts.min()) < 0 or int(verts.max()) >= int(n_nodes):
        raise ValueError(f"vertex_indices out of range [0, {n_nodes})")

    pos = np.full(int(n_nodes), -1, dtype=np.int32)
    pos[verts] = np.arange(verts.size, dtype=np.int32)

    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    mask = (pos[u] >= 0) & (pos[v] >= 0)
    if not np.any(mask):
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            verts.astype(np.int32),
        )

    uu = pos[u[mask]]
    vv = pos[v[mask]]
    ww = w[mask]
    return uu.astype(np.int32), vv.astype(np.int32), ww, verts.astype(np.int32)


def rank_communities_by_size(membership: np.ndarray, *, top_k: int = 30) -> List[Tuple[int, int]]:
    """返回 [(community_id, size), ...] 按 size 降序，最多 top_k 条。"""
    membership = np.asarray(membership, dtype=np.int64)
    uniq, cnt = np.unique(membership, return_counts=True)
    order = np.argsort(-cnt)
    out: List[Tuple[int, int]] = []
    for i in order[: max(int(top_k), 0)]:
        out.append((int(uniq[int(i)]), int(cnt[int(i)])))
    return out


def nearest_resolution_in_summary(leiden_dir: Path, r_target: float) -> float:
    """从已有 sweep 的 summary.npy 中取与 r_target 最接近的分辨率。"""
    leiden_dir = Path(leiden_dir)
    summary_path = leiden_dir / "summary.npy"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.npy not found under {leiden_dir}")
    summary = np.load(summary_path, allow_pickle=True).item()
    resolutions = [float(x) for x in np.asarray(summary["resolutions"]).tolist()]
    if not resolutions:
        raise ValueError(f"empty resolutions in {summary_path}")
    return float(min(resolutions, key=lambda x: abs(float(x) - float(r_target))))
