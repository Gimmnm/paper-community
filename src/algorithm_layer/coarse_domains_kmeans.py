from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coarse domains via k-means on embeddings; save indices per domain")
    p.add_argument("--emb-npy", type=str, default="data/paper_embeddings_specter2.npy", help="embedding npy (with leading dummy row if present)")
    p.add_argument("--umap2d-npy", type=str, default="out/umap2d.npy", help="2D coords for plotting (optional)")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="out/coarse_domains_kmeans")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def load_X(emb_path: Path) -> np.ndarray:
    embs = np.load(emb_path, mmap_mode="r")
    # repo convention: embs[0] is dummy; real papers are embs[1:]
    X = np.asarray(embs[1:], dtype=np.float32)
    return X


def kmeans_labels(X: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=int(seed),
        batch_size=4096,
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.01,
        verbose=0,
    )
    return km.fit_predict(X).astype(np.int32)


def maybe_plot(Y: np.ndarray, labels: np.ndarray, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.scatter(Y[:, 0], Y[:, 1], s=1.2, c=labels, cmap="tab10", alpha=0.65, linewidths=0)
    ax.set_title("UMAP(2D) colored by coarse k-means domains")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = load_X(Path(args.emb_npy))
    labels = kmeans_labels(X, k=int(args.k), seed=int(args.seed))

    counts = {int(i): int((labels == i).sum()) for i in range(int(args.k))}
    meta: Dict[str, object] = {
        "method": "MiniBatchKMeans",
        "k": int(args.k),
        "seed": int(args.seed),
        "emb_npy": str(Path(args.emb_npy).resolve()),
        "n_papers": int(X.shape[0]),
        "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    np.save(out_dir / "labels.npy", labels)
    for i in range(int(args.k)):
        idx = np.where(labels == i)[0].astype(np.int32)
        np.save(out_dir / f"domain_{i}_vertex_indices.npy", idx)

    if args.plot:
        Y = np.load(Path(args.umap2d_npy))
        Y = np.asarray(Y, dtype=np.float32)
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"umap2d rows {Y.shape[0]} != embeddings rows {X.shape[0]}; ensure umap2d is built from embs[1:]")
        maybe_plot(Y, labels, out_dir / f"umap_kmeans_k{int(args.k)}.png")

    print("[coarse-domains] wrote:", out_dir)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

