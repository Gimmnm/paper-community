from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class NodeKey:
    r: float
    c: int

    @property
    def node_id(self) -> str:
        return f"r={self.r:.4f}|c={self.c}"


def _read_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean: Dict[str, str] = {}
            for k, v in row.items():
                kk = (k or "").lstrip("\ufeff").strip()
                clean[kk] = v if v is not None else ""
            yield clean


def _try_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _try_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _node_id(r: float, c: int) -> str:
    return f"r={float(r):.4f}|c={int(c)}"


def iter_nodes(nodes_csv: Path) -> Iterator[Tuple[str, float, int, float]]:
    """
    Yields (node_id, r, c, size).
    Expected columns: resolution, community, size.
    """
    for row in _read_csv_rows(nodes_csv):
        r = round(_try_float(row.get("resolution") or row.get("r")), 4)
        c = _try_int(row.get("community") or row.get("community_id") or row.get("cid"))
        if c < 0:
            continue
        size = _try_float(row.get("size"), 1.0)
        if size != size:  # nan
            size = 1.0
        yield _node_id(r, c), float(r), int(c), float(size)


def iter_edges(edges_csv: Path) -> Iterator[Dict[str, Any]]:
    """
    Expected columns: r_parent, community_parent, r_child, community_child, child_share, jaccard, intersection, parent_share.
    """
    for row in _read_csv_rows(edges_csv):
        rp = round(_try_float(row.get("r_parent")), 4)
        rc = round(_try_float(row.get("r_child")), 4)
        cp = _try_int(row.get("community_parent"))
        cc = _try_int(row.get("community_child"))
        if cp < 0 or cc < 0:
            continue
        out = {
            "parent_id": _node_id(rp, cp),
            "child_id": _node_id(rc, cc),
            "r_parent": float(rp),
            "c_parent": int(cp),
            "r_child": float(rc),
            "c_child": int(cc),
            "intersection": _try_int(row.get("intersection"), 0),
            "parent_share": float(_try_float(row.get("parent_share"), 0.0) or 0.0),
            "child_share": float(_try_float(row.get("child_share"), 0.0) or 0.0),
            "jaccard": float(_try_float(row.get("jaccard"), 0.0) or 0.0),
        }
        yield out


def build_sqlite_index(*, hierarchy_dir: Path, db_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Build a queryable SQLite index for a hierarchy directory containing:
      - hierarchy_nodes.csv
      - hierarchy_edges.csv
      - breakpoints.json (optional)
      - hierarchy_viz_overview.json (optional)

    The DB stores:
      - nodes(node_id PRIMARY KEY, r, c, size)
      - edges(parent_id, child_id, r_parent, c_parent, r_child, c_child, child_share, jaccard, intersection, parent_share)
      - meta(key PRIMARY KEY, value_json)
    """
    hierarchy_dir = Path(hierarchy_dir)
    nodes_csv = hierarchy_dir / "hierarchy_nodes.csv"
    edges_csv = hierarchy_dir / "hierarchy_edges.csv"
    if not nodes_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError(f"missing hierarchy_nodes.csv / hierarchy_edges.csv under {hierarchy_dir}")

    if db_path is None:
        db_path = hierarchy_dir / "hierarchy_index.sqlite"
    db_path = Path(db_path)
    if db_path.exists():
        if not overwrite:
            return db_path
        db_path.unlink()

    t0 = time.time()
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")  # ~200MB

        cur.execute(
            """
            CREATE TABLE nodes(
              node_id TEXT PRIMARY KEY,
              r REAL NOT NULL,
              c INTEGER NOT NULL,
              size REAL NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE edges(
              parent_id TEXT NOT NULL,
              child_id TEXT NOT NULL,
              r_parent REAL NOT NULL,
              c_parent INTEGER NOT NULL,
              r_child REAL NOT NULL,
              c_child INTEGER NOT NULL,
              child_share REAL NOT NULL,
              jaccard REAL NOT NULL,
              intersection INTEGER NOT NULL,
              parent_share REAL NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX idx_edges_parent ON edges(parent_id);")
        cur.execute("CREATE INDEX idx_edges_rparent ON edges(r_parent);")
        cur.execute("CREATE INDEX idx_edges_child ON edges(child_id);")
        cur.execute(
            """
            CREATE TABLE meta(
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL
            );
            """
        )
        con.commit()

        # nodes
        n_nodes = 0
        batch: List[Tuple[str, float, int, float]] = []
        for node in iter_nodes(nodes_csv):
            batch.append(node)
            if len(batch) >= 5000:
                cur.executemany("INSERT INTO nodes(node_id, r, c, size) VALUES (?,?,?,?)", batch)
                n_nodes += len(batch)
                batch.clear()
        if batch:
            cur.executemany("INSERT INTO nodes(node_id, r, c, size) VALUES (?,?,?,?)", batch)
            n_nodes += len(batch)
            batch.clear()
        con.commit()

        # edges
        n_edges = 0
        e_batch: List[Tuple[Any, ...]] = []
        for e in iter_edges(edges_csv):
            e_batch.append(
                (
                    e["parent_id"],
                    e["child_id"],
                    e["r_parent"],
                    e["c_parent"],
                    e["r_child"],
                    e["c_child"],
                    e["child_share"],
                    e["jaccard"],
                    e["intersection"],
                    e["parent_share"],
                )
            )
            if len(e_batch) >= 5000:
                cur.executemany(
                    """
                    INSERT INTO edges(parent_id, child_id, r_parent, c_parent, r_child, c_child, child_share, jaccard, intersection, parent_share)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    e_batch,
                )
                n_edges += len(e_batch)
                e_batch.clear()
        if e_batch:
            cur.executemany(
                """
                INSERT INTO edges(parent_id, child_id, r_parent, c_parent, r_child, c_child, child_share, jaccard, intersection, parent_share)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                e_batch,
            )
            n_edges += len(e_batch)
            e_batch.clear()
        con.commit()

        # meta
        meta: Dict[str, Any] = {
            "hierarchy_dir": str(hierarchy_dir),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_nodes": int(n_nodes),
            "n_edges": int(n_edges),
        }
        for name in ["breakpoints.json", "hierarchy_viz_overview.json"]:
            p = hierarchy_dir / name
            if p.exists():
                try:
                    meta[name] = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass
        cur.execute("INSERT INTO meta(key, value_json) VALUES (?,?)", ("meta", json.dumps(meta, ensure_ascii=False)))
        con.commit()
    finally:
        con.close()

    _ = time.time() - t0
    return db_path


def query_roots(con: sqlite3.Connection, *, root_resolution: float = 0.001, top_k: int = 50) -> List[Dict[str, Any]]:
    r0 = round(float(root_resolution), 4)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT node_id, c, size FROM nodes WHERE ABS(r - ?) < 5e-5 ORDER BY size DESC LIMIT ?",
        (float(r0), int(top_k)),
    ).fetchall()
    return [{"node_id": nid, "community": int(c), "resolution": float(r0), "size": float(size)} for (nid, c, size) in rows]


def bfs_lineage(
    con: sqlite3.Connection,
    *,
    root_node_id: str,
    r_max: float,
    min_child_share: float = 0.15,
    max_nodes: int = 5000,
) -> Dict[str, Any]:
    """
    BFS using sqlite parent->children lookups.
    """
    r_max = float(r_max)
    min_child_share = float(min_child_share)
    cur = con.cursor()
    seen = set()
    q = [root_node_id]
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def _get_r(nid: str) -> float:
        try:
            # nid format: r=0.0010|c=37
            s = nid.split("|", 1)[0]
            return float(s.split("=", 1)[1])
        except Exception:
            return float("nan")

    while q and len(seen) < int(max_nodes):
        nid = q.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        rr = _get_r(nid)
        if rr != rr or rr > r_max + 1e-12:
            continue
        # node attrs
        row = cur.execute("SELECT r, c, size FROM nodes WHERE node_id = ?", (nid,)).fetchone()
        if row:
            r, c, size = row
            nodes[nid] = {"node_id": nid, "resolution": float(r), "community": int(c), "size": float(size)}
        # children
        child_rows = cur.execute(
            """
            SELECT child_id, r_child, c_child, child_share, jaccard, intersection, parent_share
            FROM edges
            WHERE parent_id = ? AND child_share >= ? AND r_child <= ?
            ORDER BY child_share DESC
            """,
            (nid, min_child_share, r_max),
        ).fetchall()
        for child_id, r_child, c_child, child_share, jaccard, inter, parent_share in child_rows:
            edges.append(
                {
                    "source": nid,
                    "target": child_id,
                    "child_share": float(child_share),
                    "jaccard": float(jaccard),
                    "intersection": int(inter),
                    "parent_share": float(parent_share),
                }
            )
            if child_id not in seen:
                q.append(child_id)

    return {"root": root_node_id, "nodes": list(nodes.values()), "edges": edges, "meta": {"r_max": r_max, "min_child_share": min_child_share}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy-dir", type=str, required=True)
    ap.add_argument("--db-path", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    db = build_sqlite_index(
        hierarchy_dir=Path(args.hierarchy_dir),
        db_path=Path(args.db_path) if args.db_path else None,
        overwrite=bool(args.overwrite),
    )
    print(str(db))


if __name__ == "__main__":  # pragma: no cover
    main()

