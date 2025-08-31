#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, pandas as pd
from sqlalchemy.orm import Session
from backend.db import SessionLocal, Base, engine
from backend.models import Paper, Edge, Community

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="data/graph/nodes.csv")
    ap.add_argument("--edges", default="data/graph/edges.csv")
    ap.add_argument("--communities", default="data/graph/communities.csv")
    ap.add_argument("--layout", default="data/graph/layout.csv")
    ap.add_argument("--names", default="data/graph/graph.json")
    args = ap.parse_args()

    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    nodes = pd.read_csv(args.nodes).fillna("")
    layout = pd.read_csv(args.layout) if args.layout else pd.DataFrame()
    pos = {int(r.node):(float(r.x), float(r.y)) for _,r in layout.iterrows()} if len(layout)>0 else {}
    comm = pd.read_csv(args.communities)
    with open(args.names, "r", encoding="utf-8") as f:
        names = { int(c["id"]): c["name"] for c in json.load(f)["communities"] }

    # communities
    size_map = comm.groupby("community").size().to_dict()
    for cid, size in size_map.items():
        db.merge(Community(id=int(cid), name=names.get(int(cid), f"Community {cid}"), size=int(size)))
    db.commit()

    # papers
    cid_map = {int(r.node): int(r.community) for _,r in comm.iterrows()}
    for i,row in nodes.iterrows():
        pid = str(row["id"]); x,y = pos.get(i, (None,None))
        db.merge(Paper(id=pid,
                       title=str(row.get("title","")),
                       abstract=str(row.get("abstract","")) if "abstract" in row else None,
                       authors=str(row.get("authors","")),
                       year=int(row.get("year")) if "year" in row and str(row["year"]).isdigit() else None,
                       arxiv_id=str(row.get("arxiv_id","")),
                       categories=str(row.get("categories","")) if "categories" in row else None,
                       community_id=cid_map.get(i, None), x=x, y=y))
    db.commit()

    # edges
    edges = pd.read_csv(args.edges)
    for _,e in edges.iterrows():
        db.add(Edge(source=str(int(e["source"])), target=str(int(e["target"])), weight=float(e["weight"])))
    db.commit(); db.close()
    print("[ok] database loaded")

if __name__ == "__main__":
    main()