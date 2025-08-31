from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .loaders import load_nodes, load_edges, load_layout, build_joined_nodes, graph_summary

app = FastAPI(title="paper-community API", version="0.1.0")

# CORS（前端 dev 服务默认 http://localhost:5173）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 需要可收紧
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/graph/summary")
def api_summary():
    return graph_summary()

@app.get("/api/graph/nodes")
def api_nodes():
    df = build_joined_nodes()
    # 精简字段给前端
    cols = [c for c in ["index","paper_id","title","community","x","y"] if c in df.columns]
    return df[cols].to_dict(orient="records")

@app.get("/api/graph/edges")
def api_edges():
    df = load_edges()
    return df[["src","dst","weight"]].to_dict(orient="records")

@app.get("/api/healthz")
def health():
    return {"ok": True}