// src/pages/HomePage.tsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Papa from "papaparse";

type EdgeRow = Record<string, any>;
type LayoutRow = Record<string, any>;

async function countCSVRows(path: string): Promise<number> {
  try {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) return 0;
    const text = await r.text();
    const p = Papa.parse(text, { header: true, skipEmptyLines: true });
    return (p.data || []).length;
  } catch {
    return 0;
  }
}

export default function HomePage() {
  const [nodeCount, setNodeCount] = useState<number>(0);
  const [edgeCount, setEdgeCount] = useState<number>(0);

  useEffect(() => {
    let canceled = false;
    (async () => {
      const [n, e] = await Promise.all([
        countCSVRows("/data/layout.csv"),
        countCSVRows("/data/edges.csv"),
      ]);
      if (!canceled) {
        setNodeCount(n);
        setEdgeCount(e);
      }
    })();
    return () => { canceled = true; };
  }, []);

  return (
    <div className="home">

      {/* 英雄区 */}
      <section className="hero-tech">
        {/* 背景装饰层 */}
        <div className="hero-bg-grid" />
        <div className="hero-glow hero-glow-1" />
        <div className="hero-glow hero-glow-2" />
        <div className="hero-glow hero-glow-3" />

        <div className="hero-inner">
          <div className="hero-badge">arXiv · Graph Explorer</div>
          <h1 className="hero-title">
            <span className="hero-title-line">探索论文社区</span>
            <span className="hero-title-grad">看见结构 · 发现关系</span>
          </h1>
          <p className="hero-sub">
            本地 CSV 即开即用 · 交互网络与列表筛选 · 按社区/四类快速着色 · 点击节点直达详情。
          </p>

          <div className="hero-ctas">
            <Link className="btn btn-primary" to="/graph">开始探索网络</Link>
            <Link className="btn btn-ghost" to="/list">浏览论文列表</Link>
          </div>

          <div className="hero-stats">
            <div className="stat">
              <div className="stat__num">{nodeCount.toLocaleString()}</div>
              <div className="stat__label">节点（论文）</div>
            </div>
            <div className="stat">
              <div className="stat__num">{edgeCount.toLocaleString()}</div>
              <div className="stat__label">边（相似/关联）</div>
            </div>
            <div className="stat">
              <div className="stat__num">CSV</div>
              <div className="stat__label">无需后端 · 纯前端</div>
            </div>
          </div>

          <div className="hero-note">
            支持 CSV 文件直接导入，存储于 <code>public/data/</code> 。
          </div>
        </div>
      </section>

    </div>
  );
}