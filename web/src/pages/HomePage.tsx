import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";

type Stat = { nodes: number; edges: number; communities: number };

export default function HomePage() {
  const [s, setS] = useState<Stat>({ nodes: 0, edges: 0, communities: 0 });
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        // 粗略统计：layout.csv 行数 ≈ 节点数；edges.csv 行数 ≈ 边数；communities.csv 行数 ≈ 记录数
        const [l, e, c] = await Promise.all([
          fetch("/data/layout.csv").then(r => r.ok ? r.text() : ""),
          fetch("/data/edges.csv").then(r => r.ok ? r.text() : ""),
          fetch("/data/communities.csv").then(r => r.ok ? r.text() : ""),
        ]);
        const lc = l ? l.trim().split("\n").length - 1 : 0; // 去表头
        const ec = e ? e.trim().split("\n").length - 1 : 0;
        const cc = c ? c.trim().split("\n").length - 1 : 0;
        setS({ nodes: Math.max(0, lc), edges: Math.max(0, ec), communities: Math.max(0, cc) });
      } catch (er: any) {
        setErr(er?.message || String(er));
      }
    })();
  }, []);

  return (
    <div className="home-hero">
      <div className="home-hero__inner">
        <h1>Paper Community</h1>
        <p className="lead">
          从论文嵌入构建图，进行社区发现与可视化浏览。支持 Louvain / Leiden / Infomap，
          并提供 CSV 导出与前端交互探索。
        </p>

        <div className="stats">
          <div className="stat">
            <div className="stat__num">{s.nodes.toLocaleString()}</div>
            <div className="stat__label">节点（论文）</div>
          </div>
          <div className="stat">
            <div className="stat__num">{s.edges.toLocaleString()}</div>
            <div className="stat__label">边（相似度）</div>
          </div>
          <div className="stat">
            <div className="stat__num">{s.communities.toLocaleString()}</div>
            <div className="stat__label">社区映射</div>
          </div>
        </div>

        {!!err && <div className="error-box" style={{maxWidth:820}}>{err}</div>}

        <div className="cta">
          <Link to="/graph" className="btn btn-primary">进入网络可视化</Link>
          <Link to="/list" className="btn">浏览论文列表</Link>
          <a href="https://arxiv.org/" target="_blank" rel="noreferrer" className="btn btn-ghost">了解数据来源</a>
        </div>

        <div className="tips">
          <div>· 把 <code>data/layout.csv</code>、<code>data/edges.csv</code>、<code>data/communities.csv</code> 放到 <code>web/public/data/</code></div>
          <div>· 默认“按社区着色”需要 <code>communities.csv</code>（至少含 <code>index,community</code> 两列）</div>
        </div>
      </div>
    </div>
  );
}