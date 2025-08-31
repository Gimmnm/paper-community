import React, { useEffect, useMemo, useState } from "react";
import GraphView, { GraphData } from "../components/GraphView";
import { loadAll } from "../data/loader";
import { useNavigate } from "react-router-dom";

export default function GraphPage() {
  const [loading, setLoading] = useState(true);
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] });
  const [layout, setLayout] = useState<Record<string, { x: number; y: number }>>({});
  const navigate = useNavigate();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const { nodes, edges, layout } = await loadAll("/data");
        if (!alive) return;
        // 把 index 当成前端的 node.id
        setGraph({
          nodes: nodes.map((n) => ({ id: String(n.index), title: n.title, community: n.community, paper_id: n.id })),
          edges: edges.map((e) => ({ source: String(e.source), target: String(e.target), weight: e.weight })),
        });
        setLayout(layout); // key 是 index
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const communityColors = useMemo<Record<number, string>>(
    () => ({}), // 可自定义颜色：{0:"#...",1:"#..."}
    [],
  );

  if (loading) {
    return <div className="container">加载中……</div>;
  }

  return (
    <div style={{ height: "calc(100% - 49px)" }}>
      <GraphView
        graphData={graph}
        layout={layout}
        communityColors={communityColors}
        onNodeClick={(id, attrs) => {
          // attrs.paper_id = arXiv id；id 是 index
          const pid = attrs.paper_id || attrs.id || id;
          navigate(`/paper/${encodeURIComponent(String(pid))}`);
        }}
        sigmaSettings={{ labelFont: "Inter, system-ui, -apple-system" }}
      />
    </div>
  );
}