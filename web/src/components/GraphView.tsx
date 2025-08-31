import React, { useEffect, useRef } from "react";
import Sigma from "sigma";
import Graph from "graphology";

export type LayoutMap = Record<string, { x: number; y: number }>;

export type NodeDatum = {
  id: string;
  label?: string;
  title?: string;
  community?: number;
  size?: number;
  color?: string;
  [k: string]: any;
};

export type EdgeDatum = {
  source: string;
  target: string;
  weight?: number;
  [k: string]: any;
};

export type GraphData = {
  nodes: NodeDatum[];
  edges: EdgeDatum[];
};

export type GraphViewProps = {
  graphData: GraphData;
  layout: LayoutMap;
  className?: string;
  style?: React.CSSProperties;
  communityColors?: Record<number, string>;
  onNodeClick?: (nodeId: string, attrs: Record<string, any>) => void;
  sigmaSettings?: Partial<ConstructorParameters<typeof Sigma>[2]>;
};

function colorFromCommunity(c?: number): string {
  if (typeof c !== "number" || Number.isNaN(c)) return "#8392a6";
  const hue = (c * 137.508) % 360;
  return `hsl(${hue}, 65%, 55%)`;
}

export default function GraphView({
  graphData,
  layout,
  className,
  style,
  communityColors,
  onNodeClick,
  sigmaSettings,
}: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<Sigma | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const g = new Graph({ multi: false, type: "undirected" });

    // nodes
    for (const n of graphData.nodes) {
      const id = String(n.id);
      const pos = layout[id] ?? { x: Math.random(), y: Math.random() };
      const c = typeof n.community === "number" ? n.community : undefined;
      g.addNode(id, {
        label: n.label ?? n.title ?? id,
        x: pos.x,
        y: pos.y,
        size: n.size ?? 2,
        color: n.color ?? (c !== undefined && communityColors?.[c]) ?? colorFromCommunity(c),
        community: c,
        ...n,
      });
    }

    // edges
    for (const e of graphData.edges) {
      if (!g.hasNode(e.source) || !g.hasNode(e.target)) continue;
      if (!g.hasEdge(e.source, e.target)) g.addEdge(e.source, e.target, { weight: e.weight ?? 1, ...e });
    }

    const renderer = new Sigma(
      g,
      containerRef.current,
      Object.assign(
        {
          renderLabels: true,
          labelDensity: 1,
          defaultNodeType: "circle",
          defaultEdgeType: "line",
          zIndex: true,
          minCameraRatio: 0.05,
          maxCameraRatio: 10,
        },
        sigmaSettings || {},
      ),
    );
    rendererRef.current = renderer;

    const handleClickNode = ({ node }: { node: string }) => {
      try {
        onNodeClick?.(node, g.getNodeAttributes(node));
      } catch {}
    };
    renderer.on("clickNode", handleClickNode);

    const ro = new ResizeObserver(() => renderer.refresh());
    ro.observe(containerRef.current);

    return () => {
      renderer.off("clickNode", handleClickNode);
      ro.disconnect();
      renderer.kill();
      rendererRef.current = null;
      g.clear();
    };
  }, [graphData, layout, communityColors, onNodeClick, sigmaSettings]);

  return <div ref={containerRef} className={className} style={{ width: "100%", height: "100%", ...(style || {}) }} />;
}