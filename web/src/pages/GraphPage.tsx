import React, { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { SigmaContainer, useSigma, useRegisterEvents } from "@react-sigma/core";
import Graph from "graphology";
import { useNavigate } from "react-router-dom";
// 顶部 import 区
import { colorForCommunity, colorForFour, colorForEdgeWeight } from "../lib/colors";

/** ---------- 类型 ---------- */
type LayoutRow = {
  node?: string | number;
  index: number | string;
  x: number | string;
  y: number | string;
  paper_id?: string;
  title?: string;
  community?: number | string;
  color?: string;
};
type EdgeRow = {
  source?: number | string;
  target?: number | string;
  weight?: number | string;
  u?: number | string;
  v?: number | string;
  sim?: number | string;
  w?: number | string;
};
type ComRow = { index?: number | string; community?: number | string };
type LabeledRow = { index?: number | string; four?: string };
type Stats = { totalNodes: number; usedNodes: number; totalEdges: number; usedEdges: number };
type LegendItem = { key: string; label: string; color: string; count: number };

/** ---------- 常量 & 工具 ---------- */
const LIMIT_INITIAL_NODES = 600;
const LIMIT_INITIAL_EDGES = 12000;
const ADD_BATCH = 2000;
type ColorMode = "community" | "four";

function useDebounced<T>(value: T, delay = 300) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}
async function fetchText(url: string, signal?: AbortSignal): Promise<string> {
  const r = await fetch(url, { cache: "no-store", signal });
  if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
  const t = await r.text();
  return t.replace(/^\uFEFF/, ""); // 去掉 BOM
}
function normalizeXY(rows: LayoutRow[]): LayoutRow[] {
  const xs = rows.map((r) => Number(r.x));
  const ys = rows.map((r) => Number(r.y));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const s = 2 / span;
  return rows.map((r) => ({ ...r, x: (Number(r.x) - cx) * s, y: (Number(r.y) - cy) * s }));
}
function resolveWeight(e: EdgeRow): number {
  const w = e.weight ?? e.w ?? e.sim;
  return w == null ? 1 : Number(w);
}
function resolveSource(e: EdgeRow): string {
  const s = e.source ?? e.u;
  return String(s);
}
function resolveTarget(e: EdgeRow): string {
  const t = e.target ?? e.v;
  return String(t);
}

/** communities.csv → index -> community（双路径） */
async function loadCommunityMap(signal?: AbortSignal): Promise<Record<number, number>> {
  const urls = ["/data/communities.csv", "/data/graph/communities.csv"];
  for (const url of urls) {
    try {
      const text = await fetchText(url, signal);
      const p = Papa.parse<ComRow>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
      const map: Record<number, number> = {};
      for (const row of (p.data || [])) {
        if (row && row.index != null && row.community != null) {
          const idx = Number(row.index);
          const c = Number(row.community);
          if (Number.isFinite(idx) && Number.isFinite(c)) map[idx] = c;
        }
      }
      const n = Object.keys(map).length;
      if (n) {
        console.log(`[graph] communities loaded from ${url}, size=${n}`);
        return map;
      }
    } catch {
      /* try next */
    }
  }
  console.warn("[graph] communities.csv not found or empty");
  return {};
}

/** nodes_labeled.csv → index -> four */
async function loadFourMap(signal?: AbortSignal): Promise<Record<number, string>> {
  try {
    const text = await fetchText("/data/nodes_labeled.csv", signal);
    const p = Papa.parse<LabeledRow>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
    const map: Record<number, string> = {};
    for (const row of (p.data || [])) {
      if (row && row.index != null && row.four != null) {
        const idx = Number(row.index);
        const v = String(row.four);
        if (Number.isFinite(idx) && v) map[idx] = v;
      }
    }
    console.log("[graph] nodes_labeled size =", Object.keys(map).length);
    return map;
  } catch {
    return {};
  }
}

function GraphEvents() {
  const sigma = useSigma();
  const register = useRegisterEvents();
  const navigate = useNavigate();

  useEffect(() => {
    register({
      clickNode: ({ node }) => {
        const g = sigma.getGraph();
        // 优先读 index 属性，兜底用 node key
        const idxAttr = g.getNodeAttribute(node, "index");
        const target = (idxAttr !== undefined && idxAttr !== null) ? String(idxAttr) : String(node);
        navigate(`/detail/${encodeURIComponent(target)}`);
      },
    });
  }, [register, sigma, navigate]);

  return null;
}

/** 悬停高亮邻居 */
function HoverHighlight({ enabled = true }: { enabled?: boolean }) {
  const sigma = useSigma();
  const register = useRegisterEvents();
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    register({
      enterNode: ({ node }) => enabled && setHovered(node as string),
      leaveNode: () => setHovered(null),
    });
  }, [register, enabled]);

  useEffect(() => {
    const g = sigma.getGraph();
    if (!enabled) {
      sigma.setSetting("nodeReducer", undefined as any);
      sigma.setSetting("edgeReducer", undefined as any);
      sigma.refresh();
      return;
    }
    const neighbors = new Set<string>();
    if (hovered) {
      g.forEachNeighbor(hovered, (n) => neighbors.add(String(n)));
      neighbors.add(hovered);
    }
    sigma.setSetting("nodeReducer", (n: string, data: any) => {
      if (!hovered) return data;
      if (neighbors.has(n)) return { ...data, zIndex: 1 };
      return { ...data, color: "#e5e7eb", label: undefined, zIndex: 0 };
    });
    sigma.setSetting("edgeReducer", (e: string, data: any) => {
      if (!hovered) return data;
      const [s, t] = sigma.getGraph().extremities(e) || [];
      if (s != null && t != null && neighbors.has(String(s)) && neighbors.has(String(t))) {
        return { ...data, color: data.color || "rgba(51,65,85,0.92)", size: Math.max(1.2, (data.size || 1)) };
      }
      return { ...data, color: "rgba(209,213,219,0.15)", size: Math.max(0.4, (data.size || 1) * 0.4) };
    });
    sigma.refresh();
  }, [sigma, hovered, enabled]);

  return null;
}

/** 边/节点尺寸实时缩放 */
function EdgeSizeUpdater({ edgeScale }: { edgeScale: number }) {
  const sigma = useSigma();
  useEffect(() => {
    const g = sigma.getGraph();
    g.forEachEdge((e) => {
      const base = Number(g.getEdgeAttribute(e, "size")) || 1;
      const baseStored = g.getEdgeAttribute(e, "sizeBase") ?? base;
      g.setEdgeAttribute(e, "sizeBase", baseStored);
      g.setEdgeAttribute(e, "size", Number(baseStored) * Math.max(0.2, edgeScale));
    });
    sigma.refresh();
  }, [edgeScale, sigma]);
  return null;
}
function NodeSizeUpdater({ sizeScale }: { sizeScale: number }) {
  const sigma = useSigma();
  useEffect(() => {
    const g = sigma.getGraph();
    g.forEachNode((n) => {
      const base = Number(g.getNodeAttribute(n, "size")) || 4;
      const baseStored = g.getNodeAttribute(n, "sizeBase") ?? base;
      g.setNodeAttribute(n, "sizeBase", baseStored);
      g.setNodeAttribute(n, "size", Number(baseStored) * Math.max(0.2, sizeScale));
    });
    sigma.refresh();
  }, [sizeScale, sigma]);
  return null;
}

/** 构图 */
function BuildGraph({
  nodeLimit,
  edgeLimit,
  minWeight,
  showEdges,
  hideEdgesOnMove,
  baseNodeSize,
  colorMode,
  onStats,
  onLegend,
  onCommunityLoad,
}: {
  nodeLimit: number;
  edgeLimit: number;
  minWeight: number;
  showEdges: boolean;
  hideEdgesOnMove: boolean;
  baseNodeSize: number;
  colorMode: ColorMode;
  onStats?: (s: Stats) => void;
  onLegend?: (items: LegendItem[]) => void;
  onCommunityLoad?: (count: number) => void;
}) {
  const sigma = useSigma();
  const [msg, setMsg] = useState<string | null>(null);
  const tokenRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  const dNodeLimit = useDebounced(nodeLimit, 200);
  const dEdgeLimit = useDebounced(edgeLimit, 200);
  const dMinWeight = useDebounced(minWeight, 200);
  const dShowEdges = useDebounced(showEdges, 0);
  const dColorMode = useDebounced(colorMode, 0);

  useEffect(() => {
    sigma.setSetting("renderEdges", dShowEdges);
    sigma.setSetting("hideEdgesOnMove", hideEdgesOnMove);
    sigma.refresh();
  }, [dShowEdges, hideEdgesOnMove, sigma]);

  useEffect(() => {
    let canceled = false;
    const myToken = ++tokenRef.current;

    // 终止上一轮 fetch
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    async function run() {
      const graph = sigma.getGraph() as Graph;
      graph.clear();

      try {
        const [comMap, fourMap] = await Promise.all([loadCommunityMap(signal), loadFourMap(signal)]);
        onCommunityLoad?.(Object.keys(comMap).length);

        setMsg("加载 /data/layout.csv ...");
        const t1 = await fetchText("/data/layout.csv", signal);
        const p1 = Papa.parse<LayoutRow>(t1, { header: true, dynamicTyping: true, skipEmptyLines: true });

        let rows = (p1.data || []).filter(r => r && r.index != null && r.x != null && r.y != null);
        if (rows.length === 0) throw new Error("layout.csv 无有效数据");
        rows = normalizeXY(rows);

        const ids = rows.map((r, i) => ({ i, idx: Number(r.index) })).sort((a, b) => a.idx - b.idx);
        const totalNodes = ids.length;
        const stride = Math.max(1, Math.ceil(totalNodes / dNodeLimit));
        const usedIdxs = ids.filter((_, i) => i % stride === 0).slice(0, dNodeLimit).map(x => x.i);

        const counterCommunity = new Map<number, number>();
        const counterFour = new Map<string, number>();

        console.debug("[graph] building nodes:", usedIdxs.length, "/", totalNodes);
        setMsg(`构建节点 ${usedIdxs.length}/${totalNodes} ...`);

        for (const ri of usedIdxs) {
          if (canceled || tokenRef.current !== myToken) return;
          const r = rows[ri];
          const key = String(r.index);
          const idx = Number(r.index);

          const cFromLayout = r.community != null ? Number(r.community) : null;
          const cFromMap = comMap[idx];
          const comm = Number.isFinite(cFromLayout as number) ? (cFromLayout as number)
                         : Number.isFinite(cFromMap) ? cFromMap : null;

          const four = fourMap[idx] || null;

          let col = (r.color ? String(r.color) : null);
          if (!col) {
            if (dColorMode === "four" && four) col = colorForFour(four);
            else col = colorForCommunity(comm);
          }
          if (comm != null) counterCommunity.set(comm, (counterCommunity.get(comm) || 0) + 1);
          if (four) counterFour.set(four, (counterFour.get(four) || 0) + 1);

          if (!graph.hasNode(key)) {
            graph.addNode(key, {
              x: Number(r.x),
              y: Number(r.y),
              label: r.title ? String(r.title) : key,
              size: baseNodeSize,
              color: col,
              paper_id: r.paper_id ?? null,
              title: r.title ?? null,
              community: comm,
              four,
              index: idx,
            });
          }
        }

        if (onLegend) {
          let items: LegendItem[] = [];
          if (dColorMode === "four" && counterFour.size > 0) {
            items = Array.from(counterFour.entries())
              .map(([k, cnt]) => ({ key: String(k), label: String(k), color: colorForFour(String(k)), count: cnt }))
              .sort((a, b) => b.count - a.count)
              .slice(0, 16);
          } else {
            items = Array.from(counterCommunity.entries())
              .map(([k, cnt]) => ({ key: `C${k}`, label: `C${k}`, color: colorForCommunity(k), count: cnt }))
              .sort((a, b) => b.count - a.count)
              .slice(0, 16);
          }
          onLegend(items);
        }

        sigma.getCamera().setState({ x: 0, y: 0, ratio: 1 });
        sigma.refresh();

        let usedEdges = 0;
        let totalEdges = 0;

        if (dShowEdges) {
          setMsg("加载 /data/edges.csv ...");
          const t2 = await fetchText("/data/edges.csv", signal);
          const p2 = Papa.parse<EdgeRow>(t2, { header: true, dynamicTyping: true, skipEmptyLines: true });

          let es = (p2.data || []).filter(Boolean) as EdgeRow[];
          totalEdges = es.length;

          es = es
            .map(e => ({ s: resolveSource(e), t: resolveTarget(e), w: resolveWeight(e) }))
            .filter(e => Number.isFinite(e.w) && e.w >= dMinWeight);

          es = es.filter(e => graph.hasNode(String(e.s)) && graph.hasNode(String(e.t)));

          es.sort((a, b) => b.w - a.w);
          if (dEdgeLimit > 0 && es.length > dEdgeLimit) es = es.slice(0, dEdgeLimit);

          console.debug("[graph] writing edges:", es.length);
          let added = 0;
          for (let i = 0; i < es.length; i += ADD_BATCH) {
            if (canceled || tokenRef.current !== myToken) return;
            const batch = es.slice(i, i + ADD_BATCH);
            setMsg(`写入边 ${i + 1} ~ ${i + batch.length} / ${es.length} ...`);
            for (const e of batch) {
              const s = String(e.s);
              const t = String(e.t);
              const a = s < t ? s : t;   // 无向边去重
              const b = s < t ? t : s;
              if (!graph.hasEdge(a, b)) {
                const edgeSize = 0.8 + Math.max(0, Math.min(1, e.w)) * 2.4;
                graph.addEdge(a, b, {
                  weight: e.w,
                  size: edgeSize,
                  color: colorForEdgeWeight(e.w),
                });
                added++;
              }
            }
            sigma.refresh();
            await new Promise(r => setTimeout(r, 0));
          }
          usedEdges = added;
        }

        onStats?.({ totalNodes, usedNodes: graph.order, totalEdges, usedEdges });
        if (Object.keys(comMap).length === 0 && dColorMode === "community") {
          setMsg("提示：未检测到 communities.csv（放到 web/public/data 或 web/public/data/graph）");
          setTimeout(() => setMsg(null), 2400);
        } else {
          setMsg(null);
        }
      } catch (e: any) {
        if (e?.name === "AbortError") return; // 被中断，忽略
        setMsg(`加载/构图失败：${e?.message ?? String(e)}`);
        console.error(e);
      }
    }

    run();

    return () => {
      canceled = true;
      abortRef.current?.abort();
    };
  }, [
    sigma,
    dNodeLimit,
    dEdgeLimit,
    dMinWeight,
    dShowEdges,
    hideEdgesOnMove,
    baseNodeSize,
    dColorMode,
    onStats,
    onLegend,
    onCommunityLoad,
  ]);

  return msg ? <div className="toast" style={{ zIndex: 20 }}>{msg}</div> : null;
}

/** ---------- 页面 ---------- */
export default function GraphPage() {
  const [nodeLimit, setNodeLimit] = useState<number>(LIMIT_INITIAL_NODES);
  const [edgeLimit, setEdgeLimit] = useState<number>(LIMIT_INITIAL_EDGES);
  const [minWeight, setMinWeight] = useState<number>(0.35);
  const [showEdges, setShowEdges] = useState<boolean>(false); // 默认不画边，避免卡住
  const [hideEdgesOnMove, setHideEdgesOnMove] = useState<boolean>(true);
  const [stats, setStats] = useState<Stats | null>(null);

  const [nodeScale, setNodeScale] = useState<number>(1.8);
  const [edgeScale, setEdgeScale] = useState<number>(1.2);
  const baseNodeSize = 7.0;

  const [colorMode, setColorMode] = useState<ColorMode>("community");
  const [legend, setLegend] = useState<LegendItem[]>([]);
  const [hoverOn, setHoverOn] = useState<boolean>(true);
  const [commCount, setCommCount] = useState<number>(0);

  const settings = useMemo(
    () => ({
      labelRenderedSizeThreshold: 10,
      labelDensity: 0.08,
      zIndex: true,
      defaultNodeType: "circle",
      defaultNodeColor: "#111827",
      defaultNodeSize: baseNodeSize,
      defaultEdgeColor: "rgba(51,65,85,0.92)",
      edgeColor: "default" as const,
      minEdgeSize: 0.6,
      maxEdgeSize: 3.6,
      renderEdges: showEdges,
      hideEdgesOnMove: hideEdgesOnMove,
    }),
    [showEdges, hideEdgesOnMove]
  );

  return (
    <div className="graph-wrap">
      <div className="toolbar soft-card" style={{ flexWrap: "wrap", gap: 10 }}>
        <div className="badge">节点上限: {nodeLimit}</div>
        <input type="range" min={50} max={8000} step={50} value={nodeLimit}
               onChange={(e) => setNodeLimit(parseInt(e.target.value, 10))} style={{ width: 220 }} />
        <button className="chip" onClick={() => setNodeLimit(300)}>300</button>
        <button className="chip" onClick={() => setNodeLimit(1000)}>1000</button>
        <button className="chip" onClick={() => setNodeLimit(3000)}>3000</button>

        <span className="sep" />

        <label className="badge">
          <input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} />
          显示边
        </label>

        <div className="badge">边上限: {edgeLimit}</div>
        <input type="range" min={0} max={200000} step={2000} value={edgeLimit}
               onChange={(e) => setEdgeLimit(parseInt(e.target.value, 10))} style={{ width: 240 }} />

        <div className="badge">最小权重: {minWeight.toFixed(2)}</div>
        <input type="range" min={0} max={1} step={0.01} value={minWeight}
               onChange={(e) => setMinWeight(parseFloat(e.target.value))} style={{ width: 160 }} />

        <label className="badge">
          <input type="checkbox" checked={hideEdgesOnMove} onChange={(e) => setHideEdgesOnMove(e.target.checked)} />
          移动时隐藏边
        </label>

        <span className="sep" />

        <div className="badge">节点大小 × {nodeScale.toFixed(1)}</div>
        <input type="range" min={0.1} max={3.0} step={0.1} value={nodeScale}
               onChange={(e) => setNodeScale(parseFloat(e.target.value))} style={{ width: 160 }} />

        <div className="badge">边粗细 × {edgeScale.toFixed(1)}</div>
        <input type="range" min={0.1} max={3.0} step={0.1} value={edgeScale}
               onChange={(e) => setEdgeScale(parseFloat(e.target.value))} style={{ width: 160 }} />

        <span className="sep" />

        <div className="badge">
          颜色：
          <select value={colorMode} onChange={(e) => setColorMode(e.target.value as ColorMode)}>
            <option value="community">按社区</option>
            <option value="four">按四类</option>
          </select>
        </div>

        <label className="badge">
          <input type="checkbox" checked={hoverOn} onChange={(e) => setHoverOn(e.target.checked)} />
          悬停高亮邻居
        </label>

        {commCount === 0 && colorMode === "community" && (
          <div className="badge" style={{ background: "#fef3c7", borderColor: "#fde68a" }}>
            未检测到 communities.csv（放到 web/public/data 或 web/public/data/graph）
          </div>
        )}

        {stats && (
          <div className="badge">
            N {stats.usedNodes}/{stats.totalNodes} | E {stats.usedEdges}/{stats.totalEdges}
          </div>
        )}

        {legend?.length > 0 && (
          <div className="legend">
            {legend.map((it) => (
              <div key={it.key} className="legend-item">
                <span className="legend-dot" style={{ background: it.color }} />
                <span className="legend-text">{it.label}</span>
                <span className="legend-count">{it.count}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <SigmaContainer className="sigma-container" settings={settings}>
        <BuildGraph
          nodeLimit={nodeLimit}
          edgeLimit={edgeLimit}
          minWeight={minWeight}
          showEdges={showEdges}
          hideEdgesOnMove={hideEdgesOnMove}
          onStats={setStats}
          baseNodeSize={baseNodeSize}
          colorMode={colorMode}
          onLegend={setLegend}
          onCommunityLoad={setCommCount}
        />
        <NodeSizeUpdater sizeScale={nodeScale} />
        <EdgeSizeUpdater edgeScale={edgeScale} />
        <HoverHighlight enabled={hoverOn} />
        <GraphEvents />
      </SigmaContainer>
    </div>
  );
}