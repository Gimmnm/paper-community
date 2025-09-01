import React, { useEffect, useMemo, useRef, useState } from "react";
import Papa from "papaparse";
import { SigmaContainer, useSigma, useRegisterEvents } from "@react-sigma/core";
import Graph from "graphology";
import { useNavigate } from "react-router-dom";
import { colorForCommunity, colorForFour } from "../lib/colors";


function useThemeName() {
  const getTheme = () => document.documentElement.dataset.theme || "light";
  const [theme, setTheme] = React.useState<string>(getTheme());

  React.useEffect(() => {
    const obs = new MutationObserver(() => setTheme(getTheme()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  return theme as "light" | "ocean" | "midnight";
}

/** ---------- 类型 ---------- */
type LayoutRow = {
  node?: string | number;
  index: number | string;
  x: number | string;
  y: number | string;
  paper_id?: string;
  title?: string;
  community?: number | string;
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

type ColorMode = "community" | "four";

/** ---------- 常量 ---------- */
const LIMIT_INITIAL_NODES = 800;       // 初始节点数（可调）
const ADD_BATCH = 2000;

/** ---------- 小工具 ---------- */
function useDebounced<T>(value: T, delay = 200) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

function colorForEdgeWeight(w: number): string {
  const ww = Math.max(0, Math.min(1, w));
  const alpha = 0.65 + ww * 0.25; // 0.65 ~ 0.90
  return `rgba(55, 65, 81, ${alpha.toFixed(2)})`; // 深灰蓝
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

/** 读取 CSV -> Map */
async function loadCommunityMap(): Promise<Record<number, number>> {
  try {
    const r = await fetch("/data/communities.csv", { cache: "no-store" });
    if (!r.ok) return {};
    const text = await r.text();
    const p = Papa.parse<ComRow>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
    const map: Record<number, number> = {};
    for (const row of (p.data || [])) {
      if (row && row.index != null && row.community != null) {
        const idx = Number(row.index);
        const c = Number(row.community);
        if (Number.isFinite(idx) && Number.isFinite(c)) map[idx] = c;
      }
    }
    return map;
  } catch { return {}; }
}
async function loadFourMap(): Promise<Record<number, string>> {
  try {
    const r = await fetch("/data/nodes_labeled.csv", { cache: "no-store" });
    if (!r.ok) return {};
    const text = await r.text();
    const p = Papa.parse<LabeledRow>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
    const map: Record<number, string> = {};
    for (const row of (p.data || [])) {
      if (row && row.index != null && row.four != null) {
        const idx = Number(row.index);
        const v = String(row.four);
        if (Number.isFinite(idx) && v) map[idx] = v;
      }
    }
    return map;
  } catch { return {}; }
}

/** 图事件：点击节点 → 详情页 */
function GraphEvents() {
  const sigma = useSigma();
  const register = useRegisterEvents();
  const navigate = useNavigate();
  useEffect(() => {
    register({
      clickNode: ({ node }) => {
        const g = sigma.getGraph();
        const idx = g.getNodeAttribute(node, "index");
        if (idx != null) navigate(`/detail/${idx}`);
      },
    });
  }, [register, sigma, navigate]);
  return null;
}
function HoverHighlight({ enabled = true }: { enabled?: boolean }) {
  const sigma = useSigma();
  const register = useRegisterEvents();
  const theme = useThemeName();                // ← 读主题
  const [hovered, setHovered] = useState<string | null>(null);

  // 按主题确定淡出/高亮颜色
  const fadedNodeColor = theme === "midnight" ? "#334155" : "#e5e7eb";                  // 深色用 slate-700，浅色用浅灰
  const fadedEdgeColor = theme === "midnight" ? "rgba(100,116,139,0.18)" : "rgba(209,213,219,0.15)";
  const neighborEdgeBoost = theme === "midnight" ? 1.15 : 1.25;                         // 深色里少量增强即可

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
      if (neighbors.has(n)) {
        // 高亮节点：保持原色、略提 zIndex
        return { ...data, zIndex: 1 };
      }
      // 非邻居：按主题淡出（深色用更深的灰，避免“发白”）
      return { ...data, color: fadedNodeColor, label: undefined, zIndex: 0 };
    });

    sigma.setSetting("edgeReducer", (e: string, data: any) => {
      if (!hovered) return data;
      const ext = g.extremities(e);
      if (!ext) return data;
      const [s, t] = ext;
      if (neighbors.has(String(s)) && neighbors.has(String(t))) {
        // 邻接边：保留原色，稍微增粗
        return { ...data, color: data.color, size: Math.max(1.1, (data.size || 1) * neighborEdgeBoost) };
      }
      // 非邻接边：按主题淡出
      return { ...data, color: fadedEdgeColor, size: Math.max(0.4, (data.size || 1) * 0.5) };
    });

    sigma.refresh();
  }, [sigma, hovered, enabled, theme, fadedNodeColor, fadedEdgeColor, neighborEdgeBoost]);

  return null;
}

/** 边粗细缩放（不重建） */
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

/** 节点大小缩放（不重建） */
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

/** 构图：节点构建、独立的边构建、颜色模式切换仅重染 */
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

  const [comMap, setComMap] = useState<Record<number, number>>({});
  const [fourMap, setFourMap] = useState<Record<number, string>>({});

  const dNodeLimit = useDebounced(nodeLimit, 150);
  const dEdgeLimit = useDebounced(edgeLimit, 150);
  const dMinWeight = useDebounced(minWeight, 150);

  /** 只改设置，不重建 */
  useEffect(() => {
    sigma.setSetting("renderEdges", showEdges);
    sigma.setSetting("hideEdgesOnMove", hideEdgesOnMove);
    sigma.refresh();
  }, [showEdges, hideEdgesOnMove, sigma]);

  /** 先把 maps 读好 */
  useEffect(() => {
    let cancel = false;
    (async () => {
      const [cm, fm] = await Promise.all([loadCommunityMap(), loadFourMap()]);
      if (!cancel) {
        setComMap(cm);
        setFourMap(fm);
        onCommunityLoad?.(Object.keys(cm).length);
      }
    })();
    return () => { cancel = true; };
  }, [onCommunityLoad]);

  /** 构建节点（只在 nodeLimit 或数据文件变化时重建） */
  useEffect(() => {
    let canceled = false;
    (async () => {
      const g = sigma.getGraph() as Graph;
      g.clear();

      try {
        setMsg("加载 /data/layout.csv ...");
        const r1 = await fetch("/data/layout.csv", { cache: "no-store" });
        if (!r1.ok) throw new Error(`layout.csv HTTP ${r1.status}`);
        const t1 = await r1.text();
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

        setMsg(`构建节点 ${usedIdxs.length}/${totalNodes} ...`);
        for (const ri of usedIdxs) {
          if (canceled) return;
          const r = rows[ri];
          const key = String(r.index);
          const idx = Number(r.index);

          // 社区优先 layout -> communities.csv
          let comm: number | null = r.community != null ? Number(r.community) : null;
          if (!Number.isFinite(comm as number)) {
            const c2 = comMap[idx];
            comm = Number.isFinite(c2) ? c2 : null;
          }
          const four = fourMap[idx] || null;

          // 初始按当前模式着色
          const col = colorMode === "four" && four ? colorForFour(four) : colorForCommunity(comm);

          if (!g.hasNode(key)) {
            g.addNode(key, {
              x: Number(r.x), y: Number(r.y),
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
          if (comm != null) counterCommunity.set(comm, (counterCommunity.get(comm) || 0) + 1);
          if (four) counterFour.set(four, (counterFour.get(four) || 0) + 1);
        }

        // 图例
        if (onLegend) {
          let items: LegendItem[] = [];
          if (colorMode === "four" && counterFour.size > 0) {
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

        // 相机复位
        sigma.getCamera().setState({ x: 0, y: 0, ratio: 1 });
        sigma.refresh();

        // 暴露调试把手
        (window as any).PC = { sigma, graph: g };

        // 统计
        onStats?.({ totalNodes, usedNodes: g.order, totalEdges: 0, usedEdges: 0 });
        setMsg(null);
      } catch (e: any) {
        setMsg(`加载/构图失败：${e?.message ?? String(e)}`);
      }
    })();
    return () => { canceled = true; };
  // 仅在这些变动时重建“节点”
  }, [sigma, dNodeLimit, baseNodeSize, colorMode, comMap, fourMap, onStats, onLegend]);

  /** 颜色模式切换 → 只重染，不重建节点 */
  useEffect(() => {
    const g = sigma.getGraph();
    if (g.order === 0) return;
    g.forEachNode((n) => {
      const comm = g.getNodeAttribute(n, "community") as number | null;
      const four = g.getNodeAttribute(n, "four") as string | null;
      const col = colorMode === "four" && four ? colorForFour(four) : colorForCommunity(comm);
      g.setNodeAttribute(n, "color", col);
    });
    sigma.refresh();
  }, [colorMode, sigma]);

  /** 构建/重建边（不动节点） */
  useEffect(() => {
    let canceled = false;
    (async () => {
      const g = sigma.getGraph() as Graph;

      // 每次重建边前，先清掉现有边
      g.clearEdges();

      if (!showEdges) {
        sigma.refresh();
        return;
      }

      try {
        setMsg("加载 /data/edges.csv ...");
        const r2 = await fetch("/data/edges.csv", { cache: "no-store" });
        if (!r2.ok) { setMsg(null); return; } // 没有 edges.csv 也算正常
        const t2 = await r2.text();
        const p2 = Papa.parse<EdgeRow>(t2, { header: true, dynamicTyping: true, skipEmptyLines: true });

        let es = (p2.data || []).filter(Boolean) as EdgeRow[];
        let totalEdges = es.length;

        es = es
          .map(e => ({ s: resolveSource(e), t: resolveTarget(e), w: resolveWeight(e) }))
          .filter(e => Number.isFinite(e.w) && e.w >= dMinWeight)
          .filter(e => g.hasNode(String(e.s)) && g.hasNode(String(e.t)));

        // 权重优先 & 上限
        es.sort((a, b) => b.w - a.w);
        if (dEdgeLimit > 0 && es.length > dEdgeLimit) es = es.slice(0, dEdgeLimit);

        let added = 0;
        for (let i = 0; i < es.length; i += ADD_BATCH) {
          if (canceled) return;
          const batch = es.slice(i, i + ADD_BATCH);
          setMsg(`写入边 ${i + 1} ~ ${i + batch.length} / ${es.length} ...`);
          for (const e of batch) {
            const s = String(e.s);
            const t = String(e.t);
            const a = s < t ? s : t;
            const b = s < t ? t : s;
            if (!g.hasEdge(a, b)) {
              const edgeSize = 0.8 + Math.max(0, Math.min(1, e.w)) * 2.4;
              g.addEdge(a, b, {
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

        onStats?.({ totalNodes: g.order, usedNodes: g.order, totalEdges, usedEdges: added });
        setMsg(null);
      } catch (e: any) {
        setMsg(`加载/构图失败：${e?.message ?? String(e)}`);
      }
    })();
    return () => { canceled = true; };
  // 仅边相关设置变化时执行
  }, [sigma, showEdges, dMinWeight, dEdgeLimit, onStats]);

  return msg ? <div className="toast" style={{ zIndex: 20 }}>{msg}</div> : null;
}

/** ---------- 页面 ---------- */
export default function GraphPage() {
  const [nodeLimit, setNodeLimit] = useState<number>(LIMIT_INITIAL_NODES);
  const [edgeLimit, setEdgeLimit] = useState<number>(12000);
  const [minWeight, setMinWeight] = useState<number>(0.35);
  const [showEdges, setShowEdges] = useState<boolean>(true);
  const [hideEdgesOnMove, setHideEdgesOnMove] = useState<boolean>(true);
  const [stats, setStats] = useState<Stats | null>(null);

  const [nodeScale, setNodeScale] = useState<number>(1.6);
  const [edgeScale, setEdgeScale] = useState<number>(1.2);
  const baseNodeSize = 6.0;

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

  /** 调试：把某个社区染红 / 恢复 */
  function paintCommunity(c: number) {
    const PC = (window as any).PC;
    if (!PC?.graph) return;
    PC.graph.forEachNode((n: string) => {
      const cc = PC.graph.getNodeAttribute(n, "community");
      if (cc === c) PC.graph.setNodeAttribute(n, "color", "hsl(0, 82%, 42%)");
    });
    PC.sigma.refresh();
  }
  function repaintByMode() {
    const PC = (window as any).PC;
    if (!PC?.graph) return;
    PC.graph.forEachNode((n: string) => {
      const cc = PC.graph.getNodeAttribute(n, "community") as number | null;
      const ff = PC.graph.getNodeAttribute(n, "four") as string | null;
      const col = colorMode === "four" && ff ? colorForFour(ff) : colorForCommunity(cc);
      PC.graph.setNodeAttribute(n, "color", col);
    });
    PC.sigma.refresh();
  }

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

        <button className="chip" onClick={() => paintCommunity(17)}>测试：把社区 17 染红</button>
        <button className="chip" onClick={repaintByMode}>清除测试颜色（重染）</button>

        {commCount === 0 && colorMode === "community" && (
          <div className="badge" style={{ background: "#fef3c7", borderColor: "#fde68a" }}>
            未检测到 communities.csv（放到 web/public/data）
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
          baseNodeSize={6.0}
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