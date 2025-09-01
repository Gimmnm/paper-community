// src/pages/DetailPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Papa from "papaparse";
import { colorForCommunity, colorForFour } from "../lib/colors";
import MathText from "../components/MathText";

type LayoutRow = {
  index: number | string;
  x?: number | string;
  y?: number | string;
  paper_id?: string;
  title?: string;
  community?: number | string;
};

type PaperRow = {
  id: string;
  title?: string;
  authors?: string;
  abstract?: string;
  field?: string;
  field_multi?: string;
  is_AP?: number | string;
  is_NA?: number | string;
};

type ComRow  = { index?: number | string; community?: number | string };
type LblRow  = { index?: number | string; four?: string };

async function fetchCSV<T>(path: string): Promise<T[]> {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) return [];
  const text = await r.text();
  const p = Papa.parse<T>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
  return (p.data || []).filter(Boolean) as T[];
}

function Pill({ text, color, invert=false }: { text: string; color: string; invert?: boolean }) {
  const style = invert ? { background: "#fff", color, border: `1px solid ${color}` } : { background: color, color: "#fff" };
  return <span className="pill" style={style}>{text}</span>;
}
function KV({ k, v }: { k: React.ReactNode; v: React.ReactNode }) {
  return (
    <>
      <div className="k">{k}</div>
      <div className="v">{v}</div>
    </>
  );
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div className="section-title">{title}</div>
      <div>{children}</div>
    </div>
  );
}

export default function DetailPage() {
  const { index } = useParams();
  const idx = useMemo(() => (index ? Number(index) : NaN), [index]);

  const [layout, setLayout] = useState<LayoutRow | null>(null);
  const [paper, setPaper] = useState<PaperRow | null>(null);
  const [community, setCommunity] = useState<number | null>(null);
  const [four, setFour] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;
    async function run() {
      try {
        setErr(null);

        // 1) layout.csv 找该 index
        const layouts = await fetchCSV<LayoutRow>("/data/layout.csv");
        const row = layouts.find(r => Number(r.index) === idx) || null;
        if (!row) {
          setErr(`未在 layout.csv 找到 index=${idx}`);
          return;
        }
        if (canceled) return;
        setLayout(row);

        // 2) 社区：优先 layout，回退 communities.csv
        let comm: number | null = row.community != null ? Number(row.community) : null;
        if (!Number.isFinite(comm as number)) {
          const cmaps = await fetchCSV<ComRow>("/data/communities.csv");
          const found = cmaps.find(r => Number(r.index) === idx);
          comm = found && found.community != null ? Number(found.community) : null;
        }
        if (!canceled) setCommunity(Number.isFinite(comm as number) ? (comm as number) : null);

        // 3) 四类：nodes_labeled.csv 可选
        const lbls = await fetchCSV<LblRow>("/data/nodes_labeled.csv");
        const lbl = lbls.find(r => Number(r.index) === idx);
        if (!canceled) setFour(lbl?.four ? String(lbl.four) : null);

        // 4) 论文元数据：papers.csv；若没有则尝试 nodes.csv
        let ps = await fetchCSV<PaperRow>("/data/papers.csv");
        if ((!ps || ps.length === 0)) ps = await fetchCSV<PaperRow>("/data/nodes.csv");

        let p: PaperRow | null = null;
        const pid = row.paper_id ? String(row.paper_id).trim() : null;
        if (pid) {
          p = ps.find(r => String(r.id).trim() === pid) || null;
        } else {
          // fallback：有的表没有 paper_id，就按 title 近似匹配
          const t = row.title ? String(row.title).trim() : null;
          if (t) p = ps.find(r => String(r.title).trim() === t) || null;
        }
        if (!canceled) setPaper(p);
      } catch (e: any) {
        if (!canceled) setErr(e?.message ?? String(e));
      }
    }
    if (!Number.isNaN(idx)) run();
    return () => { canceled = true; };
  }, [idx]);

  const title = paper?.title || layout?.title || `#${idx}`;
  const paperId = layout?.paper_id ? String(layout.paper_id) : null;
  const commColor = colorForCommunity(community ?? null);
  const fourColor = colorForFour(four || undefined);

  return (
    <div className="page-wrap">

      <div className="page-header">
        <div className="breadcrumbs">
          <Link to="/" className="link">首页</Link>
          <span> / </span>
          <Link to="/list" className="link">列表</Link>
          <span> / </span>
          <span className="muted">详情</span>
        </div>
        <h1 className="page-title">
          <MathText text={title} />
        </h1>
        <div className="badges-row" style={{ gap: 8 }}>
          <Pill text={`社区 ${community ?? "?"}`} color={commColor} />
          {four && <Pill text={four} color={fourColor} />}
          {paper?.field && <span className="chip">field: {paper.field}</span>}
          {paper?.field_multi && <span className="chip">multi: {paper.field_multi}</span>}
          {paper?.is_AP != null && <span className="chip">is_AP: {paper.is_AP}</span>}
          {paper?.is_NA != null && <span className="chip">is_NA: {paper.is_NA}</span>}
        </div>
      </div>

      {err && <div className="error-box">{err}</div>}

      <div className="card">
        <div className="kv">
          <KV k="Paper ID" v={paperId || "-"} />
          <KV k="Index" v={index} />
          <KV k="Authors" v={paper?.authors || "-"} />
        </div>

        {paper?.abstract && (
          <div className="abstract">
            <div className="section-title">Abstract</div>
            <p><MathText text={paper.abstract} /></p>
          </div>
        )}

        <div className="cta" style={{ marginTop: 16 }}>
          {paperId && (
            <a className="btn btn-primary" href={`https://arxiv.org/abs/${paperId}`} target="_blank" rel="noreferrer">
              打开 arXiv
            </a>
          )}
          <Link className="btn" to="/graph">返回网络</Link>
        </div>
      </div>
    </div>
  );
}