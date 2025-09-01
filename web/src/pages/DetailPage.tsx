import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { colorForCommunity } from "../lib/colors";
import { fetchTextNoBOM, parseCsv, getAny } from "../lib/csv";

type LayoutRow = {
  index: number | string;
  x?: number | string;
  y?: number | string;
  paper_id?: string;
  title?: string;
  community?: number | string;
};
type LabeledRow = {
  index: number | string;
  id?: string; title?: string; authors?: string; four?: string;
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
type ComRow = { index?: number | string; community?: number | string };

// ---- helpers ----
function normId(raw?: string) {
  if (!raw) return "";
  let s = String(raw).trim();
  s = s.replace(/^arxiv:/i, "");
  s = s.replace(/v\d+$/i, "");       // 去掉尾部版本号 v1/v2
  return s.toLowerCase();
}
async function loadCommunityMap(): Promise<Record<string, number>> {
  try {
    let txt: string;
    try {
      txt = await fetchTextNoBOM("/data/communities.csv");
    } catch {
      txt = await fetchTextNoBOM("/data/graph/communities.csv");
    }
    const p = parseCsv<ComRow>(txt);
    const map: Record<string, number> = {};
    for (const r of p.data || []) {
      const idx = getAny(r, ["index", "node", "paper_index"]);
      const c   = getAny(r, ["community", "cluster", "comm"]);
      if (idx != null && c != null && Number.isFinite(Number(c))) {
        map[String(idx)] = Number(c);
      }
    }
    return map;
  } catch {
    return {};
  }
}

export default function DetailPage() {
  const { index } = useParams();                     // URL /detail/:index
  const indexStr = String(index ?? "");
  const idxNumber = useMemo(() => {
    const n = Number(indexStr);
    return Number.isFinite(n) ? n : null;
  }, [indexStr]);

  const [layout, setLayout] = useState<LayoutRow | null>(null);
  const [paper, setPaper] = useState<PaperRow | null>(null);
  const [community, setCommunity] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let canceled = false;

    async function run() {
      try {
        setErr(null);

        // ---------- 0) 优先 nodes_labeled.csv（按 index 直接拿 id/title/authors） ----------
        let labeled: LabeledRow | null = null;
        try {
          const t0 = await fetchTextNoBOM("/data/nodes_labeled.csv");
          const p0 = parseCsv<LabeledRow>(t0);
          const rows0 = (p0.data || []).filter(r => r && r.index != null);
          labeled = rows0.find(r =>
            String(r.index) === indexStr ||
            (idxNumber != null && Number(r.index) === idxNumber)
          ) || null;
        } catch { /* optional */ }

        // ---------- 1) layout.csv：找该 index 的行 ----------
        const t1 = await fetchTextNoBOM("/data/layout.csv");
        const p1 = parseCsv<LayoutRow>(t1);
        const rows = (p1.data || []).filter(r => r && r.index != null);

        let row = rows.find(r => String(r.index) === indexStr)
              || rows.find(r => idxNumber != null && Number(r.index) === idxNumber)
              || null;

        if (!row) {
          setErr(`未在 layout.csv 找到 index=${indexStr}`);
          return;
        }
        if (canceled) return;
        setLayout(row);

        // ---------- 2) 社区 ----------
        let comm: number | null =
          row.community != null && Number.isFinite(Number(row.community))
            ? Number(row.community)
            : null;

        if (!Number.isFinite(comm as number)) {
          const cmap = await loadCommunityMap();
          const c2 = cmap[indexStr] ?? (idxNumber != null ? cmap[String(idxNumber)] : undefined);
          comm = Number.isFinite(c2) ? c2! : null;
        }
        if (!canceled) setCommunity(comm);

        // ---------- 3) Paper 信息 ----------
        // 3.1 候选 paper_id：layout.csv / nodes_labeled.csv / 其它变体
        const candPaperId =
          getAny(row as any, ["paper_id", "paperId", "id"]) ??
          (labeled?.id ?? undefined);
        const paperIdRaw = candPaperId ? String(candPaperId) : "";

        // 3.2 如果拿到了 id，就去 papers.csv 精确 & 模糊匹配一遍
        if (paperIdRaw) {
          try {
            const t2 = await fetchTextNoBOM("/data/papers.csv");
            const p2 = parseCsv<PaperRow>(t2);
            const ps = (p2.data || []).filter(pr => pr && pr.id);

            // 精确（去空格）
            let one = ps.find(pr => String(pr.id).trim() === paperIdRaw.trim());

            if (!one) {
              // 模糊：去掉前缀 & 版本号再比
              const key = normId(paperIdRaw);
              one = ps.find(pr => normId(pr.id) === key);
            }

            // 兜底：如果 papers.csv 也有 index 字段，可按 index 匹配
            if (!one && idxNumber != null) {
              const byIndex = (ps as any[]).find((pr) => Number((pr as any).index) === idxNumber);
              if (byIndex) one = byIndex as PaperRow;
            }

            if (!canceled && one) setPaper(one);
          } catch { /* optional */ }
        } else if (labeled) {
          // 即使没有 paper_id，labeled.csv 也能提供标题作者
          const fake: PaperRow = {
            id: "",
            title: labeled.title,
            authors: labeled.authors,
          };
          if (!canceled) setPaper(fake);
        }
      } catch (e: any) {
        setErr(e?.message ?? String(e));
      }
    }

    if (indexStr) run();
    return () => { canceled = true; };
  }, [indexStr, idxNumber]);

  // 展示字段：优先 papers.csv，然后回退 layout/nodes_labeled
  const title =
    paper?.title ||
    layout?.title ||
    `#${idxNumber ?? indexStr}`;

  const paperIdShown =
    (layout?.paper_id ? String(layout.paper_id) : "") ||
    (paper?.id ? String(paper.id) : "") ||
    "-";

  const authorsShown = paper?.authors || "-";
  const commColor = colorForCommunity(community ?? null);

  return (
    <div className="page-wrap">
      <div className="page-header">
        <div className="breadcrumbs">
          <Link to="/" className="link">首页</Link>
          <span> / </span>
          <Link to="/graph" className="link">网络</Link>
          <span> / </span>
          <span className="muted">详情</span>
        </div>
        <h1 className="page-title">{title}</h1>
        <div className="badges-row">
          <span className="pill" style={{ background: commColor, color: "#fff" }}>
            社区 {community ?? "?"}
          </span>
          {paper?.field && <span className="chip">field: {paper.field}</span>}
          {paper?.field_multi && <span className="chip">field_multi: {paper.field_multi}</span>}
          {paper?.is_AP != null && <span className="chip">is_AP: {paper.is_AP}</span>}
          {paper?.is_NA != null && <span className="chip">is_NA: {paper.is_NA}</span>}
        </div>
      </div>

      {err && <div className="error-box">{err}</div>}

      <div className="card">
        <div className="kv">
          <div><span className="k">Paper ID</span><span className="v">{paperIdShown}</span></div>
          <div><span className="k">Index</span><span className="v">{idxNumber ?? indexStr}</span></div>
          <div><span className="k">Authors</span><span className="v">{authorsShown}</span></div>
        </div>

        {paper?.abstract && (
          <div className="abstract">
            <div className="section-title">Abstract</div>
            <p>{paper.abstract}</p>
          </div>
        )}

        <div className="actions" style={{ display: "flex", gap: 10 }}>
          {paperIdShown && paperIdShown !== "-" && (
            <a className="btn" href={`https://arxiv.org/abs/${paperIdShown}`} target="_blank" rel="noreferrer">
              打开 arXiv
            </a>
          )}
          <Link className="btn ghost" to="/graph">返回网络</Link>
        </div>
      </div>
    </div>
  );
}