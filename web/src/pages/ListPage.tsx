// src/pages/ListPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import Papa from "papaparse";
import { colorForCommunity, colorForFour } from "../lib/colors";
import MathText from "../components/MathText";
import RichTitle from "../components/RichTitle";

type LayoutRow = {
  index: number | string;
  paper_id?: string;
  title?: string;
};
type PaperRow = {
  id: string;
  title?: string;
  authors?: string;
};
type ComRow  = { index?: number | string; community?: number | string };
type LblRow  = { index?: number | string; four?: string };

type Item = {
  index: number;
  paper_id?: string;
  title?: string;
  authors?: string;
  community?: number | null;
  four?: string | null;
};

async function fetchCSV<T>(path: string): Promise<T[]> {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) return [];
  const text = await r.text();
  const p = Papa.parse<T>(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
  return (p.data || []).filter(Boolean) as T[];
}

const PAGE_SIZE = 100;

export default function ListPage() {
  const [items, setItems] = useState<Item[]>([]);
  const [q, setQ] = useState("");
  const [fourFilter, setFourFilter] = useState<string>("all");
  const [page, setPage] = useState(1);

  // 加载并合并：layout + communities + nodes_labeled + papers/nodes
  useEffect(() => {
    let canceled = false;
    async function run() {
      const [layouts, communities, labels] = await Promise.all([
        fetchCSV<LayoutRow>("/data/layout.csv"),
        fetchCSV<ComRow>("/data/communities.csv"),
        fetchCSV<LblRow>("/data/nodes_labeled.csv"),
      ]);

      // 索引映射
      const cmap = new Map<number, number>();
      for (const r of communities) {
        if (r.index != null && r.community != null) cmap.set(Number(r.index), Number(r.community));
      }
      const fmap = new Map<number, string>();
      for (const r of labels) {
        if (r.index != null && r.four != null) fmap.set(Number(r.index), String(r.four));
      }

      // 论文元数据（authors）：可选
      let papers = await fetchCSV<PaperRow>("/data/papers.csv");
      if (!papers || papers.length === 0) papers = await fetchCSV<PaperRow>("/data/nodes.csv");
      const pid2paper = new Map<string, PaperRow>();
      for (const p of papers) pid2paper.set(String(p.id).trim(), p);

      const list: Item[] = [];
      for (const r of layouts) {
        const idx = Number(r.index);
        const pid = r.paper_id ? String(r.paper_id).trim() : undefined;
        const p = pid ? pid2paper.get(pid) : undefined;
        list.push({
          index: idx,
          paper_id: pid,
          title: r.title || p?.title,
          authors: p?.authors,
          community: cmap.get(idx) ?? null,
          four: fmap.get(idx) ?? null,
        });
      }
      if (!canceled) setItems(list.sort((a,b) => a.index - b.index));
    }
    run();
    return () => { canceled = true; };
  }, []);

  // 过滤 & 分页
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return items.filter(it => {
      if (fourFilter !== "all" && (it.four || "Other") !== fourFilter) return false;
      if (!needle) return true;
      const inTitle   = (it.title   || "").toLowerCase().includes(needle);
      const inAuthors = (it.authors || "").toLowerCase().includes(needle);
      const inId      = (it.paper_id|| "").toLowerCase().includes(needle);
      const inIndex   = String(it.index).includes(needle);
      return inTitle || inAuthors || inId || inIndex;
    });
  }, [items, q, fourFilter]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const pageSlice = useMemo(() => {
    const start = (pageSafe - 1) * PAGE_SIZE;
    return filtered.slice(start, start + PAGE_SIZE);
  }, [filtered, pageSafe]);

  useEffect(() => { setPage(1); }, [q, fourFilter]);

  return (
    <div className="page-wrap page-wrap--wide">

      <div className="page-header">
        <h1 className="page-title">论文列表</h1>
        <div className="badges-row" style={{ gap: 10 }}>
          <input
            className="input"
            placeholder="搜索：title / authors / id / index …"
            value={q}
            onChange={e => setQ(e.target.value)}
            style={{ width: 340 }}
          />
          <select className="input" value={fourFilter} onChange={e => setFourFilter(e.target.value)}>
            <option value="all">全部四类</option>
            <option value="AP-only">AP-only</option>
            <option value="NA-only">NA-only</option>
            <option value="AP+NA">AP+NA</option>
            <option value="Other">Other</option>
          </select>
          <span className="chip">共 {filtered.length} 条</span>
        </div>
      </div>

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 88 }}>Index</th>
              <th>Title</th>
              <th style={{ width: 160 }}>Paper ID</th>
              <th style={{ width: 120 }}>社区</th>
              <th style={{ width: 120 }}>四类</th>
              <th style={{ width: 100 }}></th>
            </tr>
          </thead>
          <tbody>
            {pageSlice.map(it => {
              const c = it.community;
              const color = colorForCommunity(c ?? null);
              return (
                <tr key={it.index} className="row">
                  <td className="col-index">{it.index}</td>

                  <td className="col-title">
                    <RichTitle
                      text={it.title || "-"}
                      className="row-title"
                    />
                    {it.authors && <div className="row-authors">{it.authors}</div>}
                  </td>

                  <td className="col-paperid">{it.paper_id || "-"}</td>

                  <td className="col-comm">
                    <span className="legend-item">
                      <span className="legend-dot" style={{ background: color }} />
                      <span className="legend-text">{c != null ? `C${c}` : "-"}</span>
                    </span>
                  </td>

                  <td className="col-four">
                    {it.four ? (
                      <span className="pill" style={{ background: colorForFour(it.four), color: "#fff" }}>
                        {it.four}
                      </span>
                    ) : "-"}
                  </td>

                  <td className="text-right">
                    <Link className="btn" to={`/detail/${it.index}`}>详情</Link>
                  </td>
                </tr>
              );
            })}
            {pageSlice.length === 0 && (
              <tr><td colSpan={6} style={{ textAlign: "center", color: "#6b7280" }}>没有匹配结果</td></tr>
            )}
          </tbody>
        </table>

        <div className="pagination">
          <button
            className="btn"
            disabled={pageSafe <= 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
          >
            上一页
          </button>
          <span className="chip">第 {pageSafe} / {totalPages} 页</span>
          <button
            className="btn"
            disabled={pageSafe >= totalPages}
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
          >
            下一页
          </button>
        </div>
      </div>
    </div>
  );
}