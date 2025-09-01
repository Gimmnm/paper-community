import React, { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import { Link } from "react-router-dom";
// ✅ 从 lib 里引，同时拼写要对：colorForCommunity
import { colorForCommunity } from "../lib/colors";

type LayoutRow = {
  index: number | string;
  paper_id?: string;
  title?: string;
  community?: number | string;
};
type ComRow = { index?: number | string; community?: number | string };

async function loadCommunityMap(): Promise<Record<number, number>> {
  try {
    const r = await fetch("/data/communities.csv", { cache: "no-store" });
    if (!r.ok) return {};
    const t = await r.text();
    const p = Papa.parse<ComRow>(t, { header: true, dynamicTyping: true, skipEmptyLines: true });
    const map: Record<number, number> = {};
    for (const row of (p.data || [])) {
      if (row && row.index != null && row.community != null) {
        const idx = Number(row.index);
        const c = Number(row.community);
        if (Number.isFinite(idx) && Number.isFinite(c)) map[idx] = c;
      }
    }
    return map;
  } catch {
    return {};
  }
}

export default function ListPage() {
  const [rows, setRows] = useState<LayoutRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    let canceled = false;
    async function run() {
      // 1) layout.csv
      const r1 = await fetch("/data/layout.csv", { cache: "no-store" });
      if (!r1.ok) return;
      const t1 = await r1.text();
      const p1 = Papa.parse<LayoutRow>(t1, { header: true, dynamicTyping: true, skipEmptyLines: true });
      let arr = (p1.data || []).filter(r => r && r.index != null);

      // 2) merge communities.csv
      const cmap = await loadCommunityMap();
      arr = arr.map((r) => {
        const cLayout = r.community != null ? Number(r.community) : null;
        const cMap = cmap[Number(r.index)];
        const community = Number.isFinite(cLayout as number) ? cLayout as number :
                          Number.isFinite(cMap) ? cMap : null;
        return { ...r, community };
      });

      if (!canceled) setRows(arr);
    }
    run();
    return () => { canceled = true; };
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return rows.slice(0, 1000);
    return rows.filter(r =>
      String(r.index).includes(s) ||
      (r.paper_id && r.paper_id.toLowerCase().includes(s)) ||
      (r.title && r.title.toLowerCase().includes(s))
    ).slice(0, 2000);
  }, [rows, q]);

  return (
    <div className="page-wrap">
      <div className="page-header">
        <h1 className="page-title">论文列表</h1>
        <div className="badges-row" style={{ gap: 8 }}>
          <input
            className="chip"
            style={{ padding: "8px 10px", minWidth: 260 }}
            placeholder="搜索：标题 / ID / index"
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
          <span className="chip">共 {rows.length} 条</span>
          <span className="chip">显示 {filtered.length} 条</span>
        </div>
      </div>

      <div className="card" style={{ padding: 0 }}>
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 80 }}>Index</th>
              <th>Title</th>
              <th style={{ width: 160 }}>Paper ID</th>
              <th style={{ width: 130 }}>Community</th>
              <th style={{ width: 100 }}>详情</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const idx = Number(r.index);
              const comm = r.community != null ? Number(r.community) : null;
              const col = colorForCommunity(comm);
              return (
                <tr key={idx}>
                  <td>{idx}</td>
                  <td>{r.title || "-"}</td>
                  <td>{r.paper_id || "-"}</td>
                  <td>
                    <span className="pill" style={{ background: col, color: "#fff" }}>
                      {comm != null ? `#${comm}` : "—"}
                    </span>
                  </td>
                  <td>
                    <Link className="btn ghost" to={`/detail/${idx}`}>查看</Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}