import React, { useEffect, useMemo, useState } from "react";
import { loadAll } from "../data/loader";
import { Link } from "react-router-dom";

export default function ListPage() {
  const [loading, setLoading] = useState(true);
  const [rows, setRows] = useState<any[]>([]);
  const [q, setQ] = useState("");
  const [comm, setComm] = useState<string>("");

  useEffect(() => {
    let alive = true;
    (async () => {
      const { nodes } = await loadAll("/data");
      if (!alive) return;
      setRows(nodes);
      setLoading(false);
    })();
    return () => {
      alive = false;
    };
  }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    const c = comm.trim();
    return rows.filter((r) => {
      const okText =
        !s ||
        String(r.title || "").toLowerCase().includes(s) ||
        String(r.id || "").toLowerCase().includes(s) ||
        String(r.authors || "").toLowerCase().includes(s);
      const okComm = !c || String(r.community ?? "") === c;
      return okText && okComm;
    });
  }, [rows, q, comm]);

  if (loading) return <div className="container">加载中……</div>;

  const commSet = Array.from(new Set(rows.map((r) => r.community).filter((x) => x !== undefined)));

  return (
    <div className="container">
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 12 }}>
          <input
            placeholder="按标题 / arXiv id / 作者 搜索"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            style={{ flex: 1, padding: "8px 10px", borderRadius: 8, border: "1px solid #1f2937", background: "#0b1117", color: "white" }}
          />
          <select
            value={comm}
            onChange={(e) => setComm(e.target.value)}
            style={{ padding: "8px 10px", borderRadius: 8, border: "1px solid #1f2937", background: "#0b1117", color: "white" }}
          >
            <option value="">全部社区</option>
            {commSet.map((c) => (
              <option key={String(c)} value={String(c)}>
                社区 {String(c)}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 120 }}>arXiv id</th>
              <th>标题</th>
              <th style={{ width: 120 }}>社区</th>
              <th style={{ width: 150 }}>标签</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={String(r.index)}>
                <td>
                  <Link to={`/paper/${encodeURIComponent(String(r.id ?? r.index))}`}>{r.id ?? r.index}</Link>
                </td>
                <td>{r.title}</td>
                <td>
                  <span className="badge">{r.community ?? "-"}</span>
                </td>
                <td>
                  <span className="badge">{r.four ?? "Other"}</span>
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td colSpan={4} style={{ color: "#9da9b5" }}>暂无结果</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}