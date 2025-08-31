import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { loadAll } from "../data/loader";

export default function DetailPage() {
  const { id } = useParams<{ id: string }>();
  const [row, setRow] = useState<any | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      const { nodes } = await loadAll("/data");
      const found =
        nodes.find((r) => String(r.id) === String(id)) ??
        nodes.find((r) => String(r.index) === String(id));
      if (alive) setRow(found ?? null);
    })();
    return () => {
      alive = false;
    };
  }, [id]);

  if (!row) {
    return (
      <div className="container">
        <div style={{ marginBottom: 12 }}>
          <Link to="/list" style={{ color: "#4cc9f0", textDecoration: "none" }}>← 返回列表</Link>
        </div>
        未找到论文（id={id}）
      </div>
    );
  }

  const arxiv = row.id ? `https://arxiv.org/abs/${row.id}` : undefined;

  return (
    <div className="container">
      <div style={{ marginBottom: 12 }}>
        <Link to="/list" style={{ color: "#4cc9f0", textDecoration: "none" }}>← 返回列表</Link>
      </div>

      <div className="grid2">
        <div className="card">
          <div style={{ fontSize: 18, marginBottom: 8 }}>{row.title || "(无标题)"}</div>
          <div style={{ color: "#9da9b5", marginBottom: 8 }}>
            <div>arXiv id: {row.id ?? "-"}</div>
            <div>作者: {row.authors ?? "-"}</div>
            <div>社区: <span className="badge">{row.community ?? "-"}</span></div>
            <div>四类标签: <span className="badge">{row.four ?? "Other"}</span></div>
            <div>field: {row.field ?? "-"}</div>
            <div>field_multi: {row.field_multi ?? "-"}</div>
            <div>is_AP: {row.is_AP ?? 0} &nbsp; is_NA: {row.is_NA ?? 0}</div>
          </div>
          {arxiv && (
            <a href={arxiv} target="_blank" rel="noreferrer" className="badge" style={{ textDecoration: "none" }}>
              打开 arXiv
            </a>
          )}
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 8 }}>原始记录</div>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#cbd5e1" }}>
            {JSON.stringify(row, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}