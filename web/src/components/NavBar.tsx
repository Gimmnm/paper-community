import React from "react";
import { NavLink } from "react-router-dom";

const linkStyle: React.CSSProperties = { textDecoration: "none", padding: "6px 10px", borderRadius: 8 };
const active: React.CSSProperties = { background: "#e5edff", color: "#1e40af" };

export default function NavBar() {
  return (
    <div className="toolbar">
      <span className="badge">Paper Community</span>
      <NavLink to="/graph" style={({ isActive }) => ({ ...linkStyle, ...(isActive ? active : {}) })}>
        图视图
      </NavLink>
      <NavLink to="/list" style={({ isActive }) => ({ ...linkStyle, ...(isActive ? active : {}) })}>
        列表
      </NavLink>
    </div>
  );
}