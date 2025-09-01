import React from "react";
import { Link, NavLink } from "react-router-dom";

export default function NavBar() {
  return (
    <div className="topbar">
      <div className="brand"><Link to="/" className="link">Paper Community</Link></div>
      <div className="nav">
        <NavLink to="/" end>首页</NavLink>
        <NavLink to="/graph">网络</NavLink>
        <NavLink to="/list">列表</NavLink>
      </div>
    </div>
  );
}