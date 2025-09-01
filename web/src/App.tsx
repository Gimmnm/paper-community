import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import HomePage from "./pages/HomePage";
import GraphPage from "./pages/GraphPage";
import ListPage from "./pages/ListPage";
import DetailPage from "./pages/DetailPage";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><Link to="/">Paper Community</Link></div>
        <nav className="nav">
          <Link to="/">首页</Link>
          <Link to="/graph">网络</Link>
          <Link to="/list">列表</Link>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/list" element={<ListPage />} />
        <Route path="/detail/:idx" element={<DetailPage />} />
      </Routes>
    </div>
  );
}