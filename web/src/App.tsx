// src/App.tsx
import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import HomePage from "./pages/HomePage";
import GraphPage from "./pages/GraphPage";
import ListPage from "./pages/ListPage";
import DetailPage from "./pages/DetailPage";
import ThemeToggle from "./components/ThemeToggle";

export default function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Link to="/" className="brand-link">
            <img className="brand-logo" src="/pic/logo.svg" alt="logo" />
            <span className="brand-word">Paper Community</span>
          </Link>
        </div>
        <nav className="nav">
          <Link to="/">首页</Link>
          <Link to="/graph">网络</Link>
          <Link to="/list">列表</Link>
          <ThemeToggle /> {/* 主题切换按钮 */}
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/list" element={<ListPage />} />
        <Route path="/detail/:index" element={<DetailPage />} />
      </Routes>
    </div>
  );
}