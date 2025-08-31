import React from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import GraphPage from "./pages/GraphPage";
import ListPage from "./pages/ListPage";
import DetailPage from "./pages/DetailPage";

export default function App() {
  return (
    <div style={{ height: "100%" }}>
      <nav className="nav">
        <NavLink to="/" end>Graph</NavLink>
        <NavLink to="/list">List</NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<GraphPage />} />
        <Route path="/list" element={<ListPage />} />
        <Route path="/paper/:id" element={<DetailPage />} />
      </Routes>
    </div>
  );
}