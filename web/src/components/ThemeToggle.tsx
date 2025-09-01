// src/components/ThemeToggle.tsx
import React, { useEffect, useState } from "react";

const THEMES = ["light", "ocean", "midnight"] as const;
type Theme = typeof THEMES[number];

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = (localStorage.getItem("pc-theme") || "light") as Theme;
    return THEMES.includes(saved) ? saved : "light";
  });

  useEffect(() => {
    // <html data-theme="..."> 用于切换样式变量
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("pc-theme", theme);
  }, [theme]);

  function next() {
    const i = THEMES.indexOf(theme);
    setTheme(THEMES[(i + 1) % THEMES.length]);
  }

  return (
    <button className="btn" onClick={next} title="切换主题">
      主题：{theme === "light" ? "浅色" : theme === "ocean" ? "海洋" : "午夜"}
    </button>
  );
}