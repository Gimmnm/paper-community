// src/components/MathText.tsx
import React from "react";
import katex from "katex";

/**
 * 将字符串中的 $...$（行内）与 $$...$$（块级）片段渲染为 KaTeX。
 * 用法：<MathText text={titleOrAbstract} />
 */
type Props = { text?: string | null };

const TOK = /(\$\$[^$]+\$\$|\$[^$]+\$)/g;

export default function MathText({ text }: Props) {
  if (!text) return null;
  const src = String(text);

  const parts = src.split(TOK).filter(Boolean);

  return (
    <>
      {parts.map((p, i) => {
        const isDisplay = p.startsWith("$$") && p.endsWith("$$");
        const isInline = !isDisplay && p.startsWith("$") && p.endsWith("$");

        if (!isDisplay && !isInline) {
          return <React.Fragment key={i}>{p}</React.Fragment>;
        }

        const inner = p.slice(isDisplay ? 2 : 1, isDisplay ? -2 : -1);
        // KaTeX 渲染为 HTML
        const html = katex.renderToString(inner, {
          displayMode: isDisplay,
          throwOnError: false,
          strict: "ignore",
          output: "html",
        });

        return isDisplay ? (
          <div
            key={i}
            className="math-block"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        ) : (
          <span
            key={i}
            className="math-inline"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      })}
    </>
  );
}