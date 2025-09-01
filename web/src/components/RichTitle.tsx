// src/components/RichTitle.tsx
import React, { useMemo } from "react";
import katex from "katex";

type Props = {
  text: string;
  className?: string;
};

function renderWithKatex(text: string) {
  // 同时处理 $$...$$（display）和 $...$（inline）
  const regex = /(\$\$[^$]+\$\$|\$[^$]+\$)/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let m: RegExpExecArray | null;

  while ((m = regex.exec(text)) !== null) {
    // 之前的普通文本
    if (m.index > lastIndex) {
      parts.push(<span key={`t-${lastIndex}`}>{text.slice(lastIndex, m.index)}</span>);
    }
    const token = m[0];
    const isBlock = token.startsWith("$$");
    const raw = token.slice(isBlock ? 2 : 1, token.length - (isBlock ? 2 : 1)).trim();

    try {
      const html = katex.renderToString(raw, {
        throwOnError: false,
        displayMode: isBlock,
        output: "html",
        strict: "ignore",
      });
      parts.push(
        <span
          key={`k-${m.index}`}
          className={isBlock ? "katex-block" : "katex-inline"}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    } catch {
      // 渲染失败就原样回退
      parts.push(<span key={`f-${m.index}`}>{token}</span>);
    }

    lastIndex = m.index + token.length;
  }

  // 结尾的普通文本
  if (lastIndex < text.length) {
    parts.push(<span key={`t-end`}>{text.slice(lastIndex)}</span>);
  }
  return <>{parts}</>;
}

export default function RichTitle({ text, className }: Props) {
  const content = useMemo(() => renderWithKatex(text), [text]);
  return <div className={className}>{content}</div>;
}