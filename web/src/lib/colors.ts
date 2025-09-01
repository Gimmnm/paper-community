// src/lib/colors.ts
export function colorForCommunity(comm?: number|null): string {
  if (comm == null || Number.isNaN(comm)) return '#334155';  // slate-700
  const h = (Number(comm) * 47) % 360;
  return `hsl(${h} 82% 42%)`;
}

export function colorForFour(four?: string|null): string {
  switch ((four || '').trim()) {
    case 'AP-only':
      return '#ef4444';
    case 'NA-only':
      return '#3b82f6';
    case 'AP+NA':
      return '#8b5cf6';
    default:
      return '#9ca3af';
  }
}

export function colorForEdgeWeight(w: number): string {
  const ww = Math.max(0, Math.min(1, w));
  const l = 22 + (1 - ww) * 20;
  const a = 0.55 + ww * 0.35;
  return `hsla(215 28% ${l}% / ${a})`;
}