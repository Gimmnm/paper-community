// src/utils/colors.ts
export function hslToHex(h: number, s: number, l: number): string {
  s /= 100;
  l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let [r, g, b] = [0, 0, 0];

  if (0 <= hp && hp < 1)
    [r, g, b] = [c, x, 0];
  else if (1 <= hp && hp < 2)
    [r, g, b] = [x, c, 0];
  else if (2 <= hp && hp < 3)
    [r, g, b] = [0, c, x];
  else if (3 <= hp && hp < 4)
    [r, g, b] = [0, x, c];
  else if (4 <= hp && hp < 5)
    [r, g, b] = [x, 0, c];
  else if (5 <= hp && hp < 6)
    [r, g, b] = [c, 0, x];

  const m = l - c / 2;
  const toHex = (v: number) =>
      Math.round((v + m) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function colorForCommunity(comm?: number|null): string {
  if (comm == null || Number.isNaN(comm)) return '#64748b';  // slate-500
  const h = (Number(comm) * 47) % 360;
  return hslToHex(h, 82, 42);
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