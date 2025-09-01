import Papa from 'papaparse';

export async function fetchTextNoBOM(url: string): Promise<string> {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
  const t = await r.text();
  return t.charCodeAt(0) === 0xfeff ? t.slice(1) : t;
}

export function parseCsv<T = any>(text: string) {
  return Papa.parse<T>(
      text, {header: true, dynamicTyping: true, skipEmptyLines: true});
}

/** 在若干候选键里取第一个存在的值 */
export function getAny<T extends object, K extends string>(row: T, keys: K[]) {
  for (const k of keys)
    if (k in (row as any) && (row as any)[k] != null) return (row as any)[k];
  return undefined;
}