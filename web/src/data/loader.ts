import Papa from 'papaparse';

export type NodeRow = {
  index: string|number;
  id?: string;
  title?: string;
  authors?: string;
  field?: string;
  field_multi?: string;
  is_AP?: string | number;
  is_NA?: string | number;
  community?: string | number;
  four?: string;
};

export type EdgeRow = {
  source: string|number; target: string | number;
  weight?: string | number;
};

export type LayoutRow = {
  index: string|number; x: string | number; y: string | number;
};

export function loadCSV<T = any>(url: string): Promise<T[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<T>(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data as T[]),
      error: (err) => reject(err)
    });
  });
}

export async function loadAll(base = '/data') {
  const [nodesRaw, edgesRaw, layoutRaw] = await Promise.all([
    loadCSV<NodeRow>(`${base}/nodes_labeled.csv`),
    loadCSV<EdgeRow>(`${base}/edges.csv`),
    loadCSV<LayoutRow>(`${base}/layout.csv`)
  ]);

  // 规范化
  const nodes = nodesRaw.map((r) => {
    const idx = String(r.index ?? '');
    return {
      index: idx,
      id: r.id ?? String(r.index ?? ''),
      title: r.title ?? '',
      authors: r.authors ?? '',
      field: r.field ?? '',
      field_multi: r.field_multi ?? '',
      is_AP: Number(r.is_AP ?? 0),
      is_NA: Number(r.is_NA ?? 0),
      community: r.community !== undefined && r.community !== '' ?
          Number(r.community) :
          undefined,
      four: r.four ?? 'Other',
    };
  });

  const edges =
      edgesRaw.filter((e) => e.source !== undefined && e.target !== undefined)
          .map((e) => ({
                 source: String(e.source),
                 target: String(e.target),
                 weight: e.weight !== undefined ? Number(e.weight) : 1,
               }));

  const layout: Record < string, {
    x: number;
    y: number
  }
  > = {};
  for (const r of layoutRaw) {
    const idx = String(r.index ?? '');
    if (!idx) continue;
    layout[idx] = {x: Number(r.x), y: Number(r.y)};
  }

  return {nodes, edges, layout};
}