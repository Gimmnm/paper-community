/* global cytoscape */

const $ = (id) => document.getElementById(id);

const SUBGRAPH_MAX_NODES = 2000;
const SUBGRAPH_MAX_EDGES = 450;
/** 全局社区图：与后端 /graph/communities 的 max_nodes 一致；可在页面修改并写入 localStorage */
const LS_KEY_GLOBAL_GRAPH_MAX_NODES = "paper_comm_global_graph_max_nodes";
const GLOBAL_GRAPH_MAX_NODES_MIN = 10;
const GLOBAL_GRAPH_MAX_NODES_MAX = 200_000;
const GLOBAL_GRAPH_MAX_NODES_DEFAULT = 1500;
const SPREAD = 145;

function clampGlobalGraphMaxNodes(n) {
  const x = Math.floor(Number(n));
  if (!Number.isFinite(x)) return GLOBAL_GRAPH_MAX_NODES_DEFAULT;
  return Math.min(GLOBAL_GRAPH_MAX_NODES_MAX, Math.max(GLOBAL_GRAPH_MAX_NODES_MIN, x));
}

function readStoredGlobalGraphMaxNodes() {
  try {
    const s = localStorage.getItem(LS_KEY_GLOBAL_GRAPH_MAX_NODES);
    if (s == null || s === "") return null;
    return clampGlobalGraphMaxNodes(parseInt(s, 10));
  } catch (_) {
    return null;
  }
}

function getGlobalGraphMaxNodes() {
  const inp = $("inpGlobalMaxNodes");
  if (inp && inp.value !== "") return clampGlobalGraphMaxNodes(parseInt(inp.value, 10));
  const stored = readStoredGlobalGraphMaxNodes();
  return stored ?? GLOBAL_GRAPH_MAX_NODES_DEFAULT;
}

function syncGlobalGraphMaxNodesInput() {
  const inp = $("inpGlobalMaxNodes");
  if (!inp) return;
  const v = readStoredGlobalGraphMaxNodes() ?? GLOBAL_GRAPH_MAX_NODES_DEFAULT;
  inp.value = String(clampGlobalGraphMaxNodes(v));
}

function persistGlobalGraphMaxNodes(n) {
  const v = clampGlobalGraphMaxNodes(n);
  try {
    localStorage.setItem(LS_KEY_GLOBAL_GRAPH_MAX_NODES, String(v));
  } catch (_) {
    /* ignore */
  }
  const inp = $("inpGlobalMaxNodes");
  if (inp) inp.value = String(v);
  return v;
}

async function applyGlobalGraphMaxNodesSetting() {
  persistGlobalGraphMaxNodes(getGlobalGraphMaxNodes());
  if (graphMode === "global") {
    await loadCommunityGraph();
  }
}

function apiUrl(path) {
  return path.startsWith("/api") ? path : `/api${path}`;
}

async function fetchJson(path, params = undefined) {
  const url = new URL(apiUrl(path), window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v === undefined || v === null) return;
      url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.toString(), { method: "GET" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
  return await res.json();
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function truncateText(s, maxLen) {
  const t = String(s ?? "").trim();
  if (!maxLen || t.length <= maxLen) return t;
  return `${t.slice(0, Math.max(0, maxLen - 1))}…`;
}

let cy = null;
let graphMode = "global";
let lastCommunityId = null;
let showEdges = true;
let lastFocus = null;
let experimentCatalog = { runs: [], algorithms: [], time_windows: [] };
let currentRunId = null;
let currentResolution = null;
let resolutionSliderValues = [];
let currentPid = null;
/** 左侧「当前论文」固定区展示用（与下方检索结果分离） */
let currentPaperHit = null;
let appMode = "user";
let adminSixwayCache = null;

function activeParams() {
  return {
    run_id: currentRunId || undefined,
    resolution: currentResolution !== null && currentResolution !== undefined ? Number(currentResolution) : undefined,
  };
}

function setDetails(obj) {
  $("rawJson").textContent = JSON.stringify(obj, null, 2);
  $("details").innerHTML = renderDetailsPanel(obj);
}

function setHealth(h) {
  const el = $("health");
  const emb = h.embeddings_loaded ? " emb=ok" : " emb=—";
  const tw = h.time_window ? ` win=${h.time_window}` : "";
  el.textContent = `papers=${h.n_papers}  comm=${h.n_communities}  r=${Number(h.resolution).toFixed(4)}  ${h.active_run_id || ""}${tw}${emb}`;
  const err = h.embeddings_error || "";
  const tc = h.topic_communities_csv;
  const tcl = h.topic_communities_csv_loaded;
  const tip = [err, tc ? `主题表: ${tc}` : "主题表: —", tcl ? "主题词已挂到社区" : "主题词未挂到社区（无表或社区 id 不匹配）"].filter(Boolean).join("\n");
  el.title = tip || "";
}

function syncResolutionSlider() {
  const rng = $("rngResolution");
  const sel = $("selResolution");
  const lab = $("rngResolutionLabel");
  if (!sel) return;
  resolutionSliderValues = [...sel.options].map((o) => Number(o.value)).filter((x) => Number.isFinite(x));
  if (resolutionSliderValues.length === 0) return;
  const sliderRow = $("userResolutionSliderRow");
  const useSlider = rng && sliderRow && !sliderRow.classList.contains("hidden");
  if (useSlider) {
    rng.min = "0";
    rng.max = String(Math.max(0, resolutionSliderValues.length - 1));
    rng.step = "1";
    let idx = resolutionSliderValues.indexOf(Number(currentResolution));
    if (idx < 0) idx = resolutionSliderValues.length - 1;
    rng.value = String(idx);
    currentResolution = resolutionSliderValues[idx];
    sel.value = String(currentResolution);
  } else if (Number.isFinite(Number(currentResolution)) && [...sel.options].some((o) => Number(o.value) === Number(currentResolution))) {
    sel.value = String(currentResolution);
  }
  if (lab) lab.textContent = `r=${Number(currentResolution).toFixed(4)}`;
}

function renderResults(hits) {
  const root = $("results");
  root.innerHTML = "";
  if (!hits || hits.length === 0) {
    root.innerHTML = `<div class="muted">无结果</div>`;
    return;
  }
  hits.forEach((h) => {
    const payload = h.payload || {};
    const div = document.createElement("div");
    div.className = "result";
    div.dataset.kind = h.kind;
    div.dataset.id = h.id;
    if (h.kind === "paper") {
      const title = payload.title || `PID ${payload.pid || h.id}`;
      const comm = payload.community;
      const commPill =
        comm === null || comm === undefined
          ? `<span class="pill">comm -</span>`
          : `<span class="pill clickable-cid" data-cid="${Number(comm)}">C${escapeHtml(String(comm))}</span>`;
      div.innerHTML = `
        <div class="result-title">${escapeHtml(title)}</div>
        <div class="result-meta">
          <span class="pill">pid ${payload.pid || h.id}</span>
          ${commPill}
          <span class="pill">score ${Number(h.score).toFixed(4)}</span>
        </div>`;
    } else {
      div.innerHTML = `<div class="result-title">${escapeHtml(h.kind)} ${escapeHtml(h.id)}</div>`;
    }
    div.addEventListener("click", async (ev) => {
      const cidEl = ev.target.closest(".clickable-cid");
      if (cidEl?.dataset?.cid != null && cidEl.dataset.cid !== "") {
        const cc = Number(cidEl.dataset.cid);
        if (Number.isFinite(cc)) {
          ev.stopPropagation();
          await enterCommunityFromCid(cc).catch((e) => setDetails({ error: String(e) }));
        }
        return;
      }
      if (h.kind === "paper") {
        const pid = Number(payload.pid || h.id);
        if (Number.isFinite(pid)) await selectPaper(pid);
      }
    });
    root.appendChild(div);
  });
}

function renderCurrentPaperSlot() {
  const root = $("currentPaperSlot");
  if (!root) return;
  const h = currentPaperHit;
  if (!h || h.kind !== "paper") {
    root.innerHTML = `<div class="muted">未选择论文</div>`;
    return;
  }
  const p = h.payload || {};
  const title = p.title || `PID ${p.pid ?? ""}`;
  const comm = p.community;
  const commPill =
    comm === null || comm === undefined
      ? `<span class="pill">comm -</span>`
      : `<span class="pill clickable-cid" data-cid="${Number(comm)}">C${escapeHtml(String(comm))}</span>`;
  root.innerHTML = `
    <div class="result current-paper-card" data-current-pid="${Number(p.pid)}">
      <div class="result-title">${escapeHtml(title)}</div>
      <div class="result-meta">
        <span class="pill">pid ${escapeHtml(String(p.pid ?? ""))}</span>
        ${commPill}
      </div>
    </div>`;
}

function makeNodeKey(kind, rawId) {
  return `${kind}:${rawId}`;
}

function parseCommNodeId(s) {
  try {
    const str = String(s || "");
    const parts = str.split("|", 2);
    if (parts.length !== 2) return null;
    const r = Number(parts[0].split("=", 2)[1]);
    const c = Number(parts[1].split("=", 2)[1]);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return null;
    return { resolution: r, cid: c, node_id: `r=${r.toFixed(4)}|c=${c}` };
  } catch (_) {
    return null;
  }
}

/** 全局社区节点 id 为 r=…|c=…；若仅有数字则视为 cid，分辨率取当前 session。 */
function parseCommunityTap(rawId) {
  const c = parseCommNodeId(String(rawId));
  if (c && Number.isFinite(Number(c.cid))) return { cid: Number(c.cid), resolution: Number(c.resolution) };
  const n = Number(rawId);
  if (Number.isFinite(n) && n >= 0) return { cid: Math.floor(n), resolution: Number(currentResolution) };
  return null;
}

function firstPidFromPaperRows(rows) {
  if (!Array.isArray(rows)) return null;
  const row = rows.find((x) => Number.isFinite(Number(x?.pid)));
  return row ? Number(row.pid) : null;
}

/** 为社区选一篇种子论文（中心 / 桥接 / 示例 / 子图节点），供与「直接点论文」相同的详情 + 子图流程。 */
async function pickSeedPidForCommunity(cid) {
  const c = Number(cid);
  if (!Number.isFinite(c) || c < 0) return null;
  try {
    const cr = await fetchJson(`/communities/${c}`, { top_papers: 12, top_neighbors: 0, ...activeParams() });
    const pay = cr?.hits?.[0]?.payload;
    let pid =
      firstPidFromPaperRows(pay?.center_papers) ??
      firstPidFromPaperRows(pay?.bridge_papers) ??
      firstPidFromPaperRows(pay?.example_papers);
    if (pid) return pid;
  } catch (_) {
    /* fall through */
  }
  try {
    const sg = await fetchJson(`/graph/community/${c}`, { max_nodes: 40, max_edges: 0, ...activeParams() });
    const nodes = sg?.nodes || [];
    const row = nodes.find((n) => Number.isFinite(Number(n?.id)));
    if (row) return Number(row.id);
  } catch (_) {
    /* ignore */
  }
  return null;
}

/**
 * 以社区为入口：对齐 session 后选种子论文并走 selectPaper（用户模式下子图 + 右侧论文/社区卡片与点论文一致）。
 * @param {number} cid
 * @param {{ resolution?: number }} opts 从全局社区图点节点时传入节点上的 r，以对齐分辨率。
 */
async function enterCommunityFromCid(cid, opts = {}) {
  const c = Number(cid);
  if (!Number.isFinite(c)) return;
  if (opts.resolution != null && Number.isFinite(Number(opts.resolution))) {
    await syncSessionResolutionNear(Number(opts.resolution));
  }
  await pushExperimentSession();
  const seed = await pickSeedPidForCommunity(c);
  if (Number.isFinite(seed)) {
    await selectPaper(seed);
    return;
  }
  await loadCommunitySubgraph(c);
}

async function syncSessionResolutionNear(targetR) {
  const tr = Number(targetR);
  if (!Number.isFinite(tr)) return;
  const sel = $("selResolution");
  const vals =
    sel && sel.options.length > 0
      ? [...sel.options].map((o) => Number(o.value)).filter(Number.isFinite)
      : resolutionSliderValues.length > 0
        ? resolutionSliderValues.slice()
        : [tr];
  let best = vals[0];
  let bd = Infinity;
  for (const v of vals) {
    const d = Math.abs(Number(v) - tr);
    if (d < bd) {
      bd = d;
      best = v;
    }
  }
  if (!Number.isFinite(best)) best = tr;
  if (Math.abs(Number(currentResolution) - Number(best)) < 1e-7) return;
  currentResolution = best;
  if (sel) {
    const opt = [...sel.options].find((o) => Math.abs(Number(o.value) - Number(best)) < 1e-5);
    if (opt) sel.value = opt.value;
  }
  syncResolutionSlider();
  await pushExperimentSession();
}

function rescalePositions(elements, factor) {
  const nodes = elements.filter((e) => e.position);
  if (nodes.length === 0) return elements;
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  nodes.forEach((n) => {
    minX = Math.min(minX, n.position.x);
    maxX = Math.max(maxX, n.position.x);
    minY = Math.min(minY, n.position.y);
    maxY = Math.max(maxY, n.position.y);
  });
  const cx = (minX + maxX) / 2;
  const cyy = (minY + maxY) / 2;
  const f = Number(factor) || 120;
  elements.forEach((e) => {
    if (!e.position) return;
    e.position = { x: (e.position.x - cx) * f, y: (e.position.y - cyy) * f };
  });
  return elements;
}

function setGraphMeta(meta) {
  if (!meta) {
    $("graphMeta").textContent = "";
    return;
  }
  const n = meta.n_nodes ?? meta.nodes ?? "-";
  const m = meta.n_edges ?? meta.edges ?? "-";
  $("graphMeta").textContent = `nodes=${n} edges=${m}`;
}

function applyVisibilityRules() {
  if (!cy) return;
  const z = cy.zoom();
  const shouldShowEdges = showEdges && z >= 0.72;
  cy.edges().forEach((e) => {
    e.style("display", shouldShowEdges ? "element" : "none");
  });
  cy.nodes().forEach((n) => {
    if (n.data("role") === "focus") {
      n.style("label", "");
      return;
    }
    n.style("label", "");
  });
}

function setGraphFromPayload(payload, { defaultKind } = {}) {
  if (!cy || !payload) return;
  let nodes = (payload.nodes || []).map((n) => {
    const kind = n.kind || defaultKind || "paper";
    /** API 的 size / n_members 为成员数；Cytoscape 节点直径单独用 vizSize，避免与「规模」混用。 */
    let vizSize;
    let nMembers;
    if (kind === "community") {
      nMembers = Number(n.n_members ?? n.size);
      if (!Number.isFinite(nMembers) || nMembers < 0) nMembers = undefined;
      vizSize =
        nMembers !== undefined && nMembers > 0
          ? Math.max(5, Math.min(26, 4 + Math.log1p(nMembers) * 2.6))
          : 10;
    } else {
      nMembers = undefined;
      vizSize = 6;
    }
    const rawId = n.id;
    const key = makeNodeKey(kind, rawId);
    const data = {
      id: String(key),
      rawId: String(rawId),
      label: n.label ?? String(rawId),
      kind,
      vizSize,
      role: n.role,
    };
    if (kind === "community" && nMembers !== undefined) data.n_members = nMembers;
    const el = { data: data };
    Object.keys(n || {}).forEach((k) => {
      if (["id", "x", "y", "label", "kind", "size", "role", "n_members", "vizSize"].includes(k)) return;
      if (el.data[k] !== undefined) return;
      el.data[k] = n[k];
    });
    if (Number.isFinite(n.x) && Number.isFinite(n.y)) {
      el.position = { x: Number(n.x), y: Number(n.y) };
      el.data.hasPos = true;
    }
    return el;
  });
  const edges = (payload.edges || []).map((e, i) => {
    const w = e.weight ?? 1.0;
    const width = Math.max(1, Math.min(5, Math.log1p(w)));
    const kind = e.kind || defaultKind || "paper";
    return { data: { id: `e${i}`, source: makeNodeKey(kind, e.source), target: makeNodeKey(kind, e.target), width } };
  });
  nodes = rescalePositions(nodes, SPREAD);
  cy.elements().remove();
  cy.add(nodes);
  cy.add(edges);
  const allHavePos = nodes.length > 0 && nodes.every((n) => n.position);
  if (allHavePos) {
    cy.layout({ name: "preset", fit: true, animate: false }).run();
    cy.fit(undefined, 30);
  } else {
    cy.layout({ name: "cose", animate: false, fit: true }).run();
  }
  setGraphMeta(payload.meta || { n_nodes: nodes.length, n_edges: edges.length });
  applyVisibilityRules();
}

async function focusNodes(kind, rawIds, { fitIds = null } = {}) {
  if (!cy) return;
  if (kind && rawIds && rawIds.length > 0) {
    lastFocus = { kind, id: Number(rawIds[0]), fitIds: fitIds ? fitIds.map((x) => Number(x)) : null };
  }
  const focusKeys = (rawIds || []).map((id) => makeNodeKey(kind, id));
  const focusEles = cy.nodes().filter((n) => focusKeys.includes(n.id()));
  const fitKeys = (fitIds || rawIds || []).map((id) => makeNodeKey(kind, id));
  const fitEles = cy.nodes().filter((n) => fitKeys.includes(n.id()));
  cy.nodes().forEach((n) => {
    if (n.data("role") === "keyword") return;
    n.data("role", undefined);
  });
  focusEles.forEach((n) => n.data("role", "focus"));
  applyVisibilityRules();
  if (fitEles.length > 0) cy.fit(fitEles, 60);
}

function initCy() {
  cy = cytoscape({
    container: $("cy"),
    elements: [],
    style: [
      {
        selector: "node",
        style: {
          "background-color": "#6aa6ff",
          label: "",
          color: "#e6eefc",
          "font-size": 8,
          "overlay-opacity": 0,
          opacity: 0.95,
          "border-width": 1,
          "border-color": "rgba(230,238,252,0.22)",
          width: "data(vizSize)",
          height: "data(vizSize)",
        },
      },
      { selector: "node[kind = 'community']", style: { "background-color": "#57d3a1" } },
      {
        selector: "node[role = 'focus']",
        style: {
          "background-color": "#ff6a8b",
          "border-width": 2,
          "border-color": "rgba(230,238,252,0.92)",
          label: "",
          "z-index-compare": "manual",
          "z-index": 9999,
          width: 11,
          height: 11,
        },
      },
      {
        selector: "edge",
        style: {
          width: "data(width)",
          "line-color": "rgba(230,238,252,0.12)",
          "target-arrow-color": "rgba(230,238,252,0.16)",
          "curve-style": "bezier",
          opacity: 0.45,
        },
      },
    ],
    layout: { name: "preset", animate: false },
    wheelSensitivity: 0.22,
  });
  cy.minZoom(0.12);
  cy.maxZoom(6.0);
  cy.on("tap", "node", async (evt) => {
    const n = evt.target;
    const kind = n.data("kind");
    const rawId = n.data("rawId");
    try {
      if (kind === "community") {
        const p = parseCommunityTap(rawId);
        if (!p) return;
        await enterCommunityFromCid(Number(p.cid), { resolution: p.resolution });
      } else if (kind === "paper") {
        await selectPaper(Number(rawId));
      }
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });
  cy.on("zoom", () => applyVisibilityRules());
  cy.on("mouseover", "node", (evt) => {
    const n = evt.target;
    const kind = n.data("kind") || "node";
    const rawId = n.data("rawId");
    if (kind === "paper") {
      const yr = n.data("year");
      const title = truncateText(n.data("label") || rawId, 140);
      const c = n.data("community");
      const head = [`pid ${rawId}`, c !== undefined && c !== "" ? `社区 C${c}` : null, yr ? `year ${yr}` : null].filter(Boolean).join("  ·  ");
      $("hoverInfo").textContent = `${head}\n${title}`;
      return;
    }
    if (kind === "community") {
      const mc = n.data("n_members");
      $("hoverInfo").textContent = Number.isFinite(Number(mc))
        ? `community ${rawId}  ·  ${mc} 篇`
        : `${kind} ${rawId}`;
      return;
    }
    $("hoverInfo").textContent = `${kind} ${rawId}`;
  });
  cy.on("mouseout", "node", () => {
    $("hoverInfo").textContent = "将鼠标悬停在节点上查看论文与社区";
  });
}

async function pushExperimentSession() {
  if (!currentRunId) return;
  const sel = $("selResolution");
  if (sel && sel.value !== "") {
    const rv = Number(sel.value);
    if (Number.isFinite(rv)) currentResolution = rv;
  }
  const resQ =
    currentResolution !== null && currentResolution !== undefined && Number.isFinite(Number(currentResolution))
      ? Number(currentResolution)
      : undefined;
  await fetchJson("/v3/session/switch", { run_id: currentRunId, resolution: resQ });
}

async function loadCommunityGraph() {
  const cap = getGlobalGraphMaxNodes();
  const payload = await fetchJson("/graph/communities", {
    max_nodes: cap,
    min_weight: 0.0,
    include_positions: true,
    ...activeParams(),
  });
  graphMode = "global";
  setGraphFromPayload(payload, { defaultKind: "community" });
  const baseHint =
    `全局社区网络（节点为社区；当前最多显示约 ${cap} 个大社区，边过弱会隐藏；坐标为成员论文在同一套 UMAP 上的质心，换算法/分辨率后布局会变）`;
  $("graphFooterHint").textContent = payload.meta?.singleton_partition_hint
    ? `${payload.meta.singleton_partition_hint} ${baseHint}`
    : baseHint;
}

async function loadCommunitySubgraph(cid) {
  const payload = await fetchJson(`/graph/community/${cid}`, {
    max_nodes: SUBGRAPH_MAX_NODES,
    max_edges: SUBGRAPH_MAX_EDGES,
    ...activeParams(),
  });
  const err = payload?.meta?.error;
  if (err) {
    setDetails({ error: String(err), meta: payload.meta });
    $("graphFooterHint").textContent = `子图未加载：${err}`;
  }
  graphMode = "community";
  lastCommunityId = cid;
  setGraphFromPayload(payload, { defaultKind: "paper" });
  if (currentPid) {
    const has = (payload.nodes || []).some((n) => Number(n?.id) === Number(currentPid));
    if (!has) {
      try {
        const coords = await fetchJson("/coords/papers", {
          ids: String(currentPid),
          include_title: true,
          max_ids: 1,
          ...activeParams(),
        });
        const n0 = coords?.nodes?.[0];
        payload.nodes.push({
          id: currentPid,
          label: n0?.label ? String(n0.label).slice(0, 120) : String(currentPid),
          x: n0 && Number.isFinite(Number(n0.x)) ? Number(n0.x) : undefined,
          y: n0 && Number.isFinite(Number(n0.y)) ? Number(n0.y) : undefined,
        });
        setGraphFromPayload(payload, { defaultKind: "paper" });
      } catch (_) {
        /* ignore */
      }
    }
    const nb = await neighborPidsForFocus(currentPid);
    await focusNodes("paper", [currentPid], { fitIds: [currentPid, ...nb] });
  }
  $("graphFooterHint").textContent = `社区 ${cid} 内论文子图（蓝点可点）`;
}

async function neighborPidsForFocus(pid) {
  try {
    const res = await fetchJson(`/papers/${pid}`, { k_neighbors: 24, k_neighbors_in_comm: 0, k_neighbor_comms: 0, ...activeParams() });
    const p0 = res?.hits?.[0]?.payload;
    const nb = Array.isArray(p0?.neighbors) ? p0.neighbors.slice(0, 16).map((x) => x.pid) : [];
    return nb.map((x) => Number(x)).filter(Number.isFinite);
  } catch (_) {
    return [];
  }
}

async function loadGlobalCommunityGraphHard() {
  await pushExperimentSession();
  const al = $("adminMiddleList");
  if (al) al.innerHTML = "";
  await loadCommunityGraph();
  lastFocus = null;
  currentPid = null;
  currentPaperHit = null;
  renderCurrentPaperSlot();
  updateVecCommButtons();
  const h = await fetchJson("/health", activeParams());
  setHealth(h);
}

async function selectPaper(pid) {
  currentPid = Number(pid);
  updateVecCommButtons();
  const k_neighbors = 30;
  const res = await fetchJson(`/papers/${currentPid}`, {
    k_neighbors,
    k_neighbors_in_comm: 12,
    k_neighbor_comms: 8,
    ...activeParams(),
  });
  const hit0 = res?.hits?.[0];
  currentPaperHit = hit0 && hit0.kind === "paper" ? hit0 : null;
  renderCurrentPaperSlot();
  await refreshUserRightDetails();
  if (appMode !== "user") return;
  const p0 = res?.hits?.[0]?.payload;
  const cid = p0?.community;
  if (cid !== null && cid !== undefined && Number.isFinite(Number(cid))) {
    await loadCommunitySubgraph(Number(cid));
    graphMode = "paper";
    const neighbors = Array.isArray(p0.neighbors) ? p0.neighbors.slice(0, 20).map((x) => x.pid) : [];
    await focusNodes("paper", [currentPid], { fitIds: [currentPid, ...neighbors] });
  }
}

async function refreshUserRightDetails() {
  if (!currentPid) {
    setDetails({ type: "empty", message: "未选择论文" });
    return;
  }
  const [pr, cr] = await Promise.all([
    fetchJson(`/papers/${currentPid}`, { k_neighbors: 20, k_neighbors_in_comm: 10, k_neighbor_comms: 10, ...activeParams() }),
    (async () => {
      const r0 = await fetchJson(`/papers/${currentPid}`, { k_neighbors: 1, k_neighbors_in_comm: 0, k_neighbor_comms: 0, ...activeParams() });
      const cid = r0?.hits?.[0]?.payload?.community;
      if (cid === null || cid === undefined) return null;
      return fetchJson(`/communities/${cid}`, { top_papers: 8, top_neighbors: 12, ...activeParams() });
    })(),
  ]);
  setDetails({ type: "user_panel", paper: pr, community: cr });
}

function renderDetailsPanel(obj) {
  if (!obj || typeof obj !== "object") return `<div class="muted">无</div>`;
  if (obj.error) return `<div class="card"><div class="card-title">错误</div><div class="hint">${escapeHtml(obj.error)}</div></div>`;
  if (obj.type === "empty") return `<div class="muted">${escapeHtml(obj.message || "")}</div>`;
  if (obj.type === "keyword_note") {
    return `<div class="card"><div class="card-title">关键词</div><div class="hint">查询 <b>${escapeHtml(obj.q)}</b>，命中 ${escapeHtml(String(obj.n))} 条（左侧列表）。点击论文打开社区子图。</div></div>`;
  }
  if ((obj.type === "vector_nn" || obj.type === "community_bundle") && Array.isArray(obj.hits)) {
    return `<div class="card"><div class="card-title">${escapeHtml(obj.type)}</div><div class="hint">共 ${obj.hits.length} 条结果，点击左侧论文打开详情与子图。</div></div>`;
  }
  if (obj.type === "admin_sixway_only") {
    return renderSixwayTableHtml(obj.six || adminSixwayCache);
  }
  if (obj.type === "user_panel") {
    const ph = obj.paper?.hits?.[0]?.payload || {};
    const ch = obj.community?.hits?.[0]?.payload || null;
    const paperCard = renderPaperCard(ph);
    const commCard = ch ? renderCommunityCardShort(ch) : `<div class="muted">无社区信息</div>`;
    return `${paperCard}${commCard}`;
  }
  if (obj.type === "admin_live") {
    return renderAdminLive(obj);
  }
  return `<div class="muted">${escapeHtml(JSON.stringify(obj).slice(0, 400))}</div>`;
}

function renderPaperCard(p) {
  const authors =
    Array.isArray(p.authors) && p.authors.length
      ? p.authors
          .map((a) => String(a.name || "").trim())
          .filter(Boolean)
          .join("；")
      : escapeHtml(p.authors_line || "");
  const kws = Array.isArray(p.keywords) ? p.keywords.join("，") : "";
  const inf = p.impact_factor ?? p.structure_influence_index ?? "—";
  return `
    <div class="card">
      <div class="card-title">论文</div>
      <div class="kv">
        <div>pid</div><b>${escapeHtml(p.pid)}</b>
        <div>社区</div><b>${
          p.community === undefined || p.community === null
            ? "—"
            : `<span class="pill clickable-cid" data-cid="${Number(p.community)}">C${escapeHtml(String(p.community))}</span>`
        }</b>
        <div>影响指数</div><b>${escapeHtml(inf)}</b>
        <div>年份</div><b>${escapeHtml(p.year || "—")}</b>
      </div>
      <div class="paper-title-lg">${escapeHtml(p.title || "")}</div>
      <div class="hint" style="margin-top:8px">作者：${authors || "—"}</div>
      ${p.abstract ? `<div class="abstract-box">${escapeHtml(p.abstract)}</div>` : ""}
      ${kws ? `<div class="hint"><b>关键词</b>：${escapeHtml(kws)}</div>` : ""}
    </div>`;
}

function renderCommunityCardShort(c) {
  const ti = c.topic_info || {};
  const terms = Array.isArray(ti.top_terms) ? ti.top_terms.slice(0, 16).join("，") : "";
  const label = ti.community_label ? String(ti.community_label) : "";
  const centers = (c.center_papers || [])
    .slice(0, 6)
    .map((x) => `<span class="pill clickable-pid" data-pid="${x.pid}">${escapeHtml(x.title || x.pid)}</span>`)
    .join(" ");
  const bridges = (c.bridge_papers || [])
    .slice(0, 6)
    .map((x) => `<span class="pill clickable-pid" data-pid="${x.pid}">${escapeHtml(x.title || x.pid)}</span>`)
    .join(" ");
  const nbs = (c.neighbor_communities || [])
    .slice(0, 10)
    .map(
      (x) =>
        `<span class="pill clickable-cid" data-cid="${x.cid}">C${escapeHtml(x.cid)} (${escapeHtml(Number(x.weight).toFixed(2))})</span>`
    )
    .join(" ");
  return `
    <div class="card" style="margin-top:10px">
      <div class="card-title">社区 C${escapeHtml(c.cid)}</div>
      <div class="kv">
        <div>规模</div><b>${escapeHtml(c.n_members ?? c.size ?? "—")}</b>
      </div>
      ${label ? `<div class="topic-label">${escapeHtml(label)}</div>` : ""}
      ${terms ? `<div class="hint"><b>主题词</b>：${escapeHtml(terms)}</div>` : ""}
      <div class="hint" style="margin-top:8px"><b>中心论文</b>：${centers || "—"}</div>
      <div class="hint"><b>桥接论文</b>：${bridges || "—"}</div>
      <div class="hint"><b>相邻社区</b>：${nbs || "—"}</div>
    </div>`;
}

function renderSixwayTableHtml(six) {
  if (!six || six.error || !(six.rows || []).length) {
    return `<div class="muted">未找到离线 six-way 表（<code>out/experiment_eval/comparison_retrieval_sixway_long.csv</code>）。</div>`;
  }
  const body = (six.rows || [])
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.method || "")}</td>
      <td>${escapeHtml(fmt(r.mean_cosine_mean))}</td>
      <td>${escapeHtml(fmt(r.mean_keyword_tfidf_mean))}</td>
      <td>${escapeHtml(fmt(r.mean_hops_mean))}</td>
      <td>${escapeHtml(fmt(r.time_sec_mean))}</td>
      <td>${escapeHtml(fmt(r.topic_top1_match_mean))}</td>
    </tr>`
    )
    .join("");
  return `
    <div class="card">
      <div class="card-title">离线检索对比（six-way）</div>
      <div class="hint">${escapeHtml(six.comparison_run_tag || "")}</div>
      <table class="tbl">
        <thead><tr><th>方法</th><th>cosine</th><th>TF‑IDF</th><th>hops</th><th>time</th><th>topic</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

function renderAdminLive(obj) {
  const s = obj.summary || {};
  const cmp = obj.sixway_comparison || {};
  const sixTbl = renderSixwayTableHtml(adminSixwayCache);
  const rows = Object.entries(cmp)
    .filter(([k]) => k !== "method_row" && k !== "offline_row")
    .map(([k, v]) => {
      if (!v || typeof v !== "object") return "";
      return `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v.live))}</td><td>${escapeHtml(String(v.mean_of_six ?? "—"))}</td><td>${escapeHtml(
        v.z_vs_mean_of_six === null || v.z_vs_mean_of_six === undefined ? "—" : Number(v.z_vs_mean_of_six).toFixed(2)
      )}</td><td>${escapeHtml(
        v.beat_fraction_vs_six_offline === null || v.beat_fraction_vs_six_offline === undefined
          ? "—"
          : Number(v.beat_fraction_vs_six_offline).toFixed(2)
      )}</td></tr>`;
    })
    .join("");
  return `
    ${sixTbl}
    <div class="card" style="margin-top:10px">
      <div class="card-title">本次检索摘要</div>
      <div class="kv">
        <div>通道</div><b>${escapeHtml(obj.method)}</b>
        <div>分区 run</div><b>${escapeHtml(obj.partition_run_id)}</b>
        <div>mean cosine</div><b>${fmt(s.mean_cosine_to_seed)}</b>
        <div>mean TF‑IDF</div><b>${fmt(s.mean_keyword_tfidf)}</b>
        <div>mean hops</div><b>${fmt(s.mean_shortest_path_hops)}</b>
        <div>time s</div><b>${fmt(s.time_sec)}</b>
        <div>topic match</div><b>${fmt(s.topic_top1_match_rate_vs_seed_community)}</b>
      </div>
    </div>
    <div class="card" style="margin-top:10px">
      <div class="card-title">相对 six-way 离线均值（6 路聚合）</div>
      <table class="tbl"><thead><tr><th>指标</th><th>本次</th><th>六路均值</th><th>z</th><th>优于占比</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="hint">优于占比：本次单种子指标相对六条离线 aggregate 均值的击败比例（粗参考）。</div>
    </div>`;
}

function fmt(x) {
  if (x === null || x === undefined) return "—";
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(4) : String(x);
}

function updateVecCommButtons() {
  const has = Number.isFinite(currentPid);
  const b1 = $("btnVecSearch");
  const b2 = $("btnCommSearch");
  if (b1) b1.disabled = !has;
  if (b2) b2.disabled = !has;
}

function setupLeftTabs() {
  document.querySelectorAll("#secUser .tab").forEach((t) => {
    t.addEventListener("click", () => {
      document.querySelectorAll("#secUser .tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const name = t.dataset.tab;
      document.querySelectorAll("#secUser [data-tab-content]").forEach((c) => {
        c.classList.toggle("hidden", c.dataset.tabContent !== name);
      });
    });
  });
}

function filteredRunsAllTime() {
  return (experimentCatalog.runs || []).filter((r) => String(r.time_window).toLowerCase() === "all");
}

async function refreshResolutionSelectorForRun(run) {
  const sel = $("selResolution");
  if (!sel) return;
  sel.innerHTML = "";
  let vals = [];
  let defaultR = Number(run?.default_resolution);
  try {
    const res = await fetchJson(`/v3/runs/${encodeURIComponent(String(run?.run_id || ""))}/resolutions`);
    vals = Array.isArray(res.resolutions) ? res.resolutions.map((x) => Number(x)).filter((x) => Number.isFinite(x)) : [];
    if (Number.isFinite(Number(res.default_resolution))) defaultR = Number(res.default_resolution);
  } catch (_) {
    vals = [];
  }
  if (vals.length === 0) {
    const d = Number(run?.default_resolution);
    if (Number.isFinite(d)) vals.push(d);
  }
  if (vals.length === 0) vals.push(0.2);
  vals = [...new Set(vals.map((x) => Number(x.toFixed(4))))].sort((a, b) => a - b);
  vals.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = String(r);
    opt.textContent = `r=${Number(r).toFixed(4)}`;
    sel.appendChild(opt);
  });
  const cur = Number(currentResolution);
  if (currentResolution === null || !Number.isFinite(cur) || !vals.includes(cur)) {
    let pick = null;
    if (Number.isFinite(defaultR) && vals.includes(defaultR)) pick = defaultR;
    else if (Number.isFinite(cur)) {
      let best = vals[0];
      let bd = Infinity;
      for (const v of vals) {
        const d = Math.abs(Number(v) - cur);
        if (d < bd) {
          bd = d;
          best = v;
        }
      }
      pick = best;
    } else pick = vals[Math.min(vals.length - 1, Math.floor(vals.length / 2))];
    currentResolution = pick;
  }
  sel.value = String(currentResolution);
  syncResolutionSlider();
}

async function rerenderUserRunDropdown() {
  const runSel = $("selRunId");
  if (!runSel) return;
  const runs = filteredRunsAllTime();
  runSel.innerHTML = "";
  runs.forEach((r) => {
    const opt = document.createElement("option");
    opt.value = String(r.run_id);
    opt.textContent = `${r.run_id} (${r.algorithm})`;
    runSel.appendChild(opt);
  });
  if (runs.length > 0) {
    if (!runs.find((r) => String(r.run_id) === String(currentRunId))) currentRunId = String(runs[0].run_id);
    runSel.value = String(currentRunId);
    const run = runs.find((r) => String(r.run_id) === String(currentRunId)) || runs[0];
    await refreshResolutionSelectorForRun(run);
  }
}

function bindResolutionControls() {
  const sel = $("selResolution");
  if (sel && !sel.dataset.bound) {
    sel.dataset.bound = "1";
    sel.addEventListener("change", async () => {
      const rv = Number(sel.value);
      if (Number.isFinite(rv)) currentResolution = rv;
      syncResolutionSlider();
      await pushExperimentSession();
      if (graphMode === "global") await loadCommunityGraph();
      else if (Number.isFinite(Number(currentPid))) await selectPaper(currentPid);
      else if (lastCommunityId !== null) await enterCommunityFromCid(lastCommunityId);
      else await loadCommunityGraph();
      const h = await fetchJson("/health", activeParams());
      setHealth(h);
    });
  }
  const rng = $("rngResolution");
  if (!rng || rng.dataset.bound) return;
  rng.dataset.bound = "1";
  rng.addEventListener("input", () => {
    const i = Number(rng.value);
    const r = resolutionSliderValues[i];
    if (!Number.isFinite(r)) return;
    currentResolution = r;
    const s = $("selResolution");
    if (s) s.value = String(r);
    const lab = $("rngResolutionLabel");
    if (lab) lab.textContent = `r=${Number(r).toFixed(4)}`;
  });
  rng.addEventListener("change", async () => {
    await pushExperimentSession();
    if (graphMode === "global") await loadCommunityGraph();
    else if (Number.isFinite(Number(currentPid))) await selectPaper(currentPid);
    else if (lastCommunityId !== null) await enterCommunityFromCid(lastCommunityId);
    else await loadCommunityGraph();
    const h = await fetchJson("/health", activeParams());
    setHealth(h);
  });
}

async function userKeywordSearch() {
  const q = $("kwInput").value.trim();
  const top_k = Number($("kwTopK").value || 25);
  if (!q) return;
  const res = await fetchJson("/search/keyword", { q, top_k, offset: 0, ...activeParams() });
  renderResults(res.hits || []);
  setDetails({ type: "keyword_note", q, n: (res.hits || []).length });
}

async function userVectorSearch() {
  if (!currentPid) return;
  const top_k = Number($("vecTopK").value || 25);
  const res = await fetchJson("/search/vector_nn", { paper_id: currentPid, top_k, ...activeParams() });
  renderResults(res.hits || []);
  setDetails(res);
}

async function userCommSearch() {
  if (!currentPid || !currentRunId) return;
  const top_k = Number($("commTopK").value || 25);
  const res = await fetchJson("/search/community_bundle", {
    paper_id: currentPid,
    partition_run_id: currentRunId,
    top_k,
    resolution: currentResolution,
  });
  renderResults(res.hits || []);
  setDetails(res);
}

async function refreshDomainsUi() {
  try {
    const res = await fetchJson("/v2/domains");
    const runs = res.runs || [];
    const block = $("userDomainBlock");
    const isKmeans = String(currentRunId || "").includes("kmeans") || String(currentRunId || "") === "coarse_kmeans";
    if (!isKmeans || runs.length === 0) {
      block.classList.add("hidden");
      return;
    }
    block.classList.remove("hidden");
    const rs = $("domRun");
    const di = $("domId");
    rs.innerHTML = "";
    runs.forEach((r) => {
      const o = document.createElement("option");
      o.value = r.run_id;
      o.textContent = r.run_id;
      rs.appendChild(o);
    });
    const fillDom = () => {
      const run = runs.find((x) => x.run_id === rs.value) || runs[0];
      di.innerHTML = "";
      (run.domains || [0, 1, 2]).forEach((d) => {
        const o = document.createElement("option");
        o.value = String(d);
        o.textContent = `domain ${d}`;
        di.appendChild(o);
      });
    };
    rs.onchange = fillDom;
    fillDom();
  } catch (_) {
    $("userDomainBlock")?.classList.add("hidden");
  }
}

async function loadDomainScatter() {
  if (appMode !== "user") return;
  const run = $("domRun").value;
  const domain = Number($("domId").value);
  const res = await fetchJson(`/v2/domains/${encodeURIComponent(run)}/points`, {
    domain,
    max_points: 5000,
    include_title: true,
  });
  const nodes = (res.nodes || []).map((n) => ({
    id: n.id,
    kind: "paper",
    x: n.x,
    y: n.y,
    label: n.label || String(n.id),
    domain: n.domain,
    size: 6,
  }));
  graphMode = "keyword";
  setGraphFromPayload({ nodes, edges: [], meta: res.meta }, { defaultKind: "paper" });
  $("graphFooterHint").textContent = `子域散点：${run} / domain ${domain}`;
}

function switchMode() {
  appMode = document.querySelector('input[name="appMode"]:checked')?.value || "user";
  $("secUser").classList.toggle("hidden", appMode !== "user");
  $("secAdmin").classList.toggle("hidden", appMode !== "admin");
  $("centerUser")?.classList.toggle("hidden", appMode !== "user");
  $("centerAdmin")?.classList.toggle("hidden", appMode !== "admin");
  if (appMode === "admin") {
    const rq = $("results");
    if (rq) rq.innerHTML = "";
    const al = $("adminMiddleList");
    if (al) al.innerHTML = "";
    loadAdminSixway().then(() => setDetails({ type: "admin_sixway_only", six: adminSixwayCache }));
  } else {
    if (currentPid) {
      refreshUserRightDetails().catch(() => {});
    } else {
      setDetails({ type: "empty", message: "未选择论文" });
    }
    requestAnimationFrame(() => {
      try {
        cy?.resize();
      } catch (_) {
        /* ignore */
      }
    });
  }
}

async function loadAdminSixway() {
  try {
    adminSixwayCache = await fetchJson("/v3/evaluations/retrieval-sixway", { comparison_run_tag: "master_breakpoints" });
  } catch (_) {
    adminSixwayCache = null;
  }
}

function syncAdminResolutionFromRun() {
  const runId = $("admPartitionRun")?.value;
  const run = (experimentCatalog.runs || []).find((r) => String(r.run_id) === String(runId));
  const sel = $("admSelResolution");
  const rng = $("admRngResolution");
  const lab = $("admRngResolutionLabel");
  if (!sel || !run) return;
  const prevAdmR = Number.isFinite(Number(sel.value)) ? Number(sel.value) : null;
  sel.innerHTML = "";
  fetchJson(`/v3/runs/${encodeURIComponent(String(run.run_id))}/resolutions`)
    .then((res) => {
      let vals = Array.isArray(res.resolutions) ? res.resolutions.map(Number).filter(Number.isFinite) : [];
      if (!vals.length) vals = [Number(run.default_resolution) || 0.2];
      vals = [...new Set(vals.map((x) => Number(x.toFixed(4))))].sort((a, b) => a - b);
      vals.forEach((r) => {
        const o = document.createElement("option");
        o.value = String(r);
        o.textContent = `r=${Number(r).toFixed(4)}`;
        sel.appendChild(o);
      });
      const defaultR = Number(res.default_resolution);
      let pick = null;
      if (prevAdmR !== null && vals.includes(prevAdmR)) pick = prevAdmR;
      else if (Number.isFinite(defaultR) && vals.includes(defaultR)) pick = defaultR;
      else pick = vals[Math.min(vals.length - 1, Math.floor(vals.length / 2))];
      sel.value = String(pick);
      const vals2 = [...sel.options].map((o) => Number(o.value));
      const sliderRow = $("adminResolutionSliderRow");
      if (rng && lab && sliderRow && !sliderRow.classList.contains("hidden")) {
        rng.min = "0";
        rng.max = String(Math.max(0, vals2.length - 1));
        let idx = vals2.indexOf(Number(sel.value));
        if (idx < 0) idx = vals2.length - 1;
        rng.value = String(idx);
        lab.textContent = `r=${Number(sel.value).toFixed(4)}`;
        rng.oninput = () => {
          const i = Number(rng.value);
          const r = vals2[i];
          if (Number.isFinite(r)) {
            sel.value = String(r);
            lab.textContent = `r=${Number(r).toFixed(4)}`;
          }
        };
      } else if (lab) {
        lab.textContent = `r=${Number(sel.value).toFixed(4)}`;
      }
    })
    .catch(() => {});
}

async function initAdminControls() {
  const pr = $("admPartitionRun");
  const meth = $("admMethod");
  if (!pr || !meth) return;
  pr.innerHTML = "";
  filteredRunsAllTime().forEach((r) => {
    const o = document.createElement("option");
    o.value = r.run_id;
    o.textContent = r.run_id;
    pr.appendChild(o);
  });
  pr.value = currentRunId || pr.options[0]?.value;
  pr.onchange = () => syncAdminResolutionFromRun();
  meth.innerHTML = "";
  [
    ["keyword", "关键词（标题+摘要）"],
    ["vector_nn", "向量近邻"],
    ["community_bundle", "社区 bundle（用下方分区 run）"],
  ].forEach(([v, t]) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    meth.appendChild(o);
  });
  const adr = $("admSelResolution");
  const arng = $("admRngResolution");
  adr?.addEventListener("change", () => {
    const vals = [...adr.options].map((o) => Number(o.value));
    const idx = vals.indexOf(Number(adr.value));
    if (arng && idx >= 0) arng.value = String(idx);
  });
  arng?.addEventListener("change", () => {
    const vals = [...adr.options].map((o) => Number(o.value));
    const i = Number(arng.value);
    if (vals[i] !== undefined) adr.value = String(vals[i]);
  });
  syncAdminResolutionFromRun();
}

function setAdminMiddleFromHits(hits) {
  const list = $("adminMiddleList");
  if (!list || appMode !== "admin") return;
  if (!hits || !hits.length) {
    list.innerHTML = `<div class="muted">无结果，请运行检索。</div>`;
    return;
  }
  list.innerHTML = hits
    .map((h) => {
      const p = h.payload || {};
      const cs = p.cosine_to_seed !== undefined ? `cos ${Number(p.cosine_to_seed).toFixed(4)}` : "";
      return `<div class="admin-hit-card" data-pid="${Number(p.pid)}">
      <div class="t">${escapeHtml(p.title || p.pid)}</div>
      <div class="m">pid ${escapeHtml(p.pid)}  ${p.community != null ? `C${escapeHtml(p.community)}` : ""}  ${escapeHtml(cs)}</div>
      ${p.abstract ? `<div class="hint" style="margin-top:6px">${escapeHtml(String(p.abstract).slice(0, 320))}</div>` : ""}
    </div>`;
    })
    .join("");
  list.querySelectorAll(".admin-hit-card").forEach((el) => {
    el.addEventListener("click", () => {
      const pid = Number(el.dataset.pid);
      if (Number.isFinite(pid)) selectPaper(pid).catch(() => {});
    });
  });
}

async function adminRunLive() {
  const seed = Number($("admSeedPid").value);
  if (!Number.isFinite(seed) || seed < 1) {
    setDetails({ error: "请填写有效 seed pid" });
    return;
  }
  const method = $("admMethod").value;
  const part = $("admPartitionRun").value;
  const top_k = Number($("admTopK").value || 20);
  const resEff = Number($("admSelResolution")?.value || currentResolution);
  const res = await fetchJson("/v3/retrieval/live", {
    seed_pid: seed,
    method,
    partition_run_id: part,
    resolution: resEff,
    top_k,
    comparison_run_tag: "master_breakpoints",
  });
  if (res.error) {
    setDetails(res);
    return;
  }
  setAdminMiddleFromHits(res.hits || []);
  setDetails({ type: "admin_live", ...res });
}

function onDetailsClick(e) {
  const p = e.target.closest(".clickable-pid");
  if (p?.dataset?.pid) {
    selectPaper(Number(p.dataset.pid));
    return;
  }
  const c = e.target.closest(".clickable-cid");
  if (c?.dataset?.cid != null && c.dataset.cid !== "") {
    const cc = Number(c.dataset.cid);
    if (Number.isFinite(cc)) enterCommunityFromCid(cc).catch((err) => setDetails({ error: String(err) }));
  }
}

async function init() {
  initCy();
  syncGlobalGraphMaxNodesInput();
  setDetails({ type: "empty", message: "选择论文后显示详情" });
  $("toggleRaw").addEventListener("click", () => $("rawJson").classList.toggle("hidden"));
  $("toggleEdges").addEventListener("change", (ev) => {
    showEdges = !!ev.target.checked;
    applyVisibilityRules();
  });
  $("btnApplyGlobalMaxNodes")?.addEventListener("click", () => {
    applyGlobalGraphMaxNodesSetting().catch((e) => setDetails({ error: String(e) }));
  });
  $("inpGlobalMaxNodes")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("btnApplyGlobalMaxNodes")?.click();
    }
  });
  $("details").addEventListener("click", onDetailsClick);

  $("currentPaperSlot")?.addEventListener("click", (e) => {
    const cidEl = e.target.closest(".clickable-cid");
    if (cidEl?.dataset?.cid != null && cidEl.dataset.cid !== "") {
      const cc = Number(cidEl.dataset.cid);
      if (Number.isFinite(cc)) {
        e.stopPropagation();
        enterCommunityFromCid(cc).catch((err) => setDetails({ error: String(err) }));
      }
      return;
    }
    const card = e.target.closest(".current-paper-card[data-current-pid]");
    if (card?.dataset?.currentPid != null && card.dataset.currentPid !== "") {
      const pid = Number(card.dataset.currentPid);
      if (Number.isFinite(pid)) selectPaper(pid).catch((err2) => setDetails({ error: String(err2) }));
    }
  });

  renderCurrentPaperSlot();

  document.querySelectorAll('input[name="appMode"]').forEach((r) => r.addEventListener("change", switchMode));

  setupLeftTabs();

  $("btnLoadCommunities").addEventListener("click", () => loadGlobalCommunityGraphHard().catch((e) => setDetails({ error: String(e) })));
  $("btnBackGlobal").addEventListener("click", () => loadGlobalCommunityGraphHard().catch((e) => setDetails({ error: String(e) })));

  $("selRunId").addEventListener("change", async () => {
    currentRunId = $("selRunId").value;
    const run = filteredRunsAllTime().find((r) => String(r.run_id) === String(currentRunId));
    await refreshResolutionSelectorForRun(run);
    await refreshDomainsUi();
    // 否则 cytoscape 仍是上一分区的图，点击社区时 run_id 已变，cid 与当前分区不一致 → 子图空或 community_not_found。
    if (appMode === "user") {
      await pushExperimentSession();
      if (graphMode === "global") await loadCommunityGraph();
      else if (lastCommunityId !== null && currentPid) await selectPaper(currentPid);
      else await loadCommunityGraph();
      const h = await fetchJson("/health", activeParams());
      setHealth(h);
    }
  });

  $("btnApplySession").addEventListener("click", async () => {
    await pushExperimentSession();
    if (graphMode === "global") await loadCommunityGraph();
    else if (lastCommunityId !== null && currentPid) await selectPaper(currentPid);
    else await loadCommunityGraph();
    const h = await fetchJson("/health", activeParams());
    setHealth(h);
  });

  $("btnKwSearch").addEventListener("click", () => userKeywordSearch().catch((e) => setDetails({ error: String(e) })));
  $("kwInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("btnKwSearch").click();
  });
  $("btnVecSearch").addEventListener("click", () => userVectorSearch().catch((e) => setDetails({ error: String(e) })));
  $("btnCommSearch").addEventListener("click", () => userCommSearch().catch((e) => setDetails({ error: String(e) })));
  $("btnDomainView").addEventListener("click", () => loadDomainScatter().catch((e) => setDetails({ error: String(e) })));

  $("admRandom").addEventListener("click", async () => {
    const r = await fetchJson("/tools/random-paper");
    $("admSeedPid").value = String(r.pid);
  });
  $("admKwPick").addEventListener("click", async () => {
    const q = $("admKwSeed").value.trim();
    if (!q) return;
    const res = await fetchJson("/search/keyword", { q, top_k: 1, offset: 0, ...activeParams() });
    const pid = res?.hits?.[0]?.payload?.pid;
    if (pid) $("admSeedPid").value = String(pid);
  });
  $("admRunLive").addEventListener("click", () => adminRunLive().catch((e) => setDetails({ error: String(e) })));

  const cat = await fetchJson("/v3/catalog");
  const sess = await fetchJson("/v3/session");
  experimentCatalog = cat || { runs: [], algorithms: [], time_windows: [] };
  currentRunId = sess.active_run_id || cat.active_run_id;
  currentResolution = Number(sess.resolution);

  await rerenderUserRunDropdown();
  bindResolutionControls();
  await refreshDomainsUi();
  await initAdminControls();

  switchMode();
  await loadGlobalCommunityGraphHard();

  const h = await fetchJson("/health", activeParams());
  setHealth(h);
}

init().catch((e) => setDetails({ error: String(e) }));
