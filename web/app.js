/* global cytoscape */

const $ = (id) => document.getElementById(id);

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

function setDetails(obj) {
  $("rawJson").textContent = JSON.stringify(obj, null, 2);
  $("details").innerHTML = renderDetailsHtml(obj);
}

function setHealth(h) {
  $("health").textContent = `papers=${h.n_papers}  communities=${h.n_communities}  r=${Number(h.resolution).toFixed(4)}`;
}

function renderResults(hits) {
  const root = $("results");
  root.innerHTML = "";
  if (!hits || hits.length === 0) {
    root.innerHTML = `<div class="muted">No results.</div>`;
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
      const year = payload.year || 0;
      const comm = payload.community;
      div.innerHTML = `
        <div class="result-title">${escapeHtml(title)}</div>
        <div class="result-meta">
          <span class="pill">pid ${payload.pid || h.id}</span>
          <span class="pill">${year ? `year ${year}` : `year -`}</span>
          <span class="pill">${comm === null || comm === undefined ? `comm -` : `comm ${comm}`}</span>
          <span class="pill">score ${Number(h.score).toFixed(4)}</span>
        </div>
      `;
    } else if (h.kind === "community") {
      const cid = payload.cid ?? Number(h.id);
      div.innerHTML = `
        <div class="result-title">Community ${cid}</div>
        <div class="result-meta">
          <span class="pill">cid ${cid}</span>
          <span class="pill">size ${payload.size ?? "-"}</span>
          <span class="pill">score ${Number(h.score).toFixed(4)}</span>
        </div>
      `;
    } else {
      div.innerHTML = `<div class="result-title">${escapeHtml(h.kind)} ${escapeHtml(h.id)}</div>`;
    }

    div.addEventListener("click", async () => {
      try {
        if (h.kind === "paper") {
          await lookupPaper(payload.pid || Number(h.id));
        } else if (h.kind === "community") {
          await lookupCommunity(payload.cid ?? Number(h.id));
        }
      } catch (e) {
        setDetails({ error: String(e) });
      }
    });

    root.appendChild(div);
  });
}

function escapeHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

let cy = null;
let graphMode = "global"; // global | community | paper | keyword
let lastCommunityId = null;
let showLabels = false;
let showEdges = true;
let spread = 200;
let lastFocus = null; // { kind: "paper", id: number, fitIds?: number[] }

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
          "text-wrap": "wrap",
          "text-max-width": 110,
          "text-valign": "center",
          "text-halign": "center",
          "overlay-opacity": 0,
          opacity: 0.95,
          "border-width": 1,
          "border-color": "rgba(230,238,252,0.22)",
          "text-background-color": "rgba(11, 18, 32, 0.72)",
          "text-background-opacity": 1,
          "text-background-shape": "roundrectangle",
          "text-background-padding": 3,
          width: "data(size)",
          height: "data(size)",
        },
      },
      {
        selector: "node[kind = 'community']",
        style: { "background-color": "#57d3a1" },
      },
      {
        selector: "node[role = 'keyword']",
        style: { "background-color": "#ffd36a" },
      },
      {
        selector: "node[role = 'focus']",
        style: {
          // Focus node should visually match the Details panel target.
          "background-color": "#ff6a8b",
          "border-width": 2,
          "border-color": "rgba(230,238,252,0.92)",
          // No label for focus node (Details already has the info).
          label: "",
          // Keep it visually on top & easier to spot.
          "z-index-compare": "manual",
          "z-index": 9999,
          width: 16,
          height: 16,
        },
      },
      {
        selector: "edge",
        style: {
          width: "data(width)",
          "line-color": "rgba(230,238,252,0.14)",
          "target-arrow-color": "rgba(230,238,252,0.18)",
          "curve-style": "bezier",
          opacity: 0.55,
        },
      },
      {
        selector: ":selected",
        style: {
          "border-width": 2,
          "border-color": "#e6eefc",
          "line-color": "rgba(106,166,255,0.6)",
        },
      },
    ],
    layout: { name: "preset", animate: false },
    wheelSensitivity: 0.22,
  });

  cy.minZoom(0.15);
  cy.maxZoom(6.0);

  cy.on("tap", "node", async (evt) => {
    const n = evt.target;
    const kind = n.data("kind");
    const rawId = n.data("rawId");
    try {
      if (kind === "community") {
        await lookupCommunity(Number(rawId));
      } else if (kind === "paper") {
        await lookupPaper(Number(rawId), { showCommunityMap: true, focus: true });
      }
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });

  cy.on("zoom", () => {
    applyVisibilityRules();
  });
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

function makeNodeKey(kind, rawId) {
  return `${kind}:${rawId}`;
}

function applyVisibilityRules() {
  if (!cy) return;
  const z = cy.zoom();
  const shouldShowEdges = showEdges && z >= 0.75;
  cy.edges().style("display", shouldShowEdges ? "element" : "none");

  // Labels:
  // - always show on hover/selected (handled in stylesheet)
  // - show all labels only when user enables and zoom is high enough
  const showAll = showLabels && z >= 1.35;
  cy.nodes().forEach((n) => {
    const role = n.data("role");
    if (role === "focus") {
      // Never show focus label in-graph (Details already contains the info).
      n.style("label", "");
      return;
    }
    if (showAll) {
      n.style("label", n.data("label"));
      return;
    }
    n.style("label", "");
  });
}

function rescalePositions(elements, factor) {
  // Map existing positions (in data space) to a nicer screen space by:
  // 1) centering at origin
  // 2) scaling by factor
  const nodes = elements.filter((e) => e.position);
  if (nodes.length === 0) return elements;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
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

function setGraphFromPayload(payload, { defaultKind }) {
  if (!payload) return;
  let nodes = (payload.nodes || []).map((n) => {
    const kind = n.kind || defaultKind || "paper";
    // Node size: keep small to reduce overlap on dense maps.
    // - paper: small constant
    // - community: log-scaled by size but capped
    const size =
      kind === "community"
        ? (n.size ? Math.max(10, Math.min(42, 6 + Math.log1p(n.size) * 4.2)) : 16)
        : 10;
    const rawId = n.id;
    const key = makeNodeKey(kind, rawId);
    const el = { data: { id: String(key), rawId: String(rawId), label: n.label ?? String(rawId), kind, size, role: n.role } };
    if (Number.isFinite(n.x) && Number.isFinite(n.y)) {
      el.position = { x: Number(n.x), y: Number(n.y) };
      el.data.hasPos = true;
    }
    return el;
  });
  const edges = (payload.edges || []).map((e, i) => {
    const w = e.weight ?? 1.0;
    const width = Math.max(1, Math.min(6, Math.log1p(w)));
    const kind = e.kind || defaultKind || "paper";
    const source = makeNodeKey(kind, e.source);
    const target = makeNodeKey(kind, e.target);
    return { data: { id: `e${i}`, source: String(source), target: String(target), width } };
  });
  nodes = rescalePositions(nodes, spread);
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

async function loadCommunityGraph() {
  const payload = await fetchJson("/graph/communities", { max_nodes: 400, min_weight: 0.0 });
  graphMode = "global";
  setGraphFromPayload(payload, { defaultKind: "community" });
}

async function loadCommunitySubgraph(cid) {
  const max_nodes = Number($("commMaxNodes").value || 60);
  const max_edges = Number($("commMaxEdges").value || 200);
  const payload = await fetchJson(`/graph/community/${cid}`, { max_nodes, max_edges });
  // If focus paper is outside the truncated subgraph nodes, still show it as an overlay node.
  try {
    if (
      lastFocus &&
      lastFocus.kind === "paper" &&
      Number.isFinite(Number(lastFocus.id)) &&
      payload &&
      Array.isArray(payload.nodes)
    ) {
      const pid = Number(lastFocus.id);
      const has = payload.nodes.some((n) => Number(n?.id) === pid);
      if (!has) {
        const coords = await fetchJson("/coords/papers", { ids: String(pid), include_title: true, max_ids: 1 });
        const n0 = coords && Array.isArray(coords.nodes) ? coords.nodes[0] : null;
        payload.nodes.push({
          id: pid,
          label: (n0 && n0.label) ? String(n0.label).slice(0, 120) : String(pid),
          x: n0 && Number.isFinite(Number(n0.x)) ? Number(n0.x) : undefined,
          y: n0 && Number.isFinite(Number(n0.y)) ? Number(n0.y) : undefined,
        });
      }
    }
  } catch (_) {
    // ignore overlay failures
  }
  graphMode = "community";
  lastCommunityId = cid;
  setGraphFromPayload(payload, { defaultKind: "paper" });
}

async function keywordSearch() {
  const q = $("kwInput").value.trim();
  const top_k = Number($("kwTopK").value || 20);
  const offset = window.__kwOffset || 0;
  const res = await fetchJson("/search/keyword", { q, top_k, offset });
  renderResults(res.hits || []);
  setDetails(res);
  window.__kwLastQuery = q;
  window.__kwTopK = top_k;
  window.__kwOffset = offset;
  $("kwPage").textContent = `offset=${offset}`;

  // Paper map for keyword hits (unified points)
  const pids = (res.hits || [])
    .filter((h) => h.kind === "paper" && h.payload && h.payload.pid)
    .map((h) => Number(h.payload.pid))
    .filter((x) => Number.isFinite(x));
  if (pids.length > 0) {
    const coords = await fetchJson("/coords/papers", { ids: pids.join(","), include_title: true, max_ids: 300 });
    const nodes = (coords.nodes || []).map((n) => ({
      id: n.id,
      kind: "paper",
      role: "keyword",
      x: n.x,
      y: n.y,
      label: n.label || String(n.id),
      size: 16,
    }));
    graphMode = "keyword";
    setGraphFromPayload({ nodes, edges: [], meta: coords.meta }, { defaultKind: "paper" });
  }
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

  // clear previous focus, keep keyword highlighting
  cy.nodes().forEach((n) => {
    const r = n.data("role");
    if (r === "keyword") return;
    n.data("role", undefined);
  });
  focusEles.forEach((n) => n.data("role", "focus"));
  applyVisibilityRules();
  if (fitEles.length > 0) cy.fit(fitEles, 60);
}

async function reapplyFocusAfterRerender() {
  if (!lastFocus || !cy) return;
  const kind = lastFocus.kind;
  const id = Number(lastFocus.id);
  if (!kind || !Number.isFinite(id)) return;
  const fitIds = Array.isArray(lastFocus.fitIds) && lastFocus.fitIds.length > 0 ? lastFocus.fitIds : [id];
  await focusNodes(kind, [id], { fitIds });
}

async function lookupPaper(pid, opts = { showCommunityMap: true, focus: true }) {
  const k_neighbors = Number($("paperK").value || 20);
  const res = await fetchJson(`/papers/${pid}`, { k_neighbors, k_neighbors_in_comm: 10, k_neighbor_comms: 8 });
  renderResults(res.hits || []);
  setDetails(res);

  // Paper page: show its community map, then zoom to paper + neighbors
  const payload = (res.hits && res.hits[0] && res.hits[0].payload) ? res.hits[0].payload : {};
  const cid = payload.community;
  const neighbors = Array.isArray(payload.neighbors) ? payload.neighbors.slice(0, 20).map((x) => x.pid) : [];
  if (opts.showCommunityMap && cid !== null && cid !== undefined && Number.isFinite(Number(cid))) {
    await loadCommunitySubgraph(Number(cid));
    graphMode = "paper";
    // 只给“当前论文”打 focus 标签，但 fit 可以包含邻居，便于 zoom 到局部
    if (opts.focus) await focusNodes("paper", [pid], { fitIds: [pid, ...neighbors] });
  } else if (res.graph_snippet && (res.graph_snippet.nodes || []).length > 0) {
    // fallback ego graph
    const nodes = (res.graph_snippet.nodes || []).map((n) => ({ id: n.id, kind: "paper", label: n.id, size: 18 }));
    const edges = (res.graph_snippet.edges || []).map((e) => ({ source: e.source, target: e.target, weight: e.weight ?? 1.0, kind: "paper" }));
    setGraphFromPayload({ nodes, edges, meta: { n_nodes: nodes.length, n_edges: edges.length } }, { defaultKind: "paper" });
  }
}

async function lookupCommunity(cid) {
  const top_papers = Number($("commTopPapers").value || 20);
  const commRes = await fetchJson(`/communities/${cid}`, { top_papers, top_neighbors: 12 });
  renderResults(commRes.hits || []);

  // Treat "community lookup" as "lookup its center paper" for the detail panel.
  // We still render the community card, but the main detail becomes a full paper detail
  // (abstract + community summary + neighbors + neighbors in community).
  let centerPid = null;
  try {
    const hit0 = commRes.hits && commRes.hits[0] ? commRes.hits[0] : null;
    const center = hit0 && hit0.payload && Array.isArray(hit0.payload.center_papers) ? hit0.payload.center_papers[0] : null;
    if (center && Number.isFinite(Number(center.pid))) centerPid = Number(center.pid);
  } catch (_) {
    centerPid = null;
  }

  let detailsRes = commRes;
  if (centerPid !== null) {
    try {
      const k_neighbors = Number($("paperK").value || 20);
      const paperRes = await fetchJson(`/papers/${centerPid}`, { k_neighbors, k_neighbors_in_comm: 10, k_neighbor_comms: 8 });
      // Compose as "expand" so UI shows paper card + community card.
      const paperHit = paperRes.hits && paperRes.hits[0] ? paperRes.hits[0] : null;
      const commHit = commRes.hits && commRes.hits[0] ? commRes.hits[0] : null;
      if (paperHit && commHit) {
        detailsRes = {
          type: "expand",
          query: { cid, center_pid: centerPid },
          hits: [paperHit, commHit],
          graph_snippet: paperRes.graph_snippet || { nodes: [], edges: [] },
          debug: { composed_from: ["lookup_community", "lookup_paper(center)"] },
        };
      } else {
        detailsRes = paperRes;
      }
    } catch (_) {
      // If center paper lookup fails, fall back to community details.
      detailsRes = commRes;
    }
  }
  setDetails(detailsRes);

  // Community page: graph is its paper subgraph, positioned on global map
  await loadCommunitySubgraph(cid);
  if (centerPid !== null) await focusNodes("paper", [centerPid]);
}

function setupTabs() {
  const tabs = document.querySelectorAll(".tab");
  const contents = document.querySelectorAll("[data-tab-content]");
  tabs.forEach((t) => {
    t.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const name = t.dataset.tab;
      contents.forEach((c) => {
        c.classList.toggle("hidden", c.dataset.tabContent !== name);
      });
    });
  });
}

function renderDetailsHtml(res) {
  if (!res || typeof res !== "object") return `<div class="muted">No details.</div>`;
  const type = res.type || "-";
  const hits = res.hits || [];
  if (hits.length === 0) return `<div class="muted">No hits.</div>`;

  const first = hits[0];
  const p = first.payload || {};

  if (type === "paper") {
    return paperDetailsHtml(p);
  }
  if (type === "community") {
    return communityDetailsHtml(p);
  }
  if (type === "expand") {
    // show paper card + community card if any
    const html = [];
    const paperHit = hits.find((h) => h.kind === "paper");
    const commHit = hits.find((h) => h.kind === "community");
    if (paperHit) html.push(paperDetailsHtml(paperHit.payload || {}));
    if (commHit) html.push(communityDetailsHtml(commHit.payload || {}));
    return html.join("");
  }
  if (type === "keyword") {
    const dbg = res.debug || {};
    return `
      <div class="card">
        <div class="card-title">Keyword results</div>
        <div class="kv">
          <div>query</div><b>${escapeHtml(res.query?.q || "")}</b>
          <div>mode</div><b>${escapeHtml(dbg.mode || "-")}</b>
          <div>hits</div><b>${hits.length}</b>
        </div>
        ${dbg.tfidf_error ? `<div class="hint" style="color: var(--danger)">TF‑IDF fallback: ${escapeHtml(dbg.tfidf_error)}</div>` : ""}
      </div>
      <div class="hint">Click a result to load paper/community details.</div>
    `;
  }
  return `<div class="muted">type=${escapeHtml(type)}</div>`;
}

function paperDetailsHtml(p) {
  const title = p.title || `Paper ${p.pid ?? "-"}`;
  const comm = p.community;
  const commSummary = p.community_summary || null;
  const neighbors = Array.isArray(p.neighbors) ? p.neighbors.slice(0, 12) : [];
  const inComm = Array.isArray(p.neighbors_in_community) ? p.neighbors_in_community.slice(0, 12) : [];

  const commHtml = commSummary
    ? `
      <div class="card">
        <div class="card-title">Community summary</div>
        <div class="kv">
          <div>cid</div><b>${commSummary.cid}</b>
          <div>size</div><b>${commSummary.size}</b>
        </div>
        ${listHtml("Center papers", commSummary.center_papers)}
        ${listHtml("Bridge papers", commSummary.bridge_papers)}
        ${listHtml(
          "Neighbor communities",
          commSummary.neighbor_communities?.map((x) => ({
            cid: x.cid,
            title: `C${x.cid}`,
            meta: `weight ${Number(x.weight).toFixed(3)}  size ${x.size ?? "-"}`,
          }))
        )}
      </div>
    `
    : "";

  return `
    <div class="card">
      <div class="card-title">${escapeHtml(title)}</div>
      <div class="kv">
        <div>pid</div><b>${escapeHtml(p.pid)}</b>
        <div>year</div><b>${escapeHtml(p.year || "-")}</b>
        <div>community</div><b>${comm === null || comm === undefined ? "-" : escapeHtml(comm)}</b>
      </div>
      ${p.abstract ? `<div class="hint" style="margin-top:10px">${escapeHtml(p.abstract)}</div>` : ""}
    </div>
    ${commHtml}
    ${listHtml(
      "Neighbors (mutual-kNN)",
      neighbors.map((x) => ({
        pid: x.pid,
        title: x.title || `PID ${x.pid}`,
        score: x.score,
        community: x.community,
        year: x.year,
      }))
    )}
    ${listHtml(
      "Neighbors in community",
      inComm.map((x) => ({
        pid: x.pid,
        title: x.title || `PID ${x.pid}`,
        score: x.score,
        year: x.year,
      }))
    )}
  `;
}

function communityDetailsHtml(c) {
  const cid = c.cid ?? "-";
  const size = c.size ?? "-";
  return `
    <div class="card">
      <div class="card-title">Community C${escapeHtml(cid)}</div>
      <div class="kv">
        <div>cid</div><b>${escapeHtml(cid)}</b>
        <div>size</div><b>${escapeHtml(size)}</b>
      </div>
    </div>
    ${listHtml("Center papers", c.center_papers)}
    ${listHtml("Bridge papers", c.bridge_papers)}
    ${listHtml("Example papers", c.example_papers)}
    ${listHtml(
      "Neighbor communities",
      c.neighbor_communities?.map((x) => ({
        cid: x.cid,
        title: `C${x.cid}`,
        meta: `weight ${Number(x.weight).toFixed(3)}  size ${x.size ?? "-"}`,
      }))
    )}
  `;
}

function listHtml(title, arr) {
  if (!arr || !Array.isArray(arr) || arr.length === 0) return "";
  const titleLow = String(title || "").toLowerCase();
  const defaultKind = titleLow.includes("community") ? "community" : "paper";
  const items = arr.slice(0, 12).map((x) => {
    if (x === null || x === undefined) return "";
    // Allow primitives (e.g., [123, 456]) as shorthand IDs.
    const obj = (typeof x === "object") ? x : (defaultKind === "community" ? { cid: x } : { pid: x });
    const t = obj.title || obj.label || (obj.pid !== undefined ? `PID ${obj.pid}` : "") || (obj.cid !== undefined ? `C${obj.cid}` : "");
    const m = (() => {
      if (obj.meta) return obj.meta;
      const parts = [
        obj.pid !== undefined ? `pid ${obj.pid}` : null,
        obj.cid !== undefined ? `cid ${obj.cid}` : null,
        obj.year ? `year ${obj.year}` : null,
        obj.community !== undefined && obj.community !== null ? `comm ${obj.community}` : null,
        obj.score !== undefined && obj.score !== null && Number.isFinite(Number(obj.score)) ? `score ${Number(obj.score).toFixed(3)}` : null,
      ].filter(Boolean);
      return parts.join("  ");
    })();
    const kind = obj.cid !== undefined ? "community" : "paper";
    const id = obj.cid !== undefined ? obj.cid : obj.pid;
    const attrs = id !== undefined && id !== null ? `data-kind="${kind}" data-id="${escapeHtml(id)}"` : "";
    return `<div class="list-item clickable" ${attrs}><div class="t">${escapeHtml(t)}</div>${m ? `<div class="m">${escapeHtml(m)}</div>` : ""}</div>`;
  });
  return `
    <div class="card">
      <div class="card-title">${escapeHtml(title)}</div>
      <div class="list">${items.join("")}</div>
    </div>
  `;
}

async function init() {
  setupTabs();
  initCy();
  setDetails({});
  $("toggleRaw").addEventListener("click", () => {
    $("rawJson").classList.toggle("hidden");
  });

  // Graph controls
  $("toggleLabels").addEventListener("change", async (e) => {
    showLabels = !!e.target.checked;
    applyVisibilityRules();
  });
  $("toggleEdges").addEventListener("change", async (e) => {
    showEdges = !!e.target.checked;
    applyVisibilityRules();
  });
  $("spread").addEventListener("input", async (e) => {
    spread = Number(e.target.value || 120);
  });
  $("spread").addEventListener("change", async () => {
    // re-render current graph by reloading appropriate payload (cheap enough)
    try {
      if (graphMode === "global") await loadCommunityGraph();
      else if (graphMode === "community" && lastCommunityId !== null) await loadCommunitySubgraph(lastCommunityId);
      else if (graphMode === "paper" && lastCommunityId !== null) await loadCommunitySubgraph(lastCommunityId);
      else if (graphMode === "keyword") {
        // re-run keywordSearch to rebuild map with new spread (keeps offset)
        await keywordSearch();
      }
      await reapplyFocusAfterRerender();
    } catch (err) {
      setDetails({ error: String(err) });
    }
  });

  // Details click-through (independent from left query)
  $("details").addEventListener("click", async (e) => {
    const el = e.target.closest(".list-item.clickable");
    if (!el) return;
    const kind = el.dataset.kind;
    const id = Number(el.dataset.id);
    if (!kind || !Number.isFinite(id)) return;
    try {
      if (kind === "paper") await lookupPaper(id, { showCommunityMap: true, focus: true });
      if (kind === "community") await lookupCommunity(id);
    } catch (err) {
      setDetails({ error: String(err) });
    }
  });
  $("btnLoadCommunities").addEventListener("click", async () => {
    try {
      await loadCommunityGraph();
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });

  $("kwSearch").addEventListener("click", async () => {
    try {
      window.__kwOffset = 0;
      await keywordSearch();
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });
  $("kwPrev").addEventListener("click", async () => {
    try {
      const top_k = Number($("kwTopK").value || 20);
      window.__kwOffset = Math.max(0, (window.__kwOffset || 0) - top_k);
      await keywordSearch();
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });
  $("kwNext").addEventListener("click", async () => {
    try {
      const top_k = Number($("kwTopK").value || 20);
      window.__kwOffset = (window.__kwOffset || 0) + top_k;
      await keywordSearch();
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });
  $("kwInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") $("kwSearch").click();
  });

  $("paperLookup").addEventListener("click", async () => {
    const pid = Number($("paperIdInput").value || 0);
    if (!pid) return;
    try {
      await lookupPaper(pid);
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });

  $("commLookup").addEventListener("click", async () => {
    const cid = Number($("commIdInput").value || 0);
    if (!cid && cid !== 0) return;
    try {
      await lookupCommunity(cid);
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });

  $("commSubgraph").addEventListener("click", async () => {
    const cid = Number($("commIdInput").value || 0);
    if (!cid && cid !== 0) return;
    try {
      await loadCommunitySubgraph(cid);
    } catch (e) {
      setDetails({ error: String(e) });
    }
  });

  try {
    const h = await fetchJson("/health");
    setHealth(h);
  } catch (e) {
    $("health").textContent = `health error: ${String(e)}`;
  }

  // initial: load community graph
  try {
    await loadCommunityGraph();
  } catch (e) {
    setDetails({ error: String(e) });
  }
}

init().catch((e) => setDetails({ error: String(e) }));

