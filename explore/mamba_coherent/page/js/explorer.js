// COPY of explore/lm_coherent/page/js/explorer.js, adapted for the Mamba page (SPEC section 9): a view switch
// point | state | stack (each with its own layouts for the layer rows; the bag row is shared), reading blocks for
// every view, depth blocks (point), and a two-panel init/trained strip instead of the training strip.
// The explorer: a grid of layouts at the trained checkpoint (bag row and six layers x six positions) and a
// strip of one layer at t = 128 at both checkpoints. One block is selected; its sets or eigenfunctions colour every
// panel. The dot painter, hungarianMax and matchedJaccard are copied from explore/cifar_coherent/site/js/explorer.js.
import { el, h, showTip, hideTip, tipRows } from "./charts.js";
import { LEVELS, divLUT, seqLUT, qDiv, qSeq, hexRgb } from "./colors.js";
import { D, A, at, stat, excess, fmt, ckName, VIEW_NAME, g3, layerName, cellRecord, blockRecord, topMembers, setRows, snippetEl, srcSwatch, token, showTok } from "./data.js";

const $ = id => document.getElementById(id);
// colour index space shared by every panel: diverging LUT, sequential LUT, 10 categorical, 8 sets, grey, missing
const C_DIV = 0, C_SEQ = LEVELS, C_CAT = 2 * LEVELS, C_SET = C_CAT + 10, C_GREY = C_SET + 8, C_MISS = C_GREY + 1;
const COLOURINGS = {
  sets4: "coherent sets, r = 4", sets2: "coherent sets, r = 2", psi1: "eigenfunction ψ₁", psi2: "eigenfunction ψ₂", nodal: "sign of ψ₁ (nodal domains)",
  tok: "token at t (10 most frequent)", src: "source", own: "own-source walk share",
};

// ------------------------------------------------------------------------------------------- copied: dot painter
const _c32 = new Map();
function css32(css) {
  if (_c32.has(css)) return _c32.get(css);
  let r, g, b;
  if (css.startsWith("#")) [r, g, b] = hexRgb(css);
  else [r, g, b] = css.match(/\d+/g).map(Number);
  const v = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  _c32.set(css, v); return v;
}
function stamp(radius) {
  const offs = [], R = Math.ceil(radius);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= radius * radius + .35) offs.push(dx, dy);
  return offs.length ? offs : [0, 0];
}
// adapted: panels carry their own u8 coordinates, colour indices and draw order
function paint(buf, W, H, panels, pal32, N, radius) {
  const offs = stamp(radius), no = offs.length;
  for (const c of panels) {
    const xy = c.xy, ci = c.ci, order = c.order;
    for (let q = 0; q < N; q++) {
      const i = order[q], px = Math.round(c.x + c.w * xy[2 * i] / 255), py = Math.round(c.y + c.w * (1 - xy[2 * i + 1] / 255));
      const v = pal32[ci[i]];
      for (let o = 0; o < no; o += 2) {
        const x = px + offs[o], y = py + offs[o + 1];
        if (x >= 0 && y >= 0 && x < W && y < H) buf[y * W + x] = v;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------- copied: matched Jaccard
function hungarianMax(M) {   // square matrix, maximise; returns assignment row -> col
  const n = M.length, INF = 1e18, u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = -M[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const a = new Int32Array(n); for (let j = 1; j <= n; j++) a[p[j] - 1] = j - 1; return a;
}
function jaccardMatrix(A_, nA, B, nB) {
  const inter = Array.from({ length: nA }, () => new Float64Array(nB)), ca = new Float64Array(nA), cb = new Float64Array(nB);
  for (let i = 0; i < A_.length; i++) { const a = A_[i], b = B[i]; if (a >= 0) ca[a]++; if (b >= 0) cb[b]++; if (a >= 0 && b >= 0) inter[a][b]++; }
  return inter.map((row, i) => Array.from(row, (x, j) => { const u = ca[i] + cb[j] - x; return u > 0 ? x / u : 0; }));
}
function matchedJaccard(A_, nA, B, nB) {   // copied without the browser null (the stored null is used instead)
  const J = jaccardMatrix(A_, nA, B, nB), n = Math.max(nA, nB);
  const sq = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i < nA && j < nB) ? J[i][j] : 0));
  const asg = hungarianMax(sq);
  const per = [...Array(nA)].map((_, i) => ({ j: asg[i] < nB ? asg[i] : -1, J: asg[i] < nB ? J[i][asg[i]] : 0 }));
  return { mean: per.reduce((s, x) => s + x.J, 0) / nA, per };
}

// ------------------------------------------------------------------------------------------- the explorer
export class Explorer {
  constructor(root) {
    this.root = root;
    const l1 = this.t1li = Math.max(0, D.layers.indexOf(D.t1_layer));
    this.st = { sel: { type: "reading", li: l1 }, color: "sets4", view: "p", pin: -1, strip: l1, hover: -1, hoverPanel: -1, focus: -1 };
    this.N = D.N;
    this.R = D.layers.length + 1;          // bag + layers
    this.Cn = D.positions.length;
    this.panels = [];
    for (let r = 0; r < this.R; r++) for (let c = 0; c < this.Cn; c++) this.panels.push({ kind: "grid", row: r, col: c, li: r - 1, ti: c, si: D.fi });
    for (let s = 0; s < D.S; s++) this.panels.push({ kind: "strip", row: -1, col: s, li: null, ti: this.Cn - 1, si: s });   // init, trained
    const pal = [...divLUT(D.pal), ...seqLUT(D.pal), ...D.palette.classes, ...D.palette.sets, D.palette.unassigned, "#e9e8e3"];
    this.pal32 = Uint32Array.from(pal, css32);
    this.palCss = pal;
    this.buildControls();
    this.buildGrid();
    this.ready = Promise.resolve();
    window.addEventListener("resize", () => { clearTimeout(this._rt); this._rt = setTimeout(() => this.layout(true), 120); });
  }

  // ---------------------------------------------------------------- selection
  // the view the selection is shown in: depth blocks exist for the point view only
  effView() { return this.st.sel.type === "depth" ? "p" : this.st.view; }
  rec() {
    const s = this.st.sel;
    if (s.type === "cell") return cellRecord(s.li < 0 ? "b" : this.st.view, s.ck, s.li, s.ti);
    if (s.type === "reading") return blockRecord("reading", this.st.view, s.li);
    return blockRecord("depth", s.ti);
  }
  selName(long = false) {
    const s = this.st.sel, P = D.positions, v = VIEW_NAME[this.effView()];
    if (s.type === "cell") {
      const whole = this.st.view === "k" && s.li === D.L - 1 ? " (the whole state)" : "";
      return `${long ? "Single cell" : "Cell"} (${layerName(s.li)}, t = ${P[s.ti]}), ${ckName(s.ck)}${s.li < 0 ? ", bag reference" : `, ${v} view${whole}`}`;
    }
    if (s.type === "reading") return `Reading block: L${D.layers[s.li]}, t = ${P[0]} → ${P[P.length - 1]}, ${v} view, ${D.final}`;
    return `Depth block: L${D.layers[0]} → L${D.layers[D.L - 1]} at t = ${P[s.ti]}, point view, ${D.final}`;
  }
  select(sel) {
    this.st.sel = sel;
    if (sel.li != null && sel.li >= 0) this.st.strip = sel.li;
    this.st.focus = -1;
    this.update();
  }
  selT() {   // the position whose tokens the side panel shows
    const s = this.st.sel;
    return s.type === "cell" || s.type === "depth" ? D.positions[s.ti] : D.positions[this.Cn - 1];
  }

  // deep links: #explorer?sel=reading:16 | depth:16 | cell:<init|trained>:<L or bag>:<t> & c=<colouring> & v=p|s|k & pin=<item>
  applyParams(q) {
    const sel = (q.get("sel") || "").split(":"), Li = l => D.layers.indexOf(+l), Ti = t => D.positions.indexOf(+t);
    if (sel[0] === "reading" && Li(sel[1]) >= 0) this.st.sel = { type: "reading", li: Li(sel[1]) };
    else if (sel[0] === "depth" && Ti(sel[1]) >= 0) this.st.sel = { type: "depth", ti: Ti(sel[1]) };
    else if (sel[0] === "cell" && D.ckpts.includes(sel[1]) && Ti(sel[3]) >= 0 && (sel[2] === "bag" || Li(sel[2]) >= 0)) this.st.sel = { type: "cell", ck: sel[1], li: sel[2] === "bag" ? -1 : Li(sel[2]), ti: Ti(sel[3]) };
    if (this.st.sel.li != null && this.st.sel.li >= 0) this.st.strip = this.st.sel.li;
    if (q.get("c") && COLOURINGS[q.get("c")]) this.st.color = q.get("c");
    if (q.get("v") && ["p", "s", "k"].includes(q.get("v"))) this.st.view = q.get("v");
    if (q.get("pin") != null) { this.st.pin = +q.get("pin"); $("ex-pin").value = String(this.st.pin); }
    this.st.focus = -1;
    this.update();
  }

  // ---------------------------------------------------------------- controls
  buildControls() {
    const col = $("ex-color");
    col.addEventListener("change", () => { this.st.color = col.value; this.st.focus = -1; this.update(); });
    $("ex-view").querySelectorAll("button").forEach(b => b.addEventListener("click", () => { this.st.view = b.dataset.v; this.update(); }));
    const pin = $("ex-pin");
    pin.append(h("option", { value: "-1", text: "none" }));
    D.follow.forEach((i, fi) => { const s = D.sources[D.src[i]], k = D.follow.slice(0, fi + 1).filter(j => D.src[j] === D.src[i]).length; pin.append(h("option", { value: i, text: `${s} ${k} (item ${i})` })); });
    pin.addEventListener("change", () => { this.st.pin = +pin.value; this.drawOverlay(); this.renderSnippets(); });
    $("ex-reset").addEventListener("click", () => { this.st.view = "p"; this.select({ type: "reading", li: this.t1li }); });
    const howto = $("ex-howto"), hb = $("ex-howto-btn");
    hb.addEventListener("click", () => { howto.open = !howto.open; });
    howto.addEventListener("toggle", () => { hb.setAttribute("aria-expanded", String(howto.open)); try { localStorage.setItem("mb.explorer.howto-seen", "1"); } catch (e) { /* file:// may refuse */ } });
    let seen = false; try { seen = !!localStorage.getItem("mb.explorer.howto-seen"); } catch (e) { /* ignore */ }
    howto.open = !seen;
  }
  syncControls() {
    $("ex-color").value = this.st.color;
    $("ex-view").querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === this.st.view)));
    $("ex-sel").textContent = this.selName();
    const s = this.st.sel;
    $("ex-view-note").textContent = this.st.view !== "p" && s.type === "depth" ? "depth blocks: point view only; showing point" :
      this.st.view !== "p" && s.type === "cell" && s.li < 0 ? "the bag row has one view" : "";
  }

  // ---------------------------------------------------------------- grid
  buildGrid() {
    const g = $("ex-grid");
    g.replaceChildren();
    this.cellbg = this.panels.map(() => g.appendChild(h("div", { class: "cellbg" })));
    this.canvas = g.appendChild(h("canvas", { class: "base", "aria-label": "Panels of layouts; hover a point to read its snippet, click a panel to select that cell" }));
    this.ov = g.appendChild(h("canvas", { class: "ov" }));
    this.colh = D.positions.map((t, c) => { const e = g.appendChild(h("button", { type: "button", class: "colh lbl-btn", text: `t = ${t}`, title: `Depth block at t = ${t}: L${D.layers[0]} → L${D.layers[D.L - 1]}, trained, point view` })); e.addEventListener("click", () => this.select({ type: "depth", ti: c })); return e; });
    this.rowh = ["bag", ...D.layers.map(l => "L" + l)].map((name, r) => {
      if (r === 0) return g.appendChild(h("div", { class: "rowh rowh-bag", title: "Bag of token embeddings: a reference view, not a layer (no reading block)" }, h("b", { text: "bag" }), "ref."));
      const e = g.appendChild(h("button", { type: "button", class: "rowh lbl-btn", title: `Reading block of ${name}: t = 4 → 128, trained, in the current view` }, h("b", { text: name })));
      e.addEventListener("click", () => this.select({ type: "reading", li: r - 1 }));
      return e;
    });
    this.axisT = g.appendChild(h("div", { class: "axis-t", text: "layer" }));
    this.stripHead = g.appendChild(h("div", { class: "strip-head" }));
    this.stripCols = D.ckpts.map(c => g.appendChild(h("div", { class: "colh strip-colh", text: ckName(c) })));
    this.stripRow = g.appendChild(h("div", { class: "rowh" }));
    this.blockbox = g.appendChild(h("div", { class: "blockbox" }));
    this.canvas.addEventListener("pointermove", e => this.onMove(e));
    this.canvas.addEventListener("pointerleave", () => { this.setHover(-1, -1); hideTip(); });
    this.canvas.addEventListener("click", e => this.onClick(e));
  }
  layout(force = false) {
    const body = $("ex-body"), bar = $("ex-bar");
    if (!body || !body.clientWidth) return;
    const sig = `${body.clientWidth}x${window.innerHeight}`;
    if (sig === this._sig && !force && this._started) { this.renderAll(); return; }
    this._sig = sig;
    document.documentElement.style.setProperty("--bar-h", bar.offsetHeight + "px");
    const small = window.innerWidth <= 640, narrow = window.matchMedia("(max-width: 1099.98px)").matches;
    const W = body.clientWidth - (small ? 24 : 48), rowW = small ? 40 : 56, gap = small ? 2 : 4, colH = 18;
    let cell, side = 360;
    if (narrow) cell = Math.floor((W - rowW) / this.Cn) - gap;
    else {
      const byW = Math.floor((W - 22 - 340 - rowW) / this.Cn) - gap;
      const byH = Math.floor((window.innerHeight - 48 - bar.offsetHeight - 150) / (this.R + 1.4)) - gap;
      cell = Math.min(byW, Math.max(byH, 108));   // real layouts have fine structure: favour size over fitting the viewport
      cell = Math.max(48, Math.min(132, cell));
      side = Math.max(340, Math.min(480, W - 22 - rowW - this.Cn * (cell + gap)));
    }
    cell = Math.max(small ? 44 : 40, Math.min(132, cell));
    body.style.setProperty("--side-w", side + "px");
    const stripTop = colH + this.R * (cell + gap) + 12, stripHeadH = 30;
    this.geom = { cell, gap, rowW, colH, small, pad: Math.max(2, Math.round(cell * .05)), stripY: stripTop + stripHeadH + colH };
    const g = $("ex-grid"), Wg = rowW + this.Cn * (cell + gap), Hg = this.geom.stripY + cell + 6;
    g.style.width = Wg + "px"; g.style.height = Hg + "px";
    this.colh.forEach((e, c) => Object.assign(e.style, { left: rowW + c * (cell + gap) + "px", top: "0px", width: cell + "px" }));
    this.rowh.forEach((e, r) => Object.assign(e.style, { left: "0px", top: colH + r * (cell + gap) + cell / 2 - 12 + "px", width: rowW - 6 + "px" }));
    Object.assign(this.axisT.style, { left: "0px", top: "2px", width: rowW - 6 + "px", textAlign: "right" });
    Object.assign(this.stripHead.style, { left: rowW + "px", top: stripTop + "px", width: this.Cn * (cell + gap) + "px", height: stripHeadH + "px" });
    this.stripCols.forEach((e, s) => Object.assign(e.style, { left: rowW + s * (cell + gap) + "px", top: stripTop + stripHeadH + "px", width: cell + "px" }));
    Object.assign(this.stripRow.style, { left: "0px", top: this.geom.stripY + cell / 2 - 12 + "px", width: rowW - 6 + "px" });
    this.panels.forEach((p, k) => {
      const r = this.panelRect(k, false);
      Object.assign(this.cellbg[k].style, { left: r.x + "px", top: r.y + "px", width: cell + "px", height: cell + "px" });
    });
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    for (const c of [this.canvas, this.ov]) {
      c.width = Math.round(Wg * dpr); c.height = Math.round(Hg * dpr);
      Object.assign(c.style, { left: "0px", top: "0px", width: Wg + "px", height: Hg + "px" });
    }
    this.dpr = dpr;
    this._started = true;
    this.renderAll();
  }
  // the panel's square in CSS px (inner = inside the padding, where dots go)
  panelRect(k, inner = true) {
    const g = this.geom, p = this.panels[k], pad = inner ? g.pad : 0;
    const x = g.rowW + p.col * (g.cell + g.gap), y = p.kind === "grid" ? g.colH + p.row * (g.cell + g.gap) : g.stripY;
    return { x: x + pad, y: y + pad, w: g.cell - 2 * pad };
  }
  panelAt(x, y) {
    for (let k = 0; k < this.panels.length; k++) { const r = this.panelRect(k, false); if (x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.w) return k; }
    return -1;
  }
  // layouts: the bag row and the point view use umap "grid"; the state and stack views their own grids for the layer
  // rows; the strip is always the point layout at t = 128 (other views' colours are painted on it)
  coords(k) {
    const p = this.panels[k], v = this.effView();
    if (p.kind === "strip") return at(A("lay_strip"), this.st.strip, p.si);
    if (p.row === 0 || v === "p") return at(A("lay_grid"), p.row, p.col);
    return at(A(v === "s" ? "lay_grid_s" : "lay_grid_k"), p.row - 1, p.col);
  }
  panelLi(k) { const p = this.panels[k]; return p.kind === "grid" ? p.li : this.st.strip; }

  // ---------------------------------------------------------------- colours
  colours() {
    const N = this.N, rec = this.rec(), c = this.st.color, out = new Array(this.panels.length);
    const sortBy = key => { const o = Uint16Array.from({ length: N }, (_, i) => i); return o.sort((a, b) => key[a] - key[b]); };
    let shared = null;
    if (c === "sets4" || c === "sets2") {
      const lab = rec.lab[c === "sets4" ? 4 : 2], ci = new Uint8Array(N), pri = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        const l = lab[i];
        ci[i] = l < 0 ? C_GREY : C_SET + (l % 8);
        pri[i] = l < 0 ? 0 : 1;
        if (this.st.focus >= 0 && l !== this.st.focus) { ci[i] = C_MISS; pri[i] = -1; }
      }
      shared = { ci, order: sortBy(pri) };
    } else if (c === "psi1" || c === "psi2" || c === "nodal") {
      const v = rec.psi[c === "psi2" ? 1 : 0], ci = new Uint8Array(N), key = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        ci[i] = c === "nodal" ? (v[i] > 0 ? C_DIV + LEVELS - 1 : v[i] < 0 ? C_DIV : C_GREY) : C_DIV + qDiv(v[i]);
        key[i] = Math.abs(v[i]);
      }
      shared = { ci, order: sortBy(key) };
    } else if (c === "src") {
      const ci = new Uint8Array(N);
      for (let i = 0; i < N; i++) ci[i] = C_CAT + D.src[i];
      shared = { ci, order: Uint16Array.from({ length: N }, (_, i) => i) };
    }
    if (shared) { out.fill(shared); return out; }
    const tokCache = {};
    for (let k = 0; k < this.panels.length; k++) {
      const p = this.panels[k], li = this.panelLi(k), ci = new Uint8Array(N), key = new Float32Array(N);
      if (c === "own") {
        if (li < 0) ci.fill(C_MISS);
        else { const own = at(A("s_own"), p.si, li, p.ti); for (let i = 0; i < N; i++) { ci[i] = C_SEQ + qSeq(own[i] / 255); key[i] = own[i]; } }
      } else if (c === "tok") {
        const t = D.positions[p.ti], top = tokCache[t] || (tokCache[t] = this.topTokens(t));
        for (let i = 0; i < N; i++) { const r = top.rank.get(at(A("tok"), i, t - 1)); ci[i] = r === undefined ? C_GREY : C_CAT + r; key[i] = r === undefined ? 0 : 1; }
      }
      out[k] = { ci, order: sortBy(key) };
    }
    return out;
  }
  topTokens(t) {
    const cnt = new Map();
    for (let i = 0; i < this.N; i++) { const v = at(A("tok"), i, t - 1); cnt.set(v, (cnt.get(v) || 0) + 1); }
    const top = [...cnt.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    return { top, rank: new Map(top.map(([v], r) => [v, r])) };
  }

  // ---------------------------------------------------------------- drawing
  update() { this.syncControls(); this.renderAll(); }
  renderAll() {
    if (!this.geom) return;
    this.syncControls();
    this.cols = this.colours();
    this.draw();
    this.drawOverlay();
    this.decorate();
    this.renderKey();
    this.renderSide();
    this.renderCaption();
  }
  draw() {
    const W = this.canvas.width, H = this.canvas.height, dpr = this.dpr, ctx = this.canvas.getContext("2d");
    const img = ctx.createImageData(W, H), buf = new Uint32Array(img.data.buffer);
    const panels = this.panels.map((p, k) => { const r = this.panelRect(k); return { x: r.x * dpr, y: r.y * dpr, w: r.w * dpr, xy: this.coords(k), ci: this.cols[k].ci, order: this.cols[k].order }; });
    paint(buf, W, H, panels, this.pal32, this.N, Math.max(1.1, Math.min(2.2, this.geom.cell / 60)) * dpr / 1.4);
    ctx.putImageData(img, 0, 0);
  }
  drawOverlay() {
    const ctx = this.ov.getContext("2d"), dpr = this.dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.ov.width, this.ov.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ring = (i, col, rad) => {
      for (let k = 0; k < this.panels.length; k++) {
        const r = this.panelRect(k), xy = this.coords(k), x = r.x + r.w * xy[2 * i] / 255, y = r.y + r.w * (1 - xy[2 * i + 1] / 255);
        ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.strokeStyle = "#fff"; ctx.lineWidth = 3.4; ctx.stroke();
        ctx.strokeStyle = col; ctx.lineWidth = 1.7; ctx.stroke();
      }
    };
    if (this.st.pin >= 0) ring(this.st.pin, D.palette.accent, 5);
    if (this.st.hover >= 0) ring(this.st.hover, "#16161a", 4.2);
  }
  decorate() {
    const s = this.st.sel, g = this.geom;
    let ks = [];          // panels in the block (outlined); the strip panel of the same cell is highlighted too
    const stripK = ck => this.R * this.Cn + D.ckpts.indexOf(ck);
    const inStrip = s.type === "cell" && s.li >= 0 && s.li === this.st.strip && s.ti === this.Cn - 1;
    if (s.type === "cell") ks = s.ck === D.final ? [(s.li + 1) * this.Cn + s.ti] : inStrip ? [stripK(s.ck)] : [];
    else if (s.type === "reading") ks = D.positions.map((_, c) => (s.li + 1) * this.Cn + c);
    else ks = D.layers.map((_, li) => (li + 1) * this.Cn + s.ti);
    this.blockbox.style.display = ks.length ? "" : "none";
    if (ks.length) {
      const rs = ks.map(k => this.panelRect(k, false));
      const x0 = Math.min(...rs.map(r => r.x)), y0 = Math.min(...rs.map(r => r.y)), x1 = Math.max(...rs.map(r => r.x + r.w)), y1 = Math.max(...rs.map(r => r.y + r.w));
      Object.assign(this.blockbox.style, { left: x0 - 3 + "px", top: y0 - 3 + "px", width: x1 - x0 + 6 + "px", height: y1 - y0 + 6 + "px" });
    }
    const inb = new Set(ks);
    if (inStrip) inb.add(stripK(s.ck));
    this.cellbg.forEach((e, k) => e.classList.toggle("inb", inb.has(k)));
    this.rowh.forEach((e, r) => e.classList.toggle("inb", (s.type === "reading" && r === s.li + 1) || (s.type === "cell" && s.ck === D.final && r === s.li + 1)));
    this.colh.forEach((e, c) => e.classList.toggle("inb", (s.type === "depth" && c === s.ti) || (s.type === "cell" && s.ck === D.final && c === s.ti)));
    // strip header: which layer, and what clicking does
    const L = D.layers[this.st.strip];
    this.stripHead.replaceChildren(h("span", { class: "strip-title", text: `L${L} at t = ${D.positions[this.Cn - 1]}: HF initialisation and trained` }),
      h("span", { class: "strip-btns" }, h("span", { class: "ctl-sub", text: this.st.view === "p" ? "click a panel for that single cell (point layouts)" : `click a panel for that ${VIEW_NAME[this.st.view]} cell; dots placed by the point layouts` })));
    this.stripRow.replaceChildren(h("b", { text: "L" + L }), "t = 128");
  }

  // ---------------------------------------------------------------- pointer
  hit(e) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
  nearest(k, x, y, maxd) {
    const r = this.panelRect(k), xy = this.coords(k);
    let best = -1, bd = maxd * maxd;
    for (let i = 0; i < this.N; i++) { const dx = r.x + r.w * xy[2 * i] / 255 - x, dy = r.y + r.w * (1 - xy[2 * i + 1] / 255) - y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
    return best;
  }
  onMove(e) {
    const { x, y } = this.hit(e), k = this.panelAt(x, y);
    if (k < 0) { this.setHover(-1, -1); hideTip(); return; }
    const i = this.nearest(k, x, y, Math.max(6, this.geom.cell * .07));
    this.setHover(i, k);
    if (i >= 0) showTip(e.clientX, e.clientY, this.pointTip(i, k)); else hideTip();
  }
  onClick(e) {
    const { x, y } = this.hit(e), k = this.panelAt(x, y);
    if (k < 0) return;
    const p = this.panels[k];
    if (p.kind === "grid") this.select({ type: "cell", ck: D.final, li: p.li, ti: p.ti });
    else this.select({ type: "cell", ck: D.ckpts[p.si], li: this.st.strip, ti: this.Cn - 1 });
  }
  setHover(i, k) {
    if (i === this.st.hover && k === this.st.hoverPanel) return;
    this.st.hover = i; this.st.hoverPanel = k;
    this.drawOverlay();
    this.renderSnippets();
  }
  pointTip(i, k) {
    const p = this.panels[k], li = this.panelLi(k), t = D.positions[p.ti];
    const rows = [];
    const rec = this.rec(), c = this.st.color;
    if (c === "sets4" || c === "sets2") { const l = rec.lab[c === "sets4" ? 4 : 2][i]; rows.push({ name: "set (selected block)", value: l < 0 ? "unassigned" : `set ${l + 1}` }); }
    if (c === "psi1" || c === "psi2" || c === "nodal") rows.push({ name: c === "psi2" ? "ψ₂ / max|ψ₂|" : "ψ₁ / max|ψ₁|", value: fmt(rec.psi[c === "psi2" ? 1 : 0][i], 2) });
    if (c === "own" && li >= 0) rows.push({ name: "own-source walk share", value: fmt(at(A("s_own"), p.si, li, p.ti)[i] / 255, 2) });
    rows.push({ name: "token at t", value: JSON.stringify(showTok(token(i, t))) });
    const d = tipRows(`item ${i} · ${D.sources[D.src[i]]} · ${layerName(li)}, t = ${t}${p.kind === "strip" ? ", " + ckName(D.ckpts[p.si]) : ""}`, rows);
    d.append(h("div", { class: "tt-snip" }, snippetEl(i, t, { before: 18, after: 4 })));
    return d;
  }

  // ---------------------------------------------------------------- side panel
  renderKey() {
    const k = $("ex-key"), c = this.st.color;
    k.replaceChildren();
    const dot = (col, label) => h("span", { class: "kk" }, h("i", { style: { background: col } }), label);
    if (c === "sets4" || c === "sets2") { const r = c === "sets4" ? 4 : 2; for (let j = 0; j < r; j++) k.append(dot(D.pal.set(j), `set ${j + 1}`)); k.append(dot(D.palette.unassigned, "unassigned")); }
    else if (c === "psi1" || c === "psi2") k.append(h("span", { class: "kt", text: "−max" }), h("i", { class: "kbar", style: { background: D.pal.divGradient() } }), h("span", { class: "kt", text: "+max" }));
    else if (c === "nodal") k.append(dot(D.pal.div(-1), "ψ₁ < 0"), dot(D.pal.div(1), "ψ₁ > 0"));
    else if (c === "src") D.sources.forEach((s, j) => k.append(dot(D.srcCol[j], s)));
    else if (c === "own") k.append(h("span", { class: "kt", text: "0" }), h("i", { class: "kbar", style: { background: D.pal.seqGradient() } }), h("span", { class: "kt", text: "1 (all of the walk stays in the item's source)" }));
    else if (c === "tok") k.append(h("span", { class: "kt", text: "each panel: its 10 most frequent tokens at t in colour, the rest grey (legend in the side panel)" }));
  }
  renderSide() {
    const side = $("ex-side");
    side.replaceChildren(this.setsCard(), this.blockCard(), this.retentionCard(), this.snippetCard());
  }
  setsCard() {
    const rec = this.rec(), c = this.st.color, t = this.selT();
    const card = h("div", { class: "card card-sets" });
    const comp = counts => { const tot = counts.reduce((a, b) => a + b, 0) || 1, bar = h("div", { class: "comp comp-lg" });
      counts.forEach((v, k) => v && bar.append(h("i", { style: { width: (100 * v / tot) + "%", background: D.srcCol[k] } })));
      bar.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows("source make-up (validation)", counts.map((v, k) => ({ name: D.sources[k], value: String(v), color: D.srcCol[k] })))));
      bar.addEventListener("pointerleave", hideTip); return bar; };
    const snip = (i, tt) => h("div", { class: "snip-row" }, srcSwatch(i), snippetEl(i, tt, { before: 14, after: 3 }));
    if (c === "sets4" || c === "sets2") {
      const r = c === "sets4" ? 4 : 2, lab = rec.lab[r];
      card.append(h("p", { class: "card-title", text: `Coherent sets, r = ${r}: snippets and source make-up` }), h("p", { class: "card-sub", text: this.selName(true) }));
      for (let j = 0; j <= r; j++) {
        const un = j === r, members = []; for (let i = 0; i < this.N; i++) if (un ? lab[i] < 0 : lab[i] === j) members.push(i);
        const counts = D.sources.map((_, s) => members.filter(i => D.src[i] === s).length);
        const row = h("div", { class: "set-row" + (this.st.focus === j ? " on" : this.st.focus >= 0 ? " dim" : "") + (un ? " minor" : "") });
        const head = h("div", { class: "set-head", role: un ? null : "button", title: un ? "" : "click: show only this set in the grid" },
          h("span", { class: "sw", style: { background: un ? D.palette.unassigned : D.pal.set(j) } }), h("span", { class: "set-name", text: un ? "unassigned" : `set ${j + 1}` }), h("span", { class: "n", text: String(members.length) }), comp(counts));
        if (!un) head.addEventListener("click", () => { this.st.focus = this.st.focus === j ? -1 : j; this.renderAll(); });
        row.append(head);
        if (!un) topMembers(rec, r, j).slice(0, this.st.focus === j ? 8 : 2).forEach(i => row.append(snip(i, t)));
        card.append(row);
      }
      card.append(this.sourceKey(), h("p", { class: "card-note", text: `Snippets: each set's members with the highest SEBA value, shown up to token t = ${t} (underlined; later tokens grey). Bars: source make-up, validation only; sources play no part in finding the sets.` }));
    } else if (c === "psi1" || c === "psi2" || c === "nodal") {
      const k = c === "psi2" ? 1 : 0, v = rec.psi[k], o = Array.from({ length: this.N }, (_, i) => i).sort((a, b) => v[a] - v[b]);
      card.append(h("p", { class: "card-title", text: `${c === "nodal" ? "Sign of ψ₁" : "Eigenfunction ψ" + (k + 1)}: the two ends` }), h("p", { class: "card-sub", text: `${this.selName(true)}; λ${k + 1} = ${fmt(rec.vals[k + 1], 4)}` }));
      for (const [label, ids, col] of [["most negative", o.slice(0, 4), D.pal.div(-1)], ["most positive", o.slice(-4).reverse(), D.pal.div(1)]]) {
        const counts = D.sources.map((_, s) => o.slice(label === "most negative" ? 0 : -100).filter(i => D.src[i] === s).length);
        card.append(h("div", { class: "set-row" }, h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: label }), h("span", { class: "n", text: "" }), comp(counts)), ...ids.map(i => snip(i, t))));
      }
      card.append(this.sourceKey(), h("p", { class: "card-note", text: "Bars: source make-up of the 100 items at each end (validation only)." }));
    } else if (c === "tok") {
      const tt = this.st.hoverPanel >= 0 ? D.positions[this.panels[this.st.hoverPanel].ti] : t, top = this.topTokens(tt);
      card.append(h("p", { class: "card-title", text: `Most frequent tokens at t = ${tt}` }), h("p", { class: "card-sub", text: "Hover a panel to see its position's tokens. Tokens are inputs: this colouring is label-free." }));
      const tb = h("div", { class: "tok-list" });
      top.top.forEach(([v, n], r) => tb.append(h("span", { class: "kk" }, h("i", { style: { background: D.palette.classes[r] } }), h("code", { text: JSON.stringify(showTok(D.tokstr[v])) }), h("span", { class: "mute", text: ` ${n}` }))));
      card.append(tb);
    } else if (c === "src") {
      card.append(h("p", { class: "card-title", text: "Source (validation only)" }), this.sourceKey(),
        h("p", { class: "card-note", text: `${D.N / D.sources.length} snippets per source. The method never sees the source; this colouring is for checking only.` }));
    } else if (c === "own") {
      card.append(h("p", { class: "card-title", text: "Own-source walk share (validation)" }),
        h("p", { class: "card-note", text: `Per item and panel: the fraction of the item's row of that cell's walk P_c that lands on items of the same source. A walk that ignores sources gives about ${((D.N / D.sources.length - 1) / (D.N - 1)).toFixed(2)}. The bag row has no walk share stored.` }));
    }
    return card;
  }
  sourceKey() { return h("div", { class: "class-key" }, ...D.sources.map((s, k) => h("span", {}, h("i", { style: { background: D.srcCol[k] } }), s))); }
  blockCard() {
    const rec = this.rec(), s = this.st.sel, card = h("div", { class: "card" });
    card.append(h("p", { class: "card-title", text: this.selName(true) }));
    const nCells = s.type === "cell" ? 1 : s.type === "reading" ? D.T : D.L;
    const lead = { cell: "One cell on its own: one layer, one position, one checkpoint, one view, one geometry.",
      reading: "The same layer at six positions: links that persist as more tokens are read dominate the average.",
      depth: "One position at six layers: links that persist through depth dominate the average." }[s.type];
    card.append(h("p", { class: "card-sub", text: lead }));
    const tb = h("table", { class: "mini" }, h("thead", {}, h("tr", {}, h("th", { text: "k" }), ...[1, 2, 3, 4, 5, 6, 7, 8].map(k => h("th", { text: String(k) })))));
    tb.append(h("tbody", {}, h("tr", {}, h("td", { text: "λₖ" }), ...[1, 2, 3, 4, 5, 6, 7, 8].map(k => h("td", { text: fmt(rec.vals[k], 3) }))),
      h("tr", {}, h("td", { text: "PN(ψₖ)" }), ...[0, 1, 2, 3, 4, 5, 6, 7].map(k => h("td", { text: Number.isFinite(rec.pn[k]) ? rec.pn[k].toFixed(0) : "n/a" })))));
    card.append(h("div", { class: "table-scroll" }, tb));
    // walk shares
    let T, T0, Tb, W, W0, Wb;
    if (s.type === "cell") {
      const view = D.vIdx[this.st.view], si = D.ckpts.indexOf(s.ck);
      [T, T0, Tb, W, W0, Wb] = ["T", "T0", "T_band", "W", "W0", "W_band"].map(n => stat(n, si, view, s.li, s.ti));
    } else ({ T, T0, Tb, W, W0, Wb } = rec.stat);
    const jac = h("div", { class: "jac" },
      h("div", { class: "jrow" }, h("span", { text: "excess walk share T̃ (token at t)" }), h("span", { class: "jv", text: fmt(excess(T, T0), 3) })),
      h("div", { class: "jrow" }, h("span", { class: "jnull", text: `raw ${g3(T)}; chance ${g3(T0)}; permutation band ${g3(Tb[0])}–${g3(Tb[1])}` })),
      h("div", { class: "jrow" }, h("span", { text: "excess walk share W̃ (source; validation)" }), h("span", { class: "jv", text: fmt(excess(W, W0), 3) })),
      h("div", { class: "jrow" }, h("span", { class: "jnull", text: `raw ${g3(W)}; chance ${g3(W0)}; permutation band ${g3(Wb[0])}–${g3(Wb[1])}` })));
    [2, 4].forEach((r, a) => { const x = rec.sj.subarray(a * 5, a * 5 + 5);
      jac.append(h("div", { class: "jrow" }, h("span", { text: `sets r = ${r} vs sources: matched Jaccard` }), h("span", { class: "jv", text: fmt(x[0], 2) })),
        h("div", { class: "jrow" }, h("span", { class: "jnull", text: `null ${fmt(x[1], 2)} (97.5 %: ${fmt(x[3], 2)}); size ceiling ${fmt(x[4], 2)}` }))); });
    // label-free comparison with T1's block (point reading block, L16), computed here (copied matchedJaccard; no browser null)
    const t1 = blockRecord("reading", "p", this.t1li);
    if (t1 && t1 !== rec) {
      const mj = matchedJaccard(rec.lab[4], 4, t1.lab[4], 4);
      jac.append(h("div", { class: "jrow" }, h("span", { text: `sets r = 4 vs T1's (point reading block L${D.t1_layer}; label-free)` }), h("span", { class: "jv", text: fmt(mj.mean, 2) })));
    }
    card.append(jac, h("p", { class: "card-note", text: `${nCells} cell${nCells > 1 ? "s" : ""}; in each, a union 15-nearest-neighbour graph scaled by Sinkhorn into a lazy walk; M̄ is their mean. λₖ: how slowly ψₖ changes along the walk (1 = not at all); PN: how many snippets ψₖ is spread over.` }));
    return card;
  }
  retentionCard() {
    const rec = this.rec(), s = this.st.sel, c = this.st.color, card = h("div", { class: "card" });
    const sets = c === "sets4" || c === "sets2", r = c === "sets2" ? 2 : 4;
    const rowsIdx = sets ? setRows(r) : [0, 1, 2, 3], rowName = j => sets ? `set ${rowsIdx.indexOf(j) + 1}` : `ψ${j + 1}`;
    card.append(h("p", { class: "card-title", text: `Retention ρ_c of ${sets ? `the r = ${r} SEBA vectors` : "ψ₁ … ψ₄"} in each cell` }));
    const table = (arr, cols, hl, title) => {
      const X = cols.length, tb = h("table", { class: "mini ret" }, h("thead", {}, h("tr", {}, h("th", { text: title }), ...cols.map((x, i) => h("th", { class: hl.includes(i) ? "inb" : "", text: x })))));
      const body = h("tbody");
      rowsIdx.forEach(j => body.append(h("tr", {}, h("td", { text: rowName(j) }), ...cols.map((_, i) => { const v = arr[j * 6 + i]; const q = (v - .5) / .5;
        return h("td", { class: "rc" + (hl.includes(i) ? " inb" : ""), style: { background: Number.isFinite(v) ? D.pal.seq(q) : "", color: q > .55 ? "#fff" : "" }, text: Number.isFinite(v) ? v.toFixed(2) : "–" }); }))));
      tb.append(body);
      return h("div", { class: "table-scroll" }, tb);
    };
    const posCols = D.positions.map(String), layCols = D.layers.map(l => "L" + l);
    if (s.type === "cell") {
      const R = k => sets ? rec.ret[k].sets : rec.ret[k].psi;
      card.append(table(R(0), posCols, [s.ti], "t"));
      if (s.li >= 0) card.append(table(R(1), layCols, [s.li], "layer"));
      card.append(h("p", { class: "card-note", text: "A single cell's functions, carried into the other cells of its row (t) and its column (layer), same view and checkpoint. Bold column: the cell itself." }));
    } else {
      const arr = sets ? rec.ret.sets : rec.ret.psi;
      if (s.type === "reading") card.append(table(arr, posCols, posCols.map((_, i) => i), "t"));
      else card.append(table(arr, layCols, layCols.map((_, i) => i), "layer"));
      card.append(h("p", { class: "card-note", text: "ρ_c(f) = fᵀM_c f / fᵀf in each cell along the block's axis (bold: cells in the block); 1: constant on every walk neighbourhood. A random mean-zero function gets about 0.5; SEBA vectors are non-negative, so their chance level is higher." }));
    }
    return card;
  }
  snippetCard() {
    this._snipCard = h("div", { class: "card" });
    this.renderSnippets();
    return this._snipCard;
  }
  renderSnippets() {
    const card = this._snipCard;
    if (!card) return;
    const blocks = [h("p", { class: "card-title", text: "Snippets" })];
    const show = (label, i, t) => blocks.push(h("div", { class: "snip-card" }, h("div", { class: "imgs-label" }, srcSwatch(i), h("b", { text: label }), h("span", { class: "mute", text: `item ${i}, ${D.sources[D.src[i]]}; up to t = ${t}` })),
      snippetEl(i, t, { before: 40, after: 10, marks: D.positions })));
    const hk = this.st.hoverPanel;
    if (this.st.hover >= 0 && hk >= 0) show("hovered", this.st.hover, D.positions[this.panels[hk].ti]);
    if (this.st.pin >= 0) show("pinned", this.st.pin, this.selT());
    if (blocks.length === 1) blocks.push(h("p", { class: "card-note", text: "Hover a point to read its snippet up to that panel's position (token t underlined, later tokens grey), or pin one of the follow items in the bar." }));
    card.replaceChildren(...blocks);
  }
  renderCaption() {
    const s = this.st.sel, c = this.st.color, cap = $("ex-caption");
    const from = "computed on the outlined block only and painted on every panel";
    const what = {
      sets4: `the item's coherent set, r = 4 (SEBA on ψ₀ … ψ₃ of the block's M̄), ${from}; grey: unassigned`,
      sets2: `the item's coherent set, r = 2 (SEBA on ψ₀, ψ₁), ${from}; grey: unassigned`,
      psi1: `ψ₁ of the block's M̄ scaled by its largest |value|, ${from}`, psi2: `ψ₂ of the block's M̄ scaled by its largest |value|, ${from}`,
      nodal: `the sign of ψ₁ (nodal domains), ${from}`, tok: "the token each item reads at the panel's position t; the 10 most frequent tokens there in colour, the rest grey (label-free: tokens are inputs)",
      src: "the item's source (validation only)", own: "the share of the item's walk in that panel's point-view cell that stays inside its source (validation; point view only; the bag row has none)",
    }[c];
    cap.innerHTML = "";
    cap.append(h("b", { text: "Block: " }), `${this.selName(true)}. `, h("b", { text: "Colour: " }), what + ". ",
      h("b", { text: "Layout: " }), `one dot per snippet (${this.N}); each row is one AlignedUMAP run across its six positions (n_neighbors 15, top 50 PCs), in the ${VIEW_NAME[this.effView()]} view for the layer rows (the bag row has its own); the strip is one run across the two checkpoints, point view; positions are for display only.`);
  }
}
