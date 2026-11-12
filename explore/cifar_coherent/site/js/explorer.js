// The explorer: grid of UMAP cells (rows = checkpoints, columns = sites) coloured by the selected block.
import * as D from "./data.js";
import { Palette, divLUT, seqLUT, qDiv, qSeq, hexRgb } from "./colors.js";
import { h, el, showTip, hideTip, tipRows, lineChart, scale, niceTicks, fmtStep } from "./charts.js";

const $ = id => document.getElementById(id);
const DIM = "#e7e6e1";
const BLOCK_KINDS = new Set(["seba", "nodal", "eig"]);
const DIAG = { loss: "loss", jac: "jac_fro", gcos: "grad_cos", gdot: "grad_dot" };
// Per-image pixel statistics (colour value "px:<stat>"), from assets/questions.json -> q_imgstats.per_image, loaded
// on first use. Definitions as in analysis_findings.py (section 3); pixel values in [0, 1]; Y = 0.299 R + 0.587 G +
// 0.114 B; border = the 448 pixels within 4 px of the edge, centre = the central 16 x 16. div: signed, diverging scale.
export const IMGSTATS = {
  mean_r: { label: "mean R", def: "mean of the red channel over the 1,024 pixels" },
  mean_g: { label: "mean G", def: "mean of the green channel over the 1,024 pixels" },
  mean_b: { label: "mean B", def: "mean of the blue channel over the 1,024 pixels" },
  luminance: { label: "luminance Y", def: "mean of Y over the pixels" },
  contrast: { label: "RMS contrast of Y", def: "standard deviation of Y over the pixels" },
  saturation: { label: "HSV saturation", def: "mean over pixels of (max − min) / max of (R, G, B)" },
  bc_r: { label: "border − centre R", def: "border-ring mean minus centre mean of R", div: true },
  bc_g: { label: "border − centre G", def: "border-ring mean minus centre mean of G", div: true },
  bc_b: { label: "border − centre B", def: "border-ring mean minus centre mean of B", div: true },
  bc_lum: { label: "border − centre Y", def: "border-ring mean minus centre mean of Y", div: true },
  bc_dist: { label: "|border − centre| RGB", def: "Euclidean norm of the border − centre differences of (R, G, B)" },
  hf_frac: { label: "high-frequency share of Y", def: "share of the DFT energy of Y − mean(Y) at radius > 8 cycles per image among all non-zero frequencies" },
};
const isPx = c => typeof c === "string" && c.startsWith("px:");
// load what a colouring needs beyond the shared context (the pixel statistics); call before colouring()
export async function prepare(ctx, st) {
  if (!isPx(st.color) || ctx.imgstats) return;
  const q = (await D.json("assets/questions.json")).q_imgstats;
  if (!q || !q.per_image) throw new Error("assets/questions.json has no q_imgstats.per_image");
  ctx.imgstats = { per: q.per_image, labels: q.stat_labels || {}, range: {} };
  for (const [k, v] of Object.entries(q.per_image)) {
    const a = Float64Array.from(v);
    ctx.imgstats.range[k] = IMGSTATS[k]?.div ? Math.max(1e-9, quantile(Array.from(a, Math.abs), .99)) : [quantile(a, .01), quantile(a, .99)];
  }
}
const fmt = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : "n/a";
const pct = v => Number.isFinite(v) ? (100 * v).toFixed(0) + "%" : "n/a";

// ------------------------------------------------------------------------------------------- shared context
let CTX = null;
export async function context() {
  if (CTX) return CTX;
  CTX = (async () => {
    const m = await D.meta();
    const [labels, layout0, pts] = await Promise.all([D.labels(), D.layout(0), D.points()]);
    const pal = new Palette(m.palette);
    const pairIdx = { layer: {}, time: {} };
    m.layer_pairs.forEach(([a, b], i) => pairIdx.layer[a + "," + b] = i);
    m.time_pairs.forEach(([a, b], i) => pairIdx.time[a + "," + b] = i);
    const ctx = { m, pal, labels, layouts: { 0: layout0 }, pts, pairIdx, T: m.steps.length, S: m.sites.length, N: m.n };
    ctx.diagRange = diagRanges(ctx);
    D.sprite().then(im => { ctx.sprite = im; });
    return ctx;
  })();
  return CTX;
}

function quantile(arr, q) {
  const a = Float64Array.from(arr.filter(Number.isFinite)).sort();
  return a.length ? a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))] : NaN;
}
function diagRanges(ctx) {
  const r = {}, p = ctx.pts;
  if (p.loss) { const l = Array.from(p.loss, v => Math.log10(Math.max(v, 1e-4))); r.loss = [quantile(l, .01), quantile(l, .99)]; }
  if (p.jac_fro) { const l = Array.from(p.jac_fro, v => Math.log10(Math.max(v, 1e-8))); r.jac = [quantile(l, .01), quantile(l, .99)]; }
  if (p.grad_cos) { const a = Array.from(p.grad_cos, Math.abs); r.gcos = Math.max(1e-6, quantile(a, .99)); }
  if (p.grad_dot) { const a = Array.from(p.grad_dot, Math.abs); r.gdot = Math.max(1e-12, quantile(a, .99)); }
  return r;
}

export function pairOf(ctx, st) { return ctx.pairIdx[st.fam][st.a + "," + st.b]; }
export function blockOf(ctx, st) { return D.block(st.fam, st.anchor, st.mode, pairOf(ctx, st)); }
// "Layer block" / "Time block" / "Single-snapshot block" (glossary terms)
export const blockKind = st => st.a === st.b ? "Single-snapshot block" : st.fam === "layer" ? "Layer block" : "Time block";
export function blockLabel(ctx, st) {
  const { m } = ctx;
  if (st.fam === "layer") {
    const span = st.a === st.b ? m.sites[st.a] : `${m.sites[st.a]} → ${m.sites[st.b]}`;
    return `${span} at step ${m.steps[st.anchor].toLocaleString("en")}`;
  }
  const span = st.a === st.b ? `step ${m.steps[st.a].toLocaleString("en")}` : `steps ${m.steps[st.a].toLocaleString("en")} → ${m.steps[st.b].toLocaleString("en")}`;
  return `${m.sites[st.anchor]}, ${span}`;
}
export const inBlock = (st, t, s) => st.fam === "layer" ? (t === st.anchor && s >= st.a && s <= st.b) : (s === st.anchor && t >= st.a && t <= st.b);
const funIndex = (m, kind, k, ri, j) => kind === "psi" ? k - 1 : 8 + m.seba_r.slice(0, ri).reduce((x, y) => x + y, 0) + j;

// one probe image from the sprite sheet, as a 32 x 32 canvas (scaled by CSS)
export function thumb(ctx, i) {
  const cv = document.createElement("canvas"); cv.width = cv.height = 32; cv.dataset.i = i;
  const cols = ctx.m.sprite_cols;
  const draw = im => cv.getContext("2d").drawImage(im, (i % cols) * 32, Math.floor(i / cols) * 32, 32, 32, 0, 0, 32, 32);
  if (ctx.sprite) draw(ctx.sprite); else D.sprite().then(draw);
  return cv;
}

// tours from content/tours.json, inlined by build_site.py
export function tours() {
  try { return JSON.parse(document.getElementById("tours-data")?.textContent || "[]"); } catch (e) { console.error(e); return []; }
}

// ------------------------------------------------------------------------------------------- colouring
// Returns {ci: [T] Uint8Array(N), order: [T] Uint16Array(N), cols: css[], legend}
export function colouring(ctx, st, blk) {
  const { m, pal, labels, pts, T, N } = ctx;
  const same = (ci, order) => ({ ci: Array(T).fill(ci), order: Array(T).fill(order) });
  const orderBy = key => Uint16Array.from({ length: N }, (_, i) => i).sort((a, b) => key[a] - key[b]);
  const hashOrder = (() => { const o = Uint16Array.from({ length: N }, (_, i) => i); const k = new Float64Array(N); for (let i = 0; i < N; i++) k[i] = Math.sin(i * 12.9898) * 43758.5453 % 1; return o.sort((a, b) => k[a] - k[b]); })();
  let out;
  const c = st.color;
  if (c === "seba") {
    const ri = m.seba_r.indexOf(st.r), r = st.r, lab = blk.lab.subarray(ri * N, (ri + 1) * N);
    const vecs = D.sebaVectors(m, blk, ri);
    const ci = new Uint8Array(N), key = new Float64Array(N);
    for (let i = 0; i < N; i++) { const l = lab[i]; ci[i] = l < 0 ? r : l; key[i] = l < 0 ? -1 : vecs[l][i]; }
    const cols = [...Array(r)].map((_, j) => pal.set(j)).concat([pal.unassigned]);
    out = { ...same(ci, orderBy(key)), cols, groups: { ids: ci, n: r + 1, names: [...Array(r)].map((_, j) => `set ${j + 1}`).concat(["unassigned"]), unassigned: r } };
  } else if (c === "nodal" || c === "eig") {
    const v = D.psi(m, blk, st.k);
    let mx = 0; for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(v[i]));
    const key = Float64Array.from(v, x => Math.abs(x));
    if (c === "eig") {
      const ci = Uint8Array.from(v, x => qDiv(x / mx));
      out = { ...same(ci, orderBy(key)), cols: divLUT(pal), cont: { kind: "div", lo: -1, hi: 1, label: `ψ${st.k} / max|ψ${st.k}|` } };
    } else {
      const ci = Uint8Array.from(v, x => x > 0 ? 0 : x < 0 ? 1 : 2);
      out = { ...same(ci, orderBy(key)), cols: [pal.p.diverging.pos, pal.p.diverging.neg, pal.unassigned],
        groups: { ids: ci, n: 3, names: [`ψ${st.k} > 0`, `ψ${st.k} < 0`, `ψ${st.k} = 0`], unassigned: 2 } };
    }
  } else if (isPx(c)) {
    const k = c.slice(3), v = ctx.imgstats?.per[k], R = ctx.imgstats?.range[k];
    if (!v) throw new Error(`no image statistic "${k}"`);
    const lab = ctx.imgstats.labels[k] || IMGSTATS[k]?.label || k;
    let ci, key, cols, cont;
    if (IMGSTATS[k]?.div) { ci = Uint8Array.from(v, x => qDiv(x / R)); key = Float64Array.from(v, Math.abs); cols = divLUT(pal); cont = { kind: "div", lo: -R, hi: R, label: lab }; }
    else { const [lo, hi] = R; ci = Uint8Array.from(v, x => qSeq((x - lo) / (hi - lo || 1))); key = Float64Array.from(v); cols = seqLUT(pal); cont = { kind: "seq", lo, hi, label: lab }; }
    out = { ...same(ci, orderBy(key)), cols, cont };
  } else if (c === "class") {
    const ci = Uint8Array.from(labels);
    out = { ...same(ci, hashOrder), cols: pal.classes.slice(), groups: { ids: ci, n: 10, names: m.classes.slice(), classes: true } };
  } else if (c === "correct") {
    const ci = [], order = [];
    for (let t = 0; t < T; t++) {
      const tt = st.at === "final" ? T - 1 : t;
      const a = Uint8Array.from({ length: N }, (_, i) => pts.correct[tt * N + i] ? 0 : 1);
      ci.push(a); order.push(orderBy(a));
    }
    out = { ci, order, cols: ["#c9c8c1", "#d03b3b"], groups: { ids: null, perRow: true, n: 2, names: ["correct", "wrong"] } };
  } else {
    const f = DIAG[c], arr = pts[f], ci = [], order = [];
    let tf, cols, cont;
    if (c === "loss" || c === "jac") {
      const [lo, hi] = ctx.diagRange[c];
      tf = v => qSeq((Math.log10(Math.max(v, c === "loss" ? 1e-4 : 1e-8)) - lo) / (hi - lo));
      cols = seqLUT(pal); cont = { kind: "seq", lo, hi, log: true, label: c === "loss" ? "cross-entropy loss" : "‖∂ logits / ∂ input‖_F" };
    } else {
      const R = ctx.diagRange[c];
      tf = v => qDiv(v / R); cols = divLUT(pal);
      cont = { kind: "div", lo: -R, hi: R, label: c === "gcos" ? "cos(g_i, mean g)" : "g_i · mean g" };
    }
    for (let t = 0; t < T; t++) {
      const tt = st.at === "final" ? T - 1 : t, a = new Uint8Array(N), key = new Float64Array(N);
      for (let i = 0; i < N; i++) { const v = arr[tt * N + i]; a[i] = tf(v); key[i] = cont.kind === "div" ? Math.abs(a[i] - 20) : a[i]; }
      ci.push(a); order.push(orderBy(key));
    }
    out = { ci, order, cols, cont };
  }
  // focus: dim everything outside the focused group, draw it first
  if (out.groups && st.focus != null && st.focus < out.groups.n) {
    const D_ = out.cols.length; out.cols = out.cols.concat([DIM]);
    out.ci = out.ci.map((a, t) => {
      const b = Uint8Array.from(a, x => x === st.focus ? x : D_);
      return b;
    });
    out.order = out.ci.map((a, t) => Uint16Array.from(out.order[t]).sort((x, y) => (a[x] === D_ ? 0 : 1) - (a[y] === D_ ? 0 : 1)));
  }
  out.pal32 = new Uint32Array(out.cols.map(css32));
  return out;
}
const _c32 = new Map();
function css32(css) {
  if (_c32.has(css)) return _c32.get(css);
  let r, g, b;
  if (css.startsWith("#")) [r, g, b] = hexRgb(css);
  else [r, g, b] = css.match(/\d+/g).map(Number);
  const v = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  _c32.set(css, v); return v;
}

// ------------------------------------------------------------------------------------------- dot painter
function stamp(radius) {
  const offs = [], R = Math.ceil(radius);
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) if (dx * dx + dy * dy <= radius * radius + .35) offs.push(dx, dy);
  return offs.length ? offs : [0, 0];
}
// paint all dots of the given cells into an ImageData-backed buffer
export function paint(buf, W, H, cells, coords, col, N, radius) {
  const offs = stamp(radius), no = offs.length;
  for (const c of cells) {
    const ci = col.ci[c.t], order = col.order[c.t], pal = col.pal32, base = (c.t * c.S + c.s) * N * 2;
    for (let q = 0; q < N; q++) {
      const i = order[q], px = Math.round(c.x + c.w * coords[base + 2 * i]), py = Math.round(c.y + c.w * (1 - coords[base + 2 * i + 1]));
      const v = pal[ci[i]];
      for (let o = 0; o < no; o += 2) {
        const x = px + offs[o], y = py + offs[o + 1];
        if (x >= 0 && y >= 0 && x < W && y < H) buf[y * W + x] = v;
      }
    }
  }
}
export function toCoords(u16) { const f = new Float32Array(u16.length); for (let i = 0; i < u16.length; i++) f[i] = u16[i] / 65535; return f; }

// ------------------------------------------------------------------------------------------- matched Jaccard
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
function jaccardMatrix(A, nA, B, nB) {
  const inter = Array.from({ length: nA }, () => new Float64Array(nB)), ca = new Float64Array(nA), cb = new Float64Array(nB);
  for (let i = 0; i < A.length; i++) { const a = A[i], b = B[i]; if (a >= 0) ca[a]++; if (b >= 0) cb[b]++; if (a >= 0 && b >= 0) inter[a][b]++; }
  return inter.map((row, i) => Array.from(row, (x, j) => { const u = ca[i] + cb[j] - x; return u > 0 ? x / u : 0; }));
}
export function matchedJaccard(A, nA, B, nB, nNull = 30) {
  const score = (A_) => {
    const J = jaccardMatrix(A_, nA, B, nB), n = Math.max(nA, nB);
    const sq = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i < nA && j < nB) ? J[i][j] : 0));
    const asg = hungarianMax(sq);
    const per = [...Array(nA)].map((_, i) => ({ j: asg[i] < nB ? asg[i] : -1, J: asg[i] < nB ? J[i][asg[i]] : 0 }));
    return { mean: per.reduce((s, x) => s + x.J, 0) / nA, per };
  };
  const real = score(A);
  let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const nulls = [];
  for (let k = 0; k < nNull; k++) {
    const P = Int8Array.from(A);
    for (let i = P.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = P[i]; P[i] = P[j]; P[j] = t; }
    nulls.push(score(P).mean);
  }
  nulls.sort((a, b) => a - b);
  return { ...real, nullMean: nulls.reduce((s, x) => s + x, 0) / nNull, null95: nulls[Math.floor(.95 * (nNull - 1))] };
}

// ------------------------------------------------------------------------------------------- the explorer
export class Explorer {
  constructor(root) { this.root = root; this.ready = this.init(); }

  async init() {
    const ctx = this.ctx = await context();
    const { m, T, S } = ctx;
    this.st = {
      fam: "layer", anchor: T - 1, a: 0, b: S - 1, mode: m.default_mode, color: "seba", k: 1, r: m.seba_r.includes(4) ? 4 : m.seba_r[0],
      kind: "set", focus: null, j: null, at: "own", lay: 0, ps: 1.6, cmp: null, tap: "end",
    };
    this.coords = { 0: toCoords(ctx.layouts[0]) };
    this.cur = this.coords[0];
    this.hover = -1; this.lensCell = [T - 1, S - 1];
    this.buildControls();
    this.buildGrid();
    new ResizeObserver(() => this.layout()).observe($("ex-body"));
    new ResizeObserver(() => { document.documentElement.style.setProperty("--bar-h", $("ex-bar").offsetHeight + "px"); }).observe($("ex-bar"));
    window.addEventListener("resize", () => this.layout());
    this.layout();
  }

  // ---------------------------------------------------------------- controls
  buildControls() {
    const { m, ctx } = { m: this.ctx.m, ctx: this.ctx };
    const segInit = (id, get, set) => {
      const seg = $(id);
      seg.querySelectorAll("button").forEach(b => {
        b.setAttribute("role", "radio");
        b.addEventListener("click", () => { if (!b.disabled) { set(b.dataset.v); this.update(); } });
      });
      seg._sync = () => seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === String(get()))));
      return seg;
    };
    this.segs = [
      segInit("ex-fam", () => this.st.fam, v => this.setFamily(v)),
      segInit("ex-at", () => this.st.at, v => this.st.at = v),
      segInit("ex-lay", () => this.st.lay, v => this.setLayout(+v)),
      segInit("ex-tap", () => this.st.tap, v => { this.st.tap = v; }),
    ];
    if (m.layouts.length < 2) { $("ex-lay-ctl").hidden = true; }
    m.modes.forEach((name, i) => $("ex-mode").add(new Option(name, i)));
    $("ex-mode").addEventListener("change", e => { this.st.mode = +e.target.value; this.st.focus = null; this.update(); });
    $("ex-anchor").addEventListener("change", e => { this.st.anchor = +e.target.value; this.st.focus = null; this.update(); });
    $("ex-a").addEventListener("change", e => { this.st.a = +e.target.value; if (this.st.b < this.st.a) this.st.b = this.st.a; this.st.focus = null; this.update(); });
    $("ex-b").addEventListener("change", e => { this.st.b = +e.target.value; if (this.st.a > this.st.b) this.st.a = this.st.b; this.st.focus = null; this.update(); });
    // disable diagnostics that the data do not have
    const have = new Set(m.points_fields);
    for (const [c, f] of Object.entries(DIAG)) if (!have.has(f)) { const o = $("ex-color").querySelector(`option[value=${c}]`); if (o) { o.disabled = true; o.textContent += " (not in data)"; } }
    const og = document.createElement("optgroup"); og.label = "Image statistics (pixels)";
    for (const [k, d] of Object.entries(IMGSTATS)) og.append(new Option(d.label, "px:" + k));
    $("ex-color").append(og);
    $("ex-color").addEventListener("change", e => this.setColor(e.target.value));
    $("ex-k").addEventListener("change", e => {
      if (this.st.kind === "psi") this.st.k = +e.target.value; else { this.st.r = +e.target.value; this.st.j = null; }
      if (BLOCK_KINDS.has(this.st.color) || this.st.color === "seba") this.st.focus = null;
      this.update();
    });
    $("ex-ps").addEventListener("input", e => { this.st.ps = +e.target.value; this.draw(); this.drawLens(); });
    const kb = $("ex-kappa-btn");
    const setHelp = open => { kb.setAttribute("aria-expanded", String(open)); $("ex-kappa-help").hidden = !open; };
    kb.addEventListener("click", () => setHelp(kb.getAttribute("aria-expanded") !== "true"));
    document.addEventListener("keydown", e => { if (e.key === "Escape") setHelp(false); });
    document.addEventListener("pointerdown", e => { if (!$("ex-kappa-help").hidden && !e.target.closest("#ex-kappa-help, #ex-kappa-btn")) setHelp(false); });
    this.buildGuide();
  }

  // ---------------------------------------------------------------- tours and the how-to
  buildGuide() {
    this.tours = tours(); this.tour = null;
    const sel = $("ex-tour-sel");
    this.tours.forEach(t => sel.add(new Option(t.label, t.id)));
    if (!this.tours.length) sel.closest(".ctl").querySelectorAll("label, select").forEach(e => e.hidden = true);
    // a tour is a URL (#explorer?tour=<id>), so the menu, back and forward all go through the router
    sel.addEventListener("change", () => { if (sel.value) location.hash = `#explorer?tour=${encodeURIComponent(sel.value)}`; });
    $("ex-tour-reset").addEventListener("click", () => { if (this.tour) this.applyParams(new URLSearchParams({ tour: this.tour.id })); });
    $("ex-tour-close").addEventListener("click", () => { this.setTour(null); this.writeHash(); });
    // how-to: open on the first visit, collapsed after that
    const how = $("ex-howto"), btn = $("ex-howto-btn"), KEY = "cm.explorer.howto-seen";
    let seen = false; try { seen = localStorage.getItem(KEY) === "1"; localStorage.setItem(KEY, "1"); } catch (e) { /* private mode */ }
    how.open = !seen;
    const sync = () => btn.setAttribute("aria-expanded", String(how.open));
    how.addEventListener("toggle", sync); sync();
    btn.addEventListener("click", () => { how.open = !how.open; if (how.open) how.scrollIntoView({ block: "nearest" }); });
  }
  setTour(t) {
    this.tour = t ? { id: t.id, base: null } : null;
    $("ex-tour").hidden = !t;
    $("ex-tour-sel").value = t ? t.id : "";
    if (!t) return;
    $("ex-tour-label").textContent = t.label;
    const note = $("ex-tour-note"); note.innerHTML = t.note;
    if (window.renderMathInElement) window.renderMathInElement(note, { delimiters: [{ left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
    $("ex-tour-reset").hidden = true; $("ex-tour").classList.remove("moved");
  }

  setColor(c) {
    const st = this.st;
    st.color = c; st.focus = null;
    if (c === "seba") st.kind = "set";
    if (c === "nodal" || c === "eig") st.kind = "psi";
    if (c === "seba" && st.j != null) st.focus = st.j;
    this.update();
  }
  setFamily(f) {
    const st = this.st, { T, S } = this.ctx;
    if (f === st.fam) return;
    if (f === "time") { const t = st.anchor, s = st.b; st.anchor = s; st.a = Math.max(0, t - 2); st.b = t; if (st.a === st.b && t > 0) st.a = t - 1; }
    else { const s = st.anchor, t = st.b; st.anchor = t; st.a = Math.max(0, s - 3); st.b = s; }
    st.fam = f; st.focus = null; st.j = null; st.cmp = null;
  }
  setLayout(seed) {
    const st = this.st;
    if (seed === st.lay) return;
    const go = coords => {
      const from = this.cur, to = coords, N = from.length, t0 = performance.now(), dur = 650;
      const tmp = new Float32Array(N);
      cancelAnimationFrame(this._anim);
      const step = now => {
        const u = Math.min(1, (now - t0) / dur), e = u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
        for (let i = 0; i < N; i++) tmp[i] = from[i] + (to[i] - from[i]) * e;
        this.cur = u < 1 ? tmp : to; this.draw(); this.drawOverlay(); this.drawLens();
        if (u < 1) this._anim = requestAnimationFrame(step);
      };
      this._anim = requestAnimationFrame(step);
    };
    st.lay = seed;
    if (this.coords[seed]) go(this.coords[seed]);
    else D.layout(seed).then(u => { this.coords[seed] = toCoords(u); if (this.st.lay === seed) go(this.coords[seed]); });
  }

  syncControls() {
    const st = this.st, { m, T, S } = this.ctx;
    this.segs.forEach(s => s._sync());
    $("ex-mode").value = st.mode;
    $("ex-color").value = st.color;
    const anc = $("ex-anchor"), a = $("ex-a"), b = $("ex-b");
    const key = st.fam + T + S;
    if (anc._key !== key) {
      anc._key = key; anc.innerHTML = ""; a.innerHTML = ""; b.innerHTML = "";
      if (st.fam === "layer") {
        m.steps.forEach((s, i) => anc.add(new Option(`step ${s.toLocaleString("en")}`, i)));
        m.sites.forEach((s, i) => { a.add(new Option(s, i)); b.add(new Option(s, i)); });
        $("ex-anchor-label").textContent = "checkpoint"; $("ex-range-label").textContent = "sites";
      } else {
        m.sites.forEach((s, i) => anc.add(new Option(s, i)));
        m.steps.forEach((s, i) => { a.add(new Option(s.toLocaleString("en"), i)); b.add(new Option(s.toLocaleString("en"), i)); });
        $("ex-anchor-label").textContent = "site"; $("ex-range-label").textContent = "steps";
      }
    }
    anc.value = st.anchor; a.value = st.a; b.value = st.b;
    const ks = $("ex-k");
    const want = st.kind === "psi" ? "psi" : "set";
    if (ks._kind !== want) {
      ks._kind = want; ks.innerHTML = "";
      if (want === "psi") for (let k = 1; k <= m.n_vecs; k++) ks.add(new Option(`ψ${k}`, k));
      else m.seba_r.forEach(r => ks.add(new Option(`r = ${r} sets`, r)));
    }
    ks.value = want === "psi" ? st.k : st.r;
    ks.title = BLOCK_KINDS.has(st.color) ? "" : "Selection shown in the side panel";
    $("ex-at").hidden = !(st.color in DIAG || st.color === "correct");
    $("ex-k-label").hidden = BLOCK_KINDS.has(st.color);
  }

  // ---------------------------------------------------------------- grid DOM
  buildGrid() {
    const { m, T, S } = this.ctx, g = $("ex-grid");
    g.replaceChildren();
    this.colh = m.sites.map(s => g.appendChild(h("div", { class: "colh", text: s })));
    this.rowh = m.steps.map((s, t) => {
      const acc = m.val_acc[t];
      return g.appendChild(h("div", { class: "rowh" }, h("b", { text: s.toLocaleString("en") }), acc != null ? `${(100 * acc).toFixed(1)}%` : ""));
    });
    this.cellbg = [];
    for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) this.cellbg.push(g.appendChild(h("div", { class: "cellbg" })));
    this.blockbox = g.appendChild(h("div", { class: "blockbox" }));
    this.axisS = g.appendChild(h("div", { class: "axis-t", text: "site →" }));
    this.axisT = g.appendChild(h("div", { class: "axis-t", text: "step ↓  val acc" }));
    this.canvas = g.appendChild(h("canvas", { class: "base", "aria-label": "Grid of panels, one UMAP layout per snapshot; hover to inspect points, click to choose the block" }));
    this.ov = g.appendChild(h("canvas", { class: "ov" }));
    const cv = this.canvas;
    cv.addEventListener("pointermove", e => this.onMove(e));
    cv.addEventListener("pointerleave", () => { this.setHover(-1); this.dragPrev = null; hideTip(); this.drawOverlay(); });
    cv.addEventListener("pointerdown", e => this.onDown(e));
    window.addEventListener("pointerup", e => this.onUp(e));
    const lens = $("lens");
    lens.addEventListener("pointermove", e => this.onLensMove(e));
    lens.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
  }

  layout() {
    const { T, S } = this.ctx, body = $("ex-body"), bar = $("ex-bar");
    if (!body.clientWidth) return;   // tab hidden
    const help = $("ex-kappa-help"), barH = bar.offsetHeight - (help.hidden ? 0 : help.offsetHeight + 8);   // the explainer does not resize the grid
    const sig = `${body.clientWidth}x${window.innerHeight}x${barH}`;
    if (sig === this._sig && this._started) return;
    this._sig = sig;
    document.documentElement.style.setProperty("--bar-h", bar.offsetHeight + "px");
    const W = body.clientWidth - 48 + (window.innerWidth <= 640 ? 24 : 0);
    const narrow = window.matchMedia("(max-width: 1099.98px)").matches, small = window.innerWidth <= 640;
    const rowW = small ? 52 : 64, colH = 18, gap = small ? 2 : 4;
    let cell, side = 340;
    if (narrow) cell = Math.floor((W - rowW) / S) - gap;
    else {
      const availH = window.innerHeight - 48 - barH - 92;
      const byW = Math.floor((W - 22 - 330 - rowW) / S) - gap;
      const byH = Math.floor(availH / T) - gap;
      cell = Math.min(byW, Math.max(byH, 76));
      cell = Math.max(40, Math.min(150, cell));
      side = Math.max(330, Math.min(500, W - 22 - rowW - S * (cell + gap)));
    }
    cell = Math.max(small ? 40 : 34, Math.min(150, cell));
    body.style.setProperty("--side-w", side + "px");
    if (this.geom && this.geom.cell === cell && this.geom.small === small) { this.renderAll(); return; }
    this.geom = { cell, gap, rowW, colH, small, pad: Math.max(2, Math.round(cell * .05)) };
    const g = $("ex-grid"), Wg = rowW + S * (cell + gap), Hg = colH + T * (cell + gap) + 4;
    g.style.width = Wg + "px"; g.style.height = Hg + "px";
    this.colh.forEach((e, s) => Object.assign(e.style, { left: rowW + s * (cell + gap) + "px", top: "0px", width: cell + "px", fontSize: cell < 46 ? "9.5px" : "" }));
    this.rowh.forEach((e, t) => Object.assign(e.style, { left: "0px", top: colH + t * (cell + gap) + cell / 2 - 14 + "px", width: rowW - 8 + "px", fontSize: small ? "9.5px" : "" }));
    this.cellbg.forEach((e, i) => { const t = Math.floor(i / S), s = i % S; Object.assign(e.style, { left: rowW + s * (cell + gap) + "px", top: colH + t * (cell + gap) + "px", width: cell + "px", height: cell + "px" }); });
    this.axisS.hidden = true; this.axisT.hidden = false;
    Object.assign(this.axisT.style, { left: "0px", top: "2px", width: rowW - 8 + "px", textAlign: "right" });
    this.axisT.textContent = small ? "step" : "step \u00b7 acc";
    this.axisT.title = "Rows: training step and validation accuracy at that checkpoint. Columns: sites, from the stem to the logits.";
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = S * (cell + gap) - gap, chh = T * (cell + gap) - gap;
    for (const c of [this.canvas, this.ov]) {
      c.width = Math.round(cw * dpr); c.height = Math.round(chh * dpr);
      Object.assign(c.style, { left: rowW + "px", top: colH + "px", width: cw + "px", height: chh + "px" });
    }
    this.dpr = dpr;
    this.img = null;
    if (!this._started) { this._started = true; this.update(); } else this.renderAll();
  }

  cellRect(t, s) { const g = this.geom; return { x: s * (g.cell + g.gap) + g.pad, y: t * (g.cell + g.gap) + g.pad, w: g.cell - 2 * g.pad }; }
  hitCell(e) {
    const r = this.canvas.getBoundingClientRect(), g = this.geom, x = e.clientX - r.left, y = e.clientY - r.top;
    const s = Math.floor(x / (g.cell + g.gap)), t = Math.floor(y / (g.cell + g.gap));
    if (s < 0 || t < 0 || s >= this.ctx.S || t >= this.ctx.T) return null;
    return { t, s, x, y };
  }
  nearest(t, s, x, y, cr, maxd) {
    const { N, S } = this.ctx, c = this.cur, base = (t * S + s) * N * 2;
    let best = -1, bd = maxd * maxd;
    for (let i = 0; i < N; i++) {
      const dx = cr.x + cr.w * c[base + 2 * i] - x, dy = cr.y + cr.w * (1 - c[base + 2 * i + 1]) - y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  // ---------------------------------------------------------------- pointer
  onMove(e) {
    const hc = this.hitCell(e);
    if (!hc) return;
    if (this.drag) {
      const st = this.st, d = this.drag;
      const along = st.fam === "layer" ? (hc.t === d.t) : (hc.s === d.s);
      if (along && (hc.t !== d.t || hc.s !== d.s)) { d.moved = true; d.end = hc; this.dragPrev = d; this.drawOverlay(); }
    }
    const cr = this.cellRect(hc.t, hc.s);
    const i = this.nearest(hc.t, hc.s, hc.x, hc.y, cr, Math.max(6, this.geom.cell * .07));
    if (this.lensCell[0] !== hc.t || this.lensCell[1] !== hc.s) { this.lensCell = [hc.t, hc.s]; this.drawLens(); }
    this.setHover(i, hc.t);
    if (i >= 0) {
      const key = `${i}/${hc.t}/${hc.s}/${this._tok}`;
      if (key !== this._tipKey) { this._tipKey = key; this._tipEl = this.pointTip(i, hc.t, hc.s); }
      showTip(e.clientX, e.clientY, this._tipEl);
    } else { this._tipKey = null; hideTip(); }
    this.canvas.title = "";
    // prefetch the block a click here would select
    if (!this.drag) { const nb = this.clickTarget(hc, e.shiftKey); if (nb) D.block(nb.fam, nb.anchor, nb.mode, pairOf(this.ctx, nb)).catch(() => {}); }
  }
  onDown(e) {
    if (e.button !== 0) return;
    const hc = this.hitCell(e); if (!hc) return;
    this.drag = { t: hc.t, s: hc.s, shift: e.shiftKey, moved: false, pointer: e.pointerType };
    if (e.pointerType === "mouse") e.preventDefault();
  }
  onUp(e) {
    const d = this.drag; this.drag = null;
    if (!d) return;
    const st = this.st;
    if (d.moved && d.end) {
      if (st.fam === "layer") { st.anchor = d.t; st.a = Math.min(d.s, d.end.s); st.b = Math.max(d.s, d.end.s); }
      else { st.anchor = d.s; st.a = Math.min(d.t, d.end.t); st.b = Math.max(d.t, d.end.t); }
    } else {
      const hc = this.hitCell(e) || d;
      const touch = d.pointer !== "mouse";
      Object.assign(st, this.clickTarget(hc, d.shift || (touch && st.tap === "start")));
      if (touch && hc.x != null) {   // tap also pins the nearest point (no hover on touch screens)
        const i = this.nearest(hc.t, hc.s, hc.x, hc.y, this.cellRect(hc.t, hc.s), Math.max(10, this.geom.cell * .12));
        this.setHover(i, hc.t);
        if (i >= 0) { const ex = e.clientX, ey = e.clientY; setTimeout(() => showTip(ex, ey, this.pointTip(i, hc.t, hc.s)), 350); } else hideTip();
      }
    }
    this.dragPrev = null;
    st.focus = st.color === "seba" ? null : st.focus; st.j = null;
    this.lensCell = [d.t, d.s];
    this.update();
  }
  clickTarget(hc, shift) {
    const st = this.st, out = { fam: st.fam, mode: st.mode };
    const [anc, pos] = st.fam === "layer" ? [hc.t, hc.s] : [hc.s, hc.t];
    out.anchor = anc;
    if (shift) { out.a = pos; out.b = Math.max(st.b, pos); }
    else { out.b = pos; out.a = Math.min(st.a, pos); }
    return out;
  }
  onLensMove(e) {
    const [t, s] = this.lensCell, c = $("lens"), r = c.getBoundingClientRect(), pad = 10;
    const cr = { x: pad, y: pad, w: r.width - 2 * pad };
    const i = this.nearest(t, s, e.clientX - r.left, e.clientY - r.top, cr, 10);
    this.setHover(i, t);
    if (i >= 0) showTip(e.clientX, e.clientY, this.pointTip(i, t, s)); else hideTip();
  }
  setHover(i, t) {
    if (i === this.hover) return;
    this.hover = i; this.hoverT = t;
    this.drawOverlay(); this.drawLensOverlay();
    document.querySelectorAll(".imgs canvas.hl").forEach(c => c.classList.remove("hl"));
    if (i >= 0) document.querySelectorAll(`.imgs canvas[data-i="${i}"]`).forEach(c => c.classList.add("hl"));
  }

  thumb(i) { return thumb(this.ctx, i); }
  pointTip(i, t, s) {
    const { m, labels, pts, N } = this.ctx, st = this.st;
    const rows = [];
    rows.push({ name: "class", value: m.classes[labels[i]] });
    const tt = st.at === "final" ? m.steps.length - 1 : t;
    if (pts.pred) rows.push({ name: `predicted at ${fmtStep(m.steps[tt])}`, value: m.classes[pts.pred[tt * N + i]] });
    if (pts.loss) rows.push({ name: "loss", value: fmt(pts.loss[tt * N + i], 3) });
    const blk = this.blk;
    if (blk) {
      if (st.kind === "set") {
        const ri = m.seba_r.indexOf(st.r), l = blk.lab[ri * N + i], v = D.sebaVectors(m, blk, ri);
        rows.push({ name: `SEBA (r = ${st.r})`, value: l < 0 ? "unassigned" : `set ${l + 1} (${fmt(v[l][i])})`, color: l < 0 ? this.ctx.pal.unassigned : this.ctx.pal.set(l) });
      } else {
        const p = D.psi(m, blk, st.k); let mx = 0; for (const x of p) mx = Math.max(mx, Math.abs(x));
        rows.push({ name: `ψ${st.k} / max|ψ|`, value: fmt(p[i] / mx) });
      }
      if (st.color in DIAG && st.color !== "loss") { const f = DIAG[st.color]; rows.push({ name: f, value: pts[f][tt * N + i].toPrecision(3) }); }
      if (isPx(st.color) && this.ctx.imgstats) { const k = st.color.slice(3); rows.push({ name: this.ctx.imgstats.labels[k] || k, value: this.ctx.imgstats.per[k][i].toFixed(3) }); }
    }
    const d = h("div", { class: "tt-img" }, this.thumb(i), tipRows(`#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`, rows));
    return d;
  }

  // ---------------------------------------------------------------- state -> render
  async update() {
    const st = this.st, ctx = this.ctx;
    this.syncControls();
    this.writeHash();
    $("ex-status").textContent = "loading…";
    $("ex-side").style.opacity = .55;
    const token = this._tok = (this._tok || 0) + 1;
    let blk;
    try { blk = await blockOf(ctx, st); } catch (e) { $("ex-status").textContent = "failed to load block"; console.error(e); return; }
    if (isPx(st.color) && !ctx.imgstats) {
      $("ex-status").textContent = "loading image statistics…";
      try { await prepare(ctx, st); } catch (e) { $("ex-status").textContent = "image statistics unavailable"; console.error(e); return; }
    }
    if (token !== this._tok) return;
    this.blk = blk;
    $("ex-status").textContent = "";
    $("ex-side").style.opacity = 1;
    this.col = colouring(ctx, st, blk);
    this.renderAll();
    this.prefetch();
  }
  renderAll() {
    if (!this.col || !this.geom) return;
    this.draw(); this.drawOverlay(); this.drawLens(); this.decorate();
    this.renderBlockCard(); this.renderLegend(); this.renderImages(); this.renderRetention(); this.renderStats(); this.renderCaption();
  }
  prefetch() {
    const st = this.st, ctx = this.ctx, lim = st.fam === "layer" ? ctx.S : ctx.T;
    const near = [[st.a, st.b + 1], [st.a + 1, st.b], [st.a - 1, st.b], [st.a, st.b - 1]].filter(([a, b]) => a >= 0 && b < lim && a <= b);
    const go = () => near.forEach(([a, b]) => D.block(st.fam, st.anchor, st.mode, ctx.pairIdx[st.fam][a + "," + b]).catch(() => {}));
    setTimeout(go, 250);
  }
  writeHash() {
    const st = this.st;
    const p = new URLSearchParams({ fam: st.fam, t: st.anchor, a: st.a, b: st.b, m: st.mode, c: st.color });
    if (st.kind === "psi") p.set("k", st.k); else p.set("r", st.r);
    if (st.j != null) p.set("j", st.j);
    if (st.color in DIAG || st.color === "correct") p.set("at", st.at);
    if (st.lay) p.set("lay", st.lay);
    let hsh = "#explorer?" + p.toString();
    if (this.tour) {
      if (this.tour.base == null) this.tour.base = hsh;     // the first state written after the tour was applied
      const on = hsh === this.tour.base;
      $("ex-tour").classList.toggle("moved", !on); $("ex-tour-reset").hidden = on;
      $("ex-tour-sel").value = on ? this.tour.id : "";
      if (on) hsh += "&tour=" + encodeURIComponent(this.tour.id);
    }
    this.lastHash = hsh;
    if (location.hash !== hsh && document.body.dataset.tab === "explorer") history.replaceState(null, "", hsh);
  }
  applyParams(p) {
    const st = this.st, { m, T, S } = this.ctx;
    // #explorer?tour=<id>: exactly the tour's own parameters (others are ignored, so the note always describes the view)
    const tid = p.get("tour"), tour = tid != null ? this.tours.find(t => t.id === tid) : null;
    if (tour) p = new URLSearchParams(tour.hash.split("?")[1] || "");
    this.setTour(tour);
    if (tid != null && !tour) setTimeout(() => { $("ex-status").textContent = `no tour called "${tid}"`; }, 0);
    const int = (k, lo, hi) => { const v = parseInt(p.get(k), 10); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; };
    if (p.get("fam") === "time" || p.get("fam") === "layer") st.fam = p.get("fam");
    const lim = st.fam === "layer" ? S - 1 : T - 1, alim = st.fam === "layer" ? T - 1 : S - 1;
    st.anchor = int("t", 0, alim) ?? Math.min(st.anchor, alim);
    st.a = int("a", 0, lim) ?? Math.min(st.a, lim); st.b = int("b", 0, lim) ?? Math.min(st.b, lim);
    if (st.a > st.b) [st.a, st.b] = [st.b, st.a];
    st.mode = int("m", 0, m.modes.length - 1) ?? st.mode;
    const c = p.get("c"); if (c && $("ex-color").querySelector(`option[value="${c}"]:not([disabled])`)) st.color = c;
    if (st.color === "seba") st.kind = "set"; if (st.color === "nodal" || st.color === "eig") st.kind = "psi";
    if (p.has("k")) { st.k = int("k", 1, m.n_vecs) ?? st.k; if (!BLOCK_KINDS.has(st.color)) st.kind = "psi"; }
    if (p.has("r")) { const r = int("r", 1, 99); if (m.seba_r.includes(r)) st.r = r; if (!BLOCK_KINDS.has(st.color)) st.kind = "set"; }
    st.cmp = null;
    st.j = int("j", 0, 15); st.focus = st.color === "seba" ? st.j : null;
    if (p.get("at") === "final" || p.get("at") === "own") st.at = p.get("at");
    const lay = int("lay", 0, 1) ?? 0; if (lay < m.layouts.length && lay !== st.lay) this.setLayout(lay);
    this.update();
  }

  // ---------------------------------------------------------------- drawing
  draw() {
    const { T, S, N } = this.ctx, cv = this.canvas, W = cv.width, H = cv.height, dpr = this.dpr;
    const g = cv.getContext("2d");
    if (!this.img || this.img.width !== W || this.img.height !== H) { this.img = g.createImageData(W, H); this.buf = new Uint32Array(this.img.data.buffer); }
    this.buf.fill(0);
    const cells = [];
    for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) { const r = this.cellRect(t, s); cells.push({ t, s, S, x: r.x * dpr, y: r.y * dpr, w: r.w * dpr }); }
    paint(this.buf, W, H, cells, this.cur, this.col, N, this.st.ps * dpr / 2 * Math.max(.5, Math.min(1, this.geom.cell / 70)));
    g.putImageData(this.img, 0, 0);
  }
  drawOverlay() {
    const { T, S, N } = this.ctx, cv = this.ov, g = cv.getContext("2d"), dpr = this.dpr, c = this.cur;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, cv.width, cv.height);
    const d = this.dragPrev;
    if (d && d.end) {
      const r0 = this.cellRect(Math.min(d.t, d.end.t), Math.min(d.s, d.end.s)), r1 = this.cellRect(Math.max(d.t, d.end.t), Math.max(d.s, d.end.s)), p = this.geom.pad;
      g.fillStyle = "rgba(22,22,26,.06)"; g.strokeStyle = "#16161a"; g.lineWidth = 1.5; g.setLineDash([4, 3]);
      g.beginPath(); g.roundRect(r0.x - p + 1, r0.y - p + 1, r1.x + r1.w + p - r0.x + p - 2, r1.y + r1.w + p - r0.y + p - 2, 6); g.fill(); g.stroke(); g.setLineDash([]);
    }
    const i = this.hover;
    if (i < 0) return;
    for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) {
      const r = this.cellRect(t, s), base = (t * S + s) * N * 2;
      const x = r.x + r.w * c[base + 2 * i], y = r.y + r.w * (1 - c[base + 2 * i + 1]);
      g.beginPath(); g.arc(x, y, 4.2, 0, 2 * Math.PI); g.strokeStyle = "#fff"; g.lineWidth = 3; g.stroke();
      g.strokeStyle = "#16161a"; g.lineWidth = 1.4; g.stroke();
    }
  }
  drawLens() {
    const c = $("lens"); if (!c || !this.col) return;
    const [t, s] = this.lensCell, { N, S, m } = this.ctx;
    const w = c.clientWidth; if (!w) return;
    const dpr = this.dpr || 1;
    if (c.width !== Math.round(w * dpr)) { c.width = c.height = Math.round(w * dpr); }
    const g = c.getContext("2d"), img = g.createImageData(c.width, c.height), buf = new Uint32Array(img.data.buffer), pad = 10 * dpr;
    paint(buf, c.width, c.height, [{ t, s, S, x: pad, y: pad, w: c.width - 2 * pad }], this.cur, this.col, N, Math.max(2.2, this.st.ps * 1.6) * dpr / 2 + .3);
    g.putImageData(img, 0, 0);
    this._lensImg = img;
    $("lens-title").textContent = `${m.sites[s]} at step ${m.steps[t].toLocaleString("en")}` + (inBlock(this.st, t, s) ? " (in the block)" : "");
    this.drawLensOverlay();
  }
  drawLensOverlay() {
    const c = $("lens"); if (!c || !this._lensImg) return;
    const g = c.getContext("2d"); g.putImageData(this._lensImg, 0, 0);
    const i = this.hover; if (i < 0) return;
    const [t, s] = this.lensCell, { N, S } = this.ctx, dpr = this.dpr || 1, pad = 10 * dpr, w = c.width - 2 * pad, base = (t * S + s) * N * 2;
    const x = pad + w * this.cur[base + 2 * i], y = pad + w * (1 - this.cur[base + 2 * i + 1]);
    g.beginPath(); g.arc(x, y, 6 * dpr, 0, 2 * Math.PI); g.strokeStyle = "#fff"; g.lineWidth = 3 * dpr; g.stroke(); g.strokeStyle = "#16161a"; g.lineWidth = 1.5 * dpr; g.stroke();
  }
  decorate() {
    const st = this.st, { T, S } = this.ctx, g = this.geom;
    this.cellbg.forEach((e, i) => e.classList.toggle("inb", inBlock(st, Math.floor(i / S), i % S)));
    this.colh.forEach((e, s) => e.classList.toggle("inb", st.fam === "layer" ? (s >= st.a && s <= st.b) : s === st.anchor));
    this.rowh.forEach((e, t) => e.classList.toggle("inb", st.fam === "layer" ? t === st.anchor : (t >= st.a && t <= st.b)));
    let t0, t1, s0, s1;
    if (st.fam === "layer") { t0 = t1 = st.anchor; s0 = st.a; s1 = st.b; } else { s0 = s1 = st.anchor; t0 = st.a; t1 = st.b; }
    Object.assign(this.blockbox.style, {
      left: g.rowW + s0 * (g.cell + g.gap) - 3 + "px", top: g.colH + t0 * (g.cell + g.gap) - 3 + "px",
      width: (s1 - s0 + 1) * (g.cell + g.gap) - g.gap + 6 + "px", height: (t1 - t0 + 1) * (g.cell + g.gap) - g.gap + 6 + "px",
    });
  }

  // ---------------------------------------------------------------- side panel
  kappaRange() {
    const st = this.st, { m } = this.ctx, ks = [], ms = [];
    for (let x = st.a; x <= st.b; x++) {
      const [t, s] = st.fam === "layer" ? [st.anchor, x] : [x, st.anchor];
      const k = m.kappa?.[st.mode]?.[t]?.[s]; if (k != null) ks.push(k);
      const q = m.nn_mass?.[st.mode]?.[t]?.[s]; if (q != null) ms.push(q);
    }
    return { ks, ms, n: st.b - st.a + 1, perPoint: !!(m.kappa_q10 && m.kappa_q10[st.mode]?.[st.fam === "layer" ? st.anchor : st.a]?.[st.fam === "layer" ? st.a : st.anchor] != null && /perplexity/i.test(m.modes[st.mode])) };
  }
  renderBlockCard() {
    const st = this.st, { m, pal } = this.ctx, blk = this.blk;
    $("blk-title").textContent = `${blockKind(st)}: ${blockLabel(this.ctx, st)}`;
    const { ks, ms, n, perPoint } = this.kappaRange();
    const fmtK = v => v >= 10 ? v.toFixed(1) : v.toPrecision(2);
    const rng = a => { const lo = Math.min(...a), hi = Math.max(...a); return Math.abs(hi - lo) < 1e-3 * hi ? fmtK(lo) : `${fmtK(lo)}–${fmtK(hi)}`; };
    const kap = ks.length ? (perPoint ? `median per-image κ ${rng(ks)}` : `κ ${rng(ks)}`) : "rank-based, no κ";
    const mass = ms.length ? (() => { const lo = Math.min(...ms), hi = Math.max(...ms); return `, median 15-NN kernel mass ${hi - lo < .005 ? pct(lo) : pct(lo) + "–" + pct(hi)}`; })() : "";
    $("blk-sub").textContent = `${m.modes[st.mode]}: ${kap}${mass} over the ${n} snapshot${n > 1 ? "s" : ""}`;
    // spectrum: lambda_1..8 as lollipops
    const vals = Array.from(blk.vals), box = $("blk-spec");
    box.replaceChildren();
    const W = box.clientWidth || 300, H = 136, ml = 34, mr = 8, mt = 8, mb = 40;
    // participation number (sum psi^2)^2 / sum psi^4: about N/3 or more for a spread-out mode, small for a localised one
    const N = this.ctx.N, pn = [0];
    for (let k = 1; k <= 8; k++) pn.push(blk.pn[k - 1]);   // computed at build time from full-precision vectors
    const LOC = Math.round(0.01 * N), nLoc = pn.slice(1).filter(x => x < LOC).length;
    const lo = Math.min(.5, Math.floor((Math.min(...vals.slice(1)) - .02) * 20) / 20), hi = 1;
    const svg = el("svg", { width: W, height: H }, box);
    const X = k => ml + (k - .5) / 8 * (W - ml - mr), Y = scale("linear", lo, hi, H - mb, mt);
    niceTicks(lo, hi, 3).forEach(v => { el("line", { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: "#efeee9" }, svg); const t = el("text", { x: ml - 5, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = v.toFixed(2); });
    const yl = el("text", { x: ml - 5, y: H - 7, "text-anchor": "end", style: "font-size:10px" }, svg); yl.textContent = "PN";
    const sel = k => st.kind === "psi" ? k === st.k : k < st.r;
    for (let k = 1; k <= 8; k++) {
      const on = sel(k), c = on ? "#16161a" : "#b9b8b1";
      el("line", { x1: X(k), x2: X(k), y1: Y(lo), y2: Y(vals[k]), stroke: c, "stroke-width": on ? 2 : 1.5 }, svg);
      el("circle", { cx: X(k), cy: Y(vals[k]), r: on ? 4.5 : 3.5, fill: c, stroke: "#fff", "stroke-width": 1.5 }, svg);
      const t = el("text", { x: X(k), y: H - 24, "text-anchor": "middle" }, svg); t.textContent = `λ${k}`;
      if (on) t.setAttribute("style", "fill:#16161a;font-weight:600");
      const tp = el("text", { x: X(k), y: H - 7, "text-anchor": "middle", style: pn[k] < LOC ? `fill:${m.palette.accent};font-weight:650;font-size:10px` : "font-size:10px" }, svg);
      tp.textContent = pn[k] >= 1000 ? (pn[k] / 1000).toFixed(1) + "k" : String(Math.round(pn[k]));
      const hit = el("rect", { x: X(k) - 12, y: mt, width: 24, height: H - mt, fill: "transparent", style: "cursor:pointer" }, svg);
      hit.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(null, [{ name: `λ${k}`, value: vals[k].toFixed(4) }, { name: "average penalty 1 − λ", value: (1 - vals[k]).toFixed(4) },
        { name: "participation number", value: `${Math.round(pn[k])} of ${N}` + (pn[k] < LOC ? " (localised)" : "") }])));
      hit.addEventListener("pointerleave", hideTip);
      hit.addEventListener("click", () => { st.kind = "psi"; st.k = k; if (st.color === "seba") st.color = "eig"; st.focus = null; this.update(); });
    }
    const locNote = nLoc ? ` <span style="color:${m.palette.accent};font-weight:600">${nLoc} localised mode${nLoc > 1 ? "s" : ""}</span> (PN below ${LOC}): a mode carried by a handful of images, not a grouping; sets built from it are suspect.` : "";
    $("blk-spec-note").innerHTML = `Spectrum of the block operator M̄, the mean of the lazy walks M<sub>c</sub> over the block's ${st.b - st.a + 1} snapshot${st.a === st.b ? "" : "s"}: λ<sub>1</sub>..λ<sub>8</sub> (λ<sub>0</sub> = 1, the constant ψ<sub>0</sub>, not shown). 1 − λ<sub>k</sub> is ψ<sub>k</sub>'s average variation across the block's walks.`
      + (st.kind === "psi" ? ` Click a bar to colour by that ψ.` : ` The r = ${st.r} coherent sets come from SEBA on ψ<sub>0</sub>..ψ<sub>${st.r - 1}</sub> (dark).`)
      + ` PN: participation number (Σψ²)²/Σψ⁴, roughly how many images a mode is spread over.` + locNote;
    this.renderBlockMap();
  }
  renderBlockMap() {
    const st = this.st, { m, pal } = this.ctx;
    const pairs = st.fam === "layer" ? m.layer_pairs : m.time_pairs, n = st.fam === "layer" ? m.sites.length : m.steps.length;
    const vals = m.vals[st.fam][st.anchor][st.mode], kk = st.kind === "psi" ? st.k : st.r - 1;
    $("blk-map-anchor").textContent = st.fam === "layer" ? "checkpoint" : "site";
    const box = $("blk-map"); box.replaceChildren();
    const W = box.clientWidth || 300, ml = 34, mt = 4, cw = Math.min(22, (W - ml - 50) / n), H = mt + n * cw + 26;
    const svg = el("svg", { width: W, height: H }, box);
    const v = pairs.map((_, p) => vals[p][kk]), lo = Math.min(...v), hi = 1;
    const names = st.fam === "layer" ? m.sites : m.steps.map(fmtStep);
    pairs.forEach(([a, b], p) => {
      const on = a === st.a && b === st.b;
      const r = el("rect", { x: ml + b * cw + .5, y: mt + a * cw + .5, width: cw - 1, height: cw - 1, rx: 2, fill: pal.seq((v[p] - lo) / (hi - lo + 1e-12)), style: "cursor:pointer" }, svg);
      if (on) { r.setAttribute("stroke", "#16161a"); r.setAttribute("stroke-width", 2); }
      r.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`${names[a]} → ${names[b]}`, [{ name: `λ${kk}`, value: v[p].toFixed(4) }])));
      r.addEventListener("pointerleave", hideTip);
      r.addEventListener("click", () => { hideTip(); st.a = a; st.b = b; st.focus = null; st.j = null; this.update(); });
    });
    names.forEach((nm, i) => {
      if ((n > 9 || cw < 30) && (n - 1 - i) % 2) return;
      const t1 = el("text", { x: ml - 4, y: mt + i * cw + cw / 2 + 3.5, "text-anchor": "end" }, svg); t1.textContent = nm;
      const t2 = el("text", { x: ml + i * cw + cw / 2, y: mt + n * cw + 12, "text-anchor": "middle" }, svg); t2.textContent = nm;
    });
    $("blk-map-note").innerHTML = `Every block at this ${st.fam === "layer" ? "checkpoint" : "site"}: row = first ${st.fam === "layer" ? "site" : "checkpoint"}, column = last; colour = λ<sub>${kk}</sub> (darker = closer to 1; range ${lo.toFixed(3)} to 1). Click to select.`;
    $("blk-map-wrap").open = this._mapOpen ?? (window.innerWidth >= 1100);
    $("blk-map-wrap").ontoggle = () => { this._mapOpen = $("blk-map-wrap").open; };
  }

  renderLegend() {
    const st = this.st, { m, pal, labels, N } = this.ctx, col = this.col, body = $("leg-body");
    body.replaceChildren(); $("leg-jac").replaceChildren();
    const titles = { seba: `Coherent sets (SEBA, r = ${st.r})`, nodal: `Sign of ψ${st.k} (nodal domains)`, eig: `Eigenfunction ψ${st.k}`, class: "CIFAR class (validation only)",
      correct: "Correct / wrong", loss: "Loss", jac: "Input-Jacobian norm", gcos: "Gradient alignment (cosine)", gdot: "Gradient alignment (dot product)" };
    const pxk = isPx(st.color) ? st.color.slice(3) : null, pxLab = pxk && (this.ctx.imgstats?.labels[pxk] || IMGSTATS[pxk]?.label || pxk);
    $("leg-title").textContent = pxk ? `Image statistic: ${pxLab}` : titles[st.color];
    if (col.groups && st.color === "seba") {
      body.append(h("div", { class: "leg-note", text: "Set sizes, class make-up and images: the first card of this panel." }));
      this.renderJaccard();
    } else if (col.groups) {
      const g = col.groups, n = g.n;
      const t = st.fam === "layer" ? st.anchor : st.b;
      const ids = g.perRow ? (() => { const tt = st.at === "final" ? m.steps.length - 1 : t; return Uint8Array.from({ length: N }, (_, i) => this.ctx.pts.correct[tt * N + i] ? 0 : 1); })() : g.ids;
      const cnt = new Array(n).fill(0), comp = [...Array(n)].map(() => new Array(10).fill(0));
      for (let i = 0; i < N; i++) { cnt[ids[i]]++; comp[ids[i]][labels[i]]++; }
      for (let j = 0; j < n; j++) {
        if (j === g.unassigned && cnt[j] === 0) continue;
        const colr = j === g.unassigned ? pal.unassigned : (g.classes ? pal.classes[j] : (st.color === "seba" ? pal.set(j) : col.cols[j]));
        const bar = h("span", { class: "comp" });
        if (!g.classes) comp[j].forEach((c, k) => { if (c) bar.append(h("i", { style: { width: (100 * c / cnt[j]) + "%", background: pal.classes[k] }, title: `${m.classes[k]}: ${c}` })); });
        const row = h("div", { class: "leg-row" + (st.focus === j ? " on" : "") + (st.focus != null && st.focus !== j ? " dim" : ""), role: "button", tabindex: 0,
          title: "Click to isolate; hover to preview" },
          h("span", { class: "sw", style: { background: colr } }), h("span", { text: g.names[j] }), h("span", { class: "n", text: String(cnt[j]) }), bar);
        const toggle = () => { st.focus = st.focus === j ? null : j; if (st.color === "seba") st.j = st.focus === g.unassigned ? null : st.focus; this.col = colouring(this.ctx, st, this.blk); this.renderAll(); this.writeHash(); };
        row.addEventListener("click", toggle);
        row.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
        row.addEventListener("pointerenter", () => { if (st.focus != null) return; const save = st.focus; st.focus = j; this.col = colouring(this.ctx, st, this.blk); this.draw(); this.drawLens(); st.focus = save; });
        row.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, this.blk); this.draw(); this.drawLens(); });
        body.append(row);
      }
      if (!g.classes) body.append(h("div", { class: "leg-note", text: g.perRow ? `Counts and class make-up at step ${m.steps[st.at === "final" ? m.steps.length - 1 : t].toLocaleString("en")}. Bars: class make-up (labels used only to validate).` : "Bars: class make-up of each group (class colours as in ‘CIFAR class’; labels only validate)." }));
      if (st.color === "seba") this.renderJaccard();
    } else if (col.cont) {
      const c = col.cont;
      body.append(h("div", { class: "cbar", style: { background: c.kind === "div" ? pal.divGradient() : pal.seqGradient() } }));
      const tick = v => c.log ? (Math.pow(10, v) >= 100 ? Math.pow(10, v).toFixed(0) : Math.pow(10, v).toPrecision(2)) : (+v.toPrecision(2)).toString();
      body.append(h("div", { class: "cbar-ticks" }, h("span", { text: (c.kind === "div" && !c.log ? "≤ " : "≤ ") + tick(c.lo) }), h("span", { text: c.kind === "div" ? "0" : "" }), h("span", { text: "≥ " + tick(c.hi) })));
      const notes = {
        eig: `Diverging colour of ψ${st.k} scaled by max|ψ${st.k}|; grey = near zero. Signs are arbitrary (fixed so the largest entry is positive).`,
        loss: `Log colour scale, fixed across rows (1st to 99th percentile over all checkpoints). At ${st.at === "final" ? "the final checkpoint for every row" : "each row's own checkpoint"}.`,
        jac: `Frobenius norm of d logits / d input (log scale, fixed across rows). At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}.`,
        gcos: `Cosine between the image's loss gradient (all parameters) and the mean gradient over the 2,000 probe images. At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}. Uses labels through the loss.`,
        gdot: `Dot product of the image's loss gradient with the mean gradient (scale: 99th percentile of |value|). At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}. Uses labels through the loss.`,
      };
      if (pxk) notes[st.color] = pxDef(pxk) + " Computed from the probe image's pixels, so it is the same in every panel. Labels do not enter it.";
      body.append(h("div", { class: "leg-note", text: notes[st.color] || "" }));
    }
  }

  async renderJaccard() {
    const st = this.st, ctx = this.ctx, { m, labels, N } = ctx, box = $("leg-jac");
    const ri = m.seba_r.indexOf(st.r), A = this.blk.lab.subarray(ri * N, (ri + 1) * N);
    const key = `${st.fam}/${st.anchor}/${st.mode}/${st.a},${st.b}/${st.r}`;
    const vsC = this._jc && this._jc.key === key ? this._jc.res : matchedJaccard(A, st.r, labels, 10);
    this._jc = { key, res: vsC };
    box.replaceChildren(
      h("div", { class: "jrow" }, h("span", { text: "Matched Jaccard vs CIFAR classes" }), h("span", {}, h("span", { class: "jv", text: vsC.mean.toFixed(2) }), h("span", { class: "jnull", text: `  null ${vsC.nullMean.toFixed(2)}` }))),
      h("div", { class: "leg-note", text: vsC.per.map((x, i) => `set ${i + 1}→${x.j >= 0 ? m.classes[x.j] : "none"} ${x.J.toFixed(2)}`).join(" · ") }),
    );
    // comparison block (same family, same r, same kappa rule)
    const lim = st.fam === "layer" ? m.steps.length : m.sites.length;
    if (!st.cmp || st.cmp.fam !== st.fam) {
      const anc = st.fam === "layer" ? (st.anchor === lim - 1 ? lim - 2 : lim - 1) : (st.anchor === lim - 1 ? lim - 2 : st.anchor + 1);
      st.cmp = { fam: st.fam, anchor: anc, a: st.a, b: st.b };
    }
    const c = st.cmp;
    const sel = (opts, val, on) => { const s = h("select"); opts.forEach((o, i) => s.add(new Option(o, i))); s.value = val; s.addEventListener("change", () => { on(+s.value); this.renderJaccard(); }); return s; };
    const ancNames = st.fam === "layer" ? m.steps.map(s => "step " + s.toLocaleString("en")) : m.sites;
    const posNames = st.fam === "layer" ? m.sites : m.steps.map(s => s.toLocaleString("en"));
    const out = h("span", { class: "jv", text: "…" }), nul = h("span", { class: "jnull" });
    box.append(h("div", { class: "jcmp" }, h("span", { text: "vs block" }), sel(ancNames, c.anchor, v => c.anchor = v), sel(posNames, c.a, v => { c.a = v; if (c.b < v) c.b = v; }), h("span", { text: "to" }), sel(posNames, c.b, v => { c.b = v; if (c.a > v) c.a = v; })),
      h("div", { class: "jrow" }, h("span", { text: `same kernel rule, r = ${st.r}` }), h("span", {}, out, nul)));
    const per = h("div", { class: "leg-note" }); box.append(per);
    box.append(h("div", { class: "leg-note", text: "Hard SEBA sets matched one-to-one by the Hungarian assignment maximising total Jaccard; mean over this block's sets. Null: same set sizes, random members (30 draws)." }));
    try {
      const other = await D.block(st.fam, c.anchor, st.mode, ctx.pairIdx[st.fam][c.a + "," + c.b]);
      const B = other.lab.subarray(ri * N, (ri + 1) * N);
      const res = matchedJaccard(A, st.r, B, st.r);
      out.textContent = res.mean.toFixed(2); nul.textContent = `  null ${res.nullMean.toFixed(2)}`;
      per.textContent = res.per.map((x, i) => `set ${i + 1}→${x.j + 1} ${x.J.toFixed(2)}`).join(" · ");
    } catch (e) { out.textContent = "n/a"; }
  }

  target() {
    // which function the side panel describes: {kind: 'psi', k} or {kind:'set', ri, j|null}
    const st = this.st, m = this.ctx.m;
    if (st.kind === "psi") return { kind: "psi", k: st.k, label: `ψ${st.k}` };
    const ri = m.seba_r.indexOf(st.r);
    return { kind: "set", ri, j: st.j, label: st.j == null ? `SEBA sets (r = ${st.r})` : `set ${st.j + 1} of ${st.r}` };
  }
  // The first side-panel card, visible without clicking: for the selected sets, one row per set (swatch, size, class
  // make-up bar, strip of its highest-membership images) plus the unassigned images; for psi_k, its two ends.
  // Images come from the build-time representatives (D.reps); class labels only describe the sets.
  renderImages() {
    const st = this.st, { m, pal, labels, N } = this.ctx, blk = this.blk, body = $("img-body"), tg = this.target();
    body.replaceChildren();
    const W = body.clientWidth || 300, TH = W >= 560 ? 44 : 40, per = Math.max(5, Math.floor((W + 3) / (TH + 3)));
    const strip = (ids, rows = 1) => {
      const g = h("div", { class: "imgs imgs-strip", style: { gridTemplateColumns: `repeat(${per}, minmax(0, 1fr))` } });
      ids.slice(0, per * rows).forEach(i => {
        const c = this.thumb(i);
        c.addEventListener("pointerenter", () => this.setHover(i, this.lensCell[0]));
        c.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, this.pointTip(i, this.lensCell[0], this.lensCell[1])));
        c.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
        g.append(c);
      });
      return g;
    };
    const compBar = (cnt, total) => {
      const bar = h("span", { class: "comp comp-lg" });
      cnt.forEach((c, k) => { if (c) bar.append(h("i", { style: { width: (100 * c / total) + "%", background: pal.classes[k] } })); });
      bar.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`class make-up (${total.toLocaleString("en")} images; validation only)`,
        cnt.map((c, k) => ({ name: m.classes[k], value: `${c} (${(100 * c / total).toFixed(0)}%)`, color: pal.classes[k], c })).filter(r => r.c).sort((x, y) => y.c - x.c))));
      bar.addEventListener("pointerleave", hideTip);
      return bar;
    };
    $("img-sub").textContent = `${blockKind(st)}: ${blockLabel(this.ctx, st)} · ${m.modes[st.mode]}`;
    if (tg.kind === "psi") {
      $("img-title").textContent = `Images at the ends of ψ${tg.k}`;
      for (const [b, name, col] of [[1, `most negative ψ${tg.k}`, pal.p.diverging.neg], [0, `most positive ψ${tg.k}`, pal.p.diverging.pos], [2, `ψ${tg.k} nearest zero`, pal.p.diverging.mid]]) {
        body.append(h("div", { class: "set-row" + (b === 2 ? " minor" : "") }, h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: name })),
          strip(D.reps(m, blk, "psi", tg.k, b), b === 2 ? 1 : 2)));
      }
      body.append(h("div", { class: "card-note", text: "Images with the largest and smallest entries of the block's eigenfunction (computed at full precision). Hover an image to ring it in every panel." }));
      return;
    }
    const r = st.r, lab = blk.lab.subarray(tg.ri * N, (tg.ri + 1) * N);
    const cnt = [...Array(r + 1)].map(() => new Array(10).fill(0)), size = new Array(r + 1).fill(0);
    for (let i = 0; i < N; i++) { const g = lab[i] < 0 ? r : lab[i]; size[g]++; cnt[g][labels[i]]++; }
    $("img-title").textContent = `Coherent sets, r = ${r}: images and class make-up`;
    const seba = st.color === "seba";
    for (let j = 0; j <= r; j++) {
      if (j === r && !size[r]) continue;
      const un = j === r, on = un ? (seba && st.focus === r) : tg.j === j, dim = (tg.j != null && !on) || (un && seba && st.focus != null && st.focus !== r);
      const head = h("div", { class: "set-head", role: "button", tabindex: 0, title: un ? "Click to isolate the unassigned images" : "Click to select this set (more images, its boundary, its retention and loadings)" },
        h("span", { class: "sw", style: { background: un ? pal.unassigned : pal.set(j) } }), h("span", { class: "set-name", text: un ? "unassigned" : `set ${j + 1}` }),
        h("span", { class: "n", text: size[j].toLocaleString("en") }), compBar(cnt[j], size[j] || 1));
      const pick = () => {
        if (un) { if (!seba) return; st.focus = st.focus === r ? null : r; st.j = null; }
        else { st.j = st.j === j ? null : j; if (seba) st.focus = st.j; }
        this.col = colouring(this.ctx, st, blk); this.renderAll(); this.writeHash();
      };
      head.addEventListener("click", pick);
      head.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      if (seba) {
        head.addEventListener("pointerenter", () => { if (st.focus != null) return; st.focus = j; this.col = colouring(this.ctx, st, blk); this.draw(); this.drawLens(); st.focus = null; });
        head.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, blk); this.draw(); this.drawLens(); });
      }
      const row = h("div", { class: "set-row" + (on ? " on" : "") + (dim ? " dim" : "") }, head);
      if (!un) {
        const rp = D.reps(m, blk, "set", tg.ri, j);
        row.append(strip(rp.top, on ? 2 : 1));
        if (on && r > 1) row.append(h("div", { class: "imgs-label", text: "boundary members (smallest lead over the runner-up set)" }), strip(rp.bnd, 1));
      }
      body.append(row);
    }
    const key = h("div", { class: "class-key" });
    m.classes.forEach((c, k) => key.append(h("span", {}, h("i", { style: { background: pal.classes[k] } }), c)));
    body.append(key, h("div", { class: "card-note", text: "Bars: CIFAR class make-up of each set (validation only: labels describe the sets and play no part in finding them). Thumbnails: each set's members with the highest SEBA value. Click a set for more of its images and its boundary; hover an image to ring it in every panel." }));
  }

  renderRetention() {
    const st = this.st, { m, pal } = this.ctx, blk = this.blk, tg = this.target(), X = blk.X, R = m.seba_r.length;
    const xs = [...Array(X).keys()], names = st.fam === "layer" ? m.sites : m.steps.map(fmtStep);
    let series = [];
    if (tg.kind === "psi") series = [{ name: `ψ${tg.k}`, y: Array.from(blk.ret_psi.subarray((tg.k - 1) * X, tg.k * X)), color: "#16161a", dots: 2.5 }];
    else {
      const js = tg.j == null ? [...Array(st.r).keys()] : [tg.j];
      const so = m.seba_r.slice(0, tg.ri).reduce((x, y) => x + y, 0);
      series = js.map(j => ({ name: `set ${j + 1}`, y: Array.from(blk.ret_seba.subarray((so + j) * X, (so + j + 1) * X)), color: pal.set(j), dots: 2.5 }));
    }
    const all = series.flatMap(s => s.y).filter(Number.isFinite);
    const lo = Math.min(.8, Math.floor((Math.min(...all) - .01) * 10) / 10);
    lineChart($("ret-body"), {
      x: xs, series, yrange: [lo, 1.0], height: 150, xticks: xs.filter(i => X <= 8 || (X - 1 - i) % 2 === 0), xfmt: i => names[i], shade: [st.a - .45, st.b + .45],
      yfmt: v => v.toFixed(2), refline: tg.kind === "psi" ? { y: blk.vals[tg.k], label: `λ${tg.k} (block mean)` } : null, ml: 36, xname: st.fam === "layer" ? "site" : "step",
    });
    $("ret-title").innerHTML = `Retention ρ<sub>c</sub>(${tg.kind === "psi" ? `ψ<sub>${tg.k}</sub>` : tg.j == null ? "SEBA vectors" : `SEBA vector ${tg.j + 1}`}) in every snapshot`;
    $("ret-note").innerHTML = `ρ<sub>c</sub>(f) = f<sup>T</sup>M<sub>c</sub>f / f<sup>T</sup>f for every ${st.fam === "layer" ? "site at this checkpoint" : "checkpoint at this site"}, with f = ${tg.kind === "psi" ? `ψ<sub>${tg.k}</sub>` : "each SEBA vector (colours as the sets)"}; the block's snapshots are shaded. ${tg.kind === "psi" ? `λ<sub>${tg.k}</sub> (dashed) is the mean of ρ<sub>c</sub> over the shaded snapshots: an <b>average</b>, not a value in every snapshot.` : "The spectrum is an <b>average</b> of retention over the shaded snapshots, not a value in every snapshot."}`;
  }

  renderStats() {
    const st = this.st, { m, T, S, N } = this.ctx, blk = this.blk, tg = this.target(), body = $("stats-body");
    const ri = tg.kind === "set" ? tg.ri : 0, j = tg.kind === "set" ? (tg.j ?? 0) : 0;
    $("stats-title").textContent = st.fam === "layer" ? `Sites at step ${m.steps[st.anchor].toLocaleString("en")}` : `Checkpoints at ${m.sites[st.anchor]}`;
    const tb = h("table", { class: "mini" });
    tb.append(h("tr", {}, h("th", { text: st.fam === "layer" ? "site" : "step" }), h("th", { text: "κ" }), h("th", { text: "15-NN mass" }), h("th", { text: "ρ_c" })));
    const X = blk.X;
    for (let x = 0; x < X; x++) {
      const [t, s] = st.fam === "layer" ? [st.anchor, x] : [x, st.anchor];
      const ret = tg.kind === "psi" ? blk.ret_psi[(tg.k - 1) * X + x] : blk.ret_seba[(m.seba_r.slice(0, ri).reduce((p, q) => p + q, 0) + j) * X + x];
      const tr = h("tr", { class: inBlock(st, t, s) ? "inb" : "" },
        h("td", { text: st.fam === "layer" ? m.sites[s] : m.steps[t].toLocaleString("en") }), h("td", { text: m.kappa?.[st.mode]?.[t]?.[s] != null ? m.kappa[st.mode][t][s].toPrecision(3) : "–" }),
        h("td", { text: pct(m.nn_mass?.[st.mode]?.[t]?.[s] ?? NaN) }), h("td", { text: fmt(ret, 3) }));
      tb.append(tr);
    }
    body.replaceChildren(tb, h("div", { class: "card-note", text: `κ (per-image median for perplexity rules; none for the kNN graph) and median 15-NN kernel mass under ${m.modes[st.mode]}; retention ρ_c of ${tg.kind === "psi" ? "ψ" + tg.k : "SEBA vector " + (j + 1)}. Bold rows are in the block.` }));
  }

  renderKey() {
    const st = this.st, { m, pal } = this.ctx, col = this.col, box = $("ex-key");
    box.replaceChildren();
    if (col.groups) {
      const g = col.groups;
      for (let j = 0; j < g.n; j++) {
        if (j === g.unassigned && !col.ci[0].includes(j) && !(st.focus != null)) continue;
        const c = j === g.unassigned ? pal.unassigned : (g.classes ? pal.classes[j] : (st.color === "seba" ? pal.set(j) : col.cols[j]));
        box.append(h("span", { class: "kk" }, h("i", { style: { background: c } }), g.names[j]));
      }
    } else if (col.cont) {
      const c = col.cont, tick = v => c.log ? (Math.pow(10, v) >= 100 ? Math.pow(10, v).toFixed(0) : Math.pow(10, v).toPrecision(2)) : (+v.toPrecision(2)).toString();
      box.append(h("span", { class: "kt", text: c.label }), h("span", { class: "kt", text: tick(c.lo) }),
        h("span", { class: "kbar", style: { background: c.kind === "div" ? pal.divGradient() : pal.seqGradient() } }), h("span", { class: "kt", text: tick(c.hi) }));
    }
  }
  renderCaption() {
    this.renderKey();
    $("ex-caption").replaceChildren(...captionParts(this.ctx, this.st, this.col));
  }
}

// definition and colour scale of a pixel statistic, as one sentence
function pxDef(k) {
  const d = IMGSTATS[k];
  return `${d ? d.def[0].toUpperCase() + d.def.slice(1) : k}; pixel values in [0, 1]${/^(luminance|contrast|bc_lum|hf_frac)$/.test(k) ? ", Y = 0.299 R + 0.587 G + 0.114 B" : ""}${/^bc_/.test(k) ? "; border = pixels within 4 px of the edge, centre = the central 16 × 16" : ""}. ` +
    (d?.div ? "Diverging scale, symmetric, clipped at the 99th percentile of |value| over the probe images; largest |value| on top." : "Sequential scale from the 1st to the 99th percentile over the probe images; largest value on top.");
}

// Definitional caption of an explorer state: block, kernel rule, colouring (with r or k), layout. Never interpretive.
export function captionParts(ctx, st, col) {
  const { m, T, S } = ctx, step = t => `step ${m.steps[t].toLocaleString("en")}`, n = st.b - st.a + 1;
  let blk;
  if (st.fam === "layer") blk = st.a === st.b ? `site ${m.sites[st.a]} at ${step(st.anchor)}`
    : `sites ${m.sites[st.a]} to ${m.sites[st.b]} at ${step(st.anchor)} (${n} snapshots)`;
  else blk = st.a === st.b ? `site ${m.sites[st.anchor]} at ${step(st.a)}`
    : `site ${m.sites[st.anchor]}, checkpoints ${step(st.a)} to ${step(st.b)} (${n} snapshots)`;
  const fromBlk = "computed on the outlined block only and painted on every panel; ";
  const focus = col && col.groups && st.focus != null && st.focus < col.groups.n ? ` Only ${col.groups.names[st.focus]} is coloured; the rest are light grey.` : "";
  const at = st.at === "final" ? `at the final checkpoint (${step(T - 1)}) in every row` : "at each row's own checkpoint";
  const side = st.kind === "psi" ? `side panel: ψ${st.k}` : `side panel: r = ${st.r} coherent sets`;
  const colour = {
    seba: `coherent sets, r = ${st.r}. Each image's hard label is the largest of the ${st.r} SEBA vectors from ψ0..ψ${st.r - 1} of the block operator M̄; grey = unassigned (no positive entry); ${fromBlk}points drawn in order of membership value, largest on top.`,
    eig: `ψ${st.k} / max|ψ${st.k}|, where ψ${st.k} is the eigenfunction of the block operator M̄ for λ${st.k}, eigenvalues in decreasing order from λ0 = 1 (the constant); diverging scale, sign fixed so the largest-magnitude entry is positive; ${fromBlk}largest |ψ| on top.`,
    nodal: `sign of ψ${st.k} of the block operator M̄ (ψ${st.k} > 0, < 0, = 0); ${fromBlk}largest |ψ| on top.`,
    class: `CIFAR-10 class label (validation only). The block sets only the side panel (${side}).`,
    loss: `per-image cross-entropy loss ${at}, log scale, range fixed across rows (1st to 99th percentile over all checkpoints). The block sets only the side panel (${side}).`,
    correct: `whether the network's prediction is correct, ${at}. The block sets only the side panel (${side}).`,
    jac: `Frobenius norm of the input Jacobian of the logits, ${at}, log scale fixed across rows. The block sets only the side panel (${side}).`,
    gcos: `cosine between the image's loss gradient (all parameters) and the mean gradient over the probe images, ${at}. The block sets only the side panel (${side}).`,
    gdot: `dot product of the image's loss gradient with the mean gradient, ${at}. The block sets only the side panel (${side}).`,
  }[st.color] || (isPx(st.color) ? `image statistic ${ctx.imgstats?.labels[st.color.slice(3)] || IMGSTATS[st.color.slice(3)]?.label || st.color.slice(3)}: ${pxDef(st.color.slice(3))} The same per-image value in every panel (a property of the probe image, not of the network). The block sets only the side panel (${side}).` : st.color);
  const lay = m.layouts.length > 1 && st.lay ? `UMAP of seed ${st.lay} (a second network; colours from seed 0)` : "UMAP of each snapshot (AlignedUMAP across sites, Procrustes across checkpoints)";
  const part = (label, text) => [h("b", { text: label + " " }), text + " "];
  return [
    ...part("Block:", `${blockKind(st).toLowerCase()}, ${blk}.`),
    ...part("Kernel rule:", `${m.modes[st.mode]}.`),
    ...part("Colour:", colour + focus),
    ...part("Positions:", `${lay}; display only, no number on this page is computed from them. Rows: checkpoints (step, validation accuracy); columns: sites; ${T * S} panels, one per snapshot, of the same ${m.n.toLocaleString("en")} probe images.`),
  ];
}
