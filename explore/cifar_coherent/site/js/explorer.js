// The explorer: grid of UMAP cells (rows = checkpoints, columns = sites) coloured by the selected block.
import * as D from "./data.js";
import { Palette, divLUT, seqLUT, qDiv, qSeq, hexRgb } from "./colors.js";
import { h, el, showTip, hideTip, tipRows, lineChart, xyChart, scale, niceTicks, fmtStep } from "./charts.js";
import * as SR from "./sources.js";

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
// fixed decimals without a "-0.00"
const nz = d => v => { if (!Number.isFinite(v)) return "n/a"; const t = v.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t; };
const pct = v => Number.isFinite(v) ? (100 * v).toFixed(0) + "%" : "n/a";

// ------------------------------------------------------------------------------------------- shared context
// One context per kernel ("cos": data root d/, the default and what the prose figures use; "dot": d_dot/ when built).
// A context carries its root; blocks, spectra, kappa and layouts come from it. Labels, points and the sprite are shared.
const CTX = new Map();
export function context(kernel = "cos") {
  if (!CTX.has(kernel)) {
    const p = kernel === "cos" ? cosContext() : kernelContext(kernel);
    p.catch(() => CTX.delete(kernel));
    CTX.set(kernel, p);
  }
  return CTX.get(kernel);
}
const pairIndex = m => {
  const pairIdx = { layer: {}, time: {} };
  m.layer_pairs.forEach(([a, b], i) => pairIdx.layer[a + "," + b] = i);
  m.time_pairs.forEach(([a, b], i) => pairIdx.time[a + "," + b] = i);
  return pairIdx;
};
async function cosContext() {
  const m = await D.meta();
  const [labels, layout0, pts, kern] = await Promise.all([D.labels(), D.layout(0), D.points(), D.json("kernels.json").catch(() => ({}))]);
  const pal = new Palette(m.palette);
  // kernels.json (build_site.py link_kernels): other data roots, each kept only if built from the same probe and grid
  const kernels = Object.fromEntries(Object.entries(kern || {}).filter(([, k]) => k && k.root && k.fingerprint && k.fingerprint === m.shared_fingerprint));
  const ctx = { m, pal, labels, layouts: { 0: layout0 }, pts, pairIdx: pairIndex(m), T: m.steps.length, S: m.sites.length, N: m.n, root: "d", kernel: "cos", kernels };
  ctx.diagRange = diagRanges(ctx);
  D.sprite().then(im => { ctx.sprite = im; });
  return ctx;
}
async function kernelContext(kernel) {
  const cos = await context("cos"), k = cos.kernels[kernel];
  if (!k) throw new Error(`no ${kernel}-kernel data root in this build`);
  const m = await D.meta(k.root);
  for (const key of ["steps", "sites", "n", "seba_r", "n_vecs", "layer_pairs", "time_pairs"])
    if (JSON.stringify(m[key]) !== JSON.stringify(cos.m[key])) throw new Error(`${k.root}/meta.json: ${key} differs from d/`);
  if (m.shared_fingerprint !== k.fingerprint) throw new Error(`${k.root}/meta.json: not the root kernels.json describes`);
  const layout0 = await D.layout(0, k.root);
  const ctx = { m, pal: cos.pal, labels: cos.labels, layouts: { 0: layout0 }, pts: cos.pts, pairIdx: pairIndex(m), T: cos.T, S: cos.S, N: cos.N,
    diagRange: cos.diagRange, sprite: cos.sprite, root: k.root, kernel, kernelLabel: k.label };
  if (!ctx.sprite) D.sprite().then(im => { ctx.sprite = im; });
  return ctx;
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
// data key of the block operator: average (layer/time) or ordered product, right (start) or left (end) singular vectors
export const dkey = (st, op = st.op) => !op || op === "avg" ? st.fam : st.fam + (op === "end" ? "prodL" : "prodR");
export const isProd = st => !!st.op && st.op !== "avg";
export const sym = st => isProd(st) ? "σ" : "λ";
export const opName = st => !isProd(st) ? "average M̄" : st.op === "start" ? "ordered product Q, start (right singular vectors)" : "ordered product Q, end (left singular vectors)";
export function blockOf(ctx, st) { return D.block(dkey(st), st.anchor, st.mode, pairOf(ctx, st), ctx.root); }
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
// coarse-graining add-ons (build_site.py coarse(); cosine root, layer family only). A chain may ship for the default rule only.
export const coarseOn = (ctx, st) => ctx.kernel === "cos" && st.fam === "layer" && !!ctx.m.coarse;
export const extOn = (ctx, st) => coarseOn(ctx, st) && D.coarseShipped(ctx.m, "ext", st.mode, st.a === st.b);
export const kmOn = (ctx, st) => coarseOn(ctx, st) && D.coarseShipped(ctx.m, "km", st.mode, st.a === st.b);
export const maxK = (ctx, st) => extOn(ctx, st) ? ctx.m.coarse.ext.last : ctx.m.n_vecs;
// k-means block colours: the set palette, then the same hues darkened (k up to 16)
const kmColour = (pal, j) => j < pal.sets.length ? pal.set(j) : (() => { const [r, g, b] = hexRgb(pal.set(j)); return `rgb(${Math.round(r * .55)},${Math.round(g * .55)},${Math.round(b * .55)})`; })();
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
  if (c === "seba" && st._S) {
    // sets from another source (js/sources.js): hard labels, colours matched to the method's sets
    const S = st._S, k = S.k, ci = new Uint8Array(N), key = new Float64Array(N);
    for (let i = 0; i < N; i++) { const l = S.lab[i]; ci[i] = l < 0 ? k : l; key[i] = l < 0 ? -1 : S.key ? S.key[i] : 0; }
    const cols = (S.classes ? pal.classes.slice() : [...Array(k)].map((_, j) => pal.set(j))).concat([pal.unassigned]);
    const order = S.key ? orderBy(key) : Uint16Array.from(hashOrder).sort((a, b) => (ci[a] === k ? 0 : 1) - (ci[b] === k ? 0 : 1));
    out = { ...same(ci, order), cols, groups: { ids: ci, n: k + 1, names: S.names.concat(["unassigned"]), unassigned: k, classes: S.classes, src: S.id } };
  } else if (c === "seba") {
    const ri = m.seba_r.indexOf(st.r), r = st.r, lab = blk.lab.subarray(ri * N, (ri + 1) * N);
    const vecs = D.sebaVectors(m, blk, ri);
    const ci = new Uint8Array(N), key = new Float64Array(N);
    for (let i = 0; i < N; i++) { const l = lab[i]; ci[i] = l < 0 ? r : l; key[i] = l < 0 ? -1 : vecs[l][i]; }
    const cols = [...Array(r)].map((_, j) => pal.set(j)).concat([pal.unassigned]);
    out = { ...same(ci, orderBy(key)), cols, groups: { ids: ci, n: r + 1, names: [...Array(r)].map((_, j) => `set ${j + 1}`).concat(["unassigned"]), unassigned: r } };
  } else if (c === "km" && st._km) {
    const K = st._km, ci = Uint8Array.from(K.lab), cols = [...Array(K.k)].map((_, j) => kmColour(pal, j));
    out = { ...same(ci, hashOrder), cols, groups: { ids: ci, n: K.k, names: [...Array(K.k)].map((_, j) => `block ${j + 1}`) } };
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
    const ctx = this.ctx = this.ctxCos = await context();
    const { m, T, S } = ctx;
    this.st = {
      fam: "layer", anchor: T - 1, a: 0, b: S - 1, mode: m.default_mode, color: "seba", k: 1, r: m.seba_r.includes(4) ? 4 : m.seba_r[0],
      kind: "set", focus: null, j: null, at: "own", lay: 0, ps: 1.6, cmp: null, tap: "end", op: "avg", src: "M0", kern: "cos",
      kmk: 4, kms: "start", cgSide: "start", cgView: "abs",
    };
    this.srcIdx = await D.srcIndex().catch(() => null);   // baselines study (build_site.py --baselines); null if absent
    // layout coordinates by "<root>:<layout file index>"; st.lay indexes the cosine root's m.layouts (see layIndex)
    this.coordCache = new Map([["d:0", toCoords(ctx.layouts[0])]]);
    this._layKey = "d:0";
    this.cur = this.coordCache.get("d:0");
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
      segInit("ex-op", () => this.st.op, v => { this.st.op = v; this.st.focus = null; this.st.j = null; }),
      segInit("ex-at", () => this.st.at, v => this.st.at = v),
      segInit("ex-lay", () => this.st.lay, v => this.setLayout(+v)),
      segInit("ex-tap", () => this.st.tap, v => { this.st.tap = v; }),
      segInit("ex-kern", () => this.st.kern, v => this.setKernel(v)),
      segInit("ex-kms", () => this.st.kms, v => { this.st.kms = v; this.st.focus = null; }),
    ];
    // coarse-graining card controls (redraw the card only)
    const cgSeg = (id, get, set) => {
      const seg = $(id);
      seg.addEventListener("click", e => { const b = e.target.closest("button"); if (b && !b.disabled) { set(b.dataset.v); seg._sync(); this.renderCoarse(); } });
      seg._sync = () => seg.querySelectorAll("button").forEach(b => { b.setAttribute("role", "radio"); b.setAttribute("aria-checked", String(b.dataset.v === get())); });
      this.segs.push(seg);
    };
    cgSeg("cg-side", () => this.st.cgSide, v => this.st.cgSide = v);
    cgSeg("cg-view", () => this.st.cgView, v => this.st.cgView = v);
    m.coarse?.ks.forEach(k => $("ex-kmk").add(new Option(`k = ${k} blocks`, k)));
    $("ex-kmk").addEventListener("change", e => { this.st.kmk = +e.target.value; this.st.focus = null; this.update(); });
    if (!m.coarse) $("ex-color-km").remove();
    // Kernel switch (more options): only when the build has a second data root (build_site.py --dot-tag)
    $("ex-kern-ctl").hidden = !ctx.kernels.dot;
    if (ctx.kernels.dot) $("ex-kern").querySelector('[data-v="dot"]').textContent = ctx.kernels.dot.label;
    // layout buttons: index into m.layouts; the spectral one finds its own index
    const specIdx = m.layouts.findIndex(l => l.kind === "spectral");
    $("ex-lay-spec").dataset.v = String(specIdx); $("ex-lay-spec").hidden = specIdx < 0;
    $("ex-lay").querySelectorAll("button").forEach(b => { if (+b.dataset.v >= m.layouts.length || (+b.dataset.v > 0 && b !== $("ex-lay-spec") && m.layouts[+b.dataset.v]?.kind)) b.hidden = true; });
    if (m.layouts.length < 2) { $("ex-lay-ctl").hidden = true; }
    if (!m.vals.layerprodR) { $("ex-op-ctl").hidden = true; }   // product blocks not in this build
    m.modes.forEach((name, i) => $("ex-mode").add(new Option(name, i)));
    $("ex-mode").addEventListener("change", e => { this.st.mode = +e.target.value; this.st.focus = null; this.update(); });
    $("ex-anchor").addEventListener("change", e => { this.st.anchor = +e.target.value; this.st.focus = null; this.update(); });
    $("ex-a").addEventListener("change", e => { this.st.a = +e.target.value; if (this.st.b < this.st.a) this.st.b = this.st.a; this.st.focus = null; this.update(); });
    $("ex-b").addEventListener("change", e => { this.st.b = +e.target.value; if (this.st.a > this.st.b) this.st.a = this.st.b; this.st.focus = null; this.update(); });
    // disable diagnostics that the data do not have
    const have = new Set(m.points_fields);
    for (const [c, f] of Object.entries(DIAG)) if (!have.has(f)) { const o = $("ex-color").querySelector(`option[value=${c}]`); if (o) { o.disabled = true; o.dataset.missing = "1"; o.textContent += " (not in data)"; } }
    const og = document.createElement("optgroup"); og.label = "Image statistics (pixels)"; og.dataset.dev = "1";
    for (const [k, d] of Object.entries(IMGSTATS)) og.append(new Option(d.label, "px:" + k));
    $("ex-color").append(og);
    $("ex-color").addEventListener("change", e => this.setColor(e.target.value));
    this.buildSources();
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
    // "more options": kernel rules, nodal domains, diagnostics, image statistics, seed-1 layout, per-snapshot table
    let dev = false; try { dev = localStorage.getItem("cm.explorer.dev") === "1"; } catch (e) { /* private mode */ }
    $("ex-dev").checked = dev; this.setDev(dev, false);
    try { if (dev && ctx.kernels.dot && localStorage.getItem("cm.explorer.kernel") === "dot") this.st.kern = "dot"; } catch (e) { /* private mode */ }
    $("ex-dev").addEventListener("change", e => { this.setDev(e.target.checked, true); this.update(); });
    this.buildGuide();
  }
  // "Sets from": the colour sources of the baselines study, grouped by kind, with status tags
  buildSources() {
    const sel = $("ex-src");
    if (!this.srcIdx) { $("ex-src-ctl").hidden = true; return; }
    let og = null;
    for (const s of SR.SOURCES) {
      if (!og || og.label !== s.group) { og = document.createElement("optgroup"); og.label = s.group; sel.append(og); }
      const o = new Option(`${s.label} · ${s.status}`, s.id); o.dataset.base = o.textContent; o.title = s.def; og.append(o);
    }
    sel.addEventListener("change", e => this.setSource(e.target.value));
    const btn = $("ex-src-btn"), help = $("ex-src-help");
    const setHelp = open => { btn.setAttribute("aria-expanded", String(open)); help.hidden = !open; };
    btn.addEventListener("click", () => setHelp(btn.getAttribute("aria-expanded") !== "true"));
    document.addEventListener("keydown", e => { if (e.key === "Escape") setHelp(false); });
    document.addEventListener("pointerdown", e => { if (!help.hidden && !e.target.closest("#ex-src-help, #ex-src-btn")) setHelp(false); });
  }
  setSource(id) {
    const st = this.st;
    st.src = id; st.focus = null;
    if (id !== "M0") { st.color = "seba"; st.kind = "set"; st.j = null; }
    this.update();
  }
  srcAvail() { return SR.availability(this.srcIdx, this.st, isProd(this.st)); }
  setKernel(k) {
    const st = this.st;
    if (k === st.kern || (k !== "cos" && !this.ctxCos.kernels[k])) return;
    st.kern = k; st.focus = null;
    try { localStorage.setItem("cm.explorer.kernel", k); } catch (e) { /* private mode */ }
  }
  // st.lay indexes the cosine root's layouts; under another kernel, the same layout (seed and kind) or -1 if not shipped
  layIndex(ctx, lay) {
    if (ctx.kernel === "cos") return lay < ctx.m.layouts.length ? lay : -1;
    const L = this.ctxCos.m.layouts[lay];
    return L ? ctx.m.layouts.findIndex(x => x.seed === L.seed && (x.kind || "") === (L.kind || "")) : -1;
  }
  // make the state valid for a context before loading from it; returns notes for the status line
  fitState(ctx, st = this.st) {
    const m = ctx.m, cm = this.ctxCos.m, out = [];
    if (ctx.kernel === "cos") return out;
    if (st.mode >= m.modes.length) {
      out.push(`Rule "${cm.modes[st.mode]}" is not shipped for the dot kernel; showing "${m.modes[m.default_mode]}".`);
      st.mode = m.default_mode; st.focus = null;
    }
    if (this.layIndex(ctx, st.lay) < 0) { out.push("No seed-1 layout for the dot kernel; showing seed 0."); st.lay = 0; }
    if (st.src !== "M0") { out.push("Sets from is computed for the cosine kernel only; showing the method."); st.src = "M0"; }
    if (isProd(st) && !m.vals.layerprodR) { out.push("No product blocks for the dot kernel; showing the average."); st.op = "avg"; }
    return out;
  }
  // psi_9..psi_32 and the k-means colouring exist only where the coarse add-ons ship (cosine root, layer blocks; chains
  // possibly for the default rule only): clamp to what the target context has, with a note for the status line
  fitCoarse(ctx, st) {
    const out = [], m = ctx.m, cm = this.ctxCos.m;
    const why = ctx.kernel !== "cos" ? "the cosine kernel only" : st.fam !== "layer" ? "layer blocks only" : !cm.coarse ? "not in this build"
      : `multi-site blocks under the default kernel rule (${cm.modes[cm.default_mode]}) only`;
    if (st.kind === "psi" && st.k > maxK(ctx, st)) {
      out.push(`ψ9..ψ32: ${why}; showing ψ${m.n_vecs}.`);
      st.k = m.n_vecs; st.focus = null;
    }
    if (st.color === "km" && !kmOn(ctx, st)) {
      out.push(`Spectral k-means partitions: ${why}; showing coherent sets.`);
      st.color = "seba"; st.kind = "set"; st.focus = null;
    }
    if (st.color === "km" && !m.coarse.ks.includes(st.kmk)) st.kmk = m.coarse.ks[0];
    return out;
  }
  // dev-only state: anything a plain view cannot show (used to switch dev mode on for shared links and tours)
  needsDev() {
    const st = this.st, m = this.ctxCos.m, L = m.layouts[st.lay] || {};
    const c = $("ex-color").querySelector(`option[value="${CSS.escape(st.color)}"]`);
    return st.kern !== "cos" || st.mode !== m.default_mode || (st.lay > 0 && L.kind !== "spectral") || !!(c && (c.dataset.dev || c.parentElement.dataset.dev));
  }
  setDev(on, remember) {
    const st = this.st, m = this.ctxCos.m;
    this.dev = on; $("ex").classList.toggle("dev", on); $("ex-dev").checked = on;
    // <option>/<optgroup> cannot be hidden by CSS everywhere: set hidden + disabled
    $("ex-color").querySelectorAll("[data-dev]").forEach(e => { e.hidden = !on; e.disabled = !on || e.dataset.missing === "1"; });
    if (remember) try { localStorage.setItem("cm.explorer.dev", on ? "1" : "0"); } catch (e) { /* private mode */ }
    if (!on && this.needsDev()) {
      if (st.kern !== "cos") { st.kern = "cos"; if (remember) try { localStorage.setItem("cm.explorer.kernel", "cos"); } catch (e) { /* private mode */ } }
      if (st.mode !== m.default_mode) st.mode = m.default_mode;
      const L = m.layouts[st.lay] || {}; if (st.lay > 0 && L.kind !== "spectral") this.setLayout(0);
      if (this.needsDev()) { st.color = st.kind === "psi" ? "eig" : "seba"; st.focus = null; }
    }
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
    if (c !== "seba") st.src = "M0";      // other sources give sets only
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
    if (seed === this.st.lay) return;
    this.st.lay = seed;
    this.showLayout();
  }
  loadCoords(ctx, li) {
    const key = `${ctx.root}:${li}`, c = this.coordCache.get(key);
    if (c || li < 0) return Promise.resolve(c);
    return D.layout(li, ctx.root).then(u => { const cc = toCoords(u); this.coordCache.set(key, cc); return cc; });
  }
  // move the points to the active context's layout for st.lay (animated); a no-op when already there
  showLayout() {
    const ctx = this.ctx, li = this.layIndex(ctx, this.st.lay), key = `${ctx.root}:${li}`;
    if (li < 0 || key === this._layKey) return;
    this._layKey = key;
    this.loadCoords(ctx, li).then(c => { if (this._layKey === key) this.animateTo(c); }).catch(e => console.error(e));
  }
  animateTo(to) {
    const from = this.cur, N = from.length, t0 = performance.now(), dur = 650;
    const tmp = new Float32Array(N);
    cancelAnimationFrame(this._anim);
    const step = now => {
      const u = Math.min(1, (now - t0) / dur), e = u < .5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      for (let i = 0; i < N; i++) tmp[i] = from[i] + (to[i] - from[i]) * e;
      this.cur = u < 1 ? tmp : to;
      if (this.col && this.geom) { this.draw(); this.drawOverlay(); this.drawLens(); }
      if (u < 1) this._anim = requestAnimationFrame(step);
    };
    this._anim = requestAnimationFrame(step);
  }

  syncControls(ctx = this.ctx) {
    const st = this.st, { m, T, S } = ctx;
    this.segs.forEach(s => s._sync());
    this.syncKernel(ctx);
    $("ex-mode").value = st.mode;
    $("ex-color").value = st.color;
    if (this.srcIdx) {
      const av = this.srcAvail(), sel = $("ex-src");
      sel.querySelectorAll("option").forEach(o => { const why = av[o.value]; o.disabled = !!why; o.textContent = o.dataset.base + (why ? ` (n/a: ${why})` : ""); });
      sel.value = st.src;
      $("ex-src-note").hidden = st.mode === this.srcIdx.kernel_rule && !isProd(st) && ctx.kernel === "cos";
      $("ex-src-note").textContent = ctx.kernel !== "cos" ? "Other sources: cosine kernel only" : st.mode !== this.srcIdx.kernel_rule ? `Other sources: kernel rule ${this.ctxCos.m.modes[this.srcIdx.kernel_rule]} only` : "Other sources: average operator only";
    }
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
    const want = st.kind === "psi" ? "psi" + maxK(ctx, st) : "set";
    if (ks._kind !== want) {
      ks._kind = want; ks.innerHTML = "";
      if (st.kind === "psi") for (let k = 1; k <= maxK(ctx, st); k++) ks.add(new Option(`ψ${k}`, k));
      else m.seba_r.forEach(r => ks.add(new Option(`r = ${r} sets`, r)));
    }
    // spectral k-means colouring: layer blocks of the cosine root that ship partitions (a chain: start or end side)
    const km = st.color === "km", kmo = $("ex-color-km");
    if (kmo) { const on = kmOn(ctx, st); kmo.disabled = !on && !km; kmo.title = on ? "" : "Shipped for layer blocks of the cosine kernel: single cells for every kernel rule, multi-site blocks for the default rule only"; }
    $("ex-kmk").hidden = !km; $("ex-kms").hidden = !km || st.a === st.b;
    if (km) $("ex-kmk").value = st.kmk;
    ks.value = st.kind === "psi" ? st.k : st.r;
    ks.title = BLOCK_KINDS.has(st.color) ? "" : "Selection shown in the side panel";
    $("ex-at").hidden = !(st.color in DIAG || st.color === "correct");
    $("ex-k-label").hidden = BLOCK_KINDS.has(st.color);
  }

  // controls a non-cosine data root does not ship are disabled with the reason; the dot badge shows which root is active
  syncKernel(ctx) {
    const dot = ctx.kernel !== "cos", m = ctx.m, cm = this.ctxCos.m, kname = dot ? "the dot kernel" : "";
    const dis = (e, off, why) => { if (e.dataset.t0 == null) e.dataset.t0 = e.title || ""; e.disabled = off; e.title = off ? why : e.dataset.t0; };
    $("ex-mode").querySelectorAll("option").forEach((o, i) => {
      const off = i >= m.modes.length;
      dis(o, off, `Not shipped for ${kname}: the site ships its first ${m.modes.length} rules only (size cap)`);
      o.textContent = cm.modes[i] + (off ? " (not shipped for dot)" : "");
    });
    $("ex-lay").querySelectorAll("button").forEach(b => { if (!b.hidden) dis(b, this.layIndex(ctx, +b.dataset.v) < 0, `Not available for ${kname}: only the seed-0 network was measured with it`); });
    $("ex-op").querySelectorAll("button").forEach(b => dis(b, b.dataset.v !== "avg" && !m.vals.layerprodR, `No product blocks for ${kname}`));
    if (this.srcIdx) dis($("ex-src"), dot, "Sets from uses the baselines study, computed for the cosine kernel only");
    $("ex-kern-badge").hidden = !dot;
    if (dot) $("ex-kern-badge").textContent = `kernel: ${ctx.kernelLabel || "dot"}`;
    $("ex").classList.toggle("kern-dot", dot);
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
    if (!this.drag) { const nb = this.clickTarget(hc, e.shiftKey); if (nb) blockOf(this.ctx, nb).catch(() => {}); }
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
    const st = this.st, out = { fam: st.fam, mode: st.mode, op: st.op };
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
      if (st.kind === "set" && st._S) {
        const S = st._S, l = S.lab[i];
        rows.push({ name: `${SR.SRC[S.id].label}${S.classes ? "" : ` (r = ${st.r})`}`, value: l < 0 ? "unassigned" : S.names[l], color: l < 0 ? this.ctx.pal.unassigned : S.classes ? this.ctx.pal.classes[l] : this.ctx.pal.set(l) });
        const l0 = blk.lab[m.seba_r.indexOf(st.r) * N + i];
        rows.push({ name: `the method (r = ${st.r})`, value: l0 < 0 ? "unassigned" : `set ${l0 + 1}`, color: l0 < 0 ? this.ctx.pal.unassigned : this.ctx.pal.set(l0) });
      } else if (st.kind === "set") {
        const ri = m.seba_r.indexOf(st.r), l = blk.lab[ri * N + i], v = D.sebaVectors(m, blk, ri);
        rows.push({ name: `SEBA (r = ${st.r})`, value: l < 0 ? "unassigned" : `set ${l + 1} (${fmt(v[l][i])})`, color: l < 0 ? this.ctx.pal.unassigned : this.ctx.pal.set(l) });
      } else {
        const p = D.psi(m, blk, st.k); let mx = 0; for (const x of p) mx = Math.max(mx, Math.abs(x));
        rows.push({ name: `ψ${st.k} / max|ψ|`, value: fmt(p[i] / mx) });
      }
      if (st.color === "km" && st._km) rows.push({ name: `spectral k-means, k = ${st._km.k}${st.a === st.b ? "" : `, ${st._km.side} side`}`, value: `block ${st._km.lab[i] + 1}`, color: this.col?.cols[st._km.lab[i]] });
      if (st.color in DIAG && st.color !== "loss") { const f = DIAG[st.color]; rows.push({ name: f, value: pts[f][tt * N + i].toPrecision(3) }); }
      if (isPx(st.color) && this.ctx.imgstats) { const k = st.color.slice(3); rows.push({ name: this.ctx.imgstats.labels[k] || k, value: this.ctx.imgstats.per[k][i].toFixed(3) }); }
    }
    const d = h("div", { class: "tt-img" }, this.thumb(i), tipRows(`#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`, rows));
    return d;
  }

  // ---------------------------------------------------------------- state -> render
  // A kernel switch commits nothing until the other root's context, block and layout have all loaded: until then the badge,
  // outline, labels, disabled controls and hash describe the kernel on screen (only the Kernel switch shows the pending
  // choice). The state is fitted on a copy and copied back at the commit. A failed load reverts the switch.
  async update() {
    const st = this.st, token = this._tok = (this._tok || 0) + 1;
    let ctx = this.ctx;
    const revert = (e, what) => {
      console.error(e);
      if (token !== this._tok) return;
      st.kern = this.ctx.kernel;
      try { localStorage.setItem("cm.explorer.kernel", st.kern); } catch (e2) { /* private mode */ }
      this.syncControls(); this.writeHash();
      $("ex-status").textContent = `the dot-kernel ${what} failed to load; still showing ${this.ctx.kernel === "cos" ? "cosine" : "dot"}`;
      $("ex-side").style.opacity = 1;
    };
    if (ctx.kernel !== st.kern) {        // kernel switch: everything below reads the other data root
      this.segs.forEach(s => s._sync());
      $("ex-status").textContent = `loading the ${st.kern} kernel…`;
      try { ctx = await context(st.kern); } catch (e) { return revert(e, "data"); }
      if (token !== this._tok) return;
    }
    const switching = ctx !== this.ctx, nst = switching ? { ...st } : st;
    const notes = this.fitState(ctx, nst).concat(this.fitCoarse(ctx, nst));
    if (switching) this.segs.forEach(s => s._sync());
    else { this.syncControls(ctx); this.writeHash(); if (notes.length) this._sticky = { msg: notes.join(" "), hash: this.lastHash }; }
    $("ex-status").textContent = switching ? `loading the ${nst.kern} kernel…` : "loading…";
    $("ex-side").style.opacity = .55;
    let blk, km = null;
    try {
      [blk] = await Promise.all([blockOf(ctx, nst), this.loadCoords(ctx, this.layIndex(ctx, nst.lay))]);
      const pi = pairOf(ctx, nst), cell = nst.a === nst.b;
      if (nst.kind === "psi" && nst.k > ctx.m.n_vecs && !blk.ext) blk.ext = await D.ext(ctx.m, dkey(nst), nst.anchor, nst.mode, pi, cell);
      if (nst.color === "km") {
        const L = await D.kmLabels(ctx.m, ctx.m.steps[nst.anchor], nst.mode, pi, cell), qi = ctx.m.coarse.km.ks.indexOf(nst.kmk);
        const side = cell ? "start" : nst.kms;
        km = { lab: L[side].subarray(qi * ctx.N, (qi + 1) * ctx.N), k: nst.kmk, side };
      }
    }
    catch (e) {
      if (switching) return revert(e, "block");
      if (token === this._tok) $("ex-status").textContent = "failed to load block"; console.error(e); return;
    }
    if (isPx(nst.color) && !ctx.imgstats) {
      $("ex-status").textContent = "loading image statistics…";
      try { await prepare(ctx, nst); } catch (e) { $("ex-status").textContent = "image statistics unavailable"; console.error(e); return; }
    }
    let S = null, srcMsg = "";
    if (nst.src !== "M0") {           // never while switching to dot: fitState sets the method
      const why = this.srcAvail()[st.src];
      if (why) { srcMsg = `Sets from: "${SR.SRC[st.src].label}" is ${why}; showing the method instead.`; st.src = "M0"; this.syncControls(); this.writeHash(); }
      else {
        try { S = await SR.load(ctx, st, st.src, blk.lab.subarray(ctx.m.seba_r.indexOf(st.r) * ctx.N, (ctx.m.seba_r.indexOf(st.r) + 1) * ctx.N)); }
        catch (e) { console.error(e); srcMsg = "this source failed to load; showing the method"; st.src = "M0"; this.syncControls(); this.writeHash(); }
      }
    }
    if (token !== this._tok) return;
    if (switching) for (const k of ["mode", "lay", "src", "op", "focus", "k", "color", "kind"]) st[k] = nst[k];
    this.ctx = ctx; this.blk = blk; st._S = S; st._km = km;    // context, block and layout change together: no cosine block under a dot label
    $("ex-side").style.opacity = 1;
    this.col = colouring(ctx, st, blk);
    this.syncControls();
    this.writeHash();
    // a clamp note stays while the view it describes is on screen (a second update of the same state would clear it)
    let msg = [srcMsg, ...notes].filter(Boolean).join(" ");
    if (notes.length) this._sticky = { msg, hash: this.lastHash };
    else if (!msg && this._sticky?.hash === this.lastHash) msg = this._sticky.msg;
    $("ex-status").textContent = msg;
    this.showLayout();
    this.renderAll();
    this.prefetch();
  }
  renderAll() {
    if (!this.col || !this.geom || this.blk?.root !== this.ctx.root) return;
    this.draw(); this.drawOverlay(); this.drawLens(); this.decorate();
    this.renderBlockCard(); this.renderLegend(); this.renderImages(); this.renderRetention(); this.renderStats(); this.renderCaption(); this.renderCoarse();
  }
  prefetch() {
    const st = this.st, ctx = this.ctx, lim = st.fam === "layer" ? ctx.S : ctx.T;
    const near = [[st.a, st.b + 1], [st.a + 1, st.b], [st.a - 1, st.b], [st.a, st.b - 1]].filter(([a, b]) => a >= 0 && b < lim && a <= b);
    const go = () => near.forEach(([a, b]) => D.block(dkey(st), st.anchor, st.mode, ctx.pairIdx[st.fam][a + "," + b], ctx.root).catch(() => {}));
    setTimeout(go, 250);
  }
  writeHash() {
    const st = this.st;
    const p = new URLSearchParams({ fam: st.fam, t: st.anchor, a: st.a, b: st.b, m: st.mode, c: st.color });
    if (st.kind === "psi") p.set("k", st.k); else p.set("r", st.r);
    if (st.j != null) p.set("j", st.j);
    if (st.color in DIAG || st.color === "correct") p.set("at", st.at);
    if (st.lay) p.set("lay", st.lay);
    if (isProd(st)) p.set("op", st.op);
    if (st.src && st.src !== "M0") p.set("src", st.src);
    if (st.color === "km") { p.set("km", st.kmk); if (st.a !== st.b && st.kms === "end") p.set("kms", "end"); }
    if (this.ctx.kernel !== "cos") p.set("kern", this.ctx.kernel);   // the kernel on screen, not a pending switch
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
    const st = this.st, { m, T, S } = this.ctxCos;     // parameters index the cosine root's lists; fitState adapts them
    // #explorer?tour=<id>: exactly the tour's own parameters (others are ignored, so the note always describes the view)
    const tid = p.get("tour"), tour = tid != null ? this.tours.find(t => t.id === tid) : null;
    if (tour) p = new URLSearchParams(tour.hash.split("?")[1] || "");
    this.setTour(tour);
    if (tid != null && !tour) setTimeout(() => { $("ex-status").textContent = `no tour called "${tid}"`; }, 0);
    const int = (k, lo, hi) => { const v = parseInt(p.get(k), 10); return Number.isFinite(v) && v >= lo && v <= hi ? v : null; };
    if (p.get("fam") === "time" || p.get("fam") === "layer") st.fam = p.get("fam");
    st.op = ["start", "end"].includes(p.get("op")) && m.vals.layerprodR ? p.get("op") : "avg";
    const lim = st.fam === "layer" ? S - 1 : T - 1, alim = st.fam === "layer" ? T - 1 : S - 1;
    st.anchor = int("t", 0, alim) ?? Math.min(st.anchor, alim);
    st.a = int("a", 0, lim) ?? Math.min(st.a, lim); st.b = int("b", 0, lim) ?? Math.min(st.b, lim);
    if (st.a > st.b) [st.a, st.b] = [st.b, st.a];
    st.mode = int("m", 0, m.modes.length - 1) ?? st.mode;
    const c = p.get("c"); if (c && $("ex-color").querySelector(`option[value="${CSS.escape(c)}"]:not([data-missing])`)) st.color = c;
    if (st.color === "seba") st.kind = "set"; if (st.color === "nodal" || st.color === "eig") st.kind = "psi";
    if (p.has("k")) { st.k = int("k", 1, m.coarse ? m.coarse.ext.last : m.n_vecs) ?? st.k; if (!BLOCK_KINDS.has(st.color)) st.kind = "psi"; }
    if (p.has("r")) { const r = int("r", 1, 99); if (m.seba_r.includes(r)) st.r = r; if (!BLOCK_KINDS.has(st.color)) st.kind = "set"; }
    st.cmp = null;
    st.src = SR.SRC[p.get("src")] && this.srcIdx ? p.get("src") : "M0";
    if (st.src !== "M0") { st.color = "seba"; st.kind = "set"; }
    st.j = int("j", 0, 15); st.focus = st.color === "seba" ? st.j : null;
    if (st.src !== "M0") st.j = null;
    if (p.get("at") === "final" || p.get("at") === "own") st.at = p.get("at");
    if (m.coarse) { st.kmk = int("km", 2, 16) ?? st.kmk; st.kms = p.get("kms") === "end" ? "end" : "start"; }
    st.lay = int("lay", 0, m.layouts.length - 1) ?? 0;      // update() moves the points (showLayout)
    st.kern = p.get("kern") === "dot" && this.ctxCos.kernels.dot ? "dot" : "cos";
    if (!this.dev && this.needsDev()) this.setDev(true, false);   // a link into an extra option opens the extras
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
    $("blk-title").textContent = `${blockKind(st)}: ${blockLabel(this.ctx, st)}${this.ctx.kernel !== "cos" ? " · dot kernel" : ""}`;
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
      const t = el("text", { x: X(k), y: H - 24, "text-anchor": "middle" }, svg); t.textContent = `${sym(st)}${k}`;
      if (on) t.setAttribute("style", "fill:#16161a;font-weight:600");
      const tp = el("text", { x: X(k), y: H - 7, "text-anchor": "middle", style: pn[k] < LOC ? `fill:${m.palette.accent};font-weight:650;font-size:10px` : "font-size:10px" }, svg);
      tp.textContent = pn[k] >= 1000 ? (pn[k] / 1000).toFixed(1) + "k" : String(Math.round(pn[k]));
      const hit = el("rect", { x: X(k) - 12, y: mt, width: 24, height: H - mt, fill: "transparent", style: "cursor:pointer" }, svg);
      hit.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(null, [{ name: `${sym(st)}${k}`, value: vals[k].toFixed(4) }, isProd(st) ? { name: `per-snapshot geometric mean ${sym(st)}^(1/n)`, value: Math.pow(vals[k], 1 / (st.b - st.a + 1)).toFixed(4) } : { name: "average penalty 1 − λ", value: (1 - vals[k]).toFixed(4) },
        { name: "participation number", value: `${Math.round(pn[k])} of ${N}` + (pn[k] < LOC ? " (localised)" : "") }])));
      hit.addEventListener("pointerleave", hideTip);
      hit.addEventListener("click", () => { st.kind = "psi"; st.k = k; if (st.color === "seba") st.color = "eig"; st.focus = null; this.update(); });
    }
    const locNote = nLoc ? ` <span style="color:${m.palette.accent};font-weight:600">${nLoc} localised mode${nLoc > 1 ? "s" : ""}</span> (PN below ${LOC}): a mode carried by a handful of images, not a grouping; sets built from it are suspect.` : "";
    $("blk-spec-note").innerHTML = (isProd(st)
      ? `Singular values of the <b>ordered product</b> Q = M<sub>b</sub>⋯M<sub>a</sub> of the block's ${st.b - st.a + 1} lazy walk${st.a === st.b ? "" : "s"} (first snapshot applied first; constant projected out): σ<sub>1</sub>..σ<sub>8</sub> (σ<sub>0</sub> = 1, the constant, not shown). ψ<sub>k</sub> here is the ${st.op === "start" ? "right singular vector v<sub>k</sub>: the function at the start of the block that keeps fraction σ<sub>k</sub> of its norm after being smoothed by every snapshot in turn" : "left singular vector u<sub>k</sub> = Qv<sub>k</sub>/σ<sub>k</sub>: what the best-surviving start function has become after the last snapshot"}. A product is multiplicative: one snapshot that erases a split is enough to sink it.`
      : `Spectrum of the block operator M̄, the mean of the lazy walks M<sub>c</sub> over the block's ${st.b - st.a + 1} snapshot${st.a === st.b ? "" : "s"}: λ<sub>1</sub>..λ<sub>8</sub> (λ<sub>0</sub> = 1, the constant ψ<sub>0</sub>, not shown). 1 − λ<sub>k</sub> is ψ<sub>k</sub>'s average variation across the block's walks.`)
      + (st.kind === "psi" ? ` Click a bar to colour by that ψ.` : ` The r = ${st.r} coherent sets come from SEBA on ψ<sub>0</sub>..ψ<sub>${st.r - 1}</sub> (dark).`)
      + ` PN: participation number (Σψ²)²/Σψ⁴, roughly how many images a mode is spread over.` + locNote;
    this.renderBlockMap();
  }
  renderBlockMap() {
    const st = this.st, { m, pal } = this.ctx;
    const pairs = st.fam === "layer" ? m.layer_pairs : m.time_pairs, n = st.fam === "layer" ? m.sites.length : m.steps.length;
    const vals = m.vals[dkey(st)][st.anchor][st.mode], kk = st.kind === "psi" ? st.k : st.r - 1;
    $("blk-map-anchor").textContent = st.fam === "layer" ? "checkpoint" : "site";
    const box = $("blk-map"); box.replaceChildren();
    if (kk > m.n_vecs) {   // the map reads meta's spectra (ψ1..ψ8); the add-on is per block
      $("blk-map-note").textContent = `Not shipped for ψ${kk}: the map of every block uses ψ1..ψ${m.n_vecs} only. Pick ψ${m.n_vecs} or lower to see it.`;
      $("blk-map-wrap").open = this._mapOpen ?? (window.innerWidth >= 1100);
      $("blk-map-wrap").ontoggle = () => { this._mapOpen = $("blk-map-wrap").open; };
      return;
    }
    const W = box.clientWidth || 300, ml = 34, mt = 4, cw = Math.min(22, (W - ml - 50) / n), H = mt + n * cw + 26;
    const svg = el("svg", { width: W, height: H }, box);
    const v = pairs.map((_, p) => vals[p][kk]), lo = Math.min(...v), hi = 1;
    const names = st.fam === "layer" ? m.sites : m.steps.map(fmtStep);
    pairs.forEach(([a, b], p) => {
      const on = a === st.a && b === st.b;
      const r = el("rect", { x: ml + b * cw + .5, y: mt + a * cw + .5, width: cw - 1, height: cw - 1, rx: 2, fill: pal.seq((v[p] - lo) / (hi - lo + 1e-12)), style: "cursor:pointer" }, svg);
      if (on) { r.setAttribute("stroke", "#16161a"); r.setAttribute("stroke-width", 2); }
      r.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`${names[a]} → ${names[b]}`, [{ name: `${sym(st)}${kk}`, value: v[p].toFixed(4) }])));
      r.addEventListener("pointerleave", hideTip);
      r.addEventListener("click", () => { hideTip(); st.a = a; st.b = b; st.focus = null; st.j = null; this.update(); });
    });
    names.forEach((nm, i) => {
      if ((n > 9 || cw < 30) && (n - 1 - i) % 2) return;
      const t1 = el("text", { x: ml - 4, y: mt + i * cw + cw / 2 + 3.5, "text-anchor": "end" }, svg); t1.textContent = nm;
      const t2 = el("text", { x: ml + i * cw + cw / 2, y: mt + n * cw + 12, "text-anchor": "middle" }, svg); t2.textContent = nm;
    });
    $("blk-map-note").innerHTML = `Every block at this ${st.fam === "layer" ? "checkpoint" : "site"}: row = first ${st.fam === "layer" ? "site" : "checkpoint"}, column = last; colour = ${sym(st)}<sub>${kk}</sub> (darker = closer to 1; range ${lo.toFixed(3)} to 1). Click to select.`;
    $("blk-map-wrap").open = this._mapOpen ?? (window.innerWidth >= 1100);
    $("blk-map-wrap").ontoggle = () => { this._mapOpen = $("blk-map-wrap").open; };
  }

  renderLegend() {
    const st = this.st, { m, pal, labels, N } = this.ctx, col = this.col, body = $("leg-body");
    body.replaceChildren(); $("leg-jac").replaceChildren();
    const titles = { seba: `Coherent sets (SEBA, r = ${st.r})`, nodal: `Sign of ψ${st.k} (nodal domains)`, eig: `Eigenfunction ψ${st.k}`, km: `Spectral k-means partition, k = ${st.kmk}${st.a === st.b ? "" : ` (${st.kms} side)`}`, class: "CIFAR class (validation only)",
      correct: "Correct / wrong", loss: "Loss", jac: "Input-Jacobian norm", gcos: "Gradient alignment (cosine)", gdot: "Gradient alignment (dot product)" };
    const pxk = isPx(st.color) ? st.color.slice(3) : null, pxLab = pxk && (this.ctx.imgstats?.labels[pxk] || IMGSTATS[pxk]?.label || pxk);
    $("leg-title").textContent = pxk ? `Image statistic: ${pxLab}` : titles[st.color];
    if (st._S && st.color === "seba") {
      $("leg-title").textContent = `Sets from: ${SR.SRC[st._S.id].label}`;
      this.renderSourceLegend(body);
    } else if (col.groups && st.color === "seba") {
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

  renderSourceLegend(body) {
    const S = this.st._S, src = SR.SRC[S.id], a = S.agree, f2 = nz(2), f3 = nz(3);
    const tags = src.status.split("+").map(t => `${t}: ${SR.STATUS[t.replace(/[^A-Z]/g, "")]}`).join(" · ");
    body.append(h("div", { class: "leg-note", text: `Status ${src.status}. ${tags}.` }));
    if (a && a.cls && Number.isFinite(a.cls.ami))
      $("leg-jac").append(h("div", { class: "jrow" }, h("span", { text: "This source against the classes: AMI (jointly assigned)" }), h("span", {}, h("span", { class: "jv", text: f2(a.cls.ami) }), h("span", { class: "jnull", text: `  permutation 95th pct ${f3(a.cls.q95)}` }))),
        h("div", { class: "jrow" }, h("span", { text: "matched Jaccard with the classes" }), h("span", {}, h("span", { class: "jv", text: f2(a.cls.mj) }), h("span", { class: "jnull", text: `  permutation mean ${f2(a.cls.mjNull)}` }))));
    const more = h("div", { class: "leg-note" }); more.innerHTML = `Definitions, the table of what each null keeps and destroys, and charts over all blocks: <a href="#baselines">Baselines and nulls</a>.`;
    body.append(more);
  }
  async renderJaccard() {
    const st = this.st, ctx = this.ctx, { m, labels, N } = ctx, box = $("leg-jac");
    const ri = m.seba_r.indexOf(st.r), A = this.blk.lab.subarray(ri * N, (ri + 1) * N);
    const key = `${ctx.root}/${dkey(st)}/${st.anchor}/${st.mode}/${st.a},${st.b}/${st.r}`;
    const vsC = this._jc && this._jc.key === key ? this._jc.res : matchedJaccard(A, st.r, labels, 10);
    this._jc = { key, res: vsC };
    box.replaceChildren(
      h("div", { class: "jrow" }, h("span", { text: "Matched Jaccard vs CIFAR classes" }), h("span", {}, h("span", { class: "jv", text: vsC.mean.toFixed(2) }), h("span", { class: "jnull", text: `  null ${vsC.nullMean.toFixed(2)}` }))),
      h("div", { class: "leg-note", text: vsC.per.map((x, i) => `set ${i + 1}→${x.j >= 0 ? m.classes[x.j] : "none"} ${x.J.toFixed(2)}`).join(" · ") }),
    );
    // comparison block (same family, same r, same kappa rule)
    const lim = st.fam === "layer" ? m.steps.length : m.sites.length;
    if (!st.cmp || st.cmp.fam !== st.fam || st.cmp.from !== st.op) {
      // product views compare against the average operator on the same block by default
      const anc = isProd(st) ? st.anchor : st.fam === "layer" ? (st.anchor === lim - 1 ? lim - 2 : lim - 1) : (st.anchor === lim - 1 ? lim - 2 : st.anchor + 1);
      st.cmp = { fam: st.fam, anchor: anc, a: st.a, b: st.b, op: isProd(st) ? "avg" : st.op, from: st.op };
    }
    const c = st.cmp;
    const sel = (opts, val, on) => { const s = h("select"); opts.forEach((o, i) => s.add(new Option(o, i))); s.value = val; s.addEventListener("change", () => { on(+s.value); this.renderJaccard(); }); return s; };
    const ancNames = st.fam === "layer" ? m.steps.map(s => "step " + s.toLocaleString("en")) : m.sites;
    const posNames = st.fam === "layer" ? m.sites : m.steps.map(s => s.toLocaleString("en"));
    const out = h("span", { class: "jv", text: "…" }), nul = h("span", { class: "jnull" });
    const OPS = ["avg", "start", "end"], opSel = m.vals.layerprodR ? [sel(["average", "product · start", "product · end"], OPS.indexOf(c.op), v => c.op = OPS[v])] : [];
    box.append(h("div", { class: "jcmp" }, h("span", { text: "vs block" }), sel(ancNames, c.anchor, v => c.anchor = v), sel(posNames, c.a, v => { c.a = v; if (c.b < v) c.b = v; }), h("span", { text: "to" }), sel(posNames, c.b, v => { c.b = v; if (c.a > v) c.a = v; }), ...opSel),
      h("div", { class: "jrow" }, h("span", { text: `same kernel rule, r = ${st.r}` }), h("span", {}, out, nul)));
    const per = h("div", { class: "leg-note" }); box.append(per);
    box.append(h("div", { class: "leg-note", text: "Hard SEBA sets matched one-to-one by the Hungarian assignment maximising total Jaccard; mean over this block's sets. Null: same set sizes, random members (30 draws)." }));
    try {
      const other = await D.block(dkey(st, c.op), c.anchor, st.mode, ctx.pairIdx[st.fam][c.a + "," + c.b], ctx.root);
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
      body.append(h("div", { class: "card-note", text: tg.k > m.n_vecs
        ? `Images with the largest and smallest entries of ψ${tg.k}, ranked in the browser from the shipped 8-bit vector (ψ9..ψ32 have no build-time representatives; ties are broken by image index). Hover an image to ring it in every panel.`
        : "Images with the largest and smallest entries of the block's eigenfunction (computed at full precision). Hover an image to ring it in every panel." }));
      return;
    }
    if (st._S) return this.renderSourceImages(body, strip, compBar);
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

  // first card for a source other than the method: agreement with the method, then sizes, class make-up and images
  renderSourceImages(body, strip, compBar) {
    const st = this.st, S = st._S, { m, pal, labels, N } = this.ctx, src = SR.SRC[S.id], k = S.k;
    $("img-title").textContent = `Sets from: ${src.label} (${src.status})${S.classes ? "" : `, r = ${st.r}`}`;
    body.append(h("div", { class: "card-note src-def-top", text: src.def }));
    const cnt = [...Array(k + 1)].map(() => new Array(10).fill(0)), size = new Array(k + 1).fill(0);
    for (let i = 0; i < N; i++) { const g = S.lab[i] < 0 ? k : S.lab[i]; size[g]++; cnt[g][labels[i]]++; }
    for (let j = 0; j <= k; j++) {
      if (j === k && !size[k]) continue;
      const un = j === k, on = st.focus === j, dim = st.focus != null && !on;
      const col = un ? pal.unassigned : S.classes ? pal.classes[j] : pal.set(j);
      const head = h("div", { class: "set-head", role: "button", tabindex: 0, title: "Click to isolate this set in every panel" },
        h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: un ? "unassigned" : S.names[j] }),
        h("span", { class: "n", text: size[j].toLocaleString("en") }), compBar(cnt[j], size[j] || 1));
      const pick = () => { st.focus = st.focus === j ? null : j; this.col = colouring(this.ctx, st, this.blk); this.renderAll(); };
      head.addEventListener("click", pick);
      head.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      head.addEventListener("pointerenter", () => { if (st.focus != null) return; st.focus = j; this.col = colouring(this.ctx, st, this.blk); this.draw(); this.drawLens(); st.focus = null; });
      head.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, this.blk); this.draw(); this.drawLens(); });
      const row = h("div", { class: "set-row" + (on ? " on" : "") + (dim ? " dim" : "") }, head);
      if (!un && size[j]) row.append(strip(S.reps ? S.reps(j) : SR.shuffledMembers(S.lab, j, 2024 + 31 * j), on ? 2 : 1));
      body.append(row);
    }
    const key = h("div", { class: "class-key" });
    m.classes.forEach((c, q) => key.append(h("span", {}, h("i", { style: { background: pal.classes[q] } }), c)));
    body.append(key, h("div", { class: "card-note", text: (S.classes ? "Each set is one class; bars show its class make-up." : "Colours: each set takes the colour of the method's set it overlaps most (Hungarian matching on Jaccard, this block and r); display only. Bars: CIFAR class make-up (validation only).") +
      (S.reps ? " Thumbnails: each set's members with the highest SEBA value." : " Thumbnails: members in a fixed pseudo-random order (this source stores hard labels only, no membership values).") + " Click a set to isolate it in every panel." }));
    body.append(this.agreementBox(S));
  }
  agreementBox(S) {
    const st = this.st, src = SR.SRC[S.id], a = S.agree, f3 = nz(3), f2 = nz(2);
    const box = h("div", { class: "src-agree" });
    if (!a) { box.append(h("div", { class: "card-note", text: "No agreement row stored for this source and block." })); return box; }
    const nB = this.srcIdx.files[`${st.fam}/${st.anchor}`]?.P;
    const what = S.id === "Y1" ? "overlap of the default construction's sets with the classes" : "overlap with the default construction's sets (same block, r = " + st.r + ")";
    box.append(h("div", { class: "src-agree-head", text: what.charAt(0).toUpperCase() + what.slice(1) }));
    const line = (label, v, extra) => h("div", { class: "src-agree-row" }, h("span", { class: "k", text: label }), h("span", { class: "v", text: v }), h("span", { class: "x", text: extra }));
    box.append(line("AMI, jointly assigned images", f2(a.ami), `coverage ${f2(a.cov)} · permutation 95th pct ${f3(a.q95)}` + (Number.isFinite(a.max95) ? ` · reference from shuffled labels, scanning the ${nB} blocks at this ${st.fam === "layer" ? "checkpoint" : "site"} (95th pct of the largest): ${f3(a.max95)}` : "")));
    box.append(line("matched Jaccard", f2(a.mj), `permutation mean ${f2(a.mjNull)}` + (Number.isFinite(a.mjMax95) ? ` · shuffled labels scanned over the ${nB} blocks (95th pct of the largest): ${f2(a.mjMax95)}` : "")));
    box.append(h("div", { class: "card-note", text: "AMI (adjusted mutual information) is about 0 when two partitions share no more than chance and 1 when they are the same; coverage is the share of images both assign. Matched Jaccard runs from 0 to 1 (1 = the same sets). These say how much two partitions coincide; neither a high nor a low value is better." }));
    box.append(h("div", { class: "card-note", text: `Permutation: the source's labels shuffled ${a.nperm.toLocaleString("en")} times, set sizes kept${a.rep ? "; computed by the site build for this stored null draw (no scanned-blocks reference: one block)" : S.id === "Y2" ? "; computed by the site build" : ""}. The 95th percentile is for this block alone; when you compare many blocks, use the scanned-blocks reference. Kernel rule ${this.ctx.m.modes[this.srcIdx.kernel_rule]}, average operator.` }));
    return box;
  }
  renderRetention() {
    const st = this.st, { m, pal } = this.ctx, blk = this.blk, tg = this.target(), X = blk.X, R = m.seba_r.length;
    const xs = [...Array(X).keys()], names = st.fam === "layer" ? m.sites : m.steps.map(fmtStep);
    if (tg.kind === "psi" && tg.k > m.n_vecs) {
      $("ret-body").replaceChildren();
      $("ret-title").innerHTML = `Retention ρ<sub>c</sub>(ψ<sub>${tg.k}</sub>) in every snapshot`;
      $("ret-note").innerHTML = `Not shipped for ψ&gt;${m.n_vecs}: per-snapshot retention${isProd(st) ? " and step gains are" : " is"} computed at build time for ψ1..ψ${m.n_vecs} only (the ψ9..ψ32 add-on carries the vectors and their ${sym(st)} values). ${sym(st)}<sub>${tg.k}</sub> = ${blk.ext ? blk.ext.vals[tg.k].toFixed(4) : "n/a"}.`;
      return;
    }
    let series = [];
    const prod = isProd(st);
    // product blocks: solid = step gain g_c (in-block only), dashed = retention ρ_c (every snapshot)
    const two = (name, ret, gain, color) => prod
      ? [{ name: `${name} gain`, y: Array.from(gain), color, dots: 2.5 }, { name: `${name} ρ`, y: Array.from(ret), color, dash: true, width: 1.2 }]
      : [{ name, y: Array.from(ret), color, dots: 2.5 }];
    if (tg.kind === "psi") series = two(`ψ${tg.k}`, blk.ret_psi.subarray((tg.k - 1) * X, tg.k * X), prod ? blk.gain_psi.subarray((tg.k - 1) * X, tg.k * X) : null, "#16161a");
    else {
      const js = tg.j == null ? [...Array(st.r).keys()] : [tg.j];
      const so = m.seba_r.slice(0, tg.ri).reduce((x, y) => x + y, 0);
      series = js.flatMap(j => two(`set ${j + 1}`, blk.ret_seba.subarray((so + j) * X, (so + j + 1) * X), prod ? blk.gain_seba.subarray((so + j) * X, (so + j + 1) * X) : null, pal.set(j)));
    }
    const all = series.flatMap(s => s.y).filter(Number.isFinite);
    const lo = Math.min(.8, Math.floor((Math.min(...all) - .01) * 10) / 10);
    lineChart($("ret-body"), {
      x: xs, series, yrange: [lo, 1.0], height: 150, xticks: xs.filter(i => X <= 8 || (X - 1 - i) % 2 === 0), xfmt: i => names[i], shade: [st.a - .45, st.b + .45],
      yfmt: v => v.toFixed(2), ml: 36, xname: st.fam === "layer" ? "site" : "step",
      refline: tg.kind !== "psi" ? null : prod ? { y: Math.pow(blk.vals[tg.k], 1 / (st.b - st.a + 1)), label: `σ${tg.k}^(1/${st.b - st.a + 1}) (geometric-mean gain)` } : { y: blk.vals[tg.k], label: `λ${tg.k} (block mean)` },
    });
    if (st._S) $("ret-note").dataset.src = "1"; else delete $("ret-note").dataset.src;
    $("ret-title").innerHTML = `${prod ? "Step gain g<sub>c</sub> and retention ρ<sub>c</sub>" : "Retention ρ<sub>c</sub>"}(${tg.kind === "psi" ? `ψ<sub>${tg.k}</sub>` : tg.j == null ? "SEBA vectors" : `SEBA vector ${tg.j + 1}`}) in every snapshot`;
    $("ret-note").innerHTML = (prod ? `Solid: step gain g<sub>c</sub> = |C<sub>c</sub>w<sub>c−1</sub>| / |w<sub>c−1</sub>|, where w is f (constant removed) pushed through the block's snapshots in order (${st.op === "start" ? "first to last" : "last to first, the adjoint direction"}); defined inside the block only, and the gains multiply to σ exactly for ψ. A dip marks the snapshot that erases the function. Dashed: ` : ``) + `ρ<sub>c</sub>(f) = f<sup>T</sup>M<sub>c</sub>f / f<sup>T</sup>f for every ${st.fam === "layer" ? "site at this checkpoint" : "checkpoint at this site"}, with f = ${tg.kind === "psi" ? `ψ<sub>${tg.k}</sub>` : "each SEBA vector (colours as the sets)"}; the block's snapshots are shaded. ${prod ? "" : tg.kind === "psi" ? `λ<sub>${tg.k}</sub> (dashed) is the mean of ρ<sub>c</sub> over the shaded snapshots: an <b>average</b>, not a value in every snapshot.` : "The spectrum is an <b>average</b> of retention over the shaded snapshots, not a value in every snapshot."}` + (st._S ? ` <b>Shown for the method's SEBA vectors</b>, not for "${SR.SRC[st._S.id].label}" (that source stores hard labels only).` : "");
  }

  // ---------------------------------------------------------------- coarse-graining (SPEC_COARSE.md section 5)
  // Spectrum strip (top 64 of the selected operator, log scale, gap marker) in the block card, and the coarse-graining
  // card: chi^2 information kept by k-block partitions vs the ceiling F_k. Cosine root, layer blocks; from d/coarse/.
  async renderCoarse() {
    const st = this.st, ctx = this.ctx, m = ctx.m, on = coarseOn(ctx, st);
    $("card-coarse").hidden = !on; $("blk-strip-wrap").hidden = !on;
    if (!on) return;
    const tok = this._cgTok = (this._cgTok || 0) + 1;
    let U;
    try { U = await D.coarseUnit(m.steps[st.anchor], st.mode); }
    catch (e) { console.error(e); $("cg-note").textContent = "The coarse-graining summary failed to load."; return; }
    if (tok !== this._cgTok || this.ctx !== ctx) return;
    const rec = U.blocks[pairOf(ctx, st)];
    this.renderStrip(rec);
    this.renderCoarseCard(rec);
  }
  renderStrip(rec) {
    const st = this.st, m = this.ctx.m, cell = st.a === st.b, prod = isProd(st) || cell;
    const vals = prod ? rec.sv : rec.avg, gap = prod ? rec.gap : rec.avg_gap, S = cell ? "λ" : sym(st), K = maxK(this.ctx, st);
    const sel = st.kind === "psi" ? st.k : null, n = Math.min(64, vals.length - 1);
    const pts = lo => [...Array(n)].map((_, i) => i + 1).filter(lo).map(k => [k, vals[k]]);
    const series = [
      { name: `${S}ₖ with a shipped ψₖ (click to colour)`, pts: pts(k => k <= K && k !== sel), color: "#55545a", dots: 2.6, noline: true },
      { name: "value only", pts: pts(k => k > K), color: "#c3c2bb", dots: 2.2, noline: true },
    ];
    if (sel) series.push({ name: `ψ${sel}`, pts: [[sel, vals[sel]]], color: m.palette.accent, dots: 4.2, noline: true });
    const gx = gap.k - .5, clear = gap.clear;
    xyChart($("blk-strip"), {
      series, ylog: true, height: 128, ml: 40, xrange: [.5, n + .5], xticks: [1, 8, 16, 24, 32, 48, 64].filter(v => v <= n), xname: "index k",
      vlines: [{ x: gx, dash: !clear, color: clear ? "#16161a" : "#8a8983", label: `gap after ψ${gap.k - 1}` }], legend: false,
      yfmt: v => (+v.toPrecision(2)).toString(),
      tipExtra: k => [{ name: `${S}${k} / ${S}${k + 1}`, value: vals[k + 1] > 0 ? (vals[k] / vals[k + 1]).toFixed(3) : "n/a" }, { name: "vector", value: k <= K ? "shipped" : "not shipped" }],
      onclick: k => { if (k > K) return; st.kind = "psi"; st.k = k; if (st.color === "seba") st.color = "eig"; st.focus = null; this.update(); },
    });
    $("blk-strip-title").textContent = `Top ${n} of the ${prod ? (cell ? "cell's walk M" : "ordered product Q") : "block average M̄"}: ${S}1..${S}${n} (log scale)`;
    const rule = cell || !prod ? "on the Laplacian scale 1 − λ" : "as σₖ / σₖ₊₁";
    $("blk-strip-note").innerHTML = `Largest ratio ${rule} for a split after 1 to 31 nontrivial functions: after ψ<sub>${gap.k - 1}</sub> (a ${gap.k}-block partition), ratio ${gap.ratio.toFixed(2)}; runner-up after ψ<sub>${gap.k2 - 1}</sub>, ${gap.ratio2.toFixed(2)}. `
      + (clear ? "Marked solid: the largest ratio is at least 1.15 and 10% above the runner-up." : "<b>No clear gap</b> (dashed): the largest ratio is below 1.15 or within 10% of the runner-up.")
      + ` Descriptive only; nothing downstream uses it.${K > m.n_vecs ? ` Dark points: ψ1..ψ${K} can colour the grid (ψ9..ψ32 from an 8-bit add-on).` : ` ψ9..ψ32 ${st.a === st.b ? "" : "for multi-site blocks "}are shipped for the default kernel rule only; here only ψ1..ψ8 can colour the grid.`}`
      + (!prod && !cell ? " Gap of the average computed by the site build with the same rule as for single cells." : "");
  }
  renderCoarseCard(rec) {
    const st = this.st, ctx = this.ctx, m = ctx.m, pal = ctx.pal, cell = st.a === st.b, C = m.coarse, K = C.K;
    const sides = cell ? [["start", "one-sided I(c(X);X′)", "Information one step of the walk keeps about the block of the start image: I(c(X); X′)"], ["two", "both ends I(c(X);c(X′))", "The same partition on both ends of one step (stationary chain)"]]
      : [["start", "start side I(c(X);Y)", "Partition the images at the first site; information about the image at the last site"], ["end", "end side I(X;c′(Y))", "Partition the images at the last site"], ["two", "two-sided I(c(X);c′(Y))", "Partition both ends (coherent pairs); ceiling at the smaller number of blocks"]];
    const seg = $("cg-side");
    if (seg._cell !== cell) {
      seg._cell = cell;
      seg.replaceChildren(...sides.map(([v, label, title]) => h("button", { type: "button", "data-v": v, title, text: label })));
    }
    if (!sides.some(x => x[0] === st.cgSide)) st.cgSide = "start";
    this.segs.forEach(s => s._sync && (s.id === "cg-side" || s.id === "cg-view") && s._sync());
    const sv = rec.sv, F = k => { let s = 0; for (let i = 1; i < k && i < sv.length; i++) s += sv[i] * sv[i]; return s; };
    const side = st.cgSide, eff = st.cgView === "eff", key = "I_" + side;
    const xOf = (mt, qi) => { const a = rec.m[mt].keff_start[qi], b = rec.m[mt].keff_end[qi]; return side === "start" ? a : side === "end" ? b : a == null || b == null ? null : Math.min(a, b); };
    const ptsOf = mt => K.map((_, qi) => { const x = xOf(mt, qi), I = rec.m[mt][key][qi]; return x == null || I == null ? null : [x, eff ? I / F(x) : I]; }).filter(Boolean).sort((p, q) => p[0] - q[0]);
    const S = pal.sets, series = [
      { name: "ceiling F_k", pts: K.map(k => [k, eff ? 1 : F(k)]), color: "#16161a", dash: true, width: 1.4 },
      { name: "spectral k-means", pts: ptsOf("spectral"), color: S[0], dots: 2.2 },
      { name: "SEBA", pts: ptsOf("seba"), color: S[1], dots: 2.2 },
      { name: "PCA k-means", pts: ptsOf("pca"), color: S[2], dots: 2.2 },
      { name: "random (null)", pts: ptsOf("random"), color: "#9a9993", dots: 2, width: 1.4 },
      { name: "predictions", pts: ptsOf("pred"), color: m.palette.accent, marker: "triangle" },
      { name: "labels", pts: ptsOf("labels"), color: "#16161a", marker: "diamond", hollow: true },
    ];
    const km = kmOn(ctx, st), hl = st.color === "km" ? st.kmk : null;
    xyChart($("cg-chart"), {
      series, ylog: !eff, yrange: eff ? [0, 1.02] : null, height: 190, ml: 44, xrange: [1.5, 32.5], xticks: [2, 4, 8, 12, 16, 20, 24, 32], xname: "number of blocks k (occupied)", hl,
      vlines: [{ x: rec.gap.k, dash: !rec.gap.clear, color: rec.gap.clear ? "#16161a" : "#8a8983", label: rec.gap.clear ? "gap" : "no clear gap" }],
      yfmt: eff ? (v => v.toFixed(1)) : undefined,
      tipExtra: k => [{ name: "ceiling F_k", value: (+F(k).toPrecision(4)).toString() }, ...(km && C.ks.includes(k) ? [{ name: "click", value: `colour by spectral k-means, k = ${k}` }] : [])],
      onclick: km ? (k => { if (!C.ks.includes(k)) return; st.color = "km"; st.kmk = k; if (!cell) st.kms = side === "end" ? "end" : "start"; st.focus = null; this.update(); }) : null,
    });
    const what = cell ? `the cell's walk M at ${m.sites[st.a]}` : `the ordered product Q = M<sub>${m.sites[st.b]}</sub>⋯M<sub>${m.sites[st.a]}</sub> (the coupling of the image at ${m.sites[st.a]} with the image at ${m.sites[st.b]})`;
    const g = rec.gap;
    $("cg-sub").innerHTML = `${m.modes[st.mode]} · total I(X;Y) = ${(+rec.total.toPrecision(4))}${cell ? " (at least N/4 − 1 from the lazy half)" : ""} · ${g.clear ? `gap at k = ${g.k}` : "no clear gap"}`;
    $("cg-note").innerHTML = `χ² information kept by a k-block partition of ${what}${!cell && !isProd(st) ? " (this card is about the product, whichever operator colours the grid)" : ""}. `
      + `<b>F<sub>k</sub></b> = σ<sub>1</sub>² + … + σ<sub>k−1</sub>² (dashed) is the most any k-block partition can keep (Eckart–Young)${side === "two" ? "; two-sided, at the smaller of the two block counts" : ""}. `
      + `Each method is placed at its number of occupied blocks. <b>Random</b> balanced partitions (mean of 4 draws) are the null. <b>Labels</b> (CIFAR class) and <b>predictions</b> are validation only, at k = 10 (predictions may occupy fewer classes). `
      + (eff ? "Efficiency I / F<sub>k</sub> is a tightness check only: it is 1 for every partition of a walk with no geometry at all, so read it next to the absolute values. " : "")
      + (km ? "Click a k (2 to 16) to colour the grid by that spectral k-means partition." : "Spectral k-means colouring is shipped for this block only under the default kernel rule.");
    // single cells: leakage against the Ky Fan (Ncut) and higher-order Cheeger (max leakage) bounds
    $("cg-cell").hidden = !cell;
    if (!cell) return;
    const KF = k => { let s = 0; for (let i = 0; i < k && i < sv.length; i++) s += 1 - sv[i]; return s; }, CH = k => (1 - sv[k - 1]) / 2;
    const ratio = (mt, q) => K.map((_, qi) => { const x = rec.m[mt].keff_start[qi], v = rec.m[mt][q][qi]; return x == null || v == null ? null : [x, v / (q === "ncut" ? KF(x) : CH(x))]; }).filter(Boolean).sort((p, r) => p[0] - r[0]);
    const cs = [["spectral", "spectral k-means", S[0]], ["seba", "SEBA", S[1]], ["pca", "PCA k-means", S[2]], ["random", "random", "#9a9993"]];
    xyChart($("cg-cell-chart"), {
      series: [...cs.map(([mt, nm, c]) => ({ name: `${nm}: Ncut / Ky Fan`, pts: ratio(mt, "ncut"), color: c, width: 1.6 })),
        ...cs.map(([mt, nm, c]) => ({ name: `${nm}: max leak / Cheeger`, pts: ratio(mt, "maxleak"), color: c, dash: true, width: 1.2 })),
        { name: "labels: Ncut / Ky Fan", pts: ratio("labels", "ncut"), color: "#16161a", marker: "diamond", hollow: true }],
      ylog: true, height: 140, ml: 44, xrange: [1.5, 32.5], xticks: [2, 4, 8, 12, 16, 20, 24, 32], xname: "number of blocks k", legend: false, hl,
      yfmt: v => (+v.toPrecision(2)).toString(),
    });
    $("cg-cell-note").innerHTML = `Solid: Ncut(c) = Σ<sub>j</sub> φ(A<sub>j</sub>) over the Ky Fan bound Σ<sub>i&lt;k</sub>(1 − λ<sub>i</sub>); dashed: the largest leakage max<sub>j</sub> φ(A<sub>j</sub>) over the higher-order Cheeger lower bound (1 − λ<sub>k−1</sub>)/2. Both ratios are at least 1; 1 means the partition meets the bound. Colours as above; hollow diamond: labels (Ncut). φ(A) = 1 − (1/|A|) Σ<sub>x,y∈A</sub> M[x,y] is the chance one step leaves block A.`;
  }

  renderStats() {
    const st = this.st, { m, T, S, N } = this.ctx, blk = this.blk, tg = this.target(), body = $("stats-body");
    const ri = tg.kind === "set" ? tg.ri : 0, j = tg.kind === "set" ? (tg.j ?? 0) : 0;
    $("stats-title").textContent = st.fam === "layer" ? `Sites at step ${m.steps[st.anchor].toLocaleString("en")}` : `Checkpoints at ${m.sites[st.anchor]}`;
    const tb = h("table", { class: "mini" });
    tb.append(h("tr", {}, h("th", { text: st.fam === "layer" ? "site" : "step" }), h("th", { text: "κ" }), h("th", { text: "15-NN mass" }), h("th", { text: "ρ_c" }), ...(isProd(st) ? [h("th", { text: "gain g_c" })] : [])));
    const X = blk.X;
    for (let x = 0; x < X; x++) {
      const [t, s] = st.fam === "layer" ? [st.anchor, x] : [x, st.anchor];
      const fi = tg.kind === "psi" ? (tg.k - 1) * X + x : (m.seba_r.slice(0, ri).reduce((p, q) => p + q, 0) + j) * X + x;
      const hi = tg.kind === "psi" && tg.k > m.n_vecs;     // psi_9..psi_32: no per-snapshot retention shipped
      const ret = hi ? NaN : tg.kind === "psi" ? blk.ret_psi[fi] : blk.ret_seba[fi];
      const gain = !isProd(st) ? null : hi ? NaN : tg.kind === "psi" ? blk.gain_psi[fi] : blk.gain_seba[fi];
      const tr = h("tr", { class: inBlock(st, t, s) ? "inb" : "" },
        h("td", { text: st.fam === "layer" ? m.sites[s] : m.steps[t].toLocaleString("en") }), h("td", { text: m.kappa?.[st.mode]?.[t]?.[s] != null ? m.kappa[st.mode][t][s].toPrecision(3) : "–" }),
        h("td", { text: pct(m.nn_mass?.[st.mode]?.[t]?.[s] ?? NaN) }), h("td", { text: fmt(ret, 3) }), ...(gain == null ? [] : [h("td", { text: Number.isFinite(gain) ? fmt(gain, 3) : "–" })]));
      tb.append(tr);
    }
    body.replaceChildren(tb, h("div", { class: "card-note", text: `κ (per-image median for perplexity rules; none for the kNN graph) and median 15-NN kernel mass under ${m.modes[st.mode]}; retention ρ_c of ${tg.kind === "psi" ? "ψ" + tg.k : "SEBA vector " + (j + 1)}${tg.kind === "psi" && tg.k > m.n_vecs ? ` (not shipped for ψ>${m.n_vecs})` : ""}. Bold rows are in the block.` }));
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
  const opM = isProd(st) ? `Q = M_b⋯M_a (ordered product, ${st.op === "start" ? "right" : "left"} singular vectors)` : "M̄";
  const focus = col && col.groups && st.focus != null && st.focus < col.groups.n ? ` Only ${col.groups.names[st.focus]} is coloured; the rest are light grey.` : "";
  const at = st.at === "final" ? `at the final checkpoint (${step(T - 1)}) in every row` : "at each row's own checkpoint";
  const side = st.kind === "psi" ? `side panel: ψ${st.k}` : `side panel: r = ${st.r} coherent sets`;
  let colour = {
    seba: `coherent sets, r = ${st.r}. Each image's hard label is the largest of the ${st.r} SEBA vectors from ψ0..ψ${st.r - 1} of the block operator ${opM}; grey = unassigned (no positive entry); ${fromBlk}points drawn in order of membership value, largest on top.`,
    eig: `ψ${st.k} / max|ψ${st.k}|${st.k > m.n_vecs ? " (ψ9..ψ32: from the 8-bit add-on)" : ""}, where ψ${st.k} is ${isProd(st) ? `the ${st.op === "start" ? "right" : "left"} singular vector of the ordered product Q for σ${st.k}, singular values in decreasing order from σ0 = 1 (the constant)` : `the eigenfunction of the block operator M̄ for λ${st.k}, eigenvalues in decreasing order from λ0 = 1 (the constant)`}; diverging scale, sign fixed so the largest-magnitude entry is positive; ${fromBlk}largest |ψ| on top.`,
    nodal: `sign of ψ${st.k} of the block operator ${opM} (ψ${st.k} > 0, < 0, = 0); ${fromBlk}largest |ψ| on top.`,
    km: `spectral k-means partition into k = ${st.kmk} blocks: Lloyd's k-means (8 restarts, fixed seed) on the top ${st.kmk - 1} nontrivial spectral coordinates σ_iψ_i of ${st.a === st.b ? "the cell's walk M (one site)" : `the ordered product Q = M_b⋯M_a, ${st.kms === "end" ? "end side (images at the last site, left singular vectors)" : "start side (images at the first site, right singular vectors)"}`}; blocks numbered by decreasing size, colours are display only; ${fromBlk}scored in the side panel's coarse-graining card. The operator switch does not change it (always from the ${st.a === st.b ? "cell's walk" : "ordered product"}).`,
    class: `CIFAR-10 class label (validation only). The block sets only the side panel (${side}).`,
    loss: `per-image cross-entropy loss ${at}, log scale, range fixed across rows (1st to 99th percentile over all checkpoints). The block sets only the side panel (${side}).`,
    correct: `whether the network's prediction is correct, ${at}. The block sets only the side panel (${side}).`,
    jac: `Frobenius norm of the input Jacobian of the logits, ${at}, log scale fixed across rows. The block sets only the side panel (${side}).`,
    gcos: `cosine between the image's loss gradient (all parameters) and the mean gradient over the probe images, ${at}. The block sets only the side panel (${side}).`,
    gdot: `dot product of the image's loss gradient with the mean gradient, ${at}. The block sets only the side panel (${side}).`,
  }[st.color] || (isPx(st.color) ? `image statistic ${ctx.imgstats?.labels[st.color.slice(3)] || IMGSTATS[st.color.slice(3)]?.label || st.color.slice(3)}: ${pxDef(st.color.slice(3))} The same per-image value in every panel (a property of the probe image, not of the network). The block sets only the side panel (${side}).` : st.color);
  if (st._S && st.color === "seba") {
    const S = st._S, src = SR.SRC[S.id];
    colour = `hard sets from "${src.label}" (status ${src.status}; ${S.classes ? "10 sets" : `r = ${st.r} sets`}): ${src.def} ` +
      (S.classes ? "Class colours." : "Each set takes the colour of the method's set of this block it overlaps most (Hungarian matching on Jaccard; display only).") +
      ` Grey = unassigned. Painted on every panel. Kernel rule ${m.modes[0]}, average operator.`;
  }
  const dot = ctx.kernel && ctx.kernel !== "cos";
  // under dot, fitState leaves st.lay at seed 0 or the spectral layout (st.lay indexes the cosine root's list)
  const L = (dot ? (st.lay ? m.layouts.find(x => x.kind === "spectral") : m.layouts[0]) : m.layouts[st.lay]) || {};
  const lay = !L.kind && dot ? "UMAP of each snapshot from the top 50 principal components of its centred dot-product Gram (AlignedUMAP across sites, Procrustes across checkpoints, then rotated onto the cosine layout)"
    : !st.lay ? "UMAP of each snapshot (AlignedUMAP across sites, Procrustes across checkpoints)"
    : L.kind === "spectral" ? `UMAP of each snapshot's diffusion coordinates λ_kψ_k, k = 1..${L.n_coords}, from that snapshot's own walk under the ${m.modes[L.mode]} rule, whatever rule colours the points (AlignedUMAP across sites, Procrustes across checkpoints)`
    : `UMAP of seed ${L.seed ?? st.lay} (a second network; colours from seed 0)`;
  const part = (label, text) => [h("b", { text: label + " " }), text + " "];
  return [
    ...(dot ? part("Kernel:", `${ctx.kernelLabel || "dot"}: half squared Euclidean distance |x_i − x_j|²/2 on the unnormalised activations in place of the cosine distance 1 − cos (data ${m.tag}).`) : []),
    ...part("Block:", `${blockKind(st).toLowerCase()}, ${blk}.`),
    ...part("Kernel rule:", `${m.modes[st.mode]}.`),
    ...part("Operator:", `${opName(st)}.`),
    ...part("Colour:", colour + focus),
    ...part("Positions:", `${lay}; display only, no number on this page is computed from them. Rows: checkpoints (step, validation accuracy); columns: sites; ${T * S} panels, one per snapshot, of the same ${m.n.toLocaleString("en")} probe images.`),
  ];
}
