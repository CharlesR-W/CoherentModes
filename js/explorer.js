// The explorer: grid of UMAP cells (rows = checkpoints, columns = sites) coloured by the selected block.
import * as D from "./data.js?v=75f651321a";
import { Palette, divLUT, seqLUT, qDiv, qSeq, hexRgb } from "./colors.js?v=75f651321a";
import { h, el, showTip, hideTip, tipRows, lineChart, xyChart, scale, niceTicks, fmtStep } from "./charts.js?v=75f651321a";
import * as SR from "./sources.js?v=75f651321a";

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
// image statistics offered in the colour menu (a few that are distinct and readable)
const PX_MENU = ["luminance", "contrast", "saturation", "bc_lum", "hf_frac"];
// "Sets from" menu: the sources worth comparing in the explorer (the Baselines tab keeps the full list)
const SRC_MENU = new Set(["M0", "L1", "L2", "G1", "G2", "C1first", "C1last", "I1", "I2", "I3", "Y1", "Y2", "N3"]);
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
// a share that may be tiny (through-depth weights): 2 significant figures down to 0.1%, then powers of ten
const share = w => !Number.isFinite(w) ? "n/a" : w >= .1 ? (100 * w).toFixed(0) + "%" : w >= 1e-3 ? (100 * w).toPrecision(2) + "%" : (100 * w).toExponential(0) + "%";

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

// ------------------------------------------------------------------------------------------- diffusion wavelets
// build_site.py wavelets(), wavelets_modal.py (Coifman-Maggioni on a single snapshot's lazy walk M; cosine root, one
// kernel rule). State: st.dwj (level j), st.dwk (1 wavelets | 0 scaling functions), st.dwf (index within that level and
// kind, pivot order), st.dwi (clicked image or null: then st.dwf is the function carrying it most). st._dw = {cell,
// step, site} once loaded.
const dwGroup = (cell, j, k) => { const g = []; cell.fun.forEach((r, f) => { if (r[0] === j && r[1] === k) g.push(f); }); return g; };
const dwLevels = cell => [...new Set(cell.fun.map(r => r[0]))].sort((a, b) => a - b);
const dwStepSite = (m, st) => st.fam === "layer" ? [m.steps[st.anchor], st.b] : [m.steps[st.b], st.anchor];
const dwName = (st, f) => `${st.dwk ? "ψ" : "φ"}${st.dwj},${f + 1}`;
// clamp the level, kind and function to what the snapshot stores; resolve a clicked image to a function
function fitDwFun(st, cell, N) {
  const levs = dwLevels(cell);
  if (!levs.includes(st.dwj)) st.dwj = levs.find(j => dwGroup(cell, j, 1).length) ?? levs[0];
  if (st.dwk !== 0 && st.dwk !== 1) st.dwk = 1;
  if (!dwGroup(cell, st.dwj, st.dwk).length) st.dwk = st.dwk ? 0 : 1;
  const g = dwGroup(cell, st.dwj, st.dwk);
  if (st.dwi != null) {     // largest |f(i)| relative to the function's own peak, precomputed at full precision (build_site.py)
    const gi = cell.groups.findIndex(x => x[0] === st.dwj && x[1] === st.dwk);
    if (gi >= 0) st.dwf = cell.win[gi * N + st.dwi];
  }
  if (!(st.dwf >= 0 && st.dwf < g.length)) st.dwf = 0;
  return g[st.dwf];
}

// ------------------------------------------------------------------------------------------- change to the next site
// build_site.py change(), change.py, SPEC_CHANGE.md (cosine root, kernel rule meta.ch.rule, checkpoint meta.ch.t,
// adjacent sites a -> a + 1): D = M_b^tau - M_a^tau. State: st.chv ("score" | "mode"), st.chts / st.chtm (walk steps
// tau of the score and pattern views, remembered separately), st.chk (pattern 0..2K-1: 0..K-1 sharpen, K..2K-1
// fade), st.chi (selected image or null). st._ch = {pair, a} once loaded; colouring() records what it painted in
// col.ch, and the side card reads only that, so a pending load never mixes two pairs on screen.
const RG_VIEW = { shape: "shape change", dil: "dilation of the fitted map", shear: "shear of the fitted map" };
const adjacent = st => st.color === "ch" || st.color === "rg";
const CH_WORDS = { 1: "1 step: direct neighbours", 4: "4 steps: close neighbourhoods", 8: "8 steps: wider neighbourhoods", 16: "16 steps: broad neighbourhoods" };
const chTau = st => st.chv === "mode" ? st.chtm : st.chts;
const chWords = t => CH_WORDS[t] || `${t} steps`;
const chSteps = t => t === 1 ? "1 step" : `${t} steps`;
const chName = (K, k) => k < K ? `sharpens ${k + 1}` : `fades ${k - K + 1}`;
const chA = t => /^(8|11|18)/.test(String(t)) ? "an" : "a";      // "an 8-step walk"
// factor from the stored score to the shown one: at one step the lazy walk's tv is doubled to the share of neighbour
// weight that moved; data format "ch-v1" stored it doubled already
const chFactor = (m, t) => t === 1 && m.ch.format !== "ch-v1" ? 2 : 1;
// a change in probability (or neighbour weight), as signed percentage points
const chChg = v => (v >= 0 ? "+" : "−") + (100 * Math.abs(v)).toPrecision(2) + "%";

// ------------------------------------------------------------------------------------------- colouring
// Returns {ci: [T] Uint8Array(N), order: [T] Uint16Array(N), cols: css[], legend}
export function colouring(ctx, st, blk) {
  if (st.color === "depth") return depthColouring(ctx, st);
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
    out = { ...same(ci, hashOrder), cols, groups: { ids: ci, n: K.k, names: [...Array(K.k)].map((_, j) => `group ${j + 1}`) } };
  } else if (c === "dw") {
    if (!st._dw) { out = { ...same(new Uint8Array(N), hashOrder), cols: [DIM] }; }
    else {
      const W = st._dw, f = dwGroup(W.cell, st.dwj, st.dwk)[st.dwf], v = D.dwFun(m, W.cell, f);
      let mx = 0; for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(v[i]));
      const ci = Uint8Array.from(v, x => qDiv(x / (mx || 1))), key = Float64Array.from(v, x => Math.abs(x));
      out = { ...same(ci, orderBy(key)), cols: divLUT(pal), cont: { kind: "div", lo: -1, hi: 1, label: `${dwName(st, st.dwf)} / max|${dwName(st, st.dwf)}|` },
        dw: { f, v, mx, row: W.cell.fun[f] } };
    }
  } else if (c === "rg" && (!m.rg || !st._rg)) {
    out = { ...same(new Uint8Array(N), hashOrder), cols: [DIM] };
  } else if (c === "rg") {
    const pairs = st._rg, view = st.rgv, [lo, hi] = m.rg.ranges[view], div = view === "dil";
    const cols = div ? divLUT(pal) : seqLUT(pal), dim = cols.length;
    const sites = pairs.map(p => Uint8Array.from(p[view], x => !Number.isFinite(x) ? dim : div ? qDiv(x / hi) : qSeq((x - lo) / (hi - lo))));
    const orders = pairs.map(p => orderBy(Float64Array.from(p[view], x => Number.isFinite(x) ? Math.abs(x) : -1)));
    sites.push(new Uint8Array(N).fill(dim)); orders.push(hashOrder);
    // The measurement is from the final checkpoint. Only the layout changes down a column.
    out = { ...same(sites.at(-1), hashOrder), ciCell: Array(T).fill(sites).flat(), orderCell: Array(T).fill(orders).flat(),
      cols: [...cols, DIM], cont: { kind: div ? "div" : "seq", lo, hi, label: RG_VIEW[view] },
      rg: { view, a: st.a, b: st.b, pair: pairs[st.a] } };
  } else if (c === "ch") {
    const C = st._ch, ti = C ? m.ch.taus.indexOf(chTau(st)) : -1;
    if (!C || ti < 0) { out = { ...same(new Uint8Array(N), hashOrder), cols: [DIM] }; }
    else {
      // the score as shown: tv on the lazy walk M; at one step doubled, the share of the neighbour weight (P = 2M - I)
      // that moved ("ch-v1" data stored it doubled already). sc, scMax, scMean are what every label and the legend show.
      const d = C.pair.tau[ti], f = chFactor(m, d.t), score = D.chScore(m, C.pair, ti), K = m.ch.k, sc = score.map(x => f * x);
      const info = { pair: C.pair, a: C.a, b: C.pair.b, ti, tau: d.t, d, view: st.chv, k: st.chk, sc, scMax: f * d.tv_max, scMean: f * d.tv_mean };
      if (st.chv === "mode") {      // one pattern (eigenvector of D), diverging, scaled to +-1; largest |v| on top
        const v = D.chMode(m, C.pair, ti, st.chk); let mx = 0; for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(v[i]));
        out = { ...same(Uint8Array.from(v, x => qDiv(x / (mx || 1))), orderBy(Float64Array.from(v, Math.abs))), cols: divLUT(pal),
          cont: { kind: "div", lo: -1, hi: 1, label: `pattern “${chName(K, st.chk)}”, scaled to ±1` }, ch: { ...info, v, mx } };
      } else {                      // per-image score, sequential from 0 to the largest at this pair and scale; largest on top
        out = { ...same(Uint8Array.from(score, x => qSeq(x / (d.tv_max || 1))), orderBy(score)), cols: seqLUT(pal),
          cont: { kind: "seq", lo: 0, hi: info.scMax, exact: true, fmt: pct, label: d.t === 1 ? "share of neighbour weight that moved" : `share of the ${d.t}-step walk that moved` }, ch: info };
      }
    }
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

// ------------------------------------------------------------------------------------------- through depth
// SPEC_SPACETIME.md sections 4 and 6; data from build_site.py spacetime() (cosine root, one kernel rule). One checkpoint
// row (st.anchor) is coloured, each site's panel by its own slice of the selected mode (u_l = f_l / |f_l|, diverging,
// scaled to max |u_l|) or by the set readout; every other row is light grey. A slice holding less than
// meta.spacetime.low_weight of its mode's squared norm is drawn in a uniform grey: a normalised numerical tail is not a
// direction. State: st.dv (variant id), st.dt (coupling index; 0 for paper), st.dm (1..K or "sets"), st.dk (k), st.dr
// (readout "a" | "s"), st.dsite (site shown in the images card; null = automatic). st._dep = {U, Sm, step} once loaded.
const DEPTH_LOW = "#bdbcb6";
export const depthReady = st => st.color === "depth" && !!st._dep;
const depthC = st => st.dv === "paper" ? 0 : st.dt;
export const depthPlane = (m, st) => m.spacetime.sets.findIndex(s => s.k === st.dk && s.readout === st.dr);
export const fracLabel = f => f >= 1 ? String(f) : `1/${Math.round(1 / f)}`;
// the per-site arrays of the current setting: {R (variant readings), c, w (weights of the mode or null), k}
export function depthView(ctx, st) {
  const R = st._dep.Sm.variants[st.dv], c = depthC(st), k = st.dm === "sets" ? null : st.dm - 1;
  return { R, c, k, w: k == null ? null : R.weight[c][k] };
}
// labels of the set readout at site s (Int8Array N, -1 = no label)
export function depthLabels(ctx, st, s) {
  const U = st._dep.U, p = depthPlane(ctx.m, st), N = ctx.N;
  return U.lab.subarray(((p * U.C + depthC(st)) * U.L + s) * N, ((p * U.C + depthC(st)) * U.L + s + 1) * N);
}
// the int8 unit slice of the mode at site s (scaled so max |u| = 127)
export function depthSlice(ctx, st, s) {
  const U = st._dep.U, N = ctx.N, off = ((depthC(st) * U.K + st.dm - 1) * U.L + s) * N;
  return U.vecs.subarray(off, off + N);
}
// per-site status of the selected row: low (grey) and why
export function depthSiteInfo(ctx, st) {
  const { m, S } = ctx, low = m.spacetime.low_weight, { w } = depthView(ctx, st);
  return [...Array(S)].map((_, s) => {
    if (w) return { w: w[s], low: !(w[s] >= low) };
    const lab = depthLabels(ctx, st, s); let n = 0; for (const l of lab) if (l >= 0) n++;
    return { labelled: n, low: n === 0 };
  });
}
function depthColouring(ctx, st) {
  const { pal, T, S, N } = ctx, t0 = st.anchor, ident = Uint16Array.from({ length: N }, (_, i) => i);
  const ciCell = Array(T * S).fill(null), orderCell = Array(T * S).fill(null), info = depthSiteInfo(ctx, st);
  let cols, grey, out;
  if (st.dm !== "sets") {
    cols = divLUT(pal).concat([DIM, DEPTH_LOW]);
    const G = cols.length - 2, LO = cols.length - 1;
    grey = new Uint8Array(N).fill(G);
    for (let s = 0; s < S; s++) {
      if (info[s].low) { ciCell[t0 * S + s] = new Uint8Array(N).fill(LO); orderCell[t0 * S + s] = ident; continue; }
      const v = depthSlice(ctx, st, s); let mx = 0;
      for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(v[i]));
      ciCell[t0 * S + s] = Uint8Array.from(v, x => qDiv(x / (mx || 1)));
      orderCell[t0 * S + s] = Uint16Array.from(ident).sort((a, b) => Math.abs(v[a]) - Math.abs(v[b]));    // largest |u| on top
    }
    out = { cont: { kind: "div", lo: -1, hi: 1, label: `mode ${st.dm}: u_l / max|u_l|, each site its own slice` } };
  } else {
    const k = st.dk, f = st.focus;
    cols = [...Array(k)].map((_, j) => kmColour(pal, j)).concat([pal.unassigned, DIM]);
    const NO = k, G = k + 1;
    grey = new Uint8Array(N).fill(G);
    for (let s = 0; s < S; s++) {
      const lab = depthLabels(ctx, st, s);
      const ci = Uint8Array.from(lab, l => l < 0 ? NO : f != null && l !== f ? G : l);
      ciCell[t0 * S + s] = ci;
      // the hash order of the other colourings, with grey (no label, or outside the isolated set) drawn first
      orderCell[t0 * S + s] = Uint16Array.from(ident).sort((a, b) => (ci[a] >= NO ? 0 : 1) - (ci[b] >= NO ? 0 : 1) || (Math.sin(a * 12.9898) * 43758.5453 % 1) - (Math.sin(b * 12.9898) * 43758.5453 % 1));
    }
    out = { sets: { k, names: [...Array(k)].map((_, j) => `set ${j + 1}`), none: NO } };
  }
  return { ...out, ci: Array(T).fill(grey), order: Array(T).fill(ident), ciCell, orderCell, cols, depth: { info, t: t0 }, pal32: new Uint32Array(cols.map(css32)) };
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
    // per-panel colours where a colouring has them (through depth: each site its own slice), else one array per row
    const cell = c.t * c.S + c.s, ci = col.ciCell?.[cell] || col.ci[c.t], order = col.orderCell?.[cell] || col.order[c.t];
    const pal = col.pal32, base = cell * N * 2;
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
      dv: "own", dt: 3, dm: 1, dk: 5, dr: "a", dsite: null,      // through depth (depthColouring)
      dwj: null, dwk: 1, dwf: 0, dwi: null,                       // diffusion wavelets (fitDwFun)
      rgv: "shape", rgi: null,
      chv: "score", chts: 1, chtm: 16, chk: 0, chi: null,         // change to the next site (fitCh)
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
      segInit("ex-at", () => this.st.at, v => this.st.at = v),
      segInit("ex-lay", () => this.st.lay, v => this.setLayout(+v)),
      segInit("ex-tap", () => this.st.tap, v => { this.st.tap = v; }),
      segInit("ex-kern", () => this.st.kern, v => this.setKernel(v)),
      segInit("ex-kms", () => this.st.kms, v => { this.st.kms = v; this.st.focus = null; }),
      segInit("ex-dv", () => this.st.dv, v => { this.st.dv = v; this.st.focus = null; this.st.dsite = null; }),
      segInit("ex-dr", () => this.st.dr, v => { this.st.dr = v; this.st.focus = null; }),
      segInit("ex-dwk", () => this.st.dwk, v => { this.st.dwk = +v; if (this.st.dwi == null) this.st.dwf = 0; }),
      segInit("ex-rgv", () => this.st.rgv, v => this.st.rgv = v),
      segInit("ex-chv", () => this.st.chv, v => { this.st.chv = v === "mode" ? "mode" : "score"; }),
    ];
    $("ex-dwj").addEventListener("change", e => { this.st.dwj = +e.target.value; if (this.st.dwi == null) this.st.dwf = 0; this.update(); });
    $("ex-dwf").addEventListener("change", e => { this.st.dwf = +e.target.value; this.st.dwi = null; this.update(); });
    if (!m.dw) $("ex-color-dw").remove();
    if (m.ch) {      // change to the next site: scale (walk steps, in words), pattern (labels filled per pair in syncCh)
      m.ch.taus.forEach(t => $("ex-cht").add(new Option(chWords(t), t)));
      $("ex-cht").addEventListener("change", e => { const t = +e.target.value; if (this.st.chv === "mode") this.st.chtm = t; else this.st.chts = t; this.update(); });
      $("ex-chk").addEventListener("change", e => { this.st.chk = +e.target.value; this.update(); });
      $("ex-ch-lock").textContent = `step ${m.ch.step.toLocaleString("en")} · cosine · ${m.modes[m.ch.rule]} (fixed)`;
    } else $("ex-color-ch").remove();
    if (m.rg) $("ex-rg-lock").textContent = `Final checkpoint (step ${m.rg.step.toLocaleString("en")}) · neighbouring sites · cosine · rule ${m.rg.rule}: ${m.modes[m.rg.rule]} (fixed)`;
    else $("ex-color-rg").remove();
    this.buildDepthControls();
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
    m.modes.forEach((name, i) => $("ex-mode").add(new Option(name, i)));
    $("ex-mode").addEventListener("change", e => { this.st.mode = +e.target.value; this.st.focus = null; this.update(); });
    // disable diagnostics that the data do not have
    const have = new Set(m.points_fields);
    for (const [c, f] of Object.entries(DIAG)) if (!have.has(f)) { const o = $("ex-color").querySelector(`option[value=${c}]`); if (o) { o.disabled = true; o.dataset.missing = "1"; o.textContent += " (not in data)"; } }
    const og = document.createElement("optgroup"); og.label = "Image statistics (pixels)"; og.dataset.dev = "1";
    for (const k of PX_MENU) og.append(new Option(IMGSTATS[k].label, "px:" + k));
    $("ex-color").append(og);
    $("ex-color").addEventListener("change", e => this.setColor(e.target.value));
    this.buildSources();
    $("ex-k").addEventListener("change", e => {
      if (this.st.kind === "psi") this.st.k = +e.target.value; else { this.st.r = +e.target.value; this.st.j = null; }
      if (BLOCK_KINDS.has(this.st.color) || this.st.color === "seba") this.st.focus = null;
      this.update();
    });
    $("ex-ps").addEventListener("input", e => { this.st.ps = +e.target.value; this.draw(); });
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
  // through depth: variant (segment, set up with the others), coupling slider, mode or sets; absent without the add-on
  buildDepthControls() {
    const m = this.ctx.m, S_ = m.spacetime;
    if (!S_) { $("ex-color-depth-grp").remove(); return; }
    const dm = $("ex-dm");
    for (let k = 1; k <= S_.K; k++) dm.add(new Option(`mode ${k}`, String(k)));
    const og = document.createElement("optgroup"); og.label = "Sets (post-hoc k-means readout)";
    [...new Set(S_.sets.map(s => s.k))].forEach(k => og.append(new Option(`sets, k = ${k}`, "s" + k)));
    dm.append(og);
    dm.addEventListener("change", e => {
      const v = e.target.value, st = this.st;
      if (v[0] === "s") { st.dm = "sets"; st.dk = +v.slice(1); } else st.dm = +v;
      st.focus = null; st.dsite = null; this.update();
    });
    const dt = $("ex-dt"); dt.max = String(S_.factors.length - 1);
    dt.addEventListener("input", e => { this.st.dt = +e.target.value; this.st.focus = null; this.syncDepthLabel(); this.update(); });
    $("ex-depth-lock").textContent = `kernel: cosine · rule: ${m.modes[S_.rule]} (fixed)`;
  }
  // coupling label: t / t* from meta, the actual t from the loaded summary of the shown checkpoint
  syncDepthLabel() {
    const st = this.st, S_ = this.ctxCos.m.spacetime;
    if (!S_) return;
    const f = S_.factors[st.dt], R = st._dep?.Sm.variants[st.dv];
    const t = R && R.t && st._dep.step === this.ctxCos.m.steps[st.anchor] ? R.t[st.dt] : null;
    $("ex-dt-val").textContent = `t/t* = ${fracLabel(f)}` + (t != null ? ` · t = ${(+t.toPrecision(3)).toString()}` : "");
    $("ex-dt").value = String(st.dt);
    $("ex-dt").setAttribute("aria-valuetext", $("ex-dt-val").textContent);
  }
  // the through-depth colouring fixes the kernel (cosine), the rule (meta.spacetime.rule) and the layer family (one
  // checkpoint row); out-of-range settings are clamped. Returns notes for the status line.
  fitDepth(st = this.st) {
    const out = [], cm = this.ctxCos.m, S_ = cm.spacetime;
    if (st.color !== "depth") return out;
    if (!S_) { out.push("Through depth is not in this build; showing coherent sets."); st.color = "seba"; st.kind = "set"; return out; }
    if (st.kern !== "cos") { out.push("Through depth exists for the cosine kernel only; showing cosine."); st.kern = "cos"; }
    if (st.mode !== S_.rule) { out.push(`Through depth exists for kernel rule "${cm.modes[S_.rule]}" only; showing it.`); st.mode = S_.rule; }
    if (st.fam !== "layer") { const t = st.b; st.fam = "layer"; st.anchor = t; st.a = 0; st.b = cm.sites.length - 1; st.cmp = null; }
    st.src = "M0";
    if (!S_.variants.some(v => v.id === st.dv)) st.dv = "own";
    if (!(st.dt >= 0 && st.dt < S_.factors.length)) st.dt = S_.factors.indexOf(1) >= 0 ? S_.factors.indexOf(1) : 0;
    if (st.dm !== "sets" && !(st.dm >= 1 && st.dm <= S_.K)) st.dm = 1;
    if (!S_.sets.some(s => s.k === st.dk)) st.dk = S_.sets[0].k;
    if (!(st.dr in S_.readouts)) st.dr = "a";
    if (st.dsite != null && !(st.dsite >= 0 && st.dsite < cm.sites.length)) st.dsite = null;
    if (st.focus != null && (st.dm !== "sets" || !(st.focus >= 0 && st.focus < st.dk))) st.focus = null;   // an isolated set must exist
    return out;
  }
  // diffusion wavelets: cosine kernel, one kernel rule, single snapshots (a multi-snapshot block keeps its last snapshot)
  fitDw(st = this.st) {
    const out = [], cm = this.ctxCos.m, W = cm.dw;
    if (st.color !== "dw") return out;
    if (!W) { out.push("Diffusion wavelets are not in this build; showing eigenvectors."); st.color = "eig"; st.kind = "psi"; return out; }
    if (st.kern !== "cos") { out.push("Diffusion wavelets exist for the cosine kernel only; showing cosine."); st.kern = "cos"; }
    if (st.mode !== W.rule) { out.push(`Diffusion wavelets exist for kernel rule "${cm.modes[W.rule]}" only; showing it.`); st.mode = W.rule; }
    if (st.a !== st.b) { out.push("Diffusion wavelets are per snapshot: showing the block's last snapshot. Click a point to pick another."); st.a = st.b; st.cmp = null; }
    if (st.dwi != null && !(st.dwi >= 0 && st.dwi < cm.n)) st.dwi = null;
    st.src = "M0";
    return out;
  }
  // change to the next site: cosine kernel, rule meta.ch.rule, the layer family at checkpoint meta.ch.t, and an adjacent
  // pair (a, a + 1); a longer block snaps to a = min(a, last site - 1). Returns notes for the status line.
  fitCh(st = this.st) {
    const out = [], cm = this.ctxCos.m, C = cm[st.color], S = cm.sites.length, nm = cm.sites;
    if (!adjacent(st)) return out;
    const what = st.color === "rg" ? "The neighbourhood stretch" : "The change between sites", view = st.color === "rg" ? "This colouring" : "The change view";
    if (!C) { out.push(`${what} is not in this build; showing coherent sets.`); st.color = "seba"; st.kind = "set"; return out; }
    if (st.kern !== "cos") { out.push(`${what} exists for the cosine kernel only; showing cosine.`); st.kern = "cos"; }
    if (st.mode !== C.rule) { out.push(`${what} exists for kernel rule "${cm.modes[C.rule]}" only; showing it.`); st.mode = C.rule; }
    const final = `step ${C.step.toLocaleString("en")}`;
    const anchor = st.color === "rg" ? C.anchor : C.t;
    if (st.fam !== "layer") {        // a time block at site s: compare s with the next site at the stored checkpoint
      const s = st.anchor; st.fam = "layer"; st.anchor = anchor; st.a = Math.min(s, S - 2); st.b = st.a + 1;
      out.push(`${view} compares two neighbouring sites at ${final}: showing ${nm[st.a]} → ${nm[st.b]}.`);
    }
    if (st.anchor !== anchor) { out.push(`${what} is stored for the final checkpoint (${final}) only; showing it.`); st.anchor = anchor; }
    if (st.b !== st.a + 1) {
      st.a = Math.min(st.a, S - 2); st.b = st.a + 1;
      out.push(`${view} compares two neighbouring sites: showing ${nm[st.a]} → ${nm[st.b]}.`);
    }
    st.src = "M0"; st.cmp = null; st.focus = null;
    if (st.color === "rg") {
      if (!Object.hasOwn(RG_VIEW, st.rgv)) st.rgv = "shape";
      return out;
    }
    if (st.chv !== "mode") st.chv = "score";
    if (!C.taus.includes(st.chts)) st.chts = 1;
    if (!C.taus.includes(st.chtm)) st.chtm = 16;
    if (!(st.chk >= 0 && st.chk < 2 * C.k)) st.chk = 0;
    if (st.chi != null && !(st.chi >= 0 && st.chi < cm.n)) st.chi = null;
    return out;
  }
  // "Sets from": the colour sources of the baselines study, grouped by kind, with status tags
  buildSources() {
    const sel = $("ex-src");
    if (!this.srcIdx) { $("ex-src-ctl").hidden = true; return; }
    let og = null;
    for (const s of SR.SOURCES.filter(s => SRC_MENU.has(s.id))) {
      if (!og || og.label !== s.group) { og = document.createElement("optgroup"); og.label = s.group; sel.append(og); }
      const o = new Option(s.label, s.id); o.dataset.base = o.textContent; o.title = s.def; og.append(o);
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
      out.push(`k-means groups: ${why}; showing coherent sets.`);
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
      if (this.col && this.geom) { this.draw(); this.drawOverlay(); }
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
      $("ex-src-note").hidden = st.mode === this.srcIdx.kernel_rule && ctx.kernel === "cos";
      $("ex-src-note").textContent = ctx.kernel !== "cos" ? "Other sources: cosine kernel only" : `Other sources: kernel rule ${this.ctxCos.m.modes[this.srcIdx.kernel_rule]} only`;
    }
    const ks = $("ex-k");
    const want = st.kind === "psi" ? "psi" + maxK(ctx, st) : "set";
    if (ks._kind !== want) {
      ks._kind = want; ks.innerHTML = "";
      if (st.kind === "psi") for (let k = 1; k <= maxK(ctx, st); k++) ks.add(new Option(`eigenvector ${k}`, k));
      else m.seba_r.forEach(r => ks.add(new Option(`${r} groups`, r)));
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
    this.syncDepth(ctx);
    this.syncDw(ctx);
    this.syncCh(ctx);
  }
  // change to the next site: its controls replace the block's selectors; kernel and rule are locked and say so (after
  // syncDepth, which unlocks them when through depth is off)
  syncCh(ctx = this.ctx) {
    const st = this.st, m = ctx.m, on = st.color === "ch" && !!m.ch;
    $("ex").classList.toggle("ch", on);
    $("ex-ch-ctl").hidden = !on;
    $("card-ch").hidden = !on;
    const rg = st.color === "rg" && !!m.rg;
    $("ex").classList.toggle("rg", rg); $("ex-rg-ctl").hidden = !rg; $("card-rg").hidden = !rg;
    if (!on && !rg) return;
    $("ex-k").hidden = true; $("ex-k-label").hidden = true;
    if (this.srcIdx) $("ex-src-ctl").hidden = true;
    const why = "Fixed while comparing neighbouring sites at the final checkpoint";
    $("ex-mode").disabled = true; $("ex-mode").title = why;
    $("ex-kern").querySelectorAll("button").forEach(b => { b.disabled = true; b.title = why; });
    if (rg) return;
    $("ex-cht").value = String(chTau(st));
    const sk = $("ex-chk"), C = st._ch, ti = m.ch.taus.indexOf(chTau(st)), key = C && ti >= 0 ? `${C.a}/${ti}` : "";
    sk.hidden = st.chv !== "mode";
    if (sk._key !== key) {        // labels carry each pattern's share of the change at this pair and scale
      sk._key = key; sk.replaceChildren();
      for (let k = 0; k < 2 * m.ch.k; k++) sk.add(new Option(`${chName(m.ch.k, k)}${key ? ` · ${share(C.pair.tau[ti].share[k])} of the change` : ""}`, k));
    }
    sk.value = String(st.chk);
  }
  syncDw(ctx = this.ctx) {
    const st = this.st, on = st.color === "dw" && !!ctx.m.dw;
    $("ex").classList.toggle("dw", on);
    $("ex-dw-ctl").hidden = !on;
    $("card-dw").hidden = !on;
    if (!on) return;
    $("ex-k").hidden = true; $("ex-k-label").hidden = true;
    const W = st._dw; if (!W) return;
    const cell = W.cell, key = `${W.step}_${W.site}`;
    const sj = $("ex-dwj");
    if (sj._key !== key) {
      sj._key = key; sj.replaceChildren();
      for (const j of dwLevels(cell)) {
        const L = cell.levels.find(r => r[0] === j), nw = dwGroup(cell, j, 1).length;
        sj.add(new Option(`scale j = ${j} (M^${2 ** j}): ${L[3].toLocaleString("en")} wavelets${nw < L[3] ? `, ${nw} stored` : ""}`, j));
      }
    }
    sj.value = st.dwj;
    $("ex-dwk").querySelectorAll("button").forEach(b => { const n = dwGroup(cell, st.dwj, +b.dataset.v).length; b.disabled = !n; b.title = b.title.replace(/ \(none stored at this scale\)$/, "") + (n ? "" : " (none stored at this scale)"); });
    const sf = $("ex-dwf"), fk = `${key}/${st.dwj}/${st.dwk}`;
    if (sf._key !== fk) {
      sf._key = fk; sf.replaceChildren();
      dwGroup(cell, st.dwj, st.dwk).forEach((f, q) => sf.add(new Option(`${dwName(st, q)}: centre #${cell.fun[f][2]}`, q)));
    }
    sf.value = st.dwf;
  }
  // through depth: its controls replace the block's selectors; kernel and rule are locked and say so
  syncDepth(ctx = this.ctx) {
    const st = this.st, on = st.color === "depth" && !!ctx.m.spacetime;
    $("ex").classList.toggle("depth", on);
    $("ex-depth-ctl").hidden = !on;
    if (on) { $("ex-k").hidden = true; $("ex-k-label").hidden = true; } else $("ex-k").hidden = false;
    if (this.srcIdx) $("ex-src-ctl").hidden = on;
    const lock = (e, why) => { if (e.dataset.t1 == null) e.dataset.t1 = e.title || ""; e.disabled = on; e.title = on ? why : e.dataset.t1; };
    lock($("ex-mode"), "Fixed while colouring through depth: the through-depth data exist for this kernel rule only");
    $("ex-kern").querySelectorAll("button").forEach(b => lock(b, "Fixed while colouring through depth: cosine kernel only"));
    if (!on) return;
    $("ex-dt-wrap").hidden = st.dv === "paper";
    $("ex-dr").hidden = st.dm !== "sets";
    $("ex-dm").value = st.dm === "sets" ? "s" + st.dk : String(st.dm);
    this.syncDepthLabel();
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
    if (this.srcIdx) dis($("ex-src"), dot, "Sets from uses the baselines study, computed for the cosine kernel only");
    for (const id of ["ex-color-ch", "ex-color-rg"]) { const o = $(id); if (o) { o.hidden = dot; o.disabled = dot; } }     // the change data: cosine root only
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
    if (this.drag && this.st.color !== "depth" && this.st.color !== "dw") {
      const st = this.st, d = this.drag;
      const along = hc.t === d.t || hc.s === d.s;
      if (along && (hc.t !== d.t || hc.s !== d.s)) { d.moved = true; d.end = hc; d.fam = hc.t === d.t ? "layer" : "time"; this.dragPrev = d; this.drawOverlay(); }
    }
    const cr = this.cellRect(hc.t, hc.s);
    const i = this.nearest(hc.t, hc.s, hc.x, hc.y, cr, Math.max(6, this.geom.cell * .07));
    this.lensCell = [hc.t, hc.s];
    this.setHover(i, hc.t);
    if (i >= 0) {
      const key = `${i}/${hc.t}/${hc.s}/${this._tok}`;
      if (key !== this._tipKey) { this._tipKey = key; this._tipEl = this.pointTip(i, hc.t, hc.s); }
      showTip(e.clientX, e.clientY, this._tipEl);
    } else { this._tipKey = null; hideTip(); }
    this.canvas.title = "";
    // prefetch the block a click here would select
    if (!this.drag && this.st.color !== "depth" && this.st.color !== "dw" && !adjacent(this.st)) { const nb = this.clickTarget(hc, e.shiftKey); if (nb) blockOf(this.ctx, nb).catch(() => {}); }
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
    if (st.color === "depth") {      // through depth: a click (or the end of a drag) picks the checkpoint row only
      const hc = this.hitCell(e) || d.end || d;
      if (d.pointer !== "mouse" && hc.x != null) {
        const i = this.nearest(hc.t, hc.s, hc.x, hc.y, this.cellRect(hc.t, hc.s), Math.max(10, this.geom.cell * .12));
        this.setHover(i, hc.t);
      }
      this.dragPrev = null; this.lensCell = [hc.t, hc.s];
      if (hc.t === st.anchor && st.fam === "layer" && !this.depthPending()) { this.setDepthSite(hc.s); return; }   // same row: show that site's images
      st.fam = "layer"; st.anchor = hc.t; st.focus = null; st.j = null;
      hideTip(); this._tipKey = null;      // the tooltip describes the old row
      this.update();
      return;
    }
    if (st.color === "dw") {      // diffusion wavelets: a click picks that panel's snapshot and the image under the pointer
      const hc = this.hitCell(e) || d.end || d;
      if (hc.x != null) {
        const i = this.nearest(hc.t, hc.s, hc.x, hc.y, this.cellRect(hc.t, hc.s), Math.max(8, this.geom.cell * .1));
        if (i >= 0) { st.dwi = i; this.setHover(i, hc.t); }
      }
      this.dragPrev = null; this.lensCell = [hc.t, hc.s];
      if (st.fam === "layer") { st.anchor = hc.t; st.a = st.b = hc.s; } else { st.anchor = hc.s; st.a = st.b = hc.t; }
      hideTip(); this._tipKey = null;
      this.update();
      return;
    }
    // change to the next site: a click selects the image under the pointer and, outside the two outlined columns, the
    // pair (s, s + 1); a drag along a row picks a block, snapped to its first adjacent pair (fitCh keeps the checkpoint)
    if (adjacent(st)) {
      const S = this.ctx.S, nm = this.ctx.m.sites, view = st.color === "rg" ? "This colouring" : "The change view";
      let note = null;
      if (d.moved && d.end) {
        if (d.fam === "layer") {
          const s0 = Math.min(d.s, d.end.s), s1 = Math.max(d.s, d.end.s);
          st.a = Math.min(s0, S - 2); st.b = st.a + 1;
          if (s1 !== s0 + 1) note = `${view} compares two neighbouring sites: showing ${nm[st.a]} → ${nm[st.b]}, the first pair of the dragged block.`;
        } else note = `${view} compares neighbouring sites at the final checkpoint: drag along a row, or click a panel.`;
      } else {
        const hc = this.hitCell(e) || d;
        if (hc.x != null) {
          const i = this.nearest(hc.t, hc.s, hc.x, hc.y, this.cellRect(hc.t, hc.s), Math.max(8, this.geom.cell * .1));
          if (i >= 0) { st[st.color === "rg" ? "rgi" : "chi"] = i; this.setHover(i, hc.t); }
        }
        if (st.color === "rg" || (hc.s !== st.a && hc.s !== st.b)) { st.a = Math.min(hc.s, S - 2); st.b = st.a + 1; }
      }
      this.dragPrev = null; this.lensCell = [d.t, d.s];
      hideTip(); this._tipKey = null;
      if (note) this._depthNotes = [note];
      this.update();
      return;
    }
    if (d.moved && d.end) {
      if (d.fam !== st.fam) { st.fam = d.fam; st.cmp = null; }
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
  setHover(i, t) {
    if (i === this.hover) return;
    this.hover = i; this.hoverT = t;
    this.drawOverlay();
    document.querySelectorAll(".imgs canvas.hl").forEach(c => c.classList.remove("hl"));
    if (i >= 0) document.querySelectorAll(`.imgs canvas[data-i="${i}"]`).forEach(c => c.classList.add("hl"));
  }

  thumb(i) { return thumb(this.ctx, i); }
  thumbnailTip(i) {
    return this.col?.rg ? this.pointTip(i, this.ctx.m.rg.anchor, this.col.rg.a) : this.pointTip(i, ...this.lensCell);
  }
  pointTip(i, t, s) {
    const { m, labels, pts, N } = this.ctx, st = this.st;
    const rows = [];
    rows.push({ name: "class", value: m.classes[labels[i]] });
    const tt = st.at === "final" ? m.steps.length - 1 : t;
    if (pts.pred) rows.push({ name: `predicted at ${fmtStep(m.steps[tt])}`, value: m.classes[pts.pred[tt * N + i]] });
    if (pts.loss) rows.push({ name: "loss", value: fmt(pts.loss[tt * N + i], 3) });
    const blk = this.blk;
    if (st.color === "depth" && this.depthPending()) {
      rows.push({ name: "through depth", value: "loading…" });
      return h("div", { class: "tt-img" }, this.thumb(i), tipRows(`#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`, rows));
    }
    if (depthReady(st)) {          // through depth: this panel's own slice or set label, on the coloured row only
      const head = `#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`;
      if (t !== st.anchor) rows.push({ name: "through depth", value: `not coloured (shows step ${m.steps[st.anchor].toLocaleString("en")})` });
      else if (st.dm !== "sets") {
        const info = this.col.depth.info[s], v = depthSlice(this.ctx, st, s); let mx = 0; for (const x of v) mx = Math.max(mx, Math.abs(x));
        rows.push({ name: `mode ${st.dm}, slice at ${m.sites[s]} (scaled to ±1)`, value: info.low ? "grey: under 1% of the mode" : fmt(v[i] / (mx || 1)) });
        rows.push({ name: "share of the mode at this site", value: share(info.w) });
      } else {
        const l = depthLabels(this.ctx, st, s)[i];
        rows.push({ name: `sets, k = ${st.dk}, ${this.ctx.m.spacetime.readouts[st.dr]}`, value: l < 0 ? "no label" : `set ${l + 1}`, color: l < 0 ? this.ctx.pal.unassigned : this.col.cols[l] });
      }
      return h("div", { class: "tt-img" }, this.thumb(i), tipRows(head, rows));
    }
    if (st.color === "rg") {
      const C = this.col?.rg, v = C && st._rg?.[s]?.[C.view]?.[i];
      rows.push({ name: C ? RG_VIEW[C.view] + " (final checkpoint)" : "neighbourhood stretch", value: !C ? "loading…" : s === m.sites.length - 1 ? "no next site" : Number.isFinite(v) ? v.toFixed(2) : "no fitted value" });
      return h("div", { class: "tt-img" }, this.thumb(i), tipRows(`#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`, rows));
    }
    if (st.color === "ch" && this.col?.ch) {      // change to the next site: the image's score (and pattern value)
      const C = this.col.ch, nm = m.sites;
      rows.push({ name: `${C.tau === 1 ? "neighbour weight" : `${C.tau}-step walk`} that moved, ${nm[C.a]} → ${nm[C.b]}`, value: pct(C.sc[i]) });
      if (C.v) rows.push({ name: `pattern “${chName(m.ch.k, C.k)}” (scaled to ±1)`, value: fmt(C.v[i] / (C.mx || 1)) });
      if (i === st.chi) rows.push({ name: "selected", value: "gained and lost in the side panel" });
      else rows.push({ name: "click", value: "see the neighbours it gained and lost" });
      return h("div", { class: "tt-img" }, this.thumb(i), tipRows(`#${i} · ${m.sites[s]} · step ${m.steps[t].toLocaleString("en")}`, rows));
    }
    if (blk) {
      if (st.color === "dw" && this.col?.dw) {
        const w = this.col.dw;
        rows.push({ name: `${dwName(st, st.dwf)} (scaled to ±1)`, value: fmt(w.v[i] / (w.mx || 1)) });
        if (i === w.row[2]) rows.push({ name: "centre", value: "this image" });
      } else if (st.kind === "set" && st._S) {
        const S = st._S, l = S.lab[i];
        rows.push({ name: `${SR.SRC[S.id].label}${S.classes ? "" : ` (r = ${st.r})`}`, value: l < 0 ? "unassigned" : S.names[l], color: l < 0 ? this.ctx.pal.unassigned : S.classes ? this.ctx.pal.classes[l] : this.ctx.pal.set(l) });
        const l0 = blk.lab[m.seba_r.indexOf(st.r) * N + i];
        rows.push({ name: `the method (r = ${st.r})`, value: l0 < 0 ? "unassigned" : `set ${l0 + 1}`, color: l0 < 0 ? this.ctx.pal.unassigned : this.ctx.pal.set(l0) });
      } else if (st.kind === "set") {
        const ri = m.seba_r.indexOf(st.r), l = blk.lab[ri * N + i], v = D.sebaVectors(m, blk, ri);
        rows.push({ name: `group (of ${st.r})`, value: l < 0 ? "unassigned" : `set ${l + 1} (membership ${fmt(v[l][i])})`, color: l < 0 ? this.ctx.pal.unassigned : this.ctx.pal.set(l) });
      } else {
        const p = D.psi(m, blk, st.k); let mx = 0; for (const x of p) mx = Math.max(mx, Math.abs(x));
        rows.push({ name: `eigenvector ${st.k} (scaled to ±1)`, value: fmt(p[i] / mx) });
      }
      if (st.color === "km" && st._km) rows.push({ name: `k-means, ${st._km.k} groups${st.a === st.b ? "" : `, ${st._km.side === "end" ? "last" : "first"} site`}`, value: `group ${st._km.lab[i] + 1}`, color: this.col?.cols[st._km.lab[i]] });
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
    const depthNotes = this.fitDepth(st).concat(this.fitDw(st), this.fitCh(st), this._depthNotes || []);   // before any kernel switch: depth, wavelets and change are cosine only
    this._depthNotes = null;
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
    const notes = depthNotes.concat(this.fitState(ctx, nst), this.fitCoarse(ctx, nst));
    if (switching) this.segs.forEach(s => s._sync());
    else { this.syncControls(ctx); this.writeHash(); if (notes.length) this._sticky = { msg: notes.join(" "), hash: this.lastHash }; }
    $("ex-status").textContent = switching ? `loading the ${nst.kern} kernel…` : "loading…";
    $("ex-side").style.opacity = .55;
    let blk, km = null, dep = null, dw = null, ch = null, rg = null;
    try {
      const step = ctx.m.steps[nst.anchor], depth = nst.color === "depth" && ctx.m.spacetime;
      let U, Sm;
      if (depth) $("ex-status").textContent = `loading through-depth data for step ${step.toLocaleString("en")}…`;
      [blk, , U, Sm] = await Promise.all([blockOf(ctx, nst), this.loadCoords(ctx, this.layIndex(ctx, nst.lay)),
        depth ? D.depthUnit(ctx.m, step, nst.dv) : null, depth ? D.depthSummary(step) : null]);
      if (token !== this._tok) return;     // superseded while loading: the newer update owns the status line and the panels
      if (depth) dep = { U, Sm, step, set: Object.fromEntries(["anchor", "dv", "dt", "dm", "dk", "dr"].map(k => [k, nst[k]])) };   // what was loaded
      const pi = pairOf(ctx, nst), cell = nst.a === nst.b;
      if (nst.kind === "psi" && nst.k > ctx.m.n_vecs && !blk.ext) {
        const ext = await D.ext(ctx.m, dkey(nst), nst.anchor, nst.mode, pi, cell);
        if (!blk.ext) blk.ext = ext;
        if (token !== this._tok) return;   // superseded while loading: write no status, start no further loads
      }
      if (nst.color === "dw") {
        const [step_, site_] = dwStepSite(ctx.m, nst), cell = await D.dwCell(ctx.m, step_, site_);
        if (token !== this._tok) return;   // superseded: nst may be the live state, which the newer update has changed
        const sel = { dwj: nst.dwj, dwk: nst.dwk, dwf: nst.dwf, dwi: nst.dwi };    // fitted on a copy: committed below only if still current
        fitDwFun(sel, cell, ctx.N);
        dw = { cell, step: step_, site: site_, sel };
      }
      if (nst.color === "rg") {
        $("ex-status").textContent = "loading neighbourhood stretches…";
        rg = await D.rgData(ctx.m);
        if (token !== this._tok) return;
      }
      if (nst.color === "ch") {      // one file per adjacent pair; committed below only if this update is still current
        $("ex-status").textContent = `loading the change from ${ctx.m.sites[nst.a]} to ${ctx.m.sites[nst.b]}…`;
        const a_ = nst.a, pair = await D.chPair(ctx.m, a_);
        if (token !== this._tok) return;
        ch = { pair, a: a_ };
      }
      if (nst.color === "km") {
        const L = await D.kmLabels(ctx.m, ctx.m.steps[nst.anchor], nst.mode, pi, cell);
        if (token !== this._tok) return;
        const qi = ctx.m.coarse.km.ks.indexOf(nst.kmk);
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
      try { await prepare(ctx, nst); } catch (e) { if (token === this._tok) $("ex-status").textContent = "image statistics unavailable"; console.error(e); return; }
      if (token !== this._tok) return;
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
    if (dw) Object.assign(st, dw.sel);
    this.ctx = ctx; this.blk = blk; st._S = S; st._km = km; st._dep = dep; st._dw = dw; st._ch = ch; st._rg = rg;    // context, block and layout change together: no cosine block under a dot label
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
    if (this.col.depth) {        // through depth: block-only cards are hidden (CSS .ex.depth), the depth card replaces them
      if (this.depthPending()) return;     // a load for other settings is under way: update() redraws when it commits
      this.draw(); this.drawOverlay(); this.decorate(); this.syncDepthLabel();
      this.renderDepthCard(); this.renderLegend(); this.renderDepthImages(); this.renderCaption();
      return;
    }
    this.draw(); this.drawOverlay(); this.decorate();
    this.renderBlockCard(); this.renderLegend(); this.renderImages(); this.renderRetention(); this.renderStats(); this.renderCaption(); this.renderCoarse();
    if (this.col.dw) this.renderDw();
    if (this.col.ch) this.renderCh();
    if (this.col.rg) this.renderRg();
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
    if (st.color === "dw") { if (st.dwj != null) p.set("dwj", st.dwj); p.set("dwk", st.dwk); if (st.dwi != null) p.set("dwi", st.dwi); else p.set("dwf", st.dwf); }
    if (st.color === "rg") { p.set("rgv", st.rgv); if (st.rgi != null) p.set("rgi", st.rgi); }
    if (st.color === "ch") { p.set("chv", st.chv); p.set("cht", chTau(st)); if (st.chv === "mode") p.set("chk", st.chk); if (st.chi != null) p.set("chi", st.chi); }
    if (st.color === "km") { p.set("km", st.kmk); if (st.a !== st.b && st.kms === "end") p.set("kms", "end"); }
    if (st.color === "depth") {      // through depth: variant, coupling (not for paper), mode or sets (k, readout), site of the images card
      p.set("dv", st.dv); if (st.dv !== "paper") p.set("dt", st.dt);
      if (st.dm === "sets") { p.set("dm", "sets"); p.set("dk", st.dk); p.set("dr", st.dr); if (st.focus != null) p.set("j", st.focus); }
      else p.set("dm", st.dm);
      if (st.dsite != null) p.set("ds", st.dsite);
    }
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
    st.op = "avg";      // the operator switch was removed: the explorer shows the block average only
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
    if (p.get("c") === "depth" && !m.spacetime) this._depthNotes = ["Through depth is not in this build."];
    if (st.color === "depth") {      // dv, dt, dm, dk, dr, ds; fitDepth clamps them and fixes kernel, rule and family
      // also read: dv=rel|native, dc (coupling) and dm=s5|s10 (the form some experiment-page links use)
      st.dv = ({ rel: "own", native: "none", nat: "none" })[p.get("dv")] || p.get("dv") || "own";
      st.dt = int("dt", 0, 99) ?? int("dc", 0, 99) ?? (m.spacetime.factors.indexOf(1) >= 0 ? m.spacetime.factors.indexOf(1) : 0);
      const dm = p.get("dm");
      if (/^s\d+$/.test(dm || "")) { st.dm = "sets"; st.dk = +dm.slice(1); st.dr = p.get("dr") || "a"; st.focus = int("j", 0, 15); }
      else if (dm === "sets") { st.dm = "sets"; st.dk = int("dk", 1, 99) ?? 5; st.dr = p.get("dr") || "a"; st.focus = int("j", 0, 15); }
      else { st.dm = int("dm", 1, 999) ?? 1; st.focus = null; }
      st.dsite = int("ds", 0, S - 1);
      if (p.get("dm") != null && st.dm !== "sets" && st.dm > m.spacetime.K) this._depthNotes = [`Mode ${st.dm}: only modes 1 to ${m.spacetime.K} are stored; showing mode 1.`];
      st.j = null;
      this._depthNotes = (this._depthNotes || []).concat(this.fitDepth(st));
    }
    if (st.color === "dw") {      // dwj, dwk, dwi | dwf; fitDwFun clamps them once the snapshot is loaded
      st.dwj = int("dwj", 0, 99); st.dwk = int("dwk", 0, 1) ?? 1; st.dwi = int("dwi", 0, m.n - 1);
      st.dwf = st.dwi == null ? int("dwf", 0, 9999) ?? 0 : 0;
      this._depthNotes = (this._depthNotes || []).concat(this.fitDw(st));
    }
    if (st.color === "rg") {
      st.rgv = Object.hasOwn(RG_VIEW, p.get("rgv")) ? p.get("rgv") : "shape";
      st.rgi = /^\d+$/.test(p.get("rgi") || "") ? int("rgi", 0, m.n - 1) : null;
      this._depthNotes = (this._depthNotes || []).concat(this.fitCh(st));
    }
    if (p.get("c") === "ch" && !m.ch) this._depthNotes = (this._depthNotes || []).concat(["The change between sites is not in this build."]);
    if (st.color === "ch") {      // chv, cht (of that view; the other view keeps its scale), chk, chi; a bad value gets the default
      st.chv = p.get("chv") === "mode" ? "mode" : "score";
      const t = int("cht", 1, 999), ok = t != null && m.ch.taus.includes(t);
      if (st.chv === "mode") st.chtm = ok ? t : 16; else st.chts = ok ? t : 1;
      st.chk = int("chk", 0, 2 * m.ch.k - 1) ?? 0; st.chi = int("chi", 0, m.n - 1);
      this._depthNotes = (this._depthNotes || []).concat(this.fitCh(st));
    }
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
    const ring = (i, col, rad) => {
      for (let t = 0; t < T; t++) for (let s = 0; s < S; s++) {
        const r = this.cellRect(t, s), base = (t * S + s) * N * 2;
        const x = r.x + r.w * c[base + 2 * i], y = r.y + r.w * (1 - c[base + 2 * i + 1]);
        g.beginPath(); g.arc(x, y, rad, 0, 2 * Math.PI); g.strokeStyle = "#fff"; g.lineWidth = 3; g.stroke();
        g.strokeStyle = col; g.lineWidth = 1.6; g.stroke();
      }
    };
    const sel = this.st.color === "ch" && this.col?.ch ? this.st.chi : this.col?.rg ? this.st.rgi : null;     // change: the selected image stays ringed
    if (sel != null && sel !== this.hover) ring(sel, this.ctx.m.palette.accent, 5);
    if (this.hover >= 0) ring(this.hover, "#16161a", 4.2);
  }
  decorate() {
    const st = this.st, { T, S } = this.ctx, g = this.geom, dep = this.col?.depth;
    // through depth: the whole checkpoint row is outlined; grey panels carry a short note; the images card's site is ringed
    const inSel = dep ? (t => t === dep.t) : null, dsite = dep ? this.depthSite() : -1;
    this.cellbg.forEach((e, i) => {
      const t = Math.floor(i / S), s = i % S, info = dep && t === dep.t ? dep.info[s] : null, low = !!info?.low;
      e.classList.toggle("inb", dep ? inSel(t) : inBlock(st, t, s));
      e.classList.toggle("lowwt", low);
      e.classList.toggle("dsite", !!dep && t === dep.t && s === dsite);
      if (low) e.dataset.note = st.dm === "sets" ? "no labels" : "< 1% of mode"; else delete e.dataset.note;
    });
    this.colh.forEach((e, s) => e.classList.toggle("inb", dep ? true : st.fam === "layer" ? (s >= st.a && s <= st.b) : s === st.anchor));
    this.rowh.forEach((e, t) => e.classList.toggle("inb", dep ? inSel(t) : st.fam === "layer" ? t === st.anchor : (t >= st.a && t <= st.b)));
    let t0, t1, s0, s1;
    if (dep) { t0 = t1 = dep.t; s0 = 0; s1 = S - 1; }
    else if (st.fam === "layer") { t0 = t1 = st.anchor; s0 = st.a; s1 = st.b; } else { s0 = s1 = st.anchor; t0 = st.a; t1 = st.b; }
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
    this.renderSpectrum(null);
    this.renderBlockMap();
  }
  renderBlockMap() {
    const st = this.st, { m, pal } = this.ctx;
    const pairs = st.fam === "layer" ? m.layer_pairs : m.time_pairs, n = st.fam === "layer" ? m.sites.length : m.steps.length;
    const vals = m.vals[dkey(st)][st.anchor][st.mode], kk = st.kind === "psi" ? st.k : st.r - 1;
    $("blk-map-anchor").textContent = st.fam === "layer" ? "checkpoint" : "site";
    const box = $("blk-map"); box.replaceChildren();
    if (kk > m.n_vecs) {   // the map reads meta's spectra (ψ1..ψ8); the add-on is per block
      $("blk-map-note").textContent = `Shipped for eigenvectors 1 to ${m.n_vecs} only; pick one of those to compare blocks.`;
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
    $("blk-map-note").innerHTML = `Every block at this ${st.fam === "layer" ? "checkpoint" : "site"}: row = first ${st.fam === "layer" ? "site" : "checkpoint"}, column = last; colour = eigenvalue ${kk} (darker = closer to 1; range ${lo.toFixed(3)} to 1). Click to select.`;
    $("blk-map-wrap").open = this._mapOpen ?? (window.innerWidth >= 1100);
    $("blk-map-wrap").ontoggle = () => { this._mapOpen = $("blk-map-wrap").open; };
  }

  renderLegend() {
    if (this.col?.depth) return this.renderDepthLegend();
    const st = this.st, { m, pal, labels, N } = this.ctx, col = this.col, body = $("leg-body");
    body.replaceChildren(); $("leg-jac").replaceChildren();
    const titles = { seba: `${st.r} groups of images (coherent sets)`, nodal: `Sign of eigenvector ${st.k}`, eig: `Eigenvector ${st.k}`, km: `k-means on the eigenvectors, ${st.kmk} groups${st.a === st.b ? "" : ` (at the ${st.kms === "end" ? "last" : "first"} site)`}`, class: "CIFAR class (for checking only)",
      dw: `Diffusion ${st.dwk ? "wavelet" : "scaling function"} ${dwName(st, st.dwf)}${st._dw ? ` of ${this.ctx.m.sites[st._dw.site]}, step ${st._dw.step.toLocaleString("en")}` : ""}`,
      rg: "Neighbourhood stretch: " + RG_VIEW[st.rgv],
      ch: col.ch ? `Change from ${m.sites[col.ch.a]} to ${m.sites[col.ch.b]}: ${col.ch.view === "mode" ? `pattern “${chName(m.ch.k, col.ch.k)}”` : "how much each image's neighbourhood changed"}` : "Change to the next site",
      correct: "Correct / wrong", loss: "Loss", jac: "Sensitivity to the input (Jacobian norm)", gcos: "Gradient agrees with the average gradient", gdot: "Gradient alignment (dot product)" };
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
        row.addEventListener("pointerenter", () => { if (st.focus != null) return; const save = st.focus; st.focus = j; this.col = colouring(this.ctx, st, this.blk); this.draw(); st.focus = save; });
        row.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, this.blk); this.draw(); });
        body.append(row);
      }
      if (!g.classes) body.append(h("div", { class: "leg-note", text: g.perRow ? `Counts and class make-up at step ${m.steps[st.at === "final" ? m.steps.length - 1 : t].toLocaleString("en")}. Bars: class make-up (labels used only to validate).` : "Bars: class make-up of each group (class colours as in ‘CIFAR class’; labels only validate)." }));
      if (st.color === "seba") this.renderJaccard();
    } else if (col.cont) {
      const c = col.cont;
      body.append(h("div", { class: "cbar", style: { background: c.kind === "div" ? pal.divGradient() : pal.seqGradient() } }));
      const tick = c.fmt || (v => c.log ? (Math.pow(10, v) >= 100 ? Math.pow(10, v).toFixed(0) : Math.pow(10, v).toPrecision(2)) : (+v.toPrecision(2)).toString());
      body.append(h("div", { class: "cbar-ticks" }, h("span", { text: (c.exact ? "" : "≤ ") + tick(c.lo) }), h("span", { text: c.kind === "div" ? "0" : "" }), h("span", { text: (c.exact ? "" : "≥ ") + tick(c.hi) })));
      const notes = {
        eig: `Diverging colour of ψ${st.k} scaled by max|ψ${st.k}|; grey = near zero. Signs are arbitrary (fixed so the largest entry is positive).`,
        dw: `Diverging colour of one diffusion ${st.dwk ? "wavelet" : "scaling function"} of the outlined snapshot, scaled by its max |value|; grey = near zero. Sign fixed so the largest entry is positive. The same function is painted on every panel.`,
        loss: `Log colour scale, fixed across rows (1st to 99th percentile over all checkpoints). At ${st.at === "final" ? "the final checkpoint for every row" : "each row's own checkpoint"}.`,
        jac: `Frobenius norm of d logits / d input (log scale, fixed across rows). At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}.`,
        gcos: `Cosine between the image's loss gradient (all parameters) and the mean gradient over the 2,000 probe images. At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}. Uses labels through the loss.`,
        gdot: `Dot product of the image's loss gradient with the mean gradient (scale: 99th percentile of |value|). At ${st.at === "final" ? "the final checkpoint" : "each row's own checkpoint"}. Uses labels through the loss.`,
      };
      if (pxk) notes[st.color] = pxDef(pxk) + " Computed from the probe image's pixels, so it is the same in every panel. Labels do not enter it.";
      if (col.rg) {
        const {view, a} = col.rg, q = m.rg.pairs[a].quart;
        notes.rg = `Each column shows ${RG_VIEW[view]} from its site to the next, at the final checkpoint. One scale for all panels; values beyond it clip to the end colours. Logits have no next site.`
          + (view === "shape" ? " Shape change needs no fit; 0 = same shape up to one uniform rescaling." : ` Grey also marks images with no fitted map. Selected pair: ${Number.isFinite(q.fit[1]) && Number.isFinite(q.plane[1]) ? `median fit ${q.fit[1].toFixed(2)}, median plane share ${q.plane[1].toFixed(2)} (how much of the neighbourhood the fitted map covers)` : "no image has a fitted map at this pair"}.`);
      }
      if (col.ch) notes.ch = col.ch.view === "mode"
        ? `The two colours are the two sides of the pattern, scaled so its largest value is ±1; grey = near zero. Which side gets which colour is arbitrary. ${col.ch.k < m.ch.k ? "The second site's walk preserves this contrast better than the first site's: images it links take more similar values on the pattern." : "The second site's walk preserves this contrast less well than the first site's: images it links take less similar values on the pattern."} The same values are painted on every panel.`
        : `Darker = more of the image's ${col.ch.tau === 1 ? "neighbour weight went to different neighbours" : `${col.ch.tau}-step walk ends somewhere else`}; the scale runs from 0 to the largest value at this pair and scale (${pct(col.ch.scMax)}), largest on top. The same values are painted on every panel, so you can see where these images sit in each layer's layout.`;
      body.append(h("div", { class: "leg-note", text: notes[st.color] || "" }));
    }
  }

  renderSourceLegend(body) {
    const S = this.st._S, src = SR.SRC[S.id], a = S.agree, f2 = nz(2), f3 = nz(3);
    body.append(h("div", { class: "leg-note", text: src.def }));
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
    const opSel = [];
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
  // diffusion wavelets: the images card shows the function's ends; the wavelet card its scale table and definition
  renderDwImages(body, strip) {
    const st = this.st, { pal, N } = this.ctx, w = this.col.dw, v = w.v, name = dwName(st, st.dwf);
    $("img-title").textContent = `Images at the two ends of ${name}`;
    const idx = Array.from({ length: N }, (_, i) => i);
    const pos = idx.filter(i => v[i] > 0).sort((a, b) => v[b] - v[a] || a - b), neg = idx.filter(i => v[i] < 0).sort((a, b) => v[a] - v[b] || a - b);
    const row = (ids, label, col, rows, minor) => body.append(h("div", { class: "set-row" + (minor ? " minor" : "") },
      h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: label })), strip(ids, rows)));
    const picked = st.dwi != null && st.dwi !== w.row[2];
    row([w.row[2]], `centre: image #${w.row[2]}${picked ? ` (you picked #${st.dwi}${Math.abs(w.v[st.dwi]) < w.mx / 254 ? ", which no stored function at this scale carries above 8-bit resolution" : ""})` : ""}`, pal.p.diverging.mid, 1, true);
    row(pos, `most positive ${name}`, pal.p.diverging.pos, 2);
    if (neg.length && -v[neg[0]] > .05 * w.mx) row(neg, `most negative ${name}`, pal.p.diverging.neg, 2);
    body.append(h("div", { class: "card-note", text: "Ranked in the browser from the shipped 8-bit function (ties by image index). Hover an image to ring it in every panel." }));
  }
  renderDw() {
    const st = this.st, { m, pal } = this.ctx, W = st._dw, cell = W.cell, w = this.col.dw, r = w.row, L = cell.levels;
    const lev = L.find(x => x[0] === st.dwj), nStored = dwGroup(cell, st.dwj, st.dwk).length;
    $("dw-sub").textContent = `${m.sites[W.site]} · step ${W.step.toLocaleString("en")} · ${m.modes[m.dw.rule]} · cosine kernel`;
    xyChart($("dw-chart"), {
      series: [
        { name: "scaling functions n_j", pts: L.map(x => [x[0], x[1]]), color: pal.set(0), dots: 2.5 },
        { name: "wavelets at scale j", pts: L.map(x => [x[0], x[3]]), color: pal.set(1), dots: 2.5 },
        { name: "#{λ^(2^j) > ε}", pts: L.map((x, a) => [x[0], cell.spec[a]]), color: "#8a8984", dash: "3 3" },
      ],
      ylog: true, xname: "scale j", xticks: L.map(x => x[0]), height: 170, hl: st.dwj,
      onclick: x => { if (dwLevels(cell).includes(x) && x !== st.dwj) { st.dwj = x; if (st.dwi == null) st.dwf = 0; this.update(); } },
    });
    const body = $("dw-body"); body.replaceChildren();
    const kind = st.dwk ? "wavelet" : "scaling function";
    body.append(h("p", { class: "card-note" },
      h("b", { text: `${dwName(st, st.dwf)}` }), `: ${kind} ${st.dwf + 1} of ${nStored}${st.dwk && lev[3] > nStored ? ` stored (of ${lev[3].toLocaleString("en")} at this scale)` : ""} in pivot order, at scale j = ${st.dwj} (the walk run for ${(2 ** st.dwj).toLocaleString("en")} steps). `,
      `Centre: image #${r[2]}, the delta it descends from through the pivots. Participation ratio 1/Σf⁴ = ${r[4].toFixed(1)} images`,
      st.dwk && lev[4] != null ? ` (median over all ${lev[3].toLocaleString("en")} wavelets at this scale: ${lev[4].toFixed(1)}).` : "."));
    body.append(h("p", { class: "card-note", text: `Coifman–Maggioni diffusion wavelets of this snapshot's lazy walk M. Level j holds M^(2^j) on an orthonormal basis Φ_j (Φ_0: one delta per image). A QR with column pivoting of that matrix keeps the columns with |R_kk| > ε = ${m.dw.eps} and orthonormalizes them in pivot order, giving Φ_{j+1}, the functions M^(2^(j+1)) still resolves; the wavelets Ψ_j are an orthonormal basis of what Φ_j has and Φ_{j+1} lacks (again by pivoted QR, which orders them by residual norm in the scaling coordinates; that is not a localization ranking). The wavelets of all scales and the last scaling function (the constant) form an orthonormal basis of the ${m.n.toLocaleString("en")} images. Every step is a QR of a power of M: nothing is fitted. The dashed line counts eigenvalues of M with λ^(2^j) > ε; n_j follows it with some lag, since the QR threshold is on |R_kk|, not on eigenvalues.` }));
    body.append(h("p", { class: "card-note", text: `Stored: the first ${m.dw.cap} wavelets of each scale in pivot order (a subset, not the most localized ones), the scaling functions of scales with at most ${m.dw.cap} of them, 8 bits each. Click the chart to change scale.` }));
  }

  // ---------------------------------------------------------------- change to the next site (SPEC_CHANGE.md)
  // The images card: the most and least changed images (score view) or the two ends of the pattern; a click selects one.
  renderChImages(body, strip) {
    const { m, pal, N } = this.ctx, C = this.col.ch, d = C.d, K = m.ch.k, A = m.sites[C.a], B = m.sites[C.b];
    $("img-sub").textContent = `${A} → ${B} · ${chWords(C.tau)} · step ${m.ch.step.toLocaleString("en")}`;
    const idx = Array.from({ length: N }, (_, i) => i);
    const row = (ids, label, col, rows, minor) => body.append(h("div", { class: "set-row" + (minor ? " minor" : "") },
      h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: label })), strip(ids, rows)));
    if (C.view === "score") {
      const v = C.sc;
      $("img-title").textContent = C.tau === 1 ? "Images whose neighbours changed most and least" : `Images whose ${C.tau}-step neighbourhood changed most and least`;
      row(idx.slice().sort((a, b) => v[b] - v[a] || a - b), `changed most (up to ${pct(C.scMax)})`, pal.seqDot(1), 2);
      row(idx.slice().sort((a, b) => v[a] - v[b] || a - b), "changed least", pal.seqDot(0), 1, true);
    } else {
      const v = C.v, name = chName(K, C.k);
      $("img-title").textContent = `Images on the two sides of the pattern “${name}”`;
      row(idx.filter(i => v[i] > 0).sort((a, b) => v[b] - v[a] || a - b), "one side", pal.p.diverging.pos, 2);
      row(idx.filter(i => v[i] < 0).sort((a, b) => v[a] - v[b] || a - b), "the other side", pal.p.diverging.neg, 2);
      body.append(h("div", { class: "card-note", text: C.k < K ? `The walk at ${B} preserves this contrast better than the walk at ${A}.` : `The walk at ${B} preserves this contrast less well than the walk at ${A}.` }));
    }
    body.append(h("div", { class: "card-note", text: "Click an image to see which neighbours it gained and lost; hover to ring it in every panel. Ranked in the browser from the shipped 8-bit values (ties by image index)." }));
    body.onclick = e => { const c = e.target.closest("canvas[data-i]"); if (c) this.selectChImage(+c.dataset.i); };
  }
  setChPair(a) { const st = this.st; if (!adjacent(st)) return; st.a = a; st.b = a + 1; this.update(); }
  setChK(k) { const st = this.st; if (st.color !== "ch") return; st.chk = k; this.update(); }
  // select (or clear, i = null) the image whose gained and lost neighbours the card lists; no reload
  selectChImage(i) {
    const st = this.st;
    if (st.color === "rg") {
      st.rgi = i != null && i >= 0 && i < this.ctx.N ? i : null;
      this.drawOverlay(); if (this.col?.rg) this.renderRg(); this.writeHash(); return;
    }
    if (st.color !== "ch") return;
    st.chi = i != null && i >= 0 && i < this.ctx.N ? i : null;
    this.drawOverlay(); if (this.col?.ch) this.renderCh(); this.writeHash();
  }
  // a thumbnail that rings its image on hover and selects it on click
  chThumb(i) {
    const c = this.thumb(i);
    c.addEventListener("pointerenter", () => this.setHover(i, this.lensCell[0]));
    c.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, this.thumbnailTip(i)));
    c.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
    c.addEventListener("click", () => { hideTip(); this.selectChImage(i); });
    return c;
  }
  // ---------- neighbourhood stretch to the next site (colouring "rg", SPEC_RIGID.md). Card + images card only; state,
  // painting and legend live elsewhere. this.col.rg = {a, b, view, pair}; pair: N-arrays shape, size, dil, shear, fit,
  // plane, cut, valid, top, bot (top / bot: rigid.separated for the most / least stretched direction), logs (N*4,
  // descending log-stretches), ends (N*24: direction [most, least] x end [one, other] x 6 images, 65535 = empty).
  // Test hooks: every [data-rg-field] / [data-rg-med] holds v.toFixed(2) ("0.00" for "-0.00", "n/a" when not finite); stretches
  // s1..s4 hold the factor exp(log s).toFixed(2); [data-rg-ends="top|bot"] holds canvases [data-i] or one .rg-nosep line;
  // [data-rg-flag="invalid|sensitive"] marks the flag lines. All user-visible words of this card are in this method.
  renderRgImages(body, strip) {
    const { m, pal, N } = this.ctx, C = this.col.rg, P = C.pair, R = m.rg, nm = m.sites;
    const W = {
      shape: ["Most and least shape change of the neighbourhood", "changed shape most", "kept their shape best (up to one uniform rescaling)"],
      dil: ["Largest and smallest dilation of the fitted map", "largest dilation (fitted map)", "smallest dilation (fitted map: contracted or not carried)"],
      shear: ["Most and least shear of the fitted map", "most shear (fitted map)", "least shear (fitted map)"],
    }[C.view];
    const v = P[C.view];
    $("img-title").textContent = W[0];
    $("img-sub").textContent = `${nm[C.a]} → ${nm[C.b]} · step ${R.step.toLocaleString("en")} (final) · cosine`;
    const idx = []; for (let i = 0; i < N; i++) if (Number.isFinite(v[i]) && (C.view === "shape" || P.valid[i])) idx.push(i);
    const row = (ids, label, col, rows, minor) => body.append(h("div", { class: "set-row" + (minor ? " minor" : "") },
      h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: label })), strip(ids, rows)));
    const dil = C.view === "dil";
    row(idx.slice().sort((x, y) => v[y] - v[x] || x - y), W[1], dil ? pal.p.diverging.pos : pal.seqDot(1), 2);
    row(idx.slice().sort((x, y) => v[x] - v[y] || x - y), W[2], dil ? pal.p.diverging.neg : pal.seqDot(0), dil ? 2 : 1, !dil);
    const left = N - idx.length;
    body.append(h("div", { class: "card-note", text: `Click an image for its stretches and the images that span its most and least stretched directions. Hover to ring it in every panel. Ranked in the browser from the shipped 16-bit values; ties by image index.${left ? ` ${left.toLocaleString("en")} images with no fitted map are left out.` : ""}` }));
    body.onclick = e => { const c = e.target.closest("canvas[data-i]"); if (c) this.selectChImage(+c.dataset.i); };
  }
  renderRg() {
    const st = this.st, { m, pal, labels, N } = this.ctx, C = this.col.rg, P = C.pair, R = m.rg, nm = m.sites, A = nm[C.a], B = nm[C.b];
    const MP = R.pairs[C.a], q = MP.quart, refs = MP.refs, nP = R.pairs.length, accent = m.palette.accent;
    const T = R.t ?? 16, WT = R.wt ?? 4, MM = R.m ?? 4, GAP = R.gap ?? .05, TIE = R.tie ?? 1e-6;
    const f2 = v => { if (!Number.isFinite(v)) return "n/a"; const t = v.toFixed(2); return t === "-0.00" ? "0.00" : t; }, pc = v => Number.isFinite(v) ? (100 * v).toFixed(0) + "%" : "n/a";
    const fac = l => Number.isFinite(l) ? "×" + Math.exp(l).toFixed(2) : "n/a";
    const num = (key, v, attr = "data-rg-field") => h("span", { class: "rg-num", [attr]: key, text: f2(v) });
    const neffs = R.pairs.map(p => p.neff), nLo = Math.floor(Math.min(...neffs) / 10) * 10, nHi = Math.ceil(Math.max(...neffs) / 10) * 10;
    const VIEW = {
      shape: { name: "shape change", fitted: false },
      dil: { name: "dilation of the fitted map", fitted: true },
      shear: { name: "shear of the fitted map", fitted: true },
    }, V = VIEW[C.view], med = q[C.view][1];
    // header and lead sentence
    $("rg-title").textContent = `Neighbourhood stretch from ${A} to ${B}`;
    $("rg-sub").textContent = `${T}-step graph coordinates, ${WT}-step neighbourhoods · step ${R.step.toLocaleString("en")} (final) · ${m.modes[R.rule]} · cosine kernel`;
    const fitWords = [` The fitted map is a least-squares linear fit from each neighbourhood's ${MM} widest directions at ${A} to its positions at ${B}. Median fit: the map explains `,
      h("b", { text: pc(q.fit[1]) }), ` of the neighbours' spread at ${B}. Median plane share: `, h("b", { text: pc(q.plane[1]) }), ` of their spread at ${A} lies on those ${MM} directions.`];
    const lead = $("rg-lead");
    if (C.view === "shape") lead.replaceChildren(`Median shape change from ${A} to ${B}: `, h("b", { text: f2(med) }),
      `. Shape change needs no fit; 0 means the neighbourhood kept its shape up to one uniform rescaling. With the images of ${B} relabelled at random the median would be `, h("b", { text: f2(refs.perm.shape) }),
      "; relabelled within their classes, ", h("b", { text: f2(refs.class.shape) }), ".");
    else if (C.view === "dil") lead.replaceChildren(`Median dilation of the fitted map from ${A} to ${B}: `, h("b", { text: f2(med) }),
      `, a geometric-mean stretch of ${fac(med)}. Dilation 0 means the geometric mean of the four stretches is 1, not that nothing changed.`, ...fitWords);
    else lead.replaceChildren(`Median shear of the fitted map from ${A} to ${B}: `, h("b", { text: f2(med) }),
      ". Shear is the spread (standard deviation) of the four log-stretches; 0 means all four directions are stretched by the same factor.", ...fitWords);
    // always visible: where the numbers come from, and what they are specific to
    $("rg-about").replaceChildren(h("b", { text: "Where these numbers come from. " }),
      `The layout only displays the scores; nothing is measured in it. Each image is measured in each site's own graph coordinates (positions built from that site's ${T}-step walk), on its neighbourhood: where its ${WT}-step walk ends, about ${nLo} to ${nHi} images. A direction's two ends are the images that span it at the earlier site, not the images that moved most. One network, one probe set, kernel rule ${R.rule} (${m.modes[R.rule]}), these two scales: values for single images are specific to these choices.`);
    // the pair's medians, with the two references
    const tab = h("table", { class: "mini rg-tab" });
    tab.append(h("thead", {}, h("tr", {}, h("th", { text: "median over images" }), h("th", { text: "this pair" }),
      h("th", { text: "random relabel", title: `The images of ${B} relabelled at random: the neighbourhoods no longer correspond` }),
      h("th", { text: "within classes", title: `The images of ${B} relabelled at random within each class: keeps only what the classes carry` }))));
    const tb = h("tbody");
    const trow = (label, key, v, ref, on, d0) => {
      const cell = (k, x) => h("td", {}, x == null ? "" : d0 ? h("span", { class: "rg-num", "data-rg-med": k, text: Number.isFinite(x) ? x.toFixed(0) : "n/a" }) : num(k, x, "data-rg-med"));
      tb.append(h("tr", { class: on ? "inb" : "" }, h("td", { text: label }), cell(key, v), cell(key + "-perm", ref?.[0]), cell(key + "-class", ref?.[1])));
    };
    trow("shape change", "shape", q.shape[1], [refs.perm.shape, refs.class.shape], C.view === "shape");
    trow("size change (log)", "size", q.size[1], [refs.perm.size, refs.class.size]);
    trow("dilation*", "dil", q.dil[1], [refs.perm.dil, refs.class.dil], C.view === "dil");
    trow("shear*", "shear", q.shear[1], [refs.perm.shear, refs.class.shear], C.view === "shear");
    trow("fit*", "fit", q.fit[1], [refs.perm.fit, refs.class.fit]);
    trow("plane share*", "plane", q.plane[1]);
    trow("effective neighbours", "neff", MP.neff, null, false, true);
    tab.append(tb);
    $("rg-tab").replaceChildren(tab, h("div", { class: "card-note", text: `* Read-out of the fitted map. Shape change needs no fit (0: same shape up to one uniform rescaling). Size change is ½ log of the ratio of the neighbourhood's spread (0: same size; 0.69: twice as wide). Dilation 0 means the four stretches have geometric mean 1, not "no change". Plane share has no reference: it depends on the earlier site only. References: the images of ${B} are relabelled and everything is recomputed; a value near its reference means a relabelling gives a similar median for that read-out; it does not by itself mean the two sites are unrelated (compare shape change and fit).` }));
    // profile over the pairs: median with the middle half of the images; click a point to select that pair
    const key = C.view, Q = R.pairs.map(p => p.quart[key]), xs = [...Q.keys()];
    const fin = pts => pts.filter(p => Number.isFinite(p[1]));      // a pair with no valid fit has no point
    const series = [{ name: "median over images", pts: fin(Q.map((r, p) => [p, r[1]])), color: "#8a8984", dots: 3, width: 1.4 }];
    if (key === "shape") series.push(
      { name: "relabelled at random", pts: R.pairs.map((p, i) => [i, p.refs.perm.shape]), color: "#b5b4ae", dash: true, width: 1.2 },
      { name: "within classes", pts: R.pairs.map((p, i) => [i, p.refs.class.shape]), color: "#6f6e69", dash: true, width: 1.2 });
    series.push({ name: `${A} → ${B} (shown)`, pts: fin([[C.a, med]]), color: accent, dots: 5.5, noline: true });
    const all = series.flatMap(s => s.pts.map(p => p[1])).concat(Q.flatMap(r => [r[0], r[2]])).filter(Number.isFinite);
    let yr;
    if (key === "dil") { const M = Math.max(.1, Math.ceil(Math.max(...all.map(Math.abs)) * 1.1 * 10) / 10); yr = [-M, M]; }
    else yr = [0, Math.max(.1, Math.ceil(Math.max(...all) * 1.08 * 10) / 10)];
    const vName = { shape: "shape change", dil: "dilation", shear: "shear" }[key];
    $("rg-prof-title").textContent = `${V.name[0].toUpperCase() + V.name.slice(1)} at each pair of neighbouring sites`;
    const ch = xyChart($("rg-prof"), {
      series, yrange: yr, xrange: [-.5, nP - .5], xticks: xs.filter(p => p % 2 === 0), xfmt: p => nm[p], hl: C.a,
      height: 160, ml: 44, ytitle: vName, legend: key === "shape", yfmt: v => (+v.toFixed(2)).toString(), title: `${V.name} for each pair of neighbouring sites`,
      tipExtra: p => [{ name: "middle half of images", value: `${f2(Q[p]?.[0])} to ${f2(Q[p]?.[2])}` }, { name: "pair", value: `${nm[p]} → ${nm[p + 1]}` }, ...(p === C.a ? [] : [{ name: "click", value: "show this pair" }])],
      onclick: p => { if (p !== C.a && p >= 0 && p < nP) this.setChPair(p); },
    });
    if (ch?.svg) {     // interquartile band (and the zero line for dilation) under the series
      const { svg, X, Y } = ch, cl = v => Math.max(yr[0], Math.min(yr[1], v)), ok = Q.every(r => Number.isFinite(r[0]) && Number.isFinite(r[2]));
      const anchor = svg.querySelector("g.axis")?.nextSibling || null, ns = "http://www.w3.org/2000/svg";
      if (ok) {
        const poly = document.createElementNS(ns, "polygon");
        poly.setAttribute("points", Q.map((r, p) => `${X(p).toFixed(1)},${Y(cl(r[2])).toFixed(1)}`).concat(Q.map((r, p) => `${X(p).toFixed(1)},${Y(cl(r[0])).toFixed(1)}`).reverse()).join(" "));
        poly.setAttribute("class", "rg-band"); poly.setAttribute("fill", "rgba(138,137,132,.18)");
        svg.insertBefore(poly, anchor);
      }
      if (key === "dil") {
        const z = document.createElementNS(ns, "line");
        Object.entries({ x1: X(-.5), x2: X(nP - .5), y1: Y(0), y2: Y(0), stroke: "#8a8984", "stroke-width": 1, "stroke-dasharray": "2 3" }).forEach(([k, v]) => z.setAttribute(k, v));
        svg.insertBefore(z, anchor);
      }
    }
    $("rg-prof-note").textContent = `Each point is one site and the next, labelled by the first. Grey band: the middle half of the images (25th to 75th percentile).${key === "shape" ? " Dashed: the median after relabelling the later site's images at random, or within classes." : " Dilation and shear describe the fitted map: read them with its fit in the table above."} Click a point to show that pair.`;
    // the selected image
    const box = $("rg-img"); box.replaceChildren();
    if (st.rgi == null) box.append(h("div", { class: "card-note", text: `Click a point in any panel, or an image in the images card, for that image's stretches from ${A} to ${B} and the images that span its most and least stretched directions.` }));
    else {
      const i = st.rgi, ok = !!P.valid[i], cut = P.cut[i], L = Array.from(P.logs.subarray ? P.logs.subarray(i * 4, i * 4 + 4) : P.logs.slice(i * 4, i * 4 + 4));
      const clear = h("button", { type: "button", class: "link-btn", text: "clear", title: "Clear the selected image" });
      clear.addEventListener("click", () => this.selectChImage(null));
      box.append(h("div", { class: "sub-h", text: "Selected image" }), h("div", { class: "ch-sel-head" }, this.chThumb(i),
        h("div", {}, h("div", {}, h("b", { text: `Image #${i}` }), ` · ${m.classes[labels[i]]} `, clear),
          h("div", { class: "ch-sel-words", text: `${A} → ${B}, on its ${WT}-step neighbourhood (about ${Math.round(MP.neff)} images for the median image).` }))));
      const kv = h("div", { class: "rg-kv" });
      const pairKV = (label, k, v, tip, extra) => kv.append(h("span", { class: "rg-k", text: label, title: tip || "" }), h("span", { class: "rg-v" }, num(k, v), extra ? h("span", { class: "rg-x", text: extra }) : ""));
      pairKV("shape change", "shape", P.shape[i], "No fit: 1 minus a local linear CKA. 0 = same shape up to one uniform rescaling", "no fit");
      pairKV("size change", "size", P.size[i], "½ log of the ratio of the neighbourhood's spread", Number.isFinite(P.size[i]) ? `spread ${fac(P.size[i])}` : "");
      box.append(kv);
      if (!ok) box.append(h("div", { class: "ch-caution", "data-rg-flag": "invalid" }, h("b", { text: "No fitted values. " }),
        `The ${MM} widest directions of this neighbourhood at ${A} are not unique (the ${MM}th and ${MM + 1}th are equally wide) or a direction collapsed, so the fitted map is not defined. Shape and size change above need no fit and still hold.`));
      else {
        // the four stretches on a log axis around 1 (fixed range ×¼ to ×4; longer bars clip with an arrow)
        const LIM = Math.log(4), bars = h("div", { class: "rg-bars", role: "img", "aria-label": "the four stretches of the fitted map" });
        bars.append(h("div", { class: "sub-h rg-bars-h", text: `Stretches of the fitted map, ${MM} directions, most stretched first` }));
        L.forEach((l, k) => {
          const c = Math.max(-LIM, Math.min(LIM, l)), x = 50 + 50 * c / LIM, clip = Math.abs(l) > LIM;
          const fill = h("i", { class: "rg-fill" + (clip ? " clip" : "") + (l < 0 ? " neg" : ""), style: { left: Math.min(50, x) + "%", width: Math.abs(x - 50) + "%", background: l >= 0 ? pal.p.diverging.pos : pal.p.diverging.neg } });
          bars.append(h("div", { class: "rg-bar" }, h("span", { class: "rg-bar-k", text: `${k + 1}` }), h("span", { class: "rg-track" }, fill),
            h("span", { class: "rg-bar-v" }, "×", h("span", { class: "rg-num", "data-rg-field": `s${k + 1}`, text: Number.isFinite(l) ? Math.exp(l).toFixed(2) : "n/a" }))));
        });
        bars.append(h("div", { class: "rg-bar rg-axis" }, h("span", { class: "rg-bar-k" }), h("span", { class: "rg-track rg-ticks" },
          ...[["×¼", 0], ["×½", 25], ["×1", 50], ["×2", 75], ["×4", 100]].map(([t, x]) => h("span", { style: { left: x + "%" }, text: t }))), h("span", { class: "rg-bar-v" })));
        box.append(bars);
        const kv2 = h("div", { class: "rg-kv" });
        const add = (label, k, v, tip, extra) => kv2.append(h("span", { class: "rg-k", text: label, title: tip || "" }), h("span", { class: "rg-v" }, num(k, v), extra ? h("span", { class: "rg-x", text: extra }) : ""));
        add("dilation", "dil", P.dil[i], "Mean log-stretch of the fitted map. 0 = geometric mean 1, not no change", Number.isFinite(P.dil[i]) ? `geometric-mean stretch ${fac(P.dil[i])}` : "");
        add("shear", "shear", P.shear[i], "Spread (standard deviation) of the fitted map's log-stretches");
        add("fit", "fit", P.fit[i], `share of the neighbours' spread at ${B} that the fitted map reproduces`, `pair median ${f2(q.fit[1])}`);
        add("plane share", "plane", P.plane[i], `share of the neighbours' spread at ${A} on the ${MM} directions`, `pair median ${f2(q.plane[1])}`);
        add("plane gap", "cut", cut, `log of the ratio of the ${MM}th to the ${MM + 1}th widest direction at ${A}`, cut < GAP ? "rounded; below 0.05 before rounding" : "");
        box.append(kv2);
        if (cut < GAP) box.append(h("div", { class: "ch-caution", "data-rg-flag": "sensitive" }, h("b", { text: "Sensitive to the choice of directions. " }),
          `At ${A} this neighbourhood's ${MM + 1}th widest direction is nearly as wide as its ${MM}th (plane gap below ${GAP}). A slightly different choice of the ${MM} directions could change the stretches, dilation, shear, fit and plane share.`));
      }
      // the images spanning the most and the least stretched direction (unordered ends), when that direction is identified
      const ends = (d, flag, label) => {
        const wrap = h("div", { class: "rg-ends", "data-rg-ends": d ? "bot" : "top" });
        wrap.append(h("div", { class: "imgs-label", text: label }));
        if (flag) {
          for (const [e, name] of [[0, "one end"], [1, "the other end"]]) {
            const g = h("div", { class: "imgs rg-end" }, h("span", { class: "rg-end-k", text: name }));
            for (let k = 0; k < 6; k++) { const j = P.ends[i * 24 + d * 12 + e * 6 + k]; if (j !== 65535 && j < N) g.append(this.chThumb(j)); }
            wrap.append(g);
          }
        } else {
          const why = !ok ? "No fitted map for this image, so no direction is defined."
            : cut < GAP ? (d ? `Not shown: plane gap below ${GAP}, as above.` : `Not shown: plane gap below ${GAP}. At ${A} the ${MM + 1}th widest direction is nearly as wide as the ${MM}th, so no single direction is identified.`)
            : `Not shown: not separated from the ${d ? "second least" : "second most"} stretched direction (stretches ${fac(d ? L[MM - 1] : L[0])} and ${fac(d ? L[MM - 2] : L[1])}; log gap below ${GAP}), so this direction is not identified.`;
          wrap.append(h("div", { class: "rg-nosep", text: why }));
        }
        box.append(wrap);
      };
      ends(0, ok && P.top[i], `Images spanning the most stretched direction at ${A}`);
      ends(1, ok && P.bot[i], `Images spanning the least stretched direction at ${A}`);
      box.append(h("div", { class: "card-note", text: `Ends: the images that span a direction at ${A} (largest share of the neighbourhood's spread along it), not the images that moved most. A direction has no sign, so the two ends are unordered.` }));
    }
    // definitions, with symbols, behind "How this is computed"
    let nInv = 0, nCut = 0, nTop = 0, nBot = 0;
    for (let i = 0; i < N; i++) { if (!P.valid[i]) { nInv++; continue; } if (P.cut[i] < GAP) nCut++; if (!P.top[i]) nTop++; if (!P.bot[i]) nBot++; }
    const rng = R.ranges || {}, rr = v => rng[v] ? `${f2(rng[v][0])} to ${f2(rng[v][1])}` : "a fixed range";
    $("rg-what-body").replaceChildren(
      h("p", { text: `Coordinates. Each site defines a lazy random walk M on the ${N.toLocaleString("en")} probe images (${m.modes[R.rule]}, cosine similarity). M is symmetric with eigenvalues in [0, 1], so M^${T} is a matrix of dot products: an image's coordinates at a site are its row of M^${T / 2}${R.K ? `, kept to the ${R.K} leading non-constant eigenvectors` : ""}. Only these dot products are used; the two sites share no coordinate system and none is needed. The UMAP layout plays no part.` }),
      h("p", { text: `Neighbourhood. Image i's neighbours are weighted by where its ${WT}-step walk at ${A} ends: w_j = M_a^${WT}(i, j); effective number 1 / Σ w_j² (median ${Math.round(MP.neff)} here). Weighted means are removed at each site, and G_a, G_b are the neighbours' centred, weighted Gram matrices at ${A} and ${B}.` }),
      h("p", { text: `Shape change = 1 − ⟨G_a, G_b⟩ / (‖G_a‖ ‖G_b‖), a linear CKA of the two sites localised to the neighbourhood. 0 exactly when the neighbourhood kept its shape up to one uniform rescaling; no fit. Size change = ½ log(tr G_b / tr G_a), the log ratio of the neighbourhood's spread.` }),
      h("p", { text: `Fitted map. Take the ${MM} widest directions of the neighbourhood at ${A} (top ${MM} eigenvectors of its local covariance) and fit, by weighted least squares, the linear map F from the neighbours' positions on them to their positions at ${B}. This is a closed-form fit at a finite scale, not the network's Jacobian. The stretches s_1 ≥ … ≥ s_${MM} are the singular values of F (its rotation part is discarded: the two sites share no coordinates). Dilation = mean_k log s_k, shear = standard deviation of log s_k. Fit = share of the neighbours' spread at ${B} that F reproduces; plane share = share of the spread at ${A} on the ${MM} directions. Stretches of 1 do not make a neighbourhood rigid, and a small stretch means "contracted or not carried"; use shape and size change together to assess rigidity.` }),
      h("p", { text: `Flags. Plane gap = log(ℓ_${MM} / ℓ_${MM + 1}), the ${MM}th and ${MM + 1}th local eigenvalues at ${A}. At or below ${TIE} the plane is not unique and no fitted values are given; under ${GAP} the card marks the fitted values as sensitive. A direction's ends are listed only when the plane gap and its own stretch gap (log(s_1 / s_2), or log(s_${MM - 1} / s_${MM})) are both at least ${GAP}. For this pair: ${nInv} images without fitted values, ${nCut} with a plane gap under ${GAP}, and the most / least stretched direction not identified for ${nTop} / ${nBot} of the others.` }),
      h("p", { text: `Ends. For the most and the least stretched direction, the six images at each end with the largest share w_j g_j² of the neighbourhood's spread along it at ${A}.` }),
      h("p", { text: `Colours. Panel s shows the step from site s to site s + 1; the last panel has no next site and is dimmed. One range for every panel, so steps compare by eye: shape change ${rr("shape")}, dilation ${rr("dil")} (diverging about 0), shear ${rr("shear")}; values beyond the range take the end colour (the legend marks this). Images without fitted values are dimmed in the dilation and shear views. Values are shipped at 16 bits and printed to 2 decimals.` }),
      h("p", { text: `References. Everything is recomputed with the images of ${B} relabelled at random (neighbourhoods no longer correspond) and relabelled within classes (only what the classes carry survives). Class labels are used only for that reference and to name images. Nothing is trained; the only fit is the closed-form least squares above.` }));
  }
  renderCh() {
    const st = this.st, { m, pal, labels } = this.ctx, C = this.col.ch, d = C.d, K = m.ch.k, nm = m.sites, A = nm[C.a], B = nm[C.b];
    const one = C.tau === 1, accent = m.palette.accent;
    $("ch-title").textContent = `Change from ${A} to ${B}`;
    $("ch-sub").textContent = `${chWords(C.tau)} · step ${m.ch.step.toLocaleString("en")} (final) · ${m.modes[m.ch.rule]} · cosine kernel`;
    // the lead sentence: the per-image summary, or how much of the change this pattern carries
    const lead = $("ch-lead");
    if (C.view === "score") lead.replaceChildren("On average ", h("b", { text: pct(C.scMean) }),
      one ? ` of an image's neighbour weight went to different neighbours between ${A} and ${B}` : ` of where ${chA(C.tau)} ${C.tau}-step walk from an image ends up changed between ${A} and ${B}`,
      ` (the most for any image: ${pct(C.scMax)}). Darker points changed more.`);
    else lead.replaceChildren("The pattern ", h("b", { text: `“${chName(K, C.k)}”` }), " carries ", h("b", { text: share(d.share[C.k]) }), " of the change; all 16 stored patterns together carry ",
      h("b", { text: share(d.top_share) }), ". ", C.k < K ? `It is a contrast that sharpens: the walk at ${B} preserves it better than the walk at ${A} (images the walk links take more similar values on it).` : `It is a contrast that fades: the walk at ${B} preserves it less well than the walk at ${A} (images the walk links take less similar values on it).`);
    const cau = $("ch-caution"), low = C.view === "mode" && d.top_share < .25;
    cau.hidden = !low;
    if (low) cau.replaceChildren(h("b", { text: "Read with care. " }), `At ${chSteps(C.tau)} the change is spread over many small patterns (these 16 carry only ${share(d.top_share)} of it), so a single pattern is close to noise. Here the per-image view (“how much each image's neighbourhood changed”) is the one to read, or pick a larger scale.`);
    // the selected image: its score, and the images it gained and lost
    const box = $("ch-img"); box.replaceChildren();
    if (st.chi == null) box.append(h("div", { class: "card-note", text: `Click a point in any panel, or an image in the first card, to see which neighbours that image gained and lost from ${A} to ${B}.` }));
    else {
      const i = st.chi, sc = C.sc[i], mv = D.chMoves(m, C.pair, C.ti, i), f = one ? 2 : 1;     // 1 step: D = (P_b - P_a) / 2, shown on P
      const clear = h("button", { type: "button", class: "link-btn", text: "clear", title: "Clear the selected image" });
      clear.addEventListener("click", () => this.selectChImage(null));
      const words = one ? `${pct(sc)} of its neighbour weight moved to other neighbours from ${A} to ${B}.` : `${pct(sc)} of where ${chA(C.tau)} ${C.tau}-step walk from it ends up changed from ${A} to ${B}.`;
      box.append(h("div", { class: "ch-sel-head" }, this.chThumb(i),
        h("div", {}, h("div", {}, h("b", { text: `Image #${i}` }), ` · ${m.classes[labels[i]]} `, clear), h("div", { class: "ch-sel-words", text: words }),
          C.v ? h("div", { class: "card-note", text: `Its value in the pattern shown: ${fmt(C.v[i] / (C.mx || 1))} (scaled to ±1).` }) : null)));
      const moves = (list, label) => {
        const g = h("div", { class: "imgs ch-moves" });
        if (!list.length) g.append(h("span", { class: "ch-none", text: "none" }));
        list.forEach(({ j, v }) => {
          g.append(h("div", { class: "ch-move" }, this.chThumb(j), h("span", { text: chChg(f * v) })));
        });
        box.append(h("div", { class: "imgs-label", text: label }), g);
      };
      moves(mv.gained, one ? "Gained: images it now gives more neighbour weight to (change in its neighbour weight, percentage points)" : `Gained: images its ${C.tau}-step walk now reaches more often (change in probability, percentage points)`);
      moves(mv.lost, one ? "Lost: images it now gives less neighbour weight to" : `Lost: images its ${C.tau}-step walk now reaches less often`);
    }
    // the size of the change for every adjacent pair at this scale; click a point to show that pair
    $("ch-prof-title").textContent = `Size of the change between each site and the next (${chSteps(C.tau)})`;
    const s = m.ch.s[C.ti];
    xyChart($("ch-prof"), {
      series: [{ name: "size of the change", pts: s.map((v, p) => [p, v]), color: "#8a8984", dots: 3, width: 1.4 },
        { name: `${A} → ${B} (shown)`, pts: [[C.a, s[C.a]]], color: accent, dots: 5.5, noline: true }],
      yrange: [0, 1], xrange: [-.5, s.length - .5], xticks: [...s.keys()].filter(p => p % 2 === 0), xfmt: p => nm[p], hl: C.a,
      height: 150, ml: 44, ytitle: "size of change", legend: false, yfmt: v => v.toFixed(1), title: "size of the change for each pair of neighbouring sites",
      tipExtra: p => [{ name: "pair", value: `${nm[p]} → ${nm[p + 1]}` }, ...(p === C.a ? [] : [{ name: "click", value: "show this pair" }])],
      onclick: p => { if (p !== C.a && p >= 0 && p < s.length) this.setChPair(p); },
    });
    $("ch-prof-note").textContent = `Each point is one site and the next (labelled by the first). 0 = the two walks are the same; about 1 = as different as two unrelated arrangements of the images. Click a point to show that pair.`;
    // pattern view: the 16 stored patterns by strength, up = sharpen, down = fade; click a bar to show one
    $("ch-modes-wrap").hidden = C.view !== "mode";
    if (C.view === "mode") {
      signedBars($("ch-modes"), {
        y: Array.from(d.lam), color: i => i === C.k ? accent : "#a3a29c", sel: C.k, groups: [[0, K - 1, "sharpen"], [K, 2 * K - 1, "fade"]],
        name: i => chName(K, i), yfmt: v => (+v.toFixed(2)).toString(), label: "strength of each pattern",
        tip: i => [{ name: "share of the change", value: share(d.share[i]) }, { name: "strength (eigenvalue λ)", value: d.lam[i].toFixed(4) }, ...(i === C.k ? [] : [{ name: "click", value: "show this pattern" }])],
        onclick: i => { if (i !== C.k) this.setChK(i); },
      });
      $("ch-modes-note").textContent = `Up: contrasts that sharpen from ${A} to ${B}; down: contrasts that fade. Taller = stronger. The highlighted bar is shown on the grid; click another to show it.`;
    }
    // definitions, with symbols, behind "What is this?"
    $("ch-what-body").replaceChildren(
      h("p", { text: `Each site defines a random walk on the ${m.n.toLocaleString("en")} probe images: from an image, step to one of its nearest neighbours in that layer's activations (${m.modes[m.ch.rule]}, cosine similarity; the walk M is lazy: it stays put half the time). Run the walk τ steps at both sites and subtract: D = M_b^τ − M_a^τ, here with a = ${A}, b = ${B}, τ = ${C.tau}.` }),
      h("p", { text: `Per image (the colour in the first view): tv_i = ½ Σ_j |D_ij|, the share of image i's τ-step distribution that moved (total-variation distance). At 1 step the walk stays put half the time, so tv_i is at most 50%; the page shows 2 tv_i, the same distance on the walk without its lazy half (P = 2M − I): the share of the image's neighbour weight that went to other neighbours. Gained and lost at 1 step are on P too (2 D_ij); at more steps they are changes in the τ-step probability D_ij. Up to 6 of each sign, largest first; the image itself is never listed.` }),
      h("p", { text: `Size of the change: s = ‖D‖²_F divided by its average when the images of ${B} are relabelled at random (closed form). Here s = ${s[C.a].toFixed(3)}.` }),
      h("p", { text: `Patterns: eigenvectors v of D, 16 stored per pair and scale (the 8 largest positive eigenvalues, “sharpens”: fᵀM_b^τ f > fᵀM_a^τ f, a contrast the walk at ${B} preserves better than the walk at ${A}, whether or not the two sides are separate at either site; and the 8 most negative, “fades”). A pattern's share is λ²/‖D‖²_F; here all 16 carry ${share(d.top_share)}, and after a random relabelling of the images the top 16 would carry ${share(d.perm_top_share)}, so a small number of patterns carrying most of the change is not by itself evidence of structure. Measured on this network: at 1 and 4 steps the 16 patterns carry only about 4–15% of the change and single patterns look like noise; at 16 steps they carry 55–90% and line up with the image classes (labels are used only to check).` }),
      h("p", { text: "Nothing is fitted: everything here is an eigendecomposition or an entrywise statistic of D. Patterns are stored at 8 bits, scores at 8 bits of the largest score, gained and lost at 16 bits." }));
  }
  renderImages() {
    const st = this.st, { m, pal, labels, N } = this.ctx, blk = this.blk, body = $("img-body"), tg = this.target();
    body.replaceChildren();
    const W = body.clientWidth || 300, TH = W >= 560 ? 44 : 40, per = Math.max(5, Math.floor((W + 3) / (TH + 3)));
    const strip = (ids, rows = 1) => {
      const g = h("div", { class: "imgs imgs-strip", style: { gridTemplateColumns: `repeat(${per}, minmax(0, 1fr))` } });
      ids.slice(0, per * rows).forEach(i => {
        const c = this.thumb(i);
        c.addEventListener("pointerenter", () => this.setHover(i, this.lensCell[0]));
        c.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, this.thumbnailTip(i)));
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
    body.onclick = null;
    if (this.col?.dw) return this.renderDwImages(body, strip);
    if (this.col?.rg) return this.renderRgImages(body, strip);
    if (this.col?.ch) return this.renderChImages(body, strip);
    if (tg.kind === "psi") {
      $("img-title").textContent = `Images at the two ends of eigenvector ${tg.k}`;
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
    $("img-title").textContent = `${r} groups of images (coherent sets) and their class make-up`;
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
        head.addEventListener("pointerenter", () => { if (st.focus != null) return; st.focus = j; this.col = colouring(this.ctx, st, blk); this.draw(); st.focus = null; });
        head.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, blk); this.draw(); });
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
    $("img-title").textContent = `Sets from: ${src.label}${S.classes ? "" : `, r = ${st.r}`}`;
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
      head.addEventListener("pointerenter", () => { if (st.focus != null) return; st.focus = j; this.col = colouring(this.ctx, st, this.blk); this.draw(); st.focus = null; });
      head.addEventListener("pointerleave", () => { if (st.focus != null) return; this.col = colouring(this.ctx, st, this.blk); this.draw(); });
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
    box.append(h("div", { class: "card-note", text: `Permutation: the source's labels shuffled ${a.nperm.toLocaleString("en")} times, set sizes kept${a.rep ? "; computed by the site build for this stored null draw (no scanned-blocks reference: one block)" : S.id === "Y2" ? "; computed by the site build" : ""}. The 95th percentile is for this block alone; when you compare many blocks, use the scanned-blocks reference. Kernel rule ${this.ctx.m.modes[this.srcIdx.kernel_rule]}.` }));
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
    $("card-coarse").hidden = !on;
    if (!on) return;
    const tok = this._cgTok = (this._cgTok || 0) + 1;
    let U;
    try { U = await D.coarseUnit(m.steps[st.anchor], st.mode); }
    catch (e) { console.error(e); $("cg-note").textContent = "The coarse-graining summary failed to load."; return; }
    if (tok !== this._cgTok || this.ctx !== ctx) return;
    const rec = U.blocks[pairOf(ctx, st)];
    this.renderSpectrum(rec);
    this.renderCoarseCard(rec);
  }
  // Eigenvalue chart of the block average: lambda_1..lambda_8 from the block, or up to lambda_32 with the largest drop when
  // the coarse-graining summary for this block is loaded (rec). Linear scale; dark = the eigenvectors behind the current groups.
  renderSpectrum(rec) {
    const st = this.st, ctx = this.ctx, m = ctx.m, blk = this.blk, N = ctx.N, cell = st.a === st.b;
    const vals = rec ? (cell ? rec.sv : rec.avg) : Array.from(blk.vals), gap = rec ? (cell ? rec.gap : rec.avg_gap) : null;
    const K = maxK(ctx, st), n = Math.min(rec ? 32 : m.n_vecs, vals.length - 1);
    const pn = k => k <= m.n_vecs ? blk.pn[k - 1] : null, LOC = Math.round(0.01 * N);
    const loc = k => pn(k) != null && pn(k) < LOC;
    const sel = st.kind === "psi" ? st.k : null, used = k => st.kind !== "psi" && k < st.r;
    const pts = f => [...Array(n)].map((_, i) => i + 1).filter(f).map(k => [k, vals[k]]);
    const series = [
      { name: "eigenvector (click to colour the grid by it)", pts: pts(k => k <= K && k !== sel && !used(k)), color: "#9a9993", dots: 3, noline: true },
      { name: `used to build the current ${st.r} groups`, pts: pts(k => used(k) && k !== sel), color: "#16161a", dots: 3.6, noline: true },
      { name: "value only (vector not shipped)", pts: pts(k => k > K), color: "#d6d5cf", dots: 2.4, noline: true },
      { name: "concentrated on a few images", pts: pts(loc), color: m.palette.accent, marker: "diamond", hollow: true },
    ];
    if (sel && sel <= n) series.push({ name: `eigenvector ${sel} (shown)`, pts: [[sel, vals[sel]]], color: m.palette.accent, dots: 5, noline: true });
    const top = vals.slice(1, n + 1).filter(Number.isFinite), lo = Math.max(0, Math.min(.9, Math.floor((Math.min(...top) - .02) * 10) / 10));
    const xt = n <= 8 ? [...Array(n)].map((_, i) => i + 1) : [1, 4, 8, 12, 16, 20, 24, 28, 32].filter(v => v <= n);
    xyChart($("blk-spec"), {
      series: series.filter(x => x.pts.length), yrange: [lo, 1], height: 160, ml: 50, xrange: [.5, n + .5], xticks: xt,
      xname: "eigenvector number", ytitle: "eigenvalue", legend: false,
      vlines: gap && gap.clear ? [{ x: gap.k - .5, color: "#16161a", top: true, label: `big drop → ${gap.k} groups` }] : [],
      yfmt: v => v.toFixed(2),
      tipExtra: k => [{ name: "average change across the block (1 − eigenvalue)", value: (1 - vals[k]).toFixed(4) },
        ...(pn(k) != null ? [{ name: "spread over about", value: `${Math.round(pn(k))} of ${N} images` + (loc(k) ? " (concentrated)" : "") }] : []),
        ...(k > K ? [{ name: "vector", value: "not shipped, value only" }] : [])],
      onclick: k => { if (k > K) return; st.kind = "psi"; st.k = k; if (st.color === "seba" || st.color === "km") st.color = "eig"; st.focus = null; this.update(); },
    });
    $("blk-spec-title").textContent = "Eigenvalues: which patterns the block keeps";
    const nLoc = [...Array(n)].map((_, i) => i + 1).filter(loc).length;
    $("blk-spec-note").innerHTML = `Each dot is one eigenvector of the block's averaged random walk between similar images (M̄, the mean over the block's ${st.b - st.a + 1} snapshot${cell ? "" : "s"}), in order. `
      + `Its height, the eigenvalue (at most 1), says how well that pattern of values over the images survives a step of the walk: near 1 means a smooth split the network keeps throughout the block; lower means finer detail or noise. `
      + (gap ? (gap.clear ? `The largest relative drop comes after eigenvector ${gap.k - 1}, which suggests ${gap.k} natural groups. ` : `There is no clear drop, so no natural number of groups stands out. `) : "")
      + (st.kind === "psi" ? "" : `The ${st.r} groups are built from the dark dots. `)
      + (nLoc ? `<span style="color:${m.palette.accent};font-weight:600">${nLoc} eigenvector${nLoc > 1 ? "s are" : " is"} concentrated on a few images</span> (diamond): a handful of outliers, not a grouping. ` : "")
      + "Click a dot to colour the grid by that eigenvector.";
  }
  renderCoarseCard(rec) {
    const st = this.st, ctx = this.ctx, m = ctx.m, pal = ctx.pal, cell = st.a === st.b, C = m.coarse, K = C.K;
    const sides = cell ? [["start", "group the start image", "Sort the image a walk step starts from into k groups; how much does knowing its group tell you about where the step lands? I(c(X); X′)"], ["two", "group both ends", "Sort both ends of one walk step with the same k groups. I(c(X); c(X′))"]]
      : [["start", "group at first site", "Sort the images at the first site into k groups; how much does that tell you about the image at the last site? I(c(X); Y)"], ["end", "group at last site", "Sort the images at the last site into k groups. I(X; c′(Y))"], ["two", "group at both", "Sort both ends into k groups each. I(c(X); c′(Y))"]];
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
      { name: "best possible", pts: K.map(k => [k, eff ? 1 : F(k)]), color: "#16161a", dash: true, width: 1.4 },
      { name: "k-means on eigenvectors", pts: ptsOf("spectral"), color: S[0], dots: 2.2 },
      { name: "coherent sets", pts: ptsOf("seba"), color: S[1], dots: 2.2 },
      { name: "k-means on activations (PCA)", pts: ptsOf("pca"), color: S[2], dots: 2.2 },
      { name: "random groups", pts: ptsOf("random"), color: "#9a9993", dots: 2, width: 1.4 },
      { name: "predicted class", pts: ptsOf("pred"), color: m.palette.accent, marker: "triangle" },
      { name: "true class", pts: ptsOf("labels"), color: "#16161a", marker: "diamond", hollow: true },
    ];
    const km = kmOn(ctx, st), hl = st.color === "km" ? st.kmk : null;
    xyChart($("cg-chart"), {
      series, ylog: !eff, yrange: eff ? [0, 1.02] : null, height: 190, ml: 52, xrange: [1.5, 32.5], xticks: [2, 4, 8, 12, 16, 20, 24, 32], xname: "number of groups k", ytitle: eff ? "share of best" : "information kept", hl,
      vlines: rec.gap.clear ? [{ x: rec.gap.k, color: "#16161a", label: "big eigenvalue drop" }] : [],
      yfmt: eff ? (v => v.toFixed(1)) : undefined,
      tipExtra: k => [{ name: "best possible", value: (+F(k).toPrecision(4)).toString() }, ...(km && C.ks.includes(k) ? [{ name: "click", value: `colour the grid by k-means, ${k} groups` }] : [])],
      onclick: km ? (k => { if (!C.ks.includes(k)) return; st.color = "km"; st.kmk = k; if (!cell) st.kms = side === "end" ? "end" : "start"; st.focus = null; this.update(); }) : null,
    });
    const what = cell ? `one step of the walk at ${m.sites[st.a]}` : `the passage from ${m.sites[st.a]} to ${m.sites[st.b]} (the walks of the block's sites applied in order)`;
    const g = rec.gap;
    $("cg-sub").innerHTML = `${m.modes[st.mode]} · total information ${(+rec.total.toPrecision(4))} · ${g.clear ? `big eigenvalue drop at ${g.k} groups` : "no clear eigenvalue drop"}`;
    $("cg-note").innerHTML = `Sort the images into k groups and ask how much the group alone tells you about ${what}; the y axis is that information (χ² mutual information). `
      + `<b>Best possible</b> (dashed) is the most any k groups could keep, from the top singular values${side === "two" ? " (at the smaller of the two group counts)" : ""}. `
      + `Each method is plotted at the number of groups it actually uses. <b>Random groups</b> (mean of 4 draws) show what chance keeps; <b>true class</b> and <b>predicted class</b> are for checking only (10 groups, or fewer if some class is never predicted). `
      + (eff ? "The share of best possible is only a tightness check: a walk with no structure at all gives 1 for every grouping, so read it next to the amounts. " : "")
      + (km ? "Click a k from 2 to 16 to colour the grid by k-means with that many groups." : "k-means colouring is shipped for this block only under the default kernel rule.");
    // single cells: leakage against the Ky Fan (Ncut) and higher-order Cheeger (max leakage) bounds
    $("cg-cell").hidden = !cell;
    if (!cell) return;
    const KF = k => { let s = 0; for (let i = 0; i < k && i < sv.length; i++) s += 1 - sv[i]; return s; }, CH = k => (1 - sv[k - 1]) / 2;
    const ratio = (mt, q) => K.map((_, qi) => { const x = rec.m[mt].keff_start[qi], v = rec.m[mt][q][qi]; return x == null || v == null ? null : [x, v / (q === "ncut" ? KF(x) : CH(x))]; }).filter(Boolean).sort((p, r) => p[0] - r[0]);
    const cs = [["spectral", "k-means on eigenvectors", S[0]], ["seba", "coherent sets", S[1]], ["pca", "k-means on activations", S[2]], ["random", "random groups", "#9a9993"]];
    xyChart($("cg-cell-chart"), {
      series: [...cs.map(([mt, nm, c]) => ({ name: `${nm}: total leak / best possible`, pts: ratio(mt, "ncut"), color: c, width: 1.6 })),
        ...cs.map(([mt, nm, c]) => ({ name: `${nm}: leakiest group / best possible`, pts: ratio(mt, "maxleak"), color: c, dash: true, width: 1.2 })),
        { name: "true class: total leak / best possible", pts: ratio("labels", "ncut"), color: "#16161a", marker: "diamond", hollow: true }],
      ylog: true, height: 140, ml: 52, xrange: [1.5, 32.5], xticks: [2, 4, 8, 12, 16, 20, 24, 32], xname: "number of groups k", ytitle: "times the minimum", legend: false, hl,
      yfmt: v => (+v.toPrecision(2)).toString(),
    });
    $("cg-cell-note").innerHTML = `A group <b>leaks</b> when a walk step starting inside it lands outside it; the leak of group A is the chance of that, φ(A). Solid lines: the leaks summed over all groups; dashed: the leakiest group. Each is divided by the smallest value the eigenvalues allow (the Ky Fan bound Σ<sub>i&lt;k</sub>(1 − λ<sub>i</sub>) for the sum, the higher-order Cheeger bound (1 − λ<sub>k−1</sub>)/2 for the leakiest group), so 1 means as tight as possible and larger means leakier. Colours as above; hollow diamonds: the true classes (sum).`;
  }

  // ---------------------------------------------------------------- through depth (SPEC_SPACETIME.md sections 4 and 6)
  // the site whose images the first card shows: chosen (st.dsite) or the one holding most of the mode (sets: of the space)
  depthSite() {
    const st = this.st;
    if (st.dsite != null) return st.dsite;
    const { R, c, w } = depthView(this.ctx, st), y = w || R.occupancy[c];
    return y.reduce((b, v, i) => v > y[b] ? i : b, 0);
  }
  // true while the depth data on screen belong to other settings than st (a load is pending): depth redraws, site
  // picks, legend clicks and tooltips wait for the commit in update()
  depthPending() {
    const st = this.st, d = st._dep, c = this.col?.depth;
    return !d || !c || ["anchor", "dv", "dt", "dm", "dk", "dr"].some(k => d.set[k] !== st[k]);
  }
  setDepthSite(s) {
    if (!depthReady(this.st) || this.depthPending()) return;
    this.st.dsite = s;
    this.decorate(); this.renderDepthImages(); this.renderDepthCard(); this.renderLegend(); this.writeHash();
  }
  depthSetting() {
    const st = this.st, m = this.ctx.m, S_ = m.spacetime, V = S_.variants.find(v => v.id === st.dv), R = st._dep.Sm.variants[st.dv];
    const coupling = st.dv === "paper" ? "no adjustable coupling (the paper's matrix as is)" : `t = ${(+R.t[st.dt].toPrecision(3))} (t/t* = ${fracLabel(S_.factors[st.dt])})`;
    return { V, coupling, step: m.steps[st.anchor].toLocaleString("en") };
  }
  renderDepthCard() {
    const st = this.st, ctx = this.ctx, { m, pal, S } = ctx, S_ = m.spacetime, { R, c, k, w } = depthView(ctx, st);
    const names = m.sites, info = this.col.depth.info, dsite = this.depthSite(), pick = s => this.setDepthSite(s), { V, coupling, step } = this.depthSetting();
    $("dp-sub").textContent = `${V.label} · ${coupling} · step ${step} · ${m.modes[S_.rule]}`;
    const siteFmt = x => Number.isInteger(x) ? names[x] : `${names[Math.floor(x)]} → ${names[Math.ceil(x)]}`;
    const xt = [...Array(S).keys()].filter(i => (S - 1 - i) % 2 === 0);     // every other site, ending at the last
    const lo01 = arr => Math.max(-1, Math.min(.8, Math.floor((Math.min(...arr.filter(Number.isFinite), 1) - .02) * 10) / 10));   // as the retention card
    if (k != null) {
      $("dp-title").textContent = `Mode ${st.dm} through depth`;
      $("dp-a-title").textContent = "Depth weight: the share of the mode at each site" + (st.dv === "paper" ? " (B-weighted, b_l |f_l|²)" : "");
      const sw = st.dm === 1 && R.scalar_weight ? R.scalar_weight[c] : null;
      barChart($("dp-a"), {
        names, y: w, color: pal.ord(.7), low: info.map(x => x.low), sel: dsite, onclick: pick, yfmt: v => (100 * v).toFixed(0) + "%",
        barName: `share of mode ${st.dm}`, overlay: sw ? { name: "what the gaps alone would give (scalar control)", y: sw } : null,
        tip: s => [{ name: `share of mode ${st.dm}`, value: share(w[s]) }, ...(sw ? [{ name: "scalar control", value: share(sw[s]) }] : []),
          ...(info[s].low ? [{ name: "slice", value: "under 1%: drawn grey" }] : []), { name: "click", value: "show this site's images" }],
      });
      $("dp-b-title").textContent = "Each slice at its own site and against the next site's";
      const ok = s => !info[s].low, at = (arr, x) => arr.map((v, l) => [l + x, ok(l) && (x === 0 || ok(l + 1)) ? v : null]);   // null: omitted, the line breaks
      const series = [
        { name: "cosine with the next site's slice", pts: at(R.adj_cos[c][k], .5), color: pal.set(0), dots: 2.4, width: 1.6, breaks: true },
        { name: "retention at its own site", pts: at(R.retention[c][k], 0), color: pal.set(1), dots: 2.4, width: 1.6, breaks: true },
        { name: "share in the site's own first 8 directions", pts: at(R.own_share[c][k], 0), color: pal.set(2), dots: 2.4, width: 1.6, breaks: true },
      ];
      const lo = lo01(series.flatMap(x => x.pts.map(p => p[1])));
      if (series.some(x => x.pts.some(p => Number.isFinite(p[1])))) xyChart($("dp-b"), { series, yrange: [lo, 1.0], xrange: [-.4, S - .6], xticks: xt, xfmt: siteFmt, height: 165, ml: 36, xname: "site", yfmt: v => v.toFixed(1) });
      else $("dp-b").replaceChildren(h("div", { class: "card-note", text: "Every slice holds under 1% of the mode." }));
      $("dp-note").innerHTML = `<b>Mode numbers are local to this setting</b>: they are not matched across the coupling slider or across checkpoints, so mode ${st.dm} here need not be the same function as mode ${st.dm} at another coupling or step. `
        + (st.dv === "paper" ? "" : `<b>Mode 1's depth weight mostly follows the sites' gaps</b>: for mode 1 the dashed line is the profile the gaps alone would give if every site's cheapest direction were one shared image function. `)
        + `Retention: u<sub>l</sub><sup>T</sup>M<sub>l</sub>u<sub>l</sub>, how much of the slice survives one step of its own site's walk. Own share: the part of u<sub>l</sub> inside the span of that site's own first 8 eigenvectors. Sites holding under 1% of the mode (grey bars) are left out of the lower chart. Click a bar for that site's images.`;
    } else {
      const p = depthPlane(m, st), Q = R.sets[p], lab = `k = ${st.dk}, ${S_.readouts[st.dr]}`;
      $("dp-title").textContent = `Sets through depth (${lab})`;
      $("dp-a-title").textContent = "Number of sets present at each site";
      const g = Q.groups[c];
      barChart($("dp-a"), {
        names, y: g, color: pal.ord(.7), low: info.map(x => x.low), sel: dsite, onclick: pick, yfmt: v => String(Math.round(v)), yhi: st.dk, barName: "non-empty sets",
        tip: s => [{ name: "non-empty sets", value: info[s].low ? "none (no labels here)" : String(g[s]) }, { name: "click", value: "show this site's sets" }],
      });
      $("dp-b-title").textContent = "Between neighbouring sites";
      const at = arr => (arr || []).map((v, l) => [l + .5, Number.isFinite(v) ? v : null]);
      const series = [{ name: "share of images changing set", pts: at(Q.churn[c]), color: pal.set(0), dots: 2.4, width: 1.6, breaks: true }];
      if (st.dr === "a") series.push({ name: "alignment strength (σmin / σmax of the rotation's cross matrix)", pts: at(R.link[c]), color: "#16161a", dash: true, dots: 2, width: 1.2, breaks: true });
      if (series.some(x => x.pts.some(p => Number.isFinite(p[1])))) xyChart($("dp-b"), { series, yrange: [0, 1], xrange: [-.4, S - .6], xticks: xt, xfmt: siteFmt, height: 150, ml: 36, xname: "site", yfmt: v => v.toFixed(1) });
      else $("dp-b").replaceChildren(h("div", { class: "card-note", text: "No two neighbouring sites both carry labels." }));
      $("dp-note").innerHTML = `<b>A post-hoc readout, not part of the construction</b> (k-means is a fit): one k-means with centroids shared across the ${S} sites on the rows of the first 8 modes' slices, each site's block scaled to unit size`
        + (st.dr === "a" ? `, every site first rotated onto the previous one (orthogonal Procrustes; an alignment strength near 0 means that link's rotation is poorly determined)` : ", rows as they are")
        + `. Sites holding almost none of the space get no labels (grey). Set numbers go by size and, like mode numbers, are local to this setting: they are not matched across the coupling slider or checkpoints. Click a set in the legend to isolate it in every panel; click a bar for that site's sets.`;
    }
    $("dp-occ-title").textContent = `Where the low space lives: occupancy per site${st.dv === "paper" ? ", B-weighted" : ""} (the ${S} values sum to ${S_.K})`;
    barChart($("dp-occ"), { names, y: R.occupancy[c], color: "#9a9993", sel: dsite, onclick: pick, yfmt: v => (+v.toFixed(1)).toString(), barName: "occupancy", height: 110,
      tip: s => [{ name: `occupancy (of ${S_.K})`, value: R.occupancy[c][s].toFixed(2) }] });
    const vals = R.vals[c], cur = k != null ? k + 1 : null;
    xyChart($("dp-vals"), {
      series: [{ name: "mode", pts: vals.map((v, i) => [i + 1, v]).filter(p => p[0] !== cur), color: "#9a9993", dots: 3, noline: true },
        ...(cur ? [{ name: `mode ${cur} (shown)`, pts: [[cur, vals[cur - 1]]], color: m.palette.accent, dots: 5, noline: true }] : [])],
      legend: false, height: 130, ml: 50, xrange: [.5, S_.K + .5], xticks: [1, 4, 8, 12, 16].filter(x => x <= S_.K), xname: "mode",
      ytitle: st.dv === "paper" ? "eigenvalue of B⁻¹A" : "eigenvalue of L_t", yfmt: v => (+v.toPrecision(3)).toString(),
      tipExtra: () => [{ name: "click", value: "colour the row by this mode" }],
      onclick: x => { st.dm = x; st.focus = null; st.dsite = null; this.update(); },
    });
  }
  renderDepthImages() {
    const st = this.st, ctx = this.ctx, { m, pal, labels, N } = ctx, body = $("img-body"), s = this.depthSite(), info = this.col.depth.info[s];
    body.replaceChildren(); body.onclick = null;
    const W = body.clientWidth || 300, TH = W >= 560 ? 44 : 40, per = Math.max(5, Math.floor((W + 3) / (TH + 3)));
    const strip = (ids, rows = 1) => {
      const g = h("div", { class: "imgs imgs-strip", style: { gridTemplateColumns: `repeat(${per}, minmax(0, 1fr))` } });
      ids.slice(0, per * rows).forEach(i => {
        const c = this.thumb(i);
        c.addEventListener("pointerenter", () => this.setHover(i, st.anchor));
        c.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, this.pointTip(i, st.anchor, s)));
        c.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
        g.append(c);
      });
      return g;
    };
    const sel = h("select", { class: "dp-site", "aria-label": "Site whose images are shown" });
    m.sites.forEach((nm, j) => sel.add(new Option(nm, j))); sel.value = String(s);
    sel.addEventListener("change", () => this.setDepthSite(+sel.value));
    const { V, step } = this.depthSetting();
    $("img-sub").replaceChildren(`Step ${step} · ${V.label} · at site`, sel);
    if (st.dm !== "sets") {
      $("img-title").textContent = `Images at the two ends of mode ${st.dm}, at ${m.sites[s]}`;
      if (info.low) { body.append(h("div", { class: "card-note", text: `The slice at ${m.sites[s]} holds ${share(info.w)} of the mode: under 1%, so it is a normalised numerical tail, not a direction, and is drawn grey. Pick a site with a taller bar in the depth-weight chart.` })); return; }
      const v = depthSlice(ctx, st, s), idx = Array.from({ length: N }, (_, i) => i), P = 2 * per;
      const ends = [[1, `most negative at ${m.sites[s]}`, pal.p.diverging.neg, (a, b) => v[a] - v[b]], [0, `most positive at ${m.sites[s]}`, pal.p.diverging.pos, (a, b) => v[b] - v[a]],
        [2, "nearest zero", pal.p.diverging.mid, (a, b) => Math.abs(v[a]) - Math.abs(v[b])]];
      for (const [b, name, col, cmp] of ends)
        body.append(h("div", { class: "set-row" + (b === 2 ? " minor" : "") }, h("div", { class: "set-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: name })),
          strip(idx.slice().sort((x, y) => cmp(x, y) || x - y).slice(0, P), b === 2 ? 1 : 2)));
      body.append(h("div", { class: "card-note", text: `Images with the largest and smallest entries of this site's slice of mode ${st.dm}, ranked in the browser from the 8-bit slice (ties by image index). The sign is one choice for the whole mode. Click a panel in the coloured row, or a bar, to change the site; hover an image to ring it in every panel.` }));
      return;
    }
    const k = st.dk, lab = depthLabels(ctx, st, s), cols = this.col.cols;
    $("img-title").textContent = `Sets at ${m.sites[s]} and their class make-up`;
    if (info.low) { body.append(h("div", { class: "card-note", text: `No labels at ${m.sites[s]}: this site holds almost none of the low space. Pick a site with a taller occupancy bar.` })); return; }
    const cnt = [...Array(k + 1)].map(() => new Array(10).fill(0)), size = new Array(k + 1).fill(0);
    for (let i = 0; i < N; i++) { const g = lab[i] < 0 ? k : lab[i]; size[g]++; cnt[g][labels[i]]++; }
    const compBar = (cn, total) => {
      const bar = h("span", { class: "comp comp-lg" });
      cn.forEach((c, q) => { if (c) bar.append(h("i", { style: { width: (100 * c / total) + "%", background: pal.classes[q] } })); });
      bar.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`class make-up (${total.toLocaleString("en")} images; validation only)`,
        cn.map((c, q) => ({ name: m.classes[q], value: `${c} (${(100 * c / total).toFixed(0)}%)`, color: pal.classes[q], c })).filter(r => r.c).sort((x, y) => y.c - x.c))));
      bar.addEventListener("pointerleave", hideTip);
      return bar;
    };
    for (let j = 0; j <= k; j++) {
      if (!size[j]) continue;
      const un = j === k, on = st.focus === j, dim = st.focus != null && !on;
      const head = h("div", un ? { class: "set-head" } : { class: "set-head", role: "button", tabindex: 0, title: "Click to isolate this set in every panel of the row" },
        h("span", { class: "sw", style: { background: un ? pal.unassigned : cols[j] } }), h("span", { class: "set-name", text: un ? "no label" : `set ${j + 1}` }),
        h("span", { class: "n", text: size[j].toLocaleString("en") }), compBar(cnt[j], size[j]));
      if (!un) {
        const toggle = () => this.depthFocus(j);
        head.addEventListener("click", toggle);
        head.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      }
      const row = h("div", { class: "set-row" + (on ? " on" : "") + (dim ? " dim" : "") }, head);
      if (!un) row.append(strip(SR.shuffledMembers(lab, j, 2024 + 31 * j), on ? 2 : 1));
      body.append(row);
    }
    const key = h("div", { class: "class-key" });
    m.classes.forEach((c, q) => key.append(h("span", {}, h("i", { style: { background: pal.classes[q] } }), c)));
    body.append(key, h("div", { class: "card-note", text: `Sizes, CIFAR class make-up (validation only: labels play no part in the sets) and members in a fixed pseudo-random order, at ${m.sites[s]}. The same set number is the same shared centroid at every site.` }));
  }
  depthFocus(j) {
    const st = this.st;
    if (this.depthPending()) return;
    st.focus = st.focus === j ? null : j;
    this.col = colouring(this.ctx, st, this.blk); this.renderAll(); this.writeHash();
  }
  renderDepthLegend() {
    const st = this.st, { m, pal } = this.ctx, col = this.col, body = $("leg-body"), S_ = m.spacetime;
    body.replaceChildren(); $("leg-jac").replaceChildren();
    const step = m.steps[st.anchor].toLocaleString("en");
    if (st.dm !== "sets") {
      $("leg-title").textContent = `Colour: mode ${st.dm} through depth, step ${step}`;
      body.append(h("div", { class: "cbar", style: { background: pal.divGradient() } }),
        h("div", { class: "cbar-ticks" }, h("span", { text: "−max" }), h("span", { text: "0" }), h("span", { text: "+max" })));
      body.append(h("div", { class: "leg-note", text: `Each panel in the outlined row shows its own site's slice u_l = f_l / |f_l| of mode ${st.dm}, scaled so its largest entry is ±1; largest |u_l| on top. The sign is one choice for the whole mode, so neighbouring panels can be compared (flipping every sign gives the same mode). A uniformly grey panel holds under ${100 * S_.low_weight}% of the mode${st.dv === "paper" ? " (B-weighted norm, b_l |f_l|²)" : ""}: a normalised numerical tail, not a direction. Other rows are not coloured.` }));
      return;
    }
    $("leg-title").textContent = `Colour: sets through depth, k = ${st.dk}, ${S_.readouts[st.dr]}, step ${step}`;
    const s = this.depthSite(), lab = depthLabels(this.ctx, st, s), cnt = new Array(st.dk).fill(0), tot = new Array(st.dk).fill(0);
    for (const l of lab) if (l >= 0) cnt[l]++;
    for (let q = 0; q < m.sites.length; q++) for (const l of depthLabels(this.ctx, st, q)) if (l >= 0) tot[l]++;
    for (let j = 0; j < st.dk; j++) {
      if (!tot[j]) continue;
      const row = h("div", { class: "leg-row" + (st.focus === j ? " on" : "") + (st.focus != null && st.focus !== j ? " dim" : ""), role: "button", tabindex: 0, title: "Click to isolate in every panel of the row" },
        h("span", { class: "sw", style: { background: col.cols[j] } }), h("span", { text: `set ${j + 1}` }), h("span", { class: "n", text: String(cnt[j]), title: `images in set ${j + 1} at ${m.sites[s]}` }));
      row.addEventListener("click", () => this.depthFocus(j));
      row.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.depthFocus(j); } });
      body.append(row);
    }
    body.append(h("div", { class: "leg-note", text: `One colour per shared centroid, the same at every site; grey = no label (the site holds almost none of the low space). Counts at ${m.sites[s]}, the site of the first card. Other rows are not coloured.` }));
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
    const sw = (c, name) => h("span", { class: "kk" }, h("i", { style: { background: c, boxShadow: "inset 0 0 0 1px rgba(0,0,0,.08)" } }), name);
    if (col.depth) {      // through depth: one row, each site its own slice (or the shared sets); grey meanings spelled out
      box.append(h("span", { class: "kt", text: `step ${m.steps[col.depth.t].toLocaleString("en")} only:` }));
      if (col.sets) {
        for (let j = 0; j < col.sets.k; j++) box.append(sw(col.cols[j], col.sets.names[j]));
        box.append(sw(pal.unassigned, "no label"));
      } else {
        box.append(h("span", { class: "kt", text: `mode ${st.dm}, each site its own slice` }), h("span", { class: "kt", text: "−max" }),
          h("span", { class: "kbar", style: { background: pal.divGradient() } }), h("span", { class: "kt", text: "+max" }), sw(DEPTH_LOW, "under 1% of the mode"));
      }
      box.append(sw(DIM, "other rows: not coloured"));
      return;
    }
    if (col.groups) {
      const g = col.groups;
      for (let j = 0; j < g.n; j++) {
        if (j === g.unassigned && !col.ci[0].includes(j) && !(st.focus != null)) continue;
        const c = j === g.unassigned ? pal.unassigned : (g.classes ? pal.classes[j] : (st.color === "seba" ? pal.set(j) : col.cols[j]));
        box.append(h("span", { class: "kk" }, h("i", { style: { background: c } }), g.names[j]));
      }
    } else if (col.cont) {
      const c = col.cont, tick = c.fmt || (v => c.log ? (Math.pow(10, v) >= 100 ? Math.pow(10, v).toFixed(0) : Math.pow(10, v).toPrecision(2)) : (+v.toPrecision(2)).toString());
      box.append(h("span", { class: "kt", text: c.label }), h("span", { class: "kt", text: (col.rg && c.kind === "div" ? "≤ " : "") + tick(c.lo) }),
        h("span", { class: "kbar", style: { background: c.kind === "div" ? pal.divGradient() : pal.seqGradient() } }), h("span", { class: "kt", text: (col.rg ? "≥ " : "") + tick(c.hi) }));
    }
  }
  renderCaption() {
    this.renderKey();
    $("ex-caption").replaceChildren(...captionParts(this.ctx, this.st, this.col));
  }
}

// Bars per site (the through-depth card): thin bars with a rounded top on a zero baseline, hairline grid, an optional
// dashed overlay (a control) on the same y scale, per-bar hover and click. o: names, y, color, low (bars drawn light
// grey), sel (outlined bar), yfmt, yhi, barName, overlay {name, y}, tip(i) -> rows, onclick(i), height.
function barChart(box, o) {
  box.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" }); box.append(wrap);
  const W = Math.max(220, box.clientWidth || 320), H = o.height || 130, mg = { l: 36, r: 8, t: 8, b: 20 }, n = o.y.length;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.barName || "bars per site" }, wrap);
  const all = o.y.concat(o.overlay ? o.overlay.y : []).filter(Number.isFinite);
  const yhi = o.yhi ?? Math.max(1e-12, ...all) * 1.06, Y = scale("linear", 0, yhi, H - mg.b, mg.t);
  const bw = (W - mg.l - mg.r) / n, X = i => mg.l + (i + .5) * bw, yfmt = o.yfmt || (v => (+v.toPrecision(2)).toString());
  const g = el("g", { class: "grid" }, svg);
  for (const v of niceTicks(0, yhi, 3)) { el("line", { x1: mg.l, x2: W - mg.r, y1: Y(v), y2: Y(v) }, g); const t = el("text", { x: mg.l - 5, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yfmt(v); }
  el("line", { x1: mg.l, x2: W - mg.r, y1: H - mg.b + .5, y2: H - mg.b + .5 }, el("g", { class: "axis" }, svg));
  const every = bw < 30 ? 2 : 1;
  o.names.forEach((nm, i) => { if ((n - 1 - i) % every) return; const t = el("text", { x: X(i), y: H - mg.b + 13, "text-anchor": "middle" }, svg); t.textContent = nm; });
  for (let i = 0; i < n; i++) {
    const v = o.y[i]; if (!Number.isFinite(v)) continue;
    const x0 = X(i) - bw / 2 + 1.5, x1 = X(i) + bw / 2 - 1.5, y0 = Y(0), y = Math.min(y0 - .5, Y(Math.min(v, yhi))), r = Math.min(3, (x1 - x0) / 2, y0 - y);
    el("path", { d: `M${x0},${y0}L${x0},${y + r}Q${x0},${y} ${x0 + r},${y}L${x1 - r},${y}Q${x1},${y} ${x1},${y + r}L${x1},${y0}Z`, fill: o.low?.[i] ? "#dddcd6" : o.color,
      ...(i === o.sel ? { stroke: "#16161a", "stroke-width": 1.5 } : {}) }, svg);
  }
  if (o.overlay) {
    const P = o.overlay.y.map((v, i) => [X(i), Y(Math.min(v, yhi))]);
    el("path", { d: P.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(""), fill: "none", stroke: "#16161a", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    P.forEach(p => el("circle", { cx: p[0], cy: p[1], r: 2.4, fill: "#16161a", stroke: "#fff", "stroke-width": 1 }, svg));
  }
  for (let i = 0; i < n; i++) {      // hit targets: the whole column, bigger than the bar
    const r = el("rect", { x: X(i) - bw / 2, y: 0, width: bw, height: H - mg.b + 14, fill: "transparent", style: o.onclick ? "cursor:pointer" : "" }, svg);
    r.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(o.names[i], o.tip ? o.tip(i) : [{ name: o.barName || "value", value: yfmt(o.y[i]) }])));
    r.addEventListener("pointerleave", hideTip);
    if (o.onclick) r.addEventListener("click", () => { hideTip(); o.onclick(i); });
  }
  if (o.overlay) {
    const lg = h("div", { class: "chart-legend" });
    lg.append(h("span", { class: "k" }, h("i", { style: { "--sw": o.color, height: "8px", borderRadius: "2px" } }), o.barName || "value"),
      h("span", { class: "k" }, h("i", { style: { background: "repeating-linear-gradient(90deg, #16161a 0 4px, transparent 4px 7px)" } }), o.overlay.name));
    wrap.append(lg);
  }
}

// Signed bars (the change card's patterns): one bar per value from a zero line, up for positive and down for negative,
// symmetric y range, per-bar hover and click on the whole column. o: y, color(i), sel (outlined), groups [[from, to,
// label]] (labelled under the axis), name(i), yfmt, label (aria), tip(i) -> rows, onclick(i), height.
function signedBars(box, o) {
  box.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" }); box.append(wrap);
  const W = Math.max(220, box.clientWidth || 320), H = o.height || 130, mg = { l: 40, r: 8, t: 8, b: 20 }, n = o.y.length;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.label || "signed bars" }, wrap);
  const M = Math.max(1e-12, ...o.y.filter(Number.isFinite).map(Math.abs)) * 1.08, Y = scale("linear", -M, M, H - mg.b, mg.t);
  const bw = (W - mg.l - mg.r) / n, X = i => mg.l + (i + .5) * bw, yfmt = o.yfmt || (v => (+v.toPrecision(2)).toString());
  const g = el("g", { class: "grid" }, svg);
  for (const v of niceTicks(-M, M, 4)) { el("line", { x1: mg.l, x2: W - mg.r, y1: Y(v), y2: Y(v) }, g); const t = el("text", { x: mg.l - 5, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yfmt(v); }
  for (const [a, b, label] of o.groups || []) { const t = el("text", { x: (X(a) + X(b)) / 2, y: H - 5, "text-anchor": "middle" }, svg); t.textContent = label; }
  for (let i = 0; i < n; i++) {
    const v = o.y[i]; if (!Number.isFinite(v)) continue;
    const y0 = Y(0), y = Y(v);
    el("rect", { x: X(i) - bw / 2 + 1.5, y: Math.min(y, y0), width: Math.max(1, bw - 3), height: Math.max(1, Math.abs(y - y0)), rx: 1.5, fill: o.color(i),
      ...(i === o.sel ? { stroke: "#16161a", "stroke-width": 1.5 } : {}) }, svg);
  }
  el("line", { x1: mg.l, x2: W - mg.r, y1: Y(0) + .5, y2: Y(0) + .5 }, el("g", { class: "axis" }, svg));
  for (let i = 0; i < n; i++) {      // hit targets: the whole column
    const r = el("rect", { x: X(i) - bw / 2, y: 0, width: bw, height: H, fill: "transparent", style: o.onclick ? "cursor:pointer" : "" }, svg);
    r.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(o.name ? o.name(i) : String(i + 1), o.tip ? o.tip(i) : [{ name: "value", value: yfmt(o.y[i]) }])));
    r.addEventListener("pointerleave", hideTip);
    if (o.onclick) r.addEventListener("click", () => { hideTip(); o.onclick(i); });
  }
  return { svg, X, Y };
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
    dw: `diffusion ${st.dwk ? "wavelet" : "scaling function"} ${dwName(st, st.dwf)} (Coifman–Maggioni) of the outlined snapshot's lazy walk M at scale j = ${st.dwj} (M run for ${2 ** st.dwj} steps), centred on image #${col?.dw?.row[2] ?? "?"}: a pivoted QR of M^(2^j) on the previous scale's basis, nothing fitted. Painted on every panel so you can follow the same images; click a point to pick a snapshot and the function of this scale carrying that image most.`,
    rg: `${RG_VIEW[st.rgv]} from each site to the next, at the final checkpoint. Measured in each site's own graph coordinates on the image's neighbourhood at the earlier site, not in this layout. Each column displays its own site's step; the last column has no next site.`,
    ch: !col?.ch ? "change to the next site (loading)." : (() => {
      const C = col.ch, A = m.sites[C.a], B = m.sites[C.b], K = m.ch.k;
      const D_ = `D = M_b^τ − M_a^τ, the change of the lazy walk from ${A} (a) to ${B} (b) at ${step(m.ch.t)}, τ = ${C.tau} walk step${C.tau > 1 ? "s" : ""}`;
      return C.view === "mode"
        ? `pattern “${chName(K, C.k)}”: the eigenvector of ${D_} for ${C.k < K ? `positive eigenvalue ${C.k + 1} (largest first)` : `negative eigenvalue ${C.k - K + 1} (most negative first)`} (λ = ${C.d.lam[C.k].toFixed(4)}, ${share(C.d.share[C.k])} of ‖D‖²_F), scaled to ±1, sign arbitrary; diverging scale, largest |v| on top. Computed from the two outlined sites only and painted on every panel; nothing fitted.`
        : `per image, tv_i = ½ Σ_j |D_ij| with ${D_}: the share of its τ-step distribution that moved${C.tau === 1 ? "; at one step shown doubled, 2 tv_i = ½ Σ_j |(P_b − P_a)_ij| with P = 2M − I the walk without its lazy half: the share of the image's neighbour weight that moved" : ""}; sequential scale from 0 to the largest value at this pair and scale (${pct(C.scMax)}), largest on top. Computed from the two outlined sites only and painted on every panel; nothing fitted.`;
    })(),
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
    colour = `hard sets from "${src.label}" (${S.classes ? "10 sets" : `r = ${st.r} sets`}): ${src.def} ` +
      (S.classes ? "Class colours." : "Each set takes the colour of the method's set of this block it overlaps most (Hungarian matching on Jaccard; display only).") +
      ` Grey = unassigned. Painted on every panel. Kernel rule ${m.modes[0]}.`;
  }
  const dot = ctx.kernel && ctx.kernel !== "cos";
  // under dot, fitState leaves st.lay at seed 0 or the spectral layout (st.lay indexes the cosine root's list)
  const L = (dot ? (st.lay ? m.layouts.find(x => x.kind === "spectral") : m.layouts[0]) : m.layouts[st.lay]) || {};
  const lay = !L.kind && dot ? "UMAP of each snapshot from the top 50 principal components of its centred dot-product Gram (AlignedUMAP across sites, Procrustes across checkpoints, then rotated onto the cosine layout)"
    : !st.lay ? "UMAP of each snapshot (AlignedUMAP across sites, Procrustes across checkpoints)"
    : L.kind === "spectral" ? `UMAP of each snapshot's diffusion coordinates λ_kψ_k, k = 1..${L.n_coords}, from that snapshot's own walk under the ${m.modes[L.mode]} rule, whatever rule colours the points (AlignedUMAP across sites, Procrustes across checkpoints)`
    : `UMAP of seed ${L.seed ?? st.lay} (a second network; colours from seed 0)`;
  const part = (label, text) => [h("b", { text: label + " " }), text + " "];
  if (col?.depth && st._dep) {       // through depth (SPEC_SPACETIME.md section 2): its own definitional caption
    const S_ = m.spacetime, R = st._dep.Sm.variants[st.dv], c = st.dv === "paper" ? 0 : st.dt, K = S_.K;
    const op = st.dv === "paper"
      ? `the paper's transition matrix B⁻¹A as is (A links site l to l+1 through the walk M_l, so image i at one site is tied to a random neighbour at the next; B = diag(1, 2, …, 2, 1) ⊗ I), its leading ${K} eigenvectors among functions of mean zero at every site; no adjustable coupling (A itself ties neighbouring sites); depth weights and occupancy use the B-weighted norm b_l |f_l|², as the paper normalises`
      : `L_t = blockdiag(H_l) + t (D_path ⊗ I) with H_l = (I − M_l)/c_l, ${st.dv === "own" ? "c_l = the mean of site l's first 8 nontrivial 1 − λ (relative site units)" : "c_l = 1 (native walk units)"}, D_path the path Laplacian over the ${S} sites (image i at one site linked to image i at the next), t = ${(+R.t[c].toPrecision(3))} = t* × ${fracLabel(S_.factors[c])}; its ${K} smallest eigenvectors among functions of mean zero at every site`;
    const colour = st.dm !== "sets"
      ? `mode ${st.dm} of ${op}. Every panel of the outlined row shows its own site's slice u_l = f_l / |f_l| of the mode, scaled so max|u_l| = 1; diverging scale, one sign for the whole mode (arbitrary), largest |u_l| on top. A panel in uniform grey holds under ${100 * S_.low_weight}% of the mode's ${st.dv === "paper" ? "B-weighted squared norm b_l |f_l|²" : "squared norm"} (a normalised numerical tail, not a direction). Mode numbers are local to this setting (not matched across couplings or checkpoints).`
      : `post-hoc sets (k-means is a fit), k = ${st.dk}, ${S_.readouts[st.dr]}: one k-means with centroids shared across the ${S} sites on the rows of the first 8 modes' slices of ${op}, each site's block scaled to unit Frobenius norm${st.dr === "a" ? ", each site first rotated onto the previous one by orthogonal Procrustes" : ""}; set numbers by decreasing size over all sites; grey = no label (the site holds almost none of the space).`;
    return [
      ...part("Through depth:", `one checkpoint, step ${m.steps[st.anchor].toLocaleString("en")} (outlined row); the other rows are not coloured (light grey). No block is used.`),
      ...part("Kernel and rule:", `cosine kernel, ${m.modes[S_.rule]} (fixed while this colouring is on: the through-depth data exist for this rule only).`),
      ...part("Colour:", colour),
      ...part("Positions:", `${lay}; display only, no number on this page is computed from them. Rows: checkpoints (step, validation accuracy); columns: sites; ${T * S} panels of the same ${m.n.toLocaleString("en")} probe images.`),
    ];
  }
  return [
    ...(dot ? part("Kernel:", `${ctx.kernelLabel || "dot"}: half squared Euclidean distance |x_i − x_j|²/2 on the unnormalised activations in place of the cosine distance 1 − cos (data ${m.tag}).`) : []),
    ...part("Block:", `${blockKind(st).toLowerCase()}, ${blk}.`),
    ...part("Kernel rule:", `${m.modes[st.mode]}.`),
    ...part("Colour:", colour + focus),
    ...part("Positions:", `${lay}; display only, no number on this page is computed from them. Rows: checkpoints (step, validation accuracy); columns: sites; ${T * S} panels, one per snapshot, of the same ${m.n.toLocaleString("en")} probe images.`),
  ];
}
