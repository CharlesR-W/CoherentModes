// COPY of explore/lm_coherent/page/js/data.js, adapted for the Mamba page: checkpoints are labels (init, trained);
// cells carry a view (p point, s state, k stack, b bag); blocks are reading:<view>:<li> and depth:<ti>.
// Data access. The page embeds two gzip+base64 blobs (build_page.py): a JSON object (meta, compare rows, trees,
// validation, the token string table, and "index", the layout of the binary pack) and a binary pack of
// little-endian typed arrays. Both are decoded with DecompressionStream, so the page works from file://.
import { Palette } from "./colors.js";

const TYPED = { f4: Float32Array, i1: Int8Array, u1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array };
export const D = { ready: null };

async function gunzipB64(id) {
  const b64 = document.getElementById(id).textContent.trim();
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}

export function loadData() {
  if (D.ready) return D.ready;
  D.ready = (async () => {
    const [jb, bb] = await Promise.all([gunzipB64("blob-json"), gunzipB64("blob-bin")]);
    const J = JSON.parse(new TextDecoder().decode(jb));
    Object.assign(D, J);
    D.A = {};
    for (const [name, [off, code, shape]] of Object.entries(J.index)) {
      const n = shape.reduce((a, b) => a * b, 1);
      const arr = new TYPED[code](bb, off, n);
      arr.shape = shape;
      D.A[name] = arr;
    }
    D.N = J.n; D.L = J.layers.length; D.T = J.positions.length;
    D.ckpts = J.checkpoints; D.S = D.ckpts.length; D.fi = D.ckpts.indexOf(J.final); D.final = J.final;
    D.VIEWS = ["p", "s", "k"]; D.vIdx = { p: 0, s: 1, k: 2 };
    D.pal = new Palette(J.palette);
    D.srcCol = J.sources.map((_, i) => J.palette.classes[i]);
    D.cellIdx = Object.fromEntries(J.cell_keys.map((k, i) => [k, i]));
    D.blockIdx = Object.fromEntries(J.block_keys.map((k, i) => [k, i]));
    D.src = D.A.src;
    return D;
  })();
  return D.ready;
}

// ------------------------------------------------------------------------------------------- array helpers
function strides(sh) { const st = new Array(sh.length); let s = 1; for (let i = sh.length - 1; i >= 0; i--) { st[i] = s; s *= sh[i]; } return st; }
// a[ix...]: a scalar when every axis is indexed, otherwise a subarray view of the trailing axes
export function at(a, ...ix) {
  const st = strides(a.shape);
  let off = 0;
  for (let i = 0; i < ix.length; i++) off += ix[i] * st[i];
  return ix.length === a.shape.length ? a[off] : a.subarray(off, off + st[ix.length - 1]);
}
export const A = name => D.A[name];
export const excess = (s, s0) => (s - s0) / (1 - s0);
export const fmt = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : "n/a";
export const ckName = c => c === "init" ? "init (HF)" : c;
export const VIEW_NAME = { p: "point", s: "state", k: "stack", b: "bag" };
export const VIEW_LONG = { p: "point view (residual at t)", s: "state view (layer l's own recurrent state)", k: "stack view (states of layers 1..l)", b: "bag reference" };
export const g3 = v => !Number.isFinite(v) ? "n/a" : v === 0 ? "0" : Math.abs(v) >= 1e-3 ? (+v.toPrecision(3)).toString() : v.toExponential(2);
export const layerName = li => li < 0 ? "bag" : "L" + D.layers[li];

// per-cell statistic: li = -1 is the bag reference (its own arrays)
export function stat(name, si, view, li, ti) {
  return li < 0 ? at(A("sb_" + name), si, ti) : at(A("s_" + name), si, view, li, ti);
}

// ------------------------------------------------------------------------------------------- records
function labelsOf(arr, k) {
  const b = at(arr, k), n = b.length, l2 = new Int8Array(n), l4 = new Int8Array(n);
  for (let i = 0; i < n; i++) { l2[i] = (b[i] & 15) - 1; l4[i] = (b[i] >> 4) - 1; }
  return { 2: l2, 4: l4 };
}
function psiOf(arr, k) {   // int8 with max |q| = 127 per vector: q / 127 in [-1, 1], the colour scale
  const q = at(arr, k), n = D.N;
  return [0, 1].map(j => { const out = new Float32Array(n); for (let i = 0; i < n; i++) out[i] = q[j * n + i] / 127; return out; });
}
const recCache = new Map();
// view "p" (point), "s" (state), "k" (stack), "b" (bag); checkpoint label; li layer index (-1 for bag); ti position index
export function cellRecord(view, ck, li, ti) {
  if (li < 0) view = "b";
  const key = `${view}:${ck}:${view === "b" ? -1 : li}:${ti}`;
  if (recCache.has(key)) return recCache.get(key);
  const k = D.cellIdx[key];
  if (k === undefined) return null;
  const ret = ax => ({ psi: at(A("c_retp"), k, ax), sets: at(A("c_rets"), k, ax) });
  const rec = {
    kind: "cell", key, view, ck, li: view === "b" ? -1 : li, ti, psi: psiOf(A("c_psi"), k), lab: labelsOf(A("c_lab"), k),
    vals: at(A("c_vals"), k), pn: at(A("c_pn"), k), ret: [ret(0), ret(1)], top: at(A("c_top"), k), sj: at(A("c_sj"), k),
  };
  recCache.set(key, rec);
  return rec;
}
// kind "reading" (a = view code, b = li), "depth" (a = ti; point view)
export function blockRecord(kind, a, b) {
  const key = kind === "reading" ? `reading:${a}:${b}` : `depth:${a}`;
  if (recCache.has(key)) return recCache.get(key);
  const k = D.blockIdx[key];
  if (k === undefined) return null;
  const st = at(A("b_stat"), k);
  const rec = {
    kind, key, a, b, psi: psiOf(A("b_psi"), k), lab: labelsOf(A("b_lab"), k), vals: at(A("b_vals"), k), pn: at(A("b_pn"), k),
    ret: { psi: at(A("b_retp"), k), sets: at(A("b_rets"), k) }, top: at(A("b_top"), k), sj: at(A("b_sj"), k),
    stat: { T: st[0], T0: st[1], Tb: [st[2], st[3]], W: st[4], W0: st[5], Wb: [st[6], st[7]] },
  };
  recCache.set(key, rec);
  return rec;
}
// top members of set j at rank r (2 or 4), from the stored lists (-1 padded)
export function topMembers(rec, r, j) {
  const row = (r === 2 ? 0 : 2) + j, t = rec.top.subarray(row * 24, row * 24 + 24);
  return Array.from(t).filter(x => x >= 0);
}
// set retention rows for rank r inside a [6, X] retention array
export const setRows = r => r === 2 ? [0, 1] : [2, 3, 4, 5];

// ------------------------------------------------------------------------------------------- tokens and snippets
export function token(i, t) { return D.tokstr[at(A("tok"), i, t - 1)]; }   // token t (1-based) of item i
export function showTok(s) { return s.replace(/\n/g, "↵").replace(/\t/g, "→"); }
// a DOM span: tokens [from..to] of item i; tokens <= t dark, > t grey, token t underlined; marks = positions to
// underline lightly (the sampled positions)
export function snippetEl(i, t, { before = 24, after = 8, marks = null, cls = "snip" } = {}) {
  const span = document.createElement("span");
  span.className = cls;
  const lo = Math.max(1, t - before + 1), hi = Math.min(128, t + after);
  if (lo > 1) span.append(Object.assign(document.createElement("span"), { className: "tk-more", textContent: "…" }));
  for (let u = lo; u <= hi; u++) {
    const e = document.createElement("span");
    e.className = u === t ? "tk tk-t" : u < t ? "tk" : "tk tk-after";
    if (marks && marks.includes(u) && u !== t) e.classList.add("tk-mark");
    e.textContent = showTok(token(i, u));
    e.title = `token ${u}`;
    span.append(e);
  }
  if (hi < 128) span.append(Object.assign(document.createElement("span"), { className: "tk-more", textContent: "…" }));
  return span;
}
export function srcSwatch(i) {
  const s = document.createElement("span");
  s.className = "swatch"; s.style.setProperty("--sw", D.srcCol[D.src[i]]);
  s.title = D.sources[D.src[i]] + " (source: validation only)";
  return s;
}
