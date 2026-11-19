// Lazy, cached data access for the static site. All URLs are relative to the page.
const cache = new Map();

function once(key, fn) {
  if (!cache.has(key)) {
    const p = fn().catch(e => { cache.delete(key); throw e; });
    cache.set(key, p);
  }
  return cache.get(key);
}

export function bytes(url) {
  return once("b:" + url, async () => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.arrayBuffer();
  });
}

export function json(url) {
  return once("j:" + url, async () => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  });
}

export const meta = () => json("d/meta.json");

export function image(url) {
  return once("i:" + url, () => new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(url));
    im.src = url;
  }));
}

export const sprite = () => image("d/sprite.webp");
export const labels = () => bytes("d/labels.bin").then(b => new Int8Array(b));

export async function layout(seed) {
  const b = await bytes(`d/umap_${seed}.bin`);
  return new Uint16Array(b);  // (T, S, N, 2)
}

export async function points() {
  const m = await meta();
  const b = await bytes("d/points.bin");
  const T = m.steps.length, N = m.n, out = {};
  let off = 0;
  for (const f of m.points_fields) { out[f] = new Float32Array(b, off, T * N); off += 4 * T * N; }
  out.pred = new Int8Array(b, off, T * N); off += T * N;
  out.correct = new Uint8Array(b, off, T * N);
  return out;
}

const blkCache = new Map();
const BLK_MAX = 400;
export function block(fam, idx, mode, pair) {
  const key = `${fam}/${idx}/${mode}/${pair}`;
  if (blkCache.has(key)) return blkCache.get(key);
  const p = (async () => {
    const m = await meta();
    const L = m.block_layout[fam];
    const r = await fetch(`d/blk/${key}.bin.gz`);
    if (!r.ok) throw new Error(`block ${key}: HTTP ${r.status}`);
    const b = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    return parseBlock(m, L, b);
  })();
  p.catch(() => blkCache.delete(key));
  blkCache.set(key, p);
  if (blkCache.size > BLK_MAX) blkCache.delete(blkCache.keys().next().value);
  return p;
}

function view(b, f) {
  if (f.dtype === "f32") return new Float32Array(b, f.offset, f.count);
  if (f.dtype === "i8") return new Int8Array(b, f.offset, f.count);
  if (f.dtype === "u16") return new Uint16Array(b, f.offset, f.count);
  if (f.dtype === "i16") return new Int16Array(b, f.offset, f.count);
  return new Uint8Array(b, f.offset, f.count);
}

function parseBlock(m, L, b) {
  const F = L.fields, N = m.n, R = m.seba_r.length;
  const o = {};
  for (const k in F) o[k] = view(b, F[k]);
  // SEBA labels: one byte per image, value + 1 (0 = unassigned) -> Int8Array (R, N) of -1..r-1
  const lab = new Int8Array(R * N);
  for (let i = 0; i < R * N; i++) lab[i] = o.labels[i] - 1;
  o.lab = lab;
  o.X = L.X;
  return o;
}

// representative image indices computed at build time from full-precision vectors (see build_site.py:
// REP_PSI, REP_TOP, REP_BND). TOP and BND are fixed; the per-eigenfunction stride PSI is derived from the stored
// block layout (count = n_vecs * 3 * PSI + sum(seba_r) * (TOP + BND)) so the two files cannot drift apart again.
export const REP = { TOP: 24, BND: 12 };
const repPsi = (m, blk) => (blk.reps.length - m.seba_r.reduce((x, y) => x + y, 0) * (REP.TOP + REP.BND)) / (m.n_vecs * 3);
export function reps(m, blk, kind, a, b) {
  // kind 'psi': a = k (1..8), b = 0 most positive | 1 most negative | 2 nearest zero
  // kind 'set': a = rank index ri, b = set j; returns {top, bnd}
  const clean = arr => Array.from(arr).filter(i => i !== 65535), P = repPsi(m, blk);
  if (kind === "psi") { const o = ((a - 1) * 3 + b) * P; return clean(blk.reps.subarray(o, o + P)); }
  const q = m.seba_r.slice(0, a).reduce((x, y) => x + y, 0) + b, o = m.n_vecs * 3 * P + q * (REP.TOP + REP.BND);
  return { top: clean(blk.reps.subarray(o, o + REP.TOP)), bnd: clean(blk.reps.subarray(o + REP.TOP, o + REP.TOP + REP.BND)) };
}

// psi_k (k = 1..8) as a Float64Array of the unit-norm eigenvector
export function psi(m, blk, k) {
  const N = m.n, out = new Float64Array(N), sc = blk.scale[k - 1] / (blk.vecs.BYTES_PER_ELEMENT === 2 ? 32767 : 127), off = (k - 1) * N;
  for (let i = 0; i < N; i++) out[i] = blk.vecs[off + i] * sc;
  return out;
}

// SEBA amplitude of each image in its own set (rank index ri), in [0, 1]; 0 for unassigned images. Computed from the
// float64 SEBA vectors at measurement time and shipped as u8 (the browser never reconstructs SEBA vectors).
export function sebaVal(m, blk, ri) {
  const N = m.n, out = new Float32Array(N), s = blk.sval.subarray(ri * N, (ri + 1) * N);
  for (let i = 0; i < N; i++) out[i] = s[i] / 255;
  return out;
}

// stability horizon of a block: the largest k whose leading-k span is reproducible (span_k >= threshold), 0 if none
export function horizon(m, blk) {
  const thr = m.stability.threshold;
  for (let k = blk.span.length; k >= 1; k--) if (blk.span[k - 1] >= thr) return k;
  return 0;
}
