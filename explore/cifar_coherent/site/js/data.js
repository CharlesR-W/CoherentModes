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

export async function chan(t, s) {
  const m = await meta();
  const C = m.channels[s], N = m.n;
  const b = await bytes(`d/chan/${t}_${s}.bin`);
  const lo = new Float32Array(b, 0, C), hi = new Float32Array(b, 4 * C, C), q = new Uint8Array(b, 8 * C, N * C);
  // dequantise, then centre each column -> Float32Array (C, N) column-major for fast correlations
  const X = new Float32Array(C * N);
  for (let c = 0; c < C; c++) {
    const a = lo[c], sc = (hi[c] - lo[c]) / 255;
    let mean = 0;
    for (let i = 0; i < N; i++) { const v = a + sc * q[i * C + c]; X[c * N + i] = v; mean += v; }
    mean /= N;
    for (let i = 0; i < N; i++) X[c * N + i] -= mean;
  }
  return { C, X };
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
  // unpack SEBA labels (value + 1 in 4 bits) -> Int8Array (R, N)
  const half = (N + 1) >> 1, lab = new Int8Array(R * N);
  for (let ri = 0; ri < R; ri++) for (let i = 0; i < N; i++) {
    const byte = o.labels[ri * half + (i >> 1)];
    lab[ri * N + i] = ((i & 1) ? byte >> 4 : byte & 15) - 1;
  }
  o.lab = lab;
  o.X = L.X;
  o._seba = {};
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

// SEBA vectors for rank index ri, recomputed as SPEC section 2 (clipped at 0, reliability order); (r arrays of N)
export function sebaVectors(m, blk, ri) {
  if (blk._seba[ri]) return blk._seba[ri];
  const N = m.n, r = m.seba_r[ri];
  const V = [new Float64Array(N).fill(1 / Math.sqrt(N))];
  for (let k = 1; k < r; k++) V.push(psi(m, blk, k));
  const mu = 0.99 / Math.sqrt(N), cols = [];
  const roff = m.seba_r.slice(0, ri).reduce((x, y) => x + y * y, 0), rot = blk.rot.subarray(roff, roff + r * r);   // R (r x r), row-major
  for (let j = 0; j < r; j++) {
    const s = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let z = 0;
      for (let c = 0; c < r; c++) z += V[c][i] * rot[c * r + j];
      const a = Math.abs(z) - mu;
      s[i] = a > 0 ? Math.sign(z) * a : 0;
    }
    let nrm = 0, sum = 0;
    for (let i = 0; i < N; i++) { nrm += s[i] * s[i]; sum += s[i]; }
    nrm = Math.sqrt(nrm) || 1;
    const sg = sum < 0 ? -1 : 1;
    let mx = -Infinity;
    for (let i = 0; i < N; i++) { s[i] = sg * s[i] / nrm; if (s[i] > mx) mx = s[i]; }
    for (let i = 0; i < N; i++) s[i] /= mx;
    cols.push(s);
  }
  const out = [];
  for (let j = 0; j < r; j++) {
    const s = cols[blk.perm[ri * 8 + j]];
    const c = new Float64Array(N);
    for (let i = 0; i < N; i++) c[i] = s[i] > 0 ? s[i] : 0;
    out.push(c);
  }
  blk._seba[ri] = out;
  return out;
}
