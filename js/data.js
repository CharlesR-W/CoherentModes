// Lazy, cached data access for the static site. All URLs are relative to the page.
// Data roots: "d" (cosine kernel, the default and the only root the prose, baselines and alignment use) and, when built,
// "d_dot" (dot kernel; d/meta.json "kernels"). meta, layout and block take the root; every cache key contains it (the
// once() keys are the URLs, which start with the root). Labels, sprite and points are kernel-independent: always d/.
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

export const meta = (root = "d") => json(`${root}/meta.json`);

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

export async function layout(seed, root = "d") {
  const b = await bytes(`${root}/umap_${seed}.bin`);
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
export function block(fam, idx, mode, pair, root = "d") {
  const key = `${root}/blk/${fam}/${idx}/${mode}/${pair}`;
  if (blkCache.has(key)) return blkCache.get(key);
  const p = (async () => {
    const m = await meta(root);
    const L = m.block_layout[fam];
    const r = await fetch(`${key}.bin.gz`);
    if (!r.ok) throw new Error(`block ${key}: HTTP ${r.status}`);
    const b = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const o = parseBlock(m, L, b);
    o.root = root;
    return o;
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
  if (kind === "psi" && a > m.n_vecs) {     // psi_9..psi_32: not stored, ranked in the browser from the 8-bit add-on vector
    const v = psi(m, blk, a), P = 12, idx = Array.from({ length: m.n }, (_, i) => i);
    const key = [i => -v[i], i => v[i], i => Math.abs(v[i])][b];
    return idx.sort((x, y) => key(x) - key(y) || x - y).slice(0, P);
  }
  const clean = arr => Array.from(arr).filter(i => i !== 65535), P = repPsi(m, blk);
  if (kind === "psi") { const o = ((a - 1) * 3 + b) * P; return clean(blk.reps.subarray(o, o + P)); }
  const q = m.seba_r.slice(0, a).reduce((x, y) => x + y, 0) + b, o = m.n_vecs * 3 * P + q * (REP.TOP + REP.BND);
  return { top: clean(blk.reps.subarray(o, o + REP.TOP)), bnd: clean(blk.reps.subarray(o + REP.TOP, o + REP.TOP + REP.BND)) };
}

// psi_k as a Float64Array of the unit-norm eigenvector: k = 1..8 from the block chunk, k = 9..32 from the add-on
// attached as blk.ext (attachExt; layer family, cosine root only)
export function psi(m, blk, k) {
  if (k > m.n_vecs) {
    const E = blk.ext, X = m.coarse.ext;
    if (!E) throw new Error(`ψ${k}: the ψ9..ψ32 add-on is not loaded for this block`);
    const N = m.n, out = new Float64Array(N), j = k - X.first, sc = E.scale[j] / 127, off = j * N;
    for (let i = 0; i < N; i++) out[i] = E.vecs[off + i] * sc;
    return out;
  }
  const N = m.n, out = new Float64Array(N), sc = blk.scale[k - 1] / (blk.vecs.BYTES_PER_ELEMENT === 2 ? 32767 : 127), off = (k - 1) * N;
  for (let i = 0; i < N; i++) out[i] = blk.vecs[off + i] * sc;
  return out;
}

// ------------------------------------------------------------------ coarse-graining add-ons (build_site.py coarse())
// meta.coarse (cosine root only): ext (psi_9..psi_32 of the layer average and both product sides), km (spectral
// k-means partitions, k = 2..16) and the per-unit coarse-graining summaries. All gzip, fetched on first use.
async function gunzip(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
// rules shipped for a layer block (a single cell is shipped for every rule; chains maybe only for the default rule)
export function coarseShipped(m, part, mode, cell) {
  const c = m.coarse?.[part];
  return !!c && (cell ? c.rules_cells : c.rules_chains).includes(mode);
}
// the psi_9..psi_32 add-on of one layer block: {vals (33, vals[k] = lambda_k or sigma_k), scale (24), vecs (24*N int8)}
export function ext(m, fam, t, mode, pair, cell) {
  const f = cell ? m.coarse.ext.cells_under : fam;
  return once(`x:d/ext/${f}/${t}/${mode}/${pair}`, async () => {
    const b = await gunzip(`d/ext/${f}/${t}/${mode}/${pair}.bin.gz`), F = m.coarse.ext.fields, o = {};
    if (b.byteLength !== m.coarse.ext.bytes) throw new Error(`ext ${f}/${t}/${mode}/${pair}: ${b.byteLength} bytes, expected ${m.coarse.ext.bytes}`);
    for (const k in F) o[k] = view(b, F[k]);
    o.pn = [];
    for (let j = 0; j < F.scale.count; j++) {     // participation number from the 8-bit vector (scale cancels)
      let s2 = 0, s4 = 0; const off = j * m.n;
      for (let i = 0; i < m.n; i++) { const v = o.vecs[off + i], q = v * v; s2 += q; s4 += q * q; }
      o.pn.push(s4 > 0 ? s2 * s2 / s4 : 0);
    }
    return o;
  });
}
// spectral k-means labels of one layer block: {start: Int8Array (15*N), end: Int8Array (15*N)} (end = start for a cell)
export function kmLabels(m, step, mode, pair, cell) {
  return once(`k:d/km/${step}_${mode}/${pair}`, async () => {
    const b = await gunzip(`d/km/${step}_${mode}/${pair}.bin.gz`), N = m.n, K = m.coarse.km.ks.length, half = (N + 1) >> 1;
    const un = off => { const u8 = new Uint8Array(b, off, K * half), out = new Int8Array(K * N);
      for (let q = 0; q < K; q++) for (let i = 0; i < N; i++) { const byte = u8[q * half + (i >> 1)]; out[q * N + i] = (i & 1) ? byte >> 4 : byte & 15; }
      return out; };
    const start = un(0);
    return { start, end: cell ? start : un(K * half) };
  });
}
// one checkpoint x rule of coarse-graining summaries: {step, rule, blocks: [per layer pair]} (see build_site.py coarse())
export function coarseUnit(step, mode) {
  return once(`c:d/coarse/${step}_${mode}`, async () => JSON.parse(new TextDecoder().decode(await gunzip(`d/coarse/${step}_${mode}.json.gz`))));
}

// ------------------------------------------------------------------ through-depth add-on (build_site.py spacetime())
// meta.spacetime (cosine root, kernel rule meta.spacetime.rule only). Fetched on first use, per checkpoint and variant.
// One (step, variant): {C, K, L, vecs: Int8Array (C, K, L, N) unit slices scaled to max |u| = 127,
// lab: Int8Array (4, C, L, N) set labels (-1 = no label), planes in the order of meta.spacetime.sets}
export function depthUnit(m, step, variant) {
  return once(`st:d/st/${step}_${variant}`, async () => {
    const S = m.spacetime, V = S.variants.find(v => v.id === variant);
    if (!V) throw new Error(`through depth: no variant "${variant}"`);
    const { K, L } = S, C = V.C, N = m.n, half = (N + 1) >> 1, nv = C * K * L * N, nl = S.sets.length * C * L;
    const b = await gunzip(`d/st/${step}_${variant}.bin.gz`);
    if (b.byteLength !== nv + nl * half) throw new Error(`d/st/${step}_${variant}: ${b.byteLength} bytes, expected ${nv + nl * half}`);
    const u8 = new Uint8Array(b, nv, nl * half), lab = new Int8Array(nl * N);
    for (let p = 0; p < nl; p++) for (let i = 0; i < N; i++) { const byte = u8[p * half + (i >> 1)]; lab[p * N + i] = ((i & 1) ? byte >> 4 : byte & 15) - 1; }
    return { C, K, L, vecs: new Int8Array(b, 0, nv), lab };
  });
}
// per-step readings of every variant: {step, rule, tstar: {own, none}, variants: {own|none|paper: {t, vals, weight,
// adj_cos, retention, own_share, occupancy, scalar_weight, sets: [{groups, churn}], link}}}
export function depthSummary(step) {
  return once(`sts:d/st/${step}`, async () => JSON.parse(new TextDecoder().decode(await gunzip(`d/st/${step}.json.gz`))));
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

// ------------------------------------------------------------------ baselines study (site/baselines.py -> d/src/)
// index.json: files {"<fam>/<anchor>": {lab: [source ids with label data], agree: [source ids with agreement rows], P,
// maxnull}}, reps {id: {fam, anchor, a, b, pair, draws, agree}}, fields (agreement row fields), seba_r.
export const srcIndex = () => json("d/src/index.json");
export function unpackNibbles(u8, count) {
  const out = new Int8Array(count);
  for (let i = 0; i < count; i++) out[i] = ((i & 1) ? u8[i >> 1] >> 4 : u8[i >> 1] & 15) - 1;
  return out;
}
const srcCache = new Map();
// one block's source labels (Int8Array (R*N) per source) and agreement rows (Float32Array (R*F) per source)
export function srcChunk(fam, anchor, pair) {
  const key = `${fam}/${anchor}/${pair}`;
  if (srcCache.has(key)) return srcCache.get(key);
  const p = (async () => {
    const [m, idx] = await Promise.all([meta(), srcIndex()]);
    const f = idx.files[`${fam}/${anchor}`];
    if (!f) return null;
    const r = await fetch(`d/src/${key}.bin.gz`);
    if (!r.ok) throw new Error(`sources ${key}: HTTP ${r.status}`);
    const b = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const N = m.n, R = m.seba_r.length, half = (N + 1) >> 1, F = idx.fields.length, out = { lab: {}, agree: {} };
    let off = 0;
    for (const s of f.lab) {
      const lab = new Int8Array(R * N);
      for (let ri = 0; ri < R; ri++) lab.set(unpackNibbles(new Uint8Array(b, off + ri * half, half), N), ri * N);
      out.lab[s] = lab; off += R * half;
    }
    for (const s of f.agree) { out.agree[s] = new Float32Array(b.slice(off, off + R * F * 4)); off += R * F * 4; }
    return out;
  })();
  p.catch(() => srcCache.delete(key));
  srcCache.set(key, p);
  if (srcCache.size > 200) srcCache.delete(srcCache.keys().next().value);
  return p;
}
// labels painted on every block: the method on the network's input pixels (R*N)
export async function srcI1() {
  const [m, b] = await Promise.all([meta(), bytes("d/src/i1.bin")]);
  const N = m.n, R = m.seba_r.length, half = (N + 1) >> 1, lab = new Int8Array(R * N);
  for (let ri = 0; ri < R; ri++) lab.set(unpackNibbles(new Uint8Array(b, ri * half, half), N), ri * N);
  return lab;
}
// the stored null draws of a representative block: {draw id: Int8Array (R*N)}
export async function srcRep(id) {
  const [m, idx] = await Promise.all([meta(), srcIndex()]);
  const rp = idx.reps[id], b = await bytes(`d/src/rep/${encodeURIComponent(id)}.bin`);
  const N = m.n, R = m.seba_r.length, half = (N + 1) >> 1, out = {};
  rp.draws.forEach((k, di) => {
    const lab = new Int8Array(R * N);
    for (let ri = 0; ri < R; ri++) lab.set(unpackNibbles(new Uint8Array(b, (di * R + ri) * half, half), N), ri * N);
    out[k] = lab;
  });
  return out;
}
