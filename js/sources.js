// "Sets from": the colour sources of the baselines study (explore/cifar_baselines/SCHEMA.md; data from site/baselines.py).
// Everything here is kernel rule 0 (kNN graph, k = 15) and the average operator only.
import * as D from "./data.js?v=633f70d53a";

export const STATUS = {
  S: "label-free spectral object built from dot products (an eigen- or singular-vector computation on the images)",
  P: "post-processing, diagnostic or null generator (SEBA rounding, a vote, a shuffle)",
  B: "fitted baseline outside the construction (k-means and spherical k-means are fits)",
  V: "uses the CIFAR class labels (validation only, or a null whose strata are classes)",
};

// group (menu), label, status tag, definition (one or two sentences, plain words)
export const SOURCES = [
  { id: "M0", group: "The method", label: "the method (default)", status: "S+P",
    def: "SEBA on ψ0..ψr−1 of the block operator M̄ (the mean of the block's lazy walks)." },
  { id: "L1", group: "Linear geometry", label: "cosine PCA + SEBA", status: "S+P",
    def: "The top r − 1 principal directions of the block's centred cosine Gram (the mean of its snapshots' Grams; the same computation as classical MDS of the chordal distances), rounded by the same SEBA step. No graph." },
  { id: "L2", group: "Linear geometry", label: "PCA + k-means", status: "B",
    def: "k-means (k = r, 20 starts) on the block's PCA scores (max(r − 1, 20) components). A fit." },
  { id: "L3", group: "Linear geometry", label: "spherical k-means", status: "B",
    def: "Exact spherical k-means (k = r, 20 k-means++ starts) on the normalised activations, through the block Gram. A fit with no spectral step." },
  { id: "G1", group: "Same graph, other rounding", label: "method eigenvectors + k-means", status: "S+B",
    def: "k-means (k = r) on the method's own ψ1..ψr−1 instead of SEBA." },
  { id: "G1n", group: "Same graph, other rounding", label: "method eigenvectors, row-normalised + k-means", status: "S+B",
    def: "k-means on the rows of ψ0..ψr−1 after scaling each row to unit length (Ng–Jordan–Weiss rounding)." },
  { id: "G2", group: "Same graph, other rounding", label: "textbook spectral clustering", status: "S+B",
    def: "Each snapshot's 15-NN graph normalised as D^−1/2 A D^−1/2 (no Sinkhorn balancing, no lazy step, no floor), averaged over the block; leading eigenvectors row-normalised, then k-means." },
  { id: "C1first", group: "One snapshot, or a vote", label: "first snapshot of the block", status: "S+P",
    def: "The method on the block's first snapshot alone." },
  { id: "C1mid", group: "One snapshot, or a vote", label: "middle snapshot of the block", status: "S+P",
    def: "The method on the block's middle snapshot alone (index ⌊(a + b)/2⌋)." },
  { id: "C1last", group: "One snapshot, or a vote", label: "last snapshot of the block", status: "S+P",
    def: "The method on the block's last snapshot alone." },
  { id: "C2", group: "One snapshot, or a vote", label: "co-assignment consensus", status: "P",
    def: "For every pair of images, the share of the block's snapshots whose own sets (same r) put the two together; the leading eigenvectors of this centred co-assignment matrix, rounded by SEBA." },
  { id: "C3", group: "One snapshot, or a vote", label: "mean-projector consensus", status: "S+P",
    def: "The leading eigenvectors of the mean, over the block's snapshots, of the projector onto each snapshot's own leading r − 1 eigenfunctions, rounded by SEBA. Uses the snapshots' spans, not their spectra." },
  { id: "I1", group: "Input and other networks", label: "pixels", status: "S+P",
    def: "The method on the network's input (normalised pixels) as one snapshot. The same sets on every block." },
  { id: "I2", group: "Input and other networks", label: "untrained network", status: "S+P",
    def: "The method on the same layer block at step 0, before training." },
  { id: "I3", group: "Input and other networks", label: "other training seed", status: "S+P",
    def: "The method on the same layer block of a second network trained from another seed (seed 1), final checkpoint, same probe images." },
  { id: "Y1", group: "Labels (validation)", label: "classes", status: "V", k10: true,
    def: "The CIFAR-10 class labels: 10 sets." },
  { id: "Y2", group: "Labels (validation)", label: "predicted class", status: "V", k10: true,
    def: "The network's predicted class at the block's checkpoint (layer block) or at its last checkpoint (time block): 10 sets." },
  { id: "N3", group: "Null draws", label: "trajectory shuffle (one draw)", status: "P", null: true,
    def: "Each snapshot's walk is relabelled by its own random permutation of the images, the block's first snapshot is kept, and the method is run (seed 0). Keeps every snapshot exactly; destroys that row i is the same image in every snapshot. On a single-snapshot block it changes nothing." },
  { id: "N2a", group: "Null draws", label: "spectrum-matched Gaussian (one draw)", status: "P", null: true, rep: true,
    def: "Gaussian activations with each snapshot's covariance spectrum, independent across snapshots, then the method. Keeps each snapshot's spectrum in expectation, before the pipeline's centring and row normalisation; destroys higher moments and every link between snapshots." },
  { id: "N2b", group: "Null draws", label: "joint second-order surrogate (one draw)", status: "P", null: true, rep: true,
    def: "One random rotation of image space, shared by all the block's snapshots, applied to each Gram, then the unit diagonal is restored and the method run. Keeps within- and cross-snapshot second moments before that restoration (the change after it is stored and shown in the Baselines tab); destroys image-level structure beyond them." },
  { id: "N4", group: "Null draws", label: "within-class trajectory shuffle (one draw)", status: "P+V", null: true, rep: true,
    def: "The trajectory shuffle with each permutation restricted to images of the same CIFAR class. Keeps the snapshots and class identity across them; destroys which image within a class is which." },
  { id: "N5", group: "Null draws", label: "degree-preserving rewiring (one draw)", status: "P", null: true, rep: true,
    def: "Each snapshot's kNN graph rewired by double-edge swaps that keep every image's degree, then the floor, Sinkhorn and lazy step. Keeps degrees; destroys which images are neighbours." },
];
export const SRC = Object.fromEntries(SOURCES.map(s => [s.id, s]));
const LAB = new Set(["L1", "L2", "L3", "G1", "G1n", "G2", "C2", "C3", "N3", "I3"]);

// the representative block (Baselines tab) that equals this explorer block, if any
export function repOf(idx, st) {
  for (const [id, r] of Object.entries(idx.reps || {})) if (r.fam === st.fam && r.anchor === st.anchor && r.a === st.a && r.b === st.b) return id;
  return null;
}
// {id: null (available) | reason (string)} for the explorer state
export function availability(idx, st, isProd) {
  const out = {}, f = idx && idx.files[`${st.fam}/${st.anchor}`], rep = idx && repOf(idx, st);
  for (const s of SOURCES) {
    let why = null;
    if (s.id === "M0") why = null;
    else if (!idx) why = "baselines data not in this build";
    else if (st.mode !== idx.kernel_rule) why = "computed for kernel rule 0 (kNN graph, k = 15) only";
    else if (isProd) why = "compared with the average operator only";
    else if (s.id === "I2" && st.fam !== "layer") why = "defined for layer blocks only";
    else if (s.id === "I3" && !(f && f.lab.includes("I3"))) why = "stored for layer blocks at the final checkpoint only";
    else if (LAB.has(s.id) && !(f && f.lab.includes(s.id))) why = "not stored for this block";
    else if (s.rep && !(rep && idx.reps[rep].draws.includes(s.id))) why = "stored for the 12 representative blocks only";
    out[s.id] = why;
  }
  return out;
}

// Hungarian assignment (maximise) on a rectangular score matrix; returns row -> column (-1 if unmatched)
function hungarian(M, nr, nc) {
  const n = Math.max(nr, nc), INF = 1e18, u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  const at = (i, j) => (i < nr && j < nc) ? M[i][j] : 0;
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) if (!used[j]) {
        const cur = -at(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const a = new Int32Array(nr).fill(-1); for (let j = 1; j <= n; j++) if (p[j] - 1 < nr && j - 1 < nc) a[p[j] - 1] = j - 1; return a;
}
// relabel src (k sets) so that the set matched (Jaccard, Hungarian) to the method's set j takes index j; display only
function matchColours(src, k, ref, r, N) {
  const inter = Array.from({ length: k }, () => new Float64Array(r)), cs = new Float64Array(k), cr = new Float64Array(r);
  for (let i = 0; i < N; i++) { const a = src[i], b = ref[i]; if (a >= 0) cs[a]++; if (b >= 0) cr[b]++; if (a >= 0 && b >= 0) inter[a][b]++; }
  const J = inter.map((row, i) => Array.from(row, (x, j) => { const un = cs[i] + cr[j] - x; return un > 0 ? x / un : 0; }));
  const asg = hungarian(J, k, r), used = new Set([...asg].filter(j => j >= 0)), map = new Int32Array(k);
  let free = [...Array(Math.max(k, r)).keys()].filter(j => !used.has(j));
  for (let i = 0; i < k; i++) map[i] = asg[i] >= 0 ? asg[i] : free.shift();
  return map;
}
// members of set j in a fixed pseudo-random order (sources that store hard labels only)
export function shuffledMembers(lab, j, seed) {
  const mem = []; for (let i = 0; i < lab.length; i++) if (lab[i] === j) mem.push(i);
  let s = seed; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let i = mem.length - 1; i > 0; i--) { const q = Math.floor(rnd() * (i + 1)); [mem[i], mem[q]] = [mem[q], mem[i]]; }
  return mem;
}

// Load the sets of source `id` for the explorer state (kernel rule 0, average operator). Returns
// {id, k, lab (display labels: Int8Array N, -1 = unassigned), names, classes, key (draw order) | null,
//  reps(j) -> image indices | null, agree {...} | null}. pairIdx: ctx.pairIdx; m0lab: the method's labels at this r.
export async function load(ctx, st, id, m0lab) {
  const { m, N } = ctx, ri = m.seba_r.indexOf(st.r), r = st.r, idx = await D.srcIndex();
  const pairs = ctx.pairIdx[st.fam], pair = pairs[st.a + "," + st.b];
  const chunk = await D.srcChunk(st.fam, st.anchor, pair);
  const F = idx.fields, fget = (row, k) => row ? row[ri * F.length + F.indexOf(k)] : NaN;
  const file = idx.files[`${st.fam}/${st.anchor}`];
  const maxBand = (sid, field) => { const v = file?.maxnull?.[sid]?.[ri]?.[idx.max_fields.indexOf(field)]; return v ? v[1] : NaN; };
  let raw, k = SRC[id].k10 ? 10 : r, reps = null, key = null, agree = null, blkSrc = null;
  const sub = a => a.subarray(ri * N, (ri + 1) * N);
  if (id === "Y1") raw = Int8Array.from(ctx.labels);
  else if (id === "Y2") { const t = st.fam === "layer" ? st.anchor : st.b; raw = Int8Array.from(ctx.pts.pred.subarray(t * N, (t + 1) * N)); }
  else if (id === "I1") raw = sub(await D.srcI1());
  else if (id.startsWith("C1") || id === "I2") {
    let fam = st.fam, anchor = st.anchor, p = pair;
    if (id === "I2") anchor = 0;
    else { const c = id === "C1first" ? st.a : id === "C1last" ? st.b : Math.floor((st.a + st.b) / 2); p = pairs[c + "," + c]; }
    blkSrc = await D.block(fam, anchor, idx.kernel_rule, p);
    raw = sub(blkSrc.lab);
  } else if (SRC[id].rep) raw = sub((await D.srcRep(repOf(idx, st)))[id]);
  else raw = sub(chunk.lab[id]);
  // agreement with the method (same block and r)
  const row = chunk && chunk.agree[id];
  if (id === "Y1") {
    const r0 = chunk.agree.M0;
    agree = { ami: fget(r0, "class_ami_joint"), cov: 1 - fget(r0, "unassigned"), q95: fget(r0, "class_ami_joint_null_q95"), max95: maxBand("M0", "class_ami_joint"),
      mj: fget(r0, "class_mj_max"), mjNull: fget(r0, "class_mj_null_mean"), mjMax95: maxBand("M0", "class_mj_max"), nperm: idx.n_perm };
  } else if (row) {
    agree = { ami: fget(row, "ami_joint"), cov: fget(row, "joint_coverage"), q95: fget(row, "ami_joint_null_q95"), max95: maxBand(id, "ami_joint"),
      mj: fget(row, "mj_max"), mjNull: fget(row, "mj_null_mean"), mjMax95: maxBand(id, "mj_max"), nperm: id === "Y2" ? idx.n_perm_site : idx.n_perm,
      cls: { ami: fget(row, "class_ami_joint"), q95: fget(row, "class_ami_joint_null_q95"), mj: fget(row, "class_mj_max"), mjNull: fget(row, "class_mj_null_mean") } };
  } else if (SRC[id].rep) {
    const a = idx.reps[repOf(idx, st)].agree[id][ri], g = k => a[F.indexOf(k)];
    agree = { ami: g("ami_joint"), cov: g("joint_coverage"), q95: g("ami_joint_null_q95"), max95: NaN, mj: g("mj_max"), mjNull: g("mj_null_mean"), mjMax95: NaN, nperm: idx.n_perm_site, rep: true };
  }
  // display labels: colours matched to the method's sets (sets of the method keep their own colours)
  let lab = raw, names;
  if (!SRC[id].k10) {
    const map = matchColours(raw, k, m0lab, r, N);
    lab = Int8Array.from(raw, x => x < 0 ? -1 : map[x]);
    const inv = new Int32Array(k); map.forEach((d, s) => inv[d] = s);
    names = [...Array(k)].map((_, j) => `set ${j + 1}`);
    if (blkSrc) {
      const vecs = D.sebaVectors(m, blkSrc, ri);
      key = Float64Array.from(lab, (l, i) => l < 0 ? -1 : vecs[inv[l]][i]);
      reps = j => D.reps(m, blkSrc, "set", ri, inv[j]).top;
    }
  } else names = m.classes.slice();
  return { id, k, lab, names, classes: !!SRC[id].k10, key, reps, agree };
}
