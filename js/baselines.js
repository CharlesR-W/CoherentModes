// Live figures of the "Baselines and nulls" tab (content/baselines.html; data d/bn/*.json from site/baselines.py).
// Every chart says what is plotted and draws its null or oracle reference; none ranks the sources.
import * as D from "./data.js?v=633f70d53a";
import { Palette } from "./colors.js?v=633f70d53a";
import { h, el, lineChart, showTip, hideTip, tipRows, scale, niceTicks, fmtStep } from "./charts.js?v=633f70d53a";
import { SRC } from "./sources.js?v=633f70d53a";

// colours: the method in ink; other sources in muted families (not the saturated set colours); nulls in muted greys/earths
export const SRC_COL = {
  M0: "#16161a", L1: "#8c5a2b", L2: "#c39060", L3: "#5a3714", G1: "#4a6274", G1n: "#8da1b3", G2: "#22384b",
  C1first: "#7d7f2c", C1mid: "#a7a95a", C1last: "#4f5118", C2: "#8e5b85", C3: "#56304f",
  I1: "#9c6b55", I2: "#b9a58f", I3: "#6b4a3a", Y1: "#8d8c86", Y2: "#b5b4ae",
  N3: "#9a9993", N2a: "#a0702a", N2b: "#4f6b8a", N4: "#8a4f7d", N5: "#5d7f4f",
  V_product: "#7a4a4a", V_two_end: "#a86f6f", V_meanM2: "#5b4a7a", V_concat_graph: "#3e6e6a", G2op: "#22384b", oracle: "#9a9993",
  T_linear: "#8c5a2b", T_nystrom: "#2f5d5d", T_diffusion: "#4a6274", T_soft_ulam: "#8e5b85",
};
const DASH = new Set(["L2", "G1n", "C1mid", "I2", "Y2", "V_two_end", "T_soft_ulam"]);
const NULL_IDS = ["N2a", "N2b", "N3", "N4", "N5"];
export const NULL_NAME = { N2a: "spectrum-matched Gaussian", N2b: "joint second-order surrogate", N3: "trajectory shuffle", N4: "within-class trajectory shuffle", N5: "degree-preserving rewiring" };
const BENCH_NAME = { M0: "the method", L1: "cosine PCA + SEBA", L2: "PCA + k-means", L3: "spherical k-means", G1: "method eigenvectors + k-means", G1n: "row-normalised + k-means",
  G2: "textbook spectral clustering", G2op: "textbook spectral clustering (operator-level)", C1first: "first cell alone", C1mid: "middle cell alone", C1last: "last cell alone", C2: "co-assignment consensus",
  C3: "mean-projector consensus", V_product: "ordered product (start)", V_two_end: "two-end product", V_meanM2: "mean of M²", V_concat_graph: "one graph on the concatenated view",
  T_linear: "transfer: linear CCA", T_nystrom: "transfer: Nyström", T_diffusion: "transfer: diffusion coordinates", T_soft_ulam: "transfer: soft Ulam" };
const srcName = id => SRC[id]?.label || BENCH_NAME[id] || id;
const fx = d => v => { if (!Number.isFinite(v)) return "n/a"; const t = v.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t; };
const f2 = fx(2), f3 = fx(3);
const num = v => v == null ? NaN : +v;
// median of the finite values; the mean of the middle two for an even count
const median = a => { const v = a.filter(Number.isFinite).sort((x, y) => x - y), n = v.length; return !n ? NaN : n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2; };

export async function render(kind, body, ds) {
  const fn = { "bn-spectra": spectra, "bn-matrix": matrix, "bn-dist": dist, "bn-haar": haar, "bn-stab": stab, "bn-s1cells": s1cells, "bn-ablate": ablate,
    "bn-linear": linear, "bn-bench": bench, "bn-tr-syn": trSyn, "bn-tr-real": trReal, "bn-tr-support": trSupport, "bn-tr-10k": tr10k, "bn-tr-comp": trComp }[kind];
  if (!fn) throw new Error(`unknown live figure "${kind}"`);
  await fn(body, ds);
}
const onResize = (elm, fn) => { let w = elm.clientWidth; new ResizeObserver(() => { if (Math.abs(elm.clientWidth - w) > 4) { w = elm.clientWidth; fn(); } }).observe(elm); };
const pal = async () => new Palette((await D.meta()).palette);
function seg(opts, val, on) {
  const s = h("div", { class: "seg", role: "radiogroup" });
  const sync = v => s.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === String(v))));
  opts.forEach(([v, label]) => { const b = h("button", { type: "button", "data-v": v, role: "radio", text: label }); b.addEventListener("click", () => { sync(v); on(v); }); s.append(b); });
  sync(val); return s;
}
function select(opts, val, on) {
  const s = h("select");
  opts.forEach(([v, label]) => s.add(new Option(label, v)));
  s.value = val; s.addEventListener("change", () => on(s.value)); return s;
}
const ctl = (...kids) => h("div", { class: "chart-controls bn-ctl" }, ...kids);
const lab = t => h("span", { class: "ctl-label", text: t });

// ------------------------------------------------------------------------------------------- representative blocks
let REPS = null;
async function reps() { if (!REPS) REPS = await D.json("d/bn/reps.json"); return REPS; }
async function repName(id) {
  const [R, m] = await Promise.all([reps(), D.meta()]), r = R.reps[id];
  if (r.fam === "layer") return `${r.a === r.b ? m.sites[r.a] : `${m.sites[r.a]} → ${m.sites[r.b]}`} at step ${m.steps[r.anchor].toLocaleString("en")}`;
  return `${m.sites[r.anchor]}, steps ${m.steps[r.a].toLocaleString("en")} → ${m.steps[r.b].toLocaleString("en")}`;
}
async function repOptions() { const R = await reps(); return Promise.all(R.order.map(async id => [id, await repName(id)])); }
const exLink = (r, rr = 4, src) => `#explorer?fam=${r.fam}&t=${r.anchor}&a=${r.a}&b=${r.b}&m=0&c=seba&r=${rr}${src ? "&src=" + src : ""}`;
function cellNames(m, r) { const out = []; for (let c = r.a; c <= r.b; c++) out.push(r.fam === "layer" ? m.sites[c] : fmtStep(m.steps[c])); return out; }

// (a) spectra against null bands
async function spectra(body, ds) {
  const [R, m, opts] = await Promise.all([reps(), D.meta(), repOptions()]);
  const st = { id: ds.block || "b4-b6", r: "4" };
  const chart = h("div"), tab = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("block"), select(opts, st.id, v => { st.id = v; draw(); }), lab("r (table)"), seg(R.seba_r.map(r => [String(r), String(r)]), st.r, v => { st.r = v; draw(); })), chart, tab, note);
  const draw = () => {
    const B = R.reps[st.id], x = [1, 2, 3, 4, 5, 6, 7, 8];
    const series = [{ name: "real block", y: B.real.lambda_P, color: SRC_COL.M0, width: 2.2, dots: 3 }];
    for (const k of NULL_IDS) if (B.bands[k]) series.push({ name: `${NULL_NAME[k]} (median of ${B.nulls[k].n} draws)`, y: B.bands[k].map(b => b.q50), color: SRC_COL[k], dash: true, width: 1.6,
      band: [B.bands[k].map(b => b.q05), B.bands[k].map(b => b.q95)] });
    lineChart(chart, { x, series, xticks: x, xfmt: k => `λ${k}`, xname: "", yrange: [0, 1], height: 280, bandMode: "each", ytitle: "λ_P = 2λ_M − 1",
      bandLabel: "bands: 5th to 95th percentile over the null draws" });
    // table: SEBA on the real block and on the null draws at rank r
    const rs = st.r, real = B.real.seba[rs], mc = B.m0_class[rs];
    const row = (name, un, mn, ca, am, sw) => h("tr", {}, h("td", {}, sw ? h("span", { class: "swatch", style: { "--sw": sw } }) : "", name), h("td", { text: un }), h("td", { text: mn }), h("td", { text: ca }), h("td", { text: am }));
    const band = b => b && Number.isFinite(b.q50) ? `${f2(b.q50)} [${f2(b.q05)}, ${f2(b.q95)}]` : "n/a";
    const t = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: `r = ${rs}` }), h("th", { text: "unassigned share" }), h("th", { text: "smallest set" }), h("th", { text: "class AMI" }), h("th", { text: "AMI with the method (joint)" })),
      row("real block", f2(real.unassigned), String(Math.min(...real.sizes)), `${f2(real.class_ami)} (perm. 95th pct ${f3(mc.class_ami_null_q95)})`, "1 (itself)", SRC_COL.M0));
    for (const k of NULL_IDS) if (B.nulls[k]) { const n = B.nulls[k].per_r[rs]; t.append(row(NULL_NAME[k], band(n.unassigned), `${Math.round(n.min_size.q50)} [${Math.round(n.min_size.q05)}, ${Math.round(n.min_size.q95)}]`, band(n.class_ami), band(n.ami_joint_M0), SRC_COL[k])); }
    tab.replaceChildren(t);
    const mm = B.nulls.N2b?.mismatch, kept = B.nulls.N5?.kept;
    note.innerHTML = `Null rows: median [5th, 95th percentile] over the draws. Class AMI counts unassigned images as one more category. ` +
      (B.a === B.b ? `This is a single-snapshot block: the trajectory shuffle keeps its one snapshot fixed, so it returns the block unchanged. ` : `The trajectory shuffle keeps the block's first snapshot fixed, so its class AMI keeps that snapshot's class structure and is not a zero reference. `) +
      (mm ? `Joint second-order surrogate: unit diagonals before restoring ranged ${f2(mm.diag_min)} to ${f2(mm.diag_max)}; after restoring, the stored cross-snapshot second-moment diagnostic changed by up to ${(100 * mm.second_moment_max).toFixed(1)}%. ` : "") +
      (kept ? `Rewiring kept ${(100 * kept[0]).toFixed(1)}% to ${(100 * kept[1]).toFixed(1)}% of the original edges. ` : "") +
      `<a class="explore-link" href="${exLink(B, +rs)}">this block in the explorer</a>`;
  };
  draw(); onResize(body, draw);
}

// generic heat map with a custom tooltip; z[r][c], color(t)
function heat(box, o) {
  box.replaceChildren();
  const nr = o.rows.length, nc = o.cols.length, Wc = Math.max(280, box.clientWidth || 480);
  const CH = 6.3, ml = o.ml ?? Math.ceil(Math.max(...o.rows.map(t => String(t).length)) * CH + 12), mt = 6;
  const cw = Math.max(13, Math.min(30, (Wc - ml - 60) / nc)), ch = Math.max(13, Math.min(22, cw));
  const mb = o.mb ?? Math.ceil(Math.max(...o.cols.map(t => String(t).length)) * CH * .87 + 16);
  const W = ml + nc * cw + 56, H = mt + nr * ch + mb;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: "display:block;margin:0 auto;max-width:none" }, box);
  for (let r = 0; r < nr; r++) {
    const t = el("text", { x: ml - 5, y: mt + r * ch + ch / 2 + 3.5, "text-anchor": "end" }, svg); t.textContent = o.rows[r];
    for (let c = 0; c < nc; c++) {
      const v = o.z[r][c], rect = el("rect", { x: ml + c * cw + .5, y: mt + r * ch + .5, width: cw - 1, height: ch - 1, rx: 1.5, fill: Number.isFinite(v) ? o.color((v - o.zrange[0]) / (o.zrange[1] - o.zrange[0])) : "#f2f1ec" }, svg);
      if (o.mark && o.mark(r, c)) el("circle", { cx: ml + c * cw + cw / 2, cy: mt + r * ch + ch / 2, r: Math.min(2.2, cw / 6), fill: "#fff", opacity: .9, "pointer-events": "none" }, svg);
      rect.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`${o.rows[r]} · ${o.cols[c]}`, o.tip(r, c))));
      rect.addEventListener("pointerleave", hideTip);
    }
  }
  for (let c = 0; c < nc; c++) {
    const x = ml + c * cw + cw / 2, y = mt + nr * ch + 6;
    const t = el("text", { x, y, "text-anchor": "end", "dominant-baseline": "middle", transform: `rotate(-60 ${x} ${y})` }, svg); t.textContent = o.cols[c];
  }
  const bx = ml + nc * cw + 12, bh = Math.min(120, nr * ch), id = "g" + Math.random().toString(36).slice(2);
  const lg = el("linearGradient", { id, x1: 0, x2: 0, y1: 1, y2: 0 }, el("defs", {}, svg));
  for (let i = 0; i <= 8; i++) el("stop", { offset: i / 8, "stop-color": o.color(i / 8) }, lg);
  el("rect", { x: bx, y: mt, width: 9, height: bh, fill: `url(#${id})`, rx: 2 }, svg);
  [[o.zrange[1], mt + 6], [o.zrange[0], mt + bh]].forEach(([v, y]) => { const t = el("text", { x: bx + 12, y, "text-anchor": "start" }, svg); t.textContent = String(v); });
}

// (b) source-by-source agreement on a representative block
async function matrix(body, ds) {
  const [M, R, m, P, opts] = await Promise.all([D.json("d/bn/matrix.json"), reps(), D.meta(), pal(), repOptions()]);
  const st = { id: ds.block || "b4-b6", r: "4", stat: "ami_joint" };
  const box = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("block"), select(opts, st.id, v => { st.id = v; draw(); }), lab("r"), seg(R.seba_r.map(r => [String(r), String(r)]), st.r, v => { st.r = v; draw(); }),
    seg([["ami_joint", "AMI (jointly assigned)"], ["mj_max", "matched Jaccard"]], st.stat, v => { st.stat = v; draw(); })), box, note);
  const draw = () => {
    const B = M.blocks[st.id], T = B.r[st.r], names = B.ids.map(id => `${srcName(id)} · ${SRC[id]?.status || ""}`);
    heat(box, { rows: names, cols: B.ids.map(srcName), z: T[st.stat], zrange: [0, 1], color: t => P.seq(t), ml: undefined,
      mark: (r, c) => { if (r === c) return false; const [v, q] = st.stat === "ami_joint" ? [T.ami_joint[r][c], T.null_q95[r][c]] : [T.mj_max[r][c], T.mj_null_q95[r][c]]; return Number.isFinite(v) && Number.isFinite(q) && v <= q; },
      tip: (r, c) => [{ name: "AMI, jointly assigned", value: f3(T.ami_joint[r][c]) }, { name: "coverage (both assign)", value: f2(T.coverage[r][c]) },
        { name: "AMI: permutation 95th pct", value: f3(T.null_q95[r][c]) }, { name: "matched Jaccard", value: f3(T.mj_max[r][c]) },
        { name: "matched Jaccard: permutation mean / 95th pct", value: `${f3(T.mj_null_mean[r][c])} / ${f3(T.mj_null_q95[r][c])}` }] });
    const q = (st.stat === "ami_joint" ? T.null_q95 : T.mj_null_q95).flat().filter(Number.isFinite);
    note.innerHTML = `${await_name(st.id)}, $r = ${st.r}$ (classes and predicted class: 10 sets). Colour: ${st.stat === "ami_joint" ? "AMI on the images both sources assign" : "matched Jaccard (sum of matched Jaccard over max(k<sub>a</sub>, k<sub>b</sub>))"}. ` +
      `Each cell has its own permutation reference (the column source's labels shuffled ${R.n_perm_site} times, set sizes kept); on this block the 95th percentiles of ${st.stat === "ami_joint" ? "the joint AMI" : "matched Jaccard"} range ${f3(Math.min(...q))} to ${f3(Math.max(...q))}. A white dot marks a cell at or below its own 95th percentile. Hover a cell for coverage and both null values.`;
    if (window.renderMathInElement) window.renderMathInElement(note, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
  };
  const names = Object.fromEntries(opts); const await_name = id => names[id];
  draw(); onResize(body, draw);
}

// (b) distribution over every explorer block of each source's agreement with the method
async function dist(body, ds) {
  const [Dd, m] = await Promise.all([D.json("d/bn/dist_index.json"), D.meta()]);
  const keys = Object.keys(Dd.files);
  const fname = k => { const [fam, a] = k.split("/"); return fam === "layer" ? `layer blocks at step ${m.steps[+a].toLocaleString("en")} (${Dd.files[k].P})` : `time blocks at ${m.sites[+a]} (${Dd.files[k].P})`; };
  const st = { f: ds.file || `layer/${m.steps.length - 1}`, r: "4", stat: "ami_joint" };
  const box = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("blocks"), select(keys.map(k => [k, fname(k)]), st.f, v => { st.f = v; draw(); }), lab("r"), seg(["2", "3", "4", "6", "8"].map(r => [r, r]), st.r, v => { st.r = v; draw(); }),
    seg([["ami_joint", "AMI (jointly assigned)"], ["mj_max", "matched Jaccard"]], st.stat, v => { st.stat = v; draw(); })), box, note);
  const draw = async () => {
    const F = await D.json(`d/bn/dist/${st.f.replace("/", "_")}.json`), [fam, anc] = st.f.split("/"), pairs = fam === "layer" ? m.layer_pairs : m.time_pairs;
    const ids = Object.keys(F.src).sort((a, b) => order(a) - order(b));
    const fi = F.fields.indexOf(st.stat), ci = F.fields.indexOf("joint_coverage");
    const W = Math.max(300, box.clientWidth || 600), ml = Math.min(340, W * .45), mr = 14, rowH = 24, H = 14 + ids.length * rowH + 30;
    box.replaceChildren();
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box), X = scale("linear", 0, 1, ml, W - mr);
    niceTicks(0, 1, 5).forEach(v => { el("line", { x1: X(v), x2: X(v), y1: 8, y2: H - 26, stroke: "#efeee9" }, svg); const t = el("text", { x: X(v), y: H - 12, "text-anchor": "middle" }, svg); t.textContent = v.toFixed(1); });
    const tt = el("text", { x: (ml + W - mr) / 2, y: H - 1, "text-anchor": "middle", class: "ttl" }, svg); tt.textContent = st.stat === "ami_joint" ? "AMI with the method's sets (jointly assigned images)" : "matched Jaccard with the method's sets";
    ids.forEach((id, i) => {
      const y0 = 10 + i * rowH, v = F.src[id].vals[st.r][fi], cov = F.src[id].vals[st.r][ci], mx = F.src[id].max[st.r][st.stat];
      const name = id === "Y1" ? "classes (the method against them)" : srcName(id);
      const t = el("text", { x: ml - 8, y: y0 + rowH / 2 + 3.5, "text-anchor": "end" }, svg); t.textContent = `${name} · ${SRC[id]?.status || ""}`;
      el("title", {}, t).textContent = `${name} · ${SRC[id]?.status || ""}`;
      if (mx && Number.isFinite(mx[1])) {
        el("rect", { x: X(0), y: y0 + 3, width: Math.max(1.5, X(Math.min(1, mx[1])) - X(0)), height: rowH - 6, fill: "#16161a", opacity: .09, rx: 2 }, svg);
        el("line", { x1: X(Math.min(1, mx[1])), x2: X(Math.min(1, mx[1])), y1: y0 + 2, y2: y0 + rowH - 2, stroke: "#6b6b70", "stroke-width": 1.5 }, svg);
      }
      v.forEach((x, p) => {
        if (!Number.isFinite(x)) return;
        const jy = y0 + rowH / 2 + ((p * 7919) % 13 - 6) * (rowH - 8) / 13;
        const c = el("circle", { cx: X(Math.max(0, Math.min(1, x))), cy: jy, r: 2.3, fill: SRC_COL[id] || "#555", opacity: .75 }, svg);
        c.addEventListener("pointermove", e => { const [a, b] = pairs[p]; const nm = fam === "layer" ? `${m.sites[a]} → ${m.sites[b]}` : `${fmtStep(m.steps[a])} → ${fmtStep(m.steps[b])}`;
          showTip(e.clientX, e.clientY, tipRows(nm, [{ name: st.stat === "ami_joint" ? "AMI (joint)" : "matched Jaccard", value: f3(x) }, { name: id === "Y1" ? "coverage (method assigned)" : "coverage", value: f2(id === "Y1" ? 1 - cov : cov) },
            { name: "reference from shuffled labels: largest over blocks, 95th pct", value: mx ? f3(mx[1]) : "n/a" }])); });
        c.addEventListener("pointerleave", hideTip);
      });
    });
    note.innerHTML = `One dot per block of the chosen file (${F.P} blocks), $r = ${st.r}$; kernel rule ${m.modes[0]}, average operator. Grey bar and tick: a reference from <b>shuffled labels</b>, not an observed value: shuffle the source's labels (set sizes kept) in every one of these ${F.P} blocks, take the largest agreement over the blocks, and repeat; the tick is the 95th percentile of that largest value. It is the reference for a value picked by scanning the blocks. ` +
      (st.stat === "ami_joint" ? "For the classes the row shows the method's sets against the class labels." : "Matched Jaccard under permutation is not near 0: it depends on the set sizes.") +
      " On single-snapshot blocks the trajectory shuffle leaves the block unchanged, so its dots sit at 1.";
    if (window.renderMathInElement) window.renderMathInElement(note, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
  };
  draw(); onResize(body, draw);
}
const ORDER = ["L1", "L2", "L3", "G1", "G1n", "G2", "C1first", "C1mid", "C1last", "C2", "C3", "I1", "I2", "I3", "Y1", "Y2", "N3"];
const order = id => { const i = ORDER.indexOf(id); return i < 0 ? 99 : i; };

// (c) SEBA on random subspaces and rotation sensitivity
async function haar(body, ds) {
  const [R, m] = await Promise.all([reps(), D.meta()]), rs = R.seba_r, N = R.n;
  const opts = await repOptions();
  const st = { stat: "unassigned", id: ds.block || "b4-b6" };
  const chart = h("div"), rot = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(seg([["unassigned", "unassigned"], ["min", "smallest set"], ["clipped", "clipped mass"]], st.stat, v => { st.stat = v; draw(); }),
    lab("rotation test"), select(opts, st.id, v => { st.id = v; draw(); })), chart, rot, note);
  const draw = () => {
    const H = R.haar, get = (r, k) => H[String(r)][k];
    const key = st.stat === "unassigned" ? "unassigned" : st.stat === "min" ? "min_size" : "clipped_negative_mass", sc = st.stat === "min" ? 1 / N : 1;
    const series = [{ name: `random subspaces (median of ${get(rs[0], "draws").toLocaleString("en")})`, y: rs.map(r => get(r, key).q50 * sc), color: "#8d8c86", dash: true, width: 1.6,
      band: [rs.map(r => get(r, key).q05 * sc), rs.map(r => get(r, key).q95 * sc)] }];
    for (const id of R.order) {
      const S = R.reps[id].real.seba;
      series.push({ name: id, y: rs.map(r => { const s = S[String(r)]; return st.stat === "unassigned" ? s.unassigned : st.stat === "min" ? Math.min(...s.sizes) / N : s.clipped_negative_mass; }), color: "#16161a", width: .01, dots: 2.6 });
    }
    lineChart(chart, { x: rs, xticks: rs, xfmt: r => `r = ${r}`, xname: "", series, height: 250, legend: false, yrange: st.stat === "clipped" ? [0, .25] : [0, .5], yfmt: v => (+v.toFixed(2)).toString(),
      ytitle: st.stat === "unassigned" ? "share of images unassigned" : st.stat === "min" ? "smallest set / N" : "clipped negative mass" });
    chart.append(h("div", { class: "chart-legend" }, h("span", { class: "k" }, h("i", { style: { "--sw": "#16161a", height: "6px", width: "6px", borderRadius: "50%" } }), "the 12 representative blocks (method)"),
      h("span", { class: "k" }, h("i", { style: { background: "repeating-linear-gradient(90deg,#8d8c86 0 4px,transparent 4px 7px)" } }), "SEBA on random subspaces: median"), h("span", { class: "k mute", text: "grey band: 5th to 95th percentile" })));
    const T = R.reps[st.id].N6_rotation, t = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: "r" }), ...rs.map(r => h("th", { text: String(r) }))));
    const stat = (r, f) => { const v = (T[String(r)] || []).map(d => d[f]).filter(Number.isFinite); return v.length ? [Math.min(...v), median(v)] : [NaN, NaN]; };
    t.append(h("tr", {}, h("td", { text: "joint AMI, smallest of 20" }), ...rs.map(r => h("td", { text: f3(stat(r, "ami_joint")[0]) }))),
      h("tr", {}, h("td", { text: "joint AMI, median of 20" }), ...rs.map(r => h("td", { text: f3(stat(r, "ami_joint")[1]) }))),
      h("tr", {}, h("td", { text: "coverage, median" }), ...rs.map(r => h("td", { text: f2(stat(r, "joint_coverage")[1]) }))),
      h("tr", {}, h("td", { text: "permutation 95th pct, largest of 20" }), ...rs.map(r => h("td", { text: f3(Math.max(...(T[String(r)] || []).map(d => num(d.ami_joint_null_q95)).filter(Number.isFinite))) }))));
    rot.replaceChildren(h("p", { class: "chart-sub", style: { textAlign: "center" }, text: `Rotation test, ${opts.find(o => o[0] === st.id)[1]}: SEBA started from 20 random rotations of ψ0..ψr−1, sets compared with the default start` }), t);
    const ref = R.reps[st.id].ref_null;
    note.textContent = `Random subspaces: SEBA applied to ${get(rs[0], "draws").toLocaleString("en")} Haar-random (r − 1)-dimensional subspaces orthogonal to the constant (N = ${N.toLocaleString("en")}); ${rs.map(r => get(r, "failures")).reduce((a, b) => a + b, 0)} SEBA failures in total. Rotation test: each comparison has its own stored permutation reference (last row).`;
  };
  draw(); onResize(body, draw);
}

// (d) stability on a representative block
async function stab(body, ds) {
  const [R, m, P, opts] = await Promise.all([reps(), D.meta(), pal(), repOptions()]), rs = R.seba_r;
  const st = { id: ds.block || "b4-b6" };
  const c1 = h("div"), t2 = h("div", { class: "bn-fig" }), c3 = h("div", { class: "bn-fig" }), t4 = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  const panel = (title, sub, kid) => h("div", { class: "bn-panel" }, h("h4", { text: title }), h("p", { class: "chart-sub", text: sub }), kid);
  body.replaceChildren(ctl(lab("block"), select(opts, st.id, v => { st.id = v; draw(); })),
    h("div", { class: "bn-panels" },
      panel("S1: ten 80% subsamples", "agreement with the sets on all 2,000 images, on the shared images; everything recomputed per subsample", c1),
      panel("S2: frozen extension to 500 held-out images", "fit on 1,500 images, extend to 500; compared with a refit on all 2,000 (test images); retention on a graph of the test images alone", t2),
      panel("S3: rank nesting", "H(sets at r | sets at r′) in nats; 0 = the finer sets refine the coarser ones", c3),
      panel("Leave one snapshot out", "retention of the other snapshots' sets on the omitted snapshot, minus its size null (mean over sets)", t4)), note);
  const draw = () => {
    const B = R.reps[st.id], S1 = B.S1, ref = B.ref_null;
    const med = median;
    const col = k => rs.map(r => S1.map(s => s[String(r)]?.[k]));
    const aj = col("ami_joint"), ov = col("overlap"), cov = col("joint_coverage"), nq = col("ami_joint_null_q95"), oc = col("overlap_chance");
    lineChart(c1, { x: rs, xticks: rs, xfmt: r => `r = ${r}`, xname: "", yrange: [0, 1], height: 210, series: [
      { name: "AMI (joint), median", y: aj.map(med), color: SRC_COL.M0, dots: 3, band: [aj.map(a => Math.min(...a)), aj.map(a => Math.max(...a))] },
      { name: "coverage, median", y: cov.map(med), color: "#8d8c86", dash: true, width: 1.4 },
      { name: "subspace overlap (mean cos²), median", y: ov.map(med), color: SRC_COL.G1, width: 1.6, dots: 2.5, band: [ov.map(a => Math.min(...a)), ov.map(a => Math.max(...a))] },
      { name: "joint AMI, permutation 95th pct (largest of 10)", y: nq.map(a => Math.max(...a.filter(Number.isFinite))), color: "#b5b4ae", dash: true, width: 1.2 },
      { name: "overlap chance q/(N − 1)", y: oc.map(med), color: "#d3d2cc", dash: true, width: 1.2 }],
      bandMode: "each", bandLabel: "bands: smallest to largest of the 10 subsamples" });
    const S2 = B.S2.src, tb = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: "source" }), h("th", { text: "r" }), h("th", { text: "AMI vs refit" }), h("th", { text: "perm. 95%" }), h("th", { text: "cov." }), h("th", { text: "ret. − null" })));
    for (const s of Object.keys(S2)) for (const r of ["2", "4", "8"]) {
      const v = S2[s][r]; if (!v) continue;
      const ex = v.test_retention.map((x, i) => x - v.test_null[i]).filter(Number.isFinite), mean = ex.reduce((a, b) => a + b, 0) / (ex.length || 1);
      tb.append(h("tr", {}, h("td", { text: srcName(s) }), h("td", { text: r }), h("td", { text: f2(v.ami_joint) }), h("td", { text: f3(num(v.ami_joint_null_q95)) }), h("td", { text: f2(v.joint_coverage) }), h("td", { text: f3(mean) })));
    }
    t2.replaceChildren(tb, h("p", { class: "bn-note", text: B.S2.flagged.length ? `Flagged directions with λ_P < 0.1: ${B.S2.flagged.join(", ")}.` : "No direction with λ_P < 0.1 was flagged." }));
    const S3 = B.S3, z = rs.map(r => rs.map(r2 => S3[`${r}|${r2}`] ?? NaN));
    heat(c3, { rows: rs.map(r => `r = ${r}`), cols: rs.map(r => `r′ = ${r}`), z, zrange: [0, 1], color: t => P.seq(t), ml: 50, mb: 44, tip: (i, j) => [{ name: "H(sets at r | sets at r′), nats", value: f3(z[i][j]) }] });
    const LO = B.LOCO, names = cellNames(m, B), tl = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: "r" }), ...(LO["4"]?.rows || []).map(x => h("th", { text: `omit ${names[x.omitted]}` })), h("th", { text: "next snapshot" })));
    for (const r of ["2", "4", "8"]) {
      const L = LO[r]; if (!L) continue;
      const ex = o => { const v = o.retention.map((x, i) => x - o.null[i]); return v.reduce((a, b) => a + b, 0) / v.length; };
      tl.append(h("tr", {}, h("td", { text: r }), ...L.rows.map(x => h("td", { text: f3(ex(x)) })), h("td", { text: L.next ? f3(ex(L.next)) : "n/a" })));
    }
    t4.replaceChildren(B.a === B.b ? h("p", { class: "bn-note", text: "Single-snapshot block: no snapshot to leave out." }) : tl);
    note.textContent = `Agreement numbers carry their own stored permutation reference (${S1[0]?.["4"]?.n_perm ?? "n/a"} permutations of the labels within the jointly assigned images: S1, the dashed grey line; S2, the column "perm. 95%"). Retention size null: t + (1 − t)(|A| − 1)/(N − 1), the expected retention of a random set of the same size.`;
  };
  draw(); onResize(body, draw);
}

// (d) S1 for every single snapshot at the final checkpoint, and the 10,000-image check
async function s1cells(body) {
  const [R, m] = await Promise.all([reps(), D.meta()]), rs = R.seba_r;
  const st = { stat: "ami_joint" };
  const c = h("div"), t = h("div", { class: "bn-fig" });
  body.replaceChildren(ctl(seg([["ami_joint", "AMI (joint)"], ["joint_coverage", "coverage of the joint AMI"], ["overlap", "subspace overlap"]], st.stat, v => { st.stat = v; draw(); })), c,
    h("p", { class: "bn-note", text: "The joint AMI is computed on the images that both the subsample's sets and the full-sample sets assign; the coverage view shows that share (median over the 10 subsamples) for the same snapshots and r." }), t);
  const draw = async () => {
    const sites = Object.keys(R.s1_cells), x = sites.map((_, i) => i), P = await pal();
    const med = median;
    lineChart(c, { x, xticks: x, xfmt: i => sites[i], xname: "site", yrange: [0, 1], height: 230, ytitle: "median over 10 subsamples",
      series: [...rs.map((r, k) => ({ name: `r = ${r}`, y: sites.map(s => med(R.s1_cells[s].map(d => d[String(r)]?.[st.stat]))), color: P.ord(.2 + .8 * k / (rs.length - 1)), dots: 2.5, width: 1.6 })),
        ...(st.stat === "joint_coverage" ? [] : [{ name: st.stat === "overlap" ? "chance q/(N − 1), largest r" : "permutation 95th pct of the joint AMI, largest over r and subsets", color: "#9a9993", dash: true, width: 1.2,
          y: sites.map(s => Math.max(...R.s1_cells[s].flatMap(d => rs.map(r => num(d[String(r)]?.[st.stat === "overlap" ? "overlap_chance" : "ami_joint_null_q95"]))).filter(Number.isFinite))) }])] });
    const N = R.n10k, tb = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: "block (final step)" }), h("th", { text: "k" }), ...rs.map(r => h("th", { text: `r = ${r}` }))));
    for (const [id, d] of Object.entries(N)) for (const [k, v] of Object.entries(d))
      tb.append(h("tr", {}, h("td", { text: await repName(id) }), h("td", { text: k.replace("k", "") }), ...rs.map(r => h("td", { text: `${f2(v.per_r[String(r)]?.ami_joint)} (${f2(v.per_r[String(r)]?.joint_coverage)}; ${f3(num(v.per_r[String(r)]?.ami_joint_null_q95))})` }))));
    t.replaceChildren(h("p", { class: "chart-sub", style: { textAlign: "center" }, text: "Default construction on the 10,000 CIFAR-10 test images (k = 15 and k = 75): AMI with the 2,000-image sets on the 2,000 probe images (coverage; permutation 95th pct)" }), tb);
  };
  draw(); onResize(body, draw);
}

// (e) construction ablations, one panel per factor
const FACTOR = { A: "graph family and scale", B: "floor", C: "normalisation", D: "diagonal and laziness", E: "block combination", F: "block weights (time blocks)", G: "rounding", H: "view preprocessing", I: "sample size" };
async function ablate(body, ds) {
  const [R, m, opts] = await Promise.all([reps(), D.meta(), repOptions()]);
  const st = { id: ds.block || "b4-b6", r: "4" };
  const grid = h("div", { class: "bn-panels wide" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("block"), select(opts, st.id, v => { st.id = v; draw(); }), lab("r"), seg(R.seba_r.map(r => [String(r), String(r)]), st.r, v => { st.r = v; draw(); })), grid, note);
  const bar = v => { const s = h("span"); if (Number.isFinite(v)) s.append(h("span", { class: "bn-bar", style: { width: `${Math.max(1, 36 * Math.max(0, Math.min(1, v)))}px` } })); s.append(f2(v)); return s; };
  const draw = () => {
    const B = R.reps[st.id], A = B.ablate, r = st.r, q = +r - 1, chance = q / (R.n - 1);
    grid.replaceChildren();
    for (const fac of Object.keys(FACTOR)) {
      const levels = A[fac] || {};
      const tb = h("table", { class: "bn-table" }, h("tr", {}, h("th", { text: "level" }), h("th", { text: "vs default: AMI (cov.)", title: "this level's sets against the default's sets: AMI on jointly assigned images (coverage)" }), h("th", { text: "perm. 95%", title: "95th percentile of that AMI with labels permuted within the jointly assigned images" }),
        h("th", { text: "match. J (perm. mean)" }), h("th", { text: "overlap (chance)" }), h("th", { text: "S1, vs own full set: AMI (cov.) [perm.]", title: "the level rebuilt on 5 subsets of 80%, against the level's own full-set sets on shared images: median joint AMI (median joint coverage, range over the 5) [largest permutation 95th pct]" }),
        h("th", { text: "λ_P1 − N3 95%", title: "first eigenvalue λ_P minus the 95th percentile over 10 trajectory-shuffle draws of the level's own operators" }), h("th", { text: "leading value (convention)" })));
      let n = 0;
      const add = (name, e) => {
        n++;
        const ag = e.agree?.[r] || e.agreement_with_M0?.[r] || (fac === "G" ? e[r] : null);
        const ov = e.overlap_with_M0?.[r] ?? e.right_overlap?.[r], ovc = e.overlap_with_M0_chance?.[r] ?? e.right_overlap_chance?.[r];
        const s1 = e.S1?.[r], cvs = s1?.coverage?.filter(Number.isFinite) || [];
        const s1t = s1 ? `${f2(median(s1.ami_joint))} (${f2(median(cvs))}, ${f2(Math.min(...cvs))}–${f2(Math.max(...cvs))}) [${f3(Math.max(...s1.null_q95.filter(Number.isFinite)))}]` : (fac === "G" ? "–" : "n/a");
        const ex = e.n3 ? (e.n3.undefined ? "n/a (1 snapshot)" : f3(num(e.n3.excess?.[0]))) : "n/a";
        const lv = e.lambda_P ? e.lambda_P[0] : e.vals ? e.vals[0] : null, cv = convOf(fac, e);
        const err = e.error || ag?.error;
        tb.append(h("tr", {}, h("td", { text: name + (e.right_overlap ? " (start)" : ""), title: e.spectrum_of || "" }),
          err ? h("td", { class: "mute", text: String(err), colspan: 3 }) : h("td", {}, bar(num(ag?.ami_joint)), h("span", { class: "mute", text: Number.isFinite(num(ag?.joint_coverage)) ? ` (${f2(num(ag?.joint_coverage))})` : "" })),
          ...(err ? [] : [h("td", { class: "mute", text: ag ? f3(num(ag.ami_joint_null_q95)) : "–" }), h("td", {}, f2(num(ag?.mj_max)), h("span", { class: "mute", text: ag && Number.isFinite(num(ag.mj_null_mean)) ? ` (${f2(num(ag.mj_null_mean))})` : "" }))]),
          h("td", {}, f2(num(ov)), h("span", { class: "mute", text: Number.isFinite(num(ovc)) ? ` (${num(ovc).toPrecision(1)})` : "" })), h("td", { text: s1t }), h("td", { text: ex }),
          h("td", { title: e.spectrum_of || cv }, lv == null ? "–" : f3(lv), lv == null ? "" : h("span", { class: "mute", text: " " + cv }))));
      };
      for (const [lev, e] of Object.entries(levels)) {
        if (lev === "matched_effective_neighbours") { for (const [k, v] of Object.entries(e.levels || {})) add(`${k}, matched n_eff`, v); continue; }
        if (lev === "commutator_fro" || lev === "kmeans_grid" || e.matrix) continue;
        add(lev, e);
      }
      const conv = [];
      for (const [lev, e] of Object.entries(levels)) if (e && e.spectrum_of && !/^P = 2M - I/.test(e.spectrum_of)) conv.push(`${lev}: ${e.spectrum_of}${e.note ? " (" + e.note + ")" : ""}`);
      const sub = fac === "F" && B.fam !== "time" ? "Time blocks only." : fac === "I" ? "Agreement with the default sets on the images both runs contain." : fac === "G" ? "SEBA threshold μ = c/√N (c = 0.99 is the default); k-means grid: class AMI only, not shown." : "";
      grid.append(h("div", { class: "bn-panel span2" }, h("h4", { text: `${fac}. ${FACTOR[fac]}` }), sub ? h("p", { class: "chart-sub", text: sub }) : "", n ? h("div", { class: "bn-fig" }, tb) : h("p", { class: "bn-note", text: "No levels stored for this block." }),
        conv.length ? h("p", { class: "bn-note", text: "Convention: " + conv.join("; ") + "." }) : ""));
    }
    const ref = B.ref_null[r];
    note.textContent = `n_eff: the median effective neighbour count 1/Σ_j P_ij², matched to the default's. Each level changes one factor of the default construction on this block and is compared with the default's sets at r = ${r}: AMI on jointly assigned images, coverage, matched Jaccard, and the subspace overlap (mean cos²) of the r − 1 leading directions with the default's (chance ${chance.toPrecision(2)}; product levels: the start span). "Leading value": the largest value that level stores, in its own convention, named beside it: λ_P = first non-trivial eigenvalue of the non-lazy block operator (constant removed); λ_M = the same for the lazy operator; σ₁ = largest singular value of the constant-projected product; "leading eig., leading vector kept" = largest eigenvalue of a normalisation (factor C) whose leading vector is not the constant and was not removed. Each agreement carries its own permutation reference (${Object.values(A.A || {}).map(e => e.agree?.[r]?.n_perm).find(Number.isFinite) ?? "n/a"} permutations, stored): "perm. 95%" for the joint AMI, the permutation mean beside matched Jaccard. "vs default" compares the level's sets with the default construction's sets on all images. "S1, vs own full set" is a different comparison: the level rebuilt on 5 subsets of 80% (the same subsets for every level) against the level's own full-set sets on the shared images; median joint AMI, then the median joint coverage with its range over the 5 subsets, and in brackets the largest of the 5 permutation 95th percentiles. "λ_P1 − N3 95%": the level's first eigenvalue minus the 95th percentile over 10 trajectory-shuffle draws of its own operators. Both exist only for levels that are Markov operators scored by the spectral summary (factors A, B, H and D's retained diagonal); "n/a" elsewhere, and on single-snapshot blocks for the shuffle.`;
  };
  draw(); onResize(body, draw);
}

// convention of an ablation level's stored leading value
function convOf(fac, e) {
  if (e.lambda_P) return e.laziness === "none" ? "λ (non-lazy operator itself)" : "λ_P";
  const so = e.spectrum_of || "";
  if (!e.vals) return "";
  if (/singular values/.test(so)) return "σ₁";
  if (e.vals_includes_leading_vector || /top vector not removed/.test(so)) return "leading eig., leading vector kept";
  if (/M\^2/.test(so)) return "λ of mean M²";
  if (/exp\(/.test(so)) return "λ of exp(−½ΣL)";
  if (/lambda_M|^M of/.test(so)) return "λ_M";
  if (fac === "F") return "λ_M (lazy, const. removed)";
  return so || "as stored";
}

// (f) linear content and CKA
async function linear(body, ds) {
  const [R, m, P, opts] = await Promise.all([reps(), D.meta(), pal(), repOptions()]), rs = R.seba_r;
  const st = { id: ds.block || "stem-logits", r: "3" };
  const c1 = h("div"), c2 = h("div"), heats = h("div", { class: "bn-panels" }), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("block"), select(opts, st.id, v => { st.id = v; draw(); }), lab("r (overlap, sets)"), seg([["3", "3"], ["6", "6"]], st.r, v => { st.r = v; draw(); })),
    h("div", { class: "bn-panels" }, h("div", { class: "bn-panel" }, h("h4", { text: "Variance capture" }), h("p", { class: "chart-sub", text: "‖Π_U G Π_U‖²_F / ‖G‖²_F of the method's span U, against the best rank-q value (G = block Gram)" }), c1),
      h("div", { class: "bn-panel" }, h("h4", { text: "Dirichlet energy under the snapshots' walks" }), h("p", { class: "chart-sub", text: "mean over the block's snapshots of f^T(I − M_c)f, M_c the lazy walk (= ½ f^T(I − P_c)f), for each unit direction f; lower = smoother" }), c2)), heats, note);
  const draw = () => {
    const B = R.reps[st.id], L = B.linear, LC = L.linear_content;
    lineChart(c1, { x: rs, xticks: rs, xfmt: r => `r = ${r}`, xname: "", yrange: [0, 1], height: 210, series: [
      { name: "method span (block Gram)", y: rs.map(r => LC[String(r)].block.value), color: SRC_COL.M0, dots: 3 },
      { name: "best rank-q span (PCA)", y: rs.map(r => LC[String(r)].block.optimum), color: SRC_COL.L1, dash: true, dots: 2.5 },
      ...(LC["2"].per_cell.length > 1 ? [{ name: "method span in each snapshot's Gram (median; band: range)", y: rs.map(r => median(LC[String(r)].per_cell.map(c => c.value))), color: "#b5b4ae", width: 1.2,
        band: [rs.map(r => Math.min(...LC[String(r)].per_cell.map(c => c.value))), rs.map(r => Math.max(...LC[String(r)].per_cell.map(c => c.value)))] }] : [])], bandMode: "each", bandLabel: "band: range over the block's snapshots" });
    const pd = L.pca_dirichlet_under_cells["8"], md = L.method_dirichlet_under_cells["8"], x = pd.mean.map((_, j) => j + 1);
    lineChart(c2, { x, xticks: x, xfmt: j => String(j), xname: "direction j", xlabel: "direction j (the method's ψ_j; the j-th principal direction)", height: 210, yrange: [0, Math.max(...pd.max, ...md.max) * 1.1], series: [
      { name: "method ψ_j", y: md.mean, color: SRC_COL.M0, dots: 3, band: [md.mean, md.max] }, { name: "PCA direction j", y: pd.mean, color: SRC_COL.L1, dash: true, dots: 2.5, band: [pd.mean, pd.max] }],
      bandMode: "each", bandLabel: "bands: mean to the largest snapshot value" });
    heats.replaceChildren();
    const names = cellNames(m, B);
    if (names.length < 2) heats.append(h("p", { class: "bn-note", text: "Single-snapshot block: the snapshot-by-snapshot comparisons need at least two snapshots." }));
    else {
      const sa = L.set_agree_cells?.[st.r];
      const mats = [["debiased linear CKA", L.cka_debiased, null], ["subspace overlap of the leading r − 1 eigenfunctions (mean cos²)", L.subspace_overlap_cells?.[st.r], null],
        ["set agreement: AMI of the single-snapshot sets (joint)", sa?.ami_joint, sa]];
      const jobs = [];
      for (const [title, z, extra] of mats) {
        if (!z) continue;
        const box = h("div", { class: "bn-fig" });
        heats.append(h("div", { class: "bn-panel" }, h("h4", { text: title }), box));
        jobs.push(() => heat(box, { rows: names, cols: names, z, zrange: [0, 1], color: t => P.seq(t), ml: 52, mb: 50,
          mark: extra ? (i, j) => i !== j && z[i][j] <= extra.null_q95[i][j] : null,
          tip: (i, j) => [{ name: title.split(":")[0], value: f3(z[i][j]) }, ...(extra ? [{ name: "coverage", value: f2(extra.coverage[i][j]) }, { name: "permutation 95th pct", value: f3(extra.null_q95[i][j]) }] : []),
            ...(title.includes("CKA") ? [{ name: "linear CKA (biased)", value: f3(L.cka[i][j]) }] : [])] }));
      }
      jobs.forEach(f => f());
    }
    const pnd = L.cka_debiased_perm_null, pnb = L.cka_perm_null_biased || L.cka_perm_null, pc = L.pca_C_vs_HCH?.[0];
    note.textContent = `Debiased CKA uses U-centred Grams. Permutation reference for the debiased CKA plotted (snapshot pair ${(L.cka_perm_null_pair || [0, 0]).map(i => names[i] ?? i).join(" and ")}, ${pnd?.n ?? "n/a"} image permutations): 95th percentile ${f3(pnd?.q95)}; for the biased CKA (hover): ${f3(pnb?.q95)}. Subspace overlap chance: q/(N − 1). The set agreement (r = ${st.r}) is computed by the site build from the stored single-snapshot sets, ${R.n_perm_site} permutations; a white dot marks a pair at or below its permutation 95th percentile.` +
      (pc ? ` Top eight eigenvectors of C against those of HCH, first snapshot: overlap ${f3(pc.mean_cos2)} (mean cos²).` : "");
  };
  draw(); onResize(body, draw);
}

// (g) planted benchmarks (NOTES_B.md chart spec)
let BENCH = null;
async function bench(body, ds) {
  if (!BENCH) BENCH = await D.json("d/bn/bench.json");
  const B = BENCH, idx = new Map();
  for (const r of B.rows) idx.set([r.bench, r.level, r.N, r.block, r.r, r.method, r.stat].join("|"), r);
  const charts = B.charts.map((c, i) => [String(i), `${c.bench}${c.chart > 1 ? " (" + c.chart + ")" : ""}: ${c.caption.split(";")[0].replace(/\.$/, "")}`]);
  const st = { i: ds.chart || "0", N: "2000", blk: "all" };
  const extra = h("span"), box = h("div", { class: "bn-panels" }), cap = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("benchmark"), select(charts, st.i, v => { st.i = v; draw(); }), extra), box, cap);
  const levelsOf = c => { const s = new Set(B.rows.filter(r => r.bench === c.bench).map(r => r.level)); return c.categorical ? c.categorical.filter(l => s.has(l)) : [...s].sort((a, b) => +a - +b); };
  const get = (c, lev, method, stat, r, N, blk) => idx.get([c.bench, lev, N, blk, r ?? null, method, stat].join("|"));
  const draw = () => {
    const c = B.charts[+st.i], levels = levelsOf(c), xs = c.categorical ? levels.map((_, i) => i) : levels.map(Number);
    extra.replaceChildren(...(c.byN ? [lab("N"), seg([["500", "500"], ["1000", "1,000"], ["2000", "2,000"]], st.N, v => { st.N = v; draw(); })] : []),
      ...(c.blocks ? [lab("block"), seg(c.blocks.map(b => [b, b]), c.blocks.includes(st.blk) ? st.blk : (st.blk = "all"), v => { st.blk = v; draw(); })] : []));
    const N = c.byN ? +st.N : 2000, blk = c.blocks ? st.blk : "all";
    box.replaceChildren();
    const jobs = [];
    const xfmt = c.categorical ? (i => levels[i]) : (v => (+v).toString());
    const panelRefs = c.ref.filter(rf => rf.panel);
    if (panelRefs.length) {
      const pbox = h("div");
      box.append(h("div", { class: "bn-panel" }, h("h4", { text: "oracle" }), pbox));
      jobs.push(() => lineChart(pbox, { x: xs, xticks: xs, xfmt, xname: c.x, xlabel: c.x, xscale: c.xlog ? "log" : "linear", height: 170, yrange: [0, 1],
        series: panelRefs.map(rf => ({ name: rf.label, y: levels.map(l => num(get(c, l, "oracle", rf.stat, null, N, "all")?.median)), color: "#8d8c86", dots: 2.5,
          band: [levels.map(l => num(get(c, l, "oracle", rf.stat, null, N, "all")?.q10)), levels.map(l => num(get(c, l, "oracle", rf.stat, null, N, "all")?.q90))] })) }));
    }
    const perLevel = c.percell;
    for (const s of perLevel ? [null] : c.series) {
      const pb = h("div");
      box.append(h("div", { class: "bn-panel" }, h("h4", { text: perLevel ? "retention above the size null, per cell (the method, r = 4)" : s.label + (s.r ? ` (r = ${s.r})` : "") }), pb));
      let series, x = xs, xf = xfmt, xname = c.x;
      if (perLevel) {
        x = [...Array(c.percell).keys()]; xf = i => `cell ${i}`; xname = "cell index";
        series = levels.map((l, li) => ({ name: `${l} noise cell${l === "1" ? "" : "s"}`, color: ["#16161a", "#5a5a5a", "#8d8c86", "#b5b4ae"][li % 4], dots: 2.5, width: 1.6,
          y: x.map(j => num(get(c, l, "M0", `ret_excess_cell${j}`, 4, N, blk)?.median)) }));
      } else {
        series = c.methods.map(meth => {
          const stat = meth.startsWith("T_") && s.Tstat ? s.Tstat : s.stat, rr = meth.startsWith("T_") && s.Tstat ? null : s.r;
          const rows = levels.map(l => get(c, l, meth, stat, rr, N, blk));
          return { name: BENCH_NAME[meth] || srcName(meth), y: rows.map(q => num(q?.median)), band: [rows.map(q => num(q?.q10)), rows.map(q => num(q?.q90))], color: SRC_COL[meth] || "#555", dash: DASH.has(meth), width: meth === "M0" ? 2.4 : 1.7, dots: 2.5 };
        });
        for (const rf of c.ref.filter(rf => rf.perMethod)) for (const meth of c.methods) {
          const rows = levels.map(l => get(c, l, meth, rf.stat, rf.r ?? null, N, blk));
          if (rows.some(q => q)) series.push({ name: `${BENCH_NAME[meth] || srcName(meth)}: ${rf.label}`, y: rows.map(q => num(q?.median)), color: SRC_COL[meth] || "#555", dash: true, width: 1 });
        }
        for (const rf of c.ref.filter(rf => !rf.panel && !rf.band && !rf.perMethod)) {
          const rows = levels.map(l => get(c, l, rf.method || "oracle", rf.stat, rf.r ?? null, N, rf.method ? blk : "all"));
          if (rows.every(q => !q)) continue;
          series.push({ name: rf.label, y: rows.map(q => num(q?.median)), color: rf.chance ? "#c9c8c1" : "#9a9993", dash: true, width: 1.3 });
        }
      }
      const bandRef = c.ref.find(rf => rf.band);
      const bandRows = bandRef ? levels.map(l => get(c, l, bandRef.method, bandRef.stat, bandRef.r ?? null, N, blk)) : null;
      const unit = !perLevel && /^(ami|contrast_captured|span_|best_jaccard|test_corr|src_span)/.test(s.stat);
      const all = series.flatMap(q => q.y.concat(q.band ? q.band[1] : [])).filter(Number.isFinite), lo = Math.min(0, ...all), hi = unit ? Math.max(1, ...all) : Math.max(...all) * 1.08 || 1;
      jobs.push(() => lineChart(pb, { x, xticks: x, xfmt: xf, xname, xlabel: xname, xscale: c.xlog && !perLevel ? "log" : "linear", height: 250, series, bandMode: "envelope", yrange: unit ? [Math.floor(lo * 10) / 10, Math.ceil(hi * 10) / 10] : [Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20],
        band: bandRows ? [bandRows.map(() => 0), bandRows.map(q => num(q?.median))] : undefined, bandLabel: bandRef ? bandRef.label + " (median over seeds), from 0" : undefined,
        refline: c.zero ? { y: 0, label: "" } : null }));
    }
    jobs.forEach(f => f());
    cap.innerHTML = `<b>Plotted:</b> ${c.caption} Median over 10 seeds; bands: 10th to 90th percentile over seeds (hover a key to see one method's band). N = ${N.toLocaleString("en")}${c.blocks ? `, block ${blk}` : ""}. Reference lines: ${c.ref.length ? c.ref.map(r => r.perMethod ? r.label + " of each method (dashed, thin, in its colour)" : r.label).join("; ") : "none"}${c.zero ? "; AMI and retention excess are 0 at chance" : ""}. Benchmark ${c.bench}: definitions in the table above.`;
  };
  draw(); onResize(body, draw);
}

// (h) transfer-operator pilot
let TR = null;
const tr = async () => { if (!TR) TR = await D.json("d/bn/transfer.json"); return TR; };
const EST = { linear: "linear CCA", soft_ulam: "soft Ulam", nystrom: "Nyström", diffusion: "diffusion coordinates" };
const ESTC = { linear: SRC_COL.T_linear, soft_ulam: SRC_COL.T_soft_ulam, nystrom: SRC_COL.T_nystrom, diffusion: SRC_COL.T_diffusion };
const pairName = (m, p) => p.src.step === p.dst.step ? `${p.src.site} → ${p.dst.site} (step ${p.src.step.toLocaleString("en")})` : `${p.src.site}: step ${p.src.step.toLocaleString("en")} → ${p.dst.step.toLocaleString("en")}`;

async function trSyn(body) {
  const T = await tr(), C = T.curve, j0 = { j: "0" };
  const c = h("div"), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("function"), seg([0, 1, 2, 3, 4, 5].map(j => [String(j), String(j + 1)]), j0.j, v => { j0.j = v; draw(); })), c, note);
  const draw = () => {
    const j = +j0.j, ns = [...new Set(Object.values(C.series).flatMap(s => Object.keys(s).map(Number)))].sort((a, b) => a - b);
    const series = Object.entries(C.series).filter(([k]) => k.endsWith("_m64") || k.endsWith("_m128")).map(([k, s]) => {
      const e = k.replace(/_m\d+$/, ""), mm = k.match(/m(\d+)$/)[1];
      return { name: `${EST[e]}, m = ${mm}`, y: ns.map(n => num(s[String(n)]?.[j])), color: ESTC[e], dash: mm === "128", dots: 2.5, width: 1.7 };
    });
    const ctl95 = Object.values(T.syn_p8 || {}).map(u => u.control_q95?.[j]).filter(Number.isFinite);
    lineChart(c, { x: ns, xticks: ns, xscale: "log", xfmt: v => v.toLocaleString("en"), xname: "fit images", xlabel: "fit images N_fit", yrange: [-.1, 1], height: 260, series,
      refline: { y: C.truth[j], label: `known value exp(−m²σ²/2) = ${f3(C.truth[j])}` }, band: ctl95.length ? [ns.map(() => -.1), ns.map(() => Math.max(...ctl95))] : undefined,
      bandLabel: "independence control at N_fit = 1,000: largest 95th percentile over estimators" });
    note.textContent = `Noisy circle doubling (benchmark P8), σ = ${C.sigma}: the transfer operator's singular values are known, exp(−m²σ²/2), two per harmonic m. Plotted: held-out correlation of fitted function ${j + 1}, mean over ${Math.max(...Object.values(C.n_seeds))} seeds, against the number of fit images. Linear CCA has two functions only.`;
  };
  draw(); onResize(body, draw);
}

async function trSupport(body) {
  const [T, m, P] = await Promise.all([tr(), D.meta(), pal()]);
  if (!T.real) { body.replaceChildren(h("p", { class: "placeholder", text: "transfer_pilot.json not in this build" })); return; }
  const box = h("div", { class: "bn-fig" }), note = h("p", { class: "bn-note" });
  const variants = Object.keys(T.real.pairs[0].results).sort();
  body.replaceChildren(box, note);
  const draw = () => heat(box, { rows: T.real.pairs.map(p => pairName(m, p)), cols: variants.map(v => v.replace("_m", ", m").replace("mjoint", "m selected").replace("soft_ulam", "soft Ulam").replace("nystrom", "Nyström")),
    z: T.real.pairs.map(p => variants.map(v => p.results[v].leading)), zrange: [0, 6], color: t => P.seq(t), ml: 190, mb: 110,
    tip: (r, c) => { const v = T.real.pairs[r].results[variants[c]]; return [{ name: "leading supported functions", value: String(v.leading) }, { name: "splits above control, functions 1..6", value: (v.n_support || []).join(" ") },
      { name: "held-out VAMP-2, median", value: f2(v.vamp2_med) }, { name: "control VAMP-2 95th pct, median", value: f3(v.ctl_vamp2_med) }]; } });
  note.textContent = `Colour: number of consecutive fitted functions, from the first, whose held-out correlation exceeds the 95th percentile of the independence control in at least ${T.real.head.n_splits >= 10 ? 8 : Math.ceil(.8 * T.real.head.n_splits)} of ${T.real.head.n_splits} splits (0 to 6). This 8-of-10 rule is a descriptive support criterion fixed before the run (splits overlap; several functions and pairs are inspected), not a 95% confidence statement.`;
  draw(); onResize(body, draw);
}

async function trReal(body, ds) {
  const [T, m] = await Promise.all([tr(), D.meta()]);
  if (!T.real) { body.replaceChildren(h("p", { class: "placeholder", text: "transfer_pilot.json not in this build" })); return; }
  const st = { p: "0", e: "nystrom", mm: "mjoint" };
  const c = h("div"), info = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("pair"), select(T.real.pairs.map((p, i) => [String(i), pairName(m, p)]), st.p, v => { st.p = v; draw(); }),
    seg(Object.entries(EST).map(([k, v]) => [k, v]), st.e, v => { st.e = v; draw(); }), seg([["m64", "m = 64"], ["m128", "m = 128"], ["mjoint", "m selected"]], st.mm, v => { st.mm = v; draw(); })), c, info);
  const draw = () => {
    const p = T.real.pairs[+st.p], v = p.results[`${st.e}_${st.mm}`], x = [1, 2, 3, 4, 5, 6];
    lineChart(c, { x, xticks: x, xfmt: j => `f${j}`, xname: "function", yrange: [-.2, 1], height: 250, series: [
      { name: "held-out correlation, median over splits", y: v.test_med, color: ESTC[st.e], dots: 3, band: [v.test_min, v.test_max] },
      { name: "independence control, 95th pct (median over splits)", y: v.ctl_med, color: "#8d8c86", dash: true, width: 1.4, band: [v.ctl_med.map(() => -.2), v.ctl_max] }],
      bandMode: "each", bandLabel: "bands: range over the 10 splits (correlation); control up to its largest 95th percentile" });
    const ag = v.agreement || {}, sb = v.stability || {};
    const ov = k => ag[k] ? `${f2(ag[k].mean_cos2_median)} (chance q/(N − 1) = ${f3(ag[k].chance)})` : "n/a";
    info.innerHTML = `${pairName(m, p)}, ${EST[st.e]}, ${st.mm === "mjoint" ? "m selected by tuning" : st.mm.replace("m", "m = ")} (pilot). Leading supported functions: <b>${v.leading}</b> (${v.rule}). Held-out VAMP-2 ${f2(v.vamp2_med)} against a control 95th percentile of ${f3(v.ctl_vamp2_med)} (medians over splits). ` +
      `Overlap of the leading source functions with the source span of the existing product Q = M<sub>b</sub>M<sub>a</sub>: ${ov("Q_src")}; with the two-snapshot average M̄: ${ov("Mbar_src")}; with linear CCA: ${ag.linear_src ? `${f2(ag.linear_src.mean_cos2_median)} (chance q/(N − 1) = ${f3(ag.Q_src?.chance)})` : "n/a"} (all images, transductive). ` +
      `Held-out stability of the leading functions across splits: mean cos² between two splits' leading q-dimensional function spans on the images held out from both fits (${T.real.head.stab_n_shared?.join(" to ")} images per pair of splits), averaged over the ${sb.src?.n_pairs ?? "n/a"} pairs: source ${f2(sb.src?.mean_pairwise_cos2)} (q = ${sb.src?.q ?? "n/a"}; chance q/(n − 1), averaged over the pairs, ${f3(v.stab_chance?.src)}), destination ${f2(sb.dst?.mean_pairwise_cos2)} (q = ${sb.dst?.q ?? "n/a"}; chance ${f3(v.stab_chance?.dst)}). Reference: product singular values ${p.reference.Q_sv.slice(0, 3).map(f2).join(", ")}…; these are on another scale (uniform mass, not validated on held-out images).`;
  };
  draw(); onResize(body, draw);
}

async function tr10k(body) {
  const [T, m, P] = await Promise.all([tr(), D.meta(), pal()]);
  if (!T.n10k) { body.replaceChildren(h("p", { class: "placeholder", text: "transfer_pilot_10k.json not in this build" })); return; }
  const st = { p: "0", v: "nystrom_m64" };
  const vars = Object.keys(T.n10k.pairs[0].curve[String(T.n10k.head.curve[0])]);
  const c = h("div"), note = h("p", { class: "bn-note" });
  body.replaceChildren(ctl(lab("pair"), select(T.n10k.pairs.map((p, i) => [String(i), `${p.id} (${p.kind})`]), st.p, v => { st.p = v; draw(); }),
    lab("estimator"), select(vars.map(v => [v, v.replace("_m", ", m = ").replace("= joint", "selected").replace("soft_ulam", "soft Ulam").replace("nystrom", "Nyström")]), st.v, v => { st.v = v; draw(); })), c, note);
  const draw = () => {
    const p = T.n10k.pairs[+st.p], ns = T.n10k.head.curve;
    const series = [0, 1, 2, 3, 4, 5].map(j => ({ name: `function ${j + 1}`, y: ns.map(n => num(p.curve[String(n)]?.[st.v]?.test_corr?.[j])), color: P.ord(.25 + .75 * j / 5), dots: 2.5, width: 1.6 }));
    const ctlmax = ns.map(n => Math.max(...(p.curve[String(n)]?.[st.v]?.control_q95 || [NaN]).filter(Number.isFinite)));
    lineChart(c, { x: ns, xticks: ns, xscale: "log", xfmt: v => v.toLocaleString("en"), xname: "fit images", xlabel: "fit images N_fit (nested subsets of the 6,000-image fit split)", yrange: [-.2, 1], height: 260, series,
      band: [ns.map(() => -.2), ctlmax], bandLabel: `independence control: largest 95th percentile over the six functions (${T.n10k.head.n_perm} permutations)` });
    note.textContent = `Pair ${p.id}, final checkpoint, the 10,000 CIFAR-10 test images split ${T.n10k.head.split_sizes.map(x => x.toLocaleString("en")).join(" / ")} (fit / tuning / test, one predeclared split). Plotted: held-out correlation of each of the first six fitted functions on the 2,000 test images.`;
  };
  draw(); onResize(body, draw);
}

async function trComp(body) {
  const [T, m] = await Promise.all([tr(), D.meta()]);
  if (!T.real) { body.replaceChildren(h("p", { class: "placeholder", text: "transfer_pilot.json not in this build" })); return; }
  const C = T.real.composition, st = { t: "0", e: "nystrom" };
  const c = h("div"), c2 = h("div"), note = h("p", { class: "bn-note" });
  const ests = [...new Set(C.flatMap(x => Object.keys(x.est)))];
  body.replaceChildren(ctl(lab("chain"), select(C.map((x, i) => [String(i), x.views.map(v => `${v.site}@${fmtStep(v.step)}`).join(" → ")]), st.t, v => { st.t = v; draw(); }),
    seg(ests.map(e => [e, EST[e] || e]), st.e, v => { st.e = v; draw(); })),
    h("div", { class: "bn-panels" }, h("div", { class: "bn-panel" }, h("h4", { text: "Real chain (pilot)" }), c), h("div", { class: "bn-panel" }, h("h4", { text: "Calibration chains (synthetic)" }), c2)), note);
  const draw = () => {
    const FACT = [0.25, 0.5, 1, 2, 4], fl = f => ({ 0.25: "×¼", 0.5: "×½", 1: "×1", 2: "×2", 4: "×4" }[f]), x = FACT.map((_, i) => i);
    const E = C[+st.t].est[st.e] || {}, fs = FACT.map(String), row = f => E[f] && E[f].mb != null ? E[f] : null;
    const ser = (k, name, col, dash) => ({ name, y: fs.map(f => num(row(f)?.[k])), color: col, dash, dots: 2.5, width: 1.7 });
    lineChart(c, { x, xticks: x, xfmt: i => `${fl(FACT[i])}` + (row(fs[i]) ? ` (${Math.round(row(fs[i]).mb)})` : ""), xname: "intermediate basis", xlabel: "intermediate basis: factor × base size (actual m_b, median)", height: 230, yrange: [0, 1.2], series: [
      ser("partial", "partial term", "#16161a"), ser("regulariser", "regulariser term", "#8d8c86", true), ser("heldout", "held-out residual cross-moment", ESTC[st.e] || "#2f5d5d")] });
    const ch = T.chain?.rows || {}, mk = ch[`markov_${st.e}`] || [], br = ch[`broken_${st.e}`] || [];
    const at = (rows, k) => FACT.map(f => { const r = rows.find(q => q.factor === f && !q.unavailable); return r ? num(r[k]) : NaN; });
    lineChart(c2, { x, xticks: x, xfmt: i => fl(FACT[i]), xname: "intermediate basis", xlabel: "intermediate basis: factor × base size", height: 230, yrange: [0, 1.2], series: [
      { name: "Markov chain: partial term", y: at(mk, "partial"), color: "#16161a", dots: 2.5 }, { name: "Markov chain: held-out residual", y: at(mk, "heldout"), color: ESTC[st.e] || "#2f5d5d", dots: 2.5 },
      { name: "broken chain: partial term", y: at(br, "partial"), color: "#16161a", dash: true, dots: 2.5 }, { name: "broken chain: held-out residual", y: at(br, "heldout"), color: ESTC[st.e] || "#2f5d5d", dash: true, dots: 2.5 }] });
    const dims = rows => FACT.map(f => { const r = rows.find(q => q.factor === f); return !r ? "–" : r.unavailable ? "n/a" : String(r.mb); }).join(", ");
    const un = fs.some(f => row(f)) ? 0 : 1;
    note.textContent = `The discrepancy D = T_bc T_ab − T_ac on a chain a → b → c is, exactly, a regulariser term (ridge at b; for soft Ulam, the soft-membership overlap) plus a partial term (the whitened fit-split partial cross-moment of the a- and c-features given the b-features). Plotted, left: the norm of the partial term, of the regulariser term and of the held-out residual cross-moment (a separate diagnostic, not a part of D); right: the partial term and the held-out residual only. Each norm is taken on the top six singular pairs of T_ac and divided by their scale, median over ${C[+st.t].n_splits} splits. Norms of the two terms do not add up to the norm of D, which is not plotted. x axis: the requested size of the intermediate basis as a factor of the base size (endpoints, ridge and bandwidths frozen), with the actual dimension m_b in brackets on the left; on the right, actual m_b at ×¼..×4: Markov ${dims(mk)}; broken ${dims(br)} (n/a: below the occupancy gate). The pilot does not separate a finite basis, view insufficiency and sampling inside the partial term. Calibration: a chain that is Markov by construction (θ → 2θ → 4θ with independent noise) and the same chain with the middle view re-paired at random, ${T.chain?.n_fit?.toLocaleString("en") ?? ""} fit images.` + (un ? " No rows stored for this estimator and chain." : "");
  };
  draw(); onResize(body, draw);
}
