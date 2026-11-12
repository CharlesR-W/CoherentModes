// Live mini-figures embedded in prose (see DESIGN.md): decod, curves, lines, heatmap, kernel, cells, images.
import * as D from "./data.js?v=00a9dd9888";
import { Palette } from "./colors.js?v=00a9dd9888";
import { h, el, lineChart, heatmap, fmtStep, showTip, hideTip, tipRows, scale, niceTicks } from "./charts.js?v=00a9dd9888";

const seg = (opts, val, on) => {
  const s = h("div", { class: "seg", role: "radiogroup" });
  const sync = v => s.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === v)));
  opts.forEach(([v, label]) => { const b = h("button", { type: "button", "data-v": v, role: "radio", text: label }); b.addEventListener("click", () => { sync(v); on(v); }); s.append(b); });
  sync(val); return s;
};
const onResize = (elm, fn) => { let w = elm.clientWidth; new ResizeObserver(() => { if (Math.abs(elm.clientWidth - w) > 4) { w = elm.clientWidth; fn(); } }).observe(elm); };

export async function render(kind, body, ds) {
  const fn = { decod, curves, lines, heatmap: heat, kernel, cells, images }[kind];
  if (!fn) throw new Error(`unknown live figure "${kind}"`);
  await fn(body, ds);
}

async function pal() { return new Palette((await D.meta()).palette); }
function getPath(obj, path) { return path ? path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj) : obj; }
async function fromSrc(src) {
  const [url, path] = src.split("#");
  const obj = getPath(await D.json(url), path);
  if (obj == null) throw new Error(`${src}: not found`);
  return obj;
}

// ---------------------------------------------------------------------------------------------- decodability
async function decod(body) {
  const [d, P, m] = await Promise.all([D.json("d/decod.json"), pal(), D.meta()]);
  const st = { view: d.views.includes("gap") ? "gap" : d.views[0], metric: "ridge_acc" };
  body.replaceChildren();
  const ctl = h("div", { class: "chart-controls" },
    h("span", { class: "ctl-label", text: "features" }), seg(d.views.map(v => [v, v === "gap" ? "global average pool (per channel)" : v === "flat" ? "flattened (all positions)" : v]), st.view, v => { st.view = v; draw(); }),
    h("span", { class: "ctl-label", text: "readout" }), seg([["ridge_acc", "ridge (closed-form fit)"], ["knn_acc", "cosine kNN vote"]], st.metric, v => { st.metric = v; draw(); }));
  const grid = h("div", { class: "charts-2" });
  const c1 = h("div"), c2 = h("div");
  grid.append(h("div", {}, h("p", { class: "chart-title", text: "By site, one line per checkpoint" }), h("p", { class: "chart-sub", text: "test accuracy (10 classes; chance 10%)" }), c1),
    h("div", {}, h("p", { class: "chart-title", text: "By training step, one line per site" }), h("p", { class: "chart-sub", text: "test accuracy; step on a log scale" }), c2));
  body.append(ctl, grid);
  if (d.fixture || d.protocol) body.append(h("p", { class: "chart-sub", text: (d.fixture ? "SYNTHETIC FIXTURE. " : "") + (d.protocol || "") }));
  const steps = d.results.map(r => r.step), T = steps.length;
  const draw = () => {
    const val = (ri, site) => { const v = d.results[ri].views[st.view]?.[site]?.[st.metric]; return Number.isFinite(v) ? v : NaN; };
    lineChart(c1, { x: d.sites.map((_, i) => i), xticks: d.sites.map((_, i) => i).filter(i => d.sites.length < 13 || i % 2 === 0), xfmt: i => d.sites[i], xname: "site",
      series: d.results.map((r, ri) => ({ name: `step ${r.step.toLocaleString("en")}`, y: d.sites.map(s => val(ri, s)), color: P.ord(T > 1 ? ri / (T - 1) : 1), width: 1.6 })),
      yrange: [0, 1], yfmt: v => (100 * v).toFixed(0) + "%", height: 280, legend: false, refline: { y: .1, label: "chance" } });
    lineChart(c2, { x: steps.map(s => Math.max(s, 0)), xscale: "symlog", xfmt: fmtStep, xname: "step",
      series: d.sites.map((s, si) => ({ name: s, y: steps.map((_, ri) => val(ri, s)), color: P.ord(si / (d.sites.length - 1)), width: 1.6 })),
      yrange: [0, 1], yfmt: v => (100 * v).toFixed(0) + "%", height: 280, legend: false, refline: { y: .1, label: "chance" } });
    // compact ordinal keys
    c1.append(rampKey(P, `step ${fmtStep(steps[0])}`, `step ${fmtStep(steps[T - 1])}`));
    c2.append(rampKey(P, d.sites[0], d.sites[d.sites.length - 1]));
  };
  draw(); onResize(body, draw);
}
function rampKey(P, a, b) {
  return h("div", { class: "chart-legend" }, h("span", { class: "k" }, a, h("i", { style: { width: "90px", height: "6px", borderRadius: "3px", background: `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(t => P.ord(t)).join(",")})` } }), b), h("span", { class: "k mute", text: "hover for values" }));
}

// ---------------------------------------------------------------------------------------------- training curves
async function curves(body) {
  const [c, P, m] = await Promise.all([D.json("d/curves.json"), pal(), D.meta()]);
  const ck = c.checkpoints.slice().sort((a, b) => a.step - b.step), x = ck.map(r => r.step);
  body.replaceChildren();
  const g = h("div", { class: "charts-2" }), a = h("div"), l = h("div");
  g.append(h("div", {}, h("p", { class: "chart-title", text: "Accuracy" }), h("p", { class: "chart-sub", text: "dots: the checkpoints shown in the explorer" }), a),
    h("div", {}, h("p", { class: "chart-title", text: "Loss" }), h("p", { class: "chart-sub", text: "cross-entropy" }), l));
  body.append(g);
  if (c.fixture) body.append(h("p", { class: "chart-sub", text: "SYNTHETIC FIXTURE" }));
  const col = [P.sets[0], P.sets[1]];
  const draw = () => {
    const ca = lineChart(a, { x, xscale: "symlog", xfmt: fmtStep, xname: "step", yrange: [0, 1], yfmt: v => (100 * v).toFixed(0) + "%", height: 240,
      series: [{ name: "train", y: ck.map(r => r.train_acc), color: col[0] }, { name: "validation", y: ck.map(r => r.val_acc), color: col[1] }] });
    const lmax = Math.max(...ck.flatMap(r => [r.train_loss, r.val_loss]).filter(Number.isFinite));
    lineChart(l, { x, xscale: "symlog", xfmt: fmtStep, xname: "step", yrange: [0, Math.ceil(lmax * 2) / 2], yfmt: v => v.toFixed(1), height: 240,
      series: [{ name: "train", y: ck.map(r => r.train_loss), color: col[0] }, { name: "validation", y: ck.map(r => r.val_loss), color: col[1] }] });
    m.steps.forEach(s => { const r = ck.find(q => q.step === s); if (r && Number.isFinite(r.val_acc)) el("circle", { cx: ca.X(s), cy: ca.Y(r.val_acc), r: 4, fill: col[1], stroke: "#fff", "stroke-width": 2 }, ca.svg); });
  };
  draw(); onResize(body, draw);
}

// ---------------------------------------------------------------------------------------------- generic JSON charts
async function lines(body, ds) {
  const [o, P] = await Promise.all([fromSrc(ds.src), pal()]);
  body.replaceChildren();
  const box = h("div", { class: "live-chart" }); body.append(box);
  // optional reference from another object (data-ceiling-src): a heatmap whose columns match x gives a dashed
  // per-x series (mean over its rows); anything else gives one horizontal line at its maximum
  let ceiling = null, ceilSeries = null;
  if (ds.ceilingSrc) {
    const c = await fromSrc(ds.ceilingSrc);
    if (Array.isArray(c.z) && Array.isArray(c.z[0]) && c.z[0].length === o.x.length) {
      const y = o.x.map((_, j) => { const v = c.z.map(r => r[j]).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; });
      ceilSeries = { name: ds.ceilingLabel || "size ceiling (mean over sites)", y, color: "#8d8c86", dash: true, width: 1.5 };
    } else {
      const vals = JSON.stringify(c.z ?? c.y ?? c).match(/-?\d+(\.\d+)?(e-?\d+)?/g)?.map(Number).filter(Number.isFinite) || [];
      if (vals.length) ceiling = { y: Math.max(...vals), label: ds.ceilingLabel || "ceiling" };
    }
  }
  const n = o.series.length;
  const color = (s, i) => s.color || (o.colors === "ordinal" ? P.ord(n > 1 ? i / (n - 1) : 1) : o.colors === "classes" ? P.classes[i % 10] : P.set(i));
  const lab = o.xticklabels;
  const draw = () => {
    const W = box.clientWidth || 600, every = lab ? Math.max(1, Math.ceil(lab.length * 64 / Math.max(200, W - 80))) : 1;
    const series = o.series.map((s, i) => ({ name: s.name, y: s.y, color: color(s, i), dash: s.dash, band: s.band, width: s.width || 2 }));
    if (ceilSeries) series.push(ceilSeries);
    lineChart(box, { x: o.x, xscale: o.xscale || "linear",
      xfmt: lab ? (v => lab[o.x.indexOf(v)] ?? String(v)) : (o.xscale && o.xscale !== "linear" ? fmtStep : (v => (+(+v).toPrecision(4)).toString())),
      xticks: lab ? o.x.filter((_, i) => i % every === 0) : undefined,
      xname: o.xlabel || "", xlabel: o.xlabel, ytitle: o.ylabel, mr: lab ? 40 : undefined, yrange: o.yrange, height: +ds.height || 280, refline: o.refline || ceiling, series,
      band: o.band, bandMode: o.band_mode, bandLabel: o.band_label });
  };
  draw(); onResize(body, draw);
}
// value format of a heatmap: o.fmt = "int" | "pct" (x100, %) | "<n>f" (n decimals) | "<n>g" (n significant figures);
// default: integers when every finite z is an integer (counts), else 3 significant figures
export function zFormat(o) {
  const m = typeof o.fmt === "string" && o.fmt.match(/^(\d)([fg])$/);
  if (o.fmt === "int") return v => Math.round(v).toLocaleString("en");
  if (o.fmt === "pct") return v => (+(100 * v).toPrecision(3)).toString() + "%";
  if (m) return m[2] === "f" ? (v => v.toFixed(+m[1])) : (v => (+v.toPrecision(Math.max(1, +m[1]))).toString());
  const flat = JSON.stringify(o.z).match(/-?\d+(\.\d+)?(e-?\d+)?/g) || [];
  if (flat.length && flat.every(x => Number.isInteger(+x))) return v => Math.round(v).toLocaleString("en");
  return v => Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en") : (+v.toPrecision(3)).toString();
}
async function heat(body, ds) {
  const [o, P] = await Promise.all([fromSrc(ds.src), pal()]);
  body.replaceChildren();
  const box = h("div"); body.append(box);
  const opts = { ...o, color: o.diverging ? (t => P.div(2 * t - 1)) : (t => P.seq(t)), zfmt: zFormat(o) };
  heatmap(box, opts);
  onResize(body, () => heatmap(box, opts));
}

// ---------------------------------------------------------------------------------------------- kernel (kappa) demo
async function kernel(body, ds) {
  const [m, P] = await Promise.all([D.meta(), pal()]);
  const T = m.steps.length, t = ds.t != null ? +ds.t : T - 1, s = ds.s != null ? +ds.s : Math.min(5, m.sites.length - 1);
  const kap = m.modes.map((_, i) => m.kappa?.[i]?.[t]?.[s] ?? null);
  // d_15 is implied by the kNN-scale rule (kappa = 1 / d_15); found by name, not by index
  const iScale = m.modes.findIndex(n => /knn.?scale/i.test(n));
  const dk = iScale >= 0 && kap[iScale] > 0 ? 1 / kap[iScale] : NaN;
  const pick = ds.m != null && kap[+ds.m] != null ? +ds.m : (iScale >= 0 && kap[iScale] != null ? iScale : kap.findIndex(k => k != null));
  let kappa = pick >= 0 ? kap[pick] : 1;
  body.replaceChildren();
  const out = h("b"), inp = h("input", { type: "range", min: Math.log(0.5), max: Math.log(200), step: 0.01, value: Math.log(kappa), style: { width: "240px" } });
  const chips = h("div", { class: "seg" });
  m.modes.forEach((name, i) => {
    if (kap[i] == null) return;
    const label = /perplexity/i.test(name) ? `${name} (median κᵢ)` : name;
    const b = h("button", { type: "button", text: label, "data-i": i });
    b.addEventListener("click", () => { kappa = kap[i]; inp.value = Math.log(kappa); draw(); }); chips.append(b);
  });
  const ctl = h("div", { class: "chart-controls" }, h("label", {}, "κ = ", out, " "), inp, chips);
  const box = h("div"), read = h("p", { class: "chart-sub" });
  const noKappa = m.modes.filter((_, i) => kap[i] == null);
  body.append(ctl, box, read, h("p", { class: "chart-sub", text: `Snapshot: ${m.sites[s]} at step ${m.steps[t].toLocaleString("en")}. The buttons set \u03ba to each rule's value for this snapshot` +
    (noKappa.length ? ` (${noKappa.join(", ")} is rank-based and has no \u03ba)` : "") + "." }));
  inp.addEventListener("input", () => { kappa = Math.exp(+inp.value); draw(); });
  const draw = () => {
    out.textContent = kappa >= 10 ? kappa.toFixed(1) : kappa.toFixed(2);
    chips.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(Math.abs(kap[+b.dataset.i] - kappa) < 1e-9)));
    const x = Array.from({ length: 121 }, (_, i) => i * 1.2 / 120);
    const c = lineChart(box, { x, xfmt: v => v.toFixed(1), xname: "cosine distance", xlabel: "cosine distance 1 − cos", yrange: [0, 1], height: 220, yfmt: v => (+v.toFixed(2)).toString(),
      series: [{ name: "kernel weight e^{−κ d}", y: x.map(d => Math.exp(-kappa * d)), color: P.sets[0] }] });
    for (const [d, lab] of [[dk, "d₁₅"], [1, "d = 1"]]) if (Number.isFinite(d) && d <= 1.2) {
      el("line", { x1: c.X(d), x2: c.X(d), y1: c.Y(0), y2: c.Y(1), stroke: "#9a9993", "stroke-width": 1 }, c.svg);
      const tx = el("text", { x: c.X(d) + 4, y: c.Y(.95), "text-anchor": "start" }, c.svg); tx.textContent = lab;
    }
    const w1 = Number.isFinite(dk) ? Math.exp(-kappa * dk) : NaN, w2 = Math.exp(-kappa);
    read.textContent = (Number.isFinite(w1) ? `Weight at d₁₅: ${w1.toPrecision(2)}; ` : "") + `at d = 1: ${w2.toPrecision(2)}` + (Number.isFinite(w1) ? ` (ratio ${(w1 / w2).toPrecision(2)}).` : ".");
  };
  draw(); onResize(body, draw);
}

// ---------------------------------------------------------------------------------------------- strip of cells
// explorer state from a live figure's data-* block parameters (the explorer-link parameters; see DESIGN.md)
function blockState(ctx, ds) {
  const { m, T, S } = ctx, num = (v, d) => (v == null || v === "" || !Number.isFinite(+v) ? d : +v);
  const st = { fam: ds.fam === "time" ? "time" : "layer", mode: num(ds.m, m.default_mode), color: ds.c || "seba", k: num(ds.k, 1), r: num(ds.r, 4),
    focus: ds.j != null ? num(ds.j, null) : null, at: ds.at || "own" };
  const lim = st.fam === "layer" ? S - 1 : T - 1;
  st.anchor = num(ds.t, st.fam === "layer" ? T - 1 : S - 1); st.a = num(ds.a, 0); st.b = num(ds.b, lim);
  if (st.a > st.b) [st.a, st.b] = [st.b, st.a];
  st.kind = st.color === "eig" || st.color === "nodal" ? "psi" : "set";
  if (!m.seba_r.includes(st.r)) throw new Error(`r = ${st.r} is not one of ${m.seba_r.join(", ")}`);
  if (st.k < 1 || st.k > m.n_vecs) throw new Error(`k = ${st.k} is outside 1..${m.n_vecs}`);
  if (Math.min(st.a, st.anchor) < 0 || st.b > lim || st.anchor > (st.fam === "layer" ? T - 1 : S - 1)) throw new Error("block outside the grid");
  return st;
}

async function cells(body, ds) {
  const E = await import("./explorer.js?v=00a9dd9888");
  const ctx = await E.context();
  const { m, T, S, N } = ctx;
  const st = blockState(ctx, ds);
  const blk = await E.blockOf(ctx, st);
  await E.prepare(ctx, st);            // pixel statistics (c = px:<stat>) load on first use
  const col = E.colouring(ctx, st, blk);
  const list = (ds.cells ? ds.cells.split(",").map(x => x.split(":").map(Number)) : [...Array(st.b - st.a + 1)].map((_, i) => st.fam === "layer" ? [st.anchor, st.a + i] : [st.a + i, st.anchor]));
  const coords = E.toCoords(ctx.layouts[0]);
  body.replaceChildren();
  const row = h("div", { style: { display: "flex", flexWrap: "wrap", gap: "10px", justifyContent: "center" } });
  body.append(row);
  const size = +(ds.size || 150), dpr = Math.min(2, window.devicePixelRatio || 1);
  const items = list.map(([t, s]) => {
    const cv = h("canvas", { style: { width: size + "px", height: size + "px", background: "#fff", borderRadius: "6px", boxShadow: E.inBlock(st, t, s) ? "0 0 0 2px #16161a" : "0 0 0 1px #efeee9", cursor: "crosshair" } });
    cv.width = cv.height = Math.round(size * dpr);
    const cap = h("div", { style: { textAlign: "center", font: "12px var(--sans)", color: "var(--ink-2)", marginTop: "4px" }, text: `${m.sites[s]} · step ${fmtStep(m.steps[t])}` });
    row.append(h("div", {}, cv, cap));
    return { cv, t, s };
  });
  let hover = -1;
  const draw = () => items.forEach(({ cv, t, s }) => {
    const g = cv.getContext("2d"), img = g.createImageData(cv.width, cv.height), buf = new Uint32Array(img.data.buffer), pad = 8 * dpr;
    E.paint(buf, cv.width, cv.height, [{ t, s, S, x: pad, y: pad, w: cv.width - 2 * pad }], coords, col, N, 1.3 * dpr);
    g.putImageData(img, 0, 0);
    if (hover >= 0) {
      const base = (t * S + s) * N * 2, w = cv.width - 2 * pad, x = pad + w * coords[base + 2 * hover], y = pad + w * (1 - coords[base + 2 * hover + 1]);
      g.beginPath(); g.arc(x, y, 5 * dpr, 0, 7); g.strokeStyle = "#fff"; g.lineWidth = 3 * dpr; g.stroke(); g.strokeStyle = "#16161a"; g.lineWidth = 1.4 * dpr; g.stroke();
    }
  });
  items.forEach(({ cv, t, s }) => {
    cv.addEventListener("pointermove", e => {
      const r = cv.getBoundingClientRect(), pad = 8, w = r.width - 2 * pad, base = (t * S + s) * N * 2;
      let best = -1, bd = 64;
      for (let i = 0; i < N; i++) { const dx = pad + w * coords[base + 2 * i] - (e.clientX - r.left), dy = pad + w * (1 - coords[base + 2 * i + 1]) - (e.clientY - r.top), d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
      if (best !== hover) { hover = best; draw(); }
      if (best >= 0) showTip(e.clientX, e.clientY, tipRows(`#${best}`, [{ name: "class", value: m.classes[ctx.labels[best]] }])); else hideTip();
    });
    cv.addEventListener("pointerleave", () => { hover = -1; draw(); hideTip(); });
  });
  draw();
}

// ---------------------------------------------------------------------------------------------- probe thumbnails
// A set (data-j, 0-based, of data-r sets) or one end of an eigenfunction (data-k with data-end = pos | neg | zero) of
// one block. Images and their order are the explorer's representative images (build time, full precision): a set's
// members by decreasing SEBA value (data-pick="boundary": smallest lead over the runner-up set), or psi_k's most
// positive / most negative / nearest-zero entries. data-pick="random" draws members uniformly (fixed seed). Past the
// stored representatives (24 per set, 12 per end of psi_k) the order is continued from the vectors in the browser.
async function images(body, ds) {
  const E = await import("./explorer.js?v=00a9dd9888");
  const ctx = await E.context();
  const { m, N, pal, labels } = ctx;
  const byPsi = ds.k != null && ds.j == null;
  const st = blockState(ctx, { ...ds, c: byPsi ? "eig" : "seba" });
  const blk = await E.blockOf(ctx, st);
  const want = Math.max(1, Math.min(400, +(ds.n || 12))), pick = ds.pick || "top";
  let ids, title, colour, of = null;
  if (byPsi) {
    const end = { pos: 0, neg: 1, zero: 2 }[ds.end || "pos"];
    if (end == null) throw new Error(`data-end must be pos, neg or zero`);
    const v = D.psi(m, blk, st.k), key = [i => -v[i], i => v[i], i => Math.abs(v[i])][end];
    ids = extend(D.reps(m, blk, "psi", st.k, end), key, N, want);
    title = [`most positive ψ${st.k}`, `most negative ψ${st.k}`, `ψ${st.k} nearest zero`][end];
    colour = [pal.p.diverging.pos, pal.p.diverging.neg, pal.p.diverging.mid][end];
  } else {
    const ri = m.seba_r.indexOf(st.r), j = st.focus ?? 0, lab = blk.lab.subarray(ri * N, (ri + 1) * N);
    if (j < 0 || j >= st.r) throw new Error(`data-j = ${j} is outside 0..${st.r - 1}`);
    const mem = []; for (let i = 0; i < N; i++) if (lab[i] === j) mem.push(i);
    of = mem.length;
    const S_ = D.sebaVectors(m, blk, ri), rp = D.reps(m, blk, "set", ri, j);
    if (pick === "random") {
      let seed = 2024 + 31 * j; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
      const a = mem.slice(); for (let i = a.length - 1; i > 0; i--) { const q = Math.floor(rnd() * (i + 1)); [a[i], a[q]] = [a[q], a[i]]; }
      ids = a.slice(0, want); title = `set ${j + 1} of ${st.r}: random members`;
    } else if (pick === "boundary") {
      const lead = i => S_[j][i] - Math.max(0, ...S_.filter((_, q) => q !== j).map(v => v[i]));
      ids = extend(rp.bnd, lead, N, want, mem); title = `set ${j + 1} of ${st.r}: boundary members (smallest lead over the runner-up set)`;
    } else {
      ids = extend(rp.top, i => -S_[j][i], N, want, mem); title = `set ${j + 1} of ${st.r}: highest SEBA value`;
    }
    colour = pal.set(j);
  }
  body.replaceChildren();
  const size = +(ds.size || 40);
  const grid = h("div", { class: "imgs live-imgs", style: { gridTemplateColumns: `repeat(auto-fill, minmax(${size}px, 1fr))` } });
  for (const i of ids) {
    const c = E.thumb(ctx, i);
    c.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`#${i}`, [{ name: "class", value: m.classes[labels[i]] }])));
    c.addEventListener("pointerleave", hideTip);
    grid.append(c);
  }
  const blkTxt = `${E.blockKind(st)}: ${E.blockLabel(ctx, st)}; kernel rule ${m.modes[st.mode]}`;
  body.append(h("div", { class: "imgs-label" }, h("span", { class: "sw", style: { background: colour } }),
    `${title}: ${ids.length}${of != null ? ` of ${of.toLocaleString("en")}` : ""} images`, h("span", { class: "mute", text: ` · ${blkTxt}` })), grid);
}
// stored representatives first, then the rest of `pool` (default: every image) in increasing key order
function extend(first, key, N, want, pool = null) {
  const out = first.slice(0, want);
  if (out.length >= want) return out;
  const have = new Set(out), rest = (pool || [...Array(N).keys()]).filter(i => !have.has(i)).sort((x, y) => key(x) - key(y));
  return out.concat(rest.slice(0, want - out.length));
}
