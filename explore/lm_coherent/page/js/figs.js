// Live figures of the Measurements and Method tabs (F2-F8, F11) and the shared renderers: small multiples of
// layer x position heatmaps, and the icicle used by F2 and the Zoom tab. Every chart draws from the embedded data.
import { el, h, showTip, hideTip, tipRows, lineChart } from "./charts.js";
import { D, A, at, stat, excess, fmt, fmtS, g3, layerName, snippetEl, cellRecord, srcSwatch } from "./data.js";

const ROWS_BAG = () => [-1, ...D.layers.map((_, i) => i)];
const ROWS = () => D.layers.map((_, i) => i);
const stepTitle = s => s === 0 ? "step 0" : `step ${s.toLocaleString("en")}`;
const LOG_CAP = 4;

// ------------------------------------------------------------------------------------------- small multiples
// o: panels [{title}], rows [labels], cols [labels], value(p, r, c) -> number, color(v) -> css, tip(p, r, c) -> rows,
// head(p, r, c) -> string, onclick(p, r, c), sel {p, r, c}, legend {gradient, lo, hi, label, note}, minPanel
export function multiples(container, o) {
  container.replaceChildren();
  const W = Math.max(280, container.clientWidth || 900), np = o.panels.length, gap = 14, minP = o.minPanel || 150;
  let perRow = np, per = Math.floor((W - gap * (np - 1)) / np);
  if (per < minP) { perRow = Math.max(1, Math.floor((W + gap) / (minP + gap))); per = Math.floor((W - gap * (perRow - 1)) / perRow); }
  const nr = o.rows.length, nc = o.cols.length, lab = 30;
  const cw = Math.max(12, Math.min(38, Math.floor((per - lab - 4) / nc))), ch = Math.min(cw, 24), top = 18, bot = 16;
  const grid = h("div", { class: "sm-grid", style: { gridTemplateColumns: `repeat(${perRow}, ${lab + nc * cw + 4}px)`, columnGap: gap + "px" } });
  container.append(grid);
  o.panels.forEach((P, p) => {
    const Wp = lab + nc * cw + 4, Hp = top + nr * ch + bot;
    const svg = el("svg", { width: Wp, height: Hp, viewBox: `0 0 ${Wp} ${Hp}`, class: "sm-svg" });
    const tt = el("text", { x: lab, y: 12, class: "sm-title" }, svg); tt.textContent = P.title;
    for (let r = 0; r < nr; r++) {
      const t = el("text", { x: lab - 4, y: top + r * ch + ch / 2 + 3.5, "text-anchor": "end" }, svg); t.textContent = o.rows[r];
      for (let c = 0; c < nc; c++) {
        const v = o.value(p, r, c);
        const rect = el("rect", { x: lab + c * cw + .5, y: top + r * ch + .5, width: cw - 1, height: ch - 1, rx: 1.5, fill: Number.isFinite(v) ? o.color(v) : "#f2f1ec" }, svg);
        rect.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(o.head ? o.head(p, r, c) : null, o.tip(p, r, c))));
        rect.addEventListener("pointerleave", hideTip);
        if (o.onclick) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => o.onclick(p, r, c)); }
        if (o.sel && o.sel.p === p && o.sel.r === r && o.sel.c === c)
          el("rect", { x: lab + c * cw + .5, y: top + r * ch + .5, width: cw - 1, height: ch - 1, rx: 1.5, fill: "none", stroke: "#16161a", "stroke-width": 2, "pointer-events": "none" }, svg);
      }
    }
    for (let c = 0; c < nc; c++) { const t = el("text", { x: lab + c * cw + cw / 2, y: top + nr * ch + 12, "text-anchor": "middle" }, svg); t.textContent = o.cols[c]; }
    grid.append(h("div", { class: "sm-panel" }, svg));
  });
  if (o.legend) container.append(legendBar(o.legend));
}
export function legendBar(lg) {
  return h("div", { class: "sm-legend" },
    h("span", { class: "sm-lab", text: lg.label }),
    h("span", { class: "sm-lo", text: lg.lo }), h("i", { class: "sm-bar", style: { background: lg.gradient } }), h("span", { class: "sm-hi", text: lg.hi }),
    lg.note ? h("span", { class: "sm-note", text: lg.note }) : null);
}
const seqGrad = () => `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(t => D.pal.seq(t)).join(",")})`;
const divGrad = () => D.pal.divGradient();
function segButtons(values, labels, cur, on) {
  const seg = h("div", { class: "seg", role: "radiogroup" });
  values.forEach((v, i) => {
    const b = h("button", { type: "button", "aria-checked": String(v === cur), text: labels[i] });
    b.addEventListener("click", () => { seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-checked", "false")); b.setAttribute("aria-checked", "true"); on(v); });
    seg.append(b);
  });
  return seg;
}
const ctlRow = (...kids) => h("div", { class: "chart-controls" }, ...kids);
const bandTxt = b => `[${g3(b[0])}, ${g3(b[1])}]`;

// ------------------------------------------------------------------------------------------- F3: token vs source
function f3(body) {
  body.replaceChildren();
  const rows = ROWS_BAG(), steps = D.steps;
  const one = (name, title, sub) => {
    const box = h("div", { class: "sm-block" }, h("p", { class: "chart-title", text: title }), h("p", { class: "chart-sub", text: sub }));
    const holder = h("div");
    box.append(holder);
    body.append(box);
    const v = (si, li, ti) => excess(stat(name, si, 0, li, ti), stat(name + "0", si, 0, li, ti));
    multiples(holder, {
      panels: steps.map(s => ({ title: stepTitle(s) })), rows: rows.map(layerName), cols: D.positions,
      value: (p, r, c) => v(p, rows[r], c), color: x => D.pal.seq(Math.sqrt(Math.max(0, Math.min(1, x)))),
      head: (p, r, c) => `${layerName(rows[r])}, t = ${D.positions[c]}, ${stepTitle(steps[p])}`,
      tip: (p, r, c) => { const li = rows[r], S = stat(name, p, 0, li, c), S0 = stat(name + "0", p, 0, li, c), b = stat(name + "_band", p, 0, li, c);
        return [{ name: `excess ${name}̃`, value: fmt(excess(S, S0), 3) }, { name: `raw ${name}`, value: g3(S) }, { name: "chance", value: g3(S0) }, { name: "permutation band", value: bandTxt(b) }]; },
      legend: { label: `excess ${name}̃`, lo: "0 (chance)", hi: "1", gradient: seqGrad(), note: "square-root colour scale (0.25 is mid-colour), the same for both rows; negative values are drawn as 0; hover for the value" },
    });
  };
  one("T", "Token just read: excess walk share T̃", "pairs of items that read the same token at position t (label-free)");
  one("W", "Source: excess walk share W̃", "pairs of items from the same source (validation only)");
}

// ------------------------------------------------------------------------------------------- F4: maths vs poetry
function f4(body, ds) {
  const S = D.sources, pairs = [];
  for (let a = 0; a < S.length; a++) for (let b = a + 1; b < S.length; b++) pairs.push([a, b]);
  const st = { pair: [S.indexOf("maths"), S.indexOf("poetry")], sel: null };
  const rows = ROWS_BAG();
  const Xof = (si, li, ti, a, b) => li < 0 ? at(A("sb_X"), si, ti, a, b) : at(A("s_X"), si, 0, li, ti, a, b);
  const Xband = (si, li, ti, a, b) => li < 0 ? at(A("sb_X_band"), si, ti, a, b) : at(A("s_X_band"), si, 0, li, ti, a, b);
  const sel = h("select", { "aria-label": "Source pair" });
  pairs.forEach(([a, b]) => sel.append(h("option", { value: `${a},${b}`, text: `${S[a]} vs ${S[b]}` })));
  sel.value = st.pair.join(",");
  const quick = [[S.indexOf("maths"), S.indexOf("poetry")], [S.indexOf("wiki"), S.indexOf("forum")]];
  const qs = segButtons(["0", "1"], quick.map(([a, b]) => `${S[a]} – ${S[b]}`), "0", v => { st.pair = quick[+v]; sel.value = st.pair.join(","); draw(); });
  sel.addEventListener("change", () => { st.pair = sel.value.split(",").map(Number); draw(); });
  const holder = h("div"), detail = h("div", { class: "x-detail" });
  body.replaceChildren(ctlRow(h("label", { class: "ctl-sub", text: "pair" }), sel, qs), holder, detail);
  const draw = () => {
    const [a, b] = st.pair;
    multiples(holder, {
      panels: D.steps.map(s => ({ title: stepTitle(s) })), rows: rows.map(layerName), cols: D.positions,
      value: (p, r, c) => Math.log2(Xof(p, rows[r], c, a, b)), color: v => D.pal.div(v / LOG_CAP), sel: st.sel,
      head: (p, r, c) => `${S[a]} vs ${S[b]} · ${layerName(rows[r])}, t = ${D.positions[c]}, ${stepTitle(D.steps[p])}`,
      tip: (p, r, c) => { const x = Xof(p, rows[r], c, a, b), bd = Xband(p, rows[r], c, a, b);
        return [{ name: "X", value: g3(x) }, { name: "log₂ X", value: fmt(Math.log2(x), 2) }, { name: "permutation band", value: bandTxt(bd) }, { name: "click", value: "full 5 × 5 X" }]; },
      onclick: (p, r, c) => { st.sel = { p, r, c }; draw(); showX(p, rows[r], c); },
      legend: { label: "log₂ X", lo: `≤ −${LOG_CAP}`, hi: `≥ ${LOG_CAP}`, gradient: divGrad(), note: "0: the walk crosses as often as for random sources; blue: it avoids crossing" },
    });
  };
  const showX = (si, li, ti) => {
    const n = S.length, cs = 46, lab = 56, svg = el("svg", { width: lab + n * cs, height: 26 + n * cs, viewBox: `0 0 ${lab + n * cs} ${26 + n * cs}` });
    for (let i = 0; i < n; i++) {
      const t = el("text", { x: lab - 6, y: 22 + i * cs + cs / 2, "text-anchor": "end" }, svg); t.textContent = S[i];
      const u = el("text", { x: lab + i * cs + cs / 2, y: 14, "text-anchor": "middle" }, svg); u.textContent = S[i];
      for (let j = 0; j < n; j++) {
        const x = Xof(si, li, ti, i, j), lx = Math.log2(x);
        el("rect", { x: lab + j * cs + .5, y: 20 + i * cs + .5, width: cs - 1, height: cs - 1, rx: 2, fill: D.pal.div(lx / LOG_CAP) }, svg);
        const tv = el("text", { x: lab + j * cs + cs / 2, y: 20 + i * cs + cs / 2 + 4, "text-anchor": "middle", class: Math.abs(lx) > LOG_CAP * .6 ? "on-dark" : "" }, svg);
        tv.textContent = x >= 10 ? x.toFixed(0) : x >= 0.01 ? x.toFixed(2) : x.toExponential(0);
      }
    }
    detail.replaceChildren(h("p", { class: "chart-sub", text: `X(A, B) at ${layerName(li)}, t = ${D.positions[ti]}, ${stepTitle(D.steps[si])} (diagonal: within-source, above 1 when the walk stays in the source)` }), h("div", { class: "chart" }, svg));
  };
  draw();
}

// ------------------------------------------------------------------------------------------- F5: follow two snippets
function f5(body) {
  const F = D.follow, S = D.sources;
  const firstOf = name => F.findIndex(i => S[D.src[i]] === name);
  const st = { a: Math.max(0, firstOf("maths")), b: Math.max(0, firstOf("poetry")), si: D.fi };
  const label = fi => { const i = F[fi], s = S[D.src[i]], k = F.slice(0, fi + 1).filter(j => D.src[j] === D.src[i]).length; return `${s} ${k} (item ${i})`; };
  const mk = which => { const s = h("select", { "aria-label": `Snippet ${which}` }); F.forEach((_, fi) => s.append(h("option", { value: fi, text: label(fi) }))); s.value = st[which]; s.addEventListener("change", () => { st[which] = +s.value; draw(); }); return s; };
  const seg = segButtons(D.steps.map((_, i) => i), D.steps.map(fmtS), st.si, v => { st.si = v; draw(); });
  const texts = h("div", { class: "f5-texts" }), grids = h("div", { class: "f5-grids" });
  body.replaceChildren(ctlRow(h("span", { class: "ctl-sub", text: "snippet A" }), mk("a"), h("span", { class: "ctl-sub", text: "snippet B" }), mk("b"), h("span", { class: "ctl-sub", text: "checkpoint" }), seg), texts, grids);
  const draw = () => {
    const ia = F[st.a], ib = F[st.b];
    texts.replaceChildren(...[["A", ia], ["B", ib]].map(([w, i]) => h("div", { class: "f5-text" },
      h("div", { class: "f5-head" }, h("b", { text: `Snippet ${w}` }), srcSwatch(i), h("span", { class: "mute", text: `${S[D.src[i]]}, item ${i}` })),
      snippetEl(i, 128, { before: 128, after: 0, marks: D.positions, cls: "snip snip-full" }))));
    grids.replaceChildren(massGrid(st.a, "A"), massGrid(st.b, "B"), coGrid(ia, ib));
    grids.append(h("div", { class: "f5-key" }, ...S.map((s, k) => h("span", {}, h("i", { style: { background: D.srcCol[k] } }), s)),
      h("span", { class: "mute", text: "bars: share of the snippet's walk (its row of P_c) that lands on each source" })));
  };
  const cs = 34, lab = 30;
  const frame = (title) => { const svg = el("svg", { width: lab + 6 * cs, height: 20 + D.L * cs + 16, viewBox: `0 0 ${lab + 6 * cs} ${20 + D.L * cs + 16}` });
    const t = el("text", { x: lab, y: 12, class: "sm-title" }, svg); t.textContent = title;
    D.layers.forEach((l, li) => { const r = D.L - 1 - li, tx = el("text", { x: lab - 4, y: 20 + r * cs + cs / 2 + 4, "text-anchor": "end" }, svg); tx.textContent = "L" + l; });
    D.positions.forEach((p, ti) => { const tx = el("text", { x: lab + ti * cs + cs / 2, y: 20 + D.L * cs + 12, "text-anchor": "middle" }, svg); tx.textContent = p; });
    return svg; };
  const massGrid = (fi, w) => {
    const svg = frame(`Snippet ${w}: where its walk goes`);
    D.layers.forEach((_, li) => D.positions.forEach((_, ti) => {
      const r = D.L - 1 - li, x0 = lab + ti * cs + 2, y0 = 20 + r * cs + 2, hh = cs - 4;
      const m = at(A("s_follow"), st.si, fi, li, ti), tot = m.reduce((a, b) => a + b, 0) || 1;
      let y = y0 + hh;
      const g = el("g", {}, svg);
      for (let k = 0; k < S.length; k++) { const hk = hh * m[k] / tot; if (hk > 0) el("rect", { x: x0, y: y - hk, width: cs - 4, height: hk, fill: D.srcCol[k] }, g); y -= hk; }
      el("rect", { x: x0, y: y0, width: cs - 4, height: hh, fill: "transparent", stroke: "#e4e3de" }, g)
        .addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(`Snippet ${w} · L${D.layers[li]}, t = ${D.positions[ti]}`, S.map((s, k) => ({ name: s, value: (100 * m[k] / tot).toFixed(0) + "%", color: D.srcCol[k] })))));
      g.lastChild.addEventListener("pointerleave", hideTip);
    }));
    return h("div", { class: "f5-grid" }, svg);
  };
  const coGrid = (ia, ib) => {
    const svg = frame("Same coherent set? (r = 4, each cell)");
    D.layers.forEach((_, li) => D.positions.forEach((_, ti) => {
      const r = D.L - 1 - li, cx = lab + ti * cs + cs / 2, cy = 20 + r * cs + cs / 2;
      const rec = cellRecord("p", D.steps[st.si], li, ti), la = rec.lab[4][ia], lb = rec.lab[4][ib];
      const un = la < 0 || lb < 0, same = !un && la === lb;
      const c = el("circle", { cx, cy, r: un ? 4 : 10, fill: un ? D.palette.unassigned : same ? "#16161a" : "#fff", stroke: un ? "none" : "#16161a", "stroke-width": 1.6 }, svg);
      const hit = el("rect", { x: cx - cs / 2, y: cy - cs / 2, width: cs, height: cs, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(`L${D.layers[li]}, t = ${D.positions[ti]}`, [{ name: "A's set", value: la < 0 ? "unassigned" : "set " + (la + 1) }, { name: "B's set", value: lb < 0 ? "unassigned" : "set " + (lb + 1) }])));
      hit.addEventListener("pointerleave", hideTip);
    }));
    return h("div", { class: "f5-grid" }, svg, h("div", { class: "f5-co-key" },
      h("span", {}, h("i", { class: "dot-fill" }), "same set"), h("span", {}, h("i", { class: "dot-open" }), "different sets"), h("span", {}, h("i", { class: "dot-grey" }), "either unassigned")));
  };
  draw();
}

// ------------------------------------------------------------------------------------------- F6: point vs slab
function f6(body) {
  const rows = ROWS(), by = {};
  for (const r of D.compare) if (r.kind === "point_slab") by[`${r.step}:${r.layer}:${r.t}`] = r;
  const holder = h("div");
  body.replaceChildren(holder);
  const get = (p, r, c) => by[`${D.steps[p]}:${D.layers[rows[r]]}:${D.positions[c]}`];
  multiples(holder, {
    panels: D.steps.map(s => ({ title: stepTitle(s) })), rows: rows.map(layerName), cols: D.positions,
    value: (p, r, c) => get(p, r, c)?.a3, color: v => D.pal.seq(Math.max(0, Math.min(1, v))),
    head: (p, r, c) => `${layerName(rows[r])}, t = ${D.positions[c]}, ${stepTitle(D.steps[p])}`,
    tip: (p, r, c) => { const x = get(p, r, c); if (!x) return [{ name: "n/a", value: "" }];
      return [{ name: "a₃", value: fmt(x.a3, 3) }, { name: "a₆", value: fmt(x.a6, 3) }, { name: "matched Jaccard r = 4", value: fmt(x.jacc, 2) }, { name: "its null (mean, 2.5–97.5 %)", value: `${fmt(x.jacc_null[0], 2)} (${fmt(x.jacc_null[1], 2)}–${fmt(x.jacc_null[2], 2)})` }]; },
    legend: { label: "a₃ (point vs slab)", lo: "0", hi: "1", gradient: seqGrad(), note: `null r/(N − 1) = ${(3 / (D.N - 1)).toFixed(3)}` },
  });
}

// ------------------------------------------------------------------------------------------- F7: through training
function f7(body) {
  const ti = D.T - 1, x = D.steps;
  const left = h("div", { class: "chart" }), right = h("div", { class: "chart" });
  body.replaceChildren(h("div", { class: "charts-2" },
    h("div", {}, h("p", { class: "chart-title", text: "Source separation at t = 128: W̃ (validation)" }), left),
    h("div", {}, h("p", { class: "chart-title", text: "Agreement with the final sets (L8, r = 4)" }), right)));
  const ser = D.layers.map((l, li) => ({ name: "L" + l, y: x.map((_, si) => excess(stat("W", si, 0, li, ti), stat("W0", si, 0, li, ti))), color: D.pal.ord(.15 + .85 * li / (D.L - 1)), dots: 2.5 }));
  ser.push({ name: "bag", y: x.map((_, si) => excess(stat("W", si, 0, -1, ti), stat("W0", si, 0, -1, ti))), color: "#6b6b70", dash: true, dots: 2.5 });
  const xo = { x, xscale: "symlog", xticks: x, xfmt: fmtS, xlabel: "training step (log scale; 0 at the left)", xname: "step", height: 250 };
  lineChart(left, { ...xo, series: ser, yrange: [Math.min(0, ...ser.flatMap(s => s.y).filter(Number.isFinite)) - .02, 1], ylabel: "W̃", ytitle: "excess W̃" });
  const rows = D.compare.filter(r => r.kind === "to_final" && r.layer === 8);
  const tf = [16, 128].map((t, k) => { const yy = x.map(s => s === D.final ? 1 : rows.find(r => r.step === s && r.t === t)?.jacc ?? NaN);
    const lo = x.map(s => rows.find(r => r.step === s && r.t === t)?.jacc_null?.[1] ?? NaN), hi = x.map(s => rows.find(r => r.step === s && r.t === t)?.jacc_null?.[2] ?? NaN);
    return { name: `(L8, t = ${t})`, y: yy, band: [lo, hi], color: D.pal.set(k), dots: 3 }; });
  lineChart(right, { ...xo, series: tf, yrange: [0, 1.02], ytitle: "matched Jaccard to step " + fmtS(D.final), bandMode: "each", bandLabel: "shaded: permutation null (2.5–97.5 %)" });
}

// ------------------------------------------------------------------------------------------- F8: order control
function f8(body) {
  const rows = D.compare.filter(r => r.kind === "real_vs_S"), x = D.layers, ti = D.T - 1;
  const byL = l => rows.find(r => r.layer === l);
  const w0 = li => stat("W0", D.fi, 0, li, ti);
  const real = x.map((l, li) => excess(byL(l)?.W_a, w0(li))), shuf = x.map((l, li) => excess(byL(l)?.W_b, w0(li)));
  const bag = rows.find(r => r.layer === "bag"), bagW = bag ? excess(bag.W_a, stat("W0", D.fi, 0, -1, ti)) : NaN;
  const left = h("div", { class: "chart" }), right = h("div", { class: "chart" });
  body.replaceChildren(h("div", { class: "charts-2" },
    h("div", {}, h("p", { class: "chart-title", text: `W̃ at t = 128, step ${fmtS(D.final)}: original vs reordered` }), left),
    h("div", {}, h("p", { class: "chart-title", text: "Sets of the original vs reordered probe (r = 4)" }), right)));
  const xo = { x, xticks: x, xfmt: v => "L" + v, xlabel: "layer", xname: "", height: 240 };
  lineChart(left, { ...xo, series: [{ name: "original (probe A)", y: real, color: D.pal.set(0), dots: 3 }, { name: "tokens 1..127 reordered (probe S)", y: shuf, color: D.pal.set(1), dash: true, dots: 3 }],
    yrange: [Math.min(0, ...real, ...shuf, bagW) - .02, 1], refline: Number.isFinite(bagW) ? { y: bagW, label: `bag (identical for both): ${fmt(bagW, 2)}` } : null, ytitle: "excess W̃" });
  lineChart(right, { ...xo, series: [{ name: "matched Jaccard, A vs S", y: x.map(l => byL(l)?.jacc), band: [x.map(l => byL(l)?.jacc_null?.[1]), x.map(l => byL(l)?.jacc_null?.[2])], color: D.pal.set(0), dots: 3 }],
    yrange: [0, 1.02], ytitle: "matched Jaccard", bandMode: "each", bandLabel: "shaded: permutation null (2.5–97.5 %)" });
}

// ------------------------------------------------------------------------------------------- F11: diagnostics
function frogue(body) { f11(body, { q: "rogue1", only: true }); }
function f11(body, ds = {}) {
  const norm = h("div", { class: "chart" }), holder = h("div");
  const Q = {
    rogue1: { label: "top-1 coordinate share of centred variance", f: (si, li, ti) => stat("rogue1", si, 0, li, ti), fmt: v => fmt(v, 3) },
    rogue5: { label: "top-5 coordinate share", f: (si, li, ti) => stat("rogue5", si, 0, li, ti), fmt: v => fmt(v, 3) },
    pn1: { label: "PN(ψ₁) (flag below 10)", f: (si, li, ti) => stat("pn1", si, 0, li, ti), fmt: v => v.toFixed(0) },
    minset: { label: "smallest r = 4 set (items)", f: (si, li, ti) => stat("minset", si, 0, li, ti), fmt: v => v.toFixed(0) },
    min_dist: { label: "smallest pairwise cosine distance", f: (si, li, ti) => stat("min_dist", si, 0, li, ti), fmt: v => g3(v) },
  };
  let q = ds.q || "rogue1";
  const seg = segButtons(Object.keys(Q), ["top-1 share", "top-5 share", "PN(ψ₁)", "smallest set", "smallest distance"], q, v => { q = v; draw(); });
  if (ds.only) { body.replaceChildren(holder); }
  else body.replaceChildren(h("p", { class: "chart-title", text: `Mean residual norm against position, step ${fmtS(D.final)}` }), norm,
    h("p", { class: "chart-title", style: { marginTop: "18px" }, text: "Per-cell diagnostics (point view; bag row: the reference view)" }), ctlRow(seg), holder);
  const nm = A("s_norm"), xs = Array.from({ length: nm.shape[1] }, (_, i) => i);
  if (!ds.only) lineChart(norm, { x: xs, xscale: "symlog", xticks: [0, 1, 4, 16, 64, 128], xlabel: "position u (0 = EOT)", xname: "u =", yscale: "log", ytitle: "mean ‖h‖", height: 230,
    series: D.layers.map((l, li) => ({ name: "L" + l, y: Array.from(at(nm, li)), color: D.pal.ord(.15 + .85 * li / (D.L - 1)) })) });
  const rows = ROWS_BAG();
  const draw = () => {
    const f = Q[q].f;
    let lo = Infinity, hi = -Infinity;
    D.steps.forEach((_, si) => rows.forEach(li => D.positions.forEach((_, ti) => { const v = f(si, li, ti); if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } })));
    multiples(holder, {
      panels: D.steps.map(s => ({ title: stepTitle(s) })), rows: rows.map(layerName), cols: D.positions,
      value: (p, r, c) => f(p, rows[r], c), color: v => D.pal.seq((v - lo) / ((hi - lo) || 1)),
      head: (p, r, c) => `${layerName(rows[r])}, t = ${D.positions[c]}, ${stepTitle(D.steps[p])}`,
      tip: (p, r, c) => [{ name: Q[q].label, value: Number.isFinite(f(p, rows[r], c)) ? Q[q].fmt(f(p, rows[r], c)) : "n/a" }],
      legend: { label: Q[q].label, lo: Q[q].fmt(lo), hi: Q[q].fmt(hi), gradient: seqGrad(), note: "one colour range for all five panels; grey: not defined (no coordinates)" },
    });
  };
  draw();
}

// ------------------------------------------------------------------------------------------- F2: controls
function f2(body) {
  const V = D.validation || {}, p1 = D.trees.find(t => t.id === "P1");
  const c2 = V.C2 || {}, c2ok = Object.values(c2).length && Object.values(c2).every(x => x.identical);
  const c2max = Math.max(0, ...Object.values(c2).map(x => x.max_abs_diff ?? 0));
  const c1 = V.C1 || {};
  const pass = ok => ok == null ? h("span", { class: "pill mute", text: "not run" }) : h("span", { class: "pill " + (ok ? "ok" : "bad"), text: ok ? "pass" : "fail" });
  const p2 = V.P2, p3 = V.P3;
  const p3dev = p3?.per_step ? Math.max(...Object.values(p3.per_step).map(x => Math.max(...Object.keys(x).filter(k => k.startsWith("dev_")).map(k => x[k])))) : NaN;
  const f32 = c1.float32_by_layer ? Object.entries(c1.float32_by_layer).map(([k, v]) => `${k} ${(+v).toExponential(1)}`).join(", ") : "";
  const rowsT = [
    ["C1", "causal indexing: a pass on tokens 1..64 gives the same residuals at t ≤ 64", "identity", c1.status ? c1.status : `${c1.precision || ""} max rel. error ${g3(c1.max_rel_err)} (gate ${g3(c1.tolerance)})${f32 ? "; float32 (hazard): " + f32 : ""}`, c1.status ? null : c1.pass ?? null],
    ["C2", "bag view at t = 128 is identical for probes A and S", "identity", `max |difference| ${g3(c2max)} over ${Object.keys(c2).length} checkpoints`, Object.keys(c2).length ? c2ok : null],
    ["P1", "planted hierarchy recovered exactly; planted splits above, structureless splits below the threshold", "planted", p1 ? `${p1.match?.exact ? "every node Jaccard 1" : "not exact"}; planted Υ ≥ ${g3(Math.min(...(p1.planted_upsilon || [NaN])))}, structureless Υ ≤ ${g3(Math.max(...(p1.structureless || []).map(s => s.upsilon)))}; threshold ${g3(p1.threshold)} (${p1.rule})` : "n/a", p1 ? !!p1.match?.exact : null],
    ["P2", "planted block demotion: transient split tops its cell, persistent split tops the block", "planted", p2?.block ? `cell: sweep / SEBA Jaccard ${fmt(p2.transient_cell.sweep_jaccard, 2)} / ${fmt(p2.transient_cell.seba2_jaccard, 2)}; block: ${fmt(p2.block.sweep_jaccard, 2)} / ${fmt(p2.block.seba2_jaccard, 2)} (fixture seed ${p2.seed})` : "asserted in test_lm.py; result not stored in the data", p2 ? !!p2.pass : null],
    ["P3", "block statistics are the means of the cell statistics (φ, T, W, X and their nulls)", "[thm]", p3 ? `largest deviation ${Number.isFinite(p3dev) ? p3dev.toExponential(1) : "n/a"} over ${Object.keys(p3.per_step || {}).length} checkpoints (tolerance 1e-12, real data); slab identity in test_lm.py` : "asserted in test_lm.py; result not stored in the data", p3 ? !!p3.pass : null],
    ["", "Sinkhorn converged in every cell", "run check", `${V.sinkhorn_failures ?? "n/a"} failures`, V.sinkhorn_failures != null ? V.sinkhorn_failures === 0 : null],
    ["", "no exact duplicates; no zero-norm items", "run check", `${(V.removed_items || []).length} items removed; ${V.duplicates ? "duplicates asserted per cell" : ""}`, true],
  ];
  const tbl = h("table", { class: "data controls" }, h("thead", {}, h("tr", {}, ...["", "check", "label", "measured", ""].map(t => h("th", { text: t })))));
  const tb = h("tbody");
  rowsT.forEach(([id, what, lab, meas, ok]) => tb.append(h("tr", {}, h("td", {}, h("b", { text: id })), h("td", { text: what }), h("td", { class: "mute", text: lab }), h("td", { text: meas }), h("td", {}, pass(ok)))));
  tbl.append(tb);
  const ic = h("div", { class: "icicle-wrap" });
  body.replaceChildren(h("div", { class: "table-scroll" }, tbl), h("p", { class: "chart-title", style: { marginTop: "14px" }, text: "Fixture tree P1: recovered nodes, coloured by the planted leaves they contain" }), ic);
  if (p1) icicle(ic, p1, { bar: "planted", label: nd => p1.match?.nodes?.find(m => m.node === nd.id) ? `J ${fmt(p1.match.nodes.find(m => m.node === nd.id).jaccard, 2)}` : "" });
}

// ------------------------------------------------------------------------------------------- icicle (F2, F9)
// tree: {nodes, threshold}; o: bar "makeup" | "planted", sel (node id), onclick(node), label(node) -> extra text
export function icicle(container, tree, o = {}) {
  container.replaceChildren();
  const nodes = tree.nodes, byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const W = Math.max(300, container.clientWidth || 900), depth = Math.max(...nodes.map(n => n.depth)) + 1, rh = W < 520 ? 50 : 56, gap = 4;
  const H = depth * rh + 4;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "icicle" });
  const thr = tree.threshold;
  const weak = nd => nd.children?.length && thr != null && Number.isFinite(nd.upsilon) && nd.upsilon < thr;
  const pos = {};
  const place = (nd, x0, x1) => { pos[nd.id] = [x0, x1]; let x = x0; for (const c of nd.children || []) { const w = (x1 - x0) * byId[c].n / nd.n; place(byId[c], x, x + w); x += w; } };
  place(nodes[0], 0, W);
  const fadedBy = {};
  for (const nd of nodes) for (const c of nd.children || []) fadedBy[c] = weak(nd) || fadedBy[nd.id];
  for (const nd of nodes) {
    const [x0, x1] = pos[nd.id], y = nd.depth * rh, w = x1 - x0 - 2, g = el("g", { class: "ic-node" + (fadedBy[nd.id] ? " ic-faded" : "") + (o.sel === nd.id ? " ic-sel" : "") }, svg);
    el("rect", { x: x0 + 1, y: y + 1, width: Math.max(1, w), height: rh - gap, rx: 4, class: "ic-box" }, g);
    const parts = o.bar === "planted" ? (nd.planted || []) : (nd.makeup || []), tot = parts.reduce((a, b) => a + b, 0) || 1;
    let bx = x0 + 4;
    parts.forEach((v, k) => { const bw = (w - 6) * v / tot; if (bw > 0) el("rect", { x: bx, y: y + rh - gap - 9, width: bw, height: 6, fill: o.bar === "planted" ? D.pal.set(k) : D.srcCol[k] }, g); bx += bw; });
    const lines = [];
    if (w > 46) lines.push(`n = ${nd.n}`);
    if (nd.children?.length && w > 74) lines.push(`Υ ${g3(nd.upsilon)}${weak(nd) ? " (weak)" : ""}`);
    else if (nd.children?.length && w > 40) lines.push(`Υ ${g3(nd.upsilon)}`);
    const extra = o.label ? o.label(nd) : "";
    if (extra && w > 100) lines[0] = (lines[0] || "") + " · " + extra;
    lines.forEach((ln, i) => {
      const t = el("text", { x: x0 + 6, y: y + 15 + i * 13, class: i ? "ic-sub" : "ic-t" }, g);
      if (ln.startsWith("Υ")) { el("tspan", { class: "ic-ups" }, t).textContent = "Υ"; el("tspan", {}, t).textContent = ln.slice(1); } else t.textContent = ln;
    });
    const ncomp = nd.n_components, flag = (Number.isFinite(ncomp) && ncomp >= 3) || nd.cut_uninformative || nd.uninformative;
    if (flag && w > 20) { const t = el("text", { x: x1 - 8, y: y + 14, "text-anchor": "end", class: "ic-flag" }, g); t.textContent = "◆"; }
    g.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(`node ${nd.id} (depth ${nd.depth})`, [
      { name: "items", value: String(nd.n) },
      ...(nd.children?.length ? [{ name: "clustering ratio Υ", value: g3(nd.upsilon) + (weak(nd) ? " (weak split)" : "") }, { name: "φ*", value: g3(nd.phi) }, { name: "cut along", value: "ψ" + nd.axis_used }] : [{ name: "leaf", value: "not split" }]),
      ...(Number.isFinite(ncomp) ? [{ name: "kNN components", value: String(ncomp) }] : []),
      ...(extra ? [{ name: "match", value: extra }] : []),
      ...(o.bar === "planted" ? (nd.planted || []).map((v, k) => ({ name: `planted leaf ${k}`, value: String(v), color: D.pal.set(k) })).filter(r => r.value !== "0")
        : (nd.makeup || []).map((v, k) => ({ name: D.sources[k], value: String(v), color: D.srcCol[k] }))),
    ])));
    g.addEventListener("pointerleave", hideTip);
    if (o.onclick) { g.style.cursor = "pointer"; g.addEventListener("click", () => o.onclick(nd)); }
  }
  container.append(svg);
  return { pos };
}

// ------------------------------------------------------------------------------------------- sources: blocks vs cells
function fsrc(body) {
  const left = h("div", { class: "chart" }), right = h("div", { class: "chart" });
  body.replaceChildren(h("div", { class: "charts-2" },
    h("div", {}, h("p", { class: "chart-title", text: `Sets (r = 4) against sources at step ${fmtS(D.final)}` }), left),
    h("div", {}, h("p", { class: "chart-title", text: "Through training, at L8" }), right)));
  const sjCell = (view, step, li, ti) => { const k = D.cellIdx[`${view}:${step}:${li}:${ti}`]; return k === undefined ? NaN : at(A("c_sj"), k, 1, 0); };
  const sjBlock = key => { const k = D.blockIdx[key]; return k === undefined ? NaN : at(A("b_sj"), k, 1, 0); };
  const x = D.layers, li8 = D.layers.indexOf(8), rbs = D.reading_by_step;
  const ser = [{ name: "reading block (t = 4 → 128)", y: x.map((_, li) => sjBlock(`reading:${li}`)), color: D.pal.set(0), width: 2.6, dots: 3.5 }];
  D.positions.forEach((t, ti) => ser.push({ name: `single cell, t = ${t}`, y: x.map((_, li) => sjCell("p", D.final, li, ti)), color: D.pal.ord(.1 + .6 * ti / (D.T - 1)), width: 1.2 }));
  ser.push({ name: "slab cell, t = 128", y: x.map((_, li) => sjCell("s", D.final, li, D.T - 1)), color: D.pal.set(1), dash: true, dots: 2.5 });
  const nullHi = Math.max(...rbs.null_hi[D.fi]);
  const bagT = sjCell("b", D.final, -1, D.T - 1);
  lineChart(left, { x, xticks: x, xfmt: v => "L" + v, xlabel: "layer", xname: "", height: 260, yrange: [0, 1], ytitle: "matched Jaccard vs sources",
    series: ser, hlines: [{ y: nullHi, label: `null (97.5 %) ≈ ${nullHi.toFixed(2)}`, dash: "3 3" }, ...(Number.isFinite(bagT) ? [{ y: bagT, label: `bag, t = 128: ${bagT.toFixed(2)}`, color: "#6b6b70", right: true }] : [])] });
  const xs = D.steps;
  lineChart(right, { x: xs, xscale: "symlog", xticks: xs, xfmt: fmtS, xlabel: "training step (log; 0 at the left)", xname: "step", height: 260, yrange: [0, 1], ytitle: "matched Jaccard vs sources",
    series: [{ name: "reading block L8", y: rbs.sj.map(r => r[li8]), color: D.pal.set(0), width: 2.6, dots: 3.5 },
      { name: "single cell (L8, 128)", y: xs.map(s => sjCell("p", s, li8, D.T - 1)), color: D.pal.ord(.7), dots: 2.5 },
      { name: "null 97.5 % (reading block)", y: rbs.null_hi.map(r => r[li8]), color: "#9a9993", dash: true, width: 1 }] });
}

// ------------------------------------------------------------------------------------------- clustering ratios
function fups(body) {
  const p1 = D.trees.find(t => t.id === "P1"), real = D.trees.filter(t => t.id !== "P1");
  const rows = [
    { name: "planted splits (fixture P1)", pts: (p1?.planted_upsilon || []).map(u => ({ u, tip: "planted split" })), col: D.pal.set(2) },
    { name: "structureless (fixture)", pts: (p1?.structureless || []).map(s => ({ u: s.upsilon, tip: s.kind === "flat root" ? "isotropic noise cell, root" : "extra split of a planted leaf" })), col: "#9a9993" },
    ...real.map(t => ({ name: `${t.id}: ${t.block?.kind === "single" ? "cell" : "reading block"} L${t.block?.layer}, step ${fmtS(t.step)}`, pts: t.nodes.filter(n => n.children?.length).map(n => ({ u: n.upsilon, tip: `node ${n.id}, depth ${n.depth}, n = ${n.n}`, depth: n.depth })), col: D.pal.set(0) })),
  ];
  const W = Math.max(320, body.clientWidth || 700), lab = Math.min(250, W * .38), rh = 26, H = rows.length * rh + 40;
  const lo = .7, hi = 1000, X = v => lab + 10 + (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (W - lab - 24);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  [1, 3, 10, 100, 1000].forEach(v => { el("line", { x1: X(v), x2: X(v), y1: 4, y2: rows.length * rh + 6, stroke: v === 3 ? D.palette.accent : "#efeee9", "stroke-width": v === 3 ? 1.5 : 1, ...(v === 3 ? { "stroke-dasharray": "4 3" } : {}) }, svg);
    const t = el("text", { x: X(v), y: rows.length * rh + 20, "text-anchor": "middle" }, svg); t.textContent = v === 3 ? "3 (reading threshold)" : String(v); });
  const t2 = el("text", { x: (lab + W) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg); t2.textContent = "clustering ratio Υ = μ₂/φ* (log scale)";
  rows.forEach((r, i) => {
    const y = i * rh + 16;
    const t = el("text", { x: lab, y: y + 4, "text-anchor": "end" }, svg); t.textContent = r.name;
    r.pts.forEach((p, k) => { const c = el("circle", { cx: X(Math.max(lo, Math.min(hi, p.u))), cy: y + (k % 3 - 1) * 3, r: 4.2, fill: r.col, "fill-opacity": .8, stroke: "#fff", "stroke-width": 1 }, svg);
      c.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(r.name, [{ name: p.tip, value: "" }, { name: "Υ", value: g3(p.u) }])));
      c.addEventListener("pointerleave", hideTip); });
  });
  body.replaceChildren(h("div", { class: "chart" }, svg));
}

export const FIGS = { fsrc, fups, frogue, f2, f3, f4, f5, f6, f7, f8, f11 };
