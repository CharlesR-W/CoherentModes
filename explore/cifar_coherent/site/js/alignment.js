// Live figures of the "Alignment" tab (content/alignment.html): W1-W4 and C1-C5 of explore/spectral_align/SPEC.md.
// Data: d/al/ from site/alignment.py (analysis by explore/spectral_align/analyze.py; layout in its docstring).
// Distances arrive as uint16 per distance (lazily), eigenvectors as int16 per snapshot (lazily, C5 only).
import * as D from "./data.js";
import { Palette } from "./colors.js";
import { h, el, lineChart, showTip, hideTip, tipRows, scale, niceTicks, fmtStep } from "./charts.js";

const AL = "d/al/";
let IDX = null;
const index = () => (IDX ||= D.json(AL + "index.json").then(prep));
const pal = async () => new Palette((await D.meta()).palette);

// ------------------------------------------------------------------------------------------------ data
function prep(I) {
  I.gid = new Map(I.groups.map((g, i) => [`${g.net}:${g.step}`, i]));
  I.selfSeg = new Map(); I.trajSeg = new Map(); I.pairSeg = new Map();
  I.segs.forEach((s, i) => {
    if (s.kind === "self") I.selfSeg.set(s.group, i);
    else if (s.kind === "traj") I.trajSeg.set(`${s.net}:${s.site}`, i);
    else { I.pairSeg.set(`${s.ga}:${s.gb}`, [i, false]); I.pairSeg.set(`${s.gb}:${s.ga}`, [i, true]); }
  });
  I.cellInfo = [];
  I.groups.forEach((g, gi) => g.cells.forEach((c, si) => { I.cellInfo[c] = { g: gi, net: g.net, step: g.step, site: si }; }));
  return I;
}
const sliceName = (did, rule) => did === "cka" ? "cka" : `${did}_r${rule}`;
const sliceCache = new Map();
function slice(did, rule) {
  const name = sliceName(did, rule);
  if (!sliceCache.has(name)) {
    const p = (async () => {
      const I = await index();
      const r = await fetch(`${AL}s/${name}.bin.gz`);
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      const b = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      const q = new Uint16Array(b), s = I.scale[name] / 65534, v = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) v[i] = q[i] === 65535 ? NaN : q[i] * s;
      return v;
    })();
    p.catch(() => sliceCache.delete(name));
    sliceCache.set(name, p);
  }
  return sliceCache.get(name);
}
const paths = (did, rule) => D.json(`${AL}s/${sliceName(did, rule)}.json`);
const summary = rule => D.json(`${AL}summary_r${rule}.json`);
const diagCache = {};
function diag(rule) {
  return (diagCache[rule] ||= (async () => {
    const I = await index(), q = new Uint16Array(await D.bytes(`${AL}diag_r${rule}.bin`)), W = I.n_diag_eig + I.n_span;
    const val = x => x === 65535 ? NaN : x / 65534;
    const eig = (c, m) => val(q[c * W + m]), span = (c, k) => val(q[c * W + I.n_diag_eig + k - 1]);
    const avail = c => Number.isFinite(eig(c, 1));            // operator available under this rule
    return { eig, span, avail, ok: (c, k) => span(c, k) >= I.stab_threshold, gap: (c, k) => eig(c, k) - eig(c, k + 1),
      mark: (c, k) => avail(c) ? span(c, k) >= I.stab_threshold : null };
  })());
}
// one shipped segment as a matrix accessor (t: transposed)
function mat(I, v, si, t = false) {
  const s = I.segs[si];
  return t ? { nr: s.nc, nc: s.nr, at: (i, j) => v[s.o + j * s.nc + i] } : { nr: s.nr, nc: s.nc, at: (i, j) => v[s.o + i * s.nc + j] };
}
// stored distance between two cells, when the pair is shipped (same group, same site over checkpoints, or a cross block)
function pairValue(I, v, ca, cb) {
  const A = I.cellInfo[ca], B = I.cellInfo[cb];
  if (A.g === B.g) return mat(I, v, I.selfSeg.get(A.g)).at(A.site, B.site);
  const ps = I.pairSeg.get(`${A.g}:${B.g}`);
  if (ps) return mat(I, v, ps[0], ps[1]).at(A.site, B.site);
  if (A.net === B.net && A.site === B.site && I.trajSeg.has(`${A.net}:${A.site}`)) {
    const ia = I.main_steps.indexOf(A.step), ib = I.main_steps.indexOf(B.step);
    if (ia >= 0 && ib >= 0) return mat(I, v, I.trajSeg.get(`${A.net}:${A.site}`)).at(ia, ib);
  }
  return null;
}

// ------------------------------------------------------------------------------------------------ labels
const netName = n => `ResNet-${n.depth} seed ${n.seed}`;
const stepTxt = s => Number(s).toLocaleString("en");
const fx = d => v => v == null || !Number.isFinite(v) ? "n/a" : v.toFixed(d);
const f2 = fx(2), f3 = fx(3);
const FAMS = [["proj", "projector dₖ"], ["heat", "diffusion hₜ"], ["eigd", "eigenvalue-only sₖ"], ["cka", "CKA (baseline)"]];
const did = st => st.fam === "cka" ? "cka" : `${st.fam}_${st.fam === "heat" ? "t" : "k"}${st.p}`;
const dShort = id => id === "cka" ? "CKA" : id.replace("proj_k", "d").replace("heat_t", "h").replace("eigd_k", "s");
function distLong(st) {
  if (st.fam === "proj") return `projector distance d_k, k = ${st.p}`;
  if (st.fam === "heat") return `diffusion-kernel distance h_t, t = ${st.p}`;
  if (st.fam === "eigd") return `eigenvalue-only distance s_k, k = ${st.p} (ignores eigenvectors)`;
  return "CKA distance (baseline, not an output of the construction)";
}
const ruleTxt = (st, I) => st.fam === "cka" ? "no kernel rule (CKA)" : `kernel rule ${st.rule} (${I.rule_names[st.rule]})`;
const zmax = (I, st) => st.fam === "proj" ? 1 : I.scale[sliceName(did(st), st.rule)];
function stageBounds(stages) { const b = []; for (let i = 0; i + 1 < stages.length; i++) if (stages[i] !== stages[i + 1] && stages[i + 1] !== 5) b.push(i); return b; }

// ------------------------------------------------------------------------------------------------ controls
function seg(opts, val, on) {
  const s = h("div", { class: "seg", role: "radiogroup" });
  const sync = v => s.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === String(v))));
  opts.forEach(([v, label, dis]) => { const b = h("button", { type: "button", "data-v": v, role: "radio", text: label }); if (dis) b.disabled = true; b.addEventListener("click", () => { sync(v); on(v); }); s.append(b); });
  s.sync = sync; sync(val); return s;
}
function select(opts, val, on) {
  const s = h("select");
  opts.forEach(([v, label]) => s.add(new Option(label, v)));
  s.value = String(val); s.addEventListener("change", () => on(s.value)); s.setOpts = (o, v) => { s.replaceChildren(); o.forEach(([vv, l]) => s.add(new Option(l, vv))); s.value = String(v); };
  return s;
}
const ctl = (...kids) => h("div", { class: "chart-controls al-ctl" }, ...kids);
const lab = t => h("span", { class: "ctl-label", text: t });
// distance family, its k or t, and the kernel rule
function distControls(I, st, on, fams = FAMS.map(f => f[0])) {
  const pSel = select([], st.p, v => { st.p = +v; on(); });
  const setP = () => {
    const grid = st.fam === "heat" ? I.t_grid : I.k_grid;
    pSel.setOpts(grid.map(v => [v, `${st.fam === "heat" ? "t" : "k"} = ${v}`]), st.p);
    pSel.hidden = st.fam === "cka";
  };
  const rSeg = seg(I.rules.map(r => [String(r), `rule ${r}: ${I.rule_names[r]}`]), st.rule, v => { st.rule = +v; on(); });
  const syncRule = () => rSeg.querySelectorAll("button").forEach(b => { b.disabled = st.fam === "cka"; });
  const fSeg = seg(FAMS.filter(f => fams.includes(f[0])), st.fam, v => {
    st.fam = v; if (v === "heat" && !I.t_grid.includes(st.p)) st.p = 16; if (v !== "heat" && !I.k_grid.includes(st.p)) st.p = 9;
    setP(); syncRule(); on();
  });
  setP(); syncRule();
  return [lab("distance"), fSeg, pSel, rSeg];
}
function fixtureTag(I) { return I.fixture ? h("div", { class: "al-fixture-tag", text: "synthetic fixture data: not a measurement" }) : null; }
const onResize = (elm, fn) => { let w = elm.clientWidth; new ResizeObserver(() => { if (Math.abs(elm.clientWidth - w) > 4) { w = elm.clientWidth; fn(); } }).observe(elm); };
// cut k|k+1 of the seed-floor entry at site i of a same-depth cross-seed segment: resolved only if both seeds' cuts are
function floorOk(I, dg, segIdx, i, k) { const s = I.segs[segIdx]; return dg.ok(I.groups[s.ga].cells[i], k) && dg.ok(I.groups[s.gb].cells[i], k); }
// floor values of a segment's diagonal; for the projector distance only sites whose floor cut is resolved
function floorVals(I, dg, v, segIdx, L, proj, k) {
  const F = mat(I, v, segIdx), all = [], ok = [];
  for (let i = 0; i < L; i++) { const x = F.at(i, i); if (!Number.isFinite(x)) continue; all.push(x); if (!proj || floorOk(I, dg, segIdx, i, k)) ok.push(x); }
  return { vals: ok.sort((a, b) => a - b), n: ok.length, nAll: all.length };
}
const med = a => a.length ? a[Math.floor(a.length / 2)] : NaN;
function cutRows(I, dg, ca, cb, k) {
  return [{ name: `cut ${k}|${k + 1}`, value: dg.ok(ca, k) && dg.ok(cb, k) ? "resolved" : "unresolved" },
    { name: "gap λk−λk+1 (A, B)", value: `${f3(dg.gap(ca, k))}, ${f3(dg.gap(cb, k))}` },
    { name: "span_k (A, B)", value: `${f2(dg.span(ca, k))}, ${f2(dg.span(cb, k))}` }];
}

// ------------------------------------------------------------------------------------------------ matrix (SVG)
// o: nr, nc, rowLabels, colLabels, value(i,j), color(v)->css, zrange [lo,hi], zlabel, zfmt, hatch(i,j), tip(i,j)->[head, rows],
//    rowMark(i)/colMark(j) -> true (filled) | false (hollow) | null, rowBounds/colBounds (line after index), rowTicks/colTicks
//    (indices marked with a short tick), overlay(g, X, Y, cs), rowTitle, colTitle, maxCell, maxH, colorbar
let PAT = 0;
function matrix(box, o) {
  box.replaceChildren();
  const nr = o.nr, nc = o.nc, Wc = Math.max(260, box.clientWidth || 600), CH = 6.4;
  const marks = !!(o.rowMark || o.colMark);
  const ml = Math.ceil(Math.max(0, ...o.rowLabels.map(t => String(t).length)) * CH) + (marks ? 22 : 10) + (o.rowTitle ? 16 : 0);
  const mr = o.colorbar === false ? 8 : 64, mt = 6;
  let cs = Math.min(o.maxCell || 26, (Wc - ml - mr) / nc, (o.maxH || 560) / nr);
  cs = Math.max(3, cs);
  const lw = Math.max(0, ...o.colLabels.map(t => String(t).length)) * CH;
  const rot = lw > cs * (Math.ceil(11 / cs)) - 2;
  const mb = (rot ? lw + 10 : 18) + (marks ? 10 : 0) + (o.colTitle ? 18 : 0);
  const W = ml + nc * cs + mr, H = mt + nr * cs + mb;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "al-matrix" }, box);
  const pid = `alh${++PAT}`;
  const pat = el("pattern", { id: pid, patternUnits: "userSpaceOnUse", width: 5, height: 5, patternTransform: "rotate(45)" }, el("defs", {}, svg));
  el("rect", { width: 5, height: 5, fill: "rgba(255,255,255,.55)" }, pat);
  el("line", { x1: 0, y1: 0, x2: 0, y2: 5, stroke: "#6b6b70", "stroke-width": 1.6 }, pat);
  const X = j => ml + j * cs + cs / 2, Y = i => mt + i * cs + cs / 2;
  const [zl, zh] = o.zrange, col = v => Number.isFinite(v) ? o.color((v - zl) / ((zh - zl) || 1)) : "#fff";
  const mid = `alm${++PAT}`, mp = el("pattern", { id: mid, patternUnits: "userSpaceOnUse", width: 6, height: 6 }, svg.querySelector("defs"));
  el("rect", { width: 6, height: 6, fill: "#fff" }, mp);
  el("path", { d: "M0,0L6,6M6,0L0,6", stroke: "#b9b8b2", "stroke-width": .8 }, mp);
  let missing = 0;
  if (nr * nc > 1600) {             // many cells: paint into a canvas and place it as one image
    const cv = document.createElement("canvas"); cv.width = nc; cv.height = nr;
    const cx = cv.getContext("2d");
    for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) { const v = o.value(i, j); if (!Number.isFinite(v)) missing++; cx.fillStyle = col(v); cx.fillRect(j, i, 1, 1); }
    el("image", { x: ml, y: mt, width: nc * cs, height: nr * cs, href: cv.toDataURL(), preserveAspectRatio: "none", style: "image-rendering: pixelated" }, svg);
  } else {
    const g = el("g", {}, svg), gh = el("g", {}, svg);
    for (let i = 0; i < nr; i++) for (let j = 0; j < nc; j++) {
      const v = o.value(i, j), fin = Number.isFinite(v), hat = fin && o.hatch && o.hatch(i, j);
      if (!fin) missing++;
      el("rect", { x: ml + j * cs, y: mt + i * cs, width: cs + .3, height: cs + .3, fill: fin ? col(v) : `url(#${mid})` }, g);
      if (hat) el("rect", { x: ml + j * cs, y: mt + i * cs, width: cs, height: cs, fill: `url(#${pid})` }, gh);
    }
  }
  const bl = el("g", { stroke: "#16161a", "stroke-width": 1, opacity: .55 }, svg);
  (o.rowBounds || []).forEach(i => el("line", { x1: ml, x2: ml + nc * cs, y1: mt + (i + 1) * cs, y2: mt + (i + 1) * cs }, bl));
  (o.colBounds || []).forEach(j => el("line", { y1: mt, y2: mt + nr * cs, x1: ml + (j + 1) * cs, x2: ml + (j + 1) * cs }, bl));
  el("rect", { x: ml, y: mt, width: nc * cs, height: nr * cs, fill: "none", stroke: "#c9c8c1" }, svg);
  const er = Math.ceil(11 / cs), ec = rot ? Math.ceil(11 / cs) : Math.ceil(Math.max(11, lw + 4) / cs);
  // which labels to draw: o.labelAt(i) if given, else every er-th / ec-th, always including the last
  const rowIdx = [], colIdx = [];
  for (let i = 0; i < nr; i++) if (o.labelAt ? o.labelAt(i) : i % er === 0 || (i === nr - 1 && (nr - 1) % er >= er / 2)) rowIdx.push(i);
  for (let j = 0; j < nc; j++) if (o.labelAt ? o.labelAt(j) : j % ec === 0 || (j === nc - 1 && (nc - 1) % ec >= ec / 2)) colIdx.push(j);
  for (const i of rowIdx) {
    const t = el("text", { x: ml - (marks ? 14 : 5), y: Y(i) + 3.8, "text-anchor": "end" }, svg); t.textContent = o.rowLabels[i];
  }
  for (const j of colIdx) {
    const y0 = mt + nr * cs + (marks ? 13 : 4);
    const t = el("text", rot ? { x: X(j) + 3.8, y: y0, "text-anchor": "end", transform: `rotate(-90 ${X(j) + 3.8} ${y0})` } : { x: X(j), y: y0 + 9, "text-anchor": "middle" }, svg);
    t.textContent = o.colLabels[j];
  }
  const mark = (x, y, filled) => el("circle", { cx: x, cy: y, r: Math.min(3.2, Math.max(2, cs / 3.5)), fill: filled ? "#16161a" : "#fff", stroke: "#16161a", "stroke-width": 1.1 }, svg);
  if (o.rowMark) for (let i = 0; i < nr; i++) { const m = o.rowMark(i); if (m != null) mark(ml - 7, Y(i), m); }
  if (o.colMark) for (let j = 0; j < nc; j++) { const m = o.colMark(j); if (m != null) mark(X(j), mt + nr * cs + 7, m); }
  (o.rowTicks || []).forEach(i => el("line", { x1: ml - 5, x2: ml - 1, y1: mt + (i + 1) * cs, y2: mt + (i + 1) * cs, stroke: "#c2410c", "stroke-width": 1.4 }, svg));
  (o.colTicks || []).forEach(j => el("line", { y1: mt + nr * cs + 1, y2: mt + nr * cs + 5, x1: ml + (j + 1) * cs, x2: ml + (j + 1) * cs, stroke: "#c2410c", "stroke-width": 1.4 }, svg));
  if (o.rowTitle) { const t = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(10,${mt + nr * cs / 2}) rotate(-90)` }, svg); t.textContent = o.rowTitle; }
  if (o.colTitle) { const t = el("text", { x: ml + nc * cs / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg); t.textContent = o.colTitle; }
  if (o.overlay) o.overlay(el("g", { "pointer-events": "none" }, svg), X, Y, cs);
  if (o.colorbar !== false) {
    const bx = ml + nc * cs + 14, bh = Math.min(150, Math.max(60, nr * cs)), gid = `alg${++PAT}`;
    const lg = el("linearGradient", { id: gid, x1: 0, x2: 0, y1: 1, y2: 0 }, el("defs", {}, svg));
    for (let i = 0; i <= 10; i++) el("stop", { offset: i / 10, "stop-color": o.color(i / 10) }, lg);
    el("rect", { x: bx, y: mt, width: 10, height: bh, fill: `url(#${gid})`, rx: 2 }, svg);
    const zf = o.zfmt || (v => (+v.toFixed(3)).toString());
    [[zh, mt + 8], [zl, mt + bh]].forEach(([v, y]) => { const t = el("text", { x: bx + 14, y, "text-anchor": "start" }, svg); t.textContent = zf(v); });
  }
  const hl = el("rect", { width: cs, height: cs, fill: "none", stroke: "#16161a", "stroke-width": 1.5, opacity: 0, "pointer-events": "none" }, svg);
  const hit = el("rect", { x: ml, y: mt, width: nc * cs, height: nr * cs, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), s = W / r.width;
    const j = Math.floor(((ev.clientX - r.left) * s - ml) / cs), i = Math.floor(((ev.clientY - r.top) * s - mt) / cs);
    if (i < 0 || j < 0 || i >= nr || j >= nc) return;
    hl.setAttribute("x", ml + j * cs); hl.setAttribute("y", mt + i * cs); hl.setAttribute("opacity", 1);
    const [head, rows] = o.tip(i, j);
    showTip(ev.clientX, ev.clientY, tipRows(head, rows));
  });
  hit.addEventListener("pointerleave", () => { hl.setAttribute("opacity", 0); hideTip(); });
  if (missing) box.append(h("div", { class: "chart-legend" }, h("span", { class: "k mute" }, h("span", { class: "al-ic al-ic-miss" }),
    o.missingText || `missing (${missing} cells): a snapshot whose operator is not available under this kernel rule`)));
  return { svg, X, Y, cs, missing };
}

// ------------------------------------------------------------------------------------------------ points and lines (SVG)
// o: x {kind: linear|log|symlog, domain, ticks?, fmt, label}, y {kind: linear|log, domain, fmt, label}, height,
//    series [{name, color, pts [{x, y, hollow?, tip?: rows}], dash?, width?, nomark?}], bands [{xs, lo, hi, label}],
//    hlines [{y, label, dash?}], vlines [{x, label, dash?}]
function plot(box, o) {
  box.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" }); box.append(wrap);
  const W = Math.max(260, box.clientWidth || 480), H = o.height || 260;
  const m = { l: 50, r: 14, t: o.y.label ? 24 : 12, b: o.x.label ? 38 : 24 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, wrap);
  const X = scale(o.x.kind, o.x.domain[0], o.x.domain[1], m.l, W - m.r);
  const Y = scale(o.y.kind === "log" ? "log" : "linear", o.y.domain[0], o.y.domain[1], H - m.b, m.t);
  const yf = o.y.fmt || (v => (+v.toPrecision(3)).toString()), xf = o.x.fmt || (v => String(v));
  const yt = o.y.kind === "log" ? (() => { const out = []; for (let e = Math.floor(Math.log10(o.y.domain[0])); e <= Math.ceil(Math.log10(o.y.domain[1])); e++) for (const mm of [1, 2, 5]) { const v = mm * 10 ** e; if (v >= o.y.domain[0] && v <= o.y.domain[1]) out.push(v); } return out.length > 7 ? out.filter(v => /^1/.test(String(v))) : out; })()
    : niceTicks(o.y.domain[0], o.y.domain[1], Math.max(3, Math.round(H / 55)));
  const g = el("g", { class: "grid" }, svg);
  for (const v of yt) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); const t = el("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yf(v); }
  const ax = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  let last = -1e9;
  for (const v of (o.x.ticks || niceTicks(o.x.domain[0], o.x.domain[1], 6))) {
    const px = X(v); if (px - last < 30) continue; last = px;
    el("line", { x1: px, x2: px, y1: H - m.b, y2: H - m.b + 4 }, ax);
    const t = el("text", { x: px, y: H - m.b + 15, "text-anchor": "middle" }, svg); t.textContent = xf(v);
  }
  if (o.x.label) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg); t.textContent = o.x.label; }
  if (o.y.label) { const t = el("text", { x: m.l - 40, y: 11, "text-anchor": "start", class: "ttl" }, svg); t.textContent = o.y.label; }
  const cy = v => Y(Math.min(o.y.domain[1], Math.max(o.y.domain[0], v)));
  for (const b of o.bands || []) {
    let d = "";
    b.xs.forEach((x, i) => { if (Number.isFinite(b.hi[i])) d += (d ? "L" : "M") + X(x) + "," + cy(b.hi[i]); });
    for (let i = b.xs.length - 1; i >= 0; i--) if (Number.isFinite(b.lo[i])) d += "L" + X(b.xs[i]) + "," + cy(b.lo[i]);
    if (d) el("path", { d: d + "Z", fill: b.fill || "#16161a", opacity: b.opacity ?? .08 }, svg);
  }
  for (const r of o.hlines || []) {
    el("line", { x1: m.l, x2: W - m.r, y1: cy(r.y), y2: cy(r.y), stroke: r.color || "#9a9993", "stroke-width": 1, "stroke-dasharray": r.dash ? "4 3" : null }, svg);
    if (r.label) { const t = el("text", { x: W - m.r - 4, y: cy(r.y) - 4, "text-anchor": "end" }, svg); t.textContent = r.label; }
  }
  for (const r of o.vlines || []) {
    el("line", { x1: X(r.x), x2: X(r.x), y1: m.t, y2: H - m.b, stroke: "#16161a", "stroke-width": 1, "stroke-dasharray": r.dash || "2 3", opacity: .7 }, svg);
    if (r.label) { const t = el("text", { x: X(r.x) + 3, y: m.t + 9, "text-anchor": "start" }, svg); t.textContent = r.label; }
  }
  const groups = [];
  const all = [];
  o.series.forEach((s, si) => {
    const gg = el("g", {}, svg); groups.push(gg);
    let d = "", pen = false;
    for (const p of s.pts) {
      if (!Number.isFinite(p.y) || (o.y.kind === "log" && p.y <= 0)) { pen = false; continue; }
      d += (pen ? "L" : "M") + X(p.x).toFixed(1) + "," + cy(p.y).toFixed(1); pen = true;
    }
    const pth = el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-linejoin": "round", opacity: s.opacity ?? 1 }, gg);
    if (s.dash) pth.setAttribute("stroke-dasharray", "5 4");
    for (const p of s.pts) {
      if (!Number.isFinite(p.y) || (o.y.kind === "log" && p.y <= 0)) continue;
      if (!s.nomark) el("circle", { cx: X(p.x), cy: cy(p.y), r: p.hollow ? 3.2 : 2.8, fill: p.hollow ? "#fff" : s.color, stroke: s.color, "stroke-width": p.hollow ? 1.4 : 1 }, gg);
      all.push({ s, si, p, px: X(p.x), py: cy(p.y) });
    }
  });
  const ring = el("circle", { r: 5.5, fill: "none", stroke: "#16161a", "stroke-width": 1.5, opacity: 0, "pointer-events": "none" }, svg);
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), sc = W / r.width, x = (ev.clientX - r.left) * sc, y = (ev.clientY - r.top) * sc;
    let best = null, bd = 18 * 18;
    for (const a of all) { const dd = (a.px - x) ** 2 + (a.py - y) ** 2; if (dd < bd) { bd = dd; best = a; } }
    if (!best) { ring.setAttribute("opacity", 0); hideTip(); return; }
    ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("opacity", 1);
    showTip(ev.clientX, ev.clientY, tipRows(best.s.name, [{ name: o.x.name || "x", value: xf(best.p.x) }, { name: o.y.name || "y", value: yf(best.p.y) }, ...(best.p.tip || [])]));
  });
  hit.addEventListener("pointerleave", () => { ring.setAttribute("opacity", 0); hideTip(); });
  const lg = h("div", { class: "chart-legend" });
  if (o.series.length > 1 && o.legend !== false) o.series.forEach((s, si) => {
    if (s.nolegend) return;
    const key = h("i", { style: { "--sw": s.color } });
    if (s.dash) key.style.background = `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`;
    const k = h("span", { class: "k" }, key, s.name);
    k.addEventListener("pointerenter", () => groups.forEach((gg, j) => gg.setAttribute("opacity", j === si ? 1 : .12)));
    k.addEventListener("pointerleave", () => groups.forEach(gg => gg.setAttribute("opacity", 1)));
    lg.append(k);
  });
  for (const x of o.extraLegend || []) lg.append(h("span", { class: "k mute" }, x.icon, x.label));
  if (lg.childNodes.length) wrap.append(lg);
  return { svg, X, Y };
}
const icon = {
  hollow: () => h("span", { class: "al-ic al-ic-hollow" }), filled: () => h("span", { class: "al-ic al-ic-filled" }),
  band: () => h("i", { style: { background: "rgba(22,22,26,.14)", height: "8px", borderRadius: "2px" } }),
  dash: (c = "#9a9993") => h("i", { style: { background: `repeating-linear-gradient(90deg, ${c} 0 4px, transparent 4px 7px)` } }),
  line: (c = "#9a9993") => h("i", { style: { "--sw": c } }),
  hatch: () => h("span", { class: "al-ic al-ic-hatch" }),
};
const symlogTicks = steps => steps;

// ------------------------------------------------------------------------------------------------ W1
async function w1(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const st = { net: +(ds.net || 0), step: I.final, fam: ds.fam || "proj", p: +(ds.p || 9), rule: +(ds.rule || 0) };
  const box = h("div", { class: "chart" }), info = h("p", { class: "al-info" });
  const stepSel = select([], st.step, v => { st.step = +v; draw(); });
  const setSteps = () => { const n = I.nets[st.net]; if (!n.steps.includes(st.step)) st.step = I.final;
    stepSel.setOpts(n.steps.map(s => [s, `step ${stepTxt(s)}${n.iso && n.iso.available && n.iso.step === s ? " (iso-accuracy)" : ""}`]), st.step); };
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("network"), select(I.nets.map((n, i) => [i, netName(n)]), st.net, v => { st.net = +v; setSteps(); draw(); }), lab("checkpoint"), stepSel),
    ctl(...distControls(I, st, () => draw())), box, info);
  setSteps();
  let tok = 0;
  const draw = async () => {
    const my = ++tok, id = did(st), [v, dg] = await Promise.all([slice(id, st.rule), diag(st.rule === undefined ? 0 : st.fam === "cka" ? I.rules[0] : st.rule)]);
    if (my !== tok) return;
    const n = I.nets[st.net], g = I.gid.get(`${st.net}:${st.step}`), M = mat(I, v, I.selfSeg.get(g)), cells = I.groups[g].cells, k = st.p, proj = st.fam === "proj";
    const bounds = stageBounds(n.stages);
    matrix(box, {
      nr: M.nr, nc: M.nc, rowLabels: n.sites, colLabels: n.sites, value: M.at, color: t => P.seq(t), zrange: [0, zmax(I, st)],
      hatch: proj ? (i, j) => i !== j && !(dg.ok(cells[i], k) && dg.ok(cells[j], k)) : null,
      rowMark: proj ? i => dg.mark(cells[i], k) : null, rowBounds: bounds, colBounds: bounds, rowTitle: "site i", colTitle: "site j",
      tip: (i, j) => [`${n.sites[i]} · ${n.sites[j]}`, [{ name: dShort(id), value: f3(M.at(i, j)) }, ...(proj && i !== j ? cutRows(I, dg, cells[i], cells[j], k) : [])]],
    });
    const adj = n.sites.slice(0, -1).map((_, i) => M.at(i, i + 1)), o = adj.map((_, i) => i).filter(i => Number.isFinite(adj[i])).sort((a, b) => adj[b] - adj[a]);
    const sib = I.nets.findIndex((m, i) => i !== st.net && m.depth === n.depth), x = I.cross.find(c => c.kind === "step" && c.step === st.step && ((c.a === st.net && c.b === sib) || (c.b === st.net && c.a === sib)));
    let fl = "";
    if (x) { const F = floorVals(I, dg, v, x.seg, n.sites.length, proj, k), d = F.vals;
      fl = d.length ? ` Seed floor at this checkpoint (same site, other seed${proj ? `; the ${F.n} of ${F.nAll} sites whose cut is resolved in both seeds` : ""}): median ${f3(med(d))}, range ${f3(d[0])}–${f3(d[d.length - 1])}.`
        : ` Seed floor at this checkpoint: no site has a resolved cut in both seeds.`; }
    const unres = proj ? cells.filter(c => dg.avail(c) && !dg.ok(c, k)).length : 0, nmiss = cells.filter(c => !dg.avail(c)).length;
    info.textContent = `${netName(n)}, step ${stepTxt(st.step)}, ${distLong(st)}, ${ruleTxt(st, I)}. ${o.length > 1 ? `Largest adjacent-site distance ${f3(adj[o[0]])} (${n.sites[o[0]]}–${n.sites[o[0] + 1]}), then ${f3(adj[o[1]])} (${n.sites[o[1]]}–${n.sites[o[1] + 1]})${o.length < adj.length ? ` (among the ${o.length} adjacent pairs with both operators)` : ""}.` : ""}${fl}` +
      (proj ? ` Cut ${k}|${k + 1} unresolved at ${unres} of ${cells.length - nmiss} sites (hollow dots); chance 1 − k/(N−1) = ${(1 - k / (I.n - 1)).toFixed(4)}.` : "") +
      (nmiss && st.fam !== "cka" ? ` ${nmiss} site${nmiss > 1 ? "s have" : " has"} no operator under this kernel rule (missing).` : "");
    if (proj) box.append(h("div", { class: "chart-legend" }, h("span", { class: "k mute" }, icon.hatch(), "unresolved cut in either snapshot"), h("span", { class: "k mute" }, icon.filled(), "site's cut resolved"), h("span", { class: "k mute" }, icon.hollow(), "unresolved")));
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ W2
async function w2(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const st = { net: 0, site: 6, fam: "proj", p: 9, rule: 0 };
  const left = h("div", { class: "chart" }), right = h("div", { class: "chart" }), info = h("p", { class: "al-info" });
  const siteSel = select([], st.site, v => { st.site = +v; draw(); });
  const setSites = () => { const n = I.nets[st.net]; st.site = Math.min(st.site, n.sites.length - 1); siteSel.setOpts(n.sites.map((s, i) => [i, s]), st.site); };
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("network"), select(I.nets.map((n, i) => [i, netName(n)]), st.net, v => { st.net = +v; setSites(); draw(); }), lab("site (right)"), siteSel),
    ctl(...distControls(I, st, () => draw())),
    h("div", { class: "al-panels" }, h("div", {}, h("p", { class: "chart-title", text: "Distance to the same site at the final checkpoint" }), left),
      h("div", {}, h("p", { class: "chart-title", text: "Checkpoint × checkpoint, one site" }), right)), info);
  setSites();
  let tok = 0;
  const draw = async () => {
    const my = ++tok, id = did(st), [v, dg] = await Promise.all([slice(id, st.rule), diag(st.fam === "cka" ? I.rules[0] : st.rule)]);
    if (my !== tok) return;
    const n = I.nets[st.net], S = I.main_steps, T = S.length, proj = st.fam === "proj", k = st.p;
    const cellAt = (t, si) => I.groups[I.gid.get(`${st.net}:${S[t]}`)].cells[si];
    const series = n.sites.map((name, si) => {
      const M = mat(I, v, I.trajSeg.get(`${st.net}:${si}`));
      return { name, color: P.ord(n.sites.length > 1 ? si / (n.sites.length - 1) : 1), width: 1.5,
        pts: S.map((s, t) => ({ x: s, y: M.at(t, T - 1), hollow: proj && t < T - 1 && !(dg.ok(cellAt(t, si), k) && dg.ok(cellAt(T - 1, si), k)),
          tip: proj && t < T - 1 ? cutRows(I, dg, cellAt(t, si), cellAt(T - 1, si), k).slice(0, 1) : [] })) };
    });
    const sib = I.nets.findIndex((m, i) => i !== st.net && m.depth === n.depth), x = I.cross.find(c => c.kind === "step" && c.step === I.final && ((c.a === st.net && c.b === sib) || (c.b === st.net && c.a === sib)));
    const bands = [];
    let flTxt = "";
    if (x) { const F = floorVals(I, dg, v, x.seg, n.sites.length, proj, k), d = F.vals;
      if (d.length) { const lo = d[0], hi = d[d.length - 1]; bands.push({ xs: [S[0], S[T - 1]], lo: [lo, lo], hi: [hi, hi] });
        flTxt = ` Seed floor at the final checkpoint: ${f3(lo)}–${f3(hi)} over ${proj ? `the ${F.n} of ${F.nAll} sites whose cut is resolved in both seeds` : "sites"}.`; }
      else flTxt = " Seed floor at the final checkpoint: no site has a resolved cut in both seeds (no band)."; }
    const ymax = Math.max(zmax(I, st) * (st.fam === "proj" ? 1 : 1), ...series.flatMap(s => s.pts.map(p => p.y)).filter(Number.isFinite));
    plot(left, { x: { kind: "symlog", domain: [0, S[T - 1]], ticks: S, fmt: fmtStep, label: "training step", name: "step" }, y: { domain: [0, st.fam === "proj" ? 1 : ymax * 1.05], label: dShort(id), name: dShort(id) },
      series, bands, height: 300, extraLegend: [...(bands.length ? [{ icon: icon.band(), label: `seed floor at the final checkpoint (range over sites${proj ? " with a resolved cut in both seeds" : ""})` }] : []), ...(proj ? [{ icon: icon.hollow(), label: "unresolved cut" }] : [])] });
    const M = mat(I, v, I.trajSeg.get(`${st.net}:${st.site}`)), lbl = S.map(fmtStep);
    matrix(right, { nr: T, nc: T, rowLabels: lbl, colLabels: lbl, value: M.at, color: t => P.seq(t), zrange: [0, zmax(I, st)], maxCell: 30,
      hatch: proj ? (i, j) => i !== j && !(dg.ok(cellAt(i, st.site), k) && dg.ok(cellAt(j, st.site), k)) : null, rowMark: proj ? i => dg.mark(cellAt(i, st.site), k) : null,
      rowTitle: "step", colTitle: "step", tip: (i, j) => [`${n.sites[st.site]}: step ${stepTxt(S[i])} · ${stepTxt(S[j])}`, [{ name: dShort(id), value: f3(M.at(i, j)) }, ...(proj && i !== j ? cutRows(I, dg, cellAt(i, st.site), cellAt(j, st.site), k) : [])]] });
    info.textContent = `${netName(n)}, ${distLong(st)}, ${ruleTxt(st, I)}.${flTxt}`;
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ W3
async function w3(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const st = { net: 0, step: I.final, rule: 0 };
  const box = h("div", { class: "chart" }), info = h("p", { class: "al-info" });
  const stepSel = select([], st.step, v => { st.step = +v; draw(); });
  const setSteps = () => { const n = I.nets[st.net]; if (!n.steps.includes(st.step)) st.step = I.final; stepSel.setOpts(n.steps.map(s => [s, `step ${stepTxt(s)}`]), st.step); };
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("network"), select(I.nets.map((n, i) => [i, netName(n)]), st.net, v => { st.net = +v; setSteps(); draw(); }), lab("checkpoint"), stepSel,
    seg(I.rules.map(r => [String(r), `rule ${r}: ${I.rule_names[r]}`]), st.rule, v => { st.rule = +v; draw(); })), box, info);
  setSteps();
  let tok = 0;
  const draw = async () => {
    const my = ++tok, K = I.k_grid, [vs, dg] = await Promise.all([Promise.all(K.map(k => slice(`proj_k${k}`, st.rule))), diag(st.rule)]);
    if (my !== tok) return;
    const n = I.nets[st.net], g = I.gid.get(`${st.net}:${st.step}`), cells = I.groups[g].cells, sg = I.selfSeg.get(g), L = n.sites.length;
    const series = [];
    for (let i = 0; i + 1 < L; i++) series.push({ name: `${n.sites[i]}–${n.sites[i + 1]}`, color: P.ord(L > 2 ? i / (L - 2) : 1), width: 1.4,
      pts: K.map((k, ki) => ({ x: k, y: mat(I, vs[ki], sg).at(i, i + 1), hollow: !(dg.ok(cells[i], k) && dg.ok(cells[i + 1], k)), tip: cutRows(I, dg, cells[i], cells[i + 1], k) })) });
    const sib = I.nets.findIndex((m, i) => i !== st.net && m.depth === n.depth);
    const x = I.cross.find(c => c.kind === "step" && c.step === st.step && ((c.a === st.net && c.b === sib) || (c.b === st.net && c.a === sib)));
    const bands = [], hl = [];
    if (x) {
      // floor band and median over sites whose cut k|k+1 is resolved in both seeds; where none is, over all sites (hollow)
      const fl = K.map((k, ki) => { const r = floorVals(I, dg, vs[ki], x.seg, L, true, k); return r.n ? { ...r, ok: true } : { ...floorVals(I, dg, vs[ki], x.seg, L, false, k), ok: false, n: 0 }; });
      // band only over runs of k where some site's floor cut is resolved; no band where none is
      let run = null;
      K.forEach((k, ki) => {
        if (fl[ki].ok && fl[ki].vals.length) { if (!run) { run = { xs: [], lo: [], hi: [] }; bands.push(run); } run.xs.push(k); run.lo.push(fl[ki].vals[0]); run.hi.push(fl[ki].vals[fl[ki].vals.length - 1]); }
        else run = null;
      });
      series.push({ name: "seed floor (median over sites)", color: "#6b6b70", dash: true, width: 1.3,
        pts: K.map((k, ki) => ({ x: k, y: med(fl[ki].vals), hollow: !fl[ki].ok, tip: [{ name: "sites with a resolved floor cut", value: `${fl[ki].n} of ${fl[ki].nAll}` }] })) });
    }
    series.push({ name: "chance 1 − k/(N−1)", color: "#9a9993", nomark: true, width: 1, pts: K.map(k => ({ x: k, y: 1 - k / (I.n - 1) })) });
    const hz = cells.map(c => I.horizon[String(st.rule)][c]).filter(x => x >= 0).sort((a, b) => a - b), hmed = hz.length ? hz[Math.floor(hz.length / 2)] : 0;
    const nmiss = cells.length - hz.length;
    const vl = [{ x: Math.max(1, Math.min(64, hmed)), label: `median horizon ${hmed}`, dash: "3 3" }];
    if (hz[0] !== hmed && hz[0] >= 1) vl.push({ x: hz[0], label: "", dash: "1 3" });
    if (hz.length && hz[hz.length - 1] !== hmed && hz[hz.length - 1] >= 1) vl.push({ x: Math.min(64, hz[hz.length - 1]), label: "", dash: "1 3" });
    plot(box, { x: { kind: "log", domain: [1, 64], ticks: [1, 2, 4, 8, 16, 32, 64], label: "k (leading eigenfunctions kept)", name: "k" }, y: { domain: [0, 1.02], label: "dₖ between adjacent sites", name: "dₖ" },
      series, bands, vlines: vl, height: 340,
      extraLegend: [{ icon: icon.band(), label: "seed floor (range over sites whose cut is resolved in both seeds; no band where no site's is; the median is then hollow and over all sites)" }, { icon: icon.filled(), label: "resolved cut" }, { icon: icon.hollow(), label: "unresolved cut" }, { icon: icon.dash("#16161a"), label: `stability horizon: median, min ${hz[0]}, max ${hz[hz.length - 1]} (dotted)` }] });
    info.textContent = `${netName(n)}, step ${stepTxt(st.step)}, kernel rule ${st.rule} (${I.rule_names[st.rule]}). Stability horizons of the ${hz.length} sites, sorted: ${hz.join(", ")}.` +
      (nmiss ? ` ${nmiss} site${nmiss > 1 ? "s have" : " has"} no operator under this rule; lines touching them are missing.` : "");
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ W4
async function w4(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const st = { rule: 0, set: "within" };
  const box = h("div", { class: "chart" }), info = h("p", { class: "al-info" });
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("pairs"), seg([["within", "within one network"], ["cross", "across networks, final checkpoint"]], st.set, v => { st.set = v; draw(); }),
    seg(I.rules.map(r => [String(r), `rule ${r}: ${I.rule_names[r]}`]), st.rule, v => { st.rule = +v; draw(); })), box, info);
  const draw = () => {
    const W = I.w4[`${st.set}_r${st.rule}`], ids = I.w4_dists, lbl = ids.map(dShort), Z = W.z.map(r => r.map(v => v == null ? NaN : v));
    const fam = ids.map(i => i.split("_")[0]), bnd = []; for (let i = 0; i + 1 < fam.length; i++) if (fam[i] !== fam[i + 1]) bnd.push(i);
    matrix(box, { nr: ids.length, nc: ids.length, rowLabels: lbl, colLabels: lbl, value: (i, j) => Z[i][j], color: t => P.div(2 * t - 1), zrange: [-1, 1], maxCell: 16, maxH: 720,
      rowBounds: bnd, colBounds: bnd, zfmt: v => v.toFixed(0), missingText: "undefined: one of the two distances is constant over these pairs (no ranking)", tip: (i, j) => [`${lbl[i]} · ${lbl[j]}`, [{ name: "Spearman", value: f3(Z[i][j]) }]] });
    info.textContent = `Spearman rank correlation over ${W.n.toLocaleString("en")} snapshot pairs${W.dropped ? ` (${W.dropped.toLocaleString("en")} pairs with a missing value dropped)` : ""} (${st.set === "within" ? "pairs of one network that share a checkpoint or a site" : "site pairs of two networks with different seeds at the final checkpoint"}), kernel rule ${st.rule}. d = projector distance, h = diffusion-kernel distance, s = eigenvalue-only distance (ignores eigenvectors), CKA = baseline; the number is k or t.`;
  };
  draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ cross-network helpers
function crossOf(I, a, b, ck) {
  const [lo, hi] = a < b ? [a, b] : [b, a];
  const xi = I.cross.findIndex(x => x.a === lo && x.b === hi && (ck === "iso" ? x.kind === "iso" : x.kind === "step" && x.step === +ck));
  return xi;
}
const ckOpts = (I, pairs) => [...I.main_steps.map(s => [String(s), `step ${stepTxt(s)}${s === 0 ? " (untrained)" : ""}`]),
  ...(pairs.some(([a, b]) => crossOf(I, a, b, "iso") >= 0) ? [["iso", "iso-accuracy (secondary)"]] : [])];
const depthPairs = I => {
  const out = new Map();
  I.cross.forEach((x, xi) => { const da = I.nets[x.a].depth, db = I.nets[x.b].depth, key = `${da}-${db}`; if (!out.has(key)) out.set(key, []); });
  return [...out.keys()];
};
const pairsOfDepth = (I, key) => { const [da, db] = key.split("-").map(Number), out = [];
  I.nets.forEach((na, a) => I.nets.forEach((nb, b) => { if (a < b && na.depth === da && nb.depth === db) out.push([a, b]); })); return out; };
const dpName = key => { const [a, b] = key.split("-"); return a === b ? `ResNet-${a}, seed 0 vs seed 1` : `ResNet-${a} vs ResNet-${b}`; };
const pairName = (I, a, b) => `${I.nets[a].id}–${I.nets[b].id}`;
// different depth, same seed: the two networks share their initial draws for the stem and the first blocks
const sameSeed = (I, a, b) => I.nets[a].depth !== I.nets[b].depth && I.nets[a].seed === I.nets[b].seed;
const SHARED = "same seed: shared initial draws";
const splitSeeds = (I, xs) => ({ diff: xs.filter(xi => !I.cross[xi].same_seed), same: xs.filter(xi => I.cross[xi].same_seed) });
function summaryRow(S, id, xi) { const r = S.dists[id] && S.dists[id][xi]; if (!r) return null; const o = {}; S.fields.forEach((f, i) => o[f] = r[i]); return o; }
const ckName = (I, x) => x.kind === "iso" ? `iso-accuracy (${netName(I.nets[x.a])} step ${stepTxt(I.groups[x.ga].step)}, ${netName(I.nets[x.b])} step ${stepTxt(I.groups[x.gb].step)})` : `step ${stepTxt(x.step)}`;

// ------------------------------------------------------------------------------------------------ C1
async function c1(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const r56 = I.nets.findIndex(n => n.depth === Math.max(...I.nets.map(m => m.depth)));
  // default: the shallowest seed-0 network against the deepest network of the other seed (different-seed pairing)
  const deepOther = I.nets.findIndex(n => n.depth === Math.max(...I.nets.map(m => m.depth)) && n.seed !== I.nets[0].seed);
  const st = { a: 0, b: deepOther >= 0 ? deepOther : r56 >= 0 ? r56 : 1, ck: String(I.final), fam: "proj", p: 9, rule: 0 };
  const box = h("div", { class: "chart" }), info = h("p", { class: "al-info" }), legend = h("div", { class: "chart-legend" }), warn = h("p", { class: "al-warn" });
  const optsFor = other => I.nets.map((n, i) => [i, netName(n) + (sameSeed(I, i, other) ? " · shared initial draws" : "")]);
  const ckSel = select([], st.ck, v => { st.ck = v; draw(); });
  const setCk = () => { const o = ckOpts(I, [[st.a, st.b]]); if (!o.some(x => x[0] === st.ck)) st.ck = String(I.final); ckSel.setOpts(o, st.ck); };
  const relabel = () => { aSel.setOpts(optsFor(st.b), st.a); bSel.setOpts(optsFor(st.a), st.b); };
  const aSel = select([], st.a, v => { st.a = +v; relabel(); setCk(); draw(); }), bSel = select([], st.b, v => { st.b = +v; relabel(); setCk(); draw(); });
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("network A"), aSel, lab("network B"), bSel, lab("checkpoint"), ckSel), ctl(...distControls(I, st, () => draw())), warn, box, legend, info);
  relabel(); setCk();
  let tok = 0;
  const draw = async () => {
    const my = ++tok;
    if (st.a === st.b) { box.replaceChildren(h("div", { class: "placeholder", text: "Choose two different networks (the two seeds of one depth are a pair of their own)." })); info.textContent = ""; legend.replaceChildren(); return; }
    const xi = crossOf(I, st.a, st.b, st.ck);
    if (xi < 0) { box.replaceChildren(h("div", { class: "placeholder", text: "This comparison is not available at the chosen checkpoint (iso-accuracy needs ResNet-20 against a deeper network, or the two seeds of a deeper network, with an accepted iso-accuracy checkpoint)." })); info.textContent = ""; legend.replaceChildren(); return; }
    const id = did(st), [v, pth, S, dg] = await Promise.all([slice(id, st.rule), paths(id, st.rule), summary(st.fam === "cka" ? I.rules[0] : st.rule), diag(st.fam === "cka" ? I.rules[0] : st.rule)]);
    if (my !== tok) return;
    const x = I.cross[xi], A = I.nets[x.a], B = I.nets[x.b], pr = pth[xi], sm = summaryRow(S, id, xi), k = st.p, proj = st.fam === "proj";
    const ca = I.groups[x.ga].cells, cb = I.groups[x.gb].cells;
    warn.hidden = !x.same_seed;
    warn.textContent = x.same_seed ? `Same seed, different depth: ${netName(A)} and ${netName(B)} share their initial draws for the stem and the first blocks (same random generator, same construction order), which shows most at early checkpoints. Summaries on this tab use different-seed pairings.` : "";
    // display: the network with more sites on the rows (y), A's sites on the columns otherwise as in the caption
    const tr = B.sites.length >= A.sites.length;          // rows = B, columns = A
    const M = mat(I, v, x.seg, tr), R = tr ? B : A, C = tr ? A : B, rc = tr ? cb : ca, cc = tr ? ca : cb;
    const pt = (i, j) => tr ? [j, i] : [i, j];               // data (A site i, B site j) -> display (row, col)
    const bR = stageBounds(R.stages), bC = stageBounds(C.stages);
    matrix(box, {
      nr: M.nr, nc: M.nc, rowLabels: R.sites, colLabels: C.sites, value: M.at, color: t => P.seq(t), zrange: [0, zmax(I, st)], maxCell: 24, maxH: 620,
      rowBounds: bR, colBounds: bC, rowTitle: `${netName(R)} (${tr ? "B" : "A"})`, colTitle: `${netName(C)} (${tr ? "A" : "B"})`,
      hatch: proj ? (i, j) => !(dg.ok(rc[i], k) && dg.ok(cc[j], k)) : null, rowMark: proj ? i => dg.mark(rc[i], k) : null, colMark: proj ? j => dg.mark(cc[j], k) : null,
      tip: (i, j) => [`${R.sites[i]} (${R.id}) · ${C.sites[j]} (${C.id})`, [{ name: dShort(id), value: f3(M.at(i, j)) }, ...(proj ? cutRows(I, dg, rc[i], cc[j], k) : [])]],
      overlay: (g, X, Y, cs) => {
        if (!pr) return;
        const P2 = []; for (let q = 0; q < pr.p.length; q += 2) P2.push(pt(pr.p[q], pr.p[q + 1]));
        el("polyline", { points: P2.map(([r, c]) => `${X(c)},${Y(r)}`).join(" "), fill: "none", stroke: "#fff", "stroke-width": 4.5, "stroke-linejoin": "round", opacity: .8 }, g);
        el("polyline", { points: P2.map(([r, c]) => `${X(c)},${Y(r)}`).join(" "), fill: "none", stroke: "#c2410c", "stroke-width": 2.2, "stroke-linejoin": "round" }, g);
        const s = Math.max(2.2, Math.min(4.5, cs / 3.2));
        x.ref_map.forEach((jb, ia) => { const [r, c] = pt(ia, jb); el("path", { d: `M${X(c)},${Y(r) - s}L${X(c) + s},${Y(r)}L${X(c)},${Y(r) + s}L${X(c) - s},${Y(r)}Z`, fill: "#16161a", stroke: "#fff", "stroke-width": .8 }, g); });
        pr.ra.forEach((jb, ia) => { const [r, c] = pt(ia, jb); el("circle", { cx: X(c), cy: Y(r), r: s * 1.25, fill: "none", stroke: "#fff", "stroke-width": 2.6 }, g); el("circle", { cx: X(c), cy: Y(r), r: s * 1.25, fill: "none", stroke: "#2a78d6", "stroke-width": 1.5 }, g); });
        pr.ca.forEach((ia, jb) => { const [r, c] = pt(ia, jb), q = s * 1.1;
          el("path", { d: `M${X(c) - q},${Y(r) - q}L${X(c) + q},${Y(r) + q}M${X(c) - q},${Y(r) + q}L${X(c) + q},${Y(r) - q}`, stroke: "#fff", "stroke-width": 3 }, g);
          el("path", { d: `M${X(c) - q},${Y(r) - q}L${X(c) + q},${Y(r) + q}M${X(c) - q},${Y(r) + q}L${X(c) + q},${Y(r) - q}`, stroke: "#16161a", "stroke-width": 1.3 }, g); });
      },
    });
    legend.replaceChildren(h("span", { class: "k" }, h("i", { style: { "--sw": "#c2410c", height: "3px" } }), "DTW path"), h("span", { class: "k" }, h("span", { class: "al-ic al-ic-diamond" }), "architectural reference"),
      h("span", { class: "k" }, h("span", { class: "al-ic al-ic-ring" }), `nearest ${B.id} site for each ${A.id} site`), h("span", { class: "k" }, h("span", { class: "al-ic al-ic-cross" }), `nearest ${A.id} site for each ${B.id} site`),
      h("span", { class: "k" }, h("i", { style: { "--sw": "#16161a", opacity: .55 } }), "stage boundary"), ...(proj ? [h("span", { class: "k mute" }, icon.hatch(), "unresolved cut")] : []));
    if (!sm || sm.cost == null) { info.textContent = `No DTW result for this comparison: ${[...ca, ...cb].filter(c => !dg.avail(c)).length} of its snapshots have no operator under this kernel rule, so the comparison is excluded from every summary.`; return; }
    const ends = x.stage_ends.map(([ia, jb], q) => `${A.sites[ia]}/${B.sites[jb]}`).join(", ");
    info.innerHTML = `<b>${netName(A)}</b> (A) and <b>${netName(B)}</b> (B), ${ckName(I, x)}, ${distLong(st)}, ${ruleTxt(st, I)}. ` +
      `DTW cost <b>${f3(sm.cost)}</b> (path of ${sm.plen} steps); permutation null 5th / 50th / 95th percentile ${f3(sm.n05)} / ${f3(sm.n50)} / ${f3(sm.n95)}, fraction of null draws at or below the cost ${f3(sm.pnull)} (${I.n_perm.toLocaleString("en")} draws); ` +
      `constant-matrix reference ${f3(sm.const)}; seed-floor cost ${A.id.slice(0, -2)} ${f3(sm.floor_a)}, ${B.id.slice(0, -2)} ${f3(sm.floor_b)}. ` +
      `Mean relative-depth gap from the architectural reference ${f3(sm.gap)}; stage ends passed ${sm.ends}/3 (${ends}). Inversion fraction of the argmins: rows ${f2(sm.inv_r)}, columns ${f2(sm.inv_c)}.` +
      (proj ? ` Share of the path with an unresolved cut: ${f2(sm.unres)}.` : "");
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ C2
async function c2(body, ds) {
  const I = await index();
  const dps = depthPairs(I), deep = Math.max(...I.nets.map(n => n.depth));
  const st = { dp: dps.find(k => k === `${I.base_depth}-${deep}`) || dps[0], ck: String(I.final), rule: 0, rows: "some" };
  const box = h("div", { class: "bn-fig" }), info = h("p", { class: "al-info" });
  const ckSel = select([], st.ck, v => { st.ck = v; draw(); });
  const setCk = () => { const o = ckOpts(I, pairsOfDepth(I, st.dp)); if (!o.some(x => x[0] === st.ck)) st.ck = String(I.final); ckSel.setOpts(o, st.ck); };
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("networks"), select(dps.map(k => [k, dpName(k)]), st.dp, v => { st.dp = v; setCk(); draw(); }), lab("checkpoint"), ckSel),
    ctl(seg(I.rules.map(r => [String(r), `rule ${r}: ${I.rule_names[r]}`]), st.rule, v => { st.rule = +v; draw(); }), lab("rows"), seg([["some", "selected k and t"], ["all", "every k and t"]], st.rows, v => { st.rows = v; draw(); })), info, box);
  setCk();
  const SOME = new Set([1, 2, 4, 8, 16, 32, 64]);
  const draw = async () => {
    const S = await summary(st.rule);
    const xsAll = pairsOfDepth(I, st.dp).map(([a, b]) => crossOf(I, a, b, st.ck)).filter(xi => xi >= 0);
    if (!xsAll.length) { box.replaceChildren(h("div", { class: "placeholder", text: "No comparison of these networks at this checkpoint." })); info.textContent = ""; return; }
    const { diff, same } = splitSeeds(I, xsAll), xs = diff.length ? diff : same;
    const ids = I.dists.filter(d => st.rows === "all" || d.fam === "cka" || d.fam === "heat" || SOME.has(d.p) && (d.fam === "proj" || d.p === 8 || d.p === 64)).map(d => d.id);
    const cols = [["gap", "gap", f3], ["ends", "stage ends", v => v == null ? "n/a" : `${v}/3`], ["cost", "cost", f3], ["floor_a", `floor ${I.nets[I.cross[xs[0]].a].depth}`, f3], ["floor_b", `floor ${I.nets[I.cross[xs[0]].b].depth}`, f3],
      ["n05", "null 5th pct", f3], ["pnull", "null ≤ cost", f2], ["const", "constant", f3], ["inv_r", "inversions", f2], ["unres", "unresolved", f2]];
    const t = h("table", { class: "bn-table al-table" }, h("tr", {}, h("th", { text: "distance" }), ...cols.map(c => h("th", { text: c[1] }))));
    let fam0 = null;
    for (const id of ids) {
      const fam = id.split("_")[0];
      if (fam !== fam0) { t.append(h("tr", { class: "al-fam" }, h("td", { colspan: cols.length + 1, text: { proj: "projector distance dₖ", heat: "diffusion-kernel distance hₜ", eigd: "eigenvalue-only distance sₖ (ignores eigenvectors)", cka: "CKA distance (baseline)" }[fam] }))); fam0 = fam; }
      const line = (xl, f, fmt) => { const vals = xl.map(xi => summaryRow(S, id, xi)).map(r => r ? fmt(r[f]) : "n/a");
        return (f === "floor_a" || f === "floor_b") && vals.every(z => z === vals[0]) ? vals[0] : vals.join(" · "); };
      t.append(h("tr", {}, h("td", { text: dShort(id) }), ...cols.map(([f, , fmt]) => {
        if (f === "unres" && fam !== "proj") return h("td", { class: "mute", text: "–" });
        const td = h("td", {}, h("div", { text: line(xs, f, fmt) }));
        if (diff.length && same.length && f !== "floor_a" && f !== "floor_b") td.append(h("div", { class: "al-same", text: line(same, f, fmt) }));
        return td;
      })));
    }
    box.replaceChildren(t);
    const x0 = I.cross[xs[0]], miss = xsAll.filter(xi => !summaryRow(S, ids[0], xi) || summaryRow(S, ids[0], xi).cost == null).length;
    info.textContent = `${dpName(st.dp)}, ${ckName(I, x0).replace(/ \(.*/, "")}, kernel rule ${st.rule} (CKA rows: no kernel rule). ${diff.length && same.length ? "First line of each cell, different-seed pairings" : "Values in each cell"}, in order: ${xs.map(xi => pairName(I, I.cross[xi].a, I.cross[xi].b)).join(" · ")}` +
      (diff.length && same.length ? `; second line (grey), same-seed pairings, which share initial draws: ${same.map(xi => pairName(I, I.cross[xi].a, I.cross[xi].b)).join(" · ")}` : "") +
      `${st.ck === "iso" ? ` (iso-accuracy checkpoint: ${[...new Set(xsAll.map(xi => `${I.nets[I.cross[xi].b].id} step ${stepTxt(I.groups[I.cross[xi].gb].step)}`))].join(", ")})` : ""}. A floor shown once is the same for every pairing. Null: ${I.n_perm.toLocaleString("en")} block permutations.` +
      (miss ? ` ${miss} pairing${miss > 1 ? "s are" : " is"} excluded (n/a): a snapshot has no operator under this kernel rule.` : "");
  };
  await draw();
}

// ------------------------------------------------------------------------------------------------ C3
async function c3(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const dps = depthPairs(I), deep = Math.max(...I.nets.map(n => n.depth));
  const st = { dp: dps.find(k => k === `${I.base_depth}-${deep}`) || dps[0], ck: String(I.final), fam: "proj", p: 9, rule: 0, pairs: "diff" };
  const top = h("div", { class: "chart" }), bot = h("div", { class: "chart" }), info = h("p", { class: "al-info" });
  const ckSel = select([], st.ck, v => { st.ck = v; draw(); });
  const setCk = () => { const o = ckOpts(I, pairsOfDepth(I, st.dp)); if (!o.some(x => x[0] === st.ck)) st.ck = String(I.final); ckSel.setOpts(o, st.ck); };
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("networks"), select(dps.map(k => [k, dpName(k)]), st.dp, v => { st.dp = v; setCk(); draw(); }), lab("checkpoint"), ckSel,
      lab("pairings"), seg([["diff", "different seeds"], ["all", `also same seed (${SHARED})`]], st.pairs, v => { st.pairs = v; draw(); })), ctl(...distControls(I, st, () => draw())),
    h("p", { class: "chart-title", text: "Distance along the DTW path (solid) and seed floor at the same point (dashed)" }), top,
    h("p", { class: "chart-title", text: "Ratio of distance to seed floor" }), bot, info);
  setCk();
  let tok = 0;
  const draw = async () => {
    const my = ++tok, id = did(st), [v, pth, dg] = await Promise.all([slice(id, st.rule), paths(id, st.rule), diag(st.fam === "cka" ? I.rules[0] : st.rule)]);
    if (my !== tok) return;
    const xs = pairsOfDepth(I, st.dp).map(([a, b]) => crossOf(I, a, b, st.ck)).filter(xi => xi >= 0 && (st.pairs === "all" || !I.cross[xi].same_seed)), proj = st.fam === "proj", k = st.p;
    const s1 = [], s2 = [], notes = [];
    let ymax = 0, rmin = Infinity, rmax = 0;
    xs.forEach((xi, q) => {
      const x = I.cross[xi], A = I.nets[x.a], B = I.nets[x.b], pr = pth[xi];
      if (!pr) { notes.push(`${pairName(I, x.a, x.b)}: excluded (a snapshot has no operator under this rule)`); return; }
      const M = mat(I, v, x.seg), LA = A.sites.length, LB = B.sites.length, ca = I.groups[x.ga].cells, cb = I.groups[x.gb].cells;
      const FA = x.floor_a >= 0 ? mat(I, v, x.floor_a) : null, FB = x.floor_b >= 0 ? mat(I, v, x.floor_b) : null;
      const color = P.set(q), d = [], f = [], r = [];
      for (let z = 0; z < pr.p.length; z += 2) {
        const i = pr.p[z], j = pr.p[z + 1], xx = (i / (LA - 1) + j / (LB - 1)) / 2, val = M.at(i, j);
        const fl = FA && FB ? (FA.at(i, i) + FB.at(j, j)) / 2 : NaN, hol = proj && !(dg.ok(ca[i], k) && dg.ok(cb[j], k));
        // the floor uses the sibling seeds' snapshots too: its cut is resolved only if all four floor snapshots' cuts are
        const fhol = proj && FA && FB && !(floorOk(I, dg, x.floor_a, i, k) && floorOk(I, dg, x.floor_b, j, k));
        const tip = [{ name: "sites", value: `${A.sites[i]} / ${B.sites[j]}` }, { name: "seed floor", value: f3(fl) },
          ...(proj ? [{ name: "cut (pair, floor)", value: `${hol ? "unresolved" : "resolved"}, ${fhol ? "unresolved" : "resolved"}` }] : [])];
        d.push({ x: xx, y: val, hollow: hol, tip }); f.push({ x: xx, y: fl, hollow: fhol, tip }); r.push({ x: xx, y: val / fl, hollow: hol || fhol, tip });
        ymax = Math.max(ymax, val, Number.isFinite(fl) ? fl : 0);
        if (Number.isFinite(val / fl) && val / fl > 0) { rmin = Math.min(rmin, val / fl); rmax = Math.max(rmax, val / fl); }
      }
      const nm = pairName(I, x.a, x.b) + (x.same_seed ? ` (${SHARED})` : "");
      s1.push({ name: nm, color, pts: d, width: 1.6 }, { name: `${nm} floor`, color, pts: f, dash: true, width: 1.2, nolegend: true, opacity: .8 });
      s2.push({ name: nm, color, pts: r, width: 1.6 });
      if (!FA || !FB) notes.push(`${nm}: no seed floor for one side`);
    });
    if (!s1.length) { top.replaceChildren(h("div", { class: "placeholder", text: xs.length ? `Every comparison shown here needs a snapshot that has no operator under this kernel rule (${notes.join("; ")}).` : "No comparison of these networks at this checkpoint." })); bot.replaceChildren(); info.textContent = ""; return; }
    const xax = { kind: "linear", domain: [0, 1], label: "relative depth along the path", name: "relative depth", fmt: v => v.toFixed(1) };
    plot(top, { x: xax, y: { domain: [0, Math.max(1e-6, ymax) * 1.05], label: dShort(id), name: dShort(id) }, series: s1, height: 260,
      extraLegend: [{ icon: icon.dash("#6b6b70"), label: "seed floor (same colour)" }, ...(proj ? [{ icon: icon.hollow(), label: "unresolved cut (for a ratio: in the pair or in the floor)" }] : [])] });
    if (Number.isFinite(rmin)) plot(bot, { x: xax, y: { kind: "log", domain: [Math.min(1, rmin) / 1.2, Math.max(1, rmax) * 1.2], label: "distance / seed floor", name: "ratio", fmt: v => (+v.toPrecision(2)).toString() },
      series: s2, height: 220, hlines: [{ y: 1, label: "= seed floor" }] });
    else bot.replaceChildren(h("div", { class: "placeholder", text: "No seed floor for this comparison." }));
    info.textContent = `${dpName(st.dp)}, ${st.ck === "iso" ? "iso-accuracy" : `step ${stepTxt(+st.ck)}`}, ${distLong(st)}, ${ruleTxt(st, I)}.${notes.length ? " " + notes.join("; ") + "." : ""}`;
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ C4
async function c4(body, ds) {
  const [I, P] = await Promise.all([index(), pal()]);
  const dps = depthPairs(I), deep = Math.max(...I.nets.map(n => n.depth));
  const st = { dp: dps.find(k => k === `${I.base_depth}-${deep}`) || dps[0], fam: "proj", p: 9, rule: 0 };
  const panels = h("div", { class: "al-panels" }), info = h("p", { class: "al-info" });
  body.replaceChildren(fixtureTag(I) || "", ctl(lab("networks"), select(dps.map(k => [k, dpName(k)]), st.dp, v => { st.dp = v; draw(); })), ctl(...distControls(I, st, () => draw())), panels, info);
  const draw = async () => {
    const id = did(st), S = await summary(st.fam === "cka" ? I.rules[0] : st.rule), steps = I.main_steps, prs = pairsOfDepth(I, st.dp);
    const rows = prs.map(([a, b]) => steps.map(s => summaryRow(S, id, crossOf(I, a, b, String(s)))));
    // different-seed pairings solid (with their null bands); same-seed pairings of different depth dashed, labelled
    const ss = prs.map(([a, b]) => sameSeed(I, a, b));
    const ser = (f, extra = {}) => prs.map(([a, b], q) => ({ name: pairName(I, a, b) + (ss[q] ? ` (${SHARED})` : ""), color: P.set(q), dots: 2.5, width: ss[q] ? 1.3 : 1.8, dash: ss[q],
      y: rows[q].map(r => r && r[f] != null ? r[f] : NaN), ...extra }));
    const cost = ser("cost").map((s, q) => ss[q] ? s : ({ ...s, band: [rows[q].map(r => r?.n05 ?? NaN), rows[q].map(r => r?.n95 ?? NaN)] }));
    const [da, db] = st.dp.split("-");
    const fl = [{ name: `seed floor ResNet-${da}`, color: "#6b6b70", dash: true, width: 1.3, y: rows[0].map(r => r?.floor_a ?? NaN) }];
    if (da !== db) fl.push({ name: `seed floor ResNet-${db}`, color: "#9a9993", dash: true, width: 1.3, y: rows[0].map(r => r?.floor_b ?? NaN) });
    const titles = ["DTW cost (band: null 5th–95th percentile)", "Gap from the architectural reference", "Inversion fraction of the row argmins", ...(st.fam === "proj" ? ["Share of the path with an unresolved cut"] : [])];
    const boxes = titles.map(() => h("div", { class: "chart" }));
    panels.replaceChildren(...titles.map((t, i) => h("div", {}, h("p", { class: "chart-title", text: t }), boxes[i])));
    const mk = t => boxes[titles.indexOf(t)];
    const gmax = Math.max(1e-6, ...rows.flat().map(r => r?.gap ?? 0));
    const common = { x: steps, xscale: "symlog", xticks: steps.filter((_, i) => i === 0 || i === 2 || i === 4 || i === steps.length - 1), xfmt: fmtStep, xname: "step", xlabel: "training step", height: 220 };
    lineChart(mk("DTW cost (band: null 5th–95th percentile)"), { ...common, series: [...cost, ...fl], bandMode: "envelope", bandLabel: "grey: permutation null of the different-seed pairings, 5th to 95th percentile (hover a key for one pairing)", yfmt: v => (+v.toFixed(3)).toString() });
    lineChart(mk("Gap from the architectural reference"), { ...common, series: ser("gap"), yrange: [0, gmax * 1.1], yfmt: v => (+v.toFixed(3)).toString() });
    lineChart(mk("Inversion fraction of the row argmins"), { ...common, series: ser("inv_r"), yrange: [0, 1] });
    if (st.fam === "proj") lineChart(mk("Share of the path with an unresolved cut"), { ...common, series: ser("unres"), yrange: [0, 1] });
    info.textContent = `${dpName(st.dp)}, ${distLong(st)}, ${ruleTxt(st, I)}; the eight common checkpoints (step 0 is the untrained reference).` +
      (ss.some(Boolean) ? ` Dashed coloured lines: same-seed pairings, whose networks share their initial draws for the stem and the first blocks; solid: different-seed pairings.` : "");
  };
  await draw(); onResize(body, draw);
}

// ------------------------------------------------------------------------------------------------ C5 (live linear algebra)
const vecCache = new Map();
function vectors(I, net, step, site) {
  const key = `${I.nets[net].id}_${step}_${site}`;
  if (!vecCache.has(key)) {
    const p = (async () => {
      const b = await D.bytes(`${AL}U/${key}.bin`), sc = new Float32Array(b, 0, 64), q = new Int16Array(b, 256), n = q.length / 64, U = new Float64Array(q.length);
      for (let i = 0; i < n; i++) for (let a = 0; a < 64; a++) U[i * 64 + a] = q[i * 64 + a] * sc[a] / 32767;
      return { U, n };
    })();
    p.catch(() => vecCache.delete(key));
    vecCache.set(key, p);
    if (vecCache.size > 12) vecCache.delete(vecCache.keys().next().value);
  }
  return vecCache.get(key);
}
// eigenvalues of a symmetric matrix (cyclic Jacobi), descending
export function jacobiEig(Ain, n) {
  const A = Float64Array.from(Ain);
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let r = 0; r < n; r++) { const arp = A[r * n + p], arq = A[r * n + q]; A[r * n + p] = c * arp - s * arq; A[r * n + q] = s * arp + c * arq; }
      for (let r = 0; r < n; r++) { const apr = A[p * n + r], aqr = A[q * n + r]; A[p * n + r] = c * apr - s * aqr; A[q * n + r] = s * apr + c * aqr; }
    }
  }
  return Array.from({ length: n }, (_, i) => A[i * n + i]).sort((a, b) => b - a);
}
async function c5(body, ds) {
  const [I, P, m] = await Promise.all([index(), pal(), D.meta()]);
  if (!I.vec_cells.length) { body.replaceChildren(h("div", { class: "placeholder", text: "No eigenvectors were shipped." })); return; }
  const grp = [...new Map(I.vec_cells.map(v => [`${v.net}:${v.step}`, v])).values()];
  const gOpts = grp.map(v => [`${v.net}:${v.step}`, `${netName(I.nets[v.net])}, step ${stepTxt(v.step)}`]);
  const sib = I.nets.findIndex((n, i) => i > 0 && n.depth === I.nets[0].depth);
  const st = { A: `0:${I.final}`, sa: 6, B: `${sib >= 0 ? sib : 0}:${I.final}`, sb: 6, k: 9 };
  if (!gOpts.some(o => o[0] === st.B)) st.B = gOpts[gOpts.length - 1][0];
  const side = key => { const [net, step] = key.split(":").map(Number); return { net, step, n: I.nets[net] }; };
  const siteSel = (which) => { const s = select([], st[which === "A" ? "sa" : "sb"], v => { st[which === "A" ? "sa" : "sb"] = +v; draw(); });
    s.refresh = () => { const sd = side(st[which]), key = which === "A" ? "sa" : "sb"; st[key] = Math.min(st[key], sd.n.sites.length - 1); s.setOpts(sd.n.sites.map((x, i) => [i, x]), st[key]); }; return s; };
  const sA = siteSel("A"), sB = siteSel("B");
  const kSel = select(I.k_grid.map(k => [k, `k = ${k}`]), st.k, v => { st.k = +v; draw(); });
  const g2 = h("div", { class: "chart" }), cum = h("div", { class: "chart" }), ang = h("div", { class: "chart" }), cls = h("div", { class: "chart" }), imgs = h("div", { class: "al-thumbs" }), info = h("p", { class: "al-info" });
  body.replaceChildren(fixtureTag(I) || "",
    ctl(lab("A"), select(gOpts, st.A, v => { st.A = v; sA.refresh(); draw(); }), sA), ctl(lab("B"), select(gOpts, st.B, v => { st.B = v; sB.refresh(); draw(); }), sB, lab("cut"), kSel), info,
    h("div", { class: "al-panels" }, h("div", {}, h("p", { class: "chart-title", text: "G² (mode i of A, mode j of B)" }), g2), h("div", {}, h("p", { class: "chart-title", text: "cumulative overlap ‖G[:i,:j]‖² / min(i, j)" }), cum)),
    h("div", { class: "al-panels" }, h("div", {}, h("p", { class: "chart-title", text: "principal angles of the leading-k spans (cos²)" }), ang), h("div", {}, h("p", { class: "chart-title", text: "per-image discrepancy δᵢ at k, mean per class" }), cls)),
    h("p", { class: "chart-title", text: "images with the largest δᵢ" }), imgs);
  sA.refresh(); sB.refresh();
  let tok = 0;
  const draw = async () => {
    const my = ++tok, a = side(st.A), b = side(st.B);
    const ca = I.groups[I.gid.get(st.A)].cells[st.sa], cb = I.groups[I.gid.get(st.B)].cells[st.sb];
    const [VA, VB, dg, stored] = await Promise.all([vectors(I, a.net, a.step, st.sa), vectors(I, b.net, b.step, st.sb), diag(0), slice(`proj_k${st.k}`, 0)]);
    if (my !== tok) return;
    const N = VA.n, K = 64, G = new Float64Array(K * K), UA = VA.U, UB = VB.U;
    for (let i = 0; i < N; i++) { const o = i * K; for (let p = 0; p < K; p++) { const x = UA[o + p]; if (!x) continue; const gp = p * K; for (let q = 0; q < K; q++) G[gp + q] += x * UB[o + q]; } }
    const G2 = G.map(x => x * x), C = new Float64Array(K * K);
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) C[i * K + j] = G2[i * K + j] + (i ? C[(i - 1) * K + j] : 0) + (j ? C[i * K + j - 1] : 0) - (i && j ? C[(i - 1) * K + j - 1] : 0);
    const k = st.k, dk = 1 - C[(k - 1) * K + k - 1] / k, sv = pairValue(I, stored, ca, cb);
    const tA = [], tB = []; for (let q = 1; q < K; q++) { if (!dg.ok(ca, q)) tA.push(q - 1); if (!dg.ok(cb, q)) tB.push(q - 1); }
    const modes = Array.from({ length: K }, (_, i) => String(i + 1));
    const eigTxt = (c, q) => f3(dg.eig(c, q));
    matrix(g2, { nr: K, nc: K, rowLabels: modes, colLabels: modes, value: (i, j) => G2[i * K + j], color: t => P.seq(t), zrange: [0, 1], maxCell: 7, maxH: 460, rowTicks: tA, colTicks: tB, labelAt: i => i === 0 || (i + 1) % 8 === 0, rowTitle: "mode of A", colTitle: "mode of B",
      tip: (i, j) => [`mode ${i + 1} of A · mode ${j + 1} of B`, [{ name: "G²", value: f3(G2[i * K + j]) }, { name: "λ (A, B)", value: `${eigTxt(ca, i + 1)}, ${eigTxt(cb, j + 1)}` }, { name: "span at these cuts (A, B)", value: `${f2(dg.span(ca, i + 1))}, ${f2(dg.span(cb, j + 1))}` }]] });
    matrix(cum, { nr: K, nc: K, rowLabels: modes, colLabels: modes, value: (i, j) => C[i * K + j] / Math.min(i + 1, j + 1), color: t => P.seq(t), zrange: [0, 1], maxCell: 7, maxH: 460, rowTicks: tA, colTicks: tB, labelAt: i => i === 0 || (i + 1) % 8 === 0, rowTitle: "i", colTitle: "j",
      overlay: (g, X, Y, cs) => el("rect", { x: X(k - 1) - cs / 2 - 1, y: Y(k - 1) - cs / 2 - 1, width: cs + 2, height: cs + 2, fill: "none", stroke: "#c2410c", "stroke-width": 1.6 }, g),
      tip: (i, j) => [`i = ${i + 1}, j = ${j + 1}`, [{ name: "‖G[:i,:j]‖² / min(i,j)", value: f3(C[i * K + j] / Math.min(i + 1, j + 1)) }]] });
    // principal angles of the leading-k spans: eigenvalues of Gkᵀ Gk
    const Mk = new Float64Array(k * k);
    for (let p = 0; p < k; p++) for (let q = 0; q < k; q++) { let s = 0; for (let r = 0; r < k; r++) s += G[r * K + p] * G[r * K + q]; Mk[p * k + q] = s; }
    const cos2 = jacobiEig(Mk, k).map(x => Math.max(0, Math.min(1, x)));
    plot(ang, { x: { kind: "linear", domain: [0.5, k + .5], ticks: niceTicks(1, k, 6).filter(v => v >= 1 && Number.isInteger(v)), label: "principal angle (largest cos² first)", name: "angle" }, y: { domain: [0, 1.02], label: "cos²", name: "cos²" },
      series: [{ name: "cos²", color: P.p.sets[0], pts: cos2.map((c, i) => ({ x: i + 1, y: c })) }], height: 220, legend: false });
    // per-image discrepancy at k
    const Gk = new Float64Array(k * k); for (let p = 0; p < k; p++) for (let q = 0; q < k; q++) Gk[p * k + q] = G[p * K + q];
    const delta = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let na = 0, nb = 0, cr = 0; const o = i * K;
      for (let p = 0; p < k; p++) { na += UA[o + p] ** 2; nb += UB[o + p] ** 2; let s = 0; for (let q = 0; q < k; q++) s += Gk[p * k + q] * UB[o + q]; cr += UA[o + p] * s; }
      delta[i] = na + nb - 2 * cr;
    }
    const L = I.probe.labels, ncl = m.classes.length, sum = new Float64Array(ncl), cnt = new Float64Array(ncl);
    let tot = 0; for (let i = 0; i < N; i++) { sum[L[i]] += delta[i]; cnt[L[i]] += 1; tot += delta[i]; }
    const mean = Array.from(sum, (s, c) => s / Math.max(1, cnt[c])), mx = Math.max(...mean) || 1;
    cls.replaceChildren(h("div", { class: "al-bars" }, ...mean.map((v, c) => h("div", { class: "al-bar-row" }, h("span", { class: "al-bar-lab", text: m.classes[c] }),
      h("span", { class: "al-bar", style: { width: `${(100 * v / mx).toFixed(1)}%`, background: P.classes[c] } }), h("span", { class: "al-bar-v", text: v.toExponential(2) })))),
      h("p", { class: "bn-note", text: `Mean δᵢ over all images ${(tot / N).toExponential(2)} (= 2k·dₖ/N). Class labels group and colour only.` }));
    const order = Array.from(delta.keys()).sort((x, y) => delta[y] - delta[x]).slice(0, 30);
    imgs.replaceChildren();
    const sprite = I.thumb ? await D.sprite() : null;
    for (const i of order) {
      let node;
      if (sprite) { node = document.createElement("canvas"); node.width = node.height = 32; const t = I.thumb[i], cols = m.sprite_cols; node.getContext("2d").drawImage(sprite, (t % cols) * 32, Math.floor(t / cols) * 32, 32, 32, 0, 0, 32, 32); }
      else node = h("span", { class: "al-noimg", text: `#${i}` });
      node.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`image ${i}`, [{ name: "class", value: m.classes[L[i]] }, { name: "δᵢ", value: delta[i].toExponential(2) }, { name: "share of Σδ", value: `${(100 * delta[i] / tot).toFixed(2)}%` }])));
      node.addEventListener("pointerleave", hideTip);
      imgs.append(node);
    }
    const isoA = !dg.ok(ca, k) || !dg.ok(cb, k);
    info.innerHTML = `A: <b>${netName(a.n)}, step ${stepTxt(a.step)}, ${a.n.sites[st.sa]}</b>; B: <b>${netName(b.n)}, step ${stepTxt(b.step)}, ${b.n.sites[st.sb]}</b>; kernel rule 0. ` +
      `d<sub>${k}</sub> from the shipped eigenfunctions ${dk.toFixed(4)}; stored all-pairs value ${sv == null ? "not shipped for this pair" : sv.toFixed(4)}. ` +
      `Cut ${k}|${k + 1}: span ${f2(dg.span(ca, k))} (A), ${f2(dg.span(cb, k))} (B)${isoA ? ", <b>unresolved</b>: the leading-k spans are not reproducible here" : ", resolved"}. ` +
      `Orange ticks on the matrix axes: unresolved cuts of A (left) and B (bottom).`;
  };
  await draw(); onResize(body, draw);
}

export async function render(kind, body, ds) {
  const fn = { "al-w1": w1, "al-w2": w2, "al-w3": w3, "al-w4": w4, "al-c1": c1, "al-c2": c2, "al-c3": c3, "al-c4": c4, "al-c5": c5 }[kind];
  if (!fn) throw new Error(`unknown live figure "${kind}"`);
  await fn(body, ds);
}
