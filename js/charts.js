// Minimal SVG charts: thin marks, hairline axes, crosshair + tooltip. No dependencies.
const NS = "http://www.w3.org/2000/svg";
export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    if (k === "style" && typeof attrs[k] === "object") { for (const [p, v] of Object.entries(attrs[k])) { if (p.startsWith("--")) e.style.setProperty(p, v); else e.style[p] = v; } }
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else if (k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}

// ---------- scales and ticks
export function scale(kind, d0, d1, r0, r1) {
  let f, inv;
  if (kind === "log") {
    const a = Math.log10(d0), b = Math.log10(d1);
    f = v => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
    inv = p => Math.pow(10, a + (p - r0) / (r1 - r0) * (b - a));
  } else if (kind === "symlog") {
    const t = v => Math.sign(v) * Math.log10(1 + Math.abs(v));
    const a = t(d0), b = t(d1);
    f = v => r0 + (t(v) - a) / (b - a) * (r1 - r0);
    inv = p => { const y = a + (p - r0) / (r1 - r0) * (b - a); return Math.sign(y) * (Math.pow(10, Math.abs(y)) - 1); };
  } else {
    f = v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
    inv = p => d0 + (p - r0) / ((r1 - r0) || 1) * (d1 - d0);
  }
  f.inv = inv; f.kind = kind; f.domain = [d0, d1];
  return f;
}
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + .5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toFixed(10));
  return out;
}
export function logTicks(lo, hi, symlog) {
  const out = symlog ? [0] : [];
  for (let e = Math.floor(Math.log10(Math.max(lo, 1))); Math.pow(10, e) <= hi * 1.0001; e++) if (Math.pow(10, e) >= (symlog ? 1 : lo)) out.push(Math.pow(10, e));
  return out;
}
export const fmtStep = v => v >= 1000 ? (v % 1000 === 0 ? v / 1000 + "k" : (v / 1000).toFixed(1) + "k") : String(v);
export const fmtPct = v => (100 * v).toFixed(v >= 0.995 || v < 0.1 ? 1 : 0) + "%";

// ---------- tooltip (one global element)
const tip = () => document.getElementById("tooltip");
export function showTip(x, y, content) {
  const t = tip();
  if (typeof content === "string") t.innerHTML = content; else { t.replaceChildren(content); }
  t.hidden = false;
  const r = t.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight;
  let lx = x + 14, ly = y + 14;
  if (lx + r.width > W - 8) lx = x - r.width - 14;
  if (ly + r.height > H - 8) ly = y - r.height - 14;
  t.style.left = Math.max(8, lx) + "px"; t.style.top = Math.max(8, ly) + "px";
}
export function hideTip() { tip().hidden = true; }
export function tipRows(head, rows) {
  const d = h("div");
  if (head) d.append(h("div", { class: "tt-head", text: head }));
  for (const r of rows) {
    const row = h("div", { class: "tt-row" });
    const k = h("span", { class: "tt-k" });
    if (r.color) k.append(h("span", { class: "tt-line", style: { "--sw": r.color } }));
    k.append(document.createTextNode(r.name));
    row.append(k, h("span", { class: "tt-v", text: r.value }));
    d.append(row);
  }
  return d;
}

// ---------- line chart
// opts: x (array), series [{name, y, color, dash (true = dashed, "dot" = dotted), width, dots (marker radius),
//       noline (markers only), band:[lo[],hi[]]}], xscale, yrange, xlabel, ylabel,
//       height, xfmt, yfmt, xticks, refline {y,label}, shade [x0,x1], legend (bool), title,
//       band [lo[],hi[]] (one band for the whole chart, grey), bandMode, bandLabel.
// bandMode for per-series bands: "each" (each in its series colour), "shared" (draw the first once, grey),
// "envelope" (one grey band from the lowest lo to the highest hi; a series' own band shows on hovering its key),
// "none", or "auto" (default): "shared" when all bands are equal, "each" for up to 3 banded series, else "envelope".
export function lineChart(container, o) {
  container.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" });
  container.append(wrap);
  const W = Math.max(240, container.clientWidth || 480), H = o.height || 240;
  // a y title that does not fit along the axis is shown horizontally above the plot instead
  const yRot = o.ytitle && o.ytitle.length * 6.4 <= H - 50;
  if (o.ytitle && !yRot) wrap.before(h("p", { class: "chart-sub", style: { textAlign: "center", margin: "0 0 4px" }, text: o.ytitle }));
  const m = { l: o.ml ?? (yRot ? 62 : 44), r: o.mr ?? 14, t: o.mt ?? 10, b: o.xlabel ? 38 : 24 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.ylabel || "chart" }, wrap);
  const xs = o.x;
  const xk = o.xscale || "linear";
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (xk === "log") x0 = Math.max(x0, Math.min(...xs.filter(v => v > 0)));
  const X = scale(xk, x0, x1, m.l, W - m.r);
  let ylo, yhi;
  if (o.yrange) [ylo, yhi] = o.yrange;
  else {
    const all = o.series.flatMap(s => s.y).filter(Number.isFinite);
    ylo = Math.min(...all); yhi = Math.max(...all);
    const pad = (yhi - ylo) * .06 || .05; ylo -= pad; yhi += pad;
  }
  const Y = scale("linear", ylo, yhi, H - m.b, m.t);
  const yfmt = o.yfmt || (v => (+v.toFixed(3)).toString());
  const xfmt = o.xfmt || (v => String(v));
  // grid + axes
  const g = el("g", { class: "grid" }, svg);
  const yt = niceTicks(ylo, yhi, Math.max(3, Math.round(H / 55)));
  for (const v of yt) el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
  const ax = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  for (const v of yt) { const t = el("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yfmt(v); }
  let xt = o.xticks || (xk === "log" || xk === "symlog" ? logTicks(Math.max(1, x0), x1, xk === "symlog") : niceTicks(x0, x1, Math.max(3, Math.round(W / 90))));
  xt = xt.filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  for (const v of xt) {
    el("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax);
    const t = el("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, svg); t.textContent = xfmt(v);
  }
  if (o.xlabel) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg); t.textContent = o.xlabel; }
  if (yRot) { const t = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)` }, svg); t.textContent = o.ytitle; }
  if (o.shade) el("rect", { x: X(o.shade[0]), y: m.t, width: Math.max(2, X(o.shade[1]) - X(o.shade[0])), height: H - m.b - m.t, fill: "rgba(22,22,26,.05)" }, svg);
  if (o.refline) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.refline.y), y2: Y(o.refline.y), stroke: "#9a9993", "stroke-width": 1 }, svg);
    if (o.refline.label) { const t = el("text", { x: m.l + 6, y: Y(o.refline.y) - 5, "text-anchor": "start" }, svg); t.textContent = o.refline.label; }
  }
  // bands
  const bandD = b => {
    let d = "";
    xs.forEach((xv, i) => { if (Number.isFinite(b[1][i])) d += (d ? "L" : "M") + X(xv) + "," + Y(Math.min(yhi, b[1][i])); });
    for (let i = xs.length - 1; i >= 0; i--) if (Number.isFinite(b[0][i])) d += "L" + X(xs[i]) + "," + Y(Math.max(ylo, b[0][i]));
    return d ? d + "Z" : "";
  };
  const GREY = { fill: "#16161a", opacity: .075 };
  const banded = o.series.filter(s => s.band), key0 = banded.length ? JSON.stringify(banded[0].band) : "";
  let bandMode = o.bandMode || "auto", bandNote = null;
  if (bandMode === "auto") bandMode = banded.length > 1 && banded.every(s => JSON.stringify(s.band) === key0) ? "shared" : banded.length > 3 ? "envelope" : "each";
  if (o.band) { el("path", { d: bandD(o.band), ...GREY }, svg); bandNote = o.bandLabel || "shaded: reference band"; }
  if (banded.length && bandMode === "shared") { el("path", { d: bandD(banded[0].band), ...GREY }, svg); bandNote = o.bandLabel || "shaded: band (the same for every series)"; }
  if (banded.length && bandMode === "envelope") {
    const lo = xs.map((_, i) => Math.min(...banded.map(s => s.band[0][i]).filter(Number.isFinite))), hi = xs.map((_, i) => Math.max(...banded.map(s => s.band[1][i]).filter(Number.isFinite)));
    el("path", { d: bandD([lo, hi]), ...GREY }, svg);
    bandNote = o.bandLabel || "shaded: from the lowest to the highest of the series' bands (hover a key for one series' band)";
  }
  // series
  const paths = [], bandEls = [];
  o.series.forEach((s, si) => {
    const gg = el("g", {}, svg);
    if (s.band && bandMode === "each") el("path", { d: bandD(s.band), fill: s.color, opacity: .1 }, gg);
    bandEls[si] = s.band && bandMode === "envelope" ? el("path", { d: bandD(s.band), fill: s.color, opacity: 0, "pointer-events": "none" }, svg) : null;
    let d = "", pen = false;
    xs.forEach((xv, i) => {
      const v = s.y[i];
      if (!Number.isFinite(v) || (xk === "log" && xv <= 0)) { pen = false; return; }
      d += (pen ? "L" : "M") + X(xv).toFixed(1) + "," + Y(v).toFixed(1); pen = true;
    });
    const p = el("path", { d, fill: "none", stroke: s.noline ? "none" : s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, gg);
    if (s.dash) p.setAttribute("stroke-dasharray", s.dash === "dot" ? "1.5 3.5" : "5 4");
    if (s.dots) xs.forEach((xv, i) => { if (Number.isFinite(s.y[i])) el("circle", { cx: X(xv), cy: Y(s.y[i]), r: s.dots, fill: s.color, stroke: "#fff", "stroke-width": 1.5 }, gg); });
    paths.push(gg);
  });
  // crosshair
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "#16161a", "stroke-width": 1, opacity: 0 }, svg);
  const dots = o.series.map(s => el("circle", { r: 4, fill: s.color, stroke: "#fff", "stroke-width": 2, opacity: 0 }, svg));
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  const nearest = px => {
    let bi = 0, bd = Infinity;
    xs.forEach((xv, i) => { if (xk === "log" && xv <= 0) return; const d = Math.abs(X(xv) - px); if (d < bd) { bd = d; bi = i; } });
    return bi;
  };
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), i = nearest(ev.clientX - r.left);
    cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("opacity", .35);
    o.series.forEach((s, si) => {
      const v = s.y[i], d = dots[si];
      if (Number.isFinite(v)) { d.setAttribute("cx", X(xs[i])); d.setAttribute("cy", Y(v)); d.setAttribute("opacity", 1); } else d.setAttribute("opacity", 0);
    });
    const rows = o.series.map(s => ({ name: s.name, value: Number.isFinite(s.y[i]) ? yfmt(s.y[i]) : "n/a", color: s.color, v: s.y[i] }));
    if (rows.length > 6) rows.sort((a, b) => (b.v ?? -1e9) - (a.v ?? -1e9));
    showTip(ev.clientX, ev.clientY, tipRows((o.xname ? o.xname + " " : "") + xfmt(xs[i]), rows));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); hideTip(); });
  if (o.legend !== false && o.series.length > 1) {
    const lg = h("div", { class: "chart-legend" });
    o.series.forEach((s, si) => {
      const key = h("i", { style: { "--sw": s.color } });
      if (s.dash) key.style.background = s.dash === "dot" ? `repeating-linear-gradient(90deg, ${s.color} 0 2px, transparent 2px 5px)` : `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`;
      if (s.noline && s.dots) Object.assign(key.style, { width: "8px", height: "8px", borderRadius: "50%", background: s.color });
      const k = h("span", { class: "k" }, key, s.name);
      k.addEventListener("pointerenter", () => { paths.forEach((p, j) => p.setAttribute("opacity", j === si ? 1 : .15)); bandEls[si]?.setAttribute("opacity", .22); });
      k.addEventListener("pointerleave", () => { paths.forEach(p => p.setAttribute("opacity", 1)); bandEls[si]?.setAttribute("opacity", 0); });
      lg.append(k);
    });
    if (bandNote) lg.append(h("span", { class: "k mute" }, h("i", { style: { background: "rgba(22,22,26,.12)", height: "8px", borderRadius: "2px" } }), bandNote));
    wrap.append(lg);
  } else if (bandNote) wrap.append(h("div", { class: "chart-legend" }, h("span", { class: "k mute", text: bandNote })));
  return { svg, X, Y };
}

// ---------- heatmap (rows x cols), optional slider over a leading axis
// o: rows, cols, z ([r][c] or [k][r][c]), zrange, color(t)->css, zfmt, rowlabel, collabel, slider {label, values}
export function heatmap(container, o) {
  container.replaceChildren();
  const ctrl = h("div", { class: "chart-controls" });
  const wrap = h("div", { class: "chart-wrap" });
  container.append(ctrl);
  if (o.zlabel) container.append(h("p", { class: "chart-sub", style: { textAlign: "center" }, text: "colour: " + o.zlabel }));
  container.append(wrap);
  const has3 = !!o.slider;
  let k = has3 ? o.slider.values.length - 1 : 0;
  if (has3) {
    const lab = h("label", {}, (o.slider.label || "") + " ");
    const val = h("b", { text: String(o.slider.values[k]) });
    const inp = h("input", { type: "range", min: 0, max: o.slider.values.length - 1, step: 1, value: k });
    inp.addEventListener("input", () => { k = +inp.value; val.textContent = String(o.slider.values[k]); draw(); });
    ctrl.append(lab, inp, val);
  } else ctrl.remove();
  const nr = o.rows.length, nc = o.cols.length;
  const draw = () => {
    wrap.replaceChildren();
    const Wc = Math.max(260, container.clientWidth || 480);
    const ml = o.ml ?? (o.rowlabel ? 70 : 56), mt = 8;
    const cw = Math.max(6, Math.min(44, (Wc - ml - 70) / nc)), ch = o.cellH || Math.max(10, Math.min(24, cw));
    // column labels: horizontal when every label fits its column, otherwise rotated 45 degrees (full name on hover)
    const colTxt = o.cols.map(c => typeof c === "number" ? fmtStep(c) : String(c)), CH = 6.1;
    const rot = colTxt.some(t => t.length * CH > cw - 3), lmax = Math.max(...colTxt.map(t => t.length)) * CH;
    const mb = rot ? 30 + Math.min(110, lmax * .72) : 46;
    const W = Math.min(Wc, ml + nc * cw + 96), H = mt + nr * ch + mb;
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: "margin: 0 auto" }, wrap);
    const Z = has3 ? o.z[k] : o.z;
    const [zl, zh] = o.zrange || [0, 1];
    for (let r = 0; r < nr; r++) {
      const t = el("text", { x: ml - 6, y: mt + r * ch + ch / 2 + 3.5, "text-anchor": "end" }, svg); t.textContent = o.rows[r];
      for (let c = 0; c < nc; c++) {
        const v = Z[r][c];
        const rect = el("rect", { x: ml + c * cw + .5, y: mt + r * ch + .5, width: cw - 1, height: ch - 1, rx: 1.5,
          fill: Number.isFinite(v) ? o.color((v - zl) / (zh - zl)) : "#f2f1ec" }, svg);
        rect.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(null, [
          { name: o.rowlabel || "row", value: String(o.rows[r]) }, { name: o.collabel || "col", value: String(o.cols[c]) },
          { name: o.zlabel || "value", value: Number.isFinite(v) ? (o.zfmt || (x => x.toFixed(3)))(v) : "n/a" }])));
        rect.addEventListener("pointerleave", hideTip);
        if (o.onclick) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => o.onclick(r, c, k)); }
      }
    }
    const every = Math.ceil(nc / Math.max(1, Math.floor(nc * cw / (rot ? 13 : 34))));
    for (let c = 0; c < nc; c += every) {
      const x = ml + c * cw + cw / 2, y = mt + nr * ch + (rot ? 8 : 14);
      const t = el("text", rot ? { x, y, "text-anchor": "end", "dominant-baseline": "middle", transform: `rotate(-45 ${x} ${y})` } : { x, y, "text-anchor": "middle" }, svg);
      t.textContent = colTxt[c];
      el("title", {}, t).textContent = `${o.collabel ? o.collabel + ": " : ""}${o.cols[c]}`;
    }
    if (o.collabel) { const t = el("text", { x: ml + nc * cw / 2, y: H - 6, "text-anchor": "middle", class: "ttl" }, svg); t.textContent = o.collabel; }
    if (o.rowlabel) { const t = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(11,${mt + nr * ch / 2}) rotate(-90)` }, svg); t.textContent = o.rowlabel; }
    // colour bar
    const bx = ml + nc * cw + 16, bh = Math.min(140, nr * ch);
    const id = "g" + Math.random().toString(36).slice(2);
    const lg = el("linearGradient", { id, x1: 0, x2: 0, y1: 1, y2: 0 }, el("defs", {}, svg));
    for (let i = 0; i <= 8; i++) el("stop", { offset: i / 8, "stop-color": o.color(i / 8) }, lg);
    el("rect", { x: bx, y: mt, width: 10, height: bh, fill: `url(#${id})`, rx: 2 }, svg);
    [[zh, mt + 4], [zl, mt + bh]].forEach(([v, y]) => { const t = el("text", { x: bx + 14, y, "text-anchor": "start" }, svg); t.textContent = (o.zfmt || (x => (+x.toFixed(2)).toString()))(v); });
  };
  draw();
  return { redraw: draw };
}

// ---------- xy chart: every series has its own x positions; log or linear y; point markers; vertical rules
// o: series [{name, pts: [[x, y], ...], color, dash, width, dots, noline, marker: "diamond"|"triangle" (markers only, no line), hollow,
//             breaks (a non-finite y breaks the line instead of joining over it)}],
//    ylog, yrange, xrange, xticks, xfmt, yfmt, height, ml, xname, ytitle, vlines [{x, label, dash, color}],
//    hl (x to emphasise), onclick(x), tipExtra(x) -> rows, legend (bool)
export function xyChart(container, o) {
  container.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" });
  container.append(wrap);
  const W = Math.max(220, container.clientWidth || 320), H = o.height || 170;
  const m = { l: o.ml ?? 42, r: o.mr ?? 10, t: o.mt ?? 8, b: o.xname ? 32 : 20 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "chart" }, wrap);
  const pts = o.series.flatMap(s => s.pts).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]) && (!o.ylog || p[1] > 0));
  const [x0, x1] = o.xrange || [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0]))];
  let [ylo, yhi] = o.yrange || [Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))];
  if (o.ylog) { ylo = Math.pow(10, Math.floor(Math.log10(ylo) * 4) / 4); yhi = Math.pow(10, Math.ceil(Math.log10(yhi) * 4) / 4); if (yhi <= ylo) yhi = ylo * 10; }
  else if (!o.yrange) { const pad = (yhi - ylo) * .06 || .05; ylo -= pad; yhi += pad; }
  const X = scale("linear", x0, x1, m.l, W - m.r), Y = scale(o.ylog ? "log" : "linear", ylo, yhi, H - m.b, m.t);
  const yfmt = o.yfmt || (v => o.ylog ? (v >= 1e3 || v < 1e-2 ? v.toExponential(0) : (+v.toPrecision(2)).toString()) : (+v.toFixed(3)).toString());
  const xfmt = o.xfmt || (v => String(v));
  const g = el("g", { class: "grid" }, svg), ax = el("g", { class: "axis" }, svg);
  let yt;
  if (o.ylog) {
    yt = [];
    for (let e = Math.floor(Math.log10(ylo)); e <= Math.ceil(Math.log10(yhi)); e++) for (const c of [1, 2, 5]) { const v = c * Math.pow(10, e); if (v >= ylo * .999 && v <= yhi * 1.001) yt.push(v); }
    const dec = yt.filter(v => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
    if (dec.length >= 2) yt = dec;
    if (yt.length > 6) yt = yt.filter((_, i) => i % Math.ceil(yt.length / 5) === 0);
    if (yt.length < 3) yt = niceTicks(ylo, yhi, 3).filter(v => v > 0);    // less than a decade: plain ticks
  } else yt = niceTicks(ylo, yhi, Math.max(3, Math.round(H / 50)));
  for (const v of yt) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); const t = el("text", { x: m.l - 5, y: Y(v) + 3.5, "text-anchor": "end" }, svg); t.textContent = yfmt(v); }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  const xt = (o.xticks || niceTicks(x0, x1, Math.max(3, Math.round(W / 60)))).filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  for (const v of xt) { el("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); const t = el("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }, svg); t.textContent = xfmt(v); }
  if (o.xname) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle", class: "ttl" }, svg); t.textContent = o.xname; }
  if (o.ytitle) { const t = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(10,${(m.t + H - m.b) / 2}) rotate(-90)` }, svg); t.textContent = o.ytitle; }
  if (o.hl != null && o.hl >= x0 && o.hl <= x1) el("rect", { x: X(o.hl) - 4, y: m.t, width: 8, height: H - m.b - m.t, fill: "rgba(22,22,26,.06)" }, svg);
  for (const v of o.vlines || []) {
    if (v.x < x0 || v.x > x1) continue;
    el("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: H - m.b, stroke: v.color || "#16161a", "stroke-width": 1.2, ...(v.dash ? { "stroke-dasharray": "4 3" } : {}) }, svg);
    if (v.label) { const t = el("text", { x: X(v.x) + (X(v.x) > W - 90 ? -4 : 4), y: v.top ? m.t + 9 : H - m.b - 5, "text-anchor": X(v.x) > W - 90 ? "end" : "start", style: `font-size:10px;fill:${v.color || "#16161a"}` }, svg); t.textContent = v.label; }
  }
  const ok = p => Number.isFinite(p[0]) && Number.isFinite(p[1]) && (!o.ylog || p[1] > 0);
  const groups = o.series.map(s => {
    const gg = el("g", {}, svg), P = s.pts.filter(ok);
    if (!s.marker && !s.noline && P.length > 1) {
      // s.breaks: a non-finite point in s.pts (e.g. [x, null]) ends the line there instead of being joined over
      const Q = s.breaks ? s.pts.filter(p => Number.isFinite(p[0])) : P;
      let pen = false;
      const d = Q.map(p => { if (!ok(p)) { pen = false; return ""; } const c = (pen ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(Math.max(ylo, Math.min(yhi, p[1]))).toFixed(1); pen = true; return c; }).join("");
      const path = el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.8, "stroke-linejoin": "round", "stroke-linecap": "round" }, gg);
      if (s.dash) path.setAttribute("stroke-dasharray", "5 4");
    }
    for (const p of P) {
      const cx = X(p[0]), cy = Y(Math.max(ylo, Math.min(yhi, p[1])));
      if (s.marker === "diamond") el("path", { d: `M${cx},${cy - 6.5}L${cx + 6},${cy}L${cx},${cy + 6.5}L${cx - 6},${cy}Z`, ...(s.hollow ? { fill: "none", stroke: s.color, "stroke-width": 1.6 } : { fill: s.color, stroke: "#fff", "stroke-width": 1.2 }) }, gg);
      else if (s.marker === "triangle") el("path", { d: `M${cx},${cy - 5.5}L${cx + 5},${cy + 4}L${cx - 5},${cy + 4}Z`, fill: s.color, stroke: "#fff", "stroke-width": 1.2 }, gg);
      else if (s.dots) el("circle", { cx, cy, r: s.dots, fill: s.color, stroke: "#fff", "stroke-width": 1 }, gg);
    }
    return gg;
  });
  const xsAll = [...new Set(pts.map(p => p[0]))].sort((a, b) => a - b);
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "#16161a", "stroke-width": 1, opacity: 0 }, svg);
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent", style: o.onclick ? "cursor:pointer" : "" }, svg);
  const nearest = px => xsAll.reduce((b, x) => Math.abs(X(x) - px) < Math.abs(X(b) - px) ? x : b, xsAll[0]);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), x = nearest(ev.clientX - r.left);
    cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("opacity", .3);
    const rows = [];
    o.series.forEach(s => { const p = s.pts.find(q => q[0] === x && ok(q)); if (p) rows.push({ name: s.name, value: yfmt2(p[1]), color: s.color }); });
    showTip(ev.clientX, ev.clientY, tipRows((o.xname ? o.xname + " = " : "") + xfmt(x), rows.concat(o.tipExtra ? o.tipExtra(x) : [])));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  if (o.onclick) hit.addEventListener("click", ev => { const r = svg.getBoundingClientRect(); hideTip(); o.onclick(nearest(ev.clientX - r.left)); });
  if (o.legend !== false && o.series.length > 1) {
    const lg = h("div", { class: "chart-legend" });
    o.series.forEach((s, si) => {
      const key = h("i", { style: { "--sw": s.color } });
      if (s.dash) key.style.background = `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`;
      if (s.marker === "diamond") Object.assign(key.style, { width: "8px", height: "8px", transform: "rotate(45deg)", borderRadius: "1px", background: s.hollow ? "transparent" : s.color, boxShadow: s.hollow ? `inset 0 0 0 1.5px ${s.color}` : "" });
      if (s.marker === "triangle") Object.assign(key.style, { width: "10px", height: "9px", borderRadius: "0", background: s.color, clipPath: "polygon(50% 0, 100% 100%, 0 100%)" });
      const k = h("span", { class: "k" }, key, s.name);
      k.addEventListener("pointerenter", () => groups.forEach((p, j) => p.setAttribute("opacity", j === si ? 1 : .15)));
      k.addEventListener("pointerleave", () => groups.forEach(p => p.setAttribute("opacity", 1)));
      lg.append(k);
    });
    wrap.append(lg);
  }
  return { svg, X, Y };
}
const yfmt2 = v => Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1e-3 ? (+v.toPrecision(4)).toString() : v.toExponential(3);
