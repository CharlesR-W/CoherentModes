// F9, the Zoom tab: an icicle of one precomputed tree (T1..T4) and, for the clicked node, (a) the items on the
// winning eigenfunction with the cut, (b) the sweep profile with its bounds, (c) the cut's retention in each cell,
// (d) SEBA r = 2 beside the cut and the node's numbers, (e) the six items at each end. Nothing is recomputed here.
import { el, h, showTip, hideTip, tipRows, lineChart } from "./charts.js";
import { D, fmt, fmtS, g3, snippetEl, srcSwatch } from "./data.js";
import { icicle, legendBar } from "./figs.js";

export class Zoom {
  constructor(root) {
    this.root = root;
    this.trees = D.trees.filter(t => t.id !== "P1");
    this.st = { tree: this.trees[0]?.id, node: 0, col: "side" };
    this.build();
  }
  tree() { return this.trees.find(t => t.id === this.st.tree); }
  applyParams(q) {   // #zoom?tree=T1&node=0
    if (q.get("tree") && this.trees.some(t => t.id === q.get("tree"))) this.st.tree = q.get("tree");
    if (q.get("node") != null) this.st.node = +q.get("node");
    if (q.get("c")) this.st.col = q.get("c");
    this.render();
  }
  describe(t) {
    const b = t.block || {}, step = t.step != null ? `step ${t.step.toLocaleString("en")}` : "";
    const role = { T1: "the headline", T2: "a layer where the token just read dominates", T3: "one cell instead of a block", T4: "the same block earlier in training" }[t.id] || "";
    const where = b.kind === "single" ? `single cell (L${b.layer}, t = ${b.positions?.[0]})` : `reading block L${b.layer}, t = ${b.positions?.[0]} → ${b.positions?.[b.positions.length - 1]}`;
    return `${where}, ${step}; ${role}.`;
  }
  build() {
    const r = this.root;
    this.seg = h("div", { class: "seg", role: "radiogroup", "aria-label": "Tree" });
    for (const t of this.trees) {
      const b = h("button", { type: "button", "aria-checked": String(t.id === this.st.tree), text: t.id });
      b.addEventListener("click", () => { this.st.tree = t.id; this.st.node = 0; this.render(); });
      this.seg.append(b);
    }
    this.desc = h("span", { class: "ctl-sub" });
    this.ic = h("div", { class: "icicle-wrap" });
    this.detail = h("div", { class: "zoom-detail" });
    const key = h("div", { class: "f5-key" }, ...D.sources.map((s, k) => h("span", {}, h("i", { style: { background: D.srcCol[k] } }), s)),
      h("span", { class: "mute", text: "bars: source make-up (validation only)" }));
    r.replaceChildren(h("div", { class: "chart-controls zoom-ctl" }, h("span", { class: "ctl-label", text: "Tree" }), this.seg, this.desc), h("p", { class: "card-note", style: { margin: "0 0 4px" }, text: "Click any node, faded or not, to see its cut below." }), this.ic, key, this.detail);
    this.render();
    window.addEventListener("resize", () => { clearTimeout(this._rt); this._rt = setTimeout(() => this.root.offsetParent && this.render(), 150); });
  }
  render() {
    const t = this.tree();
    if (!t) return;
    this.seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.textContent === t.id)));
    this.desc.textContent = this.describe(t);
    icicle(this.ic, t, { bar: "makeup", sel: this.st.node, onclick: nd => { this.st.node = nd.id; this.render(); } });
    this.renderNode(t, t.nodes.find(n => n.id === this.st.node) || t.nodes[0]);
  }
  renderNode(t, nd) {
    const byId = Object.fromEntries(t.nodes.map(n => [n.id, n])), thr = t.threshold;
    const nav = h("div", { class: "zoom-nav" });
    const go = (id, label) => { const b = h("button", { type: "button", class: "guide-btn", text: label }); b.addEventListener("click", () => { this.st.node = id; this.render(); }); return b; };
    if (nd.parent >= 0) nav.append(go(nd.parent, "↑ parent"));
    (nd.children || []).forEach((c, k) => nav.append(go(c, `child ${k} (n = ${byId[c].n})`)));
    const head = h("div", { class: "zoom-head" }, h("h3", { text: `Node ${nd.id}: depth ${nd.depth}, ${nd.n} items` }), nav);
    if (!nd.children?.length) {
      this.detail.replaceChildren(head, h("p", { class: "card-note", text: `A leaf: not split, because the recursion stops at depth 3 or below 64 items (size and depth only; Υ decides how a split is drawn, never whether it exists).` }), makeupLine(nd));
      return;
    }
    const weak = thr != null && nd.upsilon < thr;
    const verdict = thr == null ? "no reading rule (threshold dropped)" : weak ? `weak split (Υ < ${g3(thr)})` : `clear split (Υ ≥ ${g3(thr)})`;
    const comp = Number.isFinite(nd.n_components) ? ` · kNN graph of the node: ${nd.n_components} connected piece${nd.n_components === 1 ? "" : "s"}` : "";
    head.append(h("p", { class: "card-sub", text: `clustering ratio Υ = μ₂/φ* = ${g3(nd.upsilon)}: ${verdict}; cut along ψ${nd.axis_used}${comp}` }));
    const A_ = h("div", { class: "card" }), B_ = h("div", { class: "card" }), C_ = h("div", { class: "card" }), D_ = h("div", { class: "card" }), E_ = h("div", { class: "card zoom-ends" });
    this.detail.replaceChildren(head, h("div", { class: "zoom-cards" }, A_, B_, C_, D_), E_);
    this.scatter(A_, t, nd, byId);
    this.profile(B_, nd);
    this.retention(C_, t, nd);
    this.numbers(D_, nd);
    this.ends(E_, t, nd, byId);
  }
  scatter(card, t, nd, byId) {
    if (this.st.col === "seba" && !nd.seba2) this.st.col = "side";
    const n = nd.n, w = nd.axis_used - 1, o = w === 0 ? 1 : 0, sq = Math.sqrt(n);
    const xs = nd.psi[w].map(q => q / 127 * nd.psi_scale[w] * sq), ys = nd.psi[o].map(q => q / 127 * nd.psi_scale[o] * sq);
    const c0 = new Set(byId[nd.children[0]].members);
    const seg = h("div", { class: "seg" });
    const opts = [["side", "side of the cut"], ...(nd.seba2 ? [["seba", "SEBA r = 2"]] : []), ["source", "source (validation)"]];
    opts.forEach(([v, l]) => { const b = h("button", { type: "button", "aria-checked": String(this.st.col === v), text: l });
      b.addEventListener("click", () => { this.st.col = v; this.scatter(card, t, nd, byId); }); seg.append(b); });
    const Wd = Math.min(380, (card.clientWidth || 360) - 28), Ht = Math.round(Wd * .72), m = { l: 34, r: 8, t: 8, b: 30 };
    const lo = a => Math.min(...a), hi = a => Math.max(...a);
    const x0 = Math.min(lo(xs), nd.cut_theta * sq), x1 = Math.max(hi(xs), nd.cut_theta * sq), y0 = lo(ys), y1 = hi(ys);
    const X = v => m.l + (v - x0) / ((x1 - x0) || 1) * (Wd - m.l - m.r), Y = v => Ht - m.b - (v - y0) / ((y1 - y0) || 1) * (Ht - m.t - m.b);
    const svg = el("svg", { width: Wd, height: Ht, viewBox: `0 0 ${Wd} ${Ht}` });
    el("line", { x1: m.l, x2: Wd - m.r, y1: Ht - m.b + .5, y2: Ht - m.b + .5, stroke: "#c9c8c1" }, svg);
    const g = el("g", {}, svg);
    nd.members.forEach((item, i) => {
      const sb = nd.seba2 ? nd.seba2[i] : null;
      const col = this.st.col === "side" ? D.pal.set(c0.has(item) ? 0 : 1) : this.st.col === "seba" && sb != null ? (sb < 0 ? D.palette.unassigned : D.pal.set(sb + 2)) : D.srcCol[D.src[item]];
      const c = el("circle", { cx: X(xs[i]).toFixed(1), cy: Y(ys[i]).toFixed(1), r: n > 600 ? 1.6 : 2.2, fill: col, "fill-opacity": .75 }, g);
      c.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, h("div", {}, h("div", { class: "tt-head", text: `item ${item} · ${D.sources[D.src[item]]} · child ${c0.has(item) ? 0 : 1}` }), snippetEl(item, 128, { before: 20, after: 0 }))));
      c.addEventListener("pointerleave", hideTip);
    });
    el("line", { x1: X(nd.cut_theta * sq), x2: X(nd.cut_theta * sq), y1: m.t, y2: Ht - m.b, stroke: "#16161a", "stroke-width": 1.4, "stroke-dasharray": "4 3" }, svg);
    const lab = (x, y, s, a = "middle", rot) => { const e = el("text", { x, y, "text-anchor": a, class: "ttl", ...(rot ? { transform: `rotate(-90 ${x} ${y})` } : {}) }, svg); e.textContent = s; };
    lab((m.l + Wd - m.r) / 2, Ht - 8, `√n · ψ${nd.axis_used} (the cut's eigenfunction)`);
    lab(12, (m.t + Ht - m.b) / 2, `√n · ψ${o + 1}`, "middle", true);
    const keyEl = this.st.col === "side" ? h("div", { class: "f5-key" }, h("span", {}, h("i", { style: { background: D.pal.set(0) } }), `child 0 (${byId[nd.children[0]].n})`), h("span", {}, h("i", { style: { background: D.pal.set(1) } }), `child 1 (${byId[nd.children[1]].n})`), h("span", { class: "mute", text: "dashed: the cut" }))
      : this.st.col === "seba" ? h("div", { class: "f5-key" }, h("span", {}, h("i", { style: { background: D.pal.set(2) } }), "SEBA set 1"), h("span", {}, h("i", { style: { background: D.pal.set(3) } }), "SEBA set 2"), h("span", {}, h("i", { style: { background: D.palette.unassigned } }), "unassigned"), h("span", { class: "mute", text: "dashed: the cut" }))
      : h("div", { class: "f5-key" }, ...D.sources.map((s, k) => h("span", {}, h("i", { style: { background: D.srcCol[k] } }), s)));
    card.replaceChildren(h("p", { class: "card-title", text: "(a) Items on the cut's eigenfunction" }),
      h("p", { class: "card-note", style: { marginTop: "0" }, text: `Each dot is one snippet of this node: x, its value on ψ${nd.axis_used} (the function the cut was taken along); y, on ψ${o + 1}. The dashed line is the cut.` }),
      h("div", { class: "chart-controls", style: { justifyContent: "flex-start", margin: "2px 0 6px" } }, seg), h("div", { class: "chart" }, svg), keyEl);
  }
  profile(card, nd) {
    const holder = h("div", { class: "chart" });
    card.replaceChildren(h("p", { class: "card-title", text: "(b) Sweep along the winning ψ" }), holder,
      h("p", { class: "card-note", text: `Slide a threshold along ψ${nd.axis_used}: x is m, the number of snippets above it; y is the conductance φ of that split (log scale; low = few walk steps cross). Shaded: sides below m_min = ${nd.m_min}. φ* = ${g3(nd.phi)} at m = ${nd.cut_m}; φ_free = ${g3(nd.phi_free)} (unconstrained ψ₁ sweep); Cheeger: μ₁/2 = ${g3(nd.cheeger_lo)} ≤ h ≤ φ_free ≤ √(2μ₁) = ${g3(nd.cheeger_hi)}.` }));
    requestAnimationFrame(() => lineChart(holder, {
      x: nd.profile_m, series: [{ name: "φ(m)", y: nd.profile, color: "#16161a", width: 1.6 }], yscale: "log",
      yrange: [Math.min(...nd.profile.filter(v => v > 0), nd.phi, nd.cheeger_lo) / 2, Math.max(...nd.profile, nd.cheeger_hi) * 1.5], xlabel: "m (items on the high side)", xname: "m =", ytitle: "conductance φ", height: 220, legend: false,
      shades: [[nd.profile_m[0], nd.m_min], [nd.n - nd.m_min, nd.profile_m[nd.profile_m.length - 1]]],
      vlines: [{ x: nd.cut_m, color: D.palette.accent, label: "cut", dash: "4 3" }],
      hlines: [{ y: nd.phi, label: "φ*", color: D.palette.accent, right: true }, { y: nd.cheeger_lo, label: "μ₁/2", color: "#6b6b70", dash: "2 3" }, { y: nd.cheeger_hi, label: "√(2μ₁)", color: "#6b6b70", dash: "2 3" }],
    }));
  }
  retention(card, t, nd) {
    const pos = t.block?.positions || [], vals = nd.retention_cut || [];
    const row = h("div", { class: "ret-strip" });
    vals.forEach((v, i) => row.append(h("div", { class: "ret-cell", style: { background: D.pal.seq((v - .5) / .5), color: (v - .5) / .5 > .55 ? "#fff" : "#16161a" }, title: `t = ${pos[i]}: ${fmt(v, 4)}` },
      h("b", { text: fmt(v, 3) }), h("span", { text: t.block?.kind === "planted fixture" ? `cell ${i}` : `t = ${pos[i]}` }))));
    card.replaceChildren(h("p", { class: "card-title", text: "(c) Retention of the cut in each cell" }), row,
      h("p", { class: "card-note", text: "ρ_c(f) = fᵀM_c f / fᵀf for the centred cut indicator f, in each cell of the block (rebuilt on this node's items). 1: the walk almost never crosses the cut; 0.5: as for a random split." }));
  }
  numbers(card, nd) {
    const rows = [["λ₁ … λ₄ of M̄", nd.vals.map(v => fmt(v, 4)).join(", ")], ["μ₁, μ₂", `${g3(nd.mu1)}, ${g3(nd.mu2)}`], ["φ*, φ_free", `${g3(nd.phi)}, ${g3(nd.phi_free)}`], ["clustering ratio Υ = μ₂/φ*", g3(nd.upsilon)], ["winning ψ", "ψ" + nd.axis_used]];
    const tb = h("table", { class: "mini" }, h("tbody", {}, ...rows.map(([a, b]) => h("tr", {}, h("td", { text: a }), h("td", { text: b })))));
    let ct = null;
    if (nd.seba2) {
      const c0 = new Set(this.tree().nodes.find(x => x.id === nd.children[0]).members), M = [[0, 0], [0, 0], [0, 0]];
      nd.members.forEach((item, i) => { const r = nd.seba2[i] < 0 ? 2 : nd.seba2[i]; M[r][c0.has(item) ? 0 : 1] += 1; });
      ct = h("table", { class: "mini" }, h("thead", {}, h("tr", {}, h("th", { text: "SEBA r = 2 vs cut" }), h("th", { text: "child 0" }), h("th", { text: "child 1" }))),
        h("tbody", {}, ...["set 1", "set 2", "unassigned"].map((nm, r) => h("tr", {}, h("td", { text: nm }), h("td", { text: String(M[r][0]) }), h("td", { text: String(M[r][1]) })))));
    }
    const sb = nd.seba2 ? "; colour (a) by it to see where the two disagree" : "";
    card.replaceChildren(h("p", { class: "card-title", text: "(d) SEBA r = 2 beside the cut; the node's numbers" }),
      h("div", { class: "jac" }, h("div", { class: "jrow" }, h("span", { text: "SEBA r = 2 vs the cut: matched Jaccard" }), h("span", { class: "jv", text: fmt(nd.seba2_vs_cut, 3) })),
        h("div", { class: "jrow" }, h("span", { text: "items SEBA leaves unassigned" }), h("span", { class: "jv", text: String(nd.seba2_unassigned) }))),
      ...(ct ? [ct] : []), tb, h("p", { class: "card-note", text: "SEBA is computed on the same node operator and is never used to make a cut" + sb + "." }));
  }
  ends(card, t, nd, byId) {
    const single = t.block?.kind === "single";
    const col = (k, items) => h("div", { class: "zoom-end" }, h("p", { class: "card-title" }, h("span", { class: "swatch", style: { "--sw": D.pal.set(k) } }), `child ${k}: the 6 items with the ${k === 0 ? "largest" : "smallest"} ψ${nd.axis_used}`),
      ...items.map(i => h("div", { class: "snip-row" }, srcSwatch(i), single ? snippetEl(i, 128, { before: 36, after: 0 }) : snippetEl(i, 36, { before: 36, after: 0, cls: "snip snip-plain" }))));
    card.replaceChildren(h("p", { class: "card-title", text: "(e) Representative snippets" }),
      h("p", { class: "card-note", text: single ? "Tokens 93–128 of each snippet (token 128 underlined)." : "Tokens 1–36 of each snippet (the block reads every position from 4 to 128)." }),
      h("div", { class: "zoom-endcols" }, col(0, nd.ends[0]), col(1, nd.ends[1])));
  }
}
function makeupLine(nd) {
  if (!nd.makeup) return h("span");
  return h("div", { class: "f5-key" }, ...nd.makeup.map((v, k) => h("span", {}, h("i", { style: { background: D.srcCol[k] } }), `${D.sources[k]} ${v}`)));
}
