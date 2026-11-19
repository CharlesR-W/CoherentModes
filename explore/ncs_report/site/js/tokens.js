// The Mamba-130m token-context explorer: same look and conventions as the ViT/ResNet explorer (explorer.js), for
// 8,000 token contexts instead of images. Data: d/mamba/ (written by build_mamba.py). Labels (token classes,
// sources, positions) only colour and describe; the sets and eigenfunctions come from the stored spectral objects.
import * as D from "./data.js";
import { h, el, showTip, hideTip, tipRows } from "./charts.js";
import { Palette, divLUT, qDiv } from "./colors.js";
import { paint, toCoords, tours as allTours } from "./explorer.js";

const $ = id => document.getElementById(id);
const DIM = "#e7e6e1";
const OTHER = "#8d8c86";
const BASE = "d/mamba/";
const PANELS = {
  main: ["Bearly", "Bmid", "Blate", "Bpoint", "Bstate", "whole"],
  refs: ["Bearly", "Bmid", "Blate", "Bpoint", "Bstate", "whole", "bag", "recent", "pred"],
  lags: ["Bearly", "Bmid", "Blate", "Bpoint", "Bstate", "whole", "Blagpoint", "Blagwhole"],
};
const COLOURS = { seba: "coherent sets (SEBA)", eig: "eigenfunction ψₖ", nodal: "sign of ψₖ", cls_t: "token class at t", cls_n: "token class at t+1", src: "kind of source", pos: "position t in the snippet" };
const CLASS_NAMES = { newline: "newline", indent: "indentation", number: "number", latex: "LaTeX", open: "opening bracket", close: "closing bracket", sentence_punct: "sentence punctuation", symbol: "symbol", function_word: "function word", word_start: "word start", word_cont: "word continuation", other: "other" };
const SRC_COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#6250d6"];

// ------------------------------------------------------------------------------------------ data
let CTX = null;
// a share as a percentage: whole percent from 1% up, one decimal below (so 11 of 2,798 reads 0.4%, never 0%)
export function pctShare(x) { const p = 100 * x; return p > 0 && p < 0.95 ? `${p.toFixed(1)}%` : `${p.toFixed(0)}%`; }

export function mctx() {
  if (CTX) return CTX;
  CTX = (async () => {
    const [m, items, ctxb, vocab, layb, vm] = await Promise.all([D.json(BASE + "meta.json"), D.json(BASE + "items.json"), D.bytes(BASE + "ctx.bin"),
      D.json(BASE + "vocab.json"), D.bytes(BASE + "layouts.bin"), D.meta()]);
    const pal = new Palette(vm.palette);
    const N = m.n;
    const c = { m, items, toks: new Uint16Array(ctxb), vocab, coords: toCoords(new Uint16Array(layb)), pal, N, L: m.layouts.length, objs: new Map() };
    c.classCols = vm.palette.token_classes || vm.palette.classes.concat(["#8d8c86", "#3a3a3a"]);
    return c;
  })();
  return CTX;
}
export function setColour(c, j) {
  if (j < 8) return c.pal.sets[j];
  const dk = c.pal.p.sets_dark || [];
  return j < 16 && dk[j - 8] ? dk[j - 8] : OTHER;
}
export async function object(c, o, ck) {
  const key = ck + "/" + o;
  if (c.objs.has(key)) return c.objs.get(key);
  const p = (async () => {
    const r = await fetch(`${BASE}obj/${ck}/${o}.bin.gz`);
    if (!r.ok) throw new Error(`object ${key}: HTTP ${r.status}`);
    const b = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const L = c.m.obj_layout, v = f => {
      const x = L[f];
      return x.dtype === "f32" ? new Float32Array(b, x.offset, x.count) : x.dtype === "i8" ? new Int8Array(b, x.offset, x.count)
        : x.dtype === "u8" ? new Uint8Array(b, x.offset, x.count) : new Uint16Array(b.slice(x.offset, x.offset + 2 * x.count));
    };
    const ex = await D.json(`${BASE}obj/${ck}/${o}.json`);
    return { o, ck, vals: v("vals"), pn: v("pn"), vecs: v("vecs"), labels: v("labels"), memb: v("memb"), ends: v("ends"), examples: ex.examples };
  })();
  p.catch(() => c.objs.delete(key));
  c.objs.set(key, p);
  return p;
}
const rIndex = (c, r) => c.m.ranks.indexOf(r);
export function labelsOf(c, ob, r) { const ri = rIndex(c, r); return ob.labels.subarray(ri * c.N, (ri + 1) * c.N); }
export function membOf(c, ob, r) { const ri = rIndex(c, r); return ob.memb.subarray(ri * c.N, (ri + 1) * c.N); }
export function psiOf(c, ob, k) { return ob.vecs.subarray((k - 1) * c.N, k * c.N); }
// stored ends of psi_k: end 0 = the 12 largest entries, end 1 = the 12 smallest
export function endsOf(ob, k, end) { const o = ((k - 1) * 2 + end) * 12; return Array.from(ob.ends.subarray(o, o + 12)); }

// ------------------------------------------------------------------------------------------ tokens and contexts
export function tokStr(c, id) { return id === 65535 ? "" : (c.vocab[id] ?? `‹${id}›`); }
// a token as a chip: quoted, whitespace made visible (↵ newline, ⇥ tab, runs of spaces counted)
export function tokChip(c, id) {
  const s = tokStr(c, id);
  if (/^ +$/.test(s) && s.length > 1) return `${s.length} spaces`;
  return "“" + s.replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/ /g, "\u00a0") + "”";
}
// one context as a line: the tokens before t (muted), the token at t boxed, then ▸ and the next tokens
export function contextLine(c, i, opts = {}) {
  const L = c.m.ctx_len, T0 = c.m.ctx_t, before = opts.before ?? 14, after = opts.after ?? 3;
  const line = h("span", { class: "ctx-line" });
  const vis = s => s.replace(/\n/g, "↵").replace(/\t/g, "⇥");
  let pre = "";
  for (let q = Math.max(0, T0 - before); q < T0; q++) pre += tokStr(c, c.toks[i * L + q]);
  if (T0 - before > 0 && c.toks[i * L + T0 - before - 1] !== 65535) pre = "…" + pre;
  line.append(h("span", { class: "ctx-pre" }, h("span", { text: vis(pre) })), h("span", { class: "ctx-t", text: vis(tokStr(c, c.toks[i * L + T0])) || " " }));
  let post = "";
  for (let q = T0 + 1; q < Math.min(L, T0 + 1 + after); q++) post += tokStr(c, c.toks[i * L + q]);
  line.append(h("span", { class: "ctx-arrow", text: "▸" }), h("span", { class: "ctx-post", text: vis(post) }));
  return line;
}
export function contextTip(c, i, extra = []) {
  const it = c.items, m = c.m;
  const box = h("div", { class: "tt-ctx" }, h("div", { class: "tt-head", text: `context #${i} · snippet ${it.U[i]} · t = ${it.t[i]}` }), contextLine(c, i, { before: 22, after: 4 }));
  const rows = [
    { name: "token at t", value: `${tokChip(c, it.T[i])} (${CLASS_NAMES[m.classes[it.Cc[i]]]})` },
    { name: "next token", value: `${tokChip(c, it.Nx[i])} (${CLASS_NAMES[m.classes[it.Cn[i]]]})` },
    { name: "model's top-1 next", value: `${tokChip(c, it.A[i])} (p = ${it.p1[i].toFixed(2)})` },
    { name: "kind of source", value: m.sources[it.W[i]] }, ...extra];
  box.append(tipRows("", rows));
  return box;
}
// per-set summary computed from the exported arrays (equal to sets_<ck>.json for r = 8, 16: checked at build)
export function summary(c, ids) {
  const it = c.items, n = ids.length, ct = new Array(12).fill(0), src = new Array(5).fill(0), tt = new Map(), tn = new Map();
  for (const i of ids) { ct[it.Cc[i]]++; src[it.W[i]]++; tt.set(it.T[i], (tt.get(it.T[i]) || 0) + 1); tn.set(it.Nx[i], (tn.get(it.Nx[i]) || 0) + 1); }
  const top = mp => [...mp.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  return { n, ct, src, tt: top(tt), tn: top(tn) };
}
function compBar(c, counts, total, cols, names, title) {
  const bar = h("span", { class: "comp comp-lg" });
  counts.forEach((v, k) => { if (v) bar.append(h("i", { style: { width: (100 * v / total) + "%", background: cols[k] } })); });
  bar.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(title, counts.map((v, k) => ({ name: names[k], value: `${v.toLocaleString("en")} (${pctShare(v / total)})`, color: cols[k], v })).filter(x => x.v).sort((a, b) => b.v - a.v))));
  bar.addEventListener("pointerleave", hideTip);
  return bar;
}
function chips(c, s, n = 3) {
  const fmt = arr => arr.slice(0, n).map(([id, k]) => `${tokChip(c, id)} ${k}`).join(" · ");
  const share = s.n ? s.tt[0][1] / s.n : 0;
  return h("div", { class: "tok-chips" }, h("span", { class: "tk-l", text: "t" }), h("span", { class: "tk", text: fmt(s.tt) }),
    h("span", { class: "tk-l", text: "→ t+1" }), h("span", { class: "tk", text: fmt(s.tn) }),
    h("span", { class: "tk-share", text: `commonest token at t: ${pctShare(share)}`, title: "share of the set's contexts whose token at t is the commonest one" }));
}

// ------------------------------------------------------------------------------------------ explorer
export class MambaExplorer {
  constructor(root) { this.root = root; this.ready = this.init(); }
  async init() {
    const c = this.c = await mctx();
    this.st = { o: "Bpoint", ck: "trained", color: "seba", r: 16, k: 1, j: null, sel: [], lay: "main", ps: 1.4, more: 0, zoom: true };
    this.hover = -1; this.hoverSet = null;
    this.buildControls();
    new ResizeObserver(() => this.layout()).observe($("mx-body"));
    window.addEventListener("resize", () => this.layout());
    await this.update();
  }

  // ---------------------------------------------------------------- controls
  buildControls() {
    const c = this.c, m = c.m, so = $("mx-obj");
    const groups = [["residual", "Residual stream (h after layer l)"], ["state", "Recurrent state"], ["along reading", "Along reading (lag τ)"], ["references", "References (input-embedding averages; the model's predictions)"]];
    for (const [g, label] of groups) {
      const og = document.createElement("optgroup"); og.label = label;
      const names = Object.keys(m.objects).filter(o => m.objects[o].group === g);
      const blockFirst = names.sort((a, b) => (a.startsWith("B") || a === "whole" ? 0 : 1) - (b.startsWith("B") || b === "whole" ? 0 : 1) || m.order.indexOf(a) - m.order.indexOf(b));
      blockFirst.forEach(o => og.append(new Option(m.objects[o].name, o)));
      so.append(og);
    }
    so.addEventListener("change", e => { this.st.o = e.target.value; if (!m.objects[this.st.o].ck.includes(this.st.ck)) this.st.ck = "trained"; this.resetSel(); this.update(); });
    const seg = (id, get, set) => {
      const s = $(id);
      s.querySelectorAll("button").forEach(b => { b.setAttribute("role", "radio"); b.addEventListener("click", () => { if (!b.disabled) { set(b.dataset.v); this.update(); } }); });
      s._sync = () => s.querySelectorAll("button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.v === String(get()))));
      return s;
    };
    this.segs = [seg("mx-ck", () => this.st.ck, v => { this.st.ck = v; this.resetSel(); }), seg("mx-lay", () => this.st.lay, v => { this.st.lay = v; })];
    $("mx-color").addEventListener("change", e => { this.st.color = e.target.value; this.resetSel(); this.update(); });
    $("mx-k").addEventListener("change", e => { if (this.isPsi()) this.st.k = +e.target.value; else this.st.r = +e.target.value; this.resetSel(); this.update(); });
    $("mx-ps").addEventListener("input", e => { this.st.ps = +e.target.value; this.draw(); });
    $("mx-zoom").addEventListener("change", e => { this.st.zoom = e.target.checked; this.draw(); });
    // tours (shared list; each tour's hash names its network)
    const sel = $("mx-tour-sel");
    this.tours = allTours();
    for (const [net, lab] of [["mamba", "Mamba-130m"], ["vit", "DINOv2 ViT"]]) {
      const og = document.createElement("optgroup"); og.label = lab;
      this.tours.filter(t => (new URLSearchParams(t.hash.split("?")[1]).get("net") || "vit") === net).forEach(t => og.append(new Option(t.label, t.id)));
      sel.append(og);
    }
    sel.addEventListener("change", () => { if (sel.value) location.hash = `#explorer?tour=${encodeURIComponent(sel.value)}`; });
    $("mx-tour-reset").addEventListener("click", () => { if (this.tour) this.applyParams(new URLSearchParams({ tour: this.tour.id })); });
    $("mx-tour-close").addEventListener("click", () => { this.setTour(null); this.writeHash(); });
    const how = $("mx-howto"), btn = $("mx-howto-btn"), KEY = "cm.mexplorer.howto-seen";
    let seen = false; try { seen = localStorage.getItem(KEY) === "1"; localStorage.setItem(KEY, "1"); } catch (e) { /* private mode */ }
    how.open = !seen;
    const sync = () => btn.setAttribute("aria-expanded", String(how.open));
    how.addEventListener("toggle", sync); sync();
    btn.addEventListener("click", () => { how.open = !how.open; if (how.open) how.scrollIntoView({ block: "nearest" }); });
  }
  isPsi() { return this.st.color === "eig" || this.st.color === "nodal"; }
  resetSel() { this.st.j = null; this.st.sel = []; this.st.more = 0; }
  setTour(t) {
    this.tour = t ? { id: t.id, base: null } : null;
    $("mx-tour").hidden = !t; $("mx-tour-sel").value = t ? t.id : "";
    if (!t) return;
    $("mx-tour-label").textContent = t.label;
    const note = $("mx-tour-note"); note.innerHTML = t.note;
    if (window.renderMathInElement) window.renderMathInElement(note, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
    $("mx-tour-reset").hidden = true; $("mx-tour").classList.remove("moved");
  }
  syncControls() {
    const st = this.st, m = this.c.m;
    $("mx-obj").value = st.o; $("mx-color").value = st.color;
    this.segs.forEach(s => s._sync());
    $("mx-ck").querySelector('[data-v="init"]').disabled = !m.objects[st.o].ck.includes("init");
    const ks = $("mx-k"), want = this.isPsi() ? "psi" : "set";
    if (ks._kind !== want) {
      ks._kind = want; ks.innerHTML = "";
      if (want === "psi") for (let k = 1; k <= m.n_vecs; k++) ks.add(new Option(`ψ${k}`, k));
      else m.ranks.forEach(r => ks.add(new Option(`r = ${r} sets`, r)));
    }
    ks.value = want === "psi" ? st.k : st.r;
    $("mx-k-label").hidden = st.color === "seba" || this.isPsi();
  }
  writeHash() {
    const st = this.st;
    const p = new URLSearchParams({ net: "mamba", o: st.o, ck: st.ck, c: st.color });
    if (this.isPsi()) p.set("k", st.k); else p.set("r", st.r);
    if (st.j != null) p.set("j", st.j);
    if (st.sel.length) p.set("sel", st.sel.map(([r, j]) => `${r}.${j}`).join(","));
    if (st.lay !== "main") p.set("lay", st.lay);
    let hsh = "#explorer?" + p.toString().replace(/%2C/g, ",");
    if (this.tour) {
      if (this.tour.base == null) this.tour.base = hsh;
      const on = hsh === this.tour.base;
      $("mx-tour").classList.toggle("moved", !on); $("mx-tour-reset").hidden = on;
      $("mx-tour-sel").value = on ? this.tour.id : "";
      if (on) hsh += "&tour=" + encodeURIComponent(this.tour.id);
    }
    this.lastHash = hsh;
    if (location.hash !== hsh && document.body.dataset.tab === "explorer") history.replaceState(null, "", hsh);
  }
  applyParams(p) {
    const st = this.st, m = this.c.m;
    const tid = p.get("tour"), tour = tid != null ? this.tours.find(t => t.id === tid) : null;
    if (tour) p = new URLSearchParams(tour.hash.split("?")[1] || "");
    this.setTour(tour);
    const o = p.get("o"); if (o && m.objects[o]) st.o = o;
    const ck = p.get("ck"); st.ck = ck && m.objects[st.o].ck.includes(ck) ? ck : "trained";
    const col = p.get("c"); if (col && COLOURS[col]) st.color = col;
    const r = parseInt(p.get("r"), 10); if (m.ranks.includes(r)) st.r = r;
    const k = parseInt(p.get("k"), 10); if (k >= 1 && k <= m.n_vecs) st.k = k;
    const j = parseInt(p.get("j"), 10); st.j = Number.isFinite(j) && j >= 0 && j < st.r ? j : null;
    st.sel = (p.get("sel") || "").split(",").filter(Boolean).map(x => x.split(".").map(Number)).filter(([a, b]) => m.ranks.includes(a) && b >= -1 && b < a);
    if (st.j == null) st.sel = [];
    st.more = 0;
    const lay = p.get("lay"); st.lay = PANELS[lay] ? lay : "main";
    return this.update();
  }
  async update() {
    const tok = this._tok = (this._tok || 0) + 1;
    $("mx-status").textContent = "loading…";
    try {
      const ob = await object(this.c, this.st.o, this.st.ck);
      if (tok !== this._tok) return;
      this.ob = ob;
      this.syncControls();
      this.colouring();
      this.layout(true);
      this.renderSide(); this.renderCaption();
      this.writeHash();
      $("mx-status").textContent = "";
    } catch (e) { console.error(e); $("mx-status").textContent = "failed to load: " + e.message; }
  }

  // ---------------------------------------------------------------- selection (set j of r, then members under other r)
  members() {
    const st = this.st, c = this.c, N = c.N;
    if (st.j == null || st.color !== "seba") return null;
    const lab = labelsOf(c, this.ob, st.r);
    let ids = []; for (let i = 0; i < N; i++) if (lab[i] === st.j) ids.push(i);
    for (const [r, j] of st.sel) { const l2 = labelsOf(c, this.ob, r); ids = ids.filter(i => l2[i] === j); }
    return ids;
  }

  // ---------------------------------------------------------------- colouring
  colouring() {
    const st = this.st, c = this.c, N = c.N, it = c.items;
    const ci = new Uint8Array(N), key = new Float64Array(N);
    let cols, groups = null, cont = null;
    if (st.color === "seba") {
      const r = st.r, lab = labelsOf(c, this.ob, r), mb = membOf(c, this.ob, r);
      for (let i = 0; i < N; i++) { ci[i] = lab[i] < 0 ? r : lab[i]; key[i] = lab[i] < 0 ? -1 : mb[i]; }
      cols = [...Array(r)].map((_, j) => setColour(c, j)).concat([c.pal.unassigned]);
      groups = { n: r + 1, unassigned: r, names: [...Array(r)].map((_, j) => `set ${j + 1}`).concat(["unassigned"]) };
    } else if (this.isPsi()) {
      const v = psiOf(c, this.ob, st.k);
      for (let i = 0; i < N; i++) { key[i] = Math.abs(v[i]); ci[i] = st.color === "eig" ? qDiv(v[i] / 127) : v[i] > 0 ? 0 : v[i] < 0 ? 1 : 2; }
      if (st.color === "eig") { cols = divLUT(c.pal); cont = { label: `ψ${st.k} / max|ψ${st.k}|` }; }
      else { cols = [c.pal.p.diverging.pos, c.pal.p.diverging.neg, c.pal.unassigned]; groups = { n: 3, names: [`ψ${st.k} > 0`, `ψ${st.k} < 0`, `ψ${st.k} = 0`] }; }
    } else if (st.color === "pos") {
      const lo = 16, hi = 112;
      for (let i = 0; i < N; i++) { const t = (it.t[i] - lo) / (hi - lo); ci[i] = Math.max(0, Math.min(40, Math.round(t * 40))); key[i] = Math.sin(i * 12.9898) * 43758.5453 % 1; }
      cols = [...Array(41)].map((_, q) => c.pal.seqDot(q / 40)); cont = { label: "position t in the snippet", lo, hi, seq: true };
    } else {
      const arr = st.color === "src" ? it.W : st.color === "cls_t" ? it.Cc : it.Cn;
      for (let i = 0; i < N; i++) { ci[i] = arr[i]; key[i] = Math.sin(i * 12.9898) * 43758.5453 % 1; }
      if (st.color === "src") { cols = SRC_COLS; groups = { n: 5, names: c.m.sources.slice() }; }
      else { cols = c.classCols.slice(0, 12); groups = { n: 12, names: c.m.classes.map(x => CLASS_NAMES[x]) }; }
    }
    // highlighted set (hover in the side card) or selection: everything else dims
    const mem = this.members(), hl = this.hoverSet;
    let keep = null;
    if (hl) keep = new Set(hl);
    else if (mem) keep = new Set(mem);
    const D_ = cols.length;
    cols = cols.concat([DIM]);
    if (keep) for (let i = 0; i < N; i++) if (!keep.has(i)) { ci[i] = D_; key[i] = -2; }
    const order = Uint16Array.from({ length: N }, (_, i) => i).sort((a, b) => key[a] - key[b]);
    this.col = { ci: [ci], order: [order], cols, groups, cont, pal32: new Uint32Array(cols.map(css32)) };
    this.keep = keep;
  }

  // ---------------------------------------------------------------- panels
  layout(force) {
    const body = $("mx-body"); if (!body || !this.col || this.root.hidden) return;
    const names = PANELS[this.st.lay], L = names.length, grid = $("mx-grid"), W = grid.clientWidth || 600;
    const cols = W < 520 ? 2 : 3, gap = 10, cell = Math.min(300, Math.floor((W - (cols - 1) * gap) / cols));
    const key = `${names.join()}|${cell}|${cols}`;
    if (key !== this._gkey || force === "rebuild") {
      this._gkey = key;
      grid.replaceChildren();
      this.panels = names.map(o => {
        const idx = this.c.m.layouts.indexOf(o);
        const cv = h("canvas", { class: "mx-cv" }), ov = h("canvas", { class: "mx-ov" });
        const lab = h("div", { class: "mx-plab", text: this.c.m.layout_names[o] });
        const box = h("div", { class: "mx-panel", style: { width: cell + "px" } }, lab, h("div", { class: "mx-cvwrap", style: { width: cell + "px", height: cell + "px" } }, cv, ov));
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        for (const x of [cv, ov]) { x.width = x.height = Math.round(cell * dpr); x.style.width = x.style.height = cell + "px"; }
        const p = { o, idx, cv, ov, box, lab, cell, dpr };
        cv.addEventListener("pointermove", e => this.onMove(e, p));
        cv.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
        cv.addEventListener("pointerdown", e => { if (e.pointerType !== "mouse") this.onMove(e, p, true); });
        grid.append(box);
        return p;
      });
      grid.style.gridTemplateColumns = `repeat(${cols}, ${cell}px)`;
    }
    this.draw();
  }
  // coordinates of each panel, refit to the selection's extent when zooming
  panelCoords(p) {
    const c = this.c, N = c.N, base = p.idx * N * 2, src = c.coords;
    const sub = src.subarray(base, base + 2 * N);
    const ids = this.st.zoom ? this.members() : null;
    if (!ids || ids.length < 3) return { arr: sub, f: x => x };
    let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
    const xs = ids.map(i => sub[2 * i]).sort((a, b) => a - b), ys = ids.map(i => sub[2 * i + 1]).sort((a, b) => a - b);
    const q = (a, t) => a[Math.min(a.length - 1, Math.max(0, Math.round(t * (a.length - 1))))];
    x0 = q(xs, .01); x1 = q(xs, .99); y0 = q(ys, .01); y1 = q(ys, .99);
    const span = Math.max(x1 - x0, y1 - y0, 0.02) * 1.12, mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const arr = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) { arr[2 * i] = (sub[2 * i] - mx) / span + .5; arr[2 * i + 1] = (sub[2 * i + 1] - my) / span + .5; }
    return { arr, zoomed: true };
  }
  draw() {
    if (!this.panels) return;
    const own = this.ownLayout();
    for (const p of this.panels) {
      const W = p.cv.width, g = p.cv.getContext("2d"), img = g.createImageData(W, W), buf = new Uint32Array(img.data.buffer);
      const pc = this.panelCoords(p); p.pc = pc.arr;
      const pad = Math.round(6 * p.dpr);
      paint(buf, W, W, [{ t: 0, s: 0, S: 1, x: pad, y: pad, w: W - 2 * pad }], pc.arr, this.col, this.c.N, this.st.ps * p.dpr * Math.max(.6, Math.min(1, p.cell / 230)));
      g.putImageData(img, 0, 0);
      p.box.classList.toggle("own", p.o === own);
      p.lab.textContent = this.c.m.layout_names[p.o] + (pc.zoomed ? " (zoomed to the selection)" : "");
    }
    this.drawOverlay();
  }
  ownLayout() { const o = this.st.o; if (this.c.m.layouts.includes(o)) return o; const lo = o.match(/^point_L16_lag/) ? "Blagpoint" : o.match(/^whole_lag/) ? "Blagwhole" : null; return lo; }
  drawOverlay() {
    for (const p of this.panels || []) {
      const W = p.ov.width, g = p.ov.getContext("2d"); g.clearRect(0, 0, W, W);
      if (this.hover < 0 || !p.pc) continue;
      const pad = 6 * p.dpr, w = W - 2 * pad, i = this.hover, x = pad + w * p.pc[2 * i], y = pad + w * (1 - p.pc[2 * i + 1]);
      g.beginPath(); g.arc(x, y, 5.5 * p.dpr, 0, 7); g.strokeStyle = "#fff"; g.lineWidth = 3 * p.dpr; g.stroke(); g.strokeStyle = "#16161a"; g.lineWidth = 1.5 * p.dpr; g.stroke();
    }
  }
  onMove(e, p, touch) {
    const r = p.cv.getBoundingClientRect(), pad = 6, w = r.width - 2 * pad, x = e.clientX - r.left, y = e.clientY - r.top, N = this.c.N, pc = p.pc;
    if (!pc) return;
    let best = -1, bd = (touch ? 14 : 7) ** 2;
    const keep = this.keep;
    for (let i = 0; i < N; i++) {
      if (keep && !keep.has(i)) continue;
      const dx = pad + w * pc[2 * i] - x, dy = pad + w * (1 - pc[2 * i + 1]) - y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    this.setHover(best);
    if (best >= 0) showTip(e.clientX, e.clientY, contextTip(this.c, best, this.extraRows(best))); else hideTip();
  }
  extraRows(i) {
    const st = this.st, c = this.c;
    if (st.color === "seba") { const l = labelsOf(c, this.ob, st.r)[i]; return [{ name: `set (r = ${st.r})`, value: l < 0 ? "unassigned" : `set ${l + 1}`, color: l < 0 ? c.pal.unassigned : setColour(c, l) }]; }
    if (this.isPsi()) return [{ name: `ψ${st.k} / max|ψ${st.k}|`, value: (psiOf(c, this.ob, st.k)[i] / 127).toFixed(2) }];
    return [];
  }
  setHover(i) {
    if (i === this.hover) return;
    this.hover = i; this.drawOverlay();
    document.querySelectorAll("#mx .ctx-row.hl").forEach(x => x.classList.remove("hl"));
    if (i >= 0) document.querySelectorAll(`#mx .ctx-row[data-i="${i}"]`).forEach(x => x.classList.add("hl"));
  }

  // ---------------------------------------------------------------- side panel
  ctxRow(i, sw) {
    const row = h("div", { class: "ctx-row", "data-i": i }, sw ? h("span", { class: "sw", style: { background: sw } }) : null, contextLine(this.c, i));
    row.addEventListener("pointerenter", () => this.setHover(i));
    row.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, contextTip(this.c, i, this.extraRows(i))));
    row.addEventListener("pointerleave", () => { this.setHover(-1); hideTip(); });
    return row;
  }
  renderSide() {
    const st = this.st, c = this.c, m = c.m, ob = this.ob, body = $("mx-sets"), N = c.N;
    const objName = `${m.objects[st.o].name} · ${st.ck === "init" ? "untrained (initialisation)" : "trained"} · kNN graph, k = ${m.objects[st.o].k ?? m.k}`;
    $("mx-sets-sub").textContent = objName;
    body.replaceChildren();
    if (this.isPsi()) {
      $("mx-sets-title").textContent = `Contexts at the two ends of ψ${st.k}`;
      for (const [end, name, col] of [[1, `12 most negative entries of ψ${st.k}`, c.pal.p.diverging.neg], [0, `12 most positive entries of ψ${st.k}`, c.pal.p.diverging.pos]]) {
        const ids = endsOf(ob, st.k, end);
        const s = summary(c, ids);
        body.append(h("div", { class: "set-row" }, h("div", { class: "set-head mx-head" }, h("span", { class: "sw", style: { background: col } }), h("span", { class: "set-name", text: name })),
          chips(c, s), h("div", { class: "ctx-list" }, ...ids.map(i => this.ctxRow(i)))));
      }
      body.append(h("div", { class: "card-note", text: `Each line: up to 14 tokens before t, the token at t boxed, ▸ the next 3 tokens. The two ends are the stored extremes of ψ${st.k} (full precision); the sign of an eigenfunction is arbitrary. Hover a line to ring the context in every panel.` }));
    } else if (st.color !== "seba") {
      $("mx-sets-title").textContent = `Coherent sets, r = ${st.r}`;
      body.append(h("div", { class: "card-note", text: "The colouring is a label (validation only). Choose “coherent sets” or an eigenfunction under Colour by to list sets or the two ends of ψₖ here." }));
      this.renderSetRows(body, false);
    } else {
      $("mx-sets-title").textContent = `Coherent sets, r = ${st.r}: contexts and make-up`;
      this.renderSetRows(body, true);
    }
    this.renderSpectrum();
  }
  renderSetRows(body, active) {
    const st = this.st, c = this.c, m = c.m, ob = this.ob, N = c.N, r = st.r, lab = labelsOf(c, ob, r), mb = membOf(c, ob, r);
    const by = [...Array(r + 1)].map(() => []);
    for (let i = 0; i < N; i++) by[lab[i] < 0 ? r : lab[i]].push(i);
    const ex = ob.examples[String(r)];
    for (let j = 0; j <= r; j++) {
      const un = j === r;
      if (un && !by[r].length) continue;
      const s = summary(c, by[j]), on = st.j === j, dim = st.j != null && !on;
      const head = h("div", { class: "set-head", role: "button", tabindex: 0, title: un ? "" : "Click to select this set: more contexts, and how its members fall under another r" },
        h("span", { class: "sw", style: { background: un ? c.pal.unassigned : setColour(c, j) } }), h("span", { class: "set-name", text: un ? "unassigned" : `set ${j + 1}` }),
        h("span", { class: "n", text: s.n.toLocaleString("en") }), compBar(c, s.ct, s.n || 1, c.classCols, m.classes.map(x => CLASS_NAMES[x]), `token class at t (${s.n.toLocaleString("en")} contexts; validation only)`));
      const row = h("div", { class: "set-row" + (on ? " on" : "") + (dim ? " dim" : "") }, head, chips(c, s));
      if (!un && active) {
        head.addEventListener("click", () => { st.j = on ? null : j; st.sel = []; st.more = 0; this.colouring(); this.draw(); this.renderSide(); this.renderCaption(); this.writeHash(); });
        head.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); head.click(); } });
        head.addEventListener("pointerenter", () => { if (st.j != null) return; this.hoverSet = by[j]; this.colouring(); this.draw(); });
        head.addEventListener("pointerleave", () => { if (!this.hoverSet) return; this.hoverSet = null; this.colouring(); this.draw(); });
        const first = (ex && ex[j] && ex[j].length) ? ex[j] : by[j].slice().sort((a, b) => mb[b] - mb[a]).slice(0, 12);
        if (!on) row.append(h("div", { class: "ctx-list" }, this.ctxRow(first[0])));
        else this.renderSelection(row, by[j], first);
      }
      body.append(row);
    }
    const key = h("div", { class: "class-key" });
    m.classes.forEach((x, k) => key.append(h("span", {}, h("i", { style: { background: c.classCols[k] } }), CLASS_NAMES[x])));
    body.append(key, h("div", { class: "card-note", text: `Bars: token class of the token at t in each set (validation only: labels describe the sets and play no part in finding them). Chips: the commonest tokens at t and at t+1, with counts. Line: the member with the largest SEBA value (up to 14 tokens before t, the token at t boxed, ▸ the next 3). Click a set for 12 members and how they fall under another r.` + (r > 8 ? ` Colours: sets 1–8 take the eight set colours, sets 9–16 darker versions of the same hues${r > 16 ? ", sets 17 and above one grey" : ""}; hovering a row isolates its set in the panels.` : "") }));
  }
  renderSelection(row, members, first) {
    const st = this.st, c = this.c, m = c.m, ob = this.ob, N = c.N;
    const ids = this.members();
    const mbR = membOf(c, ob, st.r);
    if (st.sel.length) {
      const crumbs = h("div", { class: "crumbs" });
      const b0 = h("button", { type: "button", class: "link-btn", text: `set ${st.j + 1} (r = ${st.r})` });
      b0.addEventListener("click", () => { st.sel = []; this.refreshSel(); });
      crumbs.append(b0);
      st.sel.forEach(([r, j], q) => {
        crumbs.append(" › ");
        const b = h("button", { type: "button", class: "link-btn", text: j < 0 ? `unassigned (r = ${r})` : `set ${j + 1} (r = ${r})` });
        b.addEventListener("click", () => { st.sel = st.sel.slice(0, q + 1); this.refreshSel(); });
        crumbs.append(b);
      });
      const s = summary(c, ids);
      row.append(h("div", { class: "sub-h", text: `Selected: ${ids.length.toLocaleString("en")} contexts` }), crumbs, chips(c, s));
    }
    // members in decreasing SEBA value of the first-chosen set (stored examples first)
    const ordered = st.sel.length ? ids.slice().sort((a, b) => mbR[b] - mbR[a]) : first.concat(members.filter(i => !first.includes(i)).sort((a, b) => mbR[b] - mbR[a]));
    const shown = ordered.slice(0, 12 + 12 * st.more);
    row.append(h("div", { class: "imgs-label", text: `members in decreasing SEBA value (set ${st.j + 1}, r = ${st.r})` }), h("div", { class: "ctx-list" }, ...shown.map(i => this.ctxRow(i))));
    if (shown.length < ordered.length) {
      const more = h("button", { type: "button", class: "link-btn", text: `show 12 more (of ${ordered.length.toLocaleString("en")})` });
      more.addEventListener("click", () => { st.more++; this.renderSide(); });
      row.append(more);
    }
    // the same members under another r of the same object
    const others = m.ranks.filter(r => r !== st.r && !st.sel.some(([r2]) => r2 === r));
    if (!others.length) return;
    const want = this._otherR && others.includes(this._otherR) ? this._otherR : (others.find(r => r > st.r) || others[others.length - 1]);
    this._otherR = want;
    const pick = h("select", { "aria-label": "Another number of sets r" });
    others.forEach(r => pick.add(new Option(`r = ${r}`, r))); pick.value = want;
    pick.addEventListener("change", () => { this._otherR = +pick.value; this.renderSide(); });
    const box = h("div", { class: "under" }, h("div", { class: "sub-h" }, `Selected members under another r of the same object: `, pick));
    const l2 = labelsOf(c, ob, want), cnt = new Map();
    for (const i of ids) cnt.set(l2[i], (cnt.get(l2[i]) || 0) + 1);
    const sizes = m.objects[st.o].sizes[st.ck][String(want)];
    const rowsU = [...cnt.entries()].sort((a, b) => b[1] - a[1]);
    for (const [j2, n] of rowsU.slice(0, 12)) {
      const mem2 = ids.filter(i => l2[i] === j2), s2 = summary(c, mem2);
      const total = j2 < 0 ? N - sizes.reduce((a, b) => a + b, 0) : sizes[j2];
      const r2 = h("div", { class: "under-row", role: "button", tabindex: 0, title: "Click to narrow the selection to these contexts" },
        h("span", { class: "un-name", text: j2 < 0 ? `unassigned (r = ${want})` : `set ${j2 + 1} of ${want}` }),
        h("span", { class: "n", text: `${n.toLocaleString("en")} of the selection; ${total.toLocaleString("en")} in all` }), chips(c, s2, 2));
      r2.addEventListener("click", () => { st.sel = st.sel.concat([[want, j2]]); st.more = 0; this.refreshSel(); });
      r2.addEventListener("keydown", e => { if (e.key === "Enter") r2.click(); });
      r2.addEventListener("pointerenter", () => { this.hoverSet = mem2; this.colouring(); this.draw(); });
      r2.addEventListener("pointerleave", () => { this.hoverSet = null; this.colouring(); this.draw(); });
      box.append(r2);
    }
    if (rowsU.length > 12) box.append(h("div", { class: "card-note", text: `${rowsU.length - 12} more sets of r = ${want} hold the remaining ${rowsU.slice(12).reduce((a, b) => a + b[1], 0).toLocaleString("en")} members.` }));
    box.append(h("div", { class: "card-note", text: `The sets at another r are a different SEBA rotation of the same object's eigenfunctions, not a subdivision: a set of r = ${want} can also hold contexts outside the selection (“in all”). Clicking a row keeps the intersection.` }));
    row.append(box);
  }
  refreshSel() { this.colouring(); this.draw(); this.renderSide(); this.renderCaption(); this.writeHash(); }
  renderSpectrum() {
    const c = this.c, ob = this.ob, box = $("mx-spec"), N = c.N, st = this.st;
    box.replaceChildren();
    const W = Math.max(240, box.clientWidth || 300), H = 150, ml = 34, mr = 8, mt = 8, mb = 30;
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` }, box);
    const vals = Array.from(ob.vals).slice(1, 17), lo = Math.min(...vals), y0 = Math.max(0, Math.floor((lo - 0.02) * 20) / 20);
    const X = k => ml + (k - 0.5) * (W - ml - mr) / 16, Y = v => mt + (1 - (v - y0) / (1 - y0)) * (H - mt - mb);
    for (const v of [y0, (y0 + 1) / 2, 1]) {
      el("line", { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v), stroke: "#efeee9" }, svg);
      const t = el("text", { x: ml - 4, y: Y(v) + 3, "text-anchor": "end", "font-size": 10, fill: "#6b6b70" }, svg); t.textContent = v.toFixed(2);
    }
    vals.forEach((v, q) => {
      const k = q + 1, on = this.isPsi() ? k === st.k : (st.color === "seba" && k < st.r);
      el("line", { x1: X(k), x2: X(k), y1: Y(y0), y2: Y(v), stroke: on ? "#16161a" : "#b9b8b2", "stroke-width": 1.5 }, svg);
      const d = el("circle", { cx: X(k), cy: Y(v), r: on ? 3.6 : 2.8, fill: on ? "#16161a" : "#b9b8b2" }, svg);
      d.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(`ψ${k}`, [{ name: `λ${k}`, value: v.toFixed(4) }, { name: "participation number", value: `${ob.pn[q].toFixed(0)} of ${N.toLocaleString("en")} (${(100 * ob.pn[q] / N).toFixed(1)}%)` }])));
      d.addEventListener("pointerleave", hideTip);
      if (k === 1 || k % 4 === 0) { const t = el("text", { x: X(k), y: H - mb + 12, "text-anchor": "middle", "font-size": 10, fill: "#6b6b70" }, svg); t.textContent = k; }
    });
    const t = el("text", { x: (ml + W - mr) / 2, y: H - 4, "text-anchor": "middle", "font-size": 10, fill: "#6b6b70" }, svg); t.textContent = "k (eigenvalues λ₁ … λ₁₆; λ₀ = 1 is the constant)";
    const pn = Array.from(ob.pn), mn = Math.min(...pn);
    $("mx-spec-note").textContent = `Eigenvalues of the object's averaged operator. ${this.isPsi() ? `ψ${st.k} is marked.` : st.color === "seba" ? `The r = ${st.r} sets come from ψ0..ψ${st.r - 1} (marked up to ψ16).` : ""} Participation number of ψ1..ψ16 (how many contexts an eigenfunction effectively spreads over): smallest ${mn.toFixed(0)} of ${N.toLocaleString("en")} (${(100 * mn / N).toFixed(1)}%), median ${(pn.slice().sort((a, b) => a - b)[8]).toFixed(0)}. Hover a stem for its values.`;
  }

  // ---------------------------------------------------------------- key and caption (definitions only)
  renderCaption() {
    const st = this.st, c = this.c, m = c.m, col = this.col, key = $("mx-key");
    key.replaceChildren();
    if (col.groups) col.groups.names.forEach((nm, j) => { if (j < 33) key.append(h("span", { class: "kk" }, h("i", { style: { background: col.cols[j] } }), nm)); });
    else if (col.cont) key.append(h("span", { class: "kt", text: col.cont.label }), h("span", { class: "kt", text: col.cont.seq ? String(col.cont.lo) : "−1" }),
      h("span", { class: "kbar", style: { background: col.cont.seq ? c.pal.seqGradient() : c.pal.divGradient() } }), h("span", { class: "kt", text: col.cont.seq ? String(col.cont.hi) : "1" }));
    const obj = `${m.objects[st.o].name}, ${st.ck === "init" ? "untrained network (initialisation)" : "trained network"}; kNN graph, k = ${m.objects[st.o].k ?? m.k}`;
    const what = {
      seba: `coherent sets, r = ${st.r}: each context's hard label is the largest of the ${st.r} SEBA vectors from ψ0..ψ${st.r - 1} of this object's averaged operator; grey = unassigned (no positive entry); points drawn in order of SEBA value, largest on top.`,
      eig: `ψ${st.k} / max|ψ${st.k}|, the eigenfunction of this object's averaged operator for λ${st.k} (eigenvalues in decreasing order from λ0 = 1, the constant); diverging scale; the sign of an eigenfunction is arbitrary; largest |ψ| on top.`,
      nodal: `sign of ψ${st.k} of this object's averaged operator; largest |ψ| on top.`,
      cls_t: "class of the token at t (12 classes, validation only). The object sets only the side panel.",
      cls_n: "class of the token at t+1 (validation only). The object sets only the side panel.",
      src: "kind of source of the snippet the context comes from: maths, code, wiki, forum or poetry (validation only). The object sets only the side panel.",
      pos: "position t of the context's last token in its 128-token snippet (16 to 112). The object sets only the side panel.",
    }[st.color];
    const sel = this.members();
    const selTxt = sel ? ` Selection: set ${st.j + 1} of r = ${st.r}${st.sel.map(([r, j]) => j < 0 ? `, unassigned under r = ${r}` : `, ∩ set ${j + 1} of r = ${r}`).join("")} (${sel.length.toLocaleString("en")} contexts); others light grey${st.zoom ? "; panels refit to the selection's extent" : ""}.` : "";
    const own = this.ownLayout();
    const part = (label, text) => [h("b", { text: label + " " }), text + " "];
    $("mx-caption").replaceChildren(
      ...part("Colour from:", obj + "."),
      ...part("Colour:", what + selTxt),
      ...part("Positions:", `each panel is a 2-d UMAP of ψ1..ψ16 of the object named above it (not of activations); display only, no number on this page is computed from them. All ${PANELS[st.lay].length} panels show the same ${c.N.toLocaleString("en")} contexts. ${own ? "Outlined: the panel laid out by the colouring's own object." : "The colouring's object has no stored layout of its own; it is painted on the stored ones."}`));
  }
}

const _c32 = new Map();
function css32(css) {
  if (_c32.has(css)) return _c32.get(css);
  let r, g, b;
  if (css.startsWith("#")) { const v = parseInt(css.slice(1), 16); r = (v >> 16) & 255; g = (v >> 8) & 255; b = v & 255; }
  else [r, g, b] = css.match(/\d+/g).map(Number);
  const v = ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
  _c32.set(css, v); return v;
}
