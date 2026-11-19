r"""One-file engine for experiment notebooks: a tabbed, self-contained HTML page.

Copy this file next to your page builder (``scripts/notebook.py``) and import
it. It has no dependencies beyond the standard library. The page it writes
loads Plotly and MathJax from CDNs; everything else -- data, images, styles --
is inlined, so one ``.html`` file is the whole deliverable.

The shape it produces (see SKILL.md for why):

    Notebook -> Tab -> blocks (prose, equation, figure, image, methodology, ...)

Figures carry their numbers as JSON and draw in the browser with a JavaScript
function you write; every control in a tab shares one state object, so one
step slider can drive several figures at once. Minimal example::

    from notebook import Notebook
    nb = Notebook("Cells and clocks", "paper-MLP pair", status="MEASURED · 1 seed")
    t = nb.tab("overview", "Overview")
    t.prose("<p>The question, in two sentences.</p>")
    t.equation(r"d(\alpha) = \frac{\|F(z(\alpha))-F(z_i)\|}{\|F(z(\alpha))-F(z_i)\|+\|F(z(\alpha))-F(z_j)\|}")
    e = nb.tab("interp", "Interpolation response")
    e.controls([Slider("step", "Step", values=[0, 1, 10, 100]), Radio("cut", "Cut", ["x0", "h1"])])
    e.figure("curves", "Response along the path", data=payload,
             draw="(el, d, c) => Plotly.react(el, [{x: d.alpha, y: d.curves[c.cut][c.step]}], LAYOUT({height: 380}))")
    nb.write("pages/notebook.html")

Draw functions receive ``(el, d, c)``: the plot ``<div>``, the figure's data,
and the tab's control state (``c.step`` is the *value*, not the index). They
run when the tab is first shown and again on every control change in the tab.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
MATHJAX_CDN = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"

# --------------------------------------------------------------------- style --

STYLE = r"""
* { box-sizing: border-box; }
:root {
  --ink:#2b2a27; --dim:#6b665e; --line:#e2ddd5; --border:#d3cdc3; --bg:#fdfcfa; --soft:#f6f3ee;
  --accent:#2f6f9f; --warn-bg:#f7ecdf; --warn-ink:#8a4a2a; --interp-bg:#eef3f8; --interp-ink:#2a4d6e;
  --c0:#4477aa; --c1:#ee6677; --c2:#228833; --c3:#ccbb44; --c4:#66ccee; --c5:#aa3377; --c6:#bbbbbb;
}
html { scroll-behavior: smooth; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5;
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:1240px; margin:0 auto; padding:26px 28px 90px; }
header.hero { margin-bottom:12px; }
header.hero .kicker { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim); }
header.hero h1 { font-size:30px; line-height:1.15; margin:4px 0 6px; font-weight:700; }
header.hero .subtitle { color:var(--dim); max-width:900px; margin:0 0 10px; }
.pill { display:inline-block; font-size:12px; font-weight:650; padding:3px 10px; border-radius:999px;
  border:1px solid var(--border); background:var(--soft); margin-right:6px; }
.pill.warn { background:var(--warn-bg); color:var(--warn-ink); border-color:#e0c9ae; }
.banner { border:1px solid var(--border); border-radius:10px; background:white; padding:10px 16px 6px; margin:0 0 6px; }
.banner .label { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim); font-weight:650; margin-bottom:4px; }
.banner .archs { display:flex; flex-wrap:wrap; gap:10px 28px; align-items:flex-start; }
.banner figure.arch { margin:0; }
.banner figure.arch figcaption { font-size:12px; color:var(--dim); margin-top:2px; }
nav.tabs { position:sticky; top:0; z-index:5; background:var(--bg); padding:10px 0;
  border-bottom:1px solid var(--line); margin-bottom:18px; display:flex; flex-wrap:wrap; gap:6px; }
nav.tabs button { font:inherit; font-size:14px; font-weight:600; padding:7px 14px; border-radius:8px;
  border:1px solid var(--border); background:white; color:var(--dim); cursor:pointer; }
nav.tabs button.active { background:var(--accent); border-color:var(--accent); color:white; }
nav.tabs button .idx { opacity:.55; font-weight:500; margin-right:6px; }
section.tab { display:none; }
section.tab.active { display:block; }
h2 { font-size:22px; margin:26px 0 8px; line-height:1.2; }
h3 { font-size:17px; margin:22px 0 6px; }
h4 { font-size:14px; margin:0 0 6px; }
p, li { max-width:78ch; }
a { color:var(--accent); }
code { font:13px ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--soft); padding:1px 5px;
  border-radius:4px; }
.cols { display:grid; gap:22px; grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); align-items:start; }
.box { border:1px solid var(--border); border-radius:8px; padding:12px 16px; background:white; margin:12px 0; }
.box.bridge { border-left:4px solid var(--accent); background:#f7fafd; }
.box.bridge .label, .box.interp .label, .box.takeaway .label, .box.verdict .label { font-size:11.5px;
  letter-spacing:.06em; text-transform:uppercase; color:var(--dim); font-weight:700; margin-bottom:4px; }
.box.interp { border-left:4px solid #7a9cc0; background:var(--interp-bg); color:var(--interp-ink); }
.box.takeaway { border-left:4px solid #5d8f75; background:#f1f7f3; }
.box.verdict { border-left:4px solid #c08a3e; background:#fbf5ec; }
.box.verdict.pass { border-left-color:#3c8a5a; background:#eff7f1; }
.box.verdict.fail { border-left-color:#b04a4a; background:#faeeee; }
.eq { margin:12px 0; padding:10px 16px; border:1px solid var(--border); border-left:4px solid var(--accent);
  border-radius:8px; background:white; overflow-x:auto; }
.eq .where { font-size:13.5px; color:var(--dim); margin-top:4px; }
figure { margin:0; }
.fig { border:1px solid var(--border); border-radius:8px; background:white; padding:12px 14px 10px; margin:14px 0; }
.fig .ftitle { font-size:15px; font-weight:700; margin:0 0 4px; }
.fig .fsub { font-size:13px; color:var(--dim); margin:0 0 6px; }
.fig .plot { width:100%; min-height:120px; }
.fig figcaption, .img figcaption { font-size:13.5px; color:var(--dim); margin-top:8px; max-width:none; }
.grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); align-items:start; }
.grid > .fig, .grid > .img { margin:0; }
.img { border:1px solid var(--border); border-radius:8px; background:white; padding:10px 12px; margin:14px 0; }
.img img { display:block; max-width:100%; height:auto; margin:0 auto; }
.img .credit { font-size:12px; color:var(--dim); margin-top:4px; }
.controls { display:flex; flex-wrap:wrap; gap:10px 22px; align-items:center; padding:10px 12px;
  background:var(--soft); border:1px solid var(--line); border-radius:8px; margin:10px 0; }
.controls .ctl { display:inline-flex; align-items:center; gap:8px; font-size:13.5px; }
.controls .ctl > label { font-weight:650; color:var(--dim); }
.seg { display:inline-flex; background:white; border:1px solid var(--border); border-radius:7px; padding:2px; }
.seg button { font:inherit; font-size:13px; border:0; background:transparent; padding:4px 10px; border-radius:5px;
  color:var(--dim); cursor:pointer; }
.seg button.active { background:var(--accent); color:white; }
select { font:inherit; font-size:13px; padding:4px 6px; border:1px solid var(--border); border-radius:6px; background:white; }
input[type=range] { width:220px; }
.play { font:inherit; font-size:12px; border:1px solid var(--border); background:white; border-radius:6px;
  padding:3px 8px; cursor:pointer; }
.val { font-variant-numeric:tabular-nums; min-width:5ch; display:inline-block; }
table.doc { border-collapse:collapse; width:100%; font-size:13.5px; margin:8px 0 14px; }
table.doc th, table.doc td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; }
table.doc th { background:var(--soft); font-weight:650; }
table.doc td.num { text-align:right; font-variant-numeric:tabular-nums; }
.status { font-size:11.5px; font-weight:700; padding:2px 7px; border-radius:5px; background:var(--soft);
  border:1px solid var(--border); white-space:nowrap; }
.status.done { background:#eaf5ee; color:#2f6b45; border-color:#bfe0c9; }
.status.open { background:#fbf2e6; color:#8a5a1f; border-color:#ecd3ae; }
details { margin:10px 0; }
details summary { cursor:pointer; font-weight:600; color:var(--dim); }
footer { margin-top:50px; padding-top:12px; border-top:1px solid var(--line); color:var(--dim); font-size:12.5px; }
.mockup { position:relative; }
.mockup::after { content:"MOCKUP"; position:absolute; inset:0; display:flex; align-items:center;
  justify-content:center; font-size:64px; font-weight:800; color:rgba(180,60,60,.13); pointer-events:none;
  transform:rotate(-18deg); }
"""

# ------------------------------------------------------------------- script --

PRELUDE = r"""
const DATA = JSON.parse(document.getElementById('nb-data').textContent);
const FIGS = {};      // id -> {tab, draw}
const STATE = {};     // tab -> {control name -> value}
const DRAWN = {};     // id -> bool
const PAL = ['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb','#000000'];
function merge(a, b) {
  const out = Object.assign({}, a);
  for (const k of Object.keys(b || {})) {
    out[k] = (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a && a[k] && typeof a[k] === 'object')
      ? merge(a[k], b[k]) : b[k];
  }
  return out;
}
const AXIS = {gridcolor:'#eeebe5', zerolinecolor:'#e2ddd5', linecolor:'#d3cdc3', ticks:'outside',
  tickcolor:'#d3cdc3', title:{font:{size:13}}, automargin:true};
function LAYOUT(extra) {
  return merge({margin:{l:56,r:18,t:36,b:48}, paper_bgcolor:'white', plot_bgcolor:'white',
    font:{family:'system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif', size:12.5, color:'#2b2a27'},
    xaxis:AXIS, yaxis:AXIS, legend:{orientation:'h', y:1.06, x:0, font:{size:12}},
    hovermode:'closest', height:380}, extra || {});
}
const CONFIG = {displaylogo:false, responsive:true, modeBarButtonsToRemove:['lasso2d','select2d']};
function fmt(v) { if (v == null) return ''; const a = Math.abs(v);
  return a >= 1e5 || (a > 0 && a < 1e-3) ? v.toExponential(2) : a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toFixed(3); }
function ctrlValue(tab, name) { return STATE[tab][name]; }
function drawTab(tab) {
  for (const id of Object.keys(FIGS)) {
    if (FIGS[id].tab !== tab) continue;
    const el = document.getElementById('plot-' + id);
    if (!el) continue;
    let d = DATA[id]; if (typeof d === 'string' && d.startsWith('@')) d = DATA[d.slice(1)];
    try { FIGS[id].draw(el, d, STATE[tab]); DRAWN[id] = true; }
    catch (e) { el.innerHTML = '<p style="color:#b04a4a">draw error: ' + e.message + '</p>'; console.error(id, e); }
  }
}
function setControl(tab, name, value) { STATE[tab][name] = value; syncControl(tab, name); drawTab(tab); }
function syncControl(tab, name) {
  document.querySelectorAll('[data-tab="' + tab + '"][data-ctl="' + name + '"]').forEach(w => {
    const v = STATE[tab][name];
    w.querySelectorAll('.seg button').forEach(b => b.classList.toggle('active', b.dataset.v === String(v)));
    const s = w.querySelector('select'); if (s) s.value = String(v);
    const r = w.querySelector('input[type=range]');
    if (r) { const vals = JSON.parse(w.dataset.values); const i = vals.findIndex(x => String(x) === String(v));
      if (i >= 0) r.value = i; const lab = w.querySelector('.val'); if (lab) lab.textContent = w.dataset.fmt === 'int' ? Number(v).toLocaleString() : String(v); }
  });
}
function initControls() {
  document.querySelectorAll('.ctl[data-ctl]').forEach(w => {
    const tab = w.dataset.tab, name = w.dataset.ctl;
    STATE[tab] = STATE[tab] || {};
    const vals = JSON.parse(w.dataset.values);
    let v = w.dataset.default; if (v === undefined || v === '') v = vals[0];
    // coerce to the type of the listed values
    const match = vals.find(x => String(x) === String(v)); if (match !== undefined) v = match;
    if (!(name in STATE[tab])) STATE[tab][name] = v;
    w.querySelectorAll('.seg button').forEach(b => { b.onclick = () => setControl(tab, name, coerce(vals, b.dataset.v)); });
    const s = w.querySelector('select'); if (s) s.onchange = () => setControl(tab, name, coerce(vals, s.value));
    const r = w.querySelector('input[type=range]');
    if (r) { r.max = vals.length - 1; r.oninput = () => setControl(tab, name, vals[Number(r.value)]); }
    const p = w.querySelector('.play');
    if (p) { let timer = null;
      p.onclick = () => { if (timer) { clearInterval(timer); timer = null; p.textContent = '▶'; return; }
        p.textContent = '❚❚';
        timer = setInterval(() => { const i = vals.findIndex(x => String(x) === String(STATE[tab][name]));
          setControl(tab, name, vals[(i + 1) % vals.length]); }, Number(w.dataset.ms || 450)); }; }
    syncControl(tab, name);
  });
}
function coerce(vals, s) { const m = vals.find(x => String(x) === String(s)); return m === undefined ? s : m; }
// Tabs: #tab=<key> in the URL wins, then localStorage, then the first tab.
const TABS = Array.from(document.querySelectorAll('nav.tabs button')).map(b => b.dataset.tab);
function setTab(key, push) {
  if (!TABS.includes(key)) key = TABS[0];
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === key));
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.dataset.tab === key));
  try { localStorage.setItem(NB_KEY, key); } catch (e) {}
  if (push !== false) { try { history.replaceState(null, '', '#tab=' + key); } catch (e) {} }
  drawTab(key);
  if (window.MathJax && MathJax.typesetPromise) {
    const sec = document.querySelector('section.tab[data-tab="' + key + '"]');
    if (sec && !sec.dataset.typeset) { sec.dataset.typeset = '1'; MathJax.typesetPromise([sec]).catch(e => console.error(e)); }
  }
  window.scrollTo({top: 0});
}
function initialTab() {
  const m = location.hash.match(/tab=([\w-]+)/); if (m && TABS.includes(m[1])) return m[1];
  try { const s = localStorage.getItem(NB_KEY); if (s && TABS.includes(s)) return s; } catch (e) {}
  return TABS[0];
}
window.addEventListener('DOMContentLoaded', () => {
  initControls();
  document.querySelectorAll('nav.tabs button').forEach(b => { b.onclick = () => setTab(b.dataset.tab); });
  setTab(initialTab(), false);
});
window.addEventListener('hashchange', () => { const m = location.hash.match(/tab=([\w-]+)/); if (m) setTab(m[1], false); });
"""

# ------------------------------------------------------------------ controls --


@dataclass
class Control:
    """A control whose value lives in the tab's shared state as ``c.<name>``."""

    name: str
    label: str
    values: list[Any]
    default: Any = None
    labels: dict[Any, str] | None = None
    kind: str = "radio"  # radio | select | slider
    play_ms: int = 450

    def html(self, tab: str) -> str:
        default = self.values[0] if self.default is None else self.default
        attrs = (
            f'class="ctl" data-tab="{tab}" data-ctl="{_esc(self.name)}" '
            f"data-values='{_esc(json.dumps(self.values))}' data-default=\"{_esc(str(default))}\""
        )
        lab = lambda v: _esc(str(self.labels.get(v, v) if self.labels else v))  # noqa: E731
        if self.kind == "radio":
            buttons = "".join(
                f'<button data-v="{_esc(str(v))}">{lab(v)}</button>' for v in self.values
            )
            body = f'<span class="seg">{buttons}</span>'
        elif self.kind == "select":
            opts = "".join(f'<option value="{_esc(str(v))}">{lab(v)}</option>' for v in self.values)
            body = f"<select>{opts}</select>"
        elif self.kind == "slider":
            is_int = all(isinstance(v, int) for v in self.values)
            attrs += f' data-ms="{self.play_ms}" data-fmt="{"int" if is_int else "raw"}"'
            body = (
                '<button class="play" title="play">&#9654;</button>'
                '<input type="range" min="0" max="1" value="0"><span class="val"></span>'
            )
        else:
            raise ValueError(f"unknown control kind {self.kind!r}")
        return f"<span {attrs}><label>{_esc(self.label)}</label>{body}</span>"


def Radio(name, label, values, default=None, labels=None):
    return Control(name, label, list(values), default, labels, "radio")


def Select(name, label, values, default=None, labels=None):
    return Control(name, label, list(values), default, labels, "select")


def Slider(name, label, values, default=None, play_ms=450):
    return Control(name, label, list(values), default, None, "slider", play_ms)


# ---------------------------------------------------------------------- tabs --


@dataclass
class Tab:
    key: str
    label: str
    notebook: Notebook
    blocks: list[str] = field(default_factory=list)

    # -- prose ---------------------------------------------------------------
    def h(self, text: str, level: int = 2) -> Tab:
        self.blocks.append(f"<h{level}>{text}</h{level}>")
        return self

    def prose(self, html: str) -> Tab:
        """Raw HTML paragraphs. Inline math as \\(...\\) or $...$; display as $$...$$."""
        self.blocks.append(html)
        return self

    def bridge(self, html: str, label: str = "From the project to this experiment") -> Tab:
        self.blocks.append(f'<div class="box bridge"><div class="label">{label}</div>{html}</div>')
        return self

    def takeaway(self, html: str, label: str = "Takeaway (measured)") -> Tab:
        self.blocks.append(
            f'<div class="box takeaway"><div class="label">{label}</div>{html}</div>'
        )
        return self

    def interpretation(self, html: str, label: str = "Interpretation") -> Tab:
        """The one place hypothesis vocabulary is allowed (measurement/interpretation firewall)."""
        self.blocks.append(f'<div class="box interp"><div class="label">{label}</div>{html}</div>')
        return self

    def verdict(self, html: str, state: str = "", label: str = "Verdict") -> Tab:
        """state: 'pass' | 'fail' | '' (partial / pending)."""
        self.blocks.append(
            f'<div class="box verdict {state}"><div class="label">{label}</div>{html}</div>'
        )
        return self

    def box(self, html: str) -> Tab:
        self.blocks.append(f'<div class="box">{html}</div>')
        return self

    def equation(self, tex: str, where: str = "") -> Tab:
        where_html = f'<div class="where">{where}</div>' if where else ""
        self.blocks.append(f'<div class="eq">$$ {tex} $$</div>{where_html}')
        return self

    def details(self, summary: str, html: str) -> Tab:
        self.blocks.append(f"<details><summary>{summary}</summary>{html}</details>")
        return self

    def cols(self, *columns: str) -> Tab:
        """Side-by-side HTML columns (collapse to one column when narrow)."""
        self.blocks.append(
            '<div class="cols">' + "".join(f"<div>{c}</div>" for c in columns) + "</div>"
        )
        return self

    def table(
        self, header: list[str], rows: list[list[Any]], numeric: set[int] | None = None
    ) -> Tab:
        self.blocks.append(table_html(header, rows, numeric))
        return self

    # -- controls and figures -----------------------------------------------
    def controls(self, controls: list[Control]) -> Tab:
        """Tab-level controls: every figure in the tab redraws when one changes."""
        self.blocks.append(
            '<div class="controls">' + "".join(c.html(self.key) for c in controls) + "</div>"
        )
        return self

    def figure(
        self,
        id: str,
        title: str,
        *,
        data: Any,
        draw: str,
        subtitle: str = "",
        caption: str = "",
        controls: list[Control] | None = None,
        height: int | None = None,
        mockup: bool = False,
    ) -> Tab:
        """A figure that draws in the browser. ``draw`` is a JS function ``(el, d, c) => ...``."""
        self.notebook._register(id, self.key, data, draw)
        ctl = (
            '<div class="controls">' + "".join(c.html(self.key) for c in controls) + "</div>"
            if controls
            else ""
        )
        sub = f'<p class="fsub">{subtitle}</p>' if subtitle else ""
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        style = f' style="height:{height}px"' if height else ""
        cls = "fig mockup" if mockup else "fig"
        self.blocks.append(
            f'<figure class="{cls}" id="fig-{id}"><div class="ftitle">{title}</div>{sub}{ctl}'
            f'<div class="plot" id="plot-{id}"{style}></div>{cap}</figure>'
        )
        return self

    def grid(self, *blocks_html: str) -> Tab:
        """Lay several already-rendered blocks side by side; pass strings from ``render_*``."""
        self.blocks.append('<div class="grid">' + "".join(blocks_html) + "</div>")
        return self

    def image(
        self,
        path: str | Path,
        caption: str = "",
        *,
        credit: str = "",
        width: int | None = None,
        title: str = "",
    ) -> Tab:
        self.blocks.append(image_html(path, caption, credit=credit, width=width, title=title))
        return self

    def svg(self, markup: str, caption: str = "", title: str = "") -> Tab:
        t = f'<div class="ftitle">{title}</div>' if title else ""
        cap = f"<figcaption>{caption}</figcaption>" if caption else ""
        self.blocks.append(f'<figure class="img">{t}{markup}{cap}</figure>')
        return self

    def methodology(
        self,
        rows: list[dict[str, str]],
        intro: str = "",
        title: str = "Methodology and design decisions",
    ) -> Tab:
        """rows: dicts with keys decision, default, alternatives, test, status (free text)."""
        body = [f"<h3>{title}</h3>"]
        if intro:
            body.append(intro)
        trs = []
        for r in rows:
            st = r.get("status", "")
            cls = (
                "done"
                if any(k in st.lower() for k in ("ablated", "adopted", "done", "measured"))
                else "open"
            )
            trs.append(
                "<tr>"
                + "".join(
                    f"<td>{r.get(k, '')}</td>"
                    for k in ("decision", "default", "alternatives", "test")
                )
                + f'<td><span class="status {cls}">{st}</span></td></tr>'
            )
        body.append(
            '<table class="doc"><tr><th>Decision</th><th>Default chosen</th><th>Alternatives</th>'
            "<th>How to test</th><th>Status</th></tr>" + "".join(trs) + "</table>"
        )
        self.blocks.append("".join(body))
        return self

    def html(self) -> str:
        return "\n".join(self.blocks)


# ------------------------------------------------------------------ notebook --


class Notebook:
    def __init__(
        self,
        title: str,
        subtitle: str = "",
        *,
        status: str = "",
        kicker: str = "",
        footer: str = "",
    ):
        self.title, self.subtitle, self.status, self.kicker, self.footer = (
            title,
            subtitle,
            status,
            kicker,
            footer,
        )
        self.tabs: list[Tab] = []
        self._banner: str = ""
        self._data: dict[str, Any] = {}
        self._figs: list[tuple[str, str, str]] = []
        self._scripts: list[str] = []
        self._styles: list[str] = []

    def tab(self, key: str, label: str) -> Tab:
        if any(t.key == key for t in self.tabs):
            raise ValueError(f"duplicate tab key {key!r}")
        t = Tab(key, label, self)
        self.tabs.append(t)
        return t

    def banner(self, html: str, label: str = "Substrates") -> Notebook:
        """A strip between the title and the tabs, visible on every tab: the architecture
        diagrams (``arch_svg``) and anything else the reader must know before any figure."""
        self._banner = f'<div class="banner"><div class="label">{_esc(label)}</div>{html}</div>'
        return self

    def dataset(self, name: str, data: Any) -> str:
        """Store a payload once and share it between figures: pass the returned ``"@name"`` as ``data``."""
        if name in self._data:
            raise ValueError(f"duplicate dataset {name!r}")
        self._data[name] = data
        return "@" + name

    def _register(self, id: str, tab: str, data: Any, draw: str) -> None:
        if id in self._data:
            raise ValueError(f"duplicate figure id {id!r}")
        if isinstance(data, str) and data.startswith("@") and data[1:] not in self._data:
            raise ValueError(f"unknown dataset {data!r}; call Notebook.dataset first")
        self._data[id] = data
        self._figs.append((id, tab, draw))

    def script(self, js: str) -> Notebook:
        """Extra JS helpers available to every draw function (e.g. a dequantiser)."""
        self._scripts.append(js)
        return self

    def style(self, css: str) -> Notebook:
        self._styles.append(css)
        return self

    def render(self) -> str:
        nav = "".join(
            f'<button data-tab="{t.key}"><span class="idx">{i}</span>{_esc(t.label)}</button>'
            for i, t in enumerate(self.tabs)
        )
        sections = "".join(
            f'<section class="tab" data-tab="{t.key}">{t.html()}</section>' for t in self.tabs
        )
        figs = "\n".join(
            f"FIGS[{json.dumps(i)}] = {{tab: {json.dumps(t)}, draw: ({d})}};"
            for i, t, d in self._figs
        )
        data = json.dumps(self._data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
        pills = "".join(
            f'<span class="pill{" warn" if "MOCKUP" in s else ""}">{_esc(s)}</span>'
            for s in _split_status(self.status)
        )
        kicker = f'<div class="kicker">{_esc(self.kicker)}</div>' if self.kicker else ""
        built = _dt.date.today().isoformat()
        foot = (
            self.footer
            or f"Built {built}. Data is embedded in this file; figures draw in the browser (Plotly), math by MathJax."
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(self.title)}</title>
<script>window.MathJax = {{tex: {{inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
  processEscapes: true}}, options: {{skipHtmlTags: ['script','noscript','style','textarea','pre','code']}},
  startup: {{typeset: false}}}};</script>
<script src="{MATHJAX_CDN}" async></script>
<script src="{PLOTLY_CDN}"></script>
<style>{STYLE}{"".join(self._styles)}</style>
</head><body><main>
<header class="hero">{kicker}<h1>{_esc(self.title)}</h1><p class="subtitle">{self.subtitle}</p>{pills}</header>
{self._banner}
<nav class="tabs">{nav}</nav>
{sections}
<footer>{foot}</footer>
</main>
<script type="application/json" id="nb-data">{data}</script>
<script>
const NB_KEY = {json.dumps("nb-tab:" + self.title)};
{PRELUDE}
{"".join(self._scripts)}
{figs}
</script>
</body></html>"""

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self.render()
        path.write_text(html, encoding="utf-8")
        size = len(html.encode("utf-8")) / 1e6
        print(f"wrote {path} ({size:.1f} MB, {len(self.tabs)} tabs, {len(self._figs)} figures)")
        if size > 25:
            print("  warning: over 25 MB; browsers get sluggish. Subsample frames or fields.")
        return path


# ------------------------------------------------------------------- helpers --


def _esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def _split_status(s: str) -> list[str]:
    return [p.strip() for p in s.split("|") if p.strip()] if s else []


def table_html(header: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if j in numeric else ''}>{c}</td>" for j, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return f'<table class="doc"><tr>{head}</tr>{body}</table>'


def image_html(
    path: str | Path,
    caption: str = "",
    *,
    credit: str = "",
    width: int | None = None,
    title: str = "",
    mockup: bool = False,
) -> str:
    """A base64-embedded image card (png/jpg/gif/svg). Use for bridge figures, gifs, paper crops."""
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    if len(b64) > 2_800_000:
        print(f"  warning: {p.name} embeds at {len(b64) / 1e6:.1f} MB; shrink it (see figures.md)")
    style = f' style="max-width:{width}px"' if width else ""
    t = f'<div class="ftitle">{title}</div>' if title else ""
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    cr = f'<div class="credit">{credit}</div>' if credit else ""
    cls = "img mockup" if mockup or "MOCKUP" in p.name.upper() else "img"
    return f'<figure class="{cls}">{t}<img src="data:{mime};base64,{b64}" alt="{_esc(caption or p.name)}"{style}>{cap}{cr}</figure>'


def arch_svg(
    blocks: list[tuple[str, str, str]],
    *,
    skips: list[tuple[int, int]] | None = None,
    cuts: dict[int, str] | None = None,
    title: str = "",
    caption: str = "",
) -> str:
    """A block diagram of a network, as an inline SVG figure for ``Notebook.banner``.

    ``blocks``: ``(name, width, kind)`` left to right, kind in ``in | linear |
    act | norm | attn | mlp | out | other`` (colours only). ``skips``: identity
    paths from the output of block i to the output of block j (drawn as arcs
    above). ``cuts``: ``{i: label}`` marks the activation after block i (the
    places instruments read from). Widths are printed inside the blocks so a
    reader sees depth and width at a glance. Pure text, no dependencies.
    """

    skips = skips or []
    cuts = cuts or {}
    bw, bh, gap, x0, y0 = 92, 44, 26, 12, 40
    fill = {
        "in": "#eef2f7",
        "linear": "#dfe8f3",
        "act": "#f6e7d8",
        "norm": "#ece4f2",
        "attn": "#e2f0e6",
        "mlp": "#dfe8f3",
        "out": "#eef2f7",
        "other": "#f1efe9",
    }
    n = len(blocks)
    width = x0 * 2 + n * bw + (n - 1) * gap
    height = y0 + bh + 34
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" style="max-width:100%;'
        'font-family:system-ui;font-size:11px" role="img">',
        '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
        '<path d="M0,0 L7,3.5 L0,7 z" fill="#5a5650"/></marker></defs>',
    ]
    if title:
        parts.append(
            f'<text x="{x0}" y="14" font-weight="700" font-size="12.5">{_esc(title)}</text>'
        )
    for i, (name, w, kind) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        parts.append(
            f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="6" fill="{fill.get(kind, fill["other"])}" '
            'stroke="#8d877c" stroke-width="1"/>'
            f'<text x="{x + bw / 2}" y="{y0 + 18}" text-anchor="middle" font-weight="650">{_esc(name)}</text>'
            f'<text x="{x + bw / 2}" y="{y0 + 34}" text-anchor="middle" fill="#5a5650">{_esc(w)}</text>'
        )
        if i < n - 1:
            parts.append(
                f'<line x1="{x + bw + 2}" y1="{y0 + bh / 2}" x2="{x + bw + gap - 3}" y2="{y0 + bh / 2}" '
                'stroke="#5a5650" stroke-width="1.4" marker-end="url(#ah)"/>'
            )
        if i in cuts:
            cx = x + bw + gap / 2
            parts.append(
                f'<line x1="{cx}" y1="{y0 + bh + 2}" x2="{cx}" y2="{y0 + bh + 12}" stroke="#b04a4a" stroke-width="1.6"/>'
                f'<text x="{cx}" y="{y0 + bh + 24}" text-anchor="middle" fill="#b04a4a" font-weight="650">{_esc(cuts[i])}</text>'
            )
    for i, j in skips:
        xa = x0 + i * (bw + gap) + bw + gap / 2
        xb = x0 + j * (bw + gap) + bw + gap / 2
        parts.append(
            f'<path d="M{xa},{y0} C{xa},{y0 - 26} {xb},{y0 - 26} {xb},{y0 - 2}" fill="none" stroke="#3c8a5a" '
            'stroke-width="1.6" marker-end="url(#ah)"/>'
        )
    parts.append("</svg>")
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="arch">{"".join(parts)}{cap}</figure>'


def render_figure(
    id: str, title: str, *, subtitle: str = "", caption: str = "", height: int | None = None
) -> str:
    """The HTML shell of a figure for use inside ``Tab.grid``; register data/draw with ``Tab.figure``
    is not possible then, so use ``Notebook.add_figure`` instead."""
    sub = f'<p class="fsub">{subtitle}</p>' if subtitle else ""
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    style = f' style="height:{height}px"' if height else ""
    return f'<figure class="fig" id="fig-{id}"><div class="ftitle">{title}</div>{sub}<div class="plot" id="plot-{id}"{style}></div>{cap}</figure>'


def add_figure(nb: Notebook, tab: Tab, id: str, data: Any, draw: str) -> None:
    """Register a figure whose shell was placed with ``render_figure`` (for grids)."""
    nb._register(id, tab.key, data, draw)


def clean(x: Any) -> Any:
    """Recursively turn numpy / NaN / inf into JSON-safe values (NaN -> None)."""
    import math

    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "tolist"):
        return clean(x.tolist())
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if hasattr(x, "item"):
        return clean(x.item())
    return x


__all__ = [
    "Control",
    "Notebook",
    "Radio",
    "Select",
    "Slider",
    "Tab",
    "add_figure",
    "arch_svg",
    "clean",
    "image_html",
    "render_figure",
    "table_html",
]
