// COPY of explore/lm_coherent/page/js/main.js (verdict pills also map pass / fail / not run).
// Page shell (after explore/cifar_coherent/site/js/main.js): hash router, maths, heading anchors, side notes, live
// figures drawn when they scroll into view, and the explorer and zoom views built on first visit.
import { D, loadData } from "./data.js";
import { FIGS } from "./figs.js";
import { Explorer } from "./explorer.js";
import { Zoom } from "./zoom.js";

const TABS = ["overview", "explorer", "zoom", "findings", "measurements", "method"];
let explorer = null, zoom = null, currentTab = null;
const panelOf = tab => document.getElementById(`tab-${tab}`);

function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi);
  const query = new URLSearchParams(qi < 0 ? "" : raw.slice(qi + 1));
  let [tab, id] = path.split("/");
  if (!TABS.includes(tab)) {
    const e = tab && document.getElementById(tab), p = e && e.closest(".tab-panel");
    return { tab: p ? p.dataset.tab : "overview", anchor: p ? e : null, query };
  }
  const anchor = id ? panelOf(tab)?.querySelector(`#${CSS.escape(id)}`) : null;
  return { tab, anchor, query };
}

async function route() {
  const { tab, anchor, query } = resolveHash();
  const changed = tab !== currentTab;
  currentTab = tab;
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.dataset.tab !== tab);
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  await loadData();
  if (tab === "explorer") {
    if (!explorer) explorer = window.__ex = new Explorer(document.getElementById("ex"));
    explorer.layout();
    if ([...query.keys()].length) explorer.applyParams(query);
  }
  if (tab === "zoom") {
    if (!zoom) zoom = window.__zoom = new Zoom(document.getElementById("zoom"));
    else zoom.render();
    if ([...query.keys()].length) zoom.applyParams(query);
  }
  if (anchor) requestAnimationFrame(() => anchor.scrollIntoView({ block: "start" }));
  else if (changed) window.scrollTo(0, 0);
  if (tab === "zoom" && [...query.keys()].length) requestAnimationFrame(() => document.getElementById("zoom").scrollIntoView({ block: "start" }));
  observeLive();
}

function maths() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["no-math", "tooltip", "snip"], ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"], throwOnError: false,
  });
}
function anchors() {
  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(hd => {
    const tab = hd.closest(".tab-panel").dataset.tab;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${tab}/${hd.id}`; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    hd.prepend(a);
  });
}
function sidenotes() {
  document.querySelectorAll(".tab-panel").forEach(panel => {
    let n = 0;
    panel.querySelectorAll(".sidenote").forEach(sn => {
      n += 1;
      const b = document.createElement("button"); b.type = "button"; b.className = "sn-ref"; b.textContent = n; b.setAttribute("aria-label", `Side note ${n}`);
      const num = document.createElement("span"); num.className = "sn-num"; num.textContent = n; sn.prepend(num);
      sn.before(b);
      b.addEventListener("click", () => sn.classList.toggle("open"));
    });
  });
}

// live figures: drawn when they scroll into view, redrawn on resize
let io = null;
const drawn = new Set();
function observeLive() {
  if (!io) io = new IntersectionObserver(entries => entries.forEach(en => { if (en.isIntersecting) { io.unobserve(en.target); renderLive(en.target); } }), { rootMargin: "400px" });
  document.querySelectorAll(".tab-panel:not([hidden]) figure.live:not([data-done])").forEach(f => { f.dataset.done = "1"; io.observe(f); });
}
function renderLive(fig) {
  let body = fig.querySelector(":scope > .live-body");
  if (!body) { body = document.createElement("div"); body.className = "live-body"; fig.insertBefore(body, fig.querySelector("figcaption")); }
  try {
    const f = FIGS[fig.dataset.live];
    if (!f) throw new Error("unknown figure");
    f(body, fig.dataset);
    drawn.add(fig);
  } catch (e) {
    console.error(e);
    body.replaceChildren(Object.assign(document.createElement("div"), { className: "placeholder", textContent: `Figure "${fig.dataset.live}" failed: ${e.message}` }));
  }
}
let lastW = window.innerWidth;
window.addEventListener("resize", () => {
  clearTimeout(window.__rz);
  window.__rz = setTimeout(() => { if (Math.abs(window.innerWidth - lastW) < 40) return; lastW = window.innerWidth; drawn.forEach(f => { if (f.offsetParent) renderLive(f); }); }, 200);
});

function pills() {   // verdict statuses as pills
  document.querySelectorAll("table.verdict td:last-child").forEach(td => {
    const t = td.textContent.trim(), cls = { "met": "ok", "not met": "bad", "not determinable": "na", "pass": "ok", "fail": "bad", "not run": "na" }[t];
    if (cls) td.innerHTML = `<span class="pill ${cls}">${t}</span>`;
  });
}
function init() {
  anchors(); sidenotes(); pills();
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  window.addEventListener("hashchange", route);
  route().catch(e => { console.error(e); document.getElementById("load-error").hidden = false; document.getElementById("load-error").textContent = "The embedded data could not be decoded: " + e.message; });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
