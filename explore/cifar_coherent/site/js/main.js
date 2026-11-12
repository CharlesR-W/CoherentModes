// Page shell: hash router, maths, gifs, side notes, heading anchors, lazy live figures.
const TABS = ["overview", "explorer", "findings", "measurements", "method"];
const TAB_ALIAS = { questions: "measurements" };          // old tab names (the site is published)
// Old #<tab>/<id> whose section moved. Applied only when <id> is no longer in <tab>, so an old URL keeps
// working while the content is being moved; a target id that does not exist (yet) lands on its tab's top.
const ANCHOR_ALIAS = {
  "overview/treated-alike": "overview/procedure",
  "overview/blocks": "overview/step-3",
  "overview/through-training": "overview/step-3",
  "overview/hankel": "method/avgprod",
  "overview/kernel": "method/kernel",
  "overview/reading": "method/reading",
  "overview/panels": "method/loadings",
  "overview/limits": "method/checks",
  "method/limits": "method/checks",
  "method/glossary-cell": "method/glossary-snapshot",                       // "cell" became "snapshot"
  "method/glossary-single-cell-block": "method/glossary-single-snapshot-block",
};
let explorer = null, current = null;

const panelOf = tab => document.getElementById(`tab-${tab}`);
const inTab = (tab, id) => (id && panelOf(tab)?.querySelector(`#${CSS.escape(id)}`)) || null;
function findTabOf(id) { const e = document.getElementById(id); const p = e && e.closest(".tab-panel"); return p ? p.dataset.tab : null; }

// -> {tab, anchor (an element or null), query, canon (the hash to show, or null to leave the URL alone)}
function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi);
  let [tab, id] = path.split("/");
  let moved = false;
  if (TAB_ALIAS[tab]) { tab = TAB_ALIAS[tab]; moved = true; }
  if (!TABS.includes(tab)) {
    // a bare #id (no tab): find the tab that holds it; anything else goes to the Overview
    const t = tab && findTabOf(tab);
    return { tab: t || "overview", anchor: t ? document.getElementById(tab) : null, query: new URLSearchParams(qs.slice(1)), canon: null };
  }
  if (id && !inTab(tab, id) && ANCHOR_ALIAS[`${tab}/${id}`]) { [tab, id] = ANCHOR_ALIAS[`${tab}/${id}`].split("/"); moved = true; }
  const anchor = inTab(tab, id);
  return { tab, anchor, query: new URLSearchParams(qs.slice(1)), canon: moved ? `#${tab}${id ? "/" + id : ""}${qs}` : null };
}

async function route() {
  const { tab, anchor, query, canon } = resolveHash();
  if (canon && canon !== location.hash) history.replaceState(history.state, "", canon);
  const wanted = location.hash;
  const changed = tab !== current;
  current = tab;
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.dataset.tab !== tab);
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  if (tab === "explorer") {
    if (!explorer) {
      const { Explorer } = await import("./explorer.js");
      explorer = window.__ex = new Explorer(document.getElementById("ex"));
      await explorer.ready;
    }
    if ([...query.keys()].length && wanted !== explorer.lastHash) explorer.applyParams(query);
    else explorer.layout();
  }
  if (anchor) requestAnimationFrame(() => anchor.scrollIntoView({ block: "start" }));
  else if (changed || tab !== "explorer") window.scrollTo(0, 0);   // no anchor, or an unknown one: the tab's top
  observeLive();
}

window.addEventListener("hashchange", route);
document.addEventListener("click", e => {
  // explorer deep links from prose: let the router apply them even if the hash is equal
  const a = e.target.closest("a.explore-link");
  if (a && a.getAttribute("href") === location.hash) { e.preventDefault(); route(); }
});

function maths() {
  if (!window.renderMathInElement) return;
  window.renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["no-math", "tooltip"], throwOnError: false,
  });
}

function anchors() {
  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(hd => {
    const tab = hd.closest(".tab-panel").dataset.tab;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${tab}/${hd.id}`; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    hd.prepend(a);
  });
  // internal links of the form #id (no tab) still work: route() finds the tab
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

function gifs() {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelectorAll("img.gif").forEach(img => {
    const still = img.dataset.still, anim = img.getAttribute("src");
    const wrap = document.createElement("div"); wrap.className = "gif-wrap"; img.before(wrap); wrap.append(img);
    const btn = document.createElement("button"); btn.type = "button"; btn.className = "gif-btn"; wrap.append(btn);
    let playing = !(reduce && still);
    const set = () => {
      btn.innerHTML = playing ? "&#10074;&#10074; pause" : "&#9654; play";
      btn.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
      if (playing) img.src = anim + (img._restarted ? "" : ""); else if (still) img.src = still;
    };
    btn.addEventListener("click", () => { playing = !playing; if (playing) { img._restarted = true; img.src = anim + "?t=" + Date.now(); } set(); });
    if (!still) btn.hidden = true;
    set();
  });
}

// ---------- live figures: rendered when they scroll into view
let io = null;
function observeLive() {
  if (!io) io = new IntersectionObserver(entries => entries.forEach(en => { if (en.isIntersecting) { io.unobserve(en.target); renderLive(en.target); } }), { rootMargin: "300px" });
  document.querySelectorAll(".tab-panel:not([hidden]) figure.live:not([data-done])").forEach(f => { f.dataset.done = "1"; io.observe(f); });
}
async function renderLive(fig) {
  const body = document.createElement("div"); body.className = "live-body";
  const loading = document.createElement("div"); loading.className = "live-loading"; loading.textContent = "loading…"; body.append(loading);
  fig.insertBefore(body, fig.querySelector("figcaption"));
  try {
    const live = await import("./live.js");
    await live.render(fig.dataset.live, body, fig.dataset);
  } catch (e) {
    console.error(e);
    body.replaceChildren(Object.assign(document.createElement("div"), { className: "placeholder", textContent: `Live figure "${fig.dataset.live}" failed: ${e.message}` }));
  }
}

function init() {
  anchors(); sidenotes(); gifs();
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  route();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
