// Page shell: hash router, maths, gifs, side notes, heading anchors, lazy live figures.
const TABS = ["overview", "primer", "explorer", "experiments", "method", "appendices",
              "choices", "baselines", "method-details", "measurements", "reproduce"];
// Appendix panels: reached from the "Appendices" tab, which stays highlighted while one of them is open.
const APPENDICES = ["choices", "baselines", "method-details", "measurements", "reproduce"];
// Old tab names (the site is published). Findings became Experiments and the Alignment appendix was folded into
// the experiments and appendices (2026-11-19, EXPERIMENTS_PLAN.md).
const TAB_ALIAS = { questions: "measurements", findings: "experiments", alignment: "experiments" };
// Old #<tab>/<id> whose section moved, keyed by the tab as written in the URL (before TAB_ALIAS). Applied only when
// <id> is no longer in <tab>, so an old URL keeps working while the content is being moved; a target id that does not
// exist (yet) lands on its tab's top. A value without "/<id>" is the tab's top.
const ANCHOR_ALIAS = {
  "overview/treated-alike": "overview/procedure",
  "overview/blocks": "overview/step-3",
  "overview/through-training": "overview/step-3",
  "overview/hankel": "choices/avgprod",
  "method-details/reproduce": "reproduce",   // Appendix C's reproduce section became Appendix E
  "overview/kernel": "method/kernel",
  "overview/reading": "method/reading",
  "overview/panels": "method/reading",   // the loadings section was removed
  "overview/limits": "experiments",      // "Checks I'd like to run" went into each experiment's open checks
  "method/limits": "experiments",
  "method/checks": "experiments",
  "method/glossary-cell": "method/glossary-snapshot",                       // "cell" became "snapshot"
  "method/glossary-single-cell-block": "method/glossary-single-snapshot-block",
  // Findings -> Experiments (2026-11-19)
  "findings/f-end-sets": "experiments/x-groups",
  "findings/f-looking-back": "experiments/x-when",
  "findings/f-depth": "experiments/x-depth",
  "findings/f-block-vs-cell": "experiments/x-blocks",
  "findings/f-gradients": "experiments/x-training-signal",
  "findings/f-coarse": "experiments/x-coarse",
  "findings/f-markov": "experiments/x-markov",
  "findings/qualitative": "experiments",
  "findings/next-checks": "experiments",
  "findings/read-a-finding": "experiments",
  // Appendix A Measurements -> Experiments
  "measurements/when": "experiments/x-when",
  "measurements/va": "experiments/x-when",
  "measurements/carry": "experiments/x-depth",
  "measurements/imgstats": "experiments/x-image-stats",
  "measurements/gradients": "experiments/x-training-signal",
  // Appendix D Alignment -> Experiments and appendices (other alignment ids follow their content by the generic rule)
  "alignment/w1": "measurements/ref-depth-w1",
  "alignment/w2": "experiments/x-when",
  "alignment/w3": "choices/w3",
  "alignment/w4": "baselines/w4",
  "alignment/iso": "baselines/iso",
  "alignment/c1": "experiments/x-across-nets",
  "alignment/c2": "measurements/ref-across-nets-c2",
  "alignment/c3": "measurements/ref-across-nets-c3",
  "alignment/c4": "measurements/ref-across-nets-c4",
  "alignment/c5": "measurements/ref-across-nets-c5",
  "alignment/modes": "measurements/ref-across-nets-modes",
};
// The Depth embedding tab became its own site (2026-11-21); old #embedding?<state> links go there with their state.
const EMBEDDING_SITE = "https://charlesr-w.github.io/DepthEmbedding/";
let explorer = null, current = null;

const panelOf = tab => document.getElementById(`tab-${tab}`);
const inTab = (tab, id) => (id && panelOf(tab)?.querySelector(`#${CSS.escape(id)}`)) || null;
function findTabOf(id) { const e = document.getElementById(id); const p = e && e.closest(".tab-panel"); return p ? p.dataset.tab : null; }

// -> {tab, anchor (an element or null), query, canon (the hash to show, or null to leave the URL alone)}
function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi);
  let [tab, id] = path.split("/");
  if (tab === "embedding") { location.replace(EMBEDDING_SITE + "#embedding" + qs); tab = "overview"; id = undefined; }
  let moved = false;
  // anchor aliases are looked up under the tab as written (e.g. findings/f-markov) before the tab alias applies
  const old = id && ANCHOR_ALIAS[`${tab}/${id}`];
  if (old && !(TABS.includes(tab) && inTab(tab, id))) { [tab, id] = old.split("/"); moved = true; }
  else if (TAB_ALIAS[tab]) { tab = TAB_ALIAS[tab]; moved = true; }
  if (!TABS.includes(tab)) {
    // a bare #id (no tab): find the tab that holds it, else an alias row for that id (e.g. #f-markov);
    // anything else goes to the Overview
    const t = tab && findTabOf(tab);
    const al = !t && tab && Object.entries(ANCHOR_ALIAS).find(([k]) => k.endsWith(`/${tab}`));
    if (al) { const [t2, id2] = al[1].split("/"); return { tab: t2, anchor: inTab(t2, id2), query: new URLSearchParams(qs.slice(1)), canon: `#${al[1]}${qs}` }; }
    return { tab: t || "overview", anchor: t ? document.getElementById(tab) : null, query: new URLSearchParams(qs.slice(1)), canon: null };
  }
  // a section moved to another tab (e.g. Method -> Appendix C, 2026-11-19): follow its id
  if (id && !inTab(tab, id)) { const t = findTabOf(id); if (t && TABS.includes(t)) { tab = t; moved = true; } }
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
  const shown = APPENDICES.includes(tab) ? "appendices" : tab;
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === shown)));
  document.querySelectorAll(".appx-nav a").forEach(a => a.setAttribute("aria-current", a.dataset.tab === tab ? "page" : "false"));
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
  requestAnimationFrame(markScrolls);
}

// tables wider than their box get a "scroll sideways" cue (css .table-scroll.scrolls; as in explore/ncs_report)
function markScrolls() {
  document.querySelectorAll(".tab-panel:not([hidden]) .table-scroll, .tab-panel:not([hidden]) .bn-fig").forEach(d => {
    d.classList.add("table-scroll"); d.classList.toggle("scrolls", d.scrollWidth > d.clientWidth + 2); });
}
window.addEventListener("resize", markScrolls);
let scrollsQueued = false;
new MutationObserver(() => {
  if (scrollsQueued || document.body.dataset.tab === "explorer") return;
  scrollsQueued = true; requestAnimationFrame(() => { scrollsQueued = false; markScrolls(); });
}).observe(document.body, { childList: true, subtree: true });

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
  // an experiment's question (its h2) links to the experiment's section id
  document.querySelectorAll(".prose h2[id], .prose h3[id], .prose section.experiment[id] > h2:not([id])").forEach(hd => {
    const tab = hd.closest(".tab-panel").dataset.tab, id = hd.id || hd.parentElement.id;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${tab}/${id}`; a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
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
