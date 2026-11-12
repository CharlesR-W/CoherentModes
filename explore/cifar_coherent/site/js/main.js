// Page shell: hash router, maths, gifs, side notes, heading anchors, lazy live figures.
const TABS = ["overview", "explorer", "questions", "method"];
let explorer = null, current = null;

function parseHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const [path, query] = raw.split("?");
  const [tab, anchor] = path.split("/");
  return { tab: TABS.includes(tab) ? tab : (document.getElementById(tab) ? findTabOf(tab) : "overview"), anchor: TABS.includes(tab) ? anchor : tab, query: new URLSearchParams(query || "") };
}
function findTabOf(id) { const e = document.getElementById(id); const p = e && e.closest(".tab-panel"); return p ? p.dataset.tab : "overview"; }

async function route() {
  const { tab, anchor, query } = parseHash();
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
  if (anchor) {
    const e = document.getElementById(anchor);
    if (e) requestAnimationFrame(() => e.scrollIntoView({ block: "start" }));
  } else if (changed) window.scrollTo(0, 0);
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
