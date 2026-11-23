"""Browser check of the built site (viewer/site/) with playwright-cli (headless Chrome). Writes nothing in the repo
except, with --capture-explorer, the experiment's explorer screenshot.

    python3 site/check_browser.py [--session x-check] [--out /tmp/x-check] [--only ID[,ID]] [--no-aliases] [--no-depth]
                                  [--tabs experiments|all|TAB[,TAB]]   # all = experiments, the appendices,
                                                                       # overview and method
    python3 site/check_browser.py --capture-explorer ID      # (re)take assets/fig/x_<ID>_explorer.png from the built site

What it does:
  1. serves viewer/site/ on 127.0.0.1 (a free port; stopped at the end) and opens one playwright-cli session (-s=NAME;
     use a distinct name per agent; it is closed at the end, also on failure);
  2. opens #experiments, scrolls every figure.live of every experiment section (or of --only) into view, waits for it
     to render and asserts it holds a chart: an svg/canvas in .live-body (lines, heatmap, cells, bn-*), an alignment
     chart (.chart / svg from js/alignment.js) or an image grid (.live-imgs) or a table (bn-ablate, al-c2), and no "failed" placeholder; static
     <img> must have loaded (naturalWidth > 0);
     (--tabs: the same for other tabs, one result per tab, with every <details> opened first);
  3. screenshots each experiment section (and each other --tabs tab, full page) at desktop (1440 x 900) and mobile (390 x 844) widths, full page, into --out
     (look at them: python3 cannot judge a chart);
  4. follows every ANCHOR_ALIAS / TAB_ALIAS row of js/main.js and checks it lands on an existing element of the
     target tab. Targets in experiments that are not in this build's manifest are reported, not failed; aliases whose
     old id still exists in the old tab are dormant by design (reported).
  5. smoke-tests the explorer's "through depth" colouring (DEPTH_JS; skip with --no-depth; reported as skipped when the
     build has no meta.spacetime): checkpoint, variant, coupling, mode and sets switches, the URL round trip, one
     incompatible state, leaving the colouring; screenshots explorer-depth-*.png into --out.
     Only this test: --tabs "" --no-aliases.
Exit status 1 if any check fails. Run from anywhere; needs a previous build_site.py run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
OUT = ROOT / "viewer" / "site"
PW = shutil.which("playwright-cli") or "playwright-cli"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Session:
    def __init__(self, name: str, cwd: Path):
        self.name, self.cwd = name, cwd

    def run(self, *args, check=True, timeout=180) -> str:
        r = subprocess.run([PW, f"-s={self.name}", *args], cwd=self.cwd, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode != 0:
            raise RuntimeError(f"playwright-cli {' '.join(args[:2])} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def code(self, js: str, timeout=300):
        """Run a Playwright snippet `async page => {...}` that returns a JSON string; returns the parsed value."""
        with tempfile.NamedTemporaryFile("w", suffix=".js", dir=self.cwd, delete=False) as f:
            f.write(js)
        try:
            out = self.run("run-code", "--filename", f.name, "--raw", timeout=timeout)
        finally:
            Path(f.name).unlink(missing_ok=True)
        try:
            res = json.loads(out.strip())        # --raw prints the returned string as a JSON literal
            m = re.search(r"@@JSON@@(.*)@@END@@", res, re.S)
        except (ValueError, TypeError):
            m = None
        if not m:
            raise RuntimeError(f"no result from run-code:\n{out[-3000:]}")
        return json.loads(m.group(1))


CHECK_JS = r"""
async page => {
  const only = %ONLY%, tab = %TAB%;
  await page.evaluate(t => { location.hash = "#" + t; }, tab);
  await page.waitForSelector(`#tab-${tab}:not([hidden])`, { timeout: 20000 });
  // live figures inside closed "More views" <details> render only when opened: open them all
  await page.evaluate(t => document.querySelectorAll(`#tab-${t} details`).forEach(d => { d.open = true; }), tab);
  const ids = tab === "experiments"
    ? await page.evaluate(only => [...document.querySelectorAll("#tab-experiments section.experiment")]
        .map(s => s.id).filter(id => !only.length || only.includes(id.replace(/^x-/, ""))), only)
    : [`tab-${tab}`];
  const results = [];
  for (const sid of ids) {
    const n = await page.evaluate(sid => document.querySelectorAll(`#${sid} figure.live`).length, sid);
    const sec = { id: sid, placeholder: await page.evaluate(sid => !!document.querySelector(`#${sid} > .placeholder`), sid), figures: [] };
    for (let i = 0; i < n; i++) {
      const sel = `#${sid} figure.live >> nth=${i}`;
      const fig = page.locator(sel);
      await fig.scrollIntoViewIfNeeded();
      let state = "timeout", detail = "";
      const t0 = Date.now();
      while (Date.now() - t0 < 30000) {
        const r = await fig.evaluate(f => {
          const b = f.querySelector(".live-body");
          if (!b) return ["pending", ""];
          const err = b.querySelector(".placeholder");
          if (err) return ["error", err.textContent];
          if (b.querySelector(".live-loading")) return ["pending", ""];
          const chart = b.querySelector(".live-chart svg, .chart-wrap svg, svg, canvas, .chart, .live-imgs, .imgs, table");   // bn-ablate, al-c2 draw tables
          return chart ? ["ok", chart.tagName.toLowerCase() + "." + (chart.getAttribute("class") || "")] : ["empty", b.innerHTML.slice(0, 120)];
        });
        [state, detail] = r;
        if (state !== "pending") break;
        await page.waitForTimeout(250);
      }
      const head = await fig.evaluate(f => `${f.dataset.live}${f.dataset.src ? " " + f.dataset.src : ""}`);
      sec.figures.push({ fig: head, state, detail });
    }
    sec.images = await page.evaluate(async sid => {
      const imgs = [...document.querySelectorAll(`#${sid} img`)];
      for (const im of imgs) { im.loading = "eager"; if (!im.complete) await new Promise(r => { im.onload = im.onerror = r; setTimeout(r, 8000); }); }
      return imgs.map(im => ({ src: im.getAttribute("src"), ok: im.naturalWidth > 0 }));
    }, sid);
    results.push(sec);
  }
  return "@@JSON@@" + JSON.stringify(results) + "@@END@@";
}
"""

SHOT_JS = r"""
async page => {
  const tab = %TAB%, path = %PATH%, out = [];
  if (tab === "experiments") {
    const ids = await page.evaluate(() => [...document.querySelectorAll("#tab-experiments section.experiment")].map(s => s.id));
    for (const id of ids) {
      const p = path.replace(/\.png$/, `-${id}.png`);
      // full page width (wide figures break out of the prose column), clipped to the section's height
      const box = await page.evaluate(id => { const r = document.getElementById(id).getBoundingClientRect();
        return { y: r.top + window.scrollY, h: r.height, w: document.documentElement.clientWidth }; }, id);
      await page.screenshot({ path: p, fullPage: true, timeout: 180000, animations: "disabled",
                              clip: { x: 0, y: Math.max(0, box.y - 8), width: box.w, height: box.h + 16 } });
      out.push(p);
    }
  } else {
    await page.screenshot({ path, fullPage: true, timeout: 180000, animations: "disabled" });
    out.push(path);
  }
  return "@@JSON@@" + JSON.stringify(out) + "@@END@@";
}
"""

ALIAS_JS = r"""
async page => {
  const rows = %ROWS%;
  const out = [];
  for (const [from, to] of rows) {
    await page.evaluate(h => { location.hash = h; }, "#" + from);
    await page.waitForTimeout(150);
    const r = await page.evaluate(([from, to]) => {
      const [tab, id] = to.split("/");
      const [ftab, fid] = from.split("/");
      const shown = document.body.dataset.tab;
      const el = id ? document.querySelector(`#tab-${tab} #${CSS.escape(id)}`) : document.getElementById(`tab-${tab}`);
      const dormant = !!(fid && document.querySelector(`#tab-${ftab} #${CSS.escape(fid)}`));
      return { from, to, shown, hash: location.hash, exists: !!el, dormant };
    }, [from, to]);
    out.push(r);
  }
  return "@@JSON@@" + JSON.stringify(out) + "@@END@@";
}
"""

EXPLORER_JS = r"""
async page => {
  // the "How to use" panel opens on a first visit only: mark it seen so the shot shows the grid
  await page.evaluate(() => localStorage.setItem("cm.explorer.howto-seen", "1"));
  await page.evaluate(h => { location.hash = h; }, %HASH%);
  await page.waitForSelector("#tab-explorer:not([hidden])", { timeout: 20000 });
  await page.waitForFunction(() => window.__ex && window.__ex.ready, null, { timeout: 60000 });
  await page.evaluate(() => window.__ex.ready);
  await page.waitForFunction(() => !document.querySelector("#ex-status")?.textContent.match(/loading/i), null, { timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.addStyleTag({ content: ".ex-bar { position: static !important; } #ex-howto, #ex-hint, #ex-caption, #ex-tour { display: none !important; }" });   // no sticky bar over the shot
  await page.locator(%SEL%).screenshot({ path: %PATH% });
  const status = await page.evaluate(() => document.querySelector("#ex-status")?.textContent || "");
  return "@@JSON@@" + JSON.stringify({ status }) + "@@END@@";
}
"""

# Smoke test of the explorer's "through depth" colouring (build_site.py spacetime(), js/explorer.js depthColouring):
# open it, switch checkpoint (a real click on a panel), variant, coupling and mode, then sets; check the URL round-trips;
# visit an incompatible state (dot kernel, another rule, a time block, out-of-range coupling and mode, paper with a
# coupling) and check it is fitted and announced; leave the colouring and check the block cards come back. Screenshots
# of the grid and side panel go to --out (explorer-depth-*.png).
DEPTH_JS = r"""
async page => {
  const out = { steps: [], errors: [] }, OUTDIR = %OUT%;
  page.on("pageerror", e => out.errors.push(String(e)));
  await page.evaluate(() => localStorage.setItem("cm.explorer.howto-seen", "1"));
  const settle = async () => {
    await page.waitForSelector("#tab-explorer:not([hidden])", { timeout: 20000 });
    await page.waitForFunction(() => window.__ex && window.__ex.ready, null, { timeout: 60000 });
    await page.evaluate(() => window.__ex.ready);
    await page.waitForTimeout(150);
    await page.waitForFunction(() => !/loading/i.test(document.querySelector("#ex-status")?.textContent || "") && window.__ex.col, null, { timeout: 90000 });
    await page.waitForTimeout(400);
  };
  const go = async h => { await page.evaluate(h => { location.hash = h; }, h); await settle(); };
  const snap = async label => {
    const r = await page.evaluate(label => {
      const ex = window.__ex, st = ex.st, col = ex.col, { S, T } = ex.ctx, t = st.anchor, shown = id => getComputedStyle(document.getElementById(id)).display !== "none";
      const other = (t + 1) % T;
      return { label, hash: location.hash, color: st.color, kernel: ex.ctx.kernel, mode: st.mode, rule: ex.ctx.m.spacetime?.rule, fam: st.fam, anchor: t,
        dv: st.dv, dt: st.dt, dm: st.dm, dk: st.dk, dr: st.dr, depthOn: document.getElementById("ex").classList.contains("depth"),
        ctl: !document.getElementById("ex-depth-ctl").hidden, lock: document.getElementById("ex-depth-lock").textContent,
        ruleLocked: document.getElementById("ex-mode").disabled, dtHidden: document.getElementById("ex-dt-wrap").hidden,
        dtLabel: document.getElementById("ex-dt-val").textContent, drHidden: document.getElementById("ex-dr").hidden,
        blockCards: ["card-block", "card-coarse", "card-ret"].filter(shown), depthCard: shown("card-depth"),
        depthSvgs: document.querySelectorAll("#card-depth svg").length,
        rowColours: col.depth ? [...Array(S)].map((_, s) => new Set(col.ciCell[t * S + s]).size) : null,
        otherRowColours: new Set(col.ci[other]).size, otherRowPerPanel: !!(col.ciCell && col.ciCell[other * S]),
        lowPanels: document.querySelectorAll(".cellbg.lowwt").length, status: document.getElementById("ex-status").textContent,
        caption: document.getElementById("ex-caption").textContent.slice(0, 160), dataStep: st._dep?.step ?? null, step: ex.ctx.m.steps[t],
        legendSite: (document.querySelector("#leg-body .leg-note")?.textContent.match(/Counts at (\S+),/) || [])[1] || null };
    }, label);
    out.steps.push(r);
    return r;
  };
  const shot = async name => {
    await page.addStyleTag({ content: ".ex-bar { position: static !important; } #ex-howto, #ex-tour { display: none !important; }" });
    await page.locator("#ex-body").screenshot({ path: `${OUTDIR}/explorer-depth-${name}.png`, timeout: 60000 });
  };
  const has = await page.evaluate(async () => { const m = await (await fetch("d/meta.json")).json(); return !!m.spacetime; });
  if (!has) return "@@JSON@@" + JSON.stringify({ skipped: "no meta.spacetime in this build" }) + "@@END@@";
  await go("#explorer?fam=layer&t=7&a=0&b=11&m=0&c=depth&dv=own&dt=3&dm=1");
  await snap("open: final checkpoint, relative units, t = t*, mode 1"); await shot("final-own-mode1");
  // switch checkpoint: a real click on a panel of row 1 (step 64)
  const xy = await page.evaluate(() => { const ex = window.__ex, c = ex.cellRect(1, 5), b = ex.canvas.getBoundingClientRect(); return [b.left + c.x + c.w / 2, b.top + c.y + c.w / 2]; });
  await page.mouse.click(xy[0], xy[1]); await settle();
  await snap("click a panel in row 1"); await shot("early-own-mode1");
  await page.click("#ex-dv button[data-v='none']"); await settle(); await snap("variant: native walk units");
  await page.click("#ex-dv button[data-v='paper']"); await settle(); await snap("variant: paper transition"); await shot("early-paper-mode1");
  await page.click("#ex-dv button[data-v='own']"); await settle();
  await page.$eval("#ex-dt", e => { e.value = "0"; e.dispatchEvent(new Event("input", { bubbles: true })); }); await settle();
  await snap("coupling: t*/64");
  await page.selectOption("#ex-dm", "4"); await settle(); await snap("mode 4");
  await page.selectOption("#ex-dm", "s10"); await settle();
  await page.click("#ex-dr button[data-v='s']"); await settle(); await snap("sets k = 10, shared coordinates");
  // round trip: leave the tab and come back through the URL just written
  const h = await page.evaluate(() => location.hash);
  await page.evaluate(() => { location.hash = "#overview"; }); await page.waitForTimeout(400);
  await page.evaluate(() => { location.hash = "#explorer?fam=layer&t=7&a=0&b=11&m=0&c=seba&r=4"; }); await settle();
  await go(h); await snap("round trip of " + h);
  await go("#explorer?fam=layer&t=7&a=0&b=11&m=0&c=depth&dv=own&dt=3&dm=sets&dk=5&dr=a");
  await snap("sets k = 5, shape-aligned, final"); await shot("final-own-sets5");
  // incompatible: dot kernel, another rule, a time block, paper with a coupling, a mode out of range
  await go("#explorer?fam=time&t=9&a=2&b=5&m=3&c=depth&kern=dot&dv=paper&dt=9&dm=40");
  await snap("incompatible: kern=dot, m=3, time block, paper + dt=9, dm=40");
  await page.selectOption("#ex-color", "seba"); await settle(); await snap("leave: coherent sets");
  // the experiment page's link form (dv=rel, dc, dm=s5) and an isolated set that does not exist (j=9 with k = 5)
  await go("#explorer?fam=layer&t=7&a=0&b=11&m=0&c=depth&dv=rel&dc=3&dm=s5&j=9");
  await snap("link form dv=rel&dc=3&dm=s5, j=9");
  // the sets legend follows the images card's site
  await page.selectOption("#img-sub select.dp-site", "6"); await page.waitForTimeout(300);
  await snap("site b6 picked in the first card");
  // a slow load: click row 4 twice while its data are delayed; nothing may show the old data under the new step
  await go("#explorer?fam=layer&t=7&a=0&b=11&m=0&c=depth&dv=own&dt=3&dm=1");
  await page.route("**/d/st/2560_own.bin.gz", async r => { await new Promise(z => setTimeout(z, 2500)); await r.continue(); });
  const xy4 = await page.evaluate(() => { const ex = window.__ex, c = ex.cellRect(4, 3), b = ex.canvas.getBoundingClientRect(); return [b.left + c.x + c.w / 2, b.top + c.y + c.w / 2]; });
  await page.mouse.click(xy4[0], xy4[1]); await page.waitForTimeout(300);
  await page.mouse.click(xy4[0], xy4[1]); await page.waitForTimeout(300);
  await page.evaluate(() => window.__ex.renderAll());          // a redraw (e.g. a resize) while the load is pending
  out.pending = await page.evaluate(() => ({ sub: document.getElementById("dp-sub").textContent, onScreen: window.__ex.st._dep?.step, anchor: window.__ex.st.anchor }));
  await settle(); await page.unroute("**/d/st/2560_own.bin.gz");
  await snap("after a slow load of step 2,560");
  return "@@JSON@@" + JSON.stringify(out) + "@@END@@";
}
"""


def depth_checks(res: dict) -> list[str]:
    """Assertions on DEPTH_JS's snapshots; returns failure messages."""
    fails, steps = [], res["steps"]
    chk = lambda ok, msg: None if ok else fails.append("through depth: " + msg)
    for x in steps:
        if x["color"] != "depth":
            continue
        chk(x["depthOn"] and x["ctl"] and x["depthCard"] and x["depthSvgs"] >= 3, f"[{x['label']}] depth controls/card not shown ({x['depthSvgs']} charts)")
        chk(not x["blockCards"], f"[{x['label']}] block-only cards visible: {x['blockCards']}")
        chk(x["kernel"] == "cos" and x["mode"] == x["rule"] and x["ruleLocked"] and "cosine" in x["lock"], f"[{x['label']}] kernel/rule not locked ({x['kernel']}, rule {x['mode']}, '{x['lock']}')")
        chk(x["fam"] == "layer" and x["dataStep"] == x["step"], f"[{x['label']}] data for step {x['dataStep']} shown at step {x['step']}")
        chk(x["otherRowColours"] == 1 and not x["otherRowPerPanel"], f"[{x['label']}] other rows are not uniformly grey")
        rc = x["rowColours"] or []
        chk(len(rc) == 12 and all(n >= 1 for n in rc) and sum(n > 2 for n in rc) >= 1, f"[{x['label']}] coloured row has no varying panel: {rc}")
        chk(x["dtHidden"] == (x["dv"] == "paper"), f"[{x['label']}] coupling slider visibility wrong for {x['dv']}")
        chk(x["drHidden"] == (x["dm"] != "sets"), f"[{x['label']}] readout control visibility wrong")
        chk("c=depth" in x["hash"] and f"dv={x['dv']}" in x["hash"], f"[{x['label']}] hash lacks the depth state: {x['hash']}")
    a, b = steps[0], steps[1]
    chk(a["anchor"] == 7 and a["dv"] == "own" and a["dt"] == 3 and a["dm"] == 1 and "t/t* = 1" in a["dtLabel"] and "t = " in a["dtLabel"], f"open state wrong: {a}")
    chk(b["anchor"] == 1 and b["step"] == 64, f"clicking a panel in row 1 did not select step 64: anchor {b['anchor']}")
    chk(steps[2]["dv"] == "none" and steps[3]["dv"] == "paper" and "dt=" not in steps[3]["hash"], "variant switches")
    chk(steps[4]["dt"] == 0 and "1/64" in steps[4]["dtLabel"] and "dt=0" in steps[4]["hash"], f"coupling switch: {steps[4]['dtLabel']}")
    chk(steps[5]["dm"] == 4 and "dm=4" in steps[5]["hash"], "mode switch")
    chk(steps[6]["dm"] == "sets" and steps[6]["dk"] == 10 and steps[6]["dr"] == "s" and "dk=10" in steps[6]["hash"], "sets switch")
    rt = steps[7]
    chk(all(rt[k] == steps[6][k] for k in ("anchor", "dv", "dt", "dm", "dk", "dr")) and rt["hash"] == steps[6]["hash"], f"URL round trip changed the state: {steps[6]['hash']} -> {rt['hash']}")
    inc = steps[9]
    chk(inc["color"] == "depth" and inc["kernel"] == "cos" and inc["mode"] == inc["rule"] and inc["fam"] == "layer" and inc["anchor"] == 5
        and inc["dv"] == "paper" and inc["dm"] == 1 and "kern=dot" not in inc["hash"] and "dt=" not in inc["hash"], f"incompatible state not fitted: {inc}")
    chk("rule" in inc["status"] and "mode 40" in inc["status"].lower(), f"incompatible state not announced in the status line: '{inc['status']}'")
    out = steps[10]
    chk(out["color"] == "seba" and not out["depthOn"] and "card-block" in out["blockCards"] and not out["lowPanels"] and not out["ruleLocked"], f"leaving the colouring did not restore the block view: {out}")
    lf, site, slow = steps[11], steps[12], steps[13]
    chk(lf["color"] == "depth" and lf["dv"] == "own" and lf["dt"] == 3 and lf["dm"] == "sets" and lf["dk"] == 5 and "j=" not in lf["hash"],
        f"link form dv=rel&dc=3&dm=s5&j=9 not read as relative units, t*, sets k = 5 without focus: {lf}")
    chk(site["legendSite"] == "b6", f"sets legend did not follow the images card's site: {site['legendSite']}")
    pend = res.get("pending") or {}
    chk(f"{pend.get('onScreen', 0):,}" in pend.get("sub", ""), f"while loading, the depth card shows step {pend.get('onScreen')} data under '{pend.get('sub')}'")
    chk(slow["anchor"] == 4 and slow["dataStep"] == 2560 and slow["step"] == 2560 and "2,560" in slow["caption"], f"after the slow load: {slow}")
    for e in res["errors"]:
        fails.append(f"through depth: page error: {e}")
    return fails


def aliases():
    js = (SITE / "js" / "main.js").read_text()
    body = js[js.index("const ANCHOR_ALIAS"): js.index("};", js.index("const ANCHOR_ALIAS"))]
    rows = re.findall(r'"([a-z-]+/[A-Za-z0-9_-]+)":\s*"([a-z-]+(?:/[A-Za-z0-9_-]+)?)"', body)
    tab = js[js.index("const TAB_ALIAS"): js.index("};", js.index("const TAB_ALIAS"))]
    rows += re.findall(r'(\w+):\s*"([a-z-]+)"', tab.split("=", 1)[1])
    return rows


class Shot(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inshot, self.href, self.src = False, None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "figure" and "explorer-shot" in (a.get("class") or ""):
            self.inshot = True
        if self.inshot and tag == "a" and self.href is None:
            self.href = a.get("href")
        if self.inshot and tag == "img" and self.src is None:
            self.src = a.get("src")

    def handle_endtag(self, tag):
        if tag == "figure":
            self.inshot = False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="x-check", help="playwright-cli session name (distinct per agent)")
    ap.add_argument("--out", default="/tmp/x-check", help="directory for screenshots")
    ap.add_argument("--only", default="", help="comma-separated experiment ids to check (default: all in the build)")
    ap.add_argument("--no-aliases", action="store_true")
    ap.add_argument("--no-depth", action="store_true", help="skip the explorer's through-depth smoke test")
    ap.add_argument("--tabs", default="experiments",
                    help="tabs whose live figures to check and screenshot: experiments (default), all, or a list")
    ap.add_argument("--capture-explorer", default=None, metavar="ID",
                    help="take assets/fig/x_<ID>_explorer.png from the href of the fragment's figure.explorer-shot")
    a = ap.parse_args()
    if not (OUT / "index.html").exists():
        sys.exit(f"no build at {OUT}: run build_site.py first")
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "-d", str(OUT)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/"
    ses = Session(a.session, outdir)          # playwright-cli keeps its logs in the cwd: keep them out of the repo
    fails = []
    try:
        for _ in range(50):
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
            time.sleep(0.1)
        ses.run("open", url, "--browser=chrome")
        ses.run("resize", "1440", "900")
        if a.capture_explorer:
            xid = a.capture_explorer
            p = Shot()
            p.feed((SITE / "content" / "experiments" / f"{xid}.html").read_text())
            if not p.href or not p.src:
                sys.exit(f"{xid}.html has no figure.explorer-shot with <a href> and <img src>")
            if not p.href.startswith("#explorer?"):
                sys.exit(f"explorer-shot href must start with #explorer?: {p.href}")
            dest = SITE / p.src
            if dest.name != f"x_{xid}_explorer.png":
                sys.exit(f"explorer-shot img src must be assets/fig/x_{xid}_explorer.png, not {p.src}")
            js = (EXPLORER_JS.replace("%HASH%", json.dumps(p.href)).replace("%SEL%", json.dumps("#ex-main"))
                  .replace("%PATH%", json.dumps(str(dest))))
            r = ses.code(js)
            shutil.copy(dest, OUT / p.src)          # so the current build shows it without a rebuild
            print(f"captured {dest.relative_to(ROOT)} ({r['status'].strip() or 'explorer ready'})")
            return
        only = [x for x in a.only.split(",") if x]
        tabs = (["experiments", "choices", "baselines", "method-details", "measurements", "reproduce", "overview",
                 "method"] if a.tabs == "all" else [t for t in a.tabs.split(",") if t])
        for tab in tabs:
            ses.run("resize", "1440", "900")
            res = ses.code(CHECK_JS.replace("%ONLY%", json.dumps(only)).replace("%TAB%", json.dumps(tab)), timeout=1200)
            for sec in res:
                figs = sec["figures"]
                bad = [f for f in figs if f["state"] != "ok"]
                badimg = [i for i in sec["images"] if not i["ok"]]
                tag = "placeholder" if sec["placeholder"] else f"{len(figs) - len(bad)}/{len(figs)} live figures rendered"
                print(f"{sec['id']}: {tag}" + (f", {len(sec['images']) - len(badimg)}/{len(sec['images'])} images" if sec["images"] else ""))
                for f in bad:
                    fails.append(f"{sec['id']}: {f['fig']}: {f['state']} {f['detail']}")
                for i in badimg:
                    fails.append(f"{sec['id']}: image did not load: {i['src']}")
            for w, h, name in ((1440, 900, "desktop"), (390, 844, "mobile")):
                ses.run("resize", str(w), str(h))
                ses.code(r"""async page => { await page.evaluate(async t => { for (const f of document.querySelectorAll(`#tab-${t} figure.live`)) { f.scrollIntoView(); await new Promise(r => setTimeout(r, 400)); } window.scrollTo(0, 0); }, %TAB%); await page.waitForTimeout(800); return "@@JSON@@{}@@END@@"; }""".replace("%TAB%", json.dumps(tab)), timeout=1200)
                # very tall pages exceed playwright-cli's 5 s screenshot timeout: shoot through run-code, one file
                # per experiment section (Experiments tab) or one full page (other tabs)
                path = outdir / f"{tab}-{name}.png"
                shots = ses.code(SHOT_JS.replace("%TAB%", json.dumps(tab)).replace("%PATH%", json.dumps(str(path))),
                                 timeout=1200)
                for f in shots:
                    print(f"screenshot: {f}")
        if not a.no_aliases:
            ses.run("resize", "1440", "900")
            res = ses.code(ALIAS_JS.replace("%ROWS%", json.dumps(aliases())))
            n_ok = n_dorm = n_absent = 0
            for r in res:
                tab = r["to"].split("/")[0]
                if r["dormant"]:
                    n_dorm += 1
                elif r["shown"] == tab and r["exists"]:
                    n_ok += 1
                elif r["shown"] == tab and tab == "experiments" and r["to"].startswith("experiments/x-"):
                    n_absent += 1
                    print(f"  alias {r['from']} -> {r['to']}: target not in this build's manifest (lands on the tab top)")
                else:
                    fails.append(f"alias {r['from']} -> {r['to']}: landed on '{r['shown']}' ({r['hash']})")
            print(f"aliases: {n_ok} land on their target, {n_dorm} dormant (old id still in place), "
                  f"{n_absent} target not built yet, {len(res) - n_ok - n_dorm - n_absent} failed")
        if not a.no_depth:
            ses.run("resize", "1440", "900")
            res = ses.code(DEPTH_JS.replace("%OUT%", json.dumps(str(outdir.resolve()))), timeout=900)
            if "skipped" in res:
                print(f"through depth: skipped ({res['skipped']})")
            else:
                bad = depth_checks(res)
                fails += bad
                for x in res["steps"]:
                    print(f"  depth [{x['label']}]: step {x['step']}, {x['dv']}, dt {x['dt']}, dm {x['dm']}, kernel {x['kernel']}, rule {x['mode']}"
                          + (f"; status: {x['status']}" if x["status"] else ""))
                print(f"through depth: {len(res['steps'])} states, {len(bad)} failed; screenshots {outdir}/explorer-depth-*.png")
    finally:
        with contextlib.suppress(Exception):
            ses.run("close", check=False, timeout=30)
        srv.terminate()
        srv.wait(timeout=10)
    for f in fails:
        print("FAIL:", f)
    print("browser check: " + ("FAILED" if fails else "passed"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
