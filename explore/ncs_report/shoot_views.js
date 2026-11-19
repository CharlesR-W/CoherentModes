// Playwright driver for shoot_views.sh (run with `playwright-cli run-code`). shoot_views.sh substitutes __VIEWS__,
// __PORT__ and __TITLE__. For each view: load the tour, wait until the explorer shows exactly what the tour names
// (network, block or object, checkpoint, colouring, r or k), fail if it never does, hide the page chrome (banner,
// tabs, control bar, tour note, how-to, step hints, point-size row) and capture the panels and the side panel.
async page => {
  const views = __VIEWS__, base = "http://127.0.0.1:__PORT__/", title = __TITLE__;
  const got = [];
  // hide the page chrome before the explorer lays out, with transitions off (a headless capture caught the ViT block
  // outline mid-slide, misplaced)
  const CHROME = ".banner, .tabs, .ex-bar, .ex-tour, .ex-howto, .ex-steps, .ex-ps { display: none !important; } .ex-side { position: static !important; } * { transition: none !important; animation: none !important; }";
  await page.addInitScript(css => document.addEventListener("DOMContentLoaded", () => {
    const st = document.createElement("style"); st.textContent = css; document.head.append(st);
  }), CHROME);
  const read = () => {
    const $ = id => document.getElementById(id);
    const vit = !$("ex").hidden, mx = !$("mx").hidden;
    if (vit === mx) return { net: null };
    if (vit) return { net: "vit", title: document.title, fam: document.querySelector("#ex-fam [aria-checked=true]")?.dataset.v,
      a: $("ex-a").value, b: $("ex-b").value, m: $("ex-mode").value, c: $("ex-color").value, k: $("ex-k").value,
      cap: $("ex-caption").textContent.length, side: $("blk-title").textContent };
    return { net: "mamba", title: document.title, o: $("mx-obj").value, c: $("mx-color").value, k: $("mx-k").value,
      ck: document.querySelector("#mx-ck [aria-checked=true]")?.dataset.v, lay: document.querySelector("#mx-lay [aria-checked=true]")?.dataset.v, cap: $("mx-caption").textContent.length,
      side: $("mx-sets").children.length, sub: $("mx-sets-sub").textContent };
  };
  for (const v of views) {
    await page.goto(`${base}?shot=${v.id}#explorer?tour=${v.id}`);
    let s = null;
    try {
      await page.waitForFunction(([e, t, fn]) => {
        const s = eval(fn)();
        if (s.net !== e.net || s.title !== t || !s.cap || !s.side) return false;
        return Object.keys(e).every(key => String(s[key]) === String(e[key]));
      }, [v.expect, title, `(${read.toString()})`], { timeout: 45000, polling: 250 });
      s = await page.evaluate(`(${read.toString()})()`);
    } catch (err) {
      s = await page.evaluate(`(${read.toString()})()`);
      return JSON.stringify({ ok: false, view: v.id, expect: v.expect, got: s });
    }
    await page.waitForTimeout(2500);                                    // let the panels finish drawing
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);
    const pre = v.expect.net === "vit" ? "ex" : "mx";
    const box = await page.evaluate(pre => {
      const r = id => document.getElementById(id).getBoundingClientRect(), y0 = window.scrollY;
      const body = r(`${pre}-key`), grid = r(pre === "ex" ? "ex-gridwrap" : "mx-grid"), side = r(`${pre}-side`);
      // the panels and, beside them, the top of the side panel; the definitional caption is left out (the finding's box
      // states the view, and the caption would be illegible at the page's width)
      const top = Math.max(0, Math.min(body.top, side.top) - 8), bottom = grid.bottom + 10;
      return { x: 0, y: top + y0, width: document.documentElement.clientWidth, height: Math.ceil(bottom - top) };
    }, pre);
    await page.screenshot({ path: `/tmp/ncs-views/${v.id}.png`, clip: box, fullPage: true });
    got.push({ id: v.id, label: v.label, got: s });
  }
  return JSON.stringify({ ok: true, views: got });
}
