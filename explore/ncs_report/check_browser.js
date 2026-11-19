// Playwright driver for check_browser.sh (run with `playwright-cli run-code`). check_browser.sh substitutes __BASE__,
// __TOURS__ and __SHOTS__. Loads every tab and both explorers at 1440 x 900 and 390 x 844, and every tour at 390 x 844;
// records console errors and page errors; at 390 px requires document.documentElement.scrollWidth <= innerWidth
// (REVIEW_2 objection 3). Screenshots of each tab and explorer at both sizes go to __SHOTS__ (or nowhere if empty).
async page => {
  const base = "__BASE__", tours = __TOURS__, shots = "__SHOTS__";
  const errors = [], overflow = [], checked = [];
  page.on("console", m => { if (m.type() === "error") errors.push(`${page.url()}: ${m.text()}`); });
  page.on("pageerror", e => errors.push(`${page.url()}: ${e.message}`));
  const views = [["overview", "#overview"], ["explorer-vit", "#explorer?net=vit"], ["explorer-mamba", "#explorer?net=mamba"],
    ["findings", "#findings"], ["measurements", "#measurements"], ["method", "#method"]];
  let n = 0;
  const visit = async (name, hash, w, h, shot) => {
    await page.setViewportSize({ width: w, height: h });
    await page.goto(`${base}?c=${n++}${hash}`);
    await page.waitForTimeout(hash.startsWith("#explorer") ? 5000 : 3500);
    if (!hash.startsWith("#explorer")) {     // scroll through so lazy images and live figures load
      await page.evaluate(async () => { for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); } window.scrollTo(0, 0); });
      await page.waitForTimeout(1500);
    }
    const sw = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    if (w <= 400 && sw[0] > sw[1]) overflow.push(`${name} at ${w} px: scrollWidth ${sw[0]} > ${sw[1]}`);
    checked.push(`${name}@${w}`);
    if (shot && shots) await page.screenshot({ path: `${shots}/${name}-${w}.png`, fullPage: !hash.startsWith("#explorer") });
  };
  for (const [w, h] of [[1440, 900], [390, 844]]) for (const [name, hash] of views) await visit(name, hash, w, h, true);
  for (const t of tours) await visit(`tour-${t}`, `#explorer?tour=${t}`, 390, 844, false);
  return JSON.stringify({ ok: !errors.length && !overflow.length, errors, overflow, checked: checked.length });
}
