#!/usr/bin/env bash
# Screenshot every tab of the built page at 1440 and 390 px, exercise the explorer and zoom views, and fail on any
# console error or any request that is not the page itself or a data: URL.
#
#   bash explore/lm_coherent/page/verify.sh [shots_dir] [page.html]
#
# Default page: lm_coherent.html if it exists, else lm_coherent_MOCKUP.html. Needs playwright-cli (headless Chrome).
# Screenshots: <shots_dir>/<tab>_<width>.png (full page) plus explorer/zoom interaction states. Look at them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/var/tmp/cm-lm/shots}"
PAGE="${2:-}"
if [[ -z "$PAGE" ]]; then
  if [[ -f "$HERE/lm_coherent.html" ]]; then PAGE="$HERE/lm_coherent.html"; else PAGE="$HERE/lm_coherent_MOCKUP.html"; fi
fi
PAGE="$(cd "$(dirname "$PAGE")" && pwd)/$(basename "$PAGE")"
[[ -f "$PAGE" ]] || { echo "no page at $PAGE (build it first)"; exit 1; }
export TMPDIR="${TMPDIR:-/var/tmp/cm-lm}"
mkdir -p "$OUT" "$TMPDIR"
cd "$TMPDIR"                      # keep playwright artefacts out of the repo
S=lmverify
URL="file://$PAGE"
pw() { playwright-cli -s=$S "$@"; }
trap 'pw close >/dev/null 2>&1 || true' EXIT

pw open about:blank --browser=chrome >/dev/null   # the CLI blocks file: in open/goto; run-code may load it
# visit a tab, open collapsed details (F11) so their figures draw, scroll through to trigger lazy figures, back to top
visit() {
  pw run-code "async page => {
    await page.evaluate(t => { location.hash = '#' + t; }, '$1');
    await page.waitForTimeout(900);
    await page.evaluate(() => document.querySelectorAll('.tab-panel:not([hidden]) details.more').forEach(d => d.open = true));
    const H = await page.evaluate(() => document.body.scrollHeight);
    for (let y = 0; y < H; y += 500) { await page.evaluate(v => window.scrollTo(0, v), y); await page.waitForTimeout(60); }
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);
  }" >/dev/null
}
for size in "1440 900" "390 844"; do
  set -- $size
  w=$1
  pw resize "$1" "$2" >/dev/null
  pw run-code "async page => { await page.goto('$URL'); await page.waitForTimeout(1500); }" >/dev/null
  for tab in overview explorer zoom findings measurements method; do
    visit "$tab"
    pw screenshot --full-page --filename="$OUT/${tab}_${w}.png" >/dev/null
    echo "  $tab @ ${w}px -> $OUT/${tab}_${w}.png"
  done
done

# interactions at 1440 px: explorer (depth block, a panel, psi colouring, pin, hover, slab view) and zoom (T2, a node)
pw resize 1440 1500 >/dev/null   # tall, so every panel the checks click is inside the viewport
RES=$(pw --raw run-code "async page => {
  await page.goto('$URL#explorer'); await page.waitForTimeout(1500);
  const out = {};
  await page.locator('.ex-grid .colh.lbl-btn').nth(2).click(); await page.waitForTimeout(300);
  out.depth = await page.evaluate(() => window.__ex.st.sel.type + ':' + window.__ex.st.sel.ti);
  const box = await page.locator('#ex-grid canvas.base').boundingBox();
  const g = await page.evaluate(() => window.__ex.panelRect(4 * 6 + 5));
  await page.mouse.click(box.x + g.x + g.w / 2, box.y + g.y + g.w / 2); await page.waitForTimeout(300);
  out.cell = await page.evaluate(() => JSON.stringify(window.__ex.st.sel));
  await page.selectOption('#ex-color', 'psi1'); await page.waitForTimeout(300);
  await page.selectOption('#ex-pin', { index: 9 }); await page.waitForTimeout(200);
  const pt = await page.evaluate(() => { const ex = window.__ex, k = 4 * 6 + 5, r = ex.panelRect(k), xy = ex.coords(k); return { x: r.x + r.w * xy[10] / 255, y: r.y + r.w * (1 - xy[11] / 255) }; });
  await page.mouse.move(box.x + pt.x + 3, box.y + pt.y + 3); await page.mouse.move(box.x + pt.x, box.y + pt.y);
  await page.waitForTimeout(200);
  out.hover = await page.evaluate(() => window.__ex.st.hover);
  await page.screenshot({ path: '$OUT/explorer_interact_1440.png' });
  await page.locator('#ex-view button[data-v=s]').click(); await page.waitForTimeout(300);
  out.slab = await page.evaluate(() => window.__ex.rec().key);
  await page.locator('.strip-btns button').first().click(); await page.waitForTimeout(300);
  out.train = await page.evaluate(() => JSON.stringify(window.__ex.st.sel));
  await page.selectOption('#ex-color', 'tok'); await page.waitForTimeout(300);
  await page.screenshot({ path: '$OUT/explorer_train_tok_1440.png', fullPage: true });
  await page.evaluate(() => { location.hash = '#zoom'; }); await page.waitForTimeout(900);
  await page.locator('#zoom .seg button', { hasText: 'T2' }).click(); await page.waitForTimeout(300);
  await page.locator('#zoom .ic-node').nth(1).click(); await page.waitForTimeout(500);
  out.zoom = await page.evaluate(() => window.__zoom.st.tree + ':' + window.__zoom.st.node);
  await page.screenshot({ path: '$OUT/zoom_T2_node1_1440.png', fullPage: true });
  await page.evaluate(() => { location.hash = '#measurements/q2'; }); await page.waitForTimeout(1200);
  const f4 = page.locator('figure[data-live=f4] rect[style*=pointer]').nth(20);
  if (await f4.count()) { await f4.click(); await page.waitForTimeout(300); }
  out.f4detail = await page.evaluate(() => !!document.querySelector('.x-detail svg'));
  await page.evaluate(() => { location.hash = '#explorer?sel=cell:143000:bag:128&c=src'; }); await page.waitForTimeout(800);
  out.deeplink = await page.evaluate(() => 'dl:' + window.__ex.st.sel.li + ':' + window.__ex.st.color);
  await page.evaluate(() => { location.hash = '#zoom?tree=T3&node=2'; }); await page.waitForTimeout(800);
  out.zoomlink = await page.evaluate(() => 'zl:' + window.__zoom.st.tree + ':' + window.__zoom.st.node);
  return JSON.stringify(out);
}")
echo "interactions: $RES"

ERR=$(pw console error 2>&1 || true)
REQ=$(pw requests 2>&1 || true)
pw close >/dev/null 2>&1 || true
trap - EXIT
echo "console (errors): ${ERR:-none}"
BAD=$(echo "$REQ" | grep -Eo '(https?|file)://[^ ]+' | grep -v "^file://$PAGE" | grep -v '^data:' || true)
SIZE=$(stat -c %s "$PAGE")
printf 'page: %s (%.2f MB)\n' "$PAGE" "$(echo "$SIZE / 1000000" | bc -l)"
fail=0
if echo "$ERR" | grep -qiE '\[error\]|uncaught|TypeError|ReferenceError|failed'; then echo "FAIL: console errors"; fail=1; fi
if [[ -n "$BAD" ]]; then echo "FAIL: external requests:"; echo "$BAD"; fail=1; fi
if (( SIZE > 10000000 )); then echo "FAIL: page above 10 MB"; fail=1; fi
for k in 'dl:-1:src' 'zl:T3:2' 'depth:2' 'T2:1' 'f4detail[^,]*true' 'hover[^,]*:5' 's:143000:3:5' 'train'; do
  echo "$RES" | grep -q "$k" || { echo "FAIL: interaction check $k"; fail=1; }
done
(( fail == 0 )) && echo "verify: OK" || { echo "verify: FAILED"; exit 1; }
