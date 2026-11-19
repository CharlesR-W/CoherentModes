#!/usr/bin/env bash
# Screenshot every tab of a built page at 1440 and 390 px, exercise the explorer, and fail on console errors.
#
#   bash explore/mamba_contexts/page/verify.sh [page.html] [shots_dir]
#
# Default page: mamba_contexts.html if it exists, else mamba_contexts_MOCKUP.html. Default shots: /var/tmp/cm-mc-page/shots.
# Needs playwright-cli (headless Chrome) and network access for the Plotly and MathJax CDNs. Look at the screenshots.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PAGE="${1:-}"
OUT="${2:-/var/tmp/cm-mc-page/shots}"
if [[ -z "$PAGE" ]]; then
  if [[ -f "$HERE/mamba_contexts.html" ]]; then PAGE="$HERE/mamba_contexts.html"; else PAGE="$HERE/mamba_contexts_MOCKUP.html"; fi
fi
PAGE="$(cd "$(dirname "$PAGE")" && pwd)/$(basename "$PAGE")"
[[ -f "$PAGE" ]] || { echo "no page at $PAGE (build it first)"; exit 1; }
mkdir -p "$OUT" /var/tmp/cm-mc-page
cd /var/tmp/cm-mc-page                  # keep playwright artefacts out of the repo
S="mcverify$$"
URL="file://$PAGE"
pw() { playwright-cli -s=$S "$@"; }
trap 'pw close >/dev/null 2>&1 || true' EXIT
pw open about:blank --browser=chrome >/dev/null
TABS="overview findings explorer measurements method"
for size in "1440 900" "390 844"; do
  set -- $size
  w=$1
  pw resize "$1" "$2" >/dev/null
  pw run-code "async page => { await page.goto('$URL'); await page.waitForTimeout(2500); }" >/dev/null
  for tab in $TABS; do
    pw run-code "async page => {
      await page.evaluate(t => { location.hash = '#tab=' + t; }, '$tab');
      await page.waitForTimeout(1500);
      await page.evaluate(() => document.querySelectorAll('section.tab.active details').forEach(d => d.open = true));
      await page.waitForTimeout(400);
    }" >/dev/null
    pw screenshot --full-page --filename="$OUT/${tab}_${w}.png" >/dev/null
    echo "  $tab @ ${w}px -> $OUT/${tab}_${w}.png"
  done
done

# interactions at 1440 px: explorer colourings, a set card click, a point click, eigenfunction ends; F3/F6 controls
pw resize 1440 1000 >/dev/null
RES=$(pw --raw run-code "async page => {
  await page.goto('$URL#tab=explorer'); await page.waitForTimeout(2500);
  const out = {};
  const sel = async (name, v) => { await page.locator('section.tab.active [data-ctl=' + name + '] select').selectOption(String(v)); await page.waitForTimeout(500); };
  await sel('col', 'cls'); out.cls = await page.locator('#plot-ex_layout .legend .traces').count();
  await page.locator('#plot-ex_sets .sethead[data-set]').first().click(); await page.waitForTimeout(400);
  await sel('col', 'sets');
  out.setOn = await page.locator('#plot-ex_sets .setcard.on').count();
  await page.screenshot({ path: '$OUT/explorer_set_isolated_1440.png' });
  await page.evaluate(() => { EX.item = 5; drawTab('explorer'); }); await page.waitForTimeout(400);
  out.item = await page.locator('#plot-ex_item .itemcard').count();
  await sel('col', 'psi2'); await sel('k', 3);
  out.ends = await page.locator('#plot-ex_ends ol.exs li').count();
  await page.screenshot({ path: '$OUT/explorer_psi_1440.png', fullPage: true });
  await sel('obj', 'whole'); await sel('col', 'src');
  await page.screenshot({ path: '$OUT/explorer_whole_src_1440.png' });
  await page.evaluate(() => { location.hash = '#tab=measurements'; }); await page.waitForTimeout(1500);
  await sel('f3obj', 'v1T1'); out.f3 = await page.locator('#plot-f3 .scatterlayer .trace').count();
  await page.locator('#fig-f3').screenshot({ path: '$OUT/f3_v1T1_1440.png' });
  await page.locator('section.tab.active [data-ctl=tree] button[data-v=T2]').click(); await page.waitForTimeout(600);
  out.f6 = await page.locator('#f6-node .itemcard').count();
  await page.locator('#fig-f6').screenshot({ path: '$OUT/f6_T2_1440.png' });
  await page.locator('section.tab.active [data-ctl=f4stat] button[data-v=eta]').click(); await page.waitForTimeout(500);
  await page.locator('#fig-f4').screenshot({ path: '$OUT/f4_eta_1440.png' });
  out.errs = await page.evaluate(() => Array.from(document.querySelectorAll('.plot')).filter(p => p.textContent.includes('draw error')).map(p => p.id));
  return JSON.stringify(out);
}")
RES=$(echo "$RES" | tr -d '\\')
echo "interactions: $RES"
ERR=$(pw console error 2>&1 || true)
pw close >/dev/null 2>&1 || true
trap - EXIT
echo "console (errors): ${ERR:-none}"
SIZE=$(stat -c %s "$PAGE")
printf 'page: %s (%.2f MB)\n' "$PAGE" "$(echo "$SIZE / 1000000" | bc -l)"
fail=0
if echo "$ERR" | grep -qiE '\[error\]|uncaught|TypeError|ReferenceError|draw error'; then echo "FAIL: console errors"; fail=1; fi
echo "$RES" | grep -q '"errs":\[\]' || { echo "FAIL: a figure reported a draw error"; fail=1; }
echo "$RES" | grep -q '"setOn":1' || { echo "FAIL: set card click did not isolate a set"; fail=1; }
echo "$RES" | grep -q '"item":1' || { echo "FAIL: item card"; fail=1; }
echo "$RES" | grep -q '"ends":24' || { echo "FAIL: eigenfunction ends"; fail=1; }
echo "$RES" | grep -q '"f6":1' || { echo "FAIL: tree node card"; fail=1; }
(( fail == 0 )) && echo "verify: OK" || { echo "verify: FAILED"; exit 1; }
