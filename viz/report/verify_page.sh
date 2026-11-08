#!/usr/bin/env bash
# Adapted from CellsDevelopmental/scripts/verify_diffusion_geometry_page.sh.
# Requires globally installed playwright-cli and headless Google Chrome.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
PAGE=$(realpath "${1:-$HERE/example.html}")
OUT=${2:-$(mktemp -d /tmp/report-shots.XXXXXX)}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
[ -s "$PAGE" ] || { echo 'Missing or empty page' >&2; exit 2; }
[ "$(stat -c%s "$PAGE")" -lt 15000000 ] || { echo 'Page exceeds 15 MB' >&2; exit 2; }
SESSION="report-verify-$$"
WORK=$(mktemp -d /tmp/report-verify.XXXXXX)
cleanup() {
  playwright-cli -s="$SESSION" close >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
# Keep all browser session files out of the repository.
cd "$WORK"
run() {
  local result
  result=$(playwright-cli -s="$SESSION" "$@" 2>&1) || { echo "$result" >&2; return 1; }
  printf '%s\n' "$result" >>"$OUT/cli.log"
  # playwright-cli can report a JS exception with a zero shell exit status.
  if grep -q '^### Error' <<<"$result"; then echo "$result" >&2; return 1; fi
}
run open about:blank --browser=chrome
python3 - "$PAGE" "$OUT" >"$WORK/check.js" <<'PY'
import json
import sys
from pathlib import Path
print('async page => {')
print('const url = ' + json.dumps(Path(sys.argv[1]).as_uri()) + ';')
print('const out = ' + json.dumps(sys.argv[2]) + ';')
PY
cat >>"$WORK/check.js" <<'JS'
const errors = [], requests = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
page.on('request', r => { if (!/^(file:|data:|blob:)/.test(r.url())) requests.push(r.url()); });
await page.context().setOffline(true);
await page.setViewportSize({width:1440,height:1000});
await page.goto(url);
await page.waitForFunction(() => window.reportReady === true, null, {timeout:30000});
async function check() {
  const state = await page.evaluate(() => {
    const plots = [...document.querySelectorAll('.plot')];
    const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
    const spec = JSON.parse(document.getElementById('report-data').textContent);
    const figs = spec.sections.flatMap(s => s.figures || []);
    function matches(actual, expected) {
      if (expected !== null && typeof expected === 'object')
        return actual != null && Object.keys(expected).every(k => matches(actual[k],expected[k]));
      return actual === expected;
    }
    const correctData = plots.every((p,i) => {
      const expected = figs[i].frames ? figs[i].frames[window.reportFrame] : figs[i].data;
      return p.data?.length === expected.length && matches(p.data,expected);
    });
    return {correctData,count:plots.length, rendered:plots.every(p => p.classList.contains('js-plotly-plot') && p._fullData?.length),
      duplicateIds:ids.length !== new Set(ids).size, error:document.querySelector('#error')?.textContent,
      overflow:document.documentElement.scrollWidth > innerWidth + 1};
  });
  if (!state.correctData || !state.count || !state.rendered || state.duplicateIds || state.error || state.overflow)
    throw new Error('Rendered page check failed: '+JSON.stringify(state));
}
await check();
await page.screenshot({path:out+'/desktop.png',fullPage:true});
const slider = page.locator('#checkpoint');
if (await slider.count()) {
  const max = Number(await slider.getAttribute('max'));
  for (let i = 0; i <= max; i++) {
    await slider.evaluate((el,index) => {el.value=String(index);el.dispatchEvent(new Event('input',{bubbles:true}));},i);
    await page.waitForFunction(index => window.reportReady && window.reportFrame === index,i,{timeout:30000});
    await check();
    const valid = await page.evaluate(index => {
      const spec = JSON.parse(document.getElementById('report-data').textContent);
      return document.getElementById('checkpoint-value').textContent === String(spec.checkpoints[index]);
    },i);
    if (!valid) throw new Error('Checkpoint label did not update');
  }
  await page.screenshot({path:out+'/last-checkpoint.png',fullPage:true});
}
await page.setViewportSize({width:390,height:844});
// ResizeObserver / Plotly responsiveness settles asynchronously.
await page.waitForTimeout(600);
await check();
await page.screenshot({path:out+'/mobile.png',fullPage:true});
if (errors.length || requests.length) throw new Error(JSON.stringify({errors,networkRequests:requests}));
console.log('VERIFY_PASS: all plots rendered, all checkpoints checked, no JS/console errors or network requests');
}
JS
run run-code --filename="$WORK/check.js"
grep -q 'VERIFY_PASS' "$OUT/cli.log" || { cat "$OUT/cli.log" >&2; exit 1; }
echo "PASS: offline rendering and checkpoint checks. Screenshots: $OUT"
