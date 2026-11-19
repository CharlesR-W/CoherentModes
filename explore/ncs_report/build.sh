#!/usr/bin/env bash
# Rebuild the joint ViT / Mamba report into viewer/site/ (about 2 minutes on the VPS, plus about 3 for the browser check). Reads the four studies'
# stored artifacts in place (read-only); writes only inside explore/ncs_report/. See NOTES.md.
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR=/tmp/ncs-mpl
UV="uv run -q --with numpy"
echo "== analysis (facts and chart data)"; $UV --with scipy python -I analysis_ncs.py
echo "== ViT measurements (questions.json)"; $UV --with scipy python -I site/analysis_vit.py
echo "== figures"; $UV --with matplotlib --with pillow python -I make_figures.py
echo "== Mamba explorer data"; $UV python -I build_mamba.py
echo "== ViT explorer data and the page"; $UV --with pillow python -I build_site.py vits_b2k --clean
if [ "${NCS_SKIP_BROWSER:-0}" != 1 ] && command -v playwright-cli >/dev/null; then
  echo "== browser check (console errors; no horizontal overflow at 390 px)"; ./check_browser.sh
else echo "== browser check skipped (NCS_SKIP_BROWSER=1 or no playwright-cli)"; fi
echo "== done: viewer/site/ ($(du -sh viewer/site | cut -f1))"
