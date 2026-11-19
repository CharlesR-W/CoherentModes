#!/usr/bin/env bash
# Browser check of the built page (run by build.sh after the build; skip with NCS_SKIP_BROWSER=1, or when
# playwright-cli is not installed). Serves viewer/site on a free loopback port, loads every tab and both explorers at
# 1440 x 900 and 390 x 844 and every tour at 390 x 844 (check_browser.js), and fails on any console or page error
# or on horizontal overflow at 390 px. Screenshots go to $NCS_SHOTS (default /tmp/ncs-shots-check).
set -euo pipefail
cd "$(dirname "$0")"
S=ncs-check
SHOTS=${NCS_SHOTS:-/tmp/ncs-shots-check}
mkdir -p "$SHOTS" /tmp/ncs-check
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
python3 -m http.server "$PORT" --bind 127.0.0.1 -d viewer/site >/tmp/ncs-check/http.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; (cd /tmp && playwright-cli -s=$S close >/dev/null 2>&1 || true)' EXIT
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break; sleep 0.2; done
TOURS=$(python3 -c "import json; print(json.dumps([t['id'] for t in json.load(open('site/content/tours.json'))]))")
sed -e "s|__BASE__|http://127.0.0.1:$PORT/|" -e "s|__TOURS__|$TOURS|" -e "s|__SHOTS__|$SHOTS|" check_browser.js > /tmp/ncs-check/run.js
cd /tmp
playwright-cli -s=$S open "about:blank" --browser=chrome >/dev/null
playwright-cli -s=$S run-code --filename=/tmp/ncs-check/run.js > /tmp/ncs-check/out.txt
python3 - <<'PY'
import json, sys
lines = open("/tmp/ncs-check/out.txt").read().splitlines()
if "### Result" not in lines:
    sys.exit("browser check: no result\n" + "\n".join(lines[-30:]))
res = json.loads(json.loads(lines[lines.index("### Result") + 1]))
for e in res["errors"]:
    print("console/page error:", e)
for o in res["overflow"]:
    print("horizontal overflow:", o)
if not res["ok"]:
    sys.exit("browser check failed")
print(f"browser check: {res['checked']} page loads, no console errors, no horizontal overflow at 390 px")
PY
