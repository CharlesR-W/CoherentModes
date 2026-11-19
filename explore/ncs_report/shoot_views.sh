#!/usr/bin/env bash
# Refresh the explorer screenshots used as "What we are looking at" boxes (site/assets/fig/view_<tour>.png) and their
# manifest (site/assets/fig/views.json). Needs a build (it bootstraps one if the screenshots are missing),
# playwright-cli and Chrome. Run build.sh after; its page check fails if a view is missing, two views are identical,
# or a view does not match the manifest written here.
#
# Robustness (REVIEW_2 objection 1): the server runs on a free loopback port chosen here (a fixed port can already be
# serving another site), and before each capture shoot_views.js asserts that the page is this report and that the
# explorer shows the tour's network, object or block, checkpoint, colouring and r / k. Any mismatch aborts the script.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
S=ncs-views
PORT=$(python3 -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
NCS_ALLOW_MISSING_VIEWS=1 uv run -q --with numpy --with pillow python -I build_site.py vits_b2k --html-only >/dev/null
python3 -m http.server "$PORT" --bind 127.0.0.1 -d viewer/site >/tmp/ncs-views-http.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; (cd /tmp && playwright-cli -s=$S close >/dev/null 2>&1 || true)' EXIT
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break; sleep 0.2; done
TITLE=$(python3 -c "import json; print(json.load(open('site/content/header.json'))['title'])")
curl -sf "http://127.0.0.1:$PORT/" | grep -qF "<title>$TITLE" || { echo "port $PORT is not serving this report" >&2; exit 1; }

# the views used by the page, and what each tour must show (parsed from the tours' hashes)
rm -rf /tmp/ncs-views && mkdir -p /tmp/ncs-views
python3 - "$PORT" "$TITLE" > /tmp/ncs-views/run.js <<'PY'
import json, re, sys, glob
port, title = sys.argv[1], sys.argv[2]
ids = sorted({m for f in glob.glob("site/content/*.html") for m in re.findall(r"view_([a-z0-9-]+)\.png", open(f).read())})
tours = {t["id"]: t for t in json.load(open("site/content/tours.json"))}
views = []
for i in ids:
    if i not in tours:
        sys.exit(f"view_{i}.png has no tour {i}")
    q = dict(x.split("=", 1) for x in tours[i]["hash"].split("?", 1)[1].split("&"))
    net = q.get("net", "vit")
    exp = {"net": net, "c": q.get("c", "seba"), "k": q.get("r") if q.get("c", "seba") == "seba" else q.get("k")}
    if net == "vit":
        exp.update(a=q["a"], b=q["b"], m=q.get("m", "0"), fam=q.get("fam", "layer"))
    else:
        exp.update(o=q["o"], ck=q.get("ck", "trained"), lay=q.get("lay", "main"))
    views.append({"id": i, "label": tours[i]["label"], "expect": exp})
js = open("shoot_views.js").read()
print(js.replace("__VIEWS__", json.dumps(views)).replace("__PORT__", port).replace("__TITLE__", json.dumps(title)))
PY
cd /tmp
playwright-cli -s=$S open "about:blank" --browser=chrome >/dev/null
playwright-cli -s=$S resize 1200 800 >/dev/null
OUT=$(playwright-cli -s=$S run-code --filename=/tmp/ncs-views/run.js); echo "$OUT" > /tmp/ncs-views/out.txt
python3 - <<'PY' || { echo "shoot_views: a view failed its check (see /tmp/ncs-views/out.txt)" >&2; exit 1; }
import json
lines = open("/tmp/ncs-views/out.txt").read().splitlines()
res = json.loads(json.loads(lines[lines.index("### Result") + 1]))
json.dump(res, open("/tmp/ncs-views/result.json", "w"))
if not res.get("ok"):
    raise SystemExit(f"view {res.get('view')}: expected {res.get('expect')}, the page showed {res.get('got')}")
PY
cd "$HERE"
uv run -q --with pillow python -I - "$HERE/site/assets/fig" /tmp/ncs-views/result.json <<'PY'
import hashlib, json, sys, datetime
from PIL import Image
out, res = sys.argv[1], json.load(open(sys.argv[2]))
man = {"written_by": "shoot_views.sh", "date": datetime.date.today().isoformat(), "viewport": [1200, 800], "views": {}}
for v in res["views"]:
    im = Image.open(f"/tmp/ncs-views/{v['id']}.png").convert("RGB")
    p = f"{out}/view_{v['id']}.png"
    im.save(p, optimize=True)
    man["views"][v["id"]] = {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "size": list(im.size),
                             "label": v["label"], "checked": v["got"]}
    print(f"view_{v['id']}.png {im.size[0]}x{im.size[1]}  {v['got']}")
hs = [m["sha256"] for m in man["views"].values()]
assert len(set(hs)) == len(hs), "two views are identical"
json.dump(man, open(f"{out}/views.json", "w"), indent=1)
print("wrote", len(hs), "views and views.json")
PY
