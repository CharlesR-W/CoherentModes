"""Render viz/diagrams/D*.svg to PNG with headless Chrome (the page's real renderer).
    python3 viz/diagrams/render_png.py [--half DIR] [names...]
--half DIR also writes 50%-scale renders to DIR (legibility check)."""
import glob, os, re, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
half = None
if "--half" in args:
    i = args.index("--half"); half = args[i + 1]; del args[i:i + 2]
def shot(svg, png, scale):
    head = open(svg).read(400)
    w, h = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', head).groups())
    subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={scale}", f"--window-size={w},{h}",
                    f"--screenshot={png}", "file://" + svg], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for svg in sorted(glob.glob(os.path.join(here, "D*.svg"))):
    if args and not any(a in svg for a in args):
        continue
    shot(svg, svg[:-4] + ".png", 1)
    if half:
        os.makedirs(half, exist_ok=True)
        shot(svg, os.path.join(half, os.path.basename(svg)[:-4] + ".png"), 0.5)
    print("rendered", os.path.basename(svg)[:-4] + ".png")
