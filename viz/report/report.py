"""Small offline Plotly report builder (standard library only).

Adapted from CellsDevelopmental: scripts/notebook.py (shell, layout, palette),
report/sidecar.py (safe JSON embedding), and its section assembly pattern.
Original report code has no declared license in that checkout; see ../TEMPLATE.md.
Plotly and its notices are kept verbatim in vendor/.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOTLY = HERE / "vendor/plotly-2.35.2.min.js"
PLOTLY_SHA256 = "6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603"

STYLE = """
* {box-sizing:border-box}
:root {--ink:#2b2a27;--dim:#6b665e;--border:#d3cdc3;--bg:#fdfcfa;--soft:#f6f3ee}
body {margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main {max-width:1100px;margin:auto;padding:26px 28px 70px}
h1 {font-size:30px;line-height:1.2;margin:4px 0 10px}
h2 {font-size:22px;margin:30px 0 8px}
h3 {font-size:16px;margin:0 0 8px}
p {max-width:78ch}
.subtitle,figcaption {color:var(--dim)}
figure {border:1px solid var(--border);border-radius:8px;background:white;padding:14px;margin:16px 0}
.plot {width:100%;height:420px}
figcaption {font-size:13.5px;margin-top:8px}
.controls {position:sticky;top:0;z-index:10;background:var(--soft);border:1px solid var(--border);
 border-radius:8px;padding:12px;display:flex;flex-wrap:wrap;align-items:center;gap:12px}
.controls input {flex:1;min-width:120px;accent-color:#4477aa}
pre {white-space:pre-wrap;overflow-wrap:anywhere}
#error {color:#a12020;white-space:pre-wrap}
@media(max-width:600px) {main{padding:18px 12px 40px}figure{padding:10px}.plot{height:360px}h1{font-size:25px}}
"""

SCRIPT = r"""
const REPORT = JSON.parse(document.getElementById('report-data').textContent);
const PALETTE = ['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377','#bbbbbb'];
function merge(a,b) {
  const out = {...a};
  for (const [k,v] of Object.entries(b || {})) {
    out[k] = v && typeof v === 'object' && !Array.isArray(v)
      ? merge(a[k] || {}, v) : v;
  }
  return out;
}
const AXIS = {gridcolor:'#eeebe5',zerolinecolor:'#e2ddd5',linecolor:'#d3cdc3',
  ticks:'outside',automargin:true,title:{font:{size:13}}};
function LAYOUT(extra) {
  return merge({margin:{l:64,r:32,t:36,b:60},paper_bgcolor:'white',plot_bgcolor:'white',
    font:{family:'system-ui,sans-serif',size:13,color:'#2b2a27'},colorway:PALETTE,
    xaxis:AXIS,yaxis:AXIS,scene:{xaxis:AXIS,yaxis:AXIS,zaxis:AXIS,camera:{eye:{x:1.8,y:1.8,z:1.5}}},
    legend:{orientation:'h',y:1.08,x:0},hovermode:'closest',uirevision:'keep-view'},extra);
}
const CONFIG = {displaylogo:false,responsive:true,showLink:false,
  modeBarButtonsToRemove:['lasso2d','select2d','sendDataToCloud']};
const figures = REPORT.sections.flatMap(s => s.figures || []);
const slider = document.getElementById('checkpoint');
let pending = Promise.resolve();
window.reportReady = false;
window.reportFrame = -1;
function draw(index) {
  window.reportReady = false;
  pending = pending.then(async () => {
    for (const [i,f] of figures.entries()) {
      const data = f.frames ? f.frames[index] : f.data;
      // Clone: Plotly may mutate its arguments; embedded frames remain pristine.
      await Plotly.react('plot-'+i, JSON.parse(JSON.stringify(data)), LAYOUT(f.layout), CONFIG);
    }
    if (slider) document.getElementById('checkpoint-value').textContent = String(REPORT.checkpoints[index]);
    window.reportFrame = index;
    window.reportReady = true;
  }).catch(error => {
    document.getElementById('error').textContent = 'Figure draw error: '+error.message;
    console.error(error);
    throw error;
  });
  return pending;
}
if (slider) slider.addEventListener('input', () => draw(Number(slider.value)));
draw(0);
"""


def embed_json(value: object) -> str:
    """Strict JSON, safe inside a script element even with literal </script>."""
    return json.dumps(value, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def write_report(spec: dict, output: str | Path) -> Path:
    """Embed a JSON-compatible report spec and vendored Plotly in one HTML file.

    Text is escaped, not interpreted as HTML. Plotly trace/layout dictionaries
    follow Plotly.js's schema. Checkpoints are labels; frames are indexed arrays
    of complete trace lists, one per checkpoint. Layout is fixed across frames.
    """
    sections = spec.get("sections", [])
    if not sections or not any(s.get("figures") for s in sections):
        raise ValueError("A report needs at least one section with a figure")
    checkpoints = spec.get("checkpoints")
    if checkpoints is not None and (not isinstance(checkpoints, list) or not checkpoints):
        raise ValueError("checkpoints must be a nonempty list")
    esc = lambda value: html.escape(str(value), quote=True)
    body = []
    number = 0
    for section in sections:
        body.append(f'<section><h2>{esc(section["title"])}</h2><p>{esc(section.get("caption", ""))}</p>')
        for figure in section.get("figures", []):
            if not figure.get("title") or not figure.get("caption"):
                raise ValueError("Every figure needs a title and caption")
            if ("data" in figure) == ("frames" in figure):
                raise ValueError("Provide exactly one of data or frames per figure")
            frames = figure.get("frames")
            traces = [figure["data"]] if frames is None else frames
            if frames is not None and (checkpoints is None or len(frames) != len(checkpoints)):
                raise ValueError("frames must have one trace list per checkpoint")
            if not all(isinstance(t, list) and t and all(isinstance(x, dict) for x in t) for t in traces):
                raise ValueError("data and each frame must be nonempty lists of Plotly traces")
            body.append(f'<figure><h3>{esc(figure["title"])}</h3>'
                        f'<div class="plot" id="plot-{number}"></div>'
                        f'<figcaption>{esc(figure["caption"])}</figcaption></figure>')
            number += 1
        body.append('</section>')
    controls = '' if checkpoints is None else (
        '<div class="controls"><label for="checkpoint">Checkpoint</label>'
        f'<input id="checkpoint" type="range" min="0" max="{len(checkpoints)-1}" value="0" step="1">'
        f'<output id="checkpoint-value" for="checkpoint">{esc(checkpoints[0])}</output></div>')
    payload = embed_json(spec)
    bundle = PLOTLY.read_bytes()
    if hashlib.sha256(bundle).hexdigest() != PLOTLY_SHA256:
        raise ValueError("Vendored Plotly hash differs from the pinned bundle")
    notices = '\n\n'.join((HERE / 'vendor' / name).read_text() for name in
                           ('Plotly-MIT.txt', 'plotly.min.js.LICENSE.txt'))
    title = esc(spec["title"])
    script_bundle = bundle.decode("utf-8").replace("</script", "<\\/script")
    page = (f'<!doctype html>\n<!-- Adapted from CellsDevelopmental report and notebook machinery. -->\n'
            f'<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{STYLE}</style></head><body><main><header><h1>{title}</h1>'
            f'<p class="subtitle">{esc(spec.get("subtitle", ""))}</p></header>{controls}'
            f'<p id="error" role="alert"></p>{"".join(body)}'
            f'<details><summary>Plotly license notices</summary><pre>{esc(notices)}</pre></details></main>'
            f'<script>{script_bundle}</script>'
            f'<script id="report-data" type="application/json">{payload}</script>'
            f'<script>{SCRIPT}</script></body></html>\n')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="JSON report specification")
    parser.add_argument("output", type=Path, help="Self-contained HTML output")
    args = parser.parse_args()
    print(write_report(json.loads(args.data.read_text()), args.output))
