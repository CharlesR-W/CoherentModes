# Self-contained Plotly report template

The engine is `report/report.py`; it uses only Python's standard library. Figures
are drawn in the browser from embedded numbers, not screenshots. The output is
one movable HTML file: open it directly with `file://`, without a server or
network connection. Plotly 2.35.2 and its license notices are embedded too.

## Build and verify

From the repository root:

```sh
python3 viz/report/example_build.py
bash viz/report/verify_page.sh viz/report/example.html /tmp/report-shots
```

The example has a 3D scatter, heatmap, line plot, and three-position checkpoint
slider. It is deliberately labelled **MOCKUP**: these are invented values, not
experimental results. No GIF/filmstrip is included because this is a control and
rendering demonstration, not evidence about training evolution.

For real data, have the measurement/analysis code produce a JSON specification:

```sh
python3 viz/report/report.py /path/to/data.json /path/to/notebook.html
bash viz/report/verify_page.sh /path/to/notebook.html /tmp/notebook-shots
```

Alternatively import `write_report` from a builder next to `report.py` and call
`write_report(spec, output_path)`. The example writes `example.html` as requested;
real synthetic notebooks should use `MOCKUP` in their filename as well as title.

## Add a section or figure

The JSON file **is the report specification**, not a project-specific artifact
format. Adapt an existing artifact to this shape in a small builder; do not add
model or measurement knowledge to the engine.

```json
{
  "title": "MOCKUP · minimal report",
  "subtitle": "Synthetic illustration, not measured results.",
  "sections": [{
    "title": "A question this section answers",
    "caption": "Section-level prose introducing the readout.",
    "figures": [{
      "title": "MOCKUP · scalar readout by step",
      "caption": "Each point is an invented value. This demonstrates declaration syntax, not a result.",
      "data": [{"type": "scatter", "mode": "lines+markers", "x": [0, 1], "y": [1, 0.5]}],
      "layout": {
        "xaxis": {"title": {"text": "Step"}, "range": [0, 1]},
        "yaxis": {"title": {"text": "Readout (arbitrary units)"}, "range": [0, 1.1]}
      }
    }]
  }]
}
```

Append another object to `sections` to add a section, or to a section's `figures`
to add a figure. A prose-only section can omit `figures`; the whole report must
have at least one figure. Text fields are escaped plain text, not HTML/Markdown.
Each figure requires a title and caption. Use standalone titles (what, where,
checkpoint), labelled axes with units, and two-sentence captions explaining how
to read the figure and what it shows. State sample/seed counts and band meanings
when applicable, deriving them from the saved data rather than typed constants.

`data` is a list of Plotly.js trace dictionaries; `layout` is an optional Plotly
layout dictionary, recursively merged with the shared defaults. Multiple traces
in one list overlay in one plot; multiple figure objects create separate plots.
There are no Python Plotly, NumPy, CDN, or project-package dependencies.
Nonfinite numbers are rejected: use JSON `null` for intentionally missing values,
and disclose missingness in the caption. No quantization or clipping is applied.

### Shared checkpoint slider

Add top-level `"checkpoints": [0, 10, 20]`. For a changing figure replace `data`
with `frames`: a list of complete trace lists, **one per checkpoint, in order**.
For example `"frames": [[{"type":"scatter","x":[0],"y":[1]}],
[{"type":"scatter","x":[0],"y":[2]}]]` goes with two checkpoints.
A figure must declare exactly one of `data` or `frames`. Static `data` figures
can coexist with frame figures. The one shared slider redraws all figures from
the embedded data; no fetches occur. Layout is not animated. The engine preserves
user zoom/camera via `uirevision`; use reset view to restore the declared axes.
There is intentionally no play loop or independent per-figure control framework.

## Layout and palette conventions

- Warm off-white page, white figure cards, system fonts, muted caption text;
  one readable vertical column, responsive to mobile widths. Shared constants
  are `STYLE`, `LAYOUT`, `AXIS`, `CONFIG`, and `PALETTE` in `report.py`.
- Categorical palette inherited from the source notebook:
  `#4477aa`, `#ee6677`, `#228833`, `#ccbb44`, `#66ccee`, `#aa3377`, `#bbbbbb`.
  Assign hues consistently to conceptual categories; encode variants with dash
  or symbol as well as color. Trace order otherwise determines default hue.
- The example uses Viridis for sequential heatmap values. For signed quantities
  choose a diverging scale and explicitly document its zero/midpoint.
- Fix `xaxis.range`, `yaxis.range`, `scene.*axis.range`, and heatmap `zmin`/`zmax`
  across frames and comparable panels. **The builder does not infer these ranges**;
  calculate them from the whole dataset in the caller. Print any cap. Name the
  algorithm for any derived 2D/3D placement; do not imply synthetic coordinates
  are a measured embedding.

## Headless verification

`verify_page.sh [page.html] [output-directory]` defaults to the example and a new
`/tmp/report-shots.*` directory. It uses the globally installed `playwright-cli`
and headless Google Chrome, a unique browser session, and temporary working
files outside the repo; its browser is closed on exit. No always-on server.

It enforces the 15 MB page budget, opens the file with the browser offline,
checks every figure rendered with nonempty traces, checks unique DOM IDs, and
moves through **every checkpoint** checking plotted trace values against the
embedded specification and the checkpoint label. It also checks for horizontal
page overflow, JavaScript/console errors, and any external resource requests.
CLI-reported JavaScript exceptions fail even if the CLI exits with status zero.
It saves `desktop.png`, `last-checkpoint.png` (when a slider exists), `mobile.png`,
and `cli.log`. Inspect **every screenshot** before declaring a page finished:
DOM success cannot detect clipped WebGL axis labels or a misleading caption.
It does not establish scientific correctness, validate all Plotly attributes,
or replace hover/rotation inspection. Geo/map traces that fetch tiles and
external image/font URLs are outside this offline template's scope.

## Source, provenance, and deliberate removals

Adapted from `/home/ubuntu/code/CellsDevelopmental`:

- `scripts/notebook.py`: page/figure/caption style, recursive layout merge,
  palette, shared control-state idea and Plotly.react pattern.
- `src/cells_developmental/report/sidecar.py`: safe script-embedded JSON idea
  (hardened here by escaping all `<` characters and rejecting NaN/Infinity).
- Section assembly patterns in `scripts/build_pages.py`,
  `scripts/build_diffusion_geometry_page.py`, and `scripts/build_effective_loss_page.py`.
- `scripts/verify_diffusion_geometry_page.sh`: page-size budget, rendered-DOM
  checks and screenshots, adapted here to `playwright-cli` and offline checks.

Reviewed the source store/payload/sections/sidecar machinery, page output,
`AGENTS.md`/`CLAUDE.md`, report tests, and the `experiment-notebooks` skill.
The source tests motivate safe payload embedding, unique IDs, explicit empty-page
failure and actually running JavaScript in a browser. Their measurement-specific
contracts are **not** carried over or claimed for this smaller engine.

Removed model/substrate names, measurement loaders and part naming, trained
artifacts, experiment-specific sections, quantized payload schemas, sidecars and
fetch loaders, GitHub pins, Modal/orchestration code, family navigation/tabs,
architecture diagrams, equation/MathJax/CDN support, and unused controls/helpers.
These belong to the later experiment builder, or conflict with the requested
small, fully offline, single-file output. No measurement code runs in this engine.

### Vendoring and licensing

This source checkout does **not** actually contain a vendored Plotly JS file:
`vendor/` has other projects, and `scripts/notebook.py` uses the CDN. Therefore
`report/vendor/plotly-2.35.2.min.js` was downloaded verbatim from the exact URL
pinned by that source engine: `https://cdn.plot.ly/plotly-2.35.2.min.js`.
SHA-256 (also checked on every build):

```
6d21266ce1bd7d9e5ab4e115989c70c20de0382fd973a8f26ab58619eba4d603
```

`report/vendor/Plotly-MIT.txt` is copied verbatim from the source repository's
`LICENSES/Plotly-MIT.txt`. `report/vendor/plotly.min.js.LICENSE.txt` is the
third-party notice file referenced by the bundle, downloaded verbatim from
`https://cdn.plot.ly/plotly-2.35.2.min.js.LICENSE.txt`. Both notices are also
included in the generated page, so moving the HTML does not discard them.

No root license or explicit license for the original report Python/JS code was
found in this checkout. Its `LICENSES/Apache-2.0.txt` accompanies unrelated
vendored code; it is not asserted to govern this adaptation and was not copied.
Source attribution is retained in the engine and generated HTML. Attribution is
not a substitute for a license grant: confirm the source report code's licensing
before redistributing this derivative outside the owner's repositories.
