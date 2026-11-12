# Site design notes (for writers of content fragments)

The site is one page with four hash-routed tabs: `#overview`, `#explorer`, `#questions`, `#method`.
`build_site.py <tag> [layout_tag]` assembles `site/` into `viewer/site/` (deploy root, `.nojekyll`).

## How content is included

| fragment (writer-owned) | goes into | if missing |
|---|---|---|
| `site/content/overview.html` | Overview tab | a placeholder paragraph |
| `site/content/questions.html` | Questions tab | a placeholder with the default decodability charts |
| `site/content/method.html` | Method tab | a placeholder paragraph |
| `site/content/header.json` (optional) | page header: `{"title", "subtitle", "date", "joke", "short_title"}`; `"joke"` is the italic line under the title (set `""` to drop it; dropped automatically if the title or subtitle already says "joke"); `"short_title"` is the compact title in the tab bar | owner's title, today's date |

Fragments are plain HTML, no `<html>`/`<head>`/`<body>`, no `<script>`. They are inlined at build time into
`<article class="prose">`. Everything under `site/assets/` (figures from `make_figures.py` in `assets/fig/`,
`assets/questions.json` from `analysis.py`) is copied to `viewer/site/assets/`; reference it with relative URLs
(`assets/fig/pipeline.svg`), never absolute ones. Images get `loading="lazy"` automatically unless they set `loading`.

Fact names for per-rule numbers use both `rule<i>` (index in `meta.modes`) and a slug of the rule name (e.g. `kappa.rule3.final.mid_median` = `kappa.knn_scale.final.mid_median`); rules without a kappa give `n/a`.

Build-time numbers: write `{{fact:NAME}}` anywhere in a fragment and the build substitutes the value; it fails
and lists the known names if NAME is unknown. `python3 site/build_site.py <tag> --facts` prints all facts (kappa
and 15-NN mass medians per rule, accuracies, checkpoint list, ...).

## Type and layout

- Prose: Source Serif 4, 18 px, measure ~ 68 ch (`.prose` column, 700 px). UI and figure text: Inter. Code: JetBrains Mono.
- Headings: `h2` for sections (each gets an anchor link automatically if it has an `id`), `h3` for subsections.
  Give every `h2`/`h3` an `id`; link to it from anywhere as `href="#overview/kappa"` (tab, slash, id).
- `<p class="lede">`: the opening paragraph, slightly larger.
- `<p class="byline">`: small grey line, e.g. under a section title.

## Patterns

Figure with caption (captions say how to read the figure, then what it shows):
```html
<figure class="fig">
  <img src="assets/fig/pipeline.svg" alt="Activations to cosine Gram to kernel to operator ...">
  <figcaption><b>How to read.</b> Each panel ... <b>What it shows.</b> ...</figcaption>
</figure>
```
Width modifiers on `figure.fig`: none (prose width, 700 px), `wide` (up to 1040 px, centred, breaks out of the
column), `full` (up to 1400 px). Side by side: wrap figures in `<div class="fig-row">` (stacks below 760 px).

Gif with a play/pause control (starts paused, on the still, when the reader prefers reduced motion):
```html
<figure class="fig wide">
  <img class="gif" src="assets/fig/howto.gif" data-still="assets/fig/howto_still.png" alt="...">
  <figcaption>...</figcaption>
</figure>
```

Maths: KaTeX auto-render. Inline `$\psi_k$` or `\(\psi_k\)`; display `$$ ... $$` or `\[ ... \]`.
Numbered display equation: `<div class="eq" id="eq-penalty" data-num="1">$$ ... $$</div>`; refer to it with
`<a href="#overview/eq-penalty">(1)</a>`. A literal dollar sign is `\$`.

Callouts (`p.callout-title` optional):
```html
<aside class="callout">...</aside>              <!-- neutral note -->
<aside class="callout key">...</aside>          <!-- the one thing to take away -->
<aside class="callout caveat">...</aside>       <!-- limitation / honest caveat -->
```

Side note (a margin note at >= 1240 px wide, an inline numbered expandable note below that):
```html
... the kernel is nearly flat<span class="sidenote">At kappa = 1.2 the kernel weight at the median
pairwise distance is still e^{-1.2} = 0.3.</span> in the middle layers ...
```

Collapsible detail: `<details class="more"><summary>Why Sinkhorn?</summary> ... </details>`.

Tables: `<table class="data">` (numbers right-aligned with `class="num"` on `td`/`th`).

Deep link into the explorer (opens the Explorer tab in that state):
```html
<a class="explore-link" href="#explorer?fam=layer&t=7&a=3&b=9&m=2&c=seba&r=4">see this block in the explorer</a>
```
Parameters (all optional): `fam` = `layer` | `time`; `t` = checkpoint index (layer blocks) or site index (time
blocks); `a`, `b` = first and last site index (layer) or checkpoint index (time), 0-based, see `meta.json`;
`m` = kappa mode index; `c` = `seba` | `nodal` | `eig` | `class` | `loss` | `correct` | `jac` | `gcos` | `gdot`;
`k` = eigenfunction 1..8; `r` = SEBA rank (2, 3, 4, 6, 8); `j` = focused set (0-based); `at` = `own` | `final`
(for per-image diagnostics); `lay` = `0` | `1` (which seed's UMAP lays out the points).

## Live mini-figures

A `figure.fig.live` with a `data-live` attribute is rendered by the site's JS when it scrolls into view (data
are fetched lazily). Put a `<figcaption>` inside it as for any figure; the JS inserts the chart before it.

`data-live="cells"`: a strip of UMAP cells coloured by a block (same renderer and hover as the explorer).
```html
<figure class="fig wide live" data-live="cells" data-cells="7:0,7:4,7:8,7:11"
        data-fam="layer" data-t="7" data-a="4" data-b="9" data-m="2" data-c="seba" data-r="4">
  <figcaption>...</figcaption>
</figure>
```
`data-cells` lists `checkpoint_index:site_index` pairs (default: the block's own cells). Colour parameters are
the explorer link parameters above. `data-size` = cell size in px (default 150).

`data-live="kernel"`: interactive kappa slider showing vMF kernel weight exp(kappa (c - 1)) against cosine
distance, with the measured 15th-neighbour and median pairwise distances of one cell marked.
Attributes: `data-t`, `data-s` (cell), `data-m` (initial rule).

`data-live="lines"` and `data-live="heatmap"`: charts from a JSON object, normally in `assets/questions.json`.
`data-src="assets/questions.json#path.to.object"` (dot path into the JSON).
- lines object: `{"x": [...], "xlabel": str, "ylabel": str, "xscale": "linear"|"log"|"symlog",
  "yrange": [lo, hi] (optional, fixed axes preferred), "series": [{"name": str, "y": [...], "dash": bool?,
  "band": [[lo...],[hi...]]?}], "colors": "ordinal"|"categorical"|"classes" (default categorical; ordinal = one
  hue light to dark, for ordered series such as checkpoints or sites), "refline": {"y": v, "label": str}?}`.
  Hover shows every series' value at the nearest x.
  Optional on the figure: `data-ceiling-src="assets/questions.json#q_when.0_4.vs_class_ceiling"` draws a horizontal
  reference line at the maximum finite value of that object's `z` (or `y`), labelled by `data-ceiling-label`.
- heatmap object: `{"rows": [...], "cols": [...], "z": [[...]] or [[[...]]], "zlabel": str, "zrange": [lo, hi],
  "rowlabel": str, "collabel": str, "slider": {"label": str, "values": [...]}?, "diverging": bool?}`; with a
  slider, `z` is indexed `[slider][row][col]`. Hover shows the value; the slider steps through e.g. checkpoints.

Measurement JSON is also served: `d/decod.json`, `d/curves.json`, `d/avgprod.json`, `d/normalisation_compare.json`, `d/sweep.json` (when present), `d/meta.json`, `d/facts.json`; a lines/heatmap object can live in any of them, but the usual place is `assets/questions.json` from `analysis.py`.

`data-live="decod"` and `data-live="curves"`: the default decodability and training-curve charts (no attributes).

## Colour grammar (tokens in `site/palette.json`, used by JS, CSS and `make_figures.py`)

- `classes`: 10 CIFAR classes (cool hues = vehicles, warm/green = animals). Labels only ever colour and validate.
- `sets`: SEBA sets in reliability order (set 1 = most reliable); `unassigned` grey.
- `diverging` (`neg`, `mid`, `pos`): eigenfunctions and signed loadings; nodal domains use `neg`/`pos`.
- `sequential`: one-hue ramp, light to dark, for magnitudes (loss, Jacobian norm, R^2, Jaccard).
- `ordinal`: one-hue ramp for ordered series (checkpoints, sites) in line charts.
- Ink and surfaces: `fg`, `mute`, `line`, `bg`, `surface`. Text never takes a data colour.
CSS custom properties with the same names (`--c-class-0`, `--c-set-0`, `--c-neg`, ...) are available in fragments,
e.g. `<span class="swatch" style="--sw: var(--c-set-0)"></span>` for an inline key swatch.

## Building and previewing

```bash
uv run --with numpy --with pillow python site/build_site.py <tag> [--layout-tag r20s1b]   # full build (incremental)
uv run --with numpy --with pillow python site/build_site.py <tag> --html-only             # after editing content/, css/, js/, assets/: ~1 s
python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/site                               # http://127.0.0.1:8765/
```
Block chunks are rebuilt only when their source `.npz` changes (`--clean` forces everything). `--facts` lists the
`{{fact:NAME}}` values of the last build. The page works only over HTTP (it fetches data), not from `file://`.
