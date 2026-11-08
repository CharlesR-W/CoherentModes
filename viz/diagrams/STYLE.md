# Visual style for the CoherentModes diagrams and data figures

One system for the explanatory diagrams (D1–D7, this folder) and the data figures of
`pages/toy.html`. It is the `dataviz` skill's validated reference palette
(`references/palette.md`), used by *job*: identity, signed value, unsigned magnitude.
The values live in `make_diagrams.py` (top of file); change them there and regenerate.

## Surfaces and ink

| role | hex |
|---|---|
| chart surface (SVG background) | `#fcfcfb` |
| primary ink (labels, formulas, key marks) | `#0b0b0b` |
| secondary ink (notes, captions) | `#52514e` |
| muted ink (de-emphasised labels, axis text) | `#898781` |
| gridline / hairline | `#e1e0d9` |
| baseline, axis, sphere outline | `#c3c2b7` |

Text is always in ink, never in a series colour (a coloured mark beside it carries identity).

## Colour by job

**Signed values** (mode values ψ_i(x), loadings b, error loadings g, mode profiles, matrix
entries of g_i b_iᵀ): diverging, **blue `#2a78d6` (negative) — grey `#f0efec` (zero) — red
`#e34948` (positive)**, interpolated linearly in OKLab. The poles are categorical slots 1 and 8
of the reference palette at matched lightness (OKLCH L ≈ 0.59 / 0.62), so neither sign looks
heavier. Every figure scales symmetrically about zero; when one figure shows several signed
quantities, each has its own symmetric scale and the legend says so.

**Unsigned magnitudes** (kernel and operator entries, |g_k b_j| on edges, canonical
correlations κ_i): a **neutral ink ramp `#f0efec` → `#898781` → `#0b0b0b`** (OKLab). This
departs from the skill's default sequential hue (blue) on purpose: blue is already the
negative pole of the diverging scale, and D2 and D4 show signed and unsigned quantities side by
side, so a blue magnitude ramp would let "large" and "negative" be confused. On edges,
magnitude is also carried by stroke width.

**Identity** (which site group, which target): categorical slots in fixed order, **slot 1
blue `#2a78d6`, slot 2 orange `#eb6834`, slot 3 aqua `#1baf7a`**. These are only ever used in a
figure that has no diverging encoding (D5 groups A/B; D6 carried/read targets). Identity marks are
always direct-labelled.

Validation (`scripts/validate_palette.js` of the dataviz skill, run 2026-11-07):

- categorical, first three slots, `--pairs all`: PASS in light (worst CVD ΔE 9.2, normal-vision
  24.0) and dark (9.4 / 20.9). Light-mode aqua is 2.74:1 against the surface (WARN), relieved by
  direct labels; it is not used in the diagrams so far.
- each diverging arm and the neutral ramp, `--ordinal`: monotone lightness PASS, adjacent ΔL ≥ 0.06
  PASS, single hue PASS. The light-end contrast check fails (1.4–1.7:1), as expected and accepted:
  these are continuous scales whose near-zero end is meant to recede into the surface
  (palette.md: "the lightest step means near zero and is allowed to recede"); they are not
  ordinal marks.

Status colours are not used. Dark mode is not yet provided for the SVG diagrams: they carry
their own light surface, so they stay legible when embedded in a dark page. The dark steps of
the same palette (palette.md) are the ones to use if a dark variant is needed.

## Typography

- One family everywhere: `system-ui, -apple-system, "Segoe UI", sans-serif` (no web fonts, no
  serif, no display face; on this VPS it resolves to DejaVu Sans).
- Sizes in the diagrams (SVG px, diagrams are 1200 px wide): title 24 bold; panel headings 18–22
  bold; labels 18; secondary notes 16 (the floor, so text stays ≥ 8 px at 50% scale); caption 18
  in secondary ink, two lines, under a hairline.
- Mathematics is set in the same sans with real sub/superscripts (`<tspan>` offsets; markup
  `a_{ij}`, `a^{T}` in `make_diagrams.py`); typographic minus `−` in numbers. Symbols follow
  THEORY.md exactly (Section 16 glossary and Sections 1–7): sites x, h^L(x), K_β, P, ψ_i, λ_i,
  a_i = (H^L)ᵀψ_i, Q^{L→L′} = P^{L′}P^L, (u_i, v_i, σ_i), g_i b_iᵀ, blocks B, assumption tags [A·],
  [P·], [M·].

## Marks

- Thin strokes (1.2–2.4 px), round caps; arrows 1.6–2.2 px with solid heads drawn as polygons
  (no SVG markers, so every renderer agrees).
- Heatmap cells have a 1–2 px surface gap; matrix cells have 2 px radius.
- Spheres are drawn as an outline with a dashed equator only, without shading.
- No decoration: no drop shadows, gradients-as-ornament, icons or background textures.

## Data figures (pages/toy.html)

The same assignments apply in Plotly: `paper_bgcolor`/`plot_bgcolor` `#fcfcfb`, font family as
above at 13–15 px, gridlines `#e1e0d9`, axis lines `#c3c2b7`; diverging colourscale
blue–grey–red as above with `zmid = 0`; magnitude heatmaps on the neutral ink ramp;
categorical traces in slot order. Six planted clusters in a 3D scatter exceed the three-slot
all-pairs cap, so there they take slots 1–6 *with* direct cluster labels as secondary encoding
(the validator's WARN band requires it), or are faceted.

**3D figures: one WebGL scene per figure.** Every Plotly 3D scene is its own WebGL context and Chrome
keeps only about 16 alive per page (older ones are dropped, leaving blank panels). Small multiples of
spheres therefore share one orthographic scene, laid out as a grid of offset unit spheres
(`grid_scene`/`off` in `viz/build_page.py`), and rotate together. Pin numeric cartesian axes to
`type: "linear"`; Plotly's auto-typing occasionally made a numeric axis categorical on this page.

## Rendering and review

`uv run --with numpy python -I viz/diagrams/make_diagrams.py` writes the SVGs (and asserts the
identities each figure depicts); `python3 viz/diagrams/render_png.py [--half DIR]` renders PNGs
with headless Chrome, the renderer the page uses. (cairosvg mis-places centred text that contains
sub/superscripts, so it is not used.)
