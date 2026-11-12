# Reorganisation plan for the cifar_coherent site (draft v4)

## Owner's brief (fixed; not up for review)

- Audience: technical LessWrong readers (first venue: the owner's research org's internal Slack).
- Thesis: a **label-free procedure**, plus a **language** for describing what a network treats as alike. The
  method comes first. Using it to investigate questions about this one network comes second.
- The explorer is the centrepiece.
- The site is sorted by emphasis and by how settled each part is. It must not *look* long at a glance: the
  skimmer gets summary images, and long material lives in other tabs.
- Captions say exactly what is plotted, in terms of the definitions, and never pre-interpret. Interpretation
  is argued in its own section, with examples.
- Findings form a separate section that argues for its readings, qualitative ones included ("by step X the
  2-set split looks like vehicles vs animals"; "these sets seem to follow background or average colour").
  The section doubles as a worked example of how to think with the tool. **The owner sees the findings
  before anything is published.**
- Title: "Towards Characterizing a Neural Network's Native Ontology: Coherent Sets of Data Through Layers and
  Over Training". Subtitle: "or, Sparse Eigenbases of the Tempered Cross-Hankel Operator for Spectral
  Identification of the Kantian Categories by Neural Coherent Sets".
- Tone (owner, 2026-11-12): exploratory and not defensive. "Exploratory report" appears **once** at the top,
  and nothing is hedged after that. The confidence tags in Findings are light labels, not disclaimers.
  The limitations sections become one short **"Checks I'd like to run"** list (split-half resampling of the
  sets, a second seed's sets, other probes).
- Findings, this pass: what the method measures on this network, and class make-up over training (such as
  when the 2-set split becomes vehicles vs animals). The owner's own observations from the explorer
  (colour, background and the like) are **deferred, to work on together later**. Leave a stub section,
  build the tooling those readings will need (the `images` live grid and the image-statistics analysis),
  and do not write those readings.
- Preview: publishing directly, or to the laptop, is fine.
- First-person "I" stays. No ViT material. The kernel choice is methodology, not a headline.
- Project rules: labels, losses and gradients only colour and validate. Every number comes from build-time
  facts or analysis JSON. Nothing is pushed or committed without the owner's say.

## Problems with the current site (what the rework fixes)

1. The procedure is never stated in one place, in order.
2. Terms are used before they are defined (SEBA and UMAP in the first figure; retention, participation
   number, "static", "reliability", r).
3. The Overview mixes motivation, maths, the kernel sweep, a guide to the explorer and limitations. The
   kernel section is its longest part.
4. The findings exist only as captions in the Questions tab, and they are interpretations written into
   captions.
5. The Method tab repeats the Overview's maths and also tells the history of the kernel choice.
6. The joke title and the "Hankel" section take space away from the theory link.

## New structure: five tabs, ordered by centrality

| tab | role | settledness | target size |
|---|---|---|---|
| **Overview** (landing) | what the method is, why it is unusual, the procedure in steps, where to look | settled | ~5 min read; mostly figures |
| **Explorer** | the tool | settled | unchanged, plus guided tours |
| **Findings** | readings of this network, argued with evidence | tentative, each claim tagged | one screen per finding, details collapsed |
| **Measurements** | systematic charts (the current Questions tab), purely descriptive | descriptive | charts with definitional captions |
| **Method** | reference: exact definitions in pipeline order, the kernel choice, normalisation, SEBA, average vs product, layouts, reproduction | settled reference | long is fine |

### Overview (landing)

1. Title and subtitle. Then one lede paragraph: I gave a ResNet-20 the same 2,000 images at 8 checkpoints and
   12 sites, and asked which groups of images it treats alike, where in the network, and since when, without
   telling the method about labels.
2. **Hero:** a live `cells` strip of one site across all eight checkpoints, coloured by the SEBA sets of one
   block, with a large "Open the explorer" link to that exact state. The caption is definitional only.
3. **"What is unusual about this approach"**: five or six one-line desiderata, each with a short
   "because …":
   - label-free: nothing semantic defines the kernel or the basis;
   - each layer is judged in its own geometry, and no map between layers or between checkpoints is needed;
   - one object covers depth and training time alike (a block is a run of cells along either axis);
   - it yields *groups of data points*, not directions, with no sparsity assumption;
   - every step is a fixed computation or an eigenproblem, with no training on top of the network;
   - it is architecture-agnostic: it needs only activations on a fixed probe set.
4. **"The procedure in five steps"**, a numbered list. Each step has one sentence, one small visual from a
   redrawn pipeline strip, and a `<details>` with its maths:
   1. per cell, take cosine similarities of centred activations;
   2. turn them into a neighbourhood kernel and Sinkhorn-scale it into a symmetric random walk $P_c$, then
      set $M_c = (I + P_c)/2$;
   3. average the walks over a block of cells (consecutive layers, or consecutive checkpoints);
   4. take the leading eigenvectors of the average: the functions with the smallest *average* variation
      across the block's walks. Then show at once their **retention** in each cell, which marks the cells
      where a candidate distinction is weaker. Average is not "in every cell", and the text says so here,
      not only in Method;
   5. apply SEBA to the constant and the leading $r-1$ non-constant eigenvectors. This gives sparse
      approximations to that span (alternating rotation and thresholding, so the vectors leave the span)
      and, by hard assignment, **candidate coherent sets**, with some images unassigned. The rotation,
      thresholding, ordering and assignment details go in the step's maths disclosure.
   
   Then one paragraph naming the **vocabulary** (site, checkpoint, cell, block, eigenfunction, retention,
   coherent set, unassigned), with a link to a glossary.
5. **"Three things to look at"**: cards that lead with images, each linking to a Finding and a preset in
   the explorer.
6. (No separate status note: the one "exploratory report" line sits at the top.)

Moved out of the Overview: the kernel sweep (to Method), the Hankel section (to Method), the full
"reading the plots" guide (to the explorer's help panel and the glossary), and "extras" (to the explorer's
help).

### Explorer

Its behaviour does not change. Additions:
- a **Tours** menu: named presets, one per finding plus "the default view", that set the URL state and show
  a two-line definitional note;
- a short how-to on first visit, collapsed after that (the howto gif moves here from the Overview);
- control labels reworded to the glossary's terms (for example "static" becomes "single-cell block");
- a generated caption under the grid that states the block, rule, r and colouring precisely.

### Findings

Each finding uses the same template:
- **Claim** in one sentence, with a confidence tag: *observed* (a direct reading of a measured quantity),
  *reading* (an interpretation I would defend), or *speculation*.
- **Evidence**: one or two live embeds (cells strips, image grids) with definitional captions, an explorer
  preset link, and a few numbers taken from facts.
- **How I read it, and what else it could be**: the argument, alternative explanations, and which check
  would separate them.
- **Try it**: what to change in the explorer to probe the claim yourself.

Candidate findings, to be confirmed by analysis before writing. The list is provisional; any that fail are
dropped or turned into negative results:
1. The four late-layer sets at the end of training (b7 to pool), and what is in them, from class
   make-up and representative images.
2. Looking back through training: the time block at b9 and the r = 2 split at each checkpoint, to find the
   first checkpoint at which the 2-set split separates vehicles from animals. This needs a per-checkpoint
   class make-up of SEBA-2 sets.
3. Early layers and step 0: do the sets follow low-level image statistics (mean colour, luminance,
   saturation, border-vs-centre colour as a background proxy, high-frequency energy)? This needs a **new
   closed-form analysis**: correlations and set-mean contrasts of these statistics with SEBA vectors and
   $\psi_k$, computed from the probe pixels against permutation nulls, labelled as validation.
4. Persistence across depth: runs of sites with matching sets, and where the grouping is redrawn (the carry
   matrix).
5. A single-cell split against a block split: averaging prefers splits that later layers keep.
6. A negative or weak result: the label-free sets share only a little of the classes' gradient coherence.

Findings are written last, after the analysis exists, and go to the owner for review before any publishing.

### Measurements

These are the current Questions charts, plus the new image-statistics charts, with captions rewritten to the
caption policy. A short descriptive intro only, and no argument (Findings cites these charts).

### Method

Restructured to follow pipeline order, with a glossary at the top. Sections: data, model, probe and sites;
per-cell operator (with the "why Sinkhorn" details); **choosing the kernel** (the current Overview kernel
section, the sweep, the κ tables); blocks and retention; SEBA; what is canonical; average vs product (the
theory link and the cross-Hankel analogy, folded in briefly); layouts; per-image diagnostics; loadings;
statistics behind the Measurements tab; checks I'd like to run; reproduce; references. Prose that repeats the
Overview is cut.

## URL migration (the site is already published)

- `#questions` and `#questions/<id>` keep working: `questions` becomes an alias for the Measurements route,
  and the section ids are kept.
- Add an anchor-alias map in `js/main.js`: an old `#<tab>/<id>` whose id has moved (for example
  `#overview/kernel`, `#overview/hankel`, `#overview/reading`, `#overview/blocks`) is rewritten to the new
  tab and id with `history.replaceState`. An unknown id on a known tab falls back to that tab's top, not to
  the Overview.
- Ids that still exist keep their names. New ids do not reuse old ones with a different meaning.
- Browser checks: load representative old URLs directly; test back/forward navigation; test every Tours
  link and every `explore-link`.

## Caption policy

Each caption has a **Plotted:** part: what each axis, mark and colour is, as defined quantities with
symbols, plus the block, kernel rule, r, checkpoint(s), sort order and null. An optional **Values:** line
gives only the main quantities below. Captions contain no class make-up, no adjectives like "clear" or
"persists", and no "what it shows".

The main quantities to report wherever they apply:
1. the block spectrum $\lambda_1..\lambda_{r-1}$ and the gap $\lambda_{r-1}-\lambda_r$ (SEBA uses $\psi_0..\psi_{r-1}$);
2. retention $\rho_c(f)$ per cell (minimum and range across the block);
3. SEBA set sizes and the number unassigned;
4. matched Jaccard against the null band.

Everything else is named and defined, but not given as headline numbers.

## Figures

- Redraw the pipeline as a five-panel strip matching the five steps (from `make_figures.py`).
- New live type `images`: a grid of probe thumbnails for a set (the explorer already has representative
  images, so reuse its data). Findings need it.
- New charts for the image-statistics analysis (Measurements), produced by `analysis.py`.
- Keep `static_dynamic`, `seba` and `timeblock` (Findings and Method), with captions rewritten. Move the
  `rules`, `sweep` and `kappa` figures to Method.

## Work plan (swarm)

The owner's model choices: Opus 5.5 does all design and writing; GPT (Codex) does iterative critique; cheap
GPT handles mechanical work. Each file has one owner at a time.

| phase | who | output |
|---|---|---|
| 0 | me + `plan-reviewer` (iterate) | this plan, approved; then the owner's OK |
| 1a | Opus analyst | image-statistics analysis and SEBA-2 per-checkpoint make-up → `analysis.py` additions, `questions.json` keys, new facts; a notes file of candidate findings with numbers and alternatives |
| 1b | Opus figure/JS engineer | pipeline strip, `images` live type, Tours menu, explorer caption and label rewording, 5-tab shell |
| 1c | Opus writer | Method restructure and glossary |
| 2 | Opus writer | Overview (needs 1b's figures) |
| 3 | Opus writer | Findings and Measurements (needs 1a) |
| 4 | GPT-6.1 Sol critiques (same agent re-checks revisions); GPT-6.1 Terra fresh-eyes read for skimmability; GPT Luna for fact and link checks and the build | revisions |
| 5 | me | build, desktop and mobile screenshots inspected; findings and preview to the owner. **No publishing until the owner OKs it.** |

## Swarm conventions (phase 1 onward)

- Everything happens in the main working tree (`explore/cifar_coherent/`). No commits, no pushes, no
  publishing. Edit only the files you own (below). Read anything.
- Build with `uv run --with numpy --with pillow python site/build_site.py r20s0b --html-only`. Use a full
  build (without `--html-only`) only when assets or data change. Preview with
  `python3 -m http.server <port> --bind 127.0.0.1 -d viewer/site` on your own port, and
  `playwright-cli -s=<your-name>` from `/tmp`. Close both when done.
- Every number in prose comes from a `{{fact:NAME}}` or from a JSON the page reads. Never type a measured
  number by hand.
- Owner's tone rules: exploratory, said once at the top. No hedging, no defensive limitations. Captions follow
  the caption policy above.

### File ownership, phase 1

| agent | owns |
|---|---|
| ENG (engineering) | `index.html`, `explorer.html`, `build_site.py`, `js/*`, `css/*`, `DESIGN.md`, `content/tours.json` (schema and placeholders), placeholder `content/findings.html`; renames `content/questions.html` to `content/measurements.html` (content unchanged) |
| FIG (figures) | `make_figures.py`, `figdata.py`, `assets/fig/*` |
| ANA (analysis) | `analysis.py` (and any new analysis module), `assets/questions.json`, `FINDINGS_NOTES.md` |
| MET (Method writer) | `content/method.html` |

Phase 2 (lead writer): `content/overview.html`, `content/findings.html`, `content/measurements.html`,
`content/header.json`, and the content of `content/tours.json`.

### Contracts between agents

- **Facts from analysis:** ANA puts scalar numbers under `questions.json["facts"]` (flat
  `{name: value-or-formatted-string}`). ENG makes `build_site.py` merge them into the fact table as
  `q.<name>`.
- **Tours:** `content/tours.json` = `[{"id", "label", "hash": "#explorer?...", "note": "<two definitional
  lines, HTML allowed>"}]`. ENG renders a Tours menu in the explorer bar and supports
  `#explorer?tour=<id>`.
- **`images` live type** (ENG): `<figure class="fig live" data-live="images" ...>`, with the same block
  parameters as an explorer link plus `data-j` (set index, 0-based) or `data-k` with `data-end="pos|neg"`
  (extremes of an eigenfunction), and `data-n` (thumbnail count). It shows probe thumbnails using the
  explorer's representative-images logic.
- **Tab ids:** `overview`, `explorer`, `findings`, `measurements`, `method`. `questions` is an alias of
  `measurements`.
- **New section ids.** Overview: `why`, `procedure`, `step-1` to `step-5`, `vocabulary`, `look`. Method keeps
  its current ids, except `limits`, which becomes `checks`, and adds `glossary` (terms as
  `glossary-<term>`), `kernel` (choosing the kernel, absorbing the Overview's kernel section) and `reading`
  (how to read the layouts and colourings). Measurements keeps `when`, `carry`, `gradients`,
  `decodability`, and adds `va` (r = 2 split vs vehicles/animals), `classmix` (class make-up of sets) and
  `imgstats` (image statistics vs eigenfunctions and sets).
- **Anchor aliases for old URLs** (ENG implements):

  | old | new |
  |---|---|
  | `overview/treated-alike` | `overview/procedure` |
  | `overview/blocks` | `overview/step-3` |
  | `overview/through-training` | `overview/step-3` |
  | `overview/hankel` | `method/avgprod` |
  | `overview/kernel` | `method/kernel` |
  | `overview/reading` | `method/reading` |
  | `overview/panels` | `method/loadings` |
  | `overview/limits` | `method/checks` |
  | `method/limits` | `method/checks` |
  | `questions/*` | `measurements/*` |
- **Glossary terms** (fixed; use them everywhere, including the explorer UI): probe image, site,
  checkpoint, cell, block (layer block, time block), kernel rule, walk $P_c$, lazy walk $M_c$, block
  operator $\bar M$, eigenfunction $\psi_k$, spectrum, retention $\rho_c$, SEBA vector, coherent set,
  unassigned, single-cell block (replaces "static"), matched Jaccard.
- **Report back** in at most 250 words: files changed, how you verified, open issues, and anything another
  agent needs to know.

### Phase 2 ownership

LEAD: `content/overview.html`, `content/findings.html`, `content/header.json`, `content/tours.json`
(content). MEAS: `content/measurements.html`. ENG, ANA and MET keep their phase 1 files, for fix-ups only.
Analysis notes for writers: `site/FINDINGS_NOTES.md`.
