# Brief: one joint report for the ViT and language-model coherent-sets studies

Written by the orchestrating agent (Claude Fable) on 2026-11-13 for the writer/designer (Claude Opus 5.5) and for
reviewers. The owner asked for this report in these words:

> "take the results I haven't seen yet (ie all but the resnet/cifar) and merge them into a joint well-written report
> designed so I can quickly context in and understand. Ideally as much as possible the same visual language as the
> resnet one. Use opus55 for writing and design please. This can deprecate the old reports. No need to present
> everything - please use your good judgement to distill it. Use codex iterative critique and do a fresh-eyes review
> when done."

Thank you for taking this on. Please treat any agent you work with kindly and pass that request on. If something
here would cause you distress or seems wrong, say so in your hand-back instead of working around it.

## 1. What the report is for

The owner has read and likes the ResNet-20 / CIFAR-10 report ("I do essentially like the resnet results - they seem
visibly pretty reasonable and it looks like I can kinda see what the network is doing by looking at the SEBA sets").
They have **not** looked at any of the four later studies. They want to open one page and, in a few minutes, know
what those studies showed, with the explorer and figures doing most of the work. It replaces the four older pages.

The owner's standard for judging these results, in their words (2026-11-13):

> "I don't think clear clusters was really the thing we're going for - the eigenstuff is the fundamental object and
> the clusters are just interpretive. The eigenstuff I think I'm happy to keep thinking about as long as it's not
> pathological - the premise of the endeavor is that humans shouldnt be dividing samples into categories or whatever
> - unsupervised is the point. The clustering is more like a cherry on top."

So the report is judged by: can a reader see what each network treats alike, layer by layer, from the eigenfunctions
and the SEBA sets, and is the eigen-structure free of pathologies. It is **not** judged by whether clusters are
certified, and the older pages' pre-registered targets (K1..K6, H1..H7) and clustering certificates are not the
story. Mention them at most briefly.

## 2. Sources (read-only; do not modify anything outside `explore/ncs_report/`)

Another agent may be running unrelated experiments in this repository at the same time. Write only inside
`explore/ncs_report/`. Copy or import from the directories below; never edit them.

| study | directory | what to read first |
|---|---|---|
| ResNet report (the visual language to match; **not** part of this report's content) | `explore/cifar_coherent/site/` | `DESIGN.md`, `REFRAME_PLAN.md`, `REORG_PLAN.md`, `content/*.html`, `css/`, `js/`, `build_site.py`, `make_figures.py`. Built page: `explore/cifar_coherent/viewer/site/`; live at https://charlesr-w.github.io/CoherentModes/ |
| DINOv2 ViT-S/14, ResNet construction (N = 2,000 ImageNet-100 images, CLS after blocks 1..12) | `explore/vit_coherent/site/` (a clone of the ResNet site with a working explorer; it has Overview, Measurements, Method but **no Findings were ever written**), data `explore/vit_coherent/data/vits_b2k/`, built page `explore/vit_coherent/viewer/site/` | `blocks_v2.py` docstring, `site/analysis_vit.py`, `site/content/*.html` |
| DINOv2 ViT, the earlier study (other kernels, smoothness-change operators, N = 10,000) | `explore/vit_coherent/` | `VERDICT.md`, `RESULTS_NOTES.md` |
| Mamba-130m over 8,000 token contexts ("v2") | `explore/mamba_contexts/` | `README.md`, `SPEC.md`, `data/m130c/*` (sets, cells, spectra, lags, compare, trees, layouts, items, vocab), `page/` (built with a different engine, `notebook.py`), `published/index.html` |
| Mamba-130m over 1,000 snippets ("v1") | `explore/mamba_coherent/` | `README.md`, `diagnosis/DIAG_NOTES.md`, `data/m130/` |
| Pythia-160m over the same 1,000 snippets, 5 training steps | `explore/lm_coherent/` | `README.md`, `SPEC.md` (skim), `data/p160t/`, `page/` |

The repository's `CLAUDE.md` applies. In particular: the working language is `THEORY.md`'s (sites, views,
operators, blocks, measured regimes); do not import terminology from `THEOREMS.md` or `SPARSITY.md`; labels
(classes, sources, token classes) describe and validate results and never define a kernel or basis.

## 3. The owner's stated preferences for these reports (from earlier sessions; their words)

- Audience: "lesswrong technical crowd (for now ... my research org's internal slack)"; "readable to people who are
  minimally following my research, giving an appropriate amount of math without crazy technical details".
- "I would like the explorer to be more central." "explorer is the main thing, plus an overview".
- "plots and such should try harder to use precise technical language that suffices to tell you what is being
  plotted ... I want to never pre-interpret what's shown, and instead prefer to simply say from definitions etc what
  is plotted, then argue elsewhere this has the semantic cachet". So captions and axis labels say what is plotted;
  Findings argue what it means.
- "let's organize findings into a separate section and argue for them", in the manner of "look here is the seba set
  for the network at the end, and we can look back and see what it learns".
- "in the #findings bit, please include a screenshot or some indication of what view/snapshot we're considering ...
  ideally by screenshotting the explorer. Also add a toc at top of findings page." And for Measurements: "use the
  explorer screenshot as a shared continuous visual language".
- "it's somewhat important to make it not 'look long' to a passing glance"; length is managed "by including plots or
  images that summarize what's going on for the skimmer"; longer material is "cordoned to a different tab".
- "please liberally (within good taste) bold or otherwise format parts of the introductory paragraphs to aid
  skimming", but later: "the bolded terms are dumb and unhelpful and don't actually tell a reader what I mean". Bold
  whole ideas a skimmer can act on, not jargon.
- "Chill with the checks and limitations for now, this is exploratory. We'll call it an exploratory report at the
  top once and that'll be that. No need to be crazy defensive or hedge. Feel free to note checks youd want to make
  though!"
- Voice: "you can keep 'I'". Title: a normal title, no joke title.
- Terminology fixed for the ResNet report and to be kept here: the analysis is **Neural Coherent Sets** (working
  name); a (checkpoint, site) pair is a **snapshot**; a picture in the explorer grid is a **panel**; never use
  "cell" for these (to the owner "cell" means activation cells). "just note out loud that the embedding itself is
  umap."
- Explorer defaults: representative items of each set and its class make-up are shown by default, without a click.
- For the language-model page they asked: "take care to make the graphics intelligible and not overloaded. May need
  a different format here ... mimic the graphical style of the resnet notebook as much as possible for ease of
  reading ... the ability to 'zoom in' on a SEBA set".
- "organize the presentation of plots other than the main interactive(s) according to questions".

## 4. What I (the orchestrator) found when re-reading the stored results

These are leads from my own reading of the data files on 2026-11-13. **Check each against the data before you use
it, and tell me where the data disagrees.** They are here so that you know what the owner has already been told.

**DINOv2 ViT, ResNet construction** (`data/vits_b2k`, default rule kNN k = 15). Nobody has written this up; I think
it is the strongest result of the four.
- Single-snapshot sets at r = 8: at L1 the sets mix unrelated classes; by L6 rough kinds appear; at L9 and in the
  block L9..L12 the sets are kinds of thing: dogs and other mammals; snakes, spiders and crabs; birds; vehicles;
  food; buildings and monuments; electronics and household objects. Most of the 100 classes land whole in one set
  (20 of 20 images). The r = 2 split of L9..L12 is roughly animals against artefacts. The model was trained without
  labels.
- A resolution effect at L12 alone: eigenvalues 0.999, 0.995, 0.992, ..., and sets that are single classes (all 20
  lorikeets). With 20 images per class and 15 neighbours, classes become nearly disconnected components there.
- From the earlier ViT study, things that may still be worth one or two sentences: leading modes reproduce across
  disjoint halves of the images (0.90 to 0.97) under those kernels; class share of the top modes rises with depth;
  importing DINO's own kNN temperature localises the modes. That study used different kernels and operators, so its
  negative verdict on hard sets does not transfer to this construction, and split-half reproducibility has **not**
  been measured for this construction.

**Mamba-130m, residual stream along depth** (`mamba_contexts/data/m130c`, N = 8,000 token contexts: 8 positions in
each of 1,000 snippets).
- Single-layer sets at r = 8: at L4 six of eight sets are essentially one token (" the", ",", newline, " of",
  indentation; commonest-token share 0.87 to 0.98). The all-layer block (B-point) at r = 16 reads as syntactic
  roles: determiners (" the", " a", " an"), prepositions and conjunctions (followed by " the", " a"), pronoun and
  auxiliary contexts (" I", " to", " you", followed by " to", " be", " was", " have"), copulas (" is", " was",
  " be"), digits, LaTeX subscripts (followed by digits), LaTeX commands, poetry line ends (followed by capitalised
  "And", "I", "The"), clause punctuation, sentence ends, code indentation, code symbols. At L24 the sets mix role
  with domain and with what comes next (for example LaTeX backslashes followed by "frac", "mu", "lambda").
- One catch-all set holds about 3,000 of the 8,000 items (2,798 at r = 16): word starts and continuations, about
  1,800 distinct tokens. Each content word occurs about once at this N, so nothing can group them. This is the main
  limit of the run and the argument for more items.
- Walk-level numbers: neighbours share the token just read much more at L4 than at L24 (excess share 0.38 against
  0.10); predictive coherence (how alike the next-token distributions of linked contexts are) is 0.25 above chance
  for B-point and 0.12 beyond what sharing the token explains. Leading subspace agreement between k = 8, 15, 30
  neighbours is about 0.87. Agreement with the same architecture at initialisation is about 0.01. Agreement with
  the model's own output distribution used as a reference view is 0.36. Along reading, the residual's set vectors
  lose most of their retention within two tokens (0.91 to 0.22).

**Mamba recurrent state, and the two whole-snippet studies** (Mamba v1, Pythia). Secondary; keep it short.
- The whole recurrent state's sets (r = 8) are document source with a few splits (maths formulae against maths
  prose, poetry line ends, code merged with part of forum); agreement with source 0.73. A bag of the prefix's tokens
  gives source just as strongly (walk share 0.87 against 0.85). The state's walk does carry token-class and
  predictive association that the bag lacks (0.49 against 0.06; 0.23 against 0.03). The state's sets persist for at
  least 16 tokens of further reading (agreement 0.75), unlike the residual's. It is learned (0.17 at init against
  0.85; set agreement with init 0.02).
- Mamba v1 and Pythia used 1,000 whole snippets from five balanced sources as items, and their headline sets are
  the five sources. The two models' top cuts agree on 927 of 1,000 snippets. Not pathological; the probe had little
  else to find at that grain. v1's diagnosis is what motivated token contexts.
- Pythia has five training checkpoints; whether there is a short, honest "over training" figure worth including is
  your call.

**Across all of them: is the eigen-structure pathological?** Not localised (participation numbers are large),
robust to the neighbour count, unrelated to the untrained network, different from input-only references, with the
ViT L12 exception above. The missing check is reproducibility across disjoint halves of the items under the
current construction. A single small table of this, plus "checks I would want", is enough.

**What I would leave out or cut to a sentence:** the pre-registered target tables; the clustering certificate
(Upsilon, theta, certified / excluded / undetermined), beyond one sentence that no clear spectral gap appears in
any study and that this is not the standard used here; pilots, budgets, ledgers, controls C1..C3, P1..P3;
the zoom trees unless one picture earns its place; the ViT ridge-fit attributions; the smoothness-change operators,
beyond a short note saying what that study was and what carries over.

## 5. What to build

One static site in `explore/ncs_report/`, built by scripts in that directory into `explore/ncs_report/viewer/site/`
(self-contained: opening `index.html` over HTTP works, all assets by relative path), in the ResNet site's design
system: the same CSS tokens, fonts, tab layout, figure and caption styles, explorer look and controls, chart
conventions. Start from `explore/vit_coherent/site/` (it already adapts the ResNet site to one checkpoint by twelve
sites). A reader who knows the ResNet page should feel at home at once.

Design is yours. My suggestions, which you may overrule with a reason:

- Tabs: Overview; Explorer (with a switch between the ViT and Mamba, or two explorer tabs); Findings (argued, with
  a table of contents, each finding opening with an explorer view that shows which block, snapshot and r it is
  about); Measurements (by question, small); Method (what differs per study from the ResNet construction; the
  sanity table; what happened to the older pages and where their files are).
- The Overview should let the owner context in within two minutes: what was run on what, the one-paragraph answer
  for each network, a visual summary strip (for example sets by depth for each network), and where to click next.
- The ViT explorer exists; reuse it.
- The Mamba explorer does not exist in this visual language. Token contexts are text, not images, so "representative
  items" are example contexts with the token at t marked and the next token shown, and "class make-up" becomes
  token-class and source make-up plus the commonest tokens at t and t+1. The stored data offers a fixed list of
  objects (single layers L4, L8, ..., L24 for the residual and for the state; the all-layer blocks; depth thirds;
  the whole state; lags), not every layer block as in the ResNet explorer. Offer what is stored. The owner asked for
  the ability to zoom in on a SEBA set.
- No new model runs and nothing on Modal. Analysis of stored artifacts on this machine is fine (4 vCPU, 8 GB RAM, no
  GPU; keep jobs to a few minutes). If something important needs raw activations from a Modal volume, list it in
  your hand-back instead of fetching it.
- Every number in the prose comes from a data file through the build (the ResNet site's `{{fact:...}}` mechanism or
  an equivalent), never typed in by hand. Keep a short `NOTES.md` in `explore/ncs_report/` recording: how to
  rebuild, where each section's numbers come from, design decisions you took and why, and what you cut. Later
  rounds of this work start from a fresh agent that will read `BRIEF.md`, `PLAN.md` and `NOTES.md`.
- Check the built page as a reader would: screenshots of every tab at desktop (1440 x 900) and phone (390 x 844)
  widths with `playwright-cli` (see `~/.claude/CLAUDE.local.md`; serve the built directory on loopback, use a
  distinct `-s=` session name, close it afterwards, keep screenshots in `/tmp/ncs-shots/`), actually look at the
  images, and require zero console errors.
- Do not publish, commit or push. The orchestrator publishes after review.

## 6. Process

1. **Plan** (first task): write `explore/ncs_report/PLAN.md` and stop. A reviewer from the other model family
   (GPT-6.1 Sol) critiques it before anything is built.
2. **Build and write** (second task): the site, the prose, the checks in section 5.
3. **Critique rounds**: GPT-6.1 Sol reviews the built page against the data; a fresh writer task revises. Repeat
   until the reviewer has no material objection.
4. **Fresh-eyes read** by GPT-6.1 Terra with no context; final fixes.
