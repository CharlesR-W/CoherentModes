# Plan: joint report on the ViT and Mamba coherent-sets studies

Writer/designer: Claude Opus 5.5, 2026-11-13. Reads with `BRIEF.md`. Nothing is built yet.

## 1. Identity

- **Title:** What a Vision Transformer and a Language Model Treat Alike
- **Subtitle:** Neural Coherent Sets of images through DINOv2 ViT-S/14, and of token contexts through Mamba-130m
- **Short title (tab bar):** Neural Coherent Sets: ViT and Mamba
- **Description:** The label-free construction of the ResNet-20 report, run on a self-supervised vision transformer
  and on a state-space language model, with an explorer for each and the findings argued from it.
- Banner (once): "Exploratory report". Byline, date, "code and data" link as on the ResNet page. Voice: "I".

## 2. Tabs

Same five tabs, order, ids and look as the ResNet page: `#overview`, `#explorer` (marked), `#findings`,
`#measurements`, `#method`. **One** Explorer tab with a network switch (ViT | Mamba) in its control bar, so a
reader who knows the ResNet page sees the same shell. URL state carries `net=vit|mamba`.

### Overview (target: two minutes; mostly pictures)

| # | section (id) | reader comes away able to say | carried by | data |
|---|---|---|---|---|
| 1 | lede | "Same construction, two new networks; the ViT's late layers group images into kinds of thing, the Mamba residual groups contexts by the token just read and then by its grammatical role." | one paragraph, facts only | facts below |
| 2 | hero (`hero`) | "I can see the sets change with depth in both networks." | static `hero_vit.png` and `hero_mamba.png`, each a strip of 4 *mini explorer views* (one UMAP panel coloured by a set colouring, beside its set list: swatch, size, make-up bar, and 4 thumbnails (ViT) or the commonest token at t → t+1 (Mamba)). ViT: L1, L6, L9 own sets, block L9–L12, r = 8. Mamba (on the residual all-layer layout): L4 r = 8, all-layer block r = 16, L24 r = 8. Each strip links to its tour. | `vit_coherent/data/vits_b2k/{layer_0.npz:seba_label,probe.npz:images,labels}`, `umap_0.npy`; `mamba_contexts/data/m130c/{spectra_trained.npz:<obj>__seba_label, layouts.npz:Bpoint, sets_trained.json:<obj>.<r>[].tok_t,tok_next,class_t}` |
| 3 | `what-ran` | what was run on what | 2-row table: network, items, sites/views, blocks offered, N, kernel rule (kNN k = 15), r | `meta.json` of each |
| 4 | `answers` | the one-paragraph answer per network | two `callout key` cards (ViT, Mamba), each ending in links to its Findings and a tour | facts |
| 5 | `procedure` | "It's the ResNet procedure, unchanged in its steps." | five one-line steps (no figures), maths in one `<details>`; link to the ResNet report's `#overview/procedure` | none |
| 6 | `healthy` | "The eigen-structure is not pathological, with one resolution effect at ViT L12; split-half is the missing check." | two sentences + link to `method/sanity` | facts |
| 7 | `next` | where to click | three cards (as ResNet "Three things to look at"): ViT kinds, Mamba roles, Mamba forgetting along reading | live `images` grid; static mini-view; small lines chart |

### Explorer: see section 4.

### Findings

Lede (how to read a finding; tags *observed / reading / speculation*, as in ResNet) and a contents box. Template
per finding, unchanged from ResNet: **view box** (explorer screenshot, click opens the tour) → **Claim** with tag
→ evidence with definitional captions → **How I read it, and what else it could be** (collapsed) → **Try it**.
Every number via `{{fact:…}}`.

| # | id | claim (draft sentence) | evidence shown | opens |
|---|---|---|---|---|
| V1 | `v-kinds` | *reading.* By the last four blocks, DINOv2's sets are kinds of thing: dogs and other mammals; snakes, spiders and crabs; birds; road vehicles; food; buildings and monuments; electronics and household objects. `{{v.l9_12.r8.n_ge16}}` of the 100 classes have at least 16 of their 20 images in one set (`{{v.l9_12.r8.n_whole}}` all 20). | 8 `images` grids (`data-r=8`, `j=0..7`, 12 each), class-group heatmap (`q_classmix` L9–L12 r = 8), list of each set's top fine classes (fact strings) | tour `vit-late` (`a=8,b=11,r=8`) |
| V2 | `v-depth` | *observed.* Kinds appear gradually with depth: the number of classes kept ≥16/20 in one single-snapshot set (r = 8) rises from `{{v.site.L1.n_ge16}}` at L1 through `{{…L6…}}` and `{{…L9…}}` to `{{…L12…}}` (L12 includes the resolution effect of V4). | lines chart of n_ge16 by site (L1..L12: 0, 0, 1, 5, 11, 23, 30, 38, 39, 50, 76, 94 as computed today) with a permutation null band; beside it the largest set of L1, L6 and L9 as small `images` columns | tours `vit-L1`, `vit-L6`, `vit-L9` |
| V3 | `v-two` | *observed, then reading.* At r = 2 the block L9–L12 splits mammals and birds from everything else; invertebrates go with food and artefacts, and 70 of the 80 reptile images are unassigned. | heatmap: 10 class groups × (set 1, set 2, unassigned); two `images` grids | tour `vit-two` |
| V4 | `v-l12` | *observed.* L12 alone resolves single classes: λ₁…λ₃ = 0.999, 0.995, 0.992 and sets of exactly one class (all 20 lorikeets). With 20 images per class and 15 neighbours, classes become nearly separate components; this is a resolution effect of the probe, not a finer ontology. | spectrum stems L9 vs L12 (from block meta); `images` grid of the lorikeet set | tour `vit-l12` |
| M1 | `m-token` | *observed.* At L4 the residual's sets are the token just read: five of eight sets are one token (" the", ",", " of", two newline sets) with commonest-token share 0.87–0.98; one is indentation; one catch-all holds `{{m.L4.r8.set1.size}}` contexts. | Mamba explorer screenshot; set table (size, top-token share, top 3 tokens at t and t+1) | tour `m-L4` |
| M2 | `m-roles` | *reading.* Averaged over all 24 layers (r = 16), the residual groups contexts by role: determiners, prepositions and conjunctions (followed by " the"/" a"), pronoun and auxiliary contexts, digits, LaTeX subscripts, LaTeX commands, poetry line ends (followed by capitalised "And", "I"), clause punctuation, sentence ends, code indentation, code symbols. | the set table at r = 16 with t → t+1 chips; 3 example contexts per chosen set (token at t marked) | tour `m-roles` |
| M3 | `m-catchall` | *observed.* One set holds `{{m.Bpoint.r16.set1.size}}` of 8,000 contexts, word starts and continuations; it needs `{{…tokens_for_90}}` distinct tokens to cover 90% of it. Most content words occur about once at this N, so nothing can group them. | zoom view of that set (finer-r breakdown) | tour `m-catchall` |
| M4 | `m-late` | *observed, then reading.* Toward L24 the residual's neighbours share the token just read less (excess share `{{L4}}` → `{{L24}}`) and share the next-token distribution more beyond what the token explains (κ̃\|T `{{L4}}` → `{{L24}}`); its sets mix role with domain and with what comes next (LaTeX backslashes followed by "frac", "mu", "lambda"). | lines chart over L1..L24 of excess shares T, Cc, Cn, W and κ̃, κ̃\|T (point view) | tour `m-L24` |
| M5 | `m-reading` | *observed.* Along reading, the residual's set structure fades within two tokens, while the whole recurrent state keeps its sets (document source) for at least 16. | lines chart of lag retention (point L8/L16/L24, state L8/16/24, whole) vs τ; AMI whole vs whole_lag16 | tours `m-whole`, `m-lag` |
| M6 | `m-state` | *reading.* The whole state's sets are document source (AMI `{{…}}`) with a few splits; a bag of the prefix's tokens gives source as strongly (walk share 0.87 vs 0.85), but the state also carries token class and next-token association that the bag lacks. | 2×3 small table (walk excess shares W, Cc, κ̃ for whole vs bag vs recent) | tour `m-whole` |
| — | `snippets` | (short, untagged) The two earlier whole-snippet studies (Mamba v1, Pythia-160m; 1,000 snippets from five sources) found the five sources; their top cuts agree on 927 of 1,000 snippets. At that grain the probe had little else to offer, which is why token contexts were next. | one sentence + link to Method `older` | none |
| — | `next-checks` | Checks I'd like to run (split-half resampling; more items for Mamba; a ViT probe with more images per class or larger k at L12; the true zoom of §4) | list | — |

### Measurements (by question; small; captions say only what is plotted)

1. **How fast do the sets change with depth?** ViT: site × site matched Jaccard (`q_carry`, slider r) and n_ge16
   by site. Mamba: a8 between thirds (L1–8, L9–16, L17–24) and between residual / state / whole (`compare.json`,
   kind `thirds`, `objects`) as a small heatmap.
2. **What do the neighbours share, layer by layer?** Mamba: excess walk shares T, Nx, Cc, Cn, W, U and κ̃, κ̃|T
   for point and state at L1..L24 (`cells_trained.json:point_L*/state_L*.{T,…}.ex, kappa.kappa_ex, kappa_ex_T`),
   with permutation bands. ViT: class-group make-up per site (`q_classmix`, slider).
3. **How long do sets last along reading?** `lags.json:views.*.mean` (retention vs τ, chance line) and
   `compare.json` kind `lag` (a8 vs τ).
4. **Spectra and spread.** λ₁..λ₁₆ and participation numbers of the main objects (ViT block meta; Mamba
   `spectra_trained.npz:<obj>__vals, __pn`). Feeds the Method sanity table.

### Method (reference; long is fine)

`glossary` (ResNet terms plus: token context, item, position t, residual (point) view, state view, whole state,
lag τ, reference views bag / recent / pred, token class, all-layer block; rename the studies' "cell" to
**snapshot** everywhere) · `vit` (what differs: one checkpoint, CLS after LN at L1..L12, N = 2,000 = 20 per
ImageNet-100 class, 10 display groups from WordNet) · `mamba` (8,000 contexts = 8 positions × 1,000 snippets,
views, blocks, references, groupings; layouts are UMAP of ψ₁..ψ₁₆ of an object, not of activations) ·
`kernel` (kNN k = 15 default in both; one paragraph) · `sanity` (the table below) · `earlier-vit` (3 sentences:
other kernels, smoothness-change operators, N = 10,000; leading modes reproduced across halves 0.90–0.97 there;
class η² rises with depth; DINO's temperature localises modes; its hard-set verdict does not transfer) ·
`certificate` (one sentence: no clear spectral gap anywhere; not the standard here) · `older` (the four older
pages, what each showed in a line, where their files are) · `checks` · `reproduce` · `refs`.

**Sanity table** (`method/sanity`), columns ViT L9–L12 r = 8 · Mamba residual all-layer r = 16 · Mamba whole r = 8:
min PN of ψ₁..ψ₈ (`vecs`/`__pn`); agreement under another neighbour count or kernel rule (ViT: matched Jaccard vs
the 4 other stored rules; Mamba: `sensitivity` a8/a16 vs k = 8, 30, residual only); agreement with the untrained
network (Mamba: `init_trained` a8; ViT: not stored); agreement with an input-only reference (Mamba: a8 vs bag;
ViT: max |corr| of ψ₁..ψ₈ with pixel statistics, `q_imgstats`); split-half: **not measured** (row says so).

## 3. Leads checked against the data (details in the hand-back)

Confirmed: ViT L1 mixed / L6 rough / L9 kinds; L12 spectrum 0.999, 0.995, 0.992 and one-class sets; Mamba
B-point r = 16 roles (as listed, except a mixed "copula" set, top-token share 0.12, and a newline set followed by capitalised headings), catch-all 2,798; T excess 0.380 vs 0.104; κ̃ 0.248, κ̃|T 0.121;
k-sensitivity a8 0.866 / 0.875; init a8 0.009; pred a8 0.359; residual lag retention 0.91 → 0.22 at τ = 2;
whole AMI(W) 0.735, W share 0.849 vs bag 0.866, Cc 0.491 vs 0.058, κ̃ 0.225 vs 0.033; whole-lag16 AMI 0.753;
init whole W 0.173; whole init–trained a8 0.03; 927 / 1,000.
Not as stated: see hand-back (b). Prose will use the data's version.

## 4. Explorers

### ViT (reuse)

Reuse the ViT explorer's data path (`build_site.py` over `vits_b2k`: one row of 12 snapshots wrapping 2 × 6,
block chunks, sprite of 64 px thumbnails, fine class in the tooltip) **but on the ResNet site's current code**,
which the ViT site forked from before three later changes: snapshot/panel wording, the sets-first side card with
make-up bars and thumbnails, and dropped channel loadings. Remove: time-block family and checkpoint selector,
per-image training diagnostics (loss, correct, Jacobian, gradients), channel loadings (head norms), seed-1 layout.
Keep: pixel-statistic colourings (they are cheap and already computed). Tours for V1–V4 plus `vit-default`.

### Mamba token contexts (new, same look)

```
[ Network: ViT | Mamba ]  COLOUR FROM [object ▾] [trained|init]   COLOUR BY [sets ▾] [r = 16 ▾]   TOURS [▾]  How to use
┌──────────────── panels (6, 3 × 2; 1 column on phone) ────────────────┐  ┌─ side panel ───────────────────────────┐
│ residual L1–8   residual L9–16   residual L17–24                      │  │ Sets, r = 16, of residual all layers  │
│ residual L1–24  state L1–24      whole state      [+ references]      │  │ ■ set 1  2,798  [token-class bar]     │
│ (outline: the panel whose layout is the colour source's own)          │  │   t: " the" · " a" · " an"  → t+1: …  │
└───────────────────────────────────────────────────────────────────────┘  │   [source bar]  top-token share 0.37   │
generated definitional caption (object, r or k, layout source, grey)       │ … one row per set; unassigned row      │
                                                                            │ class and source keys                  │
                                                                            ├─ spectrum λ₁..λ₁₆ stems + PN ─────────┤
                                                                            └────────────────────────────────────────┘
```

- **Panels** are the stored UMAP layouts (`layouts.npz`): residual thirds, residual all-layer, state all-layer,
  whole state; "+ references" adds bag, recent, pred; "+ along reading" adds residual-L16-lags and whole-lags.
  Each panel's title says "UMAP of ψ₁..ψ₁₆ of <object>". Every panel shows the same 8,000 contexts; hover rings
  the context in all panels and shows a tooltip: 24 tokens of left context with the token at t boxed, the next
  token after a ▸, source, the model's top-1 prediction.
- **Colour from** (grouped select; exactly the stored spectral objects): residual single layers L4, L8, …, L24;
  residual blocks all layers, L1–8, L9–16, L17–24, all layers at k = 8 / k = 30; state single layers L4…L24; state
  all layers; whole state; along reading: residual L8/L16/L24 and state L8/L16/L24 and whole at τ = 1, 2, 4, 8, 16,
  residual-L16 lag block, whole lag block; references bag, recent, pred. `init` is enabled for residual all
  layers, whole, and residual/state single layers (`objects.init`).
- **Colour by:** coherent sets (r ∈ 2, 3, 4, 6, 8, 12, 16 from `__seba_label`); eigenfunction ψ₁..ψ₁₆ (`__vecs`,
  diverging); token class at t, at t+1 (12 classes); source (5); position t. Labels only colour and describe.
- **Side panel, sets:** one row per set (as ResNet): swatch, size, stacked bar of token class at t, source bar,
  commonest tokens at t and at t+1 as chips with counts, top-token share. Shown by default, no click.
  **Eigenfunction:** the 12 most negative, most positive and nearest-zero contexts (`__ends`).
- **Zoom into a set** (click a row; `j=` in the URL): (1) the panels show only that set's members and refit
  their extent to them (a real visual zoom), (2) the card lists 12 example contexts in membership order (soft SEBA
  value, reconstructed at build from `__vecs` and `__seba_rot`), with a "more" control, (3) **"Inside this set"**:
  how its members split among the same object's sets at the next finer stored r (r → 16 → 24 → 32), each sub-set
  with size and its t → t+1 chips; clicking a sub-set zooms again (breadcrumb back), and the panels recolour by
  the finer sets, (4) **"The same contexts elsewhere"**: a bar of its members over the sets of a second object
  (default: L4 for blocks, the all-layer block for single layers).
  This is a zoom by refinement of stored sets, not a new eigenproblem on the members; see open question 2.
- URL: `#explorer?net=mamba&o=Bpoint&ck=trained&c=seba&r=16&j=11&zr=24&k=1&lay=main|refs|lags&tour=`.

## 5. Cut from the four older pages

- Pre-registered targets K1–K6 and H1–H7, and their met / not-met tables: not the standard the owner judges by.
- Clustering certificate (Υ, θ, certified / excluded / undetermined), fixtures and calibration: one sentence in Method.
- Pilots, budgets, ledgers, controls C1–C3, P1–P3, deviations lists: provenance, not results; linked via `older`.
- Zoom trees T1, T2: their 3-level cuts duplicate what finer-r zoom shows; open question 2.
- ViT smoothness-change operators, diffusion-time grid, ridge-fit attributions, neuron loadings, head-norm
  "channels": a different construction or a fit; 3 sentences in `earlier-vit`.
- Mamba v1 and Pythia explorers, per-position and slab views, cross-architecture a3 tables: whole-snippet grain;
  one sentence plus 927 / 1,000.
- Pythia over training: the stored series are time blocks per layer, and the sets are sources at every step;
  no figure (open question 5).
- ViT decodability charts (earlier 10,000-image protocol) and kernel κ tables: off-topic here.

## 6. Build

- **Layout:** `explore/ncs_report/site/` (sources) → `explore/ncs_report/viewer/site/` (built, relative paths,
  `.nojekyll`). Copy, not import, so later edits to sibling sites cannot change this report.
- **Copied from `cifar_coherent/site` (HEAD):** `index.html`, `explorer.html`, `css/site.css`, `js/{main,charts,
  colors,live,data,explorer}.js`, `vendor/`, `palette.json`, `build_site.py`, `DESIGN.md` (trimmed).
- **ViT deltas:** 3-way merge with `git merge-file`, base = `git show 39d1266:explore/cifar_coherent/site/<f>`
  (the ViT fork point; ViT explorer.js differs from it by 163 lines), ours = ResNet HEAD, theirs =
  `vit_coherent/site/<f>`; conflicts resolved by hand. Then the removals in §4. Data read in place from
  `vit_coherent/data/vits_b2k` (read-only path argument). `analysis_vit.py` copied; its `questions.json` reused.
- **New scripts:** `build.sh` (one command: analysis → figures → ViT data → Mamba data → HTML);
  `build_mamba.py` (writes `d/mamba/`: `items.json` (contexts as token ids, vocab subset, t, snippet, source,
  class t / t+1, top-1), `layouts.bin` u16, `obj/<ck>/<obj>.bin.gz` (labels at 9 r, ψ₁..ψ₁₆ int8, soft SEBA max
  for r = 8, 16, set summaries for every r, ends, vals, PN); `js/tokens.js` (the Mamba explorer, reusing the
  ResNet canvas panel code and CSS classes); `analysis_ncs.py` (facts and chart JSON for both networks, incl.
  n_ge16 with a permutation null, the sanity table, Mamba layer and lag curves); `make_figures.py` (hero strips,
  small static figures; matplotlib in the site's palette and fonts); `shoot_views.sh` (playwright-cli: each tour →
  `assets/fig/view_<tour>.png`, the Findings view boxes).
- **Facts:** `{{fact:…}}` as in ResNet; `analysis_ncs.py` writes `facts` with prefixes `v.` (ViT), `m.` (Mamba),
  `s.` (sanity); strings preformatted. The build fails on an unknown fact. Set descriptions in prose (e.g. "dogs
  and other mammals") are my words; the numbers and top classes / tokens beside them are facts.
- **Palette:** sets need 16 colours (ResNet has 8). Add 8, check with the dataviz validator; 12 token-class
  colours and 5 source colours as two new categorical tokens.
- **Size:** ViT data ≈ 15 MB (as now), Mamba ≈ 8–10 MB (60 + 14 objects × ~100 KB gz, items ≈ 1 MB, layouts
  0.4 MB), figures ≈ 3 MB: **≈ 30 MB** total (ResNet: 135 MB). Mamba object chunks load lazily.
- **Verify:** build twice (idempotent); `--facts` lists all facts, none unknown; a link checker over every
  `#tab/id`, `explore-link` and tour; screenshots of every tab at 1440 × 900 and 390 × 844 plus each tour, in
  `/tmp/ncs-shots/`, inspected by eye; zero console errors; a grep that "cell" appears only for activation
  cells; `NOTES.md` written (rebuild, fact sources per section, decisions, cuts).

## 7. Open questions (with my defaults)

1. **One explorer tab with a network switch, or two tabs?** Default: one tab (keeps the ResNet shell; tours span both).
2. **Zoom into a set.** A true zoom (the construction rerun on a set's members) needs the Mamba Grams on the Modal
   volume. Default: zoom by refinement of stored finer-r sets plus visual refit; list the true zoom under "checks".
   The stored trees T1/T2 are such reruns but on tree nodes, not SEBA sets; default: leave them out.
3. **ViT explorer base.** Default: ResNet HEAD code with the ViT delta merged (wording and side card match the
   ResNet page); dropping head-norm loadings.
4. **Mamba panels.** Only 11 stored layouts exist (no single-layer layouts), so single layers are painted on block
   layouts. Default: 6 panels + toggles, as sketched.
5. **Pythia over training.** Default: no figure; one sentence.
6. **Split-half reproducibility** cannot be computed here (ViT CLS views on `cm-vit-coherent`, Mamba Grams on the
   Mamba volume). Default: the sanity table says "not measured"; listed for the orchestrator as a Modal job.
7. **Title.** Default as in §1; the owner may prefer a title in the ResNet report's pattern.
