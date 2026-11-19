# NOTES: the joint ViT / Mamba report (build 1, revised after REVIEW_2, REVIEW_3 and REVIEW_4, 2026-11-13)

Written by the build-and-write agent (Claude Opus 5.5); revised by a second Opus 5.5 agent against `REVIEW_2_page.md`
(see "Review 2" at the end) and by a third against `REVIEW_3_recheck.md` and `REVIEW_4_fresh_eyes.md` ("Reviews 3 and 4"). Reads with `BRIEF.md`, `PLAN.md`, `REVIEW_1_plan.md`, `REVIEW_2_page.md`.
Title: **What a Vision Transformer and a Language Model Treat Alike** (plan's draft, kept; the owner will be asked).

## Rebuild

```bash
explore/ncs_report/build.sh          # ~2 min + ~3 min browser check; fails on an unknown fact, a broken link, a bad view
                                     # image, a console error or horizontal overflow at 390 px (NCS_SKIP_BROWSER=1 skips
                                     # the browser part; NCS_SHOTS=dir keeps its tab screenshots)
explore/ncs_report/shoot_views.sh    # recaptures site/assets/fig/view_<tour>.png + views.json (needs playwright-cli);
                                     # run build.sh after
python3 -m http.server <free port> --bind 127.0.0.1 -d explore/ncs_report/viewer/site   # to look at it
```

`build.sh` runs, in order:

| step | script | writes | reads (read-only) |
|---|---|---|---|
| facts and chart data | `analysis_ncs.py` | `site/assets/ncs.json` (`facts` + chart objects) | `vit_coherent/data/vits_b2k`, `mamba_contexts/data/m130c`, `mamba_coherent/diagnosis/diag_v1.json`, `mamba_coherent/data/m130/trees.json`, `lm_coherent/data/p160t` |
| ViT measurements | `site/analysis_vit.py` (copy of the ViT site's, paths changed, L9–L12 added) | `site/assets/questions.json` (`q.*` facts; per-image pixel statistics for the explorer) | `vit_coherent/data/vits_b2k` |
| figures | `make_figures.py` | `site/assets/fig/hero_vit.png`, `hero_mamba.png`; phone versions in `site/assets/fig/phone/` (both heroes, and side-panel crops of every `view_*.png`) | same as above, and the `view_*.png` from `shoot_views.sh` |
| Mamba explorer data | `build_mamba.py` | `viewer/site/d/mamba/` | `mamba_contexts/data/m130c` |
| ViT explorer data and the page | `build_site.py vits_b2k --clean` (adapted from the ResNet site's) | `viewer/site/` (everything else) | `vit_coherent/data/vits_b2k` |

Memory: `analysis_vit.py` peaks at about 0.9 GB; everything else is under 0.2 GB. No 8,000 x 8,000 array is
loaded (none is stored; the spectral objects are eigenvectors).

## What the build checks (REVIEW_1 objection 8)

- `build_mamba.py`: for every one of the 74 stored objects (60 trained, 14 init) and all 9 ranks, SEBA labels
  reconstructed from the stored int16 eigenvectors and rotations by the `mc_schema.validate` convention (dequantise,
  prepend the constant, threshold 0.99/sqrt N, unit norm, sign, max-scale, reliability permutation, clamp) agree with
  the stored labels on >= 99% of items (worst 99.94%). For every stored set descriptor (r = 8, 16; 1,776 sets) the
  size, top tokens at t and t+1, token-class (t, t+1) and source make-up recomputed from the exported arrays equal
  `sets_<ck>.json` exactly; stored examples are members of their set (overlap with the reconstructed top-12 by SEBA
  value: at least 8 of 12). The exported context's token at t / t+1 equal the stored T / Nx. Eigenfunctions are
  shipped as int8 (16 per object): sign pattern exact, no item moves by more than one of the explorer's 41 colour
  levels, correlation >= 0.98 (worst 0.990, a localised reference mode). The explorer always uses the stored
  labels and the stored ends (12 largest, 12 smallest entries; `ends[k,0]` largest, `ends[k,1]` smallest), never
  the int8 vectors, for anything but colour.
- `build_site.py`: the ResNet build's SEBA recomputation check (worst agreement 0.9995) plus a read-back of every
  ViT block chunk: decoded labels equal the source labels and the int8 eigenvector signs equal the source's.
- Page check (fails the build): leftover `{{...}}`, "MOCKUP/TODO/XXX/lorem", placeholder blocks; every internal link
  `#tab`, `#tab/id`, bare `#id`; every `#explorer?...` link and every tour hash (network, object, checkpoint, stored
  r, block range); every `data-src` path into `ncs.json`; every image file. Unknown `{{fact:...}}` fails earlier.
- Explorer screenshots (REVIEW_2 objection 1), in the page check: every `view_<id>.png` the page shows exists, names a
  tour, has the sha256 that `shoot_views.sh` recorded in `site/assets/fig/views.json` after its state check, and no
  two are identical. `shoot_views.sh` itself serves on a free port, checks the served `<title>`, and before each
  capture waits until the explorer shows the tour's network, block / object, checkpoint, colouring, r or k and panel
  set (`shoot_views.js`), failing otherwise; transitions are disabled for the capture (a headless capture caught the
  ViT block outline mid-slide).
- Browser check (`check_browser.sh` + `check_browser.js`, run by `build.sh`): every tab and both explorers at
  1440 x 900 and 390 x 844, every tour at 390 x 844 (26 loads); fails on any console or page error or if
  `document.documentElement.scrollWidth > innerWidth` at 390 px (REVIEW_2 objection 3).
- Not automated: interaction (checked by hand with playwright: network switch, tours, set selection, the
  "under another r" breadcrumb, back/forward, reload with `j=` and `sel=`).

## Where each section's numbers come from

All prose numbers are `{{fact:NAME}}` from `site/assets/ncs.json` (prefixes `v.` ViT, `m.` Mamba, `o.` older
studies, `s.` sanity table), `site/assets/questions.json` (`q.`), or the Mamba export's validation (`x.`).
`uv run --with numpy --with pillow python -I build_site.py vits_b2k --facts` lists them all after a build.

- **Overview**: `v.*` (n, classes, L9_L12 set counts, eta2), `m.*` (n, T_ex, broad sets, PN, sensitivity, init,
  bag), `o.*`. Hero figures: `make_figures.py` from the stored labels / layouts / sets descriptors.
- **Findings 1–4 (ViT)**: `v.L9_L12.*`, `v.L1|L6|L9|L12.*` from `layer_0.npz` mode 0 (kNN k = 15) and `probe.npz`;
  `v.ge16.null_max` from 50 relabellings; `v.eta_*` computed here from the dequantised eigenvectors. Image grids:
  the explorer's own live `images` figure.
- **Findings 5–9 (Mamba)**: set facts from `sets_trained.json`; walk shares and kappa from `cells_trained.json` /
  `cells_init.json` (`ex` = (S − S0)/(1 − S0); bands converted to the same units); `a_r`, AMI from `compare.json`;
  lag retention from `lags.json`; eigenfunction ends, eta2, AMI with source from `spectra_trained.npz`; PN computed
  from the stored int16 vectors. Set tables and context lists: live `msets` / `contexts` figures over `d/mamba/`.
- **Earlier studies**: `o.py.*` from `lm_coherent/data/p160t/blocks_<step>.npz` (`reading_src_jacc[:, 1]`, r = 4,
  with null and ceiling); `o.v1.*` from `diag_v1.json` (`d1.depth`, `d1.reading`, AMI means over t); `o.rootcut.*`
  by cross-tabulating the T1 root children of the two `trees.json`.
- **Measurements**: chart objects in `ncs.json` (`v_eta`, `m_eta`, `m_shares_point|state`, `v_carry_r8`,
  `m_thirds`, `v_ge16`, `m_lags`, `m_objects`, `v_spectra`, `m_spectra`, `v_pn`).
- **Method**: `m.*`, `v.*`, `s.*`, `q.img.*`, `x.*`.

Construction parameters (k = 15, r, tau, "16 of 20", "12 contexts at each end", "128 tokens" where not a fact) are
written as definitions of what is plotted; the run sizes (N, classes, layers, snippets, positions) are facts.

## Design decisions

- **Shell**: the ResNet site's current `index.html`, CSS, `main.js`, charts and live figures; the ViT explorer is
  the ResNet `explorer.js` with the ViT fork's one-checkpoint changes 3-way merged (`git merge-file`, base
  `39d1266`), loadings dropped (head norms), "cell" → "snapshot/panel" kept. Five tabs as in ResNet.
- **One Explorer tab, network switch** (`.net-seg`) as the first control in each bar; `#explorer?net=mamba|vit`.
  A tour's hash names its network; a bare `#explorer` keeps the last network shown. The two explorers are separate
  DOM trees (`#ex`, `#mx`); the router (`main.js`) shows one and lazy-loads its module.
- **Mamba explorer** (`site/js/tokens.js`), same look: control bar (object grouped by residual / state / along
  reading / references; trained | untrained; colour by sets, ψk, sign of ψk, token class at t / t+1, source,
  position; panels: 6 layouts, + references, + along reading), the stored UMAP panels (outline = the colouring's own
  layout), definitional caption, side card of sets shown by default (size, token-class bar, commonest tokens at t
  → t+1 with counts, share of the commonest token, the top member as a context line with the token at t boxed),
  spectrum λ1..λ16 with PN on hover. ψ colouring: the side card lists the 12 contexts at each end.
- **Zoom** (objection 6): click a set → panels refit to its members (others light grey), 12 members in SEBA order
  with "show 12 more", then "Selected members under another r of the same object": rows of that r's sets with
  "n of the selection; m in all", recomputed summaries, unassigned included; clicking keeps the intersection; a
  breadcrumb steps back. Labelled as a different SEBA rotation, not a subdivision. URL `j=` and `sel=r.j,...`.
- **Palette** (objection 8): sets 1–8 use the eight validated set colours; sets 9–16 darker versions of the same
  hues (composite hue + lightness; the dataviz validator fails these as an independent palette, so identity is
  carried by the side card: hovering a row isolates its set); sets 17–32 one grey; the key says so. 12 token-class
  colours = the 10 class colours + beige (word continuation) + dark grey (other).
- **Context lines** keep the token at t visible: the left context is clipped from the left, ↵ and ⇥ shown.
- **Findings**: each opens with an explorer screenshot box (`shoot_views.sh`), claim with tag, captions that say
  what is plotted, collapsed "How I read it", Try it. Eigenfunctions are a finding of their own for the ViT (2) and
  part of findings 5 and 6 for Mamba, with tours `vit-psi1` and `m-psi1` linked from the Overview cards.
- **Overview**: lede with bolded whole ideas, two hero strips (matplotlib "mini explorer views"), a three-row
  "what was run" table (the earlier studies as a row: the four-study provenance strip of objection 1), two answer
  cards, five one-line steps with the maths collapsed, the soundness paragraph, three cards (two explorer
  screenshots, one along-reading screenshot).
- Sizes: built site 36.0 MB in 703 files after reviews 3 and 4 (`du` 37M); phone images add about 1 MB. After review 2 it was 35 MB (34.1 MB of files; ViT 15 MB data, Mamba 17 MB, figures 4 MB, vendor 1 MB).

## What I cut

Pre-registered targets and tables; clustering certificates (one paragraph in Method); pilots, ledgers, controls;
zoom trees (one sentence: a different analysis, omitted for brevity); Mamba v1 / Pythia explorers and per-position
views; the earlier ViT study's operators, diffusion-time grid, ridge fits, decodability charts (one paragraph);
the ViT explorer's per-image diagnostics and head-norm loadings; kernel κ charts (the ViT explorer's "?" help keeps
the five rules); Mamba init comparisons beyond the two a_8 values and the init whole-state source share.

## Plan items not done as planned, and what I did instead

- **ViT sanity row "other neighbour count"**: not stored for the ViT (only other kernel rules), so the row shows
  a_8 and matched Jaccard against the two per-item perplexity rules instead; untrained-network and split-half rows
  are dots / "not measured".
- **Near-zero examples** of Mamba eigenfunctions (objection 8): not derived; the explorer shows both stored ends.
- **"More" examples** beyond the stored 12 are ordered by the reconstructed SEBA value (u8 in the export).
- **Plan V3 (r = 2 split)** folded into finding 1 as a paragraph and heatmap; plan M3 (catch-all) is a subsection of
  finding 6; plan M5/M6 are findings 8 and 9.
- **Earlier ViT study numbers** (0.90–0.97 split-half etc.) are only in its `VERDICT.md` text, not a data file,
  so Method describes that study qualitatively and points to the file.
- **Pythia over training**: one chart (objection 1), reading-block source Jaccard at r = 4 by step and layer.

## Where the data differed from the plan or brief

- Brief: "six of eight sets are essentially one token at L4". Data: five sets with commonest-token share 0.87–0.98
  (1,135 contexts, 14.2%), plus an indentation set (share 0.27 of one token but 99% indentation class); the two
  broad sets hold 82.5%. Prose follows the data (objection 2).
- Brief: whole-state lag-16 "agreement 0.75" is the AMI of the r = 8 sets (`compare.json`, 0.753); the a_8 is 0.640.
  Prose quotes the AMI, Measurements the a_8.
- Brief/plan: L12 "sets that are single classes (all 20 lorikeets)": two such sets (lorikeet, robin).
- Brief: B-point r = 16 list included a "copula" set; data: set 13 is mixed (" is", "$", ")" ...; share 0.12),
  described as "copulas and closing symbols".
- Pythia: sets are not sources at every step (step 0: 0.17–0.24), as the reviewer said.

## Claims in Findings, with the facts behind them

1. *Kinds (reading).* sizes `v.L9_L12.r8.sizes`, `v.L9_L12.r8.n_unassigned`; per-set classes
   `v.L9_L12.r8.set{1..8}.top`, `.size`; classes ≥16/20 `v.L9_L12.r8.n_ge16`, all 20 `v.L9_L12.r8.n_whole`;
   heatmap `v_mix_L9_L12_r8`, read down its columns (each column sums to 100%): `v.L9_L12.r8.mix.one_set_groups` /
   `.one_set_min` (groups with ≥ 90% in one set), `.spread_groups` / `.spread_max` (groups with < 60% in any one). r = 2: `v.L9_L12.r2.set2.dogs_share`, `.birds_share`, `.share_mammal_bird`,
   `v.L9_L12.r2.set1.invert_share`, `v.L9_L12.r2.reptiles_unassigned` / `reptiles_n`, heatmap `v_mix_L9_L12_r2`;
   `v.dog_classes`. The kind names are mine from the class lists and the heatmap's group shares (set 6 "buildings and
   monuments (with some clothing and devices)", set 7 "devices, containers and clothing", set 8 mixed); the prose says
   they are names.
2. *Eigenfunctions (observed, then reading).* ends `v.L9_L12.psi{1,2,3}.{neg,pos}.classes`; η² of ψ1 with groups
   `v.L9_L12.eta2_group.psi1_pc` (as a percentage); by layer `v.eta_fine.{L1,L6,L9,L12,chance}`, chart `v_eta`; eigenvalues
   `v.L9_L12.lam{1,2,3}`. Stated as "the 12 images at each end are one or two kinds"; "food and devices",
   "primates" etc. are my words for those class lists.
3. *Depth (observed).* `v.{L1,L6,L9,L12}.r8.n_ge16`, `v.ge16.null_max`, `v.{L9,L12}.r8.largest_share`, charts
   `v_ge16`, `v_bigset`; reading: `v.L1.r8.set1.top`, `v.L6.r8.set1.top` ("mostly red things", "natural textures"
   are readings of the images).
4. *L12 (observed, then reading).* `v.L12.lam{1,2,3}`, `v.L9.lam{1,2,3}`, `v.L12.r8.n_single_class_sets`,
   `v.L12.r8.single_class_sets`, `v.L12.r8.sizes`, `v.L12.r{2,4,8}.n_ge16`. The resolution-effect explanation is a
   reading, not tested (listed under checks).
5. *Token at L4 (observed).* `m.T_ex.L4` (introduced in plain words: 0 = chance, 1 = every step of the walk), `m.T_band_hi.max`, `m.L4.r8.n_token_sets`, `.token_sets_tokens`,
   `.token_share_min/max`, `.token_sets_size`, `.token_sets_frac`, `.broad_sizes`, `.broad_frac`;
   ψ ends `m.L4.psi{1,2}.{pos,neg}.tok` (stated as "the extremes isolate frequent tokens", ends only); reading:
   `m.Cc_ex.L4`, `m.W_ex.L4`; table: live `msets` (point_L4, r = 8).
6. *Token families, punctuation and layout (observed, then reading).* tokens
   `m.Bpoint.r16.set{12,15,5,13,8,2}.tok_t|tok_next`; poem line ends `m.Bpoint.r16.set2.{size,top_class_share,
   top_source_share}`; set 13's mix `m.Bpoint.r16.set13.share_pc`; ψ ends `m.Bpoint.psi{1,2}.{pos,neg}.{tok,source}`;
   catch-all `m.Bpoint.r16.catch.{set,size,word_share,t90}`, rarity `m.word.{n,n_once}`, finer r
   `m.Bpoint.catch.r{24,32}.largest`. "Determiners", "prepositions", "pronouns and to" describe the most prominent
   members of sets whose class is ≥ 83% function words (`m.Bpoint.r16.set{12,15,5}.top_class_share`); they are not
   offered as grammatical roles.
7. *Late layers: token shared less, next-token distributions more similar (observed, then reading).* `m.T_ex.{L4,L24}`, `m.W_ex.{L4,L24}`, `m.kappaT.{L4,L24}`,
   `m.L24.r8.set2.tok_next`; chart `m_shares_point`; table `msets` (point_L24).
8. *Along reading (observed).* `m.ret.Bpoint_on_L16.{t0,t2,t16}` (the all-layer residual block's r = 8 SEBA vectors
   on residual layer 16's walk after τ tokens; `lags.json` `source.point = Bpoint`, asserted in `analysis_ncs.py`),
   `m.ret.chance`, `m.lag.whole.{t0,t16}`, `m.lagami8.whole.t16`; chart `m_lags` (series named by whose sets on
   whose walk). Title and claim say "lose most of their smoothness"; the claim states at once that this measures
   particular vectors on later walks, not what the network keeps (also in the Overview card).
9. *State's sets follow the kind of source and differ from the untrained network's (observed).* `m.whole.ami_W.r8`; table `m.ex.{whole,bag,recent}.{W,Cc,kappa}`,
   `m.init.ex.whole.W`; `m.init.a8.whole`, `m.cmp.whole.bag.a8`. bag / recent are described as order-insensitive
   averages of the trained model's input embeddings (whole prefix / last 16 tokens) that bypass the recurrent layers
   (`mamba_contexts/mc_decode.py` takes `model.backbone.embeddings.weight`; `mc_items.py` averages). Limits sentence:
   `m.snippets_per_source` (five kinds of source; split halves not run). The split descriptions (formulas vs prose, code
   comments, poem line ends) are readings of the `msets` table for `whole`, r = 8.
- *Earlier studies.* `o.n_snippets`, `o.snippet_len`, `o.per_source`, `o.n_sources`,
  `o.v1.reading.source.range`, `o.v1.n_reading`, `o.v1.depth.cur_class.range`, `o.v1.depth.source.range`,
  `o.py.step0|step512|final.min|max`, `o.py.positions`, chart `o_pythia`; `o.rootcut.agree`, `o.rootcut.n`.
- *Overview cards.* the same facts as findings 1–9 (`v.L9_L12.r8.n_ge16`, `v.L9_L12.eta2_group.psi1_pc`, `m.T_ex.{L4,L24}`,
  `m.L4.r8.broad_frac`, `m.Bpoint.r16.set{12,15}.tok_t`, `m.ret.Bpoint_on_L16.{t0,t2}`, `m.lag.whole.{t0,t16}`).

## Places where I am least sure the prose is carried by the data

- Finding 6's family names describe prominent members only; the prose now says so and names set 13 as mixed.
- Finding 4's resolution-effect explanation is plausible but untested.
- Finding 7's heading (shares the token less, predicts more alike) rests on κ̃|T rising 0.063 → 0.161 while source share also
  rises; the second bullet of its "what else" says so.
- The ViT set names "buildings and monuments" (set 6 includes a pirate ship and a boathouse) is loose; set 7 was
  renamed "devices, containers and clothing" from the heatmap (51%, 44%, 40% of those groups).

## Review 2 (REVIEW_2_page.md): what changed and where

All six objections and four minor suggestions were accepted and done; the build and browser check pass.

1. **Explorer screenshots were the ResNet explorer.** Rewrote `shoot_views.sh` and added `shoot_views.js`: free
   loopback port, `<title>` check, per-view state assertion (network, block / object, checkpoint, colouring, r or k,
   panel set), transitions off, `views.json` manifest with each image's sha256 and the state read back. Build check
   in `build_site.py` `check_page` (present, manifest hash matches, names a tour, all distinct). Crop: page chrome,
   tour note, how-to, step hints and the definitional caption hidden; the image is the colour key, the panels and the
   top of the side panel (ViT 1200 x 352, Mamba 1200 x 621–911). All eight recaptured and looked at. Tour `m-roles`
   renamed `m-all` (and its view file).
2. **"Forgets in two tokens."** Findings `#m-reading` (title, TOC, claim, "How I read it"), Overview lede, Mamba
   answer card (`#answers`) and third card (`#next`, retitled "Sets along reading"): the residual block's set vectors
   lose most of their retention within two tokens (0.91 → 0.22 on layer 16's walk; chance −0.0001; 0.06 at τ = 16);
   vectors correctly attributed to the all-layer block; "gone", "forgets" and "replaced as soon as tokens arrive"
   removed; "How I read it" now says this is smoothness of set vectors on later walks, not a memory test. Facts
   renamed `m.lag.point_L16.*` → `m.ret.Bpoint_on_L16.*`, `m.ret.chance` added, chart series renamed (whose sets,
   whose walk), Measurements `#q-reading` caption corrected (single-layer walks).
3. **Phone overflow.** `site/css/site.css`: `.ex-bar > .ctl { min-width: 0; max-width: 100% }` and
   `.ctl select { min-width: 0; max-width: min(360px, 100vw - 48px) }`. Both explorers now 390 / 390 (462 without the
   rule, reproduced in the browser). Asserted for every tab, both explorers and all 14 tours by `check_browser.sh`.
4. **k = 8 / k = 30 labels.** `build_mamba.py` writes each object's `k` (`K_OF`, checked against `meta.k_sens`);
   `site/js/tokens.js` side-panel subtitle and caption read `m.objects[o].k`.
5. **"Roles."** Findings `#m-roles` (now `6. Across all layers, the sets mix token families, punctuation and layout`):
   claim says mixtures; determiners / prepositions / pronouns introduced as the prominent members of sets that are
   ≥ 83% function words; poem line ends quantified (145 contexts, 99% newline, 99% poetry); set 13 named as mixed
   (commonest token 12%). Overview lede and Mamba card narrowed the same way; Overview card retitled "Token families".
6. **Two matched-Jaccard conventions.** `analysis_ncs.py` `jacc_matrix` / `matched_jaccard` now use full set
   membership and average over the first labelling's sets (= `explorer.js` `matchedJaccard`, and the ResNet
   `analysis.py` for equal r). L9–L12 against the perplexity rules: 0.82 / 0.76 (reviewer: 0.8160 / 0.7648). Method
   glossary (`#glossary-agreement`) and Measurements `#q-depth` caption state the convention; `v_carry_r8` regenerated.

Minor: views cropped tighter (above); Overview lede cut to four sentences with the two cards carrying the detail;
shares below 1% shown with one decimal (`pctShare` in `tokens.js`, used by the side panel, tooltips and the live
`msets` table; `pc` in `analysis_ncs.py`), so set 1 reads 0.4%; finding 5's "token indicators" → "the extremes of its
eigenfunctions isolate frequent tokens (the ends only)".

Other overstatements found and narrowed (same cause as 2 and 5):
- Findings `#v-kinds` claim: sets "each made mostly of one or two kinds", named "from their classes and the heatmap";
  set 7 renamed from the heatmap. Overview ViT card likewise.
- Findings `#v-psi` title "The leading eigenfunctions run between kinds" (was "axes between kinds"); claim states what
  the 12 end images are; reading limited to the ends. Overview ViT card: "the 12 images at one end … primates".
- Findings `#v-l12` "the late geometry is not noisy; it is resolved finer than this probe can show" → what the
  counts show.
- Findings `#m-token` "How I read it": "which token was just read dominates the geometry" was wrong as stated (at L4
  the token-class share 0.784 and the source share 0.409 both exceed the token share 0.380); now says so.
- Findings `#m-late` title "trade the token for what comes next" → "share the token less and predict more alike";
  "role" → "token families"; the predictive-coherence bullet no longer implies the network computes anything.
- Findings `#m-state` title "groups by document, and is learned" → "follow the source document and differ from the
  untrained network's"; tag (observed); "groups by source" → "links contexts from the same source" for the walk
  statistics.
- Findings `#earlier`: "Source structure is learned early in training" → "The sets come to follow the source early in
  training"; "the probe found documents" → "both models' sets follow the source document"; "the sets are sources;
  token classes" → "follow the source; the class of the token". Catch-all "does not break into roles" → "into
  smaller readable sets".
- Method `#sanity`: "has nothing in common with the untrained network's" → "shares almost none of its leading
  subspace".

Not changed: the Bk8 / Bk30 object names already contain "k = 8" / "k = 30", so the side-panel subtitle repeats it;
left as is because the "Colour from" menu needs the names to differ.

## Reviews 3 and 4 (REVIEW_3_recheck.md, REVIEW_4_fresh_eyes.md): what changed and where

Every "Do" item is done; the build and its browser check pass.

- **R3 material, bag / recent not model-free.** Method glossary `#glossary-reference` now defines each reference
  separately (bag: average of the trained model's input embeddings over positions 1..t; recent: over the last 16
  tokens; both order-insensitive and bypassing the recurrent layers; pred: the model's next-token distribution, as
  square-root probabilities). Same wording in Findings 9 (claim and table header / caption), Method `#sanity` row
  ("Reference that bypasses the network's layers", was "Input-only reference") and its reading, Measurements
  `#q-objects` caption, the explorer's "How to use" text, its menu group ("References (input-embedding averages; the
  model's predictions)") and object names (`build_mamba.py` NAMES: "mean input embedding of the prefix / of the last
  16 tokens (reference)"). The Overview soundness note no longer mentions the bag.
- **R3 / R4 "source document".** "kind of source" throughout: lede, Mamba card, Findings 5–9 and the earlier-studies
  section (titles, TOC, claims, reading bullets), Method glossary and Mamba run, explorer colouring ("kind of source"),
  its tooltip row and legend note, chart series (`analysis_ncs.py` `names` / `names_eta`), tour `m-whole`.
- **R3 / R4 phone readability.** `make_figures.py` writes phone heroes (`phone/hero_vit.png`: L1 and the block,
  four larger images per set; `phone/hero_mamba.png`: the three panels stacked, eight largest sets each, larger type)
  and side-panel crops of every explorer screenshot (`phone/view_*.png`, card edges found from the pixels). The page
  uses `<picture><source media="(max-width: 600px)">` for all of them; captions carry a phone-only note on what the
  phone image omits. Page check: every `srcset` exists and each phone view's desktop view is on the page.
- **R3 / R4 wide tables on a phone.** The Overview's "what was run" table is stacked on a phone (`.stack-phone`,
  labels from `data-label`); every other `.table-scroll` that overflows gets a "scroll sideways →" cue and an edge
  shadow (`markScrolls` in `main.js`).
- **R4 1, retention read as forgetting.** Overview Mamba card and Findings 8 (title, TOC, first lines of the claim):
  "lose most of their smoothness … a property of these vectors on later walks, not of what the network remembers".
- **R4 2, Mamba hero.** Redrawn: title "the same 8,000 token contexts at the same positions, three colourings",
  subtitle that the UMAP is identical in all panels, panel titles "coloured by: …", a column header ("size ·
  commonest tokens at t → at t+1"), token rows at 9.5 pt (was 8) placed by measured text width, and a build-time
  check that no row runs into the next column. Caption rewritten to say the same.
- **R4 4, excess share and η².** Plain definition at first use in the Overview card and Findings 5 (0 = chance, 1 =
  every step), and in the glossary; Findings 7 caption links the glossary. "92% of its variance" from the new fact
  `v.*.eta2_group.psi1_pc` (Overview card, Findings 2).
- **R4 5, heatmap text vs normalisation.** Findings 1 now reads the heatmap down its columns, from new facts
  `v.L9_L12.r8.mix.*` (normalisation unchanged, so the set names that rest on it stay valid).
- **R4 6, "SEBA value" before SEBA.** Hero caption: "the five images that belong to it most strongly (largest SEBA
  value)".
- **R4 explorer 1, colouring reset on switch.** `main.js` network switch carries the colouring kind (sets, ψ_k with
  its k, sign of ψ_k) to the other explorer, dropping that explorer's tour; label colourings are not carried (they
  differ between the networks).
- **R4 explorer 2, "Colour from".** An always-visible line under the control: "This object's sets or ψ colour every
  panel; each panel keeps its own stored layout." Not added to the ViT's "Block" control (its outline on the grid
  already shows this, and the ViT bar is fuller).
- **R4 explorer 4, tooltip left over.** `showNet` and the switch hide the global tooltip; the ViT explorer's delayed
  touch tooltip checks that its explorer is still shown.
- **R4 explorer 5.** "whole state read 1 token further" (`build_mamba.py` `name_of`).
- **R4 visual 2, blank Mamba hero.** Both heroes `loading="eager"`; checked by scrolling the Overview normally at
  1440 x 900 (both appear).
- **R4 length 1, Overview.** Captions cut to what is plotted; the five-step procedure is one sentence with a link
  to the ResNet report (maths still collapsed); the soundness section is one paragraph without a heading; the Mamba
  card is shorter; the three cards' text shortened. The section "The procedure" became "What was run, and how".
- **R4 length 2.** Kept per-finding views and "Plotted" captions (owner's preference); the Findings 7 chart caption
  shortened (the excess share is defined once, in finding 5 and the glossary).
- **R4 length 3, one layer or many.** New Overview paragraph `#one-or-many`: a block's sets keep what its layers treat
  alike together, with the " the" example (stated as a hypothetical).
- **R4 length 4.** Findings 9: one sentence on the five kinds of source (200 snippets each, sharply different) and
  one on what the missing split-half check means.
- **R4 wording.** "more alike next-token distributions" → "more similar" (Overview, Findings 7 title, TOC, claim and
  reading); "predict alike" removed.
- Explorer screenshots re-captured with `shoot_views.sh` after the UI change. Phone crops of them are capped at
  560 px tall (the top of the side card), so the Overview's three cards stay short on a phone.
- Method `#sanity` on a phone: the reference row's label shortened (links the glossary) and the first column given a
  minimum width, so the table scrolls rather than squeezing the labels.
