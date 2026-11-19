# Review 2: the built page (GPT-6.1 Sol, 2026-11-13), with the orchestrator's decisions

Verdict: **ship after fixes.** Every objection below is accepted; the orchestrator confirmed objection 1 by hash.
Paths are relative to the repository root unless they start with `/`.

## Material objections (reviewer's text, lightly condensed) and what to do

1. **Every embedded explorer screenshot shows the wrong experiment.** Overview cards and the Findings "What we are
   looking at" boxes: all eight `explore/ncs_report/viewer/site/assets/fig/view_*.png` are byte-identical (SHA-256
   `ba64a2a1…595ad`) screenshots of the older ResNet/CIFAR explorer (checkpoint "step 60", sites `c1`..`logits`,
   CIFAR classes), captioned as different ViT and Mamba views. Compare `view_m-roles.png` with the live view the
   reviewer captured at `/tmp/ncs-review1/m-roles.png`.
   **Do:** recapture all eight from this report's own server and look at each image. Make `shoot_views.sh` robust:
   pick a free port (the fixed port was probably already serving another site), assert before each capture that
   the page is this report and shows the intended network, object, r and colouring, and make the build fail if two
   view images are identical or a view is missing. Crop the views tighter (see minor suggestions).

2. **"The residual stream forgets in two tokens" exceeds and misidentifies the measurement** (Findings
   `#m-reading`; Overview opening paragraph and last card). What is measured is the retention (smoothness) of
   particular SEBA vectors under later walks, not disappearance of information. The prose says the vectors belong to
   residual layer 16, but they are the **all-layer residual block's** r = 8 vectors evaluated on layer 16's walks
   (`mamba_contexts/data/m130c/lags.json`: `source.point = Bpoint`; `mc_schema.py` lines 223–224). Mean retention
   0.909864 → 0.223134 at lag 2, chance −0.000125: weakened a lot, not zero, and not a memory test.
   **Do:** say "the residual block's set vectors lose most of their retention within two tokens"; correct the vector
   source; remove "gone", "forgets", and the statement that the grouping is replaced as soon as tokens arrive. The
   figure caption already has the source right.

3. **Both explorers overflow a phone viewport.** At 390 × 844 the tour selectors (`#mx-tour-sel`, `#ex-tour-sel`,
   `.ctl-guide`) push the document to 462 px (Mamba) and 447 px (ViT). Screenshots:
   `/tmp/ncs-review1/phone-mamba-top.png`, `/tmp/ncs-review1/phone-vit-top.png`.
   **Do:** let the control wrapper shrink and constrain the select (`min-width: 0`, a `max-width`); recheck both, and
   assert `document.documentElement.scrollWidth <= innerWidth` at 390 px in your check.

4. **The k = 8 and k = 30 objects are labelled with the wrong neighbour count.** Selecting `Bk8` shows "residual,
   all layers, k = 8 … kNN graph, k = 15" (`site/js/tokens.js` lines 402 and 540; same for `Bk30`).
   **Do:** take the displayed k from the selected object in both places.

5. **The grammatical-role headline is stronger than the evidence** (Findings `#m-roles`, around lines 179 and 188;
   the Overview's short answer). "The sets are roles" and "what the token is doing" suggest contextual grammatical
   distinctions; the evidence shows mixtures of token families, punctuation and document formatting. In
   `sets_trained.json:Bpoint["16"]`, set 2 is 143 / 145 newline tokens and 143 / 145 poetry contexts; set 13 mixes
   " is", "$" and ")" with commonest-token share 12%.
   **Do:** narrow the headline and the Overview to something like "the all-layer sets mix token families and
   role-like groupings", and introduce determiners, prepositions and so on as descriptions of the prominent members
   of particular sets.

6. **The report and the explorer use two different "matched Jaccard" metrics.** `analysis_ncs.py` lines 60–77
   first drops every item unassigned in either partition; `site/js/explorer.js` lines 234–239 keeps each set's full
   membership. Recomputed from `vit_coherent/data/vits_b2k/layer_0.npz`, L9–L12 against the two perplexity rules:
   0.821799 / 0.772441 (report) against 0.815962 / 0.764768 (explorer).
   **Do:** use the explorer's full-set convention in the report (Measurements `#q-depth`, the Method table) and
   regenerate the facts and figures.

## Minor (accepted)

- The Findings tab repeats a lot of explorer chrome in the eight views. Crop them tighter so the evidence is
  legible and the page looks shorter.
- The Overview's first paragraph carries nearly the whole report. Shorten it and let the two answer cards carry the
  detail.
- Mamba commonest-token shares round non-zero values to 0% (for example 11 / 2,798). Show one decimal below 1%.
- Finding 5: "eigenfunctions are token indicators" → "their extremes isolate frequent tokens"; twelve end items do
  not establish an indicator over all contexts.

## Checked by the reviewer and correct (no need to re-check)

All five tabs render at both sizes with no live-figure failures; sampled tours gave no page errors; set and
eigenfunction controls, set selection, network switching, back navigation, Mamba intersections, reload and
breadcrumbs worked. All 74 exported Mamba objects at all nine ranks equal the source labels, and the exported
eigenfunction ends match. ViT ≥16/20 counts 0 / 23 / 39 / 94 at L1 / L6 / L9 / L12; L9–L12 72 with 35 whole; set
sizes, leading eigenvalues, participation minima and the first three eigenfunctions' end classes agree. Mamba
catch-all intersections 2,386 / 2,798 at r = 24 and 1,969 / 2,798 at r = 32; participation minima 220.38 (B-point)
and 61.76 (whole). Pythia source-Jaccard 0.174–0.236 at the first step, 0.676–0.786 at step 512, 0.654–0.843 at the
end; 927 / 1,000 root-cut agreement. Not checked: walk statistics from raw activations or Grams (not stored here).
