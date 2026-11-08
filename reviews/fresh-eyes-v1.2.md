# Fresh-eyes readability review of v1.2 (GPT-6.1 Terra, 2026-11-07)

Items marked [done] were applied in v1.3; the rest are for a pedagogical rewrite.

## THEORY.md
- [done] Revision-history references (v0/v1, withdrawn formulas) interrupt the argument; move to an appendix.
- [done] Claim-marker promise ("every claim") is broader than practice; scope it; explain "pro-tem".
- Add a compact overview distinguishing the core operator, spectral summaries, and fitted/interpretive extensions.
- [done] "Attribution is a separate step" sounds stronger than the predictive/local-derivative content of Sections 6.4 and 7; narrow it.
- Qualify "a site has an activation at every layer" for resolution-changing architectures.
- [done] Define KV cache, SAE, CKA, HAVOK, ISAN, TMoS, VAMP, DMD, SSM, MLP, CNN, LTI, HMM, JVP, RSA, EDMD, PSR, IPR at first use or in a glossary.
- Columns of H^L are the site-space vectors, not the matrix itself.
- Signpost that the exact expansion needs the complete basis including the constant mode; state whether "top-r" counts the constant mode.
- Section 2 title overreaches (units are not kernels); zero-vector policy should be at the normalization definition; cite the isotropy/cosine evidence; specify the lag index set and document-boundary handling; one intuitive sentence for the sqrt-p output view.
- Section 3: separate the finite construction from the continuum interpretation; name the conditions at the point of use; clarify "composability" means meaningful comparison under a common measure; the tail-sum error refers to a specified F.
- Section 4: a worked two-layer example for v_i, u_i, sigma; separate theorem / localization condition / interpretation; [done] map split vs merge to directions; state layer order in the product; [done] define IPR and its direction; explain varimax and display-vs-recovery rotation; explain Hankel/HAVOK/DMD relation.
- Section 5: a one-mode example with dimensions; [done] "centroid on A and zero outside"; readability evaluation convention (held-out, intercept); carry "effective action" forward; Nyström formula or pointer.
- Section 6: concrete attention example for "the layer's inputs at x"; [done] "lower bound on what"; explain learned parametric model vs closed-form fit; dimensions-and-meaning table for the KCCA objects; define the VAMP score; summary table of the four rank notions; [done] Section 14 wording contradicting 6.3; "writes" vs loading; cross-fitting procedure (folds, extension of variates across splits); [done] define R_0; [done] carried ladder defined analogously.
- Section 7: mark "detects one coherent structure" as a definition; identify "the paper" or mark Stage 2 as placeholder; small example for Delta including loss averaging; define B, Pi_B, G_r explicitly; two-row comparison of spectral vs site-group interaction; procedure for "halves as seen at L'"; expand JVP; [done] "standalone magnitude-based weight comparisons" wording.
- Sections 8-10: A, B reused for models vs sets; define J_{L'->out}, F_out before use; aggregation formula for sensitivity-weighted unreadability; Section 9 opening says local metric first; [done] measurement sequence for stochastic resamples; [done] beta_nn convention.
- Sections 11-15: one-line setup for each hypothesis bullet; [done] add M8 to the register; unpack the long-range-context limitation; [done] flag negative eigenvalues of the sparse estimand earlier; define d_int and epsilon; [done] move "open decisions for the user" to a planning note.

## EXPERIMENTS.md
- Add a one-paragraph orientation (question, pipeline, what success means).
- Clarify what a failed planted target does and does not establish; the "trained vs fit" distinction.
- Label planning estimates as estimates.
- Shared machinery: give inputs/outputs per module; which kernel Nyström approximates in the context module; describe the notebook deliverable independent of tooling.
- Stage 0: "no trained network"; condition for the symmetric-normalized product test; expand TMoS; HMM benchmark assumptions; name the metric and reference per knob.
- Stage 1: sense in which columns of W are a reference; block selection rule when there is no gap; label expectations.
- Stage 2: [done] mark as placeholder pending the source; define the target (overlapping indicators vs partition); define cross-feature error.
- Stage 3: short grokking explanation; operational identification of used frequencies; measurable merge criterion; block matching across checkpoints.
- Stage 4: two-sentence HMM/belief setup; scope of the established result; "needs no separately trained probe" wording; stitching interface.
- Stages 5-8: specify models and stitch objective; exact-fit threshold; per-use vs per-site for convolution; beta-VAE; target labels for Stages 6-8; CIFAR/ResNet variants; access assumptions for Pythia; SAE matching convention; stitching across widths.
- Apply one template per stage: setup, labelled targets, measurements, controls, decision rule, deliverable.

## Keep as is
Section 0's bookkeeping-vs-causation framing; 6.3's carried vs read with the copy example; 7.2's gradient vs update with the frozen-input qualification; Stage 3's site-table declaration.
