# Handoff

Written 2026-11-07 at the end of the founding session. Reading order for a new agent: this file, then `REPORT.md` (note its "Update" section at the end; a few passages in its body predate v1.4–v1.5 and are superseded there), then `CONTEXT.md`, then `CLAUDE.md`.

## State of the project in one paragraph

A theory document (`THEORY.md`, v1.5) and an experiment plan (`EXPERIMENTS.md`, v1.5) exist. Versions up to v1.3 went through three review rounds by GPT models (two accuracy rounds and a fresh-eyes read); the v1.4 and v1.5 edits (split/merge fix, noise protocol, the two supplied papers) were smaller and were reviewed only by the agents that proposed them. Every formal claim is marked [thm], [cond] or [hyp]. Three constructions that changed under review are verified numerically by `checks/cm_check.py`, which runs in seconds and has assertions. An owner-facing summary is `REPORT.md`. Six design decisions remain open and are listed in `DECISIONS.md` with provisional defaults; they are the owner's to make and do not block Stage 0. No code beyond the check script has been written, no experiments have been run, and several stage protocols still need to be specified before they can be coded (see step 2 below). The programme is not yet implementation-ready end to end; Stage 0 is.

## File map

| file | purpose | who reads it |
|---|---|---|
| `REPORT.md` | owner-facing summary of everything: idea, pipeline, claims and their status, limits, experiment plan, decisions | owner, any new agent |
| `CONTEXT.md` | the owner's intent, definition of "principled", interview answers, working preferences | any new agent, before touching the theory |
| `THEORY.md` | the language itself, Sections 0–16 plus Appendix A (revision notes) | agents extending the theory or writing code |
| `EXPERIMENTS.md` | nine stages (0–8) with labelled targets; shared-machinery library plan | agents writing code |
| `DECISIONS.md` | six open choices with consequences and provisional defaults (not owner-approved) | owner |
| `CLAUDE.md` | standing instructions for agents in this repo | agents |
| `checks/cm_check.py` | asserted numerical checks: reconstruction tail (1.3), bistochastic product (3.1, 4.1), centered CKA limit (8), KCCA dual form and variates (6.3), indicator-vs-group gradient (7.2), merge example (4.1) | agents; run before and after any change to Sections 1, 3, 4, 6, 7, 8 |
| `reviews/README.md` | review history, three rounds | agents |
| `reviews/fresh-eyes-v1.2.md` | readability items; the ones marked [done] were applied, the rest await a pedagogical rewrite | a rewrite agent |
| `reviews/papers.md` | constructions and limits of the two supplied papers, with a list of *proposed* edits; the edits were applied in trimmed form in commit `a64234a` (v1.5) and are summarized in THEORY Appendix A | agents working on Sections 6, 7 or Stage 2 |

Git history on `main` records every version; there is no remote.

## The core idea in five lines (details in REPORT.md)

Sites (one per token position or input) are the atoms; every measurement used as a view becomes a kernel on sites via a dot product on a sphere (units are functions on sites to be expanded, not kernels); kernels are normalized to symmetric doubly stochastic operators (Sinkhorn) so that all layers share one stationary measure; products of layer operators are Froyland coherent-set operators with identity transport, and their SVDs give coherent modes; modes map back to activation space by loadings, and any unit or gradient expands exactly in a complete mode basis (truncation leaves a computable residual). Context is handled by a regularized canonical-correlation SVD between the past window of the stream and either the stream's future (carried) or the next layer's update (read).

## What the next session should do

1. **Owner decisions.** Get answers to the six items in `DECISIONS.md`. Provisional defaults are recorded there; if used, say so in the commit message.
2. **Protocol specification before code, for everything beyond Stage 0.** `REPORT.md` Section 10 asks for this and it has not been done. Items with no written protocol yet: out-of-sample extension of the canonical variates and the exact cross-fitting folds (Section 6.4); block selection when there is no gap, and block matching across checkpoints and temperature scans (Sections 3.2, 4.3); operational identification of the frequencies a grokking model uses (Stage 3); the stitching interface for unequal widths and architectures (Stages 4, 5, 8); quantitative success thresholds per stage; seeds, probe-bank sizes and checkpoint schedules. Write these into `EXPERIMENTS.md`, get them reviewed by an opposite-family agent (see `CLAUDE.md`), then code.
3. **Stage 0 and the library.** Build the shared machinery of `EXPERIMENTS.md` (sites table, views, kernels with the two estimands, Sinkhorn, spectra, loadings, context module, gradient module, comparison). Start from `checks/cm_check.py` for the exact formulas. Reuse `datafold`/`pydiffmap` internals only if their normalization can be replaced by Sinkhorn; `deeptime` for VAMP cross-checks; `pynndescent` or `faiss` for neighbours. Stage 0 produces the knob-to-error curves everything else quotes, and it includes four identity checks that must become unit tests.
4. **Stage 1** (toy models of superposition) with Stage 0, then **Stages 3 and 4** (grokking; HMM belief states with a transformer and a Mamba), then **Stage 2** (source task and main training recipe specified from Ferreira da Silva and Heimersheim 2026; seeds, probe banks, thresholds still to fix; its 100k-step runs go to Modal), then 5–8.
5. **Optional pedagogical rewrite** of `THEORY.md` using the open items in `reviews/fresh-eyes-v1.2.md`; the owner said they may commission this.

## Things learned the hard way (do not relearn)

- **Normalize all layers against one shared measure.** Separate `D^{-1/2} K D^{-1/2}` normalizations give each layer its own stationary vector and the product of two layers loses coherence from density mismatch alone (top singular value 0.997 instead of 1 in the check). Sinkhorn to doubly stochastic fixes it exactly.
- **Center before comparing.** Uncentered alignment tends to 1 for every pair at both temperature limits. The centered operator at low temperature is `(β/N) C Ĝ C` with `Ĝ` the Gram of row-normalized activations, so the limit is linear CKA on normalized rows.
- **Infinite temperature is the identity, not clustering** (for distinct normalized vectors; repeated directions leave averaging blocks, which matters for isolated-feature probe banks). Clustering is a finite-temperature window; only the span of a block of near-top eigenvalues is canonical, never individual modes inside it. Report blocks delimited by gaps.
- **The context estimator must be whitened.** Use smoothers `R = K̃ (K̃ + Nρ I)^{-1}` on centered Grams; canonical correlations squared are the eigenvalues of `R_p R_q` and the past variates are its eigenvectors. The first-draft formula was not whitened and its spectrum grew with N. The correlations are *not* invariant to kernel scale at fixed ridge.
- **Carried and read are different.** A state that predicts the stream's own future need not contain what the next layer reads (i.i.d. tokens, next layer copies the previous token). The predictive-risk ladder is built on the read target, held out, cross-fitted on document-level splits, reported as a sequence with no inequality signs and no zero-limit claim.
- **Split and merge.** A contrast slow at L but annihilated at L' is a *merge*. The founding draft had it backwards; `checks/cm_check.py` item 7 is the example.
- **Gradient decomposition.** Exact for the gradient, at block granularity (individual mode pressures are not rotation-invariant), with signed interaction `<G_B, G_B'>`. It is not the optimizer update; the exact optimizer-agnostic statement is on the activation side with inputs frozen. The indicator-block contribution is not the site-group loss gradient (two-site counterexample in the checks). For weights used several times per site, index the uses.
- **Cosine views erase gain.** On an isolated-feature ray the normalized hidden and output vectors are the same whatever the gain, so the attenuation central to the compute-in-superposition paper is invisible to spectra; keep raw response curves.
- **Mode profiles can be spread by rectification alone.** A neuron that is ReLU of one spectral function has a broad profile. Expand the pre-rectification input and the response both.
- **Grokking sites.** On sites = (example, final position) the embedding view is constant; use post-attention, post-MLP and logits there, and analyse token-embedding geometry separately on token sites.
- **Two estimands.** Dense kernel (Nyström at low temperature) and k-NN graph (at high temperature) are different objects; the graph operator need not be PSD. Measure their agreement on dense subsets; never assume it.
- **Claims discipline.** Every repair above came from a GPT reviewer catching an overclaim. Keep the [thm]/[cond]/[hyp] markers and send every substantive change to `THEORY.md` through a GPT critique before committing (see `CLAUDE.md`).

## Review process used

The standing rules are in `CLAUDE.md`; the history is in `reviews/README.md`. One lesson worth keeping here: the owner-facing report step (an Opus agent writing `REPORT.md` and running its own critique loop) found the split/merge error that two rounds of theory review had missed. Always do the report step.

## Open literature items

None blocking. Pointers for later: Hänni et al. 2024 (sparse binary code theory) and Gibson 2026 (learned Bloom filters), both cited in the compressed-computation paper; Marshall and Coifman 2019, arXiv:1711.06711 (bistochastic kernels); Wormell and Reich (spectral convergence of diffusion maps); Froyland 2013, arXiv:1210.7418 (finite-time coherent sets); Wu and Noé 2017, arXiv:1707.04659 (VAMP); Hsu, Kakade and Zhang, arXiv:0811.4413 (spectral HMM learning); on stitching failures, Smith et al. 2025, "Functional Alignment Can Mislead: Examining Model Stitching" (PMLR v267) and "How Not to Stitch Representations", arXiv:2412.11299.
