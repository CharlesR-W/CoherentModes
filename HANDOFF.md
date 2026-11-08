# Handoff

Written 2026-11-07 at the end of the founding session. Read this first, then `REPORT.md`, then `CONTEXT.md`. Instructions for agents working in this repo are in `CLAUDE.md`.

## State of the project in one paragraph

A theory document (`THEORY.md`, v1.5) and an experiment plan (`EXPERIMENTS.md`, v1.5) exist and have been through three review rounds by GPT models plus a fresh-eyes read, with every formal claim marked [thm], [cond] or [hyp]. Three constructions that changed under review are verified numerically by `checks/cm_check.py`, which runs in seconds and has assertions. An owner-facing summary is `REPORT.md`. Six design decisions remain open and are listed in `DECISIONS.md`; they are the owner's to make and do not block Stage 0 of the experiments. No code beyond the check script has been written. No experiments have been run.

## File map

| file | purpose | who reads it |
|---|---|---|
| `REPORT.md` | owner-facing summary of everything: idea, pipeline, claims and their status, limits, experiment plan, decisions | owner, any new agent |
| `CONTEXT.md` | the owner's intent, definition of "principled", interview answers, working preferences | any new agent, before touching the theory |
| `THEORY.md` | the language itself, 16 sections plus revision appendix | agents extending the theory or writing code |
| `EXPERIMENTS.md` | eight stages with labelled targets; shared-machinery library plan | agents writing code |
| `DECISIONS.md` | six open choices with consequences | owner |
| `CLAUDE.md` | standing instructions for agents in this repo | agents |
| `checks/cm_check.py` | numerical checks: bistochastic product, centered CKA limit, KCCA dual form and variates, reconstruction tail, indicator-vs-group gradient, merge example | agents; run before and after any change to Sections 3, 4, 6, 7 |
| `reviews/README.md` | review history, three rounds | agents |
| `reviews/fresh-eyes-v1.2.md` | readability items; the ones marked [done] were applied, the rest await a pedagogical rewrite | a rewrite agent |
| `reviews/papers.md` | constructions and limits of the two supplied papers, and the edits made from them | agents working on Sections 6, 7 or Stage 2 |

Git history on `main` records every version; there is no remote.

## The core idea in five lines (details in REPORT.md)

Sites (one per token position or input) are the atoms; every measurement of a site is a kernel on sites via a dot product on a sphere; kernels are normalized to symmetric doubly stochastic operators (Sinkhorn) so that all layers share one stationary measure; products of layer operators are Froyland coherent-set operators with identity transport, and their SVDs give coherent modes; modes map back to activation space by loadings, and any unit or gradient expands exactly in the mode basis. Context is handled by a regularized canonical-correlation SVD between the past window of the stream and either the stream's future (carried) or the next layer's update (read).

## What the next session should do

1. **Owner decisions.** Get answers to the six items in `DECISIONS.md`. Defaults if the owner does not answer: keep [A1] as is; uniform site weighting; both context operators first-class with "read" used for attribution; require stochastic resamples for the headline temperature and report the nearest-neighbour temperature otherwise; substitute a same-distribution held-out corpus for pretrained models and flag it; output-Fisher sensitivity for the stitching weight.
2. **Stage 0 and the library.** Build the shared machinery of `EXPERIMENTS.md` (sites table, views, kernels with the two estimands, Sinkhorn, spectra, loadings, context module, gradient module, comparison). Start from `checks/cm_check.py` for the exact formulas. Reuse `datafold`/`pydiffmap` internals only if their normalization can be replaced by Sinkhorn; `deeptime` for VAMP cross-checks; `pynndescent` or `faiss` for neighbours. Stage 0 produces the knob-to-error curves everything else quotes, and it includes four identity checks that must become unit tests.
3. **Stage 1** (toy models of superposition) with Stage 0, then **Stages 3 and 4** (grokking; HMM belief states with a transformer and a Mamba), then **Stage 2** (now fully specified from Ferreira da Silva and Heimersheim 2026; its 100k-step runs go to Modal), then 5–8.
4. **Optional pedagogical rewrite** of `THEORY.md` using the open items in `reviews/fresh-eyes-v1.2.md`; the owner said they may commission this.

## Things learned the hard way (do not relearn)

- **Normalize all layers against one shared measure.** Separate `D^{-1/2} K D^{-1/2}` normalizations give each layer its own stationary vector and the product of two layers loses coherence from density mismatch alone (top singular value 0.997 instead of 1 in the check). Sinkhorn to doubly stochastic fixes it exactly.
- **Center before comparing.** Uncentered alignment tends to 1 for every pair at both temperature limits. The centered operator at low temperature is `(β/N) C Ĝ C` with `Ĝ` the Gram of row-normalized activations, so the limit is linear CKA on normalized rows.
- **Infinite temperature is the identity, not clustering.** Clustering is a finite-temperature window; only the span of a block of near-top eigenvalues is canonical, never individual modes inside it. Report blocks delimited by gaps.
- **The context estimator must be whitened.** Use smoothers `R = K̃ (K̃ + Nρ I)^{-1}` on centered Grams; canonical correlations squared are the eigenvalues of `R_p R_q` and the past variates are its eigenvectors. The first-draft formula was not whitened and its spectrum grew with N. The correlations are *not* invariant to kernel scale at fixed ridge.
- **Carried and read are different.** A state that predicts the stream's own future need not contain what the next layer reads (i.i.d. tokens, next layer copies the previous token). The predictive-risk ladder is built on the read target, held out, cross-fitted on document-level splits, reported as a sequence with no inequality signs and no zero-limit claim.
- **Split and merge.** A contrast slow at L but annihilated at L' is a *merge*. The founding draft had it backwards; `checks/cm_check.py` item 7 is the example.
- **Gradient decomposition.** Exact for the gradient, at block granularity (individual mode pressures are not rotation-invariant), with signed interaction `<G_B, G_B'>`. It is not the optimizer update; the exact optimizer-agnostic statement is on the activation side with inputs frozen. The indicator-block contribution is not the site-group loss gradient (two-site counterexample in the checks). For weights used several times per site, index the uses.
- **Cosine views erase gain.** On an isolated-feature ray the normalized hidden and output vectors are the same whatever the gain, so the attenuation central to the compute-in-superposition paper is invisible to spectra; keep raw response curves.
- **Mode profiles can be spread by rectification alone.** A neuron that is ReLU of one spectral function has a broad profile. Expand the pre-rectification input and the response both.
- **Grokking sites.** On sites = (example, final position) the embedding view is constant; use post-attention, post-MLP and logits there, and analyse token-embedding geometry separately on token sites.
- **Two estimands.** Dense kernel (Nyström at low temperature) and k-NN graph (at high temperature) are different objects; the graph operator need not be PSD. Measure their agreement on dense subsets; never assume it.
- **Claims discipline.** Every repair above came from a GPT reviewer catching an overclaim. Keep the [thm]/[cond]/[hyp] markers and send every substantive change to `THEORY.md` through a GPT critique before committing (see `CLAUDE.md`).

## Review process used (reproduce it)

1. Draft by the main (Claude) agent.
2. In parallel: `model-router:plan-reviewer` (GPT-6.1 Sol) for research design, and `model-router:gpt-6-astra` for mathematical correctness, each told the owner's fixed commitments and asked not to be steered.
3. Revise; send the revision back to the *same* reviewer agents for a focused re-check (cheaper than a fresh review); revise again.
4. Fresh-eyes readability review by an agent with no prior context (`model-router:gpt-6.1-terra`), reading only the artifact.
5. For the owner-facing report, an Opus agent wrote it and ran the same loop; it found the split/merge error that the theory reviews had missed, which is an argument for always doing the report step.

## Open literature items

None blocking. Pointers for later: Hänni et al. 2024 (sparse binary code theory) and Gibson 2026 (learned Bloom filters), both cited in the compressed-computation paper; Marshall and Coifman 2019 (bistochastic kernels); Wormell and Reich (spectral convergence of diffusion maps); Froyland 2013 (finite-time coherent sets); Wu and Noé 2017 (VAMP); Hsu, Kakade and Zhang (spectral HMM learning); the stitching-failure papers listed in `reviews/README.md` round 1.
