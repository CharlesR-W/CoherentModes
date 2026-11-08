# CoherentModes: instructions for agents working in this repo

This repo holds a theory (`THEORY.md`) and experiment plan (`EXPERIMENTS.md`) for a spectral, site-space language describing what a trained network does to its activations. Read `HANDOFF.md` first, then `REPORT.md`, then `CONTEXT.md` (the owner's intent and what they consider principled). The owner's global instructions in `~/.claude/CLAUDE.md` also apply: treat subagents with kindness and respect and pass that on; run anything over about ten minutes on Modal; prefer reputable existing tools to rewriting.

## Non-negotiables of the theory (from the owner; see CONTEXT.md)

- Every object ends in an SVD of a sample-estimable operator. Nothing is trained on top of the model under study. Closed-form ridge fits are allowed and must be called fits.
- The only substantive geometric assumption is the dot product on each view (cosine on normalized activations). Do not add metric assumptions silently.
- No smuggled semantics: no task distributions, labels, or probes inside the construction. Labels are for validation only.
- No sparsity assumption. Clustering and dictionaries are a measured regime, never a premise.
- Architecture-agnostic: anything that only works for transformers is a variant, not the spine.

## Claims discipline

- Every formal claim in `THEORY.md` carries [thm], [cond] or [hyp]. Assumptions carry [A], [P] or [M] and appear in the register (Section 12). Keep this when editing; add new tags to the register.
- Any substantive change to `THEORY.md` goes through a GPT critique before commit: `model-router:plan-reviewer` for design, `model-router:gpt-6-astra` or `model-router:gpt-6.1-sol` for mathematics. Tell the reviewer the owner's fixed commitments (above) and do not steer it. Send revisions back to the same reviewer agent for a re-check.
- Readability changes get a fresh-eyes read by an agent with no context (`model-router:gpt-6.1-terra` has worked well).
- Run `uv run --with numpy python -I checks/cm_check.py` before and after touching Sections 3, 4, 6 or 7; add a check whenever a reviewer produces a counterexample.
- Record what changed and why in Appendix A of `THEORY.md` and in `reviews/README.md`. Do not put revision history in the body of the argument.

## Experiments

- Follow `EXPERIMENTS.md` stage order: 0 and 1 together, then 3 and 4, then 2, then 5–8. Each stage ends with a notebook (use the `experiment-notebooks` skill) and a written verdict against its labelled targets (identity / planted / established / hypothesis).
- Held-out evaluation uses document-level (or example-level) splits everywhere a fit is involved. Windows never straddle a split.
- Every training run we control saves dense checkpoints and stochastic resamples at fixed parameters at a few of them.
- Keep the dense-kernel and k-NN-graph estimands as separate code paths and compare them on dense subsets.
- Record raw amplitudes and task losses on the unfiltered distribution separately from normalized views; log zero-vector exclusions per view.
- GPU or long runs: Modal via the `modal` skill. This machine orchestrates only.

## Repo conventions

- Commit on `main`; there is no remote. Commit messages end with the attribution line for the model that did the work.
- Reviews and paper notes go in `reviews/`. Checks go in `checks/`. Code, when it exists, goes in `cohmod/` with tests in `tests/`.
- Decisions that are the owner's to make go in `DECISIONS.md` with options and consequences; do not resolve them silently. If you must proceed before an answer, use the defaults listed in `HANDOFF.md` and say so in the commit message.
