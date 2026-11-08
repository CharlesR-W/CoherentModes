# CoherentModes: instructions for agents working in this repo

This repo holds a theory (`THEORY.md`) and experiment plan (`EXPERIMENTS.md`) for a spectral, site-space language describing what a trained network does to its activations. Start with `HANDOFF.md`, which gives the reading order. The owner's global instructions in `~/.claude/CLAUDE.md` apply and take precedence over this file.

## Non-negotiables of the theory (from the owner; see CONTEXT.md)

- Every object *in the construction* ends in an SVD of a sample-estimable operator. Nothing is trained on top of the model under study *inside the construction*. Closed-form ridge fits are allowed and must be called fits. Trained baselines for comparison (an SAE, an actual stitch) and labelled mechanism-validation interventions are allowed and required where `EXPERIMENTS.md` lists them; they are labelled as baselines or interventions, never as outputs of the method.
- The only substantive geometric assumption is the dot product on each view (cosine on normalized activations). Do not add metric assumptions silently.
- No smuggled semantics in the spectral construction: no task distributions, semantic labels, or probes define a kernel or a basis. The studied model's own training objective is used on the gradient side ([A3]), and labels are used to validate results.
- No sparsity assumption. Clustering and dictionaries are a measured regime, never a premise.
- Architecture-agnostic: anything that only works for transformers is a variant, not the spine.

## Claims discipline

- Every formal claim in `THEORY.md` carries [thm], [cond] or [hyp]. Assumptions carry [A], [P] or [M] and appear in the register (Section 12). Keep this when editing; add new tags to the register.
- Any substantive change to `THEORY.md` or to the targets, baselines, splits or protocols in `EXPERIMENTS.md` is critiqued by an agent of the *opposite model family* from the author before it is finalized (committed or handed back). For Claude authors: `model-router:plan-reviewer` for design, `model-router:gpt-6-astra` or `model-router:gpt-6.1-sol` for mathematics. For GPT authors: an Opus or Fable agent. Tell the reviewer the owner's fixed commitments (above) and do not steer it. Within one revision cycle, send revisions back to the same reviewer agent; if it cannot be resumed or the named model is unavailable, brief a fresh opposite-family reviewer and say so.
- Readability changes get a fresh-eyes read by an agent with no context (`model-router:gpt-6.1-terra` has worked well).
- Run `uv run --with numpy python -I checks/cm_check.py` before and after touching Sections 1, 3, 4, 6, 7 or 8. It ends with `all checks passed` only if every assertion holds. Add an asserted check whenever a reviewer produces a counterexample.
- Record what changed and why in Appendix A of `THEORY.md` and in `reviews/README.md`. Do not put revision history in the body of the argument.

## Experiments

- Follow the Principles, Shared machinery and Ordering sections of `EXPERIMENTS.md`; they are authoritative and are not repeated here. Before coding any stage beyond Stage 0, its protocol must be written into `EXPERIMENTS.md` and reviewed (see `HANDOFF.md`, next steps, item 2).
- Each stage ends with a notebook (the `experiment-notebooks` skill) and a written verdict against its labelled targets (identity / planted / established / hypothesis).
- GPU or long runs: Modal via the `modal` skill. This machine orchestrates only.

## Repo conventions

- Work so far is on `main`. The repo is **public** (https://github.com/CharlesR-W/CoherentModes, remote `origin`): anything committed and pushed is published, so push only when the owner asks. Commit only when the owner asks, and follow the harness's current instructions on branching and attribution lines.
- Reviews and paper notes go in `reviews/` and are appended to `reviews/README.md`. Checks go in `checks/`. Code, when it exists, goes in `cohmod/` with tests in `tests/`.
- Decisions that are the owner's to make go in `DECISIONS.md` with options, consequences and a provisional default; do not resolve them silently. If you must proceed before an answer, use the provisional default and say so in the commit message or handback.

## THEOREMS.md and SPARSITY.md are showcase documents, not the working ontology

`THEOREMS.md` and `SPARSITY.md` were written (2026-11-07) for a theorem-minded funder audience. Their results are technically correct, but they prove things under strong named idealizations (exact gaps, exact sparsity, belief-sufficient layers, finite rank) and they coin terminology ("sparse concept cluster", "ideal dictionary regime", "belief-sufficient layer", and similar) whose names do rhetorical work. Agents working on the theory or experiments must **not** adopt that terminology, treat those idealizations as premises, or import their ontological or semantic posits into `THEORY.md`, `EXPERIMENTS.md`, code, or verdicts. The working language is `THEORY.md`'s: sites, views, operators, blocks, measured regimes. Cite a theorem from `THEOREMS.md` only as a theorem (with its assumptions), never as a description of what a trained network does. The one thing from that work that agents *should* act on is `reviews/theorems-discrepancies.md`, the list of places where the theorems show `THEORY.md` to be imprecise; it is the owner's to decide on, and changes to `THEORY.md` follow the usual review rule above. To edit `THEOREMS.md` or `SPARSITY.md`, edit `theorems/drafts/*.md` and re-run `python3 theorems/assemble.py`; then run `uv run --with numpy python -I checks/theorems_check.py`.
