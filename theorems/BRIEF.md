# Brief for agents drafting THEOREMS.md and SPARSITY.md (2026-11-07)

Thank you for helping with this. Please work carefully and kindly; if you dispatch sub-agents yourself, pass this request on to them. If something blocks you, say so in your report rather than working around it silently.

## What exists

`THEORY.md` (v1.5) defines a spectral construction on "sites" (one place where a network has an activation): sphere kernels on each view (Section 2), Sinkhorn/bistochastic normalization to a symmetric doubly stochastic operator `P` (Section 3), the cross-layer operator `Q = P^{L'} P^{L}` and its SVD (Section 4), loadings back to activation space (Section 5), context via a kernel-ridge canonical-correlation SVD (Section 6), exact unit and gradient expansions in the mode basis (Section 7), and fitting-free model comparison on shared sites (Section 8). Read `THEORY.md` in full before writing; `checks/cm_check.py` holds the numerical checks already asserted. `CLAUDE.md` (repo root) lists the owner's non-negotiables for the construction; do not change the construction and do not edit `THEORY.md`.

## What we are writing

Two new documents for a math-minded funder audience (theorem-lovers; they like belief-state geometry and computational mechanics; do NOT mention or lean on singular learning theory at all):

1. `THEOREMS.md`: theorems and bounds about the construction. Part A: guarantees that follow from the construction's choices (SVD/Sinkhorn/Davis–Kahan/Wedin, exactness, stability, sample complexity, cost). Part B: what the construction achieves in idealized regimes under strong, named assumptions (finite rank, exact gaps, exact sparsity, belief-sufficient layers): full recovery, optimality, limits that reproduce known objects (PCA, CKA, belief simplex / predictive states / causal states, dictionaries).
2. `SPARSITY.md`: the sense in which the construction recovers sparse/discrete structure when it is present: spectral graph sparsifiers of the kernel, spectral cuts and (higher-order) Cheeger inequalities, SEBA-style sparse basis extraction from a block, hierarchical clustering / diffusion wavelets / persistence of gaps along the temperature and diffusion-time knobs. Sparsity is a *measured regime*, never a premise: this document explains what is recovered when the regime holds and how its presence is certified.

## Epistemic rules for these two documents (they differ from THEORY.md's)

- Every statement must be technically correct, with a complete proof or a precise citation to a standard theorem whose hypotheses are verified in the text. No hand-waving in proofs. If a desired statement is false or you cannot prove it, say so and prove the strongest true neighbour.
- Strong assumptions are welcome if they are named and stated precisely (e.g. "finite rank", "exact spectral gap", "exactly k-sparse", "belief-sufficient layer"). Prefer a clean theorem under a strong named assumption over a vague one under a weak assumption. Where a noisy/approximate version is also easy, give both.
- Coined terminology is allowed and encouraged where a name carries weight ("coherent block", "ideal dictionary regime", "sparse concept cluster", "belief-sufficient layer"): define it precisely first. The name may do rhetorical work; the definition may not cheat.
- Every theorem gets, after its proof: (i) **What it buys** (one or two sentences on why this matters for the construction, in ambitious but honest language), and (ii) **Tractability and sampling** (what it costs to compute the objects involved, in N = number of sites, d = width, r = rank; how many sites are needed; what breaks the bound in practice). These remarks are mandatory, honest, and the main reason the owner wants the document to be usable.
- Relate to known objects where it is a genuine theorem: PCA/CKA limits, Froyland coherent sets, Hsu–Kakade–Zhang spectral HMM, predictive state representations, computational-mechanics causal states and the mixed-state presentation (belief simplex geometry of Shai et al. 2024, "Transformers represent belief state geometry in their residual stream"), SAEs as sparse dictionary learning, Cheeger inequalities, spectral sparsifiers (Spielman–Teng, Spielman–Srivastava, Batson–Spielman–Srivastava), diffusion wavelets (Coifman–Maggioni), SEBA (Froyland–Rock–Sakellariou 2019), Marshall–Coifman and Landa–Coifman–Kluger on doubly stochastic normalization, Rosasco–Belkin–De Vito / Koltchinskii–Giné on spectral convergence of kernel operators, Wu–Noé VAMP, Kostic et al. on Koopman learning rates.
- Cite precisely (author, year, venue or arXiv id, theorem number where you rely on it). Do not invent citations; if unsure of a reference, say "standard; see e.g." and name a textbook (Bhatia, Matrix Analysis; Stewart–Sun; Horn–Johnson; Chung, Spectral Graph Theory; Levin–Peres).

## Format

- GitHub-flavoured Markdown with LaTeX math (`$...$`, `$$...$$`). Use `\lbrace`, `\rbrace` not `\{`, `\}` (GitHub rendering). No HTML.
- Numbered, stably labelled environments: `**Definition A.1 (name).**`, `**Theorem A.3 (name).**`, `**Proposition**`, `**Lemma**`, `**Corollary**`, `**Remark**`, `*Proof.*` ... `∎`. Labels are `A.n` for Part A clusters, `B.n` for Part B, `S.n` for SPARSITY.md; within a draft number sequentially using your assigned prefix (given in your task).
- Notation must match `THEORY.md`: sites X, |X| = N, Hilbert space H = R^N with counting measure, view φ, normalized φ̂, kernel K_β[x,y] = exp(β⟨φ̂(x),φ̂(y)⟩), Sinkhorn vector v, P = D_v K D_v, centered P̃ = C P C, eigenpairs (λ_i, ψ_i), Q = P^{L'} P^L with singular triples (σ_i, u_i, v_i), loading a_F(ψ) = F^T ψ, block B with projector Π_B, gap δ.
- Each draft goes in `theorems/drafts/<your-file>.md` as told in your task. Do not touch other files except to add a numerical check file under `checks/` if your task says so.

## Report back

In your final message: the path you wrote; a numbered list of every theorem/proposition with its one-line statement and the named assumptions; which proofs you consider fully rigorous vs. which rely on a cited result; anything you believe is false, open, or risky; and open questions for the integrator. Keep the report under 600 words.
