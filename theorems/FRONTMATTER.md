# Coherent modes: theorems, bounds and ideal-case guarantees

Status: v1.0, 2026-11-07. Companion to `THEORY.md` (v1.5), which defines every object used here and is not modified by this document. Proofs were drafted by GPT-6 Astra agents, reviewed independently by GPT-6.1 Sol and Claude Opus 5.5 agents, and revised; reviews are in `reviews/`. Numerical checks asserting the finite-dimensional identities and the planted-regime recovery statements are in `checks/` (see Section 0.4). A subset of the exact identities is formalized in Lean 4 with Mathlib in `lean/` (22 theorems, no `sorry`).

## 0. How to read this document

### 0.1 What it is for

`THEORY.md` makes one move: every measurable property of a site (one place where the network has an activation) defines a kernel on sites, hence a symmetric doubly stochastic operator, hence a spectrum and an SVD. Everything downstream is linear algebra on one shared Hilbert space. Each kernel is a declared choice (which view, which temperature, the bistochastic balancing), and the operators are what is studied. The point of this document is to show what that move buys when it is pushed as far as it will go: which guarantees come for free from the construction's choices, and what the construction recovers *exactly* in idealized regimes where the answer is known.

Part A is the first kind of result. It contains no assumptions about trained networks: existence and uniqueness of the normalization, exactness of every expansion, perturbation bounds for every reported object, and finite-sample rates and arithmetic costs. These results hold for every network, every layer, every recording. Part B is the second kind. Under strong, named idealizations (exact angular clusters, belief-sufficient layers, exactly readable blocks) it shows that the construction's output coincides with a known object: the centroid dictionary that an ideal one-sparse autoencoder would learn, the affine image of the belief simplex, the causal-state partition of computational mechanics, the linear CKA statistic, the sensitivity-weighted stitching predictor. `SPARSITY.md` is the companion for readers who want to know in what sense discrete and sparse structure is recovered, and certified, when it is there.

### 0.2 Epistemic status, stated once

Every statement below is a theorem with a complete proof or a precise reduction to a cited result whose hypotheses are verified in the text. Where a natural-sounding statement turned out to be false, the false version is recorded and the strongest true neighbour is proved; the counterexamples are part of the document. Three things the reader should keep in mind:

- The idealizations of Part B are **named assumptions**, not claims about trained networks. Whether a given layer satisfies them is measured by the diagnostics of `THEORY.md` Sections 4.3 and 11, and the results here say what is recovered when they hold and what the error is when they nearly hold.
- Coined terms ("sparse concept cluster", "ideal dictionary regime", "belief-sufficient layer") are defined where they are first used in Part B (B.20, B.20, B.1 respectively). The name is a mnemonic; the definition is the content. `CLAUDE.md` instructs agents working on the project not to adopt this vocabulary as the working ontology, and that instruction is deliberate.
- Every theorem ends with a **Tractability and sampling** remark. These are the part of the document meant to be used, not admired. The recurring price is $e^{\beta}$: resolution costs constants that grow exponentially in the temperature, and the remarks say so wherever it bites.

### 0.3 Headline results

| | statement | where |
|---|---|---|
| exactness | The bistochastic normalization exists and is unique; spectra lie in $[0,1]$; products of layer operators are doubly stochastic with top singular value exactly one; every expansion of activations, units and gradients in the mode basis is exact with a computable tail | A.1–A.7 |
| limits | At $\beta \to 0$ the centered operator is the centered Gram matrix of the row-normalized activations with an explicit $O(\beta^2)$ remainder, so its eigenvectors are PCA scores and the alignment statistic is linear CKA (centered kernel alignment), given nonzero centered Grams and a spectral gap respectively; at $\beta \to \infty$ it is the projector onto duplicate classes with rate $2(N-1)e^{-\beta\gamma}$ | A.9–A.11, B.42 |
| stability | Isolated blocks move by $\|E\|/\delta$ (Davis–Kahan with constant 1, Wedin); the Sinkhorn normalization is Lipschitz with explicit constants, $\min(e^{4\beta}/N, e^{\beta}/\sqrt2)$ in Frobenius norm and the sharp $\sqrt{N/2}$ in log-kernel units; every reported statistic is continuous; individual vectors inside a degenerate block are not identifiable, while a block separated from its complement by a gap is stable | A.21–A.28 |
| sampling | Empirical blocks converge to population blocks at rate $O(S_{\max}^2 e^{3\beta}\sqrt{\log(1/\eta)/N}/\delta)$, with $S_{\max}$ the Sinkhorn conditioning (the unconditional worst case carries $e^{6\beta}$; vacuity thresholds stated); kernel CCA is consistent with ridge $\rho_N \gg N^{-1/3}$; a $k$-state sufficient summary forces centered predictive rank $\le k-1$; the cross-layer operator $Q$ is estimated at $N^{-1/4}$ unconditionally and at root-$N$ for finitely supported views; dependent positions are handled by document-level sampling | A.43–A.48, A.55–A.56 |
| cost | Dense cost is $O(N^2 d)$ formation plus $O(N^2)$ per Sinkhorn sweep with contraction factor $\le \tfrac12(1+\tanh\beta)$; Nyström and random-feature surrogates have certified error | A.51–A.54 |
| belief states | On a belief-sufficient, belief-Markov layer with affine target means, the predictive-state coordinates are an affine image of the belief simplex and separate exactly the causal states of computational mechanics; counterexamples show each hypothesis is needed, and a perfectly belief-sufficient Mess3 layer has infinite population context rank under a universal kernel, so finite $r_{\text{ctx}}$ is a ridge-and-window-dependent effective rank | B.3–B.14 |
| dictionaries | In the exact angular-cluster regime the top block is spanned by cluster indicators with explicit gap $1 - 2e^{-\beta\Delta}R$; a localizing rotation recovers them; the loadings are the centroid dictionary; every global optimum of a hard one-sparse autoencoder of the same width is that dictionary | B.21–B.28 |
| superposition | Under orthonormal (or explicitly separated) features with positive, balanced coefficients, sparse sums form angular clusters by active set; singleton clusters recover the feature directions; overlapping codes are not identified | B.29–B.30 |
| comparison | $P$ determines the normalized Gram exactly, so comparison across models needs no fitting; stitching damage has an exact local KL identity, and the Section 8 sensitivity-weighted-unreadability predictor is exact under named commutation and site-homogeneous Fisher hypotheses and fails without them (a zero-residual self-stitch with predictor 39.4) | B.41–B.48 |
| sparsity | The bistochastic operator has an $O(N/\epsilon^2)$-edge spectral sparsifier preserving cuts and gaps; $k$ eigenvalues near one certify $k$ low-conductance sets; SEBA (sparse eigenbasis approximation) recovers indicators exactly from a nearby start, and can fail at its published default threshold (reproduced numerical example) | `SPARSITY.md`, S.2–S.8 |

### 0.4 Checks and formalization

Run from the repo root:

```
uv run --with numpy python -I checks/cm_check.py
uv run --with numpy python -I checks/theorems_check.py
```

The second script asserts the finite-dimensional identities and the planted-regime recovery statements of this document and ends with `all theorem checks passed`. The Lean development in `lean/` (`cd lean && lake exe cache get && lake build`) formalizes the doubly stochastic closure, the two-stage dissipation identity, the eigenvalue range of a PSD doubly stochastic matrix, the exact expansion and tail identity, the gradient decomposition, and block-pressure rotation invariance; `lean/README.md` maps theorem names to labels here.

### 0.5 Notation

As in `THEORY.md`: sites $X$, $|X| = N$; $\mathcal H = \mathbb R^N$ with the counting-measure inner product; view $\phi$, normalized $\hat\phi$; kernel $K_\beta[x,y] = \exp(\beta\langle\hat\phi(x),\hat\phi(y)\rangle)$; Sinkhorn vector $v$, $P = D_v K D_v$; centering matrix $C = I - \mathbf 1\mathbf 1^\top/N$; centered $\tilde P = CPC$; eigenpairs $(\lambda_i, \psi_i)$; cross-layer $Q = P^{L'}P^{L}$ with singular triples $(\sigma_i, u_i, v_i)$; loading $a_F(\psi) = F^\top\psi$; block $B$ with projector $\Pi_B$ and gap $\delta$. Labels: A.1–A.11 exactness, A.20–A.31 stability, A.40–A.56 sampling and cost; B.1–B.14 belief states, B.20–B.34 dictionaries, B.40–B.53 comparison and stitching. Each cluster ends with its own "Discrepancies with THEORY.md" list; the consolidated list for the owner is `reviews/theorems-discrepancies.md`. `SPARSITY.md` uses S.1–S.10.
