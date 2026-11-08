# Review of A1 (exactness and structure of the spine) by GPT-6.1 Sol, 2026-11-07

Saved by the orchestrator from the reviewer's report (the reviewer agent could not write files).

**Scope.** Reviewed `theorems/drafts/A1-exactness.md` against the brief and `THEORY.md`, every numbered statement and proof. Independently stress-tested A.4, A.5, A.8, A.9, A.11. No substantive mathematical counterexample found.

## Finding

**[NIT] A.7 should distinguish the use-mediated gradient from the full loss gradient.** The displayed equation writes $G = \nabla_W \mathcal L = \sum_a \Delta_a^\top U_a$; if $\mathcal L$ includes a direct regularizer $R(W)$ the sum is $\nabla_W \mathcal L - \nabla_W R$ (scalar example $W = u = 1$, $\mathcal L = (Wu)^2/2 + W^2/2$: product 1, full gradient 2). The prose anticipates this; the displayed formulas should define $G_{\mathrm{use}}$ or declare the convention, consistently including the residual formula.

## Adversarial checks (all pass)

- **A.4 (degree-normalization contrast, iff):** strict positivity suffices, PSD unnecessary. 300 mismatched and 300 matched-degree positive-matrix tests passed. Sparse/reducible exclusion correctly stated.
- **A.5 (balanced-set relaxation):** with $M = \mathbf 1_B^\top Q \mathbf 1_A$ the complementary transported mass is exactly $N - 2aN + M$; 3702 balanced pairs, max identity error 3.85e-16. Draft correctly does not claim tightness for indicators.
- **A.8 (probe variance constants):** symmetrization essential and done; diagonal variance $(\mu_4 - 1)\sum_i S_{ii}^2$, off-diagonal $4\sum_{i<j} S_{ij}^2$. Exact Gaussian quadrature 21.194594922 = formula; exhaustive Rademacher 10.6765917224 = formula; skew distribution 22.5093453219 = formula (the wrong $2\|M\|_F^2$ would give 37.2894962058).
- **A.9 (remainder $M_b \beta^2$):** quadratic envelope $(9/4 + 3) e^{3b}\beta^2 + \tfrac12 e^{4b}\beta^2 = M_b\beta^2$ holds; 700 configurations/temperatures, max error/R = 0.0742, max R/(M_b β²) = 0.99986. Conservative but valid.
- **A.11 (large-temperature rate $2(N-1)e^{-\beta\gamma}$):** correct; constant 2 is asymptotically sharp (antipodal pair: $\|P - I\|_2 = 2a/(1+a)$, $a = e^{-2\beta}$; ratio to bound at β = 10 is 0.99999999794).

A.1–A.3, A.6, A.10 proofs complete; Knight Theorem 4.4 citation matches; costs honest; no `\{` rendering issues. Repo check prints `all checks passed`. Scripts: `/tmp/cohmod-a1-sol-scripts/check.py`.

## Verdict: approve with fixes

1. In A.7, distinguish the use-mediated gradient from the full gradient when parameter-only loss terms are present; apply consistently to displayed formulas.
