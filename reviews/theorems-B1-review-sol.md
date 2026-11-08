# Review of B1 (belief, predictive and causal states) by GPT-6.1 Sol, 2026-11-07

Saved by the orchestrator from the reviewer's report (the reviewer agent could not write files). Scripts: `/tmp/b1-review-sol-checks/recompute.py`.

## Finding requiring correction

**[FALSE] B.1 (line 35): the mixture-law assumption is sufficient for affine target means (ATM), not equivalent.** ATM constrains conditional feature means on the *reachable* belief set and need not extrapolate to probability laws at the simplex vertices. Counterexample: reachable beliefs $m_1 = (.4,.6)$, $m_2 = (.6,.4)$, binary target $Q = 0$ at $m_1$, $Q = 1$ at $m_2$; ATM holds, but a representation $\mathcal L(Q\mid b) = b_1\nu_1 + b_2\nu_2$ forces $(\nu_1(Q{=}1), \nu_2(Q{=}1)) = (3, -2)$, not probabilities. Realizable by B.6's reset construction with $M = [[.4,.6],[.6,.4]]$, $W = [[.8,.2],[.2,.8]]$ (belief chain $[[.44,.56],[.56,.44]]$). Local fix: the mixture-law condition is only used as sufficient, so the theorems survive.

## Audit (all correct)

- **B.2–B.3.** Conditional independence factors through $L^2_0(b)$; ATM supplies the finite-dimensional range; centered bound $d_b$, uncentered $k$; positive-ridge whitening preserves rank.
- **B.4.** Zero-ridge argument sound (range of unregularized prediction operator is $\mathcal B$; smoothers converge strongly; finite rank + self-adjointness upgrades to norm convergence; isolated positive block; smoother applied to perturbed singular functions preserves the limit). Belief completeness (BC) *is* satisfiable by the sphere kernel: $\exp(\beta u^\top v)$ on a finite-dimensional sphere is universal (tensor monomials of every degree, polynomials dense), so past accessibility implies BC. **[NIT]** include this universality argument explicitly.
- **B.5.** Nested spaces $V_w$, recursion, no plateau before the full observable space, $w \le k-1$. HKZ §2.3 and Lemma 2 hypotheses match; the draft correctly presents the edge-emitting finite-window result as its own.
- **B.6.** Recomputed: $P_b = MW$, $P_b^\ell = M(WM)^{\ell-1}W$; det(T) 0.0855; stationary belief masses (0.2235, 0.1901, 0.2187, 0.1877, 0.1800); within-view centered rank 4, cross rank 2; zero-ridge correlations 0.1120986861, 0.0754723813; affine reconstruction error 1.53e-15; Hankel ranks 3 / 2.
- **B.7–B.8.** Correct under linear prediction-completeness (strong but explicit). B.8: stationary (0.425, 0.300, 0.275), 3 causal states, centered predictive rank 1. **[NIT]** for the infinite family of word indicators specify summable positive weights when packaging into one Hilbert-valued feature map.
- **B.9.** Correct; $\mathsf H_c = R(I - \mathbf 1\pi^\top)O^\top$ gives the asserted rank.
- **B.10–B.11.** All checked by enumeration: B.10 carried correlations 4/9, 1/9, centered token rank 1; B.11 (V=3) delayed-cache carried correlations 1, 1; embedding carried rank 0; copy-read correlations 1, 1; raw-linear update correlations $1/\sqrt2$ twice; retained sphere-update correlations 1, 1; excluded update mass 1/3.

Shai et al. 2024 used correctly (motivation, not a claim about arbitrary trained layers). Fukumizu–Bach–Gretton rate and source conditions match. Draft script reproduces its values; `checks/cm_check.py` passes.

## Verdict: approve with fixes
1. B.1 line 35: remove the claimed equivalence; state the mixture-law condition as sufficient and generally stronger.

## Re-check of the revised draft (GPT-6.1 Sol, 2026-11-07, saved by the orchestrator)

**Verdict: approve with fixes** (one notation repair, applied by the orchestrator). Original findings resolved: ATM mixture form now "sufficient, generally stronger" with the two-point counterexample; sphere-kernel universality proved in B.4 (tensor monomials, Stone–Weierstrass; β = 0 excluded); summable weights in B.7. **Proposition B.12 approved**: ATM iff conditional target-belief laws preserve finite affine relations, under bounded characteristic kernel and injective measurable target; signed-measure extension sound; Mess3 non-affineness argument sound (invertible word matrices, infinite stationary belief support via Perron–Frobenius, disjoint finite output supports); parameterization reproduces Shai et al.'s Mess3 matrices. **Theorem B.13 approved**: the bound $\eta/(\gamma - \eta)$ follows directly because the rank-$d_b$ reference operator's range is exactly $\mathcal B$; ridge-score correction valid; Mess3 diagnostic does not overclaim certification of the whitened hypothesis. `checks/theorems_b1_check.py` passes (Mess3 carried third/first singular value 0.0150, affine residual 1.40%; read-style 0.0301, residual 4.03%; two-token indicator residual 6.7e-16). Remaining required fix: repair the corrupted `\su\eta_a` → `\sum_a` in B.1 line 13 (done).
