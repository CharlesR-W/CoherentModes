# Review of A2 (stability) by GPT-6.1 Sol, 2026-11-07

Saved by the orchestrator from the reviewer's report (the reviewer agent could not write files). Scripts: `/tmp/cm-a2-review-sol-scripts/check.py`, `perturbation.py`.

**Overall.** No false theorem, missing substantive hypothesis, or material proof gap found. Sinkhorn Frobenius constant valid for non-PSD adversarial kernels (conservative); Landa–Coifman–Kluger transfer accurately qualified; both counterexamples work.

## A.20–A.22 (Davis–Kahan, Wedin): correct
Weyl gives cross-separation δ−e and perturbed external gaps ≥ δ−2e; entrywise inversion of the diagonal Sylvester equation proves the Frobenius estimate; the √2 operator constant is justified. Yu–Wang–Samworth 2015 Theorem 2 gives $\|\sin\Theta\|_F \le 2\min(\sqrt r\|E\|_{op}, \|E\|_F)/\delta$; the draft's independent δ−e bound is consistent (e < δ/2 ⇒ 1/(δ−e) < 2/δ) and not misattributed. Numerics: 766 Davis–Kahan cases, max actual/bound 0.72 (op), 0.716 (Frob), 0.459 vs YWS; 740 Wedin cases, max 0.44 / 0.70. All assertions passed.

## A.23 (Sinkhorn Lipschitz): correct
Decisive step is the weighted orthogonal-projection inequality $\sum P_{ij} r_{ij}^2 \le \sum P_{ij} h_{ij}^2$, giving exactly $e^{4b}/N$ without PSD. Full differential map tested on 2,044 kernels (exhaustive box corners N=2,3, near-bipartite, unbalanced blocks, N ≤ 10, b ≤ 6): max actual/bound → 1 as b → 0 (0.99998 at b=1e-6), 0.073 at b=1, 2.3e-8 at b=6; finite-pair max Frobenius 0.992, operator 0.333, log-Frobenius 0.900. Finite differences agree with the derivative formula to ~1.6e-10.

**[NIT, optional]** Not sharp at fixed N for large b: retaining $p_{\max} \le 1$ gives $\|\dot P\|_F^2 \le e^{2b}\|H\|_F^2$, so the constant may be replaced by $\min(e^{4b}/N, e^{b})$. Draft already disclaims optimality.

## A.24–A.27: correct
View-displacement, norm-floor normalization, temperature continuation, contraction telescoping, pressure and alignment continuity all check. A.24 correctly avoids a √r factor.

## A.28 (nonidentifiability): correct
Matrices PSD, strictly positive, bistochastic; at θ=0.7, gap 0.099/0.01/1e-6 gives projector distance 0.644 constant while operator perturbation → 0.

## A.29–A.31 (noise): correct
Checked Landa–Coifman–Kluger (PMC8194191): Lemma 1 (hollow positive symmetric, N>2, unique scaling) and Theorem 3 (fixed points, growing d, bounded clean points, pairwise independent centered noise, covariance O(1/d), Frobenius error O_P(d^{-1/2})) as stated; width conversion ε = 2/β correct; draft's second-moment proof valid. A.31 correctly separates positive-diagonal failure from angular distortion; numerics at β=1: full Gaussian τ=1 limiting error 0.520, full sphere 0.242, hollow sphere a=(.9,.8,.7,.6) error 0.0098 with log cross-ratio change 0.03.

## Verdict: approve. Required fixes: none.
