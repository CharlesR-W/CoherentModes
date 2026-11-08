# Review of `theorems/drafts/A2-stability.md` (Part A, cluster 2: stability)

Reviewer: Claude Opus 5.5, independent of other reviewers, 2026-11-07. Scope: mathematics, citations, and honesty of the remarks, per `theorems/REVIEW-BRIEF.md`. Scripts: `/tmp/a2rev/` (Newton-polished Sinkhorn, the exact derivative (A.23.4), hill-climbing worst-case searches). (A.23.4) matches central finite differences to 1e-12.

**Summary.** Every statement I checked is true, and every proof I checked is complete. I re-derived A.21, A.22, A.23 (all steps), A.24–A.29 and the cancellation proof of A.30 line by line, and found no failing step. The problems are quantitative honesty. The Sinkhorn constants (e^{2b}, e^{4b}, e^{8b}) are proof artifacts that the proof's own inequalities already remove. The draft's remarks then present those artifacts as a real "resolution-versus-conditioning cost". At THEORY's default temperatures (β_noise = 1/mean‖δĥ‖² can be ≫1), that turns true theorems into vacuous certificates.

## Findings

**1. [OVERCLAIM] A.23.3 / A.23 derivative: the e^{2b} is an artifact; a b-free bound follows in one line.** Step (A.23.6) uses ‖Ṗ‖_F² ≤ (max P_ij)·Σ P_ij h_ij², with max P_ij ≤ e^{4b}/N. But P is doubly stochastic, so P_ij ≤ 1 as well. Using that cap gives

  ‖Ṗ‖_F ≤ min(e^{2b}, √N)·‖h‖_max,  hence ‖P'−P‖_F ≤ min(e^{2b}, √N)·‖log K'−log K‖_max.

This is sharp up to √2. Take P near block-diagonal with 2×2 blocks of 1/2 (pairs, b large), and h = (1,−1,1) on each block; this h is P-orthogonal to every gauge array a_i+a_j. Then ‖Ṗ‖_F = √(N/2):

```
pairs N=4  b=8: ||dP||_F=1.4142 sqrt(N/2)=1.4142   e^2b=8.89e+06
pairs N=8  b=8: ||dP||_F=2.0000 sqrt(N/2)=2.0000
pairs N=16 b=8: ||dP||_F=2.8284 sqrt(N/2)=2.8284
pairs N=32 b=8: ||dP||_F=4.0000 sqrt(N/2)=4.0000
```

The exact worst case over the vertices of the h-cube at N=4, after hill-climbing over K, saturates at √2 for every b. At larger N, hill-climbing finds a worst case that is flat in b:

```
N=4:  b=0.05 1.001 | b=1 1.264 | b=3 1.411 | b=5 1.414 | b=8 1.414   (bound e^2b up to 8.9e6)
N=12: b=1 1.676 | b=3 1.818 | b=6 2.216                               (sqrt N = 3.46)
```

On clustered sphere kernels (N=60, five clusters in R^16), ‖∂_βP‖_F actually *decreases* with β: it is 0.554, 0.216 and 0.146 at β = 1, 5, 20, against e^{2β} = 7.4, 2.2e4 and 2.4e17. So the tractability sentence "Large b makes the uniform bounds pessimistic: this is a quantified resolution-versus-conditioning cost", and A.25's "growing Lipschitz constants … can defeat fine resolution", describe the proof, not the normalization. The change propagates to η in A.24, to A.25.1–2 and to A.29.

**2. [OVERCLAIM] A.23.1 constant e^{4b}/N.** In the linear-path step, ‖Ṗ‖_F² ≤ (max P_ij)(max P_ij/K_ij²)‖H‖_F². Adding the caps P_ij ≤ 1 and P_ij/K_ij² ≤ 1/K_ij² ≤ e^{2b} gives ‖Ṗ‖_F ≤ [min(1, e^{4b}/N)·min(e^{2b}, e^{4b}/N)]^{1/2}‖H‖_F ≤ min(e^{4b}/N, e^{b})·‖H‖_F. The worst N·c_F found (c_F is the exact top singular value of H ↦ Ṗ at N=4, maximized over K) grows like ≈2e^{b}, not e^{4b}:

```
b=1: N*cF=3.667 (bound 54.6) | b=3: 36.68 (1.6e5) | b=6: 796.7 (2.6e10)
```

The near-1 ratio the draft reports (0.921) is the b→0 regime, where e^{4b}/N is in fact tight. State both regimes.

**3. [OVERCLAIM] A.23.2 operator constant L_b = e^{3b}+2e^{8b}.** The proof is correct, but both factors are far from sharp. The worst-case search below hill-climbs over K and H, so these are lower bounds on the sup; the quantity is N‖Ṗ‖_op/‖H‖_op:

```
            general (L_b)                     sphere/PSD kernel (L_b psd)
N=4 b=1:   3.22  (5982)                      1.64  (129)
N=4 b=3:  27.8   (5.3e10)                    0.43  (3.3e5)
N=6 b=3:  41.1   (5.3e10)                    0.64  (3.3e5)
```

Two observations about the source of the slack:
- The bound λ_min(I+P) ≥ e^{−4b} is loose by about e^{2b}: on anti-clustered kernels the true minimum is ≈2e^{−2b} (6.7e−4 at b=4, against 1.1e−7). Small values do occur.
- The amplified direction cancels anyway. The near-null eigenvector of I+P is the bipartite sign vector s, and D_sP + PD_s vanishes on the heavy cross entries. Bounding ‖w‖ and then ‖D_wP‖ separately throws away exactly this cancellation; the Frobenius route (r is a P-orthogonal projection of h) keeps it.

At minimum, add the alternative ‖Ṗ‖_op ≤ ‖Ṗ‖_F ≤ min(e^{4b}/N, e^{b})‖H‖_F (finding 2). For the PSD spine, also apply the P_ij ≤ 1 cap in the ‖D_wP‖_F step. Then state that e^{8b} is an artifact of the proof.

**4. [OVERCLAIM] A.25 "quantitative block-continuation certificate".** As written, the a priori certificate needs δ_* > √2∫e^{2t}dt over the interval. At β≈5 that is a gap above 3e4 for a unit β-step, so it is vacuous. A usable certificate between grid points needs a bound on ∂_βP̃ over the whole interval. With finding 1 that bound is √N (uniform in β). A computed derivative from one (I+P) solve per grid point plus a second-derivative bound also works, but the draft does not supply the latter. Restate "What it buys" accordingly.

**5. [NIT] A.21's √2 is unnecessary.** After Weyl, the selected eigenvalues of A' lie in [λ_t−e, λ_s+e] and the unselected eigenvalues of A lie outside (λ_t−e−g, λ_s+e+g). That is exactly the interval hypothesis of the Davis–Kahan sinΘ theorem, which gives constant 1 in every unitarily invariant norm (Davis–Kahan 1970; Bhatia, *Matrix Analysis*, Thm VII.3.1). The split-and-stack proof is correct but loses √2, which A.24 and A.25.2 inherit. The (1+√2) in A.22 is fine as stated: the dilation's selected set ±σ_B is two intervals, so the interval form does not apply directly.

**6. [NIT/OVERCLAIM] A.31 obstruction (1) is a fixed-N effect; obstruction (2) is not.** Both proofs are correct (I re-derived the cross-ratio identity and reproduced the 0.0098349 discrepancy). But the remark "the failures persist at fixed four sites as d grows, so neither a wider network nor more precise numerical scaling cures them" omits that more *sites* cure (1). For the limiting kernel (diagonal 1, off-diagonal k=e^{−βτ²}), ‖P−J‖_F ≈ (e^{βτ²}−1)/√N:

```
beta=1: N=4 0.5205 | N=100 0.1681 | N=10000 0.01718
```

Obstruction (2) does persist (hollow, a_i evenly spaced in [0.6,0.9]):

```
beta=1: N=4 0.00983 | N=50 0.00782 | N=400 0.00754;  beta=5: 0.0496 | 0.0391 | 0.0377
```

The diagonal is a finite-N, exponential-in-β nuisance; row-normalization distortion is a systematic bias. That distinction is the useful one for the spine.

**7. [NIT] Scale invariance in A.23.** S(cK) = S(K), so the hypothesis "entries in [e^{−b},e^{b}]" covers any kernel with max/min ≤ e^{2b} once K and K' are rescaled by a common constant (‖H‖ scales with it). For the sphere kernel, b=β is right either way. One sentence prevents misuse.

**8. [NIT] A.27 notation.** THEORY 7.2 uses U ∈ R^{N×p}; the draft renames it p_0 so that p can be the projector distance. Rename the distance instead (e.g. ρ). In addition, "η_A < ‖A‖_F is a sufficient condition for definedness" is redundant once A' and B' are assumed nonzero. Either drop the assumption or drop the sentence.

**9. [CITE] Checked and correct.**
- Landa–Coifman–Kluger 2021, Lemma 1: n>2, symmetric, zero diagonal, positive off-diagonal gives a unique positive scaling.
- LCK Theorem 3: ‖x_i‖≤1 with x_i deterministic; pairwise independent mean-zero noise with ‖Σ_i‖ ≤ C_η/m; n and ε fixed, m→∞; no fourth-moment assumption; Frobenius O_P(m^{−1/2}). The correspondence ε = 2/β is right.
- Yu–Wang–Samworth Theorem 2 is the population-gap variant in arXiv:1405.0680v1 numbering. Say "arXiv numbering" or check it against the Biometrika version.
- Davis–Kahan 1970, Wedin 1972 and Sinkhorn 1964: bibliographic data correct.

Suggestions: cite Stewart–Sun Thm I.5.5 in A.20, and cite Sinkhorn 1964 at the point of use in A.23 rather than only "the exactness draft".

**10. [NIT] A.30 proof** is correct and self-contained. The mean-value step should say that it uses the integral form along the segment, with the derivative bounded on the compact off-diagonal box [e^{−2β}/2, 2], which lies inside the open C¹ domain.

**11. [NIT] Reproducibility.** The draft's numerical checks live in `/tmp/cm-a2-checks/`, outside the repo. Per CLAUDE.md, the integrator should move asserted versions into `checks/`. The improved constants (findings 1–3) and the √(N/2) family are cheap assertions to add.

**Checked without findings:**
- A.20: projector/sine identities.
- A.22: the coefficient singular values |s−τ| and s+τ; the X/Y split.
- A.24: unit-vector inner-product bound; ε ≤ 2ξ/a.
- A.26: telescoping uses only contractivity of P_2' and P_1.
- A.27: telescoping and the normalization inequality.
- A.28: P = aI+(1−a)J+δff^⊤; positivity via |f_if_j| ≤ 1/2.
- A.29: invariance and the e^{2(b+η)} estimate.
- A.30: the R_ij expansion; the trivialization of sphere-valued additive noise via E‖y‖² = 1+E‖η‖².

Rendering: no `\{`, no HTML. Notation matches THEORY v1.5.

## Verdict

**Approve with fixes.** No statement is false and no proof has a gap. The required fixes are about constants and the honesty of the remarks, which the owner treats as mandatory content.

## Required fixes

1. In A.23, add the P_ij ≤ 1 cap and state ‖P'−P‖_F ≤ min(e^{2b}, √N)·‖log K'−log K‖_max. Give the √(N/2) lower-bound family, so that the constant is shown sharp up to √2 and uniform in b.
2. Improve A.23.1 to min(e^{4b}/N, e^{b})·‖H‖_F (the improvement bites when N < e^{3b}, i.e. at the large β of THEORY Section 10 defaults).
3. Replace or supplement the e^{8b} operator constant: give the Frobenius-route bound, apply the cap in the PSD case, and say that e^{8b} is an artifact of the proof (it overestimates by ≥1e9 at b=3).
4. Propagate fixes 1–3 to A.24 (η), A.25 (derivative and integral bounds) and A.29.
5. Rewrite the A.23 and A.25 tractability remarks: remove "quantified resolution-versus-conditioning cost" and "growing Lipschitz constants defeat fine resolution", or support them with an actual lower bound. State what a between-grid-point certificate needs.
6. In A.31's remark, distinguish the finite-N diagonal effect (≈(e^{βτ²}−1)/√N) from the persistent row-normalization bias.
7. A.21: either cite the Davis–Kahan interval form with constant 1, or state that √2 is an artifact of the proof.
8. Minor: scale-invariance sentence (A.23), symbol rename (A.27), Yu–Wang–Samworth numbering, standalone Sinkhorn citation in A.23, and moving the checks into `checks/`.

## Re-check of the revised draft (2026-11-07)

Scope: my eight required fixes, plus the new material in the revision. I ran `uv run --with numpy python -I checks/theorems_a2_check.py`, which ends `all A2 checks passed`, and `checks/cm_check.py`, which ends `all checks passed`. I also ran independent checks in `/tmp/a2rev/t6.py`.

**Status of the required fixes.**

| # | Fix | Status |
|---|---|---|
| 1 | P_ij ≤ 1 cap, b-free log bound, sharp family | Done, and improved to √(N/2) |
| 2 | A.23.1 Frobenius constant | Done: min(e^{4b}/N, e^b/√2) |
| 3 | e^{8b} operator constant | Done: Frobenius route, c_PSD, labelled an artifact |
| 4 | Propagate to A.24, A.25, A.29 | Done (ℓ_N throughout; A.24 operator constant now 1) |
| 5 | Tractability remarks | Done: the vacuity at Section 10 temperatures is stated, and a valid grid-step condition 2∫ℓ_N < δ is given (Weyl) |
| 6 | A.31 large-N behaviour | Done, with an exact formula and a persistence proof |
| 7 | Davis–Kahan constant 1 | Done; the interval hypothesis is verified explicitly |
| 8 | Minor items | Done: scale invariance, ρ_Π, YWS arXiv numbering, Sinkhorn cited in place, permanent check script |

**New material, checked.**

- *(A.23.7) and the √(N/2) constant: correct.* For a zero-sum row x and a probability row p, Cauchy–Schwarz gives Σx²/p ≥ (Σ|x|)² = 4s² ≥ 2Σx². Since Σ Ṗ²/P = Σ P r² ≤ Σ P h² ≤ N‖h‖²_max, the bound follows. The linear-path cap P/K² ≤ 1/m² gives e^b/√2. The finite-b family formula √(2N)/(2+(N−2)e^{−2b}) matches my computation to 12 digits (N=4 and 10; b=0.5 and 2).
- *Scaling obstruction: correct.* At K = c11^⊤, Ṗ = CHC/(Nc) to 1e−16 for random symmetric H. So the additive operator constant necessarily grows like e^b/2 at c = 2e^{−b}. This also explains why my general-kernel search values grew roughly like e^b: that part of my finding 3 was a genuine scaling effect, not an artifact. Only e^{8b} versus e^{b} was the artifact, and the draft now says exactly this.
- *Relative-unit bound* ‖P′−P‖_op ≤ √(N/2)‖H‖_op/min(K,K′): correct, since ‖h‖_max ≤ ‖H‖_max/m ≤ ‖H‖_op/m.
- *c_PSD: correct.* The step v_i² = P_ii/K_ii ≤ e^b holds for any positive kernel; PSD is used only for ‖(I+P)^{-1}‖ ≤ 1.
- *PSD non-sphere fixtures: correct, and the qualification is fair.* I reproduce 38.2659 and 56.2274, equal to N·v₁² (rank-2 PSD K, feasible PSD direction). My table header "sphere/PSD" should have read "sphere kernels": every fixture I tested was a sphere kernel.
- *A.31 kept-diagonal formula: correct.* P − J = (1−k)/(1+(N−1)k)·C exactly. The four-type replicated persistence argument is rigorous: within-type scaling is common by symmetry, the cross-ratio of aggregated entries is m-invariant, and blockwise Cauchy–Schwarz gives ‖D_agg‖_F ≤ ‖D‖_F.
- *A.27 and A.30:* the definedness sentence is now meaningful (A′ and B′ are arbitrary), and the integral-form wording is in place.

**Remaining points (all [NIT], none blocking).**

1. *Sharpness wording in A.23.* "Optimal uniformly over positive kernels" should say that the supremum over b (b→∞) equals √(N/2). At fixed b the family attains only √(2N)/(2+(N−2)e^{−2b}), so ℓ_N(b) is not claimed sharp at fixed b. The family's direction h also leaves the closed box at K_ii = e^b. The draft acknowledges this; one clause stating it would settle it.
2. *Notation in A.23.* "m = min(K_ij, K′_ij)" in the relative-unit sentence reuses m, which the proof defines as e^{−b}. Rename it (e.g. κ_min).
3. *Optional sphere-kernel refinement.* For same-temperature sphere kernels, K_ii = e^β is the maximal entry, so v_i² ≤ e^{−β}. With PSD this gives ‖P′−P‖_op ≤ (1+2√N)e^{−β}‖K′−K‖_op; the worst ratio over 200 random sphere pairs at N = 4–11 and β ≤ 6 was 0.33. This explains the decreasing sphere observations. It is weaker than the log bound for view perturbations, so it is a remark, not a fix.
4. *Optional.* Stewart–Sun Thm I.5.5 for the projector/principal-angle identities in A.20 (suggested before, not adopted).

**Verdict: approve.** All eight required fixes are addressed correctly. The new constants and counterexamples are true, proved and asserted in `checks/theorems_a2_check.py`. No remaining fix is required. Points 1–2 above are recommended wording edits that the integrator can make without re-review.
