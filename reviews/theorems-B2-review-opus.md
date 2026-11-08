# Review of `theorems/drafts/B2-dictionary.md` (Part B, ideal dictionaries / sparse concept clusters)

Reviewer: Claude Opus 5.5, independent of other reviewers, 2026-11-07. Contract: `theorems/REVIEW-BRIEF.md`. I checked every proof line by line. I also ran adversarial numerical tests (`uv run --with numpy python -I`, scripts in `/tmp/b2rev/t1.py`–`t5.py`). I did not edit the draft.

**Summary.** I found no false statement, and every proof I checked is correct. The problems: one bound is honest but practically vacuous, a few labels and names overclaim, one counterexample sits outside the draft's own m ≥ 2 setting, one standard citation is missing, some notation clashes with THEORY.md, and there is one rendering bug.

## Numerical evidence (inline output)

```
B.21: 300 adversarial exact-ray configs OK; min slack lambda_m-(1-2bR) = 0.0902
B.22: 1253 (config,beta) pairs inside the window, violations=0
 max observed/proven ratios: sin-angle 0.025, cross-mass 1.000; min within/c 1.000, min gap/proven gap 1.000
max Sinkhorn row error 9.9e-14
```
B.22's configurations were adversarial. They used sizes from {1,2,3,10,40}, negative cross dots, orthogonal spread, two antipodal sub-clumps, one outlier, and β up to 80. The cross-mass bound ℓ and the within-entry bound c/n_j are asymptotically tight (ratio → 1 as a → 1, b → 0). The sinΘ bound is conservative by at least 40×. At β = 80 there were 33 apparent "violations". All of them have c < 1e-14, below eigensolver resolution, so the bound cannot be checked there. They are float artifacts, not counterexamples.

```
B.23 exact m=3 (1,2,40) / (5,5,5) / (1,1,30): best restart F = F(U) in all cases
B.23 noisy m=2 (global grid search over O(2)):
(3,4)   s=1e-4 D=0.009 worst err/s=1.41 bound/s=1347
(3,4)   s=1e-2 D=0.905 worst err/s=1.34 bound/s=vacuous
(2,30)  s=1e-3 D=0.679 worst err/s=1.39 bound/s=vacuous
(1,40)  spike perturbation, s=0.05..0.5: err/s = 1.00..1.04
local max-IPR ascent with line search, exact case: 40/40 restarts global for
(1,1,1,1,60), (2,...,2,100), (1,...,1); fixed-step ascent fails 0/40 (step-size artifact)
B.24 N=7 m=2: 63 partitions, min projector distance 0.6614 vs claimed >= 1/N=0.1429
B.24 N=7 m=3: 301 partitions, min projector distance 0.7071; s=0.999/(2N): 0/200 wrong argmin
B.26 two-clump family (cluster of n = two clumps, plus two singletons at dot 1-Delta), m=3:
  n=8 (N-m)eps/Delta=1.58 planted optimal=True; (N-m)eps/Delta=1.93 planted optimal=False
B.27 m=2 extension (orthogonal 2nd cluster, brute force): planted assignment optimal,
  angle(atom, centroid) = 4.6e-6 rad at theta=0.05
B.29: 4000 configs (incl. S subset T, |T|=k, rho<1/(8k)), violations 0;
  max eps/(8rho^2)=0.25, max (cross-(1-gamma_k))/(4rho)=0.18
B.31 counterexample: sin = 1e-15..4e-14 for (m,r^2,beta) in {(2,.1,.3),(3,.45,5),(4,.01,.01)}
B.32 bounds/formulas, B.33 full eigenvalue list (50 random parameter sets): asserted
B.34: max exact P(Y<=Np/2)/exp(-Np/8) over N in {10..1000}, p in {.01..0.9} = 0.916
```

## Findings

**B.20–B.21.** Both are correct. The block-constancy argument (a positive matrix has a unique scaling, combined with within-cluster permutation symmetry) is valid. So are $n_js_j^2\le1$, the row-sum bound $2\ell_j$, and the product inequality for Q in "What it buys". No issues.

**B.22.** The statement is correct. I re-derived $aS_j^2\le n_j\le S_j^2+bS_jT_{-j}$, the bounds $s_x\in[1/(S_j+bT_{-j}),1/(aS_j)]$, the claim $\ell=bR/(a^{3/2}\sqrt{1-q})$, and the within-entry bound $c/n_j$. $P_0$ is PSD and stochastic and leaves $U^\perp$ invariant. The Sylvester integral converges at rate $c-\eta$.
- [OVERCLAIM, minor] "explicitly checkable recovery window": ε, Δ and R are defined relative to a partition. On real data the window can only be checked *a posteriori*, using the partition read off the localized block. Say this explicitly. It is a legitimate certificate, because the theorem applies to any partition that meets its hypotheses.

**B.23.** The exact case is correct, and the raw-varimax equivalence is correct. The noisy proof is also correct: polar alignment, the Lipschitz constant 4 on the unit ball, the counting argument for distinct columns, and $\|O_*-M\|_F^2\le2D$ all check out.
- [OVERCLAIM, moderate] Substituting the definitions, $D<1/2$ means $s<1/(16\sqrt2\,m\,n_{\max})$. At N = 7 the bound is already vacuous at s = 10⁻², and where it is non-vacuous it exceeds the observed error by 400–3700×. The observed error of the global maximizer is linear, ≈ 1.4·s, even under spike perturbations. The draft does disclaim a linear rate, but its tractability remark ("can be far more demanding") understates this. State the threshold in s explicitly.
- [Suggestion] The crude Lipschitz constant 4 can be replaced by $4(n_{\min}^{-1/2}+e)^2$, because columns of $UO$ have sup-norm at most $n_{\min}^{-1/2}$. Combined with the curvature of $\mathcal F(UO)$ at signed permutations, this plausibly gives a linear rate with threshold $s\lesssim n_{\max}^{-1/2}$. Only state it if it is proved.

**B.24.** The statement is correct. The candidate projectors are pairwise at distance ≥ 1/N, the selection is unique for s < 1/(2N), and W is the Procrustes factor.
- [OVERCLAIM, minor] The name "certified" does not fit the theorem as stated, because its hypothesis involves the unknown $U$. The true and useful form is a posteriori. Let $d_*$ be the computed minimum distance. If $d_*<1/(2N)$, then the minimizer is the unique nearest partition and $V$ lies within $d_*$ of it. State that form.
- [NIT] The separation 1/N is very loose. The observed minimum is 0.66 at N = 7. Moving one site out of a cluster of size n already gives about $n^{-1/2}$. Sharpening it would weaken the expensive $s<1/(2N)$ requirement.

**B.25.** Correct.
- [NIT] The proof lists "$\|AB\|_F\le\|A\|\|B\|_F$" twice.

**B.26.** Correct, including the "at most m groups ⇒ a mixed group" step.
- [Suggestion] Add the two-clump example as a remark: the planted optimum fails once (N−m)ε/Δ ≈ 1.6–1.9. This shows the dependence on cluster size is necessary, and that the sufficient condition is within about 2× of tight. Without it, the reader cannot tell whether the dependence on N is an artifact of the proof.

**B.27.** Correct as computed.
- [GAP, minor] The counterexample has m = 1, but Definition B.20 assumes m ≥ 2, and the statement speaks of "atoms" in the plural. Embed it in m = 2 by adding a second cluster orthogonal to the first. Assigning each cluster to its own atom is then optimal (verified by brute force), so the m = 1 computation applies blockwise. Also give the size of the effect: the deviation is third order, 4.6·10⁻⁶ rad at θ = 0.05. "Not exactly the centroid" should not be read as "materially different".

**B.28.** Correct.
- [Suggestion] With nonnegative codes, "distinct lines" can be weakened to "distinct rays": antipodal rays need separate atoms, and the signs are then fixed. This matters because TMoS solutions contain antipodal pairs, and B.21 separates such rays (Δ = 2).

**B.29.** Correct. I verified the normalization lemma, the support-overlap bound $\le\sqrt{(k-1)/k}$, and the perturbation of the cross dot product by 4ρ. The constant 8ρ² is about 4× loose, which is harmless.
- [CITE] The Elhage et al. reference is correct: *Toy Models of Superposition*, Transformer Circuits Thread, 2022, arXiv:2209.10652. Add the venue. The paragraph should also say concretely why TMoS falls outside the hypotheses:
  - active amplitudes are Uniform[0,1], so the within-site ratio is unbounded on multi-active sites;
  - the columns of W are nonorthogonal in superposition.

  Only singleton supports satisfy the hypothesis automatically. This also bears on THEORY §11's TMoS bullet ("about n modes"), which B.29/B.30 show should read "M populated supports".
- [CITE] The Ferreira da Silva & Heimersheim (2026, arXiv:2607.04800v1) title and arXiv id match `reviews/papers.md`. The listed limitations also match: signed Uniform(−1,1) inputs, ReLU gating, overlapping binary codewords, and negative off-code weights.

**B.30.** The mathematical content is correct but thin.
- [NIT/OVERCLAIM] "They neither prove sparse-additive compositionality" is not a mathematical statement. Split this into a precise proposition plus a remark. The proposition: if the only observed support is $S$ with a fixed coefficient profile, there is a continuum of nonnegative two-atom decompositions with identical $K_\beta$, $P$ and loadings.

**B.31.** Correct; the counterexample was verified.
- [NIT, label] This is a standard identity, not a theorem (Stewart–Sun 1990, Thm I.5.5; Bhatia, *Matrix Analysis*, §VII.1). The brief's item 5 applies. Call it a Lemma and drop "sound and complete" from the title.

**B.32.** Correct. The sampling remark $2s/r$ is correct. The draft's Discrepancy 1 against THEORY §4.3 (IPR max 1/r, not 1) is right, and is a useful correction.

**B.33.** Correct. I verified the full spectrum, both windows, the nonemptiness condition $\Delta/\delta>2\log(2Gb)$ and the diffusion interval.

**B.34.** Correct; the Chernoff constant holds with margin.
- [CITE] The projector bound $e_N/(g-e_N)$ is the Davis–Kahan sinΘ theorem (Davis & Kahan 1970, SIAM J. Numer. Anal. 7(1):1–46). The brief lists Davis–Kahan explicitly, so cite it alongside the self-contained proof.

**Notation and rendering.**
- [NIT] Several symbols are reused for different things: R (size ratio and rotation in B.23), s (Sinkhorn entries and subspace error), D (dictionary and a constant), c (cos θ and the mixing constant), e (error and e_N), and b (e^{−βΔ} and the fine-cluster count in B.33).
- The clashes with THEORY.md matter more: $v$ (THEORY's Sinkhorn vector) is used for $e^{-\beta\Delta}$ in B.33, $u$ for a data vector and for $e^{-\beta\delta}$, and $L$ (layer) for a Lipschitz constant.
- [NIT, rendering] In the results table, `$\|\sin\Theta\|$` sits inside a GFM table cell. GFM unescapes `\|` to `|` before math parsing, so the norm renders as single bars. Use `\lVert\sin\Theta\rVert`. No `\{` or `\}` occurrences were found.

## Verdict: **approve with fixes**

Required fixes:
1. B.23: state the noisy threshold explicitly as $s<1/(16\sqrt2\,m\,n_{\max})$, and say the bound is vacuous at typical subspace errors. Either sharpen it (local Lipschitz/curvature) or present the observed linear behaviour as a [hyp].
2. B.24: rename "certified" or restate the theorem in its a-posteriori form (computable $d_*<1/(2N)$ ⇒ unique nearest partition within $d_*$).
3. B.27: extend the counterexample to m ≥ 2 (second, orthogonal cluster), and report that the deviation is third order.
4. B.29: say concretely why Elhage et al. TMoS violates the hypotheses (Uniform[0,1] amplitudes on multi-active sites, nonorthogonal W), and give the venue in the citation.
5. B.31: relabel it as a Lemma (standard identity) with the Stewart–Sun/Bhatia citation. B.30: split the precise statement from the interpretive remarks.
6. B.34: cite Davis–Kahan (1970) for the sinΘ step.
7. B.22: say that the window is checked a posteriori against the partition read off the localized block.
8. Rename the symbols that clash with THEORY.md ($v$, $u$, $L$) and the internal collisions ($R$, $s$, $D$, $b$, $c$).
9. Fix the GFM table pipe (`\lVert…\rVert`) and the duplicated inequality in B.25's proof.

Recommended, not required: add the B.26 two-clump tightness remark, the B.28 nonnegative-code extension to antipodal rays, and the sharper B.24 separation constant.

---

## Re-check of the revised draft (2026-11-07)

Reviewer: Claude Opus 5.5, the same reviewer. I re-read the revised `theorems/drafts/B2-dictionary.md` (552 lines) and ran `uv run --with numpy python -I checks/theorems_b2_check.py`. The run took 0.2 s and ended `all B2 checks passed`. Its B.27 angle is 4.634456102e-06 rad, its B.23 error/τ ratios are 1.414214, 1.414231 and 1.414214, and its wrong-view dictionary difference is √5.

**Status of my nine required fixes**

1. **B.23 threshold: addressed.** $T_{\rm rot}=8\sqrt2\,m\,n_{\max}\tau<1/2$ is indeed equivalent to $\tau<1/(16\sqrt2\,m\,n_{\max})$. The ≈1.4τ behaviour (≈√2 in the script) is labelled empirical, and the draft says explicitly that it is not a linear-rate theorem. One residual slip: the worked number "$m=2$, $n_{\max}=4$ requires $\tau<0.00553$" should read **0.00552**, since $1/(32\sqrt2\cdot4)=0.0055243$. As written, the stated bound admits τ ∈ [0.005525, 0.00553), where the theorem does not apply.
2. **B.24: addressed.** It is restated as an a-posteriori certificate on the computed $d_*<1/(2N)$. The uniqueness argument ($2d_*<1/N$) is correct. The planted corollary ($d_*=\tau$) is correct. The disclaimer that the certificate carries no semantic interpretation is appropriate. The script computes the certificate without the planted labels.
3. **B.27: addressed.** I re-derived the two-cluster argument. Planted loss is ≤ sin²θ < 1, and any mixed group contains an orthonormal pair, whose rank-one loss is ≥ 1, so the planted assignment is the global optimum. I also re-derived the Taylor expansion: the optimal-line angle is $\theta/3-4\theta^3/81$ and the centroid angle is $\theta/3-\theta^3/81$. Their difference is $\theta^3/27+O(\theta^5)$, giving 4.63e-6 rad at θ = 0.05. Both the brute force and the ratio to θ³/27 are asserted in the script.
4. **B.29 TMoS caveat and venue: addressed.** The draft now names the Uniform[0,1] amplitudes, the nonorthogonal $W$, and the singleton exception, and adds the Transformer Circuits Thread venue. The Ferreira da Silva & Heimersheim citation is unchanged and still correct.
5. **B.31 Lemma and B.30 split: addressed.** B.31 is now a Lemma with a Stewart–Sun citation. I could not check the theorem number I.5.5 offline; please confirm it against the book. B.30 is now a precise proposition plus an "Interpretive scope" remark.
6. **Davis–Kahan: addressed.** The full reference is SIAM J. Numer. Anal. 7(1):1–46, 1970.
7. **B.22 a-posteriori: addressed.** The "What it buys" text now says the window is checked against the extracted or validation partition.
8. **Notation: addressed.** The renames are $\omega_\beta$, $\mathcal R$, $\Omega$, $\widetilde v=e^{\beta/2}v$ (correct for $A=e^{-\beta}K$), τ, $\gamma_{\rm mix}$, $T_{\rm rot}$, $J_{\rm rot}$, $E_{\rm load}$, $\xi_\theta$, $b_f$, $z_f$, $z_c$. No clashes with THEORY's $v$, $u$ or $L$ remain. Two minor reuses remain, both harmless and optional to fix: $C$ is both the centering operator (B.32) and the rate constant (B.34), and θ is both an angle (B.27) and the diffusion threshold (B.33).
9. **Rendering and B.25: addressed.** The table now uses `\lVert…\rVert`. The B.25 proof now names the right-multiplication factor $1/\sqrt{n_{\min}}$. No `\{` occurrences remain.

**New material scrutinized**

- **B.26 same-matrix requirement (from Sol): correct and necessary.** The script's example (normalized dictionary $I$ against raw $\mathrm{diag}(2,3)$, difference √5) is valid. The added cluster-size obstruction is also correct: planted cost $n\varepsilon/2$ against $\Delta$, giving 0.11 vs 0.10 at n = 8. Its scope is stated honestly as "not a claim that the constant is optimal".
- **B.28 nonnegative-code extension: correct.** Distinct oriented rays force distinct atoms, and width $m$ with unit norm forces the bijection with positive orientation. The script only asserts that one nonnegative atom cannot fit antipodal data (residual 5). That is weaker than the theorem, but the proof is complete, so this is acceptable.
- **B.30 continuum: correct.** Orthonormal frames with both $\langle w,d_i\rangle>0$ form an open interval of width π/2, and $H$, $K_\beta$, $P$ and the loadings are unchanged across it. One NIT on scope: in $\mathbb R^2$, under B.29's equal-coefficient case (ρ = 0) the frame is unique up to swap, at ±45° from $w$. The continuum therefore relies on unequal fixed ratios (allowed when ρ > 0, within the ratio band $[(1-\rho)/(1+\rho),(1+\rho)/(1-\rho)]$) or on ambient dimension ≥ 3. One sentence saying so would stop a reader from applying it to the ρ = 0 model in the plane.

No new false statements, gaps or overclaims were found.

**Verdict: approve with fixes** (one trivial fix).

Remaining required fix:
1. B.23 tractability remark: change "$\tau<0.00553$" to "$\tau<0.00552$" (exact value 0.0055243).

Optional: the B.30 ρ = 0 / ambient-dimension sentence; renaming the residual $C$ and θ reuses; confirming the theorem number of Stewart–Sun I.5.5.
