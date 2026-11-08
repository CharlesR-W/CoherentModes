# Review of SPARSITY-draft.md — GPT-6.1 Sol

Reviewed against `theorems/REVIEW-BRIEF.md`, `theorems/BRIEF.md`, and the complete `THEORY.md`. Locations below refer to the assigned draft. I checked the original Spielman–Srivastava, BSS, Lee–Oveis Gharan–Trevisan, Froyland (2013), and Froyland–Rock–Sakellariou (2019) papers. **I found no false central theorem.** The remaining fixes concern sampling language, citation hypotheses, and specification rather than the main inequalities.

## Findings

### 1. [OVERCLAIM] The coverage bound is sufficient, not necessary

**S.7, line 134.** “Requires” is not justified by a union bound. Independent sampling gives

$$
\Pr(\text{some group missing})\le r e^{-Np_{\min}},
$$

so the displayed sample count is a **sufficient** condition for coverage with probability at least $1-\zeta$. It is not necessary, even for equally likely groups. For two groups of probability $1/2$, $N=6$, and $\zeta=0.05$, direct calculation gives

```text
exact failure = 0.03125 <= 0.05
N = 6 < union-bound threshold = 7.377758908
```

Replace “requires” with “is ensured by.” The subsequent warning that coverage is not spectral accuracy is correct and important.

### 2. [CITE] The cited resistance-sampling theorem has additional range qualifications

**S.2, line 33.** Spielman–Srivastava, Theorem 1 in the linked [arXiv paper](https://arxiv.org/pdf/0803.0929), assumes $1/\sqrt N<\varepsilon\le1$ and sufficiently large $N$. The draft asserts the sampling construction for all $0<\varepsilon<1$ without addressing those qualifications. This does **not** invalidate the existence statement: retaining the entire graph handles $\varepsilon\le1/\sqrt N$ within the claimed edge budget, and bounded small $N$ can likewise be handled directly. State those cases, or cite a resistance-sampling version covering the full range. BSS Theorem 1.1 does give the other edge count after scalar rescaling.

### 3. [CITE] Froyland does not assume an invariant reference measure

**After S.4, line 100.** “Its … invariant-measure hypotheses” misdescribes Froyland (2013). His Assumption 1 requires a nonnegative square-integrable kernel, $L1_X=1_Y$, and $L^*1_Y=1_X$, between potentially different reference measures. Section 3 explicitly says that the initial measure need not be invariant. Our matrices satisfy the relevant hypotheses, but the reason is **unitality and integral preservation**, not a requirement of dynamical invariance. Name Assumptions 1–2; use Proposition 2 for the variational characterization and **Theorem 2**, not just “§3,” for the coherent-partition upper bound. The draft correctly does not attribute its bipartite rounding argument to that paper.

### 4. [NIT] Make the singular-polar convention explicit

**S.5–S.6.** An SVD-based orthogonal Procrustes update exists, but its polar factor need not be unique. The stated threshold restriction does not prevent this. For

$$
V=\begin{pmatrix}1/\sqrt2&1/\sqrt2\\1/2&-1/2\\1/2&-1/2\end{pmatrix},
\qquad \mu=0.55<1/\sqrt3,
$$

normalized thresholding gives $S=(e_1,e_1)$, and the numerical check returns

```text
rank(S^T V) = 1
S^T V = [[0.70710678, 0.70710678],
         [0.70710678, 0.70710678]]
```

Specify “choose any orthogonal Procrustes minimizer, using an orthogonal SVD completion when singular.” The convergence claim holds for any such choices; no uniqueness or nonsingularity assumption should be added. The SVD wording already points toward this interpretation.

### 5. [NIT] Resolve integration placeholders

**Lines 119, 169, 181.** `B.2x` and `A.2x` are unresolved. Replace them with stable labels or remove the dependencies. S.7 and S.10 already have self-contained proofs. I found no forbidden escaped set braces in this draft.

## Mathematical audit of all numbered statements

- **S.1:** Correct definition. Including loops in degree while cancelling them in the combinatorial Laplacian is consistent.
- **S.2:** Correct substantive bounds. PSD of the original dense operator is essential to $\mathcal L\preceq I$ and hence the norm bound $\varepsilon$. Sparsifying the already normalized operator avoids the Sinkhorn noncommutation problem. For clarity, the final surrogate has $I-P_*=\widehat{\mathcal L}/(1+\varepsilon)$, so its energies and cuts lie between $(1-\varepsilon)/(1+\varepsilon)$ and $1$ times the originals, rather than retaining the intermediate $1\pm\varepsilon$ constants. The product telescoping bound is $e_L+e_{L'}$, with no extra product term needed because all factors are contractions. The block-gap qualifications are adequate.
- **S.3:** Correct. LOGT Theorem 1.1 states $\mu_k/2\le\rho_G(k)\le O(k^2)\sqrt{\mu_k}$ in one-based Laplacian indexing, matching the draft’s $1-\lambda_{k-1}$. Holding mass belongs in the denominator; it contributes zero to differences and cuts. Disjoint sets need not cover the space. The elementary converse proof is valid.
- **S.4:** Correct, including constants. The bipartite chain has second-largest algebraic eigenvalue $\sigma_1$ despite its eigenvalue $-1$. After trimming, leakage is at most $D$ and retained size at least $(1-\kappa)(a+b)/2$; centering multiplies leakage by at most two. No macroscopic balance follows.
- **S.5:** Faithful to FRS 2019 §3, equation (1), Algorithm 3.1. Its objective is orthogonally equivalent to the published one. Calling the output a candidate approximation rather than automatically a basis is necessary.
- **S.6:** Correct for the alternating core. Thresholding is nonzero and uniquely minimizing; Procrustes minimizes the other block. At completed iterates, the current rotation gap is zero and the next thresholding gap tends to zero. Continuity of block minima then gives coordinatewise minimality at accumulation points. This is a supplied proof, not a convergence theorem established by the cited FRS passage.
- **S.7:** Correct quartic/unnormalized-varimax maximizer characterization. Positive weights $1/n_i$ force equality row by row, hence a signed permutation. The nonconstant Hadamard fixed point genuinely refutes arbitrary-initialization recovery; I reproduced residual $7.692\times10^{-16}$ and nearest signed-indicator distance $2$.
- **S.8:** Correct for equally sized matrix shapes and normalized partition indicators. Procrustes, threshold nonexpansiveness, normalization, and loading submultiplicativity yield the stated constants. It promises existence, not algorithmic recovery.
- **S.9–S.10:** Correct. The Sinkhorn Jacobian really is $I+P$; PSD ensures invertibility. Ordered eigenvalues remain continuous at crossings. The power-gap formula is unimodal, with the stated endpoint cases, so strict time superlevel sets are intervals. The minimum-rank count correctly uses strict $>\theta$ for error **at most** $\theta$.

Additional independent numerical checks passed for 100 nonvacuous random-product sweep/trim cases and 300 unequal-size noisy indicator cases; the largest observed S.8 error/bound ratio was $0.331066$. These are checks, not proofs.

## Framing and verdict

The framing respects **“sparsity is a measured regime, never a premise.”** Conditional indicator recovery is not asserted universally; semantic interpretation, overlapping sparse features, canonical spectral blocks, and optional nonconvex postprocessing remain distinct. Preserve that distinction: SEBA is not a fitting-free canonical output merely because its rotation update uses an SVD. Cost and sampling caveats are otherwise appropriately restrained.

**Verdict: approve with fixes.** Required fixes:

1. Correct the necessary-versus-sufficient coverage wording in S.7.
2. Address the parameter qualifications of the cited Spielman–Srivastava theorem.
3. Correct Froyland’s hypothesis description and identify Theorem 2 precisely.
4. Specify the singular orthogonal-Procrustes convention without claiming uniqueness.
5. Resolve or remove the integration placeholders.

## Re-check of the revised draft (GPT-6.1 Sol, 2026-11-07, saved by the orchestrator)

**Verdict: approve with fixes** (one editorial item). Fixes 1–4 addressed (coverage bound now sufficient-not-necessary; Spielman–Srivastava parameter restrictions and full-graph fallback, explicit BSS rescaling; Froyland Assumptions 1–2 and Theorem 2 eq. (8); singular Procrustes handled by any orthogonal SVD completion). New material correct: loop-free higher-order Cheeger reduction ($\mu_i/d_{\max} \le \nu_i \le \mu_i/d_{\min}$, $\Phi_P(A) \le d_{\max}\Phi_W(A)$); nearby-start SEBA basin theorem; default-μ failure reproduced (43/200 recoveries, 193/200 converged; 200/200 at smaller thresholds) and correctly described as empirical; collapsed-column handling. `checks/sparsity_check.py` ends with `all sparsity checks passed`.

Remaining required fix: 1. **[NIT]** resolve the `A.2x` / `B.2x` placeholders (lines 132, 188, 200) at integration.
