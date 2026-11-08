# Review of B3: fitting-free comparison and linear stitching

**Reviewer:** GPT-6.1 Sol. **Scope:** `theorems/drafts/B3-stitching.md`, against `theorems/REVIEW-BRIEF.md`, `theorems/BRIEF.md`, and `THEORY.md`. No draft or theory files were edited.

## Findings

### 1. [GAP] B.53's example does not distinguish the ideal least-squares stitching responses

**Lines 250–260.** The example correctly establishes different Jacobians, off-site predictions, and responses to arbitrary hidden perturbations. However, its added linear functional vanishes on **every ideal fitted stitch defined in B.43**, not just on the original sampled activations.

Let $n=(1,1,-1)^\top$. Since $Hn=0$, every source matrix $A$ satisfies

$$S_*n=A^+Hn=0,\qquad (AS_*)n=0.$$

Also $(H-tE)n=0$ for every partial stitch. Thus the two displayed downstream models give identical predictions along every such ideal-stitch path. Even applying this minimum-norm fitted map to new source inputs cannot excite the distinguishing direction. The argument's last sentence about “general perturbations” does not prove the ideal-stitch interpretation of “stitching response.”

The proposition is repairable. For the same sampled hidden cloud, instead use

$$z_B(h)=\big(h_1-h_2+c[(h_1-h_2)^2-1],h_3\big).$$

It agrees with $z_A$ at both sites but not at their mean. With source $A=\mathbf1$, the ideal stitch replaces both rows by $(1/2,1/2,1)$. For $c=1$, direct calculation gives

```text
full-stitch empirical KL_A = 0.093622630706
full-stitch empirical KL_B = 0.216890415242
```

Either supply such an explicit ideal-stitch example, or narrow the existing example's conclusion to arbitrary perturbations/non-ideal stitches.

### 2. [OVERCLAIM] B.46's full Fisher decoupling collapses to site-independent Fisher geometry

**Lines 120–146.** The inequalities and proof are correct, but the assumptions have an important consequence missing from the interpretation. When the complete decomposition includes the separate constant block, decoupling from every nonconstant block forces

$$g_x=g\quad\text{for all }x.$$

Indeed, test a constant matrix $V[x]=a$ against every centered matrix $W$: decoupling says $\sum_x a^\top g_xW[x]=0$. Since the nonconstant blocks span the centered space, $g_xa$ must be constant in $x$, for every $a$.

Consequently the sharp sensitivity interval on **every** nonzero site block is the same $[\lambda_{\min}(g),\lambda_{\max}(g)]$. Under the exact block-isotropic assumption, $g=\gamma I$ and all $\gamma_b$ coincide. Thus this exact B.46 regime does not provide genuinely different block-specific Fisher coefficients; block differentiation still comes from loadings and unreadability. The relative-interference variant and B.47's weaker loading-direction assumptions need not suffer this collapse.

State this restriction explicitly rather than leaving “strong” as the only qualification. No change to the proved inequalities is necessary.

### 3. [GAP] B.49 should explicitly require a common evaluation law

**Lines 186–196.** The Hoeffding exponent and union bound are correct, but independence alone concentrates around $m^{-1}\sum_j\mathbb E Z_b(x_j)$, not automatically the single population expectation $\mathbb E Z_b(x)$. Require evaluation sites to be conditionally i.i.d. from the specified population, independently of fitting data; alternatively define the target as the average of their possibly different risks.

For example, $f(x)=1$, $\hat f(x)=x$, and $Z=(1-x)^2$ satisfy standardization and $M=1$. If the target population is concentrated at $x=0$ while independent evaluation sites are concentrated at $x=1$, estimated and target $R^2$ differ by one for arbitrarily large $m$. This is an assumption-precision issue, not a problem with Hoeffding or its citation.

## Statements checked and positive results

- **B.40, B.43, B.51:** Definitions are precise and maintain the distinction between geometry, fitted reconstruction, and descriptive formation. Counting measure and the constant-coordinate convention are consistent.
- **B.41, both directions:** Correct. The diagonal identity $P_{xx}=v_x^2e^\beta$ really recovers every normalized Gram entry. Gram equality extends to a common orthogonal map after spanning/zero-padding; restoring positive norms gives exactly the stated cloud equivalence. Pairwise Gram preservation and the images of the standard basis prove the universal-map converse, including $d=1$. The map/group distinction is handled correctly. Independent numerical recovery/invariance errors were $3.92\times10^{-16}$ and $1.11\times10^{-16}$.
- **B.42:** Both temperature limits, duplicate classes, and the PSD alignment range/equality condition are correct. The low-temperature implicit-function derivative and high-temperature scaling bounds justify the arguments. The CKA/kernel-alignment citations are used only to identify definitions.
- **B.44:** Constants are correct: counting-measure $q(E)$ gives empirical KL coefficient $q(E)/(2N)$; the fixed-baseline Fisher logit discrepancy has neither the $1/2$ nor the empirical averaging factor. $C^2$ logits suffice for the stated remainders. The locality and full-stitch caveats are essential and correctly stated. Re-running the author's nonlinear check reproduced coefficient `0.00915415565648` and KL/quadratic ratio `0.9998633324` at $t=0.001$.
- **B.45 and B.48:** Both counterexamples work. B.45 can even use genuine target-kernel spectral blocks: $H=A=(2,1)^\top$, with constant/contrast blocks, gives exact residual energy approximately $2\times10^{-31}$ but diagonal loading-weighted unreadability `0.900000000000`. B.48's stationary Gaussian example gives empirical KL $1/2$ with zero training gradient.
- **B.46–B.47 are consistent and non-vacuous**, including with actual spectral blocks and softmax outputs. On four sites indexed by $s,t\in\lbrace-1,1\rbrace$, take $H=(2s,t)$, $A=(1,s)$, $\beta=1$, and four downstream logits $z_{u,v}(h)=uh_1+vh_2$, $u,v\in\lbrace-1,1\rbrace$. The spectral frames are $1/2,s/2,t/2,st/2$; eigenvalues are `1, 0.664036770, 0.197375320, 0.131064470`. Here $g=\operatorname{diag}(\operatorname{sech}^2 2,\operatorname{sech}^2 1)$, the two nonzero loading spaces are the coordinate axes, and unreadabilities are `0,0,1,1`. Direct calculation gives

  ```text
  q(E) = sum(s_b^2 u_b)/N = 1.679897366456
  empirical-KL quadratic coefficient = 0.209987170807
  ```

  For $H=(s,t)$ instead, the two loaded modes form one isolated rank-two block with isotropic loading energy and isotropic Fisher sensitivity. Keeping $A=(1,s)$ yields block $R^2=0.5$ and the same nonzero $q(E)$, verifying B.46's equality regime with genuinely partial block readability.
- **B.49:** With the common-law clarification, $M^4$ is the correct squared range; the reported sample count is indeed `2952`. The estimated-denominator ratio bound is valid.
- **B.50–B.52:** The Sinkhorn derivative, PSD inverse bound, Weyl separation, Frobenius sine bound, and bookkeeping margins are correct, with corresponding eigenvalue indices understood as in B.50. No unsupported perturbation theorem is needed. Costs, sampling caveats, and GitHub math notation are otherwise sound.

Verification used `/tmp/cm_b3_sol_review_checks.py`; it ended `all independent B3 reviewer checks passed`. The author's `/tmp/cm_b3_stitch_check.py` and repository `checks/cm_check.py` also passed.

## Verdict: approve with fixes

Required fixes:

1. **B.53:** Prove the ideal-stitch response claim with an explicit distinguishing stitch/example, or explicitly restrict the current example to arbitrary perturbations/non-ideal stitches.
2. **B.46:** Explain the site-independent Fisher consequence of exact complete-block decoupling and the common coefficient forced by exact full-block isotropy.
3. **B.49:** Specify the common evaluation population and conditional independence from fitting data, or change the population target to the appropriate average risk.

## Re-check of the revised draft (GPT-6.1 Sol, 2026-11-07, saved by the orchestrator)

**Verdict: approve.** All three required fixes addressed: B.53 now uses a nonlinear term vanishing at observed activations but not at the fitted mean, giving distinct ideal-stitch KLs 0.0936226307057 and 0.216890415242; B.46 proves full-feature decoupling forces site-constant Fisher geometry, adds restricted-feature variants, and the commutator correction $E_b = R_\perp \Pi_b H + [T, \Pi_b]H$ with $1 \pm \eta$ interference factors is valid (200 randomized cases with real spectral blocks and site-varying PSD Fisher forms all satisfied the corrected bounds); B.49 specifies the common population and independence from fitting data, Hoeffding and Bernstein constants correct. B.45 counterexamples now use genuine spectral blocks ($q(E) = 0$ vs predictor 0.9; $q(E) = 2$ vs bound 1). B.47 valid under site-homogeneous anisotropic Fisher. `checks/theorems_b3_check.py` ends with `all B3 checks passed`. Remaining required fixes: none.
