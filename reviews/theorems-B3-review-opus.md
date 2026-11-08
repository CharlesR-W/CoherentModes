# Review of `theorems/drafts/B3-stitching.md` (B.40–B.53)

Reviewer: Claude Opus 5.5, independent of other reviewers. Scripts are in `/tmp/b3rev/` (`check.py`, `stitch.py`, `b53.py`, `scale.py`). All were run with `uv run --with numpy python -I`. The draft's own `/tmp/cm_b3_stitch_check.py` reproduces its numbers exactly.

## What I verified as correct

B.41: Gram recovery and invariance (errors about $4\times10^{-16}$), the universal-map argument, and the $2/\sqrt5$ example. B.42: the Jacobian $I+\mathbf1\mathbf1^\top/N$, the cancellation under centering, $1/N\le w_x\le1$, and both limits (alignment $0.358134$ against CKA $0.357963$ at $\beta=10^{-3}$; duplicate-class limit $0.852803$ against predicted $0.852803$). B.44, B.48, B.52 and the B.50 derivation (200 random trials, $\max e/\text{bound}=0.026$). B.49's constant ($m\ge2952$) and its ratio bound.

## Findings

**1. [GAP] B.53 does not establish the "stitching response" claim, and its example provably cannot.**
- Every site-linear stitch (least squares, ridge or affine) gives $AS=T_\lambda H$, so each stitched row is a linear combination of rows of $H$.
- The models differ only by $c\,\ell^\top h$ with $\ell=(1,1,-1)$, and $\ell$ vanishes on $\operatorname{rowspan}(H)$.
- So $z_A=z_B$ on every stitched activation. The local quantities $q(E)$ and $s_b$ also coincide, because they use $J_x$ only on $\operatorname{rowspan}(H)$.

```
input t   ridge=0.0  stitch KL A=[0.120115 0.067131]  B=[0.120115 0.067131]
random 1-d ridge=0.5 stitch KL A=[0.104792 0.044835]  B=[0.104792 0.044835]
nonlinear variant (c*h1*h2, zero on both sites): KL A=[0.07354 0.203985] B=[0.033847 0.221004]
```

The Jacobian and off-site claims of B.53 do hold.

**2. [OVERCLAIM] The B.45 counterexample is not built from spectral blocks, and the natural spectral one is much stronger.**
- In the example $H=\psi_1+\psi_2$, so if $\psi_i=e_i$ the normalized rows are identical. Then $P^B=\mathbf1\mathbf1^\top/2$, whose blocks are $(1,\pm1)/\sqrt2$, not $\psi_1,\psi_2$.
- The proposition is stated for arbitrary decompositions, so it is not false. But Discrepancy 3 uses it against THEORY Section 8, and it is the wrong example for that.
- The decisive example is the **identity stitch $A=H^B$ with the actual spectral blocks of $P^B$**. Generically, finite-$\beta$ modes are nonlinear in $H^B$. So THEORY's predictor reports damage for a stitch whose residual is exactly zero:

```
N=30, beta=4, A=H:  ||E||_F^2=2.6e-29   sum_b l_b U_b = 39.36 (g=I)
mean in-sample R^2 of B's own nonconstant blocks on H^B: 0.134
```

- Without readout reduction the *upper* bound fails too. With $N=2$, blocks $e_1,e_2$, $H=(1,-1)^\top$, $A=(1,1)^\top$ and $g=1$: $q(E)=2$, but $\sum_b\gamma^+\ell^+U_b=1$.

**3. [GAP] B.46 has no approximate version, and readout reduction is a measure-zero condition.**
- On the exactly commuting example of item 4, adding $10^{-3}$ noise to $A$ gives $\max_b\|[T,\Pi_b]\|_2=2.2\times10^{-3}$, and the theorem then says nothing.
- The BRIEF asks for the noisy version where it is easy, and here it is. Since $\Pi_bM-M\Pi_b=[T,\Pi_b]$, we have $E_b=M\Psi_bL_b+[T,\Pi_b]H$. Hence
$$\big|\|E_b\|_F-\|M\Psi_bL_b\|_F\big|\le\|[T,\Pi_b]\|_2\|H\|_F .$$
- This gives two-sided bounds with an additive, computable commutator term. It should be stated alongside the relative-interference relaxation.

**4. Joint satisfiability of B.46 and B.47 on spectral blocks: positive, and worth adding.**
- The draft's own numerical checks use random frames, not spectral blocks of $P^B$. I tested real spectral blocks.
- Construction:
  - Take $N=24$, $\theta_x=2\pi x/N$ and $\hat h(x)=(\cos\theta,\sin\theta,0.7\cos2\theta,0.7\sin2\theta)$.
  - The raw rows are $a_x\hat h(x)$ with random $a_x\in[0.5,2]$. This leaves $P^B$ unchanged and makes the loadings anisotropic.
  - At $\beta=3$, $P^B$ is circulant, its blocks are Fourier pairs, and the minimum gap is $1.6\times10^{-4}$.
  - The source is $A=[\cos\theta,\cos2\theta]$.
- Readout reduction holds to $2.1\times10^{-15}$, and all 13 blocks carry residual.
- Gaussian head, where constant $g$ makes decoupling exact: lower bound $28.80\le q(E)=98.69\le$ upper bound $249.71$.
- Softmax head: $\eta=0.266$, and $[0.70,46.98]\ni11.99$.
- So the hypotheses are jointly satisfiable and non-vacuous, but the interval is wide (a factor of 9). "What it buys" should say so.

**5. [OVERCLAIM] B.47's hypotheses are stronger than needed, and it is not a corollary of B.46.**
- **Weaker sufficient condition.** Let $\mathcal D=\sum_b\mathcal D_b$. It suffices that $g_x$ restricted to $\mathcal D$ equals one fixed form $\bar g$ for all $x$; I'll call this site-homogeneous Fisher. Under binary readability:
  - $q(E_b)=u_b\operatorname{tr}(\bar gL_b^\top L_b)=u_bs_b^2/N$;
  - cross terms are $u_bu_c\operatorname{tr}(\bar gL_c^\top\Psi_c^\top\Psi_bL_b)=0$.
- This needs no isotropy and no directional decoupling. The draft's assumption is the special case $\bar g|_{\mathcal D_b}=\gamma_bI$.
- Check: on the circulant cloud with $A=[\cos\theta,\sin\theta]$ and the anisotropic $g=\operatorname{diag}(1.5,1.5,0.4,0.4)$, both sides equal $14.665017$.
- **Not a corollary.** B.47's directional hypotheses do not imply B.46's block-sensitivity bounds, which range over all of $\mathbb R^{d_B}$. Relabel it as a Proposition.
- **Sharper B.46.** Since the rows of $E_b$ lie in $\operatorname{rowspan}(L_b)$, B.46's $\gamma_b^\pm$ can be restricted to that span, which tightens it.
- For softmax heads, site-homogeneity is non-generic because $F_x$ varies with $p_x$. Say so in the Tractability remark.

**6. [OVERCLAIM] B.50's constant: the $\sqrt N$ is not observed, and the bound is vacuous at realistic $N$.**
- Since $\|P-P'\|_2\le1$ always, the bound is informative only if $2\beta(1+2\sqrt N)\epsilon<1$. At $N=10^4$ and $\beta=10$ that means $\epsilon<2.5\times10^{-4}$. Under random and coherent perturbations, $e/(2\beta\epsilon)$ shows no growth in $N$:

```
N=  10  max e/(2 beta eps)=0.133   draft factor 7.3
N= 160  max e/(2 beta eps)=0.093   draft factor 26.3
N= 640  max e/(2 beta eps)=0.095   draft factor 51.6
```

- **Fix.** Make the computable $e=\|P_{\tau'}-P_\tau\|_2$ (Lanczos on the difference) the primary input to the sine bound, and demote the activation bound to a remark. "Calibration against activation perturbation" overstates what the activation bound delivers.
- **[NIT]** $\theta\le(\pi/2)\sin\theta$ holds on all of $[0,\pi/2]$, so the "$\le1/2$" proviso and the word "locally" are unnecessary.
- Optionally, Yu, Wang and Samworth, *Biometrika* 102 (2015) 315–323, remove the $e<\delta/2$ condition.

**7. [GAP] and [OVERCLAIM] in B.49.**
- **[GAP]** The evaluation sites must be i.i.d. from the site law that defines $R_b^2$. If they are only independent, Hoeffding controls the average of non-identical means.
- **[OVERCLAIM]** The $M=2$ example hides the regime that matters. For a localized mode $f=\sqrt N\psi$ on a cluster of fraction $s$, $Z$ can reach $\sim1/s$. With a 1% cluster, $M^4\approx10^4$ and $m\approx1.8\times10^6$. Bernstein's inequality (variance $\le M^2\,\mathbb EZ$; Boucheron, Lugosi and Massart 2013, §2.8) gives $M^2$ rather than $M^4$ scaling. State both.

**8. [CITE] Missing citations, and verification of the existing ones.**
- Missing:
  - Uniqueness of the symmetric Sinkhorn scaling is used in B.41 and in the permutation argument of B.42 without citation. Use Sinkhorn, *Ann. Math. Statist.* 35 (1964) 876–879 (uniqueness up to scalar; the symmetric case follows), or Knight, *SIAM J. Matrix Anal. Appl.* 30 (2008) 261–275.
  - Schur product theorem: Horn–Johnson, *Matrix Analysis*, §7.5.
  - Centered normalized alignment is Cortes, Mohri and Rostamizadeh, *JMLR* 13 (2012) 795–828. Cristianini et al. define the uncentered version.
- Verified as correct:
  - Cristianini et al., NIPS 14;
  - Kornblith et al., PMLR 97:3519–3529;
  - Hoeffding, JASA 58 (1963) 13–30, Theorem 2 (one-sided; two-sided by symmetry, as used);
  - Davis–Kahan, SINUM 7 (1970) 1–46.

**9. [NIT] Notation, rendering and minor points.**
- Notation clashes:
  - $Q$ is an orthogonal matrix here but $P^{L'}P^L$ in THEORY;
  - $M$ is a model index, $I-T$, and the B.49 bound;
  - $A$ is both a model and the source matrix;
  - $a$ and $e$ are each overloaded.
- `\hbox{or}` should be `\text{or}`. There are no `\{` instances.
- B.41(4) needs a declared gap rule for "isolated blocks". THEORY Section 8 uses blocks of $\tilde P$, where the constant eigenvalue moves to 0.
- B.48: with hard labels, finite-logit cross-entropy *cannot*, rather than "need not", be stationary.
- The verification script should live in `checks/` for reproducibility; that is the integrator's call under the BRIEF.

## Verdict: **approve with fixes**

No statement is false as written. One proof does not show what its statement claims (B.53). The main stitching theorem lacks its easy approximate version. B.47 is stated under hypotheses stronger than needed. The identity-stitch counterexample is the result THEORY Section 8 most needs and is missing.

### Required fixes

1. **B.53.** Replace the linear null term with a nonlinear one (e.g. $c\,h_1h_2$, which vanishes on both sites), or exhibit a stitch into the input layer. Add the remark that site-linear stitches, and all first-order quantities, see the downstream map only on $\operatorname{rowspan}(H)$.
2. **B.45 and Discrepancy 3.** Add the spectral-block identity-stitch counterexample ($E=0$, predictor $39.36$) and the upper-bound failure ($2$ vs $1$). Note that the existing example's blocks are not spectral.
3. **B.46.** Add the commutator-perturbed bound $\big|\|E_b\|_F-\|M\Psi_bL_b\|_F\big|\le\|[T,\Pi_b]\|_2\|H\|_F$ and its consequence for $q(E)$. Optionally restrict $\gamma_b^\pm$ to $\operatorname{rowspan}(L_b)$.
4. **B.47.** Restate under site-homogeneous Fisher on $\sum_b\operatorname{rowspan}(L_b)$, relabel it as a Proposition, and note that it is non-generic for softmax heads.
5. **B.49.** Require i.i.d. sites, discuss the scaling of $M$ for localized modes, and give the Bernstein variant.
6. **B.50.** Lead with the computed $e$, state when the activation bound is vacuous, and drop the unnecessary angle proviso.
7. Add the citations in item 8.
8. Add the circulant joint-satisfiability example (item 4) to the numerical verification, with the interval widths reported honestly.
9. Address the item 9 nits.

---

## Re-check of the revised draft (2026-11-07)

Scope: my nine required fixes, plus the new material. `uv run --with numpy python -I checks/theorems_b3_check.py` ends `all B3 checks passed`; it has 25 assertions, including the corrected-interval and Bernstein budgets. `checks/cm_check.py` still ends `all checks passed`. My independent re-check script is `/tmp/b3rev/recheck.py`.

### Status of the required fixes

1. **B.53. Fixed.**
   - The new null term $d^2-1$, with $d=h_1-h_2$, vanishes on both sites, and I re-derived the argument. Both models predict first-logit probabilities $1/2$ and $\sigma(-2)$ on the sites. The stitch to the site mean gives $\sigma(-1)$ for A and $\sigma(-2)$ for B. Since $\bar p>\sigma(-1)$ by convexity and the cross-entropy derivative $\sigma(u)-\bar p<0$ on $u\le-1$, KL is strictly larger for B.
   - Recomputed independently: $0.093622630706$ vs $0.216890415242$. The B-model Jacobian row $(1+2d,-1-2d,0)$ is correct.
   - The added row-span remark is correct.
2. **B.45. Fixed.** I checked both exact two-site spectral examples by hand. With $P^B=\mathbf1\mathbf1^\top/2$ and blocks $\psi_\pm$:
   - Example 1: $(L_+^2,L_-^2,U_+,U_-)=(9/2,1/2,1/10,9/10)$, so the predictor is $9/10$ while $E=0$.
   - Example 2: $(5,5/9,1/10,9/10)$, so the putative bound is $1$ while $q(E)=\tfrac{10}{9}\cdot\tfrac95=2$.
   - The 30-site self-stitch ($39.3566$) is included, and Discrepancy 3 is rewritten appropriately.
3. **B.46 commutator correction. Fixed and correct.**
   - The identity $E_b=F_b+[T,\Pi_b]H$ has the right sign, the reverse-triangle step is right, and $c_b\le\|[T,\Pi_b]\|_2\|H\|_F$. The way the corrections combine with $\eta$ is valid.
   - The new "restriction of the full-space idealization" is correct and useful: full-feature decoupling with a constant block forces $g_x\equiv\bar g$, and full-feature isotropy forces $\bar g=\gamma I$.
   - Stress test: 150 generic trials, each with a random cloud, random non-commuting source, actual spectral blocks and a site-varying softmax Fisher, gave **0 violations**. They also show that for generic sources the interval is usually uninformative: the lower bound was 0 in 76 of 150 trials, and the median upper bound was $23\times q(E)$. The draft says this qualitatively ("a small commutator need not make the intervals narrow"); see remaining fix R2.
4. **B.47. Fixed.**
   - It is relabelled a Proposition and states site-homogeneous Fisher on $\mathcal D=\operatorname{rowspan}(H)$, without isotropy.
   - The proof is correct, and the identity $\sum_b\operatorname{rowspan}(L_b)=\operatorname{rowspan}(H)$ holds for a complete decomposition. Restricting B.46's sensitivity bounds to $\mathcal D_b$ is also adopted.
5. **B.49. Fixed.**
   - It now has an i.i.d. common law $\nu$.
   - I checked the Bernstein MGF proof line by line: $|\mathbb EY^j|\le b_0^{j-2}v_*$ and $j!\ge2\cdot3^{j-2}$ give $1+v_*t^2/(2(1-tb_0/3))$; the stated $t$ gives exponent $-m\epsilon^2/(2(v_*+b_0\epsilon/3))$; and $t<3/b_0$ holds.
   - Budgets confirmed: $2952$, $1{,}844{,}440$, $76{,}237$.
   - The localized-indicator bound $B_{\rm err}^2\le1/s$ for $s\le1/2$ is correct.
6. **B.50. Fixed.**
   - The computable $d_P$ is now primary.
   - The auxiliary bound now carries $\min(1,\cdot)$, which is justified by $-I\preceq P'-P\preceq I$, together with the explicit vacuity threshold $2.49\times10^{-4}$.
   - The angle proviso is replaced by $\theta\le(\pi/2)\sin\theta$.
7. **Citations. Fixed.** Sinkhorn (1964), Cortes–Mohri–Rostamizadeh (2012), Horn–Johnson §7.5, and Boucheron–Lugosi–Massart §2.8 are added; their hypotheses are stated and match.
8. **Circulant joint-satisfiability example. Fixed.** It is in the draft and asserted in `checks/`, including the perturbed-source run, which uses one perturbed projector consistently. The widths are reported honestly.
9. **Nits. Fixed.**
   - Notation: $O$ for orthogonal matrices, $\mathsf m$ for the model index, $R_\perp$, $H_A$, $B_{\rm err}$, $d_P$, $\epsilon_h$.
   - `\text`; there are no `\{`.
   - A declared gap rule on $\tilde P|_{\mathbf1^\perp}$.
   - B.48 now says "cannot".

### New findings

- **[NIT] Inaccurate remark in B.53, Tractability.** "The nonlinear term above vanishes only at the observed rows" is false: $d^2-1$ vanishes on the hyperplanes $h_1-h_2=\pm1$, e.g. at $(3,2,7)$. It should say it vanishes at the observed rows but not on their row span, in particular not at their mean.
- **[NIT] B.46 Tractability should quantify vacuity.** It should state that for a generic, non-commuting source the corrected interval is typically uninformative: a zero lower bound and a wide upper bound, as in the stress test above. The theorem then certifies validity, not prediction.

No new [FALSE], [GAP], [OVERCLAIM] or [CITE] findings.

### Verdict: **approve**

Remaining fixes are optional; the integrator may apply them without another review round:

- **R1.** Reword the B.53 remark as above.
- **R2.** Add one sentence to B.46's Tractability remark on the typical vacuity of the corrected interval for generic sources.
