# Review of `theorems/drafts/A3-sampling.md` (Part A, sample complexity and cost)

Reviewer: Claude Opus 5.5, independent. Scripts: `/tmp/a3rev/scripts/check{1..4}.py` (`uv run --with numpy python -I`).

## What I verified as correct

I re-derived every constant in A.41–A.54. These are correct:

- **A.41.** The Doeblin contraction $\kappa=1-a/b$, the quotient fixed point, and $\sqrt a/b\le s\le\sqrt b/a$. The bound $M=e^{4\beta}$ is **sharp**: on two antipodal atoms with mass $\pi$ on the rare one, $\sup p/e^{4\beta}=0.9999$ ($\beta=1$, $\pi=10^{-6}$) and $0.9941$ ($\beta=2$).
- **A.42 and A.43.** All steps hold, including the interior-block constant $2\epsilon/(\delta-\epsilon)\le 4\epsilon/\delta$.
- **A.44.** The algebra is right throughout: $\epsilon=Mh$ via $\|s\Phi\|\le R=\sqrt M$; $\operatorname{osc}u\le 2RD$; $|\max u+\min u|\le D$; and $E_N\le(12R+7)Mh$ under the stated side conditions.
- **A.45.** Both bounds hold, including $\|f_i\|_\infty\le\sqrt{M/\lambda_i}$ and $|\widehat\lambda^{-1/2}-\lambda^{-1/2}|\le\sqrt2d/\lambda^{3/2}$.
- **A.47.**
  - The telescoped bound $\Delta_N$ held in 200/200 trials (max err/$\Delta$ = 0.12).
  - The constant $1/(2\rho^{3/2})$ is right (worst observed ratio 0.87).
  - The Gram form $R_q^{1/2}R_p^{1/2}$ of THEORY 6.3 equals the primal $\widehat V_\rho$ to $2.5\times10^{-15}$.
  - The bias inequality and the exponent $N^{-\gamma/(2\gamma+3)}$ are correct.
- **A.48** and **A.50** are correct.
- **A.52.** The Bernstein exponent $3m/(28L)$ is right. The certificate held in 60/60 trials with $m<N$ (max $\|R\|/\tau=1.3\times10^{-3}$).
- **A.53** is correct.
- **A.54.** The Chernoff exponent is at least $\varepsilon^2/6$ (upper tail) and $3\varepsilon^2/16$ (lower tail), so $\varepsilon^2/8$ is safe. The variance check gave 64.08 against 64.25 predicted.

**Citations.** I checked FBG 2007 against the JMLR PDF. Lemma 6 is $O_p(\varepsilon_n^{-3/2}n^{-1/2})$; condition (9) is $n^{-1/3}/\varepsilon_n\to0$; Theorem 1 is about the whitened (NOCCO) directions; Theorem 2 is about centered $L^2(P_X)$ canonical functions under range conditions. Knight (2008), Theorem 4.5, is as described: an asymptotic rate $\sigma_2^2$ under full indecomposability.

The plain iteration $s\leftarrow1/(Ks)$ oscillates in gauge: mean$(x/s_*)$ = 0.5, 2, 0.5, 2, … at both $\beta=1$ and $\beta=4$. This confirms Discrepancy 2.

## Findings

**F1 [GAP] There is no sampling theorem for $Q=P^{L'}P^L$, the central object of THEORY §4.**
- No draft supplies one; A2's Corollary A.26 is deterministic.
- A.44 does not transfer, because the two layers' lifts live in different RKHSs.
- A usable reduction exists. The nonzero singular values of $Q_N=BB^\top AA^\top$ equal those of $\widehat C_{P'}^{1/2}\,\widehat C_{L'L}\,\widehat C_P^{1/2}$, where $\widehat C_{L'L}=N^{-1}\sum_j s'_N s_N\,\Phi'_j\otimes\Phi_j$ (numerically: max diff $2.0\times10^{-15}$). All three factors are bounded empirical means covered by A.42 and the A.44 argument.
- But the operator square root is only $\tfrac12$-Hölder ($\|X^{1/2}-Y^{1/2}\|\le\|X-Y\|^{1/2}$). So this route gives an immediate $O(N^{-1/4})$ bound on singular values and Wedin subspaces, not root-$N$.
- Fix: add that theorem, or state plainly that the $Q$ sampling rate is open. Either way, add a table row.

**F2 [OVERCLAIM] A.44's certificate is vacuous at every practical temperature, and the text does not say so quantitatively.**
- $\epsilon<1$ alone needs $N>e^{8\beta}(1+\sqrt{2\log(2/\eta)})^2$. At $\eta=0.1$ that is about $3.5\times10^4$ at $\beta=1$, $1.1\times10^8$ at $\beta=2$, and $3\times10^{11}$ at $\beta=3$.
- The crude form also needs $(2R+1)\epsilon\le\tfrac12$, and its block certificate scales as $e^{12\beta}$.
- Measured (30 atoms on $S^2$, 40 replicates, 90th percentile):

| $\beta$ | $N$ | lifted error | $E_N$ | crude $(12R+7)Mh$ | error$/(e^\beta h)$ | implied $L_0$ |
|---|---|---|---|---|---|---|
| 0.5 | 20000 | 0.0059 | 3.24 | 7.14 | 0.147 | 0.11 |
| 1 | 20000 | 0.0090 | ∞ | 127 | 0.136 | 0.12 |
| 2 | 20000 | 0.0120 | ∞ | $4.8\times10^4$ | 0.066 | 0.11 |
| 3 | 20000 | 0.0142 | ∞ | $1.9\times10^7$ | 0.029 | 0.044 |

- So the truth here is at most $e^\beta h$, and only the unproved conditioning hypothesis is informative.
- Fix:
  - State the non-vacuity thresholds explicitly.
  - Write the table entry as $O(e^{6\beta}h_N)$, not hidden inside $R$ and $M$.
  - Consider the cheap sharpening that replaces $\sqrt M$ by $S_0=\sup s_\mu$ in the two A.42 applications. This gives $E_N=O(R\,S_0^2e^\beta h)$, roughly $e^{3\beta}$ when $S_0=O(1)$. It is proved by the same argument, with no new hypothesis except the population quantity $S_0$.

**F3 [CITE] A.47's canonical-variate claim overstates FBG Theorem 2.** FBG Theorems 1–2 cover only the *largest* singular value. They assume a one-dimensional top eigenspace (footnote 4) and say the $m$-th case "is not difficult" by extending Lemma 10, but do not prove it. The draft applies Theorem 2 to "the selected nonzero singular value". Fix: restrict the claim to the top value, or label the extension as unproved, or prove it.

**F4 [CITE/GAP] Davis–Kahan and Wedin are invoked "from the stability draft" in an infinite-dimensional RKHS $\mathcal G$** (A.43, A.44, A.45, A.47). But A2's Theorems A.21 and A.22 are stated for finite-dimensional spaces and square matrices, so their hypotheses are not verified as cited. Fix: cite Davis–Kahan (1970, *SIAM J. Numer. Anal.* 7) or Bhatia–Davis–McIntosh (1983) in Hilbert space, or write out A.43's Sylvester argument for bounded self-adjoint operators. Also reconcile the constants: A.43 uses $4\epsilon/\delta$ with $\epsilon\le\delta/4$; A.21 uses $\sqrt2e/(\delta-e)$ with $e<\delta/2$.

**F5 [OVERCLAIM] A.49 is valid but useless at realistic noise scales.** A range-4 Hoeffding bound needs $n\propto\theta^{-2}$, and one-step noise has $\theta\ll1$. Since $X=\|Y-Y'\|^2\in[0,4]$ has $\operatorname{Var}X\le4\theta$, Bernstein applies through the same Hoeffding averaging argument:

$$\Pr(|\widehat\theta-\theta|>u)\le2\exp\!\Big(-\frac{\lfloor n/2\rfloor u^2}{8\theta+8u/3}\Big).$$

Relative error $\alpha$ then needs $n\approx16(1+\alpha/3)\log(2/\eta)/(\alpha^2\theta)$, which is $\theta^{-1}$. Fix: replace or add this bound, and correct "What it buys".

**F6 [GAP] Independent sites versus THEORY's sites $x=(\text{doc},t)$.** The only admissible design offered is one sample per document, which discards nearly all the data. A one-paragraph corollary closes most of the gap. With i.i.d. documents and $T$ positions each, apply A.42 to the document means $T^{-1}\sum_t s\Phi(z_{d,t})$ (same norm bounds), with the position-averaged marginal as $\mu$. A.43–A.48 then hold verbatim with $N$ replaced by $n_{\text{docs}}$. Fix: add it.

**F7 [GAP] A.51.**
- "Projective log error" is never defined. Define $D=\operatorname{osc}(\log c-\log c_*)$.
- $D_0$ refers to an unknown fixed point. Bound it: $s_*(w)/s_*(w')\le R$ gives $\operatorname{osc}\log s_*\le2\beta$, so $D_0\le2\beta$ from $c_0=\mathbf1$. The iteration count is then fully explicit: $\lceil e^{2\beta}\log(2\beta/\varepsilon)/2\rceil$.
- Add the honest corollary of Knight's rate. For the PSD limit, $\sigma_2=\lambda_1$, so Sinkhorn is slowest exactly when a strongly coherent block exists ($\lambda_1\to1$).
- Observed per-cycle ratios: 0.089 vs $\kappa^2=0.75$ ($\beta=1$), 0.62 vs 0.9993 ($\beta=4$); Birkhoff's $\tanh^2\beta$ is slightly better.

**F8 [CITE] A.52 omits the direct precedents.** These are El Alaoui–Mahoney (NeurIPS 2015) and Rudi–Camoriano–Rosasco ("Less is more", NeurIPS 2015). Musco–Musco (NeurIPS 2017, recursive RLS sampling) removes the leverage-oracle cost that the cost row currently leaves open. I did not re-check the numbering of Tropp (2012) Theorem 6.1 against the text; the inequality used is the standard bounded-case Bernstein.

**F9 [GAP] A.44's conditional bound.**
- "On an event of the required probability" must name it ($\ge1-\eta/2$, so the union bound closes).
- "The advertised $O(e^\beta\ldots)$ rate": nothing in THEORY advertises this rate; rephrase.
- State plainly that $L_0$ is a hypothesis about Sinkhorn sensitivity, not proved here. My measurements ($L_0\approx0.04$–$0.14$) make it plausible, not established.

**F10 [NIT] A.47.**
- On its event, $\Delta_N(\rho)$ holds **simultaneously for all $\rho>0$**, since the event involves only the covariances. Say so: the sampling term then survives data-driven ridge selection, which the preamble otherwise forbids.
- The source condition with $\gamma>0$ already implies compactness, because $C^\gamma$ is compact.

**F11 [NIT] Notation.**
- $b$ is redefined in A.46.
- $a,b$ are reused as indices in $C_{ab}$.
- $R=e^{2\beta}$ clashes with THEORY's smoothers $R_p,R_q$.
- $C$ is both THEORY's centering matrix and a covariance.
- $M$ and $M_{\mathrm{ref}}$ are easily confused.
- Results are stated for $P$, but THEORY reports modes of $\tilde P$. Add one sentence that the transfer is by dropping the constant mode, with gaps to $\lambda_0=1$ included.
- No `\{` appears; rendering is fine.

**F12 [NIT]** A.51: the projected eigensolve is $O(\ell^3)$, not $O(N\ell^2)$; Ritz vectors cost $O(N\ell r)$. A.53: the diagonal is exact (union over $N(N-1)/2$ pairs), and matrix Bernstein would sharpen the $N\epsilon$ step when $\|K/N\|\ll b$. A.43: the small-error condition is implied when $\alpha\le1$.

## Verdict

**Approve with fixes.** I found no false statement. The proofs I checked are complete apart from the cited-hypothesis gaps above. The main weaknesses are coverage (no result for $Q$) and the honesty of the constants (A.44 and A.49 certify nothing in the regimes THEORY uses).

## Required fixes

1. Add a $Q=P^{L'}P^L$ sampling result (at least the $N^{-1/4}$ cross-covariance lift of F1), or declare it open in the text and the table.
2. A.44: state the explicit non-vacuity thresholds in $N$ and $\beta$; write $e^{6\beta}$ in the table; and preferably add the $S_0$-sharpened bound (F2).
3. A.47: restrict the FBG Theorem 2 claim to the largest canonical correlation, or label or prove the extension (F3).
4. Replace the finite-dimensional A.21/A.22 citations with Hilbert-space Davis–Kahan/Wedin, or a written proof, and reconcile the constants (F4).
5. A.49: add the Bernstein (variance-adaptive) bound and correct "What it buys" (F5).
6. Add the document-block corollary for (doc, t) sites (F6).
7. A.51: define the projective error; give $D_0\le2\beta$; add the $\lambda_1^2$ slow-down remark (F7).
8. A.52: cite El Alaoui–Mahoney 2015, Rudi–Camoriano–Rosasco 2015 and Musco–Musco 2017, and cost the approximate leverage computation (F8).
9. A.44: specify the conditional event's probability; drop "advertised"; label $L_0$ as an unproved hypothesis (F9).
10. A.47: state that the bound is uniform in $\rho$ (F10); fix the notation clashes and add the transfer to $\tilde P$ (F11).

## Re-check of the revised draft (2026-11-07)

I re-read every changed theorem in full and re-derived the new constants. `uv run --with numpy python -I checks/theorems_a3_check.py` ends with `all A3 checks passed`, including the Q-lift, nonleading canonical-function, Bernstein, document-block and orthogonal-slow-block checks.

### Status of my ten required fixes

1. **Q sampling: done (A.55).**
   - The three-covariance identity $Q=U_b\mathcal K_QU_a^*$ is correct.
   - The cross-covariance factor-change bound is correct: Cauchy–Schwarz against fixed-factor covariances of norm at most $1+\epsilon_j$.
   - The five-event union with $h_N(\eta/5)$ and the Hölder step $\|T^{1/2}-S^{1/2}\|\le\|T-S\|^{1/2}$ are correct.
   - The floor-case Sylvester identity $\widehat\Sigma^{1/2}X+X\Sigma^{1/2}=\widehat\Sigma-\Sigma$ gives $E_j/\sqrt{\nu_j}$. This step uses the fact that $s_N\Phi(z)$ is a scalar multiple of $s\Phi(z)$, so empirical features stay in $\mathcal G_j^0$.
   - Table row added; root-$N$ is honestly left open.
2. **A.44 constants: done.** $B_0=bS_{\max}^2$ enters both A.42 applications correctly, and $S_{\max}^2\le e^{3\beta}$ recovers the $e^{6\beta}$ fallback. The thresholds $N>B_0^2g_\eta^2$ and $N\ge4(2\chi_\beta+1)^2B_0^2g_\eta^2$, and the scalings $e^{12\beta}$ and $S_{\max}^4e^{6\beta}$, are all right.
3. **Nonleading canonical pairs: done, with a direct proof.**
   - I checked the three-term inequality term by term:
     - $\Sigma\preceq\widehat\Sigma+eI$ gives the factor $\sqrt{1+e/\rho}$;
     - the inverse-square-root term is $\sqrt b\,e/(2\rho^{3/2})$;
     - the bias term is $(D_{p,\rho}-I)u_p$.
   - $e/\rho_N=o_P(1)$ follows from $N\rho_N^3\to\infty$.
   - The variance-renormalization step ($\|\widehat a\|\le\rho^{-1/2}$) is correct.
   - The FBG attribution is now accurate.
4. **Hilbert-space Davis–Kahan: done.**
   - The semigroup solution of the Sylvester equation converges at rate $\delta-\epsilon$.
   - The constant-rank Riesz path and the equal-rank projector identity are fine.
   - Dilation constants: the diagonal blocks of the dilation projector are half the singular projectors, so $4\to8$ is right.
5. **Bernstein noise temperature: done.** Exponent $\lfloor n/2\rfloor u^2/(8\theta+8u/3)$, count $8(1+a/3)\log(2/\eta)/(a^2\theta)$, and $a=\alpha/(1+\alpha)$ all check.
6. **Document blocks: done (A.56).** The document means are i.i.d. with the same norm radius, and their expectations are the position-averaged law. The duplicated-position example shows that $D$, not $DT$, is necessary.
7. **Sinkhorn count: done.** $D_t$ is defined; $\max c_*/\min c_*\le\chi_\beta$ gives $D_0\le2\beta$; the count $\lceil(e^{2\beta}/2)\log(2\beta/\varepsilon)\rceil$ is correct. **On the author's disagreement, I concede.** The construction is correct: duplicated antipodal views give $P=J+\tanh\beta\,uu^\top$ and $P'=J+\tanh\beta\,vv^\top$ with $u\perp v$, so $P'P=J$. My phrase "strongly coherent block" conflated a slow mode of one layer with cross-layer coherence. The revised wording ("slow-mode/near-cluster regime ... not equivalent to cross-layer coherence") is the right statement.
8. **Nyström precedents: done.** I checked Musco–Musco (arXiv:1605.07583) Theorem 7: $\tilde K\preceq K\preceq\tilde K+\lambda I$ with probability $1-3\delta$, $\delta\in(0,1/32)$, $O(ns)$ kernel evaluations and $O(ns^2)$ time. $\lambda=N\tau$ is the correct translation; the constant 384 is absorbed in the $O(\cdot)$.
9. **A.44 conditional bound: done.** Event probability $1-\eta/2$; "advertised" removed; $L_0$ labelled unproved.
10. **Done.** Uniformity in $\rho$ is stated, including in the preamble; source condition implies compactness; notation fixed ($\chi_\beta$, $B_\beta$, $\Sigma$, $b_{\mathrm{ctx}}$, $n_{\mathrm{ref}}$); the $\tilde P$ transfer and the separation from $\lambda_0=1$ are stated. No `\{` appears.

### New findings

**R1 [OVERCLAIM, minor] A.55's covariance floor is exactly a finite-support assumption for this kernel.** The sphere kernel is strictly positive definite (a restricted Gaussian kernel). So $\mathcal G_j^0$ is finite-dimensional iff $\operatorname{supp}\nu_j$ is finite, and the floor $\nu_j>0$ then holds automatically, with $\nu_j$ the smallest eigenvalue of the atom-weighted Gram. The name "finite-feature covariance floor" reads more general than it is. Say explicitly that the root-$N$ case is the **finitely supported (finite-state) population**, and that $\nu_j$ can be as small as the smallest atom mass times the smallest Gram eigenvalue.

**R2 [NIT] A.47's range condition is stronger than the $L^2$ proof needs.** With $f_p:=U_pu_p$ (the polar isometry, a unit $L^2_0$ function with the right correlation), $\|\widehat f_{p,N}-f_p\|_{L^2}=\|\Sigma_{pp}^{1/2}\widehat a_{p,N}-u_p\|$. The same three-term split then goes through without assuming $u_p\in\operatorname{ran}\Sigma_{pp}$. The condition is needed only to represent $f_p$ as an RKHS function with coefficient $a_p$. This is optional and worth a remark.

### Verdict (re-check)

**Approve with fixes.** All ten original required fixes are addressed correctly, and I found no errors in the new material.

Remaining required fix:

1. A.55, and the table row's "finite-feature covariance floors": state that for the strictly positive-definite sphere kernel the floor assumption is equivalent to finitely supported view marginals, and say how $\nu_j$ scales (R1).

R2 is optional.
