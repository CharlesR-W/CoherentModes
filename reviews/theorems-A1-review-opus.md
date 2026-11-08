# Review of `theorems/drafts/A1-exactness.md` (Part A, cluster 1)

Reviewer: Claude Opus 5.5, independent of other reviews. 2026-11-07.

**Method.** I checked every proof line by line and re-derived each constant by hand: the bounds $w_i\in[e^{-3\beta/2},e^{3\beta/2}]$, the three terms of $R(\beta)$, $M_b=\tfrac{21}{4}e^{3b}+\tfrac12e^{4b}$, the balanced-score algebra, the probe-variance constant, and the A.11 row-mass bound. `checks/cm_check.py` and the draft's `/tmp/cohmod-a1-checks/verify_a1.py` both pass. I checked Knight (2008) against the author's text: Lemma 4.1 (unique symmetric scaling), Theorem 4.4 (iteration $x\leftarrow 1/(Ax)$, $\lVert d_{k+2}\rVert\le|\lambda_2|^2\lVert d_k\rVert$) and Theorem 4.5 (the general case, rate $\sigma_2^2$) say what the draft says. Both DOIs resolve. Probe scripts: `/tmp/a1rev-scripts/`.

**Overall.** Careful and almost entirely correct. One false sentence, one hypothesis that fails routinely in practice (so its "What it buys" overclaims), one gap against THEORY's site-exclusion policy, and a tractability remark that misses a clean result about the iteration the repository uses.

## Findings

**1. [FALSE] A.4, the sentence "PSD is sufficient but not necessary here".** PSD is not sufficient. The draft's own Discrepancy 4 gives a counterexample made of PSD matrices:

```
A.4 PSD counterexample: eig K1,K2 >=0: True True  ||S2 S1|| = 1.0     (K1=diag(1,2), K2=diag(2,1))
```

The proof uses only strict positivity, so the sentence should say "PSD is neither needed nor sufficient; strict positivity is what the proof uses."

**2. [OVERCLAIM] A.11 "What it buys", and Discrepancy 6.** "Infinite temperature provably erases discriminative spectral structure … clustering must therefore be a finite-temperature regime" is only true when all normalized rows are distinct. In real data they often are not:

- every document's BOS site, and any two sites with identical prefixes, have identical activations at *every* layer;
- at layer 0 of a model without additive absolute position embeddings (for example a RoPE model), all occurrences of a token share $h^0$.

With duplicate classes $c$ the limit is $P_\infty=\sum_c \mathbf 1_c\mathbf 1_c^\top/|c|$, the exact projector onto the class indicators:

```
rows {0,1,2} identical, rows 3,4 distinct:  beta=40: P[0]=[0.333,0.333,0.333,0,0]  P[3]=[0,0,0,1.0,0]
```

The fix is to state the general limit. The proof is the same row-mass argument applied to $A=\mathrm{blockdiag}(J_c)+E$, plus continuity of the scaling, which follows from the implicit function theorem on A.1's strictly convex potential. Then reword "What it buys" and Discrepancy 6, which treats only the all-identical case. THEORY Section 2 should note it.

**3. [GAP] A.7 versus THEORY [P2].** The identity $G=\nabla_W\mathcal L$ needs every use of $W$ to be indexed by a site in $X$. Under [P2], sites with a zero row in any view are removed from $X$, but their uses of $W$ still contribute to the gradient. In that case $\sum_a\Delta_a^\top U_a$ over $X$ is not $\nabla_W\mathcal L$. Either add the hypothesis "no site with a use of $W$ was excluded", or state $G=G_X+G_{\text{excl}}$ and report $\lVert G_{\text{excl}}\rVert_F$ next to the removed mass. Smaller point: the multi-use step ("reverse-mode accumulation collects…") is informal. The rigorous version is one line: give each use its own copy $W_j$ of the parameter, set $W_j=W$, and apply the chain rule.

**4. [GAP] Sinkhorn tractability (Conventions and after A.1).** The draft analyses only the undamped iteration and says the damped update $w\leftarrow\sqrt{w/(Kw)}$ used in the repository gets no rate. That update has a simple rate.

- *Local rate.* Its Jacobian at $v$ is $\tfrac12(I-D_vPD_v^{-1})$, which is similar to $\tfrac12(I-P)$. The gauge direction gets eigenvalue $0$, because $g(cw)=g(w)$. For PSD $P$ (all sphere kernels) the local rate is $\max_{i\ge1}(1-\lambda_i)/2\le\tfrac12$, *whatever the spectral gap*.
- *Global rate.* In Hilbert's projective metric, inversion is an isometry, $K$ contracts by Birkhoff's coefficient $\tanh(\Delta(K)/4)\le\tanh\beta$ (since $\Delta(K)\le4\beta$ for sphere kernels), and the geometric mean averages, so $g$ contracts by $\tfrac12(1+\tanh\beta)$. Standard; see e.g. Lemmens–Nussbaum, *Nonlinear Perron–Frobenius Theory* (CUP 2012), and Franklin–Lorenz (1989, LAA) on Sinkhorn in the Hilbert metric.

Numerically (two-cluster sphere kernel, $N=200$):

```
beta=5:  lambda_1=0.999891; damped rate/step=0.524 (pred 0.500), 36 sweeps to 1e-12; undamped per-step rate=0.999890 (Knight |lambda_1|)
beta=20: lambda_1=1.000000; damped rate/step=0.502 (pred 0.500), 37 sweeps;          undamped stalls
Hilbert contraction over 300 random (N,d,beta): max observed / (1/2)(1+tanh beta) = 0.9409
```

This justifies THEORY's "a few dozen" locally, while the undamped iteration the draft analyses can be arbitrarily slow on clustered kernels. Also fix the Knight wording:

- Theorem 4.4 assumes $x_0=e$.
- "equivalently, any strictly larger rate" is not equivalent to Knight's statement. It is what his Lemma 4.3 proof, which has an $o(\epsilon)$ term, actually supports.
- State that the theorem numbers are those of the author's text.

**5. [OVERCLAIM] A.4 "What it buys".** A.4 shows that the *trivial* singular value falls below one under degree normalization. It does not show that the nontrivial $\sigma_i$ are free of density artefacts under bistochastic normalization. THEORY 3.1 itself says the continuum operator keeps a density drift ($\alpha=\tfrac12$). "The reason for shared-measure normalization is an exact theorem" should be narrowed to "the trivial mode is protected exactly".

**6. [OVERCLAIM, NIT] A.2 "What it buys".** "A clean separation between smoothing and finite-temperature structure" is not proved by A.2, which says nothing about $\beta$. Delete it or rephrase.

**7. [CITE] Missing relations the contract requires.** A.5 is the discrete balanced form of Froyland's coherent-set relaxation, but the draft never cites Froyland. Cite Froyland (2013, Physica D 250, "An analytic framework for identification of finite-time coherent sets…") or Froyland–Santitissadeekorn–Monahan (2010, Chaos 20), and state the differences: identity transport, two declared smoothers, counting measure. Also cite:

- Davis–Kahan (1970, SIAM J. Numer. Anal. 7) and Weyl for A.10, since THEORY names Davis–Kahan;
- Coifman–Lafon (2006, ACHA 21) for the $1/\pi$ diffusion-distance convention in A.2;
- Marshall–Coifman (2019) for bistochastic normalization;
- for A.1, the classical results: Marshall–Olkin (1968, Numer. Math. 12) for the convex-potential proof and Csima–Datta (1972, J. Combin. Theory A 12) for symmetric scaling with positive diagonal.

Check these bibliographic details before adding them; I quote them from memory.

**8. [NIT] Notation collisions with THEORY.md.**

- $J$ means both $\mathbf 1\mathbf 1^\top$ and the Jacobian $J_x$ (THEORY uses $J^L(x)$ for the Jacobian).
- $G$ means both the Gram matrix and the gradient (THEORY uses $G$ for the gradient).
- $H$ means $CGC$, $H_\beta$ and $H_A$, which clashes with THEORY's activation matrix $H^L$.
- $s$ is used for nnz, the slot count, the sketch and $s_\beta$; $d$ for both degrees and width; $b$ for the block dimension, the $\beta$-bound and $b_{i,a}$; $B$ for the block, the target set and $B_0$.

Rename these, for example $\Gamma=ZZ^\top$ and $\mathbf 1\mathbf 1^\top$ written out. There are no `\{` problems.

**9. [NIT] Labels.** A.3, A.6 and A.7 are identities. "Proposition" is more honest than "Theorem" for A.6 and A.7. A.3's simplicity claim needs only the *first* factor to be strictly positive.

**10. [NIT] Reproducibility.** The verification script lives in `/tmp`, outside the repository. Move it into `checks/` with assertions, including the A.4 PSD counterexample and the duplicate-class limit from finding 2, as CLAUDE.md asks for reviewer counterexamples.

**Correct as stated (verified):**

- **A.1:** proof complete, including the positive-diagonal extension.
- **A.2:** Jensen contraction, the components/multiplicity argument, the tensor-series PSD proof, the $N$-factor distance convention, and the fractional-power counterexample ($-0.014593$, matching the closed form).
- **A.5:** complement mass $N-2aN+M$, the score $=f_B^\top Qf_A$, and both Courant–Fischer forms.
- **A.8:** Monte Carlo with $4\cdot10^5$ draws gives variance 141.1 against the formula's 140.2 (Gaussian), 94.92 against 94.85 (Rademacher), and 112.4 against 113.0 (uniform, $\mu_4=9/5$).
- **A.9:** an adversarial search (300 configurations, including unbalanced antipodal clusters) gives max error$/R(\beta)=0.087$. The empirical $\beta^2$ constant is at most $0.46$, against $5.75$.
- **A.3, A.6, A.10.**
- **A.11 under its stated hypothesis:** $\lVert P-I\rVert$ is about $5\times$ below $2s_\beta$.

## Recommended (not required)

- **A.5 strengthening.** It formalizes THEORY's "$u_i,v_i$ concentrated on the same sites": for symmetric PSD factors, $\lVert\sigma_iu_i-v_i\rVert\le\lVert(I-P^{L'})w\rVert+\lVert(I-P^L)v_i\rVert\le2\sqrt{1-\sigma_i^2}$. Over 200 random layer pairs the observed max ratio to this bound was 0.498.
- **A.10.** The $\sqrt r$ is unnecessary. The Frobenius Sylvester bound applied to $X=(I-\Pi)\Pi_\beta$, which satisfies $HX-XA_\beta=-(I-\Pi)E\Pi_\beta$, gives $\lVert X\rVert_F\le\eta_\beta/(\delta-\eta_\beta)$, because $\eta_\beta$ already bounds $\lVert E\rVert_F$. Observed: true value $1.31\times10^{-5}$, $\eta/(\delta-\eta)=1.44\times10^{-4}$, the draft's bound $2.50\times10^{-4}$.
- **A.8.** Chebyshev with a union bound needs $m_J\propto N^2/\delta$ probes to control all pairs. For Gaussian or Rademacher probes, Hanson–Wright (Rudelson–Vershynin 2013, ECP 18) gives $\log(N^2/\delta)$; say the stated route is pessimistic. Also note that the realized normalized sketch kernel is *exactly* a sphere kernel on the sketch vectors $s_x$, so A.1 and A.2 apply to it exactly.
- **A.7.** A sharper truncation bound is $\lVert A^\top B_0\rVert_F\le\min(\lVert A\rVert_2\lVert B_0\rVert_F,\lVert A\rVert_F\lVert B_0\rVert_2)$.

## Verdict: **approve with fixes**

Required fixes:

1. A.4: replace "PSD is sufficient but not necessary" with "PSD is neither needed nor sufficient; strict positivity is used" (finding 1).
2. A.11: state the duplicate-class limit $P_\infty=\sum_c\mathbf 1_c\mathbf 1_c^\top/|c|$ (or restrict the "What it buys" explicitly), and correct Discrepancy 6 (finding 2).
3. A.7: handle sites excluded under [P2], by hypothesis or by a reported remainder; make the multi-use step rigorous with parameter copies (finding 3).
4. Sinkhorn tractability: add the damped-update local rate ($\le\tfrac12$ for PSD) and the global Hilbert-metric rate, or say explicitly that the implemented algorithm is unanalysed; fix the three Knight wording points (finding 4).
5. Narrow the A.4 and A.2 "What it buys" claims (findings 5 and 6).
6. Add the Froyland, Davis–Kahan/Weyl, Coifman–Lafon, Marshall–Coifman and classical-scaling citations, with verified details (finding 7).
7. Remove the notation collisions with THEORY.md (finding 8).
8. Move the verification script into `checks/` with assertions for the reviewer counterexamples (finding 10).

## Re-check of the revised draft (2026-11-07, Claude Opus 5.5)

I re-read the revised `theorems/drafts/A1-exactness.md` in full and ran `uv run --with numpy python -I checks/theorems_a1_check.py`. It ends `all A1 checks passed`, with 73 `assert` statements; the Hilbert-contraction regression ratio is 0.917. Probe scripts: `/tmp/a1rev-scripts/`.

**My eight required fixes.** All are addressed.

1. **A.4.** Now reads "PSD is neither needed nor, on its own, sufficient". Discrepancy 4 notes that the diagonal counterexample is positive definite.
2. **A.11.** Generalized to $\Pi_{\mathrm{dup}}$. "What it buys" and Discrepancy 6 are corrected, including the alignment limit under different duplicate partitions.
3. **A.7.** Original-loss adjoints, $G_{\mathrm{use},X}+G_{\mathrm{excl}}+G_{\mathrm{direct}}$, and a rigorous parameter-copy proof. There is a new Discrepancy 9.
4. **Sinkhorn.** The damped update is analysed in A.1. All three Knight wording points are fixed: the $x_0=\mathbf 1$ start, no "equivalently", and numbering "in the author's text".
5. **"What it buys" claims.** A.4 now says only that the trivial pair is protected exactly and that density effects remain. A.2's unproved separation claim is removed.
6. **Citations.** All added. See the bibliography check below.
7. **Notation.** $\Gamma$, $\mathbf 1\mathbf 1^\top$, $\mathbf d_i$, $n_{\mathrm{use}}$, $n_{\mathrm{nz}}$, $r_B$, $\zeta_x$ and $\beta_{\max}$ are now distinct. A grep finds no leftover $J$/$G$/$H$ collisions and no `\{`.
8. **Checks.** `checks/theorems_a1_check.py` exists and asserts every reviewer counterexample.

**New material, checked line by line.**

- **A.1, damped Sinkhorn: correct.**
  - *Projective contraction.* The chord bound $\mathrm{TV}\le(b-1)(1-a)/(b-a)$ is right, and maximizing via $\kappa a+a^{-1}\ge2\sqrt\kappa$ gives $\tanh(\Delta_K/4)$. Two rows' likelihood-ratio range is a cross-ratio of $K$, independent of $x$. Integrating the derivative and using subadditivity of osc gives $q_K=\tfrac12(1+\tanh(\Delta_K/4))$, and $\Delta_{K_\beta}\le4\beta$.
  - *The scaling vector itself converges.* $\mathscr T$ is shift-invariant and 1-Lipschitz in sup norm, and shifting by the midrange gives $\lVert\log w_{k+1}-\log v\rVert_\infty\le\tfrac12 q_K^k d_{\mathrm H}(w_0,v)$. This is valid.
  - *Local rate.* $R(\log v)=P$, so $D\mathscr T=(I-P)/2$. This is symmetric, so its spectral norm is $\max_i(1-\lambda_i)/2$, which is at most $\tfrac12$ for PSD $P$. The "eventual, with $\epsilon$ slack" phrasing is the right strength.
  - *Residual conversion.* The step from log-error to row-sum error, $[e^{-2a},e^{2a}]$, is correct.
  - *Scope.* The restriction of the iteration claims to strictly positive $K$ is honest.
- **A.11, duplicate classes: correct.**
  - Uniqueness makes the scaling constant on classes, which gives $a_c\le n_c^{-1/2}$ and $r_c\le e^{-\beta\gamma}\sum_{d\ne c}\sqrt{n_d/n_c}$. The row mass differs by $r_c$ inside the class and $r_c$ outside.
  - $P_\beta\Pi_{\mathrm{dup}}=P_\beta$, because the complement (vectors summing to zero on each class) is annihilated. Hence the difference has rank at most $m$ and norm at most 1.
  - The bound $\kappa_{\mathrm{dup}}\le N-1$ and the alignment limit are fine.
  - Independent test: 200 random configurations with unequal class sizes (1–7 per class), $\beta\in\{2,8,30\}$. The maximum of $\lVert P_\beta-\Pi_{\mathrm{dup}}\rVert_2/\varepsilon_\beta$ is $1.00000004$, at $N=2$ with $\varepsilon=1.9\times10^{-9}$. That excess is about $7\times10^{-17}$ in absolute terms, i.e. roundoff. So the interclass rate is **asymptotically sharp**: for two distinct sites $\lVert P-I\rVert=2r\approx2e^{-\beta\gamma}$.
- **A.7, gradient bookkeeping: correct.** Each untied copy is used once, so $\nabla_{W_j}\mathscr F=\delta_ju_j^\top$, and $\delta_j$ equals the tied-graph total adjoint at the diagonal. The added warning that small removed mass does not bound $\lVert G_{\mathrm{excl}}\rVert$ is right. The sharper columnwise bound $\min(\lVert\mathsf D\rVert_2\lVert\mathsf U\rVert_F,\lVert\mathsf D\rVert_F\lVert\mathsf U\rVert_2)$ is correct.
- **A.5, singular-pair bound: correct.** It follows from $Qv_i-v_i=(P^{L'}-I)w+(P^L-I)v_i$. The added caveat ("vector closeness, not localization") is appropriate.
- **A.10, dropped $\sqrt r$: correct.** It uses $\lVert(I-\Pi)u_j\rVert\le\lVert Eu_j\rVert/(\delta-\eta_\beta)$ and $\sum_j\lVert Eu_j\rVert^2=\lVert E\Pi_\beta\rVert_F^2\le\lVert E\rVert_F^2\le\eta_\beta^2$. This is legitimate because A.9 bounds the Frobenius norm. The Weyl separation step is unchanged.
- **A.8.** The realized normalized sketch kernel is now correctly stated to be exactly a sphere kernel. The Chebyshev route is labelled conservative. Hanson–Wright was not added, which is acceptable because it was optional.

**Bibliography.**

- *Matches Crossref (DOI, title, authors, venue, volume, pages):* Sinkhorn–Knopp (1967, PJM 21(2) 343–348), Csima–Datta (1972, JCTA 12(1) 147–152), Lemmens–Nussbaum (2012; Appendix A "The Birkhoff–Hopf theorem", pp. 255–283), Coifman–Lafon (2006, ACHA 21(1) 5–30), Froyland (2013, Physica D 250, 1–19), Davis–Kahan (1970, SINUM 7(1) 1–46).
- *Matches arXiv:* the titles and authors for arXiv:1711.06711 (Marshall–Coifman) and arXiv:1210.7418 (Froyland).
- *Could not confirm:* Marshall–Coifman's volume and pages, 84(3) 455–482. Crossref has no volume or pages, and the OUP page returned 403.
- *From my own knowledge:* Horn–Johnson Theorem 4.3.1 is Weyl's inequalities, as I recall.

**Remaining issues (all [NIT], none blocking).**

1. The Marshall–Olkin "digitized original" link (`digizeitschriften.de/.../log14.pdf`) returns HTTP 404. Replace it with a working link or the Springer DOI, or drop the link; the bibliographic details (Numer. Math. 12, 83–90, 1968) are right.
2. Confirm Marshall–Coifman 84(3) 455–482 against the publisher page.
3. The A.5 proof writes $P^{L\prime}$ where the statement writes $P^{L'}$; use one form.

**Verdict on the revision: approve.** No required fixes remain.
