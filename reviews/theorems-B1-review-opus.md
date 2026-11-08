# Review of `theorems/drafts/B1-belief.md` (Part B.1: belief, predictive and causal states)

Reviewer: Claude Opus 5.5 (independent), 2026-11-07.

**Summary.** The proofs of B.2–B.5, B.7, B.9 (including B.4's ridge limit) and the counterexamples B.8, B.10, B.11 are correct, and every table number reproduces. The problems are scope: B.6 is an observed first-order Markov chain, and ATM for carried/read sphere targets fails on Shai et al.'s Mess3, so two "What it buys" passages overclaim. Also: one "equivalent" is only sufficient, the FBG citation overstates the paper, B.7's hypotheses are informal, and the easy approximate B.4 (the version that could cover Mess3) is missing.

## Numerical recomputation

`/tmp/b1rev-scripts/review_b1.py` (NumPy; exact population tables; independent of the authors' script).

```text
=== B.6 showcase (k=3, m=5) ===
max |P(y3|y1,y2) - P_b[y2,y3]| = 1.11e-16  -> tokens are an order-1 Markov chain; belief = M[Y_t]
zero-ridge full-L2 CCA (lag 2): [0.112099 0.075472 0. 0.]
rho=0:     corr [0.112099 0.075472]  #>1e-9: 2  affine err of top-2 scores vs b: 1.017e-15
rho=1e-2,1e-4,1e-6: still 2 nonzero corr; affine err 1.81e-01, 7.72e-03, 7.98e-05
one-hot token CCA = same: [0.112099 0.075472]
token Hankel rank / centered: 3 2
=== B.8 ===  future length 1,2,3: Hankel rank 2, centered 1; row3 - mean(row1,row2) = 0; rank Q = 2
=== B.10 === p = [0.25 0.25 0.5], eig P_b: [1/3, 2/3, 1]
sphere-kernel carried corr: [0.444444 0.111111 0.]  (4/9, 1/9)
token Hankel L=1,2,3: rank 2, centered 1
=== B.11 ===
V=3 (V=2 analogous): (1) cache leak corr [1. 1.] rank 2 | (2) read [1. 1.], carried rank 0 | linear update [0.7071 0.7071] | sphere update [1. 1.], excluded 0.333
=== Mess3 (x=0.05, alpha=0.85), stationary filtered beliefs, N=2950, beta=5, 200 kernel test functions ===
carried b_{t+1}|b_{t-1}: relative singular values of centered E[k(.,ref)|b]: [1. 0.907 0.0147 0.0076 0.0067 0.0011]; affine residual 1.37e-02
read-style b_t|b_{t-1}:  [1. 0.907 0.0290 0.0156 0.0122 0.0103]; affine residual 3.73e-02
token 2-window target: affine residual 6.5e-16, centered rank 2
```

The finite-ridge errors differ from the draft's (0.0210, 2.4e-4, 2.4e-6) because they depend on the unspecified $A$; both fall as $O(\rho)$.

## Findings

**1. [OVERCLAIM] B.6 is a low-rank Markov chain, not a mixed-state geometry.** Given $Y_t=a$, the next hidden state is drawn from $m_a$ regardless of $Z_t$. So $Y$ is a first-order Markov chain with transition matrix $P_b=MW$ (deviation 1e-16 above), $b_t=m_{Y_t}$, and the causal states are the $m$ tokens, all synchronized. The "belief-sufficient layer" $h=Am_{Y_t}$ is a rank-$k$ token embedding. The canonical correlations equal those of one-hot token CCA, so neither the nonlinearity nor the normalization contributes anything. The theorem is correct; reword "What it buys" ("nontrivial mixed-state geometry") and "partially observed": the construction recovers the rank $k$ of a low-rank Markov chain, not its $m$ states. The hypothesis "every hidden state has positive stationary mass" is never used.

**2. [OVERCLAIM]/[GAP] With carried or read sphere-kernel targets, ATM is essentially a reset condition, and Mess3 violates it.** The draft should state and prove a lemma like the following. Suppose the target is $\iota(b_{t+\ell})$ with an injective normalized embedding, and the target kernel is characteristic. Then ATM holds iff $b\mapsto\mathcal L(b_{t+\ell}\mid b_{t-1}=b)$ is affine on the support of $b$.
*Proof.* ATM maps every finite affine relation $\sum c_jb_j=0$ with $\sum c_j=0$ to a zero-mass signed measure $\sum c_j\mathcal L(\cdot\mid b_j)$ with zero kernel embedding. The draft's own Jordan-decomposition argument (B.4, density part) then forces that measure to vanish. ∎
B.10 is one instance. On Mess3 (Shai et al.'s process), ATM fails for carried and read targets (affine residual 1.4% and 3.7%; spectrum continues past $d_b=2$); only token-future targets (B.5) satisfy it. The claim that B.4 "makes the geometry reported by Shai et al. a precise ideal-case target for the context construction" therefore holds for the token benchmark, not for the construction's carried or read activation targets, even for a perfectly belief-sufficient network.

**3. [GAP] The easy approximate version is missing; the brief asks for it.** Write the cross operator as $\mathcal T=\mathcal T_{\rm aff}+\mathcal T_{\rm def}$, where $\mathcal T_{\rm aff}$ is the projection of $\mathbb E[g(Q)\mid P]$ onto $\mathcal B$. Under PA, TS and BC, Weyl gives $\sigma_{d_b+1}\le\|\mathcal T_{\rm def}\|$. Wedin gives $\sin\Theta(\text{score span},\mathcal B)\le\|\mathcal T_{\rm def}\|/(\sigma_{d_b}(\mathcal T_{\rm aff})-\|\mathcal T_{\rm def}\|)$. Mess3's spectrum is close to rank 2 (relative third singular value 0.015), so this is the statement that could actually reach Shai-type processes.

**4. [FALSE, minor] ATM: "An equivalent sufficient assumption is $\mathcal L(Q\mid b)=\sum_ib_i\nu_i$ with fixed probability laws."** This is sufficient but not equivalent. Counterexample: $k=2$, support $\lbrace(.4,.6),(.6,.4)\rbrace$, with $Q$ a deterministic label of the support point. ATM holds trivially on two points, but the affine extension to $e_1$ is $3\delta_B-2\delta_A$, a signed measure. For a characteristic target kernel, ATM is equivalent to an affine law with unit-mass *signed* $\nu_i$. For a non-characteristic kernel, the mixture condition is strictly stronger.

**5. [CITE] Fukumizu–Bach–Gretton (2007).** I verified the paper's text. Theorem 1 covers only the **largest** singular pair. It assumes one-dimensional eigenspaces and concerns NOCCO in RKHS norm. Theorem 2 adds the range conditions. Extending to lower-order or multidimensional eigenspaces is only remarked ("not difficult to verify ... extending Lemma 10"; footnote 4), not proved. So "Theorem 1 treats isolated singular directions" and "their text extends the result to isolated eigenspaces" overstate it, and B.6's "isolated blocks when singular values repeat" rests on the remark.
For B.7's rank-by-threshold claim, the correct support is FBG Lemmas 6–7. They give $\|\widehat V_n-V\|\to0$ in operator norm, in probability, under compactness and Eq. (9), with variance term $O_p(\epsilon_n^{-3/2}n^{-1/2})$. Weyl then finishes the argument. Cite those lemmas; the variance term also belongs in "Tractability".

**6. [CITE]/[GAP] The belief object versus Shai et al.** B.1 defines $b_t$ as the infinite-past filter, which is only the recurrent part of the mixed-state presentation (MSP). As I read Shai et al. (NeurIPS 37, pp. 75012–75034, verified), they regress activations at every context position onto finite-context beliefs started from the stationary prior. Those beliefs include transient mixed states. They fit by supervised regression, concatenating layers for RRXOR. At site $(\text{doc},t)$ a transformer carries $\eta(Y_{1:t})$. TS still holds for that filter, but the site law is a mixture over positions, and $p$, $\bar b$ and $\Sigma_b$ change. State the transfer or restrict the comparison.

**7. [GAP] B.7's hypotheses are informal.** The phrase "past view preserves the relevant past information" should be formalized as: $\mathbb E[g(Y_{\rm fut})\mid Y_{\rm past}]$ is $\sigma(P)$-measurable for bounded $g$, and $\mathcal R_{\rm tok}\subseteq\mathcal V_p$. Fix whether "conditional-expectation functions" are taken given $P$ or given $Y_{\rm past}$. Reword "Pairwise separation alone would suffice...". For activation targets, prediction-completeness amounts to ATM plus BO (B.10 fails it).

**8. [GAP] B.4, BC regime 2.** The draft needs to say that $\exp(\beta\langle u,v\rangle)$ is universal on compact subsets of the sphere, which matters for infinite belief supports. This follows from the Taylor-kernel criterion; see e.g. Steinwart and Christmann (2008), *Support Vector Machines*, §4.6. Regime 1 uses unnormalized linear features, so it is the raw-linear variant and not the construction, and it should be labelled as such.

**9. [CITE, verified with nits]**
- Hsu–Kakade–Zhang: §2.3 Condition 1, Lemma 2 and the consecutive-observations remark are as claimed; the state- to edge-emitting translation, B.5 (incl. $w\le k-1$ and the gap formula) and Discrepancy #1 are correct.
- Shalizi–Crutchfield: Theorems 1–3 are prescience, minimality (in statistical complexity) and uniqueness; the "function of any sufficient statistic" step is Corollary 1 with Lemma 7 (refinement). Cite those.

**10. [NIT]**
- B.4 proof: write $\mathbb E[g(Q)\mid P]-\mathbb Eg(Q)$, because the target space is centered.
- B.3 "What it buys" says "$k-1$"; it should say $d_b$.
- Notation clashes with `THEORY.md`: $P$, $Q$ (also B.8's matrix), $B$, $R$, $D$ vs $D_p$, $M/m$. Rename the views, e.g. $\Pi,\Xi$.
- The check script is in `/tmp` with $A$ unspecified; move it to `checks/` with fixed $A$ and seed.
- No `\{` rendering problems.

## Verdict: **approve with fixes**

Required fixes:

1. Reframe B.6 and its "What it buys" as a low-rank observed Markov chain whose belief is a function of the last token. Drop "nontrivial mixed-state geometry" and "partially observed", and remove the unused hidden-mass hypothesis.
2. Add the lemma "ATM with a characteristic target kernel ⟺ the conditional target-belief law is affine on the support" (finding 2). Use it to restrict B.4's Shai et al. claim to token-future targets, and add the Mess3 non-example.
3. Add the quantitative ATM-defect version of B.3/B.4 (Weyl + Wedin, finding 3).
4. Correct "equivalent" in ATM to "sufficient", and give the signed-measure characterization.
5. Correct the FBG scope. For rank counting, cite Lemmas 6–7 plus Weyl, and either drop "isolated blocks when singular values repeat" or prove that extension.
6. Formalize B.7's past-sufficiency hypothesis and say which σ-algebra the conditional expectations are taken over.
7. Address the difference between the finite-context MSP (transient states) and the infinite-past filter when comparing with Shai et al.
8. State and cite universality of the sphere kernel for B.4 regime 2, and label regime 1 as the raw-linear variant.
9. Make the Shalizi–Crutchfield citation precise (Corollary 1, Lemma 7).
10. Move the check script into `checks/` with a fixed $A$ and seed. Fix the notation clashes and the remaining nits.

---

## Re-check of the revised draft (2026-11-07)

I ran `uv run --with numpy python -I checks/theorems_b1_check.py`; it ends `all B1 checks passed`, and its B.6, B.10, B.8, B.11 and Mess3 numbers match mine. The new B.13 check uses a finite-support law with a nonzero defect and asserts the Weyl, angle and ridge-score bounds.

**Original required fixes.** All ten are addressed.
1. B.6 is reframed as a first-order Markov calibration, and the unused hypothesis is gone.
2. B.12 is added, and B.4's "What it buys" is now restricted to token-future targets.
3. B.13 is added.
4. ATM now says "sufficient, generally stronger", with my example and the signed-measure characterization.
5. The FBG scope is correct (Lemmas 6–7 + Weyl; Theorem 1 only for the largest simple pair).
6. B.7 is formalized: predictive sufficiency, feature completeness, conditioning on $\Pi$, finite rank.
7. The finite-context filter versus infinite-past paragraph is added.
8. Universality is proved and regime 1 is labelled raw-linear.
9. Shalizi–Crutchfield is cited as Corollary 1 + Lemma 7.
10. The script is in `checks/` with fixed $A$ and seeds, and the views are renamed $\Pi,\Xi$.

**New material.**
- **B.12: correct.**
  - The equivalence (affine relations ⟺ zero-mass signed measures with zero embedding) is right.
  - The non-affineness argument is valid. Injective word maps let one choose $k+1$ affinely dependent inputs whose finite output sets are pairwise disjoint, so their laws are linearly independent.
  - The Mess3 infinite-support argument is valid. A finite support would be permuted by each injective $F_a$; Perron convergence forces a common fixed point; that point must be uniform; but $F_a(\text{uniform})=e_a$.
  - The parameterization $T_a=\mathsf F\operatorname{diag}(e_a)$ matches mine.
- **B.4 universality: correct.** Every tensor monomial is in the RKHS, and Stone–Weierstrass gives density on compact $S$.
- **B.13 algebra: correct.**
  - The residual bound $\|(I-\Pi_{\mathcal B})\widehat U\|\le\eta/(\gamma-\eta)$ follows because $(I-\Pi_{\mathcal B})\mathcal T_{\rm aff}=0$.
  - Equal dimensions turn it into the projector-difference norm.
  - The ridge-score step uses $\sigma_{\min}(S^{1/2}U)\ge1-\zeta$ and the perpendicular part $\le\eta/(\gamma-\eta)+\zeta$; both are right.

**New finding: B.13's hypotheses fail on Mess3, so the zero-ridge context rank there is infinite.** **[OVERCLAIM]**, and a correction to my own fix 3, which wrongly suggested B.13 could reach Mess3.

The argument for Mess3 at $x=0.05$, $\alpha=0.85$:
- Over the whole simplex, for $a'\ne a$, $F_{a'}(b)_a\le\beta(1-2x)/(\alpha x+\beta(1-x))=0.593407$, where $\beta=(1-\alpha)/2$. Meanwhile $F_a(b)_a>0.593407$ iff $b_a>0.0762$.
- So on $R_a=\lbrace b\in S: b_a>0.5934\rbrace$, which contains $F_a(F_a(S))$ and therefore has positive mass and infinite support, the last symbol and $b_{t-1}=F_a^{-1}(b_t)$ are determined by $b_t$.
- Take $g$ centered and supported in $R_a$. Stationarity gives $\|g\|^2=\mathbb E[p_a(b)g(F_ab)^2]$ and $\|\mathcal Tg\|^2=\mathbb E[p_a(b)^2g(F_ab)^2]\ge\inf p_a\,\|g\|^2$.
- Hence, for full-$L^2$ (universal sphere-kernel) zero-ridge CCA with a belief-valued target, **every** canonical correlation satisfies $\sigma_n\ge\sqrt{\inf p_a}=0.406$ for the read-style target ($b_t\mid b_{t-1}$).
- Restricting to $F_a(R_a)$, which is reached only by the word $aa$, gives $\sigma_n\ge\sqrt{\inf p_{aa}}=0.289$ for every $n$ for the carried target ($b_{t+1}\mid b_{t-1}$).

Condensed output of `/tmp/b1rev-scripts/mess3_noncompact.py` (801-point simplex grid; the last line is from `/tmp/b1rev-scripts/mess3_pieces.py`, 200k-step filtered attractor):
```text
max over simplex of F_a'(b)_a, a'!=a: 0.593407  (closed form 0.593407)
F_a(b)_a > that value  iff  b_a > 0.0762 (grid)
min_simplex F_a(b)_a = 0.3736  (so F_a(F_a(S)) lies in the exclusive region R_a)
read-style corr lower bound 0.4060 ; carried corr lower bound 0.2892
symbol recovered from b_t by nearest image (sampled attractor): error rate 0.0020
```

Consequences:
- $\mathcal T$ is not compact, and $\epsilon\ge\sigma_{d_b+1}(\mathcal T)\ge0.289$, so the defect is large, not small.
- The positive-ridge $\widehat{\mathcal T}$ converges to $\mathcal T$ only strongly, so $e\not\to0$.
- FBG's compactness hypothesis fails as well.
- For a perfectly belief-sufficient Mess3 stream, the population carried or read context rank is infinite. Any finite $r_{\rm ctx}$ is then set by the ridge and the sample size, not by $k-1$.

The draft's caveat that the small unwhitened third singular value "does not certify a small $\epsilon$" is correct; the truth is stronger.

**Rendering bugs from the rename.**
- Line 13: `T=\su\eta_aT_a` should be `\sum_a`.
- B.7's proof: `(\int\ldots)_{i=1}^n_c` should be `^{n_c}`.

**[NIT]** The Hankel factor $\mathsf R_{\mathrm{tok}}$ (B.9) clashes with $\mathcal R_{\mathrm{tok}}$ (B.7).

### Re-check verdict: **approve with fixes**

Remaining required fixes:
1. Add the Mess3 non-compactness result above (or an equivalent proof) to B.12. Belief-valued sphere-kernel targets have all zero-ridge canonical correlations bounded below (≥ 0.406 read-style, ≥ 0.289 carried). Then change B.12's "What it buys" ("B.13 is the appropriate route for a small … defect") and B.13's framing: B.13 does not apply to Mess3's belief-valued targets, and there the context rank is a ridge/sample quantity. Flag this to the integrator for `THEORY.md` §6.3/§11's $r_{\rm ctx}\approx k-1$ hypothesis.
2. Fix the two rendering bugs (`\su\eta`, `^n_c`) and rename the Hankel factor.

---

## Final re-check, second revision (2026-11-07)

I ran `uv run --with numpy python -I checks/theorems_b1_check.py`. It ends `all B1 checks passed` and prints B.14 bounds 0.43732139 (read-style) and 0.32518504 (carried).

**B.14: correct.** I checked each step by hand.
- **Read-style bound.**
  - $p_a=(91+527z)/800$ and $f(2/17)=2/3$ exactly. $\min f=34/91>2/17$, so $F_a^2(\operatorname{supp}\mu)\subset\mathcal R_a$.
  - The cutoff $54/91<2/3$ makes $\mathcal R_a$ exclusive.
  - So $p_a\ge153/800$ on the relevant preimage, giving $c_1=\sqrt{153/800}$.
- **Carried bound.**
  - $p_a(F_ab)\ge1327/2400$ when $(F_ab)_a>2/3$, and $153/800\cdot1327/2400=67677/640000$.
  - $\mathcal R_{aa}=F_a(\mathcal R_a)$ is reachable only by $aa$, because $f(2/3)=1258/1327>2/3$ and $F_a$ is injective.
- **Comparison with my bounds.** The constants exceed my 0.406/0.289 only because the region is smaller: I used the preimage threshold $b_a>0.0762$, B.14 uses $b_a>2/17$, so $\inf p$ is larger. Both sets of bounds are valid; B.14's are sharper.
- **Remaining steps.**
  - The norm identity $\|g\|^2=\int p_a|g\circ F_a|^2d\mu$ follows from stationarity plus exclusivity.
  - Infinite support of $\mu|_{\mathcal R_a}$ follows from the pushforward $F_a^2$ and from $F_a^3$ for $\mathcal R_{aa}$.
  - The weak-null orthonormal-sequence argument gives distance $\ge c_\ell$ from every compact operator.
  - The variational and approximation-number formulation correctly avoids presuming a discrete SVD.

**Compact versus noncompact: stated correctly.** The whitened conditional-expectation (zero-ridge correlation) operator $\mathcal T_\ell$ on $L^2_0(\mu)$ is noncompact. Each positive-ridge smoother $S^{1/2}_{p,\rho}\mathcal T_\ell S^{1/2}_{q,\rho}$ is compact, because a trace-class $\mathscr C_p$ makes $S^{1/2}_{p,\rho}$ compact. So the ridge bias stays $\ge c_\ell$ in norm and vanishes only strongly. The RKHS cross-covariance $C_{pq}=J_p^*\mathcal T_\ell J_q$ is Hilbert–Schmidt with infinite rank, by dense range of $J_q$ and injectivity of $J_p^*$. It is right that FBG's compactness hypothesis on the NOCCO operator fails. The remark that bounded-length documents give finite belief support (so the infinite-rank claim belongs to the stationary infinite-past limit) is correct and useful.

**Framing.**
- B.12's and B.13's "What it buys" now say B.13 does not apply to Mess3's belief-valued targets.
- Discrepancy 8 correctly downgrades $r_{\rm ctx}\approx k-1$ to a ridge-, threshold- and window-dependent effective-rank hypothesis.
- The rendering bugs are fixed: `\sum_a T_a` and `^{n_c}`. The Hankel factor is renamed $\mathsf H_{\rm past}$.

### Final verdict: **approve**

No remaining required fixes.
