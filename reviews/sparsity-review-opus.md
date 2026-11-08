# Review of `theorems/drafts/SPARSITY-draft.md` (Claude Opus, independent)

Contract: `theorems/REVIEW-BRIEF.md`. I checked every statement by hand, ran numerical checks (scripts in `/tmp/sparsity-review/`, run with `uv run --with numpy python -I`), and had a sub-agent check the cited sources against the primary texts. The draft is careful and nothing in it is false. The problems are one citation that relies on an unverified extension, one citation that misstates its source's hypotheses, a weak SEBA counterexample, and some framing that goes beyond what the theorems prove.

## Verified as correct

Every check below found 0 violations. Scripts are in `/tmp/sparsity-review/`.

- **S.2.** $P_*$ is nonnegative, doubly stochastic and PSD. In fact $P_*\succeq P$. The norm bound holds: at $N=60$ with $\varepsilon=0.302$, $\|P_*-P\|=0.390$ against a bound of $0.464$.
- **S.3.** The two-way inequality held in 30 instances with $N=10$, and the converse $h_3\ge(1-\lambda_2)/2$ held in 10 instances with $N=8$, both by exhaustive search. The converse proof is correct.
- **S.4.** The identity, the lift $T$ (cut $=a+b-2m$), the fraction bounds, the trimming step and the factor 2 are all correct. A sweep on $T$ over 298 instances with $\kappa<1$ found no failures.
- **S.6.** The block-coordinate argument is correct.
- **S.7(1)–(3).** Correct.
- **S.8.** Correct, checked in 500 trials.
- **S.10.** Correct, checked in 300 trials.
- **Planted example.** Correct: $e^\beta=37$, $h=0.05$, transport $0.905$, score $0.81$.

## Findings

**F1 [CITE/GAP] S.3, higher-order Cheeger with loops.** LOT (2014), §2, defines $w(v)=\sum_{u\sim v}w(u,v)$ on graphs and never mentions self-loops. The paper does not claim the theorem for reversible chains or for arbitrary vertex measures. The draft's sentence "the weighted proof allows holding weights … removing loops from the denominator would be incorrect" is therefore an unverified extension of a cited theorem. It is not a citation. Deleting loops is no reduction: it changes the vertex measure to $1-P_{xx}$, which gives $\mu_k(P)\le\lambda_k(W)$, the wrong direction for the upper bound. This matters most in exactly the clustered high-$\beta$ regime, where $P_{xx}$ is large. Fix: cite a checked statement for finite reversible Markov kernels, or prove that LOT's argument holds with vertex measure $\ge$ degree. Otherwise, restate the upper bound for loopless $W$ with explicit conversion factors. The converse is proved directly and is fine.

**F2 [CITE] §2, Froyland (2013).** The draft says "compactness and invariant-measure hypotheses". The paper explicitly states that $\mu$ need *not* be invariant; the hypotheses are its Assumptions 1–2 (an $L^2$ kernel giving compactness, normalisation conditions, and simple $\sigma_1$). The coherent-set upper bound is **Theorem 2, eq. (8)**, not a proposition. Proposition 1 (§2) is a general singular-value minimax, and Proposition 2 (§3) covers the constant pair and the variational formula.

**F3 [OVERCLAIM, needs replacement] S.7(4), the Hadamard counterexample is non-generic.** The fixed point is real, but it is unstable. From $R_0$ equal to $I$ plus a random $10^{-6}$ skew perturbation, SEBA reached the indicator basis in **100/100** runs at $\mu=0.1$ and in **100/100** runs at $\mu=0.99/\sqrt N$. A much stronger and generic failure exists. With group sizes $(20,5,3,2)$ ($N=30$), $V=H\cdot\mathrm{Haar}$, $R_0=I$ and the **published default** $\mu=0.99/\sqrt N=0.181$ (which satisfies the S.7 condition $\mu<\min_j n_j^{-1/2}=0.224$), I got:

```
mu=0.181: success 43/200, converged 193/200, indicator objective 1.7812, failure objectives 1.814–1.963
   fail col 0 nonzeros per group [0, 5, 0, 0]
   fail col 1 nonzeros per group [0, 0, 0, 2]   (+0.707)
   fail col 2 nonzeros per group [0, 0, 0, 2]   (-0.707)
   fail col 3 nonzeros per group [0, 0, 3, 0]
mu=0.100 / 0.050 / 0.020: success 200/200
```

Failing runs converge to a **rank-deficient** output, with two columns $\pm$ the same small-group indicator, and the **largest cluster is lost**. Published Step 6 makes the two copies identical with minimum 0, so the reliability ordering does not flag them. The cause: large-group entries $|U_{ij}|/\sqrt{n_i}$ fall below $\mu$ unless the column is already aligned with that group. This is the default algorithm on exact, unequal-size data, so this is the counterexample to state. Pair it with the provable positive statement that follows: if $U=OR_0^\top$ satisfies $\max_{i\ne j}|U_{ij}|/\sqrt{n_i}<\mu<\min_j|U_{jj}|/\sqrt{n_j}$, then one step returns $HD$ exactly ($D$ = signs) and stays fixed. This is a one-step basin, provable in three lines from the S.7 proof, and it makes the dependence on $\mu$ explicit.

**F4 [GAP] S.6, "well-defined".** $\operatorname{Polar}(S^\top V)$ is unique only when $S^\top V$ is nonsingular, and F3 shows that rank loss happens even in the exact regime. A simpler instance: $N=400$, $z_{1,2}=(\tfrac12,\tfrac12,\pm s,\dots,\pm s)$ with $s=\sqrt{0.5/398}$, so $z_1\perp z_2$, and $\mu=0.04<1/\sqrt N$. Both columns threshold to the same vector. The proof survives with "for any selection of a Procrustes minimizer". Also state that the guarantee is for the idealized iteration: the published code perturbs constant columns *after* `qr`, which breaks orthonormality of $V$, and stops at 5000 iterations. FRS (2019) prove no convergence result, so S.6 should be stated as new.

**F5 [OVERCLAIM] §5, "a spectral perturbation bound plus a localization certificate supplies its error bar".** S.8 bounds the SEBA output against a *candidate* $H$. It says nothing about distance to a population concept. That would need population Davis–Kahan, a sampling bound and a correct candidate, none of which is provided. Likewise, $\eta$ is a **sound** certificate that a near-indicator basis lies in the block, but it is not complete: the candidate comes from a nonconvex search that can miss it (F3). Say this, and scope the sentence comparing the method to SAEs accordingly. The protection against an optimisation "instructed to produce" a dictionary comes from $\eta$ and the cut quantities, not from SEBA.

**F6 [OVERCLAIM] S.4, "What it buys".** Give the thresholds. Any witness at all requires $\sigma_1>1/2$ ($\kappa<1$). A nontrivial trimmed transport bound requires $\kappa<1/3$, i.e. $\sigma_1>17/18\approx0.944$. A positive centered score requires $\kappa<1/5$, i.e. $\sigma_1>0.98$. "High singular coherence has an actual discrete witness" is true only in that sense.

**F7 [OVERCLAIM/framing] Scope versus the non-negotiable.** On the owner's commitment the draft passes: sparsity enters only as a named hypothesis (S.7, S.8) or a diagnostic, and SEBA and varimax are labelled optional postprocessing. Two points need fixing. (a) The ℓ1 weight $\mu$ is a sparsity *preference* in the readout, and F3 shows the output depends on it qualitatively. Reports should carry the $\mu$-independent certificates ($\eta$, rank, cut quantities) and a $\mu$ sweep, and should say plainly that SEBA output does not end in an SVD and so is outside the construction. (b) Every recovery result covers **disjoint partitions** only. The regime that SAEs target, where a site is a superposition of several active concepts (THEORY §4.2), has no theorem here. The introduction promises "a small, discrete account" in general. State in the introduction that the overlapping/$k$-sparse regime is open, and keep the three meanings of "sparse" separate: graph sparsity (S.2), low-leakage sets (§2), and localized coordinates (§3).

**F8 [NIT] S.2.**

- Spielman–Srivastava Theorem 1 needs $1/\sqrt n<\varepsilon\le1$, $n$ large, and $q=9C^2n\log n/\varepsilon^2$ samples.
- BSS Theorem 1.1 is one-sided. Show the rescaling: multiplying by $2/(1+\kappa)$ gives $\varepsilon=2\sqrt d/(d+1)$.
- With the constant $9C^2$, there are no edge savings at moderate $N$. My $N=60$ run kept 930 of 1770 edges.
- "Sparse support may obstruct scaling" applies only if the diagonal is dropped. With positive diagonal, the matrix has total support, so a symmetric scaling exists (Csima–Datta).

**F9 [NIT]**

- S.8 should require $H$ to be the $N\times r$ normalized-indicator matrix of a partition, with $r=\dim V$.
- The `B.2x` and `A.2x` placeholders remain.
- The cited `/tmp/sparsity-code/check_sparsity.py` is ephemeral. Move it to `checks/` with assertions, or mark the claims as not reproducible.
- There are no `\{` rendering problems.

**F10 [NIT] Notation.** $v$ is both the Sinkhorn vector and $Q$'s right singular vectors. §2 uses both meanings, so disambiguate. Also say at its definition that $\sigma_1$ is the top nontrivial singular value ($\sigma_0=1$).

## Verdict

**Approve with fixes.** No statement is false as written. F1 and F3 are the substantive ones: one relies on an unverified extension of a cited theorem, and the other replaces a measure-zero counterexample with a generic failure of the default algorithm.

## Required fixes

1. S.3: give a verified citation or a proof of the higher-order Cheeger upper bound with holding weights (vertex measure 1 including loops), or restate it for the loopless graph with explicit conversion factors (F1).
2. §2: correct Froyland (2013). Drop "invariant-measure", list Assumptions 1–2, and cite Theorem 2, eq. (8) for the coherent-set bound (F2).
3. S.7(4): replace or supplement the Hadamard fixed point with the generic unequal-size failure at default $\mu$ (rank loss, largest cluster lost, not flagged by the reliability ordering). Add the one-step basin proposition and state the dependence on $\mu$ (F3).
4. S.6: make the polar step a selection when $S^\top V$ is singular, note that rank loss occurs even in the exact regime, and state that S.6 is new rather than taken from FRS (F4).
5. §5: restrict "error bar" to the candidate-relative bound, and state that $\eta$ is sound but not complete (F5).
6. S.4: state the $\sigma_1$ thresholds for nonvacuous conclusions (F6).
7. Introduction and §3: state that only the disjoint-partition regime is covered and the overlapping/$k$-sparse regime is open; separate the three meanings of "sparse"; require $\mu$-independent certificates and $\mu$ sweeps in reports; label SEBA output as outside the construction (F7).
8. S.2: add the SS and BSS hypotheses and the rescaling, the practical edge-count caveat, and the total-support correction (F8).
9. Resolve the `A.2x`/`B.2x` placeholders, move the check script into `checks/` or mark it unreproducible, tighten S.8's hypotheses on $H$, and clarify notation (F9, F10).

---

## Re-check of the revised draft (2026-11-07)

Scope: my nine required fixes, plus the new material. I ran `uv run --with numpy python -I checks/sparsity_check.py`. It ends with `all sparsity checks passed`, and its unequal-size SEBA counts (43/200 at $\mu=0.1807$; 200/200 at $0.1$, $0.05$, $0.02$) reproduce my own run exactly. Independent scripts are in `/tmp/sparsity-review/recheck.py`.

### New material

- **Loop-free Cheeger reduction (S.3): correct.**
  - $D-W=I-P$ when $d_x=1-P_{xx}$ (positive, since off-diagonals are positive and $N\ge2$).
  - Writing $\nu_i$ for the eigenvalues of $D^{-1/2}(I-P)D^{-1/2}$, the generalized Rayleigh quotient $f^\top(I-P)f/f^\top Df$ gives $\mu_i/d_{\max}\le\nu_i\le\mu_i/d_{\min}$ by min–max.
  - $\Phi_P(A)=(\mathrm{vol}_W(A)/|A|)\Phi_W(A)\le d_{\max}\Phi_W(A)$, and the cap $\Phi_P\le d_{\max}$ also holds. Together these give $h_k(P)\le Ck^2d_{\max}d_{\min}^{-1/2}\sqrt{1-\lambda_{k-1}}$.
  - LOT now applies to a genuinely loop-free weighted graph, so the citation gap (F1) is closed.
  - Numerical check: eigenvalue sandwich, $h_k(P)\le d_{\max}h_k(W)$ and $\nu_{k-1}/2\le h_k(W)$ held with 0 violations over 20 clustered instances ($N=8$; exhaustive search over set families for $k=2,3$).
  - The bound is honest but often vacuous. It beats the trivial cap only when $Ck^2\sqrt{(1-\lambda_{k-1})/d_{\min}}<1$, and the check script's own instance has $d_{\min}=5.7\times10^{-4}$. The draft's "can be loose" is accurate.
- **Nearby-start basin (S.7.1): correct.**
  - Entries of $HU$ are $U_{ij}/\sqrt{n_i}$. The condition zeroes off-group entries and keeps $\operatorname{sign}(U_{jj})$ on group $j$.
  - $S^\top V=D_\pm O$ is orthogonal, so the Procrustes update is unique.
  - Persistence follows because $\mu<|U_{jj}|/\sqrt{n_j}\le n_j^{-1/2}$.
  - 1,711 random rotations satisfying the condition (random sizes 1–14) gave 0 failures after one and three steps.
- **Generic default-μ failure: incorporated accurately.** The draft gives the right numbers and mechanism, and correctly notes that the sign-corrected minima are all zero, so the reliability ranking does not flag the failure. It states the result as a reproducible empirical outcome rather than a probability claim, which is right. The Hadamard fixed point is kept, with its instability now acknowledged.
- **Collapsed-column Procrustes: correct.** Every full SVD $S^\top V=U_Z\Sigma W_Z^\top$ gives a maximizer $U_ZW_Z^\top$ of $\operatorname{tr}(R^\top S^\top V)$, including when $S^\top V$ is singular. S.6 now holds "for any selection", and the alternating-descent argument is unaffected.

### Status of the nine required fixes

1. Higher-order Cheeger with loops: **done** (the reduction above).
2. Froyland (2013): **done.** Assumptions 1–2 and Theorem 2, eq. (8) are now cited, and "invariant measure" is gone. The choice $k(x,y)=NQ_{yx}$ makes $L=Q$ and $L^*=Q^\top$ under uniform measures, so $L1=L^*1=1$.
3. SEBA counterexample, basin and μ dependence: **done.**
4. S.6 selection, rank loss and attribution: **done.**
5. Candidate-relative error bar; $\eta$ sound but not complete: **done** (§5).
6. S.4 thresholds: **done.** I checked $17/18$ and $49/50$.
7. Disjoint-partition scope, three meanings of "sparse", μ sweep, outside-construction label: **done.**
8. S.2 hypotheses: **done.** I checked the BSS rescaling ($2\sqrt b/(b+1)\le\varepsilon$ for $b\ge4/\varepsilon^2$), the full-graph fallback for $\varepsilon\le1/\sqrt N$, the edge-count caveat, and the total-support remark.
9. **Partly done.** The check script is now in `checks/` and passes, the S.8 hypotheses on $H$ are tightened, and the notation is clarified. The `A.2x`/`B.2x` placeholders remain; they are labelled integration cross-references, so the integrator can resolve them.

### Remaining fixes (none blocking)

1. Resolve `A.2x` and `B.2x` (twice) when the drafts are integrated.
2. S.2 total support: Sinkhorn–Knopp (1967) gives a two-sided scaling $D_1KD_2$, but THEORY needs the symmetric scaling $DKD$. Add the one-line step: a positive diagonal plus a connected support makes the matrix fully indecomposable, so $D_1,D_2$ are unique up to a scalar and symmetry forces $D_2\propto D_1$. Apply this per component, or cite Csima–Datta (1972).
3. S.7.1: state that the basin holds up to a column permutation, i.e. with $U_{i\pi(j)}$. This is immediate.
4. S.6: "indefinite alternating core" should read "unbounded (non-terminating) iteration".
5. S.3, two-way bound: the holding-chain form is cited to Chung. Adding a reversible-chain reference for the bottleneck-ratio inequality, such as Levin–Peres, *Markov Chains and Mixing Times*, or Jerrum–Sinclair/Lawler–Sokal, would make it explicit that the holding probabilities count in the vertex mass.

### Verdict on the revision

**Approve.** All the mathematical findings are resolved, the new material is correct, and the remaining items are editorial or for integration.
