# Review of B2: ideal dictionaries and sparse concept clusters

Reviewer: GPT-6.1 Sol, 2026-11-07. Reviewed B.20–B.34 against `theorems/BRIEF.md` and `THEORY.md`. Independent numerical tests accompanied a line-by-line proof check. No draft or construction was changed.

## Finding

### 1. [GAP] B.26 must identify the loading view in its comparison sentence

The $k$-means theorem is correct. However, line 245 says its atoms equal “the construction's size-normalized loadings,” without requiring that those loadings use **the same data matrix** $F=[f_x^\top]_x$ as the reconstruction objective. B.25 allows arbitrary loading views $H$, and raw activation loadings are explicitly supported elsewhere. The comparison is only justified with $H=F$.

A concrete exact-ray example has raw activation rows $e_1,3e_1,2e_2,4e_2$ and reconstruction data $f_x=\hat\phi(x)$. Here $N=4$, $m=2$, $\varepsilon=0$, $\Delta=1$, $R=1$. At $\beta=4$, B.21 holds. Numerical output:

```text
2 exp(-beta Delta) R = 0.03663127777746836
one-hot optimal atoms (columns) = [[1, 0], [0, 1]]
raw size-normalized spectral loadings = [[2, 0], [0, 3]]
Frobenius difference = 2.2360679774997894
```

Thus the comparison, read as including raw loadings, is false even with perfect subspace recovery and localization. This is a missing view specification, not a failure of the centroid or spectral proofs. Write the comparison explicitly as $a_F(U_j)/\sqrt{n_j}=\bar f_j$, with B.25's approximate bound likewise using $F$. Qualify “the same finite-sample dictionary” in “What it buys” accordingly. B.28 already correctly distinguishes **directions** from amplitudes.

## Adversarial numerical results

Executed independently using NumPy through `uv run --with numpy python -I`, single-threaded BLAS, seeds 20261107 and 1097. Scripts: `/tmp/cm_b2_review_sol_check.py` and `/tmp/cm_b2_review_sol_extra.py`. These are review-time scratch artifacts, not permanent repository checks. Sinkhorn maximum row residual was below $2\times10^{-14}$. Inequalities were asserted with floating-point tolerances.

**B.21:** 16 configurations, including extremely close rays and unequal sizes. At the deliberately marginal value $2e^{-\beta\Delta}R=0.999999$:

| Cluster sizes | $\Delta$ | Observed gap | Guaranteed gap |
|---|---:|---:|---:|
| (1,99) | 0.000199993333 | 0.606257714 | 0.000001 |
| (1,2,97) | 0.0199334222 | 0.703446210 | 0.000001 |
| (1,3,9,51) | 0.00000449999662 | 0.844896977 | 0.000001 |

The zero tail and exact indicator range were also checked. No counterexample: the factor $2$ follows correctly from the absolute row sums of $P-\Pi$, and symmetry makes that a spectral-norm bound. Unequal cluster frequencies are accounted for by $R$.

**B.22:** 166 admissible configurations, including 150 randomized unequal clouds and circular clouds concentrated at facing edges with rare extreme sites. The following cases sit immediately inside the theorem's threshold, $2\eta/c=0.999999$:

| Sizes | $\beta$ | Observed gap / lower bound | Observed angle / upper bound |
|---|---:|---:|---:|
| (1,2,97) | 183.728654 | 0.934892286 / 0.000000659 | 0.007915383 / 0.999998 |
| (2,98) | 1302.727260 | 0.867488059 / 0.000000724 | 0.025837065 / 0.999998 |
| (3,5,67) | 65.839551 | 0.922009566 / 0.000000640 | 0.011073078 / 0.999998 |

All intermediate leakage and entrywise mixing bounds were checked too. Near the **geometric** threshold, a separate cloud with $\varepsilon/\Delta=0.984456709$ had no admissible B.22 window in the scan; its conclusions were correctly not invoked. The constants are very conservative, not numerically contradicted.

**B.26:** Exhaustive partition searches on unequal seven-site examples also passed with $(N-m)\varepsilon/\Delta=0.999999$. Output:

```text
sizes       planted cost    minimum wrong cost
(1,2,4)     0.0105308061    0.0486185830
(1,1,5)     0.00224757694   0.00775332996
(3,4)       0.0250256759    0.149049401
```

## Results of the remaining proof checks

- **B.20–B.22:** Hypotheses and indexing are consistent with counting measure and inclusion of the constant. B.22's bounds on $S_j$ and individual balancing entries give the displayed $\ell$ and $c$. The retained principal blocks are PSD; adding leakage to the diagonal preserves PSD. The Sylvester integral has the correct sign and separation $c-\eta$. The bounded-$\beta\varepsilon$ limit is correctly distinguished from fixed positive $\varepsilon$ at infinite temperature.
- **B.23:** The exact global-maximizer claim is **true even for unequal sizes**. Each row contributes a positive weight $1/n_i$ times $\sum_jO_{ij}^4\le1$. Since all weights are positive, equality forces every row to be one-sparse; orthogonality forces a signed permutation. There is no compensating gain available in a larger cluster. Tested 6,000 random rotations, including sizes (1,1,1,1000), plus small plane rotations. The noisy proof's Lipschitz, deficit, distinct-column and rounding steps are valid. A 50,001-angle noisy $O(2)$ search also respected its bound. Practical global optimization is appropriately disclaimed.
- **B.24–B.25:** Correct. Distinct partition projectors differ in an entry by at least $1/N$; the polar alignment formula has the stated error. Exhaustive seven-site certification passed at $s=0.071296602<1/14$, with frame error $0.071342005$ versus bound $0.174640295$. Loading-error transport is valid.
- **B.26–B.28:** Apart from finding 1, correct. A wrong hard-clustering solution must contain a mixed group costing at least $\Delta$, even if it uses fewer than $m$ groups. B.27 correctly separates line fitting from centroid fitting. B.28's exact-width, unit-norm, observed-distinct-line assumptions prove the stated decoder identifiability; nonnegative codes fix orientation. It does not identify an encoder or cover an additional sparsity penalty, bias or overcomplete dictionary.
- **B.29–B.30:** Correct under the named assumptions. The maximum overlap of distinct equal-coefficient supports is bounded by $\sqrt{1-1/k}$. Normalization error gives $8\rho^2$ within supports and $4\rho$ cross-support loss. Enumerated support families for $(m,k)=(4,1),(5,2),(6,3),(6,4)$ passed. The nonorthogonal extension explicitly assumes the needed relative error and prototype separation. Positivity, near-equal coefficients, support-count rather than feature-count recovery, and missing-singleton limitations are honestly stated.
- **B.31–B.32:** Correct, including the distinct-within-cluster diagnostic counterexample. Tested 464 random projectors across all ranks up to $N=30$, and unequal/balanced cluster formulas. The sharp upper bound is $1/r$, its coordinate-projector equality characterization is valid, and the centered formula correctly uses rank $m-1$. Balanced clusters indeed attain the lower endpoint. The $2s/r$ perturbation bound is valid.
- **B.33–B.34:** Correct. Twenty-four full-spectrum hierarchy tests and 150 same-space perturbation tests passed. Diffusion's integer-time condition is explicitly necessary. The occupancy constant follows from the displayed Chernoff calculation. No universal Sinkhorn sampling rate is asserted: Part A remains a conditional interface and needs its final cross-reference during integration.

The standard linear-algebra steps are proved sufficiently explicitly; no unverified external theorem is carrying these conclusions. Costs and global-optimization caveats are appropriately scoped. No GitHub brace-rendering problem was found.

## Verdict: approve with fixes

### Required fixes

1. In B.26's loading-comparison sentence and “What it buys,” explicitly require loading view $F$ equal to the baseline reconstruction data. State the equality and transported error using $a_F$, not unspecified construction loadings. No change to the gap, localization, SAE-identifiability or IPR bounds is required.
