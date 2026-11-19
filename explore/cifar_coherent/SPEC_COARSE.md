# Delta spec: spectral coarse-graining certificates (exploratory)

Owner request (2026-11-19): apply the "spectral coarse-graining = approximate causal states + approximate
autonomy" argument (owner's note, summarised in section 1) to the ResNet-20 / CIFAR-10 explorer; keep at least
10 eigenfunctions in the viewer, ideally more, to look for a spectral gap. Owner's answers: analyse **layer-chain
products** and **single cells**; store **32** nontrivial functions per block; results go in an **explorer panel
plus a short site section**. Kernel: cosine (`r20s0b`) now; the dot kernel (`r20s0d`, another session) can reuse
the code later. Exploratory side study like the rest of `explore/`; labels only validate.

Everything not mentioned is unchanged from `SPEC.md` / `README.md` (schema v2, probe N = 2,000, 8 checkpoints,
12 sites, five rules, zero-diagonal symmetric scaling P, fixed-lazy M = (I + P)/2, float64 spectra).

## 1. The statements we use, in our notation

Every cell operator here is symmetric doubly stochastic on the N probe images (ordered products are doubly
stochastic but generally not symmetric), so the uniform measure is stationary on
both ends. A block's coupling is pi(x, y) = Q[y, x] / N, where

- single cell: Q = M_{t,s} (X = Y = image at that cell; reversible case);
- layer chain a -> b at checkpoint t: Q = M_b ... M_{a+1} M_a (X = image at site a, Y = image at site b; the network
  map is the identity on image indices, so this is the dynamic-Laplacian / coherent-pair transfer operator with
  the cosine kernel as the "noise" on each end). Q is again doubly stochastic, so it is a genuine coupling.

With mu = nu = uniform, the SVD of the post is the plain SVD of Q: sigma_1 = 1 with constant singular vectors,
u_i = sqrt(N) * (left singular vector), v_i likewise, and the spectral embedding of the start side is
Psi_X(x) = (sigma_2 v_2(x), sigma_3 v_3(x), ...). (For a single cell sigma_i = lambda_i(M) >= 0.)

For a hard partition c of X into blocks A_1..A_k and (optionally) c' of Y into B_1..B_l, with
S_{jy} = sum_{x in A_j} Q[y, x] and T_{jl} = sum_{x in A_j, y in B_l} Q[y, x] (exact, O(N^2)):

- total: I(X;Y) = ||Q||_F^2 - 1 = sum_{i>=2} sigma_i^2;
- one-sided (predictive): I(c(X); Y) = sum_{j,y} S_{jy}^2 / |A_j| - 1;
- two-sided (coherent pairs): I(c(X); c'(Y)) = sum_{j,l} T_{jl}^2 / (|A_j| |B_l|) - 1.

(I = chi^2 mutual information.) Identities and bounds checked in `test_coarse.py`:

- (T1, exact) I(X;Y) - I(c(X);Y) = (1/N) sum_x ||Psi_X(x) - mean of Psi_X over x's block||^2 using all N - 1
  nontrivial modes.
- (T2, floor) I(c(X);Y) <= F_k := sum_{i=2}^{k} sigma_i^2 (Eckart-Young; the coarse operator has rank <= k
  including the constant). Same floor for the two-sided quantity with k' = min(k, l).
- Names: F_k is the **retention ceiling** for k blocks, I(X;Y) - F_k the **loss floor**. Spectral k-means on the
  truncated embedding is a candidate partition, not a guarantee of reaching the floor.
- **Headline numbers** (revised after review): the absolute retained information I(c(X);Y) and its excess over the
  random-partition baseline at the same k, shown together with F_k. Efficiency eta_k(c) = I(c(X);Y) / F_k is a
  *tightness diagnostic* only: it is 1 for every partition when M is the lazy uniform walk (no geometry at all), so
  high eta alone says nothing. That uniform control (P = (J - I)/(N - 1)) is a test case and a reference row.
  Totals: a single cell has I(X;Y) >= N/4 - 1 because of the lazy half, so F_32 / I is at most about 6% by
  construction; long chains can have F_k of order 1e-5. Chain summaries are therefore grouped by chain length and
  rule, and absolute values are always shown next to ratios.
- Single cells are also scored on P = 2M - I (no laziness): its singular values are |lambda_i(P)|, ordered by
  magnitude, and the same partitions are evaluated with the same formulas (P is symmetric doubly stochastic but can
  have negative eigenvalues; a near -1 value is reported, not treated as slow).
- Reversible case (single cells): leakage phi(A) = 1 - (1/|A|) sum_{x,y in A} M[x,y]; Ncut(c) = sum_j phi(A_j);
  (T3, Ky Fan) Ncut >= sum_{i=1}^{k} (1 - lambda_i); (T4, higher-order Cheeger, lower side) max_j phi(A_j) >=
  (1 - lambda_k)/2. Reported as ratios Ncut / bound, max phi / bound. Also the stationary-chain identity
  I(c(X_0); c(X_1)) for the same partition on both ends. Because M = (I+P)/2, every leakage is exactly half the
  leakage under P, and 1 - lambda_i(M) = (1 - lambda_i(P))/2: ratios are laziness-invariant.
- Pair agreement (chains): match end blocks B to start blocks A by maximising matched mass (Hungarian). Reported as
  **matched agreement** sum_j T[B_j, A_j] / N, beside its independence expectation sum_j mu(A_j) nu(B_j) for the same
  matching, the endpoint masses, and per-pair forward and reverse rates T/|A_j|, T/|B_j|. This is an agreement
  score, not a coherence certificate; the certificate is the two-sided chi^2 information against its ceiling.
  Tested with unequal-mass independent partitions.

## 2. Partitions evaluated (for every block and every k in K = {2, 3, ..., 16, 20, 24, 32})

Method by endpoint (chains: start partition c on X, end partition c' on Y; single cells: X = Y and c' = c):

| method | start c | end c' |
|---|---|---|
| spectral k-means | Lloyd on Psi_X top (k-1) coords | Lloyd on Psi_Y top (k-1) coords |
| SEBA (r = k) | SEBA on the first r start vectors (constant included), hard argmax | same on the first r end vectors |
| PCA k-means (baseline) | Lloyd on top-50 PCs at site a | Lloyd on top-50 PCs at site b |
| random (null) | balanced random, 4 draws | independent balanced draws |
| labels (validation) | CIFAR class | same labels |
| predictions (validation) | predicted class at checkpoint | same |

Scores for every row: I(c(X);Y), I(X;c'(Y)), I(c(X);c'(Y)), each with its ceiling F_{k_eff}, where k_eff is the
**actual number of nonempty groups** (e.g. step-0 predictions occupy four classes). Single cells add Ncut, max
leakage and the I(c(X_0); c(X_1)) identity.

- Spectral k-means: uniform weights, k-means++ init, 8 restarts run as one batch, fixed seed, best within-cluster
  sum kept; scored by the full-operator formulas, not in the truncated space.
- SEBA: ranks r in {2, 3, 4, 6, 8, 10, 12, 16} with the existing port, on the factorization computed here; hard
  assignment = argmax over the r SEBA vectors *ignoring* the threshold (every point assigned; ties to the lower
  index; empty groups dropped and k_eff reduced). These analysis assignments are separate from the explorer's
  existing SEBA labels (which keep "unassigned").
- PCA baseline: from the saved cosine Grams (`/work/r20s0b/gram_<step>_<site>.npy`), re-centred after row
  normalisation, top 50 eigenvectors scaled by sqrt eigenvalue; k-means labels cached per (checkpoint, site, k).
- Labels and predictions never select k, modes or rules; they are validation rows.

**Gap report** (descriptive, never used downstream): for k in [2, 32] the largest sigma_k - sigma_{k+1} and largest
sigma_k / sigma_{k+1}, each with its size, a "no clear gap" verdict when the largest ratio is < 1.15 or the top two ratios are within 10% of each other,
flags for numerical ties (|sigma_k - sigma_{k+1}| < 1e-10) and for sigma_{k+1} < 1e-8 (ratio meaningless), and the
participation ratio of each of the 32 vectors (so a localised mode is not read as a group).

## 3. Scope and storage

- Single cells: 5 rules x 8 checkpoints x 12 sites = 480. Layer chains: 5 x 8 x 66 ordered pairs a < b = 2,640.
  Chains run start-to-end in network order only (reversal transposes Q; the two sides swap).
- Per block: top 129 singular values (spectrum plots, gap search) and I(X;Y); for each partition kind and k the
  quantities above. Stored as `coarse_cells.npz` and `coarse_chains_<step>.npz` under `data/r20s0b/`, float32,
  plus a `coarse_summary.json` with headline aggregates. Partitions themselves are not stored except spectral
  k-means labels for single cells and chains at k in {2..16} (int8; the explorer colours points by them).
- **32 stored functions, as an add-on payload** (revised after review). The existing per-block files and the
  site's `blk/` payload are not rewritten (the dot-kernel viewer requires equal `n_vecs` across roots, and a full
  4x rewrite would add about 0.8 GB before gzip). Instead the coarse stage writes, from the same factorization it
  analyses, an extension `ext_<family>_<step>.npz` with eigen/singular values 1..129 and vectors 9..32 (int16 +
  per-vector scale, same quantisation as now), for: single cells (all rules), layer averages (all rules), layer
  products start and end side (all rules). Vectors 1..8 are checked against the stored files (|cos| > 0.99 for
  every mode whose eigengap to its neighbours exceeds 1e-6, sign aligned to the stored one), else the block is
  flagged. The site build ships the extension as separate lazily-loaded files `ext/<fam>/<idx>/<mode>/<pair>.bin.gz`,
  loaded only when a mode above 8 is selected; which families and rules ship is decided from the measured
  compressed size (target: extension under 300 MB; if over, ship all rules for single cells and the default rule
  for chains/averages, and say so). Time-direction blocks keep 8 modes in this pass.

## 4. Compute

Modal, detached coordinator, new app file `coarse_modal.py` (does not import `measure_v2.py`), reading operators on
the volume (`/work/r20s0b/operators_<step>_<site>.npy`). One unit per (checkpoint, rule): one float64 eigh per cell,
one per layer average, and one float64 SVD per layer chain (built incrementally a -> a+1 -> ..., as now), each
factorization used for the spectrum, both endpoint embeddings, SEBA, and the 32-mode extension. **Pilot first:**
one checkpoint (64,000), one rule, all chains, timed, with the stored-vector agreement check; fan out only if the
projected total is under $10 (ceiling $15).

## 5. Explorer and site

- Explorer: the eigenfunction selector goes to 32; a spectrum strip for the selected block (top 64 sigma, log
  scale, gap markers); a **coarse-graining** panel: retained information and F_k vs k (absolute, log scale) for spectral k-means / SEBA / PCA
  k-means / random, eta_k as a secondary view, labels and predictions as validation markers at their k_eff; for single cells also Ncut/bound
  and max-leakage/bound; a colour-by "spectral k-means, k = ..." option.
- Site: a short section after the method ("what a k-block coarse-graining can keep"): the coupling, the exact
  loss identity (T1), the floor (T2) and Ky Fan (T3) stated as theorems about this operator, the measured
  efficiency, and the gap statistics. No new vocabulary beyond "block", "partition", "coupling".

## 6. Markov closure across layers (added 2026-11-19, owner request)

**Process.** At checkpoint t and rule m, Z_{-1} is uniform on the N probe images and Z_s is drawn from
M_s(. | Z_{s-1}) for s = 0..11, so the pair (Z_s, Z_{s+1}) has coupling M_{s+1} / N and (Z_a, Z_b) has
Q = M_b ... M_{a+1}. Each site s carries its own partition c_s; the **coarse process** is C_s = c_s(Z_s).
Z is Markov by construction. The question is whether C is (lumpability / closure). Because M = (I+P)/2, an
image stays put with probability 1/2 at each step, so image identity carries memory across sites unless the
partitions at different sites are aligned. That is part of what is measured, and it is reported, not corrected.

**Partitions per site.** (i) spectral k-means of that single cell (the stored `lab_start` of block (s, s),
k = 2..16); (ii) PCA k-means per site (baseline); (iii) a **shared** partition: the spectral partition of the b9 cell
(prespecified) used at every site, which separates "the partition changes" from "the walk changes"; (iv) independent
random balanced partitions per site (a reference with no alignment across sites, 4 draws); (v) classes, the same
partition at every site (validation only, k = 10). Reported with every sequence: the normalised mutual information
of neighbouring sites' partitions on the same image IDs (how aligned the sequence is).

**Matched geometry-free control** (revised after review). Every partition sequence is also run on the lazy uniform
walk U = (I + (J - I)/(N - 1))/2 at every site. That walk has no geometry, but image identity still carries memory, so
misaligned partitions produce coarse memory and CK error there too. The real operators are read *against this
control*, not against zero.

**Measures.** All exact and computed from the operators; Shannon in nats, with chi^2 alongside where it is defined.

1. **One-step strong-lumpability defect** delta_s = I(Z_s; C_{s+1} | C_s) = I(Z_s; C_{s+1}) - I(C_s; C_{s+1})
   (C_s is a function of Z_s). It is zero iff P(C_{s+1} | Z_s = x) is constant on each block of c_s. Reported raw
   and as the fraction delta_s / I(Z_s; C_{s+1}) (unavailable when the denominator is < 1e-12). The chi^2 version
   is the within-block variance identity with the end side coarse-grained.
2. **Coarse two-step memory** I(C_{s+2}; C_s | C_{s+1}) (Shannon), raw and divided by H(C_{s+2} | C_{s+1}): the
   fraction of the uncertainty left by the current block that the previous block resolves. Positive values falsify
   coarse Markovity. Zero adjacent-triple CMI and zero CK errors do **not** certify it (longer histories are not
   tested). The fine version I(C_{s+2}; Z_s | C_{s+1}) is a *diagnostic* of information hidden from the coarse state.
   It can be positive when the coarse process is exactly Markov (the image-retention path), so it never rejects
   closure.
3. **Chapman-Kolmogorov test** (the standard Markov-state-model check). For every a < b - 1, the true coarse
   transition T_true(a -> b) = P(C_b | C_a) against the composition T_{b-1 -> b} ... T_{a -> a+1} of the one-step
   coarse transitions. Error = sum_i P(C_a = i) * TV(T_true[:, i], T_ck[:, i]), reported against b - a beside the TV
   distance of T_true from the independent prediction P(C_b), so a small error is not mistaken for closure when
   everything has mixed. One-step pairs are zero by construction and are left out of summaries.

**Tests** (`test_markov.py`): brute-force agreement of every measure on a small chain; an exactly lumpable chain
gives zero defect, coarse CMI and CK error; a non-lumpable planted chain gives positive values; singleton blocks give
zero; the reviewer's counterexample (independent attributes A, B, D read at successive sites of the lazy uniform walk:
coarse CMI 0, fine CMI > 0) and its A, B, A variant (coarse CMI > 0, CK error > 0 with no geometry at all).

**Compute.** One Modal unit per (checkpoint, rule) reading the operators and the stored `lab_start`; PCA labels
recomputed with fixed seeds; the uniform control U is a dense N x N matrix like the real operators. Output `markov_<step>_<rule>.npz` (arrays over method x k x site / site pair), and
`markov_summary.json` plus facts for the site. Estimated < $2.

## 7. Not in this pass

Time-direction products, the dot kernel, and Shannon versions of the section 1 quantities (only chi^2 there; the
note's caveat about phi log(1/phi) applies to any comparison).
