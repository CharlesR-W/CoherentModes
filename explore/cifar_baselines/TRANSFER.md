# Transfer-operator variant: design note (builder C, 2026-11-15)

Status: design + pilot, PLAN.md section 6. This note describes one more variant next to the current construction
(the average of per-layer diffusion operators). It characterises, it does not rank: nothing below says that a variant
works or fails. Code: `transfer.py` (estimators), `test_transfer.py` (asserted checks), `transfer_pilot.py`
(runner), `transfer_modal.py` (Modal CPU fan-out). Outputs: `out/transfer/`. Status tags follow PLAN section 0:
every estimator here is **S** (label-free dot-product spectral object ending in an SVD); ridge terms are **B**-type
closed-form **fits** and are called fits; SEBA sets are **P**; planted truth is used only to score (**V**).

## 1. Estimand

Each probe image i gives one paired observation (x_i, y_i) = (z_{a,i}, z_{b,i}) of two views (two cells: two sites
at one checkpoint, or one site at two checkpoints). Sampling images induces a joint law mu_XY with marginals mu_X,
mu_Y (empirical, uniform over the images used; never rebalanced). The estimand is the conditional-expectation operator

    (T f)(y) = E[f(X) | Y = y],   T : L2(mu_X) -> L2(mu_Y),   (T* g)(x) = E[g(Y) | X = x].

Acting on density ratios h = d nu / d mu_X, T gives the ratio of the pushed-forward measure to mu_Y, so it is the
source-to-destination transfer operator with respect to the empirical marginals. T 1 = 1 and ||T|| = 1. Its
singular pairs (f_j, g_j, sigma_j) are the maximally correlated source and destination observables: sigma_j =
max corr(f(X), g(Y)) over f, g orthogonal (in L2 of the marginals) to the constant and to the earlier pairs.

Two facts fix what can be estimated (PLAN section 6, IDEAS 6.1):

- Each image is one deterministic pair. On the N image indices the operator is a permutation, and any estimator
  whose function space has dimension >= the number of fit images reproduces it: all singular values equal 1 in
  sample, under any pairing, with chance correlations held out (asserted: `test_memorisation_pathology`). Content
  therefore comes only from a **declared finite function space or smoothing, plus held-out validation**. Every
  number is "at this resolution".
- Normalised views need not be sufficient Markov states, so the composition of fitted operators need not equal the
  fitted endpoint operator (section 5).

## 2. Conventions

- **Orientation.** Every operator matrix `.T` is destination-by-source and acts on column vectors of source
  coefficients, as Q = M_b M_a does. Right singular vectors are source functions, left singular vectors are
  destination functions, forward composition a -> b -> c is `T_bc @ T_ab`. The joint table B and the cross-moment S
  are stored destination x source. Each model names its source side (`model.source_side`). Asserted in
  `test_orientation_and_shapes` (shapes with m_src != m_dst; on P8 the source functions span the doubled harmonic
  cos 2 theta, sin 2 theta and not cos theta, sin theta; destination functions span cos phi, sin phi) and
  `test_forward_composition_orientation` (a chain fitted in the wrong order is rejected).
- **Geometry.** Inputs are the raw uncentred Grams R_a, R_b (N x N). With w uniform on the fit images,
  G = (I - 1 w^T) R (I - w 1^T) and cos(i, j) = G_ij / sqrt(G_ii G_jj) (norm floor 1e-12, clip to [-1, 1]). A
  held-out image uses its own norm after the fit mean is removed. This cosine is the only within-view geometry.
  `transfer.View` computes cosine blocks on demand (an N = 10,000 R is never copied; it can be memory-mapped);
  `View.from_X` is the same object from features, for synthetic benchmarks. Asserted equal to builder A's
  `stats.raw_to_views` and to a direct feature-space computation (`test_view_from_R_matches_stats_and_features`).
- **Splits.** fit / tune / test are disjoint. Every feature map is built from fit images only (landmarks, PCA axes,
  graphs, Sinkhorn factors) and evaluated on any image. Tuning uses only the tune split; the test split is touched
  once per fitted model.
- **Measure.** Fit moments use w_i = 1/n_fit. Held-out statistics use the evaluation split's own mean (constant
  removed in that split), which is deeptime's convention (section 4.3).

## 3. The three estimators

### 3.1 Diffuse-transport-diffuse on image indices (exists; reference)

Q = M_b M_a with M = (I + P)/2 and P the rule-0 operator of each cell (union 15-NN 0/1 graph, floor 1e-6, zero
diagonal, symmetric Sinkhorn), on all N images with views centred on all N (as the explorer). Uniform mass on
images; identity transport on image indices. The right singular vectors of Q (source, view a) and left singular
vectors (destination, view b), constant removed, are compared with the transfer functions (section 6); the two-cell
average Mbar = (M_a + M_b)/2 likewise. `transfer.diffusion_reference`.

### 3.2 Soft-Ulam joint table

Landmarks: m random fit images per view, drawn independently per view (salt = the view's position in the bank),
as nested prefixes of one random order, so reducing m keeps a subset. Memberships (partition of unity, within-view
cosines only):

    phi_j(x) = exp(kappa_X (x.l_j - 1)) / sum_h exp(kappa_X (x.l_h - 1)),  likewise psi_k(y) with kappa_Y.

Bandwidth: kappa_X = c * kappa_ref,X with kappa_ref,X = 1 / median over landmarks of the cosine distance to the
nearest other landmark (label-free, scale-free); c in C_GRID = {1/64, 1/16, 1/4, 1, 4}, the same c for both views.

    B = sum_i w_i psi(y_i) phi(x_i)^T   (m_Y x m_X, destination x source),  q = B 1 (rows), p = B^T 1 (columns),
    T = D_q^-1/2 B D_p^-1/2,  SVD T = U Sigma V^T.

B is a joint probability table of two randomised coarse observations J | x ~ phi(x), K | y ~ psi(y) (drawn
independently given the image). Since T sqrt(p) = D_q^-1/2 B 1 = sqrt(q) and T^T sqrt(q) = sqrt(p), and
||T|| <= 1 by Cauchy-Schwarz, (sqrt q, sqrt p) is a singular pair with sigma = 1 (the trivial pair; kept in
`.trivial`, asserted to 1e-10 in `test_soft_ulam_trivial_pair`). Non-trivial coefficients a_j = D_p^-1/2 v_j,
b_j = D_q^-1/2 u_j; image functions f_j(x) = phi(x)^T a_j, g_j(y) = psi(y)^T b_j.

**What sigma_j is.** sigma_j = corr(a_j(J), b_j(K)) for the coarse observations. Because J and K are conditionally
independent given the image, Cov(f_j(X), g_j(Y)) = Cov(a_j(J), b_j(K)) = sigma_j, while Var f_j(X) =
Var E[a_j(J) | X] <= Var a_j(J) = 1. Hence the population correlation of the soft-evaluated image functions is

    corr(f_j(X), g_j(Y)) = sigma_j / (sd f_j * sd g_j) >= sigma_j.

This is an identity between moments under one and the same table and distribution (the fitted table and the
population it is evaluated in). It does not make a training sigma_j a lower bound for a held-out correlation (the
fitted functions are estimated; held-out moments differ). Equality holds when the relevant category functions have
zero conditional variance, Var(a_j(J) | X) = 0 and Var(b_j(K) | Y) = 0 almost surely, which does not require
globally hard memberships. Soft-Ulam training singular values are therefore not on the same scale as held-out
correlations of its image functions (observed on P8: training 0.86, held-out 0.92 for the first function at
N_fit = 1000). The pilot reports both and never compares Ulam's sigma_j with another estimator's sigma_j.

**Occupancy gate (PLAN).** A table is used only if every landmark of both views has effective occupancy >= 10 fit
images. The criterion used is the **soft count** n p_j = sum_i phi_j(x_i). The Kish effective sample size
(sum phi)^2 / sum phi^2 is a different criterion (it is never smaller than the soft count when all memberships are
<= 1, so it is the weaker one); it is reported alongside, never used for the gate. If the gate fails, m is halved
(nested prefix) until it holds, per bandwidth, before tuning; the m used is recorded. This pilot drops no landmark.
(Dropping landmarks by a prespecified rule and renormalising is a legitimate alternative, not a vacuous one; it
changes the coarse observation and was not chosen here.) Since the soft counts sum to n_fit, the gate implies
m <= n_fit / 10: at N_fit = 1000 at most 100 landmarks, in practice 32 (P8).

### 3.3 Whitened cross-covariance SVD (regularised CCA; a fit)

Finite feature spaces F_X (n x m_X), F_Y, centred with fit means mu_X, mu_Y:

- `linear`: cosine PCA scores. Kernel PCA of the linear cosine kernel on the fit images, centred on the fit
  images; top p = m eigenpairs with eigenvalue > 1e-10 of the largest (so at most the rank of the view);
  held-out images by the centred cross-kernel (z_i - zbar)^T (z_j - zbar), times U Lambda^-1/2.
- `nystrom`: Nystrom features of the PSD kernel k(x, x') = exp(kappa (x.x' - 1)) (PSD by the Schur product
  theorem), m random fit landmarks: F = K(., L) U Lambda^-1/2 with eigenvalues of K_LL below 1e-10 of the largest
  dropped. kappa = c * kappa_ref as in 3.2.
- `diffusion`: diffusion coordinates. psi_1..psi_m = leading non-constant eigenvectors of the fit images' own
  rule-0 operator P (built from fit-only cosines); features F(x) = sum_j P(x, j) psi(j). On fit images this is
  lambda_P psi exactly. A held-out image's row is the declared extension rule: x is joined to its 15 nearest fit
  images and to every fit image j with d(x, j) < d_15(j) (reverse-neighbour rule: x would enter j's 15-NN), floor
  1e-6 elsewhere, weights multiplied by the training Sinkhorn factors exp(u_j), row renormalised to 1
  (`test_diffusion_extension_rows`). The graph has no bandwidth; k = 15 is fixed (the default construction).

With C_XX = F_X^T F_X / n etc. and ridge rho_X = eta tr(C_XX) / m_X (eta in ETA_GRID = {1e-4, 1e-3, 1e-2, 1e-1, 1},
same eta both sides; IDEAS 6.3):

    T = (C_YY + rho_Y I)^-1/2 C_YX (C_XX + rho_X I)^-1/2   (destination x source),  SVD T = U Sigma V^T,
    source coefficients (C_XX + rho_X I)^-1/2 V,  destination coefficients (C_YY + rho_Y I)^-1/2 U.

The singular values are regularised objective values, not held-out correlations. Inverse square roots are
symmetric (basis-canonical), so the intermediate view of a chain has identical coordinates in both fits.

## 4. Evaluation, tuning and controls

### 4.1 Held-out correlations

Pearson correlation of f_j and g_j on the test images (test means and variances), and the k x k cross-correlation
matrix (k = 6). Fitted functions with little regularisation are heavy-tailed: a few images can carry a function,
so a single held-out Pearson value can be large under independence (seen in the memorisation fixture: |r| up to
0.6 at n_test = 300 under a within-split permutation). This is why support is judged against the permutation
control, never against a Gaussian formula.

### 4.2 VAMP-2 (PLAN definition)

On an evaluation split E with the fitted top-k (k = 6) non-constant coefficient matrices A (source), Bc
(destination): Sf = A^T (C^E_XX + rho_X I) A, Sg = Bc^T (C^E_YY + rho_Y I) Bc, Sfg = A^T C^E_XY Bc with E-centred
features; VAMP-2 = ||Sf^-1/2 Sfg Sg^-1/2||_F^2 (soft Ulam: rho = 0). It is the sum of squared canonical correlations
between the two k-dimensional fitted spans on E, invariant to bases within the spans, at most k.

### 4.3 Cross-check against deeptime (`test_deeptime_crosscheck`)

deeptime 0.5.0's VAMP on the same centred fit features gives the same singular values (1e-8), and
`vamp_score(model, r=2, covariances_test=..., dim=6) - 1` equals `transfer.vamp2` (1e-8). The normalisations agree
once three differences are accounted for: (i) deeptime's score adds 1 for the constant (mean-removed data), ours
excludes it; (ii) deeptime has no ridge (an eigenvalue cutoff epsilon instead), so the ridge path is checked by
passing deeptime the ridge-regularised covariances (C + rho I) through `CovarianceModel`; (iii) deeptime's
orientation is source x destination (Koopman matrix C00^-1/2 C0t Ctt^-1/2 = our T transposed), so its singular
values agree and its left / right roles swap. Biased (1/n) covariances in both (`bessels_correction=False`).

### 4.4 Tuning (label-free)

`tune` fits every grid point and selects the one with the largest tuning-split VAMP-2. Grids: soft Ulam: c in
C_GRID (with the gate); linear: eta in ETA_GRID; Nystrom: c x eta (25 points); diffusion: eta. m in {64, 128}: the
real runner reports the two fixed-m variants (`<est>_m64`, `<est>_m128`) **and** the jointly selected variant
`<est>_mjoint` that PLAN section 6 asks for (m chosen by the larger tuning VAMP-2). For `mjoint` the selection of m
is repeated inside every permutation of the control: the permutation sequence depends only on (seed, splits, mode),
so the m = 64 and m = 128 controls share permutations and the per-permutation choice is made post hoc from their
tuning VAMP-2 values. The fixed-m variants are additional characterisation (a declared deviation in the sense that
PLAN names only the selected m); the P8 sample-size curve reports fixed m only.

### 4.5 Controls

(i) **Independence control.** The pairing is permuted within each of fit, tune and test (the destination view of
image i becomes that of pi(i), pi preserving each split), and the whole procedure including tuning is repeated per
permutation; the 95% point of the j-th function's test correlation over permutations is the control band. Features
are not recomputed: every feature map is a function of the fit images' views only, and a permutation inside the fit
split leaves that set unchanged, so the permuted-pairing features are the same arrays re-indexed. (The landmark
images of a view stay the same images; a fresh landmark draw would be another seed of the same procedure.)
`mode='fit_only'` permutes the fit split only and is reported separately as "training-pairing disruption" (it leaves
real dependence in tune and test). (ii) **Composition discrepancy**, section 5. Also: P8 with independent pairing
(same marginals) as a generative control (`test_p8_independent_pairing_gives_control_band`).

## 5. Composition discrepancy

For a chain a -> b -> c fitted with one feature specification (the intermediate view's feature object and ridge
identical in T_ab and T_bc; `fit_chain` selects the common (c, eta) by the sum of the three tuning VAMP-2 scores),
write each fitted operator as T_xy = A_y^-1/2 S_yx A_x^-1/2 with S the fit cross-moment (centred features for CCA,
raw memberships for Ulam), A the regularised marginal moment (C + rho I; Ulam: D_p) and G the unregularised one
(C; Ulam: the membership Gram Phi^T W Phi). Then exactly:

    D = T_bc T_ab - T_ac
      = A_c^-1/2 S_cb (A_b^-1 - G_b^+) S_ba A_a^-1/2         regulariser term (ridge at b; for Ulam the
                                                               soft-membership overlap D_p versus Phi^T W Phi)
      + A_c^-1/2 (S_cb G_b^+ S_ba - S_ca) A_a^-1/2           partial term.

G_b^+ cancels in the sum, so the identity holds for any choice of pseudo-inverse; numerically the residual scales
with ||A_c^-1/2 S_cb|| ||G_b^+|| ||S_ba A_a^-1/2|| and is reported relative to that product
(`additivity_error_rel`, asserted < 1e-10), with the numerical rank and condition number of G_b.

The partial term is minus the whitened fit-split partial cross-moment of the a- and c-features given the
b-features: S_ca - S_cb G_b^+ S_ba = E_fit[r_c r_a^T], r = residual of the linear regression on the b-features. If
a -> b -> c is Markov (X independent of Z given Y) and the b-features (with the constant) span L2(mu_Y), the
regression equals E[. | Y] and the population partial cross-moment is E[Cov(F_c, F_a | Y)] = 0. Otherwise it is
nonzero, and a finite b basis, non-Markov views (view insufficiency, e.g. X = Z with Y constant) and sampling all
contribute; **this pilot does not separate them**, and nothing below is claimed to identify any of them.

Reported alongside, as a separate diagnostic and not as a component of D: the **held-out residual cross-moment**
(`heldout_residual_crossmoment`): regressions on the b-features fitted on fit images, residual cross-moment of the
a- and c-features on test images, fit whitening, same sign as the partial term. Norms: Frobenius over all features,
and projected on the top-6 non-trivial singular pairs of T_ac, relative to sqrt(sum_{j<=6} sigma_j^2)
(`topk_rel`). Also: top-6 singular values of the endpoint and the composition, held-out correlations of both sets
of functions, source-span overlap on test images.

**Intermediate-basis sweep (descriptive).** `composition_sweep` selects the base chain once, then varies only the
intermediate features: m_b = round(f * m_b0), f in {1/4, 1/2, 1, 2, 4}, with the endpoint feature objects, eta (and
the relative-ridge convention rho = eta tr(C)/m) and the **absolute** bandwidth of every view frozen (kappa_b at its
base value; landmarks nested). Sizes at which the b-features are inadmissible (soft Ulam below the occupancy gate)
are reported as unavailable; endpoints are never shrunk. Actual feature dimensions are recorded and plotted: at a
frozen bandwidth, Nystrom's numerical rank saturates (on the calibration chain 64 and 128 landmarks give m_b = 35),
so the actual dimension, not the landmark count, is the axis. Calibration chains: the known-Markov chain
theta -> phi = 2 theta + eta1 -> chi = 2 phi + eta2 (independent wrapped noise, s1 = s2 = 0.25; T_ac singular values
exp(-m^2 (4 s1^2 + s2^2)/2), two per m; T_ac = T_bc T_ab in population) and the broken chain (the middle view
re-paired at random). These are reference values at matched n and m, not a decomposition.

## 6. Preregistered pilot criteria

Exactly as PLAN section 6 states them (fixed before the run):

> tuning selects (m, rho, bandwidth) by tuning-split VAMP-2; a function counts as "supported at this resolution"
> only if its test correlation exceeds the 95% point of control (i) in at least 8 of 10 splits; a soft-Ulam table is
> used only if every retained landmark has effective occupancy >= 10 fit images, otherwise m is reduced; the report
> targets the first few (<= 6) held-out functions and never individual transition probabilities. VAMP-2 = sum of
> squared singular values of the test-split whitened cross-covariance in the fitted top-6 non-constant functions
> (constant removed; test covariances regularised with the tuned rho); 200 permutations per split. The 8-of-10 rule
> is a descriptive support criterion (splits overlap, several functions and pairs are inspected), not a 95%
> confidence statement, and the page says so.

Operational details fixed by this builder (reviewed in REVIEW_3_transfer.md; not in the PLAN):

- Correlations are signed (fitted pairs are oriented by the fit); the control's j-th function is compared with the
  observed j-th function. With 10 splits the threshold is >= 8; with s splits it is ceil(0.8 s).
- "Leading supported" q = the number of consecutive supported functions from j = 1.
- **Held-out stability** of each side separately: for every pair of splits (i, j), the subspace overlap (mean
  cos^2, constant removed) of the two fits' leading q functions on the images held out from both fits (not in
  fit_i or fit_j; about 500 images per pair), averaged over the 45 pairs.
- **Sets** ("sets only if the functions are stable across splits"), gated per side: a side gets sets only if q >= 1
  and its held-out stability is >= 0.8 (STABILITY_MIN, a provisional builder's threshold). For that side the
  consensus span = top-q eigenvectors of the mean over splits of the projectors onto each split's leading q
  functions on all images (as source C3 in PLAN section 2), the constant prepended, SEBA (the repository port) at r
  in {2, 3, 4, 6, 8} with r - 1 <= q. The source x destination joint table (matching sets through the images, not
  by column index) is given only when both sides have sets. These are **transductive exploratory summaries**: the
  consensus span and SEBA are computed on all images after fitting, and held-out correlations validate the
  functions, not the partitions built from them. The JSON and the page carry that label. Agreement with M_b M_a,
  Mbar and linear CCA is likewise computed on all images and labelled transductive.
- Soft-Ulam occupancy uses the soft count (section 3.2).

## 7. Asserted tests (`test_transfer.py`)

Views equal to stats.raw_to_views and the feature-space cosine; orientation and shapes; forward-composition order;
the soft-Ulam trivial pair; the memorisation pathology (full-rank unregularised Nystrom and hard-membership Ulam at
m = n_fit: all in-sample singular values > 0.99 under within-split permutations, held-out correlations centred at
zero; the occupancy gate refuses the table); split permutations; P8 recovery (tolerances below) and the linear miss;
P8 independent pairing inside the control band; P2 linear CCA against the population oracle; composition
reference values (Markov chain: held-out residual cross-moment < 0.15 of the top-6 scale; broken chain > 0.6; the
identity exact); Ulam composition keeps the trivial pair; the diffusion row extension; the occupancy gate at
N_fit = 250; deeptime. Added after review 3: the intermediate-basis sweep freezes endpoints, eta and every kappa and
marks inadmissible sizes unavailable; sets are gated per side and labelled transductive (stable source with
unstable destination gives source sets only, no joint table); held-out stability uses only images held out from
both fits; joint selection of m is repeated inside the permutations; the 10k runner on a final-checkpoint-only
fixture runs the eight final pairs, refuses time pairs and fails preflight before computing when a file is
missing. 19 tests.

P8 tolerances at N_fit = 1000, sigma = 0.35 (truth 0.941, 0.941, 0.783, 0.783, 0.576, 0.576): Nystrom and diffusion
training singular values of all 6 within 0.10; held-out correlations within 0.10 (functions 1-4) and 0.15 (5-6);
soft Ulam held-out correlations of functions 1-4 within 0.10 (5-6 reported only: the gate limits m to 32 at this
N). Linear CCA on the 2-d coordinates has 2 functions, held-out correlation inside the permutation band, test
VAMP-2 < 0.05.

## 8. Relation to the average and the product

- **Mbar** (the current construction): one function on image indices with small *average within-view* Dirichlet
  energy; order-invariant; uniform mass on images; the same per-image function in every cell.
- **Q = M_b M_a**: per-view smoothing, identity transport on image indices, smoothing again; uniform mass; separate
  left / right singular vectors; constant is the trivial pair because each M is doubly stochastic.
- **Transfer estimators**: separate source and destination function spaces, empirical marginal weighting
  (Ulam: D_p, D_q; CCA: whitening by the marginal covariances), and a declared resolution (landmarks / features /
  ridge). They measure what is predictable across views, held out. Q is close to a transfer estimate in which both
  feature spaces are the views' diffusion eigenfunctions and transport is the identity on images; the differences
  are the measure (uniform versus empirical-marginal whitening) and that Q is never validated on held-out images.
- **Multi-cell blocks** (PLAN section 6): endpoint T_ab, the chain where intermediate features are compatible
  (section 5), and generalised CCA in centred MAXVAR form, eigenvectors of the mean of the per-cell projectors onto
  span(F_s), which is again an average of operators on image indices (next to Mbar and C3). With ridge the
  projectors become hat matrices F_s (F_s^T F_s + n rho I)^-1 F_s^T, which are not projectors and not the identity
  at positive ridge. Without ridge and with centred features of full rank (m >= n_fit - 1), each projector is the
  centring matrix H = I - 11^T/n, so the average is H and every non-constant direction is degenerate (in-sample
  memorisation again); GCCA needs the same held-out discipline. Not implemented in this pilot.

## 9. Synthetic results

Filled from `out/transfer/synthetic.json` and `out/transfer/curve_p8.json`; see NOTES_C.md for the numbers and the
run record. Figures: `p8_curve_test_corr.png`, `p8_curve_train_sv.png`, `p8_occupancy_gate.png`,
`chain_composition.png`.

P2 note: for jointly Gaussian raw views the population nonlinear canonical correlations are products of the
linear ones (Hermite polynomials: rho_i rho_j, rho_i^2, ...). P2 after centring and cosine normalisation is not
jointly Gaussian, so that spectrum is not the truth for the normalised views; it only indicates why nonlinear
dependence beyond the linear span is expected. Observed (P2 at a = 1): Nystrom held-out correlations 0.61-0.67 for
functions 4-6, linear at chance.

## 10. Real-data runner

Inputs (builder A, SCHEMA.md section 6): `raw_flat_<step>_<si>.npy` (2,000 probe images, float64, probe order;
`si` = site index 0..11) and, for the final checkpoint only, `raw10k_flat_64000_<si>.npy` (float32, CIFAR-10 test
order) with `raw10k_index.npz` (`test_index`). Every run starts with `preflight`: all needed files exist, square of
the expected size, finite positive diagonal, symmetric on a sample, and (10k) `test_index` has 10,000 unique ids;
it raises with the full list before computing.

`real` (2,000 images): the 12 pairs fixed in PLAN section 6 (`PAIRS`); 10 predeclared splits 1,000 / 500 / 500
(seeded by (20261115, N, split); `splits_2000.json`); estimators linear, soft Ulam, Nystrom, diffusion; m in
{64, 128} plus the jointly selected m; 200 permutations (all splits) and 50 fit-only permutations per split; 4
composition triples (builder's addition, `TRIPLES`: stem-b4-b9, b1-b5-b9 at 64000; b5 and b9 at 1024 -> 6400 ->
64000) with the intermediate-basis sweep. Work units: (pair, split), one reference unit per pair (M_b M_a, Mbar),
(triple, split); `assemble_real` builds the JSON; the same functions run locally (`smoke-real`) and on Modal.
`real10k`: the eight final-checkpoint pairs only (time pairs are refused), one predeclared 6,000 / 2,000 / 2,000
split, N_fit in {500, 1000, 2000, 6000} as nested subsets of the fit split, 50 permutations.

### JSON for the site builder (`out/transfer/real/transfer_pilot.json`, schema `transfer-pilot-v2`)

    {schema, status: "pilot", tag, n_images, split_sizes, n_splits, n_perm, n_perm_fit_only, estimators, m_values,
     k: 6, eta_grid, c_grid, occupancy_min, stability_min, sets_status, unit_seconds,
     pairs: [ {id, kind: "adjacent"|"distant"|"time", src: {site, step}, dst: {site, step},
               reference: {Q_sv[6], Mbar_eig[6]},                  # M_b M_a singular values, Mbar eigenvalues
               results: { "<estimator>_m64" | "<estimator>_m128" | "<estimator>_mjoint": {
                   train_sv[split][6], test_corr[split][6],        # null where an estimator has < 6 functions
                   control_q95[split][6], fitonly_q95[split][6],   # control (i) and training-pairing disruption
                   vamp2_test[split], control_vamp2_q95[split],
                   n_support_splits[6], supported[6], leading_supported, support_rule,
                   selected[split] {spec, eta, m_used, vamp2_tune}, m_selected[split] (mjoint only),
                   occupancy[split] {min_soft_src, ...} | null,
                   stability {src, dst: {q, mean_pairwise_cos2, n_pairs, images}},     # on shared held-out images
                   agreement {Q_src, Q_dst, Mbar_src, Mbar_dst: {mean_cos2_median, chance}, linear_src, scope},
                   sets: null | {status, sides, src_labels {r: [N]}, dst_labels {r: [N]},
                                 joint_table {r: [[...]]} (both sides only), src_consensus_eig, dst_consensus_eig}
               } } } ],
     composition: [ {id, views: [{site, step} x 3], per_split: [ {"<estimator>": {base_selected, base_dims, kappa,
                     rows: [ {m_b_requested, factor, spec_b, eta, dims {m_a, m_b, m_c}, rho_b, kappa_b,
                              total, regulariser, partial, heldout_residual_crossmoment: {fro, topk, topk_rel},
                              additivity_error, additivity_error_rel, G_b_rank, G_b_cond, sv_endpoint,
                              sv_composed, test_corr_endpoint, test_corr_composed, source_span_overlap_test}
                            | {m_b_requested, factor, unavailable} ] }} ] } ] }

Labels: -1 = unassigned, else 0..r-1 in SEBA reliability order. Captions should state what is plotted and the
control value (the permutation band), label the variant as a pilot, and label sets and agreements as transductive
exploratory summaries.

## 11. Review record

The mathematics review (GPT-6 Astra) and the responses are in `REVIEW_3_transfer.md`. Points raised by this builder
for that review, as amended by it: (1) soft-Ulam sigma_j versus image-function correlations, section 3.2 (an
identity for moments under one table and distribution); (2) the composition terms: only the exact identity is
claimed (section 5); (3) occupancy: soft count, no landmark dropped (section 3.2); (4) GCCA with ridge uses hat
matrices; centred full-rank projectors approach H (section 8); (5) VAMP-2 regularisation is applied in feature
coordinates (section 4.2); (6) JSON size: about 190 kB per pair and m variant, mostly SEBA label vectors.
