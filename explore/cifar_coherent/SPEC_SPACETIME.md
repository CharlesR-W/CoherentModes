# Delta spec: through-depth operator (image x site), after Trower, Djurdjevac Conrad & Klus (exploratory)

Owner request (2026-11-20): the paper *Clustering time-evolving networks using the spatiotemporal graph
Laplacian* (Chaos 35, 013126, 2025) is "a different thing from what we are doing"; work out the smart version
for the base ResNet-20 / CIFAR-10 case and put it in the explorer if that makes sense. Owner's answers:
couple along **depth, per checkpoint**; ship **both** the literal port and a per-site-rescaled version,
switchable; a **slider over about six coupling strengths**; **explorer colouring plus an experiment page**.

Exploratory side study like the rest of `explore/`. Everything not mentioned is unchanged from `README.md`
(probe N = 2,000, 8 checkpoints, 12 sites stem, b1..b9, pool, logits, five kernel rules, symmetric doubly
stochastic fixed-lazy walks M_l = (I + P_l)/2, float64). Nothing is trained. Labels do not enter any
operator or basis on the fixed probe and are used for validation columns only; the inherited probe itself is
class-stratified (200 test images per class), a caveat of the whole study that is stated on the page.
Convention: one recorded site = one unit of depth and one unit of mass; free ends.

## 1. What the paper does, in our notation

The paper has a graph per time step on a fixed vertex set. Ours: vertices = the N probe images, "time" = site
index l = 0..L-1 (L = 12) at one checkpoint, graph at "time" l = the walk M_l. Our walks are symmetric and
doubly stochastic, so the paper's densities, started (as the paper does) at the uniform density, stay uniform and its Koopman and reweighted Perron-Frobenius
matrices are both M_l.

The paper maximises the summed correlation of neighbouring slices, sum_l f_l^T M_l f_{l+1}, relaxes the
per-slice unit-variance constraints to one shared multiplier, and gets the eigenproblem A f = lambda B f with
A[l, l+1] = A[l+1, l] = M_l and B = diag(1, 2, ..., 2, 1) (x) I. Its "spatiotemporal Laplacian" is I - B^{-1}A;
k-means on the rows of the leading eigenvectors labels every (vertex, time) pair. It reports two open problems:
eigenvectors that are constant on every time slice ("temporal") are mixed into the spectrum and must be
filtered by eye, and so the number of clusters cannot be read from a gap.

What this buys over what the repo has: a block **average** gives one function for a whole run of sites (rigid);
a block **product** gives a start function, an end function and the functions pushed through in between;
single cells give 12 unrelated bases. Here the slices at all sites are chosen jointly by one variational
problem and tied together softly. Neither construction shows by itself that a set split or merged.

For more than two views the paper's shared multiplier means its eigenvectors solve one global, degree-weighted
norm problem; they need not satisfy the per-slice constraints, and Algorithm III.11 clusters the stacked rows
without normalising slices.

## 2. The construction

For f = (f_0, ..., f_{L-1}), each f_l a function on the N images, t = a^2, H_l = (I - M_l)/c_l:

    E_t(f) = sum_l f_l^T H_l f_l  +  t sum_{l<L-1} |f_{l+1} - f_l|^2
    L_t    = blockdiag(H_l)  +  t (D_path (x) I_N)

(D_path: Laplacian of the path on the L sites, free ends.) This is the unnormalised supra-Laplacian of
Froyland, Kalia & Koltai (arXiv:2409.11984; the graph form of the inflated dynamic Laplacian), with the
identity as the link between sites: image i at one site *is* image i at the next. It is the operator the
Chaos paper compares itself against, not the paper's own. Three variants are shipped and named as such:

- **identity link, relative site units**: c_l = mean of site l's first 8 nontrivial 1 - lambda_k. By Ky Fan
  this is the least mean roughness of an orthonormal 8-frame at that site, so dividing by it sets that
  quantity to 1 at every site. It does not equalise best cuts or gaps. Default.
- **identity link, native walk units**: c_l = 1.
- **paper transition**: the matrix C = B^{-1}A of section 1 as is, no slider.

Statements (asserted in `test_spacetime.py` against dense matrices):

- **(S1) Sectors.** Slice-constant functions g (x) 1 and their orthogonal complement S (every slice of mean
  zero) are both invariant. On the first the spectrum is t * 4 sin^2(j pi / 2L) ("temporal"); for the paper's
  matrix cos(j pi / (L-1)). Everything is solved inside S, so temporal modes never appear. (An unrestricted
  eigensolver may still mix the sectors at a shared eigenvalue; the test checks the commutator, not
  individual dense eigenvectors.)
- **(S2) The paper's Laplacian as an energy.** For a B-unit generalized eigenvector,

      1 - lambda = sum_{l<L-1} [ f_l^T (I - M_l) f_l + f_{l+1}^T (I - M_l) f_{l+1} + (f_{l+1}-f_l)^T M_l (f_{l+1}-f_l) ]
                 = sum_{l<L-1} sum_{ij} (M_l)_{ij} (f_l(i) - f_{l+1}(j))^2 .

  Exact; all three terms are nonnegative because the walks are lazy (0 <= M <= I). The second form is the
  honest reading: the paper links image i at one site to a *random neighbour* j at the next, whereas the
  additive operator links i to itself and charges roughness separately. They are different graphs, not one
  energy at two exchange rates. Site l's walk is used for slices l and l+1; the last site's walk is unused.
- **(S3) Limits**, for isolated eigenvalues or whole bands. t -> 0: the multiset union of the scaled site
  spectra, with a site-supported eigenbasis. t -> infinity: N - 1 eigenvalues stay bounded and their
  eigenspaces tend to slice-independent eigenvectors of mean_l H_l; with c = 1 that is the existing 12-site
  block average. Identical sites: phi_k (x) cos(j pi (l + 1/2)/L) with eigenvalue h_k + t * 4 sin^2(j pi / 2L).
- **(S4) Depth harmonics are a control, not a law.** For identical sites every cheap image function drags
  along copies with sign-changing depth envelopes; the first such copy of phi_1 comes before the next image
  function exactly when h_1 + t * 4 sin^2(pi / 2L) < h_2, so with one very cheap function and moderate t they
  fill the leading modes. For different sites there is no such family in general (two sites sharing phi with costs h_0 != h_1
  have amplitude eigenvalues split by sqrt((h_0-h_1)^2 + 4 t^2)). How much of the real low space is repeated
  image directions is measured (section 4), not assumed.
- **(S5) Scalar control.** For f in S, with r_l = |f_l| and g_l the smallest eigenvalue of H_l on mean-zero functions,
  E_t(f) >= sum_l g_l r_l^2 + t sum_l (r_{l+1} - r_l)^2. The 12 x 12 matrix diag(g_l) + t D_path is thus a
  lower bound on the first eigenvalue (attained when all sites share their cheapest direction), and its squared
  ground vector is the depth profile the gap landscape alone
  would produce if the cheapest directions of all sites coincided.
- **(S6) Exact trade-off.** The first K modes minimise A + t B over orthonormal K-frames in S, with
  A = total within-site roughness and B = total between-site change. So as t grows, B cannot rise and A
  cannot fall. Label-free, no clustering.

**Coupling grid.** t* := the coupling at which the first temporal eigenvalue equals the first eigenvalue on S
(Froyland-Kalia-Koltai Algorithm 1 step 3: the smallest coupling whose first spatial eigenvalue is at or
below the first temporal one). If every walk is connected, s(t) = lambda_min on S is concave, nondecreasing,
bounded and s(0) > 0, so there is exactly one positive crossing. Connectivity of every site is what the
implementation requires (asserted); without it the calibration is unsupported, not necessarily absent.
Computed in one extremal eigensolve: 1/t* = lambda_max(H^{-1/2} (tau I - D_path (x) I) H^{-1/2}) on S,
tau = 4 sin^2(pi / 2L); cross-checked by bisection in the tests. t* balances two *eigenvalues*, not two
energy terms, and is a ruler, not an optimum. Stored: t = t* x {1/64, 1/16, 1/4, 1, 4, 16} per
(checkpoint, rule, scale); the page shows t as well as t/t*.

**Numerics.** L_t + eps is block tridiagonal with off-diagonal blocks -t I: twelve N x N inversions give its
exact inverse; Lanczos on the inverse; Rayleigh-Ritz on L_t for the values. 24 modes are computed, 16 stored.
Recorded per mode: unprojected 2-norm residual on the float64 vectors (required < 1e-8), and the eigenvalue
gaps at the cutoffs 4, 8, 16.

## 3. Prototype facts (step 64,000, kNN rule; superseded by the stage's numbers)

- Native units: the share of squared norm on b9/pool/logits for modes 1-4 is .987-.998 at t = .03 and
  .82-.93 at t = .3 (re-solved independently by the reviewer). The scalar control (S5) alone gives .985 and
  .780 for mode 1: most of this localisation is the gap landscape, not changing image geometry.
- Relative units: t* = 8.4. At t*/4 mode 1 is late (75% on b9..logits) and mode 2 early (76% on stem..b3).
  At t*, mode 1's site weights run .043-.158, neighbouring unit slices have cosine .992-.999, and its stem
  and logits slices have cosine .68; over modes 1-14 the lowest neighbour cosine is .88. The between-site
  charge is about 19% of mode 1's energy.
- Toy with one very cheap persistent split and an early-only split: modes 1-5 at t* are dominated by the
  persistent split; the early split enters at mode 6 (captured share at site 0: .0002 by rank 4, .74 by rank 8).

## 4. Reading the result

The result is the **low space** (first K = 16 modes) and its two energies; single modes are views of it.
G_l = the N x K matrix of raw slices at site l (for the paper's matrix sqrt(b_l) f_l, so columns are
orthonormal in the same sense). A single mode is intrinsic when its eigenvalue is isolated; near ties only
the space is. Mode numbers are local to a setting and are not matched across the slider or checkpoints.

Primary, all closed form and unchanged by rotating the basis of the space:

- **Where the space lives**: occupancy tr(G_l^T G_l) (sums to K) and the eigenvalues of G_l^T G_l (strengths
  of the distinct image directions present at site l).
- **How many image directions it really has**: eigenvalues of R = sum_l G_l G_l^T (trace K). K depth envelopes
  of one image function give R = K phi phi^T: one direction, not K discoveries.
- **What coupling buys**: A against B (S6) for K in {4, 8, 16} along the six couplings.
- **Neighbouring sites**: mean cos^2 of principal angles between the leading 4 local directions.
- **Gap-landscape control**: the scalar ground profile (S5) against mode 1's depth weights.

Per mode (explorer colouring): signed unit slice u_l = f_l / |f_l| with its depth weight shown; slices
holding under 1% of the mode's weight are drawn greyed (a normalised numerical tail is not a direction);
neighbour cosines, retention u_l^T M_l u_l, share of u_l in the site's own first 8 directions.

**Sets through depth: a post-hoc readout, not part of the construction** (k-means is a fit). Rows of the
rank-8 space, X_l = G_l / |G_l|_F (one scalar per site, so pooled distances are basis-independent), then one
k-means with centroids shared across sites, k in {5, 10}; sites holding almost none of the space get no
labels. Two readouts are stored, because they answer different questions: **shared coordinates** (rows as
they are: did this image change its joint spectral coordinates?) and **shape-aligned** (each site first
rotated onto the previous one by orthogonal Procrustes, an SVD: did the cloud change shape, rigid motion
aside?). Alignment minimises squared displacement, not churn, and is determined only as far as the cross
matrix is nonsingular; its sigma_min/sigma_max per link is stored and weak links are flagged. Gates, in `test_spacetime.py`: identical
sites with a cheap split give zero churn and the true split; the planted merge is recovered at rank 8 (apart
early, one set late, third group apart throughout). Known limit: at rank 12-16 the toy's late sites have
their noise directions rescaled to full size and the merged group is cut at random, so the rank is part of
the readout and is fixed at 8 in advance.

Scores per site, for these sets and for four references (each k-means on the same kind of coordinates, 8 unweighted eigenvectors: independent per site, composed
along the chain by Hungarian matching; the 12-site block average; its relative-units version; block
averages over four contiguous blocks of three sites): chi^2 information kept about one step of the **non-lazy** walk
2 M_l - I, with its ceiling (the k - 1 largest squared nontrivial eigenvalues, by magnitude) and the exact mean of a
same-sizes random relabelling, (q - 1)/(N - 1) (|P|_F^2 - 1) for q non-empty groups; number of non-empty
groups; churn between neighbouring sites both as labelled and after optimal relabelling; class NMI
(validation). The claim to test is a trade-off, shown per site: less churn than independent partitions at a
stated loss of information, or more information than a rigid partition at a stated churn. No verdict from
"inside the corner".

## 5. Stage, outputs, tests, controls

`spacetime_stage.compute` (numpy, no Modal) is run by `spacetime_modal.py`: app `cm-cifar-coherent-spacetime`,
CPU 8 cores / 24 GB, unit = (step, rule), 8 steps x rules {0, 1}, reading
`/out/work/r20s0b/operators_{step}_{si}.npy`, writing `/out/analysis/r20s0b/spacetime/st_{step}_{rule}.npz`
(leading axes: scale 2 x coupling 6; `paper_*` for the paper's matrix; `ref_*` for reference partitions;
unit slices as int16 with weights; probe indices for the viewer to verify). Estimated cost under $2.

The pilot (64000, 0) also runs controls, relative units: (i) relabel the images independently at every site
(same spectra and the same scalar bound, no correspondence), one draw, solved at the base t*: the excess of
the eigenvalues over the scalar bound says how much the real correspondence is worth; a weak effect would
limit the claim, it is not a solver failure; (ii) scale rank 4 and 16 instead of 8: share
of the K = 16 space in common; (iii) drop pool, drop logits, duplicate logits: t*, occupancy, and the share
of the space on stem..b9 in common (each at its own recalibrated t*). Each unit must pass
`spacetime_stage.accept` (residuals, both balances, scalar bound, A + tB = sum of eigenvalues, monotone
trade-off, finite outputs) before it is marked complete. A common relabelling of every site is an exact equivariance (test).
Rule 1 and the other seven checkpoints are the remaining robustness checks. One seed.

## 6. Site

- **Explorer**: colour-by "through depth" for the selected checkpoint row; each of the 12 site panels shows
  its own slice. Controls: variant (relative units / native units / paper transition), coupling (6 stops
  showing t/t* and t; none for the paper), mode 1..16 or sets (k = 5, 10). While active, kernel and rule are
  locked to cosine / kNN and labelled, and block-only cards are hidden or marked not applicable. Side card:
  depth weight with the scalar control, neighbour cosines, retention, occupancy. Data: rule 0 only, int8.
  The built site is measured against the 800 MB cap before this is kept; fallback is a separate viewer.
  `check_browser.py` gains a smoke test: switch checkpoint, variant, coupling, and an incompatible state.
- **Experiment page** `content/experiments/through-depth.html`: what the operator charges for and how it
  differs from the paper's (S2); gap-landscape control and native vs relative units; occupancy and number of
  image directions; the exact trade-off; sets with their scores and flows between sites; the controls;
  checkpoints and rule 1; caveats (one seed, class-stratified probe, unit-of-depth convention, readout rank).
