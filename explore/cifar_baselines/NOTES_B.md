# Builder B notes: planted and null benchmarks (PLAN.md section 8)

Files: `planted.py` (generators + oracles), `bench.py` (runner, scoring, summary), `bench_modal.py` (Modal CPU
fan-out), `test_planted.py` (regression tests), outputs in `out/bench/` (`summary.json`, `rows.jsonl.gz`, `parts/`).
Everything here is descriptive. Truth and oracles are status V (validation): they never feed a kernel, basis, rank or
selection.

Tests: `uv run --with torch --with numpy --with scipy --with scikit-learn --with pytest pytest explore/cifar_baselines/test_planted.py`
(31 tests, about 100 s on 2 threads). The command without `--with pytest` fails to spawn pytest in this environment.

## Common conventions

* Each generator: `make_<id>(seed, N, level, **params) -> (Xs, truth)`. Latent coordinates Y_l (N x D) per cell are
  embedded by an independent Haar isometry Q_l (d_l x D, orthonormal columns), d_l cycling through
  48, 64, 96, 128, 160, 192, 256, 320 (at least D). An isometry changes no dot product (asserted to 1e-10 in the tests),
  so ambient width is inert by construction. "s eps" below means iid N(0, s^2) per latent coordinate,
  independent across cells unless stated.
* Pipeline per cell = the production one: `coherent.centred_cosine_gram` (column-centre, row-normalise), then
  `sources.default_ops` (rule 0: union 15-NN 0/1, floor 1e-6, zero diagonal, `symmetric_balance`, M = (I+P)/2).
* One block per benchmark (`all` = every cell) except P6 (`all`, `theta` = cell 0 only) and P10 (`early` = cells 0-1,
  `late` = cells 4-5, `all`).
* Ranks r (SEBA rank, counting the constant) per benchmark are listed below; there is no rank selection.
* 10 seeds (0-9) x N in {500, 1000, 2000}, every level.

## Oracle

`planted.oracle(Xs, truth)`: knows the generative form, so it projects each **preprocessed** cell
(centred + row-normalised) onto the planted latent subspace Q_l[:, planted dims] and runs a 2-fold (random halves) kNN
(k = 15) classifier or regressor there. Reported: chance-normalised accuracy kappa = (acc - c) / (1 - c), c = largest
class share, per cell, the minimum over cells, and pooled (cells' projections concatenated). Continuous targets get
held-out R^2 (kNN, or ridge with 1e-3 relative penalty for the linear P2). P0 uses random pseudo-classes and the
linear CKA of two cells. P6 also reports the "apparent eta" 2|mean exp(i theta_hat)| of the angles in the
preprocessed view. P8 reports the phase-free harmonic correlations |mean exp(i(2m theta_hat - m phi_hat))| against
exp(-m^2 sigma^2 / 2). P11 (graphs) uses each node's edge counts to the true communities (argmax if assortative,
argmin if disassortative). P12 runs kNN on the full preprocessed state (the planted subspace is unknown).

## Benchmarks as implemented

| id | difficulty axis (levels) | generative definition (fixed parameters) | truth | ranks |
|---|---|---|---|---|
| P0 | rho: 0, .5, .9, 1 | X_l = sqrt(rho) G0 + sqrt(1-rho) E_l, entries N(0,1), d = 64, L = 4 | none; 10 balanced pseudo-classes | 2, 4 |
| P0d | d: 16, 64, 256, 1024 | P0 at rho = 0 | none | 2, 4 |
| P1 | a: 0, .1, .2, .3, .5, 1 | Y_l = a mu_g + s eps_l, mu_g = 4 regular-simplex unit vectors in latent dims 0-2, D = 16, s = .25, L = 4, balanced g | g | 4 |
| P2 | a: 0, .25, .5, 1, 2 | Y_l = a A_l u + O_l diag(j^-1/2) v_l, u ~ N(0, I_3) shared, v_l ~ N(0, I_16) per cell, A_l 16 x 3 Haar orthonormal, O_l Haar rotation, L = 4 | span of u (and the oracle's raw-vs-normalised R^2) | 4 |
| P3 | b/a: 0, 1, 1.5, 2, 3, 4 | Y_l = a g e1 + b h_l e2 + s eps_l, g, h_l balanced +-1, h_l independent per cell, a = 1, s = .5, D = 8, L = 6 | g (persistent); h_l (cell-specific nuisance) | 2 |
| P3L | L: 1, 2, 4, 8 | P3 at b/a = 2 | g; h_l | 2 |
| P4 | n_bottleneck: 0, 1, 2, 3 | P1 at a = 1, L = 6; interior cells at positions {}, {3}, {2,4}, {1,3,4} replaced by s eps (isotropic noise) | g; informative-cell mask | 4 |
| P5 | s: .05, .3, .5, .8, 1.2 | two Hopf-linked great circles on S^3: group 0 at (cos t, sin t, 0, 0), group 1 at (0, 0, cos t, sin t), t ~ U[0, 2pi), same t and g in every cell, + s eps in D = 8, L = 4. Both groups have mean 0 (no linear separator) | g | 2, 3 |
| P6 | eta: 0, .3, .6, .9 | theta ~ (1 + eta cos theta)/2pi (rejection); cell 0 = (cos theta, sin theta), cell 1 = (cos phi, sin phi), phi = theta + .5 sin theta; + .05 eps in D = 4 | none; theta, density | 2, 3, 5 |
| P7 | contamination: none, .5%-persistent, 2%-persistent, .5%-cell, 2%-cell | P1-type mixture, a = .8, s = .25, masses (.60, .25, .10, .05), spreads s x (1, .8, .6, .5), L = 4; m = frac N pairs (i, j): Y_l[j] = Y_l[i] + 1e-3 xi; persistent = same pairs every cell (truth g[j] := g[i]); cell = new pairs per cell (truth keeps g) | g; duplicate pairs | 4 |
| P8 | sigma: 0, .25, .5, 1, independent | cell 0 = (cos theta, sin theta), cell 1 = (cos phi, sin phi), phi = 2 theta + N(0, sigma^2) mod 2pi; "independent" = phi ~ U independent; no ambient noise; L = 2 | singular values exp(-m^2 sigma^2/2) (two per m); spans {cos theta, sin theta}, {cos 2theta, sin 2theta}, {cos phi, sin phi} | 3, 5 |
| P9 | a: 0, .25, .5, 1, 2 | u, v balanced +-1 independent, w = uv; cell 0 encodes u, cell 1 v, cell 2 w: Y_l = a c_l e1 + .5 eps, D = 8 | span(u, v, w); joint (u, v) 4-partition; per-cell u, v, w | 4, 2 |
| P10 | beta_max: 0, .25, .5, 1, 2 | 4 leaves, top t = leaf // 2 on e1 (+-1); leaf bit on e2 (top 0) or e3 (top 1) with amplitude beta_l = beta_max l/5, l = 0..5; + .3 eps, D = 16, L = 6 | top and leaf labels | 2, 4 |
| P11 | n_disassortative: 0..4 | operator-level fixture (graphs, not activations): 2 balanced communities, L = 4, expected degree 20; first n_d cells p_out = 4 p_in, the rest p_in = 4 p_out; P = symmetric_balance(log(A + 1e-6), zero diagonal), M = (I+P)/2 | community | 2 |
| P12 | train steps: 0, 250, 1000, 4000 | `explore/vit_coherent/mlp_planted.py` imported unmodified (ResMLP, train, sample): residual MLP trained on y = (A+B) mod 8, cells = residual states h0..h4 (width 256); probe of N random (A, B) draws; network trained once per (seed, steps) | y; A, B | 8 |

## Methods scored

All of `sources.SOURCES` (M0, L1, L2, L3, G1, G1n, G2, C1first, C1mid, C1last, C2, C3), with a shared cache so ranks
and blocks reuse eigensolves; `variants.combine` block-combination variants (PLAN 5 E), each rounded by the unchanged
SEBA on its leading q = r-1 non-constant vectors: V_meanM2 (mean of M^2), V_product (Q = M_b...M_a, right/source
span), V_product_left (left/destination span), V_two_end (M_b M_a), V_expprod (prod exp(-(1/2) L_l)), V_concat_graph
(one rule-0 graph on C_B), V_union_graph, V_intersection_graph; for P8 and P2 the transfer estimators of
`transfer.py` (T_linear, T_nystrom, T_diffusion, T_soft_ulam). P11 runs only the sources that read operators
(M0, G1, G1n, C1*, C2, C3), the operator variants, and G2op (textbook D^-1/2 A D^-1/2 of the SBM graphs, mean,
row-normalised k-means; a bench-local operator-level analogue of G2).

## Statistics (rows of `rows.jsonl.gz`; `stat` column)

Per (block, r, method): `ami`, `ari` (unassigned as a category), `ami_joint`, `mj_max` (padded matched Jaccard);
`truth_span_captured` (share of the planted indicator span captured by the method's span; chance in
`truth_span_chance`); `contrast_captured` for span truths; `span_<name>_captured` (P8); `ami_nuisance_max` (P3: max
over cells of AMI with h_l); `ami_top` (P10); `ami_u/v/w` (P9); `ami_A/B` (P12); `best_jaccard_group<g>` and
`dup_leverage_ratio` (P7); `density_corr_max`, `set_mean_density_spread` (P6); `ami_pseudo` (no-truth benchmarks).
Always: `unassigned`, `n_sets`, `min_set_share`, `eff_sets` (exp entropy of set sizes), `ret_excess_mean/min`
(stats.retention minus its size null, mean over sets, then mean / minimum over the block's cells; per cell
`ret_excess_cell<j>` for P3, P3L, P4, P9, P11), `val0..val8` (each source's own values; M0/G1/G1n/G2/C1* include the
leading mode at index 0, L1/C2/C3 and the variants do not), `gap_after_span`, `span_effective_points`,
`nest_H_given_r<r'>` (S3 conditional entropy), `seconds`. No-truth benchmarks (P0, P0d, P2, P6, P8) add
`stab_ami_0/1`: AMI between the full-sample sets and the sets recomputed on an 80% subsample (everything
recomputed), on shared points, at the first listed rank. Transfer rows (block `all`, r null): `train_sv<j>`,
`test_corr<j>`, `vamp2_test`, `theory_sv<j>`, `perm_q95_corr<j>`, `perm_q95_vamp2`, source/destination span overlaps.
`summary.json`: median and 10% / 90% quantiles over seeds per (bench, level, N, block, r, method, stat).

## Oracle curves (N = 1000, median [10%, 90%] over 10 seeds; full curves at every N in `summary.json`, method `oracle`)

| id | statistic | levels -> values |
|---|---|---|
| P0 | pseudo-class kappa, cell 0 | rho 0, .5, .9, 1: -0.00, 0.00, -0.01, -0.01 (all within [-0.03, 0.01]) |
| P0 | linear CKA cells 0-1 | 0.06, 0.29, 0.82, 1.00 |
| P0d | linear CKA cells 0-1 (rho = 0) | d 16, 64, 256, 1024: 0.02, 0.06, 0.20, 0.51 (finite-N CKA bias grows with d) |
| P1 | pooled kappa | a 0, .1, .2, .3, .5, 1: -0.00, 0.25, 0.65, 0.89, 1.00, 1.00 |
| P1 | min-cell kappa | -0.03, 0.09, 0.29, 0.50, 0.82, 1.00 |
| P2 | mean-cell R^2 of u (normalised view) | a 0, .25, .5, 1, 2: -0.01, 0.25, 0.54, 0.78, 0.86 |
| P2 | same before row normalisation | -0.01, 0.25, 0.56, 0.83, 0.95 |
| P3 | g min-cell kappa / nuisance h_l mean kappa | g 0.94 at every b/a; h: -0.01 (b/a = 0), 0.95, 1.00, 1.00, 1.00, 1.00 |
| P3L | g pooled kappa | L 1, 2, 4, 8: 0.95, 1.00, 1.00, 1.00 |
| P4 | min-cell kappa / pooled kappa | n_b 0, 1, 2, 3: 0.99, 0.00, -0.01, -0.02 / 1.00 at every level |
| P5 | min-cell kappa | s .05, .3, .5, .8, 1.2: 1.00, 0.92, 0.58, 0.20, 0.04 (pooled 1.00, 1.00, 0.97, 0.71, 0.32) |
| P6 | R^2 of (cos theta, sin theta), cell 0 | 1.00, 1.00, 1.00, 0.99 |
| P6 | apparent eta in the preprocessed view | eta 0, .3, .6, .9: 0.04, 0.16, 0.29, 0.37 (centring shifts the origin to the sample mean, so the apparent density modulation is about half of eta; identical before normalisation) |
| P7 | min-cell kappa | none .99, .5%-p .99, 2%-p .99, .5%-c .98, 2%-c .96 |
| P8 | harmonic 1 corr (theory) | sigma 0, .25, .5, 1, indep: 1.00 (1.00), 0.97 (0.97), 0.88 (0.88), 0.61 (0.61), 0.03 |
| P8 | harmonic 2 corr (theory) | 1.00 (1.00), 0.88 (0.88), 0.61 (0.61), 0.14 (0.14), 0.03; odd-source corr 0.04-0.05 throughout |
| P9 | each cell's own-contrast kappa | a 0, .25, .5, 1, 2: about -0.03..0.03, 0.34, 0.67, 0.95, 1.00 |
| P10 | leaf kappa, last cell / first cell | beta 0, .25, .5, 1, 2: 0.32, 0.71, 0.94, 1.00, 1.00 / 0.32 throughout (top known, leaf at chance: (0.5 - 0.25)/0.75 = 0.33); top pooled 1.00 |
| P11 | min-cell kappa (edge-count oracle) | 0.99 at every n_d |
| P12 | min-cell / pooled kappa of y | steps 0, 250, 1000, 4000: 0.72, 0.73, 0.73, 0.75 / 0.73, 0.97, 0.99, 0.99. At N = 1000 each (A, B) combination has about 15 probe points, so the k = 15 kNN oracle partly reads the combination itself; the untrained per-cell value is therefore far above chance (at N = 300 it was 0.04) |

## Selected contrasts in the sweep (N = 2000, median [10%, 90%])

* P3L (b/a = 2), AMI with g at L = 1, 2, 4, 8: M0 -0.00, 0.88, 0.97, 1.00; L1 0.00, -0.00, 0.28 [0.14, 0.63], 1.00;
  L3 -0.00, -0.00, 0.00 [-0.00, 1.00], 1.00. P3 (L = 6), AMI with g at b/a 0..4: M0 0.99 throughout; L1 and L3
  0.98-1.00 up to b/a = 2, then 0.00 at 3 and 4 (L3 AMI with a nuisance h_l = 1.00); C1first 0.81, 0.33, 0, 0, 0, 0.
* P5, AMI at r = 2 for s .05, .3, .5: M0 1.00, 0.74 [0.47, 0.92], 0.01; G2 1.00, 0.97, 0.00; L1 0.55 [0.00, 0.60],
  0.06, 0.01; L2 and L3 0.00 throughout (oracle min-cell 1.00, 0.92, 0.58).
* P8, share of span{cos 2theta, sin 2theta} captured at r = 3, sigma 0, .25, .5, 1, independent: M0 1.00, 0.98, 0.93,
  0.77, 0.03; C1first 0.00 throughout (it captures span{cos theta, sin theta} at 1.00); L1 0.56, 0.50, 0.42, 0.20, 0.00.
  Transfer held-out corr of the first function: T_nystrom 1.00, 0.96, 0.87, 0.54, -0.02; T_soft_ulam 0.99, 0.96, 0.86,
  0.59, -0.01; T_linear 0.05, 0.02, 0.00, 0.01, -0.02; theory 1.00, 0.97, 0.88, 0.61; permutation 95% point about 0.06.
* P11, AMI at n_d = 0, 1, 2, 3, 4: M0 1.00, 0.83, -0.00, -0.00, -0.00; V_meanM2 1.00, 0.94, 0.01, 0, 0; V_product
  1.00, 0.13, 0, 0, 0; V_two_end 0.95, 0, 0, 0, 0; C2 1.00, 0.95, 0.98, 0.84, -0.00.
* P4, AMI with g stays 1.00 for M0, L1 and V_product at every n_b. M0's mean retention excess over its size null is
  0.74, 0.62, 0.49, 0.37 for n_b = 0..3, and in bottleneck cell 3 it is 0.00.
* P9, AMI with the joint (u, v) 4-partition at a 0, .25, .5, 1, 2: L1 0.00, 0.09, 0.49, 0.96, 1.00; L3 0.00, 0.11, 0.51,
  0.96, 1.00; M0 0.00, 0.01, 0.10, 0.47, 0.59. The share of span(u, v, w) captured follows the same pattern: L1 0.99 and
  M0 0.58 at a = 2.
* P0 (no groups), at r = 4: M0 returns 3.99 effective sets with 12-13% unassigned at every rho. Its retention excess
  at r = 2 is 0.10, 0.13, 0.19, 0.24 for rho 0, .5, .9, 1. Subsample-stability AMI for M0 is 0.03-0.19 and for L1
  0.19-0.32.

## Deviations from PLAN section 8 (with reasons)

1. **P11 is operator-level.** A cosine kNN graph built from activations cannot be strongly disassortative, because
   metric neighbourhoods are nearly transitive. So the graphs are generated directly, and only the sources that read
   operators are scored (M0, G1, G1n, C1*, C2, C3, the operator variants), plus a bench-local G2op. L1-L3, G2 and the
   concat/union/intersection graph variants are not defined on P11.
2. **P12 is included.** `mlp_planted.py` is imported without modification: ResMLP, train and sample are reused;
   its `main` is not run. The probe is unbalanced, with random (A, B) draws. Note the oracle caveat above.
3. **V_union_graph is skipped at N = 2000.** In a local timing run (on a heavily loaded box) its Sinkhorn step in
   `variants.combine('union')` took about 1150 s at N = 2000, against about 2 s at N = 1000. It is run at N = 500 and
   1000 only. This was not investigated further (it is builder A's code) and is flagged to the orchestrator.
4. **Transfer settings are smaller than the pilot's.** Each N and seed uses one fit/tune/test split (50/25/25%),
   m = 64, and 50 permutations for the independence control. The PLAN pilot uses 200 permutations and 10 splits;
   those are the pilot's preregistered settings and belong to builder C's pilot run.
5. **Stability is thinner than S1.** It uses 2 subsamples of 80% (S1 says 10), only on the no-truth benchmarks
   (P0, P0d, P2, P6, P8), and only at the first listed rank. This was done to bound cost.
6. **Axes are split into named sub-benchmarks.** Each id keeps one declared axis, so P0 (rho) and P0d (d), and P3
   (b/a) and P3L (L), are separate. P7 has one categorical axis that combines contamination fraction and persistence.
7. **Scope cuts.** P4 sweeps only the isotropic-noise bottleneck; the independent-label bottleneck exists as a
   parameter but is not swept. P5 uses one construction (Hopf-linked great circles); arcs and spirals were not built.
   P9's operator-level companion and its non-commuting perturbation were not swept (the `weights` parameter exists).
   P2 uses a geometric spectrum j^-1/2, not real-data-estimated spectra. P6 has no "new sampling density" stability
   check. The Markov-chain calibration for the composition discrepancy (theta -> 2 theta -> 4 theta) is builder C's
   and is not here.
8. **P3's b/a axis does not move M0 or G2.** The clusters are well separated in every cell, so the local graph sees
   the same thing at every b/a; L1, L3 and C1 do move. P3L is the axis where M0 varies.

## Compute and spend

Local work was smoke-size only (N <= 300 in the tests and smoke run, plus timing runs at N = 1000 and one at
N = 2000). The full sweep ran on Modal CPU: app `cm-cifar-baselines-bench`, Volume `cm-cifar-baselines-bench`, 2
physical cores and 4 GiB per container, at the modal.com/pricing rates of 2026-11-15 ($0.0000131 per core-s,
$0.00000222 per GiB-s).

* Pilot `pilot-20261115-083716`: 15 units, 608 container-s, about $0.02.
* Full sweep `full-20261115-083951`: 2,050 units, 0 failed, 32,687 container-s, about $1.15.

Container start-up time is not included in those figures. Total spend is about $1.2, or under $1.5 with an allowance
for start-up, against a $3 cap. Outputs: `out/bench/summary.json` (193,473 summary rows),
`out/bench/rows.jsonl.gz` (1,934,729 rows), and the raw parts in `out/bench/parts/<run_id>/`.

## Review and fix (adversarial code review, 2026-11-15)

The reviewer read all four files and reran 26 conditions at N = 500. Its AMIs matched the saved ones to 1e-15, and it
confirmed the N = 2000 headline medians. One finding: `score()` called `stats.agreement(lab, y)` without declared set
counts. It therefore fell back to max(label) + 1, so a trailing empty SEBA set dropped out of the padded
matched-Jaccard denominator and inflated `mj_max`. Example: P5, s = 0.8, N = 500, seed 7, r = 3,
V_intersection_graph gave 0.0443 where the correct value is 0.0295. AMI, ARI and every other statistic are
unaffected.

Fix:
* `score()` now passes `ka = r` (the declared rank, so k_c = r sets with empty ones included) and `kb = n_groups`.
  `n_groups` is the generator's declared group count; `planted._truth` now requires it for every partition truth.
* Regression test `test_mj_max_counts_trailing_empty_set`: r = 3 with one empty slot, trailing or leading, must give
  2/3 both ways (the old fallback gave 1.0 for the trailing case).
* The reviewer's example now gives 0.0295287.

Labels were not stored, so the partition benchmarks (P1, P3, P3L, P4, P5, P7, P9, P10, P11, P12; 1,390 units) were
rerun on Modal as run `partition-20261115-092724`. Their parts replace the full-sweep parts for those benchmarks. The
no-truth benchmarks (P0, P0d, P2, P6, P8) keep the full-sweep parts; they have no `mj_max`. `summary.json` was
rebuilt from the combined parts in `out/bench/parts/final/`, 2,050 units. Rerun compute: 18,728 container-s,
about $0.66.

**Cross-host reproducibility, measured from the two runs.** The rerun is the same code on different Modal hosts.
Across the 4,839 (benchmark, level, N, block, r, method) AMI medians of the rerun benchmarks:
* the largest difference between runs is 0.10, the 99th percentile 0.014; 1.4% of medians differ by more than 0.01
  and 0.04% by more than 0.05;
* the differences sit in transition regimes: P1 at a = 0.2-0.3 (up to 0.06), P7 with near-duplicates (up to 0.10,
  L3), P12 (up to 0.04, training on CPU);
* P3, P3L, P4, P5, P9, P10 and P11 differ by at most 0.007.

Even the pure-numpy oracle moves by up to 0.028 (P1, a = 0.2). Floating-point differences between hosts (LAPACK and
BLAS kernels) flip near-tied kNN neighbours and votes. Degenerate eigenspaces have the same effect: V_intersection_graph
often has eigenvalues stuck at 0.499 (P eigenvalue about 0), and its sets differ between runs in 750 of 2,340 AMI rows.
So the reviewer's corrected median for its P5 example (0.0451) and this summary's (0.0561, same code) differ through
this effect, not through the fix. The headline contrasts quoted above come from benchmarks with differences of at most
0.007, apart from the P7 group-3 Jaccard. Per-seed values within one run are deterministic.

Total Modal spend: about $1.83 of compute (pilot $0.02, full $1.15, partition rerun $0.66), plus container start-up,
against the $3 cap.

## Chart spec for the site builder

Every series reads `summary.json` rows: (bench, level, N, block, r, method, stat), plotted as the median with a
10-90% band. Use N = 2000 unless an N panel is stated, and block `all` unless stated. Plot the oracle as a grey line
on the same y-axis only where its statistic has the same scale (kappa or R^2 on 0-1, against AMI or captured share);
otherwise put it in a small panel above. Null references: AMI chance is 0; captured-share chance is in the
`*_chance` rows; the retention-excess null is 0. Captions state what is plotted, nothing more.

| bench | chart | x | series (method) | stat (r) | reference | caption |
|---|---|---|---|---|---|---|
| P0 | 1 | rho | M0, L1, L3, G2, C1first | ret_excess_mean (2) | 0 | "Retention of each method's sets above the size null, isotropic Gaussian cells, by cross-cell correlation rho." |
| P0 | 2 | rho | same | stab_ami_0 (2) | 0 | "AMI between sets on all images and on an 80% subsample." |
| P0d | 1 | d (log) | M0, L1, L3 | ret_excess_mean (2); oracle cka_cells_01 panel | 0 | "Retention above size null vs ambient dimension (rho = 0); panel: CKA between two independent cells." |
| P1 | 1 | a | M0, L1, L2, L3, G1, G2, C1first, C3 | ami (4) | oracle pooled_kappa, min_cell_kappa | "AMI with the planted groups vs separation a; grey: oracle kappa." Panel by N (500/1000/2000). |
| P2 | 1 | a | M0, L1, C1first, T_linear, T_nystrom | contrast_captured (4); T_* use src_span_u_captured (r null) | oracle mean_cell_r2 and raw_mean_cell_r2; contrast_chance | "Share of the shared-factor span captured vs factor strength a; grey: oracle R^2 after / before row normalisation." |
| P3 | 1 | b/a | M0, G2, V_concat_graph, L1, L3, C1first, C3 | ami (2) | oracle min_cell_kappa | "AMI with the persistent group g vs nuisance strength b/a (L = 6)." |
| P3 | 2 | b/a | same | ami_nuisance_max (2) | oracle nuisance_cell_kappa_mean | "Largest AMI with any cell-specific nuisance h_l." |
| P3L | 1 | L | M0, L1, L3, C1first, C3 | ami (2) | oracle pooled_kappa | "AMI with g vs number of cells L at b/a = 2." |
| P4 | 1 | n_bottleneck | M0, L1, V_product, V_two_end, C1mid | ami (4) | oracle pooled_kappa and min_cell_kappa | "AMI with g vs number of noise cells; grey: oracle pooled and worst cell." |
| P4 | 2 | cell index 0-5 | M0 at each n_b level (one line per level) | ret_excess_cell<j> (4) | 0 | "Retention of M0's sets above the size null in each cell." |
| P5 | 1 | s | M0, G2, L1, L2, L3, C1first, V_product | ami (2) | oracle min_cell_kappa, pooled_kappa | "AMI with ring membership vs noise s (linked great circles)." |
| P6 | 1 | eta | M0, L1, G2, C1first | density_corr_max (2) and contrast_captured (3) | oracle cell0_eta_hat panel | "Largest correlation of a method direction with the sampling density; share of the first-harmonic span captured." |
| P7 | 1 | contamination (categorical) | M0, L1, L3, G2 | best_jaccard_group3 (4), plus ami | oracle min_cell_kappa | "Best Jaccard with the 5% group, and AMI, by contamination condition." |
| P8 | 1 | sigma (categorical, 'independent' last) | M0, C1first, C1last, L1, V_product, C3 | span_theta2_captured (3) | span chance | "Share of span{cos 2 theta, sin 2 theta} captured." Companion: span_theta1_captured. |
| P8 | 2 | sigma | T_linear, T_nystrom, T_diffusion, T_soft_ulam | test_corr0 (r null) | theory_sv0 (line), perm_q95_corr0 (band) | "Held-out correlation of the first transfer function pair; line: exp(-sigma^2/2); band: permutation 95% point." |
| P9 | 1 | a | M0, L1, L3, G2, C3 | ami (4, joint u-v partition) and contrast_captured (4) | oracle cell<j>_own_kappa | "AMI with the joint (u, v) partition and share of span(u, v, w) captured vs signal a." |
| P10 | 1 | beta_max | M0, L1 for blocks early, late, all | ami (4) and ami_top (2) | oracle leaf_last_cell_kappa, leaf_first_cell_kappa | "AMI with leaves (r = 4) and with the top split (r = 2), per block." |
| P11 | 1 | n_disassortative | M0, V_meanM2, V_product, V_two_end, G2op, C2 | ami (2) | oracle min_cell_kappa | "AMI with communities vs number of disassortative cells (operator-level fixture)." |
| P12 | 1 | train steps | M0, L1, L3, C1first, C1last | ami (8) and ami_A (8) | oracle pooled_kappa | "AMI with y = (A + B) mod 8 and with A vs training steps." |
