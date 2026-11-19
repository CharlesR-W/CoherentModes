# Builder A notes (stats, sources, nulls, variants, measure_b)

## Phase 2 (review fixes and full run)

- Review 2 (GPT adversarial code review) found three issues: NaN in public JSON, an assumed laziness convention in
  `spectral_summary`, and set counts inferred from labels. All three are fixed with regression tests; see
  `REVIEW_2_code_A.md`. Also adopted: the jointly-assigned AMI is the headline number (with coverage and a
  within-joint permutation band), and max-over-blocks null bands are stored (`maxnull_<SRC>`). Both are in
  SCHEMA.md. The test suite then passed: 100 tests, 2 skipped.
- Full run: coordinator `all`, hard cap $8, pause above $6 with stages outstanding. Each step is refused if the
  completed spend plus its projected cost would pass the cap (`PROJECTED` in `measure_b.py`).
- `python measure_b.py validate` checks a fetched `data/r20s0c` against SCHEMA.md.
- **Full run done** (coordinator `fc-01M7G24WNTNDYQGA4TTM9YNFXX`, `runs/b-all.json`): 16:55 to 18:39 Modal time
  (1 h 44 min), no failed unit. **Estimated spend $2.584** in total, including the phase-1 smoke: extraction $0.148,
  sources $1.681, reps $0.739, smoke and timing $0.016.
  - All 8 checkpoints' raw-derived Grams match the stored ones (worst 2.6e-13). The 10,000-image float32 R matches
    the float64 probe R to 5.2e-6 relative.
  - All four source tiers were stored: 20 files, 35.8 MB.
  - All 12 representative blocks x 4 tasks, plus N6, S1 for the 12 final single cells, and N = 10,000 for 6 blocks.
  - No ablation level recorded an error. The validation pass found no problems.
- Known bug, not acted on: the coordinator measures the sources' total size before reloading the volume, so its
  60 MB check lags one tier (it logged 0.0 MB after tier 0). It was harmless here (35.8 MB < 60 MB). Fix: call
  `vol.reload()` before the size glob.

- **Follow-up (schema 1.1)** (coordinator `fc-01M7GBJ3KEV44X8JN6SDP6TRWQ`, 17 min, no failed unit, **$0.571**;
  total now **$3.155**). It answers the site accuracy review:
  - every reps/stability agreement now carries its own N7 references, and every span overlap its chance value;
  - each Markov ablation level gets its own S1 (5 subsets) and N3 excess (10 draws);
  - debiased CKA null; Dirichlet energies in both the M_c and P_c forms; spectrum conventions stated per field;
  - the budget-check reload fix.
  Only the stability / linear / ablate parts, `s1_cells` and `n10k` were recomputed. The parts are versioned
  `_v2`; the 1.0 parts are untouched on the volume. Validation passes.

# Phase 1

2026-11-15. Files: `stats.py`, `sources.py`, `nulls.py`, `variants.py`, `measure_b.py`, `test_stats.py`,
`test_sources.py`, `test_nulls.py`, `test_measure_b.py`, `SCHEMA.md`, this file.

## Status

- Tests: `uv run --with modal --with pytest --with torch --with numpy --with scipy --with scikit-learn pytest
  explore/cifar_baselines` from the repo root with `OMP_NUM_THREADS=2`: 96 passed (53 of them mine, plus B's
  and C's), 2 skipped (mine: the N = 2000 fixture test, which is opt-in via `CM_SLOW=1` locally and ran on Modal
  instead). The command in the brief has no `--with pytest` and fails with "Failed to spawn: pytest";
  without `--with modal`, `test_measure_b.py` skips itself.
- Modal smoke (app `cm-cifar-baselines-a`, deployed once for the smoke; coordinator call
  `fc-01M7FZMY9Q3P9YSAB9NMW0Y4QW`, record in `runs/b-smoke.json`):
  - `extract_raw(64000)`: 12 sites, flat and GAP raw Grams. The centred cosine Gram derived from R matches the
    stored `r20s0b` Gram to **6.0e-14** at worst (assertion 1e-10). Same base image layers as `measure_v2.py`,
    A100-40GB, 21.5 s.
  - `fixture_check`: regenerated `_fixture` activations (seed 41; verified against the fixture's stored layout)
    and M0 reproduces the stored labels at every r (all equal).
  - `timing`: in addition, on the 5 timing blocks, M0 recomputed from the stored operators with
    `schema_v2.block_record` reproduces the stored `r20s0b` labels exactly (share 1.0, all 5 ranks).
  - **Spend: $0.030** (timing-record estimate, rates in `measure_b.py`; cap for the phase was $1.50).
- Not run: the full stages (`extract`, `sources`, `reps`, `assemble`). Code added after the smoke (units
  `s1_cells`, `n10k_unit`, `assemble`, the stored-kappa vMF levels, L3 on every tier) is tested locally on
  synthetic data only. The app must be redeployed before the full run (`python measure_b.py deploy`).

## Measured timings (Modal CPU, 4 cores, N = 2000; `timing.json`)

Per block, all five ranks. The figures were the same for 1 to 12 cells, except where noted.

| item | seconds |
|---|---|
| load the block's Grams and operators | 1 to 7 |
| M0 dense eigensolve (pipeline `block_modes`) / iterative eigsh | 0.7 / 0.07 to 0.6 (agree to min cosine 1 - 1e-15) |
| L1 0.15 to 0.65; L2 0.35 to 0.96; L3 1.6 to 2.6; G1, G1n about 0.3 each; G2 0.4 to 0.7; C2 under 0.15; C3 about 0.1; N3 draw 0.2 to 2.8 | |
| agreement + N7 nulls (1,000 permutations, vs M0 and vs classes, all sources) | 7 to 10 |
| all grid sources for one block | 11 to 16 |
| N2b Grams (1 / 12 cells) | 0.8 / 4.3 |
| N2a Grams (1 / 12 cells) | 0.5 / 5.7 |
| default operators (1 / 12 cells) | 0.3 / 3.4 |
| null spectral summary | 0.5 to 1.3 |
| S1, one 80% subset | 1.1 to 3.1 |
| S2, one fit/test split, 2 ranks | 1.5 to 4.0 |
| N5, one cell (rewire + balance) | about 3 |
| N6, SEBA r = 8 on one Haar subspace | 0.75 |

## Estimated cost and wall time of the full run

| stage | estimate |
|---|---|
| extraction: 8 checkpoints (about $0.014 each), seed 1 with 78 blocks, 10,000-image extraction | about $0.45 |
| sources: 1,056 blocks (8 x 78 layer, 12 x 36 time) x about 16 s, 4-core / 16 GB units | about $1.5 |
| reps: 12 blocks x (nulls, stability, linear, ablate), dominated by the 12-cell block | about $0.8 |
| N6: 1,000 Haar draws x 5 ranks | about $0.3 |
| N = 10,000 (factor I): 6 blocks, 20 dense 10k Sinkhorns at 8 cores / 64 GB | about $1.2 (least certain; not benchmarked) |
| **total** | **about $4.3; I would set the coordinator cap at $8** |

Wall time is about 2 h with the coordinator's order: extract about 5 min; the four source tiers run one after
another (so the 60 MB budget check can stop them) at about 20 min each; then reps, N6, S1-cells and N = 10,000
in parallel, 20 to 40 min. If the tiers ran in parallel the wall time would be under 1 h. Estimated explorer
data (int8 labels, compressed) is about 40 MB for all tiers, under the 60 MB budget.

## Deviations from PLAN, with reasons

1. **Stored results reused for grid sources.** On the grid, M0 labels, psi_1..8 (int16-decoded), single-cell
   labels and spans (C1, C2, C3, G1, G1n inputs) come from `r20s0b` rather than being re-solved. The timing unit
   shows that recomputing M0 from the stored operators reproduces the stored labels exactly, so this saves cost
   without changing anything. G1, G1n and C3 inherit the int16 quantisation (about 3e-5 of each vector's
   maximum). Representative blocks recompute everything in float64.
2. **Eigensolvers.** M0 and single cells use the pipeline's dense Householder solver (bit-for-bit with the
   stored data). Every other source and null uses Lanczos (`scipy eigsh`) on the constant-projected operator.
   It checks the residual (1e-8) and the gap to the (k+1)-th eigenvalue (1e-6), and falls back to dense if
   either check fails. The two solvers agreed to 1e-15 on the timing blocks.
3. **N7 implementation.** The 1,000-permutation bands are computed in vectorised form. Because a permutation
   preserves both marginals, sklearn's expected mutual information is computed once. The test checks every
   draw against `sklearn` AMI and the explicit Hungarian matched Jaccard. The grid bands are per-comparison:
   no selection over blocks or ranks is made on the grid. `stats.perm_null` repeats a max-over-candidates
   selection inside each permutation, for tab statistics that select.
4. **G1n** is NJW rounding on the rows of [psi_0..psi_{r-1}], i.e. r eigenvectors including the constant
   (as NJW uses k eigenvectors for k clusters). G1 (plain) uses psi_1..psi_q.
5. **G2** uses the 0/1 union-15NN graph **without the 1e-6 floor** (the textbook form). The mean of
   degree-normalised matrices has no constant eigenvector, so the span it reports for comparisons is psi_2..psi_r
   projected off the constant and re-orthonormalised. `top_vector_constant_cosine` is recorded.
6. **C2** uses single-cell SEBA labels at the same r as the block. Its eigenvectors of H A_B H come from the SVD
   of the centred stacked one-hot matrix, which is exact (A_B = F F^T / L).
7. **N2a** uses the sample covariance spectrum as the population spectrum: the nonzero eigenvalues of
   (H R H)/N, at most N-1 of them, with no shrinkage and no top-rank-plus-tail model. It is a Gram-only form,
   exact because right rotations do not change the Gram.
8. **N2b** is the reviewed Gram-only form. The reported mismatch is the diagonal spread before D_l, plus
   tr(C*_l C*_m)/tr(C_l C_m) - 1 (= the change in ||Z_l^T Z_m||_F^2) and 1^T C* 1 / 1^T C 1 - 1 after D_l.
   Full d x d moment matrices are not available from Grams.
9. **N3 coherence** comes from one permutation per (draw, cell key) via a stable hash (`zlib.crc32`; Python's
   `hash` of strings is salted per process). In each block the permutations are re-expressed relative to the
   block's first cell, so that cell is fixed and labels are in its frame. The spectra are exactly those of
   "fix any one cell".
10. **N5** makes 10 x |E| swap proposals per cell (rejected loops and multi-edges are counted). It reports the
    share of original edges kept as the decorrelation diagnostic, not a mixing-time argument.
11. **S2** uses one split: 1,500 fit / 500 test, seed 0. The reverse-neighbour rule is strict
    (d(x, z_j) < delta_j). Directions with lambda_P < 0.1 are flagged, not dropped. The extension rows are not
    rebalanced to a doubly stochastic enlarged operator; IDEAS 4.4 states this limitation. The test graph is the
    default rule on the test-test Gram under fit-only centring. L3's frozen extension uses the spherical
    centroids of the block Gram.
12. **Factor A, global vMF:** the two "stored kappas" are the stored rule-3 and rule-4 operators (per-cell
    kappa, the same zero-diagonal fixed-lazy construction), plus global kappa 2 and 8. Matched effective
    neighbour count bisects k (2..400), perplexity (5..1500) and kappa (0.1..300) per cell. The target is the
    block's median over cells of the default's median 1/sum_j P_ij^2. Levels that fail (non-convergent
    Sinkhorn, unreachable perplexity, a zero floor that cannot be balanced) are recorded as `error`, not
    dropped.
13. **Factor C, degree-normalised variants** are compared via the constant-free span of their eigenvectors
    2..r on uniform site weights (agreement and overlap only). No retention is reported for them. Converting
    them to Markov operators with stationary-weighted function coordinates (PLAN section 7) is **not
    implemented**. This is a gap, labelled.
14. **Factor E** `expprod` uses tau/L = 1/2, so that exp(-L_l/2) matches M_l = I - L_l/2 to first order.
    `expsum` has exactly the mean's eigenvectors (an identity, kept as a check and asserted in the tests).
15. **Factor I at N = 10,000** is run only for the representative final-step blocks with at most 3 cells
    (4 single cells, b4-b6, b7-b9), with k = 15 and k = 75. A dense Sinkhorn at 10k costs minutes per cell;
    the 12-cell and 4-cell blocks were left out on cost grounds. Factor I at N in {500, 1000} runs on all 12
    representative blocks.
16. **Extra units** not in the brief's list: `pixels` (I1, both network-input and raw /255 pixels), `s1_cells`
    (S1 for each final single cell, as PLAN section 4 asks) and `assemble`.
17. **Representative blocks**: as in PLAN section 3, no substitutions.
18. **"Deploy"**: the Modal app was deployed for the smoke, as the coordinator requires. Nothing else was
    deployed, committed or pushed.

## Where I think the plan is weak or wrong (for the orchestrator)

- **N7 on the grid versus selection.** The explorer will let a reader scan 1,056 blocks x 5 ranks x about 12
  sources. Per-comparison 95th percentiles invite exactly the selection the plan forbids. The page should
  either show those numbers only for the block in view, or quote a max-over-blocks band, which `perm_null`
  can compute for any declared family.
- **AMI with "unassigned" as a category** rewards two sources for agreeing on which points are unassigned.
  `ami_joint` with coverage is the cleaner number. Both are stored; the site should show both, or the joint one.
- **The N3 class AMI is not a zero-reference.** The first cell is fixed, so its class structure survives.
  IDEAS B4 notes this; captions must give its band, not 0.
- **The C2 rank coupling** (single-cell sets at the same r) is a choice. A block at r = 8 built from r = 8
  single-cell partitions is a different object from one built from r = 3 partitions. The plan does not say
  which is intended.
- **The retention size null** t + (1-t)(m-1)/(N-1) is exact for any stochastic P. The plan's wording
  suggests it holds only for zero-diagonal or constant-diagonal P; the formula itself is right.
