# `explore/cifar_coherent/data/r20s0c/`: baselines, nulls and stability (schema for the site builder)

**Schema version 1.1** (`meta_c.json` `schema_version`). 1.1 adds fields only; every 1.0 field keeps its meaning.
The additions are listed in section 8.

Produced by `explore/cifar_baselines/measure_b.py` (Modal volume `cm-cifar-coherent`, public files under
`/analysis/r20s0c`, fetched with `python measure_b.py fetch`). Companion to `data/r20s0b/` (schema v2, `SPEC.md`
section 2), whose conventions are reused unchanged: `STEPS`, `SITES`, `layer_pairs` (78, row-major (a, b), a <= b),
`time_pairs` (36), `seba_r = [2, 3, 4, 6, 8]`, N = 2000 probe images in `probe.npz` order (point i is the same
image in every file), label -1 = unassigned, labels 0..r-1. Everything here uses **kernel rule 0** (union 15-NN
graph) only. Nothing in this directory is a verdict: each number is a measurement with its null value next to it.

Status tags (one per colour source; show them on the "Sets from" control):
**S** label-free dot-product spectral object; **P** post-processing, diagnostic or null generator; **B** fitted
baseline outside the construction (k-means and spherical k-means are fits); **V** uses class labels (validation, or
a null whose strata are classes).

## 1. Colour sources

| id | label in the control | status | where the labels are |
|---|---|---|---|
| M0 | the method (default) | S+P | `r20s0b` `layer_<step>.npz` / `time_<site>.npz`, `seba_label[0]` (not duplicated) |
| L1 | cosine PCA + SEBA | S+P | `sources_*.npz` key `L1` |
| L2 | cosine PCA + k-means | B | key `L2` |
| L3 | spherical k-means (exact, block Gram) | B | key `L3` (every stored tier; the timing benchmark gave about 2 s per block for all five ranks) |
| G1 | method eigenvectors + k-means | S+B | key `G1` |
| G1n | method eigenvectors, row-normalised + k-means (NJW) | S+B | key `G1n` |
| G2 | textbook spectral clustering (kNN, D^-1/2 A D^-1/2, no Sinkhorn) | S+B | key `G2` |
| C1first / C1mid / C1last | first / middle / last cell of the block alone | S+P | pointer, see section 3 |
| C2 | co-assignment consensus of single-cell sets | P | key `C2` |
| C3 | mean-projector consensus of single-cell spans | S+P | key `C3` |
| I1 | pixels (network input; one cell, same colours on every block) | S+P | `I1_pixels.npz` key `M0_input` (also `L1_input`, `M0_u8`, `L1_u8` = raw 0..1 pixels) |
| I2 | untrained network (step 0, same block) | S+P | pointer, see section 3 (layer blocks only) |
| I3 | other training seed (seed 1, final checkpoint) | S+P | `I3_seed1_layer_64000.npz`, layer blocks at step 64000 only |
| Y1 / Y2 | classes / predicted class | V | `r20s0b/probe.npz` `labels`; `r20s0b/points_<step>.npz` `pred` |
| N3 | one null draw: cells' image correspondence shuffled (seed 0) | P | key `N3` |
| N2a, N2b, N4, N5 draws | one stored draw each, representative blocks only | P (N4: P+V) | `rep_<id>_nulls_draws.npz` |

For a source, the number of sets is k_c = r and r is the explorer's rank selector (no "best r" is stored).

## 2. Per-file arrays

### `sources_layer_<step>.npz` (P = 78) and `sources_time_<site>.npz` (P = 36)

| key | shape, dtype | meaning |
|---|---|---|
| `<SRC>` for SRC in L1, L2, L3, G1, G1n, G2, C2, C3, N3 | (P, 5, N) i1 | hard labels per block and rank index (rank = seba_r[ri]) |
| `agree_<SRC>` for every source above and for M0, C1first, C1mid, C1last, I1, I2 (layer files), I3 (step 64000 layer file) | (P, 5, F) f4 | agreement row, fields in order below |
| `maxnull_<SRC>` for the same sources | (5, M, 2) f4 | max-over-blocks null band: [mean, q95] of the max over this file's P blocks, per rank and field in `max_fields` order (below) |

**Conventions.**
- *Set counts are declared*: every source at rank r has exactly r sets (labels 0..r-1, -1 unassigned) and classes
  have 10. A declared set that receives no point is an **empty set**: it counts in `max(k_a, k_b)` and has Jaccard 0
  with everything. All numbers are therefore invariant to how sets (empty ones included) are numbered.
- *Missing values*: in `.npz` files NaN means "not applicable" (for example no span to compare, or fewer than two
  jointly assigned points). In every JSON file non-finite values are written as `null` (files are written with
  `allow_nan=False`, so `JSON.parse` accepts them); sizes and failure diagnostics are kept.

Agreement fields (F = 21, in this order; also in `meta_c.json` `agree_fields`). "vs M0" = against the method's labels
for the same block and r; "class" = against CIFAR classes (validation only).

**Headline agreement** (what the side card shows first): `ami_joint` together with `joint_coverage`, and its band
`ami_joint_null_q95`; for classes `class_ami_joint` with coverage 1 - `unassigned` and `class_ami_joint_null_q95`.
`ami_all` is stored next to it; it treats "unassigned" as one more category and so also rewards two sources for
agreeing on *which* points are unassigned.

| field | definition |
|---|---|
| `ami_all` | AMI(M0, source), "unassigned" as one more category (sklearn, arithmetic normalisation) |
| `ami_joint` | **headline**: AMI on the points assigned by both |
| `joint_coverage` | share of points assigned by both (always shown with `ami_joint`) |
| `ari_all` | ARI, unassigned as a category |
| `mj_max` | Hungarian-matched Jaccard, sum over matched pairs / max(r, r) (declared counts); unassigned in no set |
| `mj_legacy` | the explorer's old number: mean J over matched pairs |
| `mj_weighted` | matched J weighted by the two sets' sizes |
| `unassigned` | share of the source's points that are unassigned |
| `ami_null_q95` | 95th percentile of `ami_all` under N7 (1,000 size-preserving permutations of the source labels) |
| `mj_null_mean`, `mj_null_q95` | mean and 95th percentile of `mj_max` under the same N7 permutations |
| `overlap_mean_cos2`, `overlap_min_cos` | principal angles between M0's psi_1..psi_q and the source's q-dimensional span, constant removed (NaN for sources with no span: k-means, pointers); chance mean cos^2 = q / (N-1) |
| `class_ami`, `class_ami_null_q95` | AMI(source, classes), unassigned as a category, and its N7 95th percentile (class labels permuted) |
| `class_mj_max`, `class_mj_null_mean`, `class_mj_null_q95` | matched Jaccard with classes (denominator max(r, 10)) and its N7 mean / 95th percentile |
| `ami_joint_null_q95` | 95th percentile of `ami_joint` when the source's labels are permuted *within the jointly assigned points* (conditional on the coverage and both marginals there) |
| `class_ami_joint`, `class_ami_joint_null_q95` | AMI(source, classes) on the points the source assigns, and its 95th percentile with classes permuted within those points |

**Two kinds of null band.** The per-comparison percentiles above calibrate one block and one rank. A reader who
scans many blocks needs the band for the *largest* value in the family. `maxnull_<SRC>` gives it, with the family =
all P blocks of the file at a fixed source and rank. `max_fields` = [`ami_all`, `ami_joint`, `mj_max`,
`class_ami`, `class_ami_joint`, `class_mj_max`].
- For `ami_all`, `mj_max`, `class_ami` and `class_mj_max`, permutation t is the *same* image permutation in every
  block of the file (coherent), so the band is that of the max under one shared null draw.
- For the two jointly-assigned AMIs, the permutation is drawn within each block's own jointly assigned points,
  independently per block. That band is the max over independent blocks; for positively dependent blocks (nested
  layer blocks) it is no lower than the coherent one.

Show the max band whenever a statistic is quoted for a block chosen by scanning (a largest value, a highlighted
cell of a heat map). The representative-block tables in `reps.json` repeat any selection inside the permutation.

Suggested caption pattern (no interpretation): "AMI with the method on jointly assigned points 0.62 (coverage 0.91;
permutation 95th percentile 0.01; largest-over-blocks band 0.02); matched Jaccard 0.48 (permutation mean 0.21)".

## 3. Pointers (no new label data)

- **C1first / C1mid / C1last** of a layer block (a, b) at a step: the single-cell blocks (a, a), (m, m), (b, b) with
  m = floor((a + b) / 2), in the same `r20s0b/layer_<step>.npz`; pair index = `layer_pairs.index([s, s])`. For a time
  block (c, d) at a site: (c, c), (m, m), (d, d) in `r20s0b/time_<site>.npz`.
- **I2** of a layer block (a, b) at any step: the same pair index in `r20s0b/layer_0.npz`. Not defined for time blocks.
- Their agreement rows are stored (`agree_C1first`, ..., `agree_I2`) so the side card needs no computation.

## 4. Scope tiers (what is stored where)

Explorer-grid sources (L1, L2, L3, G1, G1n, G2, C2, C3, N3, pointers, I1, I3) are computed in this order until the
explorer data budget (60 MB added) is reached; `meta_c.json` `tiers_done` lists what exists:

| tier | blocks | files |
|---|---|---|
| 0 | all 78 layer blocks at step 64000 | `sources_layer_64000.npz` |
| 1 | all 36 time blocks at b5, b9, logits | `sources_time_{b5,b9,logits}.npz` |
| 2 | layer blocks at steps 0, 1024, 6400 | `sources_layer_{0,1024,6400}.npz` |
| 3 | everything else | remaining `sources_*.npz` |

A block outside the stored tiers shows only M0, the pointers and Y1/Y2; the control greys out the other sources.

## 5. Representative blocks (the "Baselines and nulls" tab)

Fixed in advance (PLAN section 3): ids `cell_stem`, `cell_b5`, `cell_b9`, `cell_logits` (single cells, step 64000);
`stem-b3`, `b4-b6`, `b7-b9`, `stem-logits` (layer blocks, step 64000); `b4-b6@0`, `b4-b6@1024`; `time-b5`, `time-b9`
(checkpoints 1024..64000). `meta_c.json` `reps` gives each id's family, step/site and (a, b).

`reps.json` (assembled from the per-unit parts): `{id: {nulls, stability, linear, ablate}}` with

- `nulls.real`: `lambda_P` (non-trivial spectrum of the non-lazy block operator Pbar, psi_1..psi_8), `gaps`,
  `lambda_P_bottom` (3 smallest eigenvalues of Pbar; near -1 = alternation), `laziness` and `spectrum_of` (the
  convention: `half` = the operators are M = (I + P)/2, so lambda_P = 2 lambda_M - 1 and Pbar = 2 Mbar - I; `none` =
  the operator is already non-lazy and is reported natively, used for factor D's retained diagonal), `participation`, and per r in `seba`: `sizes`, `unassigned`, `effective_points` (span
  leverage concentration), `clipped_negative_mass`, `raw_min`, `class_ami`.
- `nulls.N2a|N2b|N3|N4|N5`: a list of draws (20; N5 10), each with the same fields plus `agreement_with_M0` per r;
  N2b draws add `moment_mismatch` (diagonal spread before renormalisation; relative change of tr(C_l C_m) and of
  1^T C 1 after); N5 draws add `rewire` (edges, accepted swaps, share of original edges kept).
  `nulls.bands[null].lambda_P[j]` = {mean, sd, q05, q50, q95} over draws. N3 draws are coherent: draw t uses the same
  per-cell permutation in every block and rank.
- `stability.S1`: 10 subsets of 80%, per r agreement with full-set M0 on shared images and subspace overlap.
  `stability.S2`: fit 1,500 / test 500 (fixed seed 0), for M0, L1, L3 per r: `test_retention` / `test_null` per set
  (retention on a graph built from the test images only, mean over cells, with its size null), agreement of the
  frozen extension with the same source refit on all 2,000 (test images), `flagged_small_lambda`.
  `stability.S3`: H(sets at r | sets at r') in nats for r < r' (0 = refinement). `stability.LOCO[r]`: per omitted
  cell the retention of the remaining cells' sets on it with its size null; `next` = the block's sets on the first
  cell after the block. `stability.N6_rotation[r]`: 20 random rotations of [psi_0..psi_{r-1}], agreement with the
  unrotated SEBA labels.
- `linear`: `linear_content[r]` (variance capture of M0's span in C_B and per cell, with the rank-q optimum), `cka`,
  `cka_debiased` (cell x cell), `cka_perm_null` (first vs last cell), `pca_C_vs_HCH`, `pca_dirichlet_under_cells` and
  `method_dirichlet_under_cells` (mean / max / q90 over cells of the per-cell Dirichlet energy),
  `subspace_overlap_cells` (cell x cell, r = 3, 6).
- `ablate.A..I`: one factor at a time (PLAN section 5); each level has a spectral summary (as `nulls.real`),
  `agreement_with_M0[r]` and `overlap_with_M0[r]`, or `error` if that level could not be computed (for example a
  floor of 0 where balancing fails); `A.matched_effective_neighbours` gives the per-cell parameters matching the
  default's median effective neighbour count. `C.*` (degree / raw normalisations) report `stationary` drift and the
  top vector's constant cosine; `raw_adjacency` has no retention by construction. Every level that reports a bare
  spectrum (`vals`) also states `spectrum_of` (for example the singular values of a constant-projected product);
  levels scored through the spectral summary carry its `laziness` field.

`n6_haar.json`: per r, SEBA on 1,000 Haar random subspaces (N = 2000): bands of `unassigned`, `min_size`,
`clipped_negative_mass`, and the count of SEBA failures.

`stability_extra.json`: `s1_cells_64000` (S1 for each of the 12 single cells at the final step, same fields as
`stability.S1`) and `n10k[id]` (factor I at N = 10,000 for the representative blocks of at most 3 cells at the final
step: default construction with k = 15 and k = 75, `lambda_M`, and per r the agreement with the probe-set M0 on the
2,000 probe images and the subspace overlap restricted to them).

`rep_<id>_nulls_draws.npz`: keys `N2a`, `N2b`, `N3`, `N4`, `N5`, each (5, N) i1 (draw 0), for the "null colouring"
option on the representative blocks.

## 6. Raw Grams (work area, not fetched to the site)

Under `/work/r20s0c` on the volume (inputs for builders B and C):

- `raw_flat_<step>_<si>.npy`, `raw_gap_<step>_<si>.npy`: (N, N) float64, R = X X^T of the flattened (flat) or
  spatially averaged (gap) activations, probe order; for pool and logits the two are identical. The centred cosine
  Gram derived from R matches the stored `r20s0b` Gram (recorded per cell in `checks_extract_raw_s0_<step>.json`).
- `raw_pixels_input.npy` (network input after normalisation), `raw_pixels_u8.npy` (pixels / 255).
- `s1_raw_{flat,gap}_64000_<si>.npy`: seed 1, final checkpoint.
- `raw10k_flat_64000_<si>.npy`: (10000, 10000) **float32** (stated precision limit: about 1e-7 relative per entry,
  fine for kNN graphs and covariances, not for bitwise comparison with the float64 probe Grams); CIFAR-10 test-set
  order. `raw10k_index.npz`: `test_index` (0..9999), `probe_positions` (the 2,000 probe images' rows, = `probe.npz`
  `idx`), `labels`. `checks_extract_10k_64000.json` gives the relative max difference to the float64 probe R.

Derive views with `stats.raw_to_views(R, fit_idx)` (fit-only centring for held-out images).

## 7. `meta_c.json`

`tag`, `based_on` ("r20s0b"), `kernel_rule` (0), `seba_r`, `sources` ({id: status}), `agree_fields`, `n_perm`
(1000), `tiers_done`, `reps` (id -> family, step or site, a, b), `timing` (pointer to `timing.json`), and
`notes` (deviations from the plan, copied from `NOTES_A.md`).

## 8. Schema 1.1 additions (2026-11-15) and conventions made explicit

**Agreement with its own null, everywhere a hard-set agreement is reported** in `reps.json` and
`stability_extra.json`. The paths are:
- `stability.S1[t].per_r[r]`, `stability.S2.{M0,L1,L3}[r]`, `stability.N6_rotation[r][t]`
- every ablation level's `agreement_with_M0[r]` (factors A, B, C, D, E, F, G[`seba_c*`][r], H, I) and each level's
  `S1[t].per_r[r]`
- `stability_extra.s1_cells_64000[site][t].per_r[r]` and `stability_extra.n10k[id][k15|k75].per_r[r]`

Each such dict holds the `stats.agreement` fields (`ami_all`, `ami_joint`, `joint_coverage`, `ari_all`, `ari_joint`,
`mj_max`, `mj_legacy`, `mj_weighted`, `unassigned_a`, `unassigned_b`) plus these new fields:

| field | meaning |
|---|---|
| `ami_null_mean`, `ami_null_q95` | `ami_all` under 1,000 size-preserving permutations of the second labelling |
| `mj_null_mean`, `mj_null_q95` | `mj_max` (declared counts r, whole Hungarian matching repeated) under the same permutations |
| `ami_joint_null_mean`, `ami_joint_null_q95` | `ami_joint` with the second labelling permuted within the jointly assigned points |
| `n_perm` | 1000 |

The conventions are those of the grid rows: declared set counts and the empty-set rule of section 2.

**Chance values for span overlaps.**
- Ablation levels: `overlap_with_M0[r]`, `right_overlap[r]` and `left_overlap[r]` are now `{mean_cos2, chance}`,
  with chance = (r-1)/(N-1) over the rows compared. In 1.0 they were bare numbers; read `.mean_cos2`.
- S1 rows have `overlap` plus `overlap_chance`; N = 10,000 rows have `overlap_on_probe` plus `overlap_chance`.

**Per-level stability and null excess** (PLAN section 5), on every Markov ablation level: A (all levels, including
`matched_effective_neighbours.<family>.summary` and `vmf_stored_*`), B floors, D `retained_diagonal`, H.
- `S1`: 5 subsets of 80%, the same subsets for every level (seed 0). The level is rebuilt on each subset and
  compared with the level's *own* full-set labels on the shared images. Each subset row has `lambda_P`, and per r
  the agreement with its null and `overlap` `{mean_cos2, chance}`. Per-cell parameters found on the full set are
  kept fixed on the subsets: the matched k / perplexity / kappa, and the stored rule-3/4 kappas, which are listed in
  `vmf_stored_*.kappa`.
- `n3`: `lambda_P_band[j]` = {mean, sd, q05, q50, q95} of lambda_P over 10 N3 draws of the level's own operators,
  and `excess_over_q95[j]` = real lambda_P[j] - band q95. It is `{undefined: ...}` for single-cell blocks.
- `ablate.ablations.conventions` restates these rules.

**CKA nulls** (`linear`):
- `cka_perm_null` is the **biased** (ordinary) CKA under 100 joint row/column permutations of the last cell; it is
  kept for 1.0 readers, and `cka_perm_null_biased` is the same object.
- `cka_debiased_perm_null` is the same permutation reference for the debiased (U-centred) CKA.
- `cka_perm_null_pair` = [0, L-1], the cell pair both refer to.

**Dirichlet energies** (`linear`):
- `pca_dirichlet_under_cells` and `method_dirichlet_under_cells` are computed with the **lazy** M_c:
  f^T (I - M_c) f / f^T f, which is half of f^T (I - P_c) f / f^T f.
- The `_P` versions of both keys give the P_c form (twice the value). `dirichlet_convention` states this.

**Spectrum conventions per field:**
- **Spectral summaries** (`nulls.real`, null draws, ablation levels scored through the summary): `lambda_P` is the
  non-lazy spectrum, with `laziness` saying how it was obtained (section 5).
- **Factor C** (`degree_alpha*`, `raw_adjacency`): `vals` are eigenvalues of the mean of (I + S_alpha)/2, or of
  K/||K||_2. `vals[0]` **is the leading vector's eigenvalue, kept**; it is not the constant, which is not an
  eigenvector here. Flag: `vals_includes_leading_vector: true`.
- **Factor F** (`interval`, `log`): `vals` are **lambda_M** of the weighted mean of the lazy M (constant
  removed), `vals_convention: "lambda_M (lazy M)"`. `lambda_P` (= 2 vals - 1) is stored alongside.
  `drop_every_other` and `repeat_last` carry overlaps and agreements only.
- **Factor D** `product_gamma*` and **factor E** `product`, `twoend`, `expprod`: `vals` are singular values of the
  constant-projected product.
- **Factor E** `meansq`, `expsum`: eigenvalues of that operator, constant removed. `concat`, `union`,
  `intersection`: lambda_M of the single combined operator, constant removed.
- Every one of these carries a `spectrum_of` string.

Not recomputed in 1.1: the `nulls` task (its draws' `agreement_with_M0` has no own null; those draws are null draws
themselves, compared with M0) and the explorer-grid `sources_*.npz`, which already carried their nulls.
