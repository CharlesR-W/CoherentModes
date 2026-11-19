# Review 2: builder A's code (GPT adversarial code reviewer, 2026-11-15)

The orchestrator recorded the findings from the reviewer's hand-back; the reviewer's session was read-only and
reproduced each finding. Fixes are by builder A (Opus 5.5).

## Findings and fixes

1. **[P1] NaN in public JSON.** `reps_unit` serialised with `allow_nan=True`. An empty held-out cluster
   (retention of an empty set, `nanmean` over cells) therefore wrote bare `NaN` into `reps.json`, and browser
   `JSON.parse` rejects that. The stability-test configuration (n = 120, two cells, separation 2) already produced
   14 NaN tokens.
   *Fix.* `stats.finite_json` recursively converts numpy types to Python and every non-finite float to `null`,
   keeping sizes, counts and error strings. `stats.dumps` serialises with `allow_nan=False`. Every JSON file that
   `measure_b.py` writes now goes through `_dumps` → `stats.dumps`: timing records, checks, rep parts, `reps.json`,
   `n6_haar.json`, `stability_extra.json`, `meta_c.json`, `timing.json` and the fixture report.
   *Tests.*
   - `test_measure_b.py::test_run_rep_tasks[nulls|stability|linear]` serialise the reviewer's configuration.
     They assert there is no `NaN`/`Infinity` token, that a strict `json.loads` (rejecting constants) succeeds,
     and, for `stability`, that `null` occurs, so the empty-set case is exercised.
   - `test_stats.py::test_retention_internal_empty_set_and_finite_json` checks an empty declared set → `null`,
     with its size 0 kept.

2. **[P2] Laziness assumed in `spectral_summary`.** `spectral_summary` treated every input as M = (I + P)/2. But
   factor D's `retained_diagonal` level (`variants.operator(..., diagonal='retained')`) returns the balanced
   operator with no lazy step. Its reported lambda_P, gaps and bottom spectrum were therefore those of 2M - I
   (e.g. an actual bottom eigenvalue of -0.249 reported as -1.498).
   *Fix.* `nulls.spectral_summary(..., laziness='half'|'none')` takes an explicit argument, validated, and records
   `laziness` and `spectrum_of` in its output. With `none`, the spectrum and Pbar are the operator's own. Factor D
   `retained_diagonal` now passes `laziness='none'`.
   *Audit of every other level.*
   - These are all M = (I + P)/2 and use `half`: A levels (via `variants.operator`, default gamma 0.5), the stored
     rule-3/4 operators (fixed-lazy), B floors, H views, the N2a/N2b/N3/N4/N5 nulls, the S1 subsets and the
     default.
   - These report bare spectra not passed through the summary, and now state `spectrum_of`: C degree/raw
     (eigenvalues of the mean of (I + S)/2 or K/||K||, top vector kept), D product-gamma and E
     product/twoend/expprod (singular values of the constant-projected product), and E meansq/expsum/concat/
     union/intersection.
   *Test.* `test_nulls.py::test_spectral_summary_laziness_matches_direct_eigendecomposition` compares the reported
   lambda_P and bottom spectrum with a direct `eigvalsh`, for the lazy default and for the retained-diagonal
   operator. It also checks that an unknown convention raises.

3. **[P2] Set counts inferred from labels.** `contingency`, `matched_jaccard_detail` and `fast_ami_null` used
   `max(label) + 1` as the number of sets. Matched Jaccard (padded by max(k_a, k_b)) and its null then depended on
   label numbering whenever a set was empty: labels {0, 1} and {0, 2} for the same partition gave 1 and 2/3.
   *Fix.* Explicit set counts `ka`, `kb` are threaded through `contingency`, `matched_jaccard[_detail]`,
   `agreement`, `retention(..., k)` and `fast_ami_null` (observed value and every permutation draw). Labels
   >= the declared count raise. Every caller in builder A's code passes them: r for a source at rank r, 10 for
   classes (`measure_b.N_CLASSES`). The `None` fallback (max + 1) remains only for external callers and is
   documented as numbering-dependent. SCHEMA.md documents the convention: a declared set that receives no point
   is an empty set, counts in the denominator, and has Jaccard 0.
   *Tests.*
   - `test_matched_jaccard_empty_set_position_does_not_matter` covers internal vs trailing empty sets for all
     three summaries, the reviewer's case (both numberings give 2/3), relabelling of both sides, and
     out-of-range labels raising.
   - `test_fast_null_declared_counts_and_relabelling` checks that permutation draws are identical under the two
     numberings and exact against `matched_jaccard` per draw.

## Also adopted (from builder A's phase-1 notes, at the orchestrator's request)

- Both AMI variants are stored. The headline is the jointly-assigned AMI (`ami_joint`, `class_ami_joint`), always
  shown with its coverage. Its N7 band is new: a permutation within the jointly assigned points
  (`ami_joint_null_q95`, `class_ami_joint_null_q95`).
- Max-over-blocks null bands are stored next to the per-comparison percentiles: `maxnull_<SRC>` (5 ranks x 6
  fields x [mean, q95]). The family is all blocks of one file at a fixed source and rank. Permutations are coherent
  across blocks for the all-points statistics. For the jointly-assigned AMIs they are independent per block, which
  SCHEMA.md states.

## Bug found while fixing

`fast_ami_null` reused the names `ka, kb` for the number of distinct observed labels, which shadowed the new
parameters. The new tests caught it and it is fixed (`ua_n, ub_n`).

Test suite after the fixes: 100 passed, 2 skipped (all builders' tests in `explore/cifar_baselines`).
