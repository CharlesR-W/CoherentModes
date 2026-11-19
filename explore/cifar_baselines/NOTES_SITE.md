# NOTES_SITE: the site for the baselines study (2026-11-15)

Nothing committed, pushed or deployed. Inputs read-only; nothing in `explore/cifar_baselines/` changed except this file.

## Build

    cd explore/cifar_coherent
    OMP_NUM_THREADS=1 uv run --with numpy --with pillow --with scipy --with scikit-learn \
        python site/build_site.py r20s0b --baselines r20s0c
    python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/site

`--baselines r20s0c` runs `site/baselines.py` (new). It imports `explore/cifar_baselines/stats.py` unchanged.
First run about 5 min (3 worker processes for the computed agreement rows, see below), later runs about 20 s.
Computations are cached in `~/.cache/cm-site-bn/` (key: input sizes and mtimes). `--html-only` works as before.
Without `--baselines` the tab shows placeholders and the "Sets from" control is hidden.

Files touched: `site/baselines.py` (new), `site/build_site.py` (flag, call, `{{CONTENT_BASELINES}}`),
`site/index.html` (tab), `site/explorer.html` (control + "?" help), `site/js/sources.js` (new: registry, availability,
loading), `site/js/baselines.js` (new: tab figures), `site/js/data.js`, `site/js/explorer.js`, `site/js/main.js`,
`site/js/live.js` (dispatch `bn-*`), `site/css/site.css`, `site/content/baselines.html` (new),
`site/content/method.html` (new section `#method/baselines-defs`; the two kernel-rule tables wrapped in
`.table-scroll`), `site/DESIGN.md`.

## Sizes

Added data 41.6 MB (`bn.site_mb`, after review round 1 and schema 1.1): `d/src/` 35.0 MB of per-block chunks (1,056 blocks, about 33 kB each, gzip),
`d/bn/` about 5 MB of JSON. Largest single fetch in the explorer: one chunk plus, for pointer sources, one existing
block chunk (< 100 kB). Largest tab fetch: `d/bn/reps.json` 0.85 MB, `matrix.json` 0.75 MB; the distribution chart
loads one file per checkpoint or site (`d/bn/dist/*.json`, about 130 kB each).

## Where every number comes from

Explorer, "Sets from" (kernel rule 0, average operator only; other rules / the product operator disable every
source but the method, with the reason in the menu):

| source | labels | agreement row (side card) |
|---|---|---|
| L1 L2 L3 G1 G1n G2 C2 C3 N3 | `sources_*.npz` key | `agree_<SRC>`, `maxnull_<SRC>` |
| C1first/mid/last, I2 | existing `r20s0b` block chunks (mode 0), SCHEMA section 3 pointers | `agree_C1*`, `agree_I2` |
| I1 | `I1_pixels.npz` `M0_input` | `agree_I1` |
| I3 | `I3_seed1_layer_64000.npz` `seba_label` | `agree_I3` (layer blocks, step 64000) |
| Y1 classes | `probe.npz` | the M0 row's class fields (`class_ami_joint`, coverage = 1 - `unassigned`, `class_ami_joint_null_q95`, `class_mj_max`, `class_mj_null_mean`; band `maxnull_M0` class fields) |
| Y2 predicted class | `points_<step>.npz` `pred` (block's checkpoint; time block: its last checkpoint) | **computed by the site** (`stats.agreement` + `stats.fast_ami_null`, 200 permutations; max-over-blocks bands from independent per-block permutations) |
| N2a N2b N4 N5 (12 representative blocks) | `rep_<id>_nulls_draws.npz` (draw 0) | **computed by the site** the same way (200 permutations; no max-over-blocks band) |

Check: the site's computation of an agreement row reproduces a stored one (L1, block b4-b6 at 64000, r = 4) to the
third decimal in every field. Colours of a source's sets are matched to the method's sets by Hungarian matching on
Jaccard (display only, stated in the caption). Thumbnails: SEBA-ordered representatives for pointer sources; a fixed
pseudo-random order of members for label-only sources (stated).

Tab, by section: (a) `reps.json` `nulls` (real, bands, draws) and the grid M0 row's class N7 value;
(b) matrix **computed by the site** from the stored labels of every source on the 12 blocks (`quick_agree`: joint
AMI, coverage, matched Jaccard, 200-permutation joint-AMI 95th percentile, and, since review round 1, the
200-permutation mean and 95th percentile of matched Jaccard, column source permuted, sizes kept); distributions from the grid
`agree_*`/`maxnull_*`; (c) `n6_haar.json`, `reps.json` `stability.N6_rotation`; (d) `stability.S1/S2/S3/LOCO`,
`stability_extra.json`; (e) `ablate.ablations`; (f) `linear` plus the snapshot-by-snapshot set agreement
**computed by the site** from the single-snapshot M0 labels (r = 3, 6; 200 permutations); (g) `out/bench/summary.json`
streamed, filtered to the rows of the NOTES_B chart spec (3,387 rows); (h) `out/transfer/curve_p8.json`,
`synthetic.json` (p8 controls, chain), `real/transfer_pilot.json`, `real/transfer_pilot_10k.json`. Prose numbers are
`{{fact:bn.*}}` facts set by `baselines.py`: the worked example (`bn.ex.*`, stored grid row of cosine PCA + SEBA on
b4..b6 at step 64000, r = 4, and its `maxnull_L1` band), the joint second-order surrogate mismatch (`bn.n2b_mm_*`,
max over draws of |`second_moment_rel_change_max`|), and the planted benchmarks' run-to-run numbers (`bn.rr.*`)
from `site/content/bn_facts.json`, which copies them from NOTES_B.md ("Cross-host reproducibility") and cites it,
because no data file stores them.

## Left out, and why

- Transfer subspace overlaps print the chance value q/(N-1).
- Factor G k-means grid (class AMI only, no agreement with the default): not shown.
- Transfer-pilot SEBA sets and joint tables: not shown in the tab or the explorer (the page says so). PLAN section 6
  asked for them as an explorer source for the 12 pairs; the orchestrator's source list did not, and they are
  transductive summaries.
- Transfer pilot fit-only ("training-pairing disruption") bands and the per-split selected settings: stored, not
  charted.
- Second-wave items (NMF, Leiden), N1 sweeps beyond benchmark P0: not in the data.

## Deviations and builder's choices

- N3 on single-snapshot blocks equals the method (the fixed snapshot is the block); stated in the explorer
  definition and under the distribution chart.
- Bench chart P8 "span chance" uses the stored `contrast_chance` row (M0, r = 3; = q/(N-1)).
- Composition charts use the requested basis factor (x1/4..x4) as the x axis, with actual dimensions in the labels
  and caption, because Markov and broken calibration chains have different actual dimensions.
- Pre-existing 390 px overflow fixed on the Method tab (kernel tables, the kernel figure's buttons).

## Review round 1 (GPT-6 Astra accuracy, GPT-6.1 Terra fresh eyes), changes

- Dirichlet energy relabelled: the stored values are f^T(I - M_c)f with the lazy walk (= half of f^T(I - P_c)f);
  chart and Method definition now say so.
- One median helper (mean of the middle two for even counts) for every median in `js/baselines.js`.
- Ablation leading values carry their own convention (lambda_P, lambda_M, sigma_1, "top eig., top vector kept", ...).
- P8 transfer benchmark: each estimator's own permutation 95% point (dashed, thin) instead of the Nystrom band.
- Null qualifications (spectrum in expectation before preprocessing; second moments before the diagonal restoration,
  with the stored change quoted) in the tab table, the explorer definitions and the Method tab.
- Wording of the null-band callout, the transfer-sets sentence and the forced-extraction sentence; composition
  caption lists only what is plotted, says the parts' norms do not add, and names the x axis as the requested factor.
- Method: VAMP-2 defined as the squared Frobenius norm of the regularised test-whitened cross-covariance.
- Side card: sizes, class make-up and images first, then "overlap with the default construction's sets" with a
  two-line reminder of what AMI, coverage and matched Jaccard measure; the scanned-blocks reference reworded.
- Tab: a worked example near the top; the representative-block list and the ablation factor inventory collapsed;
  scroll cue (`markScrolls`, as in `explore/ncs_report`) on wide tables and figures; wider labels in the scatter.

## Second pass (schema 1.1 of `data/r20s0c`, SCHEMA.md section 8)

- Own permutation nulls now shown for S1 (dashed line), S2 ("perm. 95%"), the rotation test (last row), every
  ablation level ("perm. 95%", matched-Jaccard permutation mean), S1 per snapshot and the 10,000-image table.
- Ablation overlaps read in the new `{mean_cos2, chance}` shape, chance printed beside each value; per-level S1
  (median joint AMI of 5 subsets, largest null 95th percentile) and N3 excess (lambda_P1 minus the 95th percentile of
  10 shuffle draws) drawn where they exist, "n/a" for levels that are not Markov operators and for one-snapshot shuffles.
- Factor C leading values labelled "leading eig., leading vector kept" (`vals_includes_leading_vector`); factor F
  now reads the stored `lambda_P`.
- Debiased CKA shown against `cka_debiased_perm_null` (pair from `cka_perm_null_pair`); the biased value is quoted
  separately.
- Dirichlet energies stay in the lazy convention (`dirichlet_convention`), labelled as such.

## Re-check fixes

- Transfer held-out stability (`transfer_pilot.heldout_stability`: mean cos^2 of two splits' leading q-dimensional
  spans on the n_ij images outside both fits, averaged over 45 split pairs) now prints its chance, the mean over
  pairs of q/(n_ij - 1). n_ij (471 to 525) comes from reproducing `predeclared_splits` in `baselines.py` (seed read
  from `transfer_pilot.py`, which is not imported); q from the stored `stability.*.q`.
- Ablation S1 column now shows the median joint coverage and its range over the 5 subsets; the two agreement columns
  are headed "vs default" and "S1, vs own full set". The single-snapshot S1 chart gained a coverage view; the other
  S1 / S2 / rotation / 10,000-image displays already showed coverage.
- The transfer support caption uses the Method tab's VAMP-2 definition.

## Checks run

Build without error; browser (local Chrome, `playwright-cli`) at 1440 x 900 and 390 x 844: every tab scrolled to
render all live figures, zero console errors, zero page errors, no KaTeX errors, no horizontal document overflow;
explorer with 22 sources on a layer block, 11 on a time block, others on a 12-snapshot block and a second time block,
the kernel-rule and product-operator fallbacks, and the default tour. No dedicated page-check script exists in the
repo; the checks above are ad hoc (`/tmp/cmbn/*.js`).
