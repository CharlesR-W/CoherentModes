# Coherent sets over token contexts in Mamba (toy, exploratory)

> **Superseded (2026-11-13).** The report for this study is replaced by the joint report in `explore/ncs_report/` (published as `CoherentModes/ncs-vit-mamba`). The data and code here are still read by that report's build; the page in this directory is kept for reference and is no longer updated.

Spec: `SPEC.md` (delta against `../mamba_coherent/SPEC.md`, itself a delta against `../lm_coherent/SPEC.md` v2.1).
Exploratory side study; nothing here enters `THEORY.md`.

## Status

**Complete, 2026-11-13.** `data/m130c/` fetched and validating (8,000 items, 666 spectral records, worst SEBA label
reconstruction 0.999). Pilot 1 stopped on the `pred` duplicates; after the SPEC amendment (references exempt, tie
check) pilot 2 lost its U2 to a bookkeeping bug (fixed); pilot 3's gate passed; the full run completed. Ledger total
**$5.31** of $8. No economies were needed. The page is not built (a page agent works in `page/`).

**Pilot gate (pilot 3):** C1a 4.6e-7 (tol 1e-4), C1b 0.0, C3 0.0 over 1,392 item-layers, P3 1.2e-14; Sinkhorn max
row error 1.93e-12 (gate 2e-12: passes, narrowly); min off-diagonal distance over point/state/whole cells 4.0e-5;
no zero-norm item; min PN(psi_1) over blocks 188. References: `pred` 1 duplicate pair (2.3e-10), not tie-sensitive;
bag 2.4e-4 and recent 6.3e-3, none. Full-size check present and calibrated, so certification is enabled (theta 3).

**Targets (`data/m130c/targets.json`):** K1 met (AMI with Cc 0.494 vs W 0.155); K1b met (kappa 0.299 vs band top
0.204, kappa~|T 0.121); K2 **not met** (AMI W 0.735 > Cc 0.153, but U~ whole 0.215 vs B-point 0.030 + 0.2);
K3 **not met** (T~ L4 0.380 > L24 0.104, but kappa~ L24 - L4 = 0.264 - 0.187 = 0.077 < 0.1); K4 met (B-point
vectors in point L16: 0.91 -> 0.22 by tau = 2; whole: 0.93 -> 0.65 at tau = 16); K5 **not met** (nothing certified:
B-point excluded at every k; B-state and whole undetermined at k = 2 (and whole at 3), excluded elsewhere);
K6 met (init kappa~ L24 0.000 vs 0.264; init W~ whole 0.173 vs 0.849).

- **Tests:** `test_mc.py` 18/18 pass on CPU (tiny random Mamba-1, 24 layers, width 16; transformers 4.57.6).
  Upsilon_hat(2) of the sweep cut vs `lm_tree`'s Upsilon: 5e-15. Streaming state/whole Grams vs
  `mb_cells.position_grams`: bitwise equal. Running-mean block vs `block_modes` on the stacked list: 0.0. T2's node
  rebuild vs the explicit concatenation on a subset: 7e-16. Streaming tree vs `lm_tree` (cut, members, every record
  number): equal to 1e-10. Shares vs `lm_cells.walk_share` (same permutation draws): 1e-14. Snapshot rows vs a
  direct batch-1 decode at lags 0, 4, 16: 5e-7 relative (float32), and bitwise equal to a batched decode read at
  the item's step. Within-token null: predictions that depend on the token only give kappa = 0.9396 inside the band
  [0.9394, 0.9402] (kappa~|T = -0.003); with a planted context class kappa = 0.957 against the band [0.609, 0.623]
  (kappa~|T = 0.89). eta^2 of psi_1 against the planted grouping 1.000. U1 on the tiny model (N = 240): C3 0.0 (vs
  v1's own decode), P3 1.7e-15, C1a 1.6e-16, C1b 0.0, and a NaN in the v1 reference fails C3 (inf); then U2 (both
  checkpoints) and U3 on U1's own output validate (666 spectral records).
- **Fixture calibration (SPEC 8, Modal CPU, all 8 seeds, $0.23):** theta = **3.0** by v1's rule ("fixed": the
  check-seed planted minimum at s in {1, 2} is 382, the structureless maximum over k in R is 1.89); **P1' passes**
  (every planted block certified at its k; no structureless block certified at any k). Development seeds agree
  (423, 1.92). Detection at the lower strengths (check seeds): s = 0.75 certified in every case; s = 0.5 certified
  for k = 2, 4 (both sizes) and k = 8 balanced, not for k = 8 Zipf or k = 16. Structureless caps 2 mu_k/mu_{k-1}
  are 2.0 to 2.03 at every k, so noise reads "excluded", not "undetermined". `runs/calibration.json`.
- **Full-size check (bench, N = 8,000, seed 100):** planted k = 8 Zipf s = 1: Upsilon_hat(8) = 1515; structureless
  maximum 1.90; on the right sides of theta: **calibrated at this N**.
- **Page fixtures (MOCKUP), built locally through the real U2/U3 code path, both validate:** `data/_fixture` (k = 8 Zipf token classes,
  tokens nested in classes, 150 snippets of 5 sources, lags that decay) and `data/_fixture_flat` (every view
  noise; same tokens and groupings); 666 spectral records each, worst SEBA label reconstruction 0.999; 6.1 and 9.5
  min on 2 threads. Planted: B-point certified at k = 8 only (excluded at every other k); eta^2 of B-point's
  psi_1..psi_7 against the class = 1.000 and psi_8.. at chance; kappa~|T = 0.00 (predictions are a function of
  the token there, so K1 without K1b, as designed); point lag retention 1.00, 0.58, 0.24, 0.04 at tau = 0, 1, 2, 4.
  Flat: nothing certified, every k excluded, every share and eta^2 at chance, T2's root Gram route 0.0.
  `data/*/targets.json` evaluates K1..K6 on MOCKUP data only (the flat twin "meets" K1 with AMIs of -0.000 vs
  -0.002: K1 has no null).
- **Modal:** app `cm-mamba-contexts` deployed (bench, calib, diag, u1_trained, u1_init, u2_trained, u2_init,
  u3_finalize, coordinate). Stage A spend from the ledger: **$1.90** (calib $0.23, bench $1.55, diag $0.02,
  coordinators), above the $1 asked for Stage A; see "Bench" for why. Reservation at pilot launch: $7.23 of $8.

### Review fixes (GPT-6.1 Sol, adversarial, 2026-11-12), before Stage B

1. **Reservation (high):** a unit leaves the reservation only if its function will skip it: a cached kind
   (`mc_ledger.CACHED_KINDS`; U3 and `diag` always rerun), a completion record under the current key, and every
   required output present on the volume (the same output lists the functions check). The worst case now includes
   the 90 s per-attempt overhead the ledger charges: (timeout + 90 s) x rate. Pilot reservation $7.23, full $4.97
   now ($7.65 after a pilot that costs as projected).
2. **kappa^T (medium):** a null draw counts pair (i, j) only if i, j are in different snippets and their prediction
   sources q(i), q(j) are too; each draw is normalised by its own surviving walk mass; a draw with none is NaN and
   makes kappaT and its band NaN (validation then fails).
3. **Fail-closed folds (medium):** `K.worst_of` / `K.least_of` replace Python max/min over floats in every control
   and gate number: P3 (pass requires every deviation finite and below 1e-12), C3, the U2 summary (row error, min
   distance, min PN), the calibration (planted min, structureless max, row error), and C1a/C1b, whose v1 `max` over
   layers is re-decided here.
4. **Certification (medium):** `mc_schema.reading_rule` enables "certified" only with theta and a full-size check
   whose `calibrated` is True; if it is missing or false, k is read from the cap only (excluded/undetermined), the
   reason goes into meta.json, upsilon.json and K5's status, and `validate` checks all three agree. U3 attaches the
   bench's check; the fixture CLI reads it from `runs/bench.json`.

### Stage B, 2026-11-13: pilot stopped at U2 (duplicate assertion in the `pred` reference cell)

The pilot's U1 passed every control. U2 then stopped, as designed, on v1's duplicate assertion in the `pred`
reference cell: items 3312 and 7717 have centred-cosine distance **2.26e-10** (< 1e-9). Both are `code` contexts
where the trained model puts p = 1.000000 on ',' (`...and/or\n// sell copies of the Software` and
`func (c *FakePodSecurityPolicies) Watch(opts v1.ListOptions) (watch.Interface`), so sqrt(p) is the same one-hot
vector to float32 precision. 167 of 8,000 items have top-1 p > 0.9999 (333 above 0.999), so more such pairs are
likely. Per the rules for unattended runs nothing was patched, the gate was not evaluated, and the full run was not
launched. The owner's decision (options: exempt the reference cells, which are not model views, from the duplicate
assertion as SPEC 7 does for constituents; or drop/alter `pred`) is needed before a rerun; U1 `trained` is cached
and will be reused (U2 reruns from scratch).

- **U1 `trained` (12.1 min, $0.62):** C1a 4.6e-7 (float64 parameters and cache; tol 1e-4; float32 hazard 4.2e-5
  at L24), C1b 0.0 over 64 resumed steps, **C3 0.0** over 1,392 item-layers (bitwise equal to v1), **P3 1.2e-14**,
  no zero-norm item in any view or constituent, 22 de-duplication redraws, peak GPU 32.5 GB. Decode passes took
  9 s each (projected 63 s); the Grams 76-92 s per lag.
- **U2 `trained` (failed after 6.5 min, $0.37):** point (24 + 48 sensitivity walks), state (24), whole, bag and
  recent cells and their spectra completed before `pred`; no outputs are written until the end, so Sinkhorn, min
  distance and PN for those cells are not on record.
- A side issue: `lm_cells.DegenerateItems` cannot be unpickled by Modal (its `__init__` takes extra arguments), so
  the coordinator log shows "Could not deserialize remote exception"; the container log has the real message.

### Amendment after pilot 1 (SPEC 3, 7, 11; accepted by the plan reviewer), 2026-11-13

- The reference cells (bag, recent, pred) are exempt from the duplicate assertion (`mc_stats.cell_walk(...,
  assert_distinct=False)`); every point, state and whole cell, lag cells included, keeps it. Per reference: the
  smallest off-diagonal distance and the number of pairs below 1e-9 (cells json, U2 summary, meta.json
  `references`, validation.json). Zero-norm and Sinkhorn checks stay on.
- Tie sensitivity (`mc_stats.tie_check`): a reference with such a pair is rebuilt under item reorderings
  RandomState(1, 2, 3) (undone afterwards); a_8, a_16 and the AMI of the r = 8, 16 sets against the original are
  stored, and `tie_sensitive` is set if any a_r < 0.9 or any AMI < 0.8.
- Fingerprints: `AMENDMENTS` changes the construction fingerprint (U2/U3, meta, gate). U1's outputs are unchanged
  by the amendment, so U1 is keyed on `DECODE_FINGERPRINT` (the original construction, equal to pilot 1's), and pilot
  1's U1 is reused. The failed attempt stays in the ledger and the Runs table.
- Unit errors are re-raised as RuntimeError with their message (Modal could not unpickle `DegenerateItems`), and
  U2 writes `u2_partial_<checkpoint>.json` with every completed cell's record if it fails.
- `data/_fixture*` were built under the earlier fingerprint and are left in place for the page builder (the test
  reports them as not validated); the same code path was rebuilt and validated in a scratch directory.

## Runs

| stage | coordinator call | outcome | ledger |
|---|---|---|---|
| calib | `fc-01M79QDRHPPSYGWCCVVRZ4DTS0` | 8 seeds ok; theta 3.0, P1' pass | $0.23 |
| bench | `fc-01M79QTKBPYH772C3R27XVFX3X` | ok; full-size check calibrated | $1.65 |
| diag | `fc-01M79SZ5ZWQ9JXAQR0YPEABV5R` | ok | $0.02 |
| pilot 1 | `fc-01M79XRX0WSKMY3JP05MG98EXE` | U1 ok (12 min); U2 stopped (duplicate in `pred`); no gate | $1.00 |
| pilot 2 | `fc-01M7A01WVGDZR0NFTFXXP10QE7` | U1 cached; U2 computed (12 min) but its completion record failed (`seconds` clash in `done`); outputs not reused | $0.71 |
| pilot 3 | `fc-01M7A19T1CG13RT4AAS6KT9570` | U1 cached; U2 ok (11.6 min); gate passed | $0.69 |
| full | `fc-01M7A2244SGWZ7RB9B58AXVZR5` | U1 init 3 min, U2 init 6.7 min, U3 8.2 min; 18 min wall | $1.01 |

Ledger total **$5.31** of the $8 ceiling (Stage A $1.90, Stage B $3.41). Timeouts after the pilot
(`mc_modal.py calibrate`): U1_init 900 s, U2_init 900 s (floors), U3 1,656 s.

Pilot 2's failure was in the Modal wrapper, not the science: `done()` merged U2's summary (which has `seconds`)
into the completion record and raised. Fixed in `mc_ledger.completion_row` (clashing extra fields are kept as
`unit_<name>`), with a test; U2 was rerun rather than hand-writing its completion record.

## Files

| file | role |
|---|---|
| `mc_construction.py` | constants, construction dict and fingerprint (folds in v1's), import bridge, json helper |
| `mc_items.py` | items (8 positions per v1 snippet, label-free de-duplication, seeded order), lags, groupings, contexts, bag/recent vectors |
| `mc_stats.py` | streaming Grams (`ViewGrams`), running-mean blocks (`BlockMean`), spectra, both roundings and Upsilon_hat, reading rule, shares, kappa with the within-token null, eta^2, AMI / matched Jaccard, set descriptors, lag retention, P spill, streaming tree |
| `mc_decode.py` | U1: snapshot decode (one pass per lag), Grams, predictions, raw vectors for T2; C1a/C1b (v1 code), C3, P3, zero-norm scan |
| `mc_schema.py` | U2 (`measure_checkpoint`), U3 (`finalize`: trees, comparisons, Upsilon outcomes, layouts, targets, items/vocab), page fixture, `validate` |
| `mc_calib.py` | the fixture generator and calibration of theta and P1' (SPEC 8), the full-size check |
| `mc_bench.py` | the `bench` unit (synthetic, full N, no model) |
| `mc_project.py` | projected cost and timeouts from the bench -> `runs/projection.json` |
| `mc_ledger.py` | which units a stage will run and their worst case (pure functions, tested) |
| `mc_modal.py` | Modal app `cm-mamba-contexts` (units `bench`, `calib_<seed>`, `diag`, U1/U2 per checkpoint, U3; stages calib, bench, diag, pilot, full; ledger, guard, gate) |
| `test_mc.py` | CPU tests (SPEC 8 list and more) |
| `runs/` | `calibration.json`, `bench.json`, `diag.json`, `projection.json`, `timeouts.json`, launch records |
| `data/_fixture`, `data/_fixture_flat` | page fixtures (MOCKUP, untracked) |

## Reused, generalised, replaced

- **Reused by import, unchanged:** `mb_model` (HF decode step and cache handling: `Decoder`, `Hooks`, `sequence`,
  `assert_slow_path`, `load_model`, `parameter_sha`; C1a/C1b via `decode_views(c1b=True)`, `decode_residuals`, `c1a`;
  `_rel`), `mb_construction` (cache_position, model pin, ledger arithmetic via `lm_construction`),
  `mb_cells.explicit_concat_gram` and `position_grams` (P3 reference; test reference), `lm_cells` (`unit_rows`,
  `renormalise`, `SlabAccumulator`, `walk`, `lazy`, `permutation_bank`, `matched_jaccard`, `zero_norm_items`,
  `walk_share` as test reference), `lm_tree` (`sweep`, `m_min`, `components`, `threshold_rule`, `to_json`; and
  `node_operators`/`analyse_node`/`grow` as test references), `lm_schema.Probe`, `coherent.block_modes` and
  `symmetric_balance`, `seba` (`seba`, `seba_labels`, `seba_reconstruct`), `schema_v2.comparison`,
  `diagnosis/token_classes.py`.
- **Replaced by streaming versions here (SPEC 8), each tested against the original at small N:**
  `mb_cells.StateStack`/`position_grams` -> `mc_stats.ViewGrams` (bitwise equal); `block_modes` on a stacked list ->
  `BlockMean` + `spectrum` (eigenvalues within 1e-13); `lm_tree.node_operators`/`analyse_node`/`grow` ->
  `node_operators_streaming`/`analyse_node_streaming`/`grow_streaming` (same cut, members and record, P spilled to
  local disk for the retention pass); `lm_cells.unit_rows`/`standard_gram` -> in-place `unit_rows_`/`view_gram`
  (bitwise equal, fewer N x D float64 copies).
- **No file outside this directory was modified.** (So `test_lm.py` and `test_mb.py` were not rerun.)

## DEVIATIONS (from SPEC.md; smallest sensible choices, made while the owner was asleep)

1. **De-duplication compares the multiset of the last 16 tokens, not the sequence.** `recent` (mean embedding of
   tokens t-15..t) is a function of that multiset, so two items whose windows are permutations of each other (on
   the real probe, typically (i, t) and (i, t+1) when the token leaving the window equals the one entering) are
   exact duplicates in a reference cell and the v1 duplicate assertion stops U2. The fixture hit this. A shared
   sequence is a shared multiset, so this is the stricter reading of "share their last 16 tokens". On the real
   probe: 22 redraws; no two items share their bag multiset either.
2. **kappa is normalised by the walk mass it scores:** kappa = sum_{different snippet} P_ij BC_ij / sum_{different
   snippet} P_ij, which equals SPEC's (1/N) sum when no same-snippet pair carries walk mass; without it, a walk
   that stays inside snippets would read as predictively incoherent. The SPEC form is stored as `kappa_raw` with
   the mass (`mass_diff`). The within-token null permutes prediction rows within token groups and keeps the
   exclusion on the walk pair (i, j), so kappa^T uses the same pairs and normaliser as kappa.
3. **Block statistics are computed on the block's Pbar.** Shares on Pbar equal the mean of the cells' (linear in P);
   kappa is a ratio, so a block's kappa is kappa(Pbar) (K1b reads this), not the mean of its cells' kappa.
4. **Pilot = U1 and U2 on `trained`, then the gate** (SPEC: "U1 and the gate"): the gate's Sinkhorn, duplicate and
   PN checks are U2 outputs. C1a/C1b run inside U1 `trained` with v1's code on the 1,000 snippets.
5. **Units added:** `calib_<seed>` (the fixture calibration on Modal CPU; locally it would take about 45 min),
   `diag` (a CPU probe of SEBA's cost, below). **No U0:** U1 copies the probe and asserts its sha256 (v1's U0
   established the tokenizer facts on the same file). U1 mounts v1's volume to read `act_trained.npy` for C3.
6. **The guard reserves per stage, and each (unit, checkpoint) is its own Modal function with its own timeout.**
   v1 reserved every remaining unit at every launch. Here that would be $10.3 at 2x timeouts although the projected
   study total is $6.1; per stage, the pilot reserves $7.23 and the full run (if the pilot costs as projected)
   $7.65. The ceiling is still enforced before every dispatch, so nothing can spend past $8; the risk is only that
   the full run is refused after an expensive pilot.
7. **Groupings and objects:** lag cells are scored against T, Cc, U, W (lag 0) and T_tau, without kappa; B-lag-point
   and B-lag-whole against T, Cc, U, W. `recent` also gets a spectral object (SPEC lists bag and pred); the
   references are not computed at `init`. Set descriptors of lag cells use the lag-0 tokens. The matched-Jaccard
   null keeps v1's 200-permutation bank; shares, eta^2 and kappa use 50.
8. **Also stored (v1 inheritance):** each block's retention of psi_1..8 and of its r = 8 SEBA vectors in its own
   cells (M-form, v1 `block_record`), next to the SPEC's lag retention (P-form, chance -1/(N-1)).
9. **eta^2's lambda-weighted mean uses lambda_k(Mbar)** (k = 1..16). SEBA at r uses the constant and psi_1..psi_{r-1}
   (v1 and Froyland's convention), so r = 32 uses the 33rd eigenpair's predecessors only.
10. **Calibration seeding:** each block draws from `default_rng([seed, k, sizes_index, round(100 s)])` (structureless:
    `[seed, 0, 0, 0]`); one structureless block per seed. The SPEC fixes seeds but not how cases derive from them.
11. **Implementation speed-ups that do not change the mathematics:** SEBA and k-means for the 9 ranks run
    concurrently, each single-threaded; UMAP layouts (fixed random_state, single-threaded) run in parallel
    processes; NUMBA_NUM_THREADS=1 in the image.
12. **The decode was not benchmarked** (Stage A spends nothing on the model). The projection takes v1's measured
    decode (42 s per pass over the 1,000 snippets on an L40S) with a 1.5x margin.
13. **v1's two-way Upsilon in the trees** is read against v1's threshold 3.0 (its P1 value); theta for
    Upsilon_hat(k) is this study's calibration (also 3.0).

## Bench (Modal L40S, 8 cores, 64 GiB; synthetic data at N = 8,000; `runs/bench.json`)

| operation | seconds | peak GPU (GB, allocated) | notes |
|---|---|---|---|
| U1 buffers (point, c, z, sqrt p) + simulated pass | 0.5 | 23.0 | 24.6 GB of float32 buffers fit |
| one constituent Gram at z's width (24,576) | 2.72 | 24.9 | c's width (4,608): 0.48 s |
| BC and `pred` Grams from sqrt p | 10.3 | 26.9 | |
| 512 MB array: volume write+commit / read | 3.5 / 0.09 | - | read was cached; projection uses the write rate |
| one cell (view Gram, kNN, Sinkhorn, 7 shares x 51, kappa x 51) | 2.27 | 4.8 | walk 1.42 s (13 iterations), shares 0.48 s, kappa 0.34 s |
| one spectral object (33 pairs, SEBA all r, both roundings, eta^2, AMI, sets) | 9.4 | 5.3 | eigh 3.0 s; clustered data (SEBA converged early) |
| tree node, 24-cell block, n = 8,000 / 2,000 | 42.5 / 5.5 | 6.3 / 1.9 | 24 walks 33 s at n = 8,000 |
| one UMAP layout (CPU, 1 thread) | 90 | - | |
| full-size calibration check (2 blocks x 24 cells) | 1814 | 4.8 | anomaly, below |

Host RSS stayed below 6 GB. `diag` (CPU, 8 cores): SEBA run to max_iter on structureless vectors at N = 8,000 costs
9.4 s at r = 32 and 36.5 s over all r (no thread-contention effect in a fresh process).

**Anomaly.** The full-size check should take about 3 minutes (48 GPU walks at about 1.4 s, two spectra); it took 30
minutes, and that is most of the bench's $1.55. The walks match the bench's own cell (13 iterations, 3 Newton steps
at N = 8,000, checked locally), and SEBA and k-means are fast in a fresh process. It ran right after UMAP in the
same process; my best guess is CPU thread contention from numba's pool (UMAP) with torch's and scikit-learn's
OpenMP pools. Production U2 never imports UMAP; U3 runs layouts last, in subprocesses; the image sets
NUMBA_NUM_THREADS=1. Not confirmed (confirming it would cost another bench).

## Projected spend and timeouts (`runs/projection.json`, `runs/timeouts.json`)

| unit | projected | cost | timeout |
|---|---|---|---|
| U1 `trained` (6 passes, 6 x 48 Grams, 47 GB to the volume, C1a/C1b) | 28 min | $1.40 | 3,356 s |
| U2 `trained` (87 cells + 48 sensitivity walks, 60 spectral objects at SEBA's worst case) | 26 min | $1.28 | 3,072 s |
| U1 `init` | 4.5 min | $0.29 | 900 s |
| U2 `init` (49 cells, 14 objects) | 7.7 min | $0.43 | 920 s |
| U3 (T1, T2, layouts, comparisons) | 14 min | $0.72 | 1,656 s |

Pilot about **$2.68 and 55 minutes**; full run about **$1.45 and 26 minutes**; with the $1.90 already spent the study
projects to about **$6.1** against the $8 ceiling. None of SPEC 8's economies is needed. The reservation at pilot
launch is $7.23 (with the per-attempt overhead, review fix 1); if the pilot costs more than about $3.0, the guard
will refuse the full run, and the first
economy (trees at depth 2) or a smaller U3 timeout would be the next step.

## Reproduce

```bash
export TMPDIR=/var/tmp/cm-mc
U="uv run -q --with torch --with numpy --with scipy --with scikit-learn"
$U --with transformers==4.57.6 python -I test_mc.py                         # tests (about 5 min on 2 threads)
uv run -q --with modal python mc_modal.py deploy
uv run -q --with modal python mc_modal.py launch calib                     # then: status calib; pull calib
uv run -q --with modal python mc_modal.py launch bench                     # then: status bench; pull bench
uv run -q --with modal python mc_modal.py launch diag                      # then: pull diag
uv run python -I mc_project.py runs/bench.json runs/diag.json --spent=1.896   # -> runs/projection.json
uv run -q --with modal python mc_modal.py timeouts && uv run -q --with modal python mc_modal.py deploy
$U --with umap-learn python -I mc_schema.py fixture data/_fixture runs/calibration.json --threads=2
$U --with umap-learn python -I mc_schema.py fixture data/_fixture_flat runs/calibration.json --flat --threads=2
$U python -I mc_schema.py validate data/_fixture
# Stage B (not run): launch pilot; status pilot; launch full; status full; fetch; validate data/m130c
```
