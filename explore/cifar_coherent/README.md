# Coherent sets across layer blocks (exploratory side study)

Not an `EXPERIMENTS.md` stage. One trained seed supplies the measurements; a
second seed supplies layouts only. Labels are used for display and validation,
not to construct kernels, choose spectral bases, or select the final kernel rules.

## Status — complete, 2026-11-12

- **`data/r20s0b/`**: complete schema-v2 measurements, 264.6 MB. Eight checkpoints,
  twelve sites, 2,000 fixed probe images, five kernel rules; all 78 layer blocks
  per checkpoint and all 36 time blocks per site. Includes SEBA transforms,
  retention, channels, per-image losses and derivatives, gradient cosine Grams,
  comparisons, all 73 training-curve checkpoints, and decodability at 14 checkpoints.
- **`data/r20s1b/`**: eight top-50-PC and AlignedUMAP/Procrustes layouts, with
  exactly the same probe indices. Seed-1 training took 800.6 s; final test accuracy
  0.9151. No seed-1 spectra or validation fits were constructed.
- **`data/_fixture/`**: complete synthetic N=2,000 fixture through the real
  affinity / scaling / eigensystem / SEBA / retention path. Identical synthetic
  cells are deliberately reused across sites and checkpoints. Its linear-model
  derivative fields and gradient Grams are also synthetic. It is not evidence
  about either trained network; its validation fits are explicitly fixture-only.
- **Validation:** every required file was checked for layout, shape, dtype,
  finite values and prescribed NaN padding. All **5,280 block records** pass;
  the worst SEBA label-reconstruction agreement is **99.85%** for real data and
  **100%** for the fixture. All eight mathematical tests and the stale-cache
  regression pass. `data/r20s0b/validation.json` records the real-data verdict.
- Final detached coordinator: `fc-01M77KA5BY6WGXZ3N1TM1CTZFQ`;
  seed-1 training: `fc-01M77H21TW76R921QRC4104NH8`.
  Launch records are in `runs/`.

### Spend

The extension's ceiling was **$25**. Completed seed-0 worker timings, including
initial extractions, the 285-cell/rule sweep, derivatives, validation fits,
spectra and comparisons, give **$2.673**. Seed-1 layout workers add **$0.102**;
seed-1 training's GPU time is **$0.434**, budgeted as **$0.50** including CPU/RAM.
Thus completed work totals approximately **$3.28**. We reserve a further **$2**
for the cancelled initial spectra, failed initial float32 derivative checks,
container startup and coordinator overhead: **a conservative accounted estimate
of $5.3**, well below the ceiling. This is a timing/rate estimate, not an invoice
or an exact account-billing query.

Prices checked at <https://modal.com/pricing> on 2026-11-12: L40S
$0.000542/s; A100-40GB $0.000583/s; CPU $0.0000131/physical-core/s;
RAM $0.00000222/GiB/s. Final double-precision spectral work used A100-40GB.
Worker timing records live on `cm-cifar-coherent:/work/r20s0b/` and
`/work/r20s1b/`; the seed-1 layout ledger is also in
`runs/seed1-layout-spend.json`.

## Construction and final five rules

All rules start with cosine distances between flattened, probe-centred,
row-normalized activations. In this **stated variant** of `THEORY.md` section 3.1,
the affinity diagonal is zeroed, the symmetric affinity is scaled to a
zero-diagonal doubly stochastic operator **P**, and every subsequent computation
uses the fixed-lazy **M = (I + P)/2**. This gives every cell self-mass exactly 1/2
and moves the spectrum into [0,1], avoiding alternating near-bipartite modes in
product singular vectors. It equalizes self-loop laziness, not spectral gaps or
mixing rates.

Every final rule uses `coherent.symmetric_balance`: up to ten damped log-domain
scaling warmups, then line-searched Newton steps on the same symmetric scaling
equations if necessary. A row-sum residual **strictly below 2e-12** is required;
nonconvergence is an error. The sweep uses the same solver. Retained-diagonal vMF
**diagnostics**, not the construction, use the multiplicative symmetric-Sinkhorn
iteration. Operators stay float64 on disk and in eigensolves. A Householder
change of coordinates isolates the analytical constant mode before solving for
nontrivial eigenvectors, including in nearly-identity diagnostic cases.

Final rules, in file order:

0. **kNN graph, k = 15** — default. Union of directed 15-nearest-neighbour graphs;
   edge weights exactly 1, nonedge weights 1e-6, diagonal 0.
1. **Per-point perplexity 100** — each conditional row gets its own inverse
   temperature; symmetrize the conditional probabilities before scaling.
2. **Per-point perplexity 300** — the coarser counterpart.
3. **kNN scale** — one kappa per cell, inverse median 15th-neighbour distance;
   a nearly flat reference in the trained middle layers.
4. **Half-mass, calibrated** — one kappa per cell, calibrated to median raw
   nonself 15-NN mass 1/2; retained as the pair-localization cautionary case.

These were selected using **unlabelled** participation, gaps, set sizes and
numerical stability from `sweep.json` (19 rules × 15 real cells). Class Jaccard
was computed only as validation. The exploratory sweep's union graph added the
1e-6 floor to all off-diagonal entries, giving edge weight 1.000001; the final
contract uses edge weight exactly 1. This tiny convention difference is explicit.

SEBA follows Froyland–Rock–Sakellariou's normalized soft-threshold iteration and
returns the thresholded vectors, sign-fixed, max-scaled and reliability-ordered.
The saved rotation is transformed out of the internal QR basis into the original
signed eigenbasis. `vecs` uses **int16 / 32767**, not int8: int8 failed the
high-dimensional reconstruction regression. Float32 scales and rotations remain.

### Final-checkpoint headline numbers

Kappa and NN mass below are medians across the twelve sites; brackets give the
site range. For perplexity rules, kappa means each cell's **median per-point**
kappa. NN mass is measured on the actual symmetric **pre-Sinkhorn** affinity,
not on the final diffusion operator. Eigenvalues are for the full stem–logits block.

| Rule | Kappa | 15-NN mass | lambda_1, lambda_2, lambda_3 |
|---|---:|---:|---|
| Union 15-NN | not applicable | .7499 [.7142, .7894] | .906484, .854894, .824574 |
| Perplexity 100 | 30.887 [15.091, 44.037] | .4236 [.3263, .4849] | .874789, .819573, .782627 |
| Perplexity 300 | 24.299 [6.497, 33.850] | .2405 [.1249, .3085] | .834440, .770043, .732984 |
| kNN scale | 1.220 [1.162, 13.386] | .0097 [.0090, .2419] | .598419, .585656, .581501 |
| Calibrated half-mass | 30.390 [16.678, 43.242] | .5000 [.5000, .5000] | .930524, .919770, .892782 |

Average/product subspace agreement (mean cos² principal angles), averaged over
all checkpoints and nontrivial layer blocks:

| Rule, r | Chronological right / left | Two-end right / left |
|---|---|---|
| Union 15-NN, 3 | .98259 / .98515 | .94240 / .94368 |
| Union 15-NN, 6 | .97930 / .98203 | .93854 / .93966 |
| Perplexity 100, 3 | .98568 / .98700 | .94898 / .94886 |
| Perplexity 100, 6 | .98520 / .98693 | .94867 / .94822 |

Agreement is not identity: the final full block's chronological right/left
agreements are .83594/.88114 (graph, r=3) and .87202/.89717 (perplexity100, r=3).

For normalization comparisons, mean agreement across final-step layer blocks is:

| Rule, r | Old alpha=1 vs final | Retained-diagonal Sinkhorn vs final |
|---|---:|---:|
| kNN scale, 3 | .99098 | .99992 |
| kNN scale, 6 | .98984 | .99880 |
| Calibrated half-mass, 3 | .04364 | .08998 |
| Calibrated half-mass, 6 | .03578 | .06170 |

The calibrated rule is genuinely changed by the self-loop treatment; the old
construction was not an interchangeable normalization. Its single-cell modes can
still isolate near-duplicate pairs after the fix. Counts of **nontrivial**
single-cell eigenvalues above .995 at the final checkpoint, in site order
stem, b1–b9, pool, logits:

- Graph: `[0,0,0,0,0,0,0,0,0,0,1,1]`.
- Perplexity100, perplexity300, kNN scale: all zero.
- Calibrated half-mass: `[0,1,1,2,2,2,2,1,1,0,0,1]`.

Final decodability is a **closed-form validation fit**, not part of the spectral
construction. The unchanged protocol uses 10k train images for fit/GCV and all
10k test images for evaluation, with fit-mean centering and row normalization.

| Site | Flat ridge | Flat weighted kNN | GAP ridge | GAP weighted kNN |
|---|---:|---:|---:|---:|
| stem | .4664 | .4039 | .3172 | .2839 |
| b1 | .5223 | .4338 | .3862 | .3586 |
| b2 | .6157 | .5343 | .4410 | .4122 |
| b3 | .6502 | .6129 | .4393 | .4451 |
| b4 | .6930 | .6723 | .5372 | .4923 |
| b5 | .7342 | .7109 | .5876 | .5431 |
| b6 | .7632 | .7468 | .6183 | .5754 |
| b7 | .8263 | .8113 | .7510 | .6898 |
| b8 | .8713 | .8486 | .8422 | .8039 |
| b9 | .9094 | .9122 | .9093 | .9150 |
| pool | .9093 | .9150 | .9093 | .9150 |
| logits | .9104 | .9153 | .9104 | .9153 |

## Reproduce and validate

```bash
# From this directory; keep scratch off the VPS's quota-limited /tmp tmpfs.
export TMPDIR=/var/tmp/cm-pipeline
mkdir -p "$TMPDIR"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python test_coherent.py
python3 test_provenance.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python schema_v2.py fixture data/_fixture
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python schema_v2.py validate data/_fixture
uv run --with modal python measure_v2.py deploy
uv run --with modal python measure_v2.py launch all
uv run --with modal python measure_v2.py status all
# Wait until status reports completion before launching derivatives.
uv run --with modal python measure_v2.py launch derivatives
uv run --with modal python measure_v2.py status derivatives
# Wait for derivatives, then validate and fetch.
uv run --with modal python measure_v2.py launch validate
uv run --with modal python measure_v2.py status validate
uv run --with modal python measure_v2.py fetch r20s0b
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python schema_v2.py validate data/r20s0b
# Optional: repeat the documented candidate-rule sweep.
uv run --with modal python measure_v2.py launch sweep
# Seed 1: the same recipe, with seed 1; layout-only measurement afterwards.
uv run --with modal python modal_train.py deploy
uv run --with modal python modal_train.py launch --seed 1
# Wait for its training record to report complete before this stage.
uv run --with modal python measure_v2.py launch layouts
uv run --with modal python measure_v2.py fetch r20s1b
```

`SPEC.md` section 2 is the public contract. `meta.kappa/q10/q90` are null for
the graph, per-point quantiles for perplexity, and identical per-cell scalars for
rules 3–4. `self_weight` is null for rules 0–2 and the retained-vMF diagnostic for
3–4. The comparison files contain `rows` with `step, mode, a, b, r`:
`right/left/end_right/end_left` for average/product, and
`agreement/retained_agreement` for normalization. `decod.json` contains `results`,
one per checkpoint, with `views[view][site].ridge_acc/knn_acc`; GAP at pool/logits
aliases flat. `curves.json` is the run manifest with its `checkpoints` list.

### Cache provenance

`construction.py` defines the shared construction fingerprint. Construction-
dependent completion records must match it exactly and their outputs must exist
before reuse; old or missing fingerprints are rejected. Metadata includes the
same fingerprint. A one-time audit of the in-flight final-five run checked all
**8 final extractions, 29 downstream units and 2 aggregate outputs**: **no stale markers or outputs**
were found. The latest extraction completed at 10:02:49 UTC; every dependent
output was newer. Its already-running completion records were explicitly adopted
by that timestamp audit, rather than pretending they had originally contained
fingerprints. See `runs/provenance-final.json`; `test_provenance.py` asserts cache
rejection for mismatched/missing fingerprints and missing outputs.

## Licence note

`seba.py` and `test_seba.py` are ports/translations of
<https://github.com/gfroyland/SEBA>, by Gary Froyland, Christopher P. Rock and
Konstantinos Sakellariou, and are marked **GPL-3.0-or-later** as derived from
upstream. Both retain the upstream copyright notice and the citation to their
2019 paper. No other file in this directory contains code ported from that
repository; no licence grant for the rest of this repository is asserted here.
The upstream notice also contains a separate commercial-use sentence, retained
verbatim. The owner should confirm the repository's licensing arrangement before
publishing these source changes. No commits or pushes were made for this task.

## Legacy run (`r20s0`)

- Model: ModelZoo `resnet20/relu-seed0`, retrained on Modal (L40S, 849 s, final
  accuracy .9156; zoo original .9217). Volume:
  `cm-cifar-coherent:/runs/resnet20-relu-seed0`, 73 checkpoints.
- Probe: 2,000 CIFAR-10 test images, 200 per class, RandomState(0), sorted indices.
- Legacy operator: vMF kernel, alpha=1 density normalization, symmetric degree
  normalization. The old explorer offered kNN-scale, a two-distance half-mass
  heuristic and global kappas {2,5,12,30,75}. These are not the final v2 rules.
- Legacy spend (2026-11-11): approximately $1–1.5, separate from the extension above.

```bash
uv run --with modal python measure.py launch resnet20-relu-seed0 r20s0 <steps>
uv run --with modal python measure.py respectral r20s0
uv run --with modal python measure.py fetch r20s0
uv run --with numpy --with pillow python build_viewer.py r20s0
```

## Publishing the site

```
uv run --with numpy --with pillow python site/build_site.py r20s0b --layout-tag r20s1b   # -> viewer/site/
# deploy: copy viewer/site/ into an empty directory, add .nojekyll, commit as a single
# orphan commit on branch gh-pages and force-push (the branch holds only the built site)
```
Live at https://charlesr-w.github.io/CoherentModes/ .
