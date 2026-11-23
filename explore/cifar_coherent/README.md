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

## Product blocks (added 2026-11-15)

Alongside the average M̄, every layer and time block now has the **ordered product**
Q = C_b⋯C_a, C = M − 11ᵀ/N (first snapshot applied first; `product.py`). Files
`<family>prodR_<name>.npz` (right singular vectors v_k: start of block) and
`<family>prodL_<name>.npz` (left, u_k = Qv_k/σ_k: end of block) have the average files'
schema, with `vals` = σ (σ_0 = 1 for the constant), plus `gain_psi (5,P,8,X)` and
`gain_seba (5,P,5,8,X)`: per-snapshot norm ratios of the function pushed through the block
(start: first→last; end: last→first), NaN outside the block, multiplying to σ for ψ.
Float64 eigh of QᵀQ / QQᵀ, then the unchanged SEBA/export path (`schema_v2.block_record`).
Fingerprint `construction.PRODUCT_FINGERPRINT`. All 40 files pass `product.validate_product`;
worst gain-product identity error 1.4e-7. Modal stage `prodspectra`: 20 units, about **$1.10**
(completed seed-0 estimate now $3.78). `test_product.py`: single cell = eigenmodes; gains;
reversal transposes Q (same σ, sides swap), other permutations change σ; a planted split that
is strong in all but one cell beats a moderate persistent split under the average and loses
under the product; planted three-group clouds.

`product_analysis.py` writes `product_summary.json` (every block, rule, r: average↔product set
Jaccard, span agreement, class Jaccard, minimum-gain snapshot). At step 64,000, kNN graph, r = 4,
average↔product matched Jaccard falls with block length: .99 (2 sites), .95 (4), .90 (6), .83 (8),
.76 (10), .64 start / .77 end for stem→logits. Full-block class Jaccard: average .41, end .42,
start .29 (r = 8: .61, .74, .38). The explorer's **Operator** switch (average / product·start /
product·end) shows them; the retention strip adds step gains. The build ships all four product
families or none. The site grew from 135 MB to 362 MB.

## Coarse-graining certificates and 32 modes (added 2026-11-19)

Spec: `SPEC_COARSE.md` (plan-reviewed by GPT-6.2 Sol, two rounds). For every single cell and every layer chain
Q = M_b...M_a (all 8 checkpoints, 5 rules): full float64 SVD, chi^2 information kept by k-block partitions
(spectral k-means, SEBA, PCA k-means, random; classes/predictions as validation) against the Eckart-Young ceiling
F_k, leakage against Ky Fan / Cheeger bounds, gap reports, and eigen/singular vectors 9..32 for averages and both
product sides. Code `coarse.py` (tests `test_coarse.py`), Modal `coarse_modal.py` (app `cm-cifar-coherent-coarse`,
40 units, about **$6.3**), aggregation `coarse_summary.py` (`coarse_summary.json`, `coarse_facts.json`). Stored
vectors 1..8 agree with the existing files to |cos| >= 0.99999994. Site: explorer psi1..psi32 (all rules for cells,
default rule for longer blocks: 85 MB gz), spectrum strip, coarse-graining card, k-means colouring; Findings section 6.

Markov closure (SPEC_COARSE.md section 6, reviewed): per-site partitions on the walk through the layers, one-step
lumpability defect, coarse two-step memory, Chapman-Kolmogorov test, each against the lazy uniform walk as a
geometry-free control; partitions per site (spectral, PCA), shared (b9), random, classes. `coarse_modal.py markov`
(40 units, about **$1.9**), `test_markov.py`, `markov_summary.py` (-> `markov_summary.json`, facts merged into
`coarse_facts.json`); Findings section 7.

## Through-depth operator (added 2026-11-20)

Spec `SPEC_SPACETIME.md` (GPT-6.2 Sol plan review and GPT-6.2 Astra mathematics review, two rounds each). After
Trower, Djurdjevac Conrad & Klus (Chaos 35, 013126, 2025): one operator on (image, site) pairs at a checkpoint,
L_t = blockdiag((I - M_l)/c_l) + t (D_path (x) I), i.e. each site's roughness plus a charge for a function changing
between neighbouring sites (the supra-Laplacian of Froyland, Kalia & Koltai, with the identity as the link between
sites). Variants: relative site units (c_l = mean of the site's first 8 nontrivial 1 - lambda), native units
(c_l = 1), and the paper's own matrix as a reference. Six couplings around t*, where the first temporal and first
spatial eigenvalues meet. Code `spacetime.py` (operator, exact block-tridiagonal solver, rotation-invariant
readings, post-hoc set readouts), `spacetime_stage.py` (one unit's outputs and its acceptance checks),
`spacetime_modal.py` (app `cm-cifar-coherent-spacetime`, CPU, 16 units = 8 checkpoints x rules 0-1, about **$0.36**),
`test_spacetime.py` (nine tests). Outputs `data/r20s0b/spacetime/st_<step>_<rule>.npz`.

Pilot facts (step 64,000, kNN rule): t* = 8.36 (relative), .655 (native). Relabelling the images independently at
every site raises the first eigenvalue from .570 to 2.76 (gap-only lower bound .42). Dropping pool, dropping
logits or duplicating logits keeps 99% of the 16-mode space on stem..b9; scale rank 4 keeps 98.7%, rank 16 86%.
Mode 1's depth weights follow the 12-number gap-only control closely at every coupling, so a single mode's depth
profile is not evidence of a set appearing or vanishing. Sets (a fitted readout, rank 8): at k = 5 and t*, churn
between neighbouring sites .05 with .74 of the information ceiling kept, against .36 / .83 for independent
per-site partitions, .13 / .79 for block averages over four blocks of three sites and 0 / .63 for the 12-site
average; at k = 10, .10 / .57 against .29 / .80, .14 / .77 and 0 / .56. Caveat: the probe is class-stratified
(`DECISIONS.md` item 7).

```bash
OMP_NUM_THREADS=2 uv run --with numpy --with scipy python test_spacetime.py
uv run --with modal python spacetime_modal.py deploy && uv run --with modal python spacetime_modal.py pilot
uv run --with modal python spacetime_modal.py launch && uv run --with modal python spacetime_modal.py fetch
```

## Diffusion wavelets (added 2026-11-21)

Coifman–Maggioni diffusion wavelets of every single snapshot's lazy walk M (kernel rule 0, cosine), in the explorer
as Colour by "diffusion wavelets". `wavelets.py` (algorithm; `test_wavelets.py` checks orthonormal completeness,
nesting, rank against the eigenvalue count and fine-scale localization on a planted walk), `wavelets_modal.py`
(`pilot | launch | fetch`; 96 Modal CPU units, 0.2 container-hours, about $0.05; launch record
`runs/wavelets-launch.json`), `build_site.py --wavelets-only` (`d/dw/`, about 36 MB). Precision ε = 1e-4.

What it shows, over all 96 snapshots: levels 0–2 keep all 2,000 functions (3 in some late-site snapshots), since most
eigenvalues of M sit near 1/2 (the lazy half) and M^4 still resolves everything; the wavelets appear at j = 3 (0–222),
j = 4 (1,151–1,696) and j = 5 (198–584), then a few dozen more; 10–15 levels in all reach the constant. The wavelets are not sparse: at b6, final checkpoint, the finest
have a median participation ratio of about 30–45 images but ~1,100–1,400 entries above 8-bit resolution. Hence only
the first 64 per level in pivot order are shipped: a subset, not the most localized (pivoted QR ranks by residual norm
in scaling coordinates; at step 2,560, stem, scale 5 the shipped 64 have median participation ratio ~240 against ~102
for all 546). Exploratory: no second-network or held-out check.

## Spectral layout (added 2026-11-15)

Third explorer layout, "spectral": per snapshot, diffusion coordinates λ_kψ_k (k = 1..32) of
that snapshot's own lazy walk under rule 0 (kNN graph), then the same AlignedUMAP
(n_neighbors 15, min_dist .1, alignment .02) across sites and Procrustes across checkpoints
as the activation layouts. Modal stage `specembed` (`measure_v2.spectral_embed`, CPU,
about $0.07); outputs `umapspec_<step>.npy`. Display only.

## Dot (Euclidean) kernel and GGN comparison (added 2026-11-19)

Spec `SPEC_DOT.md` (GPT plan review, one revision). Tag **`r20s0d`**: every rule's 1 − cos replaced by
|x_i − x_j|²/2 of the unnormalized activations (global scale only; with unit rows it is exactly the cosine
pipeline; `test_dot.py`). Full schema-v2 export, validated (5,280 records). Gradient-side Grams per checkpoint
(`kernels_modal.py`, `test_kernels.py`): GGN ⟨H_i^{1/2}J_i, H_j^{1/2}J_j⟩_F (label-free), trace NTK, and the stored
loss-gradient cosine. Comparisons in `data/kernels/compare_{single,blocks,ggn}.json`. Spend ≈ $4.5.

**Verdict: keeping the norm makes things worse here, mostly because of hubness, and moves sites further from the GGN, not closer.**
- **Euclidean kNN-15 breaks in the conv sites (stem–b8).** Images near the probe mean become hubs: the max 15-NN
  in-degree is 860–1,700 of 2,000, against about 140 for cosine. Everyone's out-edges go to them, so the remaining images form
  near-disconnected mutual pairs. The leading modes then have participation ratio **2** (λ ≈ .98, flat), against
  hundreds of images for cosine. Agreement with cosine is about .00–.15 (r = 4) at stem–b5, and class-span agreement is about .01–.15.
  These are artefacts of the rule, not structure.
- **Perplexity 100 copes better:** agreement with cosine is .37–.79 (r = 4) in conv sites and ≥ .96 at b9/pool/logits
  (final step). Past 3–4 modes it also localizes. Class alignment is somewhat lower than cosine
  (median class-span r = 10: .22 vs .29).
- **Late sites (b9, pool, logits): the norm barely matters.** Agreement is .86–.97 under both rules. Here the
  norm is mostly confidence.
- **Full stem→logits average** (final step): agreement .80 (r = 4) under kNN and .89 under perplexity, with SEBA
  sets matching .67–.82. Averaging over sites dilutes the broken snapshots.
- **GGN / NTK:** against every *fixed* gradient-side reference (GGN-cos, GGN-dot, NTK-cos, NTK-dot,
  loss-gradient), Euclidean site spectra are **farther** than cosine at almost every site. Δagreement at r = 10 runs
  from −.1 to −.4 in the conv sites and is about 0 at the late sites. Linear CKA barely moves in mid sites and drops at
  b9–logits against GGN (e.g. pool .099 → .066 at the final step). The full-block average is also slightly farther
  (−.02 to −.04).
- **Raw dot ranking (maximum inner product), checked as a diagnostic:** it behaves like cosine, not like Euclidean
  (neighbour overlap .71 vs .17; span agreement with cosine .86–.92), with mild hubs (max in-degree about 205).
- **Arc-cosine:** skipped. In its angular form it ranks neighbours exactly as cosine does.
- **Caveats:** one seed; the inherited label-stratified probe; near-tied cutoffs make single fixed-rank spans
  noisy (gaps recorded in the JSON). The loss-gradient reference starts from the stored unit-gradient view, since
  reconstructing raw gradients gave ties.

Explorer: a **Kernel** switch (cosine / dot) under *more options*. The dot root `viewer/site/d_dot/` ships rules
0–1 only, because of size. The seed-1 layout is cosine-only, and the dot PCA layout is rotated onto the cosine one for display.

## Reproduce and validate

```bash
# From this directory; keep scratch off the VPS's quota-limited /tmp tmpfs.
export TMPDIR=/var/tmp/cm-pipeline
mkdir -p "$TMPDIR"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python test_coherent.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with torch --with numpy python test_product.py
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
# with the explorer's Kernel switch (dot-kernel root viewer/site/d_dot/, SPEC_DOT.md section 5), baselines and alignment:
uv run --with numpy --with pillow --with scipy --with scikit-learn python site/build_site.py r20s0b --layout-tag r20s1b \
    --baselines r20s0c --alignment ../spectral_align/data --dot-tag r20s0d
# deploy: copy viewer/site/ into an empty directory, add .nojekyll, commit as a single
# orphan commit on branch gh-pages and force-push (the branch holds only the built site)
```
Live at https://charlesr-w.github.io/CoherentModes/ .

The Depth embedding viewer that used to be a tab of this site is its own repo since 2026-11-21:
https://github.com/CharlesR-W/DepthEmbedding (site https://charlesr-w.github.io/DepthEmbedding/). It took the tab's
page and data export, the runners for the other networks and the PCA/UMAP baselines, and a copy of `spacetime*.py`.
This repo keeps `spacetime*.py` for the Through-depth experiment and the explorer's "through depth" colouring; the
old `#embedding` and `embedding.html` URLs redirect to the new site.
