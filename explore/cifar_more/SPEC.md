# Coherent-set explorers for two more networks: spec (delta against `../cifar_coherent/SPEC.md`)

Owner request (2026-11-13, dictated): "do a similar thing [to the ResNet-20/CIFAR-10 report] for CIFAR100 and/or
imagenet; scale architecture appropriately - fewer checkpoints needed. Separately please do one with VGG on Cifar10.
No need to do multiple seeds - I mainly want to look at the 'explorer' with all the UMAPs etc. so we can pare things
down to be reasonable. Also idk how expensive it is to find higher eigenmodes but can we go for more? Like for
CIFAR10 we of course want at least 10, but I'd be interested to see more if that isn't nonsensical."

Exploratory side study, not an `EXPERIMENTS.md` stage. One seed per model. The owner's standard: the eigenfunctions
are the object; SEBA sets are an interpretive layer on top.

## 0. Fixed commitments (owner and repo `CLAUDE.md`)

- Nothing is trained on top of the studied model inside the construction; every construction object ends in an
  eigendecomposition of a sample-estimable operator. Ridge decodability is a closed-form *validation fit*, labelled so.
- Labels only colour and validate. No label, class, or probe enters a kernel, a basis, the probe-subset split of the
  stability check (section 5), or any choice of rule. CIFAR-100 superclasses are a display grouping only.
- Only geometric assumption: cosine on centred activations. No sparsity premise.
- Construction identical to `../cifar_coherent` rules 0 and 1 (below), imported from its modules, not rewritten.
- `../cifar_coherent` outputs, tests and its built site stay byte-for-byte unaffected. ImageNet is out of scope
  (no affordable training trajectory). One seed, no layout-only second seed.
- Working vocabulary: sites, views, operators, blocks, measured regimes. Not the `THEOREMS.md`/`SPARSITY.md` terms.

## 1. Two studies

| | A: `r18c100s0` | B: `v16s0` |
|---|---|---|
| Data | CIFAR-100 (python tarball from cs.toronto.edu, fine + coarse labels) | CIFAR-10 (zoo Volume `crw-modelzoo-train`) |
| Model | ResNet-18, standard CIFAR variant (3x3 stride-1 stem, no max-pool; kuangliu/pytorch-cifar definition), 100-way | ModelZoo `vgg16/relu-seed0` (VGG-16-BN), unchanged recipe |
| Training | own script (ModelZoo has no CIFAR-100/ResNet-18 and must not be modified): SGD 0.1, momentum .9, Nesterov, wd 5e-4, batch 128, 60 epochs (23,400 steps, drop_last), cosine to 0, pad-4 crop + flip, bf16 autocast for training only; checkpoints on the zoo schedule (0..16, 32..1024 powers of 2, every 2%); train-subset and test loss/acc at every checkpoint | `zoo.train.train_run` exactly as `../cifar_coherent/modal_train.py` does for resnet20, run id parameterised; 64,000 steps |
| Explorer checkpoints (T = 5) | 0, 256, 2,340 (6 epochs), 9,360 (24 epochs), 23,400 | 0, 256, 2,560, 25,600, 64,000 (a subset of the ResNet-20 report's checkpoints) |
| Probe | N = 5,000 CIFAR-100 test images, 50 per fine class, `RandomState(0)` per class as in the reference, sorted | exactly the reference 2,000-image CIFAR-10 probe (same indices; asserted equal to `r20s0b/probe.npz`) |
| Sites (S = 11) | stem (ReLU(BN(conv1))), b1..b8 (the eight BasicBlock outputs), pool (global average, 512), logits (100) | ReLU outputs of convs c1, c2, c4, c6, c7, c9, c10, c12, c13 (all 13 convs except the first conv of stages 2–5: c3, c5, c8, c11), pool (512, after the fifth max-pool, the classifier input), logits (10) |
| Colouring labels | 20 CIFAR-100 superclasses (the dataset's own coarse labels) by default; fine labels as a second colouring and in hover | the 10 CIFAR-10 classes |
| Eigenfunctions stored | K = 64 non-trivial | K = 48 non-trivial |
| SEBA ranks r (incl. ψ0) | 2, 4, 10, 20, 30, 40 | 2, 4, 6, 8, 10, 12, 16, 20, 24 |

## 2. Construction (unchanged; imported)

Per cell (checkpoint t, site s): `coherent.centred_cosine_gram` of the flattened activations; two kernel rules exactly
as `sweep_math.final_kernel_bank` rules 0 and 1:

0. **union 15-NN graph** (default): weight 1 on the union of directed 15-NN edges, 1e-6 elsewhere, zero diagonal.
1. **per-point perplexity 100**: `sweep_math.perplexity_logkernel(c, 100)`.

Edge case the reference never met (found by the smoke run, where a diverged 60-step VGG maps hundreds of probe
images to identical activations): a row whose tied-nearest set (cosine distance within 1e-9 of its minimum, well above float32 noise) has ≥ 100
members cannot reach perplexity 100 at any temperature. Such rows take the β → ∞ limit (uniform over the tied
nearest images); every other row is calibrated exactly as the reference (per-row bracketing and bisection; the
bracket allows 60 doublings instead of 40 so rows with near-ties still bracket; the kernel is bit-identical to the
reference whenever no row is affected; tested). Counts per cell are recorded in
`meta.perplexity_unattainable`; κ quantiles are over the calibrated rows.

(The graph rule's 4 lines are replicated so the three unused rules — including the expensive retained-diagonal
diagnostic — are not computed; a test asserts equality with `final_kernel_bank` rule 0 and 1 on a fixture.)
Then `coherent.symmetric_balance` (float64 log-domain, row residual < 2e-12 or fail), M = (I + P)/2,
block operator Mbar = mean of the block's cells' M, `coherent.block_modes(..., shared_stationary=True)` (Householder
deflation of the constant), SEBA via `seba.seba(..., return_transform=True)`, retention ψᵀ M_c ψ for every cell
of the block's row/column, int16/32767 export. Generalised only in the counts K and r (a generalised copy of
`schema_v2.block_record`; a test asserts it reproduces `schema_v2.block_record` exactly at K = 8, r = [2,3,4,6,8]).
`meta.json` keeps `kappa/kappa_q10/kappa_q90` (null for rule 0, per-point quantiles for rule 1) and `nn_mass`.

Block families as before: all layer pairs (a ≤ b, P_L = 66) at each checkpoint; all time pairs (c ≤ d, P_T = 15)
at each site.

## 3. Layouts and per-image data

- UMAP exactly as the reference (`umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, random_state=0,
  alignment_regularisation=0.02)` across the S sites of a checkpoint, on the top-50 PCs of the centred normalised
  activations; centre, unit-RMS and Procrustes across checkpoints at build time). Display only.
- `points_<step>.npz`: `loss`, `pred`, `correct` (eval mode, no augmentation). No channels, derivatives, gradient Grams.
- Decodability: final checkpoint only, both models, protocol of `../cifar_coherent/decodability.py` with the class
  count parameterised (10 / 100 fine classes; 10k train images, balanced, `RandomState(0)`; all 10k test images;
  fit-mean centring and row normalisation; closed-form ridge with GCV λ; weighted cosine 20-NN), views flat and GAP.
  Validation fit only. Dropped first if it threatens budget.

## 4. Dropped relative to the reference

Second seed, multi-step decodability, per-sample derivatives and gradient Grams, the 19-rule sweep, rules 2–4, the
normalisation and average/product comparisons, channel loadings, pixel statistics, the Findings/Measurements analysis
pipeline (`analysis.py`, `make_figures.py`).

## 5. New: per-mode stability (is mode k reproducible, or past the horizon?)

Label-free split with a shared core. A fixed permutation of the probe positions (`RandomState(20261113)`, no labels)
is cut into three parts of sizes ⌊N/3⌋, ⌊N/3⌋, rest: core C, A', B'. Subsets S1 = C ∪ A', S2 = C ∪ B' (A: 3,333
images each, |C| = 1,666; B: 1,333 each, |C| = 666). For each subset independently and for every cell, the
centred cosine Gram is recomputed from the subset's activations (centring over the subset), the same kernel rule is
built among the subset's images only (15-NN / perplexity 100 among them), scaled and made lazy exactly as above.
For every block (same family, pair and rule) the subset block operators give eigenvectors ψ^(1)_k, ψ^(2)_k,
k = 1..K. Compare on the shared core: U^(i) = rows of ψ^(i) at C, each column centred over C and normalised.

- **Leading-span reproducibility** (the primary number): for k = 1..K+16 (a guard band of 16 extra subset modes),
  span_k = min cos² of the principal angles between span{U^(1)_1..k} and span{U^(2)_1..k} (orthonormalised).
  A cut at k is *reproducible* if span_k ≥ 0.5. Individual ψ_k inside a near-degenerate group are not canonical,
  so cuts inside such a group are expected to fail even when the whole group is reproduced; only spans at group
  boundaries are tested meaningfully, and the measure needs no explicit grouping rule.
- **Horizon** of a block: h = the largest k ≤ K+16 with span_k ≥ 0.5 (0 if none). Modes 1..h lie in a reproducible
  leading span; h > K is shown as "beyond the stored modes". Summarised per model: median and range over
  single-snapshot blocks per site at the final checkpoint, over all layer blocks, over all time blocks, per rule.
- Also stored, descriptive only: cum_k = mean cos² of the same principal angles, and the windowed index agreement
  local_k = ½[‖Q^(2)_{W(k)}ᵀ u^(1)_k‖² + ‖Q^(1)_{W(k)}ᵀ u^(2)_k‖²], W(k) = {j : |j − k| ≤ 2} (it under-reads
  inside groups wider than 5; it is not used for the horizon). Chance level of span_k for random subspaces is
  ≈ 0 (≤ k/|C| for the mean).
- Subset eigenvalues λ^(1), λ^(2) (K+17 each) are stored for eigengap context.

Caveats stated on the page: this is reproducibility at sample size 2N/3 under resampling of the probe, not a
guarantee for the full-N vector shown; a reproducible mode can still be a property of the probe's image statistics
rather than the network (as for any mode); the 0.5 threshold, guard 16 and window 2 are fixed here, before any real data. Synthetic checks: a planted
5-dim cluster span gives h = 5 with span_5 = 1.0 while cuts inside it fail; an 11-dim symmetric group with K = 8
gives h = 11 via the guard band while local_k < .5 for most of its modes; an isotropic cloud gives h = 0.

## 6. Data contract (schema_version 3; `data/<tag>/`)

| file | content |
|---|---|
| `meta.json` | `schema_version: 3`, `construction_fingerprint` (reference construction contract + K, ranks, stability parameters), `model`, `dataset`, `steps`, `sites`, `modes` (2 names), `kappa`, `kappa_q10`, `kappa_q90`, `nn_mass` [M][T][S], `seba_r`, `n_vecs` = K, `layer_pairs`, `time_pairs`, `n`, `classes` (colouring classes), `fine_classes` (A only), `stability` {seed, core, subset sizes, window, threshold}, `default_mode` 0, `rule` text |
| `layer_<step>.npz`, `time_<site>.npz` | as schema v2 with K and R = len(seba_r), rmax = max(seba_r): `vals (M,P,K+1) f32`, `vecs (M,P,K,N) i16`, `scale (M,P,K) f32`, `seba_label (M,P,R,N) i8`, `seba_rot (M,P,R,rmax,rmax) f32`, `seba_perm (M,P,R,rmax) i8`, `retention_psi (M,P,K,X) f32`, `retention_seba (M,P,R,rmax,X) f32` (NaN pad); new `stab_span (M,P,K+16) f32`, `stab_cum (M,P,K+16) f32`, `stab_local (M,P,K) f32`, `sub_vals (M,P,2,K+17) f32`, and the browser's SEBA amplitudes from the float64 vectors: `seba_val (M,P,R,N) u8` = round(255·s_label(i)) (0 if unassigned), `seba_lead (M,P,R,N) u8` = round(255·(s_label(i) − max of the other sets at i)) |
| `points_<step>.npz` | `loss (N) f32`, `pred (N) i8`, `correct (N) bool` |
| `probe.npz` | `idx`, `labels` (colouring labels), `fine` (A only), `images (N,32,32,3) u8` |
| `umap_<step>.npy` | `(S,N,2) f32` |
| `curves.json` | run manifest, `checkpoints` with step, train/val loss/acc |
| `decod.json` | `{step, sites, views, results: {view: {site: {ridge_acc, knn_acc}}}, protocol, validation_only}` |
| `validation.json` | structural checks and minimum SEBA-label reconstruction agreement (≥ 99% required) |

## 7. Pipeline (Modal, `modal` skill; detached coordinator; Volume `cm-cifar-coherent`, new paths)

`/runs/<run>` (checkpoints), `/data/cifar100`, `/analysis/<tag>` (public schema), `/work/<tag>` (scratch, timing
records). Units: `train_c100` (L40S), `train_vgg` (L40S, zoo); `extract(step)` (A100-40GB): activations, the
full and two subset centred cosine Grams per site (float64, scratch), top-50 PCs, points, probe; `embed(step)`
(CPU): AlignedUMAP; `blocks(family, index)` (A100-40GB, float64): rebuilds the cells' operators from the Grams
(full and subsets, both rules), every block record and its stability; layer units also write the per-cell kernel
metadata; `decode` (final checkpoint); `validation`. Completion records carry the construction fingerprint; reuse
requires an exact match (`construction.cached_completion`). A spend guard pauses the coordinator at $16 of
estimated compute (timings × Modal rates checked 2026-11-13).

Budget enforcement (ceiling $20): every unit writes its timing record in a `finally` (failed attempts count), plus
60 s assumed startup per unit. Before submitting a stage the coordinator *reserves* its forecast cost
(units × 1.5 × the pilot-measured seconds per unit, + startup) and refuses the stage if completed + reserved
compute would exceed $18; `max_containers` bounds concurrency. Order: (1) smoke: B with T = 2 checkpoints through
to a site build; (2) pilot: the real A extraction of one checkpoint plus its layer unit and one A time unit, timed
(kept, not discarded) — this measures the dominant cost (dense float64 eigensolves at N = 5,000 and SEBA up to
r = 40) before the full launch; (3) the rest. Fallback order if the forecast does not fit: drop decodability; drop
A's rule-1 time blocks; reduce A's K from 64 to 48. Scratch Grams deleted after validation.

## 8. Site

A generalised copy of `../cifar_coherent/site/` in `site/` (the reference site sources stay untouched), output
`viewer/<tag>/`. Browser precision (explicit): ψ_k ships as int8 round(127 ψ/max|ψ|) with the reference's non-zero
floor, for display only (41 colour levels; hover to 2 decimals; the builder asserts max error ≤ 0.5/127 of max|ψ|
plus the floor). SEBA labels ship exactly as computed; SEBA amplitudes ship as `seba_val`/`seba_lead` (u8, from
float64); the browser never reconstructs SEBA vectors, so sets, draw order and representative images (build time,
from the archive) agree with the stored labels by construction. Generalised: K and ranks from meta; SEBA labels one byte (r > 14); class count and names from
meta; CIFAR-100 colour-by superclass (20 hues) and fine class (superclass hue, five shades); a set palette long
enough for r = 40 (first 8 = reference); spectrum panel λ_1..λ_K with the per-mode stability strip and the block's
horizon marked; ψ selector 1..K with past-horizon modes flagged. Tabs: Overview (what model, data, checkpoints,
rules; link to https://charlesr-w.github.io/CoherentModes/ for method), Explorer, Notes ("first look"
observations), Measurements (training curve, final decodability, stability summary). Budget: whole site ≤ 400 MB;
a typical interaction fetches ≤ ~1 MB (A: one block chunk ≈ 0.3 MB gzipped). Visual check at 1440 and 390 px
with headless Chrome. Published with `publish-results`; not pushed, not committed.

## 9. Tests (local, synthetic, before any paid run)

1. Generalised `block_record` equals `schema_v2.block_record` at K = 8, reference ranks.
2. Kernel bank (rules 0, 1) equals `final_kernel_bank` rules 0, 1.
3. Stability: a planted K-mode structure is reproduced (stab ≈ 1 for planted modes, low past them); a pure-noise
   cloud gives low stab past the first few; the split is label-free and deterministic.
4. SEBA labels reconstruct from the stored transform on ≥ 99% of points at r = 40, N = 5,000 (int16) [passed: 100%].
   Stability: planted span, wide degenerate group crossing K, isotropic noise (section 5) [passed].
5. Fixture end to end through the site builder.
Adversarial code review (`model-router:adversarial-code-reviewer`) before the paid launch.

## 10. Budget

Ceiling $20 for both studies. Estimate: training VGG-16 (~40 min L40S) ≈ $1.5, ResNet-18 C100 (~30 min) ≈ $1.1;
extraction ≈ $0.7; blocks A ≈ $3.5, B ≈ $0.6 (A100 float64 eigensolves: 990 full + 1,980 subset per model); UMAP
≈ $0.5; decodability ≈ $0.3; smoke and overhead ≈ $3. Total ≈ $11.
