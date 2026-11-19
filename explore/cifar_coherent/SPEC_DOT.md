# Delta spec: norm-keeping (dot-product) kernel, GGN/NTK sample kernels, arc-cosine (exploratory)

Owner request (2026-11-19): repeat the ResNet-20 / CIFAR-10 coherent-set measurements with the dot product
*without* row normalization (keep the norm), compare to the existing cosine construction, offer it in the explorer
as a secondary option (eigenfunctions, layouts, everything), and compare both with a GGN (NTK-like) sample kernel.
Arc-cosine: low priority. Exploratory side study, like the rest of `explore/`; labels only validate.

Delta against `SPEC.md` / `README.md` (schema v2, tag `r20s0b`). Everything not mentioned is unchanged: probe
(2,000 images), 8 checkpoints, 12 sites, flattening, five rules, zero-diagonal symmetric scaling, M = (I+P)/2,
float64 spectra, SEBA, export, products, schema.

## 1. The dot kernel (`r20s0d`)

Every cosine rule is a function of the cosine distance d_ij = 1 − cos_ij, which on unit rows equals
|x̂_i − x̂_j|²/2. The dot variant replaces it with the half squared Euclidean distance induced by the same dot
product on the unnormalized activations:

  D_ij = (G_ii + G_jj)/2 − G_ij = |x_i − x_j|²/2,  G = X Xᵀ, X flattened activations,

divided by mean_i G_ii of the probe-centred activations (a single global scale per cell, reported; all five rules
are scale-invariant: kNN ranks, per-point perplexity, and the two calibrated global kappas). With unit rows D is
exactly 1 − cos, so the cosine pipeline is the special case. D is translation invariant, so probe-centring does not
matter. Implementation: pass c_eff = 1 − D wherever the code takes the centred cosine Gram (`final_kernel_bank`,
the retained-diagonal diagnostic), so no rule code changes.

Identity to test: for the global-kappa rules (3, 4), exp(κ G_ij) and exp(−κ D_ij) differ by a_i a_j with
a_i = exp(κ G_ii/2); after zero-diagonal symmetric scaling they give the same P. So "exponentiated dot product" is
not a separate rule. For kNN (rule 0) and per-point perplexity (1, 2) the raw dot product (maximum inner product)
is *not* a metric and is not used.

Pipeline stages for `r20s0d`: extract (operators, kappa/meta, own PCs), layer/time spectra, average/product
comparisons, product spectra, spectral layout (rule 0), seed-0 PCA layout (top-50 PCs of the centred **dot** Gram,
then the existing AlignedUMAP + Procrustes recipe), validation. Points, probe, curves, decodability and the
derivative fields are kernel-independent and copied from `r20s0b` (decodability is a validation fit with its own
fixed protocol). No seed-1 dot layout. New construction fingerprint (`kernel: dot-halfsqeuclid-meanG-scale`).

Pre-measured (local, kNN-15 only, step 64,000): Euclidean neighbours agree with cosine neighbours 21–62% per site;
in the conv sites b1–b8 Euclidean kNN has strong hubs (max in-degree 790–1,725 of 2,000 vs 76–253 for cosine;
in-degree correlates −0.4 to −0.5 with the centred norm, i.e. images near the probe mean are everyone's
neighbour) and lower class purity (b8: .43 vs .63). Symmetric scaling caps each image's total mass, so the
operator-level effect is the thing to measure, not assumed.

## 2. GGN and NTK sample kernels (analysis only, all 8 checkpoints)

Per checkpoint: per-sample parameter Jacobian of the logits J_i ∈ R^{10×P} (all trainable parameters, eval-mode BN,
torch.func), empirical NTK Θ_ij = J_i J_jᵀ (10×10 blocks), softmax cross-entropy Hessian in logits
H_i = diag(p_i) − p_i p_iᵀ (label-free). Kernels (N×N):

- **GGN**: K_ij = tr(H_i^{1/2} Θ_ij H_j^{1/2}) = ⟨H_i^{1/2} J_i, H_j^{1/2} J_j⟩_F: the dot product on the per-sample
  GGN-factor view (Σ_i of the view's outer products is the GGN). Symmetric PSD square root.
- **NTK (trace)**: K_ij = tr Θ_ij.
- **Loss-gradient** (existing `gradgram_<step>.npy`, cosine of g_i = J_iᵀ(p_i − y_i); uses labels via the training
  loss, allowed on the gradient side by [A3]); cosine only, f16 as stored.

Each Gram goes through rules 0 and 1 in both forms: cos (row-normalized) and dot (half squared distance as in §1).
Note: in GGN-dot, confident images (H_i → 0) collapse to the origin; the cos form removes this. Both reported.
Precision: J in float32, Θ accumulated in float64. Store the three Grams (float32) and their operators' leading
33 modes on the volume, not in the public data.

## 3. Arc-cosine (lowest priority, analysis only)

Angular arc-cosine kernels (order 0, or order 1 on unit rows) are monotone in cos, so the kNN rule is unchanged and
is not rerun. Run only the order-1 Cho–Saul kernel on centred, unnormalized activations,
k(x,y) = |x||y|(sin θ + (π − θ) cos θ)/π, through rules 0 and 1 via its induced half squared distance, single
snapshots only.

## 4. Comparisons (labels only in validation columns)

Agreement = mean squared cosine of principal angles between leading nontrivial spans (existing
`schema_v2.comparison`). Chance level for r modes is about r/N.

1. **Single snapshots** (8 steps × 12 sites × rules 0, 1): cos vs dot agreement at r = 4, 10; λ1..10 for both;
   kNN in-degree max / skew and the pre-scaling mass on the top 1% in-degree images; SEBA r = 10 class Jaccard
   (validation) and the agreement of each span with the 9-dim class-indicator span (validation).
2. **Blocks** (from `layer_*.npz` / `time_*.npz` of both tags): cos vs dot agreement and Hungarian-matched SEBA
   set Jaccard (r = 4, 8) for every block, rule 0 and 1; headline for the full stem→logits block.
3. **GGN / NTK / loss-gradient vs sites**: per step, for each site and each site kernel (cos, dot, arc-cos), agreement
   of the site's modes with each gradient-side kernel's modes (r = 4, 10, rules 0, 1), and linear CKA between the
   centred site Gram and each gradient-side Gram. Also full-block average modes vs GGN modes. The comparison of
   interest is whether dot sites sit closer to GGN than cos sites do.

Outputs: `data/r20s0d/` (schema v2) and `data/kernels/` (`compare_single.json`, `compare_blocks.json`,
`compare_ggn.json`, small). A short written verdict in `README.md`.

## 5. Explorer

Secondary switch "Kernel: cosine (default) / dot" behind the existing *more options* checkbox. The dot data is a
second data root (`d_dot/`) built by `build_site.py` from `r20s0d`, same formats; the switch changes the data root
and reloads views (sets, eigenfunctions, spectra, retention, products, PCA and spectral layouts). The seed-1 layout
is unavailable under dot. A short paragraph in Findings with the comparison headline. Size cap: the dot root ships
all five rules only if the total site stays under ~800 MB; otherwise rules 0–1 only.

## 6. Budget

Modal ≤ $15 total (estimate: dot pipeline ≈ $3.5, GGN/NTK ≈ $1, comparisons ≈ $0.5). No commits or pushes without
the owner's request.

## 7. Revisions after plan review (GPT-6.2 Sol, 2026-11-19)

1. **Fixed GGN reference.** The headline is a paired difference against a *fixed* reference: for each gradient-side
   Gram R (GGN-cos, GGN-dot, NTK-cos, NTK-dot, loss-gradient-cos), agreement(dot site, R) − agreement(cos site, R),
   reported for every R (the full site-form × reference-form grid). Gradient views follow the activation order:
   centre the features over the probe first (double-centring the Gram, HKH), then (cos form) normalize rows; rows
   with centred norm below 1e-8 × median raise. CKA uses the linear, double-centred Grams of the *feature-centred*
   views named explicitly: site-dot G, site-cos C, GGN K, NTK K, each normalized as stated.
2. **Raw dot product (maximum inner product) diagnostic.** Euclidean is the matched "no row normalization" variant
   and is labelled "dot (Euclidean)" in the explorer. Separately, rule 0 built from raw-dot ranking (top-15 by G_ij
   on centred activations, union, same scaling) at steps 2,560 and 64,000: neighbour overlap with cosine and
   Euclidean, in-degree hubness, and r = 4/10 span agreement with both. The exp(κG) ≡ exp(−κD) identity holds at a
   fixed κ; the calibrated rules 3–4 calibrate on D.
3. **Separate float64 analysis pass.** The schema exports 8 nontrivial modes and SEBA r ≤ 8, so the comparisons do
   not read the viewer files. A Modal analysis pass recomputes from the saved float64 operators (and averages for
   blocks) the leading 33 modes; span ranks exclude the constant, SEBA ranks include it; each comparison records
   λ_r − λ_{r+1} at its cutoff (both operators) so a near-tie is visible.
4. **Probe.** The 2,000-image probe was stratified by label (200 per class) for the whole study; this comparison
   inherits it for pairing with the existing results. Kernels and bases use no labels, but the sample was chosen
   with them. Disclosed; not re-probed.
5. **Explorer.** Two data roots (`d/`, `d_dot/`) built together; the builder must not delete the other root;
   every client cache and the explorer context are keyed by kernel; prose figures, baselines and alignment stay on
   cosine. Seed-1 layout and any unshipped rule are disabled under dot. Test cosine → dot → cosine on one block,
   with rapid switching.
6. **Budget gate.** Coordinator guard for `r20s0d` at $10. GGN: one-checkpoint pilot (peak memory, time) before
   fan-out; J (2,000 × 10 × 0.27M, float32 ≈ 22 GB) held on an 80 GB GPU, Grams by tiled matmul with float64
   accumulation. Arc-cosine is dropped first if the budget tightens.
