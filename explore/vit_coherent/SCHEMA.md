# Data contract: compute -> viewer (`explore/vit_coherent/data/dinov2s/`)

Changelog (newest first):
- 2026-11-12 v2 (final run `dinov2s`): no format change. `kernels_full` = `default_kernels` = [vmf5, st7];
  total 778 MB (857 binaries, verified against `manifest.json`). Values at t = 16, 64 are ~0 (degenerate; see
  RESULTS_NOTES.md).
- 2026-11-11 v2: Delta-mode attributions moved from `delta.json` to binary `cls/<k>/delta_attr/` files
  (JSON would be ~20 MB per kernel); `selection.json.alpha1` has no `stability_mean`; added
  `selection.json` keys `stability_capped`, `scores`, `clear_cut`, `uncentred_sensitivity`; added
  `delta.json` keys `cv_evalA` order note, `all_vals_extreme`; `seba.json` key `baseline` moved to
  `flows.json.baseline`; `meta.json` gains `kernel_ids`, `delta_pairs`, `reconstruction`, `n_attr`.
- 2026-11-11 v1: first version, written before any compute.

All paths below are relative to `data/dinov2s/`. Everything is produced by `explore/vit_coherent/pipeline.py`.
`manifest.json` lists every binary file with `{path, dtype, shape, bytes}`; the viewer can rely on it
(and the compute side verifies it against this document).

## Conventions

- **Binary files** are raw, little-endian, C order (last index fastest), no header. Extension = dtype:
  `.f32` float32, `.f16` IEEE half (decode in JS with `Float16Array` or a 10-line decoder), `.u8` uint8,
  `.i8` int8, `.i16` int16, `.u16` uint16, `.i32` int32. Shapes are given as `[d0, d1, ...]`.
- **N = 10,000 probe images**, index `i` in `[0, N)`. Order: by class id `c` in `[0,100)` (classes sorted by
  WordNet id), and within a class the 50 ImageNet-val images first, then the 50 train images, each in the
  dataset's own order. So image `i` has class `i // 100`. The held-out 10,000 images are never shipped.
- **Layers.** View `cls`: depth `l` in `1..12`, array axis index `l-1`, file suffix `L01..L12`.
  View `mp` (mean patch): depth `l` in `0..12`, array axis index `l`, file suffix `L00..L12`.
  Block `l` (1-based) maps `h_{l-1}` to `h_l`; "neurons of layer l" are block l's MLP neurons (1536), "heads
  of layer l" are block l's 6 attention heads.
- **Kernels** (PLAN §2), ids used in paths: `st7`, `st15`, `st30` (self-tuned, k = 7/15/30 nearest
  neighbours), `vmf14` (vMF, kappa = 1/0.07 = 14.2857), `vmf5` (kappa = 5), `vmf30` (kappa = 30).
  `meta.json.kernels` gives family, parameter and formula. All operators are Sinkhorn bistochastic `P = DKD`.
- **Diffusion times** `t` in `[1, 4, 16, 64]`, axis index `ti` in `0..3`; `R^t = P^{2t}` (PLAN §2).
- **SEBA state counts** `r` in `[8, 16, 32, 64]`, axis index `ri`; memberships of all r are concatenated
  along one axis with row offsets `[0, 8, 24, 56]` (total 120 rows).
- **Mode vectors** (eigenfunctions, Delta modes, singular vectors) are unit-norm on the N probe images and
  are stored **multiplied by sqrt(N) = 100** (so RMS = 1). Sign convention: the entry of largest absolute
  value is positive. Constant mode is never stored (all stored vectors are orthogonal to 1).
- **Pairs** `(a, b)` with `a < b` over CLS layers 1..12, in row-major order: `(1,2),(1,3),...,(1,12),(2,3),...,(11,12)`
  (66 pairs); `meta.json.pairs` lists them. File suffix `a01_b02`.
- **Declared pairs** (all kernels): `(1,3),(3,5),(1,5),(1,6),(6,12),(1,12),(6,9),(9,12),(3,6),(3,9)`
  = the pairs of the triples `(1,3,5),(1,6,12),(6,9,12),(3,6,9)` (PLAN §3).
- **Default kernel(s)**: chosen label-free in `selection.json` (`default_kernels`, 1 or 2 ids; the first is
  primary). Kernels in `meta.json.kernels_full` (= the defaults) get the full product set marked [full];
  all other kernels get the reduced set marked [all].
- NaN is used for "not applicable" in float arrays (e.g. components at layers above the readout layer).

## Top-level files

| file | dtype / format | shape | content |
|---|---|---|---|
| `meta.json` | JSON | | see below |
| `manifest.json` | JSON | | list of `{path, dtype, shape, bytes}` for every binary |
| `images/class.u8` | uint8 | [N] | fine class id 0..99 (validation/display only) |
| `images/superclass.u8` | uint8 | [N] | WordNet superclass id (validation/display only; names in meta) |
| `images/source.u8` | uint8 | [N] | 0 = ImageNet val split, 1 = train split |
| `images/half.u8` | uint8 | [N] | split-half membership: 0 = half A, 1 = half B (random, seed 0, not stratified) |
| `thumbs/sheet_000.jpg .. sheet_099.jpg` | JPEG | 640x640 px | 10x10 grid of 64x64 thumbs (centre crop 224 resized to 64). Image `i` is on sheet `i // 100` at row `(i % 100) // 10`, column `i % 10`. Sheet `c` = class `c`. |
| `selection.json` | JSON | | PLAN §2 selection criteria per view/kernel/layer, alpha=1 comparison, label-based validation (separate key), chosen default with reasoning in words |
| `controls.json` | JSON | | PLAN §3 controls |
| `decodability.json` | JSON | | PLAN §5 (ViT part) |
| `weights/wout_ls.f16` | f16 | [12, 1536, 384] | `LayerScale_l ⊙ W_out[:, n]` for block l (axis 0 = l-1), neuron n: the raw-residual write direction of each neuron |
| `attn/idx.i16` | int16 | [1500] | probe indices of the attention-map exemplar subset (15 per class, random seed 0) |
| `attn/L01.u8 .. L12.u8` | uint8 | [1500, 6, 256] | CLS->patch attention of each head of block l, patches in 16x16 raster order (row-major, top-left first); value = round(255 * a / max_patches(a)) per (image, head) |
| `attn/max.f32` | f32 | [12, 1500, 6, 2] | per (layer, image, head): `[max patch weight, CLS->CLS weight]` (softmax weights over 257 tokens), so raw weights = u8/255 * max |
| `unit_tests.json`, `mlp.json` | JSON | | written by the PLAN §6 agents, not by this pipeline |

### `meta.json`
```
{ "tag": "dinov2s", "model": "dinov2_vits14", "N": 10000, "N_heldout": 10000,
  "views": {"cls": {"layers": [1..12]}, "mp": {"layers": [0..12]}},
  "kernels": [{"id": "st15", "family": "self-tuned", "param": {"k": 15},
               "formula": "K_ij = exp(-d_ij^2/(s_i s_j)), s_i = dist to k-th NN, d^2 = 2-2cos"}, ...],
  "kernels_full": ["..."], "default_kernels": ["..."],
  "t": [1,4,16,64], "seba_r": [8,16,32,64], "seba_offsets": [0,8,24,56],
  "pairs": [[1,2],...], "declared_pairs": [[1,3],...], "triples": [[1,3,5],[1,6,12],[6,9,12],[3,6,9]],
  "n_modes": 64, "n_delta": 8, "n_attr_modes": 16, "n_q": 32,
  "classes": [{"id": 0, "wnid": "n01...", "name": "...", "superclass": 3}, ...],
  "superclasses": [{"id": 0, "name": "...", "wnid": "n0..."}, ...],
  "mode_scale": 100.0, "noise_snr": ..., "seed": 0, "dataset": "clane9/imagenet-100", ... }
```

## Per view `<v>` in {`cls`, `mp`}

| file | dtype | shape | content | PLAN |
|---|---|---|---|---|
| `<v>/umap.f32` | f32 | [L, N, 2] | aligned UMAP (umap-learn AlignedUMAP, n_neighbors 15, min_dist 0.1, cosine metric on the view, identity relations between consecutive layers); L = 12 (cls) or 13 (mp) | §7 |
| `<v>/<k>/spectrum.f32` | f32 | [L, N] | **full** spectrum of P_l, descending (index 0 is the constant mode's eigenvalue 1) [all] | §2(c) |
| `<v>/<k>/spectrum_alpha1.f32` | f32 | [L, 65] | top 65 eigenvalues of the Coifman–Lafon alpha=1 operator (comparison only) [all] | §2 |
| `<v>/<k>/evals.f32` | f32 | [L, 64] | eigenvalues of the stored modes (= spectrum[:, 1:65]) [all] | §2 |
| `<v>/<k>/modes/Lxx.f16` | f16 | [64, N] | top 64 nonconstant eigenvectors psi_i of P_l, row i = mode i (eigenvalue descending), x sqrt(N) [all for cls; full only for mp] | §2 |
| `<v>/<k>/smooth.f32` | f32 | [L, 64, L, 4] | `S[a, i, b, ti] = s_b^t(psi_i^a) = psi^T R_b^t psi` (unit psi) [all for cls; full only for mp] | §3 |
| `<v>/<k>/seba/Lxx.u8` | u8 | [120, N] | SEBA memberships, rows by r with offsets [0,8,24,56]; each vector scaled to max 1, clipped at 0, x255 [full] | §2 |
| `<v>/<k>/seba/Lxx_hard.i8` | i8 | [4, N] | hard state per r (argmax membership), -1 = unassigned (max scaled membership < 0.1) [full] | §2 |
| `<v>/<k>/seba.json` | JSON | | `{"r", "offsets", "layers": [{"layer", "r": {"8": {...}}}]}`; per layer per r: `gap` (lambda_{r-1} - lambda_r of the nonconstant modes), `split_half_min_cos`, `split_half_mean_cos` (principal cosines at r-1), `dist_from_span` (Frobenius distance of pre-clip SEBA vectors from input subspace), `neg_mass` (fraction of L1 mass removed by clipping), `coverage` (fraction assigned), `sizes` [full] | §2 |
| `<v>/<k>/delta/aXX_bYY.f16` | f16 | [4, 16, N] | Delta_t^{a->b} = R_b^t - R_a^t eigenvectors, axis 0 = ti; rows 0-7 = top positive eigenvalues (descending, "became smooth"), rows 8-15 = most negative (ascending, i.e. most negative first, "stopped being smooth"); x sqrt(N) [full: all 66 pairs; all: declared pairs] | §3 |
| `<v>/<k>/delta.json` | JSON | | see below [same pairs as the vectors] | §3 |
| `<v>/<k>/flows/counts_rXX.u16` | u16 | [P, r+1, r+1] | contingency counts between hard states at a (rows) and b (cols) for every pair p in `meta.pairs` (cls) or all mp pairs; index r = unassigned [full] | §3 |
| `<v>/<k>/flows.json` | JSON | | `{"pairs", "r", "H_b_given_a": {"8": [P]}, "H_a_given_b": {...}, "MI": {...}, "baseline": {"8": [{"layer", "H_B_given_A", "H_A_given_B", "H_A", "H_B", "r_used"}]}}` (bits; unassigned counted as a state). Baseline = same-layer split-half (DEVIATIONS item 12) [full] | §3 |

For `mp`: pairs for flows and Delta are over layers 0..12 (78 pairs, row-major, listed in `mp/<k>/delta.json.pairs`);
`mp` Delta vectors are stored for the declared pairs only (default kernels only).

### `delta.json` (per view and kernel)
```
{ "pairs": [[a,b],...],            # pairs for which delta/aXX_bYY.f16 exists
  "results": { "a01_b02": {
      "evals": [[16 floats] x 4 t],                  # eigenvalues of Delta for the 16 stored rows
      "cv": [[16] x 4],                              # cross-validated value: Rayleigh quotient under Delta_B of the
                                                     #   half-A mode extended to B (endpoint-spectral extension)
      "cv_evalA": [[16] x 4],                        # its eigenvalue under Delta_A (for the matching row)
      "cv_discarded": [[16] x 4],                    # projection energy discarded (|lambda| < 0.05 at the endpoint)
      "cv_subspace_cos": [[ [8 cos +], [8 cos -] ] x 4],  # principal cosines of extended leading +/- 8-subspaces (A->B vs B's own)
      "unsupported": [[16 bool] x 4],                # cv < 0.25 * |evalA| or sign flip
      "terms": [[[11 floats] x 16] x 4],             # f^T Delta_t^{l->l+1} f for l = 1..11 (index l-1); only l in [a, b-1] are part of the identity, others are given for context
      "net": [[16] x 4], "path": [[16] x 4],         # sum and sum of |.| of terms over l in [a, b-1]
      "smooth": [[[12] x 16] x 4],                   # s_l^t(f) for every layer l (cls: 12, mp: 13)
      "all_vals_extreme": [[min, max] x 4],          # extreme eigenvalues of the full-probe Delta
  }},
  "triples": [{"abc": [1,3,5], "t": 1, "sign": "+",
               "cos_ac_ab": [8], "cos_ac_bc": [8], "cos_ac_sum": [8]}, ...],  # principal cosines between leading 8-dim eigenspaces
  "controls": {...} }
```

## CLS-only, kernel-independent (PLAN §4)

| file | dtype | shape | content |
|---|---|---|---|
| `cls/neurons/Lxx.u8` | u8 | [N, 1536] | post-GELU activation g_{l,n}(x) of block l's MLP at the CLS position, quantised per neuron: `g = lo + u8/255 * (hi - lo)` |
| `cls/neurons/range.f32` | f32 | [12, 1536, 2] | per (layer, neuron): `[lo, hi]` = min and max of g over the probe set |
| `cls/neurons/stats.f32` | f32 | [12, 1536, 2] | per (layer, neuron): `[mean, std]` of g over the probe set (full precision) |
| `cls/heads/Lxx.f16` | f16 | [6, N] | `||A_{l,h}(x)||_2`, norm of head h's output to CLS (after output projection and LayerScale), raw residual coordinates: the head's scalar function on sites for colouring |

## CLS, per default kernel `<k>` in `kernels_full` (PLAN §4)

| file | dtype | shape | content |
|---|---|---|---|
| `cls/<k>/load/Lxx.f16` | f16 | [64, 1536] | neuron-mode loading: cosine between centred g_{l,n} (block l neurons) and psi_i^l |
| `cls/<k>/auroc/Lxx.f16` | f16 | [120, 1536] | AUROC of g_{l,n} for membership of hard SEBA set j (rows as seba offsets); NaN if the set is empty |
| `cls/<k>/attr/Lxx.f32` | f32 | [16, 1 + 72 + 12*1536] | direct attribution of layer l's top 16 modes. Column 0 = baseline (embedding + all biases + offset o), columns 1..72 = heads (block l' head h at `1 + 6(l'-1) + h`), columns 73.. = neurons (block l' neuron n at `73 + 1536(l'-1) + n`). Value = cov_x(w·J_l(x)c(x), f(x)) / cov_x(V_l w, f), so each row sums to 1. NaN for blocks l' > l. |
| `cls/<k>/writecos/Lxx.f16` | f16 | [16, 12, 1536] | write cosine cos(LayerScale ⊙ W_out[:, n] of block l', w_res) for each of the 16 modes; NaN for l' > l |
| `cls/<k>/attr.json` | JSON | | per layer: `r2_probe` [16], `r2_heldout` [16], `ridge_lambda` [16], `total_cov` [16], `w_res` [16][384] |
| `cls/<k>/q.json` | JSON | | per declared pair: `sv` top 32 singular values of Q^{a->b} = P_b P_a; `mean_evals` top 32 nonconstant eigenvalues of the predecessor's mean operator (1/(b-a+1)) sum_{l=a..b} P_l; `cos_q_modes` principal cosines between Q's top-8 right singular subspace and P_a's top-8 modes, and left vs P_b's |

**Attributions of Delta modes** (cls, kernels_full), one pair of files per pair in `delta.json.pairs`.
Each of the 16 rows of each t is attributed at its smooth endpoint (rows 0-7 at layer b, rows 8-15 at layer a):

| file | dtype | shape | content |
|---|---|---|---|
| `cls/<k>/delta_attr/aXX_bYY.f32` | f32 | [4, 16, 480] | per (ti, row): col 0 = baseline fraction; cols 1..72 = head fractions (block l' head h at `1 + 6(l'-1) + h`); cols 73..84 = per-block MLP total fraction (block l' at `72 + l'`); cols 85..468 = `w_res` (384, raw-residual readout direction; write cosine of neuron n of block l' = cos(`weights/wout_ls[l'-1, n]`, w_res)); col 469 = R^2 probe; 470 = R^2 held-out; 471 = total_cov; 472 = readout layer l; 473 = ridge lambda; 474..479 unused (NaN). Fractions are of total_cov = cov(V_l w, f); NaN for blocks above l. |
| `cls/<k>/delta_attr/aXX_bYY_top.f32` | f32 | [4, 16, 32, 2] | the 32 neurons with largest |fraction|: `[col, fraction]`, col = `1536(l'-1) + n` (stored as float, exact) |

## JSON summaries

`selection.json`:
```
{ "views": { "cls": { "<k>": { "per_layer": [ {           # one per layer
        "stability": {"8": [8 cos], "16": [...], "32": [...]},   # principal cosines (desc) between half-A and half-B top-r
                                                                 # extended subspaces on the held-out set
        "stability_mean": {"8": x, "16": x, "32": x},
        "n_extended": {"A": x, "B": x},                           # modes with |lambda| >= 0.05 (r capped by this)
        "eff_n_modes": [64],                                      # 1/sum_x psi(x)^4 per mode (unit psi)
        "eff_n_proj": {"8": x, "16": x, "32": x},                 # same for the top-r projector, see DEVIATIONS.md
        "spectrum": {"lambda": [first 65], "min": x, "neg_frac": x,   # sum lambda_-^2 / sum lambda^2
                     "gaps": [64], "decay_r": {"0.5": n, "0.1": n}},  # number of eigenvalues above 0.5 / 0.1
        "stability_capped": {"8": bool, ...},                   # true if fewer than r modes were extendable
        "alpha1": {"lambda": [65], "eff_n_modes": [64]}        # symmetric alpha=1 operator; index 0 = sqrt(degree)
      } ] } }, "mp": {...} },
  "validation": { "cls": { "<k>": { "knn_acc": [L], "eta2": [L][64] } }, "mp": ... },   # label-based, after the fact
  "default_kernels": ["..."], "reasoning": "words (includes the rule)", "clear_cut": bool,
  "scores": {"<k>": {"pass": [12 bool], "score": n, "mean_stab16": x, "n_stable": n, "n_nonlocal": n}},
  "order": [k ranked], "uncentred_sensitivity": {"<k default>": [{"layer", "cos16_vs_centred", "eff_n_proj16", "lambda"}]} }
```
`controls.json`: identical-view, random-rotation and noise controls (max |eigenvalue| of Delta and of
`R_b - R_a` per kernel and t; noise SNR) for representative layers.
`decodability.json`: `{ "cls": {"ridge": {"fine": [L], "super": [L]}, "knn": {...}, "ridge_lambda": [...]},
"mp": {...} }` test accuracy on the held-out 10,000.
