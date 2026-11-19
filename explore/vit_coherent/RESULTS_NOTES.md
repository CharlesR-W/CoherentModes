# ViT side study: results notes (DINOv2 ViT-S/14, ImageNet-100, run `dinov2s`)

Exploratory. Nothing here enters `THEORY.md`. All numbers are from `data/dinov2s/` (key numbers reproduced
by `uv run --with numpy python -I explore/vit_coherent/summarize.py dinov2s`). N = 10,000 probe images
(100 per class), 10,000 held-out. Layers are CLS depths 1-12 unless "mp" (mean patch, 0-12). Labels were
used only for the validation numbers marked as such. Departures from PLAN.md are in `DEVIATIONS.md`.

Integrity checks: the §4 decomposition reconstructs h_l to max |error| 1.1e-13 (|h| up to 19.1); the view
identity v = J h + o holds to 6.3e-15; the float64 re-implementation matches the hub model's float32 readout to
1.3e-4 (TF32 off); attributions sum to cov(V w, f) to 2e-7 relative; identical-view and random-rotation
controls give |Delta| <= 3e-15.

## 1. Which kernel settings give stable, non-degenerate modes

Label-free rule (pre-registered, DEVIATIONS item 7), CLS view, layers passing both stability (top-16
split-half mean principal cosine >= 0.8) and non-localisation:

| kernel | layers passing | mean stab16 | note |
|---|---|---|---|
| vmf5 (kappa 5) | 12/12 | 0.965 | stab16 0.95-0.99 at L1-11, 0.81 at L12 |
| st7 | 11/12 | 0.925 | fails L12 (0.73) |
| st15 | 11/12 | 0.922 | fails L12 (0.60) |
| st30 | 11/12 | 0.902 | fails L12 (0.18); stab32 drops to 0.58-0.79 at L5-11 |
| vmf14 (kappa 1/0.07) | 1/12 | 0.388 | degenerate at L1-11 |
| vmf30 | 0/12 | 0.154 | degenerate everywhere |

- **Defaults: vmf5 (primary) and st7, kept together because the choice is not clear-cut** (one layer apart;
  st7/st15/st30 are within 0.02 of each other in mean stability, so "st7" stands for the self-tuned family).
- **vmf14 and vmf30 give the predecessor's "isolated points" failure.** At L1-11, P is close to the identity:
  eigenvalue 16 is >= 0.91, 6,000-10,000 eigenvalues exceed 0.1, and the top-16 projector lives on 16-19 images,
  i.e. each mode sits on about one image. vmf14 is the DINO kNN temperature. It becomes non-degenerate only at
  L12 (stab16 0.92), which fits a temperature tuned on the final layer's geometry.
- The self-tuned kernel's possible negative eigenvalues (the reviewer's PSD concern) are negligible in
  practice: the negative fraction of squared spectrum is <= 6e-4 (at L12) and <= 6e-6 elsewhere.
- alpha = 1 (comparison only) gives top eigenvalues within about 0.05 of Sinkhorn's at every layer. At early
  layers its modes are somewhat more localised (st7 L1: median effective size 878 vs 1593 images).
- Uncentred view (sensitivity): st7's top-16 subspace barely moves (mean cosine 0.78-1.0). vmf5's moves
  substantially at L1-6 (0.63-0.67), so for the global kernel, centring is a real choice at early layers.
- Mean-patch view: the same picture (self-tuned and vmf5 stable at nearly all layers, vmf14 and vmf30
  degenerate). Its stability degrades at the last layers for larger k (st30 L12 0.56).

## 2. Leading modes and clusters by depth (validation numbers are label-based)

- Spectra: for vmf5, eigenvalue 1 is 0.58-0.83 at L1-11 and **0.24 at L12**. About 31-41 eigenvalues exceed
  0.1 at all layers.
- Class alignment of the leading modes rises with depth. Class eta^2 of the top 8 modes (vmf5) goes from 0.13
  (L1) to 0.50 (L9), 0.69 (L11) and 0.81 (L12); a random function gives about 0.01. Kernel-weighted kNN accuracy
  on held-out images goes from 0.11 (L1) to 0.91 (L12).
- SEBA (vmf5): coverage at r = 16 is 0.96-0.99 at L1-11 but **0.40 at L12**; all r = 16 runs converged.
  Negative mass removed by clipping is 0.02-0.06 (r = 16) and 0.11-0.16 (r = 64). Clusters are a measured
  regime here: at L12 most images are not assigned at r = 16.

## 3. Smoothness across depth and resolution

- Smoothness of layer a's top-8 modes measured at layer b (t = 1) is band-diagonal: about 0.2-0.35 at b = a,
  falling with |a - b|. Two groups are visible: L1-2 (vmf5 S[1,2] = 0.20) and L9-11 (S between them
  0.17-0.22, while S from L9-11 to L1-4 is <= 0.08).
- **Layer 12's modes are rough everywhere, including at L12 itself** (<= 0.08). Layer 12's view is far more
  spread out than the others.
- **The resolution axis is degenerate above t = 4.** Every nonconstant eigenvalue is <= 0.83, so lambda^(2t)
  is about 0 at t = 16 and 64: all smoothness values, Delta eigenvalues and profiles at t = 16/64 are 0 up to
  rounding, and their eigenvectors are numerical noise. t = 4 already shrinks everything about 10x. Only t = 1
  (and weakly t = 4) carries information with these kernels. This is a consequence of the plan's t grid, not a
  finding about the network.

## 4. Endpoint smoothness changes: reproducible? localised?

- **Reproducible across disjoint halves at t = 1.** For vmf5, the cross-validated value / half-A eigenvalue
  has a median of 0.98 for the leading +/- mode over all 66 pairs, and no leading mode is unsupported. For st7
  the ratio is 0.85-0.90 (none unsupported). At t = 4, 37-39% of the 16 stored modes per pair are unsupported
  (small eigenvalues).
- **Against the noise control (10% isotropic noise, SNR 10):** under vmf5, noise produces |Delta| of 0.02-0.05,
  versus 0.11-0.27 for adjacent layers (0.65 at 11->12), so the adjacent changes are well above a
  non-computational perturbation. **Under st7, noise produces 0.18-0.30, as large as adjacent-layer Delta at L1
  and L6 (0.13-0.25).** st7's adjacent changes are sample-reproducible, but no larger than what a small
  perturbation of the view does to a 7-NN-scaled operator. This is an artefact risk for st7, and I would not
  read st7's adjacent-layer modes as computation.
- Largest single change: **11 -> 12, "stopped being smooth"** (vmf5 -0.65, cross-validated -0.65; st7 -0.56,
  cross-validated -0.52). "Became smooth" is largest at 8 -> 9 (st7 +0.41) and 9 -> 10 (vmf5 +0.27).
- Long blocks (t = 1): for the leading mode of 1 -> 12, the "became smooth" eigenspace aligns with 6 -> 12
  (mean top-4 principal cosine 0.99) and not with 1 -> 6 (0.27). The "stopped being smooth" eigenspace aligns
  with 1 -> 6 (0.99), and 6 -> 12 gives 0.63. The union of the two sub-blocks' spaces contains the long
  block's top-4 (0.99-1.00) in all four triples.
- Depth localisation: the largest adjacent step carries a median 43% (vmf5) / 44% (st7) of the path for blocks
  of three or more layers, so changes are spread over several layers. Net = path (no cancellation) for most
  modes. The exception is negative modes ending at L12 (6 -> 12: net -0.36, path 0.64; 9 -> 12: -0.34 vs 0.54),
  which first become smoother and then rougher.

## 5. Components (CLS-position decomposition, closed-form ridge fits)

- Fits are good: R^2 is 0.96-1.00 on the probe set and on Nyström-extended held-out images for the top-4
  modes at L1-11 (vmf5), and 0.87-0.90 at L12. st7 is worse at late layers (L12 R^2 0.55-0.66; one L10 mode
  has held-out R^2 -0.77).
- **Attribution at L1-5 is dominated by cancellation.** Fractions of the readout covariance assigned to heads,
  MLP and baseline have opposite signs and magnitudes above 1. The summed |MLP| fraction is 6-76, i.e.
  thousands of neurons cancel. Per-component credit there does not say who wrote the mode.
- From L6 on, attribution is concentrated and interpretable in kind: heads carry 0.2-1.0 and the MLP 0.2-0.8
  of the covariance, and the MLP block contributing most is the readout layer's own block. At L12 the heads
  dominate (0.76-0.99).
- The block-1 MLP is the top MLP contributor for layer 1-4 modes; it may be the large-norm early writes known
  in ViTs, but that is unverified.
- Neurons: each top mode has a neuron with |loading| 0.5-0.95, and hundreds of neurons per layer exceed 0.5
  on some top-16 mode. Neurons with AUROC > 0.9 for some r = 16 SEBA set number 29-311 per layer at L1-11 and
  605 at L12.

## 6. Coarse-state flows against the same-layer baseline (vmf5)

- For adjacent pairs, H(c_b | c_a) is 1.3-2.7 bits at r = 16. That exceeds the same-layer split-half
  baseline at every adjacent pair, but the baseline is itself large: 0.4-1.8 bits at r = 16 and 1.3-2.5 bits
  at r = 64. **Hard SEBA assignments are noisy**, so only differences well above about 1 bit mean anything.
- Across the whole network (1 -> 12, r = 16): H(c12 | c1) = 2.2 bits, H(c1 | c12) = 3.8 bits. The asymmetry
  goes the way of coarse late states merging early ones, but the L12 coverage of 0.40 makes the L12 side
  unreliable.
- The predecessor's mean operator retains large eigenvalues (0.47-0.64) for every declared block. Q = P_b P_a
  loses them fast for blocks reaching L12 (second singular value 0.06-0.10, versus 0.3-0.4 for early blocks).

## 7. Decodability by layer (held-out accuracy; closed-form ridge / DINO kNN)

| view | L1 | L4 | L8 | L10 | L11 | L12 |
|---|---|---|---|---|---|---|
| CLS ridge (100 classes) | 0.16 | 0.36 | 0.57 | 0.70 | 0.82 | 0.90 |
| CLS kNN | 0.10 | 0.23 | 0.41 | 0.57 | 0.77 | 0.91 |
| mean-patch ridge | 0.19 | 0.35 | 0.58 | 0.70 | 0.74 | 0.87 |
| mean-patch kNN | 0.12 | 0.20 | 0.42 | 0.57 | 0.62 | 0.82 |

Same depth l in both views (mean-patch at l = 0: ridge 0.07). CLS kNN at L12 is 0.91 for fine classes and
0.95 for superclasses. Mean-patch is about as decodable as CLS up to L10, then CLS pulls ahead at L11-12.
Inputs are upsampled from 160 px (DEVIATIONS item 1), so absolute numbers sit below published DINOv2 figures.

## What looks like an artefact

- t = 16 and 64 (degenerate; see §3). The viewer should grey them out or label them.
- st7's adjacent-layer Delta modes are comparable to the noise control.
- Early-layer attributions (cancellation).
- vmf5 may be stable partly because it is coarse. With kappa = 5 the kernel is broad, and its modes may be close
  to PCA directions; the predecessor's other failure mode. This was not measured. `extras.py` computes the
  principal cosines between each kernel's modes and the top PCs (minutes of CPU) but was not run.
- L12 behaves differently on every measure (spectrum, SEBA coverage, Delta into L12, attribution fit for st7).
  Under a global kernel, part of that is the declared resolution: a layer that spreads its views makes every
  function rougher. The self-tuned kernel, which is invariant to uniform rescaling, also shows it (st7 11 -> 12:
  -0.56), so it is not only scale.

## What surprised me

- The DINO-standard temperature (kappa = 14.3) gives an identity-like operator at 11 of 12 layers.
- Endpoint changes at t = 1 reproduce almost exactly across disjoint halves (vmf5 ratio 0.98).
- The single biggest event is a loss of smoothness at the last block, not a gain.
- The self-tuned kernel's PSD problem is a non-issue on this data.
