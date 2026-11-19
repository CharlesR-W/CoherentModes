# Smoothness across depth in a pretrained ViT (exploratory side study) — plan v2

Side study in `explore/`; not an `EXPERIMENTS.md` stage, no claim enters `THEORY.md` from here.
Purpose: give the owner a *feel* for what layer-wise diffusion modes, their clusters, and their change
across depth look like on a real network, with neurons/heads/weights linked to them, and find out which
kernel/normalisation settings give stable, non-degenerate objects. Predecessor: `explore/cifar_coherent`
(ResNet-20, flattened conv maps, N=2,000), which failed for a diagnosed reason: flattened maps have
concentrated cosine distances (15th NN at 0.83 vs typical 1.0), so every global bandwidth gave either PCA
or isolated points; and the block operator there was a *mean* of layer operators, whose modes are by
construction "what no layer changes".

Owner's fixed commitments (from repo CLAUDE.md): every object is an SVD/eigendecomposition of a
sample-estimable operator; nothing is trained on top of the studied model (closed-form ridge fits allowed,
called fits); only geometric assumption is the dot product (cosine) on a view; no labels/probes define a
kernel or basis (labels validate only); no sparsity premise (clusters are a measured regime);
architecture-specific parts are variants. Owner is explicitly undecided on kernel family, bandwidth rule
and density normalisation and wants a principled choice; owner's top priority is an inspectable viewer.

v2 changes follow the GPT plan review (v1 findings: alpha=1 breaks the shared measure; self-tuned kernel
is not PSD; resolution changes masquerade as operations; k-jitter is not a sampling null; the ideal merge
has no positive Delta eigenvalues; SEBA needs the constant mode; soft-membership entropy is confounded;
component panel mixed coordinate systems).

## 1. Model, sites, views

- Model: DINOv2 ViT-S/14 (`facebookresearch/dinov2`, `dinov2_vits14`, no registers), 12 blocks, width 384,
  6 heads, MLP width 1536, LayerScale. Pretrained, frozen. Chosen because (i) the model's own whole-image
  summary at every layer is one vector (CLS token), so no flattening; (ii) its standard evaluation is
  cosine kNN on L2-normalised CLS with temperature 0.07, i.e. a vMF kernel with kappa = 1/0.07, so
  "cosine" and a reference bandwidth come from standard practice rather than from us.
- Sites X: N = 10,000 images, ImageNet-100 (`clane9/imagenet-100`), 100 per class (50 val + 50 train,
  fixed seed), standard eval transform (resize 256, centre crop 224). Labels and a WordNet superclass map
  are stored for validation and display only. A disjoint 10,000 more train images (100/class) are held
  out for decodability testing and out-of-sample checks.
- View at depth l in {1..12}: v_l(x) = final-LayerNorm(h_l(x))[CLS], h_l the residual stream after block l
  (`get_intermediate_layers(norm=True)`, the model's standard intermediate readout); then centre over the
  probe set and L2-normalise. Layer 0 excluded for CLS (constant across images).
  Secondary view (variant, same pipeline): mean of patch tokens, l in {0..12}. Sensitivity: uncentred.
- **CLS and mean-patch are partial views of the network state.** CLS at l is not a function of CLS at
  l-1 (patch tokens write into it), so no data-processing monotonicity is expected of CLS-view objects;
  every statement is about the geometry of the view, not about information in the network.

## 2. Per-layer operator

d_ij^2 = 2 - 2 cos(v_i, v_j).

- Kernel candidates (each a deterministic function of the view with a layer-independent parameter, so
  identical views give identical operators):
  K-st(k) (self-tuned, Zelnik-Manor–Perona / Berry–Harlim): K_ij = exp(-d_ij^2 / (s_i s_j)), s_i = distance
  from i to its k-th nearest neighbour, k in {7, 15, 30}. Resolution is declared in neighbour count;
  invariant to a layer uniformly rescaling distances.
  K-vmf(kappa) (global): K_ij = exp(kappa (cos_ij - 1)), kappa in {1/0.07, 5, 30}. Resolution is declared
  in cosine units; a layer that uniformly contracts distances changes every function's smoothness.
  Layer-dependent bandwidth rules are not used.
- Normalisation: Sinkhorn symmetric bistochastic, P = D K D, P1 = 1 (THEORY §3.1): one shared (uniform)
  stationary measure across layers, required for anything that compares or multiplies operators.
  Coifman–Lafon alpha=1 is computed only as a separate per-layer comparison (its stationary vector is
  sqrt(degree), not constant); it is shown in the settings panel and cannot be the downstream default.
- K-st is symmetric and positive entrywise but not guaranteed PSD (reviewer counterexample), so P may have
  negative eigenvalues. All downstream objects therefore use the two-step operator
  **R_l = P_l^2** (= THEORY's Q^{l->l}), which is symmetric, bistochastic and PSD for any symmetric
  bistochastic P. The full spectrum of P is computed (dense eigh) and its negative part reported (min
  eigenvalue, sum of squares of negative eigenvalues / total) per layer and candidate.
- Resolution as an explicit axis: diffusion time t in {1, 4, 16, 64}, R_l^t = P_l^{2t} (from the full
  eigendecomposition).
- Label-free selection among candidates, reported for every candidate and layer:
  (a) split-half stability: build P on disjoint halves A, B (5,000 each); Nyström-extend the top-r
      nonconstant eigenfunctions of each to the 10,000 held-out images by the **Sinkhorn-consistent
      extension**: with P_ij = d_i K_ij d_j on the half, a new point x gets weights
      p(x, j) = K(x, x_j) d_j / sum_k K(x, x_k) d_k (for K-st, x's own s is its k-th-NN distance to the
      half), and psi(x) = sum_j p(x, j) psi(x_j) / lambda; modes with |lambda| < 0.05 are not extended.
      This one extension is used everywhere an out-of-sample value is needed. Principal cosines between the two extended subspaces,
      r in {8, 16, 32}.
  (b) localisation: effective number of images 1/sum_x psi(x)^4 per mode and for the top-r projector
      (THEORY §4.3).
  (c) spectrum: decay, gaps, negative part.
  Validation afterwards only: weighted-kNN class accuracy under each kernel, class eta^2 of modes.
  The downstream default is the candidate that is stable and non-localised at most layers; if none is,
  that is the finding. No selection on label-based numbers. All candidates' results are kept.
- Outputs per layer: top 64 nonconstant eigenpairs of P_l.
- Soft clusters (display + coarse states): for r states, SEBA on the span of the constant vector plus the
  top r-1 nonconstant modes, r in {8, 16, 32, 64}. Reported per (layer, r): gap after mode r-1, split-half
  stability at r-1, distance of the SEBA vectors from the input subspace before clipping, negative mass
  removed by clipping, coverage. Hard label = argmax membership; unassigned if max membership < 0.1 after
  scaling each vector to max 1.

## 3. Across depth: smoothness and endpoint smoothness-change modes

For a unit, centred function f on images: smoothness at layer l and time t,
s_l^t(f) = f^T R_l^t f = ||P_l^t f||^2 in [0,1].

- Smoothness profiles: S[a, i, b, t] = s_b^t(psi_i^a). At which depths and resolutions is each layer's
  leading distinction smooth.
- Endpoint smoothness-change operator: Delta_t^{a->b} = R_b^t - R_a^t (symmetric, annihilates constants).
  Top positive eigenfunctions: functions whose smoothness is most increased between the endpoints
  ("became smooth"); most negative: most decreased ("stopped being smooth"). These are *endpoint
  smoothness-change modes*, not identified operations: (i) they depend on the declared resolution (kernel
  parameter and t), shown side by side for K-st and K-vmf and all t; (ii) a block that rearranges and
  restores has Delta = 0. Exact identity: f^T Delta^{a->c} f = sum_{l=a}^{c-1} f^T Delta^{l->l+1} f. For each
  reported mode f of Delta^{a->c} the per-layer terms are shown, together with net change and path length
  sum_l |f^T Delta^{l->l+1} f| (path >> |net| exposes cancellation).
  Relation of long block to sub-blocks (owner's question): principal cosines between the leading +/-
  eigenspaces of Delta^{a->c} and those of Delta^{a->b}, Delta^{b->c}, for (a,b,c) in {(1,3,5), (1,6,12),
  (6,9,12), (3,6,9)}; and the per-layer profile above.
- Controls: identical-view control (Delta must be 0 to numerical precision); a random-rotation control
  (same view rotated: Delta = 0); a noise control (view plus small isotropic noise at a declared SNR) to
  show how much Delta a perturbation that is not computation produces.
- Sampling reproducibility (replaces v1's k-jitter): compute Delta on half A and on half B separately. For
  each leading eigenfunction phi of Delta_A (**endpoint-spectral extension reproducibility**): expand phi
  in the P_A eigenbasis of the chosen endpoint (layer b for positive modes, layer a for negative), keep
  eigenfunctions with |lambda| >= 0.05, extend each to half B by the Sinkhorn-consistent extension of
  Section 2 dividing by its own eigenvalue, recombine with the expansion coefficients, report the
  discarded projection energy, centre and normalise on B, and report its Rayleigh quotient under Delta_B
  next to its eigenvalue under Delta_A (cross-validated value; projection-dependent and labelled so),
  plus principal cosines of the extended leading subspaces. Modes whose cross-validated value is near zero are marked unsupported in
  the viewer. k-variation is reported separately as bandwidth sensitivity.
- Product operator (THEORY §4.1): Q^{a->b} = P_b P_a, top 32 singular triples, for the declared triples'
  pairs; and the predecessor's mean operator for comparison.
- Coarse states: hard SEBA assignments c_l(x) (incl. unassigned) give a contingency table between layers
  a and b; reported: row-normalised flow matrix, H(c_b | c_a), H(c_a | c_b), with a same-layer split-half
  baseline (assignments from half-A-built vs half-B-built operators, compared on held-out images) so that
  assignment noise is not read as depth change. Soft memberships are displayed but not used for
  entropies.

## 4. Linking to components (ViT-specific variant), in one coordinate system

Raw residual decomposition at the CLS position, biases included once:
h_l = e + sum_{l'<=l} [ sum_h A_{l',h} + b^attn_{l'} + sum_n M_{l',n} + b^mlp_{l'} ],
e = CLS embedding + position; A = head h's output to CLS after the output projection and LayerScale
(exact per head because the projection is linear); M_{l',n}(x) = g_{l',n}(x) * (LayerScale ⊙ W_out[:, n]),
g = post-GELU activation of MLP neuron n at CLS.

View map with frozen per-image scales: v_l(x) = J_l(x) h_l(x) + o_l(x), where
J_l(x) u = gamma ⊙ (u - mean(u)) / (sigma_l(x) rho_l(x)), sigma = the final LayerNorm's std of h_l(x),
rho = the row norm after centring, o_l(x) = (beta - m_l) / rho_l(x), m_l the probe mean. This is an exact
identity at the actual sigma, rho; the "frozen scale" approximation is only in treating J_l(x) as not
depending on the component being attributed. Component c's view-space contribution is J_l(x) c(x); the
offset o and the embedding/bias terms form a single baseline component. Reconstruction of v_l from the
sum is asserted numerically before any attribution is computed.

- Neuron as a function on sites: g_{l,n} in R^N; mode loading = cosine between centred g_{l,n} and psi
  (THEORY §7.1); cluster selectivity = AUROC of g_{l,n} for membership of a hard SEBA set.
- Direct attribution of a target function f (a mode of P_l, or a Delta mode evaluated at its smooth
  endpoint l): closed-form ridge readout w, f ≈ V_l w (a fit; lambda by GCV; R^2 on the probe and on
  Nyström-extended held-out reported). Contribution of component c: cov_x(w . J_l(x) c(x), f(x)); these sum
  exactly to cov(V_l w, f). Attributions are shown only with the fit's R^2 beside them.
- Weight-level, in raw residual coordinates: readout direction w_res = mean_x J_l(x)^T w; write cosine of
  neuron n = cos(LayerScale ⊙ W_out[:, n], w_res). (No weight-level "read" cosine: the MLP input passes
  through a per-image LayerNorm of the post-attention state and has no single global direction;
  the activation loading above is the read-side quantity.)
- Head attention maps CLS -> patches for a display subset of ~1,500 exemplar images.

## 5. Decodability by layer (owner's separate request)

Closed-form ridge one-vs-rest classifier (GCV lambda) and DINO-protocol weighted kNN (k=20, T=0.07), fit
on the 10,000 probe images, tested on the held-out 10,000, per layer, for CLS and mean-patch views; fine
(100) and superclass labels. Same two numbers on the predecessor ResNet-20/CIFAR-10 for flattened vs
spatially-averaged block outputs, to quantify what flattening cost there.

## 6. Tests of the Section 3 objects

**6a. Operator unit tests on explicitly planted geometry (asserted; no training).** Point clouds on a
sphere built by hand, 64 "fine" groups of 100 points with a map y: 64 -> 8.
 (i) identical views: Delta = 0.
 (ii) ideal merge (endpoint a: 64 separated clusters; endpoint b: 8 separated clusters by y): y-contrasts
     have |f^T Delta f| < 0.05; within-y between-fine-group contrasts have f^T Delta f < -0.8; no positive
     eigenvalue above 0.05. (At fine resolution a merge only removes smoothness.)
 (iii) resolution dependence of the same merge: endpoint a's 64 clusters placed so that clusters sharing y
     are not neighbours; at large t (cross-cluster mixing at a) the y-contrast span appears in the
     positive eigenspace (principal cosines > 0.9), while at t = 1 it does not. This is the planted form of
     "a distinction becoming smooth at coarse resolution although no information was added".
 (iv) rearrange-and-restore path a -> b -> a: endpoint Delta = 0, path length > 0.
 (v) telescoping identity to 1e-10; R PSD; SEBA with the constant recovers planted indicators
     (coverage > 0.95, purity > 0.95) and the hard-assignment flow of (ii) sends each fine group to its y
     state with mass > 0.9.
 Thresholds are fixed here before running; failures are reported and the object reconsidered.
**6b. Trained residual MLP (measured, not asserted).** 4 blocks, width 256, trained on y = (A + B) mod 8
with x = E_A[A] + E_B[B] + Gaussian nuisance. Here the view is the complete state. First measure whether
hidden layers merge (A,B) states at all (decodability of A, B, y per layer; cluster counts); then report
the Section 3 objects per layer and resolution, and whether conditional entropies of hard states are
monotone in depth. No recovery threshold: the network may keep A and B and the result is reported as
found.

## 7. Viewer (designed and built by an Opus 5.5 agent, `experiment-notebooks` skill)

Self-contained static page(s). Requirements:
- Layer selector; per layer an aligned UMAP of the 10,000 images, colourable by: mode value, cluster,
  class/superclass, a neuron's activation, a smoothness-change mode of a chosen block.
- Click a cluster or mode -> exemplar image strip (top / bottom / random-in-set), its smoothness profile
  over depth and t, its flows to adjacent layers, and the neurons/heads panel coloured by loading /
  selectivity / direct attribution (with fit R^2) and write cosine. Click a neuron or head -> UMAP
  coloured by its activation, its top images, its loadings on that layer's modes and clusters, attention
  maps for heads.
- Block view: pick (a, b), kernel, t -> became-smooth and stopped-being-smooth modes with image strips,
  per-layer terms (net vs path), cross-validated eigenvalue, component attribution.
- Settings panel: kernel-selection criteria per candidate and layer, controls, decodability-by-layer
  plot, unit-test and MLP results.
- Every object shown with its mathematical definition (compact, next to where it appears).

## 8. Data contract (compute -> viewer), under `explore/vit_coherent/data/<tag>/`

The compute author writes `SCHEMA.md` first (names, shapes, dtypes, definitions' parameter values) so the
viewer is built in parallel against it. Contents: `meta.json`; `thumbs/` sprite sheets (64 px) and labels;
per layer: modes (64 x N), eigenvalues (full spectrum summary), SEBA memberships and hard labels per r
with their diagnostics, UMAP, neuron activations at CLS (N x 1536 float16), neuron-mode loadings,
neuron-cluster AUROC, write cosines, head/neuron/baseline attributions for the layer's top modes; across
layers: smoothness profiles, Delta results for all a < b and all t for the default kernel (top/bottom 8
eigenpairs, per-layer terms, cross-validated values, attributions) and for declared pairs for the other
kernels, flows and entropies, Q results; `selection.json`, `controls.json`, `decodability.json`,
`unit_tests.json`, `mlp.json`, `attn_subset.npz`.

## 9. Compute and cost

Modal: one GPU job for activations (20,000 images), dense 10,000 x 10,000 float64 eigendecompositions per
layer and candidate on GPU; detached coordinator, results on a Volume. Expected well under $10. Unit tests
and MLP on CPU.

## 10. What is reported

A notebook with a verdict per question: which kernel settings give stable non-degenerate modes (or none);
what the leading modes/clusters are at each depth; how smoothness moves across depth and resolution;
whether endpoint smoothness changes are reproducible, localised in depth and in components; what the
coarse-state flows look like against their same-layer baseline; decodability by layer; unit-test and MLP
outcomes. Everything is labelled exploratory; nothing is selected on labels; all candidates are kept.
