# Spectral alignment of layers, checkpoints and networks (exploratory side study)

Owner request, 2026-11-18. Not an `EXPERIMENTS.md` stage. Labels colour and validate only; nothing is trained on
top of the networks; every compared object is an eigendecomposition of a sample-estimated operator. Linear CKA
appears as a labelled baseline, never as an output of the method.

## Questions

1. **Within one network** (ResNet-20, CIFAR-10, the main explorer's run): how far apart are the single-layer image
   graphs (a) of different layers at the same checkpoint and (b) of the same layer at different checkpoints?
2. **Across networks of different depth** (ResNet-20/32/56, same recipe): at the end of training, which layer of
   one network matches which layer of another, and does a monotone alignment (dynamic time warping, DTW) recover
   the correspondence the architecture suggests (stages, relative depth)?
3. Does the number of eigenmodes k matter for either answer?

Only **single-layer** operators are compared. No multi-layer blocks (owner, 2026-11-18).

## Objects

- **Networks.** ResNet-(6n+2), n = 3, 5, 9, seeds 0 and 1: six networks. ResNet-20 seeds 0 and 1 already exist
  (`cm-cifar-coherent:/runs/resnet20-relu-seed{0,1}`); ResNet-32/56 are trained by `train_modal.py` with
  ModelZoo's unchanged resnet20 config and loop, only the block count differing (`nets.py`). 64,000 steps each.
- **Checkpoints.** The explorer's eight: 0, 64, 256, 1024, 2560, 6400, 25600, 64000, for every network. Plus, for
  each ResNet-32/56 run, one **iso-accuracy** checkpoint s*: the earliest saved checkpoint whose test accuracy is
  at least the final test accuracy of the same-seed ResNet-20.
- **Sites.** stem, every BasicBlock output, pool, logits: 12 / 18 / 30 sites.
- **Probe.** The explorer's 2,000 test images (200 per class, `RandomState(0)`), identical for all networks, so
  all graphs share one node set and no node matching is needed.
- **Operator per cell** (network, checkpoint, site): exactly the explorer's construction. Centred-cosine Gram →
  affinity → zero diagonal → symmetric doubly stochastic P → M = (I + P)/2. Two kernel rules: union 15-NN graph
  (default) and per-point perplexity 100, taken from `cifar_coherent/sweep_math.final_kernel_bank` (rules 0, 1)
  with `coherent.symmetric_balance`. M is symmetric with constant top eigenvector; ψ_1..ψ_K are the next K = 64
  eigenvectors (eigenvalue-descending), U_k = [ψ_1..ψ_k], P_k = U_k U_kᵀ.

About 1,056 cells per rule.

## Measured quantities, for a pair of cells A, B (all on the shared 2,000 nodes)

- **Projector distance** d_k(A,B) = 1 − ‖U_k(A)ᵀ U_k(B)‖_F² / k = ‖P_k(A) − P_k(B)‖_F² / 2k ∈ [0,1].
  k ∈ {1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,32,40,48,64}. Chance for independent random subspaces orthogonal to
  the constant: 1 − k/(N−1).
  A leading-k projector is only canonical when the cut k | k+1 separates eigenvalues. Every d_k is therefore
  reported with the **cut diagnostics of both cells**: the gap λ_k − λ_{k+1} and the subsample reproducibility
  score span_k (below). A cut with min span_k < 0.5 in either cell is drawn as *unresolved*; d_k at such a cut is
  a sensitivity value, not a property of the layers. No meaning is inferred from "k ≤ horizon".
- **Diffusion-kernel distance** (no cut; discrete time, scale t) h_t(A,B) = 1 − ⟨A_t, B_t⟩_F / (‖A_t‖_F ‖B_t‖_F), A_t = M_Aᵗ − 11ᵀ/N, using the full
  spectrum, t ∈ {1, 4, 16, 64, 256}. This is one minus CKA with the diffusion kernel M^t; it weights modes by
  λ^t instead of cutting at k.
- **Eigenvalue-only distance** s_k = ‖λ_{1..k}(A) − λ_{1..k}(B)‖₂ / √k, same k grid (the classical graph
  spectral distance; it ignores eigenvectors and would not need shared nodes).
- **Baseline, labelled as such:** CKA distance 1 − CKA(HC_AH, HC_BH) on the centred-cosine Grams (CKA of the
  probe-centred, row-normalized activations; not raw-activation linear CKA).
- **Mode-overlap matrix** G = U_K(A)ᵀ U_K(B) (64×64) and **per-image discrepancy**
  δ_i = [(P_k(A) − P_k(B))²]_ii (sums to 2k·d_k). Not stored per pair: the top-64 eigenvectors of selected cells
  are shipped and these are computed on demand. They say *which* modes are kept, mixed or reordered and *which*
  images move, which the scalar distances cannot. Entries of G are sign- and, inside near-degenerate groups,
  rotation-dependent: the viewer shows G², the cumulative block overlap ‖G[:i,:j]‖_F² and the principal angles
  (rotation-invariant), and says that single-mode correspondence is meaningful only at isolated, stable
  eigenvalues.
- **Cut reproducibility per cell**, span_k for every k = 1..80 (so a boundary beyond 64 is visible): as in
  `cifar_more/cm3.py` (`split`, `subset_grams`, `stability`): rebuild the operator on two overlapping label-free
  subsets of the probe (2/3 each, sharing 1/3) and take the minimum cos² principal angle between the two
  rebuilds' leading-k spans on the shared images. Stored per k, with all 2,000 eigenvalues of the full operator.

All scalar distances are stored for **all pairs of cells** (≈1,056² per rule), which contains every comparison
below as a slice.

## Replicates, floors, nulls

- Replicate = training seed (2 per depth). Cross-depth comparisons have 4 seed pairings; they are shown as
  individual values, not as a confidence interval. Question 1 is repeated on ResNet-20 seed 1 (and is available
  for the deeper nets).
- **Seed floor:** distance between the same layer and checkpoint of two seeds of the same depth. A cross-depth
  distance is only "different" relative to this.
- **Chance:** 1 − k/(N−1) for d_k; for DTW, the optimum recomputed after randomly permuting the residual-block
  sites of one network (stem, pool, logits fixed; 1,000 permutations), and a constant-cost matrix.
- **Untrained reference:** every cross-network analysis is also run at checkpoint 0, so that correspondence
  produced by architecture alone is not attributed to learning.

## Alignment across depth

For networks A, B at the final checkpoint and one distance, D is the L_A × L_B layer-by-layer matrix.

- **DTW path:** monotone path from (stem, stem) to (logits, logits), steps (1,0), (0,1), (1,1), minimizing the
  `symmetric2` weighted sum (diagonal steps weight 2; `dtw-python` on the precomputed matrix). The reported cost
  is that minimum divided by L_A + L_B − 1 (the total step weight, the start cell counting once), which is the
  same normalizer for every path, so the optimized and the
  reported quantity agree and short paths are not favoured. Path length is reported too. On a constant matrix
  every path ties: a diagonal-looking path alone is not evidence, only improvement over the permutation null and
  the checkpoint-0 result is.
- **Unconstrained match:** row and column argmins of D. Reported next to DTW because DTW imposes monotonicity;
  the fraction of order inversions in the argmin sequence says whether monotonicity is found or imposed.
- **Architectural reference:** site (stage s, block j of n_A) ↔ (stage s, block ⌈j·n_B/n_A⌉ of n_B), stem,
  pool, logits to themselves. Reported: mean |relative-depth gap| between the DTW path and this reference, and
  whether the path passes through the three stage ends.
- Also run for same-depth cross-seed pairs, where the expected answer is the diagonal, and at every checkpoint
  (iso-step). Final-step comparisons are primary.
- **Iso-accuracy (secondary, outcome-conditioned; uses test labels, so it is a validation analysis and not part
  of the label-free construction).** Target for seed s = final test accuracy of ResNet-20 seed s. For each
  ResNet-32/56 seed s, s* is searched over the full saved schedule (73 checkpoints); it is accepted only if its
  accuracy is within +0.5 points of the target, otherwise the comparison is marked unavailable. Step, accuracy
  and overshoot are reported. Compared: ResNet-20 final against deeper nets at s*, seed pairings as above.

## Controls with a known answer (`test_align.py`, CPU, small N)

1. d_k(A,A) = 0, h_t(A,A) = 0; independent random operators give d_k ≈ 1 − k/(N−1).
2. Planted spectra, r = c−1 signal dimensions, D = N−1:
   (a) repeated eigenvalue (symmetric partition), two independent small perturbations: d_r ≈ 0 at the complete
   span, while internal cuts k < r are unstable (E d_k = 1 − k/r for isotropic bases);
   (b) well-separated signal eigenvalues: d_k ≈ 0 for every k ≤ r;
   (c) shared r-dimensional signal plus independent random complement: E d_k = 1 − [r + (k−r)²/(D−r)]/k for k > r.
3. DTW on a planted warp (a path of operators sampled at L_A and L_B different positions) recovers the warp;
   the reviewer's matrix [[1, 0.1], [1, 1]] gives the path the stated objective predicts; a constant matrix gives
   normalized cost equal to the constant.
4. Chunked all-pairs code equals the direct formulas on a small set; heat distance equals the dense-matrix formula.
5. `nets.build_resnet(3)` has the same state-dict keys and forward output as `zoo.models.ResNet20`.
6. On Modal: the rebuilt operators of ResNet-20 seed 0, final checkpoint, one site, equal the stored
   `work/r20s0b/operators_64000_<si>.npy` rules 0 and 1 to 1e-9.

## Figures (one reader sentence each)

- **W1** layer × layer heatmap of distance at a chosen checkpoint (control: checkpoint, distance, k/t, rule,
  seed). *"At the end of training, ResNet-20's layers fall into … groups and the largest jump is at …."*
- **W2** per layer, distance to the final checkpoint against training step (log x), one line per layer; plus a
  checkpoint × checkpoint heatmap for a chosen layer. *"Layers … settle by step …; layers … keep moving."*
- **W3** distance against k for adjacent layers (final checkpoint), with the seed floor, the chance line and the
  stability horizon marked. *"Beyond k ≈ …, the distance is mostly not a reproducible property of the layer."*
- **W4** k × k (and t, baseline) matrix of Spearman rank correlation between the pair-distance vectors. *"The
  ordering of pairs is / is not sensitive to k."*
- **C1** layer × layer distance matrix between two networks, with the DTW path, the unconstrained argmins, the
  architectural reference and stage boundaries (control: network pair incl. same-depth cross-seed, distance,
  checkpoint incl. iso-accuracy). *"DTW maps ResNet-56 layer … to ResNet-20 layer …; it does / does not follow
  the stages."*
- **C2** table: for each distance, DTW gap from the architectural reference, DTW cost against seed floor and
  permutation null, inversion fraction. *"… distances recover the architectural correspondence; … do not."*
- **C3** distance along the matched path against relative depth, with the seed floor at the same depth.
  *"Networks of different depth differ most at …, and by … times the seed floor."*
- **C4** the same summary over training checkpoints. *"The correspondence appears by step …."*
- **C5** live mode-overlap matrix and per-image discrepancy for a chosen pair of cells (final-checkpoint cells of
  all networks and all ResNet-20 seed-0 cells). *"Matched layers share modes 1..m nearly one-to-one; later modes
  mix."*

## Decisions made for the owner (alternative in brackets)

- Depth 20/32/56 (owner chose over width); image graphs on the shared probe (owner chose over channel graphs).
- Two seeds per depth as the floor [one seed]; iso-step at eight checkpoints plus one iso-accuracy checkpoint
  [dense iso-accuracy matching]; stability horizon as in `cifar_more`.
- Projector distance kept as asked; heat distance added as the k-free version, the mode-overlap matrix as the
  information-rich version, eigenvalue-only distance and linear CKA as contrasts. No Gromov–Wasserstein or
  other node-matching method: nodes are already matched.
- DTW endpoints pinned at stem and logits [open-ended DTW].
- Output: an "Alignment" tab in the `cifar_coherent` site [standalone page].

## Work and spend

**Cost gate.** Before the fan-out, one pilot worker runs the largest unit (ResNet-56, final checkpoint: 30 cells
× 2 rules, subset rebuilds included) and its measured seconds set the projected extraction spend; the fan-out
proceeds only if projected total spend stays under $35. Worker timeouts are 3× the pilot time; training workers
are checked and cancelled if they exceed 90 minutes. All-pairs diffusion-kernel distances are streamed: per rule
and t, the 1,056 centred M^t are formed from stored eigendecompositions in float32 and accumulated into the
Gram of flattened matrices in row chunks, never all on disk at once; checked against the dense formula on a
sample of pairs.


Training: 2 × ResNet-32 (~25 min) + 2 × ResNet-56 (~40 min) on L40S ≈ $4. Extraction: one worker per
(network, checkpoint), ≈ 54 workers, each building up to 30 cells × 2 rules plus two subset rebuilds, ≈ 3–6 min
on A100-40GB ≈ $6. All-pairs distances: a few GPU workers, < 30 min each ≈ $3. Projected total ≈ $15, ceiling
$50 (owner). Wall-clock ≈ 2–3 h.
