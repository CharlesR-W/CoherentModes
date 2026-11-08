# Experiments: calibrating the coherent-mode language on cases with known answers

Status: v0 draft, 2026-11-07. Companion to `THEORY.md`; section numbers below refer to it.

## Principles for the programme

- Every experiment states the **ground truth** it checks against and the **theory sections** it exercises, so that a failure localizes to a section.
- Knobs are charted before they are used: the synthetic stage produces the knob-to-accuracy curves of §13 that all later stages quote.
- Nothing is trained on top of the model under study. Where a comparison baseline needs training (an SAE, an actual stitch), the baseline is trained and the method is not.
- Compute: everything up to CIFAR runs on CPU or a single small GPU in minutes; Pythia-scale runs on Modal. Checkpoints are saved densely in every training run we control, because the training-dynamics decomposition (§7.2) and the noise temperature (§10) both need them.

## Shared machinery (one small library)

Build once, reuse everywhere:

1. **Sites table.** A dataframe of sites with document/position (or sample id) and, per layer, the activation matrix $H^L$ stored as float16 memmaps. For sequence models, also the per-site past/future window indices.
2. **Views** (§2): geometric, output ($\sqrt p$), context (windowed), action (sketched Jacobians via `torch.func.jvp`), and arbitrary unit activations.
3. **Kernels** (§2–3): dense sphere kernel for $N \le 3 \times 10^4$; $k$-NN sparse graph via `pynndescent`/`faiss` beyond; Nyström landmarks for the Hankel operators. Symmetric normalization, $\alpha$ as a parameter.
4. **Spectra**: Lanczos (`scipy.sparse.linalg.eigsh`) for top-$r$ eigenpairs; products of operators as `LinearOperator`s, never materialized (§4.1).
5. **Loadings and readability** (§5): $a_F(\psi)$, ridge $R^2$, IPR, Nyström extension.
6. **Hankel module** (§6.3): past/future kernels, regularized CCA, state coordinates, effective rank, explained-variance ladder (§6.4).
7. **Gradient decomposition** (§7.2): per-site $(u(x), \delta(x))$ capture by hooks for every weight matrix; mode-wise $g_i, b_i$, pressure, interference matrix.
8. **Comparison** (§8): principal angles, alignment, stitching prediction.
9. **Reports**: per experiment, an HTML notebook (spectra, IPR, loadings, ladders, pressure over training) via the existing `experiment-notebooks` skill.

Estimated size: 1.5–2.5k lines of Python. Reuse: `pydiffmap` or `datafold` for the diffusion-maps core if their normalization options suffice; `pykoopman`/`d3s` for kernel transfer-operator estimation as a cross-check.

## Stage 0: synthetic calibration (no network)

Ground truth: planted geometry. Theory: §2, §3, §5, §13.

- Gaussian mixtures on the sphere with controlled separation; union of $n$ one-dimensional subspaces with sparse $k$-of-$n$ mixing (the TMoS activation geometry without a network); a product manifold (torus) for the dSprites case.
- Chart: eigenvalue and eigenvector error vs $N$, $\beta$, $k$ (graph sparsity), $m$ (landmarks); IPR vs mixing sparsity; loading alignment with planted directions vs $\beta$; where the $\beta \to 0$ limit becomes PCA.
- Deliverable: the knob-to-accuracy tables that every later stage cites.

## Stage 1: Toy models of superposition (Elhage et al. 2022)

Setup: $x \in \mathbb R^n$ sparse (feature probability $1 - S$), importance-weighted reconstruction $\mathrm{ReLU}(W^\top W x + b)$, hidden $h = Wx \in \mathbb R^m$, $m < n$. Sweep $S$, $n/m$.

Ground truth: the columns of $W$ (feature directions) and the known phase diagram (monosemantic / antipodal / polytope regimes).

Theory exercised: §2 (sphere kernel), §3 (spectrum, gaps), §4 (hidden $\to$ output dynamic operator), §5 (loadings, IPR), §10 (temperature from training noise: adjacent-checkpoint jitter).

Measurements:
1. At $\beta_{\text{noise}}$ and across a $\beta$ scan: top-$n$ eigenfunctions, their IPR, and mean max-cosine of hidden-layer loadings vs columns of $W$.
2. Spectral gap at $r = n$ vs regime; polytope regimes should show the geometry's symmetry group in the degenerate blocks (§3, [P2]).
3. $Q^{\text{hidden} \to \text{output}}$: singular values near 1 for feature modes; sharpening of IPR relative to the hidden layer alone.
4. Baseline: an SAE with the same dictionary size trained on the same hidden activations. Compare dictionary recovery and report where the two disagree.
5. Failure mode on purpose: dense features ($S$ small). IPR should fall and max-cosine recovery should fail, and the report should say so.

## Stage 2: Computation in superposition

Setup (operationalized from the description; to be confirmed against the Hänni et al. construction): $x \in \mathbb R^n$ sparse, target $y = \mathrm{ReLU}(x)$ (or an $L^p$ variant), network $W_2\,\mathrm{ReLU}(W_1 x)$ with $m < n$ hidden neurons, trained to convergence.

Ground truth: the paper's virtual-neuron construction; the exact target map.

Theory exercised: §7.1 (mode profiles of neurons, virtual neurons), §7.2 (gradient decomposition, interference), §7.3 (action kernel: affine cells of the hidden layer).

Measurements:
1. Modes of the input-geometry operator vs input features (recovery as in Stage 1).
2. Mode profile of every hidden neuron and loading of every mode on the hidden layer: the two matrices whose product structure is "virtual neurons". Compare to the paper's construction.
3. Interference matrix $I_{ij}$ over training vs the measured cross-feature error of the trained network.
4. Action-kernel coherent sets vs the "which features are active" partition.

## Stage 3: Modular addition and grokking (Nanda et al. 2023)

Setup: one-layer transformer on $a + b \bmod p$, dense checkpoints through the grokking transition.

Ground truth: Fourier-basis circuits; the frequencies used; the transition time.

Theory exercised: §3 (degenerate blocks: harmonics come in $\cos/\sin$ pairs), §4 (embed $\to$ post-MLP dynamic operator; frequency clusters), §4.4 (training-time axis), §7.2 (pressure as a precursor of the transition), §10 (noise temperature across checkpoints).

Measurements:
1. Eigenfunctions of the token-embedding geometry vs circle harmonics; eigenfunctions of post-MLP geometry on the $(a, b)$ site grid vs the product harmonics.
2. $Q^{\text{embed} \to \text{post-MLP}}$ singular functions grouped by frequency; compare the set of high-$\sigma$ frequencies to the frequencies Nanda's analysis recovers.
3. Across checkpoints: gap formation, and the pressure (§7.2) on each frequency mode vs step. Claim to test: pressure on the eventual frequencies leads the test-loss drop.
4. Training-axis Hankel: the dynamic-mode decomposition of the checkpoint sequence of $S^L$ (§4.4), as the HAVOK-style view of training.

## Stage 4: Belief-state geometry (Shai et al. 2024) with transformer and SSM

Setup: Mess3 (and a second HMM with a different state count); a small transformer and a small Mamba trained on each; the true belief states from the known HMM.

Ground truth: the true belief simplex and its fractal structure; the true Hankel rank (number of hidden states).

Theory exercised: §6 entire (context view, Hankel, explained-variance ladder, carried vs read), §8 (transformer vs SSM comparison on the same sites, stitching prediction between them).

Measurements:
1. $r_{\text{ctx}}(L)$ per layer for both architectures vs the HMM state count.
2. Alignment of the predictive-state coordinates $s^L$ with the true belief states (linear readability, §5) per layer; compare to Shai et al.'s linear-probe result, which here needs no probe.
3. Explained-variance ladder per layer: how much of each layer is "an MLP in disguise" vs context-reading.
4. Principal angles between the carried and read Hankel state subspaces ([P5]).
5. Cross-architecture: alignment of $S^{\text{Tr}}_L$ and $S^{\text{Mamba}}_{L'}$; predicted stitching fidelity vs an actually-trained linear stitch.

## Stage 5: MNIST (MLP and small CNN), several seeds

Ground truth: classes; a weak hierarchy among classes (e.g. 4/9, 3/5/8); seed-to-seed stitching that is known to work.

Theory exercised: §3 (diffusion time vs truncation, hierarchy), §4 (coherent sets through all layers), §7.2 over training, §8 across seeds, §10.

Measurements:
1. Coherent sets across all layers at $\beta_{\text{noise}}$: do they coincide with classes at coarse $t$ and refine within classes at fine $t$; does the refinement match the known confusion structure.
2. Stitching prediction between seeds vs actual stitch accuracy, layer by layer.
3. Training dynamics: pressure and interference matrices per layer over training; where class modes separate.
4. Diffusion-wavelet hierarchy (§3) at the penultimate layer; compare to the class hierarchy.

## Stage 6: dSprites

Ground truth: generative factors with a product structure.

Theory exercised: §3 (product manifolds yield product eigenfunctions), §5 (factors as loadings), §4 (which layers keep factors coherent).

Measurements: factor-wise readability of eigenfunctions; whether the top modes factorize; where in depth factors merge.

## Stage 7: CIFAR ResNet, then an ImageNet model (Inception or ResNet-50)

Ground truth: labels only, used for validation and never in the construction.

Theory exercised: scaling of §13 (sparse graphs, Nyström); whether the language localizes a failure.

Measurements: coherent sets vs labels; the explained-variance ladder is trivial (no context) so the depth axis carries everything; a report of gaps per layer. If the structure is poor, state which diagnostic says so.

## Stage 8: Pythia (checkpoints available) on a Pile sample

Setup: Pythia-70m/160m, $N \approx 10^5$–$10^6$ sites, sparse graphs, Nyström Hankel, Modal.

Theory exercised: everything, including §10 option 1 with real checkpoints and §8 against published Pythia SAEs.

Measurements:
1. $r_{\text{ctx}}(L)$ and the explained-variance ladder per layer; comparison across model sizes.
2. Coherent modes at $\beta_{\text{noise}}$ vs features of a published SAE at the same layer: for each SAE feature, its readability from the top-$r$ modes, and for each mode, its IPR.
3. Across checkpoints: mode pressure and interference at a few layers through training.
4. Across Pythia sizes on the same sites: subspace distances and predicted stitching.

## Ordering and effort

Stage 0 and 1 first, together (a week of work including the library). Stage 3 and 4 next, since they test the two time axes other than depth. Stage 2 when the toy construction is confirmed. Stages 5–8 as the library matures. Each stage ends with a notebook and a short written verdict against its ground truth.
