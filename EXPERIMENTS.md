# Experiments: calibrating the coherent-mode language on cases with known answers

Status: v1.3, 2026-11-07. A fresh-eyes readability review with per-stage items for a later rewrite is in `reviews/fresh-eyes-v1.2.md`. Companion to `THEORY.md`; section numbers refer to it.

## Principles

- Every measured target is labelled **identity** (an algebraic fact we check the code against), **planted** (ground truth we built in), **established** (a published empirical phenomenon), or **hypothesis** (ours). A failure on an identity is a bug; on a planted target it falsifies a [cond] or [hyp] claim; on a hypothesis it is a result.
- Knobs are charted before they are used: Stage 0 produces the knob-to-error curves of §13 that later stages quote.
- Nothing is trained on top of the model under study. Baselines that need training (an SAE, an actual stitch) are trained and labelled as baselines; closed-form ridge fits in the method are labelled as fits.
- Every training run we control saves dense checkpoints and, at a few of them, stochastic resamples at fixed parameters (repeated minibatch steps or dropout passes) for the noise temperature of §10.
- Held-out evaluation is by **document-level** (or example-level) splits everywhere a fit is involved; past and future windows never straddle a split.
- Compute: everything through CIFAR runs on CPU or one small GPU in minutes to an hour; Pythia-scale on Modal.

## Shared machinery (one small library)

1. **Sites table**: one row per site with document/position (or example id), per-layer activations as float16 memmaps, window indices, split id. Zero-vector policy [P2] applied here with the excluded mass logged.
2. **Views** (§2): geometric, output ($\sqrt p$), context (lag-weighted window), action (sketched Jacobian via `torch.func.jvp`, optional magnitude/offset coordinates), unit activations.
3. **Kernels and normalization** (§2–3): dense sphere kernel to $N \approx 10^4$; Nyström landmarks (leverage or $k$-means, oversampled) at low $\beta$; symmetrized $k$-NN graph at high $\beta$; Sinkhorn to bistochastic; centering. The two estimands of §13 are separate code paths and Stage 0 compares them on dense subsets.
4. **Spectra**: Lanczos for top blocks; block boundaries from gaps; products as `LinearOperator`s (§4.1).
5. **Loadings, readability, localization** (§5, §4.3): $a_F(\psi)$, ridge $R^2$, indicator-subspace principal angles, block support size, localizing rotation as a named algorithm.
6. **Context module** (§6.3–6.4): centered past/target Grams, smoothers $R_p, R_q$, eigenvectors of $R_p R_q$, held-out canonical correlations (VAMP score) for choosing $\rho, w$; carried and read targets; cross-fitted predictive-risk ladder.
7. **Gradient module** (§7.2): hooks capturing per-use $(u, \delta)$ for chosen matrices; block pressure, signed interaction, explicit truncation residual; activation-side expansion of the realized optimizer update.
8. **Comparison** (§8): principal angles, centered alignment, $\beta \to 0$ CKA check, sensitivity-weighted readability.
9. **Reports**: per stage an HTML notebook (`experiment-notebooks` skill) with the labelled targets and verdicts.

Reuse where possible: `datafold`/`pydiffmap` for diffusion-maps internals if their normalization can be replaced by Sinkhorn; `deeptime` for VAMP/kernel-CCA cross-checks; `pynndescent`/`faiss` for neighbours.

## Stage 0: synthetic calibration (no network)

Targets and labels:

- **identity**: reconstruction tail sum (§1.3); bistochastic product has trivial singular value exactly 1 and the symmetric-normalized product does not (§3.1); centered alignment → linear CKA of normalized rows as $\beta \to 0$ (§8); eigenvectors of $R_p R_q$ equal primal canonical variates on linear features (§6.3). The script `checks/cm_check.py` already passes these on random data; Stage 0 turns them into unit tests.
- **planted**: Gaussian mixtures on the sphere with controlled separation: the $\beta$ window in which an isolated block spans the indicator subspace (principal angles) and where it closes at both ends (identity limit at high $\beta$, PCA at low). Union of one-dimensional subspaces with $k$-of-$n$ sparse mixing (TMoS geometry without a network): indicator-subspace recovery vs sparsity; localizing-rotation recovery vs block-only recovery. Product manifold (torus): factorized eigenfunctions under a product kernel, and their loss under correlated sampling.
- **planted, context**: i.i.d. symbols with an embedding "layer" and a next-"layer" that copies the previous symbol: the carried ladder must stay flat and the read ladder must drop within the first $V - 1$ coordinates for a $V$-ary symbol (§6.3–6.4). An HMM token stream: the token-level Hankel benchmark has rank $k$ and the centered KCCA state has $k - 1$ coordinates above threshold at sufficient window.
- **estimand comparison**: $k$-NN graph vs dense operator on subsets across $\beta$; Nyström vs dense across $m$ and landmark rules; bootstrap-over-documents error bars vs $N$.

Deliverable: the knob tables and the regime boundaries every later stage quotes.

## Stage 1: Toy models of superposition (Elhage et al. 2022)

Setup: $x \in \mathbb R^n$ with each feature active independently with probability $1 - S$, importance-weighted reconstruction $\mathrm{ReLU}(W^\top W x + b)$, hidden $h = W x \in \mathbb R^m$. Sweep $S$, $n/m$. All-zero inputs are excluded under [P2] and their mass $S^n$ reported.

Targets: **planted**: columns of $W$ and the single-feature-active site sets; **established**: the monosemantic / antipodal / polytope phase diagram; **hypothesis**: §11 bullet 1.

Measurements:
1. At $\beta_{\text{noise}}$ (stochastic resamples at the final checkpoint) and across the scan: block structure of the hidden-layer operator; principal angles between the top block and the span of the $n$ single-feature indicators; one-to-one (Hungarian) matching of the localizing rotation's loadings to the columns of $W$, reported with the block-only recovery so the rotation's contribution is visible.
2. Polytope regimes: the symmetry group should appear as exact degeneracy; report blocks, not modes.
3. $Q^{\text{hidden} \to \text{output}}$: coherence $\sigma$ of the feature block; whether the output geometry is more or less clustered than the hidden one is measured, not assumed.
4. Baseline: an SAE of matching dictionary size trained on the same hidden activations; compare recovery and report disagreements.
5. Deliberate failure: dense features. The block should lose its gap and the principal angles should grow, and the report should say so.

## Stage 2: Computation in superposition

**Placeholder stage.** The source construction (a computation-in-superposition toy whose solution is "virtual neurons") has not yet been identified in the literature, so there is no established target yet. Provisional setup: $x \in \mathbb R^n$ sparse, target $y = \mathrm{ReLU}(x)$ or an $L^p$ variant, network $W_2\,\mathrm{ReLU}(W_1 x)$ with $m < n$, trained to convergence with dense checkpoints.

Targets: **planted**: the target map and the active-feature partition; **established**: the paper's virtual-neuron construction once identified; **hypothesis**: §11 bullet 2.

Measurements:
1. Input-geometry blocks vs input features, as in Stage 1.
2. Mode profiles of hidden neurons and block loadings on the hidden layer: the two matrices whose structure is "virtual neurons"; compare to the construction.
3. Signed interaction matrix $I_{BB'}$ (§7.2) over training vs the measured cross-feature error of the trained network (correlation, with the caveat that interaction is first-order and the error is not).
4. Action-view (§7.3) blocks vs the active-feature partition, with and without magnitude/offset coordinates, to show what normalized-derivative similarity does and does not recover.

## Stage 3: Modular addition and grokking (Nanda et al. 2023)

Setup: one-layer transformer on $a + b \bmod p$, inputs $(a, b, =)$, dense checkpoints through the transition, stochastic resamples at several checkpoints.

**Site table (declared up front):** sites are (example, final position), $N = p^2$. The embedding view at these sites is constant and is *not* used. Views: post-attention residual at the final position (first point where $a$ and $b$ combine), post-MLP residual, logits. Token-embedding geometry is a separate analysis on the $p$ token sites and is not composed with the example sites.

Targets: **established**: Fourier-frequency circuits and the frequencies used; **hypothesis**: §11 bullet 3; the pressure precursor is a hypothesis, not a known answer.

Measurements:
1. Post-attention operator on the $(a, b)$ grid: is there an isolated block, and what are its principal angles to the span of the product harmonics $\{\cos, \sin\}(\omega a) \otimes \{\cos, \sin\}(\omega b)$ at the frequencies the trained network uses (and at others, as a control)? Fourier structure is expected only if the kernel is near shift-invariant on the grid, which is itself measured.
2. $Q^{\text{post-attn} \to \text{post-MLP}}$: coherence of the frequency blocks; the merge of sites with equal $a + b \bmod p$ (compare the post-MLP block to the span of the $p$ sum-class indicators).
3. Training axis, $\beta$ fixed across checkpoints: gap formation; block pressure (§7.2) on the eventual frequency blocks vs step vs test loss. Report the adaptive-$\beta$ version alongside.
4. Lagged regression of block coordinates across checkpoints (the DMD step of §4.4), stated as a fitted linear model.

## Stage 4: Belief-state geometry (Shai et al. 2024) with transformer and SSM

Setup: Mess3 (3 hidden states, 3 symbols) and a second HMM with a different state count; a small transformer and a small Mamba trained on each; the true belief states from the known HMM; the token-level Hankel benchmark from Stage 0.

Targets: **planted**: the belief simplex and the HMM; **established**: linear representation of beliefs in the transformer residual stream; **hypothesis**: §11 bullet 4, the SSM result, and the ladder shapes.

Measurements:
1. $r_{\text{ctx}}(L)$ per layer (carried and read, held-out threshold) for both architectures, against $k - 1$ and against the token-level benchmark.
2. Linear readability of the true belief state from $s^L$ and of $s^L$ from the true belief state, per layer; compare to Shai et al.'s probe result, which here needs no probe.
3. Read ladder per layer: how much of each layer is "an MLP in disguise".
4. Principal angles between carried and read state subspaces [P5].
5. Cross-architecture on the same sites: centered alignment and principal angles between $\tilde P^{\text{Tr}}_L$ and $\tilde P^{\text{Mamba}}_{L'}$; predicted vs trained stitching, both kinds.

## Stage 5: MNIST (MLP and small CNN), several seeds

Targets: **established**: classes separate with depth; seed-to-seed stitching works; **hypothesis**: coarse-to-fine refinement matches confusion structure; the stitching predictor.

Measurements:
1. Blocks across all layers at fixed $\beta$: principal angles to class indicators at coarse $t$; within-class structure at fine $t$ vs the confusion matrix.
2. Stitching predictor (§8, sensitivity-weighted) vs unweighted readability vs activation ridge reconstruction vs CKA, all against trained stitches of both task-loss and direct-matching kinds, layer by layer. Include an exact-fit control (a seed trained to zero training loss) where gradient pressure vanishes.
3. Training dynamics: block pressure and signed interaction per layer; activation-side expansion of the realized Adam update vs the gradient decomposition, to show the difference.
4. Diffusion-wavelet hierarchy at the penultimate layer; compare to the class confusion hierarchy. For the CNN, declare whether weights are decomposed per use or per site [§7.2].

## Stage 6: dSprites

Setup: a small CNN trained to predict all generative factors (multi-head), and a β-VAE as a second model. Factor product structure is a property of the data; whether the representation inherits it is a hypothesis.

Measurements: factor-wise readability of blocks; whether the top blocks factorize (principal angles to products of per-factor eigenfunctions); where in depth factors merge.

## Stage 7: CIFAR ResNet, then an ImageNet model

Targets: labels, for validation only. Purpose: scaling of §13 (sparse graphs, Nyström) and whether the language localizes a failure.

Measurements: blocks vs labels; gaps per layer; estimand comparison at scale. If structure is poor, the report states which diagnostic says so.

## Stage 8: Pythia (checkpoints and stochastic resamples) on a Pile sample

Setup: Pythia-70m/160m, $N \approx 10^5$–$10^6$ sites, $k$-NN graphs, Nyström context module, Modal; $\beta_{\text{noise}}$ from a few minibatch steps at each used checkpoint.

Measurements:
1. $r_{\text{ctx}}(L)$ (read and carried) and the read ladder per layer; across model sizes.
2. Blocks at fixed $\beta$ vs a published SAE at the same layer: readability of each SAE feature from the top blocks, and block support sizes.
3. Across checkpoints: block pressure and interaction at a few layers.
4. Across sizes on the same sites: principal angles and predicted stitching.

## Ordering

Stage 0 and 1 together (library plus calibration, about a week). Stage 3 and 4 next, since they exercise the training and sequence axes. Stage 2 when its source construction is identified. Stages 5–8 as the library matures. Each stage ends with a notebook and a short verdict against its labelled targets.
