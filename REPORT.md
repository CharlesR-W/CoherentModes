# CoherentModes: where the project stands

*Report for the project owner, 2026-11-07. Sources: `THEORY.md` v1.3, `EXPERIMENTS.md` v1.3, `DECISIONS.md`, `reviews/`, `checks/cm_check.py`.*

## 1. Summary

We have a theory, an experiment plan and six decisions for you. Nothing has been run and no analysis library exists yet.

The theory (`THEORY.md`, about 9,000 words) is a spectral language for what a network does to its activations. Its backbone is a family of operators estimated from samples, each read through its eigendecomposition or SVD (singular value decomposition). No model is trained on top of the network; the few closed-form ridge fits are labelled as fits, and trained sparse autoencoders (SAEs) and stitches appear only as comparison baselines. The core's algebra is exact, but its meaning rests on declared choices (sites, cosine, scale). The context and stitching parts are fitted or hypothetical, and say so.

The theory does not define "ontology". Read operationally (my gloss, not THEORY's), its output is a set of blocks of modes at a declared scale, with loadings in each layer; a block counts as found structure if it is gapped, localized, readable, stable under resampling and coherent across layers.

Three review rounds fixed real mathematical errors (Section 7). The experiment plan has a synthetic calibration stage and eight network stages, from toy models of superposition to Pythia. One substantive error remains in the theory: "split" and "merge" are swapped in one paragraph (end of this report). Two literature references are needed from you; one blocks Stage 2.

## 2. The picture

### 2.1 Sites

A **site** is one place where the network has an activation: a (document, position) pair for a language model, one input for a classifier, one (trial, time bin) for a recording. The set of sites is $X$, with $N$ elements.

The same site exists at every layer. So each scalar channel at each layer (a neuron, a logit, a residual coordinate, a backpropagated error) is a vector in $\mathbb R^N$: a function on sites. A layer's activation matrix $H^L \in \mathbb R^{N\times d_L}$ is $d_L$ such functions side by side. All site-indexed quantities live in this one space.

### 2.2 A four-site example

Take four sites $a, b, c, d$. At layer $L$, $\{a,b\}$ and $\{c,d\}$ are two tight groups far apart. At a later layer $L'$, all four sit close together.

Build a random walk on sites that steps mostly to nearby sites. At layer $L$ it stays inside its group; at $L'$ it mixes all four. The contrast $f = (+1,+1,-1,-1)/2$ is barely changed by a step at $L$: it is **slow** there. A step at $L'$ averages it towards zero. So $f$ is a distinction $L$ draws and $L'$ erases: a **merge**.

Now diffuse at $L$, then at $L'$. A function that survives both steps varies slowly within the neighbourhoods of both layers: it describes a grouping the network keeps together from $L$ to $L'$. Such functions are the **coherent modes**.

### 2.3 The general construction

**Kernel.** Normalize each activation to unit length, $\hat h(x) = h(x)/\lVert h(x)\rVert$, and set
$$K_\beta[x,y] = \exp\big(\beta\,\langle \hat h(x), \hat h(y)\rangle\big), \qquad \beta \ge 0 .$$
On the sphere this is a Gaussian kernel in chordal distance, positive semidefinite (PSD) for every $\beta$. $\beta$ is an inverse temperature: larger $\beta$ means finer resolution. Sites with a zero vector are removed from every view [P2].

**Operator.** Rescale $K$ by Sinkhorn iteration into a symmetric matrix $P^L$ whose rows and columns sum to one ("bistochastic"). This is the random walk of the example. Every layer's walk has the same uniform stationary distribution, so walks from different layers are compared on a common footing. The constant function is a trivial eigenvector, removed by the centering matrix $C = I - \mathbf 1\mathbf 1^\top/N$.

**Blocks.** A **block** is the span of the eigenvectors between two spectral gaps. An isolated block moves by at most about (perturbation size)/(gap) under a small perturbation (Davis–Kahan); individual eigenvectors inside a block are not stable, so results are reported per block.

**Coherent modes.** The product $Q = P^{L'}P^{L}$ (acting on functions) diffuses at $L$, then at $L'$. Its SVD $Q = \sum_i \sigma_i u_i v_i^\top$ pairs an input function $v_i$ with its normalized image $u_i$, and $\sigma_i = \lVert Q v_i\rVert \in [0,1]$. After removing the trivial pair, $\sigma_i$ near one means $v_i$ is slow at $L$ and its image stays slow at $L'$. This is Froyland's finite-time coherent-set construction, with depth as time. A singular function is a signed function, not a set; turning it into groups of sites is a separate, validated step.

**Loadings.** A mode $\psi$ maps back to any layer by $a = (H^L)^\top \psi$. Over a complete orthonormal basis (constant mode included), $H^L = \sum_i \psi_i a_i^\top$ exactly: codes $\psi_i(x)$ times directions $a_i$, the form of a dictionary. Dropping modes costs exactly the sum of the dropped $\lVert a_i\rVert^2$ in squared Frobenius error.

**Two limits.** As $\beta \to 0$, the centered operator, rescaled by $N/\beta$, becomes the centered Gram matrix $C\hat H\hat H^\top C$ of the normalized activations. Its eigenvectors are then normalized PCA (principal component analysis) scores, and centered alignment (Section 3) becomes linear CKA (centered kernel alignment). As $\beta \to \infty$ the operator tends to the identity (for distinct normalized activations) and says nothing. Clustering is a window of intermediate $\beta$ where well-separated groups give a block spanning roughly the group indicators; a rotation of the block chosen to concentrate each mode on few sites (varimax or similar) then gives codes like an SAE dictionary. Whether the window exists is measured.

### 2.4 The pipeline

```
sites → views → kernels K_β → bistochastic P → spectra, blocks
      → cross-layer product Q = P^{L'} P^{L} → SVD → loadings a = Hᵀψ
```

Views are normalized vectors at each site: a layer's activation, the square root of the output distribution, a past window, or a flattened local Jacobian (estimated from random probes). All share one site set, so all operators act on one space.

Two extensions work in the same site space.

- **Context.** For sequence models, a regularized canonical correlation analysis between the layer's past and a target gives **predictive-state coordinates** (Section 4). They are not an orthonormal basis; they define a further view with its own operator.
- **Units and gradients.** Any neuron or head output expands exactly in a mode basis, and so do the per-site contributions to a weight gradient. Let a weight matrix $W$ act as $z = Wu$, let $U$ stack the inputs $u$ over sites ($N \times$ input width), and let $\Delta$ stack the backpropagated errors $\partial\mathcal L/\partial z$ ($N \times$ output width), with $\mathcal L$ the training loss. If each site uses $W$ once,
$$\nabla_W \mathcal L = \Delta^\top U = \sum_i (\Delta^\top \psi_i)(U^\top \psi_i)^\top ,$$
one rank-one term per mode. If a site uses $W$ several times (convolution offsets, recurrent steps), the uses are summed too, and a mode's term can have higher rank. Weight decay is added separately. The Frobenius norm of a block's summed terms, $\lVert \Delta^\top \Pi_B U\rVert_F$ with $\Pi_B$ the block projector, is its **pressure**; the inner product of two blocks' terms is their signed **interaction** (negative means interference).

### 2.5 What we actually inspect

Spectra and gaps; localization of a block (principal angles to known group indicators in toys, and the rotation-invariant effective number of sites the block lives on); **readability**, the held-out $R^2$ for predicting a mode's site values from a layer's activations, linear and kernel; and the reconstruction tail. New sites get mode values by Nyström extension, only for modes above an eigenvalue floor.

A loading is a linear association, not a decoder. Identifying sites across layers is bookkeeping, not causation: statements about what a layer *did* are a separate, weaker step, either predictive (Section 4) or first-order (Jacobians, gradients).

## 3. Your commitments, one by one

"Pro-tem" below means a provisional choice, adopted because the natural quantity is intractable or not unique. The tags [A], [P] and [M] are THEORY.md's labels for substantive assumptions, pro-tem choices and methodological choices.

| commitment | status | main caveat |
|---|---|---|
| (a) ends in an SVD; nothing trained | Met for the backbone. Trained SAEs and stitches are only baselines. | Labelled ridge fits: readability, the context ladder, a layer's "effective action", stitching readability, a lagged regression across checkpoints. Judge whether these fit (a). |
| (b) only the dot product is assumed | Cosine on each normalized view is the one geometric commitment [A2]. | Also needed: [A1] the same site at two layers is the same site; [A3] only the training loss on training sites enters the gradient side. The norm is dropped [P1]. |
| (c) nothing incalculable; no smuggled semantics | Built only from the network, its training data and its objective. Labels only validate. | Uniform site weighting is a declaration; the temperature needs a further choice. |
| (d) architecture-agnostic | The core is: it needs only activations at sites, so transformers, state-space models (SSMs, e.g. Mamba), convolutional networks (CNNs) and recordings are alike. | Cross-model work needs a site correspondence; gradients and Jacobians need a differentiable model; pooling CNNs need whole inputs as sites. |
| (e) comparable across layers, checkpoints, models; predicts stitching | All operators act on one space: principal angles between blocks, and centered alignment $\langle \tilde P^A, \tilde P^B\rangle_F / (\lVert\tilde P^A\rVert_F\lVert\tilde P^B\rVert_F)$ of centered operators. No Procrustes step. $\beta$ is fixed across checkpoints. | Stitching prediction is [hyp], tested in Stage 5. |
| (f) toy answers without assuming sparsity; MNIST; CIFAR failure localized | Each toy has a stated target; the clustering window is measured, not assumed. Failure is named by diagnostics: no gap, no localization, a ladder that never drops. | Nothing has run. Stage 2 waits for your reference. |
| (g) activation space first; weights only constrain how the cloud moves | Exact mode decomposition of the gradient: block pressure, signed interaction. The change any update makes to a weight matrix's outputs on the sites (inputs held fixed) expands exactly, whatever the optimizer. | The gradient, not the Adam update. "Pressure to split" is [hyp]. Spectral interaction need not equal the alignment of site groups' loss gradients. |
| (h) Hankel-style context on the layer-$L$ stream; no marginalizing, no hidden-state model | The "carried" operator: past window against future window of the same stream, in closed form. | A fit (ridge and window chosen on held-out data). No hidden-state-dimension bound on its rank for activation streams. Decision 3. |
| (i) pro-tem choices flagged | Register [P1]–[P8], Section 6. | none |

## 4. The context and KV-cache answer

**In the core there is no problem.** The activation at site $x$ in a later layer already contains whatever attention or recurrence delivered to $x$. So the core needs no model of the layer map, and the KV cache (the keys and values of earlier positions that attention reads) is never modelled.

**It returns when you ask what a layer did.** For attention or an SSM, the layer's inputs at $x$ are $h^L(x)$ *and* the past of the stream. No map takes one position's activation at $L$ to its activation at $L+1$.

**Rejected:** marginalizing over the past (it only bounds from below how much of the layer is explained) and fitting a hidden-state model (training a recurrent summary).

**Adopted: a predictive state.** Fix layer $L$ and a site $x = (\text{doc}, t)$. The *past features* are a lag-weighted window of normalized layer-$L$ activations at positions before $t$. Correlate them with a *target* by regularized kernel canonical correlation analysis (KCCA). With centered sphere kernels $\tilde K_p$ on pasts and $\tilde K_q$ on targets, and ridge $\rho$, form the smoothers $R_p = \tilde K_p(\tilde K_p + N\rho I)^{-1}$ and $R_q$ likewise. The canonical correlations $\kappa_i \in [0,1]$ satisfy $\kappa_i^2 = \mathrm{eig}(R_pR_q)$. For $\kappa_i > 0$ the eigenvectors are functions of the past; scaled to unit variance, they are the predictive-state coordinates $s_i$, collected as $s^L$. How to evaluate them on new documents is an open protocol item. Ridge and window maximize the held-out sum of squared canonical correlations (the VAMP-2 score, from the variational approach to Markov processes), on splits by document.

**Two targets answer different questions.**

| target | what it is | question it answers |
|---|---|---|
| **carried** | future window of the same layer-$L$ stream | what the stream carries about its own future. Intrinsic; works for recordings; this is your Hankel construction. |
| **read** | the next layer's update $\Delta h^{L+1}(x) = h^{L+1}(x) - h^L(x)$ | what the past predicts about what the layer computes now (some of which the current activation may already carry; the ladder below separates this) |

Neither implies the other [thm]. Example: independent tokens, layer $L$ is the embedding, layer $L+1$ copies the previous token. The stream's past predicts nothing about its future, so the carried state is empty; the read state is one lag (the previous token's information). The default [P5] computes both and uses "read" for attribution.

**Effective context rank** $r_{\text{ctx}}(L)$ is the number of held-out $\kappa_i$ above a threshold: empirical and scale-dependent. What theory says:

- *Hidden Markov models (HMMs).* With $k$ hidden states, the Hankel matrix of token past/future probabilities has rank at most $k$; equality needs minimality and long enough windows [cond]. The **belief state** (posterior over hidden states given the past) lives on a simplex of dimension $k-1$. This is about tokens and serves as a benchmark.
- *Activation streams.* For a network trained on that HMM, no bound follows. That $r_{\text{ctx}}$ is near $k-1$ at the belief layer is [hyp] (Stage 4).
- *SSMs.* A linear time-invariant system's Hankel rank is at most its state dimension [cond]; the activation stream's kernel rank is not bounded.
- *Golowich, Liu and Shetty* ("Sequences of Logits Reveal the Low Rank Structure of Language Models", 2025 preprint, ICLR 2026; your "lingen"). Their Theorem 4.3: the extended logit matrix (histories × (future continuation, next token), mean-centered logits) has rank at most $d$ at all positions iff the sequence distribution is a time-varying input-switched affine network of dimension $d$ [thm, theirs]. Ours shares the predictive-factorization idea but does not contain theirs (they score each history against chosen alternative futures; we correlate observed past–target pairs). No connecting theorem is claimed.

**How much context a layer reads** is a ladder of held-out prediction errors for the update $\Delta h^{L+1}$:

1. $\mathcal R_0$: predict the training-fold mean update;
2. $\mathcal R(h^L)$: kernel ridge regression on the current activation;
3. $\mathcal R(h^L, s_1,\dots,s_r)$ for $r = 1, 2, \dots$

Errors are squared and averaged over held-out sites. Fitting the state coordinates, choosing the ridge, and evaluating use disjoint sets of documents. The drop from step 1 to 2 measures how far the layer is predictable as "an MLP (multilayer perceptron) in disguise". The further drop with $r$ is the predictive information the context adds at this window and scale, not proven architectural use. Fitted risks need not decrease; they are reported as a sequence.

**Limits.** Only positions inside the window are observed directly; earlier context enters only as far as those positions' own activations carry it. Window weights [P3], ridge [P6] and the weighting of state coordinates [P7] are provisional.

## 5. What is proved, what is conditional, what is hypothesis

THEORY.md marks claims **[thm]** (exact identity, or a standard theorem whose hypotheses hold by construction), **[cond]** (true under named conditions the network need not meet) or **[hyp]** (empirical, to be tested, never relied on later).

| claim | marker |
|---|---|
| Exact expansion in any orthonormal basis; truncation costs the tail sum (§1.3) | thm |
| Sphere kernel PSD; identity limit at large $\beta$; centered Gram and linear CKA at small $\beta$ (§2, §8) | thm |
| Bistochastic $P$ exists, is unique, eigenvalues in $[0,1]$; $Q$ has trivial singular value 1 (§3.1, §4.1) | thm |
| Isolated blocks are stable relative to their gaps (§3.2) | thm |
| Loading formulas: $\sqrt{\lvert A\rvert}$ times the centroid for a normalized indicator; $\sigma_i$ times the principal direction for a linear-Gram eigenvector (§5) | thm |
| Exact mode decomposition of the gradient; block pressure rotation-invariant; indicator-block and site-group gradients need not agree (§7.2) | thm |
| KCCA dual form; carried and read states differ in general (§6.3) | thm |
| Random-probe estimate of raw Jacobian inner products: unbiased, standard error $O(m_J^{-1/2})$ for $m_J$ shared i.i.d. isotropic probes with finite fourth moments (§7.3) | thm |
| Continuum limit of $P$ is the density-dependent Coifman–Lafon $\alpha = \tfrac12$ generator (§3.1) | cond |
| HMM and linear-SSM Hankel rank bounds (§6.3); kernel convergence rates (§13) | cond |
| Clustering window (§2, §4.3) | measured regime |
| Split/merge reading (§4.1); toy targets (§11); stitching predictor (§8); "pressure to split" (§7.2); layer-constant noise scale (§10) | hyp |

`checks/cm_check.py` spot-checks five [thm] items on random examples (printed values, no assertions); all agree.

## 6. Limitations and provisional choices

**Clustering is a regime, not a guarantee.** When many features are active at once, blocks are expected to lose their gaps, and then the dictionary reading disappears. The method still reports spectral directions of variation, but they need not be PCA directions. This is the main risk for large models.

**Temperature is a choice.** The default $\beta_{\text{noise}}$ is the inverse mean squared pairwise distance between normalized activations after independent stochastic minibatch steps from one checkpoint, mean displacement removed: a heuristic, not a theorem. The fallback is a nearest-neighbour scale, from the data rather than the network. A $\beta$ scan is always reported.

**Other limits.** Rotations inside a block are declared algorithms [P4], tested for recovery. Uniform site weights make dense regions matter more. Jacobians and gradients are first-order, context results are predictive, and ablations are outside the core by design. Token sites are dependent, so error bars come from bootstrapping over documents.

**Scale needs a second estimand.** Dense kernels reach about $10^4$ sites on a laptop, $3\times10^4$ on a workstation. Beyond that: a sparse $k$-nearest-neighbour graph (a different operator, possibly with negative eigenvalues) or Nyström landmarks, each validated against the dense operator on subsets [M8].

**Provisional choices** (THEORY's [P] register): P1 drop the activation norm (restorable as a $\log\lVert h\rVert$ coordinate); P2 remove zero-vector sites from all views, reporting the mass; P3 lag weights in the context window; P4 rotations inside near-degenerate blocks; P5 compute both carried and read context, read for attribution; P6 ridge and window by held-out canonical correlation; P7 weighting of state coordinates in the joint kernel; P8 noise temperature, else nearest-neighbour, fixed across checkpoints.

## 7. What changed under review

Three rounds: a design review and mathematical audit, re-checks, and a readability review. The fixes that matter: a shared normalization so layers compose (separate normalization made density differences look like incoherence); large $\beta$ is the identity limit, not clustering; comparisons are centered, recovering CKA at small $\beta$; the context estimator is properly whitened and splits carried from read; gradient statistics are now rotation-invariant per block; and the stitching predictor, one false invariance claim and several theorems were downgraded or removed. These were audits, not proof certification. The per-stage protocol items in EXPERIMENTS.md are still open.

## 8. The experiment programme

Targets are labelled **identity** (checks the code), **planted** (built-in ground truth), **established** (published) or **hypothesis** (ours). A failed identity is a bug. Once identities pass and conditions are verified, a failed planted target refutes a recovery hypothesis or marks its regime's edge, not a theorem.

| stage | system | question |
|---|---|---|
| 0 | synthetic, no network | Identities in code; where the clustering window is; sparse and Nyström vs dense; context tools on a copy task and an HMM |
| 1 | toy models of superposition (TMoS) | Does a block span the single-feature indicators, with rotated loadings matching the columns of $W$? Comparison with a trained SAE; visible failure with dense features |
| 2 | computation in superposition | Do neuron profiles and block loadings show "virtual neurons"? **Placeholder until you supply the source.** |
| 3 | modular addition, through grokking (delayed generalization) | A block spanned by the network's Fourier harmonics? Sum classes merging after the MLP? Block pressure rising before test loss falls? |
| 4 | HMM belief states (Mess3, a 3-state, 3-symbol HMM, and one other), transformer and Mamba | $r_{\text{ctx}}$ near $k-1$? True belief readable from $s^L$? How much each layer reads |
| 5 | MNIST, MLP and CNN, several seeds | Blocks vs classes and confusions; stitching predictor vs CKA, readability and ridge reconstruction, against trained stitches |
| 6 | dSprites, CNN and β-VAE (a variational autoencoder) | Do the top blocks factorize over generative factors? |
| 7 | CIFAR ResNet, then ImageNet | Does the method scale, and does it say where structure fails? |
| 8 | Pythia-70m/160m (open models with public checkpoints) on the Pile (their public training corpus) | Context rank and ladder per layer; blocks against a published SAE; trends over checkpoints and sizes. |

Key controls: Stage 3 uses final-position views only and tests the kernel's shift-invariance separately. Stage 5 trains stitches both on the task loss and by directly matching activations, and includes a model trained to zero loss, where gradient pressure vanishes. Stage 8 validates the sparse operator on dense subsets. No stage yet has numerical pass/fail thresholds.

**Cost** (planning estimates). Through CIFAR: a CPU or one small GPU, minutes to an hour per run. Pythia: Modal. Stages 0 and 1 with the shared library: about a week of development. Stages 3 and 4 next, as they exercise the training and sequence axes.

## 9. Decisions for you

**1. Is the site the atom? [A1]** The same site at two layers is the same indexed observation (not an unchanged activation); transport between layers is the identity on site labels; attribution is a separate, weaker step.
- *Accept:* no layer-to-layer map is needed, the context problem leaves the core, and the core is architecture-agnostic. Cross-model work needs shared tokenization or a declared matching. For recordings, sites are (trial, time bin) and brain areas are views of the same site.
- *Reject:* a different construction would need a declared correspondence or transport between layers (known, chosen or fitted), with its extra assumptions made explicit.

**2. Uniform site weighting? [M3]** Bistochastic normalization weights every site equally. A common token therefore weighs more in total, because it occupies more sites: frequency enters only through the dataset.
- *Accept:* the theory stands as is.
- *Non-uniform* (say, down-weighting repeated tokens): allowed by the algebra if every operator uses the same weighting; not yet worked out.

**3. Which context operator is primary? [P5]**
- *Both* (current): carried describes the stream; read drives attribution and the ladder.
- *Carried:* closest to your Hankel request; works for recordings; blind to a layer that reads the past without carrying it (the copy example).
- *Read:* best for predicting what the layer computes from the past (predictive, not causal). Needs a next layer, so not available for a single recorded stream.

**4. Require stochastic resamples for the temperature? [P8]**
- *Yes:* a few extra minibatch steps or dropout passes per checkpoint give a network-derived $\beta$. Easy for our runs; for Pythia, first check the optimizer state is available. The protocol needs pinning down (end of report).
- *No:* the nearest-neighbour scale is the headline: cheaper, but from the data, not the network.

**5. Pretrained models without their corpus [A3].**
- *Substitute:* evaluate the training-loss gradient at the checkpoint on a held-out sample from the same distribution, flagged as such.
- *Restrict* the gradient side to models whose corpus we have (our toys; Pythia on the Pile).

**6. Weight in the stitching predictor (THEORY §8).** The hypothesis: a linear stitch from model A's layer $L$ into model B's layer $L'$ works to the extent that the blocks of B which B's downstream network is sensitive to are linearly readable from A's layer. The predictor weights each B-block's unreadability (one minus ridge $R^2$ from A's activations) by the block's importance; it is scored against actual stitches. The decision is the weight.
- *Output sensitivity* (proposed): how much moving along the block's directions changes the predicted distributions, in the output Fisher metric. (Gradient pressure is rejected: it vanishes at an exact fit, where stitching can still fail.)
- *Unweighted readability:* simpler. Stage 5 compares both, plus CKA and ridge reconstruction, as baselines anyway.

**Two references to identify (the ones you had in mind could not be found by search).**
- The paper describing neurons as "coherent structure detectors" (HAVOK-related; HAVOK is the Hankel alternative view of Koopman). THEORY §7.1 claims to make this picture literal (units expanded in coherent modes) and should cite it.
- The computation-in-superposition toy whose solution is "virtual neurons" (blocks Stage 2 only).

## 10. Suggested next steps

1. Answer the six decisions. Stage 0 can start under the current provisional defaults, but decisions 1–3 may require revising it. The missing references do not block Stage 0.
2. Correct the split/merge direction in THEORY §4.1 (below).
3. Write the missing protocol details before coding them: block selection when there is no gap; cross-fitting and out-of-sample state coordinates; matching blocks across checkpoints; how the Fourier frequencies a modular-addition network uses are identified; the stitching interface; the noise-temperature protocol; pass/fail thresholds per stage. Then have the plan reviewed once more.
4. Build the shared library and Stage 0, turning `checks/cm_check.py` into asserted unit tests and adding reconstruction-tail and split/merge tests. Reuse `datafold`/`pydiffmap`, `deeptime` and `pynndescent`/`faiss` where they fit.
5. Run Stage 1 with Stage 0, then Stages 3 and 4; Stage 2 when the reference arrives.

## Possible errors in the source documents

1. **Split and merge are swapped in THEORY §4.1.** THEORY calls "slow at $L$, image not slow at $L'$" a *split* (one cluster at $L$, two at $L'$). It is a merge. Exactly, as in Section 2.2: take normalized activations $(e_1, e_1, e_2, e_2)$ at $L$ and $(e_1,e_1,e_1,e_1)$ at $L'$, with $f = (1,1,-1,-1)/2$. Then $\lVert P^L f\rVert = \tanh(\beta/2)$, near 1, while $\lVert P^{L'} f\rVert = 0$. A split appears instead as a new slow contrast at $L'$. The claim is [hyp] and nothing downstream relies on it; THEORY's caveat (reading a split off signed functions needs a validated partition step) still applies.
2. **Citation consistency.** THEORY §6.3 cites Golowich, Liu and Shetty as 2025. That is the preprint year; the paper appeared at ICLR 2026. Cite one version consistently.
3. **The noise temperature is under-specified.** THEORY §10 does not say which optimizer, optimizer state, step size or batch sampling the independent minibatch steps use, nor the finite-sample pairwise convention. The jitter depends on all of them, so different protocols define different temperatures. For Pythia (Stage 8), check whether the optimizer state is available.
