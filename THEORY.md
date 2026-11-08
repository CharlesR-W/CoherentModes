# Coherent modes: a site-space spectral language for what a network does to its activations

Status: v1.5, 2026-11-07. Review history and the numerical checks behind three of the constructions are in `reviews/` and `checks/`; revision notes are in Appendix A. Unresolved design choices that are the project owner's to make are in `DECISIONS.md`.

**Roadmap.** Section 1 fixes the shared index set (sites) and the one exact identity everything rests on. Section 2 turns each kind of measurement into a kernel. Section 3 normalizes kernels into operators and names the coarse-graining knobs. Section 4 composes operators across layers (the coherent-mode construction). Section 5 maps modes back to activation space. Section 6 handles context (what other positions contribute). Section 7 expands units and gradients in the mode basis. Section 8 compares models. Sections 9–10 are variants and the temperature choice. Sections 11–15: hypotheses, the assumptions register, computation, related work, and a glossary. The *core* is Sections 1–5 and 7.1–7.2 (exact or standard); Sections 6, 7.3 and 8 add fitted or interpretive extensions and say so.

Formal technical claims carry one of three markers (prose recommendations and interpretations are unmarked):

- **[thm]** an exact identity or a standard theorem whose hypotheses are met by construction;
- **[cond]** true under explicitly stated conditions that the network need not satisfy; the conditions are named where the marker appears;
- **[hyp]** an empirical hypothesis about trained networks, to be tested in `EXPERIMENTS.md`, never relied on by a later section.

Assumptions and choices are additionally tagged **[A]** substantive (a modelling commitment we believe matters), **[P]** pro-tem (a provisional default adopted because the natural quantity is intractable or non-unique; an open choice), **[M]** methodological (an implementation choice not expected to matter at the level of the theory), collected in Section 12. Acronyms are expanded in the glossary, Section 16.

## 0. What this document is for

We want a description of a trained network that

1. is built only from the network, its training data, and its training objective: no posited "task distribution", no trained probe or dictionary, no semantic labels in the construction;
2. is scale-sensitive, with explicit coarse-graining knobs whose roles are visible;
3. ends in SVDs, so every object has a spectrum, nested subspaces, and perturbation bounds (Davis–Kahan, Wedin) that give an honest notion of "close";
4. lives on an object shared across layers, checkpoints, and models trained on the same data, so comparison needs no fitting;
5. pulls back to the weights, so that training can be read in the same language;
6. assumes neither sparsity nor linear features nor a particular architecture.

The construction makes one move: **everything is a function on the set of sites.** A site is one place where the network has an activation. Every measurable property of a site at any layer defines a kernel on sites, hence an operator on functions on sites, hence an SVD. Operators built from different layers, models, or kinds of measurement act on the same space and so compose and compare directly.

What this move buys is bookkeeping, not causation. Identifying a site across layers lets us compare geometries on a common index set; it does not by itself say which inputs produced an output. Saying what a layer did with its inputs is a separate, weaker step (Sections 6.4, 7): predictive and first-order local statements, each with its own assumptions, never a causal identification. Representational similarity analysis makes the same shared-index move. None of the ingredients below is claimed as new: coherent-set transfer spectra, canonical-correlation (VAMP) estimation, and the reading of neurons as thresholded Koopman singular functions with lag-vector past–future CCA (Pughe-Sanford et al. 2025, arXiv:2510.26955) are established. What is proposed is their synthesis on one index set: cross-view operators, their composition across layers, and unit, gradient and action reporting in the resulting spectral bases.

## 1. The spine: sites and the Hilbert space over them

### 1.1 Sites

$X$ is a finite set of sites, $|X| = N$. For a causal sequence model on a corpus, $x = (\text{document}, t)$. For a classifier, $x$ is one input. For a recording, $x$ is one (trial, time bin). The site is the atom: it is the thing that has an activation at every layer.

**[A1] The same site at two layers is the same site.** We never need a map from the layer-$L$ activation to the layer-$L'$ activation as a function of the former alone. Whatever other positions contributed to $x$ at $L'$ is already in $h^{L'}(x)$ by definition. Section 6 explains how this both removes and then honestly re-poses the context problem: in a transformer the activation at a position depends on the keys and values cached from all earlier positions (the KV cache), so there is no map from one position's activation at layer $L$ to its activation at $L+1$. [A1] requires a site correspondence; it is immediate for one model, and across models it requires a shared tokenization or a declared matching (e.g. whole inputs as sites), which must be stated.

### 1.2 Functions on sites and the measure convention

$\mathcal H = \mathbb R^N$ with the counting-measure inner product $\langle f, g\rangle = \sum_x f(x) g(x)$. Every quantity is a vector or operator in $\mathcal H$: the activation matrix $H^L \in \mathbb R^{N \times d_L}$ (columns are functions on sites), a neuron's activation, a head's output, a logit, a backpropagated error, a kernel $K \in \mathbb R^{N \times N}$.

**[M1] Counting measure throughout.** All orthonormality is Euclidean, $\Psi^\top \Psi = I$. The uniform probability measure differs by constants ($1/N$ in cross-moments, $1/\sqrt N$ in normalized indicators) and changes no construction once those are carried consistently; a non-uniform site weighting $\mu$ is handled by the substitution $f \mapsto M^{1/2} f$ and is not used in this document. (Every identity below is in the Euclidean convention; see Appendix A.)

### 1.3 The SVD duality, which the whole theory generalizes

The plain SVD $H^L = U \Sigma V^\top$ has the two faces we want: left singular vectors $u_i \in \mathcal H$ are functions on sites, right singular vectors $v_i \in \mathbb R^{d_L}$ are directions in activation space, and the left factor is computable from the Gram matrix $H^L (H^L)^\top$ alone.

The theory replaces the Gram matrix by a nonlinear kernel on sites (Section 2), keeps the left factor as the primary object (Section 3), and recovers the right factor as a loading (Section 5). For any Euclidean-orthonormal basis $\Psi$ of $\mathcal H$ and any $F \in \mathbb R^{N \times p}$,

$$F = \Psi \Psi^\top F = \sum_i \psi_i\, a_i^\top, \qquad a_i := F^\top \psi_i, \qquad \Big\| F - \sum_{i \le r} \psi_i a_i^\top \Big\|_F^2 = \sum_{i > r} \|a_i\|^2 . \quad \text{[thm]}$$

Here $\Psi$ is a *complete* orthonormal basis of $\mathbb R^N$, including the constant mode; "top-$r$" elsewhere in this document counts non-constant modes after the constant one has been removed by centering (Section 3.2).

The expansion is exact in any orthonormal basis, and the truncation error is the tail sum of the coefficients. What the basis choice changes is *which* functions are in the head and which in the tail. The best rank-$r$ reconstruction of $F$ in Frobenius norm is its own SVD; a nonlinear-kernel basis is not optimal for that and is not meant to be. It is chosen so that its head is interpretable, and the tail sum is the price, always computable.

## 2. Views: every measurement used as a kernel is a dot product on a sphere

A *view* is a map $\phi : X \to \mathbb R^p$, normalized to $\hat\phi = \phi / \|\phi\|$, with the kernel

$$K_\beta[x, y] = \exp\big(\beta\, \langle \hat\phi(x), \hat\phi(y)\rangle\big), \qquad \beta \ge 0 .$$

On the unit sphere $\exp(\beta\, u^\top v) = e^{\beta} \exp(-\beta \|u - v\|^2 / 2)$ **[thm]**, so this is the chordal Gaussian kernel on the sphere; it is positive semidefinite for all $\beta \ge 0$ **[thm]**. Its two limits at fixed finite data:

- $\beta \to 0$: $K_\beta = \mathbf 1 \mathbf 1^\top + \beta\, \hat\Phi \hat\Phi^\top + O(\beta^2)$. After centering by $C = I - \tfrac1N \mathbf 1\mathbf 1^\top$ and rescaling by $1/\beta$, the kernel is the centered Gram matrix of the *row-normalized* activations **[thm]**; Section 8 gives the matching statement for the normalized operator. **Linear structure (PCA of normalized activations, linear CKA) is the $\beta \to 0$ limit.**
- $\beta \to \infty$: $e^{-\beta} K_\beta \to I$ for distinct normalized vectors **[thm]**. The operator becomes the identity, all eigenvalues tend to one, and nothing is identifiable. **Clustering is not a limit of this theory; it is a finite-$\beta$ regime** in which weakly connected clusters give a block of near-top eigenvalues whose span is near the span of the cluster indicators (Section 4.3). Roughly, the regime needs cluster separation above $1/\sqrt\beta$ and within-cluster spacing below it, and it also depends on cluster sizes, connectivity bottlenecks and within-cluster gaps; it is a measured window, charted in Stage 0 of the experiments, not a theorem.

**[A2] The only geometric commitment is the dot product on each view.** The residual stream is taken with cosine similarity. This is a modelling claim and we make it deliberately: LayerNorm, the near-isotropy results, and the success of cosine-based probes and dictionaries support it, and no theorem makes it right. Where the norm matters (attention sinks, confidence scaling), $\log\|\phi\|$ can be appended as a coordinate before normalization, which introduces a scale choice **[P1]**. Sites with $\phi(x) = 0$ in any view used (all-features-off inputs, dead ReLU blocks, zero Jacobians) need a declared policy: the site is removed from $X$ for every view, so that all operators keep a common site set, and the removed mass is reported **[P2]**.

| view | $\phi(x)$ | the kernel is large when two sites... |
|---|---|---|
| geometric, layer $L$ | residual $h^L(x)$ | have similar activations at $L$ |
| output | $\sqrt{p(\cdot \mid x)}$ | predict similar distributions |
| context, layer $L$ | lag-weighted past window of the stream (Section 6.3) | arrive with similar recent pasts |
| action, layer $L$ | sketched local Jacobian $J^L(x)$ (Section 7.3) | receive similar local linear maps |
| any unit | neuron, head, state coordinate | used as a function to expand, not as a kernel (Section 7.1) |

**Output view.** Under $p \mapsto \sqrt p$ the Fisher–Rao metric tensor is four times the round sphere metric and Fisher–Rao distance is twice the angle **[thm]**; the Bhattacharyya affinity $\sum_k \sqrt{p_k q_k}$ is the dot product on that sphere **[thm]**. So "Fisher–Rao at the output" costs nothing new: the same kernel on $\sqrt p$. This is a Fisher-geometry kernel, not the Fisher–Rao heat kernel, and it is the only place Fisher information enters by default **[M2]**. Pulling Fisher back to intermediate layers is a variant (Section 9).

**Window and product kernels.** For non-negative lag weights $\beta_j$ with $B = \sum_j \beta_j > 0$, $\exp(\sum_j \beta_j \langle \hat h(x_{-j}), \hat h(y_{-j})\rangle) = \exp(B\, z(x)^\top z(y))$ with $z$ the unit vector of $\sqrt{\beta_j / B}$-weighted concatenated lags **[thm]**. A product of sphere kernels is a sphere kernel, so [A2] covers windows, with the lag weights as a pro-tem choice **[P3]**.

## 3. Operators: bistochastic normalization and spectra

### 3.1 Normalization relative to the shared measure

Each view's kernel is normalized to a **symmetric doubly stochastic** operator by Sinkhorn scaling,

$$P = D_v K D_v, \qquad P \mathbf 1 = \mathbf 1, \quad P = P^\top, \quad D_v = \mathrm{diag}(v),$$

with $v$ the unique positive vector making the row sums one (exists for strictly positive $K$) **[thm]**. $P$ is a congruence of a PSD matrix, so $0 \le \lambda_i \le 1$ with $\lambda_0 = 1$ on the constant function $\mathbf 1 / \sqrt N$ **[thm]**. This is the bistochastic kernel normalization of Marshall and Coifman (2019). In the manifold limit with sampling density $q$ it preserves the sampling measure $q\,d\mathrm{vol}$ and its generator is $\Delta + \nabla \log q \cdot \nabla$ up to scale, i.e. the Coifman–Lafon $\alpha = \tfrac12$ (Fokker–Planck) normalization **[cond]**, not the density-free $\alpha = 1$ one: uniform weighting of sampled sites still yields a density-dependent continuum operator.

Why this and not the usual $D^{-1/2} K D^{-1/2}$ **[M3]**: every view's operator must be a Markov operator for the *same* measure on sites, the counting measure. With the usual normalization each layer's operator has its own stationary vector $\sqrt{d_L}$, the identity transport of [A1] then does not identify the two stationary coordinate systems, and the product of two layers' operators has top singular value strictly below one whenever their degree vectors are not proportional. Density differences between layers would then masquerade as loss of coherence. (Verified in `checks/cm_check.py`: top singular value $0.997$ under the old normalization against exactly $1$ under the bistochastic one, on the same kernels.) The shared stationary measure is the price of composability, and it is cheap in practice: Sinkhorn on a dense or sparse kernel typically converges in a few dozen matrix–vector products.

The density question of v0 is thereby answered by a declaration rather than removed: sites are weighted uniformly, so a region the model populates densely has many sites and matters accordingly, and the continuum operator carries the corresponding density drift. Frequency enters through $X$ and, in the limit, through that drift.

### 3.2 Spectrum, coarse-graining knobs, degeneracy

$P \psi_i = \lambda_i \psi_i$. The constant mode is trivial and is removed by centering, $\tilde P = C P C = P - \tfrac1N \mathbf 1\mathbf 1^\top$. Three knobs act on the spectrum and none is applied in the spine:

- **temperature $\beta$**: resolution of the view (Section 10);
- **diffusion time $t$**: $P^t$ has eigenvalues $\lambda_i^t$, a smooth damping of fine modes **[thm]**; the embedding distance $\|P^t(\delta_x - \delta_y)\|$ is the diffusion distance for this symmetric Markov operator;
- **truncation $r$**: keep the top-$r$ eigenspace; error is the tail sum of Section 1.3.

Time and truncation disagree when there is no gap, which is correct: there is then no canonical number of modes and the smooth knob should be preferred.

**Degeneracy.** Isolated eigenspaces (the span of modes between two gaps) are canonical and stable: by Davis–Kahan, an eigenspace with gap $\delta$ to the rest of the spectrum moves by $O(\|E\|/\delta)$ in subspace distance under a perturbation $E$ with $\|E\|$ sufficiently smaller than $\delta$ **[thm]**. A top-$r$ cut inside a near-degenerate block is canonical but unstable (and non-unique at exact multiplicity), and individual vectors inside a block are not identifiable in practice. We therefore report **blocks** (delimited by gaps) and their spectra, and when a basis inside a block is wanted for display we rotate by a declared rule **[P4]**; everything in Section 7 that depends on individual modes is stated at block granularity. This is the perturbation-theoretic sense of "really close" behind the insistence on SVDs.

**Hierarchy.** Dyadic powers $P, P^2, P^4, \dots$ with a declared compression threshold define the Coifman–Maggioni diffusion-wavelet multiscale basis by rank-revealing QR; the threshold is what produces a dimension-reducing hierarchy, since exact powers of a full-rank matrix share one range. Whether the coarse scaling functions are indicator-like depends on the geometry **[hyp]**. We treat this as an extension of the spine.

## 4. Coherence across layers: the dynamic operator on sites

### 4.1 Construction

Let $P^L, P^{L'}$ be the bistochastic operators of two layers. Because transport between layers is the identity on $X$ and both operators are Markov for the same measure, the product

$$Q^{L \to L'} = P^{L'} P^{L}, \qquad Q = \sum_i \sigma_i\, u_i\, v_i^\top, \qquad 1 = \sigma_0 \ge \sigma_1 \ge \dots \ge 0,$$

is doubly stochastic with trivial pair $(u_0, v_0) = (\mathbf 1, \mathbf 1)/\sqrt N$, which we remove by centering **[thm]**. This is a finite-state realization of Froyland's finite-time coherent-set operator (diffuse, transport, diffuse) with identity transport and the two layers' bistochastic operators declared as the two diffusions; with the shared measure the realization is exact, though the smoothers are our choice, not canonically determined by the geometries.

What a singular pair means **[thm]**: for a unit vector $v$, $1 - \|P^{L'} P^L v\|^2 = (1 - \|P^L v\|^2) + (\|P^L v\|^2 - \|P^{L'} P^L v\|^2)$, so $\sigma_i \approx 1$ says $v_i$ loses little under diffusion at $L$ *and* its diffused image loses little under diffusion at $L'$: $v_i$ is slow for both geometries. For a set $A$, $\langle Q \mathbf 1_A, \mathbf 1_B\rangle / |A|$ is the fraction of $A$'s mass that, after diffusing at $L$ and then at $L'$, lies in $B$; the centered SVD is the $L^2$-normalized relaxation of maximizing this over pairs $(A, B)$ subject to a balance constraint ($|A| = |B|$ and complements compared), which is what rules out the trivial $B = X$. Interpretation in words:

- **coherent**: $\sigma_i$ near one, $v_i$ and $u_i$ both concentrated on the same sites: a set the network keeps together between $L$ and $L'$;
- **split / merge** **[hyp]**: a contrast function that is slow at $L$ (it separates two clusters there) but whose image is not slow at $L'$ (the clusters have collapsed) indicates a *merge*, two clusters at $L$ becoming one at $L'$; a contrast that is not slow at $L$ but is slow at $L'$ indicates a *split*, one cluster at $L$ becoming two at $L'$, read off $Q^\top = P^L P^{L'}$. (Worked check: activations $(e_1, e_1, e_2, e_2)$ at $L$ and four copies of $e_1$ at $L'$; the contrast $(1,1,-1,-1)/2$ keeps norm $\tanh(\beta/2)$ under $P^L$ and is annihilated by $P^{L'}$: a merge.) Reading a specific split off a pair of signed, possibly oscillatory singular functions requires a level-set or partition step, which is interpretive and must be validated on the extracted sets. When $L = L'$, $\sigma_i = \lambda_i^2$ and small $\sigma$ just means fine-scale, so "small $\sigma$" alone is never evidence of a new distinction.

For several layers $\lbrace L_k\rbrace$ the sequential object is $\prod_k P^{L_k}$ and the simultaneous one is $\tfrac1m \sum_k P^{L_k}$; if each $P^{L_k} = I - \epsilon \mathcal L_k + O(\epsilon^2)$ **[cond]**, the product is $I - \epsilon \sum_k \mathcal L_k$ (total time $m\epsilon$) and the mean is $I - \epsilon\, \tfrac1m \sum_k \mathcal L_k$, so they share the averaged generator only when the product uses bandwidth $\epsilon/m$ per step; at finite bandwidth they differ. The product composes and is the default **[M4]**.

### 4.2 Why this is compatible with independently treated features

If activations were sparse sums of concept directions, a *partition* into coherent sets would be wrong: a site with two active concepts belongs to both. But the SVD does not partition. Singular functions superpose, and a site's coordinates $(v_i(x))_i$ record how much it participates in each coherent mode. Partitions are level sets and an interpretive afterthought; the basis is primitive.

### 4.3 Localization is measured, not assumed

Whether a block of modes is a set of features (its span is near the span of indicators of well-separated clusters) or a set of directions of variation is a property of the data at the working $\beta$. Two diagnostics: the **indicator-subspace recovery**, the principal angles between a block's span and the span of candidate indicators when candidates exist (toy models), and the **support size** of a block when they do not: with $\Pi_B$ the block projector and $w(x) = \Pi_B[x,x] / \dim B$ (a probability over sites), the inverse participation ratio $\mathrm{IPR} = \sum_x w(x)^2$ ranges from $1/N$ (spread over all sites) to $1$ (one site); its reciprocal is the effective number of sites the block lives on. A single mode's IPR is basis-dependent inside a block and a point-localized mode has maximal IPR while meaning nothing, so neither is used alone. When the regime is present, a localizing rotation of the block (varimax or similar) produces a dictionary; the rotation is an algorithm whose recovery is tested, not a display convention [P4].

### 4.4 Three time axes, one machinery

The product construction never used that $L$ indexes depth. The same machinery applies with the index being depth (what the network keeps together as it computes), sequence position via the context view (Section 6), or training step via checkpoints (what the network keeps together as it learns). Delay-embedded past–future CCA and HAVOK share a trajectory (Hankel) matrix but are different estimators. Pughe-Sanford et al. (2025) pair lag vectors to estimate whitened cross-covariances and Koopman singular functions, which is the closest precedent for the carried context construction of Section 6. HAVOK and DMD additionally fit the evolution of reduced lag coordinates (classical HAVOK with a forcing coordinate). Neither follows from a product of contemporaneous smoothers. For the training axis the DMD-like step is a least-squares regression of block coordinates at checkpoint $s+1$ on those at $s$ (closed form, an SVD), labelled as a separate fit in `EXPERIMENTS.md` Stage 3. For the sequence axis the lagged pairing is the construction of Section 6.

## 5. Loadings: getting back to activation space

For a mode $\psi \in \mathcal H$ and any $F \in \mathbb R^{N \times p}$, the loading is $a_F(\psi) = F^\top \psi \in \mathbb R^p$, the coefficient of $\psi$ in the expansion of $F$ **[thm]**. It is a summed cross-moment ($N$ times the empirical one), a covariance only when $F$ or $\psi$ is centered. Properties:

- for the normalized indicator $\psi_A = \mathbf 1_A / \sqrt{|A|}$, $a_F(\psi_A) = \sqrt{|A|}\, \bar F_A$, and the reconstructed contribution $\psi_A a_F(\psi_A)^\top$ equals the centroid $\bar F_A$ on every site of $A$ and zero outside **[thm]**: the "replace by the centroid" re-embedding, derived rather than posited;
- for the linear Gram matrix's eigenvectors $u_i$, $a_{H}(u_i) = \sigma_i v_i$ **[thm]**: normalized PCA score directions and the scaled principal directions. (The eigenvectors of the centered $\beta \to 0$ operator agree with those of the centered Gram of the *row-normalized* activations, Section 2.)
- **readability**: the ridge $R^2$ of regressing $\psi$ on $H^L$ measures whether the mode is a linear function of the layer; it is distinct from the loading (regression uses the inverse covariance) and is reported alongside it. A zero loading forces zero *linear* readability **[thm]**, but a mode on a curved cluster can have zero loading and be a perfect *nonlinear* function of the layer, so a kernel-ridge readability (sphere kernel on $H^L$) is reported too;
- the same mode has a loading at every layer, and the pair $(a_{H^L}(v_i), a_{H^{L'}}(u_i))$ records where a coherent set sits at each layer. The *effective action* of the layer on the mode is a fitted object (the ridge map from $H^L$ to $H^{L'}$ restricted to the block) and is reported as such, not as "the rank-one piece".

Writing $H^L[x] = \sum_i \psi_i(x) a_i$ **[thm]** says: the activation at $x$ is a combination of directions $a_i$ with coefficients $\psi_i(x)$. This has the *form* of a dictionary decomposition. In the finite-$\beta$ clustering regime with a localizing rotation, the codes are near-indicators and the dictionary is SAE-like; in the $\beta \to 0$ limit the codes are PCA scores. Which regime holds is measured (Section 4.3). Out-of-sample sites are handled by the Nyström extension of the eigenfunctions of $P$, which includes the normalization factors and division by $\lambda_i$ and is unstable for small $\lambda_i$; only modes above a declared eigenvalue floor are extended **[M5]**.

## 6. Context: the KV-cache problem, removed and then re-posed honestly

### 6.1 At the level of the spine there is no problem

Because $x$ at $L'$ is the site with its context, $P^{L'}$ already reflects whatever attention or recurrence delivered to $x$. The dynamic operator, the loadings, the unit expansions, and the gradient decomposition (Sections 4, 5, 7.1, 7.2) are all defined without any model of the layer map. This is why the construction is the same for a transformer, an SSM, a CNN, and a recording.

### 6.2 Where it returns

To say *what the layer did* to produce $h^{L+1}(x)$ we need the layer's inputs at $x$. For an MLP that is $h^L(x)$; for attention or an SSM it is $h^L(x)$ together with the past of the stream. Marginalizing the past (treating the layer as a stochastic map of the current activation) gives a lower bound on how much of the layer's action is accounted for, and we do not adopt it as a description. Fitting a hidden-state model means training a parametric recurrent summary of the past; what follows instead is a closed-form statistical summary (a regularized canonical-correlation SVD), which is a fit but not a trained model. What we adopt is a fitted predictive summary: **a predictive state of the stream, obtained as a regularized canonical-correlation SVD.** The direct precedent is the linear lag-vector CCA of Pughe-Sanford et al. (2025, Section 4): rows of past and future lag matrices, centered, whitened by their covariances, and the SVD of $C_{pp}^{-1/2} C_{pq} C_{qq}^{-1/2}$. Ours is the sphere-kernel, ridge-regularized extension of that estimator, with their raw-feature version as the $\beta \to 0$ calibration case (Stage 0), and with a second target (read) they do not consider.

### 6.3 Two context operators

Fix a layer $L$ and a site $x = (\text{doc}, t)$. The **past features** are the context view $z_{\text{past}}(x)$ of Section 2 (lag-weighted window of $\hat h^L$ at positions $< t$). Two choices of **target**:

- **carried**: the future window of the same stream, $\hat h^L$ at positions $> t$. This asks what the layer-$L$ stream carries forward about its own future. It is intrinsic to the stream and is the recording-compatible object.
- **read**: the next layer's update at the site, $\Delta h^{L+1}(x) = h^{L+1}(x) - h^{L}(x)$, or $h^{L+1}(x)$ itself. This asks what the past contributes to what the layer computes *now*.

These are different objects and neither implies the other **[thm]** (example: i.i.d. tokens, layer $L$ the embedding, layer $L+1$ copies the previous token: the past predicts nothing about the stream's future, so the carried state is empty, while the read state is one lag). v0 conflated them; v1 keeps both, with **read** as the operator for attribution and the explained-variance ladder, and **carried** as the architecture-free characterization of the stream **[P5]**.

For either choice, with centered Gram matrices $\tilde K_p = C K_{\text{past}} C$ and $\tilde K_q = C K_{\text{target}} C$ (sphere kernels of Section 2 on past and target views) and ridge $\rho > 0$ **[P6]**, define the kernel-ridge smoothers

$$R_p = \tilde K_p (\tilde K_p + N\rho I)^{-1}, \qquad R_q = \tilde K_q (\tilde K_q + N\rho I)^{-1}.$$

(For the carried target the same construction with the future window in place of $\Delta h^{L+1}$ gives the carried operator and, in Section 6.4, a carried ladder formed analogously.) The regularized canonical correlations between past and target are the singular values of $R_q^{1/2} R_p^{1/2}$, equivalently $\kappa_i^2 = \mathrm{eig}(R_p R_q)$, and, for $\kappa_i > 0$, the **past canonical variates** $s_i \in \mathcal H$ are the corresponding eigenvectors of $R_p R_q$, normalized to unit variance **[thm]** (nullspace eigenvectors are not variates). (This is the Gram-matrix form of $(C_{qq} + \rho)^{-1/2} C_{qp} (C_{pp} + \rho)^{-1/2}$ with centered feature covariances; verified against the primal form in `checks/cm_check.py`.) Properties: $0 \le \kappa_i \le 1$ and sample-normalized **[thm]**. They are **not** invariant to kernel scale at fixed $\rho$ (scaling $\tilde K$ by $c$ rescales the effective ridge to $\rho / c$); sphere kernels have entries in $[e^{-\beta}, e^{\beta}]$, so $\rho$ is comparable across views at a fixed $\beta$ and is chosen by the held-out score in any case. The ridge and the window are chosen by maximizing the sum of squared held-out canonical correlations (the VAMP-2 score of Wu and Noé, from the variational approach to Markov processes) on document-level splits.

The **predictive-state coordinates** of layer $L$ are $s^L(x) = (s_1(x), \dots, s_r(x))$, and the **effective context rank** $r_{\text{ctx}}(L)$ is the number of $\kappa_i$ above a declared threshold on held-out data. This rank is **scale-dependent and empirical**. The stable Ornstein–Uhlenbeck case worked out by Pughe-Sanford et al. (leading functions are linear projections in whitened coordinates) is a linear-CCA calibration case, not a bound for nonlinear activation kernels; their unstable-dynamics extension leaves the reference measure open and nothing from it is imported here. The following are what can and cannot be said about it:

- For an HMM with $k$ hidden states, the Hankel matrix of *observation* past/future probabilities has rank at most $k$ (equal under minimality and sufficiently long windows), and after centering the belief simplex has affine dimension $k - 1$ **[cond]** (spectral HMM learning: Hsu–Kakade–Zhang). This is a statement about the token process and we use it as a separate benchmark computed from tokens.
- For the *activation* stream of a network trained on that HMM, no such bound follows: future activations may retain past tokens, past and future windows need not be conditionally independent given the HMM state, and a nonlinear kernel on a $k$-dimensional continuous state can have infinite population rank. That $r_{\text{ctx}}(L)$ at the belief-representing layer is close to $k - 1$, and that $s^L$ is linearly readable from the true belief state, are **[hyp]**, tested in Stage 4.
- For an SSM, the linear-system Hankel rank is bounded by the state dimension **[cond]** (LTI, linear observation); the nonlinear kernel rank of the activation stream is not, and is measured.
- Golowich, Liu and Shetty (ICLR 2026; preprint 2025) prove that an *extended logit matrix* (histories × (future, next token), mean-centered logits) has rank at most $d$ for all positions **if and only if** the sequence distribution is expressible as a time-varying input-switched affine network of hidden dimension $d$ (their Theorem 4.3; an extra constant coordinate absorbs offsets) **[thm, theirs]**. This is a statement about logits, not about probabilities (softmax raises rank), not about intermediate layers, and not about the implementation being an ISAN. Their matrix evaluates each history against *alternative* future continuations, whereas our estimator uses covariances of *observed* past/target pairs; the two share the predictive-factorization idea, and our construction is inspired by theirs rather than containing it as a special case. A connecting theorem would be needed and we do not claim one.

### 6.4 How context acts on the current site, and the explained-variance ladder

The layer's action at $x$ is modelled as a function of (current activation, predictive state): $\Delta h^{L+1}(x) \approx g(h^L(x), s^L(x))$. In site space this is the product kernel $K^L \cdot K_s$ with $K_s$ a sphere kernel on $s^L(x)$ weighted by the $\kappa_i$; the weighting is a modelling choice, not implied by the SVD **[P7]**. Its operator's eigenfunctions are joint current-and-context modes; their loadings on $H^{L+1}$ are the directions the layer writes for each.

The amount of context the layer reads is a **held-out predictive-risk curve**, not a conditional variance and not a measurement of architectural use:

$$\mathcal R_0,\quad \mathcal R(h^L),\quad \mathcal R(h^L, s^L_{1..r})\ (r = 1, 2, \dots), \qquad \mathcal R(Z) = \text{held-out } \mathbb E\, \|\Delta h^{L+1} - \hat g_Z\|^2,$$

with $\mathcal R_0$ the risk of predicting the mean update and $\hat g_Z$ a kernel ridge regression on the variables $Z$, cross-fitted on document-level splits so that the state coordinates, the ridge, and the evaluation never share sites (an unregularized in-sample kernel regression interpolates everything and would show zero residual regardless of what the layer reads). The corresponding *Bayes* risks are nested and monotone **[thm]**; the fitted held-out risks carry no monotonicity guarantee and no zero limit (finite windows, finite features), and are reported as a sequence, not an inequality. The drop from $\mathcal R_0$ to $\mathcal R(h^L)$ is the part of the layer predictable from the current activation alone ("an MLP in disguise"); the further drop with $r$ is the incremental predictive information in the context, at this window and scale. It is predictive information, not identified architectural use. The i.i.d. previous-token-copy construction is an early calibration case (Stage 0) where the carried curve must stay flat and the read curve must drop within the first few coordinates (a $V$-ary previous token has up to $V - 1$ independent centered predictive coordinates, so the drop is not at $r = 1$).

## 7. Pulling back to the computation

### 7.1 Every unit is a function on sites

A neuron $n \in \mathcal H$, a head output $O \in \mathbb R^{N \times d}$, an SSM state coordinate: each has the exact expansion $n = \sum_i \langle n, \psi_i\rangle \psi_i$ **[thm]**. The **mode profile** of a unit is $(\langle n, \psi_i\rangle)_i$, reported at block granularity. It measures association with this basis at this scale; it is not itself a mechanism. In particular, spread across blocks can come from rectifying a *single* spectral function rather than from responding to several features: Pughe-Sanford et al. (2025) model a sensory neuron as $n = H(\pm v_i)$ or $\mathrm{ReLU}(\pm v_i)$ for one Koopman singular function $v_i$, and the threshold puts energy into the constant and many other modes. So the pre-rectification projection and the rectified response are both expanded, and "detects one coherent structure" means: the pre-rectification input is concentrated on one block **[hyp]**. A unit whose response is concentrated on the small-$\sigma$ tail is responding to what the basis treats as fine structure, which may be noise, response harmonics, or essential computation; the gradient side (7.2) and the functional checks of Stage 2 are what distinguish them. For compressed computation, Ferreira da Silva and Heimersheim (2026, arXiv:2607.04800) find each input feature encoded by a sparse binary codeword of 5–7 hidden neurons with a near-pseudoinverse decoder; the effective feature computation $W_2[j,:]\,\mathrm{ReLU}(W_1 x) \approx (x_j)_+$ is what we call a "virtual neuron" (our term, not theirs). A block whose hidden-layer loading is spread over several neurons is a *candidate* signature of such a codeword **[hyp]**, to be validated against the codeword supports, decoder structure and output gains in Stage 2; distributed loading is evidence of distributed linear association, not of a joint computation.

### 7.2 The exact mode decomposition of a gradient

Let $W$ be a weight matrix applied as $z = W u$. For each **use** of $W$ there is an input $u$ and a backpropagated error $\delta = \partial \mathcal L / \partial z$. When every site uses $W$ once (every projection in a transformer or SSM block, counting key/value projections at the key site), $U \in \mathbb R^{N \times p}$, $\Delta \in \mathbb R^{N \times q}$ are functions on sites and

$$G = \nabla_W \mathcal L = \Delta^\top U = \Delta^\top \Psi \Psi^\top U = \sum_i g_i\, b_i^\top, \qquad g_i = \Delta^\top \psi_i, \quad b_i = U^\top \psi_i . \quad \text{[thm]}$$

The gradient is exactly a sum over modes of rank-one terms: the descent step $-\eta G$ moves the output of inputs along $b_i$ in the direction $-g_i$. When a site uses $W$ several times (convolution offsets, recurrent steps), index the uses by a slot $a$ so that $U_a, \Delta_a$ are functions on sites for each slot; then $G = \sum_a \Delta_a^\top U_a = \sum_a \sum_i (\Delta_a^\top \psi_i)(U_a^\top \psi_i)^\top$ exactly, and the per-mode piece $\sum_a g_{i,a} b_{i,a}^\top$ has rank up to the number of slots **[thm]**.  A composite such as $W_Q^\top W_K$ is not a parameter; each factor is decomposed separately with its own uses. Direct parameter terms in the loss (weight decay) are added separately. Only the training loss and the training sites enter **[A3]**; for a pretrained model without its corpus a held-out sample from the same distribution is substituted and flagged.

What can be read off, and its limits:

- **block pressure** $\|G_B\|_F$ with $G_B = \Delta^\top \Pi_B U$ the contribution of an isolated block $B$: invariant to rotation within the block **[thm]**. Individual $\|g_i\|\|b_i\|$ are not invariant and are not reported.
- **signed interaction** between blocks, $I_{BB'} = \langle G_B, G_{B'}\rangle_F$: the alignment of the two spectral contributions to the gradient; positive is cooperative, negative is interference. This is **not** in general the alignment of site-group loss gradients: the gradient of the loss on a site group $A$ is $\Delta^\top D_A U$ with $D_A = \mathrm{diag}(\mathbf 1_A)$, and $\Pi_B$ equals $D_A$ only when $B$ is the whole coordinate subspace supported on $A$, not when it is the single indicator (which averages $u$ and $\delta$ over the group before taking the outer product) **[thm]**. The two statistics are both reported where site groups exist; the spectral one is the one defined for every block. Even the site-group gradient equals a group *loss* gradient only when the loss separates over sites; attention couples sites, so there the error $\delta_x$ already aggregates downstream cross-site terms. 
- **truncation**: $G - G_r = \sum_{i > r} g_i b_i^\top$ exactly, but the pieces are not Frobenius-orthogonal, so the residual is computed, not summed; the bound $\|G - G_r\|_F \le (\sum_{i>r} \|g_i\|^2)^{1/2} (\sum_{i>r} \|b_i\|^2)^{1/2}$ holds **[thm]**.
- **gradient, not update.** This decomposes the gradient. Momentum, Adam's preconditioning, clipping and weight decay change the update and are not rank-one over modes. The optimizer-agnostic statement is on the activation side: any update $\Delta W$ changes the layer's output on the sites, *with its inputs frozen*, by $\delta Z = U\, \Delta W^\top$, a function on sites, whose expansion $\delta Z = \Psi \Psi^\top \delta Z$ is exact for any optimizer **[thm]**. This is the direct effect; the realized change also includes upstream updates and, for shared uses, the change of $W$'s own inputs. This is the precise form of "the pullback to the weights only constrains the update to the activation cloud": the activation-side decomposition of the realized update is always exact; the weight-side rank-one decomposition is exact for the gradient.
- **refinement and maintenance** **[hyp]**: a coherent block at $L$ whose halves (as seen at $L'$) have opposing error loadings is under pressure to split; a small-$\sigma$ block with persistent pressure across checkpoints is being maintained rather than being noise. Both are hypotheses the decomposition lets us test, not consequences of it. The method reports $\sigma$, pressure, and interaction and does not decide ex ante.

### 7.3 The action view: similarity of local derivatives

The same-site Jacobian $J^L(x) = \partial h^{L+1}(x) / \partial h^L(x)$ (cross-site blocks exist for attention and are what the context view summarizes) defines the action view $\phi(x) = \mathrm{vec}\, J^L(x)$, normalized, with the sphere kernel $\exp(\beta \langle \hat J(x), \hat J(y)\rangle_F)$ so that Section 3 applies unchanged (the raw cosine Gram has negative entries and is not a diffusion kernel) **[M6]**. It says two sites are treated alike when the layer applies positively proportional local linear maps to them. For a ReLU MLP, $J = W_2\, \mathrm{diag}(m(x))\, W_1$ with $m(x)$ the active mask, away from activation boundaries **[thm]**, so the action view is a signed, weighted comparison of masks: the activation-plateau / stable-region structure of Janiak et al. and the per-input affine map of the ISAN picture, each as a kernel on sites. Equal normalized Jacobians do not imply one affine map or equal gain (magnitude and offset $h^{L+1}(x) - J(x) h^L(x)$ can differ); when affine-cell recovery is the target, magnitude and offset are appended as coordinates with declared scales [P1]. In a bias-free network the cellwise offset is zero and adds nothing, while normalized Jacobians still hide the output attenuation (gain about $0.81$) reported by Ferreira da Silva and Heimersheim (2026), so gain is retained whenever it is a target; whether the view is hidden-layer or end-to-end is declared. An action block is a class of similar local maps, not an identified feature code. $J$ is never formed: with $m_J$ i.i.d. probes $v_r$ shared across sites, $\mathbb E[v v^\top] = I$ and finite fourth moments (Gaussian or Rademacher), $\tfrac1{m_J}\sum_r \langle J(x) v_r, J(y) v_r\rangle$ is unbiased for $\langle J(x), J(y)\rangle_F$ with standard error $O(m_J^{-1/2})$ **[thm]**, from $m_J$ Jacobian-vector products per site. Vector diffusion maps (Singer–Wu) would use $J$ as a connection rather than only its Gram matrix; noted, not used.

Standalone comparisons of weight magnitudes are deliberately absent (weights enter through the Jacobian and the gradient): comparing a weight to zero is comparison to zero-ablation, and the Jacobian and gradient compare the network to itself.

## 8. Comparing models and predicting stitching

Two models $A, B$ on the same sites give operators on the same $\mathcal H$, so comparison needs no Procrustes step:

- **subspace distance**: principal angles between isolated blocks of $\tilde P^A_L$ and $\tilde P^B_{L'}$ (Davis–Kahan stable relative to the gaps) **[thm]**;
- **centered alignment**: $\langle \tilde P^A, \tilde P^B\rangle_F / (\|\tilde P^A\|_F \|\tilde P^B\|_F)$ on the *centered* operators. As $\beta \to 0$, $C P_\beta C = \tfrac{\beta}{N} C \hat\Phi \hat\Phi^\top C + O(\beta^2)$ **[thm]** (the non-constant first-order terms of the Sinkhorn scaling cancel under centering; numerically verified, relative error $\approx \beta/4$), so the centered alignment tends to **linear CKA of the row-normalized activations** (verified to four digits in `checks/cm_check.py`). The uncentered statistic tends to one for every pair at both $\beta \to 0$ and $\beta \to \infty$ and is not used; the centered one also tends to one as $\beta \to \infty$ (both operators tend to $C$), which is one more reason the scan, not an endpoint, is the report. Finite $\beta$ changes which pathologies the statistic has (it removes magnitude sensitivity and adds bandwidth and density sensitivity); it does not remove them, and we report the $\beta$-scan rather than claim robustness.
- **stitching prediction [hyp]**: a linear stitch from $A$'s layer $L$ into $B$'s layer $L'$ succeeds to the extent that the blocks of $B$ that the downstream network is sensitive to are linearly readable from $H^A_L$. Per-block readability is the ridge $R^2$ of $B$'s block coordinates on $H^A_L$ (a closed-form fit, which is a fit). Sensitivity is **not** gradient pressure, which vanishes at an exact fit while stitching can still fail; it is the output sensitivity of the block, $\big(\sum_{\psi \in B} \sum_x \|J_{L' \to \text{out}}(x)\, a_{H^B_{L'}}(\psi)\|^2_{F_{\text{out}}(x)}\big)^{1/2}$ in the output Fisher norm (Section 9), summed over an orthonormal basis of the block so that it is rotation-invariant, and over sites: the first-order effect on the predicted distributions of perturbing along the block's directions. The predictor is sensitivity-weighted unreadability. It is compared in Stage 5 against unweighted readability, direct activation ridge reconstruction, CKA, and actual trained stitches of both task-loss and direct-matching kinds, which the stitching literature shows can disagree.

Across training ($A, B$ two checkpoints) and across layers ($A = B$) the same primitives apply.

## 9. Metrics other than cosine (variant, not spine)

The output Fisher information $F_{\text{logits}} = \mathrm{diag}(p) - p p^\top$ pulls back through the downstream network to the PSD form $g_L(x) = J_{L \to \text{out}}(x)^\top F_{\text{out}} J_{L \to \text{out}}(x)$ on the layer-$L$ stream at each site **[thm]**. "Two activations are the same if the output cannot tell them apart to first order" is the kernel built from it. Per-site it is $d_L \times d_L$, anisotropic, and does not by itself define a PSD kernel on sites (exponentiated geodesic distance for a general metric need not be PSD); the site-averaged form defines a Mahalanobis sphere kernel that is PSD (with the zero-vector policy [P2] applied in its nullspace). The choice (cosine; averaged Fisher pullback; per-site, via a sketched metric) is **[M7]**, decided by experiment, since the output view already carries Fisher–Rao at the end and $Q^{L \to \text{out}}$ already couples layer-$L$ geometry to it.

## 10. Temperature

$\beta$ is the one knob the theory cannot take from the data without a further choice **[P8]**. Candidates, in decreasing order of how principled we judge them:

1. **Noise-induced scale.** The heuristic: two activations closer than the perturbation the network itself undergoes during training are not worth distinguishing at that resolution. This is a bandwidth heuristic, not an indistinguishability theorem. Defined as the pairwise variation of $\hat h^L(x)$ under *stochastic variation around fixed parameters*: from one checkpoint, take $n$ independent single minibatch steps using the training optimizer, its saved state, step size and batch sampling (when the state is unavailable, plain SGD at the training learning rate, declared as such), evaluate $\hat h^L(x)$ after each, remove the mean displacement, and take the mean squared pairwise difference (or use dropout / batch-composition resamples at the fixed parameters); $\beta_{\text{noise}} = 1 / \overline{\|\delta \hat h\|^2}$. Adjacent-checkpoint displacement is **not** this (it mixes drift with noise and depends on the checkpoint interval) and is used only as a heuristic with the interval declared. Whether the noise scale is layer-wise constant late in training, as edge-of-stability and batch-noise results might suggest, is **[hyp]**. The noise is anisotropic and we use its mean square; an anisotropic version is the Fisher-type variant of Section 9.
2. **Nearest-neighbour scale**: $\beta_{\text{nn}} = 1 / \mathrm{median}_x \|\hat\phi(x) - \hat\phi(x_{(k)})\|^2$ with $x_{(k)}$ the $k$-th nearest neighbour ($k \approx 10$), a global version of the self-tuning heuristic; data-driven, not network-driven.
3. **Scan**: spectra along $\beta$ with a mode-matching convention across the scan. This is the coarse-graining story and is reported as supporting material; a persistence diagram would need a declared filtration and we do not claim one.

Default: $\beta_{\text{noise}}$ when stochastic resamples can be produced (every training run we control; pretrained models whose training data and optimizer settings are public, such as Pythia), else $\beta_{\text{nn}}$, plus the scan. For comparisons across training, $\beta$ is **held fixed** across checkpoints (otherwise a changing measurement scale confounds a changing representation) and the adaptive-$\beta$ result is reported alongside. The $\beta \to 0$ limit is always reported as the point of contact with linear CKA and PCA.

## 11. Hypotheses on cases where something is known

All **[hyp]**; details and controls in `EXPERIMENTS.md`, where each target is labelled identity / planted / established / hypothesis.

- **Toy models of superposition**: in a $\beta$ window, an isolated block of about $n$ modes whose span recovers the span of single-feature-active indicators (principal angles), and whose localizing rotation has hidden-layer loadings matching the columns of $W$ one-to-one; the hidden $\to$ output operator keeps those blocks coherent. As sparsity decreases, the block loses its gap and the indicator-subspace recovery degrades, and the report says so.
- **Computation in superposition** (Ferreira da Silva and Heimersheim 2026; bias-free, residual-free $100 \to 50 \to 100$ ReLU network, mean fourth-power loss, independently sparse signed inputs): blocks or their localizing rotations recover the feature-code structure (sparse binary codewords, near-pseudoinverse decoder, attenuated gains) when compared with it directly, and the $L^2$-trained control (which serves half the features exactly and drops the rest) does not show it. Isolated-positive hidden responses have rank at most 50 while the 100 centered feature indicators have rank 99, so not every feature can be linearly readable from the hidden layer; nonlinear-kernel recovery does not imply it. Under natural sampling active-feature sets overlap and are not a partition. Correlation of the signed gradient interaction with functional leakage and pairwise non-additivity is a separate hypothesis.
- **Modular addition / grokking**: on sites = (example, final position), the post-attention geometry has an isolated block spanned by the product harmonics at the network's frequencies; the post-attention $\to$ post-MLP operator keeps the frequency blocks coherent and merges sites with equal $a + b \bmod p$; across checkpoints the block pressure on the eventual frequencies rises before the test loss falls.
- **HMM belief states**: at the belief-representing layer, $r_{\text{ctx}}(L)$ is close to $k - 1$ and $s^L$ is linearly readable from the true belief state, for a transformer and an SSM alike; the token-level Hankel benchmark has rank $k$.
- **MNIST / dSprites / CIFAR**: coherent blocks across layers align with classes at coarse $t$ and with confusion structure at fine $t$; on dSprites with a factor-prediction objective the top blocks are products of per-factor functions; predicted stitching fidelity between seeds correlates with trained stitches.

A failure on a large model is informative only if the language localizes it: a layer with no gap, a block that does not localize, a ladder that never drops. Those are the diagnostic statements the method is built to make.

## 12. Assumptions register

Substantive:
- **[A1]** Same site at two layers is the same site; a site correspondence must be declared across models.
- **[A2]** Dot product on each view after normalization.
- **[A3]** Only the network's own training objective on the network's own sites enters the gradient side.

Pro-tem:
- **[P1]** Norm dropped (restorable as $\log\|h\|$, a scale choice); magnitude/offset for the action view likewise.
- **[P2]** Zero-vector policy: excluded, mass reported.
- **[P3]** Lag weights in the context view.
- **[P4]** Rotation inside near-degenerate blocks; blocks are the canonical object.
- **[P5]** Carried and read context operators both computed; read used for attribution.
- **[P6]** Ridge $\rho$ and window $w$ by held-out canonical correlation.
- **[P7]** $\kappa$-weighting of state coordinates in the joint kernel.
- **[P8]** Temperature from stochastic variation at fixed parameters, else nearest-neighbour scale; fixed across checkpoints for training comparisons.

Methodological:
- **[M1]** Counting measure and Euclidean orthonormality.
- **[M2]** Fisher only at the output by default.
- **[M3]** Bistochastic normalization. This is not cosmetic: it is what makes layer operators composable, and it fixes the manifold-limit density exponent.
- **[M4]** Product over layers; the mean is the alternative at finite bandwidth.
- **[M5]** Nyström extension above an eigenvalue floor.
- **[M6]** Exponentiated Jacobian cosine as the action kernel.
- **[M7]** Cosine vs averaged vs per-site Fisher pullback.
- **[M8]** Dense-kernel and $k$-NN-graph estimands kept as separate code paths (Section 13); the graph operator need not be PSD, so the non-negative-spectrum statements of Section 3 apply to the dense estimand only.

Conditions that stronger claims need and the network need not satisfy: a finite-$\beta$ clustering window with separation; minimality and window sufficiency for Hankel rank statements; conditional independence of past and target given a finite state; sampling regularity and weak dependence of sites for spectral convergence; a first-order relation between sensitivity and stitching loss. Each appears with its [cond] or [hyp] marker where used.

Known limitations:
- Clustering is a regime, not a limit; when many features are simultaneously active the blocks lose their gaps and the dictionary reading is unavailable.
- Long-range context enters only through the stream at past positions within the window.
- The Jacobian and gradient parts are first-order; ablation-style counterfactuals are outside the spine by design.
- Identity transport is bookkeeping; attribution of an output to input sites is the separate step of Sections 6.4 and 7.

## 13. Computation: estimands, costs, and what each knob does

There are **two estimands** and they must not be confused **[M8]**:

- the **dense-kernel operator**, the object of Sections 2–4, approximated at low $\beta$ (nearly low-rank kernel) by Nyström with $m$ landmarks; error depends on landmark coverage as well as the kernel's spectral tail, so landmarks are chosen by leverage or $k$-means and oversampled; Nyström preserves PSD but not entrywise positivity, so scalability by Sinkhorn is checked before normalizing the approximation;
- the **$k$-NN graph operator**, a different kernel (sparse, symmetrized, not guaranteed PSD, so its eigenvalues can be negative and fractional powers need the graph-Laplacian convention), appropriate at high $\beta$ where the dense kernel is nearly sparse. Whether it approximates the dense operator at a given $\beta$ is measured on dense subsets, not assumed; "$k \gg \log N$" is a connectivity condition, not a spectral guarantee.

Costs (configurations to be measured, not guarantees): a dense $N = 3 \times 10^4$ kernel is $3.6$ GB in float32, so dense work tops out near $N \approx 10^4$ on a laptop and $3 \times 10^4$ on a workstation; sparse graphs with $k \approx 50$ reach $N \approx 10^6$ with Lanczos cost per sweep $O(N k)$ and total cost depending on gaps and tolerance; products of operators are applied as matvecs and never formed; the action view costs $m_J$ JVPs per site; the gradient decomposition costs one backward pass plus storage of per-site $(u, \delta)$ for the chosen matrices.

Convergence: for a fixed smooth function, the kernel Laplacian applied to it has bias $O(\epsilon)$ and standard deviation $O\big((N \epsilon^{d_{\text{int}}/2 + 1})^{-1/2}\big)$ with $\epsilon \asymp 1/\beta$, under interior, smoothness, and independent-sampling assumptions **[cond]**; eigenvector convergence additionally needs gaps, and token sites are dependent. We use this only to set expectations and rely on **resampling stability** (bootstrap over documents) for actual error bars.

Knob roles (replacing v0's "every knob is monotone"):

| knob | role | guaranteed monotone? |
|---|---|---|
| $N$ sites | statistical accuracy | no; typically decreasing, with resampling error bars |
| $\beta$ | resolution; too large → identity limit | no (bias–variance) |
| $t$ diffusion time | smooth coarse-graining, $\lambda^t$ | yes for the PSD dense operator [thm]; sparse graphs need the Laplacian-semigroup convention |
| $r$ modes kept | truncation; tail sum of loadings | yes for reconstruction [thm]; no for estimation or gradient residual |
| $k$ neighbours | graph estimand fidelity | no |
| $m$ landmarks | Nyström fidelity | no under arbitrary selection |
| $w$, $\rho$ window, ridge | context operator bias–variance | no; chosen by held-out score |
| $m_J$ probes | action-view variance | yes under the i.i.d. finite-moment probe assumptions [thm] |

## 14. Relation to existing tools

- **RSA**: the shared-index comparison move; here extended by cross-view operators, their composition, and the exact unit and gradient expansions.
- **PCA / linear CKA**: the $\beta \to 0$ limit of the centered geometric operator (Section 8).
- **SAEs and hierarchical dictionaries**: the finite-$\beta$ clustering regime with a localizing rotation (Section 5), hierarchy via diffusion wavelets; no training, and the regime's presence is measured.
- **Froyland's finite-time coherent sets**: Section 4 with identity transport and a shared measure, made exact by bistochastic normalization.
- **Kernel transfer operators / kernel EDMD, VAMP, HAVOK**: shared spectral and predictive-factorization tools, not interchangeable estimators. Pughe-Sanford, Ding, Moore, Sengupta, Epstein, Greengard and Chklovskii, "Neurons as Detectors of Coherent Sets in Sensory Dynamics" (NeurIPS 2025, arXiv:2510.26955), is the direct precedent for both the neural coherent-set reading of Section 7.1 and the lag-vector CCA of the carried context view; it is not a HAVOK implementation and says nothing about products of layer smoothers. The HAVOK/DMD lagged-evolution fit is separate (Section 4.4).
- **Compressed computation / learned sparse codes**: Ferreira da Silva and Heimersheim, "Compressed Computation under L4 Loss is likely Computation in Superposition" (2026, arXiv:2607.04800), supplies Stage 2's toy and its partially reverse-engineered mechanism. Constructions and limits of both papers are in `reviews/papers.md`.
- **Subspace identification, predictive state representations, spectral HMM learning, the extended logit matrix of Golowich et al.**: Section 6.3 shares their predictive-factorization idea; their rank theorems concern linear or token-level objects and are not special cases of our estimator.
- **ISAN and stable regions**: Section 7.3.
- **Vector diffusion maps, Hodge Laplacians**: extensions using Jacobians as connections; not in the spine.
- **Model stitching and its failure analyses**: Section 8, as a hypothesis with controls.

## 15. Unresolved choices

Six design choices remain open and are the project owner's to make; they are listed with their consequences in `DECISIONS.md`. In brief: the site-as-atom framing [A1]; uniform site weighting under bistochastic normalization [M3]; carried vs read context as the spine [P5]; the requirement of stochastic resamples for the noise temperature [P8]; the substitute corpus for pretrained models [A3]; output-sensitivity rather than gradient pressure as the stitching weight (Section 8).

## 16. Glossary

- **CKA**: centered kernel alignment, a similarity between two representations of the same inputs via their centered Gram matrices; *linear CKA* uses the linear kernel.
- **CNN / MLP / SSM**: convolutional network / multilayer perceptron / state-space model (e.g. Mamba), the latter a recurrent architecture with a continuous hidden state.
- **DMD / HAVOK**: dynamic mode decomposition, a least-squares linear model between successive snapshots; HAVOK (Hankel alternative view of Koopman) applies it to delay-embedded (Hankel) coordinates.
- **EDMD**: extended DMD, the same with a feature or kernel lift.
- **Hankel matrix (of a process)**: the matrix indexed by pasts and futures whose entries are joint or conditional probabilities (or features); its rank bounds the dimension of a state that mediates past and future.
- **HMM**: hidden Markov model; the *belief state* is the posterior over hidden states given the past, which lives on the probability simplex.
- **IPR**: inverse participation ratio, Section 4.3.
- **ISAN**: input-switched affine network, a sequence model whose state update is an affine map chosen by the current input.
- **JVP**: Jacobian-vector product, computed by forward-mode automatic differentiation.
- **KV cache**: in a transformer, the keys and values of all earlier positions that attention at the current position reads.
- **LTI**: linear time-invariant system.
- **PSR**: predictive state representation, a model of a process by the predictions of its future from its past.
- **RSA**: representational similarity analysis, comparing systems through their pairwise similarity matrices over shared stimuli.
- **SAE**: sparse autoencoder, a trained dictionary that writes activations as sparse combinations of learned directions ("features").
- **TMoS**: toy models of superposition (Elhage et al. 2022), sparse features compressed through a narrow layer.
- **VAMP**: variational approach to Markov processes (Wu and Noé 2017); its VAMP-2 score is the sum of squared canonical correlations between past and future, used for model selection.

## Appendix A. Revision notes

- **v0 → v1.** The draft mixed $L^2(\mu)$ and Euclidean conventions; v1 uses the counting measure throughout. The draft normalized each layer's kernel separately ($D^{-1/2} K D^{-1/2}$), so layer operators had different stationary vectors and their product had top singular value below one from density mismatch alone; v1 uses bistochastic normalization. The draft compared uncentered operators, whose alignment tends to one for every pair at $\beta \to 0$; v1 centers. The draft described $\beta \to \infty$ as a clustering limit; it is the identity limit. The draft's context estimator $(K_q + \rho)^{-1/2} K_q K_p (K_p + \rho)^{-1/2}$ was not covariance-whitened and its singular values grew with $N$ and kernel scale; v1 uses the regularized canonical-correlation form. The draft conflated the predictive state of the stream's own future with the state the next layer reads; v1 separates carried and read. The draft's per-mode pressure $\|g_i\|\|b_i\|$ was not invariant inside degenerate blocks and its interference $\langle b_i, b_j\rangle \|g_i\|$ was unsigned and omitted the error side; v1 uses block pressure and signed interaction.
- **v1 → v1.2.** Kernel-scale invariance of the canonical correlations at fixed ridge was wrong and removed. The indicator-block gradient contribution is not the site-group loss gradient (counterexample in `checks/cm_check.py`). The bistochastic continuum limit is the $\alpha = \tfrac12$ normalization, not $\alpha = 1$. The multi-use gradient expansion was rewritten with a use slot. The fitted-risk ladder lost its inequality signs. Several [thm] markers were softened to [cond] (generator expansions, Davis–Kahan constant, probe moments).
- **v1.4 → v1.5.** The two previously unidentified references were supplied and incorporated (Pughe-Sanford et al. 2025; Ferreira da Silva and Heimersheim 2026): attribution in Sections 0, 4.4, 6.2, 7.1, 14; the pre- vs post-rectification distinction in 7.1; the precise Stage 2 mechanism in 7.3 and 11.
- **v1.3 → v1.4.** Split and merge were stated the wrong way round in Section 4.1 (found while writing `REPORT.md`); noise-temperature resampling protocol specified; citation year fixed.
- **v1.2 → v1.3.** Readability: roadmap, glossary, scoped claim markers, definitions moved to first use, revision history moved here, open decisions moved to `DECISIONS.md`.
