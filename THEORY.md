# Coherent modes: a site-space spectral language for what a network does to its activations

Status: v0 draft, 2026-11-07. Not yet reviewed.

## 0. What this document is for

We want a description of a trained network that

1. is built only from quantities we can compute from the network, its training data, and its training objective: no posited "task distribution", no trained probe or dictionary, no semantic labels in the construction;
2. is scale-sensitive: it has explicit coarse-graining knobs whose effect on resolution and on statistical accuracy is visible and monotone;
3. ends in SVDs, so that every object comes with a spectrum, a nested family of subspaces, and perturbation bounds (Davis–Kahan) that give us an honest notion of "two descriptions are close";
4. lives on an object that is shared across layers, across training checkpoints, and across models trained on the same data, so that comparison needs no fitting;
5. says something about the weights: given a decomposition of what happens to the activations, we want the decomposition to pull back to a decomposition of the parameter update, so that training dynamics can be read in the same language;
6. does not assume sparsity, linearity of features, or any particular architecture (transformer, SSM, RNN, CNN, or a recorded brain).

The construction below meets these by making one move that the rest follows from: **everything is a function on the set of sites.** A site is one place where the network has an activation: one (input, position) pair for a sequence model, one input for a classifier, one (trial, time bin) for a recording. Every measurable thing about a site, at any layer, defines a kernel on sites, hence an operator on functions on sites, hence an SVD. Operators built from different layers, different models, or different kinds of measurement act on the same space, so they compose and compare without any alignment step.

Assumptions are tagged throughout:

- **[A]** substantive: a real modelling commitment that we believe matters and are making on purpose;
- **[P]** pro-tem: a choice made because the natural quantity is intractable or does not exist, to be revisited;
- **[M]** methodological: an implementation choice we expect not to matter at the level of the theory.

Section 12 collects them.

## 1. The spine: sites and the Hilbert space over them

### 1.1 Sites

Let $X$ be a finite set of sites, $|X| = N$, with the empirical measure $\mu$ (uniform unless stated). For a causal sequence model run on a corpus, $x = (\text{document}, t)$. For a classifier, $x$ is one input. The site is the atom of the theory: it is the thing that "has" activations at every layer and that the network "treats" in some way.

**[A1] The same site at two layers is the same site.** This sounds vacuous and is the central move. It means we never need a map from activations at layer $L$ to activations at layer $L'$ as a function of the layer-$L$ activation alone. The transport from $L$ to $L'$ is the identity on $X$. Whatever information arrives at site $x$ at layer $L'$ from other positions (attention, convolution, a recurrent state) is already accounted for, because $x$ at layer $L'$ is, by definition, the site with that context. Section 6 explains how this both dissolves and then honestly re-poses the KV-cache problem.

### 1.2 Functions on sites

$\mathcal H = L^2(X, \mu) \cong \mathbb R^N$ with inner product $\langle f, g\rangle = \sum_x \mu(x) f(x) g(x)$. Everything we compute is an element or operator of $\mathcal H$:

- the activation matrix at layer $L$, $H^L \in \mathbb R^{N \times d_L}$, is a $d_L$-tuple of functions on sites (its columns);
- a neuron's activation, an attention head's output, an SSM state coordinate, a logit, the backpropagated error at a unit: all are functions on sites (vector-valued where appropriate);
- a kernel $K[x, y]$ is an operator on $\mathcal H$.

### 1.3 The SVD duality, which the whole theory generalizes

The plain SVD $H^L = U \Sigma V^\top$ already has the two faces we want: the left singular vectors $u_i \in \mathcal H$ are functions on sites; the right singular vectors $v_i \in \mathbb R^{d_L}$ are directions in activation space; the singular values connect them. The left factor can be computed from the Gram matrix $H^L (H^L)^\top$, a kernel on sites, alone.

The theory replaces the linear Gram matrix by a nonlinear kernel on sites (Section 2), keeps the left factor as the primary object (functions on sites, Section 3), and recovers the right factor as a *loading* (Section 5). For any orthonormal basis $\Psi = [\psi_1, \dots, \psi_N]$ of $\mathcal H$,

$$H^L = \Psi \Psi^\top H^L = \sum_i \psi_i\, a_i^\top, \qquad a_i := (H^L)^\top \psi_i \in \mathbb R^{d_L}.$$

This is an identity, not an approximation. Truncating to $r$ modes has reconstruction error exactly $\sum_{i > r} \|a_i\|^2$. So the spine never truncates, truncation is allowed at any step, and its cost is always a computable tail sum. This is the precise sense in which the construction "hell or high water ends in an SVD" and in which spectral truncation guarantees come for free.

## 2. Views: every measurement of a site is a dot product on a sphere

A *view* is a map from sites to vectors, $\phi: X \to \mathbb R^p$, together with the normalization $\hat\phi = \phi / \|\phi\|$, and the kernel

$$K_\beta[x, y] = \exp\big(\beta\, \langle \hat\phi(x), \hat\phi(y) \rangle\big).$$

On the unit sphere $\exp(\beta \cos\theta) = e^{\beta} \exp(-\beta \|\hat\phi(x) - \hat\phi(y)\|^2 / 2)$, so this is the Gaussian kernel restricted to the sphere (the von Mises–Fisher kernel), and all diffusion-maps theory applies. Two limits:

- $\beta \to 0$: $K_\beta \approx 1 + \beta \langle \hat\phi(x), \hat\phi(y)\rangle$. After centering, the operator is the linear Gram matrix: eigenfunctions are PCA scores, and comparison between two views is linear CKA. **The linear-features picture is the $\beta \to 0$ limit of this theory.**
- $\beta \to \infty$: the kernel becomes a nearest-neighbour graph: eigenfunctions localize on clusters. **The clustering / dictionary picture is the $\beta \to \infty$ limit.**

**[A2] The only geometric commitment is the dot product.** We choose the inner product on each view and nothing else. The residual stream is taken with its Euclidean dot product after normalization (cosine). This is the one place where we knowingly import a substantive assumption and we believe it is right for residual streams (LayerNorm, the empirical near-isotropy results, the success of linear probes and SAEs under cosine). Where the norm matters (attention sinks, confidence scaling), $\log\|\phi\|$ can be appended as one more coordinate before normalization **[P1]**.

The views we use:

| view | $\phi(x)$ | what the kernel says two sites share |
|---|---|---|
| geometric, layer $L$ | residual stream $h^L(x)$ | "look alike to the next layer" |
| output | $\sqrt{p(\cdot \mid x)}$, the square root of the next-token (or class) distribution | Fisher–Rao proximity of what the network predicts |
| context, layer $L$ | the past window of the stream, $(h^L(x_{-1}), \dots, h^L(x_{-w}))$, lag-weighted | "arrive with a similar recent past" (Section 6) |
| action, layer $L$ | the local Jacobian $J^L(x) = \partial h^{L+1}(x) / \partial h^L(x)$, sketched | "the layer applies a similar local linear map" (Section 7) |
| any unit | a neuron, head, state coordinate | used not as a kernel but as a function to expand (Section 7) |

The output view deserves a remark. The Fisher–Rao metric on the probability simplex is, up to a factor 2, the round metric on the sphere under $p \mapsto \sqrt p$, and the Bhattacharyya affinity $\sum_k \sqrt{p_k q_k}$ is the dot product there. So "use Fisher–Rao at the logit layer" is exactly "use the same kernel on $\sqrt p$." Nothing new enters. This is the only place Fisher information is used by default **[M1]**; pulling a Fisher metric back to intermediate layers is possible (Section 9) but we treat it as an experimental variant, not a part of the spine.

## 3. Operators and their spectra

From a kernel $K$ on $X$ form the degree $d(x) = \sum_y K[x, y]$ and the symmetric normalized operator

$$S = D^{-1/2} K D^{-1/2}, \qquad S \psi_i = \lambda_i \psi_i, \quad 1 = \lambda_0 \ge \lambda_1 \ge \dots \ge 0.$$

$S$ is conjugate to the random-walk operator $P = D^{-1} K$. In the large-$N$, small-$1/\beta$ limit it converges to a diffusion semigroup on the view manifold; which one depends on the Coifman–Lafon density exponent $\alpha$ **[M2]**. We use $\alpha = 0$ (density-weighted) by default: when the network puts many sites in a region, that region matters more, and since the dataset is shared across everything we compare, density differences reflect the model and not the sampling.

Three coarse-graining knobs act on the spectrum, and none is applied in the spine:

- **temperature $\beta$** sets which sites are "the same" (Section 10 for how to set it);
- **diffusion time $t$**: $S^t$ has eigenvalues $\lambda_i^t$; large $t$ damps fine modes smoothly. Diffusion distance at time $t$, $\|S^t(\delta_x - \delta_y)\|$, is the metric induced by the coarse-graining;
- **truncation $r$**: keep the top $r$ eigenfunctions; error is the tail sum of Section 1.3.

Diffusion time and truncation disagree when there is no spectral gap; that is the correct behaviour (there is then no canonical number of clusters, and the smooth knob should be preferred). For comparison over training we hold $t$ or $\beta$ fixed and let $r$ be reported, never the reverse.

**Degeneracy.** The SVD gives canonical nested *subspaces* and their spectra. Within a near-degenerate block the basis is not canonical. We report subspaces and gaps; where a basis is wanted for display (a "feature list") we rotate within the block by a declared rule (varimax, or the localizing rotation in Section 4.3) **[P2]**. Davis–Kahan makes this precise: a subspace is stable under perturbations small relative to its gap, and nothing within a gapless block should be trusted individually. This is the perturbation-theoretic sense of "really close" that motivates the insistence on SVDs.

**Hierarchy.** Dyadic powers $S, S^2, S^4, \dots$ define, via the Coifman–Maggioni diffusion-wavelet construction, an orthogonal multiscale basis of $\mathcal H$ by QR factorizations alone: coarse scaling functions that are indicator-like on large clusters, finer wavelets inside them. This is the construction that plays the role of a hierarchical dictionary without any training or sparsity assumption. It is heavier than a single eigendecomposition and we treat it as an extension of the spine rather than its default.

## 4. Coherence across layers: the dynamic operator on sites

### 4.1 Construction

Let $S^L, S^{L'}$ be the geometric operators at two layers. Because transport between layers is the identity on $X$ ([A1]), the Froyland coherent-set operator is simply the product

$$Q^{L \to L'} = S^{L'} S^{L}, \qquad Q = \sum_i \sigma_i\, u_i\, v_i^\top,$$

with $v_i \in \mathcal H$ a function on sites "seen at layer $L$" and $u_i$ the matching function "seen at layer $L'$". Interpretation of a singular pair:

- $\sigma_i \approx 1$: the set of sites where $v_i$ is large is tight at $L$ and still tight at $L'$: **a coherent set, something the network keeps together across those layers**;
- $v_i$ concentrated on one region at $L$ while the corresponding $u_i$ splits it: **the network distinguishes, between $L$ and $L'$, two things it had not distinguished**;
- the converse is a **merge**.

The small-time version is the dynamic Laplacian $\tfrac12 (S^L + S^{L'})$, and over a set of layers $\{L_k\}$ the two choices are the product $\prod_k S^{L_k}$ (sequential coherence) and the mean $\tfrac1m \sum_k S^{L_k}$ (simultaneous coherence). The product is the one that composes and is our default **[M3]**.

### 4.2 Why this answers the "sparse concepts are treated independently" worry

If activations were sparse sums of concept directions, a strict partition into coherent sets would be wrong: a site with two active concepts belongs to both. But the SVD of $Q$ does not partition. Its singular functions superpose, and a site's coordinates $(v_i(x))_i$ record *how much* it participates in each coherent mode. Partitions are level sets of singular functions and are an interpretive afterthought. The basis is primitive; the cut is derived.

### 4.3 Localization is measured, not assumed

Whether a mode is "a feature" (localized on a cluster of sites) or "a direction of variation" (spread over the cloud) is read off its inverse participation ratio, $\mathrm{IPR}(\psi) = \sum_x \psi(x)^4 / (\sum_x \psi(x)^2)^2$. High IPR at the working $\beta$ means the dictionary picture is accurate for that mode; low IPR means it is not, and the method says so instead of forcing a sparse code. When a dictionary is wanted for display, the localizing rotation of a near-degenerate top-$r$ block is a declared [P2] choice.

### 4.4 Three time axes, one machinery

Nothing above used that $L$ indexes depth. The same product-of-operators construction applies with the "time" index being

- **depth** (layers), as above: what the network keeps together as it computes;
- **sequence position**, via the context view: what the stream keeps together as context accumulates (Section 6);
- **training step**, via checkpoints: what the network keeps together as it learns, and the dynamic-mode / HAVOK decomposition of training.

The HAVOK and "neurons as coherent-structure detectors" picture is the case where the coherent modes are computed on a time-lagged view and a neuron's activation is expanded in them (Section 7.1).

## 5. Loadings: getting back to activation space

Given any mode $\psi \in \mathcal H$ and any vector-valued function on sites $F \in \mathbb R^{N \times p}$ (activations at some layer, a unit's output, a gradient component), the loading is

$$a_F(\psi) = F^\top \psi \in \mathbb R^p .$$

It is the covariance of $F$ with $\psi$, the direction in $F$'s space along which the mode varies. Properties:

- for a cluster indicator $\psi$, $a_{H^L}(\psi)$ is the cluster centroid at layer $L$: the "replace with the centroid" re-embedding, obtained canonically rather than posited;
- for the linear kernel, $a_{H^L}(\psi_i) = \sigma_i v_i$: PCA directions;
- for a mode supported on a curved cluster, several modes carry it and their loadings span its tangent directions; the loss in representing $\psi$ by a single direction is measured by the $R^2$ of regressing $\psi$ on $H^L$ (linear readability of the mode) and is reported, not hidden;
- the same mode has a loading at every layer: $a_{H^L}(v_i)$ is where the coherent set sits at $L$, $a_{H^{L'}}(u_i)$ where it sits at $L'$. The pair is the rank-one piece of the layer's effective action on that mode.

Writing $H^L[x] = \sum_i \psi_i(x)\, a_i$ (Section 1.3) says: the activation at site $x$ is a combination of the directions $a_i$ with coefficients $\psi_i(x)$. This is the form of a dictionary decomposition, with dictionary $\{a_i\}$ and codes $\psi_i(x)$. In the high-$\beta$, high-IPR regime the codes are near-indicators and the dictionary is an SAE-like set of concept directions. In the low-$\beta$ regime the codes are PCA scores. The regime is chosen by $\beta$ and reported by IPR, never assumed. Out-of-sample sites are handled by the Nyström extension of $\psi_i$ (kernel regression of $\psi_i$ on the view), which is the same formula as a kernel feature **[M4]**.

## 6. Context: the KV-cache problem, dissolved and then re-posed honestly

### 6.1 At the level of the spine there is no problem

Because $x$ at layer $L'$ is the site with its context, $S^{L'}$ already reflects whatever attention or recurrence delivered to $x$. The dynamic operator $Q^{L \to L'}$ needs no model of the layer map. Coherent sets, splits, and merges are well-defined and architecture-free. So is everything in Sections 5, 7.1, 7.2 and 8. This is why the construction works equally for a transformer, an SSM, a CNN, and a recording.

### 6.2 Where it returns: attributing the change at a site to its causes

To say *what the layer did* to produce $h^{L+1}(x)$ from what it had, we need the layer's inputs at $x$. For an MLP the input is $h^L(x)$ alone and the map is a function. For attention or an SSM the input is $h^L(x)$ together with the past of the stream. Marginalizing the past (treating the layer as a stochastic map on $h^L(x)$) is a lower bound on what the layer does and we do not adopt it. Fitting a model of the hidden state is training something on top of the network and we do not adopt that either.

What we adopt is the construction the low-rank-logits / Hankel literature and subspace identification both use: **the predictive state of the stream, as an SVD.**

### 6.3 The context view and the layer-$L$ Hankel operator

For a site $x = (\text{doc}, t)$ define past and future windows of the layer-$L$ stream,

$$\pi_w(x) = \big(\hat h^L(x_{-1}), \dots, \hat h^L(x_{-w})\big), \qquad \varphi_{w'}(x) = \big(\hat h^L(x_{+1}), \dots, \hat h^L(x_{+w'})\big),$$

with kernels $K_{\text{past}}[x, y] = \exp\big(\sum_{j=1}^w \beta_j \langle \hat h^L(x_{-j}), \hat h^L(y_{-j})\rangle\big)$ and the same for the future, with lag weights $\beta_j$ (geometric decay, [P3]). A product of sphere kernels is a sphere kernel on the weighted concatenation, so [A2] is the only assumption in play.

The *Hankel operator* of the layer-$L$ stream is the conditional-expectation operator from past-features to future-features. Its regularized sample form is the kernel-CCA / Hilbert-space-embedded predictive-state-representation estimate

$$\mathcal T^L = (K_{\text{fut}} + \rho I)^{-1/2} K_{\text{fut}}\, K_{\text{past}}\, (K_{\text{past}} + \rho I)^{-1/2}, \qquad \mathcal T^L = \sum_i \tau_i\, \eta_i\, \xi_i^\top ,$$

with $\rho$ a ridge **[P4]**. The right singular functions $\xi_i \in \mathcal H$ are the **predictive-state coordinates** $s^L_i(x)$: functions on sites that summarize what the past of the stream at $x$ says about its future. Their number with $\tau_i$ above the noise floor is the **effective state dimension** $r_{\text{ctx}}(L)$ of the layer-$L$ stream.

This is Ho–Kalman / subspace identification with a kernel, and it is the same construction that gives the belief-state simplex for an HMM (the Hankel matrix of the process has rank equal to the number of hidden states) and the "low-rank logits" bound for an autoregressive LM (the Hankel of $p(\text{future} \mid \text{past})$ has rank at most the width of the unembedding). We apply it to **every layer's stream treated as its own process**, which is exactly the "Hankel$^L(\text{past}, \text{fut})$ computed from $h^L$ alone" the user asked for. For an SSM, $r_{\text{ctx}}(L)$ is bounded by the state dimension; for a transformer it is measured; both are compared on equal terms.

### 6.4 How the context acts on the current site: the type signature

The layer's action at $x$ is approximated as a function of the pair (current activation, predictive state):

$$\Delta h^{L+1}(x) \approx g\big(h^L(x),\, s^L(x)\big).$$

In site space this is one more kernel, the product $K_{\text{cur} \otimes \text{ctx}}[x, y] = K^L[x, y] \cdot K_s[x, y]$ where $K_s$ is the sphere kernel on $(s_1, \dots, s_r)(x)$ with the natural scale given by the $\tau_i$ (the Hankel singular values are already a metric on the state space, so there is no extra temperature to choose). Its operator's eigenfunctions are **joint current-and-context modes**; its loadings on $H^{L+1}$ are the directions the layer writes for each joint mode.

The amount of context the layer actually reads is then a measured curve, the **explained-variance ladder**:

$$V_0 = \mathrm{Var}\,\Delta h^{L+1} \;\ge\; V_{\text{cur}} = \mathrm{Var}\big(\Delta h^{L+1} \mid h^L(x)\big) \;\ge\; V_{\text{cur}, r} = \mathrm{Var}\big(\Delta h^{L+1} \mid h^L(x), s^L_{1..r}(x)\big) \;\xrightarrow[r \to \infty]{}\; 0,$$

each term a kernel ridge regression residual (an SVD). $V_{\text{cur}}$ is the "marginalize context" lower bound, now a number rather than an assumption. The $r$ at which the ladder saturates is how much state the layer reads; the gap between $V_{\text{cur}}$ and $V_0$ is how much of the layer is "an MLP in disguise".

Two Hankels are available and their relation is itself informative **[P5]**: the one above (what the stream *carries*, intrinsic to layer $L$) and the cross-layer one with future replaced by $h^{L+1}(x)$ (what the layer *reads*). The first is the architecture-free, recording-compatible object; the second is the one to use for attribution to the layer's computation. We compute both and report the principal angles between their state subspaces.

## 7. Pulling back to the computation

### 7.1 Every unit is a function on sites: neurons as coherent-structure detectors

A neuron's activation $n \in \mathcal H$, a head's output $o \in \mathbb R^{N \times d}$, an SSM state coordinate: each is a (vector-valued) function on sites and has an exact expansion $n = \sum_i \langle n, \psi_i\rangle \psi_i$ in any coherent basis. The **mode profile** of a unit is the vector $(\langle n, \psi_i\rangle)_i$. A unit concentrated on one mode detects one coherent structure; one spread across several is polysemantic *with respect to this basis*; one spread over the small-$\sigma$ tail is responding to what the basis calls noise. All of this is stated relative to the basis and the scale $\beta$, which is the correct amount of relativity. "Virtual neurons" (a computation carried by a combination of real neurons none of which carries it alone) appear as a mode whose loading on the neuron layer is spread across several neurons; this is the object Section 11's compute-in-superposition test looks for.

### 7.2 The exact mode decomposition of a gradient step

Let $W$ be any weight matrix in the network with per-site input $u(x) \in \mathbb R^p$ and per-site backpropagated error $\delta(x) \in \mathbb R^q$ at its output. The gradient of the training loss over the sites is

$$G = \nabla_W \mathcal L = \sum_x \mu(x)\, \delta(x)\, u(x)^\top = \Delta^\top M U,$$

with $\Delta \in \mathbb R^{N \times q}$, $U \in \mathbb R^{N \times p}$ functions on sites and $M = \mathrm{diag}(\mu)$. Inserting $\Psi \Psi^\top = I$:

$$G = \sum_i \underbrace{(\Delta^\top M^{1/2} \psi_i)}_{g_i \in \mathbb R^q}\, \underbrace{(U^\top M^{1/2} \psi_i)^\top}_{b_i^\top,\; b_i \in \mathbb R^p} = \sum_i g_i\, b_i^\top .$$

The gradient is **exactly** a sum over coherent modes of rank-one updates: mode $i$ pushes $W$ to map its input loading $b_i$ toward its error loading $g_i$. This holds for every weight matrix in every architecture (for a pairwise weight such as $W_Q^\top W_K$ the input is a function on $X \times X$ and the basis is $\psi_i \otimes \psi_j$). No approximation, no extra assumption: only the training loss and the training sites the network itself sees. From it:

- **pressure on mode $i$**: $\|g_i\| \|b_i\|$, the size of the rank-one piece;
- **interference of mode $i$ on mode $j$**: the update $g_i b_i^\top$ acts on $j$'s inputs by $g_i \langle b_i, b_j\rangle$, so $I_{ij} = \langle b_i, b_j\rangle\, \|g_i\|$; mode $i$'s learning moves mode $j$'s representation when their input loadings overlap;
- **refinement**: a coherent set at $L$ whose two halves (as seen at $L'$) have opposite error loadings is under pressure to split; the basis of $Q^{L \to L'}$ exhibits this as a mode with $\sigma < 1$ and large pressure;
- **noise versus error-correction**: a small-$\sigma$ mode with small pressure is noise at this scale; a small-$\sigma$ mode with consistent nonzero pressure across checkpoints is being actively maintained. The method reports both numbers and does not decide ex ante which it is.

Truncating the basis projects $G$ onto the top modes; the part of the update left out is exactly the tail of the decomposition. This is the sense in which "the pullback to the weights only constrains the update to the activation cloud": the activation-space description (modes and their loadings) is primary, and the weights appear only through how their update decomposes over that description.

### 7.3 The action view: coherence in tangent space

The forward-pass analogue of 7.2 is the local Jacobian $J^L(x) = \partial h^{L+1}(x) / \partial h^L(x)$ (the same-site block; cross-site blocks exist for attention and are what the context view summarizes). The action kernel $K_J[x, y] = \langle J(x), J(y)\rangle_F / (\|J(x)\|_F \|J(y)\|_F)$ says two sites are treated alike when the layer applies the same local linear map to them. For a ReLU MLP, $J = W_2 \,\mathrm{diag}(m(x))\, W_1$ with $m(x)$ the active mask, so $K_J$ is a weighted mask-overlap kernel: the activation-plateau / affine-cell structure, and the per-site affine map of the ISAN picture, each as a kernel on sites. Coherent sets of $K_J$ are sets on which the layer is one affine map. The Jacobian is never formed: $m_J$ random probes $v_r$ give $\langle J(x), J(y)\rangle_F \approx \tfrac1{m_J}\sum_r \langle J(x) v_r, J(y) v_r\rangle$ from $m_J$ Jacobian-vector products per site **[M5]**. Vector diffusion maps (Singer–Wu) are the extension that uses $J$ as a connection between layers rather than only its Gram matrix; we note it and do not use it in the spine.

Weight magnitudes are deliberately absent. Comparing a weight to zero is comparison to zero-ablation; the Jacobian and the gradient compare the network to itself.

## 8. Comparing models and predicting stitching

Two models $A$, $B$ run on the same sites give operators $S^A_L, S^B_{L'}$ on the same $\mathcal H$. Comparison is therefore operator comparison with no Procrustes step:

- **subspace distance**: principal angles between the top-$r$ eigenspaces, stable by Davis–Kahan relative to the gap;
- **alignment**: $\langle S^A_L, S^B_{L'}\rangle_F / (\|S^A_L\|_F \|S^B_{L'}\|_F)$, which at $\beta \to 0$ is linear CKA. CKA's known fragilities (sensitivity to dominant directions, to outliers) are the $\beta \to 0$ pathologies; at finite $\beta$ they are tempered, and the spectrum tells us by how much;
- **stitching prediction**: a linear stitch from $A$'s layer $L$ into $B$'s layer $L'$ succeeds to the extent that $B$'s modes that carry pressure (7.2) are linearly readable from $H^A_L$. Per-mode readability is the ridge $R^2$ of $\psi^B_i$ on $H^A_L$; the predicted stitching loss is the pressure-weighted unreadability. No stitch is trained to make the prediction; the ridge regression is a closed-form SVD.

Across training, the same holds with $A$ and $B$ two checkpoints, and across layers of one model with $A = B$. This is the single comparison primitive the theory needs.

## 9. Metrics other than cosine (variant, not spine)

The Fisher information of the output pulls back through the downstream network to a metric $g_L(x) = J_{L \to \text{out}}(x)^\top F_{\text{out}} J_{L \to \text{out}}(x)$ on the layer-$L$ stream at each site, and "two activations are the same if the output cannot tell them apart" is the kernel built from it. It is the mega-principled version and it is per-site, $d_L \times d_L$, expensive, and anisotropic; the intermediate version uses the site-averaged metric. Both are computable in sketched form. We treat the choice (cosine; averaged Fisher pullback; per-site Fisher pullback) as methodological [M6], decided by experiment, since the output view already carries the Fisher–Rao information at the end of the network and the dynamic operator $Q^{L \to \text{out}}$ already couples layer-$L$ geometry to it.

## 10. Temperature

$\beta$ is the one knob the theory cannot take from the data without a further choice. Options, in decreasing order of how principled we judge them **[P6]**:

1. **Noise-induced scale.** Two activations closer than the jitter the network itself experiences cannot be reliably distinguished by it. Jitter is measurable: the displacement of $h^L(x)$ between adjacent late checkpoints (SGD noise), or across dropout / batch-composition resamples. $\beta_{\text{noise}} = 1 / \overline{\|\delta \hat h\|^2}$. Edge-of-stability and batch-noise results suggest this scale is set by the optimizer and is roughly layer-wise constant late in training; this is exactly the kind of literature anchor that should be checked rather than assumed.
2. **Self-tuning scale**: $\beta$ from the median $k$-nearest-neighbour distance (Zelnik-Manor–Perona), the standard diffusion-maps default; data-driven but not network-driven.
3. **Scan and persistence**: compute spectra along $\beta$ and report which modes persist (a persistence diagram over scale). This is the coarse-graining story and is useful for display; we do not treat functions of $\beta$ as primitives.

Default: report at $\beta_{\text{noise}}$ when checkpoints exist (every training-dynamics experiment; Pythia), $\beta_{\text{nn}}$ otherwise, with a scan as supporting material. The $\beta \to 0$ (CKA/PCA) limit is always also reported since it is the comparison point with the linear-features literature.

## 11. What this predicts in cases where the answer is known

Details are in `EXPERIMENTS.md`; the predictions that would falsify the framing are:

- **Toy models of superposition** (sparse features, narrow hidden layer): at $\beta \gtrsim \beta_{\text{nn}}$ the top-$n$ modes localize (high IPR) on single-feature-active sites and their hidden-layer loadings align with the columns of $W$; the dynamic operator hidden $\to$ output sharpens them (the ReLU makes the output geometry more clustered than the hidden one). As feature sparsity decreases, IPR falls and the method reports delocalization rather than a wrong dictionary.
- **Computation in superposition** (sparse $x \mapsto \mathrm{ReLU}(x)$ through a narrow layer): modes correspond to input features; the mode profile of each real neuron spans several modes and each mode's loading on the neuron layer spans several neurons (virtual neurons); the interference matrix $I_{ij}$ of Section 7.2 matches the feature-interference pattern.
- **Modular addition / grokking**: the embedding-layer geometry at the Fourier scale yields eigenfunctions that are the circle harmonics; $Q^{\text{embed} \to \text{post-MLP}}$ groups them by frequency; across checkpoints a spectral gap opens at the frequencies the network uses, and the mode-wise pressure of Section 7.2 on those frequencies rises before the test loss drops.
- **HMM belief states** (Mess3 and relatives): $r_{\text{ctx}}(L)$ at the layer where the belief simplex is represented equals the number of hidden states, and the predictive-state coordinates $s^L$ align with the true belief states computable from the HMM; this holds for a transformer and an SSM alike, which is the architecture-agnosticism test.
- **MNIST / dSprites / CIFAR**: coherent sets across layers coincide with classes at coarse scale and with within-class structure at fine scale; on dSprites the product structure of the factors appears as a product structure of eigenfunctions; predicted stitching fidelity between seeds correlates with actual stitching.

A failure on a large model is informative only if the language localizes it: a layer with no gap, a basis that does not localize, an explained-variance ladder that never saturates. Those are the diagnostic statements the method is built to make.

## 12. Assumptions register

Substantive:
- **[A1]** The same site at two layers is the same site (identity transport on $X$).
- **[A2]** Dot product on each view after normalization (cosine on the residual stream; Bhattacharyya on outputs; weighted concatenation on windows).
- **[A3]** The only "task" information used is the network's own training objective evaluated on the network's own training sites (Section 7.2).

Pro-tem:
- **[P1]** Dropping the norm (optionally restored as $\log\|h\|$).
- **[P2]** Basis within near-degenerate blocks by a declared rotation; subspaces are the canonical object.
- **[P3]** Lag weights in the context view.
- **[P4]** Ridge $\rho$ in the Hankel estimate.
- **[P5]** Two Hankels (carried vs read); both computed.
- **[P6]** Temperature from training noise, else from nearest-neighbour scale.

Methodological:
- **[M1]** Fisher only at the output by default.
- **[M2]** Density exponent $\alpha = 0$.
- **[M3]** Product rather than mean over layers.
- **[M4]** Nyström extension for out-of-sample sites.
- **[M5]** Sketched Jacobians.
- **[M6]** Cosine vs Fisher-pullback metrics at intermediate layers.

Known limitations:
- Eigenfunctions delocalize when many features are simultaneously active; the IPR reports this but the dictionary reading is then unavailable.
- Long-range context enters only through the stream itself at past positions (windows of length $w$); this is right for what the stream carries and incomplete for what attention can read far back.
- Everything is first-order in the weights (Jacobians, gradients); ablation-style counterfactuals are outside the spine by design.

## 13. Computational budget and the knob-to-accuracy map

| object | dense cost | scalable form | error controlled by |
|---|---|---|---|
| kernel $K$ on $N$ sites | $O(N^2 d)$ | $k$-NN sparse graph, $O(N k d)$ via approximate NN | $k \gg \log N$ preserves the top spectrum |
| eigenpairs of $S$ | $O(N^3)$ | Lanczos, $O(r N k)$ per sweep | tail sum of loadings (exact) |
| $Q^{L \to L'}$ | product densifies | matvecs only, never formed | same |
| Hankel $\mathcal T^L$ | $O(N^3)$ | Nyström with $m$ landmarks, $O(N m^2)$ | kernel spectral tail beyond $m$, ridge $\rho$ |
| action kernel | $O(N^2 d^2)$ | $m_J$ JVPs per site then $k$-NN | $O(1/\sqrt{m_J})$ on inner products |
| gradient decomposition | one backward pass over sites | exact; truncate to $r$ modes | tail of the rank-one sum |

Statistical accuracy: graph-Laplacian eigenpairs converge with bias $O(1/\beta)$ and variance $O(N^{-1/2} \beta^{d_{\text{int}}/4 + 1/2})$ in the intrinsic dimension $d_{\text{int}}$ of the view; Nyström error decays with the kernel's spectral tail beyond $m$. Dense is fine to $N \approx 3 \times 10^4$ on a laptop; sparse graphs reach $N \approx 10^6$ tokens. Every knob ($N, \beta, k, m, r, t, w, \rho, m_J$) has a monotone, visible effect on either resolution or error, and the synthetic experiments in `EXPERIMENTS.md` chart them before any real model is touched.

## 14. Relation to existing tools

- **PCA / CKA / CCA across layers**: the $\beta \to 0$ limit of the geometric view and of Section 8.
- **SAEs and hierarchical dictionaries**: the high-$\beta$, high-IPR regime of Section 5, with the hierarchy given by diffusion wavelets (Section 3); no sparsity penalty, no training.
- **Froyland's coherent sets and dynamic Laplacian**: Section 4 with identity transport.
- **Kernel EDMD / kernel transfer operators (Klus et al.), HAVOK**: Section 4.4 and 6, with the operator estimated on sites and the time axis being depth, position, or training.
- **Subspace identification, predictive-state representations, low-rank-logits Hankel**: Section 6.3 at every layer.
- **ISAN and activation plateaus**: Section 7.3, per-site affine maps as a kernel.
- **Vector diffusion maps, Hodge Laplacians**: extensions that use the Jacobians as connections; not in the spine.
- **Model stitching**: Section 8, predicted rather than fit.

## 15. Open decisions for the user

1. The identity-transport move [A1] is the load-bearing design decision; everything else follows. Does the "site is the atom" framing match the intent, including for recordings?
2. The two Hankels of Section 6.4 (carried vs read): keep both as first-class, or make "carried" the spine and "read" an attribution tool?
3. Noise-induced temperature (Section 10, option 1) requires checkpoints or resampling; is that an acceptable requirement for the headline number?
4. The gradient decomposition (7.2) uses the actual training loss and sites. For a pretrained model without its training corpus we would substitute a held-out corpus from the same distribution; is that acceptable [A3]?
5. Density exponent [M2]: density-weighted (frequent things matter more) vs manifold-intrinsic.
