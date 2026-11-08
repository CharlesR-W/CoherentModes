# The two identified references: constructions, limits, and proposed document edits

Review date: 2026-11-07. Compared against `THEORY.md` v1.4 and `EXPERIMENTS.md` v1.3. **This review does not modify either document.** Equation and section references below refer to the papers unless prefixed “our”. Statements marked **deduction** are consequences worked out here, not reported experimental findings.

## Sources and reading record

Both PDFs were downloaded into separate, newly created empty directories and treated as untrusted data. No embedded instructions, attachments, or paper code were executed. `pdftotext` was unavailable; text was extracted with isolated Python (`python -I`) from `/tmp`, outside both download directories, using an independently installed PDF library. All pages were read, including references and supplements: 25 pages for paper 1 and 8 pages for paper 2. A second extraction was used after the initial extractor warned about mathematical fonts.

- Paper 1: <https://arxiv.org/pdf/2510.26955v1>; local PDF `/tmp/papers/coherent-rx3xMj/paper.pdf`, extracted text `/tmp/papers/coherent-rx3xMj/paper.txt`; SHA-256 `76e2dc02b49f7bf66cb8a1a6a4175e503b91a8f66fd282a5b3c16c5fbca5808e`.
- Paper 2: <https://arxiv.org/pdf/2607.04800>; downloaded PDF identifies itself as **2607.04800v1**; local PDF `/tmp/papers/virtual-dWFK9R/paper.pdf`, extracted text `/tmp/papers/virtual-dWFK9R/paper.txt`; SHA-256 `e53df8babf51d1f20772cbdf3d6f193218d8c11a3f0f6d1b245f8d8a73cf6319`.

Two identification corrections matter throughout:

1. Paper 1 says **coherent sets**, not merely unspecified “coherent structures”. It uses delay coordinates and past–future CCA. It does **not** present itself as HAVOK and does not use the term HAVOK or Hankel in its text. The matrix below is a Hankel/trajectory matrix by construction; that terminology is our bridge, not a quoted paper definition.
2. Paper 2 does **not** use the phrase “virtual neuron”. Its identified mechanism is a sparse binary code over real neurons, decoded by a near-pseudoinverse. “Virtual neuron” can be our explicitly defined interpretation of that mechanism, not an attributed paper term or a theorem of exact functional decomposition.

## 1. Neurons as coherent-set detectors

### Full citation

Joshua L. Pughe-Sanford, Xuehao Ding, Jason J. Moore, Anirvan M. Sengupta, Charles Epstein, Philip Greengard, and Dmitri B. Chklovskii. **“Neurons as Detectors of Coherent Sets in Sensory Dynamics.”** 2025. arXiv:**2510.26955v1**, 30 October 2025. PDF carries the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) designation. Pughe-Sanford, Ding, and Moore are marked equal contributors.

### One-paragraph summary

The authors propose that sensory neurons learn membership indices for finite-time coherent sets of an input-generating stochastic dynamical system: regions whose trajectories remain together over a chosen horizon. A nontrivial singular-function pair of the stochastic Koopman operator gives a relaxed coherent-set pair, with one function predicting a common future and the other retrospecting a common past; thresholding or rectifying those functions yields neural responses. For stable multivariate Ornstein–Uhlenbeck (OU) dynamics the leading nonconstant functions are linear projections in whitened state coordinates, and data-driven Galerkin estimation reduces to past–future CCA, including when the observations are scalar signals represented by lag vectors. The authors connect the resulting filters to sustained/transient responses, ON/OFF rectification, and predictive/retrospective physiological cell classes. Their extension to unstable/saddle dynamics is explicitly less settled: the formal invariant density can be nonnormalizable, an appropriate localized reference measure is unresolved, and equivalence between ordinary data-driven CCA and the unstable analytic construction remains open (Discussion items i–iii). This is a model of biological sensory computation, not evidence that arbitrary trained artificial-network units optimize this objective.

### 1.1 The object a neuron is claimed to detect

Use `Z_t` for the dynamical state, reserving our `X` for the finite site index set. With horizon `τ`, define

\[
(\mathcal K_\tau f)(z)=\mathbb E[f(Z_{t+\tau})\mid Z_t=z].
\]

This maps future observables in `L²(ν)` to present observables in `L²(μ)`, where `ν` is the pushforward of `μ` under the actual stochastic dynamics. The weighted adjoint is the reverse-time conditional-expectation operator, **not** an unweighted matrix transpose unless the measure coordinates have been transformed appropriately (main Eqs. 1–8; Supplement §1).

A balanced coherent pair `(A,B)` has `μ(A)=ν(B)` and high `P(Z_{t+τ}∈B | Z_t∈A)`, with the complementary pair included in the optimization (Eq. 4). Relax the indicator optimization and solve

\[
\mathcal K_\tau u_i=\lambda_i v_i,\qquad
\mathcal K_\tau^\dagger v_i=\lambda_i u_i;
\quad
\mathcal F_\tau=\mathcal K_\tau\mathcal K_\tau^\dagger,
\quad
\mathcal B_\tau=\mathcal K_\tau^\dagger\mathcal K_\tau.
\]

Thus `v_i` is an eigenfunction of `F_τ`, `u_i` of `B_τ`, with eigenvalue `λ_i²`. The constant pair is trivial. The sign regions `{±v_i>0}` and `{±u_i>0}` approximate present/future coherent pairs (Eq. 9); the spectral relaxation does not guarantee that every zero threshold solves the exact balanced discrete-set optimization. The neuron computes `H(±v_i(z))` or `H(±u_i(z))` (Eq. 27), with ReLU-like graded rectification discussed as soft membership (§5.2). **The function before rectification and the neural response after rectification are distinct objects.**

Orientation warning: the paper calls `v_i` the present/predictive function and `u_i` the future/retrospective function. For the Koopman operator as written above, `v_i` is its **left** function and `u_i` its **right** function. For a forward state-transition matrix these vector roles reverse. Our `Q=P^{L'}P^L` instead maps source columns forward, with `v` the right/source and `u` the left/destination vector. Preserve time roles rather than transplanting “left” and “right” without the operator convention.

### 1.2 Exactly how the time-lagged matrix and CCA are built

Main Eq. 20 first introduces an `n`-lag state reconstruction from a scalar observation `y(t)=Cz(t)`. In §4 the general neuron-input version uses

\[
\ell_t=(I(t),I(t-\Delta t),\ldots,I(t-(d-1)\Delta t))^\top.
\]

The accessible Galerkin features are `ϕ_i(ℓ)=ℓ_i` in this example, but can be arbitrary upstream measurement functions in the general construction (Eqs. 22–26). To express it in our sites-as-rows convention, choose valid anchor times `t_s`, not crossing trial boundaries, and set

\[
A_p[s,k]=I(t_s-k\Delta t),\quad
A_q[s,k]=I(t_s+\tau-k\Delta t),\quad k=0,\ldots,d-1.
\]

Each is a sliding-window trajectory matrix, Hankel up to reversal of the lag index. The pairing `s↔s` supplies the temporal relation. It is **not** just two independently sampled clouds. Mean-center columns for CCA, as the dead-leaves experiment explicitly does. Its sliding windows have length **10 time steps** (Supplement §7.1). The paper does not provide a unique numerical `τ` for that experiment, so do not invent one. Past/future lag windows can overlap for short `τ`; a disjoint-window experiment is a useful additional control, not something to attribute to the paper.

For `N` pairs the linear-feature baseline is

\[
C_{pp}=A_p^\top A_p/N,\quad C_{qq}=A_q^\top A_q/N,
\quad C_{pq}=A_p^\top A_q/N;
\quad T=C_{pp}^{-1/2}C_{pq}C_{qq}^{-1/2}=U\Lambda V^\top.
\]

Use supported subspaces/pseudoinverses or declare ridge when the empirical covariances are singular. The paired canonical functions evaluated on sites are

\[
f_i=A_p C_{pp}^{-1/2}U_i,\qquad g_i=A_q C_{qq}^{-1/2}V_i.
\]

With unregularized whitening, `f_iᵀf_j/N=g_iᵀg_j/N=δ_ij` and `f_iᵀg_j/N=λ_i δ_ij`. Convert to our counting-measure convention with `ψ_{p,i}=f_i/√N`, `ψ_{q,i}=g_i/√N`. These are orthonormal within their respective supported subspaces, not automatically a complete basis of all site functions.

In the stationary, same-feature case the paper writes `Σ_0=E[ϕ_tϕ_tᵀ]`, `Σ_τ=E[ϕ_tϕ_{t+τ}ᵀ]` (uncentered second moments as written in Eq. 23), `K_τ=Σ_0^{-1}Σ_τ`, and

\[
F_\tau=\Sigma_0^{-1}\Sigma_\tau\Sigma_0^{-1}\Sigma_{-\tau},
\qquad
B_\tau=\Sigma_0^{-1}\Sigma_{-\tau}\Sigma_0^{-1}\Sigma_\tau.
\]

Their coefficient eigenvectors yield `v_i(z)=v⃗_iᵀϕ(z)`, `u_i(z)=u⃗_iᵀϕ(z)` (Eqs. 24–25). The proper inner product on coefficient vectors is covariance-weighted. An ordinary Euclidean SVD of unwhitened `K_τ`, or an SVD of the trajectory matrix alone, is **not** this CCA. Centering removes the constant contribution; a constant feature can instead be carried explicitly. Local temporal weighting/exponential forgetting is suggested for slowly nonstationary data, not proved to repair arbitrary nonstationarity.

### 1.3 OU closed form and the eigenfunction qualification

For stable `dz=Az dt+noise`, let `Σ` be the positive-definite stationary covariance, solving `AΣ+ΣAᵀ=−D`. Define the whitened forward state transition

\[
B=\Sigma^{-1/2}e^{A\tau}\Sigma^{1/2}
=\widetilde U\Lambda\widetilde V^\top.
\]

Then the predictive and retrospective linear functions are respectively

\[
v_i(z)=\widetilde V_i^\top\Sigma^{-1/2}z,
\qquad u_i(z)=\widetilde U_i^\top\Sigma^{-1/2}z.
\]

These agree with Eq. 18's covariance-metric eigenproblems, not an unwhitened eigendecomposition of `A`. Higher polynomial functions have product eigenvalues; in the stable case they do not outrank the leading nonconstant linear one (Supplement §3). Supplement §6 proves the stable OU/CCA equivalence using `C_τ=Σe^{Aᵀτ}`. For purely repulsive dynamics the text selects nontrivial eigenvalues closest to one rather than blindly the largest; mixed saddle dominance and a normalized reference measure need additional care. **Do not import its formal expanding eigenvalues into our positive, bistochastic contractions**, whose singular values cannot exceed one. Ordinary conditional expectation between proper pushed-forward probability measures is an `L²` contraction; the unstable formal density is precisely where the standard probabilistic interpretation is not already justified.

### 1.4 Mapping onto our sites → views → kernels → operators

- **Sites:** anchor observations/trial-times, with each row carrying paired lag views. This is already our shared-index move, but actual temporal pairing is extra information beyond cross-layer identity correspondence.
- **Views:** the raw lag features above are the paper's linear baseline; nonlinear upstream features give its Galerkin extension. Our normalized concatenations and exponential sphere kernels are different feature/geometry choices, not neutral re-expressions of its raw-current CCA.
- **Kernels:** `K_p=A_pA_pᵀ`, `K_q=A_qA_qᵀ` provide the dual Gram implementation of linear CCA. Replacing them with centered sphere-kernel Grams and adding ridge gives our §6.3 **extension**, not the same estimator. Whitening and Sinkhorn balancing solve different problems.
- **Operators:** our **carried** context operator is the closest analogue: lagged past versus future of the same stream, regularized KCCA/VAMP. Our **read** target changes the question to predicting the next-layer update now, which this paper does not study. The geometric product `P^{L'}P^L` compares two chosen smoothers and is not an estimate of the sensory process's Koopman operator simply because both objects admit an SVD.
- **HAVOK:** delay coordinates are shared, but HAVOK additionally models evolution of reduced delay coordinates, classically including a forcing coordinate. The paper estimates canonical dependence/transfer singular functions, not a HAVOK regression. Distinguish the raw trajectory/Hankel matrix, the lagged covariance operator, and the probability Hankel matrix used in HMM rank theorems.

### 1.5 Mapping onto our §7.1 unit profile

For our orthonormal site basis `Ψ`, every response has `n=ΨΨᵀn` and block coefficients `Ψ_Bᵀn`. In the special case where our basis contains a normalized sampled **pre-rectification** predictive function `v_i`, the linear filter output has a one-coordinate profile (up to centering and scale). But the paper's neuron is `n=H(v_i)` or `ReLU(v_i)`, not `v_i`. In general a nonlinear threshold/rectification spreads energy across the constant and many nonconstant modes; it need not concentrate in the singular-function block it was computed from. Even in an OU/Hermite basis, a threshold has higher-order components.

Therefore a broad profile is not evidence against the paper, and concentration on one of our geometric blocks is not proof of its neural learning hypothesis. A proper calibration separately records the input projection, its sign/rectified output, block reconstruction energy, and out-of-sample coherent-set retention. If thresholding introduces essential harmonics, a profile can be broad without representing several independent features. The exact expansion is generic algebra; the empirical detector interpretation requires alignment of basis, measure, timescale, and response nonlinearity.

### 1.6 What this changes about novelty claims

The 2025 paper already proposes individual neurons as thresholded/rectified Koopman singular-function detectors, predictive versus retrospective neural types, data-driven past–future CCA with lag vectors, and composition using upstream neural features. Its related-work section explicitly links this to VAMP/VAMPnets. We cannot present those ingredients or the phrase “neurons as coherent-structure detectors” as our discovery. Our possible contribution is the particular common-site framework, choice of kernels/shared measure, cross-layer smoothing products, and joint unit/gradient/action reporting. That is a narrower proposed synthesis, not an established priority claim over all relevant literature. The paper does not establish our cross-layer or gradient hypotheses, and its acknowledged unstable-regime gaps should not be promoted to our theorems.

## 2. Compressed computation with sparse binary codes

### Full citation

Francisco Ferreira da Silva and Stefan Heimersheim. **“Compressed Computation under L4 Loss is likely Computation in Superposition.”** 2026. arXiv:**2607.04800v1** (arXiv stamp 6 July 2026; PDF front matter says “Preprint. July 7, 2026”). The review uses the version returned by the supplied unversioned URL, which identifies itself as v1.

### One-paragraph summary

The authors revisit compressed computation with a bias-free single-hidden-layer ReLU network that must approximate 100 independent sparse scalar ReLUs using only 50 hidden neurons. Removing the residual connection and training with mean fourth-power error, rather than mean squared error, produces approximately uniform low per-feature error instead of exactly handling half the features and ignoring the rest. They reverse-engineer the trained encoder into sparse, near-biregular binary supports, typically 5–7 neurons per feature, with large positive on-support and small negative off-support weights, and show that a scaled encoder pseudoinverse closely describes the decoder. Codeword swaps and synthetic low-overlap codes support this account; an encoder with only two weight values plus one decoder scale recovers most, but not all, of trained performance. The resulting computation is approximate, interferes between simultaneous features, and attenuates isolated positive responses. The authors call the evidence likely computation in superposition and leave the nonuniform encoder weights and residual decoder structure partly unexplained; they do not claim an exact decomposition into 100 independent hidden nonlinear units.

### 2.1 Exact task and training protocol

To avoid clashing with our `N=number of sites`, write `F=100` features and `m=50` hidden neurons. Each coordinate is independently sampled as

\[
x_j=b_j u_j,\qquad b_j\sim\operatorname{Bernoulli}(0.02),
\quad u_j\sim\operatorname{Uniform}(-1,1).
\]

There are two **nonzero** inputs in expectation, but only one positive input in expectation. Sparsity concerns nonzero features, including negative values, not a restriction to nonnegative inputs or exactly two active coordinates. The task and model are

\[
y_j=(x_j)_+,\qquad h(x)=\operatorname{ReLU}(W_1x),\qquad
\hat y(x)=W_2h(x),
\quad W_1\in\mathbb R^{50\times100},\quad W_2\in\mathbb R^{100\times50}.
\]

There is **no encoder bias, decoder bias, residual connection, or final output ReLU**. The trained objective is

\[
\mathcal L_4=\mathbb E_x\left[\frac1{100}\sum_{j=1}^{100}
(\hat y_j(x)-(x_j)_+)^4\right],
\]

estimated by minibatch averages. `L4` means the **mean fourth power**, not the fourth root of that mean. The `p=0.02` activation probability is separate from the loss exponent. The target remains `ReLU(x)` when the exponent changes; it is not an “Lᵖ variant” of the target. Adam, learning rate `0.01`, cosine annealing, batch size `8,192`, and `100,000` steps are reported; encoder/decoder entries start uniformly in `(-0.1,0.1)`/`(-0.15,0.15)`. Exponents `2.5,3,4,6,8` yield the reported approximately uniform-error regime; exponent `2` gives the half-feature baseline (§§3–4.1). Error uniformity is assessed with **per-feature MSE conditional on `x_j>0`**, not by relabelling that diagnostic as the training L4 loss.

Appendix A embeds features into 1000 dimensions with a fixed near-orthogonal random matrix and recovers the same mechanism in effective feature coordinates. That variant is not the main 100→50→100 model. It adds near-orthogonal embedding/unembedding cross-talk, which the paper also checks by substituting pseudoinverse unembedding; it is not a second hidden layer or the cause of the L4 result. Earlier compressed-computation models with a residual and ordinary L2 are not established instances of this same mechanism.

### 2.2 The known solution: what is established, and how approximate?

Use `C∈{0,1}^{m×F}` for the feature code matrix, so `c_j=C[:,j]` is feature `j`'s support over hidden neurons. This is the transpose of the paper's `M∈{0,1}^{F×m}`; the paper has a confusing “matches the shape” phrase, but its ansatz explicitly uses `Mᵀ`.

The trained encoder has two weight populations: on-code approximately `+0.34±0.06`, off-code approximately `−0.02±0.01`. The reported support threshold is `W_1[n,j]>0.05`. Each feature uses 5–7 neurons (mean 5.5), and each neuron participates in 10–12 feature codewords (mean 10.9). A three-scalar ansatz is

\[
A(a,b)=aC+b(\mathbf1-C),\qquad
\widehat W_2=c\,A(a,b)^+,\qquad
\widehat y(x)=c\,A(a,b)^+\operatorname{ReLU}(A(a,b)x).
\]

Here `a>0`, `b<0`, and `c` is a decoder scale. `+` denotes the Moore–Penrose pseudoinverse. This is a low-dimensional description **given a code**, not a claim that a 5000-bit support is literally three scalar parameters. Code supports can be replaced by designed low-overlap codes with similar performance, which is why the particular trained support seems to carry little additional information within this family.

Reported quantitative checks (§§4.2–5, Table 1, Appendices B–C):

- The trained decoder has Frobenius cosine **0.996** with a scaled encoder pseudoinverse. Post-hoc replacement costs **1.15×** trained L4 loss; co-training with a tied scaled pseudoinverse reaches **1.05×**. A high weight cosine does not imply equal function performance.
- Fitting `(a,b,c)` on the trained support gives **1.13×** trained loss. Freeing encoder entries with a pseudoinverse decoder gives **1.05×**; a free decoder on the three-scalar encoder gives **1.06×**. Full performance remains partly unexplained.
- Equal-length codeword swaps decode to the destination feature in **100%** of the main-model tests (embedded: 99.9%). This establishes the importance of support positions in that intervention, not a general disentanglement theorem.
- Designed biregular `K=5` codes with overlap minimization reach **1.12×**; the overlap objective is `Σ_{i≠j}(c_iᵀc_j)²`. Degree-preserving edge swaps lower this objective. Without overlap minimization the designs are appreciably worse.
- Single-coordinate positive input/output slopes are approximately **0.81**, not one. The paper interprets attenuation as trading active-feature undershoot against leakage into many inactive outputs. The naive and emulate-bias baselines have roughly **28×** and **26×** the L4 loss.

These are published empirical targets to reproduce with uncertainty, not exact identities or guaranteed outcomes for all seeds, sparsities, or widths.

### 2.3 What a “virtual neuron” can mean mathematically here

**Our operational terminology, not the paper's:** feature `j` behaves approximately like a distributed effective ReLU, addressed by support `c_j` and read through decoder row `d_jᵀ=W_2[j,:]`,

\[
\widetilde n_j(x)=d_j^\top\operatorname{ReLU}(W_1x)\approx(x_j)_+.
\]

The codeword specifies the shared physical neurons recruited for that effective computation. Merely naming the output coordinate `ñ_j` is tautological; the substantive mechanism is its independently verified support, encoder two-level structure, decoder pseudoinverse relation, collision behavior, and counterfactual swaps. There is no identified extra physical unit with its own independent gate.

For an isolated **positive** input `x=s e_j`, `s>0`, positive homogeneity gives the exact identities

\[
h(s e_j)=s\,(W_1[:,j])_+,\quad
\hat y(s e_j)=s\,W_2(W_1[:,j])_+.
\]

In the ideal two-level encoder this is `h=s a c_j`, so the hidden direction is the codeword and the output is `s a W_2c_j`, approximately an attenuated `e_j` plus leakage. For `s<0`, instead `h(s e_j)=(-s)(-W_1[:,j])_+`; the negative off-code weights now activate many off-code neurons. **Do not silently treat negative isolated inputs as producing zero hidden state.**

On arbitrary inputs the exact Jacobian and featurewise linear expansion within a fixed activation cell are

\[
D(x)=\operatorname{diag}(\mathbf1_{W_1x>0}),\quad
J(x)=W_2D(x)W_1,\quad
\hat y(x)=\sum_j x_j W_2D(x)W_1[:,j].
\]

The gate `D(x)` is shared and depends on all features. Thus `h(x)≈Σ_j a c_j(x_j)_+` is at most a sparse-input approximation: overlaps, mixed signs, and off-code weights generate interference. It is not an exact decomposition into independent virtual gates. In particular `rank J≤50`, whereas the exact target Jacobian has rank equal to the number of positive coordinates. **Deduction:** exact realization on all orthants with more than 50 positive coordinates is impossible; those events have tiny but nonzero probability under the product distribution. The paper claims useful approximate computation, not an exact solution of the full distribution.

### 2.4 Which current Stage 2 measurements recover what?

| Current measurement | What it can recover | What it cannot establish; required addition |
|---|---|---|
| 1. Input-geometry blocks versus features | On isolated positive probes, normalized inputs are one direction per feature; feature-indicator subspaces can be tested. Independent feature permutations can make the whole contrast subspace degenerate. On natural multi-active data, functions may overlap. | Isolated input separation is nearly built into the probe design and says nothing about the learned code or decoder. Natural active-feature sets are **not a partition**. Use both a natural-distribution split and separate signed isolated/pairwise probes; compare centered spans and count removed sites. |
| 2. Unit profiles and hidden loadings | On isolated positive probes, centered hidden responses are linear mixtures of feature functions. If a basis localizes those functions, coefficients show one unit serving several features and one feature loading on several units. Summing the retained reconstruction, including its mean, can approximate the isolated-response matrix and expose support directions. | A spread loading is not a codeword proof. Raw loadings can have negative entries after centering even though activations are nonnegative. Individual modes can rotate in degenerate blocks. `rank(H)≤50`, while 100 centered exclusive feature indicators on a balanced isolated-positive bank have dimension 99; they cannot all be linearly readable from `H`. A nonlinear kernel can have more than 50 modes, but this does not make its hidden loadings invertible or independent. Recover codewords from reconstructed isolated responses/encoder columns and compare support precision/recall after alignment, rather than identifying every spectral vector with a virtual neuron. |
| 3. Signed gradient interaction versus cross-feature error | A well-defined description of objective-dependent first-order gradient contributions during training; possibly an empirical correlate of interference. | It is not the codeword overlap matrix, cross-feature output leakage, or a theorem about collisions. L4 backprop weights scalar errors cubically, unlike L2; the objective must be retained. Compute `O_ij=c_iᵀc_j`, isolated leakage, and pairwise nonadditivity separately, then test correlations. Near stationarity aggregate pressure can cancel without eliminating interference. |
| 4. Action-view blocks versus active features | With the **end-to-end** Jacobian, positive isolated directions have stable masks and hence stable local maps. It can distinguish learned piecewise-linear regimes and expose collision-dependent gates. The hidden-layer Jacobian `D(x)W_1` answers a different question and should be distinguished. | The mask is not generally determined by a single active feature; mixed-sign/multi-active inputs are not a feature partition. Normalized Jacobians discard overall gain; for a bias-free network the affine offset is zero within every cell, so adding offset adds no information. Gain-augmented Jacobians help quantify scale, but even full local derivatives do not on their own identify a unique binary code/pseudoinverse mechanism. Add encoder/decoder and functional tests. |

**A major normalization blind spot (deduction).** On an isolated positive ray, both `h(s e_j)` and `ŷ(s e_j)` scale with `s`, so their sphere-normalized views erase the magnitude. A model computing `0.81(x_j)_+` can have exactly the same normalized isolated-ray geometry as one with gain one. Geometry, `Q` coherence, and normalized-Jacobian agreement therefore cannot alone recover the attenuation central to Appendix B. Retain raw amplitudes in explicit response curves, function losses and leakage matrices; do not turn missing amplitude into evidence of exact computation.

**Zero-site accounting and regression outputs.** The all-zero input occurs with mass `0.98^100≈0.133`; all inputs with no positive feature have zero *target*, with mass `0.99^100≈0.366`. Negative-only inputs can still have nonzero hidden/model output. Our shared exclusion policy can therefore condition the site distribution very differently depending on whether a target or model-output view is used. Compute the paper's training/evaluation losses and per-feature errors on the **original** distribution; keep any geometry-only filtered analysis separate and report exclusions per view. The regression output is not a probability vector, so the default `√p` output view is unavailable: declare a normalized prediction-vector view on nonzero predictions and an optional target-vector validation view, not an arbitrary softmax that changes the task.

### 2.5 Recommended Stage 2 protocol

1. Reproduce the main `F=100,m=50,p=0.02` L4 model and the otherwise matched L2 control; retain dense checkpoints and stochastic resamples. Treat the 100k-step run as a remote compute job under the project's compute policy, not a promise of a quick CPU run.
2. Evaluate full-distribution L4/MSE, conditional-positive per-feature MSE/CV, error by active-count/sign pattern, and half-feature baselines. Keep natural held-out evaluation distinct from diagnostic probe banks.
3. On signed isolated probes `s e_j`, retain hidden magnitudes and all 100 outputs across `s`; estimate positive gains, inactive-output leakage, and negative-input errors. Threshold encoder columns and reconstructed isolated-positive hidden responses to compare with the binary code. Preserve the mean when reconstructing from centered spectral coordinates.
4. Measure column/row degrees, overlap matrix, weight populations, fitted decoder scale and pseudoinverse cosine. Fit the three-scalar ansatz on a fit split, then compare held-out **function** loss after decoder replacement, with all loss ratios and uncertainties. The scalar fit is a labelled mechanism-validation baseline, not an intrinsic spectral output.
5. Perform equal-cardinality codeword transplants and degree-preserving support/overlap controls. These interventions are additional validation outside the noninterventional spine; explicitly label that scope rather than banning the paper's strongest evidence to satisfy the spine.
6. On two-feature probes with controlled signs and amplitudes, measure `ŷ(u e_i+v e_j)−ŷ(u e_i)−ŷ(v e_j)`, output error/leakage, and mask changes. Compare with `O_ij` and separately with spectral gradient interactions. Report where the additive effective-neuron approximation fails.
7. Run the current spectral measurements across `β` with block/projector reporting. Compare isolated probe geometry against natural multi-active geometry, localizing rotations against block-only recovery, and raw-response functional checks against cosine-only geometry. Count code recovery as success only when support and decoder/function checks agree; a nice input block alone is not success.

### 2.6 Novelty, contradictions, and scope

This paper already supplies the toy model and partial mechanistic solution our placeholder lacked: loss-dependent feature coverage, sparse near-biregular codewords, pseudoinverse decoding, codeword-swap evidence, overlap-controlled designs, and attenuation. Applying our measurements to rediscover those is **calibration**, not a new mechanism claim. Its Related Work also identifies earlier sparse-binary-code theory (Hänni et al., 2024) and concurrent learned Bloom-filter work (Gibson, 2026); those are pointers for later literature review, not papers fully reviewed here. The original residual compressed-computation model and L2 training are explicitly cautioned against as evidence of computation in superposition. Our current description of an “active-feature partition” conflicts with independent multi-active sampling, and the suggestion that spread loadings are *the* virtual-neuron construction is unsupported. The weaker claims already in our §7.1 and §11 are appropriately hypotheses, but should now target the concrete known mechanism, with its approximation error and normalization limits.

## 3. Concrete proposed edits (not applied)

Each item supplies a section, a current excerpt identifying the edit location, and proposed replacement text. Ellipses in an excerpt only delimit the location; the replacement is intended as actual prose, not a command to rewrite the source automatically. Mathematical notation here follows the source documents where possible.

### T1 — THEORY §0: narrow the prior-art framing

**Current excerpt:**
> Representational similarity analysis in neuroscience makes the same shared-index move; what is added here is the family of cross-view operators on that index set, their composition across layers, the exact expansion of units and gradients in the resulting bases, and the context construction.

**Proposed replacement:**
> Representational similarity analysis makes the same shared-index move. The proposal here is a synthesis of cross-view operators on that index set, their composition across layers, and unit/gradient/action reporting in their spectral bases. These ingredients are not individually claimed as new: coherent-set transfer spectra and VAMP/CCA are established, and Pughe-Sanford et al. (2025, arXiv:2510.26955v1) already propose neurons as thresholded or rectified Koopman singular-function detectors with lag-vector past–future CCA. Our carried context estimator extends that family with our declared kernel and regularization choices; the read target and common-site cross-layer reporting answer different questions. Priority for the synthesis itself requires a broader literature comparison.

### T2 — THEORY §4.4: distinguish delay CCA from HAVOK

**Current excerpt:**
> The HAVOK / dynamic-mode picture requires one more thing than the product of contemporaneous operators: a lagged pairing and a fitted linear relation between lagged coordinates.

**Proposed replacement:**
> Delay-embedded past–future CCA and HAVOK share a trajectory-matrix representation but are different estimators. Pughe-Sanford et al. (2025) use paired lag vectors to estimate whitened cross-covariances and Koopman singular functions; this is the closest precedent for our carried context construction. HAVOK/DMD additionally fits evolution of reduced lag coordinates (classical HAVOK includes a forcing coordinate). Neither estimator follows from a product of contemporaneous geometric smoothers alone. Checkpoint regression in Stage 3 is a separately labelled DMD-like fit, not the CCA construction or a full HAVOK implementation.

### T3 — THEORY §6.2–6.3: add the exact reference baseline

**Current excerpt:**
> What we adopt is the subspace-identification object: **a predictive state of the stream, obtained as a regularized canonical-correlation SVD.**

**Proposed replacement:**
> What we adopt is a fitted predictive summary obtained as a regularized canonical-correlation SVD. A direct precedent is Pughe-Sanford et al. (2025, §4), whose linear lag-vector construction uses rows A_p[s,k]=I(t_s−kΔt), A_q[s,k]=I(t_s+τ−kΔt). With centered columns, C_pp=A_pᵀA_p/N, C_qq=A_qᵀA_q/N, C_pq=A_pᵀA_q/N, the SVD of C_pp^{-1/2} C_pq C_qq^{-1/2} gives the linear CCA functions. Their functions have covariance-weighted coefficient normalization; sampled unit-variance functions are divided by √N for our counting-measure convention. Our sphere-kernel, ridge-regularized estimator is an extension, not the identical raw-feature estimator. The carried target is closest to their sensory past–future question; the read target changes the prediction target to the current layer update. Delay trajectories, lagged cross-covariances and probability Hankel matrices must be kept distinct.

### T4 — THEORY §6.3: bound what is imported from the OU analysis

**Current excerpt:**
> This rank is **scale-dependent and empirical**.

**Proposed replacement:**
> This rank is **scale-dependent and empirical**. Pughe-Sanford et al.'s stable-OU calculation provides a linear-CCA calibration case, not a bound for nonlinear activation kernels. Their unstable/saddle extension explicitly leaves the localized reference measure and equivalence with ordinary CCA open (Discussion i–iii). Our Markov contractions do not inherit formal expanding eigenvalues from a nonnormalizable invariant density.

### T5 — THEORY §7.1: replace the detector/virtual-neuron interpretation

**Current excerpt:**
> A unit concentrated on one block detects one coherent structure; one spread over several is polysemantic *relative to this basis at this scale*; ... This is the "neurons as coherent-structure detectors" picture made literal. A "virtual neuron" ... appears as a block whose loading on the neuron layer is spread across several neurons; ...

**Proposed replacement:**
> A unit's block profile measures association with this basis at this scale; concentration is not itself a mechanism, and spread can arise from rectifying a single spectral function rather than detecting several independent features. Pughe-Sanford et al. (2025) propose n=H(±v_i) or ReLU(±v_i) for a Koopman singular function v_i; the pre-rectification projection may be one mode while the response occupies many. For compressed computation, Ferreira da Silva and Heimersheim (2026, arXiv:2607.04800v1) find sparse binary feature codewords shared across hidden neurons and a near-pseudoinverse decoder. They do not use the term “virtual neuron”; we use it only for the approximate effective feature computation W_2[j,:] ReLU(W_1x)≈(x_j)_+. Distributed loadings are a candidate signature [hyp], to be validated against isolated-response codewords, decoder structure, collisions and output gains in Stage 2. They are not proof of independent hidden gates. Profiles remain block-reported; small-σ content may be noise, nonlinear response harmonics, or essential computation, to be assessed separately rather than named from its spectral rank alone.

### T6 — THEORY §7.3: add the specific action-view limitation

**Current excerpt:**
> Equal normalized Jacobians do not imply one affine map (magnitude and offset ... can differ); when affine-cell recovery is the target, magnitude and offset are appended as coordinates with declared scales [P1].

**Proposed replacement:**
> Equal normalized Jacobians do not imply equal local gain or affine map. In the bias-free compressed-computation model the cellwise affine offset is exactly zero, so offset augmentation adds nothing; normalized Jacobians can still conceal the output attenuation reported by Ferreira da Silva and Heimersheim (2026). Declare whether the action view is hidden-layer or end-to-end, retain magnitude when gain is a target [P1], and measure isolated-input response curves and multi-input nonadditivity separately. An action block is a local-map similarity class, not an identified feature codeword or independent virtual gate.

### T7 — THEORY §11, computation-in-superposition bullet: state the real hypothesis

**Current excerpt:**
> **Computation in superposition**: blocks correspond to input features; neuron mode profiles span several blocks and block loadings span several neurons; the signed interaction matrix correlates with the measured cross-feature error.

**Proposed replacement:**
> **Computation in superposition** (Ferreira da Silva and Heimersheim 2026): in the no-bias, no-residual 100→50→100 ReLU model trained with mean fourth-power error on independently sparse signed inputs, spectral blocks/localizing rotations may recover feature-code structure [hyp]. Compare against the established sparse-code, near-pseudoinverse and attenuation signatures, not merely distributed loadings. On isolated-positive probes hidden responses have rank at most 50 while the 100 centered feature indicators have rank 99; nonlinear spectral feature recovery does not imply independent linear readability of every feature. Under natural multi-active sampling feature supports overlap and are not a partition. Codeword overlap and functional leakage/nonadditivity are direct mechanism measurements; correlation with signed gradient interaction remains a separate hypothesis. L2 is the reported negative control.

### T8 — THEORY §14: identify both sources and their scope

**Current excerpt:**
> **Kernel transfer operators / kernel EDMD, VAMP, HAVOK**: Sections 4.4 and 6 with the time axis being depth, position, or training; the lagged regression that HAVOK and DMD need is stated separately.

**Proposed replacement:**
> **Kernel transfer operators / kernel EDMD, VAMP, HAVOK**: shared spectral and predictive-factorization tools, not interchangeable estimators. Pughe-Sanford, Ding, Moore, Sengupta, Epstein, Greengard and Chklovskii, “Neurons as Detectors of Coherent Sets in Sensory Dynamics” (2025), arXiv:2510.26955v1, directly precedes the neural coherent-set interpretation and lag-vector CCA of our carried context view. It is not a HAVOK implementation or a proof about arbitrary layer-smoother products. HAVOK/DMD's lagged evolution fit is separate (§4.4).
>
> **Compressed computation / learned sparse codes**: Ferreira da Silva and Heimersheim, “Compressed Computation under L4 Loss is likely Computation in Superposition” (2026), arXiv:2607.04800v1, supplies Stage 2's approximate sparse-binary-code and near-pseudoinverse mechanism. Its functional tests calibrate our method; distributed loadings alone do not rediscover or prove that mechanism. See `reviews/papers.md` for constructions and limitations.

### E1 — EXPERIMENTS shared machinery item 2: handle regression correctly

**Current excerpt:**
> **Views** (§2): geometric, output ($\sqrt p$), context ...

**Proposed replacement:**
> **Views** (§2): geometric; output √p for probabilistic predictions, or a declared normalized prediction-vector view for regression; context; action; and unit activations. For Stage 2, target vectors are validation-only, not probabilistic outputs. Record raw amplitudes and functional errors separately from normalized views, and report zero exclusions per view. Full-distribution task losses are never evaluated only on the geometry-filtered sites.

### E2 — EXPERIMENTS Stage 0 context calibration: add paper 1's solvable baseline

**Current excerpt:**
> **planted, context**: i.i.d. symbols with an embedding "layer" ...

**Proposed replacement:**
> **planted, context**: retain the i.i.d. previous-token-copy and HMM checks. Add a stable multivariate OU process with known stationary covariance Σ and transition e^{Aτ}. Compare the raw-feature paired-CCA singular values/functions with the SVD of Σ^{-1/2}e^{Aτ}Σ^{1/2}, accounting for predictive/retrospective orientation and √N normalization (Pughe-Sanford et al. 2025, Supplement §6). On scalar observations, build explicit lag matrices and sweep window/horizon; compare linear CCA to our normalized sphere-kernel/ridge extension rather than asserting equality. Separately expand the linear projection and its thresholded/rectified response to calibrate profile broadening. Split by independent trajectories or separated temporal blocks; windows never cross splits. Unstable/saddle cases are exploratory and not exact calibration targets without a declared proper reference measure.

### E3 — EXPERIMENTS Stage 2 setup and targets: replace the placeholder

**Current excerpt:**
> **Placeholder stage.** The source construction ... has not yet been identified ... Provisional setup ... target $y = \mathrm{ReLU}(x)$ or an $L^p$ variant ...
>
> Targets: **planted**: the target map and the active-feature partition; **established**: the paper's virtual-neuron construction once identified; **hypothesis**: §11 bullet 2.

**Proposed replacement:**
> Reference: Ferreira da Silva and Heimersheim (2026), “Compressed Computation under L4 Loss is likely Computation in Superposition”, arXiv:2607.04800v1. Independently sample x_j=0 with probability 0.98 and Uniform(−1,1) otherwise, for 100 features. Train ŷ=W_2 ReLU(W_1x), W_1∈R^{50×100}, W_2∈R^{100×50}, with no biases, residual, or final output nonlinearity, against y=ReLU(x). Use mean fourth-power elementwise error, not an L4 root; compare an otherwise matched L2 control. Reported protocol: Adam, initial learning rate 0.01 with cosine annealing, batch 8192, 100000 steps, uniform encoder/decoder initialization ±0.1/±0.15. Save checkpoints/resamples and run remotely if the training exceeds the local compute budget.
>
> Targets: **planted**: the target map, input feature coordinates, and diagnostic isolated-feature probes. Under natural sampling active-feature sets overlap; they are not a partition. **Established empirical**: near-biregular binary feature supports, positive on-code/negative off-code weights, near-pseudoinverse decoder, codeword-swap behavior, approximate three-scalar reconstruction and attenuated positive responses. The L2 half-feature solution is a negative control. “Virtual neuron” is our approximate effective-feature terminology, not the paper's term. **Hypothesis**: recovery of this structure by our spectral measurements and correlation of spectral gradient interactions with functional interference. Evaluate all task losses on the unfiltered original distribution and log geometry-only zero exclusions separately.

### E4 — EXPERIMENTS Stage 2 measurements 1–2: make recovery identifiable

**Current excerpt:**
> 1. Input-geometry blocks vs input features, as in Stage 1.
> 2. Mode profiles of hidden neurons and block loadings on the hidden layer: the two matrices whose structure is "virtual neurons"; compare to the construction.

**Proposed replacement:**
> 1. Analyze natural held-out inputs and separate signed isolated/pairwise probe banks. Compare spectral projector spans and declared localizing rotations to centered feature indicators on the probe banks and overlapping feature functions on natural inputs; do not require a one-feature-one-isolated-eigenvector correspondence inside a degenerate block. State which views are included and which sites are excluded.
> 2. Measure unit profiles and loadings, reconstruct isolated-positive hidden responses including their means, and compare inferred code supports to W_1[n,j]>0.05 using support precision/recall and code overlap after alignment. The hidden activation matrix has rank ≤50, so 100 centered feature indicators cannot all be independently linearly readable. Add direct mechanism baselines: on/off weight populations, column/row degrees, decoder cosine with a fitted scaled pseudoinverse, held-out function loss after decoder replacement, and the fitted three-scalar encoder/decoder ansatz. Spread loadings alone count as distributed association, not code recovery.

### E5 — EXPERIMENTS Stage 2 measurements 3–4: add functional and intervention controls

**Current excerpt:**
> 3. Signed interaction matrix ... vs the measured cross-feature error ...
> 4. Action-view ... blocks vs the active-feature partition ...

**Proposed replacement:**
> 3. Retain signed block gradient interactions under each model's actual L2 or L4 objective. Separately measure binary code overlap O_ij, isolated-input output leakage, and pairwise nonadditivity ŷ(u e_i+v e_j)−ŷ(u e_i)−ŷ(v e_j), stratified by signs and amplitudes; test their relation rather than equating the matrices. Report full-distribution L4/MSE and conditional-positive per-feature MSE/CV as different statistics.
> 4. Compare hidden-layer and end-to-end action views against response masks and controlled feature probes, not an active-feature partition. Include gain-aware and cosine-only variants. Bias-free cell offsets are zero. Explicitly measure positive response slopes (paper: about 0.81), inactive-output leakage, and negative-input responses, since cosine views discard isolated-ray amplitude.
> 5. As labelled mechanism-validation interventions outside the noninterventional spine, reproduce equal-length codeword transplants, decoder replacement, and designed degree/overlap-controlled codes. Report held-out functional loss ratios and uncertainties; paper values such as 1.13× for the three-scalar trained-support ansatz are replication targets, not identities. A spectral recovery verdict must be compared with these functional/mechanistic checks, including cases where block geometry looks good but computation is wrong.

### E6 — EXPERIMENTS ordering: remove the obsolete dependency

**Current excerpt:**
> Stage 2 when its source construction is identified.

**Proposed replacement:**
> Stage 2's source is now identified (Ferreira da Silva and Heimersheim 2026); schedule it after the Stage 0–1 kernel/localization checks, with the raw-feature OU/CCA calibration added to Stage 0. Its purpose is calibration against an already partially reverse-engineered approximate mechanism, not a claim to discover that mechanism first.

## Bottom line

Both missing references are identified. Paper 1 is a direct conceptual and technical predecessor for neural coherent-set detection and carried lag-vector CCA, but not a synonym for our geometric product or a HAVOK implementation. Paper 2 supplies an exact experimental specification and a concrete, approximate sparse-code mechanism, not an exact virtual-gate theorem. The most important document changes are attribution, the pre-/post-rectification distinction, a precise L4 Stage 2 specification, overlapping rather than partitioned feature targets, and functional/codeword checks that expose what cosine spectra and distributed loadings cannot identify.
