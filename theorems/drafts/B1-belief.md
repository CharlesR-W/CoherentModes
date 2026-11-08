# Part B, cluster 1 — Ideal case: predictive states, belief states and causal states

Labels B.1–B.14 (B.14 follows B.12 to keep the Mess3 results together). Population results below concern the regularized kernel CCA of `THEORY.md` §6.3, not the cross-layer smoother product of §4. Every additional assumption is explicit. Hidden-state beliefs are ground truth for the proofs and validation; they do not enter the construction's kernel or choose its basis.

**The central positive result.** If the chosen target's conditional feature mean is affine in the belief, the past view exposes the belief, and the target observes every varying belief direction, the nonzero, zero-ridge population canonical variates are exactly an invertible affine coordinate system for the reachable belief set. Theorem B.6 gives an exact calibration: a stationary low-rank, first-order Markov token process where the actual sphere-kernel carried construction satisfies these hypotheses. Its beliefs are determined by the last token; it is not a recovery theorem for non-Markov or fractal mixed-state geometry. B.12 explains why exact affine target means fail for belief-valued Mess3 activation targets; B.14 strengthens this to infinite zero-ridge rank and noncompactness. B.13 supplies a conditional approximate-recovery bound for other regimes meeting its hypotheses, not for these Mess3 targets.

**Essential qualification.** A belief is a point in a simplex, not a categorical variable with $k$ possible values. Conditional independence through a belief therefore does **not** alone imply rank $k-1$. Nor do injective conditional means imply an *affine* recovery, or finitely many causal states imply linearly independent future laws. B.8 and B.10 give explicit counterexamples. These qualifications are necessary to make the showcase theorem true, not reasons to abandon it.

## Setup and named assumptions

**Definition B.1 (belief-sufficient layer, belief-Markov stream, and the observable boundary).** Let $(Y_t,Z_t)_{t\in\mathbb Z}$ be a stationary ergodic, finite-alphabet, edge-emitting HMM with $k$ hidden states. Write

$$T_a[i,j]=\Pr(Y_{t+1}=a,Z_{t+1}=j\mid Z_t=i),\qquad T=\sum_a T_a,$$

where $T$ is row-stochastic. The **belief** after the past through time $t$ is the column vector

$$b_t(i)=\Pr(Z_t=i\mid Y_{-\infty:t}),\qquad b_t\in\Delta^{k-1}:=\lbrace b\geq0:\mathbf1^\top b=1\rbrace.$$

Its update on an event of positive conditional probability is

$$b_{t+1}=\frac{T_{Y_{t+1}}^\top b_t}{\mathbf1^\top T_{Y_{t+1}}^\top b_t}.$$

The stationary law of $b_t$ and these updates describe the stationary/recurrent part of the **mixed-state presentation** of the chosen generator. Its support may be finite, countable, or an uncountable/fractal subset of the simplex; it need not fill the simplex. The full finite-word mixed-state presentation can additionally contain transient beliefs.

**Finite documents versus the infinite past.** Shai et al. regress activations at finite context positions onto $\eta_t=\Pr(Z_t\mid Y_{1:t})$, initialized from the stationary hidden prior, rather than onto the infinite-past $b_t$ defined here. Their validation uses supervised regression (including concatenated layers for RRXOR); those fits do not define our kernels. B.2–B.4 and B.12 apply equally to a finite-context filter under their stated accessibility/shielding/feature assumptions, with $\bar\eta$ and $\operatorname{Cov}(\eta)$ computed under the declared site law. If positions are pooled, that law is a mixture over positions and includes transient beliefs. For time-homogeneous HMM filters and future-relative targets, shielding still holds given the correct finite-context filter; arbitrary position-dependent network targets need it checked separately. The stationary example B.6 and the stationary-support argument in B.12 are not claims that every transient finite-context belief has the same distribution.

A layer is **belief-sufficient**, in the strong sense used here, if $h_t^L=\iota(b_t)$ almost surely for a measurable injection with measurable inverse on its image. It contains exactly belief information, not belief plus an arbitrary cache. It is **linear belief-sufficient** if $h_t^L=A b_t$ for a linear map injective on the affine hull of the reachable beliefs. “A linear probe can recover $b_t$ from $h_t^L$” is strictly weaker and is not this definition.

The stream is **belief-Markov** at a cut if its future and its past are conditionally independent given the belief at that cut. For a causal stream which is a deterministic function of the correct HMM filter, this follows by generating subsequent observations and filter updates from that belief. It does not say that future stream features have *linear* conditional expectations in the belief.

For an actual site $x=(\mathrm{doc},t)$ in §6.3, let $\Pi$ denote the declared past view at positions $<t$, and $\Xi$ either the carried view at positions $>t$ or the read target at $t$. We reserve $P,Q$ for `THEORY.md`’s operators; $\Pi,\Xi$ denote random views here, with past-view law $\nu_p$. We use the following separate assumptions:

- **Past accessibility (PA):** the mediator $b$ is a measurable function of $\Pi$. With a one-lag past and $h_{t-1}^L=\iota(b_{t-1})$, the available belief is $b=b_{t-1}$. Pointwise sufficiency of $h_t^L$ does not make $b_t$ accessible from positions $<t$.
- **Target shielding (TS):** $\Pi\perp \Xi\mid b$. Belief-Markov supplies this for a carried target at an aligned cut; for a read update it is an additional hypothesis, not a consequence of the token HMM.
- **Affine target mean (ATM):** for the *actual* chosen target feature map $\Phi_q:\Xi\mapsto\mathcal G_q$,
  $$\mathbb E[\Phi_q(\Xi)\mid b]=\mu_q+\mathsf L_q(b-\bar b),\qquad \bar b=\mathbb E b,$$
  for a linear map $\mathsf L_q:\mathbb R^k\to\mathcal G_q$. For an arbitrary collection of target observables, ATM means this identity for their closed linear feature span. A sufficient, generally stronger assumption is a conditional law $\mathcal L(\Xi\mid b)=\sum_i b_i\nu_i$ with fixed probability laws $\nu_i$. This is not equivalent on a restricted reachable set: at beliefs $(.4,.6)$ and $(.6,.4)$, a target taking deterministic values $0$ and $1$ has affine feature means, but a probability-mixture extension would force $(\nu_1(\lbrace1\rbrace),\nu_2(\lbrace1\rbrace))=(3,-2)$. For characteristic kernels the correct equivalence uses an affine law on the reachable set, with a possibly signed-measure extension to vertices; see B.12.
- **Belief-complete past features (BC):** the closed, centered $L^2$ span $\mathcal V_p$ of the chosen past features contains
  $$\mathcal B=\operatorname{span}\lbrace b_i-\bar b_i:1\leq i\leq k\rbrace.$$
- **Belief observability (BO):** writing $\Sigma_b=\operatorname{Cov}(b)$ and $E=\operatorname{range}\Sigma_b$, the map $\mathsf L_q$ is injective on $E$. Equivalently, $\inf_{v\in E,\|v\|=1}\|\mathsf L_qv\|>0$ when $E\neq0$.

Let $d_b=\dim E=\dim\mathcal B\leq k-1$. All feature maps have finite second moments; the sphere kernels at finite $\beta$ are bounded. Population inner products use probability measure. On $N$ sampled sites, a unit-variance score corresponds to the counting-measure vector $s/\sqrt N$; this changes no span or canonical correlation. The final object remains an SVD of sample-estimable, centered covariance operators, a **closed-form fit**, not a learned recurrent model.

**Timing and normalization are hypotheses, not bookkeeping to hide.** In the one-lag carried construction, the two included positions $t-1,t+1$ are two transitions apart. Theorems below allow this gap explicitly. If one instead wants coordinates of $b_t$, PA must be established at that cut. Sphere normalization must also preserve the belief information: assume $b\mapsto\widehat{\iota(b)}$ is injective on the support. For $h=Ab$ with $A$ full column rank, $Ab/\|Ab\|=Ab'/\|Ab'\|$ implies $b=cb'$ and hence $c=1$ by the simplex constraint, so this condition holds. It need not hold for an arbitrary injective nonlinear or affine activation map.

## What conditional independence proves — and what it does not

**Lemma B.2 (factorization through a sufficient variable).** Under TS, for square-integrable scalar features $f(\Pi),g(\Xi)$,

$$\mathbb E[f(\Pi)g(\Xi)]=\mathbb E\big[\mathbb E[f(\Pi)\mid b]\mathbb E[g(\Xi)\mid b]\big].$$

Consequently the centered cross-covariance factors through $L^2_0(\mathcal L(b))$. If $b$ takes exactly $n_b$ values with positive masses, its rank is at most $n_b-1$. A $k$-coordinate continuous belief gives no $k-1$ bound from this identity alone.

*Proof.* Conditional independence gives $\mathbb E[fg\mid b]=\mathbb E[f\mid b]\mathbb E[g\mid b]$; take expectations. For centered functions the two conditional-expectation maps take values in $L^2_0(\mathcal L(b))$, whose dimension is $n_b-1$ on $n_b$ atoms. On an infinite support this function space can be infinite-dimensional even when $b$ lies in a line. ∎

**What it buys.** This identifies the actual bottleneck: the dimension of a space of conditional expectations, not the number of entries in the state vector. It prevents importing a token-Hankel rank theorem into an activation estimator without its missing hypothesis.

**Tractability and sampling.** The factorization is a population identity and does not supply a finite-sample rank certificate. A finite past window may not determine the desired belief, and a finite future window may distinguish fewer predictive directions. Dense kernel estimation has the shared costs and document-level requirements stated below; no number of documents repairs an invalid shielding assumption.

**Theorem B.3 (finite-rank predictive bottleneck under affine target means).** Under TS and ATM, for **any** square-integrable past feature map $\Phi_p$, the population centered cross-covariance has rank at most $d_b\leq k-1$. The uncentered cross-moment has rank at most $k$. The population regularized CCA operator

$$\mathcal K_\rho=(C_{pp}+\rho I)^{-1/2}C_{pq}(C_{qq}+\rho I)^{-1/2}$$

has at most $d_b$ nonzero singular values for every $\rho>0$, not just as $\rho\downarrow0$. Every well-defined zero-ridge population limit has the same upper bound. If $\Pi$ itself is an injective view of $b$, every past score is measurable in $b$; for a larger past window, the corresponding statement for the nonzero limiting scores follows under PA and BC as in B.4, and is not asserted for arbitrary restricted past feature families or positive ridge.

*Proof.* Put $\widetilde\Phi_p=\Phi_p-\mu_p$ and $\widetilde b=b-\bar b$. B.2 and ATM give the operator identity

$$C_{pq}=\mathbb E[\widetilde\Phi_p\,\widetilde b^\top]\mathsf L_q^*.$$

For every $v\in\ker\Sigma_b$, $v^\top\widetilde b=0$ almost surely, so the first factor vanishes on $\ker\Sigma_b$. Its rank is at most $d_b$. An affine function on the simplex is linear in its $k$ barycentric coordinates: set $m_i=\mu_q+\mathsf L_q(e_i-\bar b)$, so that $\mathbb E[\Phi_q\mid b]=\sum_i b_i m_i$. Thus the uncentered cross-moment factors through $\mathbb R^k$. Left and right multiplication by the bounded invertible ridge factors preserves rank. In the zero-ridge $L^2$ formulation, conditional expectation still has range contained in a space of dimension at most $d_b$ (or its projection onto the available past function space), so no extra singular directions arise. If $\Pi=\iota(b)$, all its feature evaluations and scores are functions of $b$. The larger-window limiting assertion is proved next. ∎

**What it buys.** Under a precise, testable mechanism assumption, the construction has an exact population spectral cutoff: no more than $d_b$ independent predictive directions can survive centering. ATM is the condition that turns a low-dimensional belief *manifold* into a finite-rank predictive *operator*.

**Tractability and sampling.** For a token-future target, ATM holds for every window, while observability may require the longer windows in B.5. For carried activations, ATM must be checked for the nonlinear kernel feature map, not merely for the raw activation mean. Positive empirical singular values beyond $d_b$ are sampling noise in this regime, not algebraically forbidden; at fixed $N$, sending ridge to zero can instead interpolate and generate $N-1$ in-sample correlations near one. Use independent-document held-out ranks and the ridge/sample discussion below.

## Exact recovery of belief geometry

**Theorem B.4 (canonical coordinates are affine belief coordinates).** Assume PA, TS, ATM, BC, and BO. In the zero-ridge population $L^2$ CCA problem, exactly $d_b$ canonical correlations are nonzero, and their past canonical variates satisfy

$$\operatorname{span}\lbrace s_1^L,\ldots,s_{d_b}^L\rbrace=\mathcal B.$$

Thus, for a matrix $A_s$ invertible on $E$,

$$s^L=A_s(b-\bar b),\qquad b=\bar b+\mathsf G_s s^L\quad\text{almost surely}.$$

For $d_b=0$, this means the empty coordinate vector for an almost surely constant belief; the singular-gap argument below is needed only when $d_b>0$. The score set is an affine image of the **reachable belief set**. If $d_b=k-1$, this map extends to an affine isomorphism of the ambient simplex with its image, but the sampled scores do not thereby fill that simplex. The subspace of the $d_b$ nonzero population ridge scores converges to $\mathcal B$ as $\rho\downarrow0$ in $L^2$ subspace distance. Repeated canonical correlations identify a block, not preferred individual axes.

BC is satisfied in either of these concrete regimes:

1. **Raw-linear calibration variant, not the sphere-kernel construction:** the past features are linear in an unnormalized, linear belief-sufficient activation, or otherwise contain the centered belief coordinates in their linear span.
2. PA holds and the centered RKHS is dense in $L^2_0(\nu_p)$. In particular, a continuous universal kernel on a compact past-view space suffices. A bounded characteristic kernel also suffices for this centered $L^2$ density; the argument is included below. Injectivity must hold **after** any view normalization.

Injectivity of $b\mapsto\mathbb E[\Phi_q\mid b]$ is BO only **after ATM has been imposed**. Without ATM, injectivity guarantees neither a $k-1$ rank nor an affine recovery.

*Proof.* Let $\mathcal V_q$ be the centered $L^2$ closure of the target RKHS, and define the cross-correlation operator $\mathcal T:\mathcal V_q\to\mathcal V_p$ by the bilinear form

$$\langle f,\mathcal T g\rangle=\mathbb E[f(\Pi)g(\Xi)].$$

For target RKHS functions $g$, TS and PA imply the centered identity

$$\mathbb E[g(\Xi)\mid \Pi]-\mathbb E g(\Xi)=\langle g,\mathsf L_q(b-\bar b)\rangle.$$

The right side belongs to $\mathcal B\subseteq\mathcal V_p$. The identity extends by $L^2$ continuity to the closure. BO means that the vectors $\mathsf L_q^*g$, projected onto $E$, span $E$: otherwise a nonzero vector in their orthogonal complement would be annihilated by $\mathsf L_q$. Therefore $\operatorname{range}\mathcal T=\mathcal B$. A finite-rank operator's left singular functions with positive singular values form an orthonormal basis of its range. This proves the rank and span assertions. Two bases of $\mathcal B$ are related by an invertible linear change of coordinates on $E$, giving $A_s,\mathsf G_s$.

For completeness, the ridge limit does not require assuming that every limiting function is an RKHS element. Let $J_p:\mathcal G_p\to\mathcal V_p$ be centered feature evaluation and $\mathscr C_p=J_pJ_p^*$, and similarly on the target. The population smoother on functions is

$$S_{p,\rho}=\mathscr C_p(\mathscr C_p+\rho I)^{-1}.$$

Since $\mathcal V_p$ is the closure of the feature range, $S_{p,\rho}^{1/2}\to I$ strongly on $\mathcal V_p$; the same holds on $q$. Polar decomposition of $J_p(C_{pp}+\rho I)^{-1/2}$ shows that the nonzero singular values of $\mathcal K_\rho$ equal those of

$$S_{p,\rho}^{1/2}\mathcal T S_{q,\rho}^{1/2}.$$

The past score functions are $S_{p,\rho}^{1/2}$ applied to the latter operator's left singular functions, up to their normalization. Since $\mathcal T$ has finite rank, strong convergence of these uniformly bounded smoothers is uniform on its finite-dimensional range and adjoint range. Hence this displayed operator converges to $\mathcal T$ in operator norm. The gap between its $d_b$th positive singular value and zero gives convergence of the nonzero singular subspaces by finite-rank SVD perturbation (equivalently, apply spectral projection continuity to the self-adjoint block dilation). Applying $S_{p,\rho}^{1/2}$ does not change the limit. This proves the score-subspace assertion. Positive ridge need not yield an exactly affine score.

For the density claims, universality gives uniform density in continuous functions on a compact space, which are dense in $L^2$ for a Borel probability measure. In particular, for $\beta>0$ the sphere kernel is universal on every compact subset $S$ of a finite-dimensional sphere. Its expansion

$$e^{\beta u^\top v}=\sum_{n=0}^{\infty}\frac{\beta^n}{n!}\langle u^{\otimes n},v^{\otimes n}\rangle$$

includes tensor monomials of every degree with strictly positive weights, so every polynomial restriction belongs to its RKHS. Polynomial restrictions contain constants and separate points of $S$, and are therefore uniformly dense in $C(S)$ by Stone–Weierstrass. This verifies the Taylor-kernel universality criterion discussed in Steinwart–Christmann (2008), §4.6; the argument here also covers compact subsets rather than requiring the entire sphere. At $\beta=0$ it is false, since the kernel is constant. For a bounded characteristic kernel, suppose centered $u\in L^2(\nu_p)$ is orthogonal to the RKHS. The finite signed measure $u\,d\nu_p$ has total mass zero and zero kernel embedding. Its positive and negative parts have equal finite mass; if that mass were nonzero, normalization would give two probability measures with equal embeddings. Characteristicness forces equality, and mutual singularity of the parts forces zero mass, a contradiction. Thus $u=0$ and the centered RKHS is dense. Finally, the linear regime satisfies BC by the assumed linear inverse on the reachable affine hull. ∎

**What it buys.** This is exact recovery of belief-state geometry, not merely correlation with a belief probe: the spectral coordinates and the belief contain the same information and are linearly interconvertible after centering. For **token-future targets**, it makes belief geometry of the kind studied by Shai et al. an exact ideal-case benchmark, subject to the finite-context distinction in B.1. It does not establish that geometry for the construction's carried/read sphere-kernel activation targets: B.12 shows exact ATM fails for belief-valued Mess3 targets, and B.13 states what an approximate extension would require.

**Tractability and sampling.** A finite past window must expose the belief at the correct boundary, and BO needs the future windows in B.5 or a direct rank check. Small positive canonical correlations make the recovered block statistically fragile even when recovery is exact in the population. Dense Gram storage is $O(N^2)$, the generic dense eigensolve is $O(N^3)$, and a small final $r=d_b$ does not by itself eliminate the high-rank within-view whitening problem. For statistical consistency, Fukumizu–Bach–Gretton (2007), Lemmas 6–7 and Eq. (9), give operator-norm consistency only in a controlled joint ridge/sample limit, not ridge-first interpolation; see the verified conditions below.

**Theorem B.5 (finite-window observability of token futures).** For a word $u=a_1\cdots a_w$, write $T_u=T_{a_1}\cdots T_{a_w}$ and define the future-window matrix

$$O_w[u,i]=e_i^\top T_u\mathbf1.$$

The indicator vector $G_w=(\mathbf1_{Y_{t+1:t+w}=u})_u$ satisfies

$$\mathbb E[G_w\mid b_t]=O_wb_t.$$

Thus ATM holds, and BO on the whole simplex is equivalent to $\operatorname{rank}O_w=k$. Under **linear observability**, meaning

$$\operatorname{span}\lbrace T_u\mathbf1:u\text{ any finite word}\rbrace=\mathbb R^k,$$

some $w\leq k-1$ suffices, allowing $w=0$ when $k=1$. More generally one only needs injectivity on $E$. If the future window begins after $g$ unobserved transitions, its matrix is $O_w(T^g)^\top$; full-rank $T$ preserves the result, while a singular gap transition may destroy it.

*Proof.* Condition on $Z_t=i$, sum hidden paths, and then average with weights $b_t(i)$ to obtain the expectation formula. Since each column of $O_w$ sums to one, $O_wv=0$ implies $\mathbf1^\top v=0$. Therefore failure of full column rank is exactly failure of injectivity on the simplex's tangent space; a nonzero tangent null vector can be added and subtracted with small coefficient from an interior belief.

Let $V_w=\operatorname{span}\lbrace T_u\mathbf1:|u|=w\rbrace$. Then $V_0=\operatorname{span}\lbrace\mathbf1\rbrace$, $V_w\subseteq V_{w+1}$ by summing over the last symbol of an extension, and $V_{w+1}=\operatorname{span}_a T_a V_w$. If $V_{w+1}=V_w$, this space is invariant under every $T_a$, so it never grows again. Under linear observability it must therefore grow in dimension at every step until it is $\mathbb R^k$, requiring at most $k-1$ steps. Marginalizing the initial $g$ symbols multiplies by $T^g$, proving the gap formula. ∎

**What it buys.** “Future-rich enough” becomes an explicit observability matrix and a finite test: the future predictions are an injective linear image of the belief. This is the predictive-state/spectral-HMM bridge, rather than an appeal to expressivity of an unspecified kernel.

**Tractability and sampling.** The $w\leq k-1$ bound is an exact identifiability bound, not a well-conditioning guarantee; there are $|\mathcal A|^w$ indicator features and some words can be extremely rare. Hsu–Kakade–Zhang (2012), §2.3 Condition 1 and Lemma 2, assume positive hidden masses and full column rank of both emission and transition matrices, proving $\operatorname{rank}P_{2,1}=k$; their block-observation discussion explains why longer windows can restore observability. Mere minimality in the number of nonnegative hidden states is not the linear observability/reachability assumption used here. These indicators are a **token benchmark**, not a replacement for the activation kernel: applying this theorem to a carried/read target requires that its actual feature means satisfy ATM and expose these predictive directions. Dense activation Gram costs remain $O(N^2)$ storage and $O(N^3)$ eigensolving; explicit indicator covariance estimation trades this for potentially exponential feature dimension and rare-word sample requirements.

**Theorem B.6 (exact sphere-kernel calibration on a low-rank observed Markov chain).** Choose distinct beliefs $\eta_1,\ldots,\eta_{n_b}\in\Delta^{k-1}$ and put them in the rows of $\mathsf E\in\mathbb R^{n_b\times k}$. Let $\mathsf W\in\mathbb R^{k\times n_b}$ be strictly positive and row-stochastic. Define an edge-emitting HMM by

$$T_a=\mathsf W[:,a]\eta_a^\top,\qquad T=\mathsf W\mathsf E.$$

Assume $\mathsf E$ has column rank $k$, $\mathsf W$ has row rank $k$, and $T$ is invertible. Take the stationary version, and let $h_j^L=A b_j$ with $A$ full column rank. For a one-lag past $h_{t-1}^L$ and one-lag carried future $h_{t+1}^L$, using the actual sphere kernels at any $\beta>0$, the zero-ridge population context rank is exactly $k-1$ and the nonzero past scores are affine belief coordinates as in B.4. The same holds for any fixed positive separation between the two included positions.

*Proof.* After symbol $a$, Bayes' formula gives $b_j=\eta_a$ regardless of the previous belief, since $b^\top T_a=(b^\top \mathsf W[:,a])\eta_a^\top$. In particular $b_j=\eta_{Y_j}$, so the **token process itself is first-order Markov**, with transition matrix $\mathsf T_b=\mathsf E\mathsf W$. The belief process is the same chain with its $n_b$ symbols relabelled by distinct beliefs. Strict positivity of $\mathsf W$ makes every entry of this transition matrix positive. Its stationary masses $p_a$ are all positive. For separation $\ell\geq1$,

$$\mathsf T_b^\ell=\mathsf E(\mathsf W\mathsf E)^{\ell-1}\mathsf W.$$

Consequently the conditional mean of **every** feature of the future belief is affine (indeed linear) in the present belief. Its $n_b$-category future distribution is an injective linear function of $b$, because $T^{\ell-1}$ is invertible and $\mathsf W$ has row rank $k$. The support of $b$ affinely spans the simplex, since $\mathsf E$ has rank $k$ and its rows sum to one. Thus $d_b=k-1$.

The normalized activations at the distinct beliefs are distinct by the full-rank argument in B.1. The kernel $\exp(\beta u^\top v)$ is strictly positive definite on finitely many distinct unit vectors: its power-series feature map contains every tensor monomial of every degree with positive weight; polynomials separate and interpolate a finite collection of distinct points. A zero quadratic form would therefore annihilate every interpolating polynomial and force every coefficient to vanish. Each finite-support RKHS consequently contains all functions of the corresponding belief. PA, TS, ATM, BC, and BO hold, and B.4 applies. The two positions in §6.3 correspond to $\ell=2$, not $\ell=1$. ∎

**What it buys.** This calibrates the sphere-kernel estimator against a low-rank observed Markov chain: it recovers a $k-1$ dimensional predictive span, not $n_b-1$ independent state contrasts. There can be $n_b>k$ distinct posterior points, including interior points, but the belief is a function of the last token and full one-hot token CCA gives exactly the same zero-ridge correlations. Neither kernel nonlinearity nor normalization adds predictive information in this example. It is not a demonstration of non-Markov mixed-state or fractal-geometry recovery.

**Tractability and sampling.** One lag on each side is enough here, so there is no window-search identifiability problem. Independent stationary documents with one eligible pair each give i.i.d. samples; a document must contain the two positions and intervening transition. To see every belief with probability at least $1-\delta$, $n_{\mathrm{doc}}\geq p_{\min}^{-1}\log(n_b/\delta)$ independent past observations suffice by a union bound, but accurate recovery also depends on the smallest positive canonical correlation and kernel-conditioning constants. Generic computation still forms $N\times N$ Grams; grouping truly identical views gives an exact $n_b$-support implementation in this ideal model. Finite support verifies compactness, bounded moments, and the supported covariance-range conditions. Fukumizu–Bach–Gretton Lemmas 6–7 give operator-norm consistency; Weyl gives consistent rank counting, and the same isolated spectral-projection perturbation used in B.4 gives block convergence. Their Theorem 1 itself concerns only the largest simple singular pair.

## Causal states, not hidden labels

**Theorem B.7 (causal-state recovery and the correct rank count).** Define two observation pasts to have the same **causal state** when they induce the same conditional law of the entire observation future, identifying null-set ambiguities. Let $\mathscr F_-=\sigma(Y_{\mathrm{past}})$ and $\mathscr F_+=\sigma(Y_{\mathrm{future}})$, and assume the past view $\Pi$ is $\mathscr F_-$-measurable. Require formally:

1. **Past predictive sufficiency:** for every bounded $\mathscr F_+$-measurable $g$, $\mathbb E[g\mid\mathscr F_-]$ is $\sigma(\Pi)$-measurable, so it equals $\mathbb E[g\mid \Pi]$ almost surely.
2. **Past feature completeness:** with $\mathcal R_{\mathrm{tok}}$ the closed linear span in $L^2_0(\nu_p)$ of $\mathbb E[g\mid \Pi]-\mathbb Eg$ for these $g$, require $\mathcal R_{\mathrm{tok}}\subseteq\mathcal V_p$.
3. **Linear prediction-completeness of the actual target:** the closed span of $\mathbb E[g(\Xi)\mid \Pi]-\mathbb E g(\Xi)$ for target RKHS functions $g$ is exactly $\mathcal R_{\mathrm{tok}}$. All target conditional expectations here are given $\Pi$, not silently given the entire token past. This assumption rules out both missing token-predictive directions and extra activation-cache directions.
4. **Finite predictive rank:** $\dim\mathcal R_{\mathrm{tok}}=r<\infty$.

All finite-word future indicators satisfy target completeness when packaged as $\Phi_{\mathrm{word}}=(\sqrt{w_u}\,\mathbf1_{Y_{\mathrm{future}}\text{ starts with }u})_u$, with strictly positive weights $\sum_u w_u<\infty$, so that the Hilbert-valued feature has bounded norm. For example, on an alphabet of size $a$, use $w_u=2^{-|u|}a^{-|u|}$ for nonempty words. B.5 makes a sufficiently long finite window enough under observability. PA, TS, ATM, BC, and BO together with token observability give another sufficient regime. Without linear prediction-completeness, a family that only separates pairs of causal states need not yield the measure-span rank formula below.

Then the $r$ nonzero, zero-ridge past canonical coordinates separate exactly the causal states: for suitable versions on a common full-measure set,

$$s^L(p)=s^L(p')\quad\Longleftrightarrow\quad \mathcal L(Y_{\mathrm{future}}\mid p)=\mathcal L(Y_{\mathrm{future}}\mid p').$$

For an observable HMM satisfying B.4–B.5, these coordinates are the affine belief coordinates, and distinct reachable beliefs are distinct causal states. Without hidden-state observability, the target recovers the predictive equivalence class of a belief, not its unobservable coordinates.

If there are $n_c$ causal states with positive probabilities $p_1,\ldots,p_{n_c}$ and future laws $\nu_1,\ldots,\nu_{n_c}$, then

$$r=\dim\operatorname{span}\lbrace\nu_i-\bar\nu:1\leq i\leq n_c\rbrace\leq n_c-1,\qquad \bar\nu=\sum_i p_i\nu_i.$$

Equality $r=n_c-1$ holds **if and only if** these future laws are affinely independent (equivalently, linearly independent as probability measures). Pairwise distinction is enough for separation, but not for this rank equality. The empirical $r_{\mathrm{ctx}}$ of §6.3 equals this rank only asymptotically under consistent estimation and a threshold strictly between zero and the smallest positive canonical correlation, with estimation and ridge errors smaller than both threshold margins.

*Proof.* The causal state is sufficient by its definition: conditioning on its value fixes the full conditional future law. If another past statistic $U$ is sufficient, every past with the same $U$ has the same future law, so the causal state is a function of $U$ almost surely. It is therefore the minimal sufficient statistic, up to invertible relabelling. This is the sufficient-statistic/refinement argument of Shalizi–Crutchfield (2001), **Corollary 1 and Lemma 7**; their Theorems 1–3 concern prescience, statistical-complexity minimality, and uniqueness. It is not an intervention claim.

Let $\mathcal T$ denote the centered prediction operator. Its range is spanned by conditional-expectation functions, and its $r$ left singular functions form a basis of that range. Equality of their values is equivalent to equality of every function in the range. Linear prediction-completeness makes this equivalent to equality of future laws. To handle versions, use the countable cylinder events generating a finite-alphabet future, choose a basis from their finite-dimensional prediction span, and discard the countable union of the relevant null sets. The resulting identities hold simultaneously there.

For finitely many causal states, use linear prediction-completeness to compute the range with token observables: the map sending a bounded token-future observable $g$ to $(\int g\,d(\nu_i-\bar\nu))_{i=1}^{n_c}$ has rank equal to the span dimension of these signed measures: finite-dimensional duality and separation of distinct signed measures by bounded measurable functions give this equality. Weighting by positive $p_i$ and whitening do not alter rank. The single relation $\sum_i p_i(\nu_i-\bar\nu)=0$ bounds it by $n_c-1$. There are no additional relations exactly when the $\nu_i$ are affinely independent. Since each has total mass one, a linear relation among them automatically has coefficients summing to zero, proving equivalence to linear independence. Finally, singular values move by at most an operator-norm perturbation (Weyl), so a threshold with the stated margins counts exactly the positive population values under consistency. Fukumizu–Bach–Gretton (2007), Lemmas 6–7 and Eq. (9), supply this operator-norm convergence in the i.i.d., bounded-moment, compact-correlation setting; this rank argument does not invoke their largest-pair Theorem 1 for lower-order pairs. ∎

**What it buys.** Spectral coordinates can recover the minimal predictive statistic without clustering or semantic labels. The meaningful object is the quotient by future-law equivalence; its number of states, its geometric embedding dimension, and its linear predictive rank are different quantities.

**Tractability and sampling.** Finitely many distinct future laws can be distinguished by some finite collection of cylinder probabilities, since a different pair must disagree on some finite word; taking the maximum of finitely many distinguishing lengths gives a sufficient window. This does not guarantee affine independence or good conditioning. Observability supplies the explicit length bound in B.5 when its stronger assumptions hold. Document counts depend on the rarest causal state, separating probability differences, and spectral gap, not just $n_c$. Dense Gram memory/eigensolve costs are again $O(N^2)/O(N^3)$; the sampling results below apply only with independent documents or a separately justified dependence analysis.

**Belief simplex versus causal-state set.** Hidden states are generator states, not necessarily predictive equivalence classes. A non-unifilar generator generally yields mixed beliefs, sometimes infinitely many, and distinct mixtures can still have identical future laws if the generator is not observable. A synchronized finite $\epsilon$-machine uses one-hot beliefs over its causal states, but mixed states from finite, unsynchronized histories of even a unifilar presentation can be nontrivial. Thus non-unifilarity is a common source of the distinction, not an iff characterization. The construction always recovers states **relative to its chosen target**: a carried activation target may distinguish beliefs whose *token* futures agree, while an impoverished read target may merge token causal states. Token causal-state recovery requires the stated prediction-completeness for tokens, not merely for the activation stream; the rank formula requires its linear-span form.

**Proposition B.8 (finite causal states need not give rank $n_c-1$).** There exists a stationary ergodic process with three causal states and a finite mixed-state presentation whose centered token-future prediction rank is one.

*Proof.* Observe the state of a three-symbol Markov chain with transition matrix

$$\mathsf T_c=\begin{pmatrix}.6&.3&.1\\.2&.3&.5\\.4&.3&.3\end{pmatrix}.$$

All entries are positive, so its stationary law has full support and it is ergodic. Its three rows are distinct, hence the last observed symbol gives exactly three distinct causal states. The third row is the average of the first two. For any event in the entire future, its conditional probability given the present state is the corresponding row of $\mathsf T_c$ applied to the vector of probabilities of that event conditional on the next state; hence the third future law is the same average of the first two. The first two differ already at the next symbol. Their affine span has dimension one, so B.7 gives centered rank one. The observed-state presentation has only the three vertex beliefs and is unifilar, so neither finiteness nor unifilarity repairs the proposed $n_c-1$ equality. ∎

**What it buys.** A low-dimensional predictive embedding can faithfully label more causal states than its rank plus one. Rank is not a state-count estimator without affine independence.

**Tractability and sampling.** A single future token already distinguishes all three states here; longer windows cannot create the absent independent direction. Estimate a $3\times3$ table to verify the benchmark rather than building a large Gram matrix, though the general kernel route has the same $O(N^2)$ storage and $O(N^3)$ solve. Rare-state coverage and the nonzero singular gap, not the number of labels alone, control document requirements.

## Readability and the token Hankel benchmark

**Corollary B.9 (bidirectional linear readability and Hankel rank).** Under B.4, population ridge regression with an unpenalized intercept satisfies $R^2\to1$ as its ridge tends to zero, both for $s^L$ predicted from $b$ and for $b$ predicted from $s^L$, provided the relevant total variance is nonzero. With linear belief-sufficiency, the same holds between $s^L$ and $h^L$ at the accessible boundary. This is a statement about the limiting score block; finite positive kernel-CCA ridge generally introduces nonlinear distortions.

For token past words $v$ and future words $u$, let

$$\mathsf H[v,u]=\Pr(Y_{\mathrm{past}}=v,Y_{\mathrm{future}}=u)=\mathsf H_{\mathrm{past}}[v,:]O[u,:]^\top,$$

where $\mathsf H_{\mathrm{past}}[v,i]=\Pr(Y_{\mathrm{past}}=v,Z_{\mathrm{cut}}=i)$ and $O$ is the appropriate future-window matrix. Then $\operatorname{rank}\mathsf H\leq k$. If **linear reachability** ($\operatorname{rank}\mathsf H_{\mathrm{past}}=k$) and **linear observability** ($\operatorname{rank}O=k$) both hold, the rank equals $k$, and its marginal-centered version has rank $k-1$. This is the precise linear-minimal, sufficiently long-window token benchmark; the same numerical rank for an activation estimator requires B.3–B.4 separately.

*Proof.* The identities in B.4 supply affine zero-error predictors in both directions. In a finite-dimensional population least-squares problem, diagonalizing the predictor covariance shows that ridge coefficients converge to a zero-error pseudoinverse solution on its support. Thus residual variance tends to zero and $R^2$ tends to one. A linear injective activation admits a linear inverse on the varying belief subspace, giving the activation assertion. A coordinate with zero variance has undefined individual $R^2$ and is excluded or handled via a nonzero total-variance score.

The hidden-state cut yields $\mathsf H=\mathsf H_{\mathrm{past}}O^\top$. Let $\pi$ be its stationary hidden law; for exhaustive fixed-length word partitions, $\sum_v\mathsf H_{\mathrm{past}}[v,:]=\pi^\top$ and $O^\top\mathbf1=\mathbf1$. The marginals are $\mathsf H_{\mathrm{past}}\mathbf1$ and $O\pi$. Therefore

$$\mathsf H_c=\mathsf H_{\mathrm{past}}(I-\mathbf1\pi^\top)O^\top.$$

The middle factor has rank $k-1$, and full-column-rank outer matrices have left/right inverses on their ranges. Consequently the uncentered and centered ranks are $k$ and $k-1$. A skipped gap inserts $T^g$; equality then requires its appropriate full-rank action, as in B.5. No hidden labels or belief fits are used in forming the token probability table. ∎

**What it buys.** Exact geometric recovery has an operational validation criterion: a pair of closed-form linear fits with held-out $R^2$ approaching one. It also makes explicit which token-level rank is a known calibration and which activation-level agreement is an additional achievement.

**Tractability and sampling.** Regression from width $d$ to $r=d_b$ coordinates can use a covariance/SVD implementation, with dense predictor factorization $O(d^3)$ and moment formation $O(Nd^2+Ndr)$; in a known rank-$r$ synthetic model this can be reduced to the supported subspace. Token tables have dimensions exponential in the selected past/future word lengths. Use document-level train/test splits and estimate uncertainty across documents, not the training fit. A raw-activation linear kernel is an explicitly labelled calibration variant: the $\beta\to0$ sphere kernel is linear in **normalized** activations, which is not generally linear in $b$.

## Sharp failure cases

**Proposition B.10 (belief-sufficiency and belief-Markov do not imply the requested rank or recovery).** There is a stationary ergodic, linearly observable, minimal two-hidden-state HMM with exactly three reachable beliefs, an exactly linear belief-sufficient layer, and a belief-Markov activation stream, for which the sphere-kernel carried context rank is two, exceeding $k-1=1$. The target conditional mean embedding is injective in the belief, yet the two nonzero canonical variates cannot span just the one centered belief-coordinate direction.

*Proof.* With $\pi=(1/2,1/2)^\top$ and $c=a=d=1/3$, take four edge symbols $H,0,1,R$ with

$$T_H=cI,\quad T_0=a\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad T_1=a\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad T_R=d\mathbf1\pi^\top.$$

Their sum is $(c+a)I+d\mathbf1\pi^\top$, a positive, irreducible, aperiodic hidden transition matrix. Symbol $H$ leaves the belief unchanged; $0$ resets it to $e_1$; $1$ resets it to $e_2$; and $R$ resets it to $\pi$. Almost surely a reset occurred in the finite past, so the stationary belief support is exactly $\lbrace e_1,e_2,\pi\rbrace$. Its transition matrix and stationary law are

$$\mathsf T_b=\begin{pmatrix}c+a&0&d\\0&c+a&d\\a/2&a/2&c+d\end{pmatrix},\qquad p=(1/4,1/4,1/2)^\top.$$

This chain is reversible, with eigenvalues $1,c+a,c=1,2/3,1/3$. Set $h_j^L=b_j$. This is linearly belief-sufficient and belief-Markov. Distinct normalized beliefs yield a strictly positive definite sphere Gram matrix by B.6, so the kernel represents all functions on the three-point support. CCA between positions $t-1,t+1$ is therefore full $L^2$ CCA for $\mathsf T_b^2$. Reversibility makes its two centered canonical correlations exactly $4/9$ and $1/9$. The linear belief coordinate has dimension one, while the additional direction distinguishes the mixed belief from the pure beliefs.

The missing condition is ATM: for one transition, the probability that the next belief equals $\pi$ is $d$ at both pure beliefs but $c+d$ at their midpoint, not an affine function. Nevertheless the future-belief laws are distinct. More strongly, on the whole simplex the one-transition law is

$$c\,\delta_b+a b_1\delta_{e_1}+a b_2\delta_{e_2}+d\delta_\pi,$$

which is injective in $b$: distinct non-special beliefs have distinct movable atoms; the three special-point laws are also distinct and cannot equal a non-special law. A characteristic target kernel preserves this injectivity. Two-transition laws likewise retain a movable atom of mass $c^2$ and distinguish the special points. Thus even injective conditional mean embeddings do not replace ATM.

For token futures, $0$ and $1$ have likelihood vectors $(a,0)^\top$ and $(0,a)^\top$, so hidden-state observability holds. Positive-probability histories ending in $0$ and $1$ yield both pure beliefs, giving reachability. The token Hankel rank is exactly two, proving minimality against a one-state generator and a centered token rank of one. In this example there are three token causal states, whose future laws are affinely dependent, while the activation-future operator has centered rank two. ∎

**What it buys.** This is a direct stress test of the proposed bridge: even a perfect belief layer can have more nonlinear carried modes than hidden-state contrasts. It pinpoints why the affirmative theorem must constrain target conditional means, not merely assert a Markov belief or an injective embedding.

**Tractability and sampling.** Everything is visible in three belief categories and a four-symbol alphabet; one lag on each side suffices. A $3\times3$ joint table verifies the two correlations exactly, and the numerical check below verifies the sphere-kernel implementation. Generic dense Grams are unnecessary for this calibration, but larger instances retain their $O(N^2)/O(N^3)$ costs. No amount of sampling makes the false $k-1$ claim true; sufficiently many independent documents instead resolve the second genuine mode.

**Proposition B.11 (raw-past leakage, and exact carried/read ranks for copying).** Let $(Y_j)$ be i.i.d. uniform on $V\geq2$ symbols, a one-hidden-state HMM.

1. With $h_j^L=2^{-1/2}(e_{Y_j},e_{Y_{j-2}})$, one-lag carried CCA at positions $t-1,t+1$ has population centered rank exactly $V-1$, even with the actual sphere kernels at $\beta>0$. For $V=2$ this is the smallest nontrivial alphabet and gives rank one although $k-1=0$.
2. With $h_j^L=e_{Y_j}$ and $h_j^{L+1}=e_{Y_{j-1}}$, the carried rank is exactly zero for all disjoint past/future windows, while the read rank using target $h_t^{L+1}$ and a past containing $h_{t-1}^L$ is exactly $V-1$. Its zero-ridge nonzero correlations are all one. Using the raw, unnormalized **linear** update target $e_{Y_{t-1}}-e_{Y_t}$ gives the same rank and all correlations $1/\sqrt2$.

*Proof.* In (1), the two views contain $(Y_{t-1},Y_{t-3})$ and $(Y_{t+1},Y_{t-1})$. Conditional on the shared token $Y_{t-1}$ their nuisance tokens are independent. The centered prediction operator therefore has rank at most $V-1$, while its range contains every centered function of that shared token, giving equality and correlations one. Sphere kernels represent all functions of these finitely many distinct unit vectors by B.6. The belief is constant; these activations retain raw-past information and are not functions of that belief, violating the strong belief-sufficiency assumption.

In (2), all tokens in the embedding past are independent of all tokens in the carried future, so the centered cross-covariance is zero. The read target is exactly the previous token already in the past. Its centered function space has dimension $V-1$ and is shared perfectly. For the linear update, let $\Sigma=\operatorname{Cov}(e_Y)=I/V-\mathbf1\mathbf1^\top/V^2$. The previous-token covariance is $\Sigma$, its cross-covariance with the update is $\Sigma$, and the update covariance is $2\Sigma$. Whitening on the contrast subspace gives $I/\sqrt2$.

The update is zero when $Y_{t-1}=Y_t$, so applying the sphere kernel directly invokes `THEORY.md`'s zero-view exclusion policy. On the retained sites the normalized difference identifies its positive coordinate, hence the previous token. With uniform tokens the conditional previous-token law remains uniform, so the sphere-update read rank is still $V-1$ with correlations one. The carried future remains independent of the retained past because the exclusion event uses only $Y_{t-1},Y_t$. The excluded mass is $1/V$ and must be reported; these are conditional-on-retention results, not the original population. ∎

**What it buys.** The first case proves the raw-cache caution in §6.3 with the minimal hidden-state count. The second proves, with exact ranks, why “what the stream carries” and “what the next layer reads” are genuinely different predictive objects.

**Tractability and sampling.** A one-lag past and one-lag future suffice; the two-step delay in (1) is deliberate because §6.3 excludes the current position. With categorical features the exact computation is a small probability table; a dense kernel implementation costs $O(N^2)$ memory and $O(N^3)$ eigensolving. Use independent documents containing the necessary boundary positions, count removed zero updates, and evaluate rank/correlations out of sample. Additional overlapping windows within a document are not additional independent samples.

## Characteristic activation targets and approximate recovery

**Proposition B.12 (affine mean embeddings, affine conditional laws, and Mess3).** Fix versions of conditional laws on a full-measure set $\mathcal S$ of realized beliefs. Let the target be $\Xi=\chi(B^+)$, where $B^+$ is a future belief (or a belief window), and $\chi$ is measurable and injective with measurable inverse on its image, including after any normalization. Suppose the kernel on $\Xi$ is bounded and characteristic. Then ATM holds on $\mathcal S$ **if and only if** the conditional-law map

$$b\longmapsto\nu_b:=\mathcal L(B^+\mid b)$$

is affine on $\mathcal S$, in the precise sense that every finite affine relation among inputs is respected:

$$\sum_j c_j=0,\quad\sum_jc_j b_j=0
\quad\Longrightarrow\quad\sum_jc_j\nu_{b_j}=0.$$

Equivalently, this law has an affine extension to the affine hull of $\mathcal S$ with values in finite signed measures of total mass one. One can further extend to the simplex hyperplane and write $\nu_b=\sum_i b_i\nu_i$, but the fixed $\nu_i$ need not be nonnegative probability measures. The two-point example in B.1 witnesses precisely this distinction.

In particular, consider an HMM with strictly positive, invertible symbol matrices $T_a$ and an infinite stationary belief support. For every fixed lag $\ell\geq1$, the conditional-law map of $b_{t+\ell}$ given $b_t$ is **not affine** on any full-measure belief set. Hence ATM fails for every injectively normalized belief-valued sphere-kernel target at $\beta>0$.

This applies to Mess3 at $x=0.05$, $\alpha=0.85$, with symbol matrices

$$T_a=\mathsf F\operatorname{diag}(e_a),\qquad
\mathsf F_{ij}=\begin{cases}1-2x&i=j,\\x&i\ne j,\end{cases}
\qquad e_a(j)=\begin{cases}\alpha&j=a,\\(1-\alpha)/2&j\ne a.\end{cases}$$

Thus exact ATM fails for both the one-lag read-style target $b_t\mid b_{t-1}$ and the strict-past/strict-future carried target $b_{t+1}\mid b_{t-1}$. The read-style statement concerns a read target that actually encodes that belief; it is not a statement about an unspecified trained model's next-layer update. For a general update, the characteristic-kernel equivalence concerns the conditional law of its actual normalized target, which need not be an injective function of one belief. Token-future targets still satisfy ATM by B.5.

*Proof.* If ATM holds, a finite affine input relation gives a zero embedding for the zero-total-mass signed measure $\sum_jc_j\chi_*\nu_{b_j}$. A bounded characteristic kernel is injective on finite signed measures of mass zero: take the Jordan decomposition, whose two parts have equal mass, normalize them to probability measures, and invoke characteristicness. Consequently that signed measure is zero, and the injective measurable map $\chi$ lets us pull the equality back to the laws of $B^+$. Conversely, an affine conditional-law map gives affine feature means by integration. Choose an affine basis from $\mathcal S$; preservation of all affine relations gives a well-defined signed-measure extension by the unique affine coordinates in that basis. Its mass is one. Extending this finite-dimensional affine map to the simplex hyperplane gives the signed vertex measures asserted above. This proves the equivalence without incorrectly requiring those extensions to be probabilities.

For the non-affineness criterion, the word update $F_u(b)=T_u^\top b/(\mathbf1^\top T_u^\top b)$ is injective when $T_u$ is invertible. Every word has positive probability. For a fixed lag, the conditional law is supported on finitely many such images. Each possible output belief has at most one preimage under each word map. Starting from an infinite full-measure belief set, we can therefore choose arbitrarily many beliefs whose finite sets of possible lag-$\ell$ outputs are pairwise disjoint: each new choice avoids only finitely many preimages of the outputs selected earlier. Select $k+1$ such inputs. They have a nonzero affine dependence in the $(k-1)$-simplex, but the corresponding probability laws, having disjoint supports, have no nonzero linear dependence. The law map is not affine, and the first part applies.

For Mess3, $\sum_aT_a=\mathsf F$, whose eigenvalues are $1,1-3x,1-3x$. At the stated parameters $\mathsf F$ is invertible and each positive diagonal emission matrix is invertible; every $T_a$ is strictly positive and invertible. It remains to verify infinite stationary belief support. If that support were finite, each continuous word-one map $F_a$ would map it into itself (all symbol probabilities are positive) and, by injectivity, permute it. Iterating a strictly positive matrix and normalizing converges to its unique Perron left eigenvector, by Perron–Frobenius for strictly positive matrices (standard; see Horn–Johnson, *Matrix Analysis*, second edition, Chapter 8). A permutation orbit in a finite set can converge only if it is fixed. Thus the support would be a singleton fixed by every $F_a$. Summing $b^\top T_a=p_a b^\top$ would then make $b$ stationary for $\mathsf F$, so $b=(1,1,1)^\top/3$. But $F_a(b)=e_a\ne b$ when $\alpha\ne1/3$, a contradiction. Therefore the support is infinite. The sphere-kernel universality argument in B.4 implies characteristicness (integrals of its dense polynomial family determine probability measures), completing the claimed activation-target non-example. ∎

**What it buys.** For the construction's rich belief-valued activation targets, ATM constrains the entire conditional belief law, not merely the raw conditional mean. Exact token-future recovery therefore does not automatically transfer to carried/read sphere-kernel recovery—even with perfect linear belief activations. Mess3 supplies an analytically justified non-example. In fact B.14 shows its universal belief-valued zero-ridge operator is noncompact with a defect bounded away from zero; the compact small-defect theorem B.13 does not apply to it.

**Tractability and sampling.** The proof is population-level and needs no fitted probe. A diagnostic can sum over the $|\mathcal A|^\ell$ future words to compute conditional means of finitely many kernel sections at sampled beliefs, then measure their affine residual. This costs $O(N|\mathcal A|^\ell n_{\mathrm{ref}}d)$ after word matrices are formed and is a validation computation, not a change to the estimator. A nonzero residual refutes ATM on those evaluated points; a small residual does not bound the full whitened prediction operator. The deterministic parameter check below verifies positivity/invertibility, and the fixed-seed numerical diagnostic quantifies—not proves by sampling alone—the resulting near-affine behavior. The full construction still has $O(N^2)$ Gram storage and a generic $O(N^3)$ eigensolve; independent-document and window requirements remain as above.

**Proposition B.14 (Mess3 has infinite zero-ridge context rank).** In the stationary infinite-past Mess3 model of B.12 at $x=1/20$, $\alpha=17/20$, let the layer be exactly $h=Ab$, with $A$ full column rank. Use a universal sphere kernel with any $\beta>0$ on each injectively normalized belief view. Let $\mu$ be the stationary belief law, and let

$$\mathcal T_\ell:L^2_0(\mu)\to L^2_0(\mu),\qquad
(\mathcal T_\ell g)(b)=\mathbb E[g(b_{t+\ell})\mid b_t=b].$$

For the read-style belief target ($\ell=1$) and the strict-past/strict-future carried target ($\ell=2$), respectively, every variational canonical level obeys

$$\kappa^{\mathrm{var}}_n(\mathcal T_1)\geq c_1:=\sqrt{\frac{153}{800}}=0.4373213921\ldots,\qquad
\kappa^{\mathrm{var}}_n(\mathcal T_2)\geq c_2:=\sqrt{\frac{67677}{640000}}=0.3251850435\ldots
\quad(n\geq1).$$

Here $\kappa^{\mathrm{var}}_n(\mathcal T)=\sup_{\dim V=n}\inf_{g\in V,\|g\|=1}\|\mathcal T g\|$. For a compact operator these are its usual singular values. Here the operators are **not compact**, so we do not presume a discrete singular-function expansion; the variational statement is the precise meaning of arbitrarily many canonical directions bounded away from zero. The same bounds hold for the approximation numbers $a_n(\mathcal T)=\inf_{\operatorname{rank}K<n}\|\mathcal T-K\|$.

Consequently both zero-ridge population context operators have infinite rank. Their affine-prediction defects satisfy $\|(I-\operatorname{proj}_{\mathcal B})\mathcal T_\ell\|\geq c_\ell$, and their distance from **any compact operator**, including any positive-ridge population smoother operator, is at least $c_\ell$. Neither the compactness hypothesis of B.13 nor the compact-limit consistency hypothesis in Fukumizu–Bach–Gretton holds for these targets.

*Proof.* Write $r=(1-\alpha)/2=3/40$, $z=b_a$, and $q_a=x+(1-3x)z$. The probability of symbol $a$ and the $a$-coordinate after that symbol are

$$p_a(b)=r+(\alpha-r)q_a=\frac{91+527z}{800},\qquad
F_a(b)_a=f(z):=\frac{\alpha[x+(1-3x)z]}{r+(\alpha-r)[x+(1-3x)z]}.$$

The function $f$ is strictly increasing. For $a'\ne a$, the whole-simplex bounds $q_a\leq1-2x$ and $q_{a'}\geq x$, attained together at $b=e_a$, give the exact cutoff

$$F_{a'}(b)_a\leq\frac{r(1-2x)}{\alpha x+r(1-x)}=\frac{54}{91}=0.5934065934\ldots.$$

Thus an output above this cutoff can only have been produced by symbol $a$, and invertibility of $T_a$ identifies its previous belief. To obtain simple exact lower-bound constants rather than a grid-rounded cutoff, use the smaller exclusive region

$$\mathcal R_a=\lbrace b\in\operatorname{supp}\mu:b_a>2/3\rbrace.$$

Direct substitution gives $f(z)>2/3$ iff $z>2/17$, while $\min_{z\in[0,1]}f(z)=34/91>2/17$. Therefore $F_a^2(\operatorname{supp}\mu)\subset\mathcal R_a$. By B.12 the stationary support is infinite, and $F_a^2$ is a continuous injection. The measure obtained by weighting $\mu$ by the strictly positive word probability $p_{aa}$ and pushing it through $F_a^2$ has infinite support, is supported in $\mathcal R_a$, and is dominated by $\mu$ by stationarity. Hence $\mu$ restricted to $\mathcal R_a$ has infinite support. Its centered supported functions form an infinite-dimensional subspace of $L^2_0(\mu)$.

For such a function $g$, exclusivity removes all but one term:

$$\mathcal T_1g(b)=p_a(b)g(F_ab),\qquad
\|g\|^2=\int p_a(b)|g(F_ab)|^2\,d\mu(b).$$

On the integrand's nonzero support, $b_a>2/17$ and therefore $p_a(b)\geq153/800$. Consequently

$$\|\mathcal T_1g\|^2=\int p_a(b)^2|g(F_ab)|^2\,d\mu(b)
\geq\frac{153}{800}\|g\|^2.$$

For lag two use the target region $\mathcal R_{aa}=F_a(\mathcal R_a)$. It lies inside $\mathcal R_a$ and is reachable only by the word $aa$: the last symbol is $a$, its inverse belief lies in $\mathcal R_a$, and the preceding symbol must also be $a$. Its restricted stationary measure has infinite support by the same argument, now using the image $F_a^3(\operatorname{supp}\mu)$. For centered $g$ supported there,

$$\mathcal T_2g(b)=p_{aa}(b)g(F_a^2b),\qquad
\|g\|^2=\int p_{aa}(b)|g(F_a^2b)|^2\,d\mu(b).$$

Nonzero terms require $b_a>2/17$ and $(F_ab)_a>2/3$, so

$$p_{aa}(b)=p_a(b)p_a(F_ab)\geq\frac{153}{800}\frac{1327}{2400}
=\frac{67677}{640000}.$$

The same norm calculation proves the lag-two lower bound. Each infinite-dimensional supported subspace contains an $n$-dimensional subspace for every $n$, proving the variational bounds. For any rank-$<n$ operator, an $n$-dimensional such subspace contains a unit vector in its kernel, which proves the approximation-number bounds. An orthonormal sequence from either supported subspace converges weakly to zero, so every compact operator sends it to zero in norm, whereas $\|\mathcal T_\ell g_j\|\geq c_\ell$. Thus $\mathcal T_\ell$ has distance at least $c_\ell$ from every compact operator and is itself noncompact. Since $\operatorname{proj}_{\mathcal B}\mathcal T_\ell$ is finite-rank, the defect bound follows.

Universality and injective normalization identify the closed centered kernel feature spaces with all of $L^2_0(\mu)$, as proved in B.4, so these are the zero-ridge limits relevant to kernel CCA. Bounded kernels make the evaluation covariance $\mathscr C_p$ trace-class; hence $S_{p,\rho}^{1/2}$ is compact for every $\rho>0$ (its eigenvalues are $\sqrt{\lambda_j/(\lambda_j+\rho)}\to0$), and similarly on the target. Their product with $\mathcal T_\ell$ is compact, proving the asserted nonvanishing ridge-bias norm. **The unwhitened RKHS cross-covariance $C_{pq}$ remains Hilbert–Schmidt and compact**; it has infinite rank: the centered evaluation maps have dense ranges and their adjoints are injective, so a finite-rank $C_{pq}=J_p^*\mathcal T_\ell J_q$ would force $\mathcal T_\ell$ to have finite rank. Noncompactness belongs to the whitened correlation/conditional-expectation operator, not to that bounded-kernel cross-covariance. ∎

**What it buys.** A perfect belief representation of Mess3 does not turn universal carried/read-style kernel CCA into a two-dimensional predictive-state recovery. Infinitely many functions of the previous belief are readable on exclusive target regions. Small unwhitened feature tails therefore coexist with a nonvanishing infinite-dimensional correlation tail; B.13 is not an applicable small-defect theorem for these zero-ridge Mess3 targets.

**Tractability and sampling.** The stated constants are analytic and checked with rational arithmetic; no finite-grid singular spectrum is used as a proof of infinite rank. Positive ridge produces a compact population operator, so a fixed positive threshold can yield a finite effective rank; finite samples additionally cap rank at $N-1$. Any observed finite $r_{\mathrm{ctx}}$ here depends on the ridge, threshold, kernel scale, window, and sampling, not on a population $k-1$ cutoff. Fukumizu–Bach–Gretton's compact-limit theorem does not apply, though fixed-ridge covariance estimation remains possible. For fixed maximum document/context length the belief support can instead be finite, so the infinite-rank conclusion is specifically the stationary infinite-past limit, not a claim about a finite corpus. The ordinary dense construction still costs $O(N^2)$ memory and a generic $O(N^3)$ eigensolve.

**Theorem B.13 (quantitative recovery with an affine-prediction defect).** Assume PA, TS, and BC, but do **not** assume ATM. On the centered $L^2$ feature spaces define

$$\mathcal T g=\operatorname{proj}_{\mathcal V_p}\big(\mathbb E[g(\Xi)\mid\Pi]\big),\qquad
\mathcal T_{\mathrm{aff}}=\operatorname{proj}_{\mathcal B}\mathcal T,\qquad
\mathcal T_{\mathrm{def}}=(I-\operatorname{proj}_{\mathcal B})\mathcal T.$$

Assume $\mathcal T$ is compact (automatic for finite-dimensional feature spaces), $d_b\geq1$, and $\operatorname{range}\mathcal T_{\mathrm{aff}}=\mathcal B$. Set

$$\gamma=\sigma_{d_b}(\mathcal T_{\mathrm{aff}})>0,\qquad
\epsilon=\|\mathcal T_{\mathrm{def}}\|.$$

Then $\sigma_{d_b+1}(\mathcal T)\leq\epsilon$. More generally, for a compact operator $\widehat{\mathcal T}$ on the same $L^2$ spaces satisfying $\|\widehat{\mathcal T}-\mathcal T\|\leq e$ (for example a population regularized function-space operator), write $\eta=\epsilon+e$. If $\gamma>2\eta$, its top-$d_b$ past singular subspace $\widehat{\mathcal U}$ is isolated and satisfies

$$\sigma_{d_b}(\widehat{\mathcal T})\geq\gamma-\eta,\qquad
\sigma_{d_b+1}(\widehat{\mathcal T})\leq\eta,\qquad
\|\operatorname{proj}_{\widehat{\mathcal U}}-\operatorname{proj}_{\mathcal B}\|
\leq\frac{\eta}{\gamma-\eta}.$$

At zero ridge these past singular functions are the canonical scores. Thus the RMS-normalized top score block is close to an affine function of the belief, with an explicit subspace error; no identification of individual axes inside a repeated block is claimed. A threshold $\eta<\tau<\gamma-\eta$ counts exactly $d_b$ directions.

For actual positive-ridge population scores, take $\widehat{\mathcal T}=S_{p,\rho}^{1/2}\mathcal T S_{q,\rho}^{1/2}$ as in B.4, and let $U_\rho:\mathbb R^{d_b}\to\mathcal V_p$ be an orthonormal frame of its leading past subspace. Define the additional score-smoothing error

$$\zeta_\rho=\|(I-S_{p,\rho}^{1/2})U_\rho\|.$$

If $\zeta_\rho<1$, the span $\mathcal U_{\mathrm{score},\rho}$ of the actual ridge scores satisfies

$$\|\operatorname{proj}_{\mathcal U_{\mathrm{score},\rho}}-\operatorname{proj}_{\mathcal B}\|
\leq\min\!\left(1,\frac{\eta/(\gamma-\eta)+\zeta_\rho}{1-\zeta_\rho}\right).$$

*Proof.* Weyl's singular-value inequality and $\operatorname{rank}\mathcal T_{\mathrm{aff}}=d_b$ give all three singular-value assertions, since $\|\widehat{\mathcal T}-\mathcal T_{\mathrm{aff}}\|\leq\eta$. The strict gap follows from $\gamma>2\eta$. For leading singular frames, $\widehat{\mathcal T}\widehat V=\widehat U\widehat\Sigma$. Because $(I-\operatorname{proj}_{\mathcal B})\mathcal T_{\mathrm{aff}}=0$,

$$\|(I-\operatorname{proj}_{\mathcal B})\widehat U\|
\leq\frac{\|(I-\operatorname{proj}_{\mathcal B})(\widehat{\mathcal T}-\mathcal T_{\mathrm{aff}})\widehat V\|}{\sigma_{d_b}(\widehat{\mathcal T})}
\leq\frac{\eta}{\gamma-\eta}.$$

For equal-dimensional subspaces this is the operator norm of their projector difference. This is the residual-form Wedin bound specialized to a rank-$d_b$ reference; the displayed argument proves the version used here. For ridge scores, the smallest singular value of $S_{p,\rho}^{1/2}U_\rho$ is at least $1-\zeta_\rho$, whereas its component perpendicular to $\mathcal B$ has norm at most $\eta/(\gamma-\eta)+\zeta_\rho$. Orthonormalizing this score frame proves the final bound. Compactness also ensures that the population smoother operator converges in norm to $\mathcal T$ as $\rho\downarrow0$, by the same compact-range approximation used in B.4. This proves a ridge-bias limit, not an unspecified finite-sample source bound. ∎

**What it buys.** Exact affine recovery has an explicit robust neighbor: sufficiently small non-affine predictive dependence preserves a separated belief block and controls its angle to the true belief-coordinate span. The condition is on the **whitened $L^2$ prediction operator**; a nearly rank-two plot of unwhitened conditional kernel means does not certify the bound. Specifically, B.14 proves that Mess3’s universal belief-valued targets fail the compactness premise and have a defect bounded away from zero; this theorem is not an approximate-recovery result for that example.

**Tractability and sampling.** Estimate the positive gap and defect in the same feature/whitening convention, with held-out uncertainty. For i.i.d. document pairs in the compact regime assumed here, Fukumizu–Bach–Gretton Lemmas 6–7 give the operator-estimation and ridge-bias ingredients on their common covariance-space representation; transferring an empirical score bound to the common $L^2$ spaces above additionally requires out-of-sample evaluation/source control. There is no dimension-free conversion from a finite collection's Frobenius residual to $\epsilon$. In the exact finite-support checks below every operator, whitening, and projector is available directly; a noisy full-kernel implementation retains $O(N^2)$ memory and generic $O(N^3)$ decomposition cost. Window truncation, ill-conditioned covariances, and a small $\gamma$ can dominate sampling error even with many documents.

## Shared tractability and sampling details

These qualifications apply to every result above.

- **How many documents?** Stationarity and ergodicity alone give no distribution-free finite-sample convergence rate, nor a universal sufficient document count. A rigorous i.i.d. regime is $n_{\mathrm{doc}}$ independent stationary documents, one eligible past/target pair per document, giving $N=n_{\mathrm{doc}}$ independent pairs. More sites per document require a mixing/dependence argument; multiplying by document length is not justified. Bootstrap and split at document level. For a known finite joint table with $J$ cells, $n_{\mathrm{doc}}\geq\log(2J/\delta)/(2\epsilon^2)$ suffices for entrywise absolute error at most $\epsilon$ with probability $1-\delta$ by Hoeffding and a union bound. Translating this to rank needs the table dimensions, covariance conditioning, ridge, and the positive singular gap. There is no parameter-free claim such as “a few documents suffice.”
- **The ridge/sample limit.** Fukumizu, Bach and Gretton (2007), *JMLR* 8:361–383, **Lemmas 6–7 and Eq. (9)** give operator-norm convergence in probability for the estimated normalized correlation operator under i.i.d. pairs, the moment hypotheses of their covariance bounds (satisfied by our bounded sphere kernels), compactness of the population correlation operator, and
  $$\rho_{n_{\mathrm{doc}}}\to0,\qquad n_{\mathrm{doc}}^{-1/3}/\rho_{n_{\mathrm{doc}}}\to0.$$
  Lemma 6's stochastic term is $O_p(\rho_{n_{\mathrm{doc}}}^{-3/2}n_{\mathrm{doc}}^{-1/2})$; Lemma 7 controls the population regularization bias under compactness, without a universal bias rate. Weyl then yields the rank-by-threshold claim in B.7 for the entire singular spectrum. Theorem 1 itself covers only the **largest simple singular pair** (NOCCO convergence in RKHS norm); Theorem 2 adds covariance-range hypotheses for $L^2$ canonical-variate convergence. Their lower-order/multidimensional extension is a remark, not a proved theorem we attribute to Theorem 1. Here isolated-block convergence instead follows directly from their operator-norm convergence and the spectral-projection argument in B.4; conversion to variate functions additionally needs source/smoothing control, as explicitly displayed in B.13. Finite-support full-feature models verify the range conditions on supported subspaces. Generic continuous-belief kernels need those conditions checked separately; B.14 proves the compactness condition fails for the universal belief-valued Mess3 targets, so these zero-ridge consistency conclusions do not cover that example. The ridge must decay **slower than** $n_{\mathrm{doc}}^{-1/3}$; for example $\rho_{n_{\mathrm{doc}}}=n_{\mathrm{doc}}^{-1/4}$ meets the condition, without any optimality claim.
- **Why not zero ridge first?** If both empirical centered Gram matrices have rank $N-1$, then $R_p,R_q\to C$ as $\rho\downarrow0$ at fixed $N$, so all $N-1$ nontrivial in-sample canonical correlations tend to one even for independent views. This is precisely why §6.3 requires document-held-out selection and evaluation. Population rank and empirical positive rank are not interchangeable.
- **What computation costs.** For lag widths $w_p,w_q$ and activation width $d$, forming dense dot-product Gram matrices is $O(N^2(w_p+w_q)d)$ in a direct implementation; the two Grams require $O(N^2)$ memory and generic dense decompositions cost $O(N^3)$. With a justified $m$-feature/Nyström approximation, moment formation and reduced linear algebra are typically $O(Nm^2+m^3)$ after feature/kernel evaluation. The accuracy and conditioning of that approximation must be checked; final context rank $r\ll N$ alone does not prove a rank-$r$ approximation of either within-view Gram is adequate. Direct finite-feature CCA and exactly grouped finite-support models avoid the $N\times N$ matrices.
- **What is estimated versus assumed.** All kernels use observed activations or token windows in the separately labelled benchmark. True beliefs, hidden labels, and causal-state identities are used only to verify ideal assumptions and recovery. No supervised probe chooses the construction's basis. The two affine readability regressions are explicitly labelled fits for validation.

## Numerical verification

Executed with

```text
uv run --with numpy python -I checks/theorems_b1_check.py
```

The script uses only NumPy, runs locally in under a minute on the orchestration VPS, and asserts every claimed rank and recovery error. It is a theorem sanity check, not a trained-model experiment or a substitute for a Stage 4 protocol.

The positive case is the exact stationary HMM of B.6 with $k=3$, $n_b=5$,

$$\mathsf E=\begin{pmatrix}1&0&0\\0&1&0\\0&0&1\\.6&.3&.1\\.1&.3&.6\end{pmatrix},\quad
\mathsf W=\begin{pmatrix}.4&.1&.1&.25&.15\\.1&.4&.15&.1&.25\\.15&.1&.4&.2&.15\end{pmatrix}.$$

Synthetic width-five activations are $h=Ab$ with the fixed full-column-rank matrix

$$A=\begin{pmatrix}1&.2&0\\0&1&.3\\.1&0&1\\.4&.2&.1\\.1&.3&.5\end{pmatrix}.$$

The complete runnable source is `/home/ubuntu/code/CoherentModes/checks/theorems_b1_check.py`. The sphere Gram is evaluated at $\beta=5$ and scaled by the harmless fixed factor $e^{-5}$; positive-ridge values below refer to this declared kernel scale. Exact pair probabilities are $\operatorname{diag}(p)\mathsf T_b^2$, matching the one-lag strict-past/strict-future separation in §6.3. No trajectory truncation or estimated filter is used for the population calculation.

| asserted quantity | result |
|---|---:|
| centered within-view feature covariance rank | 4 |
| centered carried cross-covariance rank | 2 |
| nonzero zero-ridge canonical correlations | 0.1120986861, 0.0754723813 |
| relative affine reconstruction error, scores from beliefs | $<2\times10^{-14}$ |
| relative affine reconstruction error, beliefs from scores | $<2\times10^{-14}$ |
| token Hankel rank / centered rank | 3 / 2 |
| B.10 two-state counterexample carried correlations | $4/9$, $1/9$ |
| B.10 carried / centered-token ranks | 2 / 1 |
| B.8 causal-state count / centered-token rank | 3 / 1 |
| B.11 $V=3$ carried/read benchmark ranks | 0 / 2 |

The finite-ridge relative affine score error decreases from $0.02096$ at $\rho=10^{-2}$ to $2.435\times10^{-4}$ at $10^{-4}$ and $2.439\times10^{-6}$ at $10^{-6}$. This checks the important distinction between limiting affine recovery and exact positive-ridge recovery.

A multinomial sample of $N=200{,}000$ independent stationary pairs (seed 20261107), at ridge $10^{-3}$, gives correlations $0.109768,0.077067,0.003584,0.001718$ on the four-dimensional centered support. The last two are finite-sample tails, not additional population modes; no exact finite-sample rank claim is made. Output ends with `all B1 checks passed`. The existing `/home/ubuntu/code/CoherentModes/checks/cm_check.py` was also run and ended with `all checks passed`.

**Additional review checks.** The script asserts the B.1 signed vertex masses $(3,-2)$, B.6's first-order token Markov identity and equality to one-hot token CCA, every variant of B.11 (including zero-update exclusion), and the Weyl/Wedin and actual ridge-score smoothing bounds in B.13 on a positive finite joint distribution with a known affine defect. Every numerical value reported here is covered by an assertion, allowing explicit floating-point tolerances rather than asserting a particular roundoff error.

**Mess3 diagnostic (not population CCA).** With $x=0.05$, $\alpha=0.85$, identity belief activations $h=b$, $\beta=5$, and seed **20261108**, the script filters 60,000 steps from the uniform prior, discards the first 1,000, and keeps every twentieth belief, yielding 2,950 beliefs. This is a dependent long-run approximation, not 2,950 independent stationary documents. It selects 200 reference beliefs and sums exactly over the intervening one- or two-token words to evaluate conditional sphere-kernel means at those references. In that **unwhitened conditional-mean feature matrix** it finds:

| target | third/first singular value | relative affine residual |
|---|---:|---:|
| carried $b_{t+1}\mid b_{t-1}$ | 0.0150055709 | 0.0140056748 |
| read-style $b_t\mid b_{t-1}$ | 0.0301274417 | 0.0402949768 |
| two-token indicator target | centered rank 2 | $<2\times10^{-14}$ |

The carried residual is about 1.40% and the read-style residual about 4.03% for this fixed seed/reference set; reference selection changes these diagnostics. The exact non-affineness conclusion follows from B.12, not from treating this simulation as a population proof. Conversely, the small third feature singular value does **not** certify a small $\epsilon$ in B.13: it lacks the target $L^2$ whitening and an operator-norm bound over the full feature space. B.14 proves the stronger obstruction: the full whitened operator is noncompact, has infinite rank, and its carried defect is at least $0.3251850435\ldots$. No arbitrary trained next-layer update is simulated.

**Noncompactness check.** Exact rational arithmetic verifies B.14’s exclusive cutoff $54/91$, the safer target region $b_a>2/3$, its preimage threshold $2/17$, and the probability bounds $153/800$ and $67677/640000$. Their square roots are the rigorously established variational correlation lower bounds **0.4373213921** (read-style) and **0.3251850435** (carried), stronger than bounds 0.406 and 0.289 obtained from the earlier coarse-grid diagnostic. Matrix-level checks at simplex vertices and on a deterministic slice independently verify the cutoff and probability inequalities. The proof of infinite rank remains B.14, not a finite numerical rank estimate.

## Discrepancies with THEORY.md

No edits to `THEORY.md` were made.

1. **§6.3's token Hankel equality needs linear minimality.** “Equal under minimality and sufficiently long windows” should specify observable and reachable linear dimension $k$, not only minimality among nonnegative $k$-state HMM realizations. Hsu–Kakade–Zhang's explicit rank condition is stronger than bare hidden-state minimality.
2. **§6.3's simplex dimension is an ambient dimension.** The ambient belief simplex has affine dimension $k-1$; the stationary reachable belief distribution may have smaller affine span. Recovery and rank should use $d_b=\operatorname{rank}\operatorname{Cov}(b)$ until full affine support is established.
3. **§6.2's low-temperature calibration wording.** The sphere kernel's low-temperature limit is linear CCA on the chosen **normalized** views (with kernel/ridge scaling handled), not in general raw-activation CCA. A raw-linear activation calibration is a separate declared variant. Linear belief-sufficiency before normalization does not alone give exact linear readability from the normalized activation.
4. **The boundary matters for §6.3 and §11's belief comparison.** With the literal past positions $<t$, belief-sufficiency at the current site does not ensure access to its current belief. In the one-lag convention above the recovered state is $b_{t-1}$; comparing it with $b_t$ requires another assumption or an explicitly different alignment.
5. **§6.3 and §11 correctly label activation rank and readability as hypotheses.** Even strengthening them to exact belief-sufficiency plus belief-Markov would not make them theorems: B.10 is a counterexample. ATM, feature completeness, and observability yield a valid sufficient regime. This is a qualification of a proposed strengthening, not a contradiction of the existing caution.
6. **Effective context rank is not exact population rank.** Positive ridge, finite windows, held-out thresholds, and sampling remain part of §6.3's definition. Statements equating it with $k-1$ or causal-state count minus one require the extra threshold/consistency conditions in B.7. Finite causal-state count additionally requires affine independence of future laws.
7. **Zero read updates alter the evaluated population.** In the previous-token-copy example, a sphere kernel on the update excludes mass $1/V$ under the shared zero policy. Using $h^{L+1}$ as allowed in §6.3 avoids this issue; the exact conditional-on-retention statement is in B.11.

8. **The Mess3 qualification of §6.3/§11 is stronger than failure of exact ATM.** B.14 proves that even an exactly linear belief-sufficient stationary Mess3 layer has infinite zero-ridge population carried/read-style rank under universal sphere kernels, with all variational canonical levels bounded below. Thus $r_{\mathrm{ctx}}\approx k-1$ can only be a declared **ridge-, threshold-, and window-dependent effective-rank hypothesis**, not an intrinsic population rank limit in that regime. An alternative is an explicitly non-universal/raw-linear kernel on belief-linear activations, or the token-future benchmark; a low-temperature sphere kernel on normalized activations is not automatically that raw-linear variant. This does not refute the existing scale-dependent empirical hypothesis, but it rules out the proposed perfect-belief rationale for a universal zero-ridge $k-1$ cutoff. Fukumizu–Bach–Gretton compact-limit consistency cannot supply that missing limit. Neither this document nor its checks edits the theory or experiment protocol.

## Sources

- Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel and Paul M. Riechers (2024). **Transformers Represent Belief State Geometry in their Residual Stream.** *Advances in Neural Information Processing Systems* 37, 75012–75034; [arXiv:2405.15943](https://arxiv.org/abs/2405.15943). Empirical motivation and mixed-state geometry, not a theorem that arbitrary trained layers satisfy PA/ATM/BO.
- Cosma Rohilla Shalizi and James P. Crutchfield (2001). **Computational Mechanics: Pattern and Prediction, Structure and Simplicity.** *Journal of Statistical Physics* 104, 817–879; [arXiv:cond-mat/9907176](https://arxiv.org/abs/cond-mat/9907176). Corollary 1 and Lemma 7 for sufficient statistics and refinement; Theorems 1–3 for prescience, statistical-complexity minimality, and uniqueness. The sufficient-statistic argument used here is also proved directly in B.7.
- Daniel Hsu, Sham M. Kakade and Tong Zhang (2012). **A Spectral Algorithm for Learning Hidden Markov Models.** *Journal of Computer and System Sciences* 78(5), 1460–1480; [arXiv:0811.4413](https://arxiv.org/abs/0811.4413). §2.3 Condition 1 and Lemma 2, including the factorization $P_{2,1}=OT\operatorname{diag}(\pi)O^\top$ under their column-transition convention. The more general finite-window result needed here is proved in B.5.
- Kenji Fukumizu, Francis R. Bach and Arthur Gretton (2007). **Statistical Consistency of Kernel Canonical Correlation Analysis.** *Journal of Machine Learning Research* 8(14), 361–383; [paper and PDF](https://www.jmlr.org/papers/v8/fukumizu07a.html). Lemmas 6–7 and Eq. (9) for operator-norm consistency and the stochastic error term; Theorems 1–2 only within their largest-pair/source-condition scope, as specified above.
- Standard spectral facts used in B.4: polar decomposition, the spectral theorem, and continuity of an isolated singular subspace; see, e.g., G. W. Stewart and Ji-guang Sun (1990), *Matrix Perturbation Theory*, for the finite-dimensional singular-subspace result. The compact finite-rank reduction and the strong-to-norm convergence step are given explicitly in the proof.

- Ingo Steinwart and Andreas Christmann (2008). **Support Vector Machines.** Springer, §4.6 (universal kernels and Taylor-kernel criteria). The sphere specialization is proved in B.4.
- Per-Åke Wedin (1972). **Perturbation bounds in connection with singular value decomposition.** *BIT Numerical Mathematics* 12, 99–111. B.13 proves the residual-form subspace estimate used here directly, alongside Weyl’s singular-value inequality.

- Roger A. Horn and Charles R. Johnson (2013). **Matrix Analysis**, second edition. Cambridge University Press, Chapter 8, for Perron–Frobenius and convergence of normalized powers of a strictly positive matrix; the positivity and invertibility needed in B.12 are verified there directly.
