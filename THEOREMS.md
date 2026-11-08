# Coherent modes: theorems, bounds and ideal-case guarantees

Status: v1.0, 2026-11-07. Companion to `THEORY.md` (v1.5), which defines every object used here and is not modified by this document. Proofs were drafted by GPT-6 Astra agents, reviewed independently by GPT-6.1 Sol and Claude Opus 5.5 agents, and revised; reviews are in `reviews/`. Numerical checks asserting the finite-dimensional identities and the planted-regime recovery statements are in `checks/` (see Section 0.4). A subset of the exact identities is formalized in Lean 4 with Mathlib in `lean/` (22 theorems, no `sorry`).

## 0. How to read this document

### 0.1 What it is for

`THEORY.md` makes one move: every measurable property of a site (one place where the network has an activation) defines a kernel on sites, hence a symmetric doubly stochastic operator, hence a spectrum and an SVD. Everything downstream is linear algebra on one shared Hilbert space. Each kernel is a declared choice (which view, which temperature, the bistochastic balancing), and the operators are what is studied. The point of this document is to show what that move buys when it is pushed as far as it will go: which guarantees come for free from the construction's choices, and what the construction recovers *exactly* in idealized regimes where the answer is known.

Part A is the first kind of result. It contains no assumptions about trained networks: existence and uniqueness of the normalization, exactness of every expansion, perturbation bounds for every reported object, and finite-sample rates and arithmetic costs. These results hold for every network, every layer, every recording. Part B is the second kind. Under strong, named idealizations (exact angular clusters, belief-sufficient layers, exactly readable blocks) it shows that the construction's output coincides with a known object: the centroid dictionary that an ideal one-sparse autoencoder would learn, the affine image of the belief simplex, the causal-state partition of computational mechanics, the linear CKA statistic, the sensitivity-weighted stitching predictor. `SPARSITY.md` is the companion for readers who want to know in what sense discrete and sparse structure is recovered, and certified, when it is there.

### 0.2 Epistemic status, stated once

Every statement below is a theorem with a complete proof or a precise reduction to a cited result whose hypotheses are verified in the text. Where a natural-sounding statement turned out to be false, the false version is recorded and the strongest true neighbour is proved; the counterexamples are part of the document. Three things the reader should keep in mind:

- The idealizations of Part B are **named assumptions**, not claims about trained networks. Whether a given layer satisfies them is measured by the diagnostics of `THEORY.md` Sections 4.3 and 11, and the results here say what is recovered when they hold and what the error is when they nearly hold.
- Coined terms ("sparse concept cluster", "ideal dictionary regime", "belief-sufficient layer") are defined where they are first used in Part B (B.20, B.20, B.1 respectively). The name is a mnemonic; the definition is the content. `CLAUDE.md` instructs agents working on the project not to adopt this vocabulary as the working ontology, and that instruction is deliberate.
- Every theorem ends with a **Tractability and sampling** remark. These are the part of the document meant to be used, not admired. The recurring price is $e^{\beta}$: resolution costs constants that grow exponentially in the temperature, and the remarks say so wherever it bites.

### 0.3 Headline results

| | statement | where |
|---|---|---|
| exactness | The bistochastic normalization exists and is unique; spectra lie in $[0,1]$; products of layer operators are doubly stochastic with top singular value exactly one; every expansion of activations, units and gradients in the mode basis is exact with a computable tail | A.1–A.7 |
| limits | At $\beta \to 0$ the centered operator is the centered Gram matrix of the row-normalized activations with an explicit $O(\beta^2)$ remainder, so its eigenvectors are PCA scores and the alignment statistic is linear CKA (centered kernel alignment), given nonzero centered Grams and a spectral gap respectively; at $\beta \to \infty$ it is the projector onto duplicate classes with rate $2(N-1)e^{-\beta\gamma}$ | A.9–A.11, B.42 |
| stability | Isolated blocks move by $\|E\|/\delta$ (Davis–Kahan with constant 1, Wedin); the Sinkhorn normalization is Lipschitz with explicit constants, $\min(e^{4\beta}/N, e^{\beta}/\sqrt2)$ in Frobenius norm and the sharp $\sqrt{N/2}$ in log-kernel units; every reported statistic is continuous; individual vectors inside a degenerate block are not identifiable, while a block separated from its complement by a gap is stable | A.21–A.28 |
| sampling | Empirical blocks converge to population blocks at rate $O(S_{\max}^2 e^{3\beta}\sqrt{\log(1/\eta)/N}/\delta)$, with $S_{\max}$ the Sinkhorn conditioning (the unconditional worst case carries $e^{6\beta}$; vacuity thresholds stated); kernel CCA is consistent with ridge $\rho_N \gg N^{-1/3}$; a $k$-state sufficient summary forces centered predictive rank $\le k-1$; the cross-layer operator $Q$ is estimated at $N^{-1/4}$ unconditionally and at root-$N$ for finitely supported views; dependent positions are handled by document-level sampling | A.43–A.48, A.55–A.56 |
| cost | Dense cost is $O(N^2 d)$ formation plus $O(N^2)$ per Sinkhorn sweep with contraction factor $\le \tfrac12(1+\tanh\beta)$; Nyström and random-feature surrogates have certified error | A.51–A.54 |
| belief states | On a belief-sufficient, belief-Markov layer with affine target means, the predictive-state coordinates are an affine image of the belief simplex and separate exactly the causal states of computational mechanics; counterexamples show each hypothesis is needed, and a perfectly belief-sufficient Mess3 layer has infinite population context rank under a universal kernel, so finite $r_{\text{ctx}}$ is a ridge-and-window-dependent effective rank | B.3–B.14 |
| dictionaries | In the exact angular-cluster regime the top block is spanned by cluster indicators with explicit gap $1 - 2e^{-\beta\Delta}R$; a localizing rotation recovers them; the loadings are the centroid dictionary; every global optimum of a hard one-sparse autoencoder of the same width is that dictionary | B.21–B.28 |
| superposition | Under orthonormal (or explicitly separated) features with positive, balanced coefficients, sparse sums form angular clusters by active set; singleton clusters recover the feature directions; overlapping codes are not identified | B.29–B.30 |
| comparison | $P$ determines the normalized Gram exactly, so comparison across models needs no fitting; stitching damage has an exact local KL identity, and the Section 8 sensitivity-weighted-unreadability predictor is exact under named commutation and site-homogeneous Fisher hypotheses and fails without them (a zero-residual self-stitch with predictor 39.4) | B.41–B.48 |
| sparsity | The bistochastic operator has an $O(N/\epsilon^2)$-edge spectral sparsifier preserving cuts and gaps; $k$ eigenvalues near one certify $k$ low-conductance sets; SEBA (sparse eigenbasis approximation) recovers indicators exactly from a nearby start, and can fail at its published default threshold (reproduced numerical example) | `SPARSITY.md`, S.2–S.8 |

### 0.4 Checks and formalization

Run from the repo root:

```
uv run --with numpy python -I checks/cm_check.py
uv run --with numpy python -I checks/theorems_check.py
```

The second script asserts the finite-dimensional identities and the planted-regime recovery statements of this document and ends with `all theorem checks passed`. The Lean development in `lean/` (`cd lean && lake exe cache get && lake build`) formalizes the doubly stochastic closure, the two-stage dissipation identity, the eigenvalue range of a PSD doubly stochastic matrix, the exact expansion and tail identity, the gradient decomposition, and block-pressure rotation invariance; `lean/README.md` maps theorem names to labels here.

### 0.5 Notation

As in `THEORY.md`: sites $X$, $|X| = N$; $\mathcal H = \mathbb R^N$ with the counting-measure inner product; view $\phi$, normalized $\hat\phi$; kernel $K_\beta[x,y] = \exp(\beta\langle\hat\phi(x),\hat\phi(y)\rangle)$; Sinkhorn vector $v$, $P = D_v K D_v$; centering matrix $C = I - \mathbf 1\mathbf 1^\top/N$; centered $\tilde P = CPC$; eigenpairs $(\lambda_i, \psi_i)$; cross-layer $Q = P^{L'}P^{L}$ with singular triples $(\sigma_i, u_i, v_i)$; loading $a_F(\psi) = F^\top\psi$; block $B$ with projector $\Pi_B$ and gap $\delta$. Labels: A.1–A.11 exactness, A.20–A.31 stability, A.40–A.56 sampling and cost; B.1–B.14 belief states, B.20–B.34 dictionaries, B.40–B.53 comparison and stitching. Each cluster ends with its own "Discrepancies with THEORY.md" list; the consolidated list for the owner is `reviews/theorems-discrepancies.md`. `SPARSITY.md` uses S.1–S.10.

---

# Part A. Guarantees that follow from the construction

No assumption about trained networks enters Part A. Everything here holds for every finite site set, every view, every layer.
## Part A, cluster 1 — Exactness and structure of the spine

This cluster concerns finite, fixed data. It introduces no fitted representation, semantic labels, sparsity assumption, or additional metric. A theorem about these finite matrices is not a population-consistency theorem. In particular, none of the algebraic guarantees below requires independent sites; independence enters only where expressly assumed for random probes.

### Conventions and computational accounting

Let $X$ be the common site set, $N=|X|\ge 1$, $\mathcal H=\mathbb R^N$ with counting measure, $e=\mathbf 1/\sqrt N$, and $C=I-ee^\top$. Write $\mathcal H_0=e^\perp$. Vector norms are Euclidean, $\|\cdot\|_2$ on matrices is the operator norm, and $\|\cdot\|_F$ is the Frobenius norm. An entrywise nonnegative matrix is **bistochastic** if every row and column sums to one. "Positive entries" and "positive semidefinite (PSD)" are distinct conditions.

A view has nonzero rows $\phi(x)\in\mathbb R^d$, normalized to $z_x=\hat\phi(x)$; set $Z=\hat\Phi$, $\Gamma=ZZ^\top$, and $K_\beta[x,y]=\exp(\beta \Gamma_{xy})$, $\beta\ge0$. The declared common-site exclusion policy for zero rows is required before these definitions. The scaling vector is $v$, while a right singular function of $Q$ is $v_i$. The Gram matrix is $\Gamma$ (not the gradient $G$), centered Grams use $\Gamma^c$ or $\Gamma_c$, and the all-ones matrix is written $\mathbf1\mathbf1^\top$ (reserving $J_x$ for Jacobians). Feature matrices $H^L$ retain THEORY's activation meaning. The target set $B$ is local to A.5; the spectral block $B$ is local to A.7. Width $d$, degree vectors $\mathbf d_i$, slot count $n_{\mathrm{use}}$, nonzero count $n_{\mathrm{nz}}$, block dimension $r_B$, and sketch $\zeta_x$ are distinct.

The following cost convention is used in the remarks. Building a dense sphere kernel costs $O(N^2d)$ arithmetic and $O(N^2)$ storage. A kernel or normalized-operator matvec costs $O(N^2)$ dense, or $O(n_{\mathrm{nz}})$ for an explicitly stored sparse kernel with $n_{\mathrm{nz}}=\operatorname{nnz}(K)$. One alternating row/column Sinkhorn sweep uses a constant number of matvecs; one damped symmetric step uses one. Its iteration count is **not** universally a few dozen: it depends on the kernel, tolerance, and conditioning of the scaling problem. The implemented damped step has local linearization rate at most $1/2$ for PSD kernels and global Hilbert-metric contraction factor at most $(1+\tanh\beta)/2$ for a sphere kernel, proved in A.1. The global bound deteriorates at large $\beta$; the local bound explains rapid final convergence without a spectral-gap requirement. Knight (2008, Theorem 4.4 in the author's text) concerns the different, undamped symmetric alternating iteration, initialized at the all-ones vector. Its transverse two-step rate is governed by $\max_{i\ne0}|\lambda_i(P)|^2$ under full indecomposability, which strict positivity guarantees. These algorithm-specific rates are not interchangeable.

Lanczos with $\ell$ iterations for the top $r\le\ell$ eigenpairs costs $O(\ell N^2+N\ell^2)$ dense with full reorthogonalization, or $O(\ell n_{\mathrm{nz}}+N\ell^2)$ sparse, plus the small projected eigensolve. Working-vector storage is $O(N\ell)$. The number $\ell$ depends on gaps, clustering, accuracy, and restarts; $O(rN^2)$ is not an unconditional total-cost guarantee. A matrix-free SVD uses matvecs with $Q$ and $Q^\top$ and has analogous accounting. Sparsification changes the estimand and can destroy PSD or scalability; dense-kernel results are not automatically graph results.

### Normalization and diffusion

**Theorem A.1 (unique symmetric scaling and damped convergence).** Assume $K\in\mathbb R^{N\times N}$ is symmetric with strictly positive entries. There is a unique $v\in(0,\infty)^N$ satisfying

$$v_i(Kv)_i=1\quad(1\le i\le N).$$

Consequently $P=D_vKD_v$ is symmetric, strictly positive, and bistochastic. The same existence and uniqueness conclusion holds for symmetric entrywise nonnegative $K$ with strictly positive diagonal, without an irreducibility assumption. Multiplying $K$ by a scalar $a>0$ replaces $v$ by $a^{-1/2}v$ and leaves $P$ unchanged.

For strictly positive $K$, the implemented damped iteration

$$w_{k+1}=g(w_k):=\sqrt{w_k/(Kw_k)},\qquad w_0>0,$$

converges to $v$. Define $\operatorname{osc}(h)=\max_i h_i-\min_i h_i$ and Hilbert's projective distance $d_{\mathrm H}(a,b)=\operatorname{osc}(\log a-\log b)$ on positive vectors (zero distance means proportional vectors). With

$$\Delta_K=\max_{i,j,k,l}\log\frac{K_{ij}K_{kl}}{K_{il}K_{kj}},\qquad q_K=\frac12\left(1+\tanh\frac{\Delta_K}{4}\right)<1,$$

one has

$$d_{\mathrm H}(g(a),g(b))\le q_Kd_{\mathrm H}(a,b),\qquad
\|\log w_{k+1}-\log v\|_\infty\le\frac12q_K^k d_{\mathrm H}(w_0,v).$$

For sphere kernels $\Delta_K\le4\beta$, hence $q_K\le(1+\tanh\beta)/2$. In log coordinates the local derivative at the solution is $(I-P)/2$. Its spectral radius is $\rho_{\mathrm{damp}}=\max_i(1-\lambda_i(P))/2$, at most $1/2$ when $K$ is PSD. Precisely, for every $\epsilon>0$ the Euclidean log-error eventually contracts by at most $\rho_{\mathrm{damp}}+\epsilon$ (unless convergence has already occurred). These iteration claims require strictly positive entries; the existence extension to positive-diagonal sparse matrices alone does not establish them.

*Proof.* Define on $\mathbb R^N$

$$F(x)=\frac12\sum_{i,j}K_{ij}e^{x_i+x_j}-\sum_i x_i.$$

For a direction $h$,

$$D^2F(x)[h,h]=\frac12\sum_{i,j}K_{ij}e^{x_i+x_j}(h_i+h_j)^2.$$

The diagonal terms are $2K_{ii}e^{2x_i}h_i^2$, so the Hessian is positive definite. Moreover,

$$F(x)\ge\sum_i\left(\frac12K_{ii}e^{2x_i}-x_i\right).$$

Each summand is bounded below and tends to $+\infty$ as its argument tends to either infinity. Hence $F$ is coercive: its sublevel sets are bounded and closed. It attains a minimum, uniquely by strict convexity. Symmetry of $K$ gives

$$\partial_iF(x)=e^{x_i}\sum_jK_{ij}e^{x_j}-1.$$

The minimizer therefore gives the required $v_i=e^{x_i}$. Conversely every positive solution is a critical point of this strictly convex function, proving uniqueness. Row sums of $P$ are one by the equation; symmetry gives column sums. Positive diagonal scaling preserves signs and symmetry. Finally, substituting $a^{-1/2}v$ in the equation for $aK$ and using uniqueness proves scale invariance.

For the iteration, write $\mathscr L(x)=\log(Ke^x)$ and $\mathscr T(x)=(x-\mathscr L(x))/2$, componentwise, so $\mathscr T(\log w)=\log g(w)$. The derivative of $\mathscr L$ is the positive row-stochastic matrix

$$R(x)_{ij}=\frac{K_{ij}e^{x_j}}{\sum_lK_{il}e^{x_l}}.$$

We prove its oscillation contraction explicitly. For two probability rows $p,q$, let $a=\min_j p_j/q_j$, $b=\max_j p_j/q_j$. If $p=q$ their total variation is zero. Otherwise $a<1<b$. The chord of the convex function $|r-1|$ on $[a,b]$, averaged with weights $q_j$ and using $\sum_jq_j(p_j/q_j)=1$, gives

$$\frac12\sum_j|p_j-q_j|\le\frac{(b-1)(1-a)}{b-a}.$$

Put $\kappa=b/a>1$. The right side equals $(\kappa+1-\kappa a-a^{-1})/(\kappa-1)$, at most $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ by $\kappa a+a^{-1}\ge2\sqrt\kappa$. For any two rows of $R(x)$, their likelihood-ratio range obeys $b/a\le e^{\Delta_K}$, by canceling row normalizers in the cross-ratio. Thus their total variation is at most $\tau_K=\tanh(\Delta_K/4)$, uniformly in $x$. For any vector $h$, shifting its minimum to zero shows $|(p-q)^\top h|\le\operatorname{TV}(p,q)\operatorname{osc}(h)$. Consequently

$$\operatorname{osc}(R(x)h)\le\tau_K\operatorname{osc}(h).$$

Integrate the derivative along the line from $y$ to $x$ and use subadditivity of oscillation to obtain $\operatorname{osc}(\mathscr L(x)-\mathscr L(y))\le\tau_K\operatorname{osc}(x-y)$. Hence $\operatorname{osc}(\mathscr T(x)-\mathscr T(y))\le q_K\operatorname{osc}(x-y)$, proving the projective contraction without invoking an external convergence theorem. This is the finite positive-matrix Birkhoff–Hopf estimate; see Lemmens–Nussbaum (2012, Appendix A) for its general setting.

Actual scaling, not just its projective class, also converges. The map $\mathscr T$ is invariant under $x\mapsto x+c\mathbf1$ and is 1-Lipschitz in the sup norm, because $\mathscr L$ is 1-Lipschitz there: $\|x-y\|_\infty\le a$ implies $e^{-a}Ke^y\le Ke^x\le e^aKe^y$ entrywise. Shift $x-\log v$ by its midrange so its sup norm is half its oscillation; the shift leaves $\mathscr T(x)$ unchanged. This gives $\|\mathscr T(x)-\log v\|_\infty\le\tfrac12\operatorname{osc}(x-\log v)$, which combined with the projective estimate proves the displayed absolute log-error bound. For $K_\beta$, each log cross-ratio is at most $4\beta$, proving the advertised global rate.

At $x_*=\log v$, the scaling equation gives $R(x_*)=P$. Therefore $D\mathscr T(x_*)=(I-P)/2$; in the original variables $Dg(v)=\tfrac12(I-D_vPD_v^{-1})$. Symmetry makes the log-coordinate derivative's Euclidean norm equal to $\max_i(1-\lambda_i(P))/2$, since $\lambda_i(P)\le1$ by A.2's contraction argument. PSD makes this at most $1/2$. Continuity of the derivative and the mean-value integral give the stated eventual rate with any positive slack. The constant direction has derivative zero, consistent with $g(cw)=g(w)$. ∎


**What it buys.** Every dense view has one canonical normalization against the same counting measure, not merely a separately chosen stationary coordinate system. No learned parameters or semantic supervision enter this operation.

**Tractability and sampling.** One damped step costs $O(N^2)$ dense or $O(n_{\mathrm{nz}})$ sparse; the global strict-positivity proof does not automatically extend to sparse kernels. For PSD sphere kernels the eventual local log-error rate is at most $1/2$, independently of the top spectral gap, while the explicit global bound worsens at high temperature. Thus rapid final convergence is justified, but a uniform total iteration count from arbitrary starts is not. A log-scaling error at most $a$ gives row sums within $[e^{-2a},e^{2a}]$ of one by entrywise comparison with the exact scaling. This result is exact for every finite $N$ and gives no population sample-size guarantee. Numerical implementations must monitor residuals and avoid overflow.

For comparison, the undamped iteration is $f(w)=1/(Kw)$; its derivative at $v$ is $-D_vPD_v^{-1}$, and that of $f^2$ is similar to $P^2$. It can alternate between scalar multiples of $v$. Knight (2008, Theorem 4.4 **in the author's text**) assumes the initial vector $x_0=\mathbf1$ (denoted $e$ in that paper, not our normalized $e$) and full indecomposability; it gives convergence to the scaling line and a transverse two-step rate governed by $\max_{i\ne0}|\lambda_i(P)|^2$. The derivative and local-remainder argument in Lemma 4.3 support eventual local rates with any positive slack above that factor; we do not equate this weaker formulation with the theorem's exact inequality. These are properties of the undamped alternating algorithm, not the implemented damped algorithm just proved.

Sinkhorn (1964) and Sinkhorn–Knopp (1967) are the foundational scaling references. Marshall–Olkin (1968) give an extremal-problem approach; Csima–Datta (1972) address symmetric nonnegative scaling. The existence and convergence arguments above are supplied in full, rather than importing their broader theorems. Marshall–Coifman (2019) place bistochastic kernel normalization in its manifold-learning setting; no manifold assumptions are used here.

**Theorem A.2 (PSD Markov spectrum, diffusion, and the meaning of time).** Let $K$ be symmetric and entrywise nonnegative, let a positive scaling $v$ exist, and let $P=D_vKD_v$ be bistochastic.

1. $P$ is a reversible Markov kernel with uniform stationary measure $\pi_x=1/N$. Its eigenvalues lie in $[-1,1]$. If $K$ is PSD, they lie in $[0,1]$.
2. $Pe=e$. The multiplicity of eigenvalue one is the number of connected components of the undirected graph with edges $K_{xy}>0$. Thus eigenvalue one is simple if and only if $K$ is irreducible (this equivalence concerns a nonnegative symmetric matrix with a scaling already in hand). In particular it is simple for the strictly positive kernels of A.1.
3. The sphere kernel $K_\beta$ is PSD for every $\beta\ge0$, so all the preceding nonnegative-spectrum conclusions apply to the spine.
4. For integers $t\ge0$, $P^t$ is symmetric, bistochastic, and reversible; if $P$ is PSD it is also PSD, with eigenpairs $(\lambda_i^t,\psi_i)$. For $t=0$, use $\lambda_i^0=1$, including zero eigenvalues. In an orthonormal eigenbasis with $\psi_0=e$,

$$\|P^t(\delta_x-\delta_y)\|^2=\sum_{i=1}^{N-1}\lambda_i^{2t}\big(\psi_i(x)-\psi_i(y)\big)^2.$$

This is the counting-measure diffusion distance squared. The Coifman–Lafon (2006) convention that weights squared transition-probability differences by $1/\pi_z$ gives $N$ times this quantity. For irreducible PSD $P$ and $N\ge2$,

$$\|P^t-ee^\top\|_2=\lambda_1^t\longrightarrow0\quad(t\to\infty).$$

For real $t>0$, spectral powers of a PSD $P$ retain symmetry, PSD, contraction, the constant eigenvector, and the displayed distance identity, but are not asserted to have nonnegative entries or to be Markov kernels.

*Proof.* Symmetry and row sums give detailed balance $\pi_xP_{xy}=\pi_yP_{yx}$ and stationarity. For any $f$, Jensen's inequality followed by the column-sum condition gives

$$\|Pf\|^2=\sum_x\left(\sum_yP_{xy}f_y\right)^2\le\sum_{x,y}P_{xy}f_y^2=\|f\|^2.$$

The real spectral theorem implies $|\lambda_i|\le1$. If $K$ is PSD, $f^\top Pf=(D_vf)^\top K(D_vf)\ge0$, proving nonnegativity. Also

$$f^\top(I-P)f=\frac12\sum_{x,y}P_{xy}(f_x-f_y)^2.$$

Thus $Pf=f$ if and only if $f$ is constant on every graph component: necessity follows by setting the nonnegative sum to zero, and sufficiency follows from no transitions between components. Scaling preserves the graph. This proves the multiplicity assertion, including the reducible direction.

For the sphere kernel, the tensor-power feature identity yields, for every real $c\in\mathbb R^N$,

$$c^\top K_\beta c=\sum_{k=0}^{\infty}\frac{\beta^k}{k!}\left\|\sum_x c_x z_x^{\otimes k}\right\|^2\ge0.$$

The $k=0$ tensor is the scalar one. The interchange with the finite site sum is legitimate because $|z_x^\top z_y|\le1$ and the exponential series converges absolutely. Integer matrix powers preserve nonnegative entries and row and column sums; their spectral properties follow by diagonalization. Expanding $\delta_x-\delta_y$ in the same orthonormal basis proves the distance identity by Parseval; its constant coefficient vanishes. Symmetry identifies the column difference in that expression with a difference of transition-probability rows. Dividing each squared entry by $\pi_z=1/N$ explains the alternate factor. In the irreducible PSD case $1>\lambda_1\ge\cdots\ge0$, so diagonalizing $P^t-ee^\top$ gives the convergence and norm formula. The same diagonalization proves the asserted properties of real powers, without a positivity assertion. ∎

**What it buys.** Integer powers describe an actual diffusion on the observed sites; irreducibility identifies its unique constant stationary mode. Counting measure fixes the normalization constants in the embedding distance, without establishing a separation between useful structure and smoothing.

**Tractability and sampling.** Eigenpairs use the Lanczos accounting above. Applying an integer power by repeated matvecs costs $t$ matvecs; a retained spectral expansion costs $O(Nr)$ per application after eigenpairs are available but omits a spectral tail. The conclusions require only the observed finite sites. Population mixing or geometric recovery needs additional assumptions; a sparse graph can lose PSD, and a small spectral gap makes slow modes expensive to resolve.

### Shared-measure composition and the coherent-set relaxation

**Proposition A.3 (composition without stationary-measure leakage).** Let $P^{L_1},\ldots,P^{L_m}$ be any finite nonempty collection of $N\times N$ bistochastic matrices, not necessarily symmetric or PSD. Then

$$Q=P^{L_m}\cdots P^{L_1}$$

is bistochastic, $Qe=Q^\top e=e$, $\|Q\|_2=1$, and all singular values lie in $[0,1]$. The pair $(e,e)$ is a singular pair with singular value one. Moreover $\tilde Q=CQC=Q-ee^\top$ acts on $\mathcal H_0$ and is the nonconstant block of $Q$ in the orthogonal decomposition $\mathcal H=\operatorname{span}(e)\oplus\mathcal H_0$. If the first-applied factor $P^{L_1}$ is strictly positive, the singular value one is simple.

*Proof.* Products preserve nonnegative entries, and multiplying their row-sum and column-sum identities proves bistochasticity. The Jensen argument in A.2 does not require symmetry, so each factor, and hence $Q$, is a contraction. Since $Qe=e$, its norm is exactly one; since also $Q^\top e=e$, this is a left/right singular pair. Both constant and centered subspaces reduce $Q$, giving $CQC=Q-ee^\top$ and its block decomposition. For a strictly positive bistochastic matrix, equality in every row's Jensen inequality requires all entries of $f$ to be equal. Thus a nonconstant vector has strict norm loss in the first positive factor, which subsequent contractions cannot undo. The only norm-preserving subspace is the constant line, proving simplicity. ∎

**What it buys.** Changing layers cannot fabricate a loss of the trivial singular value through a mismatch of stationary measures. All nontrivial coherence is in a single shared centered site space.

**Tractability and sampling.** Never form a dense product: one $Q$-matvec uses $m$ factor matvecs, and $Q^\top$ reverses their order and transposes them. Top singular pairs use these two routines; iteration counts depend on singular gaps. This is an exact finite-sample result with no independence or minimum-$N$ condition beyond the desired number of nonconstant modes being at most $N-1$.

**Theorem A.4 (precise degree-normalization contrast).** Assume $K_1,K_2$ are symmetric matrices with strictly positive entries. Set

$$\mathbf d_i=K_i\mathbf1,\qquad S_i=D_{\mathbf d_i}^{-1/2}K_iD_{\mathbf d_i}^{-1/2},\qquad q_i=\frac{\sqrt{\mathbf d_i}}{\|\sqrt{\mathbf d_i}\|}.$$

Then $\|S_i\|_2=1$ and its only norm-preserving directions are multiples of $q_i$. Consequently

$$\|S_2S_1\|_2=1\quad\Longleftrightarrow\quad \mathbf d_1=c\,\mathbf d_2\text{ for some }c>0.$$

If the degree vectors are not proportional, the product's top singular value is strictly below one. PSD is neither needed nor, on its own, sufficient for this conclusion; the proof uses strict positivity, which supplies irreducibility and aperiodicity.

*Proof.* $S_iq_i=q_i$. To analyze its other real eigenvalues, if $S_if=\lambda f$, put $h=D_{\mathbf d_i}^{-1/2}f$. Then $R_ih=\lambda h$, where $R_i=D_{\mathbf d_i}^{-1}K_i$ is a strictly positive row-stochastic matrix. Choose an index where $|h_x|$ is maximal. The triangle inequality gives $|\lambda|\le1$. If $|\lambda|=1$, equality in this weighted average with all weights positive forces every $h_y$ to have the same magnitude and sign. Thus $h$ is constant and $\lambda=1$. All other eigenvalues of the symmetric $S_i$ therefore have modulus strictly less than one. Orthogonal diagonalization proves both its norm and the claimed norm-preserving line.

The unit sphere is compact, so if $\|S_2S_1\|_2=1$ there is a unit $f$ attaining it. Both inequalities in

$$1=\|S_2S_1f\|\le\|S_1f\|\le\|f\|=1$$

must be equalities. Thus $f\in\operatorname{span}(q_1)$, $S_1f=f$, and $f\in\operatorname{span}(q_2)$. The two positive unit vectors agree, which is equivalent to proportional degree vectors. Conversely, proportional degrees give a common unit eigenvector with eigenvalue one; the product fixes it and is a contraction, hence has norm one. ∎

**What it buys.** Bistochastic normalization protects the trivial constant singular pair exactly; conventional symmetric normalization can lose a unit top singular value solely through stationary-vector mismatch. This does not remove density effects from the nontrivial spectrum; the continuum density dependence discussed by Marshall–Coifman (2019) remains a separate issue.

**Tractability and sampling.** Degrees cost one kernel matvec; proportionality can be checked through the angle between $\sqrt{\mathbf d_1}$ and $\sqrt{\mathbf d_2}$. Estimating the product norm has the same matrix-free SVD costs as A.3. No sampling assumptions are needed. Strictness need not be quantitatively large, and the theorem supplies no universal gap below one. Reducible or periodic sparse kernels need a separate norm-preserving-subspace analysis; nonproportional degrees alone then do not suffice.

**Theorem A.5 (two-stage dissipation and balanced-set variational meaning).** Assume $N\ge2$, let $P^L,P^{L'}$ be bistochastic, and put $Q=P^{L'}P^L$, $T=Q|_{\mathcal H_0}$. Write the singular values of $T$ as $\sigma_1\ge\cdots\ge\sigma_{N-1}\ge0$.

For every unit $f$,

$$1-\|Qf\|^2=\big(1-\|P^Lf\|^2\big)+\big(\|P^Lf\|^2-\|P^{L'}P^Lf\|^2\big),$$

and both terms are nonnegative. Equivalently the terms are

$$f^\top\big(I-(P^L)^\top P^L\big)f,\qquad (P^Lf)^\top\big(I-(P^{L'})^\top P^{L'}\big)P^Lf.$$

For symmetric PSD factors, a right singular vector $v_i$ and $w=P^Lv_i$ satisfy the additional geometric bounds

$$\|(I-P^L)v_i\|^2\le1-\sigma_i^2,\qquad \|(I-P^{L'})w\|^2\le1-\sigma_i^2.$$

For a nonzero singular value with $Qv_i=\sigma_i u_i$, these also give

$$\|\sigma_i u_i-v_i\|\le 2\sqrt{1-\sigma_i^2}.$$

This is a vector-closeness bound, not by itself a localization or common-support guarantee.

For nonempty $A$, $\langle Q\mathbf1_A,\mathbf1_B\rangle/|A|$ is the mass fraction arriving in $B$ from an initial uniform distribution on $A$, using column-vector probability evolution. If $|A|=|B|=aN$ with $0<a<1$, define

$$f_A=\frac{\mathbf1_A-a\mathbf1}{\sqrt{Na(1-a)}},\qquad f_B=\frac{\mathbf1_B-a\mathbf1}{\sqrt{Na(1-a)}}.$$

Then the balanced two-set retention score has the exact identity

$$\frac{\langle Q\mathbf1_A,\mathbf1_B\rangle}{|A|}+\frac{\langle Q\mathbf1_{A^c},\mathbf1_{B^c}\rangle}{|A^c|}-1
=\langle Tf_A,f_B\rangle\le\sigma_1.$$

Replacing the two-level vectors $(f_A,f_B)$ by arbitrary centered unit vectors is an exact continuous relaxation of this score, with optimum $\sigma_1$. The higher singular values obey the Courant–Fischer formulas, with $n=N-1$,

$$\sigma_k^2=\max_{\substack{S\subset\mathcal H_0\\\dim S=k}}\ \min_{\substack{f\in S\\\|f\|=1}}\|Tf\|^2
=\min_{\substack{S\subset\mathcal H_0\\\dim S=n-k+1}}\ \max_{\substack{f\in S\\\|f\|=1}}\|Tf\|^2.$$

For fixed $a$, maximizing the first mass fraction alone is equivalent to maximizing this two-set score; if $a$ varies, this equivalence does not hold without the complementary/balanced normalization.

*Proof.* The dissipation identity telescopes, and each term is nonnegative by contraction. For a symmetric PSD contraction, $(1-\lambda)^2\le1-\lambda^2$ for $\lambda\in[0,1]$. Expanding in its eigenbasis shows $\|(I-P)h\|^2\le\|h\|^2-\|Ph\|^2$. Apply this to $h=v_i$ and $h=w$, and bound each nonnegative summand of the telescope by $1-\sigma_i^2$. The additional singular-pair bound follows from $Qv_i-v_i=(P^{L'}-I)w+(P^L-I)v_i$ and the triangle inequality.

Because $Q$ is column-stochastic, $Q\mathbf1_A/|A|$ is a probability vector, establishing the mass assertion. Let $M=\mathbf1_B^\top Q\mathbf1_A$. Bistochasticity gives the complement mass $N-2aN+M$. Direct substitution yields

$$\frac{M}{aN}+\frac{N-2aN+M}{(1-a)N}-1
=\frac{M-a^2N}{Na(1-a)}=f_B^\top Qf_A.$$

The $f$'s are centered and unit, and $Q$ restricts to $T$ there. Cauchy–Schwarz gives $\langle Tf,g\rangle\le\|T\|_2=\sigma_1$. A top right singular vector and its left partner attain equality if $\sigma_1>0$; if $T=0$, every pair attains zero. This establishes the stated relaxation without asserting that any maximizing singular vectors are indicators.

For completeness, diagonalize the symmetric PSD operator $T^\top T$ on $\mathcal H_0$, with orthonormal eigenvectors $v_1,\ldots,v_n$ and eigenvalues $\sigma_j^2$. The span of $v_1,\ldots,v_k$ gives a minimum Rayleigh quotient $\sigma_k^2$. Every other $k$-dimensional subspace intersects $\operatorname{span}(v_k,\ldots,v_n)$ nontrivially by dimension counting, so its minimum cannot exceed $\sigma_k^2$. This proves the first formula. The span of $v_k,\ldots,v_n$ gives a maximum $\sigma_k^2$, and every $(n-k+1)$-dimensional subspace intersects $\operatorname{span}(v_1,\ldots,v_k)$ nontrivially; this proves the second. Finally, for fixed $a$ the score is an increasing affine function of $M/(aN)$, whereas varying $a$ changes both its offset and scale. ∎

**What it buys.** A near-one singular value has a certified two-stage low-dissipation meaning. This is the finite-state balanced relaxation in the spirit of Froyland (2013): transport here is the identity, the two bistochastic factors are declared smoothers, and counting measure is shared rather than evolved between distinct measures. The coherent-set language has a precise variational foundation, while extraction of a partition or overlapping semantic features remains a separate, unguaranteed interpretation.

**Tractability and sampling.** Retention for declared sets uses two factor matvecs. The continuous relaxation uses the centered matrix-free SVD; centering costs $O(N)$ per vector. Optimizing over discrete sets is not solved by this theorem, and thresholding singular functions need not attain the relaxation bound. All statements concern the observed site mass; no independent-sampling assumption or population retention guarantee is implicit.

### Exact expansions, loadings, gradients, and probes

**Proposition A.6 (Parseval reconstruction and re-embedding).** Let $\Psi=(\psi_0,\ldots,\psi_{N-1})$ be a complete Euclidean-orthonormal basis of $\mathcal H$, let $F\in\mathbb R^{N\times p}$, and let $I\subseteq\lbrace0,\ldots,N-1\rbrace$ with projector $\Pi_I=\sum_{i\in I}\psi_i\psi_i^\top$. Define $a_F(\psi_i)=F^\top\psi_i$. Then

$$F=\sum_i\psi_i a_F(\psi_i)^\top,\qquad
\|F-\Pi_IF\|_F^2=\sum_{i\notin I}\|a_F(\psi_i)\|^2.$$

For a nonempty set $A$, $\psi_A=\mathbf1_A/\sqrt{|A|}$ has loading $a_F(\psi_A)=\sqrt{|A|}\,\bar F_A$ and contribution $\psi_Aa_F(\psi_A)^\top=\mathbf1_A\bar F_A^\top$, where $\bar F_A$ is the row centroid as a column vector. Thus projection onto normalized indicators of a partition replaces rows by their cell centroids.

If $F=U\Sigma V^\top$ is an SVD and $u_i$ has singular value $s_i>0$, then $a_F(u_i)=s_iv_i$. For centered $F$, these are the usual scaled PCA directions, with site eigenvectors equal to normalized PCA scores. Zero singular values give zero loadings and no uniquely determined principal direction.

*Proof.* Completeness gives $\Psi\Psi^\top=I$. Multiplication by $F$ proves the expansion. The omitted matrices are orthogonal in the Frobenius inner product because

$$\langle\psi_i a_i^\top,\psi_j a_j^\top\rangle_F=(\psi_i^\top\psi_j)(a_i^\top a_j),$$

and each squared norm is $\|a_i\|^2$. This proves the tail identity for any index subset. Summing the rows of $F$ on $A$ proves the centroid formula. Indicators of disjoint cells are orthogonal, so their projector sums the cellwise contributions. Finally $F^\top u_i=s_iv_i$ follows from the SVD. The centered data covariance is $F^\top F/N$ (or $F^\top F/(N-1)$ under the alternate sample convention), with the same principal directions, and $Fv_i=s_iu_i$ is the score vector. ∎

**What it buys.** A nonlinear spectral basis is still an exact, loss-accounted coordinate system for every observed quantity. Centroid re-embedding and PCA loadings are special cases of one identity, not competing reconstruction rules.

**Tractability and sampling.** Given $r$ retained modes, their loadings cost $O(Npr)$, with $O(Nr+pr)$ storage besides $F$. The residual can be computed as $\|F\|_F^2-\|\Psi_I^\top F\|_F^2$, avoiding the complete basis (subject to cancellation near zero). Exactness holds for all finite data; a spectral basis is not generally the best rank-$r$ basis for reconstructing $F$. If only nonconstant modes are retained, either retain the mean separately or center $F$; otherwise the constant loading is part of the tail.

**Proposition A.7 (use-mediated gradients, excluded sites, and optimizer-independent activation exactness).** Let $W\in\mathbb R^{q\times p}$ be used in linear operations $z=Wu$ in a differentiable finite computational graph. There are finitely many uses per retained site, indexed by slots $a=1,\ldots,n_{\mathrm{use}}$; unused slots can be zero-padded. Let $U_a\in\mathbb R^{N\times p}$ contain retained-site inputs and $\Delta_a\in\mathbb R^{N\times q}$ the **total adjoints for the original scalar training loss**, not adjoints recomputed after deleting sites or loss terms. Define

$$G_{\mathrm{use},X}=\sum_a\Delta_a^\top U_a.$$

Let $G_{\mathrm{excl}}$ sum $\delta_j u_j^\top$ over parameter uses omitted from $X$, including uses at sites removed by the zero-row policy [P2]. Let $G_{\mathrm{direct}}$ be the derivative of explicit parameter-only loss terms. Assume every dependency of the loss on $W$ is either one of these uses or a declared direct term. Then

$$G_{\mathrm{use}}=G_{\mathrm{use},X}+G_{\mathrm{excl}},\qquad
\nabla_W\mathcal L=G_{\mathrm{use}}+G_{\mathrm{direct}}.$$

In particular $G_{\mathrm{use}}=G_{\mathrm{use},X}$ requires that omitted uses contribute zero; no omitted uses is a sufficient hypothesis. If any sites are excluded, report $\|G_{\mathrm{excl}}\|_F$ alongside their removed mass, or state that this remainder was not computed. The retained-site exact expansion is

$$G_{\mathrm{use},X}=\sum_i\sum_a g_{i,a}b_{i,a}^\top,\qquad
 g_{i,a}=\Delta_a^\top\psi_i,\quad b_{i,a}=U_a^\top\psi_i.$$

For a spectral block $B$ of dimension $r_B$,

$$G_B=\sum_a\Delta_a^\top\Pi_BU_a$$

is invariant under orthogonal basis rotations within $B$, so $\|G_B\|_F$ and $\langle G_B,G_{B'}\rangle_F$ are invariant. A per-mode contribution has rank at most $\min(n_{\mathrm{use}},p,q)$; a block contribution has rank at most $\min(n_{\mathrm{use}}r_B,p,q)$.

For retained modes $I$, put $G_{X,I}=\sum_a\Delta_a^\top\Pi_IU_a$. Then

$$G_{\mathrm{use},X}-G_{X,I}=\sum_{i\notin I}\sum_a g_{i,a}b_{i,a}^\top,$$

$$\|G_{\mathrm{use},X}-G_{X,I}\|_F\le
\left(\sum_a\sum_{i\notin I}\|g_{i,a}\|^2\right)^{1/2}
\left(\sum_a\sum_{i\notin I}\|b_{i,a}\|^2\right)^{1/2}.$$

For any finite optimizer update $\Delta W$, the direct change at a retained use with inputs frozen is exactly

$$\delta Z_a=U_a\Delta W^\top
=\sum_i\psi_i\big(\Delta W\,b_{i,a}\big)^\top.$$

A realized activation difference with upstream inputs changing still has an exact complete-basis expansion on $X$, but need not equal this frozen-input formula.

*Proof.* Temporarily assign a distinct independent parameter copy $W_j$ to each use and a separate copy $W_{\mathrm{direct}}$ to the explicit parameter-only terms. This defines an untied loss $\mathscr F((W_j)_j,W_{\mathrm{direct}})$ whose restriction to the diagonal $W_j=W_{\mathrm{direct}}=W$ is the original loss. The multivariable chain rule gives

$$\nabla_W\mathcal L=\sum_j\nabla_{W_j}\mathscr F+\nabla_{W_{\mathrm{direct}}}\mathscr F.$$

At the common parameter values, each total adjoint $\delta_j$ gives $\nabla_{W_j}\mathscr F=\delta_j u_j^\top$ by differentiating the single operation $z_j=W_ju_j$. Partitioning the uses into retained and excluded proves the full-gradient bookkeeping without a separability assumption on the loss. Insert $I=\Psi\Psi^\top$ into each retained $\Delta_a^\top U_a$ to obtain the expansion.

Replacing $\Psi_B$ by $\Psi_BR$ with orthogonal $R$ preserves $\Psi_BRR^\top\Psi_B^\top=\Pi_B$, proving invariance. Each outer product has rank at most one; rank subadditivity and the ambient dimensions prove the stated mode and block bounds.

Put $E=I-\Pi_I$. Stack $E\Delta_a$ vertically into $\mathsf D$ and $EU_a$ into $\mathsf U$. Since $E^2=E=E^\top$, the retained gradient residual is $\mathsf D^\top\mathsf U$. Columnwise Cauchy–Schwarz gives

$$\|\mathsf D^\top\mathsf U\|_F^2
=\sum_{j,k}|\mathsf D_{:,j}^\top\mathsf U_{:,k}|^2
\le\|\mathsf D\|_F^2\|\mathsf U\|_F^2.$$

A.6 converts these norms to the two loading-tail sums. The sharper bound $\|\mathsf D^\top\mathsf U\|_F\le\min(\|\mathsf D\|_2\|\mathsf U\|_F,\|\mathsf D\|_F\|\mathsf U\|_2)$ also follows by applying the operator norm columnwise, then transposing for the second alternative. Distinct mode contributions need not be Frobenius-orthogonal: in the single-slot case their cross inner product is $(g_i^\top g_j)(b_i^\top b_j)$. Thus the activation Parseval tail identity is not a gradient residual identity.

Finally, with $U_a$ fixed, $U_a(W+\Delta W)^\top-U_aW^\top=U_a\Delta W^\top$ holds exactly. Expanding $U_a$ by A.6 proves the optimizer-independent formula. Applying A.6 to the realized activation-difference matrix proves the final assertion. ∎

**What it buys.** The retained-site part of the weight gradient and observed activation changes share exact site-space coordinates, with invariant block summaries and a truncation bound. Direct regularization and excluded uses remain visible remainders rather than being silently attributed to the retained spectral modes.

**Tractability and sampling.** A forward/backward pass supplies inputs and total adjoints, with storage or streaming. Computing $r$ loadings costs $O(n_{\mathrm{use}}Nr(p+q))$; explicitly assembling their gradient pieces costs $O(n_{\mathrm{use}}rpq)$. A sampled loss gives that sampled loss's gradient, not automatically the full-corpus gradient. Removed site mass does not bound excluded gradient norm: even a small mass can carry large adjoints. A block pressure is not generally a site-group loss gradient. Nondifferentiable points require a declared autodiff/subgradient convention.

**Theorem A.8 (action-view probe estimator with exact variance).** Fix two Jacobians $J_x,J_y\in\mathbb R^{q\times p}$ and $m_J\ge1$ independent probes $\xi_r\in\mathbb R^p$. Assume each probe has independent, identically distributed coordinates with mean zero, variance one, and fourth moment $\mu_4<\infty$. Use the same probe bank at all sites. Define

$$\widehat h_{xy}=\frac1{m_J}\sum_{r=1}^{m_J}\langle J_x\xi_r,J_y\xi_r\rangle,\quad
M=J_x^\top J_y,\quad S=\frac12(M+M^\top).$$

Then

$$\mathbb E\widehat h_{xy}=\langle J_x,J_y\rangle_F,$$

$$\operatorname{Var}(\widehat h_{xy})=\frac1{m_J}\left(2\|S\|_F^2+(\mu_4-3)\sum_i S_{ii}^2\right).$$

In particular Gaussian probes give $2\|S\|_F^2/m_J$, and independent Rademacher probes give

$$\frac{2}{m_J}\sum_{i\ne j}S_{ij}^2
=\frac{2}{m_J}\left(\|S\|_F^2-\sum_iS_{ii}^2\right).$$

The entire estimated raw Gram matrix $(\widehat h_{xy})$ is PSD for each realized shared probe bank. Its normalized cosine entries and their exponentiated kernel are not asserted to be unbiased estimators of the exact cosine or exact sphere kernel.

*Proof.* Since $\xi^\top M\xi=\xi^\top S\xi$ and $\mathbb E\xi\xi^\top=I$, the mean is $\operatorname{tr}M=\langle J_x,J_y\rangle_F$. For a symmetric $S$,

$$\xi^\top S\xi-\operatorname{tr}S
=\sum_i S_{ii}(\xi_i^2-1)+2\sum_{i<j}S_{ij}\xi_i\xi_j.$$

Distinct terms in this display have zero covariance. For two different off-diagonal pairs, at least one independent coordinate appears once and has mean zero; for a diagonal/off-diagonal pair the other off-diagonal coordinate does so. Diagonal terms are independent. The diagonal variances are $(\mu_4-1)S_{ii}^2$, and the off-diagonal variances are $4S_{ij}^2$. Thus one probe has variance $(\mu_4-1)\sum_iS_{ii}^2+4\sum_{i<j}S_{ij}^2$, which is the stated expression. Independence across probes divides it by $m_J$. Gaussian coordinates have $\mu_4=3$, Rademacher coordinates $\mu_4=1$.

Finally let $\zeta_x=m_J^{-1/2}(J_x\xi_1,\ldots,J_x\xi_{m_J})$ be the concatenated sketch. Then $\widehat h_{xy}=\zeta_x^\top\zeta_y$, proving PSD. When each $\zeta_x\ne0$, normalizing these sketches and exponentiating their dot products gives exactly a sphere kernel on the sketches, so A.1 and A.2 apply to that realized kernel. Nonlinear normalization and exponentiation do not commute with expectation; unbiasedness of the raw inner product alone makes no claim about them. ∎

**What it buys.** Local derivative similarity is estimable without materializing the Jacobian, with a concrete noise scale rather than an unspecified $O(m_J^{-1/2})$. Sharing probes also preserves an actual Gram geometry in each realization.

**Tractability and sampling.** The sketch costs $m_J$ JVPs per site and, if stored, $O(Nm_Jq)$ memory; dense sketch-Gram construction adds $O(N^2m_Jq)$. If the one-probe variance is $V_{xy}$, Chebyshev gives $\Pr(|\widehat h_{xy}-h_{xy}|\ge\epsilon)\le V_{xy}/(m_J\epsilon^2)$; a simultaneous finite-pair guarantee follows by summing these bounds. This Chebyshev/union-bound route is conservative, not a sharp all-pairs probe-complexity claim. Thus sample size here means independent probes, not independent sites. Relative accuracy requires nonvanishing signal, and cosine estimation additionally requires norms bounded away from zero. A single JVP perturbs all input sites unless implemented otherwise: obtaining the same-site Jacobian block must isolate that input-site direction or use an explicitly justified batched/block method, not silently include attention's cross-site blocks.

### The two temperature endpoints

**Theorem A.9 (explicit small-temperature remainder after Sinkhorn scaling).** For any fixed collection of unit vectors $z_1,\ldots,z_N$, let $P_\beta$ be the uniquely scaled sphere-kernel operator. Then for every $\beta\ge0$,

$$\left\|CP_\beta C-\frac{\beta}{N}C\Gamma C\right\|_F\le R(\beta),$$

where

$$R(\beta)=\big(e^{3\beta/2}-1\big)^2+\beta\big(e^{3\beta}-1\big)+e^{3\beta}\big(e^\beta-1-\beta\big).$$

In particular, for $0\le\beta\le\beta_{\max}$ with any fixed $\beta_{\max}>0$,

$$R(\beta)\le M_{\beta_{\max}}\beta^2,\qquad M_{\beta_{\max}}=\frac{21}{4}e^{3\beta_{\max}}+\frac12e^{4\beta_{\max}}.$$

The same bounds hold in operator norm. Hence the claimed $CP_\beta C=(\beta/N)CZ Z^\top C+O(\beta^2)$ has an explicit, dimension-independent absolute Frobenius remainder bound.

*Proof.* Set $w=\sqrt N\,v$ and $\bar w=N^{-1}\sum_iw_i$. The scaling equation is

$$w_i\left(\frac1N\sum_j e^{\beta \Gamma_{ij}}w_j\right)=1.$$

Since $|\Gamma_{ij}|\le1$,

$$e^{-\beta}\le w_i\bar w\le e^\beta.$$

Average over $i$ to obtain $e^{-\beta}\le\bar w^2\le e^\beta$, and divide the previous bounds by these bounds on $\bar w$. Therefore

$$e^{-3\beta/2}\le w_i\le e^{3\beta/2}.$$

Write $a=w-\mathbf1$, $E=D_w-I$, and $\eta=e^{3\beta/2}-1$. Then $\|E\|_2\le\eta$, $\|a\|\le\sqrt N\eta$, and $\|D_w\|_2\le e^{3\beta/2}$. Entrywise Taylor expansion gives

$$K_\beta=\mathbf1\mathbf1^\top+\beta \Gamma+\mathcal E_\beta,\qquad |(\mathcal E_\beta)_{ij}|\le e^\beta-1-\beta,$$

because the absolute tail of the exponential series is bounded by the same series at $\beta$. Thus $\|\mathcal E_\beta\|_F\le N(e^\beta-1-\beta)$ and $\|\Gamma\|_F\le N$.

Now $P_\beta=N^{-1}D_wK_\beta D_w$. Since $C\mathbf1=0$, the constant term after centering is

$$\frac1N CD_w\mathbf1\mathbf1^\top D_wC=\frac1N Caa^\top C,$$

whose Frobenius norm is at most $\eta^2$. The linear term differs from $(\beta/N)C\Gamma C$ by

$$\frac{\beta}{N}C(E\Gamma+\Gamma E+E\Gamma E)C,$$

with norm at most $\beta(2\eta+\eta^2)=\beta(e^{3\beta}-1)$. The Taylor-tail contribution has norm at most $e^{3\beta}(e^\beta-1-\beta)$. The triangle inequality proves $R(\beta)$.

For $0\le\beta\le\beta_{\max}$, the mean-value or integral remainder bounds give

$$e^{3\beta/2}-1\le\frac32e^{3\beta_{\max}/2}\beta,\quad e^{3\beta}-1\le3e^{3\beta_{\max}}\beta,\quad e^\beta-1-\beta\le\frac12e^{\beta_{\max}}\beta^2.$$

Adding their contributions gives $M_{\beta_{\max}}$. Finally $\|A\|_2\le\|A\|_F$. At $\beta=0$, $v=N^{-1/2}\mathbf1$ and $P_0=ee^\top$, so the formula and bounds also hold at the endpoint itself. ∎

**What it buys.** Sinkhorn normalization preserves the linear centered geometry to first order; the cancellation is certified, not just numerically observed. The explicit bound separates an absolute approximation guarantee from the potentially much harder task of relative or spectral identification.

**Tractability and sampling.** The bound holds uniformly over all configurations of $N$ unit rows and needs no stochastic sampling model. It is deliberately conservative. Relative error compared with $(\beta/N)C\Gamma C$ can still be large if $\|C\Gamma C\|_F/N$ is small; a sample-size-only guarantee would be misleading. Dense scaling costs are those of A.1. At very small $\beta$, subtracting the constant component causes numerical cancellation; use stable centering and tolerances materially below the $O(\beta)$ signal.

**Corollary A.10 (linear CKA and gapped PCA are the small-temperature limits).** Consider two fixed views $Z_A,Z_B$ on the same sites, with $\Gamma_A^c=CZ_AZ_A^\top C\ne0$ and $\Gamma_B^c=CZ_BZ_B^\top C\ne0$. Their centered-operator alignment satisfies

$$\lim_{\beta\downarrow0}
\frac{\langle CP^A_\beta C,CP^B_\beta C\rangle_F}{\|CP^A_\beta C\|_F\,\|CP^B_\beta C\|_F}
=\frac{\langle \Gamma_A^c,\Gamma_B^c\rangle_F}{\|\Gamma_A^c\|_F\|\Gamma_B^c\|_F}.$$

This is linear CKA of the **row-normalized** activations. Let $X_A=CZ_A$, $X_B=CZ_B$. Equivalently it is

$$\frac{\|X_A^\top X_B\|_F^2}{\|X_A^\top X_A\|_F\,\|X_B^\top X_B\|_F}.$$

For one view, put $\Gamma_c=C\Gamma C$ and work on $\mathcal H_0$. Let a spectral block of $\Gamma_c|_{\mathcal H_0}$ have dimension $r$ and gap $\delta>0$ from all eigenvalues outside that block. For

$$\eta_\beta=\frac{NR(\beta)}{\beta}<\frac\delta2,$$

the corresponding block of $CP_\beta C$ has a well-defined rank-$r$ spectral projector $\Pi_\beta$, and its limit is the PCA site-score projector $\Pi$ of $CZ$. More explicitly,

$$\|(I-\Pi)\Pi_\beta\|_F\le\frac{\eta_\beta}{\delta-\eta_\beta},\qquad
\|\Pi_\beta-\Pi\|_F\le\frac{\sqrt2\,\eta_\beta}{\delta-\eta_\beta}.$$

An isolated simple eigenvector therefore converges up to sign. An exactly repeated eigenvalue identifies only its eigenspace, not a preferred individual PCA basis. If the whole centered space is the block, the projector is simply $C$ and no exterior gap is needed.

*Proof.* For $\beta>0$, define $A_\beta=(N/\beta)CP_\beta C$. A.9 gives $\|A_\beta-\Gamma_c\|_F\le\eta_\beta\to0$ for fixed $N$. Normalizing a nonzero matrix by its Frobenius norm is continuous, so applying this separately to the two views gives the alignment limit. A quantitative version follows from

$$\left\|\frac A{\|A\|_F}-\frac{\Gamma_c}{\|\Gamma_c\|_F}\right\|_F\le\frac{2\|A-\Gamma_c\|_F}{\|\Gamma_c\|_F}\quad(A\ne0,\Gamma_c\ne0).$$

Indeed, add and subtract $A/\|\Gamma_c\|_F$ and use the reverse triangle inequality. If $\eta_\beta<\min(\|\Gamma_A^c\|_F,\|\Gamma_B^c\|_F)$, the alignment error is at most $2\eta_\beta(\|\Gamma_A^c\|_F^{-1}+\|\Gamma_B^c\|_F^{-1})$. Trace cyclicity gives $\langle X_AX_A^\top,X_BX_B^\top\rangle_F=\|X_A^\top X_B\|_F^2$ and $\|XX^\top\|_F=\|X^\top X\|_F$, proving the linear-CKA formula.

For the eigenspace statement write $A_\beta=\Gamma_c+E$ on $\mathcal H_0$, with $\|E\|_2\le\eta_\beta$. Weyl's eigenvalue perturbation inequality (Horn–Johnson, 2013, Theorem 4.3.1) and the Davis–Kahan (1970) subspace principle have the following self-contained specialization here. The variational characterization of ordered eigenvalues implies that each ordered eigenvalue moves by at most $\eta_\beta$: every unit-vector Rayleigh quotient changes by at most that amount, and the min–max formula preserves these two inequalities. Thus the block stays separated when $2\eta_\beta<\delta$, with unchanged multiplicity. For a unit eigenvector $u$ of $A_\beta$ in the corresponding block, with eigenvalue $\mu$, projection of $(\Gamma_c-\mu I)u=-Eu$ onto the exterior eigenspace gives

$$\|(I-\Pi)u\|\le\frac{\|E\|_2}{\delta-\eta_\beta}.$$

Here the restriction of $\Gamma_c-\mu I$ to the exterior has all eigenvalues of magnitude at least $\delta-\eta_\beta$, so its inverse has norm at most the reciprocal. To retain the sharper Frobenius constant, use $\|(I-\Pi)u\|\le\|Eu\|/(\delta-\eta_\beta)$ in the same argument, and sum its square over an orthonormal basis $u_1,\ldots,u_r$ of the perturbed block. Then $\sum_j\|Eu_j\|^2=\|E\Pi_\beta\|_F^2\le\|E\|_F^2\le\eta_\beta^2$, proving the first bound without a $\sqrt r$ loss. Equal-rank orthogonal projectors satisfy $\|\Pi_\beta-\Pi\|_F^2=2\|(I-\Pi)\Pi_\beta\|_F^2$, by expanding traces, giving the second. Finally $\Gamma_c=(CZ)(CZ)^\top$, so its positive-eigenvalue site vectors are the normalized PCA score vectors by A.6. A zero-eigenvalue block is a PCA nullspace, not a set of nonzero score directions. ∎

**What it buys.** Linear representational comparison and PCA are genuine limiting calibrations of the same construction. The gap and nonzero-signal qualifications say precisely when a limit of matrices yields an identifiable limit of reported objects.

**Tractability and sampling.** Linear CKA can be evaluated through centered feature cross-products without an $N\times N$ kernel, with cost depending on the two feature widths; PCA can use an SVD of $CZ$. The nonlinear route has the dense/Sinkhorn/Lanczos costs already stated. No finite-$N$ independence assumption is needed, but a vanishing Gram norm or eigenvalue gap destroys the useful relative bound. This is a fixed-data limit, not a joint $N\to\infty$, $\beta\to0$ sample-complexity result.

**Theorem A.11 (large-temperature duplicate-class projector and angular rate).** Partition the fixed unit rows into exact duplicate classes $\mathcal C_1,\ldots,\mathcal C_m$: two sites are in the same class if and only if their normalized vectors coincide. Put $n_c=|\mathcal C_c|$ and define the orthogonal projector

$$\Pi_{\mathrm{dup}}=\sum_{c=1}^m\frac{\mathbf1_{\mathcal C_c}\mathbf1_{\mathcal C_c}^\top}{n_c}.$$

If $m\ge2$, let $z_c$ denote the distinct class representatives and define

$$\theta_{\min}=\min_{c\ne d}\arccos(z_c^\top z_d)>0,\quad
\gamma=1-\cos\theta_{\min}=\frac12\min_{c\ne d}\|z_c-z_d\|^2>0,\quad
\kappa_{\mathrm{dup}}=\max_c\sum_{d\ne c}\sqrt{n_d/n_c}\le N-1.$$

Then with $\varepsilon_\beta=\min(1,2\kappa_{\mathrm{dup}}e^{-\beta\gamma})$,

$$\|P_\beta-\Pi_{\mathrm{dup}}\|_2\le\varepsilon_\beta,\qquad
\|P_\beta-\Pi_{\mathrm{dup}}\|_F\le\sqrt m\,\varepsilon_\beta,$$

$$\|CP_\beta C-(\Pi_{\mathrm{dup}}-ee^\top)\|_2\le\varepsilon_\beta.$$

Thus $P_\beta\to\Pi_{\mathrm{dup}}$, with $m$ eigenvalues tending to one and the other $N-m$ identically zero for every $\beta$. For fixed integer $t\ge1$, $\|P_\beta^t-\Pi_{\mathrm{dup}}\|_2\le t\varepsilon_\beta$. If $m=1$, $P_\beta=ee^\top=\Pi_{\mathrm{dup}}$ exactly at every temperature.

The distinct-row case has $m=N$, $\Pi_{\mathrm{dup}}=I$, and $\kappa_{\mathrm{dup}}=N-1$, recovering the identity limit and rate $2(N-1)e^{-\beta\gamma}$. For two views with at least two classes each, the limiting centered alignment is

$$\frac{\langle\Pi_{\mathrm{dup}}^A-ee^\top,\Pi_{\mathrm{dup}}^B-ee^\top\rangle_F}
{\sqrt{(m_A-1)(m_B-1)}}.$$

It is one when both views have distinct rows, but need not be one for different duplicate partitions. With one class the corresponding centered operator is zero and its alignment is undefined.

*Proof.* If $m=1$, the scaled kernel is $\mathbf1\mathbf1^\top$ and its scaling is $N^{-1/2}\mathbf1$, proving that case directly. In the rest of the proof assume $m\ge2$. Scale the kernel by $e^{-\beta}$ using A.1. The resulting matrix has entries one within each duplicate class and entries $e^{-\beta(1-z_c^\top z_d)}\le e^{-\beta\gamma}$ between classes. Permuting sites within a class leaves the kernel unchanged; uniqueness of the symmetric scaling implies that the scaling vector is constant on that class. Write its value as $a_c>0$. Each row in class $c$ satisfies

$$1=n_ca_c^2+a_c\sum_{d\ne c}n_d e^{-\beta(1-z_c^\top z_d)}a_d.$$

All terms are nonnegative, so $a_c\le n_c^{-1/2}$. The total probability sent out of that class is therefore

$$r_c=1-n_ca_c^2
\le e^{-\beta\gamma}\sum_{d\ne c}\sqrt{n_d/n_c}
\le\kappa_{\mathrm{dup}}e^{-\beta\gamma}.$$

Within the class, each entry of $P_\beta$ is $a_c^2=(1-r_c)/n_c$; the corresponding projector entry is $1/n_c$. Thus each row's absolute difference mass is $r_c$ inside the class and $r_c$ outside. Symmetry gives $\|P_\beta-\Pi_{\mathrm{dup}}\|_2\le\|P_\beta-\Pi_{\mathrm{dup}}\|_\infty\le2\kappa_{\mathrm{dup}}e^{-\beta\gamma}$. The displayed bound $\kappa_{\mathrm{dup}}\le N-1$ follows from $\sqrt{n_d/n_c}\le n_d$ and $\sum_{d\ne c}n_d=N-n_c\le N-1$.

Both rows and columns of $P_\beta$ are constant within each duplicate class, so

$$P_\beta\Pi_{\mathrm{dup}}=\Pi_{\mathrm{dup}}P_\beta=P_\beta.$$

On the class-constant subspace, $\Pi_{\mathrm{dup}}$ is the identity and $P_\beta$ is a PSD contraction by A.2; on its orthogonal complement both vanish. Hence the norm difference is also at most one, its rank at most $m$, and its Frobenius norm at most $\sqrt m$ times its operator norm. This proves all norm bounds, since $C\Pi_{\mathrm{dup}}C=\Pi_{\mathrm{dup}}-ee^\top$ and centering cannot increase operator norm. Restricting to the class-constant subspace shows that all its $m$ eigenvalues tend to one, while the complementary $N-m$ eigenvalues are always zero. The matrix-power telescoping identity for $P_\beta^t-\Pi_{\mathrm{dup}}^t$, with both factors contractions and $\Pi_{\mathrm{dup}}^t=\Pi_{\mathrm{dup}}$, proves the power bound. Finally $\Pi_{\mathrm{dup}}-ee^\top$ is a rank-$(m-1)$ orthogonal projector; its Frobenius norm is $\sqrt{m-1}$, so continuity of normalized inner products proves the alignment limit. ∎

**What it buys.** The endpoint identifies exact equality classes of normalized activations, not a generic finite-scale clustering. When all rows are distinct it is the identity and carries no preferred nonconstant basis; with duplicates it retains their class-indicator subspace. Such duplicates can be systematic, for example repeated token embeddings or identical prefixes, so the general projector limit matters in practice.

**Tractability and sampling.** For at least two classes, operator error at most $\epsilon<1$ is ensured by $\beta\ge\gamma^{-1}\log(2\kappa_{\mathrm{dup}}/\epsilon)$. Exact equality classes can be found by grouping identical rows; treating near-equality as equality introduces a new tolerance and is not this theorem. Minimum interclass separation may shrink with growing samples, so the rate is fixed-data, not uniform sample complexity. Form $e^{-\beta}K_\beta$ to avoid overflow. Individual eigenvectors within the limiting class-indicator eigenspace remain noncanonical; no semantic interpretation follows from equality alone.

### Numerical verification

The repository's existing `checks/cm_check.py` was run unchanged with `uv run --with numpy python -I` and printed `all checks passed`. It checks the normalization/product distinction, small-temperature alignment, reconstruction tail, and several neighboring theory identities.

The reproducible script `checks/theorems_a1_check.py` is run from the repository root with `uv run --with numpy python -I checks/theorems_a1_check.py`; its final success line is `all A1 checks passed`, observed after the review revisions. It asserts scaling and scale invariance; the convex Hessian; spectra, powers, and diffusion distance; composition, degree-normalization contrast, dissipation, balanced-set identities, and the singular-subspace min-max formula; exact reconstruction, centroid and PCA loadings; multi-slot gradients against finite differences, block rotation invariance, slot-rank bounds, truncation bounds, and frozen-input finite updates; both probe variance constants by exhaustive Rademacher enumeration and degree-four-exact Gaussian quadrature (one probe and an independent two-probe average); the explicit small-temperature remainder across four sample sizes and six temperatures; CKA and gapped-projector limits; and the high-temperature angular rate. It also checks the non-PSD-positive, fractional-power, reducible-degree, and duplicate-row boundary examples. Maximum observed small-temperature error divided by the displayed bound was **0.031902**; the CKA discrepancy at temperature 0.00001 was **0.00000141958**. The fractional-power counterexample gave a negative entry **-0.014593245**. These deterministic checks are regression evidence, not substitutes for the proofs. Review regressions additionally assert the PSD degree counterexample, duplicate-class projector and rate, excluded-use/direct-term gradient accounting, damped local Jacobian, global projective contraction and absolute log-error bounds, and the sharpened A.5/A.10 bounds.

### References

All theorem proofs above are given explicitly. The following references establish provenance and the relationships to standard constructions; the separate discussion of the undamped algorithm specifically uses Knight's stated hypotheses.

- Richard Sinkhorn (1964), "A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices," *The Annals of Mathematical Statistics* **35**(2), 876–879. [DOI: 10.1214/aoms/1177703591](https://doi.org/10.1214/aoms/1177703591).
- Richard Sinkhorn and Paul Knopp (1967), "Concerning Nonnegative Matrices and Doubly Stochastic Matrices," *Pacific Journal of Mathematics* **21**(2), 343–348. [DOI: 10.2140/pjm.1967.21.343](https://doi.org/10.2140/pjm.1967.21.343); [publisher's text](https://msp.org/pjm/1967/21-2/pjm-v21-n2-p14-p.pdf). Classical alternating-scaling and support conditions; A.1 proves its own sufficient conditions.
- Albert W. Marshall and Ingram Olkin (1968), "Scaling of Matrices to Achieve Specified Row and Column Sums," *Numerische Mathematik* **12**, 83–90. doi:10.1007/BF02170999. Extremal formulations of scaling; the exact log-potential argument used here is written out, not attributed verbatim to this paper.
- J. Csima and B. N. Datta (1972), "The DAD Theorem for Symmetric Non-negative Matrices," *Journal of Combinatorial Theory, Series A* **12**(1), 147–152. [DOI: 10.1016/0097-3165(72)90090-8](https://doi.org/10.1016/0097-3165(72)90090-8); [verified bibliographic record](https://dblp.org/rec/journals/jct/CsimaD72.html). Symmetric nonnegative scaling; A.1 independently proves the positive-diagonal sufficient case.
- Philip A. Knight (2008), "The Sinkhorn–Knopp Algorithm: Convergence and Applications," *SIAM Journal on Matrix Analysis and Applications* **30**(1), 261–275. [DOI: 10.1137/060659624](https://doi.org/10.1137/060659624); [author's text](https://personal.strath.ac.uk/p.a.knight/skapp.pdf). Lemma 4.1, Lemma 4.3, and Theorem 4.4 are numbered as in the author's text. Theorem 4.4 assumes the undamped iteration starts at the all-ones vector; its rate is not the damped rate of A.1.
- Bas Lemmens and Roger Nussbaum (2012), *Nonlinear Perron–Frobenius Theory*, Cambridge University Press, Cambridge Tracts in Mathematics **189**, Appendix A, "The Birkhoff–Hopf theorem," 255–283. [Book DOI: 10.1017/CBO9781139026079](https://doi.org/10.1017/CBO9781139026079); [Appendix A](https://doi.org/10.1017/CBO9781139026079.011). A.1 proves the finite positive-matrix projective estimate it needs directly.
- Ronald R. Coifman and Stéphane Lafon (2006), "Diffusion Maps," *Applied and Computational Harmonic Analysis* **21**(1), 5–30. [DOI: 10.1016/j.acha.2006.04.006](https://doi.org/10.1016/j.acha.2006.04.006). The stationary-density-weighted diffusion-distance convention compared with counting measure in A.2.
- Nicholas F. Marshall and Ronald R. Coifman (2019), "Manifold Learning with Bi-stochastic Kernels," *IMA Journal of Applied Mathematics* **84**(3), 455–482. [DOI: 10.1093/imamat/hxy065](https://doi.org/10.1093/imamat/hxy065); [arXiv:1711.06711](https://arxiv.org/abs/1711.06711). Bistochastic kernels with respect to a specified measure; not a claim that their continuum conclusions follow from A.1 alone.
- Gary Froyland (2013), "An Analytic Framework for Identifying Finite-Time Coherent Sets in Time-Dependent Dynamical Systems," *Physica D: Nonlinear Phenomena* **250**, 1–19. [DOI: 10.1016/j.physd.2013.01.013](https://doi.org/10.1016/j.physd.2013.01.013); [arXiv:1210.7418](https://arxiv.org/abs/1210.7418). A.5's finite-state variant uses identity transport, two declared smoothers, and shared counting measure; its balanced-score formula is proved independently.
- Chandler Davis and W. M. Kahan (1970), "The Rotation of Eigenvectors by a Perturbation. III," *SIAM Journal on Numerical Analysis* **7**(1), 1–46. [DOI: 10.1137/0707001](https://doi.org/10.1137/0707001). The eigenspace perturbation principle specialized and proved in A.10.
- Roger A. Horn and Charles R. Johnson (2013), *Matrix Analysis*, 2nd ed., Cambridge University Press, especially Theorem 4.3.1 (Weyl's eigenvalue inequalities). A.5 proves the needed Courant–Fischer specialization; A.10 derives the needed Weyl perturbation bound directly by min–max, then gives the Frobenius eigenspace proof.

### Discrepancies with THEORY.md

1. **Positive entries are not PSD.** The spectrum claim in Section 3.1 is correct for the sphere kernels because they are PSD, but must not be generalized to every strictly positive symmetric $K$. For example $K=\begin{pmatrix}1&2\\2&1\end{pmatrix}$ scales to $K/3$ and has eigenvalues $1,-1/3$. A.1 and A.2 deliberately separate the assumptions. This is a scope clarification, not a counterexample to the actual dense sphere-kernel spine.
2. **Diffusion time and distance conventions need specification (Section 3.2).** Integer powers are Markov kernels. General fractional spectral powers of PSD stochastic matrices need not be entrywise nonnegative. For an explicit positive PSD example, let $P_0$ be the symmetric tridiagonal matrix with diagonal $(3/4,1/2,3/4)$ and adjacent off-diagonal entries $1/4$, and set $P=0.99P_0+0.01\mathbf1\mathbf1^\top/3$. Its eigenvalues are $1,0.99(3/4),0.99(1/4)$ on eigenvectors proportional to $(1,1,1),(1,0,-1),(1,-2,1)$, respectively. Hence $(P^{1/2})_{13}=1/3-\sqrt{0.99}(\sqrt3/4-1/12)<0$. This is a counterexample in the general PSD Markov-matrix class, not an assertion that this particular matrix is a sphere-kernel normalization. Separately, the counting-measure distance in THEORY differs by $\sqrt N$ from the common stationary-density-weighted diffusion distance. A.2 declares both conventions rather than treating them as numerically identical.
3. **The balanced-set relaxation (Section 4.1) needs its objective written out.** The exact relaxed objective is the sum of retention of a set and its complement minus one, for equally sized source/target sets, or equivalently twice their average retention minus one. Raw $A$-to-$B$ retention is equivalent only when the size fraction is fixed. A.5 supplies the precise identity; the SVD does not itself produce a partition or solve the discrete optimum.
4. **Degree-normalization contrast needs positivity/aperiodicity (Section 3.1).** A.4 proves the claim for the strictly positive dense kernels used in the spine. It is false for unrestricted sparse kernels: $K_1=\operatorname{diag}(1,2)$ and $K_2=\operatorname{diag}(2,1)$ are themselves positive definite and have nonproportional degrees but $S_1=S_2=I$. This is consistent with the separate-estimand warning in Section 13.
5. **PCA and CKA endpoint qualifications (Sections 2, 5, 8).** CKA needs two nonzero centered Gram matrices; an all-identical view has zero centered operator at every temperature and its alignment is undefined. Individual eigenvectors have no canonical convergence claim inside repeated eigenvalue blocks. A.10 proves convergence of gapped spectral projectors and of isolated eigenvectors up to sign. These are qualifications to informal endpoint language, not changes to the operator construction.
6. **High-temperature comparison depends on duplicate classes (Sections 2 and 8).** In full generality the endpoint is $\Pi_{\mathrm{dup}}=\sum_c\mathbf1_{\mathcal C_c}\mathbf1_{\mathcal C_c}^\top/|\mathcal C_c|$, not $I$. Section 2's identity statement correctly assumes distinct normalized rows, but any broader endpoint interpretation needs the duplicate-class qualification. Duplicates occur naturally for repeated embeddings or identical prefixes. A.11 proves the general limit and an interclass angular rate. Section 8's comparison endpoints need the same qualification: different duplicate partitions need not align perfectly, and an all-identical view has an undefined centered alignment. For example its uncentered alignment with a distinct-row view tends to $1/\sqrt N$, not one.
7. **Probe unbiasedness is only before cosine normalization (Section 7.3).** The raw inner-product statement is correct. It does not make the normalized action kernel unbiased. Also, shared random probes across sites do not by themselves isolate same-site Jacobian blocks in a cross-site-coupled network; the JVP implementation must honor the declared block. A.8 specifies the variance constant and this implementation condition.
8. **Sinkhorn rate must identify the algorithm (Section 3.1).** The implemented damped update has local log-coordinate rate at most $1/2$ for PSD kernels, proved in A.1, so rapid final convergence has a mathematical explanation even for clustered data. Its global projective contraction bound depends on temperature. Knight's Theorem 4.4 instead concerns the undamped alternating iteration with the specified all-ones start. Neither statement makes "a few dozen" a uniform total iteration bound for arbitrary initial errors and tolerances.
9. **Full-gradient exactness needs excluded uses and direct terms (Section 7.2 versus [P2]).** Removing sites with zero rows in any kernel view does not remove their contributions to the original training-loss gradient. A.7 separates $G_{\mathrm{use},X}$, $G_{\mathrm{excl}}$, and $G_{\mathrm{direct}}$ and requires original-loss adjoints. The retained-site expansion equals the full gradient only if the combined excluded-plus-direct remainder vanishes; separately vanishing terms are a sufficient condition. This is a genuine missing qualification in the displayed full-gradient identity, not a limitation of the expansion on the retained site set.

No changes were made to `THEORY.md` or to the construction. The fixed-data spine identities proved here are sound under the stated qualifications; none is a causal, semantic-recovery, or population sample-complexity theorem.

---
## Part A, cluster 2 — Stability: perturbation theory for blocks

This cluster makes the stability statements in THEORY Sections 3.2, 4 and 8 quantitative. All results are finite-site and deterministic except the explicitly labelled high-dimensional noise theorem. They do not assume sparsity, a semantic partition, a manifold, or independent sampling of sites. Sampling error enters only after it has been converted into an operator perturbation; the sample-complexity cluster supplies that separate step.

### A.20–A.22. Which spectral object is stable?

**Definition A.20 (isolated block and subspace distance).** Let $J=\mathbf1\mathbf1^\top/N$, $C=I-J$, and $\mathcal H_0=\mathbf1^\perp$. For the centered operator $\tilde P=CPC$, use its restriction to $\mathcal H_0$ when defining nonconstant blocks: the extra zero eigenvalue on the removed constant direction is not part of the spectrum being cut. Likewise use $CQC|_{\mathcal H_0}$ for nonconstant singular blocks of $Q=P^{L'}P^L$.

For a real symmetric matrix $A$ with eigenvalues in decreasing order, a **contiguous isolated block** $B=\lbrace s,\ldots,t\rbrace$ has rank $r=t-s+1$ and external gap

$$
\delta=\min\bigl(\lambda_{s-1}-\lambda_s,\lambda_t-\lambda_{t+1}\bigr)>0,
$$

omitting a boundary term when its neighbor does not exist. The whole ambient space needs no bound: its projector is constant. Internal multiplicities are allowed. Under a sufficiently small perturbation, $B'$ means the same ordered indices, not a fresh threshold selection. Write $\Pi_B=\Psi_B\Psi_B^\top$. For two equal-rank subspaces with orthonormal bases $V,W$, the singular values of $(I-VV^\top)W$ are the sines of their principal angles, including zeros. Thus

$$
\|\Pi_B-\Pi_{B'}\|_{\mathrm{op}}=\|\sin\Theta(B,B')\|_{\mathrm{op}},\qquad
\|\Pi_B-\Pi_{B'}\|_F=\sqrt2\|\sin\Theta(B,B')\|_F.
$$

These identities follow by decomposing the two projectors into their two-dimensional principal-angle planes; they include the coincident and orthogonal directions. A singular block is defined analogously using ordered singular values, with separate left and right projectors $\Pi_B^u,\Pi_B^v$. The zero singular block is permitted if it is separated from all singular values outside the block. A basis of a zero singular block does not define a unique pairing of left and right null vectors.

We write $\|\cdot\|_{\max}$ for maximum absolute entry. All other norms have their indicated Euclidean operator or Frobenius meanings.

**Theorem A.21 (Davis–Kahan bounds for an isolated symmetric block).** Let $A,A'=A+E$ be real symmetric on the same finite-dimensional Euclidean space. Assume the isolated-block condition of Definition A.20 and

$$e=\|E\|_{\mathrm{op}}<\delta/2,\qquad g=\delta-e.$$

Then the corresponding block $B'$ is isolated, has rank $r$, and

$$
\|\sin\Theta(B,B')\|_{\mathrm{op}}
 \le \min\left(1,\frac{e}{g}\right),
\qquad
\|\sin\Theta(B,B')\|_F
 \le \min\left(\sqrt r,\frac{\min(\sqrt r e,\|E\|_F)}{g}\right).
\tag{A.21.1}
$$

The operator constant is $1$ for interior as well as top/bottom blocks. The projector bounds follow from Definition A.20. In particular, both are $O(e/\delta)$ in operator norm, independently of internal spacings.

*Proof.* Weyl's eigenvalue inequality, obtained directly from the Courant–Fischer min–max formula, gives $|\lambda_i(A')-\lambda_i(A)|\le e$. Consequently both neighboring gaps of $B'$ are at least $\delta-2e>0$. Every selected eigenvalue of $A'$ is separated by at least $g$ from every unselected eigenvalue of $A$.

Let $V'$ be an orthonormal eigenbasis for $B'$, and $V_\perp$ an eigenbasis of its unperturbed complement. With $X=V_\perp^\top V'$ and diagonal eigenvalue matrices $\Lambda_\perp,\Lambda'$, the eigenvalue equation gives

$$\Lambda_\perp X-X\Lambda'=-V_\perp^\top E V'.$$

Entrywise division by $\lambda_j-\lambda_i'$ proves

$$\|X\|_F\le g^{-1}\|V_\perp^\top EV'\|_F
\le g^{-1}\min(\sqrt r e,\|E\|_F).$$

For the operator norm use the interval form of the Davis–Kahan theorem (Bhatia, *Matrix Analysis*, 1997, Theorem VII.3.1): if one self-adjoint spectral subspace has spectrum in $[a,b]$ and the other operator's complementary subspace has spectrum outside $(a-g,b+g)$, their sine-angle operator norm is at most the operator perturbation norm divided by $g$. Here choose $[a,b]=[\lambda_t(A)-e,\lambda_s(A)+e]$ for the selected spectrum of $A'$. Every unselected eigenvalue of $A$ lies outside $(a-g,b+g)$ by the definition of $\delta$ and $g=\delta-e$. This checks the interval hypothesis, not merely pairwise spectral separation. The theorem gives $\|X\|_{\mathrm{op}}\le e/g$; the Frobenius estimate above retains its residual/rank improvement. Since $X$ has the sine-angle singular values, the claims follow. ∎

**What it buys.** A gap protects a whole representation even when its displayed coordinates rotate wildly. The operator constant uses the sharp interval form of Davis–Kahan, rather than a claim that eigenvectors are uniformly stable.

**Tractability and sampling.** A full dense eigendecomposition costs $O(N^3)$ time and $O(N^2)$ storage; an iterative block calculation needs the block and both neighboring eigenvalues, with costs determined by matvecs, gaps and residual tolerances. Certified eigenvalue error $\tau$ turns an observed adjacent gap $\hat\delta$ into the lower bound $\hat\delta-2\tau$. The inequality is an a posteriori certificate once an upper bound on $e$ is available; the observed gap alone does not estimate population error. No universal sample count follows without a sampling model. Work on $\mathcal H_0$ avoids a spurious gap to the discarded constant direction.

**Theorem A.22 (Wedin bounds for a singular block).** Let $Q,Q'=Q+E$ be square real matrices on the same space, and let $B$ be a contiguous block of $r$ singular values with external gap $\delta>0$ as in Definition A.20. Assume $e=\|E\|_{\mathrm{op}}<\delta/2$ and set $g=\delta-e$. Let $\Theta_u,\Theta_v$ compare the corresponding left and right singular subspaces. Then

$$
\max\bigl(\|\sin\Theta_u\|_{\mathrm{op}},\|\sin\Theta_v\|_{\mathrm{op}}\bigr)
\le\min\left(1,\frac{(1+\sqrt2)e}{g}\right),
\tag{A.22.1}
$$

$$
\left(\|\sin\Theta_u\|_F^2+\|\sin\Theta_v\|_F^2\right)^{1/2}
\le\frac{\sqrt2\min(\sqrt r e,\|E\|_F)}{g}.
\tag{A.22.2}
$$

Consequently each left/right projector satisfies the operator bound (A.22.1), and

$$
\left(\|\Pi_B^u-\Pi_{B'}^u\|_F^2+\|\Pi_B^v-\Pi_{B'}^v\|_F^2\right)^{1/2}
\le\frac{2\min(\sqrt r e,\|E\|_F)}{g}.
\tag{A.22.3}
$$

These are a Wedin residual bound in Frobenius norm and a conservative dimension-independent operator-norm version. The constants are specified because different formulations of Wedin use different separation conditions.

*Proof.* The singular-value min–max principle gives $|\sigma_i(Q')-\sigma_i(Q)|\le e$. Thus selected singular values $\tau_i$ of $Q'$ are separated by at least $g$ from all complementary singular values $s_j$ of $Q$, and the selected block remains isolated.

Choose full SVDs of both matrices, writing $U',V',T$ for the selected singular triples of $Q'$ and $U_2,V_2,S$ for the complementary triples of $Q$. Define

$$R=QV'-U'T=-EV',\qquad Z=Q^\top U'-V'T=-E^\top U',$$

$$A=U_2^\top U',\quad B=V_2^\top V',\quad R_2=U_2^\top R,\quad Z_2=V_2^\top Z.$$

Their equations are $SB-AT=R_2$ and $SA-BT=Z_2$. For each entry the coefficient matrix on $(A_{ji},B_{ji})$ has singular values $|s_j-\tau_i|$ and $s_j+\tau_i$, both at least $g$. Summing the entrywise inverse estimates gives

$$\|A\|_F^2+\|B\|_F^2\le g^{-2}(\|R_2\|_F^2+\|Z_2\|_F^2)
\le 2g^{-2}\min(r e^2,\|E\|_F^2).$$

For completeness, an operator estimate without a hidden rank factor follows by setting $X=(A+B)/2$, $Y=(A-B)/2$:

$$SX-XT=(R_2+Z_2)/2,\qquad SY+YT=(Z_2-R_2)/2.$$

Both right-hand sides have operator norm at most $e$. For ordered spectra separated by $g$, the Sylvester solution is an integral of the form $\int_0^\infty e^{-tM}Re^{tT}\,dt$, with integrand norm at most $e^{-tg}\|R\|_{\mathrm{op}}$. Splitting $S$ above and below the selected interval, applying this bound to each row block, and vertically stacking gives $\|X\|_{\mathrm{op}}\le\sqrt2 e/g$. For $Y$, all diagonal entries are nonnegative, and

$$\min_j s_j+\min_i\tau_i\ge g,$$

since every pair has $s_j+\tau_i\ge|s_j-\tau_i|\ge g$. The integral $\int_0^\infty e^{-tS}[(Z_2-R_2)/2]e^{-tT}\,dt$ therefore gives $\|Y\|_{\mathrm{op}}\le e/g$. Hence $\|A\|_{\mathrm{op}},\|B\|_{\mathrm{op}}\le(1+\sqrt2)e/g$. The matrices $A,B$ measure the left/right sine angles, and the projector identities finish the proof. This argument also covers a separated full zero-singular-value block; it does not identify particular pairings within it. ∎

**What it buys.** Both ends of a coherent block are stable under perturbations of $Q$, without replacing the product by a symmetric matrix or squaring a small singular-value gap. Internal singular multiplicity is harmless for the two projectors.

**Tractability and sampling.** Apply $Q$ and $Q^\top$ as pairs of layer matvecs; a dense explicit product is unnecessary. A rank-$r$ iterative SVD must also resolve the neighboring singular values; its iteration count is not gap-independent. The same a posteriori gap and perturbation qualifications as A.21 apply. The bound controls subspaces, not a deterministic choice of signed singular vectors.

### A.23. Stability of the normalization, not just of the last SVD

**Theorem A.23 (explicit Lipschitz bounds for symmetric Sinkhorn normalization).** Let $K,K'$ be symmetric, strictly positive $N\times N$ matrices with entries in $[e^{-b},e^b]$, $b\ge0$. They need not be PSD. Let

$$\mathcal S(K)=D_vKD_v=P,\qquad \mathcal S(K')=P',$$

where the row sums are one. The entry-range assumption is scale-relative: $\mathcal S(cK)=\mathcal S(K)$. If the **joint** range of both kernels is $[m,M]$, a common rescaling by $(mM)^{-1/2}$ gives $b=\tfrac12\log(M/m)$; use the correspondingly rescaled kernel difference in the additive bounds. Individual bounded ratios do not by themselves bound the joint range, although each kernel may also be rescaled independently if the difference is recomputed after those rescalings.

Define

$$c_F(b,N)=\min\left(\frac{e^{4b}}N,\frac{e^b}{\sqrt2}\right),\qquad
\ell_N(b)=\min\left(e^{2b},\sqrt{N/2}\right).$$

Then

$$
\|P'-P\|_F\le c_F(b,N)\|K'-K\|_F,
\tag{A.23.1}
$$

$$
\|P'-P\|_{\mathrm{op}}\le\frac{L_b}{N}\|K'-K\|_{\mathrm{op}},\qquad
L_b=e^{3b}+2e^{8b}.
\tag{A.23.2}
$$

If both kernels are PSD, one may use $L_b=e^{3b}+2e^{4b}$, or the capped coefficient

$$\|P'-P\|_{\mathrm{op}}\le c_{\mathrm{PSD}}(b,N)\|K'-K\|_{\mathrm{op}},$$

$$c_{\mathrm{PSD}}(b,N)=\min(e^{3b}/N,e^b)
+2\sqrt{\min(1,e^{4b}/N)}\min(e^{2b}/\sqrt N,e^b).$$

For all the kernels under consideration, also use the cancellation-preserving alternatives

$$\|P'-P\|_{\mathrm{op}}\le c_F(b,N)\|K'-K\|_F
\le\sqrt N\,c_F(b,N)\|K'-K\|_{\mathrm{op}}.$$

Take the smallest available bound. The $e^{8b}$ in (A.23.2) is a proof artifact, not an established conditioning rate; that route bounds an amplified scaling-vector component before the component cancels in $\dot P$. Independently,

$$
\|P'-P\|_{\mathrm{op}}\le\|P'-P\|_F
\le\ell_N(b)\|\log K'-\log K\|_{\max},
\tag{A.23.3}
$$

where logarithms are entrywise. Every bound also applies after centering, since $C$ is a contraction. In particular the weaker constants $\min(e^{4b}/N,e^b)$ and $\min(e^{2b},\sqrt N)$ are valid. The sharper $\sqrt{N/2}$ log-Frobenius coefficient is the supremum over positive kernels of the ratio as $b \to \infty$ when $N$ is even (attained asymptotically by the paired-block family below, not at any fixed $b$); the other constants are upper bounds, not claimed optimal.

Along any differentiable positive symmetric kernel path, set $h_{ij}=\dot K_{ij}/K_{ij}$ and $w_i=\dot v_i/v_i$. The exact derivative is

$$
(I+P)w=-(P\odot h)\mathbf1,\qquad
\dot P_{ij}=P_{ij}(h_{ij}+w_i+w_j),
\tag{A.23.4}
$$

and, on a path with the same entry bounds,

$$\|\dot P\|_{\mathrm{op}}\le\|\dot P\|_F\le\ell_N(b)\|h\|_{\max}.$$

*Proof.* Existence and uniqueness of positive symmetric scaling for strictly positive symmetric matrices is the finite Sinkhorn theorem (Sinkhorn, 1964, *Annals of Mathematical Statistics* 35, 876–879; also Theorem A.1). Here are the quantitative steps, including differentiability, so no unquoted robustness theorem is needed.

Put $m=e^{-b}$, $M=e^b$, $s=\sum_i v_i$. From $1=v_i(Kv)_i$,

$$\frac1{Ms}\le v_i\le\frac1{ms},\qquad
\frac NM\le s^2\le\frac Nm.$$

Consequently

$$
\frac{\sqrt m}{M\sqrt N}\le v_i\le\frac{\sqrt M}{m\sqrt N},\quad
\|v\|_2^2\le\frac1m,\quad
\frac{m^2}{M^2N}\le P_{ij}\le\frac{M^2}{m^2N}.
\tag{A.23.5}
$$

The bound on $\|v\|_2^2$ follows from $\sum_i v_i^2\le(\max_i v_i)s\le1/m$.

In log-scaling coordinates $z=\log v$, the Jacobian of the row-sum equations at a solution is $I+P$. It is positive definite, since

$$x^\top(I+P)x=\frac12\sum_{ij}P_{ij}(x_i+x_j)^2>0\quad(x\ne0);$$

the strictly positive diagonal already forces the strict inequality. The implicit-function theorem proves smoothness locally, and uniqueness joins these neighborhoods along any positive path. Differentiating gives (A.23.4).

Define the weighted inner product $\langle a,c\rangle_P=\sum_{ij}P_{ij}a_{ij}c_{ij}$, and put $r_{ij}=\dot P_{ij}/P_{ij}=h_{ij}+w_i+w_j$. Because $\dot P$ is symmetric with zero row and column sums, $r$ is orthogonal in this inner product to every array $a_i+a_j$. In particular, $r$ is orthogonal to $w_i+w_j$. Therefore

$$\sum_{ij}P_{ij}r_{ij}^2\le\sum_{ij}P_{ij}h_{ij}^2.$$

Every $P_{ij}\le1$, so multiplication by $\max P_{ij}$ yields

$$
\|\dot P\|_F^2
\le\min\left(1,\frac{M^2}{m^2N}\right)\sum_{ij}P_{ij}h_{ij}^2.
\tag{A.23.6}
$$

For the linear path $K_t=K+t(K'-K)$, let $H=K'-K$. Since $P_{ij}/K_{ij}^2=v_iv_j/K_{ij}\le M/(m^3N)$, (A.23.6) gives

$$\|\dot P_t\|_F\le\frac{M^{3/2}}{m^{5/2}N}\|H\|_F=\frac{e^{4b}}N\|H\|_F.$$

There is a stronger stochastic-tangent estimate that retains the cancellation. For any row vector $x$ with $\sum_jx_j=0$ and probability row $p_j>0$,

$$\sum_j\frac{x_j^2}{p_j}\ge\left(\sum_j|x_j|\right)^2\ge2\sum_jx_j^2.$$

The first inequality is Cauchy–Schwarz. For the second, the positive and negative parts have the same total $s$, so $\sum x_j^2\le2s^2$ and $(\sum|x_j|)^2=4s^2$. Apply this rowwise to $x_j=\dot P_{ij}$ to obtain

$$2\|\dot P\|_F^2\le\sum_{ij}\frac{\dot P_{ij}^2}{P_{ij}}
\le\sum_{ij}P_{ij}h_{ij}^2.\tag{A.23.7}$$

On the linear path $h_{ij}=H_{ij}/K_{ij}$ and $P_{ij}/K_{ij}^2\le1/m^2$, so $\|\dot P\|_F\le e^b\|H\|_F/\sqrt2$. Combining with the preceding $e^{4b}/N$ estimate and integrating proves (A.23.1). Keeping only the entry cap in (A.23.6) would already give the weaker $e^b$ alternative requested above.

For the geometric path $K_t=\exp((1-t)\log K+t\log K')$, $h=\log K'-\log K$ and $\sum P_{ij}=N$. Equations (A.23.6) and (A.23.7) give respectively $\|\dot P_t\|_F\le e^{2b}\|h\|_{\max}$ and $\|\dot P_t\|_F\le\sqrt{N/2}\|h\|_{\max}$. Their minimum and integration prove (A.23.3) and the derivative claim. Thus the logarithmic bound is genuinely $b$-free at fixed $N$ and does not use $(I+P)^{-1}$.

For the operator bound return to the linear path and set $A=D_vHD_v$. Equation (A.23.4) reads $w=-(I+P)^{-1}A\mathbf1$ and $\dot P=A+D_wP+PD_w$. By (A.23.5),

$$\|A\|_{\mathrm{op}}\le\frac{e^{3b}}N\|H\|_{\mathrm{op}},\qquad
\|A\mathbf1\|_2=\|D_vHv\|_2\le\frac{e^{2b}}{\sqrt N}\|H\|_{\mathrm{op}}.$$

Let $\alpha=e^{-4b}$. The entry lower bound gives $P=\alpha J+(1-\alpha)R$ with $R$ symmetric stochastic when $\alpha<1$. Since its eigenvalues belong to $[-1,1]$, $\lambda_{\min}(I+P)\ge\alpha$. The case $b=0$ has $P=J$ and the same inequality. Thus $\|(I+P)^{-1}\|_{\mathrm{op}}\le e^{4b}$. If $K,K'$ are PSD, the linear path and its congruence $P_t$ are PSD, and this inverse bound improves to $1$.

Finally,

$$\|D_wP\|_{\mathrm{op}}\le\|D_wP\|_F
\le\sqrt{\max_{ij}P_{ij}}\,\|w\|_2
\le\frac{e^{2b}}{\sqrt N}\|w\|_2,$$

where each row's squared sum is at most its maximum entry times its row sum. Combining the estimates gives (A.23.2) pointwise in $t$. For the PSD refinement, also use $v_i^2=P_{ii}/K_{ii}\le e^b$, hence $\|A\|_{\mathrm{op}}\le\min(e^{3b}/N,e^b)\|H\|_{\mathrm{op}}$ and $\|A\mathbf1\|_2\le\min(e^{2b}/\sqrt N,e^b)\|H\|_{\mathrm{op}}$. Cap $\max P_{ij}$ by $1$ in the displayed estimate for $D_wP$ and use $\|(I+P)^{-1}\|\le1$. This gives $c_{\mathrm{PSD}}$; integration proves all the operator alternatives.

To check sharpness of the $b$-free Frobenius coefficient, take even $N$, partition the indices into pairs, set $K_{ij}=e^b$ within each $2\times2$ pair block and $K_{ij}=e^{-b}$ elsewhere. Every row sum is $s_b=2e^b+(N-2)e^{-b}$, so $P=K/s_b$. Let $h$ be $\left(\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\right)$ on each pair block and zero elsewhere. Then $(P\odot h)\mathbf1=0$, $w=0$, and

$$\|\dot P\|_F=\frac{\sqrt{2N}}{2+(N-2)e^{-2b}}\longrightarrow\sqrt{N/2},\qquad\|h\|_{\max}=1.$$

Each finite-$b$ kernel is strictly positive; the derivative is taken in the positive-kernel domain, so its entries need not stay inside the same closed range at the boundary. Difference quotients give the same lower bound on any uniform logarithmic Lipschitz constant. Thus the $b$-free Frobenius coefficient is exact for even $N$ (and in particular the weaker $\sqrt N$ bound is sharp within $\sqrt2$).

No $b$-free absolute additive bound $\|\dot P\|\le C(N)\|H\|$ is possible: at $K=c\mathbf1\mathbf1^\top$, direct differentiation gives $\dot P=CHC/(Nc)$. Taking symmetric $H$ with $H\mathbf1=0$ makes the operator derivative ratio $1/(Nc)$, unbounded as $c\downarrow0$. One may take $c=2e^{-b}$ and sufficiently small perturbations inside the entry range. This scaling obstruction is different from the removable inverse-$(I+P)$ artifact. In relative units with $\kappa_{\min}=\min_{ij}\min(K_{ij},K'_{ij})$, the proved additive alternative is $\|P'-P\|_{\mathrm{op}}\le\sqrt{N/2}\,\|K'-K\|_{\mathrm{op}}/\kappa_{\min}$. ∎

**What it buys.** Bistochastic normalization removes endpoint affinity factors and controls the remaining log-affinity changes without an exponential-in-temperature penalty. These guarantees describe the normalized operator, not the potentially ill-conditioned scaling vector: that vector can move strongly while its contribution to the operator cancels.

**Tractability and sampling.** Dense kernel formation costs $O(N^2d)$ and storage $O(N^2)$; each dense Sinkhorn sweep costs $O(N^2)$, without a fixed iteration-count promise. The additive Frobenius coefficient is tight in scale near $b=0$ ($1/N$) and is capped by $e^b/\sqrt2$ at large $b$. Review searches found $N\|\dot P\|_{\mathrm{op}}/\|H\|_{\mathrm{op}}$ of about $27.8$ ($N=4,b=3$) and $41.1$ ($N=6,b=3$) for general positive kernels, against the old $L_b\approx5.3\times10^{10}$; all tested sphere-kernel values were below $2$. These are attained numerical lower bounds on worst-case constants, not universal upper bounds for all PSD kernels. Indeed, the permanent checks include an explicit **PSD, nonsphere** example giving larger values $38.2659$ and $56.2274$ at $N=4,6$, $b=3$: set $K=m\mathbf1\mathbf1^\top+(M-m)tt^\top$ with $m=e^{-3}$, $M=e^3$, $t=(0,0,1,\ldots,1)$, and use the feasible PSD direction $H=\operatorname{diag}(1,1,0,\ldots,0)$. This is a sum of two PSD matrices and has entries in the prescribed box. The contrast of the first two sites is an eigenvector of $\dot P$ with eigenvalue $v_1^2$, giving the displayed lower bounds (and the computed norms equal them). Thus the review's below-$2$ observation applies to its tested **sphere kernels**, not to arbitrary PSD kernels. The large exponential constant is therefore not evidence that the normalization itself has exponential sensitivity.

At the potentially large $\beta$ of THEORY Section 10 defaults, the old exponential a priori block certificate is vacuous for practical perturbation sizes. Even the corrected $\sqrt{N/2}$-capped certificate can be vacuous for a coarse perturbation budget; it is not uniformly vacuous for arbitrarily small perturbations. An a posteriori certificate using a measured or certified actual operator error and external gap does not inherit that exponential artifact, though a genuinely small gap can still make it uninformative. No site sample complexity follows here. Zeroed/sparsified kernels remain outside this positive-kernel theorem, and sampling a few entries does not certify a maximum error.

### A.24–A.27. Propagating view and temperature changes to the reported objects

**Corollary A.24 (a perturbed view gives a perturbed block).** Assume the same retained site set and unit feature maps $\hat\phi,\hat\phi':X\to\mathbb R^d$ with

$$\max_x\|\hat\phi(x)-\hat\phi'(x)\|_2\le\varepsilon.$$

At a common temperature $\beta\ge0$, form the sphere kernels and their normalized operators. Put

$$h=\beta\min(2\varepsilon,2),\qquad \eta=\ell_N(\beta)h.$$

Then

$$\|P'-P\|_{\mathrm{op}}\le\|P'-P\|_F\le\eta.$$

If a rank-$r$ nonconstant block has external gap $\delta>2\eta$, its corresponding projector obeys

$$
\|\Pi_B'-\Pi_B\|_{\mathrm{op}}
\le\min\left(1,\frac{\eta}{\delta-\eta}\right),\qquad
\|\Pi_B'-\Pi_B\|_F
\le\frac{\sqrt2\eta}{\delta-\eta}.
\tag{A.24.1}
$$

The right-hand sides can be tightened using actual operator and Frobenius errors in A.21 instead of the common upper bound $\eta$.

*Proof.* For unit vectors,

$$|\hat\phi(x)^\top\hat\phi(y)-\hat\phi'(x)^\top\hat\phi'(y)|
\le\|\hat\phi(x)-\hat\phi'(x)\|+\|\hat\phi(y)-\hat\phi'(y)\|\le2\varepsilon,$$

and the difference is also at most $2$. The entrywise log-kernel difference is at most $h$. Apply (A.23.3) with $b=\beta$, then restrict the centered difference to $\mathcal H_0$ and use A.21. ∎

**What it buys.** Nearby normalized activations give nearby isolated representations, by an explicit chain from the measurement to the final block. This is not a theorem that arbitrary upstream parameter perturbations cause small normalized activation changes.

**Tractability and sampling.** The maximum displacement costs $O(Nd)$ once both views are evaluated; the remaining costs are kernel scaling and the spectral calculation. If raw activations have norm at least $a>0$ at both endpoints and maximum raw perturbation $\xi$, then $\varepsilon\le2\xi/a$, by adding and subtracting $\phi'/\|\phi\|$ in the normalization map. There is no such uniform bound near zero: changing the retained sites under [P2] changes the ambient space and is outside this comparison. At small $\beta$ the operator perturbation is $O(\beta\varepsilon)$, but a block's gap can also be $O(\beta)$, so smooth operators do not imply vanishing subspace error.

**Corollary A.25 (temperature derivatives and Lipschitz continuation).** Fix the unit view and write $G_{xy}=\hat\phi(x)^\top\hat\phi(y)$. The map $\beta\mapsto P_\beta=\mathcal S(e^{\beta G})$ is smooth for finite $\beta\ge0$ and

$$
\|\partial_\beta P_\beta\|_{\mathrm{op}}\le\|\partial_\beta P_\beta\|_F\le\ell_N(\beta).
$$

For $0\le\beta_0\le\beta_1$,

$$
\|P_{\beta_1}-P_{\beta_0}\|_F
\le\int_{\beta_0}^{\beta_1}\ell_N(t)\,dt
\le\min\left(\frac{e^{2\beta_1}-e^{2\beta_0}}2,\sqrt{N/2}|\beta_1-\beta_0|\right).
\tag{A.25.1}
$$

A block that remains isolated along an interval has a smooth projector, even if internal eigenvalues cross. At a point with gap $\delta(\beta)>0$,

$$
\|\partial_\beta\Pi_B\|_{\mathrm{op}}\le\frac{\|\partial_\beta\tilde P_\beta\|_{\mathrm{op}}}{\delta(\beta)},\quad
\|\partial_\beta\Pi_B\|_F\le\frac{\sqrt2\min(\sqrt r\|\partial_\beta\tilde P_\beta\|_{\mathrm{op}},\|\partial_\beta\tilde P_\beta\|_F)}{\delta(\beta)}.
\tag{A.25.2}
$$

In particular the operator and Frobenius bounds are at most $\ell_N(\beta)/\delta(\beta)$ and $\sqrt2\ell_N(\beta)/\delta(\beta)$, respectively. If $\delta(\beta)\ge\delta_*>0$ on the interval, integration bounds the projector operator/Frobenius differences by $\delta_*^{-1}\int\ell_N(t)\,dt$ and $\sqrt2\delta_*^{-1}\int\ell_N(t)\,dt$, respectively. For a single finite temperature change one may instead use A.21 with (A.25.1), provided the resulting error is less than half the initial gap.

*Proof.* Smoothness and (A.23.4) apply with $h=G$, whose maximum absolute entry is at most one. This proves the operator derivative and its integral. A local contour enclosing only the selected real eigenvalues gives

$$\Pi_B(\beta)=\frac1{2\pi\mathrm i}\oint_\Gamma(zI-\tilde P_\beta)^{-1}\,dz.$$

For nearby $\beta$ the contour stays outside the spectrum by continuity of eigenvalues, so entrywise differentiation under this finite contour integral proves smoothness. Apply A.21 to $\tilde P_{\beta+h}-\tilde P_\beta=h\partial_\beta\tilde P_\beta+o(h)$, divide the projector bounds by $|h|$, and let $h\to0$. This gives (A.25.2) and the stated integral estimate. ∎

**What it buys.** Temperature changes the normalized operator continuously, with a uniform-in-temperature slope bound at fixed $N$. Continuation of a particular block is certified only while its external gap is resolved; internal crossings do not matter. At the large temperatures allowed by Section 10 defaults, the old exponential a priori certificate is vacuous for ordinary grid steps: already the step $\beta=5\to6$ gives an error upper bound $(e^{12}-e^{10})/2>70{,}000$, while a nonconstant PSD Markov gap is at most $1$. Actual operator errors and resolved gaps can still give a useful a posteriori certificate; for example an actual error below half the actual external gap meets A.21 regardless of temperature.

**Tractability and sampling.** Equation (A.23.4) computes the derivative using one linear solve with $I+P$ after scaling; for the PSD sphere kernel its eigenvalues lie in $[1,2]$. A full dense solve costs $O(N^3)$, while iterative products cost $O(N^2)$ each. A finite grid of observed positive gaps does not certify a positive gap between grid points. For a step of width $h$ from a point with gap $\delta$, the whole step is certified if $2\int_{\beta}^{\beta+h}\ell_N(t)\,dt<\delta$, by Weyl throughout the interval. This sufficient grid-spacing condition may be restrictive. A computed derivative at a single endpoint is not a bound over the interval; using such values for a sharper certificate needs an additional certified derivative variation or second-derivative bound, not supplied here. At $\beta=0$, $\tilde P=0$, so no proper nonconstant block is separated. At large $\beta$ the proved slope cap does not grow further, but disappearing external gaps can still defeat identification. Smaller observed derivatives can improve interpretation; without interval control they do not certify between-grid behavior. No monotonicity of individual eigenvalues or gaps is claimed.

**Corollary A.26 (cross-layer coherent blocks inherit layerwise stability).** Let $P_1,P_2,P_1',P_2'$ be symmetric bistochastic operators on the same sites, and put $Q=P_2P_1$, $Q'=P_2'P_1'$. Assume

$$\|P_j'-P_j\|_{\mathrm{op}}\le e_j,\qquad
\|P_j'-P_j\|_F\le f_j.$$

Then

$$\|Q'-Q\|_{\mathrm{op}}\le e_1+e_2,\qquad
\|Q'-Q\|_F\le f_1+f_2.$$

The same bounds hold for centered restrictions. Hence A.22 applies to each nonconstant singular block whenever $e_1+e_2<\delta/2$, with $e=e_1+e_2$ and Frobenius error $f_1+f_2$ as upper bounds. For a product of $m$ bistochastic contractions the corresponding bounds are $\sum_{j=1}^m e_j$ and $\sum_{j=1}^m f_j$.

*Proof.* A bistochastic nonnegative matrix has operator norm at most $\sqrt{\|P\|_1\|P\|_\infty}=1$. The exact telescoping identity

$$Q'-Q=(P_2'-P_2)P_1+P_2'(P_1'-P_1)$$

therefore gives both claims by the operator and Frobenius ideal inequalities. Repeated telescoping proves the $m$-factor assertion. Multiplication by $C$ cannot increase either norm. ∎

**What it buys.** Layerwise errors accumulate additively, without a spurious extra $e_1e_2$ term, because both endpoints remain contractions. A coherent block's certification needs the singular gap of the product, not a gap in every layer individually.

**Tractability and sampling.** Error budgets can be attached to each kernel evaluation or approximate matvec, then combined before the product SVD. A product matvec uses $m$ layer matvecs; no $N\times N$ product need be stored. A stable layer spectrum does not imply a large product gap. Approximate operators that are not contractions need their actual norm factors in the telescoping bound; the displayed contraction bound is not automatic for unfinished Sinkhorn iterates.

**Proposition A.27 (continuity of block pressure and centered alignment).** Let $\Pi,\Pi'$ be equal-rank block projectors with $\rho_\Pi=\|\Pi'-\Pi\|_{\mathrm{op}}$, and let $U,U'\in\mathbb R^{N\times p}$, $\Delta,\Delta'\in\mathbb R^{N\times q}$ be same-site input and error matrices. Then

$$
\left|\|\Delta'^\top\Pi'U'\|_F-\|\Delta^\top\Pi U\|_F\right|
\le\|\Delta'-\Delta\|_F\|U'\|_{\mathrm{op}}
+\|\Delta\|_{\mathrm{op}}\rho_\Pi\|U'\|_F
+\|\Delta\|_{\mathrm{op}}\|U'-U\|_F.
\tag{A.27.1}
$$

In particular, for fixed inputs and errors the bound is $\|\Delta\|_{\mathrm{op}}\|U\|_F\rho_\Pi$ (or $\|\Delta\|_F\|U\|_{\mathrm{op}}\rho_\Pi$, whichever is smaller). The same statement holds for a left or right coherent-block projector. For multiple uses of a weight, sum these bounds over the use slots.

For nonzero centered operators $A,B$, define $a(A,B)=\langle A,B\rangle_F/(\|A\|_F\|B\|_F)$. For arbitrary $A',B'$, set $\eta_A=\|A'-A\|_F$, $\eta_B=\|B'-B\|_F$. Whenever $A',B'$ are nonzero,

$$
|a(A',B')-a(A,B)|
\le 2\frac{\eta_A}{\|A\|_F}+2\frac{\eta_B}{\|B\|_F}.
\tag{A.27.2}
$$

One sufficient condition ensuring the perturbed statistic is defined is $\eta_A<\|A\|_F$, $\eta_B<\|B\|_F$. Thus a centered Frobenius-norm floor, not a spectral gap, is the conditioning requirement for alignment.

*Proof.* Subtract the two pressure matrices and telescope in the order

$$
(\Delta'-\Delta)^\top\Pi'U'
+\Delta^\top(\Pi'-\Pi)U'
+\Delta^\top\Pi(U'-U).
$$

The reverse triangle inequality for Frobenius norms and $\|XYZ\|_F\le\|X\|_{\mathrm{op}}\|Y\|_{\mathrm{op}}\|Z\|_F$ (with the Frobenius factor moved when needed) give (A.27.1). Summing slots uses the same triangle inequality, not orthogonality of gradient pieces.

For any nonzero $A,A'$, adding and subtracting $A'/\|A\|_F$ gives

$$\left\|\frac{A'}{\|A'\|_F}-\frac A{\|A\|_F}\right\|_F\le2\frac{\|A'-A\|_F}{\|A\|_F}.$$

Apply this to each factor of the inner product between unit-Frobenius matrices and use Cauchy–Schwarz. The reverse triangle inequality proves the sufficient nonvanishing condition. ∎

**What it buys.** The construction's basis-invariant reports inherit the block guarantees: pressure is continuous when the projector and the actual input/error data are controlled. Centered alignment is stable away from zero centered signal even without an eigengap; it is not uniformly stable at a vanishing signal.

**Tractability and sampling.** With $\Pi=\Psi_B\Psi_B^\top$, compute pressure as $(\Delta^\top\Psi_B)(\Psi_B^\top U)$, costing $O(Nr(p+q)+rpq)$ without forming an $N\times N$ projector. Dense alignment costs $O(N^2)$ once the operators are available. Counting-measure $U,\Delta$ norms can grow with the number of sites; these are bounds for the stated summed gradient, not an automatically normalized population gradient. The theory of small view perturbations alone says nothing about changes in $U$ or backpropagated $\Delta$; those need their own bounds. At $\beta=0$ centered alignment is undefined. Its nonzero-$\beta$ limit requires the separate low-temperature expansion and nonzero centered Gram matrices.

### A.28. Why the gap and the block cannot be omitted

**Proposition A.28 (no gap, no vector identifiability).** If a symmetric operator has an eigenvalue of multiplicity $r>1$, every orthogonal rotation of an eigenbasis of that eigenspace is another eigenbasis. The projector onto the entire eigenspace is uniquely determined; a cut strictly inside it is not. There is no perturbation bound for individual eigenvector lines, or for such a cut, uniform as the relevant gap tends to zero—even among strictly positive PSD bistochastic matrices.

More explicitly, there are $3\times3$ strictly positive PSD bistochastic matrices $P_\delta,P_\delta'$ with identical spectra $\lbrace1,a+\delta,a\rbrace$, such that

$$\|P_\delta'-P_\delta\|_{\mathrm{op}}=\delta|\sin\theta|,\qquad
\|\Pi_{\mathrm{top},0}'-\Pi_{\mathrm{top},0}\|_{\mathrm{op}}=|\sin\theta|,$$

where $\Pi_{\mathrm{top},0}$ is the top nonconstant one-dimensional projector and $\theta$ is any fixed angle. The whole two-dimensional nonconstant block is unchanged. Analogous nonuniqueness holds for a repeated positive singular value: left and right singular bases can be rotated by the same orthogonal matrix.

*Proof.* For a repeated eigenvalue, $A\Psi=\lambda\Psi$ implies $A\Psi R=\lambda\Psi R$ for every orthogonal $R$, and $(\Psi R)(\Psi R)^\top=\Psi\Psi^\top$. A partial-rank projector inside that eigenspace changes under such rotations.

For the quantitative construction choose $a=1/5$, an orthonormal basis $T$ of $\mathbf1^\perp\subset\mathbb R^3$, and a planar rotation $R_\theta$. Let

$$P_\delta=J+T\operatorname{diag}(a+\delta,a)T^\top,\qquad
P_\delta'=J+TR_\theta\operatorname{diag}(a+\delta,a)R_\theta^\top T^\top.$$

For $0<\delta<1/10$ both matrices are PSD, have row sums one, and are strictly positive: write either as $aI+(1-a)J+\delta ff^\top$ with $f$ a unit nonconstant vector, so an off-diagonal entry is at least $(1-a)/3-\delta/2>0$, using $|f_if_j|\le1/2$. Their difference on $\mathbf1^\perp$ is $\delta$ times the difference of two line projectors at angle $\theta$, which has norm $\delta|\sin\theta|$. The nonconstant eigenline angle remains $|\theta|$ modulo principal-angle conventions, and its sine does not shrink as $\delta\to0$. Positive repeated singular values admit the simultaneous rotation $U_B\mapsto U_BR$, $V_B\mapsto V_BR$ because $R(\sigma I)R^\top=\sigma I$. ∎

**What it buys.** “Report blocks, not vectors” is an identifiability statement with a counterexample, not just a plotting preference. A declared rotation rule [P4] supplies coordinates within a block; it does not make those coordinates intrinsic to the original operator or automatically stable.

**Tractability and sampling.** Determining whether an internal gap is resolved requires comparing it with numerical and statistical error, not testing floating-point equality. No increase in computational accuracy resolves genuine multiplicity. Localizing rotations can have their own ties and discontinuities; uniqueness and stability of a rotation require additional assumptions, not contained in Davis–Kahan. The counterexample is already three-dimensional and does not depend on sample size.

### A.29–A.31. Noise cancellation: what transfers to the sphere kernel?

**Proposition A.29 (exact diagonal-gauge cancellation and an approximate version).** Let $K$ be symmetric strictly positive, and $a_i>0$. Then

$$\mathcal S(D_aKD_a)=\mathcal S(K).$$

More generally suppose

$$K'=D_a\bigl(K\odot e^R\bigr)D_a,$$

with symmetric $R$, $\|R\|_{\max}\le\eta$, and $K_{ij}\in[e^{-b},e^b]$. Then

$$\|\mathcal S(K')-\mathcal S(K)\|_F\le\ell_N(b+\eta)\eta.$$

The exact invariance also holds for symmetric zero-diagonal matrices with all off-diagonal entries positive and $N\ge3$, for which positive symmetric scaling exists uniquely. The quantitative displayed estimate is for the strictly positive case only.

*Proof.* If $v$ scales $K$, then $v/a$ scales $D_aKD_a$ to the same $P$, proving invariance by uniqueness. Apply it to remove $D_a$ in the approximate claim. The entries of $K\odot e^R$ and $K$ lie in $[e^{-(b+\eta)},e^{b+\eta}]$ and their log difference is $R$; A.23.3 gives the bound. For the hollow case, the same algebra applies; existence and uniqueness are Lemma 1 of Landa–Coifman–Kluger (2021), with the stated $N\ge3$, symmetry and positive off-diagonal support checking all its hypotheses. ∎

**What it buys.** Normalization removes arbitrary positive per-site multiplicative affinity factors exactly, even if they are large and heterogeneous. The only error that remains is the part of the log affinity that cannot be written as a sum of one term per endpoint.

**Tractability and sampling.** The cancellation is algebraic and needs no fitted noise variances. An empirical claim that a real noise process has this form still needs either a generative assumption or a measured residual bound. The hollow-kernel version is a different estimand and need not be PSD; it cannot inherit the dense sphere kernel's nonnegative-spectrum claims.

**Theorem A.30 (the precise Landa–Coifman–Kluger transfer).** Fix $N\ge3$ and $\beta>0$. For each ambient dimension $d$, let deterministic clean points $x_i^{(d)}\in\mathbb R^d$ satisfy $\|x_i^{(d)}\|\le1$. Observe $y_i^{(d)}=x_i^{(d)}+\eta_i^{(d)}$, where the noises are pairwise independent, have mean zero, and obey

$$\|\operatorname{Cov}(\eta_i^{(d)})\|_{\mathrm{op}}\le c/d$$

uniformly in $i,d$. Form **hollow Gaussian kernels**

$$K_{ij}=\exp(-\beta\|x_i-x_j\|^2/2),\quad
K'_{ij}=\exp(-\beta\|y_i-y_j\|^2/2)\quad(i\ne j),\qquad K_{ii}=K'_{ii}=0.$$

Then

$$\|\mathcal S(K')-\mathcal S(K)\|_F=O_{\mathbb P}(d^{-1/2}).\tag{A.30.1}$$

The constants may depend on $N,\beta,c$ and on the desired probability level. There is no asserted uniform rate as $N$ or $\beta$ grows.

This is Theorem 3 of Landa, Coifman and Kluger (2021), with their number of points $n=N$, ambient dimension $m=d$, and Gaussian width $\epsilon=2/\beta$. If the clean $x_i$ are unit vectors, their hollow Gaussian kernel differs from the hollow sphere kernel $\exp(\beta x_i^\top x_j)$ by the common factor $e^{-\beta}$, so the two normalized clean operators are exactly equal. For any two unit-vector configurations, the same kernel identity holds on both sides. But this alone supplies no nontrivial additive-noise theorem for normalized observations: if deterministic $x_i$ and almost surely unit $y_i=x_i+\eta_i$ both have unit norm and $\mathbb E\eta_i=0$, then $1=\mathbb E\|y_i\|^2=1+\mathbb E\|\eta_i\|^2$, forcing $\eta_i=0$ almost surely. Thus the original statistical theorem applies nontrivially to the unnormalized noisy Gaussian observations, not directly to two sphere-valued views related by centered additive noise.

*Proof and verification of the reduction.* All the conditions of the cited theorem have been stated: fixed $N>2$, fixed positive width, bounded clean norms, pairwise independence and centered noise with the covariance-operator bound. The kernels have exactly the required zero diagonal and positive off-diagonal entries. No fourth-moment assumption or concentration of each noise norm is needed.

Here is also the cancellation argument, including why it gives the rate. For $i\ne j$ expand the squared distance to obtain

$$K'_{ij}=a_i a_j K_{ij}\exp(R_{ij}),\qquad
 a_i=\exp(-\beta\|\eta_i\|^2/2),$$

$$R_{ij}=-\beta\langle x_i-x_j,\eta_i-\eta_j\rangle+\beta\langle\eta_i,\eta_j\rangle.$$

The linear noise term has variance at most a constant times $d^{-1}$, because $\|x_i-x_j\|\le2$ and the noises are centered and pairwise independent. Independence also gives

$$\mathbb E\langle\eta_i,\eta_j\rangle^2
=\operatorname{tr}(\Sigma_i\Sigma_j)\le d(c/d)^2=c^2/d.$$

Using $(a+b)^2\le2a^2+2b^2$ and Chebyshev's inequality proves $R_{ij}=O_{\mathbb P}(d^{-1/2})$. A union bound over the fixed number of pairs proves $\max_{i\ne j}|R_{ij}|=O_{\mathbb P}(d^{-1/2})$. No bound on the realized $a_i$ is required: A.29 removes them exactly, including in the hollow case.

For clarity, the remaining smoothness step is valid despite the zero diagonal. The clean off-diagonal entries lie in the compact interval $[e^{-2\beta},1]$. On the complete off-diagonal support with $N\ge3$, symmetric scaling exists uniquely by Landa–Coifman–Kluger Lemma 1. Its log-coordinate Jacobian is again $I+P$ and

$$z^\top(I+P)z=\frac12\sum_{i\ne j}P_{ij}(z_i+z_j)^2.$$

This is positive definite: vanishing requires $z_i=-z_j$ on every edge, and any triangle forces all three coordinates to vanish; connectivity forces the others to vanish. The implicit-function theorem therefore gives smooth scaling throughout this positive off-diagonal domain. On a compact neighborhood with off-diagonal entries, for example, in $[e^{-2\beta}/2,2]$, its derivative is uniformly bounded. On the event $\max|R|\le\log2$, the corrected kernel and the line segment to $K$ lie in this neighborhood. Consequently the integral of the derivative along the segment, uniformly bounded on that compact box inside the open smooth-scaling domain, converts its $O_{\mathbb P}(d^{-1/2})$ entry error into the same Frobenius error after scaling. This proves (A.30.1) directly. Finally, the unit-sphere identity

$$e^{\beta x_i^\top x_j}=e^\beta e^{-\beta\|x_i-x_j\|^2/2}$$

and invariance to global scaling prove the stated algebraic identification of sphere and Gaussian kernels. The expectation identity in the statement proves why the direct sphere-valued centered-additive-noise specialization is trivial. ∎

**What it buys.** There is a genuine noise-cancellation theorem behind bistochastic normalization: high-dimensional heterogeneous noise contributes endpoint factors that normalization removes, leaving only small cross terms. The sphere identity transfers the kernel algebra, not an arbitrary statistical noise model.

**Tractability and sampling.** Here $d$, not $N$, is the asymptotic variable. The theorem permits $\mathbb E\|\eta_i\|^2=O(1)$ and heterogeneous variances, while forbidding noise concentrated in a few fixed directions. A block with gap bounded below independently of $d$ then has $O_{\mathbb P}(d^{-1/2}/\delta)$ subspace error by A.21 or A.22; if gaps shrink, the ratio must still vanish. No token-site independence or large-$N$ rate is supplied by this result. Removing the diagonal changes the operator, its PSD status, and potentially its gaps; it must be declared rather than silently used to justify the spine.

**Proposition A.31 (two obstacles to importing the noise theorem into the spine).** Neither of the following modifications preserves the conclusion of A.30 in general:

1. keeping the positive kernel diagonal, as the dense sphere-kernel construction does;
2. applying the sphere kernel to row-normalized noisy observations and assuming that the unnormalized additive-noise hypotheses still apply, even if the diagonal is removed.

These failures occur with bounded clean unit vectors and independent Gaussian noises satisfying the covariance condition in A.30.

*Proof.* For the first claim take fixed $N=4$, all clean points equal to the same unit vector $u$, and independent Gaussian $\eta_i$ in $u^\perp$ with covariance $\tau^2(I-uu^\top)/(d-1)$, where $\tau>0$ is fixed. The covariance norm is $O(d^{-1})$. As $d\to\infty$, $\|\eta_i\|^2\to\tau^2$ and $\langle\eta_i,\eta_j\rangle\to0$ in probability. With the full Gaussian kernel, the clean matrix is all ones, whereas the noisy matrix tends to the matrix with diagonal $1$ and off-diagonal $k=e^{-\beta\tau^2}<1$. Its stochastic normalization has diagonal $1/(1+3k)$, not $1/4$. Thus even homoskedastic noise need not disappear when the diagonal is retained.

The same failure holds for the spine's full sphere kernel after row-normalization: put $z_i=(u+\eta_i)/\|u+\eta_i\|$. Then $z_i^\top z_j\to a^2=(1+\tau^2)^{-1}$ for $i\ne j$, while $z_i^\top z_i=1$. Dividing the sphere kernel by $e^\beta$, its limiting diagonal is $1$ and off-diagonal is $k=e^{-\beta(1-a^2)}<1$, giving the same discrepancy from the clean $J$.

For the second claim allow fixed heterogeneous $\tau_i>0$ and write $a_i=(1+\tau_i^2)^{-1/2}$. With the same Gaussian construction, $z_i^\top z_j\to a_i a_j$ off the diagonal. The clean hollow sphere kernel is constant off-diagonal and scales to $(\mathbf1\mathbf1^\top-I)/3$. If the limiting noisy hollow kernel also scaled to that matrix, it would be diagonally equivalent to the clean kernel. Endpoint diagonal factors cancel in the cross ratio $K_{12}K_{34}/(K_{13}K_{24})$, but the limiting log cross-ratio change is

$$\beta(a_1a_2+a_3a_4-a_1a_3-a_2a_4)=\beta(a_1-a_4)(a_2-a_3).$$

Choose, for example, $(a_1,a_2,a_3,a_4)=(0.9,0.8,0.7,0.6)$. This change is nonzero for $\beta>0$, so the two normalized hollow kernels differ. Smoothness of hollow scaling, proved in A.30, transfers convergence of kernel entries to convergence of their scalings, proving nonvanishing limiting error. The map $u+\eta_i\mapsto z_i$ has introduced a systematic angular distortion; the residual $z_i-u$ is not centered additive noise satisfying A.30. ∎

**What it buys.** The honest robustness claim for the actual dense sphere construction is A.23–A.24, or the explicit diagonal-gauge condition A.29—not unconditional immunity to heteroskedastic noise. This separates a real advantage of normalization from an unjustified transfer of a statistical theorem.

**Tractability and sampling.** The two obstructions have different large-$N$ behavior. For the first limiting kernel with diagonal $1$ and off-diagonal $k=e^{-\beta\tau^2}$, direct diagonalization gives

$$P-J=\frac{1-k}{1+(N-1)k}C,\qquad
\|P-J\|_F=\frac{\sqrt{N-1}(e^{\beta\tau^2}-1)}{N+e^{\beta\tau^2}-1}
\sim\frac{e^{\beta\tau^2}-1}{\sqrt N}.$$

The equivalence holds as $N\to\infty$ at fixed $\beta\tau^2$; it is not uniform when $\beta\tau^2$ grows with $N$. Thus the kept-diagonal effect fades with more sites in this family, although the required $N$ can be exponentially large in temperature/noise. Replace $\tau^2$ by $1-a^2$ for its normalized-sphere counterpart.

The row-normalization effect is systematic rather than merely a diagonal nuisance. For the hollow attenuation family in the proof, taking attenuation factors evenly spaced in $[0.6,0.9]$ gives observed Frobenius discrepancies $0.00983,0.00782,0.00754$ at $N=4,50,400$ and $\beta=1$. These finite-$N$ computations show persistence across this range, not by themselves a theorem about its infinite-$N$ limit. There are also families with provably nonvanishing large-$N$ bias: repeat each of four distinct attenuation types $m$ times, including all cross-type and within-type edges except self-loops. If the noisy normalization approached the clean hollow normalization in Frobenius norm, its four-type aggregated transition matrix would approach the uniform-type matrix. Symmetry gives common scaling within each type; for distinct types the normalized per-entry cross ratio remains $\exp(\beta(a_1-a_4)(a_2-a_3))\ne1$. Four-type aggregation multiplies each cross-type entry by the same $m$, preserving this ratio, in contradiction with convergence to uniform entries $1/4$. More explicitly, if $D$ is the difference from clean normalization, Cauchy–Schwarz on each $m\times m$ type block bounds the Frobenius norm of its aggregate matrix by $\|D\|_F$, so vanishing site-level error would indeed imply the asserted aggregate convergence. No subsequence can have vanishing error, proving a positive limiting lower bound.

Neither greater ambient dimension nor numerical accuracy cures these fixed-noise biases. A negligible-diagonal regime addresses the first obstruction, not the second; the latter needs vanishing angular distortion or a verified endpoint-factor model. No labels or sparsity assumptions enter either example.

### Numerical checks performed

The permanent script is `checks/theorems_a2_check.py`, runnable from the repository root as

```bash
uv run --with numpy python -I checks/theorems_a2_check.py
```

It requires only NumPy, uses deterministic seeds/fixtures, and ends with `all A2 checks passed` only if every assertion succeeds. It replaces the original temporary-only check. These numerical checks supplement, not replace, the proofs.

- **Revised normalization constants:** symmetric positive bounded-entry pairs at $N\in\lbrace4,8,17\rbrace$, $b\in\lbrace0.01,0.2,1,2\rbrace$, eight pairs per configuration. Maximum actual/bound ratios were $0.921$ for additive Frobenius, $0.469$ for the minimum of the two operator routes, and $0.501$ for log-Frobenius. PSD-cap checks on sphere-kernel pairs attained $0.322$.
- **Derivative and cancellation:** central differences agreed with (A.23.4) within $2\times10^{-8}$ in Frobenius norm. Derivative row sums vanished, the weighted Pythagorean identity held within $10^{-10}$, and the new rowwise inequality $2\|\dot P\|_F^2\le\sum\dot P_{ij}^2/P_{ij}$ was asserted. Near-bipartite kernels verified $\lambda_{\min}(I+P)=2/(e^{2b}+1)$ and cancellation of pure endpoint gauges for $b$ up to $8$.
- **Sharp log-Frobenius family:** paired blocks at $b=8$ gave slopes $1.4142134,1.9999993,2.8284249,3.9999932$ for $N=4,8,16,32$, within $10^{-5}$ relative error of the exact uniform constants $\sqrt{N/2}$. The finite-$b$ formula in the proof was asserted to $10^{-12}$.
- **Additive worst-case regimes:** an almost-zero-$b$ fixture approached the $1/N$ Frobenius coefficient within one part per million. Explicit PSD nonsphere kernels at $b=3$ gave $N\|\dot P\|_{\mathrm{op}}/\|H\|_{\mathrm{op}}=38.2659256$ and $56.2273672$ at $N=4,6$. The fixtures check the improved bounds, PSD, feasible one-sided finite differences, and the distinction between arbitrary PSD and sphere-kernel observations. They exceed, rather than attempt to reproduce, the review's random-search lower bounds $27.8$ and $41.1$.
- **Spectral/statistic bounds:** ninety symmetric and ninety nonsymmetric perturbation cases verified Davis–Kahan with operator constant $1$ and the stated Wedin bounds. Maximum actual/bound ratios were $0.629/0.445$ for Davis–Kahan operator/Frobenius, $0.194/0.396$ for Wedin, $0.256$ for view displacement, $0.570$ for the exact capped-temperature integral, $0.336$ for pressure and $0.164$ for alignment. The no-gap counterexample, exact gauge invariance and residual-gauge bound were also asserted.
- **Different noise asymptotics:** the kept-diagonal formula gave $0.520463,0.168079,0.0171790$ at $N=4,100,10000$, with its $N^{-1/2}$ asymptotic asserted. The hollow normalized-sphere discrepancies were $0.00983493,0.00781857,0.00753939$ at $N=4,50,400$. Replicated four-type examples checked the invariant nontrivial cross ratio and the aggregation norm inequality used to prove persistence.

The existing `checks/cm_check.py` also passed. No changes were made to THEORY.md or the existing check script.

### References and exact reliance

- Bhatia, R. (1997). *Matrix Analysis*. Springer, Graduate Texts in Mathematics 169, [Theorem VII.3.1](https://doi.org/10.1007/978-1-4612-0653-8): interval-separated spectral subspaces and the Davis–Kahan bound with operator constant $1$. Exact reliance in A.21; its interval hypothesis is checked there.

- Davis, C., and Kahan, W. M. (1970). “The rotation of eigenvectors by a perturbation. III.” *SIAM Journal on Numerical Analysis* **7**(1), 1–46. [DOI](https://doi.org/10.1137/0707001). A.21 uses the interval sin-$\Theta$ theorem with constant $1$, with its hypotheses verified explicitly.
- Yu, Y., Wang, T., and Samworth, R. J. (2015). “A useful variant of the Davis–Kahan theorem for statisticians.” *Biometrika* **102**(2), 315–323. [arXiv:1405.0680](https://arxiv.org/abs/1405.0680). Theorem 2 (arXiv numbering) is a standard population-gap Frobenius alternative; no additional conclusion from it is needed here.
- Wedin, P.-Å. (1972). “Perturbation bounds in connection with singular value decomposition.” *BIT Numerical Mathematics* **12**, 99–111. [DOI](https://doi.org/10.1007/BF01932678). A.22 proves the residual argument and the particular constants used here, rather than silently importing a different gap convention.
- Sinkhorn, R. (1964). “A relationship between arbitrary positive matrices and doubly stochastic matrices.” *The Annals of Mathematical Statistics* **35**(2), 876–879. [DOI](https://doi.org/10.1214/aoms/1177703591). Standard positive matrix scaling; existence/uniqueness is also Theorem A.1. Symmetry gives a symmetric scaling, whose positive vector is unique.
- Landa, B., Coifman, R. R., and Kluger, Y. (2021). “Doubly Stochastic Normalization of the Gaussian Kernel Is Robust to Heteroskedastic Noise.” *SIAM Journal on Mathematics of Data Science* **3**(1), 388–413. [DOI](https://doi.org/10.1137/20M1342124); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8194191/); [arXiv:2006.00402](https://arxiv.org/abs/2006.00402). Exact reliance: Lemma 1 for existence/uniqueness on positive complete off-diagonal support; Theorem 3 is stated with its actual fixed-$N$, growing-ambient-dimension assumptions in A.30, which also gives the cancellation proof.

The direct derivative proof in A.23 replaces a possible Hilbert-projective-metric/Franklin–Lorenz route. It gives explicit norm constants without needing to quote an unstated contraction theorem.

### Discrepancies with THEORY.md

1. **External gap, restricted space, and matched rank must be specified.** THEORY Section 3.2's $O(\|E\|/\delta)$ statement is correct for isolated blocks. A.20–A.22 make precise the external gap, the same-index continuation, and the restriction to $\mathbf1^\perp$ after removal of the constant mode. A fixed numerical cutoff can change rank and is not covered without its own threshold margin.
2. **SVD stability does not by itself prove stability of the construction.** Kernel formation and Sinkhorn normalization intervene before the final spectrum. A.23 supplies the missing quantitative link. Dense strict positivity matters; these constants do not automatically cover the separate $k$-NN estimand or a sign-indefinite Nyström approximation.
3. **The noise theorem does not apply unchanged to the spine.** Landa–Coifman–Kluger uses a zero diagonal and additive high-dimensional noise before any row-normalization. THEORY's dense kernel retains its diagonal and normalizes each view to the sphere. A.31 disproves an unqualified transfer; the identity between sphere and chordal Gaussian kernels transfers algebra, not these statistical hypotheses.
4. **Centered alignment needs a nonzero signal condition.** Section 8's formula is undefined when either centered operator vanishes, including $\beta=0$ and a view constant across sites. Its low-temperature CKA limit also needs both centered normalized-feature Gram matrices nonzero. A.27 gives the appropriate norm-floor condition for perturbation stability.
5. **A stable block does not imply stable training pressure without stable inputs and errors.** Section 7.2's rotation invariance is exact, but perturbation continuity includes $U$ and $\Delta$ terms as well as the projector term. A.27 keeps them explicit.
6. **[P4] is a coordinate convention unless extra recovery assumptions are added.** At exact multiplicity the full isolated eigenspace is intrinsic but an individual basis is not. A declared rotation can itself have ties; no stability of that rotation follows merely from the external spectral gap. The distinction between a display rotation and an empirically validated dictionary-recovery algorithm in Sections 3.2 and 4.3 should remain explicit.

---
## Part A, cluster 3 — Sample complexity, estimation and cost

This cluster separates **statistical error** (sites), **numerical approximation** (landmarks/features and solver tolerance), and **measurement error** (stochastic resamples or Jacobian probes). They are not interchangeable sample sizes. Confidence levels satisfy $0<\eta\le1/2$. All population statements below fix the view, model parameters, temperature and sampling distribution before drawing sites. Choosing these on the same data requires a uniform bound, a finite-search union bound, or an independent split. The covariance event in A.47 is already uniform over ridge values for fixed views and windows, so its sampling bound permits data-selected ridge values; a zero-ridge consistency claim still needs the selected schedule to satisfy the stated conditions.

Indexed $\Sigma$ symbols denote RKHS covariance operators; $C$ remains THEORY's site-centering matrix. The kernel ratio $\chi_\beta$ is distinct from the ridge smoothers $R_p,R_q$, and $B_\beta$ is a kernel bound, not a reference-set size.

### Population and empirical operators

**Definition A.40 (fixed-temperature population estimand).** Let $S\subseteq\mathbb S^{d-1}$ be the compact support of the distribution $\mu$ of the normalized view $\hat\phi$. Assume **independent sites**: $z_1,\ldots,z_N\sim\mu$ independently, and let $\mu_N=N^{-1}\sum_j\delta_{z_j}$. Repeated view values are allowed; the sites remain distinct indices. Set

$$k(z,w)=e^{\beta z^\top w},\quad a=e^{-\beta},\quad b=e^\beta,\quad \chi_\beta=b/a=e^{2\beta},\qquad \beta<\infty.$$

The unnormalized integral operator is $T_k f(z)=\int k(z,w)f(w)\,d\mu(w)$ on $L^2(\mu)$. Its RKHS is $\mathcal G$, with feature $\Phi(z)=k(z,\cdot)$ and $\|\Phi(z)\|^2=b$. We use $\mathcal G$ only to compare operators on a **common space**, not as an additional learned representation.

If $s_\mu>0$ solves

$$s_\mu(z)\int k(z,w)s_\mu(w)\,d\mu(w)=1,$$

define $p_\mu(z,w)=s_\mu(z)k(z,w)s_\mu(w)$ and $P_\mu f=\int p_\mu(\cdot,w)f(w)\,d\mu(w)$. Define $s_N=s_{\mu_N}$ on all of $S$ by the same equation. The finite matrix of THEORY is exactly

$$P_N[i,j]=\frac1N s_N(z_i)k(z_i,z_j)s_N(z_j),\qquad v_i=\frac{s_N(z_i)}{\sqrt N}.$$

Thus the continuum operator uses probability measure, whereas finite eigenvectors $\psi_i$ retain THEORY's counting-measure norm $\|\psi_i\|_2=1$. A population-normalized sample function has values $\sqrt N\,\psi_i(z_j)$, not $\psi_i(z_j)$.

These statements also apply to a data-generating process that supplies independent past/target pairs or independent documents with one declared sample per document. They do **not** make overlapping token windows independent. No effective sample size for arbitrary dependent streams is asserted.

**Theorem A.41 (continuous symmetric Sinkhorn normalization).** Under Definition A.40, there is a unique positive continuous $s_\mu$ satisfying the scaling equation. It obeys

$$\frac{\sqrt a}{b}\le s_\mu(z)\le\frac{\sqrt b}{a},\qquad
0<p_\mu(z,w)\le B_\beta:=\frac{b^2}{a^2}=e^{4\beta}.$$

The same bounds hold for $s_N$, including its extension to all of $S$. The operator $P_\mu$ is compact, positive semidefinite, self-adjoint and Markov on $L^2(\mu)$, with norm one, simple constant eigenfunction and all eigenvalues in $[0,1]$.

*Proof.* For any probability $\nu$ on $S$, define on $C(S)$

$$F_\nu(f)(z)=-\log\int k(z,w)e^{f(w)}\,d\nu(w).$$

Write $\operatorname{osc}(f)=\sup f-\inf f$. The derivative of $F_\nu$ in direction $h$ is minus the expectation of $h$ under the probability with density proportional to $k(z,w)e^{f(w)}$. Each such probability dominates $(a/b)\,e^f\nu/\int e^f d\nu$. Subtracting this common measure shows that the difference of the expectations at two values of $z$ is at most $(1-a/b)\operatorname{osc}(h)$. Integrating the derivative along the segment between $f$ and $g$ gives

$$\operatorname{osc}(F_\nu(f)-F_\nu(g))\le\kappa\operatorname{osc}(f-g),\qquad \kappa=1-a/b<1.$$

The quotient $C(S)/\mathbb R\mathbf1$, with oscillation norm, is complete; $F_\nu$ is well-defined there because $F_\nu(f+c)=F_\nu(f)-c$. Banach's fixed-point theorem gives a unique fixed class. If its representative satisfies $F_\nu(f)=f+c$, replacing $f$ by $f+c/2$ gives an actual fixed point. Two actual fixed points in the same class differ by a constant equal to its negative, hence coincide. Their exponentials give the desired continuous positive scaling, even if $\nu$ is empirical.

Put $m=\int s_\nu d\nu$. The equation gives $1/(bm)\le s_\nu\le1/(am)$; integration gives $b^{-1/2}\le m\le a^{-1/2}$, proving the displayed bounds. The kernel $p_\mu$ is bounded, hence Hilbert–Schmidt, symmetric and PSD by multiplication of the PSD kernel $k$ by positive factors. Its rows integrate to one, and symmetry makes $\mu$ stationary. Jensen's inequality gives $\|P_\mu f\|_2\le\|f\|_2$. If $P_\mu f=f$, equality in this Jensen inequality forces $f$ to be constant $\mu$-almost surely because every transition density is strictly positive. The constant has eigenvalue one. ∎

**What it buys.** The finite matrix estimates a precisely defined, measure-preserving population object; no manifold or semantic labels are needed. This is the continuous Sinkhorn construction, not the separate small-bandwidth generator limit.

**Tractability and sampling.** Existence costs no additional model fitting. The worst-case bound $B_\beta=e^{4\beta}$ is deliberately conservative: high resolution can make rare regions and normalization expensive. Marshall–Coifman (2019) study the additional manifold limit; Borwein–Lewis–Nussbaum (1994) and Nutz–Wiesel (2023) cover broader continuous scaling settings. The elementary bounded-kernel proof above avoids importing unverified general hypotheses.

**Lemma A.42 (a reusable Hilbert-space sampling bound).** If $Y_1,\ldots,Y_N$ are i.i.d. in a separable real Hilbert space and $\|Y_j\|\le L$ almost surely, then, with probability at least $1-\eta$,

$$\left\|\frac1N\sum_jY_j-\mathbb EY\right\|\le L h_N(\eta),\qquad
h_N(\eta):=\frac{1+\sqrt{2\log(1/\eta)}}{\sqrt N}.$$

*Proof.* Independence and centering give expected squared error at most $L^2/N$, hence expected error at most $L/\sqrt N$. Changing one sample changes the error norm by at most $2L/N$. The bounded-differences inequality gives tail $\exp(-Nt^2/(2L^2))$ above the expectation. Substitute $t=L\sqrt{2\log(1/\eta)/N}$. Bounded differences itself follows by the Doob martingale and the scalar Hoeffding exponential bound for each bounded martingale difference. ∎

**What it buys.** One dimension-free bound controls kernel means, covariance operators and cross-covariance operators; the relevant dimension enters through gaps and effective ranks instead.

**Tractability and sampling.** Error $\epsilon$ requires $N\ge L^2(1+\sqrt{2\log(1/\eta)})^2/\epsilon^2$. The lemma requires independent draws, not merely many rows, and uniform boundedness, not a claim of finite variance alone.

**Theorem A.43 (unnormalized spectral convergence).** Let

$$\Sigma_k=\mathbb E[\Phi(z)\otimes\Phi(z)],\qquad
\widehat \Sigma_k=\frac1N\sum_j\Phi(z_j)\otimes\Phi(z_j).$$

With probability at least $1-\eta$,

$$\|\widehat \Sigma_k-\Sigma_k\|_{\mathrm{op}}\le\|\widehat \Sigma_k-\Sigma_k\|_{\mathrm{HS}}\le b h_N(\eta).$$

The nonzero spectra of $\Sigma_k,T_k$ coincide; those of $\widehat \Sigma_k,K/N$ coincide. Consequently all ordered eigenvalues, padded by zero, differ by at most $b h_N(\eta)$. An isolated contiguous finite population block of nonzero eigenvalues with gap $\delta$ has, after lifting to $\mathcal G$, projector error at most $4b h_N(\eta)/\delta$ when $b h_N(\eta)\le\delta/4$.

*Proof.* Apply Lemma A.42 in the Hilbert space of Hilbert–Schmidt operators: $\|\Phi\otimes\Phi\|_{\mathrm{HS}}=b$. If $A:\mathcal G\to L^2(\mu)$ sends $u$ to $\langle\Phi(\cdot),u\rangle$, then $A^*A=\Sigma_k$ and $AA^*=T_k$. The analogous empirical evaluation map, with a factor $1/\sqrt N$, gives $\widehat \Sigma_k$ and $K/N$. The $A^*A,AA^*$ eigenvalue identity and the min–max principle give the eigenvalue conclusion. Here is the Hilbert-space block argument, rather than an appeal to a finite-matrix theorem. For bounded self-adjoint operators $T,\widehat T=T+E$, let a finite spectral block of $T$ lie in $[l,u]$, separated from its complement by $\delta$, with $\|E\|=\epsilon\le\delta/4$. The spectral path $T+tE$ keeps the surrounding gaps open, so its Riesz projector has constant finite rank. Decompose the complementary projector into spectral parts below $l-\delta$ and above $u+\delta$. The cross term $X=\Pi_-\widehat\Pi$ satisfies a Sylvester equation with right-hand side of norm at most $\epsilon$. Its solution is the norm-convergent semigroup integral
$X=\int_0^\infty e^{t(T_- -cI)}Y e^{-t(\widehat T_B-cI)}dt$,
with a suitable sign in $Y$ and $c$ between the separated spectra. The integrand norm is at most $\epsilon e^{-t(\delta-\epsilon)}$, hence $\|X\|\le\epsilon/(\delta-\epsilon)$. The upper complementary part is identical with signs reversed. Summing yields
$\|(I-\Pi)\widehat\Pi\|\le2\epsilon/(\delta-\epsilon)\le4\epsilon/\delta$.
Equal finite projector ranks imply $\|\Pi-\widehat\Pi\|=\|(I-\Pi)\widehat\Pi\|$; this identity reduces to principal angles in the finite-dimensional sum of their ranges. This is the Hilbert-space Davis–Kahan argument (Davis and Kahan 1970), with a deliberately looser constant than the sharper finite-dimensional statement elsewhere in Part A. ∎

**What it buys.** This is a direct bounded-kernel version of the spectral-convergence programme of Koltchinskii–Giné (2000) and Rosasco–Belkin–De Vito (2010). It gives the requested explicit $e^\beta/\sqrt N$ scale before normalization, rather than confusing a matrix spectrum with an integral-operator spectrum.

**Tractability and sampling.** A projector tolerance $\alpha$ is certified by $N\ge16e^{2\beta}(1+\sqrt{2\log(1/\eta)})^2/(\alpha^2\delta^2)$, with the stated small-error condition automatic when $0<\alpha\le1$. Eigenspaces in different sample-size Euclidean spaces cannot be subtracted without this lift or another declared identification. This theorem does not yet include estimated Sinkhorn factors.

**Theorem A.44 (normalized spectral convergence, with normalization cost exposed).** Define the common-space lifts

$$\Sigma_P=\mathbb E[s_\mu(z)^2\Phi(z)\otimes\Phi(z)],\qquad
\widehat \Sigma_P=\frac1N\sum_j s_N(z_j)^2\Phi(z_j)\otimes\Phi(z_j).$$

Their nonzero spectra are respectively those of $P_\mu$ and $P_N$. Let $S_{\max}=\sup_z s_\mu(z)$ and $B_0=bS_{\max}^2\le B_\beta$; $S_{\max}$ is a population quantity, not an assumed absolute constant. Set

$$h=h_N(\eta/2),\quad \epsilon=B_0h,\quad
D=-\log(1-\epsilon),\quad t=(\chi_\beta+\tfrac12)D,$$

and assume $\epsilon<1$. With probability at least $1-\eta$,

$$\|\log s_N-\log s_\mu\|_\infty\le t,$$

$$\|\widehat \Sigma_P-\Sigma_P\|\le E_N:=\epsilon+(e^{2t}-1)(1+\epsilon).$$

Hence ordered eigenvalue errors are at most $E_N$. For an isolated contiguous finite block of nonzero eigenvalues with population gap $\delta$, the lifted projector error is at most $4E_N/\delta$ whenever $E_N\le\delta/4$.

In particular, if $\epsilon\le1/2$ and $(2\chi_\beta+1)\epsilon\le1/2$, then

$$E_N\le(12\chi_\beta+7)B_0 h
=(12e^{2\beta}+7)e^\beta S_{\max}^2h
\le(12e^{2\beta}+7)e^{4\beta}h.$$

Thus the proved population-sensitive bound is $O(S_{\max}^2e^{3\beta}h)$, becoming $O(e^{3\beta}h)$ when $S_{\max}=O(1)$. This does not turn the worst-case $O(e^{6\beta}h)$ bound into an unconditional $O(e^{3\beta}h)$ bound: $S_{\max}$ can grow with temperature.

A further **bounded-normalization-conditioning hypothesis** may improve this constant: if $\sup s_\mu\le S_0$ and, on an event $\mathcal E_{\mathrm{sc}}$ with probability at least $1-\eta/2$, $\|\log s_N-\log s_\mu\|_\infty\le L_0 b h$, then, provided $S_0^2 b h\le1/2$ and $L_0bh\le1/2$,

$$\|\widehat \Sigma_P-\Sigma_P\|\le (S_0^2+6L_0)e^\beta h.$$

The conditional conclusion holds with probability at least $1-\eta$, after union with the covariance event of probability $1-\eta/2$. The sensitivity constant $L_0$ is an additional, unproved hypothesis here. For $S_0,L_0$ bounded independently of $\beta$, this gives an $O(e^\beta\sqrt{\log(1/\eta)/N}/\delta)$ subspace rate with an absolute numerical multiplier. Without such conditioning, the multiplier must not be suppressed.

*Proof.* Let $s=s_\mu$. Apply Lemma A.42 to $s\Phi$, whose norm is at most $S_{\max}\sqrt b=\sqrt{B_0}$. Uniformly in $z$,

$$\left|s(z)\int k(z,w)s(w)\,d(\mu_N-\mu)(w)\right|\le B_0h=\epsilon.$$

Therefore $r(z)=\log(s(z)\int ks\,d\mu_N)$ has $\|r\|_\infty\le D$. Put $f=\log s$, $f_N=\log s_N$, $u=f_N-f$. The quotient contraction in A.41 gives

$$\operatorname{osc}(u)\le\frac{\operatorname{osc}(F_{\mu_N}(f)-F_\mu(f))}{1-\kappa}\le2\chi_\beta D.$$

The two scaling equations also give $0=u(z)+\log\mathbb E_{\nu_z}e^{u}+r(z)$, where $\nu_z$ is proportional to $k(z,w)s(w)\mu_N(dw)$. Evaluating at a maximum and a minimum of $u$ shows $|\max u+\min u|\le D$. Thus $\|u\|_\infty\le \chi_\beta D+D/2=t$.

Applying A.42 also to $s^2\Phi\otimes\Phi$, whose Hilbert–Schmidt norm is at most $B_0$, gives

$$\left\|\widetilde \Sigma_P-\Sigma_P\right\|\le\epsilon,\qquad
\widetilde \Sigma_P:=N^{-1}\sum_j s(z_j)^2\Phi(z_j)\otimes\Phi(z_j).$$

A union bound makes both events simultaneous. The relative scaling bound implies, in quadratic-form order,

$$-(e^{2t}-1)\widetilde \Sigma_P\preceq\widehat \Sigma_P-\widetilde \Sigma_P
\preceq(e^{2t}-1)\widetilde \Sigma_P.$$

Since $\|\Sigma_P\|=\|P_\mu\|=1$, this gives $E_N$. For $\epsilon\le1/2$, $D\le2\epsilon$; for $t\le1/2$, $e^{2t}-1\le4t$. Hence $E_N\le\epsilon+6t\le(12\chi_\beta+7)\epsilon$. The same proof with covariance radius $S_0^2b$ proves the conditional bound. The spectral and block conclusions follow from the Hilbert-space argument in A.43. For THEORY's centered $\tilde P$, drop the exact constant eigenpair: the nonconstant eigenvalues and eigenfunctions are unchanged. A retained block must still be separated from $\lambda_0=1$ as well as from the remaining spectrum; centering does not silently eliminate this identification requirement. ∎

**What it buys.** Fixed-$\beta$ spectra and isolated lifted eigenspaces are genuinely estimated at root-$N$ rate, including the data dependence of the normalization. The theorem supplies one explicit population-to-empirical Sinkhorn stability bound; sharper bounds from Theorem A.23 can replace it without changing the spectral argument.

**Tractability and sampling.** Write $g_\eta=1+\sqrt{2\log(2/\eta)}$. The logarithm bound requires $N>B_0^2g_\eta^2$; the simplified linear bound additionally requires $N\ge4(2\chi_\beta+1)^2B_0^2g_\eta^2$. A block certificate of tolerance $\alpha$ needs $N$ of order $[(12\chi_\beta+7)B_0]^2\log(1/\eta)/(\alpha^2\delta^2)$ and the side conditions. If only the universal $B_\beta=e^{4\beta}$ bound is used, even the prerequisite $\epsilon<1$ needs about $3.5\times10^4$, $1.1\times10^8$, and $3.1\times10^{11}$ sites at $\beta=1,2,3$, respectively, with $\eta=0.1$. This prerequisite is not sufficient for a nonvacuous projector bound. The worst-case block requirement scales as $e^{12\beta}$; the population-sensitive version as $S_{\max}^4e^{6\beta}$, ignoring the gap and confidence factors.

The independent Opus review measured a 30-atom $S^2$ toy, 40 replicates, $N=20{,}000$, reporting 90th percentiles. These are simulations, not improved certificates:

| $\beta$ | Actual lifted error | Original universal $E_N$ | Actual error divided by $e^\beta h$ |
|---|---:|---:|---:|
| 0.5 | 0.0059 | 3.24 | 0.147 |
| 1 | 0.0090 | unavailable ($\epsilon\ge1$) | 0.136 |
| 2 | 0.0120 | unavailable ($\epsilon\ge1$) | 0.066 |
| 3 | 0.0142 | unavailable ($\epsilon\ge1$) | 0.029 |

Thus the universal certificate can be useless at practical temperatures even when estimation works well; empirical sensitivity around $L_0=0.04$–$0.14$ in that toy is not a proof of a bounded $L_0$. The $S_{\max}$ refinement is proved but requires estimating or bounding that population quantity to obtain a numerical certificate. Fixed-$\beta$ convergence is not uniform for $\beta_N\to\infty$ or a Laplacian-generator rate. Multiplying $k$ by a constant leaves $P$ unchanged, so kernel scale alone cannot determine normalization conditioning.

### Out-of-sample eigenfunctions

**Theorem A.45 (Nyström extension and the eigenvalue floor).** Let $P_N\psi_i=\widehat\lambda_i\psi_i$, $\|\psi_i\|_2=1$, with $\widehat\lambda_i>0$. Define the **out-of-sample row** and the population-normalized extension by

$$s_N(z)=\left(\frac1N\sum_j k(z,z_j)s_N(z_j)\right)^{-1},\qquad
w_N(z,j)=\frac{s_N(z)k(z,z_j)s_N(z_j)}N,$$

$$\widehat f_i(z)=\frac1{\widehat\lambda_i}\sum_jw_N(z,j)\sqrt N\,\psi_i(z_j).$$

Then $\sum_jw_N(z,j)=1$ and $\widehat f_i(z_j)=\sqrt N\,\psi_i(z_j)$. This is extension, not rebalancing the enlarged $(N+1)$-site matrix.

For a population eigenfunction $P_\mu f_i=\lambda_i f_i$, $\|f_i\|_{L^2(\mu)}=1$, set

$$d_i=|\widehat\lambda_i-\lambda_i|,\quad
r_i=\left(\frac1N\sum_j|\sqrt N\psi_i(z_j)-f_i(z_j)|^2\right)^{1/2},\quad
F_i=\left(\frac1N\sum_j|f_i(z_j)|^2\right)^{1/2},$$

$$q_i=\sup_z\left|\int p_\mu(z,w)f_i(w)\,d(\mu_N-\mu)(w)\right|.$$

If $d_i\le\lambda_i/2$ and the scaling error is at most $t$, then

$$\|\widehat f_i-f_i\|_\infty\le
\frac{2}{\lambda_i}\left[\sqrt{B_N}\,r_i+B_\beta(e^{2t}-1)F_i+q_i+d_i\|f_i\|_\infty\right],\quad
B_N:=\sup_{z,w}p_{\mu_N}(z,w)\le B_\beta.$$

For a fixed population mode, $q_i\le B_\beta\|f_i\|_\infty h_N(\eta)$ with probability $1-\eta$, and $\|f_i\|_\infty\le\sqrt{B_\beta/\lambda_i}$.

There is also a direct lifted-eigenvector version. If $u_i,\widehat u_i\in\mathcal G$ are aligned unit eigenvectors of $\Sigma_P,\widehat \Sigma_P$, respectively, then

$$\|\widehat f_i-f_i\|_\infty\le
\sqrt{\frac{2B_\beta}{\lambda_i}}\left(e^t-1+\|\widehat u_i-u_i\|+\frac{d_i}{\lambda_i}\right).$$

For a simple isolated eigenvalue, A.44 bounds $\|\widehat u_i-u_i\|$ by $4\sqrt2 E_N/\delta$; within a multiple block only a jointly aligned basis/span is identified.

*Proof.* The first identities follow from the scaling equation and the matrix eigen-equation. Subtract $\lambda_i f_i(z)=\int p_\mu(z,w)f_i(w)d\mu(w)$ from the numerator of the extension. The sample-value error is at most $\sqrt{B_N}r_i$: Cauchy–Schwarz and $N^{-1}\sum_jp_{\mu_N}(z,z_j)^2\le B_N$ use the row-sum identity. The scaling term is at most $B_\beta(e^{2t}-1)F_i$; the integration term is $q_i$. Moving the eigenvalue difference to the numerator and using $\widehat\lambda_i\ge\lambda_i/2$ proves the first bound.

For the quadrature bound, apply A.42 to $s_\mu\Phi f_i$, whose norm is at most $\sqrt B_\beta\|f_i\|_\infty$, and pair its mean error with $s_\mu(z)\Phi(z)$. The evaluation-map factorization gives

$$f_i(z)=\frac{s_\mu(z)\langle\Phi(z),u_i\rangle}{\sqrt{\lambda_i}},\qquad
\widehat f_i(z)=\frac{s_N(z)\langle\Phi(z),\widehat u_i\rangle}{\sqrt{\widehat\lambda_i}}.$$

This proves the population sup-norm bound. Subtract the two formulas, separating scaling, eigenvector and denominator differences; use
$|\widehat\lambda_i^{-1/2}-\lambda_i^{-1/2}|\le\sqrt2 d_i/\lambda_i^{3/2}$. Finally, aligned unit eigenvectors differ by at most $\sqrt2$ times their rank-one projector distance. ∎

**What it buys.** THEORY [M5] receives an explicit, correctly normalized extension formula and an error budget. Division by $\lambda_i$ is a real inverse problem; a small training residual is not an out-of-sample certificate for a tiny-eigenvalue mode.

**Tractability and sampling.** Dense extension of $r$ modes costs $O(Nd+Nr)$ per new site after storing $s_N$ and the eigenvectors. Declare an eigenvalue floor, gap and numerical tolerance; errors in the fitted factors and eigenvectors must be added to the bounds. Root-$N$ convergence holds for fixed positive eigenvalues and gaps, not uniformly along the spectral tail.

### Context as regularized cross-covariance

**Definition A.46 (population kernel CCA).** Let $(Z_p,Z_q)$ be the past/target pair, i.i.d. across sampled pairs. In their sphere-kernel RKHSs $\mathcal G_p,\mathcal G_q$, let $\xi=\Phi_p(Z_p)-\mathbb E\Phi_p$ and $\zeta=\Phi_q(Z_q)-\mathbb E\Phi_q$. Write

$$\Sigma_{pp}=\mathbb E[\xi\otimes\xi],\quad \Sigma_{qq}=\mathbb E[\zeta\otimes\zeta],\quad \Sigma_{qp}=\mathbb E[\zeta\otimes\xi],$$

$$V_\rho=(\Sigma_{qq}+\rho I)^{-1/2}\Sigma_{qp}(\Sigma_{pp}+\rho I)^{-1/2},\qquad \rho>0.$$

The population regularized canonical correlations are the singular values of $V_\rho$. Replacing means and covariances by their empirical counterparts gives $\widehat V_\rho$. Its nonzero singular values equal those of $R_q^{1/2}R_p^{1/2}$ in THEORY 6.3, with $N\rho$ in the Gram ridge. The empirical evaluation maps establish this equality by polar decomposition; no Sinkhorn normalization is used in this context estimator.

On covariance supports there is a unique contraction $V$ with $\Sigma_{qp}=\Sigma_{qq}^{1/2}V\Sigma_{pp}^{1/2}$. One can construct it directly: the covariance Cauchy–Schwarz inequality bounds the bilinear form by $\|\Sigma_{qq}^{1/2}g\|\|\Sigma_{pp}^{1/2}f\|$, so it extends to the closures of these ranges. This $V$ is the unregularized correlation operator; it need not be compact. Set $b_{\mathrm{ctx}}=\max(e^{\beta_p},e^{\beta_q})$ below.

**Theorem A.47 (kernel-CCA sampling, ridge consistency and bias).** For empirical centering, with probability at least $1-\eta$, simultaneously for index pairs $(u,v)=(p,p),(q,q),(q,p)$,

$$\|\widehat \Sigma_{uv}-\Sigma_{uv}\|\le e_N:=3b_{\mathrm{ctx}} h_N(\eta/5),$$

and, **simultaneously for every $\rho>0$** on this same event,

$$\|\widehat V_\rho-V_\rho\|\le
\Delta_N(\rho):=e_N\left(\rho^{-1}+\sqrt{b_{\mathrm{ctx}}}\,\rho^{-3/2}\right).$$

Suppose additionally **compact correlation**: $V$ is compact. Then $B(\rho):=\|V_\rho-V\|\to0$. If $\rho_N\to0$ and $N\rho_N^3\to\infty$, then $\widehat V_{\rho_N}\to V$ in probability in operator norm. Its singular values converge, and any isolated singular block converges by the Hilbert-space dilation bound proved below, with error controlled by $[\Delta_N(\rho_N)+B(\rho_N)]/\delta$.

A quantitative bias rate requires another condition. Under the **two-sided source condition**

$$V=\Sigma_{qq}^{\gamma}W \Sigma_{pp}^{\gamma},\qquad 0<\gamma\le1,\quad \|W\|<\infty,$$

one has $B(\rho)\le2b_{\mathrm{ctx}}^\gamma\|W\|\rho^\gamma$. At fixed confidence and fixed kernel scales, $\rho_N\asymp N^{-1/(2\gamma+3)}$ therefore gives error $O(N^{-\gamma/(2\gamma+3)})$ to $V$, including ridge bias. Without a source condition the sampling rate is explicit but the vanishing bias need not have a universal power law.

For actual canonical variates, add the **variate range condition**: the selected limiting unit singular directions $u_p,u_q$ belong to $\operatorname{ran}(\Sigma_{pp})$ and $\operatorname{ran}(\Sigma_{qq})$, and the selected nonzero singular value is simple and isolated. Define coefficients

$$a_p=\Sigma_{pp}^{-1/2}u_p,\qquad \widehat a_{p,N}=(\widehat \Sigma_{pp}+\rho_N I)^{-1/2}\widehat u_{p,N},$$

with inverses on covariance supports, and likewise on the target side. The limiting canonical function is $f_p(z)=\langle a_p,\Phi_p(z)-\mathbb E\Phi_p\rangle$. A fitted function evaluated using the training-centered feature map is $\widehat f_{p,N}^{\mathrm{emp}}(z)=\langle\widehat a_{p,N},\Phi_p(z)-\overline\Phi_{p,N}\rangle$; the convergence claim concerns its **population-marginal centering**

$$\widehat f_{p,N}(z)=\widehat f_{p,N}^{\mathrm{emp}}(z)-\mathbb E_{Z_p}\widehat f_{p,N}^{\mathrm{emp}}(Z_p)
=\langle\widehat a_{p,N},\Phi_p(z)-\mathbb E\Phi_p\rangle,$$

where the expectation is conditional on the fitted sample. THEORY's unit-empirical-variance normalization divides the fitted coefficient by $[\langle\widehat a_{p,N},\widehat\Sigma_{pp}\widehat a_{p,N}\rangle]^{1/2}$; this factor tends to one under the conditions below. After aligned signs, $\widehat f_{p,N}\to f_p$ in $L^2(\mu_p)$ and similarly for $q$. This is proved below for **any selected simple isolated nonzero pair**, not claimed as a direct application of FBG's leading-pair theorem. Fukumizu–Bach–Gretton (2007), Theorems 1–2, cover the leading pair; the remark immediately following Theorem 2 suggests the nonleading extension. Their Lemmas 6–7, Weyl's inequality, and the explicit argument below supply that extension here.

*Proof.* Apply A.42 to the three raw feature outer products (Hilbert–Schmidt norm at most $b_{\mathrm{ctx}}$) and the two raw feature means (norm at most $\sqrt{b_{\mathrm{ctx}}}$), using failure probability $\eta/5$ each. Expanding empirical centering adds at most $2b_{\mathrm{ctx}} h_N(\eta/5)$ to each raw covariance error because both population and sample feature means have norm at most $\sqrt{b_{\mathrm{ctx}}}$. This gives $e_N$. Moreover $\|\Sigma_{pp}\|,\|\Sigma_{qq}\|,\|\widehat \Sigma_{pp}\|,\|\widehat \Sigma_{qq}\|\le b_{\mathrm{ctx}}$, since their traces are at most $b_{\mathrm{ctx}}$.

For positive operators $A,B\succeq\rho I$, the integral formula for $A^{-1/2}$ and the resolvent identity give

$$\|A^{-1/2}-B^{-1/2}\|\le\frac{\|A-B\|}{\pi}\int_0^\infty\frac{t^{-1/2}}{(\rho+t)^2}\,dt
=\frac{\|A-B\|}{2\rho^{3/2}}.$$

The joint covariance is PSD, and its Cauchy–Schwarz factorization gives

$$\|\widehat \Sigma_{qp}(\widehat \Sigma_{pp}+\rho I)^{-1/2}\|\le\sqrt{b_{\mathrm{ctx}}},\qquad
\|(\Sigma_{qq}+\rho I)^{-1/2}\Sigma_{qp}\|\le\sqrt{b_{\mathrm{ctx}}}.$$

Telescope the two whiteners and the cross-covariance, using empirical cross-covariance in the left-whitener term, population cross-covariance in the right-whitener term, and one empirical/one population whitener in the middle term. The bounds are respectively $e_N\sqrt{b_{\mathrm{ctx}}}/(2\rho^{3/2})$, the same for the other whitener, and $e_N/\rho$. Their sum is $\Delta_N$.

Write $D_{a,\rho}=\Sigma_{aa}^{1/2}(\Sigma_{aa}+\rho I)^{-1/2}$. These contractions converge strongly to the covariance-support projectors, and $V_\rho=D_{q,\rho}VD_{p,\rho}$. Strong convergence is uniform on a finite-dimensional subspace. Approximate compact $V$ in norm by finite-rank operators to conclude norm convergence. Under the source condition, spectral calculus and

$$\sup_{x\ge0}x^\gamma\left(1-\sqrt{x/(x+\rho)}\right)\le\rho^\gamma$$

bound each of the two bias terms, giving the stated rate. The scalar inequality follows by putting $x=\rho t$: for $t\le1$ use $t^\gamma\le1$; for $t\ge1$ use $1-\sqrt{t/(t+1)}\le1/t$.

The source condition itself implies compactness: positive powers of the compact covariance operators are compact. To make the singular-subspace reduction rigorous on Hilbert spaces, apply A.43 to the compact self-adjoint dilation

$$\mathscr D(V)=\begin{pmatrix}0&V^*\\V&0\end{pmatrix}\quad\text{on }\mathcal G_p\oplus\mathcal G_q.$$

The perturbation norm is $\|\widehat V-V\|$, and its nonzero positive eigenvalues are the singular values. Let $\delta$ be the gap of the selected positive singular block in the **dilation spectrum**, including separation from zero and negative eigenvalues. The dilation projector moves by at most $4\|\widehat V-V\|/\delta$ when the perturbation is at most $\delta/4$. Its two diagonal blocks are one half of the right and left singular projectors, so each singular projector moves by at most $8\|\widehat V-V\|/\delta$. Thus no infinite-dimensional version of a finite-matrix Wedin theorem is being assumed. Weyl controls each singular value; aligned simple singular vectors converge by the rank-one projector identity.

For the arbitrary-pair canonical-function assertion, put $e=\|\widehat \Sigma_{pp}-\Sigma_{pp}\|\le e_N$. The covariance inequality $\Sigma_{pp}\preceq\widehat \Sigma_{pp}+eI$ gives
$\|\Sigma_{pp}^{1/2}(\widehat \Sigma_{pp}+\rho I)^{-1/2}\|\le\sqrt{1+e/\rho}$. Consequently

$$\|\widehat f_{p,N}-f_p\|_{L^2(\mu_p)}
\le\sqrt{1+e/\rho_N}\,\|\widehat u_{p,N}-u_p\|
+\frac{\sqrt{b_{\mathrm{ctx}}}\,e}{2\rho_N^{3/2}}
+\|(D_{p,\rho_N}-I)u_p\|.$$

This follows by inserting the population regularized coefficient between the empirical and limiting coefficients, then using the inverse-square-root inequality already proved. The first term vanishes by the dilation argument and $e/\rho_N=o_P(1)$; the second vanishes since $N\rho_N^3\to\infty$; the last vanishes by strong convergence on the covariance support. The range condition makes $a_p$ a well-defined RKHS coefficient; the target proof is identical. Since $\|\widehat a_{p,N}\|\le\rho_N^{-1/2}$, empirical and population variances of the fitted function differ by at most $e/\rho_N=o_P(1)$. The latter tends to $\|f_p\|_{L^2}^2=1$, proving that unit-empirical-variance renormalization has the same limit. Thus this proof does not rely on extending a leading-eigenvector theorem by assertion. The bounded kernels ensure the finite kernel moments and separability required in FBG's Lemmas 6–7, and the paper's ridge condition is exactly our $N\rho_N^3\to\infty$. ∎

**What it buys.** The context SVD is a consistent estimator of a specified regularized population predictive object, with a visible ridge bias–variance tradeoff. Finite sample size does not by itself make the unregularized predictive state well-defined or finite-dimensional.

**Tractability and sampling.** A direct dense implementation forms two Gram matrices and uses $O(N^3)$ algebra and $O(N^2)$ storage; $m$-dimensional features reduce the covariance/SVD work to $O(Nm^2+m^3)$. For sampling error $\epsilon$, require $N\gtrsim b_{\mathrm{ctx}}^2(\rho^{-1}+\sqrt{b_{\mathrm{ctx}}}\rho^{-3/2})^2\log(1/\eta)/\epsilon^2$. In particular, $\rho_N=N^{-1/3}$ itself is **not** covered by this sufficient consistency condition. Compactness fails, for example, for an infinitely rich identical past/target view where $V$ is the identity on an infinite-dimensional support. The sampling term is uniform over ridge choices at fixed windows/views; selecting windows or kernels still needs a separate argument. A.56 handles within-document dependence.

**Theorem A.48 (finite-rank predictive summary: a conditional result).** Let $\mathcal T:L^2_0(\mu_p)\to L^2_0(\mu_q)$ be $\mathcal T f=\mathbb E[f(Z_p)\mid Z_q]$. Under the **rank-$k$ conditional-expectation assumption**, $\operatorname{rank}\mathcal T\le k$, the operators $V$ and $V_\rho$ have rank at most $k$ for every $\rho>0$. On the event of A.47,

$$\widehat\kappa_{k+1}(\rho)\le\Delta_N(\rho),$$

and the same holds for all subsequent empirical canonical correlations.

A sufficient condition is a **$k$-state sufficient summary**: a random variable $S_*$ taking $k$ values of positive probability satisfies $Z_p\perp Z_q\mid S_*$. Then the stronger centered rank bound is $k-1$, at every ridge, not only as $\rho\downarrow0$. No such conditional independence is implied by a $k$-state token-generating HMM for an arbitrary network's activation windows.

*Proof.* The centered evaluation maps $A_p:\mathcal G_p\to L^2_0(\mu_p)$ and $A_q:\mathcal G_q\to L^2_0(\mu_q)$ satisfy $\Sigma_{pp}=A_p^*A_p$, $\Sigma_{qq}=A_q^*A_q$ and $\Sigma_{qp}=A_q^*\mathcal T A_p$. Their polar decompositions $A_a=U_a \Sigma_{aa}^{1/2}$ give $V=U_q^*\mathcal T U_p$ on covariance supports. Thus $\operatorname{rank}V\le k$, and multiplication by the ridge contractions cannot raise rank. The singular-value min–max inequality gives $\widehat\kappa_{k+1}\le\|\widehat V_\rho-V_\rho\|$.

Under the sufficient-summary assumption, conditional independence gives

$$\mathbb E[f(Z_p)\mid Z_q]=\mathbb E[\mathbb E[f(Z_p)\mid S_*]\mid Z_q].$$

This factors through $L^2_0(S_*)$, which has dimension $k-1$. Equivalently, $\Sigma_{qp}$ is a sum of $k$ conditional-mean outer products with one weighted linear dependence on each side. ∎

**What it buys.** A genuinely finite predictive mediator yields a finite-rank population context spectrum, and a sampling threshold beyond which empirical tail singular values cannot be called new states. This theorem can be passed to the belief-state cluster without asserting that activation streams satisfy its hypothesis.

**Tractability and sampling.** Rank $k$ is an upper bound, not guaranteed recovery of all $k$ directions: nonzero strengths and gaps are still needed. To resolve a smallest retained correlation $\kappa_*$, make $\Delta_N(\rho)$ and any relevant ridge bias substantially smaller than $\kappa_*$. A continuous $k$-dimensional belief vector with a nonlinear kernel does not satisfy a rank-$k$ conclusion merely by its coordinate dimension.

### Temperature estimators

**Theorem A.49 (noise-temperature concentration).** Conditional on a fixed checkpoint, saved optimizer state and fixed evaluation sites, let $Y_1,\ldots,Y_n$ be conditionally i.i.d. stochastic resamples of normalized activations from **one fixed resampling protocol**, with $Y,Y'$ independent draws from that common conditional law. A resample can be the whole vector of activations across sites; average the following distances over those fixed sites. Define

$$\theta=\mathbb E\|Y-Y'\|^2\in[0,4],\qquad
\widehat\theta=\binom n2^{-1}\sum_{r<s}\|Y_r-Y_s\|^2.$$

Then for $n\ge2$,

$$\Pr(|\widehat\theta-\theta|>u)\le2\exp\left(-\frac{\lfloor n/2\rfloor u^2}{8}\right).$$

A variance-adaptive alternative is

$$\Pr(|\widehat\theta-\theta|>u)\le2\exp\left(-\frac{\lfloor n/2\rfloor u^2}{8\theta+8u/3}\right).$$

For relative scale error $a_{\mathrm{rel}}>0$, it suffices that

$$\lfloor n/2\rfloor\ge\frac{8(1+a_{\mathrm{rel}}/3)\log(2/\eta)}{a_{\mathrm{rel}}^2\theta}.$$

For a desired relative **temperature** error $\alpha$, use $a_{\mathrm{rel}}=\alpha/(1+\alpha)$. The resample count now scales as $\theta^{-1}$, not $\theta^{-2}$. A distribution-free absolute radius remains $u_n=\sqrt{8\log(2/\eta)/\lfloor n/2\rfloor}$. If $\theta>u_n$, the reciprocal temperature satisfies

$$\frac{|\widehat\beta_{\mathrm{noise}}-\beta_{\mathrm{noise}}|}{\beta_{\mathrm{noise}}}
\le\frac{u_n}{\theta-u_n},\qquad \beta_{\mathrm{noise}}=1/\theta.$$

Subtracting the sample mean displacement leaves every pairwise distance unchanged. For one site,

$$\widehat\theta=\frac{2n}{n-1}\left(\frac1n\sum_r\|Y_r-\overline Y\|^2\right),\qquad
\theta=2\,\mathbb E\|Y-\mathbb EY\|^2.$$

*Proof.* Distances between unit vectors have squared magnitude at most four, including when expressed as differences of displacements from the same checkpoint. For any permutation of the $n$ samples, pair the first $2\lfloor n/2\rfloor$ samples and average the pair distances. Those terms are independent and in $[0,4]$. Averaging this paired statistic over all permutations recovers the complete U-statistic. Jensen's inequality applied to its exponential moment and Hoeffding's bound for the independent pairs give the first tail. For the second, the pair variable $X\in[0,4]$ has $\operatorname{Var}X\le\mathbb EX^2\le4\theta$ and $|X-\theta|\le4$. The scalar Bernstein exponential bound, followed by the same permutation averaging, gives $2\exp[-\lfloor n/2\rfloor u^2/(2(4\theta+4u/3))]$. Substitute $u=a_{\mathrm{rel}}\theta$ for the resample requirement. Averaging over fixed evaluation sites preserves the bound $[0,4]$, without requiring site independence. The reciprocal inequality is algebra; the variance identities follow by expanding the squared distances. ∎

**What it buys.** The proposed noise scale has both an absolute error bar and a variance-adaptive certificate that remains substantially more useful for small noise: relative accuracy needs order $1/\theta$ rather than $1/\theta^2$ resamples. The factor of two between pairwise variation and variance is fixed. The uncertainty belongs to the resampling protocol, not to a claim that this scalar temperature captures anisotropic noise.

**Tractability and sampling.** All pairwise distances can be accumulated from sums and squared norms in $O(nd)$ per site, rather than $O(n^2d)$; model resampling dominates. For relative temperature error $\alpha$, it suffices that $u_n\le\alpha\theta/(1+\alpha)$, which gives the conservative Hoeffding requirement $\theta^{-2}\alpha^{-2}\log(1/\eta)$; the Bernstein alternative improves it to $\theta^{-1}\alpha^{-2}\log(1/\eta)$ for bounded $\alpha$. Zero or tiny noise gives an infinite or ill-conditioned reciprocal. Independent one-step replicas must all restart from the same checkpoint/state; consecutive training steps are not these draws.

**Theorem A.50 (nearest-neighbour calibration needs a quantile condition).** Fix an independent reference set $\mathcal R$ of size $n_{\mathrm{ref}}$, and let $D_{\mathcal R}(z)\in[0,4]$ be the squared distance from a query to its $k$th nearest reference neighbour. Draw an odd number $n$ of independent queries and let $\widehat q$ be the median of their $D_{\mathcal R}$ values. Conditional on $\mathcal R$, suppose the query-distance distribution has median $q>0$ and satisfies the **quantile margin condition**

$$F_{\mathcal R}(q-u)\le\tfrac12-cu,\qquad
F_{\mathcal R}(q+u)\ge\tfrac12+cu\qquad(0<u\le u_0),\quad c>0.$$

Then $\Pr(|\widehat q-q|>u\mid\mathcal R)\le2e^{-2nc^2u^2}$ for $u\le u_0$. For $u<q$, the reciprocal temperature error is at most $u/(q-u)$ in relative terms on this event.

Separately, for a fixed query $z$ and i.i.d. reference draws, let $F_z(r)=\mu(\|z-Z\|^2\le r)$. Suppose $F_z$ is continuous. If $0<k/n_{\mathrm{ref}}-t<k/n_{\mathrm{ref}}+t<1$ and $r_\pm$ are quantiles satisfying $F_z(r_\pm)=k/n_{\mathrm{ref}}\pm t$, then

$$\Pr(D_{\mathcal R}(z)\notin[r_-,r_+])\le2e^{-2n_{\mathrm{ref}}t^2}.$$

*Proof.* A median exceeding $q+u$ requires at least half the independent query distances to exceed that value; the corresponding Bernoulli mean is at most $1/2-cu$. Hoeffding gives $e^{-2nc^2u^2}$, and the other tail is identical. The reference bound follows by counting reference points inside each ball: these counts are binomial with means $k\pm n_{\mathrm{ref}}t$, whereas a $k$th order statistic outside the stated interval forces the corresponding count to cross $k$. Apply the binomial Hoeffding bound. The reciprocal bound is the same algebra as A.49. ∎

**What it buys.** There is a rigorous error bar for a split-reference nearest-neighbour temperature, but a bounded squared distance alone is insufficient to control a reciprocal median. This theorem identifies the extra quantile regularity instead of mistaking a mean bound for a median bound.

**Tractability and sampling.** Brute-force queries cost $O(nn_{\mathrm{ref}}d)$; approximate neighbour search adds an algorithmic distance error. This theorem does not change THEORY's default estimator: its use of the same sites for neighbours and medians needs a separate dependent-statistic argument. At fixed $k$ and increasing reference size, neighbour radii generally shrink and $\beta_{\mathrm{nn}}$ can diverge; there is no universal fixed positive population median to estimate. Ties, atoms, zero median and absence of the margin condition break this reciprocal-rate guarantee.

### Computation and approximation

**Proposition A.51 (dense and sparse arithmetic, and Sinkhorn iteration count).** Count one real addition, multiplication, division or exponential evaluation as one operation; input normalization is separate. For $N$ unit vectors in $\mathbb R^d$, computing the off-diagonal upper triangle of $K$ and reflecting it uses exactly $N(N-1)/2$ dot products, each with $d$ multiplications and $d-1$ additions, followed per pair by one multiplication by $\beta$ and one exponential. All diagonal entries equal $b$. Thus formation is $O(N^2d)$ and dense storage is $O(N^2)$.

One row-plus-column Sinkhorn cycle

$$r\leftarrow(Kc)^{-1},\qquad c\leftarrow(K^\top r)^{-1}$$

uses two matrix–vector products and $2N$ reciprocals. A naive dense matvec has $N^2$ multiplications and $N(N-1)$ additions, so a cycle uses $4N^2$ scalar operations in this convention. A sparse cycle is $O(\operatorname{nnz}K+N)$; for a scalable matrix with no empty rows/columns this is $O(\operatorname{nnz}K)$.

For positive column factors define the projective log error

$$D_t=\operatorname{osc}(\log c_t-\log c_*),$$

where $c_*$ is any limiting column factor; scalar gauges cancel. For the strictly positive sphere kernel, this error contracts by at most $\kappa^2=(1-e^{-2\beta})^2$ per full cycle. Starting with positive scaling vectors, reducing initial projective error $D_0$ to $\varepsilon_{\mathrm{alg}}$ takes at most

$$\left\lceil\frac{\log(D_0/\varepsilon_{\mathrm{alg}})}{-2\log\kappa}\right\rceil$$

cycles when $0<\kappa<1$ and $D_0>\varepsilon_{\mathrm{alg}}$. With the declared initialization $c_0=\mathbf1$, the scaling equation gives $\max c_*/\min c_*\le \chi_\beta$, so $D_0\le\log \chi_\beta=2\beta$. For $0<\varepsilon_{\mathrm{alg}}<2\beta$, a completely explicit sufficient count is

$$\left\lceil\frac{e^{2\beta}}2\log\frac{2\beta}{\varepsilon_{\mathrm{alg}}}\right\rceil;$$

if $\varepsilon_{\mathrm{alg}}\ge2\beta$, zero cycles already satisfy the projective-error target. At $\beta=0$ one cycle suffices. Since $-\log\kappa\ge1-\kappa=e^{-2\beta}$, the bound is $O(e^{2\beta}\log(D_0/\varepsilon_{\mathrm{alg}}))$ cycles. This is a worst-case guarantee, not a claim of a few dozen cycles for every temperature.

For Lanczos with $\ell$ Krylov steps, dense matvec work is $O(\ell N^2)$, plus $O(N\ell^2)$ for full reorthogonalization, $O(\ell^3)$ for a dense projected eigensolve, and $O(N\ell r)$ to form $r$ Ritz vectors. A restarted sweep with $\ell=O(r)$ therefore has $O(rN^2+Nr^2+r^3)$ cost. A sparse sweep replaces $rN^2$ by $r\operatorname{nnz}K$. Neither bound promises convergence in a single sweep.

*Proof.* The operation counts follow directly from the stated loops. The contraction is A.41 applied alternately to the two log-scaling vectors: each map contracts oscillation by $\kappa$, so their composition contracts by $\kappa^2$. Scalar normalization is a gauge for two-sided scaling; for symmetric $K$, recover the unique symmetric scaling from the limiting left/right factors by their componentwise geometric mean. Knight (2008), Theorem 4.5, gives the sharper local linear rate governed by the square of the second singular value of the scaled matrix, after gauge removal. Its full-indecomposability hypothesis holds for a strictly positive matrix; this local statement does not furnish a uniform bound on entry into the asymptotic regime. Finally, each Lanczos step uses a matvec and orthogonalization against the current Krylov vectors; summing those costs gives the claim. ∎

**What it buys.** The construction has explicit, auditable arithmetic costs and separates iteration costs from convergence costs. Products such as $P^{L'}P^L$ are applied as successive matvecs, not formed densely.

**Tractability and sampling.** Statistical sample requirements are A.44, not this arithmetic proposition. Sparse kernels may have zeros, lack total support or full indecomposability, and need not be PSD: neither the positive-kernel contraction nor the dense spectral estimand transfers automatically. Small spectral gaps increase both sampling requirements and Krylov iterations. For a PSD scaled layer operator, Knight's worst-case local factor is $\lambda_1^2$, where $\lambda_1$ is the largest nonconstant eigenvalue. Thus the slow-mode/near-cluster regime ($\lambda_1\to1$) is exactly where the local contraction bound deteriorates. A particular warm start may not excite that slow direction. This is **not** equivalent to cross-layer coherence: for orthogonal balanced contrasts $u,v\in\mathbb R^4$ and $0<a_*<1$, the stochastic PSD matrices $P^L=J+a_*uu^\top$ and $P^{L'}=J+a_*vv^\top$ each have a slow contrast, but $P^{L'}P^L=J$, where $J=\mathbf1\mathbf1^\top/4$. Duplicate antipodal sphere views realize these two incompatible partitions with $a_*=\tanh\beta$. In exact arithmetic $\operatorname{nnz}$ is $N^2$ for the sphere kernel; truncating entries creates an approximation or a different estimand.

**Theorem A.52 (ridge-leverage Nyström approximation).** Conditional on any fixed sites, let $A=K/N$, choose $\tau>0$, and define ridge leverage scores and effective dimension

$$\ell_j=[A(A+\tau I)^{-1}]_{jj},\qquad d_{\mathrm{eff}}(\tau)=\sum_j\ell_j.$$

Draw $m$ landmark indices independently with replacement from probabilities $q_j$ satisfying

$$q_j\ge\frac{\ell_j}{c_\ell d_{\mathrm{eff}}(\tau)},\qquad c_\ell\ge1.$$

Let $K_{\mathrm{land}}=K[:,I]$, $W=K[I,I]$, and $K_{\mathrm{Ny}}=K_{\mathrm{land}}W^\dagger K_{\mathrm{land}}^\top$. If

$$m\ge10c_\ell d_{\mathrm{eff}}(\tau)\log(2N/\eta),$$

then with probability at least $1-\eta$,

$$0\preceq K-K_{\mathrm{Ny}},\qquad
\left\|\frac{K-K_{\mathrm{Ny}}}{N}\right\|\le\tau.$$

Given the sampling probabilities, landmark kernel formation and factorization cost $O(Nmd+Nm^2+m^3)$ time and $O(Nm+m^2)$ storage. For $m\le N$, this is $O(Nmd+Nm^2)$. Once represented as $BB^\top$, a kernel matvec costs $O(Nm)$.

*Proof.* Work in the span of the $N$ features, of dimension at most $N$. Put $T=N^{-1}\sum_j\Phi_j\otimes\Phi_j$ and $H=(T+\tau I)^{-1/2}T(T+\tau I)^{-1/2}$. The identity between feature and Gram spectra gives

$$\ell_j=N^{-1}\|(T+\tau I)^{-1/2}\Phi_j\|^2.$$

For one sampled index $J$, define

$$X_J=\frac{(T+\tau I)^{-1/2}\Phi_J\otimes(T+\tau I)^{-1/2}\Phi_J}{Nq_J}.$$

Then $\mathbb EX_J=H$, $0\preceq H\preceq I$, and $0\preceq X_J\preceq L I$ with $L=c_\ell d_{\mathrm{eff}}$. Also $H\preceq LI$, $\|X_J-H\|\le L$, and
$\mathbb E(X_J-H)^2=\mathbb EX_J^2-H^2\preceq L H\preceq LI$.

The self-adjoint matrix Bernstein inequality (Tropp 2012, Theorem 6.1, applied to both signs, dimension at most $N$) consequently gives

$$\Pr\left(\left\|m^{-1}\sum_{t=1}^m X_{J_t}-H\right\|>\tfrac12\right)
\le2N\exp\left(-\frac{3m}{28L}\right)\le\eta.$$

All its hypotheses—independence, zero mean after centering, norm bound and variance bound—were just verified. On this event, the unwhitened sampled covariance $T_I=m^{-1}\sum_t\Phi_{J_t}\otimes\Phi_{J_t}/(Nq_{J_t})$ satisfies $T\preceq2T_I+\tau I$. Let $\Pi_I$ project onto the landmark feature span. Compression to its orthogonal complement gives

$$\|(I-\Pi_I)T(I-\Pi_I)\|\le\tau.$$

The Nyström identity is $K_{\mathrm{Ny}}[i,j]=\langle\Pi_I\Phi_i,\Pi_I\Phi_j\rangle$ even with repeated or dependent landmarks, because the Moore–Penrose inverse represents the orthogonal projector. Thus the residual Gram is PSD, and its normalized nonzero spectrum equals that of $(I-\Pi_I)T(I-\Pi_I)$. This proves the bound. The costs count $Nm$ dot products, an $m\times m$ eigendecomposition and multiplication by its inverse square-root factors. ∎

**What it buys.** A sample-specific effective dimension, rather than an assumed sparsity or ambient width, controls the landmark budget. Direct precedents are El Alaoui–Mahoney (2015) for ridge-leverage importance sampling and Rudi–Camoriano–Rosasco (2015) for Nyström computational regularization, following Bach (2013); it does not import a supervised prediction-risk theorem as an operator-error theorem.

**Tractability and sampling.** Exact leverage probabilities can themselves require dense work: the i.i.d.-sampling cost above assumes an oracle or an approximation with the stated domination factor. This is an algorithmic choice, not an unavoidable barrier. Musco–Musco (2017), Algorithm 2 and Theorem 7, give a recursive ridge-leverage route with the same residual conclusion, including score acquisition. Apply their theorem to the PSD matrix $K$ with ridge $\lambda=N\tau$, and failure parameter $\eta_{\mathrm{alg}}=\min(\eta/3,1/64)$, within their required $(0,1/32)$ range. Their output has $m_{\mathrm{rec}}=O((1+d_{\mathrm{eff}})\log((1+d_{\mathrm{eff}})/\eta_{\mathrm{alg}}))$ columns, or all $N$ columns when appropriate, uses $O(Nm_{\mathrm{rec}})$ kernel evaluations and $O(Nm_{\mathrm{rec}}^2)$ further arithmetic, and satisfies $0\preceq K-K_{\mathrm{Ny}}\preceq N\tau I$ with probability at least $1-3\eta_{\mathrm{alg}}\ge1-\eta$. Here the added ones merely cover small effective dimensions/base cases; no supervised assumption is used. The sphere kernel is PSD and each oracle evaluation costs $O(d)$, so the end-to-end cost is $O(Nm_{\mathrm{rec}}d+Nm_{\mathrm{rec}}^2)$. This recursive Bernoulli scheme is a separately cited algorithm, not identical to the i.i.d. sampling proof above. Uniform sampling has $c_\ell=N\max_j\ell_j/d_{\mathrm{eff}}$, which can be large; arbitrary or $k$-means landmarks have no guarantee from this theorem. PSD approximation does **not** ensure entrywise positivity. Sinkhorn requires an additional positivity check/certificate and a scaling-stability bound; the spectral residual bound alone is not that certificate. If $d_{\mathrm{eff}}\approx N$, the theorem offers no useful compression and one should use the dense matrix. Statistical site error remains separate from the landmark error $\tau$.

**Proposition A.53 (random Fourier features for the sphere kernel).** Draw independent $\omega_t\sim\mathcal N(0,\beta I_d)$, $t=1,\ldots,m$, independently of the sites, and use the $2m$-dimensional feature

$$Z(z)=\sqrt{b/m}\,(\cos(\omega_1^\top z),\sin(\omega_1^\top z),\ldots,\cos(\omega_m^\top z),\sin(\omega_m^\top z)).$$

Then $\mathbb E\langle Z(z),Z(w)\rangle=k(z,w)$. Conditional on any fixed $N$ sites, with probability $1-\eta$,

$$\max_{i,j}|K_{\mathrm{RF}}[i,j]-K[i,j]|\le
\epsilon_m:=b\sqrt{\frac{2\log(2N^2/\eta)}m},\qquad
\|(K_{\mathrm{RF}}-K)/N\|\le\epsilon_m.$$

The matrix $K_{\mathrm{RF}}=ZZ^\top$ is PSD and its diagonal is exactly $b$, so the union bound can in fact use only $N(N-1)/2$ off-diagonal pairs; the displayed $N^2$ version remains a safe simplification. If $\epsilon_m<a$, it is also strictly entrywise positive on these sites. Feature formation costs $O(Nmd)$, a feature-space eigensolve costs $O(Nm^2+m^3)$, and a kernel matvec costs $O(Nm)$.

*Proof.* Gaussian characteristic functions give $\mathbb E\cos(\omega^\top(z-w))=e^{-\beta\|z-w\|^2/2}$, and the sphere identity multiplies this by $b$ to give $k$. Each summand lies in $[-b,b]$, so Hoeffding gives tail $2\exp(-m\epsilon^2/(2b^2))$ for one pair; union bound over $N^2$ pairs. An $N\times N$ matrix with entrywise error at most $\epsilon$ has operator norm at most $N\epsilon$. PSD is the Gram identity, and $K[i,j]\ge a$ proves the positivity certificate. Costs follow from forming the features and their covariance. ∎

**What it buys.** A fully explicit, label-free approximation applies to this exact sphere kernel. Random frequencies and Nyström landmarks are different approximations; both can end in the same SVD machinery, but their hypotheses and error budgets should not be conflated.

**Tractability and sampling.** Entrywise accuracy $\epsilon$ needs $m\ge2e^{2\beta}\epsilon^{-2}\log(2N^2/\eta)$. The sufficient positivity certificate alone needs $m>2e^{4\beta}\log(2N^2/\eta)$, which may eliminate the computational advantage. Faster prediction-risk rates in Rudi–Rosasco (2017) use additional learning/source/effective-dimension assumptions and are not automatically Sinkhorn spectral bounds. At high resolution, low rank is a regime to check, not a promise.

**Theorem A.54 (shared Hutchinson probes for the action view).** Fix nonzero Jacobians $J_1,\ldots,J_N$ with a common input/output shape and draw shared independent Gaussian probes $g_t\sim\mathcal N(0,I)$. Define

$$\widehat G_{ij}=\frac1{m_J}\sum_t\langle J_i g_t,J_j g_t\rangle.$$

Then $\mathbb E\widehat G_{ij}=\langle J_i,J_j\rangle_F$, and, with $B_{ij}=\operatorname{sym}(J_i^\top J_j)$,

$$\operatorname{Var}(\widehat G_{ij})=\frac2{m_J}\|B_{ij}\|_F^2.$$

For $0<\varepsilon<1/2$, if

$$m_J\ge\frac8{\varepsilon^2}\log\frac{2N^2}{\eta},$$

then with probability at least $1-\eta$, simultaneously for all pairs,

$$|\widehat G_{ij}-\langle J_i,J_j\rangle_F|
\le\varepsilon\|J_i\|_F\|J_j\|_F,$$

$$\left|\frac{\widehat G_{ij}}{\sqrt{\widehat G_{ii}\widehat G_{jj}}}
-\frac{\langle J_i,J_j\rangle_F}{\|J_i\|_F\|J_j\|_F}\right|
\le\frac{2\varepsilon}{1-\varepsilon}\le4\varepsilon.$$

The resulting action sphere-kernel entries differ by at most $4\beta e^\beta\varepsilon$. Rademacher probes also give unbiased raw inner products, with variance $2m_J^{-1}\sum_{r\ne s}B_{ij}[r,s]^2$; the stated logarithmic simultaneous bound is proved here for Gaussian probes, not for arbitrary finite-fourth-moment probes.

*Proof.* Since $g^\top J_i^\top J_jg=g^\top B_{ij}g$, Gaussian second and fourth moments give the expectation and variance. For any fixed matrix $A\ne0$, diagonalize $A^\top A/\|A\|_F^2$, with eigenvalues $w_s\ge0$ summing to one. The relative sketch-norm error is a mean of variables $\sum_s w_s(Z_s^2-1)$. For $0<t<1/2$, its one-probe log-mgf is at most $t^2/(1-2t)$: expand $-\log(1-2tw_s)-2tw_s$ and use $\sum_s w_s^2\le1$. The negative-mgf bound is at most $t^2$. Chernoff with $t=\varepsilon/4$ gives

$$\Pr\left(\left|m_J^{-1}\sum_t\|Ag_t\|^2-\|A\|_F^2\right|>\varepsilon\|A\|_F^2\right)
\le2e^{-m_J\varepsilon^2/8}.$$

Apply this to the $N$ normalized matrices $A_i=J_i/\|J_i\|_F$ and to $A_i+A_j,A_i-A_j$ for $i<j$ (zero matrices need no bound). There are at most $N^2$ matrices. Polarization makes the inner-product error at most
$\varepsilon(\|A_i+A_j\|_F^2+\|A_i-A_j\|_F^2)/4=\varepsilon$.
The estimated squared norms lie in $[1-\varepsilon,1+\varepsilon]$. Dividing the inner product by their geometric mean gives at most $2\varepsilon/(1-\varepsilon)$ error. Both exact and sketched cosines are in $[-1,1]$, so the mean-value theorem gives the kernel error. Rademacher fourth moments yield the stated alternative variance. ∎

**What it buys.** The action kernel has a simultaneous, dimension-free probe budget for relative inner products and cosine accuracy, without forming Jacobians. Unbiasedness applies to the raw Gram estimates; normalized cosines and exponentiated kernels are not generally unbiased.

**Tractability and sampling.** Cost is $Nm_J$ JVPs, plus storage/Gram construction for sketches of width $m_Jd_{\mathrm{out}}$ (or blocked equivalents). For desired action-kernel entry error $\zeta$, the sufficient budget scales as $\beta^2e^{2\beta}\zeta^{-2}\log(N/\eta)$, before Sinkhorn amplification. The same probes must be shared across sites, and zero Jacobians require THEORY's exclusion policy. Arbitrary probes with merely finite fourth moments give the variance statement but need weaker tails or robust estimation; they do not inherit the Gaussian logarithmic budget.

### Cross-layer products and document-level sampling

**Theorem A.55 (sampling the coherent product on a common population).** Let a site carry two normalized views $(z_a,z_b)$ with joint law $\nu$, sampled i.i.d. across sites; their within-site dependence is unrestricted. For each view use its own finite temperature, kernel feature $\Phi_j$, marginal Sinkhorn factor $s_j$, and covariance $\Sigma_j=\mathbb E[\eta_j\otimes\eta_j]$, where $\eta_j=s_j\Phi_j$ and $j\in\lbrace a,b\rbrace$. Let $A_j:\mathcal G_j\to L^2(\nu)$ be evaluation, $A_j u=\langle\eta_j(\cdot),u\rangle$. Then $P^j=A_jA_j^*$ acts on the common site space, and

$$Q=P^bP^a,\qquad \Xi_{ba}=A_b^*A_a=\mathbb E[\eta_b\otimes\eta_a],\qquad
\mathcal K_Q=\Sigma_b^{1/2}\Xi_{ba}\Sigma_a^{1/2}.$$

The nonzero singular values of $Q$ equal those of $\mathcal K_Q:\mathcal G_a\to\mathcal G_b$. The analogous empirical identity is exact, with all three empirical covariances using the **estimated** sample Sinkhorn factors; in particular it gives the singular values of THEORY's $Q_N=P_N^bP_N^a$.

Here are explicit simultaneous error budgets. Put $h=h_N(\eta/5)$ and, for each view, $B_{0j}=e^{\beta_j}(\sup s_j)^2$, $\chi_j=e^{2\beta_j}$,

$$\epsilon_j=B_{0j}h<1,\quad t_j=(\chi_j+\tfrac12)[-\log(1-\epsilon_j)],\quad
E_j=\epsilon_j+(e^{2t_j}-1)(1+\epsilon_j).$$

Define

$$E_{ba}=\sqrt{B_{0a}B_{0b}}h+
(e^{t_a+t_b}-1)\sqrt{(1+\epsilon_a)(1+\epsilon_b)}.$$

With probability at least $1-\eta$, all three covariance errors satisfy their corresponding bounds, and

$$\|\widehat{\mathcal K}_Q-\mathcal K_Q\|\le
E_Q:=\sqrt{E_a}+E_{ba}+\sqrt{E_b}.$$

Thus every singular-value error is at most $E_Q=O(h^{1/2})=O(N^{-1/4})$ for fixed kernels. For an isolated finite positive singular block whose dilation-spectrum gap is $\delta$, each left/right **lifted** singular projector has error at most $8E_Q/\delta$ when $E_Q\le\delta/4$.

Under the additional **finite-feature covariance-floor assumption**, each deterministic population feature support $\mathcal G_j^0=\overline{\operatorname{span}}\lbrace\eta_j(z):z\in\operatorname{supp}\nu\rbrace$ is finite-dimensional and $\Sigma_j\succeq\nu_j I$ on it, with $\nu_j>0$. For the sphere kernel this assumption is exactly the statement that each view's population has **finite support** (the sphere-kernel feature span of an infinite set of distinct directions is infinite-dimensional), so the name should not be read as more general than that; and $\nu_j$ is the smallest nonzero eigenvalue of the population Gram operator on that support, which can be very small when two supported directions are nearly parallel, which then weakens the constant accordingly. If $E_j\le\nu_j/2$, the stronger bound is

$$\|\widehat{\mathcal K}_Q-\mathcal K_Q\|\le
\frac{E_a}{\sqrt{\nu_a}}+E_{ba}+\frac{E_b}{\sqrt{\nu_b}}=O(N^{-1/2}).$$

*Proof.* Polar decomposition gives $A_j=U_j\Sigma_j^{1/2}$, with $U_j$ a partial isometry on the feature support. Then $Q=U_b\mathcal K_QU_a^*$. Both sides of $\mathcal K_Q$ lie in the respective initial spaces, so multiplication by these partial isometries preserves its nonzero singular values. The empirical evaluation maps into counting-measure $\mathbb R^N$ have rows $s_{j,N}(z_{j,i})\Phi_j(z_{j,i})/\sqrt N$; the same identity applies. It does not incorrectly identify the two layer RKHSs.

Apply A.42 to five bounded random objects: the two feature means $\eta_j$ used in scaling stability, the two tensors $\eta_j\otimes\eta_j$, and $\eta_b\otimes\eta_a$. Their norm bounds are respectively $\sqrt{B_{0j}},B_{0j},\sqrt{B_{0a}B_{0b}}$. A union bound gives probability $1-\eta$. A.44's deterministic argument now gives $t_j,E_j$. For the empirical cross-covariance, first keep population scaling factors: the sampling error is at most $\sqrt{B_{0a}B_{0b}}h$. Changing to estimated factors multiplies each outer product by a scalar in $[e^{-(t_a+t_b)},e^{t_a+t_b}]$. For unit $u\in\mathcal G_a,w\in\mathcal G_b$, Cauchy–Schwarz bounds the resulting bilinear error by

$$(e^{t_a+t_b}-1)
\left(\frac1N\sum_i|\langle\eta_a(z_i),u\rangle|^2\right)^{1/2}
\left(\frac1N\sum_i|\langle\eta_b(z_i),w\rangle|^2\right)^{1/2},$$

which is at most the second term of $E_{ba}$ because the fixed-factor covariance norms are at most $1+\epsilon_j$.

Both population and fully normalized empirical covariance operators have norm one; the cross-covariances have norm at most one by $\Xi=A_b^*A_a$. For positive bounded operators, $\|T^{1/2}-S^{1/2}\|\le\|T-S\|^{1/2}$. For completeness: $T\preceq S+eI$ and $S\preceq T+eI$ with $e=\|T-S\|$; square-root operator monotonicity and scalar calculus give $-\sqrt e I\preceq T^{1/2}-S^{1/2}\preceq\sqrt e I$. Monotonicity follows from the positive-resolvent integral for the square root and the inverse-order property. Telescoping the three factors of $\mathcal K_Q$ proves $E_Q$. Weyl and the Hilbert-space dilation argument of A.47 give the spectral assertions.

Under the covariance floor, empirical features lie in the same deterministic support, and $\widehat\Sigma_j\succeq\nu_j I/2$ there. The identity
$\widehat\Sigma_j^{1/2}X+X\Sigma_j^{1/2}=\widehat\Sigma_j-\Sigma_j$ for $X=\widehat\Sigma_j^{1/2}-\Sigma_j^{1/2}$, together with its semigroup-integral solution, gives
$\|X\|\le E_j/(\sqrt{\nu_j/2}+\sqrt{\nu_j})\le E_j/\sqrt{\nu_j}$.
Replace the two square-root terms in the telescoping bound. ∎

**What it buys.** The central cross-layer product now has a population estimand and an honest sampling theorem even when its two geometries use different RKHSs. The common-space lift identifies what the empirical singular values and subspaces estimate, rather than composing incompatible covariance lifts.

**Tractability and sampling.** Computing the lift is unnecessary: apply $P_N^bP_N^a$ as two matvecs and its adjoint in reverse order. In the unconditional certificate, singular-value tolerance $\alpha$ requires order $\alpha^{-4}$ independent sites and a projector tolerance also incurs $\delta^{-4}$, with the severe temperature constants inherited from A.44. Under a fixed covariance floor the exponents improve to two, with dependence on $\nu_j^{-1}$. A generic infinite-rank sphere-kernel covariance cannot have a positive floor on its whole support. **Open question for this programme:** can one prove a general root-$N$ bound for isolated nonzero singular blocks of $Q$ without that floor, exploiting the special joint-covariance structure rather than the generic square-root inequality? This document does not establish such a rate and does not claim it is an unsolved problem in the entire operator-learning literature. The lifted projector metric is explicit; transferring to out-of-sample site singular functions needs the corresponding evaluation/extension control. Drop the constant singular pair for the centered product, while including its separation in the gap.

**Corollary A.56 (independent documents, arbitrarily dependent positions).** Suppose $D$ documents are i.i.d., each supplying $T$ declared positions (or past/target pairs) with arbitrary within-document dependence and possibly position-specific distributions. Define

$$\mu=\frac1T\sum_{t=1}^T\operatorname{Law}(z_{d,t}),\qquad
\mu_{D,T}=\frac1D\sum_{d=1}^D\frac1T\sum_{t=1}^T\delta_{z_{d,t}}.$$

Use the analogous position-averaged joint law for paired views. All sampling bounds in A.43–A.48 and A.55 remain true with **statistical** $h_N$ replaced by $h_D$, and ridge consistency requires $D\rho_D^3\to\infty$. The finite site matrices still have $N=DT$ rows, counting measure, $s_N/\sqrt{DT}$ factors and $DT\rho$ Gram ridges. Their arithmetic costs still depend on $DT$, not $D$. No mixing condition within a document is required.

*Proof.* Every empirical feature mean, covariance tensor and cross-covariance used in those proofs is an average of the document contributions $T^{-1}\sum_t Y_{d,t}$. The documents make these contributions i.i.d.; their norms are bounded by the same radius as one contribution $Y_{d,t}$, and their expectations are exactly the integrals against the position-averaged law. Apply A.42 to these $D$ bounded Hilbert-valued observations. All subsequent scaling, centering, spectral, whitening and canonical-function arguments are deterministic functions of those concentration events. In particular, the direct proof in A.47, not an i.i.d.-pairs citation, gives the canonical-function conclusion. ∎

**What it buys.** The construction can retain every declared position while obtaining valid error bars from independent documents. Effective replication is measured in documents in this conservative guarantee, not inflated by the number of overlapping windows.

**Tractability and sampling.** Perfectly duplicated positions give the same statistical information as one position per document, showing that the $D$ rather than $DT$ dependence is necessary without extra dependence assumptions. Unequal document lengths require a declared weighting: equal-document averages lead to a weighted-site construction unless lengths are equal, while indiscriminate pooling changes the marginal to a length-biased one. Correlated documents, changing position-selection rules and sample-selected windows are outside this corollary. This corollary concerns the spectral/context sampling bounds, not every unrelated resample or nearest-neighbour bound in this cluster.

### Numerical checks performed

The persistent script `checks/theorems_a3_check.py` was run with `uv run --with numpy python -I checks/theorems_a3_check.py` (base seed 4417; revision-check seed 9055). All assertions passed. It is a finite-dimensional sanity check, not a proof or an experiment on a trained network.

- On a five-point nonuniform sphere population, across $N=128,512,2048,8192$ and 350 repetitions per size, mean raw-operator, normalized lifted-operator and eigenvalue errors had log–log slopes $-0.515$, $-0.517$, and $-0.574$, respectively. Every sampled eigenvalue error satisfied the operator perturbation bound.
- Nyström eigenfunction extension reproduced the training values with the required $\sqrt N$ conversion to probability normalization.
- A two-state conditionally independent past/target toy had population centered CCA rank one. Across 200 repetitions per size, fixed-ridge operator error and the empirical second canonical correlation had slopes $-0.500$ and $-0.503$; every empirical tail was below its actual operator error and the explicit covariance-to-CCA bound.
- A ridge-leverage landmark sample satisfied the PSD residual and normalized operator-error certificate. The conservative bound requested 301 draws for 60 sites, correctly illustrating a case in which this certificate does not promise compression.
- Pairwise-noise and centered-variance formulas agreed to numerical precision; mean scale-estimation error had slope $-0.516$.
- Across 6,000 shared-Gaussian probe replicas, the measured raw-Gram variance divided by the exact variance was $0.998$.
- Revision checks verify the population-sensitive Sinkhorn bound; the exact three-covariance $Q$ lift and both its Hölder and covariance-floor perturbation bounds (finite-support observed slope $-0.494$); the nonleading canonical-function inequality and marginal centering; Bernstein tails against exact binomial enumeration; duplicated within-document positions; projective Sinkhorn contraction; and the orthogonal-slow-contrast counterexample to inferring cross-layer coherence from slow Sinkhorn convergence.
- The existing `/home/ubuntu/code/CoherentModes/checks/cm_check.py` also passes. This revision does not modify THEORY.md; it adds the persistent asserted check above.

### Discrepancies with THEORY.md

1. **Section 13's sampling rate is for a shrinking-bandwidth generator applied to a smooth function.** It is not the fixed-$\beta$ integral-operator or eigenspace rate proved here. Do not substitute one for the other. A manifold, a dimension, boundary conditions and an explicit dependence regime are additional hypotheses, not consequences of the dot product.
2. **Section 3's “few dozen” Sinkhorn products is an empirical expectation, not a uniform cost guarantee.** Worst-case iteration counts depend exponentially on $\beta$ (or on spectral conditioning). Plain symmetric iteration $s\leftarrow1/(Ks)$ can oscillate in its scalar gauge; alternating left/right factors or another convergent implementation must be specified.
3. **Section 5 [M5] needs a measure conversion.** The probability-normalized extension takes sample values $\sqrt N\psi_i$; its counting-normalized version is $\widehat f_i/\sqrt N$. Both normalization factors and division by the eigenvalue are necessary.
4. **Section 6.3 does not currently promise universal consistency, correctly.** Such a claim needs i.i.d./quantified dependence, a ridge schedule, compactness for the zero-ridge limit, gaps for directions, and range conditions for canonical functions. Choosing ridge/window by held-out score is not itself proof of these conditions. The sufficient schedule is strictly slower than $N^{-1/3}$, not equal to it.
5. **Section 10's pairwise displacement and centered variance differ by a factor of two** (and a finite-$n$ correction); the mean-removal step does not change pairwise differences. Its nearest-neighbour median is a different statistic, with a moving reference-size-dependent estimand and no distribution-free reciprocal concentration from boundedness alone.
6. **Section 13's Nyström warning is essential.** PSD and small normalized operator error do not ensure positive entries or a well-conditioned Sinkhorn solve. A rank-$r$ Lanczos “sweep” also needs its Krylov length and reorthogonalization cost specified; $O(rN^2)$ is matvec work for $O(r)$ steps, not a guaranteed total eigensolve cost.
7. **Section 7.3's unbiasedness is correctly limited to raw inner products.** The finite-fourth-moment assumption gives root-probe standard error, but logarithmic high-probability simultaneous cosine bounds need stronger tails (Gaussian probes suffice here).

No edits to THEORY.md are made here; these are distinctions or implementation clarifications for the owner.

### Sources

- [Borwein, Lewis and Nussbaum (1994), “Entropy minimization, DAD problems, and doubly stochastic kernels,” *Journal of Functional Analysis* 123, 264–307](https://doi.org/10.1006/jfan.1994.1089). Context for continuous scaling; A.41 is independently proved.
- [Marshall and Coifman (2019), “Manifold learning with bi-stochastic kernels,” *IMA Journal of Applied Mathematics* 84(3), 455–482](https://doi.org/10.1093/imamat/hxy065). Context for the continuum and generator constructions, not used to infer a manifold here.
- [Nutz and Wiesel (2023), “Stability of Schrödinger potentials and convergence of Sinkhorn's algorithm,” *Annals of Probability* 51(2), 699–722; arXiv:2201.10059](https://arxiv.org/abs/2201.10059). General continuous scaling/stability context; our bounded-kernel proof is self-contained.
- [Koltchinskii and Giné (2000), “Random matrix approximation of spectra of integral operators,” *Bernoulli* 6(1), 113–167](https://doi.org/10.2307/3318636); [Rosasco, Belkin and De Vito (2010), “On learning with integral operators,” *Journal of Machine Learning Research* 11, 905–934](https://www.jmlr.org/papers/v11/rosasco10a.html). Spectral-estimation precedents; A.43 supplies its own bounds and verifies its own hypotheses.
- [Fukumizu, Bach and Gretton (2007), “Statistical consistency of kernel canonical correlation analysis,” *JMLR* 8, 361–383](https://www.jmlr.org/papers/v8/fukumizu07a.html), Lemmas 6–7 and Theorems 1–2. Theorems 1–2 cover the leading pair; the remark after Theorem 2 suggests the nonleading extension. A.47 supplies a direct proof of that extension using the covariance estimates and Hilbert-space dilation.
- [Knight (2008), “The Sinkhorn–Knopp algorithm: convergence and applications,” *SIAM Journal on Matrix Analysis and Applications* 30(1), 261–275](https://doi.org/10.1137/060659624), Theorem 4.5. Local rate; the global projective bound in A.51 is separately proved.
- [Bach (2013), “Sharp analysis of low-rank kernel matrix approximations,” *COLT*, PMLR 30, 185–209](https://proceedings.mlr.press/v30/Bach13.html). Effective-dimension precedent, not a substituted prediction-risk guarantee.
- Tropp (2012), “User-friendly tail bounds for sums of random matrices,” *Foundations of Computational Mathematics* 12, 389–434, Theorem 6.1. The exact Bernstein reduction and its hypotheses are in A.52.
- Rudi and Rosasco (2017), “Generalization properties of learning with random features,” *NeurIPS 30*, arXiv:1602.04474. Learning-rate context only; A.53 uses an elementary, explicitly proved operator-approximation bound instead.
- Davis and Kahan (1970), “The rotation of eigenvectors by a perturbation. III,” *SIAM Journal on Numerical Analysis* 7(1), 1–46. The Hilbert-space separated-spectrum argument is written in A.43; A.47 obtains singular-subspace bounds by self-adjoint dilation, rather than invoking a finite-dimensional Wedin theorem.
- [El Alaoui and Mahoney (2015), “Fast randomized kernel ridge regression with statistical guarantees,” *NeurIPS 28*](https://arxiv.org/abs/1411.0306); [Rudi, Camoriano and Rosasco (2015), “Less is more: Nyström computational regularization,” *NeurIPS 28*](https://arxiv.org/abs/1507.04717). Direct Nyström/ridge-leverage precedents.
- [Musco and Musco (2017), “Recursive sampling for the Nyström method,” *NeurIPS 30*](https://arxiv.org/abs/1605.07583), Algorithm 2 and Theorem 7. PSD kernel oracle, positive ridge and permitted failure parameter are verified in A.52.

### Summary: cost, sampling and failure modes

Here $h_N=O(\sqrt{\log(1/\eta)/N})$, $B_\beta=e^{4\beta}$, $\chi_\beta=e^{2\beta}$; $r$ is requested rank, $\ell$ the Krylov length, $m$ the approximation budget, and $m_J$ the probe budget. Formation costs do not include model forward/backward work unless stated.

| Object | Cost | Sample/accuracy requirement | What breaks |
|---|---|---|---|
| Raw sphere kernel spectrum | $O(N^2d)$ formation; $O(N^2)$ storage | Error $\le e^\beta h_N$; block error $O(e^\beta h_N/\delta)$ | Dependent sites, closing gaps, changing $\beta$ |
| Population-normalized spectrum | Same kernel, plus Sinkhorn and eigensolve | Conservative error $O(e^{6\beta}h_N)$, refined to $O(S_{\max}^2e^{3\beta}h_N)$; divide by $\delta$ for blocks | Unreported normalization conditioning; high-$\beta$ constants |
| Cross-layer $Q=P^{L\prime}P^L$ | Two layer matvecs per application; adjoint in reverse order | $E_Q=O(N^{-1/4})$; root-$N$ under finite-feature covariance floors (= finitely supported views for the sphere kernel) | Different RKHSs require the three-covariance lift; unrestricted root-$N$ not proved here |
| Independent-document design | Same $DT$-site computational cost | Replace statistical $N$ by $D$, not matrix dimensions or Gram ridge factors | Dependent documents, unequal-length weighting, adaptive windows |
| Dense Sinkhorn | $O(N^2)$ per cycle | From all-ones initialization, at most $\lceil(e^{2\beta}/2)\log(2\beta/\varepsilon_{\mathrm{alg}})\rceil$ cycles when $\varepsilon_{\mathrm{alg}}<2\beta$ in projective error | Underflow, nearly reducible matrices, insufficient solver tolerance |
| Sparse Sinkhorn | $O(\operatorname{nnz})$ per cycle | Requires scalable support; local rate depends on scaled operator | No full indecomposability/total support; different estimand |
| Lanczos | $O(\ell N^2+N\ell^2+\ell^3)$; sparse replaces $N^2$ by $\operatorname{nnz}$ | Restarted $\ell=O(r)$ sweep is not a total iteration guarantee | Small gaps, orthogonality loss, omitted residual checks |
| Eigenfunction extension | $O(Nd+Nr)$ per new site | Error includes $1/\lambda_i$, sample error, gap and factor error | Tiny eigenvalues, ambiguous basis within blocks |
| Kernel CCA | Dense $O(N^3)$; feature version $O(Nm^2+m^3)$ | Sampling $O(b_{\mathrm{ctx}}^{3/2}\rho^{-3/2}h_N)$ for small $\rho$; $N\rho_N^3\to\infty$ for consistency | Overlapping windows, noncompact limit, missing source/range conditions |
| Finite-summary tail test | Same CCA cost | Beyond rank $k$, $\widehat\kappa_i\le\Delta_N(\rho)$ under rank-$k$ assumption | Activation conditional independence not actually true |
| Noise temperature | $n$ independent model replicas; $O(nd)$ statistics/site | Hoeffding absolute error; Bernstein relative certificate needs $n=O(\theta^{-1}\alpha^{-2}\log(1/\eta))$ | Drift substituted for noise; tiny scale; factor-of-two ambiguity |
| Split nearest-neighbour temperature | Brute-force $O(nn_{\mathrm{ref}}d)$ | Median error $O(c^{-1}\sqrt{\log(1/\eta)/n})$ conditional on reference | No quantile margin, ties, same-sample dependence, shrinking radii |
| Ridge-leverage Nyström | $O(Nmd+Nm^2)$ for $m\le N$, including recursive score acquisition | I.i.d. budget $m\ge10c_\ell d_{\mathrm{eff}}(\tau)\log(2N/\eta)$; recursive budget as A.52; raw error $\tau$ | Naive exact-score costs (avoidable recursively), high effective dimension, nonpositive approximation |
| Random Fourier features | $O(Nmd+Nm^2+m^3)$ | $m=O(e^{2\beta}\epsilon^{-2}\log(N/\eta))$ for entry/raw-operator error $\epsilon$ | Positivity certificate may require $e^{4\beta}$ features |
| Action-view sketch | $Nm_J$ JVPs plus sketch Gram costs | $m_J\ge8\varepsilon^{-2}\log(2N^2/\eta)$ for relative Gram error | Unshared probes, zero Jacobians, weak probe tails, $\beta e^\beta$ amplification |

---

# Part B. What the construction recovers in idealized regimes

Each cluster opens with named idealizations. Whether a layer satisfies them is a measurement (`THEORY.md` Sections 4.3 and 11), never a premise; the theorems say what is recovered when they hold, with what error when they nearly hold, and what fails when they do not.
## Part B, cluster 1 — Ideal case: predictive states, belief states and causal states

Labels B.1–B.14 (B.14 follows B.12 to keep the Mess3 results together). Population results below concern the regularized kernel CCA of `THEORY.md` §6.3, not the cross-layer smoother product of §4. Every additional assumption is explicit. Hidden-state beliefs are ground truth for the proofs and validation; they do not enter the construction's kernel or choose its basis.

**The central positive result.** If the chosen target's conditional feature mean is affine in the belief, the past view exposes the belief, and the target observes every varying belief direction, the nonzero, zero-ridge population canonical variates are exactly an invertible affine coordinate system for the reachable belief set. Theorem B.6 gives an exact calibration: a stationary low-rank, first-order Markov token process where the actual sphere-kernel carried construction satisfies these hypotheses. Its beliefs are determined by the last token; it is not a recovery theorem for non-Markov or fractal mixed-state geometry. B.12 explains why exact affine target means fail for belief-valued Mess3 activation targets; B.14 strengthens this to infinite zero-ridge rank and noncompactness. B.13 supplies a conditional approximate-recovery bound for other regimes meeting its hypotheses, not for these Mess3 targets.

**Essential qualification.** A belief is a point in a simplex, not a categorical variable with $k$ possible values. Conditional independence through a belief therefore does **not** alone imply rank $k-1$. Nor do injective conditional means imply an *affine* recovery, or finitely many causal states imply linearly independent future laws. B.8 and B.10 give explicit counterexamples. These qualifications are necessary to make the showcase theorem true, not reasons to abandon it.

### Setup and named assumptions

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

### What conditional independence proves — and what it does not

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

### Exact recovery of belief geometry

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

### Causal states, not hidden labels

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

### Readability and the token Hankel benchmark

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

### Sharp failure cases

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

### Characteristic activation targets and approximate recovery

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

### Shared tractability and sampling details

These qualifications apply to every result above.

- **How many documents?** Stationarity and ergodicity alone give no distribution-free finite-sample convergence rate, nor a universal sufficient document count. A rigorous i.i.d. regime is $n_{\mathrm{doc}}$ independent stationary documents, one eligible past/target pair per document, giving $N=n_{\mathrm{doc}}$ independent pairs. More sites per document require a mixing/dependence argument; multiplying by document length is not justified. Bootstrap and split at document level. For a known finite joint table with $J$ cells, $n_{\mathrm{doc}}\geq\log(2J/\delta)/(2\epsilon^2)$ suffices for entrywise absolute error at most $\epsilon$ with probability $1-\delta$ by Hoeffding and a union bound. Translating this to rank needs the table dimensions, covariance conditioning, ridge, and the positive singular gap. There is no parameter-free claim such as “a few documents suffice.”
- **The ridge/sample limit.** Fukumizu, Bach and Gretton (2007), *JMLR* 8:361–383, **Lemmas 6–7 and Eq. (9)** give operator-norm convergence in probability for the estimated normalized correlation operator under i.i.d. pairs, the moment hypotheses of their covariance bounds (satisfied by our bounded sphere kernels), compactness of the population correlation operator, and
  $$\rho_{n_{\mathrm{doc}}}\to0,\qquad n_{\mathrm{doc}}^{-1/3}/\rho_{n_{\mathrm{doc}}}\to0.$$
  Lemma 6's stochastic term is $O_p(\rho_{n_{\mathrm{doc}}}^{-3/2}n_{\mathrm{doc}}^{-1/2})$; Lemma 7 controls the population regularization bias under compactness, without a universal bias rate. Weyl then yields the rank-by-threshold claim in B.7 for the entire singular spectrum. Theorem 1 itself covers only the **largest simple singular pair** (NOCCO convergence in RKHS norm); Theorem 2 adds covariance-range hypotheses for $L^2$ canonical-variate convergence. Their lower-order/multidimensional extension is a remark, not a proved theorem we attribute to Theorem 1. Here isolated-block convergence instead follows directly from their operator-norm convergence and the spectral-projection argument in B.4; conversion to variate functions additionally needs source/smoothing control, as explicitly displayed in B.13. Finite-support full-feature models verify the range conditions on supported subspaces. Generic continuous-belief kernels need those conditions checked separately; B.14 proves the compactness condition fails for the universal belief-valued Mess3 targets, so these zero-ridge consistency conclusions do not cover that example. The ridge must decay **slower than** $n_{\mathrm{doc}}^{-1/3}$; for example $\rho_{n_{\mathrm{doc}}}=n_{\mathrm{doc}}^{-1/4}$ meets the condition, without any optimality claim.
- **Why not zero ridge first?** If both empirical centered Gram matrices have rank $N-1$, then $R_p,R_q\to C$ as $\rho\downarrow0$ at fixed $N$, so all $N-1$ nontrivial in-sample canonical correlations tend to one even for independent views. This is precisely why §6.3 requires document-held-out selection and evaluation. Population rank and empirical positive rank are not interchangeable.
- **What computation costs.** For lag widths $w_p,w_q$ and activation width $d$, forming dense dot-product Gram matrices is $O(N^2(w_p+w_q)d)$ in a direct implementation; the two Grams require $O(N^2)$ memory and generic dense decompositions cost $O(N^3)$. With a justified $m$-feature/Nyström approximation, moment formation and reduced linear algebra are typically $O(Nm^2+m^3)$ after feature/kernel evaluation. The accuracy and conditioning of that approximation must be checked; final context rank $r\ll N$ alone does not prove a rank-$r$ approximation of either within-view Gram is adequate. Direct finite-feature CCA and exactly grouped finite-support models avoid the $N\times N$ matrices.
- **What is estimated versus assumed.** All kernels use observed activations or token windows in the separately labelled benchmark. True beliefs, hidden labels, and causal-state identities are used only to verify ideal assumptions and recovery. No supervised probe chooses the construction's basis. The two affine readability regressions are explicitly labelled fits for validation.

### Numerical verification

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

### Discrepancies with THEORY.md

No edits to `THEORY.md` were made.

1. **§6.3's token Hankel equality needs linear minimality.** “Equal under minimality and sufficiently long windows” should specify observable and reachable linear dimension $k$, not only minimality among nonnegative $k$-state HMM realizations. Hsu–Kakade–Zhang's explicit rank condition is stronger than bare hidden-state minimality.
2. **§6.3's simplex dimension is an ambient dimension.** The ambient belief simplex has affine dimension $k-1$; the stationary reachable belief distribution may have smaller affine span. Recovery and rank should use $d_b=\operatorname{rank}\operatorname{Cov}(b)$ until full affine support is established.
3. **§6.2's low-temperature calibration wording.** The sphere kernel's low-temperature limit is linear CCA on the chosen **normalized** views (with kernel/ridge scaling handled), not in general raw-activation CCA. A raw-linear activation calibration is a separate declared variant. Linear belief-sufficiency before normalization does not alone give exact linear readability from the normalized activation.
4. **The boundary matters for §6.3 and §11's belief comparison.** With the literal past positions $<t$, belief-sufficiency at the current site does not ensure access to its current belief. In the one-lag convention above the recovered state is $b_{t-1}$; comparing it with $b_t$ requires another assumption or an explicitly different alignment.
5. **§6.3 and §11 correctly label activation rank and readability as hypotheses.** Even strengthening them to exact belief-sufficiency plus belief-Markov would not make them theorems: B.10 is a counterexample. ATM, feature completeness, and observability yield a valid sufficient regime. This is a qualification of a proposed strengthening, not a contradiction of the existing caution.
6. **Effective context rank is not exact population rank.** Positive ridge, finite windows, held-out thresholds, and sampling remain part of §6.3's definition. Statements equating it with $k-1$ or causal-state count minus one require the extra threshold/consistency conditions in B.7. Finite causal-state count additionally requires affine independence of future laws.
7. **Zero read updates alter the evaluated population.** In the previous-token-copy example, a sphere kernel on the update excludes mass $1/V$ under the shared zero policy. Using $h^{L+1}$ as allowed in §6.3 avoids this issue; the exact conditional-on-retention statement is in B.11.

8. **The Mess3 qualification of §6.3/§11 is stronger than failure of exact ATM.** B.14 proves that even an exactly linear belief-sufficient stationary Mess3 layer has infinite zero-ridge population carried/read-style rank under universal sphere kernels, with all variational canonical levels bounded below. Thus $r_{\mathrm{ctx}}\approx k-1$ can only be a declared **ridge-, threshold-, and window-dependent effective-rank hypothesis**, not an intrinsic population rank limit in that regime. An alternative is an explicitly non-universal/raw-linear kernel on belief-linear activations, or the token-future benchmark; a low-temperature sphere kernel on normalized activations is not automatically that raw-linear variant. This does not refute the existing scale-dependent empirical hypothesis, but it rules out the proposed perfect-belief rationale for a universal zero-ridge $k-1$ cutoff. Fukumizu–Bach–Gretton compact-limit consistency cannot supply that missing limit. Neither this document nor its checks edits the theory or experiment protocol.

### Sources

- Adam S. Shai, Sarah E. Marzen, Lucas Teixeira, Alexander Gietelink Oldenziel and Paul M. Riechers (2024). **Transformers Represent Belief State Geometry in their Residual Stream.** *Advances in Neural Information Processing Systems* 37, 75012–75034; [arXiv:2405.15943](https://arxiv.org/abs/2405.15943). Empirical motivation and mixed-state geometry, not a theorem that arbitrary trained layers satisfy PA/ATM/BO.
- Cosma Rohilla Shalizi and James P. Crutchfield (2001). **Computational Mechanics: Pattern and Prediction, Structure and Simplicity.** *Journal of Statistical Physics* 104, 817–879; [arXiv:cond-mat/9907176](https://arxiv.org/abs/cond-mat/9907176). Corollary 1 and Lemma 7 for sufficient statistics and refinement; Theorems 1–3 for prescience, statistical-complexity minimality, and uniqueness. The sufficient-statistic argument used here is also proved directly in B.7.
- Daniel Hsu, Sham M. Kakade and Tong Zhang (2012). **A Spectral Algorithm for Learning Hidden Markov Models.** *Journal of Computer and System Sciences* 78(5), 1460–1480; [arXiv:0811.4413](https://arxiv.org/abs/0811.4413). §2.3 Condition 1 and Lemma 2, including the factorization $P_{2,1}=OT\operatorname{diag}(\pi)O^\top$ under their column-transition convention. The more general finite-window result needed here is proved in B.5.
- Kenji Fukumizu, Francis R. Bach and Arthur Gretton (2007). **Statistical Consistency of Kernel Canonical Correlation Analysis.** *Journal of Machine Learning Research* 8(14), 361–383; [paper and PDF](https://www.jmlr.org/papers/v8/fukumizu07a.html). Lemmas 6–7 and Eq. (9) for operator-norm consistency and the stochastic error term; Theorems 1–2 only within their largest-pair/source-condition scope, as specified above.
- Standard spectral facts used in B.4: polar decomposition, the spectral theorem, and continuity of an isolated singular subspace; see, e.g., G. W. Stewart and Ji-guang Sun (1990), *Matrix Perturbation Theory*, for the finite-dimensional singular-subspace result. The compact finite-rank reduction and the strong-to-norm convergence step are given explicitly in the proof.

- Ingo Steinwart and Andreas Christmann (2008). **Support Vector Machines.** Springer, §4.6 (universal kernels and Taylor-kernel criteria). The sphere specialization is proved in B.4.
- Per-Åke Wedin (1972). **Perturbation bounds in connection with singular value decomposition.** *BIT Numerical Mathematics* 12, 99–111. B.13 proves the residual-form subspace estimate used here directly, alongside Weyl’s singular-value inequality.

- Roger A. Horn and Charles R. Johnson (2013). **Matrix Analysis**, second edition. Cambridge University Press, Chapter 8, for Perron–Frobenius and convergence of normalized powers of a strictly positive matrix; the positivity and invertibility needed in B.12 are verified there directly.

---
## Part B, cluster 2 — Ideal dictionaries, sparse concept clusters, and ideal sparse autoencoders

This cluster concerns a **conditional regime**, not an assumption of the construction. Its clusters are not supplied to the sphere kernel or its SVD. They describe an idealized data-generating geometry, and are used to state and validate recovery. An ideal sparse autoencoder below is a **comparison baseline**, not an extra trained component of the construction.

Throughout, $\mathcal H=\mathbb R^N$ has counting measure, $K_\beta[x,y]=\exp(\beta\langle\hat\phi(x),\hat\phi(y)\rangle)$, and $P=D_vK_\beta D_v$ is the symmetric bistochastic normalization. Eigenvalues here are numbered $\lambda_1\ge\cdots\ge\lambda_N$, **including the constant mode**. Thus an $m$-cluster block means the constant plus $m-1$ nonconstant modes. One must restore the constant before rotating this block into cluster indicators. All operator norms are Euclidean spectral norms.

### The finite-temperature dictionary window

**Definition B.20 (angular cluster regime and ideal dictionary regime).** A partition $X=C_1\sqcup\cdots\sqcup C_m$, with $n_j=|C_j|>0$, is an **$(\varepsilon,\Delta)$ angular cluster regime** if

$$
\langle\hat\phi(x),\hat\phi(y)\rangle\ge1-\varepsilon\quad(x,y\in C_j),\qquad
\langle\hat\phi(x),\hat\phi(y)\rangle\le1-\Delta\quad(x\in C_j,y\in C_l,j\ne l),
$$

where $0\le\varepsilon<\Delta\le2$. Write $U=[\mathbf1_{C_j}/\sqrt{n_j}]_{j=1}^m$, $\Pi=UU^\top$, and

$$
\mathcal R=\max_j\sum_{l\ne j}\sqrt{n_l/n_j}\le(m-1)\sqrt{n_{\max}/n_{\min}}.
$$

The **ideal dictionary regime** is $\varepsilon=0$: each cluster is a single normalized direction, although raw activation norms may vary. We assume $2\le m<N$ in the spectral-gap statements; the cases $m=1$ and $m=N$ have the evident missing contrast or missing tail. A claimed dictionary of original activations additionally specifies the loading view $H$; cosine geometry does not fix its radial amplitudes.

**Theorem B.21 (exact rays: exact indicator span and an explicit spectral gap).** In the ideal dictionary regime, put $\omega_{\beta}=e^{-\beta\Delta}$ and assume $2\omega_{\beta}\mathcal R<1$. Then

$$
\operatorname{range}P=\operatorname{range}U,\qquad
\lambda_m(P)\ge1-2\omega_{\beta}\mathcal R>0,\qquad \lambda_{m+1}(P)=\cdots=\lambda_N(P)=0.
$$

Consequently precisely $m$ eigenvalues exceed $(1-2\omega_{\beta}\mathcal R)/2$, the top block's principal angles to $\operatorname{range}U$ are **zero**, and its gap is at least $1-2\omega_{\beta}\mathcal R$. This is stronger than an asymptotic recovery assertion.

*Proof.* Rescale the kernel by $e^{-\beta}$, which does not change $P$ after balancing, and call it $A$. Within each cluster $A$ is identically one; across clusters its entries are at most $\omega_{\beta}$. Equal normalized vectors have identical inner products with every vector, so $A$ is constant on every cluster rectangle. Uniqueness of the positive balancing vector and permutation symmetry within a cluster imply that its rescaled balancing entries, denoted $\widetilde v_x$, equal a number $\widetilde v_j$ on $C_j$. Hence $P$ is block-constant and annihilates $U^\perp$.

The row equation gives $n_j\widetilde v_j^2\le1$. Its cross-cluster leakage is therefore

$$
\ell_j=\sum_{l\ne j}n_l\widetilde v_j\widetilde v_l A_{jl}
\le \omega_{\beta}\sum_{l\ne j}\sqrt{n_l/n_j}\le \omega_{\beta}\mathcal R.
$$

Within $C_j$, $P[x,y]=(1-\ell_j)/n_j$. Each row of $P-\Pi$ consequently has absolute sum $2\ell_j$. Symmetry implies $\|P-\Pi\|\le2\omega_{\beta}\mathcal R$. The Rayleigh quotient on $\operatorname{range}U$ is at least $1-2\omega_{\beta}\mathcal R$, establishing both its full rank $m$ and the asserted spectrum. ∎

**What it buys.** Repeated, separated directions are recovered as an exact site-space dictionary subspace at finite temperature, even with unequal frequencies. For two views satisfying this theorem on the same partition, write $P^L=UM_LU^\top$: then $Q=P^{L'}P^L=UM_{L'}M_LU^\top$ has both nonzero singular subspaces exactly equal to $\operatorname{range}U$ and smallest nonzero singular value at least $(1-2\omega_{\beta,L'}\mathcal R)(1-2\omega_{\beta,L}\mathcal R)$, where $\omega_{\beta,L}=e^{-\beta_L\Delta_L}$, by the elementary minimum-singular-value product inequality.

**Tractability and sampling.** Forming a dense kernel costs $O(N^2d)$, each balancing iteration $O(N^2)$, and each dense block eigensolver application $O(N^2m)$; iteration counts depend on conditioning and tolerance. One observed site per ray is enough for this deterministic conclusion, but rare-ray coverage and the size factor $\mathcal R$ matter. B.34 gives the occupancy and statistical-gap interface. Separation is amplified by $e^{\beta\Delta}$; exact duplicates, unlike distinct nearby sites, do not split into singleton modes as $\beta\to\infty$.

**Theorem B.22 (angular clouds: leakage, within-cluster mixing, and recovery).** In Definition B.20 set

$$
\alpha_{\beta}=e^{-\beta\varepsilon},\quad \omega_{\beta}=e^{-\beta\Delta},\quad
q=\frac{\omega_{\beta}\mathcal R}{\alpha_{\beta}},\quad
\ell=\frac{q}{\sqrt{\alpha_{\beta}(1-q)}},\quad
\gamma_{\rm mix}=\frac{\alpha_{\beta}^2}{(1+\omega_{\beta}\mathcal R)^2},\quad \eta=2\ell.
$$

Assume $q<1$ and $2\eta<\gamma_{\rm mix}$. Then

$$
\lambda_m(P)\ge1-\eta,\qquad
\lambda_{m+1}(P)\le1-\gamma_{\rm mix}+\eta,\qquad
\lambda_m(P)-\lambda_{m+1}(P)\ge \gamma_{\rm mix}-2\eta>0.
$$

Exactly $m$ eigenvalues exceed $1-\gamma_{\rm mix}/2$. If $V$ is any orthonormal basis for their span, then

$$
\|VV^\top-\Pi\|=\|\sin\Theta(V,U)\|
\le \tau:=\frac{\eta}{\gamma_{\rm mix}-\eta}.
$$

For bounded $\mathcal R$, $\beta\Delta\to\infty$ with $\beta\varepsilon$ bounded implies $\tau\to0$ and eventually satisfies the hypotheses. For fixed $\varepsilon>0$, this does **not** justify sending $\beta$ to infinity.

*Proof.* Again use $A=e^{-\beta}K_\beta$, with balancing vector $\widetilde v=e^{\beta/2}v$; within-cluster entries lie in $[\alpha_{\beta},1]$, across-cluster entries in $(0,\omega_{\beta}]$. Put $S_j=\sum_{x\in C_j}\widetilde v_x$ and $T_{-j}=\sum_{l\ne j}S_l$. Summing the row equations within $C_j$ yields

$$
\alpha_{\beta}S_j^2\le n_j\le S_j^2+\omega_{\beta}S_jT_{-j}.
$$

The upper bound $S_j\le\sqrt{n_j/\alpha_{\beta}}$ implies

$$
\omega_{\beta}S_jT_{-j}\le (\omega_{\beta}/\alpha_{\beta})n_j\sum_{l\ne j}\sqrt{n_l/n_j}\le qn_j,
\qquad S_j\ge\sqrt{n_j(1-q)}.
$$

Individual row equations give

$$
\frac1{S_j+\omega_{\beta}T_{-j}}\le \widetilde v_x\le\frac1{\alpha_{\beta}S_j}\quad(x\in C_j).
$$

It follows that every row's cross-cluster mass is at most

$$
\omega_{\beta}\widetilde v_xT_{-j}\le\frac{\omega_{\beta}\mathcal R}{\alpha_{\beta}^{3/2}\sqrt{1-q}}=\ell,
$$

and every within-cluster entry satisfies

$$
P[x,y]=\widetilde v_xA[x,y]\widetilde v_y\ge\frac{\alpha_{\beta}^2}{n_j(1+\omega_{\beta}\mathcal R)^2}=\frac{\gamma_{\rm mix}}{n_j}.
$$

Construct $P_0$ by deleting cross-cluster entries of $P$ and adding each deleted row sum to that row's diagonal. It is symmetric, stochastic and block-diagonal. It is PSD: the retained principal blocks of $P$ are PSD and the added diagonal is nonnegative. It fixes $U$, and $P_0-\gamma_{\rm mix}\Pi$ is entrywise nonnegative with row sum $1-\gamma_{\rm mix}$. Thus its restriction to $U^\perp$ has norm at most $1-\gamma_{\rm mix}$. Also $E=P-P_0$ is symmetric with absolute row sums at most $2\ell$, so $\|E\|\le\eta$. The variational characterization of eigenvalues now proves the two spectral inequalities.

For completeness the needed subspace estimate can be proved directly, without an unspecified perturbation constant. Let $D_0=P_0|_{U^\perp}$, let $\Lambda$ be the top-$m$ eigenvalue matrix of $P$, and write $Z=U_\perp^\top V$. Its equation is

$$
D_0Z-Z\Lambda=-U_\perp^\top EV.
$$

Since $D_0\le(1-\gamma_{\rm mix})I$ and $\Lambda\ge(1-\eta)I$, its solution is

$$
Z=\int_0^\infty e^{tD_0}(U_\perp^\top EV)e^{-t\Lambda}\,dt,
\qquad \|Z\|\le\frac{\eta}{\gamma_{\rm mix}-\eta}.
$$

For equal-dimensional subspaces this norm equals the largest principal-angle sine and the projector distance. These identities follow directly from the singular values of $U^\top V$ (the cosine-sine decomposition). Finally, bounded $\beta\varepsilon$ bounds $\alpha_{\beta}$ away from zero while $\omega_{\beta}\to0$, proving the limit assertion. ∎

**What it buys.** The whole sphere-kernel–Sinkhorn–top-block pipeline has a finite-sample recovery window that is checked **a posteriori against the partition extracted from the localized block** (or against a specified validation partition). The constants depend on that partition; they are not available from the spectrum alone. This is a legitimate geometric certificate because the theorem applies to every partition satisfying its inequalities. The proof avoids a generic Sinkhorn stability claim: it bounds the actual balanced matrix directly, and allows nonconstant within-cluster geometry even when $\beta\varepsilon$ is bounded but not small.

**Tractability and sampling.** Costs are as in B.21; computing the geometric constants exactly takes an additional $O(N^2)$ pass over the Gram matrix. The lower mixing bound pays roughly $e^{-2\beta\varepsilon}$, while leakage pays $e^{-\beta\Delta}$ with further factors of $e^{\beta\varepsilon}$. These conservative constants expose both the separation amplifier and the within-cluster price. To see the empirical gap above numerical/statistical error, apply B.34 with $g=\gamma_{\rm mix}-2\eta$; no number of samples repairs failure of the geometric window itself.

### From an identifiable block to a dictionary

**Theorem B.23 (the exact maximizers of a localizing rotation).** For an orthonormal frame $V$ define

$$
\mathcal F(V)=\sum_{x=1}^N\sum_{j=1}^m V[x,j]^4.
$$

If $\operatorname{range}V=\operatorname{range}U$, every global maximizer of $\mathcal F(V\Omega)$ over $\Omega\in O(m)$ is $U$ up to signed column permutation. This is an unweighted maximum-IPR rotation. The usual unnormalized varimax objective has the same maximizers, since the subtracted squared column norms are constant on orthonormal frames.

There is also a quantitative, though deliberately conservative, noisy statement. If $\|VV^\top-\Pi\|\le \tau$, put

$$
e_{\rm rot}=\sqrt{2m}\,\tau,\qquad J_{\rm rot}=4\sqrt m\,e_{\rm rot},\qquad T_{\rm rot}=2n_{\max}J_{\rm rot}.
$$

If $T_{\rm rot}<1/2$, equivalently $\tau<1/(16\sqrt2\,m\,n_{\max})$, any global maximizer $W=V\Omega$ obeys

$$
\min_{M\text{ signed permutation}}\|W-UM\|_F\le e_{\rm rot}+\sqrt{2T_{\rm rot}}.
$$

This bound is of square-root order in the subspace error, not a claim of linear-rate robustness for practical varimax.

*Proof.* In the exact case write $V\Omega=UO$. Then

$$
\mathcal F(UO)=\sum_{i=1}^m\frac1{n_i}\sum_{j=1}^m O_{ij}^4
\le\sum_{i=1}^m\frac1{n_i}.
$$

Each row of $O$ has squared norm one. Equality in $\sum_jO_{ij}^4\le(\sum_jO_{ij}^2)^2=1$ requires exactly one nonzero entry in each row. Orthogonality then makes $O$ a signed permutation. All such permutations attain equality.

For the noisy statement use the polar/Procrustes alignment of $V$ to $U$. Its singular-value formula gives a frame $V_0$ with $\|V_0-U\|_F\le\sqrt{2m}\tau=e_{\rm rot}$: each squared contribution is $2(1-\cos\theta_i)\le2\sin^2\theta_i$. On unit-norm columns the gradient of the fourth-power sum has Euclidean norm at most four. The segment between two unit-norm columns lies in the unit ball. Hence

$$
|\mathcal F(V_0O)-\mathcal F(UO)|\le4\sqrt m\|V_0-U\|_F\le J_{\rm rot}
$$

uniformly over $O\in O(m)$. For an optimizer $O_*$, comparison with $O=I$ gives

$$
\sum_i\frac1{n_i}\left(1-\sum_jO_{*ij}^4\right)\le2J_{\rm rot},
\qquad\sum_i\left(1-\sum_jO_{*ij}^4\right)\le T_{\rm rot}.
$$

Every row therefore has a squared entry greater than $1/2$, since its maximum squared entry is at least its fourth-power sum. These entries must occupy distinct columns, because each column's squared entries sum to one. Choose their signs to obtain a signed permutation $M$. Then

$$
\|O_*-M\|_F^2
=2\sum_i(1-|O_{*i,\pi(i)}|)
\le2\sum_i(1-\max_jO_{*ij}^2)
\le2T_{\rm rot}.
$$

The triangle inequality completes the proof. ∎

**What it buys.** A specified rotation, not merely the existence of an indicator basis, provably returns the centroid dictionary in the exact regime. The noisy result separates block stability from the harder nonconvex basis-selection problem.

**Tractability and sampling.** A dense pairwise varimax/max-IPR sweep costs $O(Nm^2)$, with implementation-dependent $O(m^3)$ orthogonal-factor work. This is a cost per sweep, **not** a global-optimization guarantee: a local optimizer need not attain the theorem's maximum. B.34 controls the input subspace error. The threshold $\tau<1/(16\sqrt2\,m\,n_{\max})$ makes this bound inapplicable at many ordinary subspace errors: already for $m=2$, $n_{\max}=4$, it requires $\tau<0.00552$, so $\tau=0.01$ is outside the guarantee. Where applicable it can still be extremely loose. Reviewer experiments with exhaustive $O(2)$ angle scans observed approximately $1.4\tau$ frame error in tested small-noise examples; this is an **empirical observation, not a linear-rate theorem**, and not a guarantee for an iterative varimax implementation.

**Theorem B.24 (finite-search, a posteriori indicator certificate).** Given only $V$, enumerate partitions of $X$ into $m$ nonempty sets and compute

$$
d_*:=\min_{\mathcal C}\|VV^\top-\Pi_{\mathcal C}\|,
$$

where $\Pi_{\mathcal C}$ is the normalized-indicator projector. If the **computed** value $d_*<1/(2N)$, the minimizing unordered partition is unique. Let $U_*$ be its indicator frame. The declared rotation

$$
W=VV^\top U_*\,(U_*^\top VV^\top U_*)^{-1/2}
$$

is orthonormal, belongs to $\operatorname{range}V$, and satisfies

$$
\|W-U_*\|_F\le\sqrt{2m}\,d_*.
$$

Moreover, if a specified planted partition has projector error $\tau<1/(2N)$, it is this unique minimizer, $d_*=\tau$, and $U_*=U$ up to permutation. The computable certificate alone establishes proximity to a unique nearby partition, not that its groups have a particular latent or semantic interpretation.

*Proof.* Two different unordered partitions disagree on whether some pair of sites belongs to the same set. Their projector entries on that pair differ by at least $1/N$, so their operator-norm distance is at least $1/N$. If two minimizers had distance $d_*<1/(2N)$ from $VV^\top$, their mutual distance would be at most $2d_*<1/N$, a contradiction. Invertibility in the displayed formula follows from $d_*<1$. Its Gram matrix is the identity, and its singular-angle alignment with $U_*$ has squared error $2\sum_i(1-\cos\theta_i)\le2md_*^2$, as in B.23. If a planted partition has error $\tau<1/(2N)$, any other candidate has distance at least $1/N-\tau>\tau$, proving the last assertion. ∎

**What it buys.** Exhaustive finite search gives a data-checkable certificate of a uniquely nearby indicator dictionary and a frame-error bound linear in the computed distance. Identifying this dictionary with a planted partition additionally requires the stated recovery hypothesis; no practical varimax convergence is assumed.

**Tractability and sampling.** This is a certification benchmark, not a scalable recommendation: there are $S(N,m)$ set partitions. Each candidate distance can be obtained from the singular values of its $m\times m$ overlap matrix with $V$, costing at most $O(Nm^2+m^3)$. The threshold $d_*<1/(2N)$ is conservative and can demand high accuracy; use B.34 to assess whether sampling error leaves a margin below it. No labelled partition enters the finite-search rule.

**Theorem B.25 (loadings are the centroid dictionary, with error transport).** For any loading view $H\in\mathbb R^{N\times d}$,

$$
a_H(U_j)=H^\top U_j=\sqrt{n_j}\,\bar h_j,
\qquad
UU^\top H[x]=\bar h_j\quad(x\in C_j).
$$

If a recovered frame $W$ is aligned to $U$ and $\|W-U\|_F\le E_{\rm load}$, then

$$
\|H^\top W-H^\top U\|_F\le\|H\|E_{\rm load},
\qquad
\|H^\top W\operatorname{diag}(n_j^{-1/2})-[\bar h_j]_j\|_F
\le\frac{\|H\|E_{\rm load}}{\sqrt{n_{\min}}}.
$$

Also $\|VV^\top H-UU^\top H\|_F\le \tau\|H\|_F$ whenever the projector error is at most $\tau$.

*Proof.* Expand $H^\top\mathbf1_{C_j}/\sqrt{n_j}$ as a sum over the cluster. The loading and projector differences follow from $\|AB\|_F\le\|A\|\|B\|_F$; right multiplication by $\operatorname{diag}(n_j^{-1/2})$ contributes at most its operator norm $1/\sqrt{n_{\min}}$. ∎

**What it buys.** The recovered activation-space dictionary is not inferred by analogy with an SAE: it is exactly the cluster centroid dictionary, with the counting-measure factor $\sqrt{n_j}$ made explicit. Norm variation lost by the sphere kernel reappears in the raw loadings, but is not recovered as a sitewise amplitude code by the indicator basis alone.

**Tractability and sampling.** Loadings cost $O(Nmd)$; centroid accumulation costs $O(Nd)$. B.34 bounds the spectral contribution to error. Population-centroid accuracy is a separate concentration problem requiring amplitude/moment assumptions on $H$; increasing ray coverage alone gives no uniform guarantee for heavy-tailed raw amplitudes.

### What an ideal sparse autoencoder must — and need not — recover

**Theorem B.26 (hard one-hot reconstruction equals the recovered centroids).** Set $f_x=\hat\phi(x)$, or more generally use a common positive multiple of these vectors. Consider the **one-hot** baseline

$$
\min_{d_1,\ldots,d_m,\,z_x\in\lbrace e_1,\ldots,e_m\rbrace}
\sum_x\|f_x-Dz_x\|^2.
$$

This is $k$-means with $k=m$, not free-amplitude one-sparse coding. Under Definition B.20, if

$$
(N-m)\varepsilon<\Delta,
$$

every global optimum uses exactly the planted partition, up to permutation, and its atoms are the cluster centroids. When B.21 or B.22 and a localization guarantee also apply, use **the same reconstruction matrix** $F=[f_x^\top]_x$ as the loading view. Then $a_F(U_j)/\sqrt{n_j}=\bar f_j$, and for an aligned recovered frame with $\|W-U\|_F\le E_{\rm load}$, B.25 gives dictionary error at most $\|F\|E_{\rm load}/\sqrt{n_{\min}}$. Loadings from another view, including raw activations with unequal norms, need not equal this baseline's atoms.

*Proof.* For a set $A$ of $s$ vectors,

$$
\sum_{x\in A}\|f_x-\bar f_A\|^2
=\frac1s\sum_{\lbrace x,y\rbrace\subset A}\|f_x-f_y\|^2.
$$

The planted partition therefore costs at most $\sum_j(n_j-1)\varepsilon=(N-m)\varepsilon$. Any different partition into at most $m$ nonempty reconstruction groups must have a mixed group: if all its groups were contained in true clusters, using at most $m$ groups to cover $m$ nonempty clusters would force exactly the original partition. A mixed group contains $x,y$ from different clusters. For any center $d$ its cost is at least

$$
\|f_x-d\|^2+\|f_y-d\|^2\ge\tfrac12\|f_x-f_y\|^2\ge\Delta.
$$

Thus no wrong partition can minimize. For a fixed nonempty group the unique least-squares center is its mean. Common rescaling multiplies both comparison costs by the same square. ∎

**What it buys.** In an explicitly tight angular regime, the spectral construction and an ideal hard autoencoder agree on the same finite-sample dictionary **when their loading and reconstruction matrices are identical**. This is a global-optimum characterization, not a claim about training dynamics.

**Tractability and sampling.** Global $k$-means is a baseline optimization problem; Lloyd iterations cost $O(Nmd)$ but do not certify its optimum. The sufficient separation condition is intentionally conservative and depends on $N$; it should not be interpreted as a sharp population threshold. B.34 governs spectral sample error, while the displayed inequality must be checked independently. Angular separation alone is insufficient for this claim on raw activations with arbitrary norms.

**Cluster-size obstruction (example).** Let an even-sized cluster of $n$ unit vectors consist of two repeated directions, each occurring $n/2$ times, with mutual dot product $1-\varepsilon$. In an orthogonal plane place two singleton clusters whose directions have dot product $1-\Delta$, where $0<\varepsilon<\Delta\le1$. Here $N=n+2$, $m=3$, and the planted cost is $n\varepsilon/2$. Splitting the large cluster and merging the two singletons costs $\Delta$, so the planted partition cannot minimize when $\varepsilon>2\Delta/n$. This proves that an inverse-cluster-size separation requirement is not merely a proof artifact; it does not claim that B.26's constant is optimal. For $n=8$, $\Delta=0.1$, $\varepsilon=0.0275$, the costs are $0.11$ versus $0.10$ and $(N-m)\varepsilon/\Delta=1.925$.

**Proposition B.27 (free-amplitude one-sparse coding is not centroid fitting).** The optimization

$$
\min_{\|z_x\|_0\le1,\,D}\sum_x\|f_x-Dz_x\|^2
$$

is not $k$-means. Even in Definition B.20 with two clusters and arbitrarily small nonzero angular spread, its optimal atom directions need not equal the centroid directions. Without atom normalization, its scale is unidentifiable.

*Proof.* For a fixed group assigned to one atom, eliminating scalar codes projects its data onto a line. The optimal line is the leading eigendirection of its uncentered second-moment matrix, by the Rayleigh principle. In $\mathbb R^3$ take

$$
C_1=(e_1,e_1,\xi_\theta),\quad
\xi_\theta=(\cos\theta,\sin\theta,0),\qquad
C_2=(e_3,e_3,e_3),\qquad0<\theta<\pi/2.
$$

These clusters satisfy $\varepsilon=1-\cos\theta<\Delta=1$. Assigning a separate line to each cluster has total loss at most $\sin^2\theta<1$. Any different assignment into at most two nonempty groups has a mixed group. Such a group contains one unit vector from the first plane and one copy of $e_3$, an orthonormal pair whose best rank-one reconstruction loss is one. Thus every global optimum uses the planted assignment.

The first centroid ray is $2e_1+\xi_\theta$, whereas

$$
(2e_1e_1^\top+\xi_\theta\xi_\theta^\top)(2e_1+\xi_\theta)
=2(2+\cos\theta)e_1+(2\cos\theta+1)\xi_\theta.
$$

Independence of $e_1$ and $\xi_\theta$ implies parallelism only if $2+\cos\theta=2\cos\theta+1$, impossible here. The two planar eigenvalues are distinct, so the optimal line is unique. In the small-$\theta$ branch its angle is $\tfrac12\arctan(\sin(2\theta)/(2+\cos(2\theta)))$, whereas the centroid angle is $\arctan(\sin\theta/(2+\cos\theta))$. Taylor expansion gives their difference $\theta^3/27+O(\theta^5)$, approximately $4.63\times10^{-6}$ radians at $\theta=0.05$. Finally $D[:,j]\mapsto tD[:,j]$, $z_{xj}\mapsto z_{xj}/t$ leaves reconstruction unchanged for $t\ne0$. ∎

**What it buys.** This prevents a false exact equivalence between centroid fitting and general sparse dictionary learning, without implying a large discrepancy for tight clusters: the exhibited directional difference is third order in their angular spread.

**Tractability and sampling.** The example requires only two line fits; exhaustive two-atom assignments are checked in the accompanying script. Repeating these data arbitrarily many times does not remove the discrepancy. B.34 handles sampling perturbations only after the baseline estimand has been specified correctly.

**Theorem B.28 (an ideal free-code SAE on exact rays).** Suppose the raw data are $h_x=t_xd_j$ for $x\in C_j$, with $t_x>0$, unit vectors $d_j$, and **pairwise distinct unoriented lines** $\operatorname{span}(d_j)$. Give the baseline exactly $m$ unit-norm atoms, arbitrary scalar codes with at most one nonzero per site, and squared reconstruction loss. Every global minimum has zero loss and dictionary columns $d_j$ up to signed permutation. For the **nonnegative-code** baseline, the hypothesis can instead be weakened to pairwise distinct **oriented rays**, allowing antipodal pairs: every global minimum then has exactly the positively oriented atoms $d_j$, up to permutation. Its atom directions therefore equal the directions of the construction's exact loadings whenever B.21's separation condition holds:

$$
a_H(U_j)=\sqrt{n_j}\,\bar t_jd_j.
$$

*Proof.* The displayed dictionary and codes $t_xe_j$ attain zero loss. At any global minimum each nonzero datum must lie on the line of the single atom used for it. The $m$ observed distinct lines require at least $m$ atom lines. Width exactly $m$ forces a bijection, and unit norm fixes the atom up to sign. For nonnegative codes each datum must instead lie on the positively oriented ray of its assigned atom. Distinct observed rays require distinct atoms, including separate atoms for antipodal rays; width exactly $m$ and unit norm again force the claimed bijection. The loading formula is B.25. ∎

**What it buys.** This is an actual one-sparse SAE identifiability theorem, rather than a relabelled clustering objective: on exact, distinct rays its learned directions must coincide with the spectral dictionary. It does not identify an encoder network away from the observed sites.

**Tractability and sampling.** Every line (signed-code case) or oriented ray (nonnegative-code case) must be observed; B.34 supplies the corresponding rare-component coverage bound. This is a theorem about global minimizers, not an optimization-time guarantee. Antipodal rays violate the signed-code hypothesis but are covered by the nonnegative-code version. Overcomplete width permits duplicate or unused atoms and removes either uniqueness conclusion. Raw amplitudes are recovered by the baseline's scalar codes, not by the indicator coordinates alone.

### Sparse sums: recovery of active sets versus recovery of features

**Theorem B.29 (a controlled sparse-sum model produces angular clusters).** Let $d_1,\ldots,d_m$ be orthonormal unit feature directions, and let every nonzero site be

$$
h_x=\sum_{i\in S_x}a_{xi}d_i,\qquad 1\le|S_x|\le k,\qquad a_{xi}>0.
$$

Assume that for each site some $t_x>0$ satisfies

$$
(1-\rho)t_x\le a_{xi}\le(1+\rho)t_x\quad(i\in S_x),\qquad0\le\rho<1.
$$

Equivalently a uniform within-site coefficient ratio bound $A\ge1$ permits $\rho=(A-1)/(A+1)$. Define

$$
\gamma_k=1-\sqrt{1-1/k},\quad
\varepsilon_*=8\rho^2,\quad\Delta_*=\gamma_k-4\rho.
$$

If $\varepsilon_*<\Delta_*$, the nonempty populated active sets $C_S=\lbrace x:S_x=S\rbrace$ form Definition B.20's regime with these constants. B.21 applies when $\rho=0$; B.22 applies in its stated finite-temperature window otherwise. Their cluster count is the number $M$ of populated supports, **not** necessarily the number $m$ of features; replace $m$ by $M$ in those theorems. Every populated singleton cluster has centroid a positive multiple of its feature direction, exactly.

More generally, orthogonality can be replaced by the following explicit data assumption: the normalized equal-coefficient prototypes $p_S=\sum_{i\in S}d_i/\|\sum_{i\in S}d_i\|$ are nonzero and mutually separated by $\langle p_S,p_T\rangle\le1-\gamma$, and every site has $\|h_x/t_x-\sum_{i\in S_x}d_i\|\le\rho\|\sum_{i\in S_x}d_i\|$. Then the same conclusion holds with $\gamma$ replacing $\gamma_k$.

*Proof.* For nonzero $\xi,\chi$,

$$
\left\|\frac{\xi}{\|\xi\|}-\frac{\chi}{\|\chi\|}\right\|
\le\frac{2\|\xi-\chi\|}{\|\chi\|},
$$

by adding and subtracting $\xi/\|\chi\|$ and using the reverse triangle inequality. Orthogonality gives relative error at most $\rho$ from the equal-coefficient prototype, hence $\|\hat h_x-p_{S_x}\|\le2\rho$. Two sites on the same support differ by at most $4\rho$, so their dot product is at least $1-8\rho^2$.

For distinct supports of sizes at most $k$, the prototype dot product is $|S\cap T|/\sqrt{|S||T|}$. If one is strictly contained in the other, it is at most $\sqrt{(k-1)/k}$. If neither contains the other, writing $r=|S\cap T|$ gives an upper bound $r/(r+1)\le(k-1)/k\le\sqrt{(k-1)/k}$. The statement also holds at $k=1$, where the dot products are zero. Perturbing each of two unit vectors by at most $2\rho$ changes their dot product by at most $4\rho$, proving the cross-support bound. The same argument proves the general version directly from its stated prototype assumptions. On a singleton, $h_x=a_{xi}d_i$, whose mean is a positive multiple of $d_i$. ∎

**What it buys.** Sparse generative structure can produce exactly the geometry that the construction recovers, and isolated-feature clusters expose the original feature directions through their loadings. Orthogonality here is a strong hypothesis about this toy, not an additional metric imposed by the construction; the general prototype formulation allows nonorthogonal directions but requires checking their separation.

**Tractability and sampling.** There can be $M\le\sum_{s=1}^k\binom ms$ populated supports; all spectral/localization costs and occupancy bounds use $M$. Sparsity alone does not ensure that only roughly $m$ singletons and a few pairs are observed. Under independent Bernoulli feature activation with probability $p$, a particular singleton has probability $p(1-p)^{m-1}$ and the probability of at least two active features is at most $\binom m2p^2$, by a union bound over active pairs. Thus $mp\ll1$ makes multi-active sites rare, but coverage still depends on sample size; condition on and report any all-zero exclusion. B.34 must cover the supports actually targeted. Merely saying the coefficient ratio is bounded is insufficient: the bound must be small enough relative to the prototype separation.

This theorem is a calibration related to Elhage et al., *Toy Models of Superposition* (Transformer Circuits Thread, 2022, arXiv:2209.10652), not a theorem about their trained solutions. Their active amplitudes are Uniform$[0,1]$, so multi-active sites have no uniformly bounded within-site coefficient ratio, and the columns of $W$ are nonorthogonal in superposition. Neither the near-equal-coefficient orthogonal model nor its explicitly separated-prototype extension is therefore automatic. Singleton supports automatically have a fixed normalized ray when their feature direction is nonzero; separation between different singleton rays must still be checked. Likewise, Ferreira da Silva and Heimersheim, *Compressed Computation under L4 Loss is likely Computation in Superposition* (2026, arXiv:2607.04800v1), supplies an empirical comparison, not these hypotheses. Its signed sparse inputs, gates, nonorthogonal codewords and off-code effects do not satisfy an exact positive additive model in general; `reviews/papers.md` explains those limitations.

**Proposition B.30 (support-span representation and nonidentifiability without singleton observations).** If $C_S$ are nonempty active-set clusters, the feature-presence functions obey

$$
g_i(x)=\mathbf1_{i\in S_x}=\sum_{S\ni i}\mathbf1_{C_S}(x),
$$

so all $g_i$ belong to the active-set indicator span. They need not be orthogonal, whereas every exact maximum-IPR frame from B.23 consists of disjointly supported indicators, up to signed permutation.

Furthermore, for fixed nonzero $w\in\mathbb R^2$ and observations $h_x=t_xw$ with $t_x>0$, there is a continuum of distinct unit-atom, nonnegative, two-sparse decompositions, each with sole observed support $\lbrace1,2\rbrace$ and a fixed coefficient ratio across sites. All give identical observed $H$, sphere kernels $K_\beta$, normalized operators $P$, and loadings of every fixed site-space function.

*Proof.* The displayed feature-presence identity is pointwise, and the maximum-IPR assertion is B.23. For the second claim choose any continuously varying orthonormal frame $(d_1(\alpha),d_2(\alpha))$ for which both $a_i(\alpha)=\langle w,d_i(\alpha)\rangle$ are positive; such frames form a nonempty open angular interval. Then

$$
h_x=t_xa_1(\alpha)d_1(\alpha)+t_xa_2(\alpha)d_2(\alpha)
$$

for every site. Restricting to a small interval gives distinct unordered dictionaries, with positive coefficients and constant within-dataset ratio $a_1(\alpha)/a_2(\alpha)$. The observed matrix $H$ is unchanged, hence so are its normalized Gram matrix, $K_\beta$, $P$, and $H^\top\psi$ for each fixed $\psi$. ∎

**Interpretive scope.** Exclusive active-set localization is not itself a choice of overlapping feature codes or evidence for a unique sparse-additive mechanism. A further factorization or identifiable singleton observations are required. This is not a limitation on the SVD's ability to represent overlapping functions: the first identity shows that it can. If amplitudes vary within a support, the indicator span contains only support-constant functions and cannot reproduce arbitrary varying coefficient functions.

**What it buys.** The dictionary interpretation remains compatible with THEORY §4.2's “not a partition” principle, while the explicit continuum of indistinguishable dictionaries prevents a stronger latent-recovery claim from the same observations.

**Tractability and sampling.** More observations on the same ray, with the same fixed coefficient profile, do not identify the latent atoms in the counterexample. Singleton coverage or another explicit dictionary-identifiability assumption is necessary before adding feature-direction claims. B.34 controls only the recovered support subspace, not a new latent factorization; the latter's optimization and sample complexity are separate.

### Diagnostics: exact certificates and their limits

**Lemma B.31 (indicator-subspace projector identity).** For a specified candidate partition and a rank-$m$ spectral projector $\Pi_B$, the largest principal-angle sine satisfies

$$
\|\sin\Theta(B,U)\|=\|\Pi_B-UU^\top\|.
$$

It vanishes if and only if the block **is** the candidate indicator subspace. For the centered contrast block use rank $m-1$ and the candidate projector $UU^\top-\mathbf1\mathbf1^\top/N$; the same assertion holds. This is not an if-and-only-if certificate of exact activation rays, high eigenvalues, semantic features, or causal mechanisms.

*Proof.* This is the standard projector/principal-angle identity; see Stewart and Sun, *Matrix Perturbation Theory* (1990), Chapter I, Section 5 (Theorem 5.5 there, on the relation between projector differences and principal angles). The singular values of the overlap of two orthonormal frames are their principal-angle cosines. The cosine-sine decomposition gives the displayed projector identity; zero difference is exactly equality of their ranges. The constant vector belongs to the indicator span, so removing its projector gives the centered formulation.

For a concrete limitation, take two sites in each of $m\ge2$ groups,

$$
\hat\phi_{j,\pm}=\sqrt{1-r^2}\,e_j\pm r f_j,
\qquad0<r^2<1/2,
$$

where all $e_j,f_j$ are orthonormal. The within-group off-diagonal dot product is $1-2r^2$ and all cross-group dot products are zero. Every kernel row has sum

$$
Z=e^\beta+e^{\beta(1-2r^2)}+2(m-1),
$$

so $P=K/Z$. Its group-contrast eigenvalue is $(e^\beta+e^{\beta(1-2r^2)}-2)/Z$, and its within-pair contrast eigenvalue is $(e^\beta-e^{\beta(1-2r^2)})/Z$. The former is strictly larger for every $\beta>0$. Hence the top-$m$ block is exactly the group-indicator span although all within-group vectors are distinct. As $\beta\downarrow0$, its nonconstant eigenvalues tend to zero, not to one. ∎

**What it buys.** Zero indicator-subspace error is a genuine exact recovery certificate for the object the diagnostic names. Its scope is algebraic; it must be accompanied by the spectrum and by independent semantic or mechanism validation.

**Tractability and sampling.** Overlap-matrix SVD costs $O(Nm^2+m^3)$, or less with explicit sparse indicators. Candidate indicators are validation information, never kernel inputs. B.34 quantifies nonzero empirical errors; finite precision cannot prove exact zero. For real data without candidate sets, this supervised diagnostic is unavailable.

**Proposition B.32 (sharp projector-IPR bounds and cluster values).** For any nonzero rank-$r$ orthogonal projector $\Pi_B$, let $w_x=\Pi_B[x,x]/r$. Then

$$
\frac1N\le\operatorname{IPR}(B):=\sum_xw_x^2\le\frac1r.
$$

The lower equality holds exactly when all diagonal entries equal $r/N$. The upper equality holds exactly when $\Pi_B$ is the coordinate projector onto $r$ sites. For the complete $m$-cluster indicator span,

$$
\operatorname{IPR}(U)=\frac1{m^2}\sum_{j=1}^m\frac1{n_j}.
$$

For its centered contrast block, $m>1$,

$$
\operatorname{IPR}(CU)=\frac{\sum_j1/n_j-(2m-1)/N}{(m-1)^2}.
$$

*Proof.* An orthogonal projector has $0\le p_x:=\Pi_B[x,x]\le1$ and $\sum_xp_x=r$. Cauchy–Schwarz gives $\sum_xp_x^2\ge r^2/N$, with equality precisely for constant diagonal. The inequality $p_x^2\le p_x$ gives the upper bound, with equality precisely when every $p_x\in\lbrace0,1\rbrace$. Since $p_x=\|\Pi_Be_x\|^2$, $p_x=1$ fixes $e_x$ and $p_x=0$ annihilates it, proving the coordinate-projector characterization. For indicators $p_x=1/n_j$ on $C_j$, giving the first formula; centering subtracts $1/N$ from every diagonal entry and the rank becomes $m-1$, giving the second by expansion. ∎

**What it buys.** Projector IPR measures where a subspace lives, not whether it is a cluster dictionary. Balanced clusters attain the minimum $1/N$ even when their indicator recovery is perfect. The rank-specific upper bound corrects the potentially misleading “one site” endpoint for multidimensional blocks.

**Tractability and sampling.** Compute the diagonal as row squared norms of a frame in $O(Nr)$ time, without forming the projector. Sampling uncertainty in the block is controlled through B.34; specifically, for same-rank projectors at operator distance $\tau$, each diagonal changes by at most $\tau$, and $|\operatorname{IPR}(B)-\operatorname{IPR}(B')|\le2\tau/r$, by factoring the difference of squares and using both diagonal sums $r$. Rare-cluster imbalance can change IPR without changing the qualitative cluster interpretation.

### Two scales give a dictionary hierarchy

**Theorem B.33 (an exact two-level dictionary hierarchy).** Let there be $G\ge2$ coarse groups, each containing $b_f\ge2$ fine clusters, with $n\ge1$ identical normalized sites in each fine cluster. Suppose prototype inner products are one within a fine cluster, $1-\delta$ between different fine clusters in the same coarse group, and $1-\Delta$ between coarse groups, with $0<\delta<\Delta\le1$. These hypotheses are realizable, for example by prototypes

$$
\sqrt{1-\Delta}\,e_0+\sqrt{\Delta-\delta}\,e_g+\sqrt\delta\,e_{g,j}.
$$

Put $z_f=e^{-\beta\delta}$, $z_c=e^{-\beta\Delta}$ and $Z=1+(b_f-1)z_f+b_f(G-1)z_c$. Then $P$ has eigenvalues

$$
1;\qquad
\lambda_c=1-\frac{Gb_fz_c}{Z}\quad(G-1\text{ times});\qquad
\lambda_f=\frac{1-z_f}{Z}\quad(G(b_f-1)\text{ times});\qquad
0\quad(Gb_f(n-1)\text{ times}).
$$

The constant and coarse contrasts span the coarse-group indicator space; adding fine contrasts gives the fine-cluster indicator space. For every $\beta>0$, $1>\lambda_c>\lambda_f>0$.

In particular, if

$$
Gb_f e^{-\beta\Delta}<1/2,\qquad\beta\delta<1/2,
$$

precisely $G$ eigenvalues exceed $1/2$, and their space is the coarse dictionary. If instead $e^{-\beta\delta}<1/(Gb_f+1)$, precisely $Gb_f$ exceed $1/2$, recovering the fine dictionary. The first window is nonempty whenever $\Delta/\delta>2\log(2Gb_f)$.

At fixed $\beta$, diffusion does not rotate these spaces. For a declared threshold $0<\theta<1$, it retains exactly the coarse space at any positive integer $t$ satisfying

$$
\frac{\log\theta}{\log\lambda_f}<t<\frac{\log\theta}{\log\lambda_c}.
$$

Such an integer is an extra condition; the interval is always nonempty over real times.

*Proof.* Equal row sums make balancing division by the row sum $ne^\beta Z$. Split site space into constants, coarse contrasts constant within each coarse group, fine contrasts whose sums vanish within each coarse group, and within-fine-cluster zero-sum functions. Direct multiplication on these four mutually orthogonal spaces gives the displayed eigenvalues and multiplicities. The difference $\lambda_c-\lambda_f=b_f(z_f-z_c)/Z$ is positive. Since $Z\ge1$, $\lambda_c\ge1-Gb_fz_c$ and $\lambda_f\le1-z_f\le\beta\delta$, proving the coarse threshold statement. Also $z_c\le z_f$ gives

$$
\lambda_f\ge\frac{1-z_f}{1+(Gb_f-1)z_f}>\frac12
$$

under the fine-window hypothesis. The coarse window asks for $\log(2Gb_f)/\Delta<\beta<1/(2\delta)$, giving its sufficient nonemptiness condition. Finally powers preserve eigenvectors and raise their eigenvalues to $t$; taking logarithms proves the time interval. ∎

**What it buys.** A real nested angular geometry yields different, exactly identifiable centroid dictionaries at two temperatures, or after thresholded diffusion at one temperature. “Hierarchy” here is not an unexplained change in the range of $P^t$: the declared spectral threshold is what discards fine modes.

**Tractability and sampling.** The balanced ideal model reduces computation to $Gb_f$ prototypes; the general dense costs remain those of B.21. In unbalanced or noisy samples the exact formulas need not hold, but operator perturbation and B.34 protect a threshold when its margin to both adjacent eigenvalues exceeds the sampling error. Both coarse-group coverage and fine-cluster coverage are needed for their respective claims. Larger time shrinks absolute eigenvalues and can eventually erase the coarse contrasts too.

### The sampling interface used by these theorems

**Proposition B.34 (coverage and the operator-error bound of Theorem A.44).** These recovery theorems are deterministic statements about the observed finite site set. For an i.i.d. mixture with $m$ target clusters of probabilities $p_j\ge p_{\min}>0$, if

$$
N\ge\frac8{p_{\min}}\log\frac m\zeta,
$$

then, with probability at least $1-\zeta$, every cluster has at least $Np_j/2$ sites.

Separately, suppose a sampling/approximation theorem from Part A supplies a **same-space** self-adjoint error bound $\|\widehat P-P\|\le e_N(\zeta)$ with its own hypotheses. If $P$ has a top-block gap at least $g>0$ and $e_N<g/2$, then the empirical gap is at least $g-2e_N$, and its top-block projector moves by at most

$$
\frac{e_N}{g-e_N}.
$$

If that particular Part A theorem has the form $e_N\le C\sqrt{\log(2m/\zeta)/n_{\min}}$, then

$$
n_{\min}>\frac{16C^2}{g^2}\log\frac{2m}{\zeta}
$$

is sufficient for $e_N<g/4$. This conditional substitution is **not** an asserted universal sampling rate for Sinkhorn kernels; its constant and assumptions must come from the sampling theorem actually used (Theorem A.44 of Part A, whose bound carries the Sinkhorn conditioning factor and the common-space lift explicitly).

*Proof.* If $Y\sim\operatorname{Binomial}(N,p)$, Markov's inequality with $t=\log2$ gives

$$
\Pr(Y\le Np/2)\le e^{tNp/2}(1-p+pe^{-t})^N
\le\exp\big(Np((\log2)/2-1/2)\big)\le e^{-Np/8}.
$$

A union bound proves the occupancy statement. The variational eigenvalue characterization moves each eigenvalue by at most $e_N$, proving the empirical gap bound. The projector bound is the Davis–Kahan sin$\Theta$ estimate (Davis and Kahan, “The rotation of eigenvectors by a perturbation. III,” *SIAM Journal on Numerical Analysis* **7**(1):1–46, 1970). For the self-contained proof here, the same Sylvester integral as B.22, now separating the old lower block's spectrum from the new upper block's spectrum by at least $g-e_N$, proves the projector bound. Solving the stated conditional inequality for $n_{\min}$ proves its last assertion. ∎

**What it buys.** Rare-cluster coverage, geometric separation, and statistical estimation error are three distinct obstacles. This interface says exactly how a valid sampling theorem turns a geometric gap into an empirical recovery certificate, without hiding its dependence on temperature or sampling measure.

**Tractability and sampling.** For exact empirical geometry no additional stochastic theorem is needed: every observed pair can be checked. Population comparisons require the common-space embedding/lifting specified by Part A; independently sized kernel matrices cannot simply be subtracted. Dependent token sites require that theorem's effective-sample or document-level assumptions, not the binomial formula above. Under the displayed conditional rate, the expected number per rare cluster must clear both coverage and $C^2/g^2$ accuracy requirements; $C$ may deteriorate strongly with temperature. The integrator should insert the actual Part A sampling-theorem cross-reference, and allocate failure probabilities across occupancy and operator-error events.

### Numerical verification

Executed with `uv run --with numpy python -I checks/theorems_b2_check.py`, seed 20261107. The toy has $N=90$, $m=3$, cluster sizes $(20,30,40)$, ambient dimension six, and orthogonal prototypes. The perturbed case adds independent Gaussian coordinate noise of standard deviation $0.035$ before row normalization. Its realized pairwise constants are $\varepsilon=0.03202905$, $\Delta=0.83444403$. Balancing is iterated to maximum row error below $3\times10^{-14}$.

| Case | $\beta$ | Observed $\lambda_m-\lambda_{m+1}$ | Proven gap lower bound | Observed $\lVert\sin\Theta\rVert$ | Proven upper bound |
|---|---:|---:|---:|---:|---:|
| Exact rays | 6 | 0.99158338 | 0.98691735 | $7.75\times10^{-16}$ | 0 |
| Exact rays | 9 | 0.99957878 | 0.99934865 | $6.82\times10^{-16}$ | 0 |
| Exact rays | 12 | 0.99997902 | 0.99996757 | $6.11\times10^{-16}$ | 0 |
| Perturbed | 6 | 0.97767416 | 0.56218041 | 0.00171247 | 0.07812354 |
| Perturbed | 9 | 0.97977506 | 0.55131277 | 0.00014296 | 0.00802087 |
| Perturbed | 12 | 0.97364362 | 0.46266534 | 0.00001080 | 0.00090884 |

The tiny nonzero exact-case angles are floating-point error. All B.21/B.22 spectral and angle inequalities were asserted, not merely printed. The script also asserts the exact maximum-IPR inequality on 100 random orthogonal rotations, analytically optimized $O(2)$ noisy rotations, B.24's computed certificate, the same-view loading identity and Sol's wrong-view counterexample (dictionary difference $\sqrt5$), the two-clump hard-clustering obstruction, B.27's two-cluster free-code counterexample (angle $4.6344561\times10^{-6}$ radians), the nonnegative antipodal-ray extension, the fixed-profile nonidentifiability example, projector-IPR formulas, and B.33's complete eigenvalue list. It additionally checks the diagnostic counterexample and the B.34 perturbation/Chernoff inequalities. The tested $O(2)$ frame errors divided by $\tau$ were $1.414214$, $1.414231$, and $1.414214$; these are empirical ratios, not an asserted general upper bound. A nested toy with $(G,b_f,n)=(2,3,4)$ and $(\delta,\Delta)=(0.01,0.8)$ has $(\lambda_c,\lambda_f)=(0.96284114,0.01649100)$ at $\beta=5$ and $(1.00000000,0.86416450)$ to displayed precision at $\beta=300$: the number above $1/2$ changes from two to six. Output ended `all B2 checks passed`. These finite tests check the formulas, not their universal validity; the proofs supply that. The existing `checks/cm_check.py` was also run and ended `all checks passed`.

### Discrepancies with THEORY.md

No changes to `THEORY.md` are made here.

1. **Section 4.3, projector IPR.** The broad bound $1/N\le\mathrm{IPR}\le1$ is valid but not sharp at fixed block rank. A rank-$r>1$ block cannot attain the stated “one site” endpoint: its sharp upper bound is $1/r$, attained by a coordinate subspace on $r$ sites. Balanced indicator dictionaries can attain the *minimum* IPR. B.32 supplies both corrections.
2. **Sections 3.2–4.3, centering and localization.** A partition into $m$ clusters has $m$ uncentered indicator dimensions but only $m-1$ centered contrasts. A localizing rotation to the original indicators must reattach the constant. Rotating a centered contrast block alone cannot produce nonnegative, nonzero cluster indicators.
3. **Sections 4.3 and 5, a localizing rotation.** Exact maximum-IPR recovery is proved here, but practical varimax convergence and a dimension-independent linear noisy recovery rate are not. B.23 distinguishes its global optimum from its iterative implementation; B.24 gives a costly certified alternative.
4. **Section 11, sparse features.** Active-set indicators, singleton indicators and overlapping feature-presence functions are different targets. Strong positive-sum/prototype-separation assumptions justify the first; singleton coverage exposes feature directions. They do not establish exact code recovery in the cited signed, gated compressed-computation model.
5. **Section 5, “SAE-like.”** The wording is appropriately weaker than an equality theorem. One-hot centroids and free-amplitude sparse coding are different objectives; the requested unrestricted equivalence would be false. B.26 and B.28 state distinct true equivalences. B.26 also requires the loading view to equal the baseline reconstruction matrix; normalized-data centroids cannot be equated with raw-data centroids when amplitudes differ.
6. **Section 2, the high-temperature endpoint.** The identity limit is already qualified there by distinct normalized vectors. Exact repeated rays are the relevant exception: their limit is the cluster-indicator projector, consistent with B.21. For clouds of distinct sites, the finite-window warning remains essential.

### Notes on imported results

The only results imported without re-proving their general existence theory are the construction's already-established positive Sinkhorn scaling and PSD sphere kernel. Spectral estimates, localization, SAE statements, diagnostics and hierarchy are proved above; the eigenvalue variational principle and the cosine-sine/SVD identities are standard finite-dimensional linear algebra (see, e.g., Horn and Johnson, *Matrix Analysis*, 2nd ed., 2013, and Stewart and Sun, *Matrix Perturbation Theory*, 1990). The Davis–Kahan citation accompanies a self-contained estimate. No unverified external sampling theorem is used: B.34 is an explicit interface pending the sampling cluster's actual assumptions and constants.

---
## Part B, cluster 3 — Ideal case: fitting-free comparison and linear stitching

This cluster distinguishes **comparison**, which is a function of two site-space operators, from **stitching prediction**, which additionally uses a labelled linear fit, activation loadings, and derivatives of the downstream model. The latter is not fitting-free. All finite-sample identities use the counting-measure space $\mathcal H=\mathbb R^N$; an empirical expectation has an explicit factor $1/N$. No new metric is imposed on the geometric construction: output Fisher geometry enters only the declared stitching diagnostic of THEORY Sections 8–9.

Cross-references `A.9–A.10` and `A.21 and A.23` are editorial anchors for the exactness and stability clusters. The needed finite-dimensional arguments are included below, so these anchors are not gaps in the proofs.

### What comparison identifies, and its invariances

**Definition B.40 (shared-site comparison).** Fix an ordered, common site set $X$, $N\ge2$, nonzero activation rows, and a common $\beta>0$. For model $\mathsf m\in\lbrace A,B\rbrace$, let $\hat\Phi^{\mathsf m}$ contain its normalized rows, $K^{\mathsf m}_{xy}=\exp(\beta\hat\phi^{\mathsf m}(x)^\top\hat\phi^{\mathsf m}(y))$, and $P^{\mathsf m}=D_{v^{\mathsf m}}K^{\mathsf m}D_{v^{\mathsf m}}$ be its symmetric bistochastic scaling. Write $C=I-\mathbf1\mathbf1^\top/N$ and $\tilde P^{\mathsf m}=CP^{\mathsf m}C$.

Fix in advance a deterministic gap-selection rule (including any gap threshold). Comparison blocks are the isolated spectral subspaces selected by that rule on $\tilde P|_{\mathbf1^\perp}$, so the removed constant mode is not confused with nonconstant zero modes. When a complete decomposition is needed below, add the constant block separately and include the full tail. A block has projector $\Pi_B$ and an arbitrary orthonormal frame $\Psi_B$. For two blocks of ranks $r,s$, their principal-angle cosines are the singular values of $\Psi_{B_A}^\top\Psi_{B_B}$. The centered alignment is

$$\mathcal A_\beta(A,B)=\frac{\operatorname{tr}(\tilde P^A\tilde P^B)}{\|\tilde P^A\|_F\|\tilde P^B\|_F},$$

when both denominators are nonzero. Sites must be paired, not activation coordinates; widths can differ.

**Theorem B.41 (exact angular identifiability, invariance, and no fitting).** Under **shared sites, nonzero rows, and fixed positive temperature**:

1. $P$ is unchanged by $\phi(x)\mapsto a_xO\phi(x)$, with $a_x>0$ and a common orthogonal $O$. Consequently its spectra, isolated blocks, and the above comparisons are unchanged by independent such transformations of the two views.
2. Conversely, $P$ determines the entire normalized Gram matrix:
   $$G_{xy}=\hat\phi(x)^\top\hat\phi(y)=1+\frac1\beta\log\frac{P_{xy}}{\sqrt{P_{xx}P_{yy}}}.$$
   Thus, in equal ambient dimension, two activation clouds give the same $P$ if and only if they differ by a common orthogonal transformation and positive per-site rescaling. In unequal dimensions the statement is an isometry between their normalized row spans, or an orthogonal transformation after zero-padding both spaces.
3. A map $T:\mathbb R^d\setminus\lbrace0\rbrace\to\mathbb R^d\setminus\lbrace0\rbrace$ preserves every finite-data operator at this $\beta$ if and only if $T(z)=a(z)Oz$ for a fixed $O\in O(d)$ and an arbitrary positive function $a$. If $T$ is linear and invertible, this reduces to $T=cO$, $c>0$. General maps of the displayed form need not be invertible: the **group** is obtained by requiring bijectivity on each positive ray (after the rotation). For a fixed indexed cloud the invariance action is $O(d)$ together with $N$ independent positive row scalings.
4. Spectra, spectral-projector principal angles, and centered alignment are functions of $(P^A,P^B)$ alone, with the declared gap rule fixed. In this precise sense their computation requires no cross-model fit or feature-coordinate correspondence.

*Proof.* Orthogonal transformations and positive row rescaling preserve every normalized dot product, hence $K$ and its unique bistochastic scaling. Since $K_{xx}=e^\beta$, $P_{xx}=v_x^2e^\beta$. Dividing $P_{xy}$ by $\sqrt{P_{xx}P_{yy}}$ cancels both scaling factors and proves the reconstruction formula.

Equal normalized Grams yield an isometry of their spans: send $\sum_x c_x\hat\phi^A(x)$ to $\sum_x c_x\hat\phi^B(x)$. Equality of Grams makes its norm and inner product well-defined and preserved. Extend orthonormal bases to obtain the stated orthogonal map. Restore row norms with positive scalars.

For the universal-map assertion apply Gram recovery to every pair. The vectors $q_i=T(e_i)/\|T(e_i)\|$ are orthonormal. For any nonzero $z$, inner-product preservation gives $q_i^\top T(z)/\|T(z)\|=z_i/\|z\|$. Therefore $T(z)/\|T(z)\|=Oz/\|z\|$, $O=[q_1\cdots q_d]$. The converse is immediate. If $T$ is linear, compare $e_i,e_j,e_i+e_j$: the positive coefficients of $q_i,q_j$ must agree, so all radial factors equal one constant (the one-dimensional case is immediate). Finally, spectral projectors are uniquely determined by a matrix and an isolated spectral set; changing their frames rotates the overlap matrix on the left and right without changing its singular values. Alignment is already an explicit function of the two matrices. ∎

**What it buys.** Models of unequal width can be compared without learning a map between their neurons. At a known positive temperature, normalization does not secretly discard any additional sampled angular geometry: the losses are precisely radial information and the choice of feature-space frame.

**Tractability and sampling.** Once rank-$r$ block frames are available, their overlap costs $O(Nr^2)$ and its small SVD costs $O(r^3)$; unequal ranks cost $O(Nrs+rs\min(r,s))$. Dense alignment costs $O(N^2)$; building the kernels costs $O(N^2d)$ per view. These are exact finite-cloud statements, requiring no statistical sample-size assumption. Population comparisons still require sampling control, gaps, and coverage; very small entries make the inverse formula numerically ill-conditioned. One cannot replace paired sites by independently ordered samples.

**Normalization reference.** Sinkhorn, “A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices,” *Annals of Mathematical Statistics* 35 (1964), 876–879, supplies existence and uniqueness of the bistochastic scaling for strictly positive square matrices (diagonal factors unique up to reciprocal scalar). Here $K$ is square, strictly positive, and symmetric; symmetry and uniqueness yield the symmetric scaling, whose positive vector is unique. These are exactly the hypotheses used in B.41 and in B.42's within-class permutation argument.

**Comparison with CKA and Procrustes.** Linear CKA **after row normalization** shares these invariances. Ordinary linear CKA on raw activations is invariant to orthogonal transformations and one global nonzero scale per view, not arbitrary positive per-row rescaling. Neither method is generally invariant to arbitrary invertible feature maps. For example, $e_1$ and $(e_1+e_2)/\sqrt2$ have cosine $1/\sqrt2$, changed to $2/\sqrt5$ by $\operatorname{diag}(2,1)$; B.41 then forces a changed $P$. Orthogonal Procrustes instead selects a cross-model feature map by a closed-form SVD optimization: it is a fit, though not iterative training. Some optimized Procrustes **values** can also be expressed through Gram matrices; “no fitting” here means no alignment-map optimization is needed to define or evaluate our statistics, not that Gram representations make all other statistics impossible.

### The endpoints and the intermediate-temperature meaning

**Theorem B.42 (CKA limit and the duplicate-aware high-temperature limit).** Under **fixed finite nonzero data**, set $G_{\mathsf m}=\hat\Phi^{\mathsf m}(\hat\Phi^{\mathsf m})^\top$ and $Z_{\mathsf m}=C\hat\Phi^{\mathsf m}$.

- If $Z_A,Z_B$ are nonzero, then
  $$\lim_{\beta\downarrow0}\mathcal A_\beta(A,B)=\frac{\|Z_A^\top Z_B\|_F^2}{\|Z_A^\top Z_A\|_F\|Z_B^\top Z_B\|_F},$$
  the linear CKA of row-normalized activations.
- Let $\mathcal E_{\mathsf m}$ partition sites with identical normalized rows in view $\mathsf m$, and let $R_{\mathsf m}$ average within each equivalence class: $R_{\mathsf m}[x,y]=1/|E|$ if $x,y\in E$, zero otherwise. Then $P_\beta^{\mathsf m}\to R_{\mathsf m}$. If $CR_AC$ and $CR_BC$ are nonzero, alignment tends to their normalized Frobenius inner product. In particular, if each view has pairwise distinct normalized rows, $R_A=R_B=I$ and $\mathcal A_\beta\to1$.
- For every finite positive temperature at which it is defined, $\mathcal A_\beta$ is exactly normalized kernel-target alignment of the two **centered bistochastic kernels**, with one regarded as the target of the other. It is not, in general, alignment of the unnormalized exponential kernels. It lies in $[0,1]$, and equals one precisely when $\tilde P^A=c\tilde P^B$ for some $c>0$.

*Proof.* The expansion in `A.9–A.10` is

$$CP_\beta^{\mathsf m}C=\frac\beta N CG_{\mathsf m}C+O(\beta^2).$$

For completeness, put $v(\beta)=N^{-1/2}(\mathbf1+\beta a+O(\beta^2))$. Existence of this expansion follows from the implicit function theorem for the row-sum equations: in log-scalings their derivative at $\beta=0$ is $I+\mathbf1\mathbf1^\top/N$, which is invertible. Expanding $K=\mathbf1\mathbf1^\top+\beta G+O(\beta^2)$ gives

$$P_\beta=\frac1N\big[\mathbf1\mathbf1^\top+\beta(G+a\mathbf1^\top+\mathbf1a^\top)\big]+O(\beta^2).$$

Centering kills the two terms containing $a$. Nonzero $Z_{\mathsf m}$ implies $CG_{\mathsf m}C=Z_{\mathsf m}Z_{\mathsf m}^\top\ne0$, so dividing numerator and denominator of alignment by $(\beta/N)^2$ and taking limits is legitimate. Trace cyclicity gives $\langle Z_AZ_A^\top,Z_BZ_B^\top\rangle_F=\|Z_A^\top Z_B\|_F^2$ and $\|ZZ^\top\|_F=\|Z^\top Z\|_F$.

For the other endpoint, replace $K$ by $\bar K=e^{-\beta}K$, which does not change its bistochastic normalization. Now $\bar K_{xx}=1$, all entries lie in $(0,1]$, and entries between unequal directions vanish. Its scaling $w$ satisfies $1=w_x\sum_y\bar K_{xy}w_y$, hence $w_x\le1$ and $w_x\ge1/N$. Permuting sites within a duplicate class leaves $\bar K$ unchanged, so uniqueness of scaling makes $w$ constant within each class. Along any convergent subsequence the row equations become $1=|E|w_E^2$, whence $w_E=|E|^{-1/2}$. Every subsequential limit is the same, proving $P\to R$. The alignment conclusion follows by continuity away from zero denominators.

Finally $\tilde P^{\mathsf m}$ is PSD, so $\operatorname{tr}(\tilde P^A\tilde P^B)\ge0$; Frobenius Cauchy–Schwarz gives the upper bound and its equality condition. The statistic is the definition of normalized kernel alignment applied to these centered Gram matrices; no probabilistic or semantic assertion is being added. ∎

**What it buys.** The low-temperature endpoint is an exact bridge to CKA, not an analogy. The high-temperature endpoint explains why a large alignment can be uninformative, while the finite-temperature scan compares which site contrasts receive similar diffusion weights.

**Tractability and sampling.** The CKA endpoint can be computed without $N\times N$ matrices using feature cross-products, costing $O(N(d_A^2+d_B^2+d_Ad_B))$. Finite-temperature dense comparison costs $O(N^2)$ once operators are built. The limits are fixed-$N$ statements, not uniform in growing $N$, vanishing inter-site separation, or vanishing centered Gram norm. Duplicate directions are especially relevant to isolated-feature probe banks; no extra sample count repairs an incorrect distinctness assumption.

**References for the comparison terminology.** Cristianini, Shawe-Taylor, Elisseeff and Kandola, “On Kernel-Target Alignment,” *Advances in Neural Information Processing Systems 14* (conference 2001, proceedings 2002), define normalized kernel alignment. The centered version is treated by Cortes, Mohri and Rostamizadeh, “Algorithms for Learning Kernels Based on Centered Alignment,” *Journal of Machine Learning Research* 13 (2012), 795–828; our two centered bistochastic PSD matrices meet that definition. Kornblith, Norouzi, Lee and Hinton, “Similarity of Neural Network Representations Revisited,” ICML 2019, PMLR 97:3519–3529, give the linear-CKA form used above. These are identifications of definitions; the limits and normalizations were proved here.

### What an ideal linear stitch can predict

**Definition B.43 (ideal least-squares stitch and unreadability).** Let $H_A=H^A_L\in\mathbb R^{N\times d_A}$ and $H=H^B_{L'}\in\mathbb R^{N\times d_B}$ be **raw** activation matrices. The ideal linear stitch is any minimizer of $\|H-H_A S\|_F^2$; choose $S_*=H_A^+H$ for definiteness. Its fitted activation matrix is unique: $H_A S_*=TH$, where $T=H_A H_A^+$ is the orthogonal projector onto $\operatorname{col}(H_A)$. Indeed, columnwise orthogonal projection gives $\|H-H_A S\|_F^2=\|(I-T)H\|_F^2+\|TH-H_A S\|_F^2$, minimized exactly at $H_A S=TH$. Put $R_\perp=I-T$ and $E=H-H_A S_*=R_\perp H$.

Choose a complete orthogonal decomposition $I=\sum_b\Pi_b$, for example the spectral blocks of $P^B$, **including the constant block and all discarded-tail blocks**. Let $\Psi_b$ be a frame for block $b$, $r_b=\operatorname{rank}\Pi_b$, and $L_b=\Psi_b^\top H\in\mathbb R^{r_b\times d_B}$ its loading matrix. Define

$$U_b=\|R_\perp\Psi_b\|_F^2=r_b(1-R_b^2),\qquad R_b^2=1-\frac{\|\Psi_b-T\Psi_b\|_F^2}{r_b}.$$

This is an **in-sample, ridge-free, aggregate zero-baseline $R^2$** for the block coordinates. For centered modes it agrees with the usual aggregate variance-normalized $R^2$; the constant coordinate needs this stated convention. An affine stitch is a separate variant obtained by adding a constant column to $H_A$. Held-out ridge $R^2$ is an estimator used in practice, not automatically the same quantity.

**Theorem B.44 (the local KL identity, with its constants).** Assume **local differentiability and site-local downstream evaluation**: for each site $x$, the downstream logits $z_x(h)$ are $C^2$ in a neighborhood of $H[x]$, and output probabilities are $p_x(h)=\operatorname{softmax}(z_x(h))$. Write

$$J_x=Dz_x(H[x]),\quad F_x=\operatorname{diag}(p_x)-p_xp_x^\top,\quad g_x=J_x^\top F_xJ_x,$$

and define the quadratic form on activation perturbations

$$q(V)=\sum_x V[x]^\top g_xV[x].$$

For the partial stitch $H_t=H+t(H_A S_*-H)=H-tE$,

$$\frac1N\sum_x\operatorname{KL}\big(p_x(H[x])\,\|\,p_x(H_t[x])\big)=\frac{t^2}{2N}q(E)+o(t^2).$$

More generally, as an arbitrary finite-array perturbation $V\to0$, empirical KL equals $q(V)/(2N)+o(\|V\|_F^2)$. The fixed-baseline Fisher-metric logit discrepancy, with no factor $1/2$ in its definition, satisfies

$$\sum_x\|z_x(H[x]+V[x])-z_x(H[x])\|_{F_x}^2=q(V)+o(\|V\|_F^2).$$

The KL conclusion extends to any regular output family whose local KL Hessian is its Fisher information. It does **not** state that the ordinary task loss has zero linear term.

*Proof.* Fix $p=\operatorname{softmax}(z)$. As a function of $u$, $D(u)=\operatorname{KL}(p\|\operatorname{softmax}(z+u))$ has $D(0)=0$, $\nabla D(0)=0$, and Hessian $\operatorname{diag}(p)-pp^\top$. Taylor's theorem gives $D(u)=u^\top Fu/2+o(\|u\|^2)$. Substituting $u=J_xV[x]+O(\|V[x]\|^2)$ gives $D=q_x(V[x])/2+o(\|V[x]\|^2)$. The finite sum preserves the remainder order, and $V=-tE$ gives the displayed identity. Squaring the first-order logit expansion proves the Fisher-discrepancy formula. For a regular output family the same argument starts with the assumed local Fisher Hessian. ∎

**What it buys.** The local output damage of a fitted stitch is exactly a sensitivity-weighted activation residual to leading order. It predicts distributional discrepancy near the original activation, not necessarily the loss of a full, large replacement.

**Tractability and sampling.** A single multi-output least-squares fit costs $O(Nd_A^2+d_A^3+Nd_Ad_B)$ by a direct factorization when $d_A\le N$, or can use an SVD. Once its residual is known, $q(E)$ needs **one JVP direction per site**, followed by an output-Fisher quadratic form. A ridge variant similarly needs one multi-output ridge fit, but changes the residual. This is an exact conditional finite-cloud statement; population KL needs independent validation sites and control of the Taylor remainder. For attention or recurrence on token sites, the displayed same-site Jacobians are insufficient when all token activations are replaced together: use the full cross-site Jacobian, or define a site as a whole independent input state. The proof then applies to the stacked state, with its full quadratic form. ReLU models require a neighborhood with fixed masks or another explicit differentiability condition.


**Proposition B.45 (two residual decompositions and spectral self-stitch obstruction).** Without additional assumptions,

$$E=\sum_b\Pi_bE=\sum_bR_\perp\Pi_bH=\sum_b R_\perp\Psi_b L_b.$$

The first decomposition is Frobenius-orthogonal; the second attributes residuals to the *original target blocks* and need not be orthogonal. In general $\Pi_bE\ne R_\perp\Pi_bH$. For either decomposition $E=\sum_bE_b$,

$$q(E)=\sum_bq(E_b)+2\sum_{b<c}\sum_xE_b[x]^\top g_xE_c[x].$$

Even with genuine spectral blocks of the target operator and constant isotropic Fisher, the uncorrected scalar unreadability predictor can be strictly positive for an **identity stitch**, and its proposed upper bound can also fail.

*Proof.* Insert $\sum_b\Pi_b=I$ on either side of $R_\perp$ and expand the quadratic form. For an exact spectral counterexample, take two sites, scalar target activations $H=(2,1)^\top$, source $H_A=H$, and $g_x=1$. Both normalized target rows equal $1$, so $P^B=\mathbf1\mathbf1^\top/2$ at every positive temperature. Its true spectral blocks are $\psi_+=(1,1)^\top/\sqrt2$ and $\psi_-=(1,-1)^\top/\sqrt2$. Direct projection gives

$$E=0,\quad L_+^2=9/2,\quad L_-^2=1/2,\quad U_+=1/10,\quad U_-=9/10.$$

Thus $\sum_b L_b^2U_b=9/10>0=q(E)$. Even omitting the constant block leaves a false positive $9/20$. The corresponding Section 8 squared-sensitivity weights satisfy $s_b^2/N=L_b^2$; hence the failure is exactly in the proposed spectral predictor, not an artifact of choosing a nonspectral basis.

For failure of the upper bound, retain the same true spectral blocks but take $H=(\sqrt{10}/3)(2,1)^\top$ and $H_A=(1,2)^\top$. Here $T=H_AH_A^\top/5$, $U_+=1/10$, $U_-=9/10$, $L_+^2=5$, and $L_-^2=5/9$. Therefore the putative upper bound is $5/10+(5/9)(9/10)=1$, whereas $E=(\sqrt{10}/3)(6/5,-3/5)^\top$ has $q(E)=2$. ∎

**What it buys.** A model stitched to itself can receive a nonzero sensitivity-weighted unreadability score because its kernel coordinates need not be linear in its own activations. Readout reduction, or an explicit correction for its failure, is indispensable to this scalar theorem; orthogonal target modes alone do not make their regression residuals orthogonal.

**Tractability and sampling.** The same obstruction is not confined to duplicate directions: the numerical check below uses 30 distinct four-dimensional target rows, $H_A=H$, $\beta=4$, and the actual rank-one spectral blocks, finding residual energy below $10^{-27}$ but a predictor of $39.36$. One multi-output fit computes block readability; block reconstruction costs $O(Nrd_B)$ for $r$ retained modes, with an explicit tail. Full cross-term accounting needs block JVPs rather than only the one residual JVP for total damage. This is an algebraic failure, not sampling noise.


**Theorem B.46 (weighted unreadability, with commutator and interference corrections).** In addition to B.43–B.44, first assume:

- **Readout reduction:** $T\Pi_b=\Pi_bT$ for every block.
- **Block Fisher decoupling:** for matrices $V,W$ supported in different site blocks ($\Pi_bV=V$, $\Pi_cW=W$, $b\ne c$), $\sum_xV[x]^\top g_xW[x]=0$.
- **Block sensitivity bounds:** on matrices supported in block $b$, $\gamma_b^-\|V\|_F^2\le q(V)\le\gamma_b^+\|V\|_F^2$, for $0\le\gamma_b^-\le\gamma_b^+$. Zero lower bounds are allowed.

Let $\ell_b^-,\ell_b^+$ be the extreme eigenvalues of $L_bL_b^\top$, including zero eigenvalues. Then

$$\sum_b\gamma_b^-\ell_b^-U_b\le q(E)\le\sum_b\gamma_b^+\ell_b^+U_b,\qquad U_b=r_b(1-R_b^2).$$

Under **block-isotropic sensitivity** ($\gamma_b^-=\gamma_b^+=\gamma_b$) and **isotropic target loading energy** ($L_bL_b^\top=\ell_bI_{r_b}$), both bounds equal $\sum_b\gamma_b\ell_bU_b$.

Here is a computable approximate version. Drop exact readout reduction, put

$$E_b=\Pi_bE,\quad c_b=\|[T,\Pi_b]H\|_F\le\|[T,\Pi_b]\|_2\|H\|_F,$$

and replace exact Fisher decoupling, if necessary, by a declared **relative interference bound**

$$\left|2\sum_{b<c}\langle E_b,\mathcal G E_c\rangle_F\right|\le\eta\sum_bq(E_b),\quad0\le\eta<1,\qquad (\mathcal GV)[x]=g_xV[x].$$

With $z_+=\max(z,0)$, the corrected bounds are

$$(1-\eta)\sum_b\gamma_b^-\big(\sqrt{\ell_b^-U_b}-c_b\big)_+^2\ \le q(E)\le\ (1+\eta)\sum_b\gamma_b^+\big(\sqrt{\ell_b^+U_b}+c_b\big)^2.$$

Exact decoupling allows $\eta=0$. Exact reduction allows $c_b=0$. Under exact reduction, sensitivity bounds and pairwise Fisher decoupling need only hold for block-supported matrices with rows in $\mathcal D_b=\operatorname{rowspan}(L_b)$; this can sharpen the bounds. Without reduction, a sufficient feature domain is instead $\operatorname{rowspan}(H)$, which contains every row of $E_b$.

*Proof.* Set $F_b=R_\perp\Psi_bL_b$ and $D_b=\Psi_b^\top R_\perp\Psi_b$. Since $R_\perp$ is an orthogonal projector,

$$D_b\succeq0,\quad\operatorname{tr}D_b=U_b,\quad\|F_b\|_F^2=\operatorname{tr}(L_bL_b^\top D_b).$$

PSD ordering gives $\ell_b^-U_b\le\|F_b\|_F^2\le\ell_b^+U_b$. Moreover,

$$E_b=\Pi_bR_\perp H=R_\perp\Pi_bH+[T,\Pi_b]H=F_b+[T,\Pi_b]H.$$

The triangle and reverse triangle inequalities give

$$\big|\|E_b\|_F-\|F_b\|_F\big|\le c_b,$$

and hence $(\sqrt{\ell_b^-U_b}-c_b)_+\le\|E_b\|_F\le\sqrt{\ell_b^+U_b}+c_b$. Apply the sensitivity bounds to $E_b$, then bound the sum of cross terms by $\eta\sum_bq(E_b)$. This proves the corrected result and its exact special cases. With $c_b=\eta=0$, both isotropy hypotheses turn all inequalities into equalities. Finally, when $c_b=0$, $E_b=R_\perp\Psi_bL_b$ has rows in $\mathcal D_b$; without that condition $E_b=\Pi_bR_\perp H$ has rows in $\operatorname{rowspan}(H)$. These are the only matrices to which the sensitivity bounds and the decoupling assumption were applied. ∎

**Restriction of the full-space idealization.** If exact Fisher decoupling is required on *all* feature directions for a complete decomposition containing the separate constant block, it forces $g_x=\bar g$ independent of $x$. Indeed, test a constant matrix $V[x]=a$ against arbitrary centered $W$, which decomposes into the nonconstant blocks. Decoupling says $\sum_xa^\top g_xW[x]=0$, so $g_xa$ is constant in $x$, for every $a$. Conversely, a site-constant form gives decoupling by orthogonality of site blocks. The sharp full-feature sensitivity interval is then $[\lambda_{\min}(\bar g),\lambda_{\max}(\bar g)]$ for every nonzero block: evaluate $V[x]=\psi(x)a$ with a unit block vector $\psi$. Full-feature exact block isotropy further forces $\bar g=\gamma I$, with all $\gamma_b$ equal. Distinct block weights in that regime arise from loadings and unreadability, not different Fisher scalars. The restricted-feature and interference variants avoid this forced collapse.

**What it buys.** The predictor has a two-sided certificate even when readout reduction holds only approximately, with explicit, observable penalties for block mixing. These intervals can be wide: the spectral circle check below gives Gaussian bounds about $[28.80,249.71]$ around $98.69$, so validity is not the same as a useful point prediction.

**Tractability and sampling.** One multi-output fit gives coordinate unreadabilities; loadings cost $O(Nrd_B)$. The correction $c_b$ can be computed from projector applications without forming a dense commutator, by $T\Psi_bL_b-\Psi_b(\Psi_b^\top TH)$. The total $q(E)$ is one residual JVP per site, but certifying all restricted $\gamma_b^\pm$ and interference generally needs multiple directions. All rank-$r$ loading sensitivities require $r$ JVP directions per site or an explicit randomized trace estimator; a batched pass does not eliminate these directions. The exact full-space assumptions are restrictive, as just proved, and a small commutator need not make the loading/sensitivity intervals narrow. No dimension-free sample count certifies these assumptions; B.49 controls evaluation error for frozen readability predictors. For generic sources whose fitted reconstruction does not commute with the target blocks, the corrected interval certifies that the bounds hold but rarely gives a useful prediction: in a stress test on 150 random non-commuting sources with a softmax head the lower bound was zero in about half the trials and the median upper bound exceeded the true value by a factor of about 23 (reviewer check, `reviews/theorems-B3-review-opus.md`). The predictor is informative in the near-commuting regime and is reported with its interval, not as a point estimate.


**Proposition B.47 (binary-readable blocks under site-homogeneous Fisher).** Suppose each block is **exactly readable or exactly unreadable**:

$$T\Pi_b=\Pi_b\quad\text{or}\quad T\Pi_b=0,\qquad u_b=1-R_b^2\in\lbrace0,1\rbrace.$$

Let $\mathcal D=\sum_b\operatorname{rowspan}(L_b)=\operatorname{rowspan}(H)$. Assume **site-homogeneous Fisher on the loading span**: a fixed PSD form $\bar g$ satisfies $a^\top g_xc=a^\top\bar g c$ for all $a,c\in\mathcal D$ and all sites $x$. No isotropy or orthogonality of different loading spans is assumed. Define exactly the squared sensitivity appearing under the square root in THEORY Section 8,

$$s_b^2=\sum_{i=1}^{r_b}\sum_x\|J_xa_H(\psi_{b,i})\|_{F_x}^2.$$

Then

$$q(E)=\sum_bu_b\operatorname{tr}(L_b\bar gL_b^\top)=\frac1N\sum_b s_b^2u_b,$$

and empirical stitched KL is $t^2\sum_b s_b^2(1-R_b^2)/(2N^2)+o(t^2)$. All expressions are invariant under rotation of a block frame. A directional-isotropy special case is recovered when $\bar g$ restricts to $\gamma_b I$ on each loading span, but that stronger assumption is unnecessary.

*Proof.* Binary readability gives $E_b=u_b\Psi_bL_b$. Site homogeneity and $\Psi_b^\top\Psi_c=0$ for $b\ne c$ imply

$$\sum_xE_b[x]^\top g_xE_c[x]=u_bu_c\operatorname{tr}(\bar gL_c^\top\Psi_c^\top\Psi_bL_b)=0.$$

For $b=c$ the same calculation gives $q(E_b)=u_b\operatorname{tr}(L_b\bar gL_b^\top)$, using $u_b^2=u_b$. Also $s_b^2=N\operatorname{tr}(L_b\bar gL_b^\top)$. Sum and apply B.44. Frame rotations conjugate $L_b\bar gL_b^\top$ without changing its trace. Completeness of the frames shows their loading row spans sum to $\operatorname{rowspan}(H)$. ∎

**What it buys.** An anisotropic but site-homogeneous downstream geometry still permits an exact scalar predictor for binary-readable blocks. The KL coefficient uses **squared sensitivity**, not its square root, with explicit counting-measure factors.

**Tractability and sampling.** Gaussian linear heads have site-constant Fisher pullbacks; softmax heads generally do not because their output Fisher varies with the predicted probabilities. Homogeneity need only hold on the loading span but must be checked or assumed, not inferred from a spectral gap. A Gaussian loading sketch $\xi$ with $\mathbb E\xi\xi^\top=I$ makes $\|J_xL_b^\top\xi\|_{F_x}^2$ an unbiased trace estimate, requiring one JVP per block per site; exact traces generally require $r_b$ directions. The circle check below has genuine spectral blocks and anisotropic $\bar g$, and verifies this equality. Partial readability needs a matrix-valued unreadability calculation or further hypotheses; this proposition does not promote it to an exact scalar formula.

**Proposition B.48 (zero gradient pressure does not imply successful stitching).** Under **exact fitted targets and a differentiable data-fit loss stationary at those predictions, with no direct parameter-loss terms**, all backpropagated errors, and hence all parameter-gradient block pressures, vanish; downstream Fisher sensitivity and stitch damage need not vanish.

*Proof.* On two sites let $H=(1,-1)^\top$, let the source activation be $H_A=(1,1)^\top$, and take the output family $\mathcal N(h,1)$ with training targets $(1,-1)^\top$. The squared-error training gradient is zero at $h=H$, but the best linear stitch gives $H_A S_*=0$; its empirical output KL is $1/2$, while the Fisher sensitivity is $g_x=1$. ∎

**What it buys.** Gradient pressure answers whether training currently requests a change; sensitivity answers whether removing an activation direction changes predictions. These are different even in the simplest exact-fit model.

**Tractability and sampling.** The counterexample uses two sites and no estimation. In real networks, gradient pressure needs a backward pass with the model's training objective; sensitivity instead needs downstream JVPs. Exact interpolation of squared-error or matched distributional targets supplies the stationary-loss assumption; for at least two classes, per-example cross-entropy with hard labels and finite logits cannot have zero logit gradient (or zero loss): its gradient is $p-y\ne0$. Parameter gradients can nevertheless vanish through network degeneracy or aggregation; that is a different claim.

### What held-out readability estimates require


**Theorem B.49 (conditional i.i.d. evaluation: Hoeffding and Bernstein bounds).** Assume **frozen, common-law evaluation**: conditional on the fitting data, the coordinate maps $f_b(x)\in\mathbb R^{r_b}$ and fitted predictors $\hat f_b(x)$ are fixed, and evaluation sites $x_1,\ldots,x_m$ are i.i.d. from a specified law $\nu$, independently of all fitting data. Every expectation below is under that same $\nu$, conditional on the frozen maps. Assume **known standardized target energy** $\mathbb E_\nu\|f_b(x)\|^2/r_b=1$ and **bounded normalized squared errors**

$$0\le Z_b(x)=\|f_b(x)-\hat f_b(x)\|^2/r_b\le B_{\rm err}^2.$$

For $k$ blocks, put $\mu_b=\mathbb E_\nu Z_b$, $R_b^2=1-\mu_b$, and $\widehat R_b^2=1-m^{-1}\sum_jZ_b(x_j)$. The simultaneous error is at most $\epsilon$ with probability at least $1-\alpha$ if

$$m\ge\frac{B_{\rm err}^4}{2\epsilon^2}\log\frac{2k}{\alpha}.$$

Alternatively, if $v_*\ge\max_b\operatorname{Var}_\nu Z_b$ is known, it suffices that

$$m\ge\frac{2(v_*+B_{\rm err}^2\epsilon/3)}{\epsilon^2}\log\frac{2k}{\alpha}.$$

One valid choice is $v_*=B_{\rm err}^2\mu_*$ for any known $\mu_*\ge\max_b\mu_b$, since $\operatorname{Var}Z_b\le\mathbb E Z_b^2\le B_{\rm err}^2\mu_b$. This improves the dependence on the range when $\mu_*$ stays bounded, not for arbitrary risks as large as the range.

*Proof.* Hoeffding's inequality conditional on the fitting data gives $\Pr(|m^{-1}\sum_jZ_b-\mu_b|>\epsilon)\le2\exp(-2m\epsilon^2/B_{\rm err}^4)$; take a union bound.

For completeness, the Bernstein step follows directly from a bounded-variable moment-generating-function bound. For $Y=Z_b-\mu_b$, $|Y|\le b_0:=B_{\rm err}^2$, $\mathbb EY=0$, and $\mathbb EY^2\le v_*$. For $0<t<3/b_0$, use $|\mathbb EY^j|\le b_0^{j-2}v_*$ for $j\ge2$ and $j!\ge2\cdot3^{j-2}$ to obtain

$$\mathbb E e^{tY}\le1+\frac{v_*t^2}{2(1-tb_0/3)}\le\exp\!\left(\frac{v_*t^2}{2(1-tb_0/3)}\right).$$

Independence multiplies the bounds for $m$ sites. Chernoff's inequality with $t=\epsilon/(v_*+b_0\epsilon/3)$ gives the upper-tail bound $\exp[-m\epsilon^2/(2(v_*+b_0\epsilon/3))]$; apply the same argument to $-Y$ and take a union bound over blocks. If $v_*=0$, all the variables are deterministic and the conclusion is immediate. Both conditional guarantees integrate to unconditional guarantees when their assumptions hold almost surely. ∎

**What it buys.** Evaluation budgets explicitly depend on localization and error variance, rather than only mode count. The result evaluates the actual frozen predictor under a declared population law; it does not bound its excess risk over the population-optimal stitch.

**Tractability and sampling.** For $k=1$, $B_{\rm err}=2$, $\epsilon=0.1$, $\alpha=0.05$, Hoeffding gives $m\ge2952$ i.i.d. sites. Localization can change this dramatically: a standardized indicator contrast for a cluster of mass $s$ has squared amplitude of order $1/s$. With zero prediction, $\mu_b=1$ and $B_{\rm err}^2\le1/s$ for $s\le1/2$. For $s=0.01$, the conservative choice $B_{\rm err}^2=100$ requires about $1.85$ million sites by Hoeffding, versus about $7.63\times10^4$ by the Bernstein bound with $v_*=100$. Thus localization can make the simple range bound impractical; the variance-aware bound helps but is not free of the rare-cluster cost.

Sample-orthonormal modes must be extended out of sample and put on a probability convention (typically $\sqrt{N_{\rm train}}\psi$ before further standardization); sample orthonormality does not establish population unit variance. If the target variance is estimated, numerator and denominator both need concentration and a positive variance floor. If both squares are bounded by $B_{\rm err}^2$ and the target second moment is at least $v_0>0$, moment errors at most $a_0\le v_0/2$ bound the ratio error by $2a_0/v_0+2B_{\rm err}^2a_0/v_0^2$, by writing the difference of two ratios and using the denominator floor. Training-size requirements depend on feature covariance, regularization, and effective rank; $\operatorname{rank}(H_A)=N$ makes all in-sample coordinates readable. Ridge replaces $T$ by a nonprojector and generally breaks B.46's projector proof. Independent but differently distributed evaluation sites estimate their average risk, not $\mathbb E_\nu Z_b$; dependent tokens require a justified dependence bound or i.i.d. document-level units with the evaluation law declared.

**References.** Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” *JASA* 58 (1963), 13–30, Theorem 2 (two-sided by applying both tails); Boucheron, Lugosi and Massart, *Concentration Inequalities* (2013), §2.8, for Bernstein bounds. Independence, common law, boundedness, and the variance proxy used here have all been stated explicitly; the Bernstein specialization was also proved above.

### Shared-site comparison through training


**Theorem B.50 (operator-calibrated checkpoint drift).** At **fixed positive $\beta$ and fixed shared sites**, let

$$d_P=\|P_{\tau'}-P_\tau\|_2,$$

the directly computable operator change. Let a rank-$r$ block of $P_\tau$ occupy a consecutive group of ordered eigenvalues, separated from the remaining spectrum by a gap $\delta>0$. If $d_P<\delta/2$, its continuation at the same eigenvalue indices is isolated and

$$\|\sin\Theta(B_\tau,B_{\tau'})\|_F\le\frac{\sqrt r\,d_P}{\delta-d_P}\le\frac{2\sqrt r\,d_P}{\delta}.$$

Since $\theta\le(\pi/2)\sin\theta$ on $[0,\pi/2]$, the angle vector itself satisfies the same bound times $\pi/2$, without an extra small-angle proviso.

*Proof.* Weyl's variational eigenvalue inequality places each corresponding new eigenvalue within $d_P$ of its old one. Boundary gaps stay at least $\delta-2d_P>0$, and each new block eigenvalue is separated from every old outside eigenvalue by at least $\delta-d_P$. For a new block eigenvector $u_j'$ and an old outside eigenvector $w_i$,

$$(\lambda_j'-\lambda_i)\langle w_i,u_j'\rangle=\langle w_i,(P_{\tau'}-P_\tau)u_j'\rangle.$$

Square and sum over $i,j$ to obtain $\|(I-\Pi_\tau)\Psi_{\tau'}\|_F\le\sqrt r\,d_P/(\delta-d_P)$. This norm is $\|\sin\Theta\|_F$. The scalar angle inequality follows from concavity of sine above its chord on $[0,\pi/2]$, and summing squares proves the angle-vector bound. ∎

**What it buys.** Fitting-free checkpoint comparison has a direct operator-and-gap error scale; it need not rely on a loose activation-space estimate. A drifting subspace and an ill-resolved spectral boundary are explicitly distinguished.

**Tractability and sampling.** After block extraction, overlaps cost $O(Nr^2)$; the operator difference admits Lanczos/power-type norm estimation using difference matvecs, with numerical tolerance reported (a rigorous certificate needs a certified upper bound). No paired feature coordinates are needed. Sampling different sites at different checkpoints destroys the shared-space identity unless a separate extension or correspondence is supplied. Fixed-data perturbation bounds do not remove population sampling uncertainty. Background: Davis and Kahan, *SIAM Journal on Numerical Analysis* 7 (1970), 1–46, and Weyl's min–max inequality, standard; see Bhatia, *Matrix Analysis* (1997). The particular subspace bound needed here was proved directly.

**Auxiliary activation bound (conservative, not the primary calibration).** Let $\epsilon_h=\max_x\|\hat h_\tau(x)-\hat h_{\tau'}(x)\|$. Then

$$d_P\le\min\big(1,\,2\beta(1+2\sqrt N)\epsilon_h\big).$$

If both raw row norms are at least $m_0>0$ and $d_h=\max_x\|h_{\tau'}(x)-h_\tau(x)\|$, one may use $\epsilon_h\le2d_h/m_0$. The bound from normalized rows is informative beyond $d_P\le1$ only when $\epsilon_h<[2\beta(1+2\sqrt N)]^{-1}$; at $N=10^4$, $\beta=10$, this is about $2.49\times10^{-4}$. Even then, certifying block continuation needs the smaller gap threshold. Sharper constants from `A.21 and A.23` can replace this auxiliary bound.

*Proof of the auxiliary bound.* The normalized Grams satisfy $\|G'-G\|_{\max}\le2\epsilon_h$. Interpolate $G_s=(1-s)G+sG'$ and $K_s=\exp(\beta G_s)$ entrywise. Each $G_s$ is PSD with unit diagonal, and $K_s$ is PSD by its nonnegative power series and the Schur product theorem (Horn and Johnson, *Matrix Analysis*, 2nd ed., 2013, §7.5). The positive symmetric scaling is differentiable: with $z=\partial_s\log v_s$, its row equations yield

$$(I+P_s)z=-\big(P_s\odot[\beta(G'-G)]\big)\mathbf1.$$

The derivative in log-scalings is invertible since $P_s\succeq0$, so the implicit function theorem justifies differentiation and $\|(I+P_s)^{-1}\|_2\le1$. Thus $\|z\|_\infty\le\sqrt N\beta\|G'-G\|_{\max}$. Differentiating $P_s$ gives $D_zP_s+P_sD_z+P_s\odot[\beta(G'-G)]$. Its norm is at most $(1+2\sqrt N)\beta\|G'-G\|_{\max}$, because $\|P_s\|_2=1$ and the last symmetric matrix has absolute row sums at most $\beta\|G'-G\|_{\max}$. Integrate over $s$. Also $0\preceq P_\tau,P_{\tau'}\preceq I$, so $-I\preceq P_{\tau'}-P_\tau\preceq I$, giving $d_P\le1$. The normalized-row bound follows by adding and subtracting a vector divided by the other row's norm and applying the reverse triangle inequality. ∎

**Definition B.51 (forming block at a declared scale).** At fixed $\beta$, a gap-delimited, rank-$r$ block path is **$(\theta_0,\kappa)$-forming** over a checkpoint interval if at every consecutive pair (i) the largest principal angle is at most $\theta_0<\pi/2$ and (ii) its mean eigenvalue $\bar\lambda_b=r^{-1}\operatorname{tr}(P\Pi_b)$ increases by at least $\kappa>0$. This defines a slowly moving site subspace receiving increasing diffusion weight, not a causal account of training or a claim that every individual eigenvalue increases. Requiring all ordered block eigenvalues to rise is a distinct, stronger definition.

**Lemma B.52 (well-posed and stable formation bookkeeping).** For isolated blocks, the forming definition is independent of frames and of all transformations in B.41. Along the continuation of B.50, $|\bar\lambda_b(\tau')-\bar\lambda_b(\tau)|\le d_P$. If the operator at each checkpoint is estimated to spectral error at most $\zeta$ and corresponding block ranks are retained, an observed mean-eigenvalue increase greater than $\kappa+2\zeta$ certifies a true increase greater than $\kappa$. If estimated block projectors have error at most $\omega$ in operator norm, an observed projector distance at most $\sin\theta_0-2\omega$ certifies the angle condition.

*Proof.* Traces and principal angles do not depend on frames, and B.41 preserves operators. Weyl's inequality bounds the change in each ordered eigenvalue by $d_P$, hence also their average. It similarly bounds each estimated average's error by $\zeta$, giving $2\zeta$ for a difference. For equal-rank projectors $\|\Pi-\Pi'\|_2=\sin\theta_{\max}$; the triangle inequality inserts the two estimation errors and proves the final claim. ∎

**What it buys.** “Forming” is a reproducible descriptive label with explicit margins, rather than an unstable comparison of individual eigenvectors. An increase smaller than estimation error is not certified formation.

**Tractability and sampling.** The same block overlap and eigenvalues suffice. A rank change or loss of the separating gap ends this block path; it is not silently counted as continuity. Bounds on $\zeta,\omega$ require the stability/sampling hypotheses of Part A or justified resampling; there is no sample-size guarantee from checkpoint count alone.

### What is not identified


**Proposition B.53 (equal sampled representations, unequal ideal-stitch responses).** At fixed known $\beta>0$, equality $P^A=P^B$ identifies the sampled normalized activation geometry up to B.41's invariances. Nevertheless, even identical raw activations at every corresponding layer need not determine Jacobians, off-site predictions, or the response to an ideal least-squares stitch.

*Proof.* On two input sites $t\in\lbrace-1,1\rbrace$, let both models share the ReLU hidden layer

$$h(t)=(\operatorname{ReLU}(t),\operatorname{ReLU}(-t),1),\qquad d(h)=h_1-h_2.$$

Use downstream logits $z_A(h)=(d(h),h_3)$ and $z_B(h)=(d(h)+d(h)^2-1,h_3)$. If intermediate downstream features are recorded, both networks can share the same extra layer $(d(h),d(h)^2,h_3)$, followed by different affine logit heads. Thus the two networks can have the same architecture; their input, hidden, intermediate, logit, and probability activations coincide on both sites, since $d(h)=\pm1$. Every corresponding geometric operator agrees for every temperature. Their hidden-to-first-logit derivatives, however, are $(1,-1,0)$ and $(1+2d,-1-2d,0)$, which differ at both sites.

Take the common source matrix $H_{\rm src}=\mathbf1\in\mathbb R^{2\times1}$ and stitch into $h$. The ideal fitted activation at both sites is their mean $(1/2,1/2,1)$. Model A then predicts first-logit probability $\sigma(-1)$ and model B predicts $\sigma(-2)$, where $\sigma(u)=(1+e^{-u})^{-1}$. The original first-logit probabilities are $1/2$ and $\sigma(-2)$ for both models. Their mean is $\bar p=(1/2+\sigma(-2))/2>\sigma(-1)$: the strict inequality follows from strict convexity of $\sigma$ on $(-\infty,0)$, or its derivative $\sigma''=\sigma(1-\sigma)(1-2\sigma)>0$ there. Average Bernoulli cross-entropy as a function of candidate log odds $u$ has derivative $\sigma(u)-\bar p<0$ for $u\le-1$. Hence its value at $-2$ strictly exceeds its value at $-1$. The reference entropies are identical, so empirical full-stitch KL is strictly larger in model B. Numerically the two KLs are $0.093622630706$ and $0.216890415242$. ∎

**What it buys.** Identical sampled representations can support different downstream reactions to the very same optimal linear reconstruction. The missing information is mechanistic, not additional hidden angular geometry in the sampled cloud.

**Tractability and sampling.** The obstruction uses two sites and a one-column source. Every site-linear fitted reconstruction $T_\lambda H$, including ridge, has rows in $\operatorname{rowspan}(H)$; its residuals and the loading vectors also lie there. Thus the stitch sees the downstream function on this span, and its first-order diagnostics see $J_x$ only on this span. A *linear* null term vanishing on the entire row span cannot distinguish such stitches; the nonlinear term above vanishes at the observed rows (and on the hyperplanes $h_1 - h_2 = \pm 1$ through them) but not at their mean, and so distinguishes the two stitches. Adding sites can expose this example, but arbitrary off-sample behavior remains unidentified without structural assumptions. Equal-$P$ sampled angular geometry beyond B.41's invariances is impossible at fixed positive temperature; $\beta=0$ is excluded.

### Numerical verification

The asserted script `checks/theorems_b3_check.py` was run from the repository root with `uv run --with numpy python -I checks/theorems_b3_check.py`, including the spectral, commutator, and nonlinear-head checks below. It ended `all B3 checks passed`.

**Local identity.** On a 24-site tanh/softmax network, $\|E\|_F=3.75714159887$ and $q(E)/(2N)=0.00915415565648$. Ratios of actual KL to $t^2q(E)/(2N)$ at $t=0.1,0.03,0.01,0.003,0.001$ were $0.9857628505$, $0.9958458141$, $0.9986276011$, $0.9995896098$, $0.9998633324$. Complete orthogonal-block reconstruction error was $1.87\times10^{-15}$. An eight-site binary-readable isotropic check gave direct and predicted quadratic discrepancy $4.4890625$, differing by $8.88\times10^{-16}$.

**Genuine spectral self-stitch counterexample.** With NumPy seed 11, $N=30$, four-dimensional standard-normal target rows, $H_A=H$, $\beta=4$, and the actual rank-one eigenspaces of $P^B$, the residual energy was $2.60439052043\times10^{-29}$ but $\sum_b\ell_bU_b=39.3566151378$. B.45's exact two-site spectral examples additionally returned $(q,\text{diagonal predictor})=(0,0.9)$ and $(2,1)$, to floating-point accuracy. Thus both claimed inequality directions fail without readout reduction, even when the blocks really come from the target operator.

**Jointly satisfiable spectral regime.** Let $N=24$, $\theta_x=2\pi x/N$, and

$$z_x=(\cos\theta_x,\sin\theta_x,0.7\cos2\theta_x,0.7\sin2\theta_x),\qquad H[x]=a_xz_x,\quad a_x\in[0.5,2].$$

The $a_x$ are reproducible uniform draws in the script. Since $\|z_x\|^2=1.49$, normalization removes $a_x$ and the normalized Gram depends only on $\theta_x-\theta_y$. Thus the $\beta=3$ kernel and its bistochastic operator are circulant, with Fourier spectral subspaces. The numerical spectrum has 13 distinct blocks (constant, Nyquist, and eleven Fourier pairs), with minimum inter-block gap $0.00015599257286$.

For $H_A=[\cos\theta,\cos2\theta]$, its column space is the direct sum of a one-dimensional subspace within each of two Fourier pairs. Its projector therefore commutes with all spectral blocks exactly; the measured largest commutator norm was $1.48\times10^{-15}$. Varying the positive raw norms makes the loading energies anisotropic without changing the blocks. The checks give:

| Downstream form | Lower bound | Direct $q(E)$ | Upper bound | Interval width |
|---|---:|---:|---:|---:|
| Constant Gaussian-head Fisher | 28.80216891 | 98.69086022 | 249.71080155 | 220.90863264 |
| Site-varying softmax Fisher, interference-corrected | 0.69894284 | 11.98913426 | 46.97803114 | 46.27908830 |

The softmax interference ratio was $\eta=0.266100238141$. These are valid but broad intervals, not accurate scalar predictions. After adding $10^{-3}$ Gaussian noise to the source matrix, using **one** perturbed projector consistently, the maximum commutator norm was $0.00137234126938$, the maximum direct correction $c_b$ was $0.00509480410441$, and B.46's corrected Gaussian interval $[28.70867837,250.38777680]$ contained $q(E)=98.69035003$.

For B.47, instead use $H_A=[\cos\theta,\sin\theta]$ and the site-constant anisotropic form $\bar g=\operatorname{diag}(1.5,1.5,0.4,0.4)$. Each complete spectral block is exactly readable or unreadable. Both the direct discrepancy and $N^{-1}\sum_bs_b^2u_b$ were $14.6650169533$.

**Distinguishable ideal stitches.** B.53's nonlinear-null construction returned full-stitch empirical KLs $0.0936226307057$ and $0.216890415242$ for the same fitted activation matrix, establishing the claimed difference in actual stitching response rather than only arbitrary perturbation response.

The existing `checks/cm_check.py` also ended `all checks passed`. These are identity and counterexample checks, not an experiment protocol, a sampling validation, or evidence that trained networks satisfy the ideal assumptions. The repository script asserts the quoted nonzero reference values, interval inequalities, sample budgets, and exact-zero identities; machine-roundoff diagnostics are bounded by absolute tolerances rather than pinned to BLAS-dependent last bits.

### Discrepancies with THEORY.md

1. **High-temperature qualification.** Section 8 says centered operators tend to $C$; this needs pairwise distinct normalized rows in both views, as already qualified in Section 2. With duplicate directions the correct limit is the centered within-class averaging projector (B.42), and alignment need not tend to one.
2. **Stitch loss and sensitivity units.** Section 8 leaves “sensitivity-weighted unreadability” as a hypothesis and uses a square-root sensitivity. A local KL coefficient is quadratic, with a factor $1/2$ and an additional $1/N$ for empirical expectation. Under the ideal conditions that connect its particular sensitivity to the residual, the weight is $s_b^2/N$ for counting-measure quadratic discrepancy, not $s_b$ (B.47).
3. **Section 8's scalar predictor fails without control of readout/block commutation.** For the identity stitch $H_A=H^B$, the exact residual is zero, yet B.45 and the 30-site genuine-spectral-block check give strictly positive sensitivity-weighted unreadability (the latter gives $39.3566$ in counting-measure quadratic units). The exact spectral upper-bound counterexample gives true damage $2$ against a putative bound $1$. Kernel coordinates need not be linear in their own raw activations. Readout reduction $[T,\Pi_b]=0$, or B.46's explicit commutator correction, is therefore a substantive missing condition for this scalar theorem; downstream Fisher interference and loading anisotropy also matter. This falsifies an unconditional equality/bound reading, not Section 8's separately labelled empirical correlation hypothesis. B.46–B.47 do not assert these assumptions hold in real networks.
4. **Locality and linearization.** Same-token $J_{L'\to\mathrm{out}}(x)$ is not the full derivative of simultaneous token replacement in a context-coupled model. Use whole-input sites or the full cross-site Jacobian. Moreover, local output KL is not automatically task-loss increase, and a local $t\to0$ result is not a guarantee for the full stitch $t=1$.
5. **Fit and evaluation distinction.** Section 8 proposes held-out ridge readability; the exact projector theorem concerns in-sample ridge-free optimal reconstruction. Substituting held-out ridge scores requires a separately stated statistical and regularization argument, not a renaming of $R^2$. The frozen-predictor evaluation bound B.49 is one limited bridge, not that entire argument.
6. **No extra loss of angular geometry.** A possible informal reading that bistochastic normalization hides additional sampled angles is false at fixed known positive temperature: B.41 recovers the entire normalized Gram. The limitation is mechanistic, as B.53 shows. At $\beta=0$ every operator is $\mathbf1\mathbf1^\top/N$, and this identifiability statement intentionally does not apply.
7. **Full-block decoupling is more restrictive than block-specific weights suggest.** With the constant block included, full-feature exact Fisher decoupling forces a site-independent pullback; full-feature isotropy forces the same scalar sensitivity in every block. B.46 proves this restriction and supplies restricted-feature, interference, and commutator variants. B.47 instead needs only site-homogeneous Fisher on the loading span, without isotropy.
8. **JVP budget.** One residual direction per site suffices for total local stitch damage. The full rank-$r$ loading-sensitivity sum in Section 8 generally needs $r$ directions per site or an explicitly randomized trace estimate; calling either simply “one pass” hides the cost.

No changes were made to THEORY.md. The other requested qualifications—raw CKA versus row-normalized CKA, the impossibility of an equal-$P$ angular counterexample, and the difference between a map-preserving family and an invertible transformation group—are resolved in B.41 rather than attributed as errors to THEORY.
