# Part A, cluster 1 — Exactness and structure of the spine

This cluster concerns finite, fixed data. It introduces no fitted representation, semantic labels, sparsity assumption, or additional metric. A theorem about these finite matrices is not a population-consistency theorem. In particular, none of the algebraic guarantees below requires independent sites; independence enters only where expressly assumed for random probes.

## Conventions and computational accounting

Let $X$ be the common site set, $N=|X|\ge 1$, $\mathcal H=\mathbb R^N$ with counting measure, $e=\mathbf 1/\sqrt N$, and $C=I-ee^\top$. Write $\mathcal H_0=e^\perp$. Vector norms are Euclidean, $\|\cdot\|_2$ on matrices is the operator norm, and $\|\cdot\|_F$ is the Frobenius norm. An entrywise nonnegative matrix is **bistochastic** if every row and column sums to one. "Positive entries" and "positive semidefinite (PSD)" are distinct conditions.

A view has nonzero rows $\phi(x)\in\mathbb R^d$, normalized to $z_x=\hat\phi(x)$; set $Z=\hat\Phi$, $\Gamma=ZZ^\top$, and $K_\beta[x,y]=\exp(\beta \Gamma_{xy})$, $\beta\ge0$. The declared common-site exclusion policy for zero rows is required before these definitions. The scaling vector is $v$, while a right singular function of $Q$ is $v_i$. The Gram matrix is $\Gamma$ (not the gradient $G$), centered Grams use $\Gamma^c$ or $\Gamma_c$, and the all-ones matrix is written $\mathbf1\mathbf1^\top$ (reserving $J_x$ for Jacobians). Feature matrices $H^L$ retain THEORY's activation meaning. The target set $B$ is local to A.5; the spectral block $B$ is local to A.7. Width $d$, degree vectors $\mathbf d_i$, slot count $n_{\mathrm{use}}$, nonzero count $n_{\mathrm{nz}}$, block dimension $r_B$, and sketch $\zeta_x$ are distinct.

The following cost convention is used in the remarks. Building a dense sphere kernel costs $O(N^2d)$ arithmetic and $O(N^2)$ storage. A kernel or normalized-operator matvec costs $O(N^2)$ dense, or $O(n_{\mathrm{nz}})$ for an explicitly stored sparse kernel with $n_{\mathrm{nz}}=\operatorname{nnz}(K)$. One alternating row/column Sinkhorn sweep uses a constant number of matvecs; one damped symmetric step uses one. Its iteration count is **not** universally a few dozen: it depends on the kernel, tolerance, and conditioning of the scaling problem. The implemented damped step has local linearization rate at most $1/2$ for PSD kernels and global Hilbert-metric contraction factor at most $(1+\tanh\beta)/2$ for a sphere kernel, proved in A.1. The global bound deteriorates at large $\beta$; the local bound explains rapid final convergence without a spectral-gap requirement. Knight (2008, Theorem 4.4 in the author's text) concerns the different, undamped symmetric alternating iteration, initialized at the all-ones vector. Its transverse two-step rate is governed by $\max_{i\ne0}|\lambda_i(P)|^2$ under full indecomposability, which strict positivity guarantees. These algorithm-specific rates are not interchangeable.

Lanczos with $\ell$ iterations for the top $r\le\ell$ eigenpairs costs $O(\ell N^2+N\ell^2)$ dense with full reorthogonalization, or $O(\ell n_{\mathrm{nz}}+N\ell^2)$ sparse, plus the small projected eigensolve. Working-vector storage is $O(N\ell)$. The number $\ell$ depends on gaps, clustering, accuracy, and restarts; $O(rN^2)$ is not an unconditional total-cost guarantee. A matrix-free SVD uses matvecs with $Q$ and $Q^\top$ and has analogous accounting. Sparsification changes the estimand and can destroy PSD or scalability; dense-kernel results are not automatically graph results.

## Normalization and diffusion

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

## Shared-measure composition and the coherent-set relaxation

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

## Exact expansions, loadings, gradients, and probes

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

## The two temperature endpoints

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

## Numerical verification

The repository's existing `checks/cm_check.py` was run unchanged with `uv run --with numpy python -I` and printed `all checks passed`. It checks the normalization/product distinction, small-temperature alignment, reconstruction tail, and several neighboring theory identities.

The reproducible script `checks/theorems_a1_check.py` is run from the repository root with `uv run --with numpy python -I checks/theorems_a1_check.py`; its final success line is `all A1 checks passed`, observed after the review revisions. It asserts scaling and scale invariance; the convex Hessian; spectra, powers, and diffusion distance; composition, degree-normalization contrast, dissipation, balanced-set identities, and the singular-subspace min-max formula; exact reconstruction, centroid and PCA loadings; multi-slot gradients against finite differences, block rotation invariance, slot-rank bounds, truncation bounds, and frozen-input finite updates; both probe variance constants by exhaustive Rademacher enumeration and degree-four-exact Gaussian quadrature (one probe and an independent two-probe average); the explicit small-temperature remainder across four sample sizes and six temperatures; CKA and gapped-projector limits; and the high-temperature angular rate. It also checks the non-PSD-positive, fractional-power, reducible-degree, and duplicate-row boundary examples. Maximum observed small-temperature error divided by the displayed bound was **0.031902**; the CKA discrepancy at temperature 0.00001 was **0.00000141958**. The fractional-power counterexample gave a negative entry **-0.014593245**. These deterministic checks are regression evidence, not substitutes for the proofs. Review regressions additionally assert the PSD degree counterexample, duplicate-class projector and rate, excluded-use/direct-term gradient accounting, damped local Jacobian, global projective contraction and absolute log-error bounds, and the sharpened A.5/A.10 bounds.

## References

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

## Discrepancies with THEORY.md

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
