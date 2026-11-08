# Part B, cluster 2 — Ideal dictionaries, sparse concept clusters, and ideal sparse autoencoders

This cluster concerns a **conditional regime**, not an assumption of the construction. Its clusters are not supplied to the sphere kernel or its SVD. They describe an idealized data-generating geometry, and are used to state and validate recovery. An ideal sparse autoencoder below is a **comparison baseline**, not an extra trained component of the construction.

Throughout, $\mathcal H=\mathbb R^N$ has counting measure, $K_\beta[x,y]=\exp(\beta\langle\hat\phi(x),\hat\phi(y)\rangle)$, and $P=D_vK_\beta D_v$ is the symmetric bistochastic normalization. Eigenvalues here are numbered $\lambda_1\ge\cdots\ge\lambda_N$, **including the constant mode**. Thus an $m$-cluster block means the constant plus $m-1$ nonconstant modes. One must restore the constant before rotating this block into cluster indicators. All operator norms are Euclidean spectral norms.

## The finite-temperature dictionary window

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

## From an identifiable block to a dictionary

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

## What an ideal sparse autoencoder must — and need not — recover

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

## Sparse sums: recovery of active sets versus recovery of features

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

## Diagnostics: exact certificates and their limits

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

## Two scales give a dictionary hierarchy

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

## The sampling interface used by these theorems

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

## Numerical verification

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

## Discrepancies with THEORY.md

No changes to `THEORY.md` are made here.

1. **Section 4.3, projector IPR.** The broad bound $1/N\le\mathrm{IPR}\le1$ is valid but not sharp at fixed block rank. A rank-$r>1$ block cannot attain the stated “one site” endpoint: its sharp upper bound is $1/r$, attained by a coordinate subspace on $r$ sites. Balanced indicator dictionaries can attain the *minimum* IPR. B.32 supplies both corrections.
2. **Sections 3.2–4.3, centering and localization.** A partition into $m$ clusters has $m$ uncentered indicator dimensions but only $m-1$ centered contrasts. A localizing rotation to the original indicators must reattach the constant. Rotating a centered contrast block alone cannot produce nonnegative, nonzero cluster indicators.
3. **Sections 4.3 and 5, a localizing rotation.** Exact maximum-IPR recovery is proved here, but practical varimax convergence and a dimension-independent linear noisy recovery rate are not. B.23 distinguishes its global optimum from its iterative implementation; B.24 gives a costly certified alternative.
4. **Section 11, sparse features.** Active-set indicators, singleton indicators and overlapping feature-presence functions are different targets. Strong positive-sum/prototype-separation assumptions justify the first; singleton coverage exposes feature directions. They do not establish exact code recovery in the cited signed, gated compressed-computation model.
5. **Section 5, “SAE-like.”** The wording is appropriately weaker than an equality theorem. One-hot centroids and free-amplitude sparse coding are different objectives; the requested unrestricted equivalence would be false. B.26 and B.28 state distinct true equivalences. B.26 also requires the loading view to equal the baseline reconstruction matrix; normalized-data centroids cannot be equated with raw-data centroids when amplitudes differ.
6. **Section 2, the high-temperature endpoint.** The identity limit is already qualified there by distinct normalized vectors. Exact repeated rays are the relevant exception: their limit is the cluster-indicator projector, consistent with B.21. For clouds of distinct sites, the finite-window warning remains essential.

## Notes on imported results

The only results imported without re-proving their general existence theory are the construction's already-established positive Sinkhorn scaling and PSD sphere kernel. Spectral estimates, localization, SAE statements, diagnostics and hierarchy are proved above; the eigenvalue variational principle and the cosine-sine/SVD identities are standard finite-dimensional linear algebra (see, e.g., Horn and Johnson, *Matrix Analysis*, 2nd ed., 2013, and Stewart and Sun, *Matrix Perturbation Theory*, 1990). The Davis–Kahan citation accompanies a self-contained estimate. No unverified external sampling theorem is used: B.34 is an explicit interface pending the sampling cluster's actual assumptions and constants.
