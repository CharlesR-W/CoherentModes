# Part B, cluster 3 — Ideal case: fitting-free comparison and linear stitching

This cluster distinguishes **comparison**, which is a function of two site-space operators, from **stitching prediction**, which additionally uses a labelled linear fit, activation loadings, and derivatives of the downstream model. The latter is not fitting-free. All finite-sample identities use the counting-measure space $\mathcal H=\mathbb R^N$; an empirical expectation has an explicit factor $1/N$. No new metric is imposed on the geometric construction: output Fisher geometry enters only the declared stitching diagnostic of THEORY Sections 8–9.

Cross-references `A.9–A.10` and `A.21 and A.23` are editorial anchors for the exactness and stability clusters. The needed finite-dimensional arguments are included below, so these anchors are not gaps in the proofs.

## What comparison identifies, and its invariances

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

## The endpoints and the intermediate-temperature meaning

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

## What an ideal linear stitch can predict

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

## What held-out readability estimates require


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

## Shared-site comparison through training


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

## What is not identified


**Proposition B.53 (equal sampled representations, unequal ideal-stitch responses).** At fixed known $\beta>0$, equality $P^A=P^B$ identifies the sampled normalized activation geometry up to B.41's invariances. Nevertheless, even identical raw activations at every corresponding layer need not determine Jacobians, off-site predictions, or the response to an ideal least-squares stitch.

*Proof.* On two input sites $t\in\lbrace-1,1\rbrace$, let both models share the ReLU hidden layer

$$h(t)=(\operatorname{ReLU}(t),\operatorname{ReLU}(-t),1),\qquad d(h)=h_1-h_2.$$

Use downstream logits $z_A(h)=(d(h),h_3)$ and $z_B(h)=(d(h)+d(h)^2-1,h_3)$. If intermediate downstream features are recorded, both networks can share the same extra layer $(d(h),d(h)^2,h_3)$, followed by different affine logit heads. Thus the two networks can have the same architecture; their input, hidden, intermediate, logit, and probability activations coincide on both sites, since $d(h)=\pm1$. Every corresponding geometric operator agrees for every temperature. Their hidden-to-first-logit derivatives, however, are $(1,-1,0)$ and $(1+2d,-1-2d,0)$, which differ at both sites.

Take the common source matrix $H_{\rm src}=\mathbf1\in\mathbb R^{2\times1}$ and stitch into $h$. The ideal fitted activation at both sites is their mean $(1/2,1/2,1)$. Model A then predicts first-logit probability $\sigma(-1)$ and model B predicts $\sigma(-2)$, where $\sigma(u)=(1+e^{-u})^{-1}$. The original first-logit probabilities are $1/2$ and $\sigma(-2)$ for both models. Their mean is $\bar p=(1/2+\sigma(-2))/2>\sigma(-1)$: the strict inequality follows from strict convexity of $\sigma$ on $(-\infty,0)$, or its derivative $\sigma''=\sigma(1-\sigma)(1-2\sigma)>0$ there. Average Bernoulli cross-entropy as a function of candidate log odds $u$ has derivative $\sigma(u)-\bar p<0$ for $u\le-1$. Hence its value at $-2$ strictly exceeds its value at $-1$. The reference entropies are identical, so empirical full-stitch KL is strictly larger in model B. Numerically the two KLs are $0.093622630706$ and $0.216890415242$. ∎

**What it buys.** Identical sampled representations can support different downstream reactions to the very same optimal linear reconstruction. The missing information is mechanistic, not additional hidden angular geometry in the sampled cloud.

**Tractability and sampling.** The obstruction uses two sites and a one-column source. Every site-linear fitted reconstruction $T_\lambda H$, including ridge, has rows in $\operatorname{rowspan}(H)$; its residuals and the loading vectors also lie there. Thus the stitch sees the downstream function on this span, and its first-order diagnostics see $J_x$ only on this span. A *linear* null term vanishing on the entire row span cannot distinguish such stitches; the nonlinear term above vanishes at the observed rows (and on the hyperplanes $h_1 - h_2 = \pm 1$ through them) but not at their mean, and so distinguishes the two stitches. Adding sites can expose this example, but arbitrary off-sample behavior remains unidentified without structural assumptions. Equal-$P$ sampled angular geometry beyond B.41's invariances is impossible at fixed positive temperature; $\beta=0$ is excluded.

## Numerical verification

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

## Discrepancies with THEORY.md

1. **High-temperature qualification.** Section 8 says centered operators tend to $C$; this needs pairwise distinct normalized rows in both views, as already qualified in Section 2. With duplicate directions the correct limit is the centered within-class averaging projector (B.42), and alignment need not tend to one.
2. **Stitch loss and sensitivity units.** Section 8 leaves “sensitivity-weighted unreadability” as a hypothesis and uses a square-root sensitivity. A local KL coefficient is quadratic, with a factor $1/2$ and an additional $1/N$ for empirical expectation. Under the ideal conditions that connect its particular sensitivity to the residual, the weight is $s_b^2/N$ for counting-measure quadratic discrepancy, not $s_b$ (B.47).
3. **Section 8's scalar predictor fails without control of readout/block commutation.** For the identity stitch $H_A=H^B$, the exact residual is zero, yet B.45 and the 30-site genuine-spectral-block check give strictly positive sensitivity-weighted unreadability (the latter gives $39.3566$ in counting-measure quadratic units). The exact spectral upper-bound counterexample gives true damage $2$ against a putative bound $1$. Kernel coordinates need not be linear in their own raw activations. Readout reduction $[T,\Pi_b]=0$, or B.46's explicit commutator correction, is therefore a substantive missing condition for this scalar theorem; downstream Fisher interference and loading anisotropy also matter. This falsifies an unconditional equality/bound reading, not Section 8's separately labelled empirical correlation hypothesis. B.46–B.47 do not assert these assumptions hold in real networks.
4. **Locality and linearization.** Same-token $J_{L'\to\mathrm{out}}(x)$ is not the full derivative of simultaneous token replacement in a context-coupled model. Use whole-input sites or the full cross-site Jacobian. Moreover, local output KL is not automatically task-loss increase, and a local $t\to0$ result is not a guarantee for the full stitch $t=1$.
5. **Fit and evaluation distinction.** Section 8 proposes held-out ridge readability; the exact projector theorem concerns in-sample ridge-free optimal reconstruction. Substituting held-out ridge scores requires a separately stated statistical and regularization argument, not a renaming of $R^2$. The frozen-predictor evaluation bound B.49 is one limited bridge, not that entire argument.
6. **No extra loss of angular geometry.** A possible informal reading that bistochastic normalization hides additional sampled angles is false at fixed known positive temperature: B.41 recovers the entire normalized Gram. The limitation is mechanistic, as B.53 shows. At $\beta=0$ every operator is $\mathbf1\mathbf1^\top/N$, and this identifiability statement intentionally does not apply.
7. **Full-block decoupling is more restrictive than block-specific weights suggest.** With the constant block included, full-feature exact Fisher decoupling forces a site-independent pullback; full-feature isotropy forces the same scalar sensitivity in every block. B.46 proves this restriction and supplies restricted-feature, interference, and commutator variants. B.47 instead needs only site-homogeneous Fisher on the loading span, without isotropy.
8. **JVP budget.** One residual direction per site suffices for total local stitch damage. The full rank-$r$ loading-sensitivity sum in Section 8 generally needs $r$ directions per site or an explicitly randomized trace estimate; calling either simply “one pass” hides the cost.

No changes were made to THEORY.md. The other requested qualifications—raw CKA versus row-normalized CKA, the impossibility of an equal-$P$ angular counterexample, and the difference between a map-preserving family and an invertible transformation group—are resolved in B.41 rather than attributed as errors to THEORY.
