# Part A, cluster 2 — Stability: perturbation theory for blocks

This cluster makes the stability statements in THEORY Sections 3.2, 4 and 8 quantitative. All results are finite-site and deterministic except the explicitly labelled high-dimensional noise theorem. They do not assume sparsity, a semantic partition, a manifold, or independent sampling of sites. Sampling error enters only after it has been converted into an operator perturbation; the sample-complexity cluster supplies that separate step.

## A.20–A.22. Which spectral object is stable?

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

## A.23. Stability of the normalization, not just of the last SVD

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

## A.24–A.27. Propagating view and temperature changes to the reported objects

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

## A.28. Why the gap and the block cannot be omitted

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

## A.29–A.31. Noise cancellation: what transfers to the sphere kernel?

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

## Numerical checks performed

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

## References and exact reliance

- Bhatia, R. (1997). *Matrix Analysis*. Springer, Graduate Texts in Mathematics 169, [Theorem VII.3.1](https://doi.org/10.1007/978-1-4612-0653-8): interval-separated spectral subspaces and the Davis–Kahan bound with operator constant $1$. Exact reliance in A.21; its interval hypothesis is checked there.

- Davis, C., and Kahan, W. M. (1970). “The rotation of eigenvectors by a perturbation. III.” *SIAM Journal on Numerical Analysis* **7**(1), 1–46. [DOI](https://doi.org/10.1137/0707001). A.21 uses the interval sin-$\Theta$ theorem with constant $1$, with its hypotheses verified explicitly.
- Yu, Y., Wang, T., and Samworth, R. J. (2015). “A useful variant of the Davis–Kahan theorem for statisticians.” *Biometrika* **102**(2), 315–323. [arXiv:1405.0680](https://arxiv.org/abs/1405.0680). Theorem 2 (arXiv numbering) is a standard population-gap Frobenius alternative; no additional conclusion from it is needed here.
- Wedin, P.-Å. (1972). “Perturbation bounds in connection with singular value decomposition.” *BIT Numerical Mathematics* **12**, 99–111. [DOI](https://doi.org/10.1007/BF01932678). A.22 proves the residual argument and the particular constants used here, rather than silently importing a different gap convention.
- Sinkhorn, R. (1964). “A relationship between arbitrary positive matrices and doubly stochastic matrices.” *The Annals of Mathematical Statistics* **35**(2), 876–879. [DOI](https://doi.org/10.1214/aoms/1177703591). Standard positive matrix scaling; existence/uniqueness is also Theorem A.1. Symmetry gives a symmetric scaling, whose positive vector is unique.
- Landa, B., Coifman, R. R., and Kluger, Y. (2021). “Doubly Stochastic Normalization of the Gaussian Kernel Is Robust to Heteroskedastic Noise.” *SIAM Journal on Mathematics of Data Science* **3**(1), 388–413. [DOI](https://doi.org/10.1137/20M1342124); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8194191/); [arXiv:2006.00402](https://arxiv.org/abs/2006.00402). Exact reliance: Lemma 1 for existence/uniqueness on positive complete off-diagonal support; Theorem 3 is stated with its actual fixed-$N$, growing-ambient-dimension assumptions in A.30, which also gives the cancellation proof.

The direct derivative proof in A.23 replaces a possible Hilbert-projective-metric/Franklin–Lorenz route. It gives explicit norm constants without needing to quote an unstated contraction theorem.

## Discrepancies with THEORY.md

1. **External gap, restricted space, and matched rank must be specified.** THEORY Section 3.2's $O(\|E\|/\delta)$ statement is correct for isolated blocks. A.20–A.22 make precise the external gap, the same-index continuation, and the restriction to $\mathbf1^\perp$ after removal of the constant mode. A fixed numerical cutoff can change rank and is not covered without its own threshold margin.
2. **SVD stability does not by itself prove stability of the construction.** Kernel formation and Sinkhorn normalization intervene before the final spectrum. A.23 supplies the missing quantitative link. Dense strict positivity matters; these constants do not automatically cover the separate $k$-NN estimand or a sign-indefinite Nyström approximation.
3. **The noise theorem does not apply unchanged to the spine.** Landa–Coifman–Kluger uses a zero diagonal and additive high-dimensional noise before any row-normalization. THEORY's dense kernel retains its diagonal and normalizes each view to the sphere. A.31 disproves an unqualified transfer; the identity between sphere and chordal Gaussian kernels transfers algebra, not these statistical hypotheses.
4. **Centered alignment needs a nonzero signal condition.** Section 8's formula is undefined when either centered operator vanishes, including $\beta=0$ and a view constant across sites. Its low-temperature CKA limit also needs both centered normalized-feature Gram matrices nonzero. A.27 gives the appropriate norm-floor condition for perturbation stability.
5. **A stable block does not imply stable training pressure without stable inputs and errors.** Section 7.2's rotation invariance is exact, but perturbation continuity includes $U$ and $\Delta$ terms as well as the projector term. A.27 keeps them explicit.
6. **[P4] is a coordinate convention unless extra recovery assumptions are added.** At exact multiplicity the full isolated eigenspace is intrinsic but an individual basis is not. A declared rotation can itself have ties; no stability of that rotation follows merely from the external spectral gap. The distinction between a display rotation and an empirically validated dictionary-recovery algorithm in Sections 3.2 and 4.3 should remain explicit.
