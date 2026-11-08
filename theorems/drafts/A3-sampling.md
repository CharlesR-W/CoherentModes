# Part A, cluster 3 — Sample complexity, estimation and cost

This cluster separates **statistical error** (sites), **numerical approximation** (landmarks/features and solver tolerance), and **measurement error** (stochastic resamples or Jacobian probes). They are not interchangeable sample sizes. Confidence levels satisfy $0<\eta\le1/2$. All population statements below fix the view, model parameters, temperature and sampling distribution before drawing sites. Choosing these on the same data requires a uniform bound, a finite-search union bound, or an independent split. The covariance event in A.47 is already uniform over ridge values for fixed views and windows, so its sampling bound permits data-selected ridge values; a zero-ridge consistency claim still needs the selected schedule to satisfy the stated conditions.

Indexed $\Sigma$ symbols denote RKHS covariance operators; $C$ remains THEORY's site-centering matrix. The kernel ratio $\chi_\beta$ is distinct from the ridge smoothers $R_p,R_q$, and $B_\beta$ is a kernel bound, not a reference-set size.

## Population and empirical operators

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

## Out-of-sample eigenfunctions

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

## Context as regularized cross-covariance

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

## Temperature estimators

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

## Computation and approximation

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

## Cross-layer products and document-level sampling

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

## Numerical checks performed

The persistent script `checks/theorems_a3_check.py` was run with `uv run --with numpy python -I checks/theorems_a3_check.py` (base seed 4417; revision-check seed 9055). All assertions passed. It is a finite-dimensional sanity check, not a proof or an experiment on a trained network.

- On a five-point nonuniform sphere population, across $N=128,512,2048,8192$ and 350 repetitions per size, mean raw-operator, normalized lifted-operator and eigenvalue errors had log–log slopes $-0.515$, $-0.517$, and $-0.574$, respectively. Every sampled eigenvalue error satisfied the operator perturbation bound.
- Nyström eigenfunction extension reproduced the training values with the required $\sqrt N$ conversion to probability normalization.
- A two-state conditionally independent past/target toy had population centered CCA rank one. Across 200 repetitions per size, fixed-ridge operator error and the empirical second canonical correlation had slopes $-0.500$ and $-0.503$; every empirical tail was below its actual operator error and the explicit covariance-to-CCA bound.
- A ridge-leverage landmark sample satisfied the PSD residual and normalized operator-error certificate. The conservative bound requested 301 draws for 60 sites, correctly illustrating a case in which this certificate does not promise compression.
- Pairwise-noise and centered-variance formulas agreed to numerical precision; mean scale-estimation error had slope $-0.516$.
- Across 6,000 shared-Gaussian probe replicas, the measured raw-Gram variance divided by the exact variance was $0.998$.
- Revision checks verify the population-sensitive Sinkhorn bound; the exact three-covariance $Q$ lift and both its Hölder and covariance-floor perturbation bounds (finite-support observed slope $-0.494$); the nonleading canonical-function inequality and marginal centering; Bernstein tails against exact binomial enumeration; duplicated within-document positions; projective Sinkhorn contraction; and the orthogonal-slow-contrast counterexample to inferring cross-layer coherence from slow Sinkhorn convergence.
- The existing `/home/ubuntu/code/CoherentModes/checks/cm_check.py` also passes. This revision does not modify THEORY.md; it adds the persistent asserted check above.

## Discrepancies with THEORY.md

1. **Section 13's sampling rate is for a shrinking-bandwidth generator applied to a smooth function.** It is not the fixed-$\beta$ integral-operator or eigenspace rate proved here. Do not substitute one for the other. A manifold, a dimension, boundary conditions and an explicit dependence regime are additional hypotheses, not consequences of the dot product.
2. **Section 3's “few dozen” Sinkhorn products is an empirical expectation, not a uniform cost guarantee.** Worst-case iteration counts depend exponentially on $\beta$ (or on spectral conditioning). Plain symmetric iteration $s\leftarrow1/(Ks)$ can oscillate in its scalar gauge; alternating left/right factors or another convergent implementation must be specified.
3. **Section 5 [M5] needs a measure conversion.** The probability-normalized extension takes sample values $\sqrt N\psi_i$; its counting-normalized version is $\widehat f_i/\sqrt N$. Both normalization factors and division by the eigenvalue are necessary.
4. **Section 6.3 does not currently promise universal consistency, correctly.** Such a claim needs i.i.d./quantified dependence, a ridge schedule, compactness for the zero-ridge limit, gaps for directions, and range conditions for canonical functions. Choosing ridge/window by held-out score is not itself proof of these conditions. The sufficient schedule is strictly slower than $N^{-1/3}$, not equal to it.
5. **Section 10's pairwise displacement and centered variance differ by a factor of two** (and a finite-$n$ correction); the mean-removal step does not change pairwise differences. Its nearest-neighbour median is a different statistic, with a moving reference-size-dependent estimand and no distribution-free reciprocal concentration from boundedness alone.
6. **Section 13's Nyström warning is essential.** PSD and small normalized operator error do not ensure positive entries or a well-conditioned Sinkhorn solve. A rank-$r$ Lanczos “sweep” also needs its Krylov length and reorthogonalization cost specified; $O(rN^2)$ is matvec work for $O(r)$ steps, not a guaranteed total eigensolve cost.
7. **Section 7.3's unbiasedness is correctly limited to raw inner products.** The finite-fourth-moment assumption gives root-probe standard error, but logarithmic high-probability simultaneous cosine bounds need stronger tails (Gaussian probes suffice here).

No edits to THEORY.md are made here; these are distinctions or implementation clarifications for the owner.

## Sources

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

## Summary: cost, sampling and failure modes

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
