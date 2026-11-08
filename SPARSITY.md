# Sparsity as a certified regime

Status: v1.0, 2026-11-07. Companion to `THEORY.md` (v1.5) and `THEOREMS.md`; written for readers who want to know in what sense this construction recovers sparse or discrete structure. Sparsity is never a premise here: `THEORY.md` treats clustering as a measured regime (Sections 4.2–4.3), and this document proves what is recovered, and how its presence is certified, when that regime holds. Drafted by a GPT-6 Astra agent, reviewed independently by GPT-6.1 Sol and Claude Opus 5.5 agents (`reviews/sparsity-review-*.md`), revised and re-approved. Asserted numerical checks: `uv run --with numpy python -I checks/sparsity_check.py`. The caution in `CLAUDE.md` about the vocabulary of `THEOREMS.md` applies here too.

The sparsity viewpoint asks an important question: can a distributed activation cloud have a small, discrete account? Here the answer can be yes, without making sparsity a condition of entry. The primitive is the spectral basis; partitions are level sets interpreted afterwards (THEORY §4.2). A gap identifies a stable subspace, while cuts and indicator-subspace comparisons determine whether it supports a discrete account (§4.3). Localization and spectral separation answer different questions. All dictionary/indicator recovery results below cover **disjoint partitions**; recovery of the overlapping or $k$-sparse superpositions targeted by SAEs remains open here.

Four readings of the **same measured operators** are useful: sparse graph surrogates; low-conductance cuts; sparse coordinates within a spectral block; and multiscale survival under diffusion or temperature changes. None inserts semantic labels into the operator. Keep graph sparsity, low-leakage sets, and localized coordinates distinct. SEBA/varimax are optional optimization **outside the construction**: their output is not itself an SVD output or a canonical measurement, even though SEBA uses SVD updates.

Throughout, $X$ has $N\ge2$ sites, $\mathcal H=\mathbb R^N$ has counting measure, and $P=D_vK_\beta D_v$ is the symmetric, nonnegative, doubly stochastic **dense-kernel** operator of THEORY §3. Its spectrum is $1=\lambda_0\ge\lambda_1\ge\cdots\ge0$. The unsubscripted $v$ denotes Sinkhorn scaling; $v_i$ would denote a right singular vector of $Q$. Constants are included when discussing indicator spans. A centered block alone cannot contain nonzero nonnegative indicators; restore the constant when the proposed partition covers $X$.

## 1. Sparse operators without sparse features

**Definition S.1 (spectral sparsifier).** For symmetric nonnegative weights $W$, let $D=\operatorname{diag}(W\mathbf1)$ and $\mathcal L_W=D-W$. Loops contribute to $D$ but cancel in $\mathcal L_W$. An $\varepsilon$-spectral sparsifier, $0<\varepsilon<1$, is a reweighted subgraph with Laplacian $\widehat{\mathcal L}$ satisfying
$$
(1-\varepsilon)\mathcal L_W\preceq\widehat{\mathcal L}\preceq(1+\varepsilon)\mathcal L_W.
$$
This preserves every quadratic energy and every cut, not individual entries of $W$.

**Theorem S.2 (sparse surrogate of a bistochastic kernel).** For the stated $P$, $\mathcal L=I-P$ is both its combinatorial and normalized Laplacian, using degree one including loops. There exists a sparsifier with $O(N\log N/\varepsilon^2)$ edges by effective-resistance sampling with the small-parameter exceptions below, or $O(N/\varepsilon^2)$ edges by Batson–Spielman–Srivastava. If $\mu_i=1-\lambda_i$ and $\widehat\mu_i$ are the increasingly ordered eigenvalues of $\mathcal L$ and $\widehat{\mathcal L}$, then
$$
(1-\varepsilon)\mu_i\le\widehat\mu_i\le(1+\varepsilon)\mu_i,
\qquad \|\widehat{\mathcal L}-\mathcal L\|_2\le\varepsilon.
$$
An isolated spectral block therefore has Davis–Kahan subspace error $O(\varepsilon/\delta)$ when its separating gap $\delta$ dominates $\varepsilon$. A nonnegative, PSD, doubly stochastic sparse surrogate is
$$
P_* = I-\widehat{\mathcal L}/(1+\varepsilon),\qquad
\|P_*-P\|_2\le 2\varepsilon/(1+\varepsilon).
$$
It has the sparsifier's eigenvectors; its cut/energy factors are $[(1-\varepsilon)/(1+\varepsilon),1]$. These are **relative guarantees for $1-\lambda_i$**, not multiplicative guarantees for $\lambda_i$. If both layer operators are replaced with such contraction surrogates having errors $e_L,e_{L'}$, the product error is at most $e_L+e_{L'}$; isolated nonzero singular blocks of $Q$ then have Wedin error $O((e_L+e_{L'})/\delta)$ when this error is small relative to their singular gap $\delta$.

*Proof.* Degree one gives $D=I$. For every $f$,
$$
f^\top\mathcal Lf=\frac12\sum_{x,y}P_{xy}(f_x-f_y)^2.
$$
Apply Spielman–Srivastava (2011), Theorem 1, to the weighted undirected graph with off-diagonal weights $P_{xy}$; these weights are positive, so it is connected for $N>1$. Its edge leverage scores are $P_{xy}R_{xy}$, where $R_{xy}=(e_x-e_y)^\top\mathcal L^\dagger(e_x-e_y)$. Sample with probabilities proportional to these scores and reweight by inverse sampling probability. Their Theorem 1 assumes sufficiently large $N$ and $1/\sqrt N<\varepsilon\le1$, taking $q=9C_0^2N\log N/\varepsilon^2$ samples for its universal constant $C_0$, with constant success probability. For $\varepsilon\le1/\sqrt N$, or bounded small $N$, retain the whole graph: its $O(N^2)$ edges fit the stated budget. Batson–Spielman–Srivastava (2012), Theorem 1.1, applies to any weighted undirected graph and gives at most $\lceil b(N-1)\rceil$ edges, $b>1$, with
$$
\mathcal L\preceq\mathcal L_H\preceq\kappa_b\mathcal L,\qquad
\kappa_b=\frac{b+1+2\sqrt b}{b+1-2\sqrt b}.
$$
Multiplying $\mathcal L_H$ by $2/(1+\kappa_b)$ gives error $2\sqrt b/(b+1)\le\varepsilon$ when $b\ge4/\varepsilon^2$. Loops do not enter either application.

Courant–Fischer applied to the Loewner inequalities gives the eigenvalue bounds. Since $0\preceq\mathcal L\preceq I$, those inequalities also give the norm bound. The standard Davis–Kahan theorem applies to these real symmetric matrices; for a top/bottom spectral cut with gap $\delta$ and perturbation norm $e<\delta/2$, the projector bound is $2e/\delta$ (standard; see Stewart–Sun, *Matrix Perturbation Theory*, 1990, Chapter V).

The off-diagonal entries of $P_*$ are nonnegative, its rows sum to one, and $\widehat{\mathcal L}\preceq(1+\varepsilon)I$ makes it PSD with nonnegative diagonal. Subtracting $P$ proves the final norm bound. For products, insert and subtract $P_*^{L'}P^L$ and use that all factors have norm at most one. Apply the standard Wedin singular-subspace theorem (Stewart–Sun, Chapter V), with separation from excluded singular values and zero. ∎

**What it buys.** The operator admits a sparse surrogate with energy, cut, and block guarantees, without learning a dictionary. The guaranteed object is canonical in its target, not a unique sparsifying edge set.

**Tractability and sampling.** Spielman–Srivastava's resistance sketch uses nearly-linear-time Laplacian solves, building on Spielman–Teng; “nearly linear” is in the input edge count $m$, which is $\Theta(N^2)$ for a dense kernel. Forming that kernel costs $O(N^2d)$ and dense Sinkhorn costs $O(N^2)$ per iteration. Subsequent sparse matvecs cost $O(N\log N/\varepsilon^2)$. The sampling constants can eliminate practical edge savings at moderate $N$; the edge counts are asymptotic. BSS's edge-optimal construction does not supply the same practical runtime. Edge sampling is conditional on the observed sites, not a statistical generalization bound.

One may instead sparsify $K$, preserving its **unnormalized graph Laplacian**. But Sinkhorn of a sparsified $K$ is not sparsification of Sinkhorn of $K$: nonlinear rescaling changes the target. Sparse support can obstruct scaling if the diagonal is dropped. Retaining a positive diagonal prevents that obstruction for a symmetric nonnegative matrix: each positive entry belongs to a positive permutation diagonal (use a transposition and otherwise diagonal entries), giving total support and hence a two-sided scaling by the Sinkhorn–Knopp theorem (1967); for a symmetric matrix with total support the scaling can be taken symmetric and is then unique (Csima and Datta 1972; or: if $D_1 K D_2$ is doubly stochastic so is its transpose $D_2 K D_1$, and uniqueness of the scaling up to a scalar forces $D_1 \propto D_2$). This does not guarantee PSD or the desired normalized approximation. Normalize first for S.2. A symmetrized $k$NN truncation is cheaper to store but has no general spectral-sparsifier guarantee; THEORY §13 correctly treats it as a separate estimand.

## 2. Cuts: when near-one eigenvalues certify discrete structure

For nonempty proper $A\subset X$, define
$$
\Phi_P(A)=\frac{\sum_{x\in A,y\notin A}P_{xy}}{|A|},\quad
h(P)=\min_{0<|A|\le N/2}\Phi_P(A),\quad
h_k(P)=\min_{A_1,\ldots,A_k\text{ disjoint, nonempty}}\max_j\Phi_P(A_j).
$$
The $k$ sets need not cover $X$.

**Theorem S.3 (Cheeger, with explicit loop removal).** For this $P$,
$$
\frac{1-\lambda_1}{2}\le h(P)\le\sqrt{2(1-\lambda_1)},\qquad
\frac{1-\lambda_{k-1}}2\le h_k(P),\quad 2\le k\le N.
$$
The two-way upper bound has a second-eigenvector sweep witness. To apply higher-order Cheeger **without an unproved self-loop extension**, let $W=P-\operatorname{diag}(P)$, $d_x=1-P_{xx}>0$, $D=\operatorname{diag}(d)$, and $d_{\min},d_{\max}$ be the extreme degrees. Let $\nu_i$ be the increasingly ordered eigenvalues of $D^{-1/2}(I-P)D^{-1/2}$, and let $h_k(W)$ use conductance denominator $\sum_{x\in A}d_x$. Then
$$
\frac{\nu_{k-1}}2\le h_k(W)\le Ck^2\sqrt{\nu_{k-1}},
$$
$$
\frac{1-\lambda_i}{d_{\max}}\le\nu_i\le\frac{1-\lambda_i}{d_{\min}},\qquad
h_k(P)\le\min\left(d_{\max},\ Ck^2\frac{d_{\max}}{\sqrt{d_{\min}}}\sqrt{1-\lambda_{k-1}}\right).
$$
The last conversion factor is explicit; this document does not claim the factor-free higher-order bound for holding kernels merely by citing the loop-free theorem.

*Proof.* The first statement is the finite reversible-chain Cheeger inequality (standard; Chung, *Spectral Graph Theory*, 1997, Chapter 2; for reversible chains with holding probabilities, Levin and Peres, *Markov Chains and Mixing Times*, 2nd ed., 2017, Theorem 13.10). Nonnegative probabilities, uniform positive stationary mass, and symmetry verify its hypotheses; PSD is unnecessary, so it also applies to S.4's bipartite chain.

For the direct converse, on the span of disjoint indicators write $f=\sum_jc_j\mathbf1_{A_j}$. Edgewise $(c_i-c_j)^2\le2(c_i^2+c_j^2)$ gives $f^\top(I-P)f\le2\sum_jc_j^2\operatorname{cut}(A_j)\le2\max_j\Phi_P(A_j)\|f\|^2$. Min–max proves $1-\lambda_{k-1}\le2h_k(P)$.

$W$ is a weighted undirected loop-free graph with positive degrees, so Lee–Oveis Gharan–Trevisan (2014), Theorem 1.1, applies exactly. Its Laplacian is $D-W=I-P$, but its normalized Laplacian and volume measure differ from those of $P$. The generalized Rayleigh quotient is $f^\top(I-P)f/(f^\top Df)$. Since $d_{\min}\|f\|^2\le f^\top Df\le d_{\max}\|f\|^2$, min–max gives **both directions** of the eigenvalue comparison above. Also $\Phi_P(A)=(\sum_{x\in A}d_x/|A|)\Phi_W(A)\le d_{\max}\Phi_W(A)$. Apply this to the disjoint sets furnished by the cited theorem, then use $\nu_{k-1}\le(1-\lambda_{k-1})/d_{\min}$. Finally every $\Phi_P(A)\le d_{\max}$, proving the trivial cap. ∎

**What it buys.** Near-one eigenvalues certify low-leakage sets with the displayed losses; the higher-order conversion can be loose for heterogeneous holding probabilities. A separating gap stabilizes a span, but a gap alone certifies neither clusters nor semantics.

**Tractability and sampling.** A two-way sweep costs $O(m+N\log N)$ after the eigenvector. Higher-order rounding uses $W$'s normalized spectral embedding and polynomial-time postprocessing; report its $d_{\min},d_{\max}$ factors. Exact minimum conductance is unnecessary. These finite-site statements need coverage, dependence control, and spectral-estimation bounds for population interpretation.

**Theorem S.4 (coherent-pair relaxation and rounding).** Let $Q=P^{L'}P^L$, let $\sigma_1$ be its top **nontrivial** singular value ($\sigma_0=1$), and write $m(A,B)=\langle Q\mathbf1_A,\mathbf1_B\rangle$. Then $m(A,B)/|A|$ is a transported mass fraction. For $0<a=|A|=|B|<N$, define $f_A=(\mathbf1_A-a\mathbf1/N)/\sqrt{a(1-a/N)}$. Then
$$
\langle Qf_A,f_B\rangle
=\frac{m(A,B)/a-a/N}{1-a/N}\le\sigma_1,
\qquad
\sigma_1=\max_{f,g\perp\mathbf1,\ \|f\|=\|g\|=1}\langle Qf,g\rangle.
$$
Conversely, put $\kappa=\sqrt{2(1-\sigma_1)}$. If $\kappa<1$, a spectral sweep yields nonempty $A,B$ with $a+b\le N$ and
$$
\frac{|a-b|}{a+b}\le\kappa,\qquad
\min\left(\frac{m(A,B)}a,\frac{m(A,B)}b\right)\ge\frac{1-\kappa}{1+\kappa}.
$$
Trimming the larger set to equal cardinality yields a pair with transport fraction at least $1-2\kappa/(1-\kappa)$; its centered score is at least $1-4\kappa/(1-\kappa)$. The witness condition is $\sigma_1>1/2$; the trimmed transport bound is positive only for $\sigma_1>17/18\approx0.9444$, and the centered bound is positive only for $\sigma_1>49/50=0.98$. There is no lower bound on either set's population fraction.

*Proof.* Column stochasticity conserves source mass; nonnegativity proves the fraction claim. Expanding centered indicators proves the identity. The variational equality is the SVD/Courant–Fischer characterization on $\mathbf1^\perp$.

Apply S.3's two-way bound to
$$
T=\begin{pmatrix}0&Q^\top\\Q&0\end{pmatrix}.
$$
It is symmetric doubly stochastic on two copies of $X$, with eigenvalues $\pm\sigma_i$; its second largest eigenvalue is $\sigma_1$. A sweep set $S=A\sqcup B$ of size $a+b\le N$ has boundary weight $D=a+b-2m\le\kappa(a+b)$. Since $m\le\min(a,b)$, $|a-b|\le D$; since $\kappa<1$, both sets are nonempty. Also $m\ge(1-\kappa)(a+b)/2$ and $\max(a,b)\le(1+\kappa)(a+b)/2$, proving the fractions.

Assume $a\ge b$. Removing $a-b$ source sites loses at most $a-b$ mass because each source column sums to one. The new leakage is at most $a-m\le D$. As $b\ge(1-\kappa)(a+b)/2$, the transport bound follows. Finally $b\le N/2$ and the centered identity multiply the leakage by at most two. ∎

**What it buys.** High singular coherence has a discrete witness with the stated nonvacuity thresholds, not merely continuous coordinates. “Balanced” must distinguish equal source/target cardinality from a macroscopic lower mass requirement; only the former follows here.

**Tractability and sampling.** Apply $Q$ and $Q^\top$ as products, avoiding dense product formation. Singular-vector computation uses these matvecs; evaluating all sweep cuts of a dense $Q$ may still cost quadratic work and is not automatically cheap from a matvec oracle. Small extracted sets are especially vulnerable to inadequate site coverage.

Froyland (2013), Propositions 1–2, gives the variational characterization; **Theorem 2, equation (8)** bounds the sum of the two complementary transport fractions by $1+\sigma_1$ in our indexing. His Assumption 1 requires a nonnegative square-integrable kernel, $L1=1$, and $L^*1=1$—unitality and integral preservation, **not an invariant initial measure**. With uniform probability measures take $k(x,y)=NQ_{yx}$: all three conditions hold. Finite dimension gives compactness, and strict positivity of $Q^\top Q$ gives the simple constant singular pair required by Assumption 2. **The bipartite rounding proof above is supplied here**, not attributed to that paper. Froyland–Padberg's coherent-set program is related; Froyland's dynamic-Laplacian Cheeger theorem (2015, Theorem 3.2) instead concerns evolving geometric boundaries under smooth volume-preserving dynamics. Those manifold hypotheses are not imposed here, so that theorem is not directly invoked for arbitrary layer kernels.

## 3. Sparse coordinates inside a block

**Definition S.5 (SEBA postprocessing).** Given an orthonormal block matrix $V\in\mathbb R^{N\times r}$, SEBA seeks
$$
\min_{R^\top R=I,\ \|s_j\|_2=1}
\frac12\|VR^\top-S\|_F^2+\mu\|S\|_{1,1}.
$$
Froyland–Rock–Sakellariou (2019), §3, equation (1), Algorithm 3.1, alternates normalized soft-thresholding $s_j=C_\mu((VR^\top)_j)/\|C_\mu((VR^\top)_j)\|$ and an orthogonal Procrustes update: choose **any** minimizer $R=U_ZW_Z^\top$ from a full orthogonal SVD $S^\top V=U_Z\Sigma_ZW_Z^\top$. When singular, use an orthogonal SVD completion; a unique polar factor is not assumed. It then changes signs, rescales maxima to one, and ranks negative minima as a reliability diagnostic. Nonnegativity is encouraged, not constrained. Thresholding can leave the original span and even lose rank: the output is a **candidate sparse approximation**, not automatically an orthonormal basis. The $\ell^1$ weight $\mu$ is a sparsity preference in this external readout. Reports must include a $\mu$ sweep and certificates defined independently of that penalty: candidate-subspace distance, rank, and cut quantities (their evaluated values may change with the candidate).

**Proposition S.6 (convergence of the idealized core; proved here).** Fix orthonormal $V$ and $0<\mu<1/\sqrt N$. For any selection of orthogonal Procrustes minimizers, the unbounded iteration of the alternating core is well-defined, its objective values converge, and every accumulation point is coordinatewise minimizing. This is our argument, not a convergence theorem attributed to Froyland–Rock–Sakellariou. This does not establish full-sequence convergence, global optimality, nonnegativity, or exact indicator recovery.

*Proof.* Every unit column has an entry of magnitude at least $1/\sqrt N$, so thresholding is nonzero. At fixed $R$, minimizing the column objective is maximizing $\langle z,s\rangle-\mu\|s\|_1$ over $\|s\|=1$. Matching signs and Cauchy–Schwarz give the normalized soft-threshold solution. At fixed $S$, an orthogonal SVD completion exactly solves Procrustes even when thresholded columns coincide and $S^\top V$ is singular. Thus each step decreases a nonnegative objective. The feasible set is compact, so accumulation points exist. The decrease in each block update tends to zero; continuity of the objective and of its minimum over a compact block implies that at every accumulation point both block suboptimalities vanish. ∎

**What it buys.** There is a genuine optimization guarantee, but no license to treat successful-looking sparse outputs as recovered concepts. The published implementation perturbs near-constant columns after QR, potentially breaking orthonormality, and stops after at most 5000 iterations. S.6 concerns fixed orthonormal input and indefinite exact updates; it is not a theorem about that perturbed, capped implementation.

**Tractability and sampling.** Each core iteration costs $O(Nr^2+r^3)$, after the block is computed. Iteration count and initialization sensitivity must be reported. The guarantee is finite-dimensional and does not supply a required number of sites.

**Theorem S.7 (exact indicator regime, and the necessary initialization qualification).** Suppose the block equals the span of normalized indicators $H_j=\mathbf1_{A_j}/\sqrt{n_j}$ of a partition into $r$ nonempty sets. This is the disjoint sparse-concept-cluster regime of the **dictionary theorem, B.21–B.23 of `THEOREMS.md`**. Then:

1. A rotation with $VR^\top=H$ exists. With that initialization, no constant-column perturbation, and $0<\mu<\min_jn_j^{-1/2}$, one SEBA core step returns $H$ exactly and is fixed thereafter. Published maximum-one rescaling returns the unnormalized indicators. More generally write $V=HO$, set $U=OR_0^\top$, and assume the **one-step basin condition**
   $$
   \max_{i\ne j}\frac{|U_{ij}|}{\sqrt{n_i}}<\mu<\min_j\frac{|U_{jj}|}{\sqrt{n_j}}.
   $$
   Then one step from $R_0$ returns $HD_\pm$, $D_\pm=\operatorname{diag}(\operatorname{sign}U_{jj})$, and remains fixed. (The statement is up to relabelling the clusters: a start aligned with a permuted indicator basis returns the correspondingly permuted signed indicators.) For $r=1$, the maximum over the empty index set is taken as zero.
2. Exact grouping of equal rows of $V$ recovers the partition and supplies such an initialization, without labels.
3. The global maximizers of $\sum_{x,j}(VR^\top)_{xj}^4$ are precisely the signed indicator bases. Unnormalized orthogonal varimax has these same maximizers.
4. The span assumption and threshold condition **alone do not imply arbitrary-initialization SEBA recovery**.

*Proof.* Write $V=HO$ with $O$ orthogonal and choose $R=O$. Thresholding $H_j$ multiplies it by $1-\mu\sqrt{n_j}>0$, canceled by normalization; $\operatorname{Polar}(H^\top V)=O$. Under the basin condition, thresholding removes every off-diagonal group contribution and retains only the signed diagonal contribution in each column, which normalization turns into $HD_\pm$. Its Procrustes update is $D_\pm O$, giving $VR^\top=HD_\pm$ thereafter. Rows within group $j$ equal $O_{j,:}/\sqrt{n_j}$, and different groups' rows are nonzero orthogonal vectors, proving the grouping claim.

For any orthogonal $U$, the quartic objective for $HU$ is $\sum_i n_i^{-1}\sum_jU_{ij}^4\le\sum_i n_i^{-1}$. Equality requires every row of $U$ to have just one nonzero entry, hence a signed permutation. Standard unnormalized varimax subtracts a constant when all column norms are one, giving the same maximizers; Kaiser-normalized variants are different objectives.

For the negative claim, take four equal-sized groups and $V=HO$, where $O$ is a normalized $4\times4$ Hadamard matrix with a row sign pattern chosen so no column is constant. Every entry of $V$ then has magnitude $1/\sqrt N$. At $R=I$ and $0<\mu<1/\sqrt N$, normalized thresholding returns $V$, and $\operatorname{Polar}(V^\top V)=I$. This fixed point is not an indicator basis and does not trigger the constant-column safeguard. It refutes universal recovery, although small generic perturbations can escape it. ∎

**Empirical failure beyond the exceptional fixed point.** On exact indicators of sizes $(20,5,3,2)$, 200 seeded Haar-rotated starts of the core at the published default $\mu=0.99/\sqrt{30}=0.180748$ recovered only **43/200** times (193/200 met rotation tolerance $10^{-13}$ within 20,000 iterations). The first failed converged output has rank three, loses the largest group, and duplicates the smallest indicator with opposite signs. Sign correction makes the duplicate identical and every column minimum zero: the published negativity ranking does not flag this failure. The same 200 starts at each $\mu=0.1,0.05,0.02$ all recovered. These are reproducible numerical outcomes, not a claim of recovery probability or success for every $\mu\le0.1$.

**What it buys.** The exact regime really contains a recoverable dictionary, including an explicit label-free initialization. It does not turn a nonconvex heuristic into a universally successful recovery method. Varimax's global characterization likewise says nothing about every local optimizer.

**Tractability and sampling.** Exact row grouping can use sorting in $O(Nr\log N)$; toleranced grouping under noise needs a separation rule. Observing every group of probability at least $p_{\min}$ is **ensured** by $N\ge p_{\min}^{-1}\log(r/\zeta)$ under independent sampling, since the union bound gives missing-group probability at most $r e^{-Np_{\min}}$. This is sufficient, not necessary. This is coverage only, not spectral accuracy; dependent token samples require separate treatment.

**Proposition S.8 (approximate indicators and candidate-relative loading bounds).** Let $V\in\mathbb R^{N\times r}$ be orthonormal, $H\in\mathbb R^{N\times r}$ the normalized-indicator matrix of a candidate partition into exactly $r$ nonempty sets, and $0<\mu<\min_jn_j^{-1/2}$. Set $\eta=\|(I-VV^\top)H\|_F$ and $c=\min_j(1-\mu\sqrt{n_j})>0$. There exists an orthogonal alignment with $\|VR^\top-H\|_F\le\sqrt2\eta$. If $\sqrt2\eta<c$, its normalized threshold step satisfies
$$
\|S-H\|_F\le\frac{2\sqrt2\eta}{c}.
$$
For any actually returned candidate $S$, independently of its initialization,
$$
\|F^\top(S-H)\|_F\le\|F\|_2\|S-H\|_F.
$$
The existence bound is not a guarantee for every SEBA run.

*Proof.* If $q_i\in[0,1]$ are the singular values of $V^\top H$, orthogonal Procrustes gives squared alignment error $2\sum_i(1-q_i)\le2\sum_i(1-q_i^2)=2\eta^2$. Soft-thresholding is coordinatewise 1-Lipschitz. Its value on $H_j$ is $c_jH_j$, where $c_j\ge c$. The column error before normalization is at most $e_j=\|(VR^\top-H)_j\|_2<c_j$. For nonzero $y$ and unit $h$, $\|y/\|y\|-h\|\le2\|y-c_jh\|/c_j$, by the triangle and reverse-triangle inequalities. Sum the squared column bounds. The loading bound is submultiplicativity. ∎

**What it buys.** Subspace closeness supplies a quantitative sparse-approximation benchmark, and verified vector errors propagate to activation-space loadings. Davis–Kahan can control a separately bounded block perturbation; it does not establish that the candidate is a population concept or remove localization/matching error.

**Tractability and sampling.** Given $H$, principal-angle/alignment computations cost $O(Nr^2+r^3)$ and loadings cost $O(Ndr)$. Searching over *all* candidate partitions can be combinatorial. Report achieved residuals, rank, negativity, and resampling stability; no universal sample count follows from the finite-site bound.

## 4. Hierarchy and persistence without a hidden filtration

Coifman–Maggioni (2006) diffusion wavelets apply localized, rank-revealing compression to dyadic powers $P^{2^j}$. A declared compression tolerance makes scaling spaces smaller; exact powers of an invertible matrix do not. Retained dimensions chart how long blocks survive diffusion. Localized QR scaling functions are not automatically indicators, and their counts need not equal ideal SVD ranks.

**Definition S.9 (gap-persistence region).** For rank $r$ **including the constant**, $1\le r<N$, and threshold $\tau>0$, define
$$
g_r(\beta,t)=\lambda_{r-1}(\beta)^t-\lambda_r(\beta)^t,\quad
\mathcal G_r(\tau)=\lbrace(\beta,t):\beta\ge0,\ t>0,\ g_r(\beta,t)>\tau\rbrace.
$$
This records a rank cut, not automatically the identity of one moving block.

**Theorem S.10 (well-defined diffusion-time persistence).** For fixed finite normalized views, $g_r$ is continuous in $\beta$ and $t>0$. At fixed $\beta$, its strict superlevel set in $t$ is an interval, possibly empty or unbounded. The gap itself need not be monotone. For $0<\theta<1$, the ideal retained dimension
$$
n_j(\theta)=\#\lbrace i:\lambda_i^{2^j}>\theta\rbrace
$$
is nonincreasing in $j$ and is the minimum rank of an approximation to $P^{2^j}$ with spectral-norm error at most $\theta$.

*Proof.* Write $s=\log v$. The Sinkhorn equations are $s_i+\log\sum_jK_{ij}(\beta)e^{s_j}=0$. Their Jacobian in $s$ at the solution is $I+P$, invertible since $P$ is PSD. The implicit-function theorem and uniqueness of positive scaling give local smoothness in $\beta$; equivalently one may use Theorems A.23 and A.25 of `THEOREMS.md`. Weyl's inequality gives eigenvalue continuity, and $(a,t)\mapsto a^t$ is continuous for $a\in[0,1]$, $t>0$.

Fix $a=\lambda_{r-1}\ge b=\lambda_r$. For $0<b<a<1$, $a^t-b^t$ increases and then decreases, with its unique maximum at $t=\log(\log b/\log a)/\log(a/b)$, by differentiating. If $a=1$ it is nondecreasing; if $b=0$ it is nonincreasing; if $a=b$ it vanishes. These include the constant endpoint cases and prove the interval claim. Finally each $\lambda_i^{2^j}$ is nonincreasing in $j$, and diagonalization plus the spectral-norm Eckart–Young theorem gives the exact rank characterization. ∎

**What it buys.** Persistence in diffusion time is a precise stability-of-scale question. Persistence in $\beta$ is not asserted monotone and can have multiple components; a persistence diagram would need an additional declared filtration, exactly as THEORY §10 warns.

**Tractability and sampling.** Once eigenvalues are known, time scans cost $O(N)$ per time, or $O(r)$ for a retained spectrum. Temperature scans require rebuilding/renormalizing operators; crossings complicate block matching. Diffusion-wavelet speedups depend on compressibility and local support, not a universal near-linear bound for dense data. Numerical and sampling errors create a margin around the chosen gap threshold.

At fixed $(\beta,t)$, single linkage on $\|P^t(e_x-e_y)\|$ or Ward merging on the corresponding Euclidean diffusion coordinates gives a dendrogram after declaring tie-breaking. Neither tree inherits Cheeger bounds merely from its distance. An alternative is recursive spectral bipartition, applying S.3 separately within each current graph and specifying whether outgoing mass is removed or converted to holding probability. Its node-wise cut bounds refer to those modified graphs, not automatically to a globally optimal or temperature-persistent hierarchy.

## 5. What is certified, and what is refused

The dictionary theorems B.21–B.23 of `THEOREMS.md` should be read with its named regime assumptions, not as completeness for every overlapping sparse representation. Given a proposed $H$, zero $\eta$ certifies exact indicator-subspace equality, and small $\eta$ certifies the approximate alignment in S.8. This is a **sound candidate-relative test, not a complete discovery procedure**: nonconvex candidate search can miss the regime, as S.7 demonstrates. Gaps and block IPR alone are not sound-and-complete indicators of concepts. For equal-sized partition blocks, $\Pi[x,x]=r/N$, so the block IPR is $1/N$ despite perfect indicator recovery.

An SAE learns a dictionary by fitting another model. Here $a_F(H_j)=\sqrt{n_j}\,\bar F_{A_j}$ is a measured loading; S.8 bounds its discrepancy from a specified candidate loading. A population-concept error bar would additionally require a correct population target, a sampling/operator-error bound, and gap-controlled spectral perturbation; none is supplied by SEBA. Failure of a search need not mean absence of sparse structure. The protection against an optimization instructed to produce sparsity comes from independently checking subspace distance and cuts, not from the sparse output itself.

## Numerical verification

Executed `uv run --with numpy python -I checks/sparsity_check.py` from the repository root. The planted operator has four groups of three sites, $P=0.9HH^\top+0.1\mathbf1\mathbf1^\top/12$; it is exactly the sphere-kernel construction for repeated orthogonal group directions at $\beta=\log37$.

- Exhaustive two-way cuts: $\lambda_1=0.9$, lower bound $0.05$, exact $h=0.05$, upper bound $0.447213595500$.
- Four planted disjoint sets: maximum conductance $0.075$, higher-order converse lower bound $0.05$.
- Indicator-aligned SEBA at $\mu=0.1$: Frobenius error $3.846\times10^{-16}$; minimum indicator height $0.577350269190$.
- Nonconstant Hadamard initialization: fixed-point residual $7.692\times10^{-16}$, distance from every signed/permuted indicator basis $2.000000000000$.
- Noisy subspace: $\eta=0.010275560863$, thresholded error $0.006756789372$, S.8 bound $0.035152217789$.
- $Q=P^2$, equal half-site pair: transport $0.905$, centered score $0.81=\sigma_1$.
- Added assertions for loop-free eigenvalue/conductance conversion, the nearby-start basin, collapsed-column Procrustes, and the 200-start unequal-size $\mu$ sweep described above.

All assertions passed. These are identity/planted checks, not statistical evidence about trained models.

## Discrepancies with THEORY.md

1. §4.1's “balanced relaxation” needs an explicit normalization and distinction between equal pair sizes and macroscopic balance; S.4 supplies both the formula and the limitation.
2. §4.3's stated IPR range $[1/N,1]$ is a valid loose bound, but for a rank-$r$ projector the sharp upper bound is $1/r$, since $\Pi[x,x]\le1$. One-site support is possible only at rank one. Equal-sized partition blocks can have the minimum IPR, so it is not an indicator-recovery certificate.
3. §4.3 rightly treats rotation recovery as tested. No stronger unconditional SEBA-recovery or complete-diagnostic claim follows; the explicit counterexample above should accompany any proposed strengthening.
4. §14's “no training” description should distinguish the SVD measurement from SEBA/varimax's optional nonconvex postprocessing. No auxiliary model is learned, but optimization is still performed.

No changes were made to THEORY.md.

## Sources

- Sinkhorn and Knopp (1967), “Concerning nonnegative matrices and doubly stochastic matrices,” *Pacific Journal of Mathematics* 21(2), 343–348, total-support scaling theorem.
- Csima, J. and Datta, B. N. (1972), “The DAD theorem for symmetric non-negative matrices,” *Journal of Combinatorial Theory, Series A* 12(1), 147–152, symmetric scaling.
- Levin, D. A. and Peres, Y. (2017), *Markov Chains and Mixing Times*, 2nd ed., AMS, Theorem 13.10 (Cheeger for reversible chains).

- Spielman and Teng (2011), “Spectral Sparsification of Graphs,” *SIAM Journal on Computing* 40(4), 981–1025, [doi:10.1137/08074489X](https://doi.org/10.1137/08074489X).
- Spielman and Srivastava (2011), “Graph Sparsification by Effective Resistances,” *SIAM Journal on Computing* 40(6), 1913–1926, Theorem 1 and resistance-sketch construction, [arXiv:0803.0929](https://arxiv.org/abs/0803.0929).
- Batson, Spielman and Srivastava (2012), “Twice-Ramanujan Sparsifiers,” *SIAM Journal on Computing* 41(6), 1704–1721, Theorem 1.1, [arXiv:0808.0163](https://arxiv.org/abs/0808.0163).
- Lee, Oveis Gharan and Trevisan (2014), “Multiway Spectral Partitioning and Higher-Order Cheeger Inequalities,” *JACM* 61(6), Article 37, Theorem 1.1, [doi:10.1145/2665063](https://doi.org/10.1145/2665063).
- Froyland (2013), “An analytic framework for identifying finite-time coherent sets in time-dependent dynamical systems,” *Physica D* 250, 1–19, Assumptions 1–2, Propositions 1–2, Theorem 2 equation (8), [paper](https://web.maths.unsw.edu.au/~froyland/froyland_analytic_coherent_sets.pdf).
- Froyland and Padberg (2009), “Almost-invariant sets and invariant manifolds—connecting probabilistic and geometric descriptions of coherent structures in flows,” *Physica D* 238, 1507–1523, [doi:10.1016/j.physd.2009.03.002](https://doi.org/10.1016/j.physd.2009.03.002) (context, not a proof dependency).
- Froyland (2015), “Dynamic isoperimetry and the geometry of Lagrangian coherent structures,” *Nonlinearity* 28, 3587–3622, Theorem 3.2, [arXiv:1411.7186](https://arxiv.org/abs/1411.7186) (comparison, not invoked under our hypotheses).
- Froyland, Rock and Sakellariou (2019), “Sparse eigenbasis approximation: Multiple feature extraction across spatiotemporal scales with application to coherent set identification,” *Communications in Nonlinear Science and Numerical Simulation* 77, 81–107, §3, equation (1), Algorithm 3.1, [published paper](https://web.maths.unsw.edu.au/~froyland/FRS18b.pdf).
- Coifman and Maggioni (2006), “Diffusion wavelets,” *Applied and Computational Harmonic Analysis* 21, 53–94, [doi:10.1016/j.acha.2006.04.004](https://doi.org/10.1016/j.acha.2006.04.004).
