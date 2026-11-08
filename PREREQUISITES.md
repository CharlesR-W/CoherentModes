# Mathematical prerequisites

What you need to read `THEORY.md` and judge whether it is right. Section numbers (§) refer to `THEORY.md`. This is a map for planning study, not a syllabus: for each topic it says what is used, where, and one or two places to learn it, with the part of each source that is enough.

## Where to start

**If you know linear algebra, probability and some ML** (the default route):

1. Read `REPORT.md` §2 (about 15 minutes). It gives the whole construction with a four-site example; note which terms are unfamiliar.
2. Use the readiness checklist below as the gate, and consult the *before the core* sources only to fill gaps it reveals. For most readers that means von Luxburg's spectral clustering tutorial (§1–4, 6 and 7 are enough; §7 is the perturbation view) and Yu, Wang and Samworth on Davis–Kahan (Theorem 2, with Corollary 1 as the single-eigenvector case; theorem numbers are those of the published Biometrika version).
3. Read Froyland (2013) far enough to see its main construction: normalize a transfer operator, take its SVD, and read coherent sets off the leading singular vectors.
4. Read `THEORY.md` §1–5 and §7.1–7.2 (the core), looking up the *when you reach it* topics as they come.
5. Later, for context in sequence models (§6): Hardoon et al. on CCA, the VAMP score in Wu and Noé, and the Hankel/predictive-state item.

**You are ready for the core when you can:** explain why the SVD of a matrix gives its best rank-$r$ approximation and what the discarded singular values cost; state Davis–Kahan in words and say why an eigenvector inside a cluster of nearly equal eigenvalues is not meaningful while the span of the cluster is; take a small kernel matrix, turn it into a random walk, and say why its top eigenvectors approximate cluster indicators when clusters are well separated; and say what a doubly stochastic matrix's stationary distribution is. Go deeper only when a specific claim in `THEORY.md` needs checking.

**Dependencies**, where they matter (a suggested order, not a rule that each item needs the previous one in full):

- SVD and symmetric eigendecomposition → perturbation of subspaces.
- Finite Markov chains and PSD kernels → diffusion maps → coherent sets (SVDs of products of normalized walks). Sinkhorn normalization is a detail of the middle step and can be learned when reached.
- Covariance and ridge regression → CCA → kernel CCA and VAMP.
- Separately: finite-state probability and matrix rank → HMM Hankel matrices and predictive states. The two context branches meet in §6 but neither is needed for the other.

**Other backgrounds.** Dynamical systems: you already have the coherent-set and Koopman side; add von Luxburg and Hardoon et al. and go straight to `THEORY.md`. Pure mathematics: the linear algebra will be quick; the unfamiliar parts are likely the statistics (ridge, held-out selection, dependent data) and the interpretability context at the end.

## Before the core

**Linear algebra with the SVD as a tool.** The SVD and its two readings (rank-one expansion, best low-rank approximation); spectral theorem for symmetric matrices; orthogonal projectors; Frobenius inner product and norm; the centering matrix $C = I - \mathbf 1\mathbf 1^\top/N$; principal angles between subspaces. Used in §1.3 (the identity every expansion rests on), §3–5, §7.2, §8.
*Learn:* Trefethen and Bau, *Numerical Linear Algebra*, Lectures 1–5. Principal angles: Golub and Van Loan, *Matrix Computations*, 4th ed., §6.4.3.

**Perturbation of eigenvectors and singular vectors.** Davis–Kahan and Wedin: an invariant subspace moves by at most about (perturbation size) / (spectral gap). For a first pass Davis–Kahan suffices; Wedin is the analogue for singular vectors. This is why the theory reports *blocks* of modes delimited by gaps and never trusts individual eigenvectors inside a near-degenerate block. Used in §0, §3.2, §8.
*Learn:* Yu, Wang and Samworth, "A useful variant of the Davis–Kahan theorem for statisticians" (Biometrika 2015; [published version](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf)), short and readable. For the full picture, Stewart and Sun, *Matrix Perturbation Theory*, or Bhatia, *Matrix Analysis*, Ch. VII.

**Finite Markov chains.** Stochastic and doubly stochastic matrices; stationary distribution; reversibility, and why a reversible chain is self-adjoint in a weighted inner product; eigenvalues, spectral gap, and "slow" functions as those one step barely changes. The operators in §3–4 are random walks on sites.
*Learn:* Levin and Peres (with Wilmer), *Markov Chains and Mixing Times*, 2nd ed., Ch. 1 and Ch. 12 (eigenvalues).

**PSD kernels and Gram matrices.** Positive semidefinite kernels; why $\exp(\beta\langle u,v\rangle)$ is PSD (a power series with non-negative coefficients in a PSD kernel, by the Schur product theorem); on the unit sphere it equals a Gaussian in chordal distance up to a constant; kernel ridge regression and its dual ("kernel trick") form. Used in §2, §5, §6.
*Learn:* Schölkopf and Smola, *Learning with Kernels*, Ch. 1–2 and 4.

**Diffusion maps and spectral clustering.** Building a random walk from a kernel; why its top eigenvectors give coordinates and, when clusters are well separated, nearly span the cluster indicators; diffusion time $t$ as smooth coarse-graining; the $\alpha$-normalization family and its continuum limit (a Laplace-type operator on the data manifold, with density dependence). Used in §2–3, §10, §13. The closest single body of prior work.
*Learn:* von Luxburg, "A tutorial on spectral clustering" (Statistics and Computing 2007); then Coifman and Lafon, "Diffusion maps" (Applied and Computational Harmonic Analysis 2006).

**Statistics for held-out estimates.** Covariance and PCA; ridge regression in closed form; held-out evaluation, and cross-fitting so that whatever is fitted is not evaluated on the data it was fitted on; grouped splits and resampling for dependent observations (token sites in one document are not independent, so splits and bootstraps are by document). Used for readability scores (§5), the context ladder (§6.4) and all error bars (§13).
*Learn:* Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, Ch. 3 and 7; for dependent data, Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" (Ecography 2017).

## When you reach it

**Sinkhorn scaling (§3.1, §4.1).** A symmetric matrix with positive entries can be scaled uniquely as $DKD$ to be doubly stochastic, and the alternating iteration converges. The theory uses this instead of the usual degree normalization so that every layer's walk has the same (uniform) stationary distribution and products of layers stay comparable.
*Learn:* Knight, "The Sinkhorn–Knopp algorithm: convergence and applications" (SIAM J. Matrix Anal. Appl. 2008); Marshall and Coifman, "Manifold learning with bi-stochastic kernels" (arXiv:1711.06711) for the diffusion-maps connection.

**Transfer operators and finite-time coherent sets (§4).** Koopman and Perron–Frobenius operators as linear operators on functions of the state; Froyland's construction: normalize, compose forward, take the SVD, read off sets that stay together over a time window. §4 is this construction with layer depth in place of time and the identity as transport.
*Learn:* Froyland, "An analytic framework for identifying finite-time coherent sets in time-dependent dynamical systems" (Physica D 2013, arXiv:1210.7418). Koopman background, optional: Brunton, Budišić, Kaiser and Kutz, "Modern Koopman theory for dynamical systems" (SIAM Review 2022).

**Canonical correlation, kernel CCA and VAMP (§6.3–6.4).** CCA as the SVD of a whitened cross-covariance; the regularized kernel version in dual form (smoothers $R = \tilde K(\tilde K + N\rho I)^{-1}$, with squared canonical correlations the eigenvalues of $R_pR_q$); the VAMP-2 score (sum of squared canonical correlations) used to choose the ridge and window.
*Learn:* Hardoon, Szedmak and Shawe-Taylor, "Canonical correlation analysis: an overview with application to learning methods" (Neural Computation 2004); Wu and Noé, "Variational approach for learning Markov processes from time series data" (J. Nonlinear Science 2020, arXiv:1707.04659).

**The bridge paper (§6.2–6.3, §7.1).** Pughe-Sanford et al., "Neurons as detectors of coherent sets in sensory dynamics" (NeurIPS 2025, arXiv:2510.26955) already reads neurons as thresholded singular functions of a transfer operator, estimated by past–future CCA on lag vectors. It joins the coherent-set and CCA items above and is the direct precedent for the "carried" context operator; read it after both.

**Hankel matrices and predictive states (§6.3, Stage 4 of `EXPERIMENTS.md`).** The past-by-future probability matrix of a process. For an HMM with $k$ hidden states its rank is at most $k$, so the rank is a lower bound on the number of hidden states; under realization conditions it equals the dimension of a minimal linear predictive state. It does not bound arbitrary nonlinear state, and the theory claims no rank bound for activation streams. HMM belief states (posteriors over hidden states) live on a simplex.
*Learn:* Hsu, Kakade and Zhang, "A spectral algorithm for learning hidden Markov models" (JCSS 2012, arXiv:0811.4413), up to the algorithm and its observable-operator form; Shai et al., "Transformers represent belief state geometry in their residual stream" (NeurIPS 2024).

**Backpropagation as linear algebra (§7.2–7.3).** A weight gradient as a sum over sites of outer products (backpropagated error × layer input), which is what lets it decompose by mode; Jacobian-vector products by forward-mode autodiff; random-probe estimates of inner products of Jacobians.
*Learn:* Baydin, Pearlmutter, Radul and Siskind, "Automatic differentiation in machine learning: a survey" (JMLR 2018), the sections on forward and reverse mode.

**Information geometry, one fact (§2, §8).** Under $p \mapsto \sqrt p$ the probability simplex maps onto part of a sphere, Fisher–Rao distance is twice the angle there, and the Bhattacharyya coefficient $\sum_k\sqrt{p_kq_k}$ is the dot product. That is all the output view needs.
*Learn:* Amari, *Information Geometry and Its Applications* (Springer 2016), §3.5–3.6.

## Optional background

- **Delay embeddings and HAVOK (§4.4).** A different, trajectory-matrix use of Hankel structure, mentioned as a separate lagged-evolution fit: Brunton et al., "Chaos as an intermittently forced linear system" (Nature Communications 2017).
- **Representation similarity and stitching (§8, Stage 5).** RSA (Kriegeskorte, Mur and Bandettini 2008); CKA (Kornblith et al., "Similarity of neural network representations revisited", ICML 2019); stitching (Bansal, Nakkiran and Barak, "Revisiting model stitching to compare neural representations", NeurIPS 2021).
- **Interpretability context (§11, Stages 1–3).** Elhage et al., "Toy models of superposition" (2022); Bricken et al., "Towards monosemanticity" (2023) for sparse autoencoders; Nanda et al., "Progress measures for grokking via mechanistic interpretability" (ICLR 2023). These supply the known answers the experiments test against.
- **Scaling (§13).** Nyström approximation (Williams and Seeger, NIPS 2000); Lanczos / ARPACK; randomized SVD (Halko, Martinsson and Tropp, SIAM Review 2011).
- **Convergence of graph Laplacians (§13).** Belkin and Niyogi; Singer, "From graph to manifold Laplacian: the convergence rate" (2006). Only for the [cond] rates quoted; the theory relies on resampling for error bars, not on these rates.
- **Localization inside a subspace (§4.3, §5).** Varimax rotation (from factor analysis) and the inverse participation ratio.

## A readiness exercise

`checks/cm_check.py` checks seven of the theory's identities and examples numerically in under sixty lines of numpy. Explaining why each assertion should hold is a good test of whether the core will read smoothly; it is not a test of the whole theory.
