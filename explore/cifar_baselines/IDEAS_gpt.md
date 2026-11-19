# Independent proposal: characterising neural coherent sets

**Status:** proposed design, not an implementation protocol or a claim that one method should win. Prepared 2026-11-15 after reading `HANDOFF.md`, the requested sections of `explore/cifar_coherent/SPEC.md`, `coherent.py`, `sweep_math.py`, `seba.py`, and the relevant measurement routines in `measure_v2.py`. No changes to the existing construction are proposed as settled decisions.

## 1. What I would try to learn first

The useful output is a map of equivalences, differences, and failure regimes—not a leaderboard of class agreement. In particular:

1. **How much is already ordinary covariance geometry?** Compare the cosine-Gram eigenspace and direct clustering with the graph eigenspace, before comparing their hard sets.
2. **How much comes from using several aligned views?** Compare against concatenated-view geometry and single-cell methods, then destroy cross-cell image correspondence while preserving every cell individually.
3. **How much comes from the set extractor?** Apply exactly the same SEBA procedure to PCA subspaces and random subspaces. Compare SEBA with clustering the same spectral embedding.
4. **How much is generic high-dimensional sampling geometry?** Use a hierarchy of isotropic, spectrum-matched, cross-view-covariance-matched, and graph-conditional nulls. Each preserves different facts; none is a universal null.
5. **What changes when the estimand changes?** Distinguish average smoothness, sequential diffusion, and density-weighted predictability between two representations. They answer different questions.

My first deliverable would be eight extra colour sources/control families (B1–B8 below), plus a compact comparison table on a predeclared selection of cells and blocks. I would not immediately rerun the full explorer over a large factorial sweep.

### 1.1 Notation and fairness conventions

For cell \(\ell\), let \(X_\ell\in\mathbb R^{N\times d_\ell}\) be the flattened activations. Let \(Z_\ell\) denote column-centred activations subsequently normalised to unit row length, exactly as in the code. Set

\[
C_\ell=Z_\ell Z_\ell^T,\quad H=I-\mathbf1\mathbf1^T/N,\quad
P_\ell=D_\ell K_\ell D_\ell,\quad M_\ell=(I+P_\ell)/2.
\]

Here the positive diagonal scaling makes \(P_\ell\) doubly stochastic; \(K_{ii}=0\). For block \(B\), \(L=|B|\), \(\bar M_B=L^{-1}\sum_{\ell\in B}M_\ell\). The code's SEBA dimension \(r\) **includes the constant**, whereas subspace comparisons below use \(q=r-1\) nonconstant directions. The number of hard clusters \(k_c\) is a separate variable; initially use \(k_c=r\), not \(r-1\).

**A subtle PCA distinction:** centring before row normalisation does not ensure that the normalised rows have zero mean. Ordinary PCA of \(Z\) diagonalises \(HCH\); the existing saved “PCs” in `measure_v2.py` diagonalise \(C\). Show the centred PCA baseline, and check the latter convention as an inexpensive companion. Do not silently call the two identical.

Use the same probe IDs, block, rank, and input view for a comparison. Keep raw-feature or global-average-pooling variants separately labelled; a flattened convolutional view and a channel-average view are not interchangeable. For multi-cell baselines, standardise each cell's contribution **before** combination. Do not choose hyperparameters by CIFAR class agreement.

Commitment labels in this document:

- **S:** a label-free, dot-product-derived spectral construction; compatible with the owner's construction commitments.
- **P:** a diagnostic/postprocessing step or a null generator, not a proposed canonical construction. SEBA's sparsity preference is an extraction device, not an assumption that the data are sparse.
- **B:** a deliberately labelled fitted baseline outside the strictly spectral construction. Iterative k-means and sparse-PCA optimisation belong here, even though they are unsupervised.
- **V:** validation using labels. Never feeds back into a kernel, basis, rank, or bandwidth choice.

Averages and products below use image identity as the coupling between views. They do not require equal activation dimensions or a cross-layer dot product.

### 1.2 Cost vocabulary and scope

Use \(N=2000\), approximately 100 cells and 100 **selected** blocks, \(r\leq 9\), and initially one kernel family. The current full explorer actually has 96 cells and 1,056 cell-indexed layer/time blocks before multiplying by five kernel rules; do not confuse that with the requested 100-block design budget.

Useful units, excluding model inference:

- One dense \(N\times N\) matrix is 4 million entries: 32 MB float64 or 16 MB float32. A 100-cell float64 bank is about 3.2 GB per matrix type; streaming is important on this VPS.
- **E:** one dense eigendecomposition/SVD, \(O(N^3)\), with \(N^3=8\times10^9\). A 100-block sweep means about 100 E, not 100 cheap vector operations. Iterative top-spectrum solvers can reduce this to \(O(mN^2)\) for \(m\) total matrix-vector products, but must be residual-checked and can struggle at small gaps.
- **G:** building a full Gram from features costs \(O(N^2d_\ell)\). Reuse the cached `gram_<step>_<site>.npy` where available; top-50 saved PCs cannot reconstruct the full Gram or its small-scale neighbourhoods.
- **Sbal:** one balancing calculation. Ordinary iterations cost \(O(tN^2)\); the existing Newton-accelerated routine can additionally perform dense linear solves, \(O(N^3)\) each. Do not budget all Sinkhorn runs as inexpensive matrix-vector iterations.
- **SEBA:** \(O(t_sNr^2+t_sr^3)\) per rank, where \(t_s\) is actual iterations; code permits 5,000. Usually cheaper than E, but record convergence, runtime, and restarts.
- Clustering a \(p\)-dimensional embedding with \(k_c\) centroids, \(J\) iterations and \(R\) starts costs \(O(RJNpk_c)\). For \(p=50,k_c=8,J=50,R=10\), this is roughly \(4\times10^8\) elementary coordinate-distance terms per case. Full flattened-feature clustering costs more; do not conceal it by calling a truncated-PCA variant “direct k-means.”

Costs below state one replicate. Generative nulls multiplied by 20–50 replicates over all 200 cases are a large compute job. Use 8–12 representative blocks and cells for initial null calibration, then expand only informative contrasts. Run long/GPU work on Modal; no pipeline is built by this proposal. Wall times and monetary costs need a short measured pilot rather than invented estimates.

## 2. Prioritised baselines and nulls

### B1 — Highest priority: cosine PCA / classical MDS, with the same extractors

**Definition.** For each cell diagonalise \(G_\ell=HC_\ell H\). For a block use

\[
G_B=H\left(L^{-1}\sum_{\ell\in B}C_\ell\right)H
=HZ_BZ_B^TH,\qquad
Z_B=L^{-1/2}[Z_{\ell_1},\ldots,Z_{\ell_L}].
\]

Use the top \(q\) eigenvectors \(U_q\) as a spectral colour source. Run SEBA on \([\mathbf1/\sqrt N,U_q]\), exactly the same \(r\), threshold, and display conventions as the current pipeline. As a separate baseline cluster ordinary PCA **scores** \(U_p\Lambda_p^{1/2}\), with \(p\in\{q,20,50\}\) fixed in advance. Whitened scores \(U_p\) are a distinct variant, not ordinary PCA scores.

**Question/trivial explanation.** Are the purported sets just the leading covariance directions, made visually discrete by SEBA? Does graph processing add anything beyond averaging raw dot-product geometry? The block Gram is also the exact concatenated-view linear-kernel baseline, so a block-method advantage over one cell alone is not enough.

**Compare.** Principal-angle overlap of the nonconstant graph and Gram subspaces; held-out reconstruction of \(G_B\); SEBA-set AMI/ARI and size distributions; each method's cross-view smoothness profile; class validation relative to permutation chance. Compare SEBA-to-SEBA before drawing conclusions about the graph.

**Cost.** Up to 100 cell E + 100 block E; reuse Grams and block prefix sums. Extra SEBA and small-embedding clustering. No new G if cached. **Status:** eigenspaces **S**, SEBA **P**, k-means on scores **B**.

### B2 — Highest priority: direct spherical k-means, including concatenated views

**Definition.** For unit rows \(z_i\), minimise

\[
\sum_i(1-z_i^Tc_{a_i}),\qquad \|c_j\|=1,
\]

with cosine assignment and unit-normalised cluster sums. Use \(k_c\in\{2,3,4,6,8\}\), a fixed restart budget, and no class-based selection. For a block use \(Z_B\) above, whose rows are already unit length. Also include ordinary Euclidean k-means on the same unit rows as a small companion; its unconstrained centroids make it a different objective.

A full-Gram implementation can implement exact spherical clustering without storing concatenated feature matrices: for membership vector \(h_j\), score point \(i\) by \((C_Bh_j)_i/(h_j^TC_Bh_j)^{1/2}\). This is exact for the block cosine geometry, not a top-PC approximation.

**Question.** Is a nonlinear graph and spectral relaxation unnecessary to recover the same grouping? Does multi-view concatenation account for the apparent benefit of “dynamic” sets?

**Compare.** Set agreement and size/coverage; per-cell graph leakage on independently constructed evaluation graphs; cosine within/between separation, which favours k-means in-sample and therefore must not be the sole score; resampling stability; class agreement against size-matched chance. For a genuinely different answer, show exemplars of disagreements, not only a scalar ranking.

**Cost.** Full features: \(O(RJNk_c\sum_{\ell\in B}d_\ell)\) per block. Exact Gram route: roughly \(O(RJN^2k_c)\) per case; 200 full cases can be expensive. First run the exact baseline on selected blocks, then consider a declared 50-PC approximation with a fidelity audit. **Status:** **B**; iterative fitted centroids are not outputs of the spectral construction.

### B3 — Highest priority: single-cell and simple consensus baselines

**Definition.** Paint every panel using the selected block's first-cell, midpoint-cell, and last-cell spectral sets separately. No new operator is needed. Add a block consensus from single-cell partitions:

\[
A_B(i,j)=L^{-1}\sum_{\ell\in B}\mathbf1\{a_i^{(\ell)}=a_j^{(\ell)}\geq0\}.
\]

For transparency, report missing assignments; do not count two unassigned points as a common cluster. Diagonalise \(HA_BH\), then apply the same set extractor. Start with single-cell SEBA partitions so the baseline does not import k-means; optionally repeat with spherical k-means partitions.

A continuous consensus companion is the averaged projector \(L^{-1}\sum U_{\ell,q}U_{\ell,q}^T\), followed by an eigendecomposition. It tests whether only each cell's leading span matters, rather than the full spectrum.

**Question.** Does the block add information beyond one convenient cell? Is averaging full operators necessary, or does elementary agreement voting suffice? “Best cell” chosen retrospectively is an oracle upper envelope and must be labelled as such, not used as an ordinary competing estimator.

**Compare.** Agreement with block sets; worst-cell and average leakage; successor-cell transfer; stable shared structure in the weak-signal benchmark. Use fixed endpoint choices first, not whichever cell maximises a target after inspection.

**Cost.** Endpoint colourings essentially free. Building consensus costs \(O(LN^2)\) per block if dense, plus 100 E. Projector consensus can use \(N\times Lq\) factors instead. **Status:** spectra **S**, hard-set consensus **P**, k-means source partitions **B**. Consensus deliberately discards information and need not be a replacement construction.

### B4 — Highest priority: break cross-cell image correspondence, preserve each cell exactly

**Definition.** Independently sample permutations \(\Pi_\ell\) for the cells in a block, fixing one reference permutation to identity, and replace

\[
C_\ell\mapsto\Pi_\ell C_\ell\Pi_\ell^T,
\quad P_\ell\mapsto\Pi_\ell P_\ell\Pi_\ell^T.
\]

The second route is exact and avoids rebuilding/balancing. Use the same permutations across all methods in a replicate. A **single common permutation** of all cells is an invariance test, not a cross-cell null: every result should simply relabel.

**Question.** Would alignment of arbitrary high-dimensional cells produce equally impressive block spectra and sets? This null preserves each cell's spectrum, graph structure, set sizes, localisation, and pairwise distances, but removes the fact that row \(i\) is the same image. It is conditional on all within-cell structure.

**Compare.** Real versus null nontrivial eigenvalues, eigengaps, subspace stability, per-cell retention profile, set coverage/reliability, and held-out aligned-view scores. Null class agreement with the reference cell's labels need not vanish when the reference itself carries strong class information; compare against its full empirical null distribution, not zero.

**Cost.** Permuting is \(O(100N^2)\); 100 block E per replicate, no Sbal or G. Initially 20 replicates on 8–12 blocks, rather than 2,000 full dense decompositions. **Status:** **P**, no labels.

### B5 — Highest priority: isotropic Gaussian null, with persistence as a separate factor

**Definition.** Generate \(X_\ell=G_\ell\), \(G_{ij}\overset{iid}{\sim}N(0,1)\), using actual widths where feasible, then perform exactly the real centring/row-normalisation/graph pipeline. Include

- independent \(G_\ell\) across cells;
- shared \(G\) across equal-width cells, transformed by orthogonal feature maps;
- \(G_\ell=\sqrt\rho G_0+\sqrt{1-\rho}E_\ell\) at a common width, \(\rho\in\{0,.5,.9,1\}\).

The latter two are **persistent geometry without planted groups**, not no-dependence nulls. Sweep \(d/N\) using \(d\in\{16,64,256,1024,4096\}\), plus representative actual widths; orthogonal invariance lets large isotropic feature factors be handled without retaining unnecessary rotations.

**Question.** Do kNN graphs and SEBA return plausible-looking, reasonably balanced sets on a distribution with no finite mixture or privileged partition? How do persistence and dimensional concentration separately affect apparent coherence?

**Compare.** Both spectral ends of \(P\), nontrivial \(M\) spectrum, gap, localisation, SEBA sizes/coverage, restart variability, and resampling stability. Assign independent pseudo-classes with the real class counts and report their AMI/Jaccard distributions. A forced \(r\)-way partition is expected even under this null; “found sets” is not evidence.

**Cost.** New G + Sbal for 100 cells and approximately 200 E per replicate if all cell and block spectra are requested. Actual convolutional widths can make G dominant. Use a small design grid and representative blocks, not the Cartesian product of every width, \(\rho\), rule, and block. **Status:** **P**, compatible geometry; pseudo-labels **V** only.

### B6 — Highest priority: spectrum-matched and cross-view second-order nulls

These answer different questions and should not be conflated.

**B6a: population-spectrum-matched Gaussian.** Let \(\hat\Sigma_\ell\) be the empirical covariance of the pre-row-normalised, centred activations, with eigenvalues \(\hat\lambda_j\). Generate

\[
X_\ell^{null}=G_\ell\operatorname{diag}(\sqrt{\hat\lambda_j}),
\]

then re-centre and row-normalise. Right feature rotations do not affect the Gram. A shrinkage estimate or top-rank-plus-isotropic-tail version is acceptable if explicitly specified. This matches an **estimated covariance spectrum**, not the realised sample spectrum after normalisation. It asks whether anisotropy, not discrete grouping or nonlinear geometry, accounts for the result.

**B6b: exact sample-spectrum random-orientation null.** For a centred Gram \(G=U\Lambda U^T\), draw a Haar orthonormal frame \(Q\) in \(\mathbf1^\perp\), form \(Y=Q\Lambda^{1/2}\), and then normalise its rows before recomputing the cosine graph. Before this final normalisation, \(YY^T\) has exactly the target nonzero eigenvalues; afterwards it generally does not. Show the mismatch. Alternatively analyse the unnormalised surrogate as a separate Gram-level null, not a purported exact cosine-spectrum match.

A stronger **joint second-order surrogate** concatenates scaled, centred cell features \(F=[F_1,\ldots,F_L]\), with SVD \(F=USV^T\), and uses \(F^*=QSV^T\), with one shared Haar frame \(Q\perp\mathbf1\). Then \((F^*)^TF^*=F^TF\): within- and cross-cell second moments are preserved exactly before row normalisation, but image-level higher-order structure is randomised. It is not an iid Gaussian sample. An iid Gaussian alternative draws shared Gaussian row scores against the same factor; covariance is then matched only in expectation.

For feasibility, first fit a joint Gaussian model to 20–50 PC scores per selected cell, with shrinkage, preserving observed cross-cell covariance of those scores. Report lost variance; omitted high-dimensional tails are not automatically negligible to kNN graphs.

**Question.** Is there information beyond covariance anisotropy and persistent linear cross-view dependence? Independent spectrum-matched cells are a weaker null than the joint construction and cannot answer that question alone.

**Compare.** Real versus surrogate localisation, graph spectrum, stability, block-over-single-cell advantage, set extraction behaviour, and higher-order discrepancies. Always verify the moments claimed to be matched, both before and after row normalisation.

**Cost.** Spectrum estimation up to 100 E or feature SVDs; surrogate generation \(O(Nd_{eff}^2)\) or \(O(N^2d_{eff})\) depending on factorisation; then B5's operator costs. Joint full-rank Haar generation/factor multiplication can itself be cubic. A joint 10-cell, 50-PC surrogate is much cheaper than the full flattened model, but changes the null. **Status:** **P**; label-free covariance estimation is not training another model on labels.

### B7 — Highest priority: SEBA on random subspaces, and basis-rotation sensitivity

**Definition.** Draw \(G\in\mathbb R^{N\times q}\) iid Gaussian; QR-factorise \(HG\) to obtain Haar \(U\perp\mathbf1\). Feed \([\mathbf1/\sqrt N,U]\) to the unchanged SEBA implementation at each \(r\). Optionally attach the real eigenvalue list to the random basis for a **spectral-display null**: this preserves displayed eigenvalues but does not create a valid nonnegative Markov operator, and must not be treated as one.

Separately, for a real basis \(V\), replace it by \(VO\) for random orthogonal \(O\), preserving exactly its span, and rerun SEBA. The optimiser's objective is span-based, but its identity initial rotation and finite/nonconvex optimisation need not find the same solution. Include multiple initial rotations, not only the deterministic default.

**Question.** How much apparent sparsity, coverage, reliability, and balanced partitioning is manufactured by thresholding and argmax? Are particular displayed sets stable properties of a span or an optimiser/tie-breaking artefact?

**Compare.** Raw (pre-clipping) column minima, mass removed by clipping, support fraction, best-versus-run-to-run objective, hard-set agreement, overlap/margins, and participation. Random subspace overlap has expectation \(q/(N-1)\) for equal dimensions under this specific isotropic null; real-data nulls require separate calibration.

**Cost.** QR \(O(Nq^2)\) and SEBA only: inexpensive enough for 100–1,000 draws over the five ranks. Real basis-rotation checks cost about 10–20 SEBA runs per selected block, no E. **Status:** **P**. This null does not assert that real spectral subspaces are Haar.

### B8 — Highest priority: size-preserving random assignments and validation-label permutations

**Definition.** For each actual partition \(a\), randomly permute its assignment vector, preserving every set size and the unassigned count. For class validation, instead permute the class-label vector with its observed counts, holding the fitted representation and partition fixed. Repeat the *whole matching calculation*, including Hungarian matching, on every permutation.

If a reported statistic selects the best of many ranks/blocks, repeat that same maximisation in every null replicate. A single-test permutation interval does not calibrate an explorer-selected “best” finding.

**Question.** Are Jaccard, purity, apparent examples, or “some sets align with classes” mostly consequences of set sizes and searching many possibilities? This null conditions on an actual partition and does **not** explain where that partition came from.

**Compare.** Observed AMI/ARI, rectangular matched Jaccard, class entropy, and within/between validation quantities against their own size-preserving distributions. Purity alone is particularly misleading when cluster counts differ. For non-label quantities, assignment permutations test a different hypothesis from label permutations; name which was used.

**Cost.** About \(O(RN)\) per partition plus tiny contingency tables and matching problems; 1,000 permutations for roughly 200 selected partitions are feasible without eigensolvers. Additional ranks multiply this. **Status:** **P/V**, never kernel or basis selection.

### B9 — Second wave: degree-preserving graph rewiring

**Definition.** Starting from the pre-floor undirected union-kNN adjacency, run degree-preserving double-edge swaps, rejecting loops/multiple edges. Make enough swaps to diagnose decorrelation rather than relying on an unexplained fixed number. Preserve the degree sequence, add the same off-edge floor, and rerun the same balancing and laziness. Do not rewire the balanced matrix and claim degree preservation there.

**Question.** Are hubness and the degree distribution enough to account for the graph spectrum and sets, without meaningful neighbourhood topology? The resulting graph need not be realisable as a cosine kNN graph; this is a graph-conditional null, not an activation-generating model.

**Compare.** Real/null gaps, localisation, component statistics before floor, Sinkhorn scaling-factor spread, and SEBA behaviour. Run independent rewirings per cell to remove persistence; shared structural rewiring across cells would be a separate hypothesis.

**Cost.** Edge swaps roughly proportional to \(Nk\) times a mixing budget; Sbal + E per cell and E per block per replicate. Default floor gives a sparse-plus-low-rank representation in principle, but current code is dense. **Status:** **P**.

### B10 — Second wave: input geometry and the existing untrained network

**Definition.** Apply B1/B2 and the default graph construction to centred, row-normalised flattened images, and to the step-0 activations already measured. Compare trained block sets with these fixed colour sources. Optionally compare raw norm, brightness, colour summaries, and loss as validation covariates, not new basis inputs.

**Question.** Is the grouping inherited from input geometry or random-network architecture rather than acquired during training? Agreement is informative even if classes also agree. Step 0 of one random seed is not a distribution over architectures or initialisations.

**Compare.** Set/subspace agreement, conditional descriptive tables of class and simple image statistics, and paired evolution across training. Avoid interpreting “less agreement with pixels” as automatically better.

**Cost.** One new pixel G + Sbal + E; mostly reuse existing step-0 outputs. Independent random-network replicates require inference but no training and are a later extension. **Status:** spectral part **S**, fitted clustering **B**, image/class annotations **V/P**.

## 3. Prioritised construction ablations

### A1 — First: graph scale and graph family, with matched effective smoothing

Vary union-kNN \(k\in\{5,15,50,150\}\), mutual-kNN as a stress test, per-point perplexity \(p\in\{30,100,300\}\), and a modest global-temperature grid including the existing flat and pair-localised references. Add a locally scaled kernel only if the first contrasts remain ambiguous:

\[
K_{ij}=\exp[-(1-C_{ij})/\sqrt{\delta_{i,k}\delta_{j,k}}],\quad i\ne j,
\]

where \(\delta_{i,k}\) is the kth cosine distance. This is a specific dot-product-derived adaptive kernel, not an extra learned metric. State the convention; locally scaled Gaussian formulas are often written using Euclidean distances, not cosine distances.

**Measure:** graph components and degree distribution before the floor; final \(P\)-row entropy and effective neighbour count \(1/\sum_{j\ne i}P_{ij}^2\); nearest-neighbour mass; both spectral ends; localisation; subspace/set stability; evaluation leakage. Compare families both at their native parameter settings and at approximately matched median effective neighbour count **after** normalisation. Matching \(k=15\) to perplexity 15 does not match operators.

Separate union versus mutual symmetrisation from neighbourhood width. A global cosine kernel is not “density-independent”: fixed bandwidth interacts with sampling density. kNN adapts radii; diffusion normalisation changes limiting density effects; Sinkhorn fixes a stationary measure on sampled sites. These are three distinct issues.

### A2 — First: normalisation, explicitly changing the measure

Hold the nonnegative affinity \(K\) fixed. Let \(q_i=\sum_jK_{ij}\), \(K^{(\alpha)}_{ij}=K_{ij}/(q_iq_j)^\alpha\), \(d_i=\sum_jK^{(\alpha)}_{ij}\). Compare:

1. Symmetric Sinkhorn \(D K D\), uniform stationary mass on images.
2. Symmetric degree normalisation \(S_\alpha=\operatorname{diag}(d)^{-1/2}K^{(\alpha)}\operatorname{diag}(d)^{-1/2}\), \(\alpha\in\{0,1\}\), equivalent to a reversible random walk in its own stationary measure \(\pi_i\propto d_i\).
3. Raw adjacency eigenvectors, with scalar scale \(K/\|K\|_2\) only to make per-cell averaging comparable. This is **not** a Markov chain; do not report its Rayleigh quotients as retention probabilities.

**Measure:** stationary-vector drift between cells; degree/scaling correlation with leading modes; uniform-site and stationary-weighted subspace comparisons; local density sensitivity; null behaviour; set-size redistribution. For a single degree-normalised cell, nontrivial means orthogonal to \(\sqrt\pi\) in symmetric coordinates. For comparisons on common image space, map to function values and explicitly choose the evaluation measure.

Averaging separately degree-normalised symmetric matrices has no shared stationary vector. Dropping its leading eigenvector does not magically remove a known constant mode. Report its top vector and its constant overlap. Do not compose independently normalised symmetric matrices and interpret attenuation from stationary-measure mismatch as physical loss of coherence.

### A3 — First: average versus products versus two-endpoint products

Compare

\[
\bar M=L^{-1}\sum M_\ell,\quad
Q=M_b\cdots M_a,\quad Q_{end}=M_bM_a,
\]

using a declared column-vector convention. For products show singular values and both right/source and left/destination singular subspaces, not eigenvalues of a generally nonsymmetric product. Compare each separately against the averaged eigenspace.

Also test a common total diffusion budget: with \(L_\ell=I-P_\ell\), compare \(\prod_\ell\exp[-(\tau/L)L_\ell]\) with \(\exp[-(\tau/L)\sum_\ell L_\ell]\) on a small subset. This costs more, but removes the most obvious confounding between block length and total smoothing. A cheaper first-order product \(\prod_\ell[I-(\tau/L)L_\ell]\), \(\tau/L\le1/2\), is a clearly labelled approximation.

**Measure:** principal angles; singular decay with length; real-space source/destination assignment differences; per-cell leakage; endpoint versus all-cell sensitivity; commutator norm \(\|M_iM_j-M_jM_i\|_F\) as a descriptive noncommutativity diagnostic. A reverse ordering of symmetric factors gives exactly \(Q^T\), so it has **the same singular values** and swaps singular sides. Reverse-order singular-value equality is not evidence of order irrelevance. Other interior permutations can matter.

Short-step products and the average Laplacian agree to first order in smoothing; big agreement in that regime is expected, not independent corroboration. Through-training averages currently weight listed checkpoints equally although their spacings are very unequal. Equal-checkpoint weighting, training-step-interval weighting, and log-step weighting answer different questions; test the first two on selected blocks, and do not give layer depth a physical time scale by analogy.

### A4 — First: SEBA versus k-means on exactly the same spectral span

Keep \([\mathbf1/\sqrt N,U_q]\) fixed. Compare current SEBA; ordinary k-means on \(U_q\); and row-normalised spectral embedding followed by k-means (Ng, Jordan and Weiss, 2002). Report the distinction: row normalisation changes the embedding geometry and can magnify low-leverage rows.

Use \(k_c=r\), matched restart budgets, then a small cross-grid in \((q,k_c)\) to separate rank from cluster count. For SEBA vary the threshold \(\mu=c/\sqrt N\), \(c\in\{.5,.99,1.5\}\), and record failures/empty columns, unassigned mass and clipped negative mass. This is a postprocessing sensitivity, not evidence for a sparsity premise.

**Measure:** span held fixed but hard sets changing; raw SEBA reconstruction error; coverage, margins and reliability; set agreement and resampling stability; same-operator and independent-view leakage. Include B7's basis-rotation check. Do not compare only the best-looking restart of SEBA against one k-means run.

### A5 — First as an analytic control: laziness, diagonal mass, and artificial connectivity

With a common \(\gamma\), \(M_\ell^{(\gamma)}=\gamma I+(1-\gamma)P_\ell\). Then

\[
\bar M^{(\gamma)}=\gamma I+(1-\gamma)\bar P.
\]

For \(0\le\gamma<1\), **all averaged eigenspaces are identical**. Only eigenvalues, raw retention, and numerical conditioning change. Therefore a large set change under a common fixed-laziness sweep indicates degeneracy/numerics or a changed downstream rank rule, not a new geometric effect. Do this identity check, not an expensive exploratory sweep. At \(\gamma=1\) every direction is degenerate. For products, laziness changes the singular vectors in general; test \(\gamma\in\{0,.5,.9\}\) there, explicitly inspecting alternating modes when \(\gamma<.5\).

Retaining the original kernel diagonal before Sinkhorn is a genuinely different ablation: self-mass becomes cell-dependent and can overwhelm comparison. Measure its diagonal distribution and off-diagonal mixing separately.

For the kNN floor, vary \(\epsilon\in\{10^{-8},10^{-6},10^{-4}\}\), plus zero only where balancing feasibility is verified. Pre-balance off-edge mass scales roughly as \(N\epsilon/k\), not simply \(\epsilon\); balancing can alter it further. Report final off-edge mass, connected components without the floor, and sensitivity of tiny eigenvalue gaps. Floor-zero failure is a result, not an excuse to accept a poor Sinkhorn residual.

### Second-wave ablations

- **View preprocessing:** centred cosine versus uncentred cosine, and a raw dot-product PCA comparator. Activation norms are erased by cosine; inspect them as covariates before inventing another geometry. Raw-feature normalisation changes the estimand.
- **Sampling density:** equal images versus explicit label-free importance weights on a known synthetic sampling scheme. Never use inverse class frequency inside the construction. For real data, uniform sampled images is a defensible reference measure, not “uniform manifold volume.”
- **Block sampling/weights:** omit every other checkpoint, repeat a cell, and use interval weighting. Repeating an identical cell changes its weight in an average and adds diffusion in a product; this is a useful implementation/interpretation check.
- **Numerical precision:** float64 reference versus stored/quantised vectors; subspace residuals and hard-label changes near margins. Existing int16 vectors are suitable for display, not necessarily fine eigengap inference.

## 4. Comparison statistics and evaluation design

### 4.1 Spectra and continuous subspaces first

For orthonormal nonconstant bases \(U,V\) of equal dimension \(q\), report

\[
A_{sub}=\|U^TV\|_F^2/q
\]

and the full principal-angle spectrum, not only its mean. Also report the smallest principal cosine when a whole subspace is being claimed stable. For unequal dimensions, give captured-energy fractions with **both** denominators, or the projector distance; a larger span trivially captures more.

Compare subspaces in the same sample weighting. Near-degenerate eigenvectors can rotate freely, so sign matching individual modes is not a stability assessment. Report residual norms and gaps bordering the selected span; do not claim a rank-4 span is identified when the fourth and fifth eigenvalues are unresolved. A single mode's participation \(1/\sum_i u_i^4\) is basis-dependent in a degenerate block. Add projector leverage \(h_i=(UU^T)_{ii}\), its concentration, and \((\sum h_i)^2/\sum h_i^2\), which are span-invariant.

Raw eigenvalues are not comparable across arbitrary scaling, laziness, and block length. For fixed \(M=(I+P)/2\), convert \(\lambda_P=2\lambda_M-1\); compare gaps to matched nulls. Inspect small eigenvalues of \(P\) to detect near-bipartite alternation. With \(\gamma\ge1/2\), \(M\) is PSD but the original graph kernel need not be.

### 4.2 Hard sets and soft memberships

Report ARI and AMI, rectangular contingency matrices, and matched Jaccard. For different set counts, pad unmatched sets with zero and divide a symmetric matched-Jaccard total by \(\max(k_A,k_B)\), or report both directional coverage summaries. The existing rectangular matching mean over matched pairs can obscure all unmatched classes when \(r<10\); keep that legacy number if desired but add the denominator-explicit version.

Always show set-size histograms, minimum size, entropy/effective number of sets, unassigned fraction, and assignment margins. Report scores both on all points (with unassigned explicitly represented) and conditional on the jointly assigned subset **with its coverage**. A method must not win by leaving difficult points unassigned. Also show macro and size-weighted summaries; the latter can hide small-set failures.

SEBA's max-scaled vectors are not probabilities and do not sum to one. Compare soft vectors through column correlations or matched cosine similarity, and label any extra row-normalisation used to create probabilistic memberships. Keep raw negative minima and clipped mass as diagnostics rather than reporting only the clipped display.

### 4.3 Smoothness/retention: a useful objective, not independent proof

For a hard set \(A\),

\[
R_P(A)=\mathbf1_A^TP\mathbf1_A/|A|,
\quad R_M(A)=\tfrac12+\tfrac12R_P(A).
\]

For any zero-diagonal stochastic \(P\), a uniformly random subset of fixed size \(m\) has expected \(R_P=(m-1)/(N-1)\). Thus both set size and the compulsory lazy diagonal create large raw retention. Report \(P\)-retention and size-preserving nulls. A chance-adjusted retention can be useful, but must display its denominator and becomes unstable for almost-all-point sets.

For \(f\), report \(f^T(I-P_\ell)f/f^Tf\) per cell, its mean, its maximum, and a high quantile. For set leakage do the analogous per-cell calculation, not only the block mean. The block construction optimises average penalty, not worst-cell persistence. A source-based partition evaluated on its own \(\bar M\) is an in-sample objective check and is biased in favour of that construction.

Use three separate evaluation tracks:

1. **Native objective:** what each method optimises, clearly in-sample.
2. **Common geometric evaluation:** graphs built with predeclared alternate scales on fresh probes or omitted cells. These still privilege a notion of geometry; report more than one, without selecting the winner after seeing results.
3. **External validation:** class labels, error/loss, or planted truth, which were not used for construction/selection.

There is no neutral scalar “method works” statistic.

### 4.4 Resampling and out-of-sample checks

Use image-level subsampling without replacement first: e.g. ten 80% subsets, recomputing centring, row normalisation, neighbourhoods and scaling. Compare on shared image IDs. Ordinary bootstrap duplicates can create exact near-neighbours and radically change zero-diagonal kNN graphs; use it only with a carefully specified weighted-empirical estimator. Distinguish robustness to deleting images from repeating the same optimiser on a fixed graph.

For disjoint fit/test images, freeze preprocessing estimated on the fit images. Fit the kernel/bandwidth/landmarks on fit data, and define a genuine extension rule. For a dense Sinkhorn kernel with training factors \(a_j\), one possible row extension is

\[
p(x,j)=\frac{K(x,z_j)a_j}{\sum_kK(x,z_k)a_k}.
\]

It preserves constants but is not itself a jointly rebalanced bistochastic enlarged operator; state that limitation. A union-kNN extension additionally needs a declared reverse-neighbour rule using stored training radii. For a block, average the extended rows using the same block weights. With \(\bar P u=\lambda_Pu\), use \(\hat u(x)=\bar p(x,\cdot)u/\lambda_P\) only for eigenvalues safely away from zero. Here \(\lambda_P=2\lambda_M-1\), not \(\lambda_M\): the lazy identity has no new-point training neighbour. Refit/rebalance on a new probe bank is a different, transductive reproducibility test, not this extension.

Apply the frozen SEBA rotation and its training-derived threshold/sign/column scaling to extended functions. Do not independently rerun SEBA on test points and call that prediction. Compare assignments and smoothness on an independently constructed test graph; construction on that graph is allowed for scoring but cannot leak back into the fit. For k-means, use frozen centroid assignments. For kernel PCA, use the correct fit-set centring of cross-Grams.

For multi-layer predictive checks, omit one cell from a block and score its geometry; also score a successor outside the block. These test generalisation across views, not across images. Checkpoints from the same run are dependent; do not treat them as independent statistical replicates.

### 4.5 Rank selection and multiplicity

First compare every method on the common \(r\in\{2,3,4,6,8\}\) grid without declaring an optimal \(r\). Then allow a label-free selection rule using fit-set spectral gaps plus resampling stability, calibrated against the null and applied identically where meaningful. Always permit the answer **no separated nontrivial subspace/no supported discrete partition**. A fixed-r explorer can still show descriptive colours, with that status visible.

Separate rank \(q\), cluster count \(k_c\), bandwidth and block length. Do not choose \(r\) because there are ten CIFAR classes. Null-versus-real spectral tests need a full prespecified selection procedure; local significance of the best-looking one among 1,000 blocks is not meaningful.

Use paired differences across identical probe replicates and synthetic seeds. Bootstrap/permutation intervals over probe images quantify image-sampling uncertainty conditional on the trained network, not training-seed uncertainty or a confidence interval over “neural networks.”

### 4.6 Explorer presentation

Keep the current UMAP positions fixed when comparing colour sources, and display exactly which block/cell supplies the colours. Add “cosine PCA”, “direct clustering”, “single-cell”, and named null draws rather than a single ambiguous “baseline” dropdown. Annotate fit/postprocessing/validation status. UMAP and default kNN share local-neighbour geometry; apparent compactness in UMAP is not independent validation. Do not score methods in 2-D UMAP coordinates. Use exemplar selection rules and seeds fixed across methods, including boundary and unassigned points.

## 5. CKA, classical MDS, and kernel reconstruction

### 5.1 CKA is a diagnostic of representational similarity, not a clustering algorithm

For PSD Grams \(C_1,C_2\), centred linear/kernel alignment is

\[
\operatorname{CKA}(C_1,C_2)=
\frac{\langle HC_1H,HC_2H\rangle_F}
{\|HC_1H\|_F\|HC_2H\|_F}.
\]

Use a cell-by-cell CKA matrix to ask whether a long block combines substantially different geometry. Compare it with overlap of low-rank subspaces and with set agreement. High CKA can reflect broadly shared variance without shared discrete groups; low CKA can coexist with a shared low-variance direction.

The high-dimensional null is important: independent isotropic row-normalised clouds can have \(C\approx I\) when dimension is large relative to sample size. Their centred Grams both approach \(H\), so biased CKA can approach one despite independent data. Include a permutation calibration and a diagonal-bias correction rather than reading a high number as learned correspondence. One concrete companion is U-centering: zero each Gram diagonal, then for \(i\ne j\),

\[
A^U_{ij}=A_{ij}-\frac{A_{i+}}{N-2}-\frac{A_{+j}}{N-2}
+\frac{A_{++}}{(N-1)(N-2)},\qquad A^U_{ii}=0.
\]

Report the normalised Frobenius inner product of U-centred matrices alongside ordinary CKA, declaring the convention; it may be negative and is unstable/undefined when the corrected norm is nearly zero. Do not replace a null calibration with a correction formula. Kornblith et al. (2019) is the relevant CKA reference; unbiased HSIC corrections have an earlier literature.

A technical reason to expect PCA agreement: around a flat off-diagonal kernel, after constant projection the first-order geometric perturbation is a projected cosine Gram, up to scalar and identity terms. That motivates B1; it does not mean every finite-temperature graph is equivalent to PCA.

### 5.2 “The kernel reconstruction one”: classical MDS / kernel PCA

For unit rows, squared Euclidean distance is \(\Delta_{ij}^2=2(1-C_{ij})\). Classical MDS (Torgerson, 1952; Gower, 1966) gives

\[
-\tfrac12H\Delta^{(2)}H=HCH.
\]

Thus classical MDS on these distances is exactly the centred cosine-Gram PCA baseline, not a distinct competitor. If someone treats \(1-C\) as an ordinary distance and then squares it, they have changed the kernel. Kernel PCA (Schölkopf, Smola and Müller, 1998) is the corresponding eigendecomposition for another PSD kernel.

For \(G=U\Lambda U^T\succeq0\), the best rank-q Frobenius reconstruction is \(G_q=U_q\Lambda_qU_q^T\), with squared residual \(\sum_{j>q}\lambda_j^2\). Report this residual, its held-out/cross-Gram analogue, and how well the graph subspace spans the Gram:

\[
\frac{\|\Pi_U G\Pi_U\|_F^2}{\|G\|_F^2},\qquad \Pi_U=UU^T.
\]

The latter is descriptive variance capture, not a coherence objective. Conversely measure graph-energy capture by PCA directions.

A zero-diagonal affinity or a sparsified kNN graph need not be PSD, so it is not automatically a valid kernel-PCA Gram. The fixed-lazy symmetric stochastic \(M\) is PSD, but reconstructing it reconstructs the engineered Markov operator, not the original cosine kernel. Negative-eigenvalue truncation silently changes an indefinite affinity; avoid presenting it as the same method.

I would not start with sparse PCA. Its coordinate sparsity is basis-dependent in activation space and introduces a trained/fitted penalised optimisation baseline. SEBA on PCA already asks the cheaper and more relevant question: does sparse **site-space** extraction of an ordinary spectral subspace reproduce the displayed sets?

## 6. A density-aware layer-to-layer transfer estimator

### 6.1 First clarify the estimand

For paired probes \((x_i,y_i)=(z_{a,i},z_{b,i})\), the sample is from a joint distribution \(\mu_{XY}\) induced by sampling images. The marginals \(\mu_X\) and \(\mu_Y\) are generally different. On \(L^2(\mu_X)\) and \(L^2(\mu_Y)\), a normalised transfer/conditional-expectation pair is

\[
(\mathcal T f)(y)=\mathbb E[f(X)\mid Y=y],\quad
(\mathcal T^*g)(x)=\mathbb E[g(Y)\mid X=x].
\]

This convention makes \(\mathcal T\) source-to-destination for density ratios/functions and its adjoint the destination-observable pullback. Constants map to constants; singular functions identify maximally correlated source/destination observables at a stated regularisation/resolution. This is not a label-based operator, and it need not return clusters.

For a deterministic invertible map with the correct pushed-forward marginal, the unsmoothed operator is unitary: every resolvable singular value can equal one. With one unique observed pair per image, an unconstrained finite-sample estimator can likewise memorise the pairing and return an identity/permutation. **The density-aware estimator is not meaningful without a declared finite function space, diffusion, or ridge regularisation and held-out validation.**

The network supplies correspondence already: image ID tells us the source/destination pair. At the site-index level transport is identity, which the current construction exploits. A new estimator gains something by estimating conditional dependence between neighbourhoods/functions under their empirical marginals, not by discovering a missing matching. Moreover, centred row-normalised activations discard gain, so even when full hidden states evolve deterministically, the normalised view need not be a sufficient state with a deterministic next-view map. Across training checkpoints there is an image-paired joint distribution, not automatically a stationary dynamical system on activation vectors.

### 6.2 Cheap, interpretable density-aware estimator: soft Ulam joint table

Select \(m_X,m_Y\) random landmark images on fit data (e.g. 128 each, independent label-free selections). Define nonnegative partition-of-unity memberships using only within-view dot products:

\[
\phi_j(x)=\frac{\exp[\kappa_X(x^T\ell_j^X-1)]}{\sum_h\exp[\kappa_X(x^T\ell_h^X-1)]},\quad
\psi_k(y)=\frac{\exp[\kappa_Y(y^T\ell_k^Y-1)]}{\sum_h\exp[\kappa_Y(y^T\ell_h^Y-1)]}.
\]

Use fit-set centring/normalisation for new points. Landmark subsampling is not a trained dictionary. Define weights \(w_i=1/N_{fit}\),

\[
B_{jk}=\sum_iw_i\phi_j(x_i)\psi_k(y_i),\quad
p_j=\sum_k B_{jk},\quad q_k=\sum_jB_{jk},\quad
T=D_p^{-1/2}BD_q^{-1/2}.
\]

Remove negligible-mass landmarks by a prespecified rule and renormalise memberships if necessary. With partition-of-unity memberships, \(B\) is a genuine joint probability table over two *randomised coarse observations*. \(D_p^{-1}B\) is the row-stochastic forward conditional transition table; \(D_q^{-1}B^T\) is its reverse conditional table. SVD \(T=A\Sigma V^T\) has a trivial pair \((\sqrt p,\sqrt q)\) with singular value one. Nontrivial functions on the coarse categories are \(D_p^{-1/2}a_j\) and \(D_q^{-1/2}v_j\), with unit marginal-weighted norm, maximising cross-moment correlation under this coarse stochastic model.

Evaluate these functions on images by \(\phi(x)^TD_p^{-1/2}a_j\) and \(\psi(y)^TD_q^{-1/2}v_j\). The resulting soft-evaluated image functions need not themselves have unit empirical variance or correlation equal to \(\sigma_j\), because memberships are soft. Report their held-out covariance/variance directly. The estimator is a coarse Markov model, **not** exactly feature-covariance-whitened CCA; the next estimator supplies that comparison.

This construction respects empirical marginal density rather than forcing equal mass on landmark cells. Sinkhorn-balancing \(B\) to uniform cell marginals would deliberately change the measure and answer another question. Equal probability per sampled image and density weighting in activation space are compatible, not contradictory.

**What the SVD gives.** Matched source/destination functions, then optionally separately extracted source/destination sets and their empirical transfer mass. It allows a source grouping and a different destination grouping, rather than imposing one fixed indicator on all cells. Report both directions and transition tables, not only one colouring. Applying SEBA separately to source and destination spans can destroy column correspondence; match the resulting sets by the joint transfer table, not by inherited vector index.

**Cost.** Memberships \(O(Nm d)\) if using activations or \(O(Nm)\) lookup/exponentiation from cross-Grams; table accumulation \(O(Nm_Xm_Y)\), and SVD \(O(m^3)\) for equal \(m\). At \(m=128,N=2000\), the table multiplication is about 33 million multiply-accumulate terms and the SVD cubic unit is about 2.1 million. Twelve representative pairs, a few bandwidths, and split repeats are feasible; 100 pairs are still far below 100 full 2000-by-2000 SVDs. Dense coarse tables have many entries relative to 2,000 samples, so inspect occupancy/effective sample sizes and do not overinterpret fine transition probabilities.

**Status:** sample-estimable operator + SVD, **S**; no trained basis and no labels. Bandwidth is a resolution choice, not a learned semantic metric.

### 6.3 Companion: regularised kernel CCA / whitened cross-covariance SVD

Use within-view **PSD** kernels (linear cosine or retained-diagonal exponential dot-product kernel), not the indefinite zero-diagonal graph affinity. Build label-free Nyström features \(F_X,F_Y\) with 64–256 random landmarks; include the landmark Gram inverse square root so these are declared Nyström features, not an unexplained raw similarity matrix. Centre the features using fit means. With \(n=N_{fit}\),

\[
C_{XX}=F_X^TF_X/n,\quad C_{YY}=F_Y^TF_Y/n,\quad C_{XY}=F_X^TF_Y/n,
\]

\[
W_\rho=(C_{XX}+\rho_XI)^{-1/2}C_{XY}(C_{YY}+\rho_YI)^{-1/2}.
\]

An SVD \(W_\rho=A\Sigma V^T\) produces source coefficients \((C_{XX}+\rho_XI)^{-1/2}a_j\) and destination coefficients \((C_{YY}+\rho_YI)^{-1/2}v_j\). Centre away constants, or include the constant separately as an exact known component. These are regularised canonical variates/transfer singular functions in a finite feature space. The ridge singular values are regularised objective values, not automatically raw held-out correlations; measure the latter.

Choose ridge from a small label-free inner-validation grid relative to covariance scale, e.g. \(\rho=\eta\,\operatorname{tr}(C)/m\), \(\eta\in\{10^{-4},10^{-2},1\}\). Include small-eigenvalue truncation and sensitivity to rank. A near-zero ridge with a near-full-rank empirical kernel can return almost perfect in-sample correlations under random pairing. Use paired holdout and independently permuted fit-pairing controls. A closed-form ridge fit is allowed, but must be called a **fit**.

**Relation to the literature.**

- Froyland (2013) uses normalised transfer operators and singular vectors to identify finite-time coherent pairs under suitable regularisation; the marginal-weighted perspective above is the relevant connection. Froyland (2015) develops the dynamic-Laplacian viewpoint. Merely taking any cross-covariance SVD does not inherit a theorem about coherent sets without the necessary operator/smoothing assumptions.
- EDMD (Williams, Kevrekidis and Rowley, 2015) estimates a Koopman action in a feature space, typically involving a one-sided covariance inverse. Kernel EDMD uses kernel features (Williams, Rowley and Kevrekidis, 2015). That regression/eigendecomposition is not numerically the same as the two-sided whitening above. For changing marginals and dimensions, the two-sided SVD is the more natural diagnostic than stationary Koopman eigenvalues.
- Kernel CCA supplies the correlation-maximising, marginal-whitened formulation. VAMP (Wu and Noé, 2020; preprint 2017) supplies an operator/SVD perspective and out-of-sample scoring for nonreversible processes. Use existing `deeptime` routines for a cross-check where normalisations agree rather than writing another VAMP implementation.
- Banisch and Koltai (2017), *Understanding the geometry of transport: diffusion maps for Lagrangian trajectory data unravel coherent sets*, connects trajectory-based local diffusion and transport geometry. It is a relevant relative of the existing common-site, multi-view averaging approach, not a reason to equate that average with empirical conditional expectation.
- Diffusion-map-based transfer constructions use local diffusion to estimate/smooth spatial function spaces and trajectory pairing to transport between them. Coifman and Lafon (2006) explain how density normalisation changes diffusion-map limits. Choosing diffusion eigenfunctions as \(F_X,F_Y\) above is a useful second pilot, but changes the approximation space and needs a held-out extension.
- Klus, Schuster and Muandet (2020), *Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces*, is a reference for RKHS transfer-operator estimation. This proposal does not assert that every “kernel transfer operator” paper uses the same normalisation or estimates the same operator.

**Cost/status.** Approximately \(O(Nm^2+m^3)\) after cross-kernel construction for each pair, plus landmark factorisation. **S**, with explicitly labelled closed-form fits; no trained nonlinear feature map. Kernel bandwidth/ridge selection uses no semantic labels.

### 6.4 How this differs from the existing average and product

- **Average:** one scalar function on image IDs, small *average within-view* Dirichlet energy; order-invariant and forced to use the same per-image function through the block.
- **Product:** sequential application of per-view smoothing on shared image IDs; a meaningful diffuse–identity-transport–diffuse estimator at a specified scale. It already uses trajectory correspondence, but its empirical marginal on site indices is forced uniform and it is not the same as estimating a finite-feature conditional expectation between activation distributions.
- **Density-aware paired SVD:** distinct source and destination function spaces with empirical marginal weighting/whitening, estimating cross-view dependence at a declared resolution. It separates what is predictable across views from what merely has a small graph boundary in each. For many views, composing fitted transfer maps requires compatible intermediate features and measures; an endpoint fit and a composition are not interchangeable.

All three are functions of the same paired data and can agree substantially. The proposed estimator is an alternative estimand, not automatically a more “real” transfer operator. A nonlinear deterministic map with no low-dimensional bottleneck may have many predictable functions but no preferred discrete partition.

### 6.5 Recommendation: a small paired pilot, not a new full explorer yet

At \(N=2000\), start with 6–12 pairs: adjacent early/middle/late layers, two distant endpoint pairs, and two through-training pairs. Use 1,000 images for fitting, 500 for label-free bandwidth/ridge selection, and 500 as an untouched test split; repeat predeclared splits to quantify variability without selecting the best test outcome. Fit soft Ulam at \(m=64,128\) and whitened linear/Nyström features at \(m=64,128\). Compare with \(M_bM_a\), the two-cell average, and linear regularised CCA. Include independent-pair permutations.

Report top 6 nonconstant training singular values, actual held-out canonical correlations/cross-moment matrices, marginal effective ranks/occupancy, sample-size curves, and source/destination sets only if the functions support stable extraction. Validate first on the noisy circle-doubling benchmark below. A successful pilot is stable held-out dependence beyond the permutation control, and a meaningful difference from simpler linear/endpoint smoothing—not high training singular values.

**Do we need more than 2,000?** Not to run this controlled low-rank pilot. We probably do need more for stable fine neighbourhoods, rare sets, small spectral gaps, or high-rank nonlinear conditional estimation. Use \(N\in\{500,1000,2000\}\), fixed-resolution and fixed-neighbour-count curves separately. Extend to 4,000–8,000 only if the uncertainty is dominated by sampling and the coarse estimator is already sensible. No dimension-free guarantee makes 2,000 sufficient for arbitrary convolutional-state transfer. More points cannot cure unspecified smoothing or finite-sample memorisation without holdout.

## 7. Concrete planted and null benchmark suite

Run multiple generative seeds, applying the identical centring/normalisation and the whole parameter-selection procedure. Labels below are ground truth **for scoring only**. Use \(N\in\{500,1000,2000\}\) first. Where a benchmark uses coordinates, embed them by independent random isometries into different ambient widths; this checks architecture/dimension independence without changing dot-product geometry. Do not claim a method must win outside a measured parameter regime.

### T1 — Isotropic cloud, independent versus persistent: no privileged partition

Generate B5's Gaussian family over \(d/N\) and \(\rho\). Ground truth: no planted finite groups, independent pseudo-classes, with known cross-cell dependence when \(\rho>0\). Measure spectral edges, apparent gaps, SEBA reliability/sizes, stability and validation chance.

**Discriminates:** high-dimensional graph artefacts, finite-sample effects, and persistence of unclustered geometry. Persistent noise may be stable across cells but unstable across fresh samples. “No groups” does not mean a finite sample cannot be partitioned or that its graph has no leading modes.

### T2 — Anisotropic Gaussian with matched spectrum and shared factors

Generate

\[
x_{\ell i}=A_\ell u_i+B_\ell v_{\ell i},\quad
u_i\sim N(0,I_h),\quad v_{\ell i}\sim N(0,I_{d_\ell}),
\]

where singular values of \(A_\ell,B_\ell\) create a chosen decaying, spiked, or flat covariance spectrum. Use both real-data estimated spectra and simple geometric decay. Shared \(u_i\) gives linear cross-view dependence but no mixture label. Set \(h=0\) for independent anisotropic cells. Optional class labels remain independent.

**Ground truth:** covariance ranks and shared-factor dimensions, not clusters. **Discriminates:** covariance-only explanations and whether CCA recovers a known predictable subspace while SEBA invents discrete sets. Linear CCA is expected to succeed when its assumptions fit; this is a useful calibration, not an embarrassment to the nonlinear method.

### T3 — Easy persistent separated mixture: everyone should succeed

Let \(g_i\in\{1,\ldots,4\}\) be balanced, \(\mu_g\) regular-simplex vertices, and

\[
x_{\ell i}=aR_\ell\mu_{g_i}+\sigma\epsilon_{\ell i},\quad
\epsilon_{\ell i}\sim N(0,I_d).
\]

Choose \(a/\sigma\) sufficiently large at a modest intrinsic dimension, then sweep through the overlap transition. Ground truth is \(g_i\), unchanged across cells.

**Discriminates:** basic implementation and sample-size effects. PCA plus k-means, direct spherical k-means, static spectral methods, and block methods should all recover the obvious groups in the easy regime. Success here alone establishes no special advantage.

### T4 — Weak repeated group, strong changing nuisance: multi-view advantage with a smart cheap baseline

Let \(g_i\in\{-1,+1\}\) be fixed across cells, while \(h_{\ell i}\in\{-1,+1\}\) is independent across cells. Generate

\[
x_{\ell i}=a g_i e_{\ell,1}+b h_{\ell i}e_{\ell,2}+\sigma\epsilon_{\ell i},\qquad b>a>0,
\]

with orthogonal axes within each view. Sweep \(L\), \(b/a\), and noise. There is a regime where the shared signal in the averaged Gram dominates a nuisance restricted to one cell: roughly shared strength \(a^2\) versus single-cell contribution \(b^2/L\), before finite-N/noise corrections. Very local graphs may completely suppress the weak shared signal; that is a useful failure, not something to tune away using truth.

**Ground truth:** persistent \(g\), transient \(h_\ell\). **Discriminates:** single-view versus multi-view inference. Block PCA/concatenated k-means should also benefit. A claim that only the proposed dynamic graph method can win is not justified; this benchmark specifically tests that cheap multi-view explanation.

### T5 — One-cell bottleneck hidden by averaging

Use T3's persistent mixture in \(L-1\) cells, but replace one interior cell by independent Gaussian noise (or a mixture with independent labels). Keep the endpoints informative and vary \(L\) and the position/number of bottlenecks. In an idealised projector version, a group contrast has \(P\)-eigenvalue one in informative cells and zero at the bottleneck. Its average lazy eigenvalue is \(1-1/(2L)\), but its lazy-product attenuation from that bottleneck is \(1/2\); repeated bottlenecks multiply this penalty.

**Ground truth:** the common grouping exists in most cells but not throughout; no group passes a strict all-cell persistence criterion. **Discriminates:** average objective versus a universal-persistence interpretation, endpoint-only blindness, and worst-cell retention. The average has not “failed” its own objective when it retains this grouping. This benchmark is essential to prevent changing the success definition after seeing results.

### T6 — Density variation without a discrete partition

Draw \(\theta_i\) on the circle with smooth positive density

\[
p(\theta)=(1+\eta\cos\theta)/(2\pi),\quad 0\le\eta\le .9,
\]

and set \(x_i=(\cos\theta_i,\sin\theta_i)\), optionally with small isotropic noise and an isometric ambient embedding. For paired views let \(\phi=\theta+c\sin\theta\) with \(|c|<1\), embedded as another circle; the map is smooth and invertible but changes density. Apply the method's centring and row normalisation, noting that these themselves transform the geometry.

**Ground truth:** connected smooth support and known sampling density/Jacobian, no privileged finite partition. Score correlation of modes/sets with density and stability to a new sampling density. **Discriminates:** density effects, the reference measure implicit in normalisation, and stationary-vector mismatch. Do not declare density-sensitive results erroneous: ask whether they match the claimed estimand.

### T7 — Unequal planted groups, rare groups, and near-duplicate contamination

Use T3 with masses \((.60,.25,.10,.05)\) and different within-group variances. Separately contaminate 0%, 0.5%, and 2% of samples with paired near-duplicates: replace selected pairs by \(x_j=x_i+\delta\xi\), with \(\delta\) much smaller than typical spacing. Make duplicate pairs persistent across cells in one condition and cell-specific in another.

**Ground truth:** mixture membership plus duplicate provenance. **Discriminates:** whether small true groups are discarded, whether Sinkhorn/SEBA tends toward balanced-looking partitions, and whether pair-localised modes outcompete the planted large-scale grouping. Report macro as well as size-weighted recovery, duplicate leverage, class-size-matched chance, and resolution dependence. A rare group absent in a fit subset is not recoverable by any method; count that explicitly.

### T8 — Known noisy transfer on a circle: same marginals, different pairing

Draw \(\theta_i\sim\mathrm{Unif}[0,2\pi)\), set

\[
\phi_i=(2\theta_i+\eta_i)\bmod2\pi,\qquad
\eta_i\sim\text{wrapped }N(0,\sigma^2),
\]

and observe \(x_i=(\cos\theta_i,\sin\theta_i)\), \(y_i=(\cos\phi_i,\sin\phi_i)\). Compare with \(\phi_i\) independently uniform, preserving exactly the same population marginals. Add ambient noise only as a separate difficulty factor.

**Ground truth:** under uniform marginals, matched source/destination harmonics \((\cos 2m\theta,\cos m\phi)\) and their sine partners have correlation \(e^{-m^2\sigma^2/2}\); source odd harmonics have zero conditional expectation given \(\phi\). The nonzero transfer singular values follow this sequence (two real harmonics per \(m\)), before estimator smoothing/ridge. There is no planted finite partition.

**Discriminates:** actual paired transfer versus static marginal clustering, nonlinear versus linear dependence, correct source/destination singular functions, and regularisation bias. For example, linear CCA of the two displayed 2-D circle coordinates misses the doubled-angle dependence; kernel features can recover it. Aligned multi-view averages/products can also detect pairing-dependent geometry, so this is not a proof that only the new estimator contains the information. It tests whether the estimator recovers the *right operator functions* and its known loss of odd source information. At \(\sigma=0\), many singular values equal one; no unique finite-rank truncation should be claimed.

### T9 — Rotating/alternating partitions: no single “true” hard answer

Draw independent balanced \(u_i,v_i\in\{-1,+1\}\). Let one cell strongly encode \(u\), another strongly encode \(v\), and a third encode \(w_i=u_iv_i\), with weak isotropic noise. In an operator-level companion, use stochastic block-averaging projectors for these partitions, adding a common teleportation component before constructing fixed-lazy operators. Operator-level fixtures are labelled as such, not claimed to arise from the exact activation kernel.

**Ground truth:** three known competing distinctions, each visible in its own cell; no privileged common binary grouping. For balanced independent variables, the ideal block-average has a degenerate contrast space. **Discriminates:** near-degenerate basis rotation, arbitrary hard-set choices, rank sensitivity, and overconfident semantic naming. Score recovery of the *span* of planted contrasts and per-cell retention rather than force a single partition target. To test genuine order effects beyond reversal, perturb memberships/weights so the operators no longer commute and compare interior permutations; keep the reversible-order SVD identity as an exact check.

### My three highest-value benchmarks

**T1 (high-dimensional Gaussian null), T4 (weak shared signal plus changing nuisance), and T8 (known noisy nonlinear transfer).** Together they test “nothing planted,” “multi-view information beyond a single view,” and “a genuinely different paired-operator estimand.” T3 remains a compulsory smoke test; T5 is nearly as important because it directly audits the language of persistence.

## 8. A minimal staged execution plan

1. **No new model training.** Reuse exact cached Grams/operators. Verify rank conventions, constant removal, preprocessing and sample IDs. Add the cheap analytic controls for common laziness, common row permutation, orthogonal feature rotations, and product reversal.
2. **B1–B3 and B7–B8 first.** On all selected 100 cells/100 blocks, start with cosine PCA, single-cell controls, random-subspace extraction, and cheap validation nulls. Run exact full-feature/Gram spherical k-means on a representative subset before deciding how much approximation is acceptable.
3. **Run T1/T3/T4/T5 at modest scale.** Calibrate expected failures and plot distributions, not just one pleasing random draw. Only then spend on B4–B6 null ensembles for real data.
4. **A1–A5 on a blocked design.** Choose 8–12 representative cells/blocks before inspecting class outcomes: early/middle/late layers, step 0/intermediate/final training, short/long blocks. Change one factor at a time, then investigate the few interactions suggested by the first pass—especially graph family × normalisation and block length × product smoothing.
5. **Transfer pilot on T8 and a dozen real pairs.** Keep it a separate panel/estimand until it has held-out evidence of stable low-rank dependence and its relationship to linear CCA and endpoint products is understood.
6. **Expand only useful contrasts.** Freeze a smaller palette of genuinely different colour sources for the full explorer. Keep failure cases selectable and explain why they are included. One model/seed remains one model/seed.

## 9. Things I would not bother with initially

- A very broad clustering zoo. DBSCAN/HDBSCAN, GMMs, agglomerative clustering, sparse PCA, NMF and spectral biclustering all add assumptions and tuning. PCA, spherical k-means, ordinary spectral clustering and consensus already isolate the central questions. A density-clustering baseline may become useful if T6 reveals a specific ambiguity, but is not first-line.
- Training an SAE, learned stitch, deep clustering head or contrastive embedding for this initial characterisation. They are valid labelled baselines in a larger study, but confound representational learning with a much narrower operator comparison and are costly.
- Claiming MDS and linear kernel PCA are independent corroborations of PCA when they are algebraically the same calculation here.
- Spending a full sweep on fixed laziness for averages: the eigenspace invariance is exact.
- Exhaustive k × perplexity × normalisation × rank × block × seed sweeps without a question-led design. They create selection opportunities faster than understanding.
- Any score in UMAP coordinates, or an image gallery chosen to make one method look more semantic.
- Full-rank unregularised kernel CCA at N=2000. Near-perfect training singular values are an expected interpolation pathology, not a pilot success.
- A theorem that high-dimensional Gaussian data “cannot have clusters” used in place of running the exact finite-N construction. The sample graph, sparsifier and extractor matter.

## 10. Risks / ways this study could mislead

1. **Equating nontrivial with class-aligned.** A method can reveal stable, non-class structure; CIFAR classes can also align trivially with input geometry. Labels are one validation axis, not the target definition.
2. **Using a too-weak null.** Independent isotropic Gaussians discard anisotropy and cross-cell dependence. A real/null difference there does not establish information beyond covariance. Conversely, a null preserving the signal of interest cannot test its absence.
3. **Confusing forced extraction with discovery.** SEBA and k-means return answers at chosen ranks/counts. Reliability, a gap, stability, and a coherent held-out effect are separate questions; none should be assumed from the existence of coloured sets.
4. **Normalisation changing the target silently.** Uniform mass per image, uniform landmark mass, degree-stationary mass, and manifold-volume approximations are not the same. Density correction is not a universally superior engineering detail.
5. **Optimising and scoring on the same graph.** Spectral methods naturally excel at their own Rayleigh quotient. Independent-view/image checks and baseline-native scores are required to interpret that excellence.
6. **Matching unequal partitions unfairly.** Unmatched classes, unassigned images, tiny duplicate sets and massive majority sets can all inflate a single headline score. Always publish the contingency and coverage.
7. **Selection over blocks and r.** The explorer makes post-selection unavoidable for exploration; formal claims need a frozen subset or a selection-aware null. Distinguish illustrations from tests.
8. **Treating adjacent layers/checkpoints as independent replications.** Shared images and one trained network induce strong dependence. Probe resampling is not network replication.
9. **Forgetting the sampling resolution.** Fixed k changes geometric radius with N; fixed bandwidth changes expected neighbour count. Report both scaling regimes. Retuning on every N can conceal nonconvergence.
10. **Ignoring numerical/tie-breaking instability.** Small spectral gaps, hard argmax boundaries, SEBA clipping, failed balancing, and quantised vectors can create visually dramatic differences without a robust geometric change.
11. **Calling average smoothness “present in every layer.”** T5 makes this error explicit. Likewise a long operator product can suppress everything by total diffusion alone.
12. **Calling predictability coherence without regularisation.** Deterministic one-to-one transport can make all functions predictable; noisy/coarse transfer singular functions become informative only at a stated scale. Density-aware transfer is not automatically a discrete-set detector.
13. **Overinterpreting a successful synthetic regime.** T3 validates mechanics; T4 validates a multi-view advantage but may favour concatenated PCA just as much; T8 validates operator estimation rather than a claim about learned semantics. State what each actually identifies.

## References and confidence boundaries

References below are established connections used in the proposal, not claims that their hypotheses automatically hold for this network.

- Torgerson (1952), *Multidimensional scaling: I. Theory and method*; Gower (1966), *Some distance properties of latent root and vector methods used in multivariate analysis*.
- Schölkopf, Smola and Müller (1998), *Nonlinear component analysis as a kernel eigenvalue problem*.
- Ng, Jordan and Weiss (2002), *On spectral clustering: analysis and an algorithm*.
- Coifman and Lafon (2006), *Diffusion maps*.
- Froyland (2013), *An analytic framework for identifying finite-time coherent sets in time-dependent dynamical systems*.
- Froyland (2015), *Dynamic isoperimetry and the geometry of Lagrangian coherent structures*.
- Williams, Kevrekidis and Rowley (2015), *A data-driven approximation of the Koopman operator: extending dynamic mode decomposition*.
- Williams, Rowley and Kevrekidis (2015), *A kernel-based method for data-driven Koopman spectral analysis*.
- Banisch and Koltai (2017), *Understanding the geometry of transport: diffusion maps for Lagrangian trajectory data unravel coherent sets*.
- Froyland, Rock and Sakellariou (2019), *Sparse eigenbasis approximation: multiple feature extraction across spatiotemporal scales with application to coherent set identification*.
- Kornblith, Norouzi, Lee and Hinton (2019), *Similarity of neural network representations revisited*.
- Klus, Schuster and Muandet (2020), *Eigendecompositions of transfer operators in reproducing kernel Hilbert spaces*.
- Wu and Noé (2020; preprint 2017), *Variational approach for learning Markov processes from time series data*.

I have not relied on an unverified theorem asserting that this exact zero-diagonal, union-kNN, Sinkhorn, fixed-lazy, SEBA pipeline consistently estimates a particular population transfer operator. Establishing such a theorem would require a separate, carefully specified asymptotic regime. The study proposed here is designed to characterise the finite-sample construction without smuggling in that claim.
