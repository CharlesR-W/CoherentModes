# Baselines, nulls, construction ablations and planted benchmarks for the coherent-sets explorer (ResNet-20 / CIFAR-10)

Draft 3 (2026-11-15; revised after `REVIEW_1_plan.md`). Orchestrator: Claude Fable 5.2. Merges the orchestrator's draft 1 with the independent
proposal `IDEAS_gpt.md` (GPT-6 Astra), which is the reference for formula-level detail wherever this plan cites
"IDEAS §x". Where this plan and IDEAS differ, this plan wins.

## 0. Owner's request and fixed decisions

- Add null / baseline views to the explorer (clusters from k-means, PCA, sparse versions, "whatever the smart
  baselines are"). Purpose: **characterise which versions of the method give non-trivial information**, and make sure
  nothing read off the explorer follows from a data-agnostic argument about statistics in high dimension.
  **No verdicts**: captions say what is plotted and what the null value is; nothing says a method works or fails.
- Look at the engineering choices (graph type, density dependence, Sinkhorn, ...).
- Consider CKA and classical MDS ("the kernel reconstruction one").
- Design and pilot an estimate of the density-aware layer-to-layer transfer operator itself (points as probes).
- Planted-structure benchmarks.
- Decided by the owner (2026-11-15): ResNet-20/CIFAR first (`explore/cifar_coherent`, tag `r20s0b`); deliverable = a
  colour-source selector in the explorer + a "Baselines and nulls" tab; transfer operator = design + pilot as one more
  variant; <= $25 Modal; local only (publish to the laptop mirror; no commit, no gh-pages deploy).
- Repo commitments apply to *the construction*. Everything here is labelled with its status (IDEAS §1.1):
  **S** label-free dot-product spectral object; **P** post-processing, diagnostic or null generator; **B** fitted
  baseline outside the construction (k-means, NMF, ridge CCA are fits and are called fits); **V** validation with
  labels (never feeds a kernel, basis, rank, bandwidth or selection). No THEOREMS/SPARSITY vocabulary. Exploratory side
  study, not an `EXPERIMENTS.md` stage; no change to `THEORY.md`.

## 1. What is being characterised

Default object: per cell (checkpoint t, site s), Z = column-centred then row-normalised activations, C = Z Z^T;
union 15-NN 0/1 graph with 1e-6 floor; zero diagonal; symmetric Sinkhorn to P; M = (I+P)/2; block operator
Mbar = mean of M over the block's cells; psi_1..psi_8; SEBA at r in {2,3,4,6,8} (r counts the constant; q = r-1
non-constant directions); hard sets by argmax, number of sets k_c = r.

Explanations a reader should be able to check by looking, without being told the answer:

- T1 *covariance geometry*: the sets are leading principal directions (of one cell or of the concatenated block) made
  discrete by SEBA; the graph adds nothing.
- T2 *one cell / simple voting*: the block's sets are those of one of its cells, or of a vote among single-cell sets.
- T3 *input*: the sets belong to the images (pixels, an untrained network), not to the trained network.
- T4 *labels / output*: the sets are the classes or the predicted classes.
- T5 *high-d, finite-N, forced extraction*: eigenvalues near 1, gaps, r plausible-looking sets and tidy UMAP colourings
  also arise from structureless data with the same N, k and second moments; SEBA and k-means always return r sets.
- T6 *rule artefact*: the answer moves with an arbitrary choice (k, kernel, normalisation, floor, r, rounding, average
  vs product, block weights, probe subsample, optimiser start, network seed).
- T7 *different estimand*: average within-view smoothness, sequential diffusion, and cross-view predictability are
  three questions; where do their answers differ?

## 2. Colour sources added to the explorer ("Sets from")

Inputs. The stored cosine Grams (`/out/work/r20s0b/gram_<step>_<site>.npy`, Modal volume `cm-cifar-coherent`) have
lost the activation norms, so they cannot give fit-only centring, uncentred views or pre-normalisation covariances.
The first Modal unit therefore re-extracts, per cell, the **raw uncentred Gram** R = X X^T (float64, N x N); the
centred cosine Gram of any subset, cross-Grams to held-out images under fit-only centring, norms and the
pre-normalisation covariance spectrum all derive from R (asserted against the stored C to 1e-10).

Scope. Cheap sources (L1, G1, G2, C2, C3, one N3 draw; pointers C1, I2; L3 exact spherical k-means only as far as the timing
benchmark allows, and first to be cut back to the representative blocks) are computed for the grid in this order
until the explorer data budget (60 MB added, <= 2 MB per interaction) is reached: all 78 layer blocks at the final
checkpoint; all 36 time blocks at b5, b9, logits; layer blocks at steps 0, 1024, 6400; the rest. Everything else is
computed on the 12 representative blocks of section 3. Top-9 eigenpairs use an iterative solver with residual
checks (dense fallback at small gaps); matrices are streamed, never the whole bank in memory. A timing benchmark on
5 blocks precedes any expansion. Ranks r in {2,3,4,6,8}. For a block, linear baselines use the block Gram C_B = mean of C over its cells (= Gram of the
concatenated per-cell-normalised activations, IDEAS B1). Each source gives hard labels for the 2,000 images; spectral
sources also give a q-dimensional non-constant subspace.

| id | source | definition | status | exposes |
|---|---|---|---|---|
| M0 | the method | as stored | S+P | |
| L1 | cosine PCA + SEBA | top q eigenvectors of H C_B H (H = centring; = classical MDS of chordal distances, IDEAS §5.2), then the unchanged SEBA with the constant | S+P | T1; holds the extractor fixed |
| L2 | cosine PCA + k-means | k-means (k_c = r, 20 starts) on PCA scores U_p Lambda_p^(1/2), p = max(q, 20) | B | T1 |
| L3 | spherical k-means | exact, in the Gram (IDEAS B2 scoring rule), k_c = r, 20 k-means++ starts, block Gram for blocks | B | T1 with no spectral step |
| G1 | method eigenvectors + k-means | k-means on psi_1..psi_q, plain and row-normalised (NJW rounding) | B on S | SEBA vs k-means on the same span |
| G2 | textbook spectral clustering | same kNN graph, D^-1/2 A D^-1/2 (no Sinkhorn, no lazy step), per cell; block = mean of these; row-normalise; k-means | S+B | what Sinkhorn + lazy + SEBA change together |
| C1 | first / middle / last cell of the block | the method on that single cell (already stored as single-cell blocks; a pointer, no new data) | S+P | T2 |
| C2 | co-assignment consensus | A_B(i,j) = share of the block's cells in which i, j are in the same single-cell SEBA set (unassigned never co-assigned); eigenvectors of H A_B H; SEBA | P | T2: voting vs averaging operators |
| C3 | mean-projector consensus | eigenvectors of mean over cells of U_q U_q^T (single-cell leading spans); SEBA | S+P | T2: spans only vs full spectra |
| I1 | pixels | the full method, and L1, on centred row-normalised raw pixels (one cell; same labels painted on every block) | S+P | T3 |
| I2 | untrained network | the method for the same block at step 0 (stored; a pointer) | S+P | T3 |
| I3 | other training seed | the method on `resnet20-relu-seed1`, same probe images, final checkpoint, layer blocks only | S+P | T3 vs noise: what two trainings share |
| Y1 | classes; Y2 predicted class | from `probe.npz`, `points_<step>.npz` | V | T4 |
| N-draws | one stored draw each of nulls N2, N3, N4, N6 below | so a reader can *see* a null colouring on the same panels | P | T5 |

Second wave, only if the first wave leaves a question open (owner asked about "sparse versions"; IDEAS §9 argues SEBA
on PCA is the relevant sparse baseline): symmetric NMF of the non-negative part of C_B (B); Leiden modularity on the
kNN graph (B). Not in this round: SAE or any trained dictionary.

Per colour source the explorer's side card works unchanged (sizes, class make-up, images) and gains: status tag
(S/P/B/V), agreement with M0 for the same block and r (AMI and matched Jaccard, with the size-preserving null value),
coverage (share unassigned).

## 3. Nulls

Each null preserves something stated and destroys something stated; none is "the" null (IDEAS §10.2).

| id | null | preserves | destroys | scope |
|---|---|---|---|---|
| N1 | isotropic Gaussian cells, N = 2000, d in {16, 64, 256, 1024, 4096}; across cells: independent, or G_l = sqrt(rho) G_0 + sqrt(1-rho) E_l, rho in {0, .5, .9, 1} | N, d, pipeline | everything else; rho > 0 = persistent geometry with no groups | synthetic (section 8, P0) |
| N2a | per-cell covariance-spectrum-matched Gaussian (IDEAS B6a), independent across cells | each cell's feature-covariance spectrum, in expectation | higher moments; cross-cell dependence | 12 representative blocks x 20 draws |
| N2b | joint second-order surrogate (IDEAS B6): one Haar orthogonal O (N x N) with O 1 = 1, shared by all cells of the block; C*_l = D_l O C_l O^T D_l with D_l diagonal restoring unit diagonal (exact Gram-only form; equals rotating the rows of the concatenated features by O). Before D_l this preserves the mean and all within- and cross-cell second moments exactly; the mismatch after D_l is reported | within- and cross-cell second moments | all image-level structure beyond second moments | same 12 blocks x 20 draws |
| N3 | trajectory shuffle: P_l -> Pi_l P_l Pi_l^T with independent permutations (one cell fixed) | every cell exactly | that row i is the same image in every cell | 12 blocks x 20 draws for bands (coherent permutations shared across candidate blocks and ranks, so selected spectral excess can be calibrated); 1 stored draw per explorer block |
| N4 | N3 within CIFAR class | cells + class identity across cells | within-class correspondence | 12 blocks x 20 draws (V) |
| N5 | degree-preserving rewiring of each cell's pre-floor kNN graph, then floor + Sinkhorn + lazy | degree sequence / hubness | neighbourhood topology | 12 blocks x 10 draws (second wave) |
| N6 | SEBA on Haar random q-subspaces orthogonal to the constant; and SEBA on the real span after a random rotation / different initial rotations | N, r (and the span, for the rotation test) | the operator | all r, 1,000 draws; rotation test on 12 blocks x 20 |
| N7 | size-preserving random assignment and label permutation, with the *whole* matching (Hungarian) and any max-over-r / max-over-blocks selection repeated inside each permutation | set sizes | membership | every reported *agreement* number, 1,000 draws (does not calibrate spectral excess: N3/N2 do) |

"12 representative blocks" are fixed **before** looking at class outcomes: single cells stem, b5, b9, logits at the
final step; layer blocks stem..b3, b4..b6, b7..b9, stem..logits at the final step; b4..b6 at steps 0 and 1024; time
blocks 1024..64000 at b5 and at b9. (The builder may substitute a block only for a technical reason and says so.)

Reported per block against the nulls: lambda_1..8 (as lambda_P = 2 lambda_M - 1) with the null band; gaps; both
spectral ends of P; participation number and span leverage concentration (IDEAS §4.1); SEBA set sizes, unassigned
share, clipped negative mass; class AMI with its N7 value. Analytic controls run as asserted tests rather than sweeps:
a common permutation of all cells only relabels; orthogonal feature rotations change nothing; reversing a product
gives Q^T; common laziness gamma leaves every averaged eigenspace unchanged (IDEAS A5).

## 4. Stability and out-of-sample checks (no new model)

- S1 image subsampling: ten 80% subsets without replacement (everything recomputed: centring, normalisation, graph,
  balancing), compared on shared images (AMI, matched Jaccard, subspace overlap). No bootstrap with duplicates.
- S2 disjoint fit/test with a frozen extension (centring from the fit images only, via R; a held-out image uses its own norm after subtracting the fit mean): fit on 1,500, extend to 500 with the row extension of IDEAS §4.4
  (training Sinkhorn factors; union-kNN reverse-neighbour rule declared; frozen SEBA rotation, threshold, scaling);
  compare extended assignments with a refit on all 2,000, and score smoothness on a graph built from the test images.
  k-means sources use frozen centroids, PCA the fit-set centring. Applies to M0, L1, L3.
- S3 rank nesting: conditional entropy of the sets at r given those at r' > r (are finer sets refinements?).
- S4 seed (I3) and SEBA start (N6).
- Leave-one-cell-out: block sets from all cells but one, scored on the omitted cell's graph; and on the first cell
  after the block (generalisation across views, not images).

All of S1-S4 on the 12 representative blocks; S1 additionally for every single cell at the final checkpoint.

## 5. Construction ablations

One factor at a time around the default on the 12 representative blocks and on the planted benchmarks; then only the
interactions the first pass points to (expected: graph x normalisation; block length x product smoothing). Every
variant is scored by section 7's three tracks.

| factor | levels | note |
|---|---|---|
| A graph family and scale | union kNN k in {5, 15, 50, 150}; mutual kNN 15; per-point perplexity 30/100/300; global vMF (the two stored kappas + 2 more); locally scaled kernel exp(-(1-C_ij)/sqrt(delta_i delta_j)), delta = 15th-neighbour cosine distance; epsilon-ball at matched mean degree | compare at native settings **and** at matched median effective neighbour count 1/sum_j P_ij^2 after normalisation (IDEAS A1) |
| B floor | 1e-8, 1e-6, 1e-4; none where balancing is feasible (largest component reported) | report final off-edge mass |
| C normalisation | Sinkhorn; D^-1/2 K^(alpha) D^-1/2 with alpha in {0, 1/2, 1}; raw adjacency scaled by its norm (not a Markov operator; no retention reported) | report stationary-vector drift across the block's cells and the constant's overlap with the top averaged vector (IDEAS A2) |
| D diagonal | zero + lazy (default); retained kernel diagonal; non-lazy P for products (gamma in {0, .5, .9}) | laziness for averages is the analytic identity, not a sweep |
| E block combination | mean of M; product Q = M_b...M_a (left and right singular spans); two-end product M_b M_a; matched total diffusion: prod exp(-(tau/L) L_l) vs exp(-(tau/L) sum L_l); mean of M^2; one graph on the concatenated view (kNN on C_B); intersection and union of the cells' kNN graphs | commutator norms as a descriptive diagnostic |
| F block weights (time blocks) | equal per listed checkpoint (default); training-step-interval weights; log-step weights; drop every other checkpoint; repeat a cell | |
| G rounding | SEBA at mu = c/sqrt N, c in {.5, .99, 1.5}; k-means; row-normalised k-means; small (q, k_c) cross-grid | |
| H view preprocessing | centred cosine (default); uncentred cosine; GAP view vs flat view (a separate GAP raw Gram is extracted); PCA of C vs of HCH (the stored "PCs" diagonalise C: IDEAS §1.1) | activation norm inspected as a covariate |
| I sample size | N in {500, 1000, 2000} with k fixed and with k proportional to N; N = 10,000 at the final checkpoint | |

## 6. Transfer-operator variant (design + pilot; mathematics to be checked by the reviewer before coding)

Estimand (IDEAS §6.1). For paired views (x_i, y_i) = (z_{a,i}, z_{b,i}) with empirical marginals mu_X, mu_Y:
(T f)(y) = E[f(X) | Y = y] on L2(mu_X) -> L2(mu_Y), and its adjoint. **Convention**: every matrix below is written
destination-by-source acting on column vectors of source coefficients (as Q = M_b M_a is), so right singular vectors
are source functions, left are destination functions, and forward composition is T_bc T_ab; the joint table and the
cross-covariance are stored transposed accordingly, and each estimator's output names its source side. Its singular functions are the maximally
correlated source / destination observables at a stated resolution. Because each image is one deterministic pair, an
unregularised estimator returns the identity on image indices: content comes only from a declared finite function
space or smoothing, plus **held-out** validation. Three estimators, all closed form:

1. *Diffuse-transport-diffuse on image indices* (exists): Q = M_b M_a and the chain product. Uniform mass on images.
2. *Soft-Ulam joint table* (IDEAS §6.2): m landmarks per view (random fit images), partition-of-unity memberships
   phi, psi from within-view cosines; B = sum_i w_i psi(y_i) phi(x_i)^T (destination x source), row sums q, column sums p; T = D_q^-1/2 B D_p^-1/2; SVD. A joint
   probability table of two coarse observations; marginals are the empirical ones (not rebalanced to uniform).
3. *Whitened cross-covariance SVD* (regularised CCA; IDEAS §6.3): features F_X, F_Y = (i) linear cosine PCA scores,
   (ii) Nystrom features of a PSD within-view kernel, (iii) Nystrom-extended leading eigenfunctions of each cell's own
   P (diffusion coordinates); W = (C_YY + rho I)^-1/2 C_YX (C_XX + rho I)^-1/2; SVD; features centred with fit means. Ridge is a fit and is called one.
   Cross-check against `deeptime`'s VAMP where normalisations agree.

Blocks of more than two cells: endpoint operator T_ab, the chain T_{b-1,b} ... T_{a,a+1} where intermediate feature
spaces are compatible, and (as the multi-view analogue) generalised CCA (centred MAXVAR form) = eigenvectors of the mean of the per-cell
projectors onto span(F_s), which is again an average of operators on image indices and so sits next to Mbar and C3.

What it adds that sections 2-5 cannot: held-out correlations / VAMP-2 on images outside the fit set; separate source
and destination groupings matched through the joint table rather than one indicator forced on every cell.

Controls. (i) Independence null: the pairing is permuted **within each of fit, tuning and test**, and the whole
procedure including tuning is repeated per permutation (a fit-only permutation leaves real dependence in tuning and
test and is reported separately as "training-pairing disruption"). (ii) *Composition discrepancy* T_ac vs T_bc T_ab on
held-out images: equality is **not** expected (normalised views need not be sufficient Markov states; projection,
ridge and smoothing each break it; e.g. X = Z with Y constant), so it is reported as a discrepancy with its parts
separated (finite basis; ridge; view insufficiency) and calibrated on a synthetic chain that is Markov by
construction (P8 extended to theta -> 2 theta -> 4 theta with independent noise).

Preregistered pilot criteria (finite resolution; fixed before the run): tuning selects (m, rho, bandwidth) by
tuning-split VAMP-2; a function counts as "supported at this resolution" only if its test correlation exceeds the
95% point of control (i) in at least 8 of 10 splits; a soft-Ulam table is used only if every retained landmark has
effective occupancy >= 10 fit images, otherwise m is reduced; the report targets the first few (<= 6) held-out
functions and never individual transition probabilities. VAMP-2 = sum of squared singular values of the test-split
whitened cross-covariance in the fitted top-6 non-constant functions (constant removed; test covariances regularised
with the tuned rho); 200 permutations per split. The 8-of-10 rule is a descriptive support criterion (splits overlap,
several functions and pairs are inspected), not a 95% confidence statement, and the page says so.

Pilot. (a) Synthetic first: noisy circle doubling (P8) with known singular values exp(-m^2 sigma^2 / 2), and the
shared-factor Gaussian (P2). (b) Real, final checkpoint, 12 pairs fixed in advance: adjacent pairs early / middle /
late (stem-b1, b4-b5, b8-b9, pool-logits), distant pairs (stem-b9, b3-b9, b6-logits, b1-b5), through-training pairs at
b5 and b9 (1024 -> 64000, 6400 -> 64000). N = 2,000 probe
images split 1,000 fit / 500 tuning (m, rho, bandwidth: label-free) / 500 test, 10 predeclared splits; **and** a
10,000-image extraction (the CIFAR-10 test set, final checkpoint, 12 sites; 6,000 / 2,000 / 2,000) for the sample-size
curve N in {500, 1000, 2000, 6000}. m in {64, 128}; linear, Nystrom and diffusion-coordinate features. Report: top 6
non-constant training singular values; actual held-out correlations; permutation-control band; occupancy / effective
sample size of the joint table; agreement of the leading source span with M_b M_a, with the two-cell Mbar and with
linear CCA; sets only if the functions are stable across splits. In the explorer the transfer variant appears as a
colour source for those 12 pairs only, labelled as a pilot.

## 7. Comparison statistics (IDEAS §4)

Three tracks, never collapsed into one score: **native objective** (what each method optimises; in-sample),
**common geometric evaluation** (graphs at two predeclared alternate scales, on held-out images or omitted cells),
**external validation** (labels, loss, planted truth).

- Subspaces: full principal-angle spectrum, mean cos^2 and smallest cosine, constant removed, chance q/(N-1) printed;
  gaps bordering the span; span leverage h_i = (U U^T)_ii and its concentration.
- Hard sets: AMI, ARI, contingency table, matched Jaccard with the denominator max(k_A, k_B) (the legacy mean over
  matched pairs kept alongside); on all points with "unassigned" as a category, and on jointly assigned points with
  coverage; macro and size-weighted.
- Retention: R_P(A) = 1_A^T P 1_A / |A| per cell with its size null t + (1-t)(|A|-1)/(N-1), t = tr(P)/N (t = 0 for
  the default zero-diagonal P); degree-normalised symmetric variants are first converted to their Markov operator and
  functions are compared in function coordinates under a stated measure; raw-adjacency variants get no retention; mean, maximum and 90th percentile
  over the block's cells of the per-cell Dirichlet energy (the block optimises the mean, not the worst cell).
- Linear content ("kernel reconstruction"): ||Pi_U G Pi_U||_F^2 / ||G||_F^2 for the method's span U against the
  rank-q optimum sum_{j<=q} lambda_j^2 / sum lambda_j^2, per cell and for C_B; and the reverse, the Dirichlet energy
  of the PCA span under Mbar.
- CKA: S x S linear CKA between cells' centred Grams, with the U-centred (debiased) version and its permutation
  value (IDEAS §5.1), next to S x S subspace overlap and S x S matched set agreement.
- Rank: common grid r in {2,3,4,6,8} with no "best r"; a label-free gap + stability summary against the nulls, which
  may say "no separated subspace".
- Selection: any statistic that picks a best block or r repeats the pick inside the appropriate null (N7 for
  agreement numbers; N3 / N2 with coherent draws for spectral excess).

## 8. Planted and null benchmarks (synthetic, local CPU; IDEAS §7 has the generative formulas)

Every benchmark measures, per condition, how much of its planted information is available after the pipeline's
preprocessing (centring + row normalisation), with an oracle that knows the generative form: asserted high in the
informative regimes (positive controls), asserted at chance in bottleneck / null cells (negative controls), and
reported as a curve along the difficulty axis.
Sweeps run on Modal CPU, not on the orchestration box; only smoke-size runs are local.
Each benchmark is `make(seed, **params) -> ([X_1..X_L], truth)`, run through the production code path (the same
functions the real pipeline calls), swept along one difficulty axis, 10 seeds, N in {500, 1000, 2000}, coordinates
embedded by independent random isometries into different ambient widths. All of M0, L1-L3, G1-G2, C1-C3, product
variants and (where paired) the transfer estimators are scored.

| id | benchmark | truth | what it separates |
|---|---|---|---|
| P0 | isotropic Gaussian, independent vs persistent (N1) | no groups | T5 reference: spectra, SEBA sizes, stability, pseudo-class agreement |
| P1 | easy persistent mixture swept through the overlap transition (IDEAS T3) | group | smoke test: everything should succeed, then fail together |
| P2 | anisotropic Gaussian with shared linear factors, no groups (IDEAS T2) | shared-factor span of the raw latent, and separately what of it survives row normalisation (measured by the oracle) | covariance-only explanations; CCA recovers a span while SEBA still returns sets |
| P3 | weak repeated group + strong cell-specific nuisance (IDEAS T4), sweep L, b/a | persistent g | single-cell vs multi-view; block PCA is expected to benefit too |
| P4 | one-cell bottleneck hidden by averaging (IDEAS T5) | grouping absent in one cell | average vs product vs worst-cell retention; audits the word "persists" |
| P5 | persistent groups that are not linearly separable: angular structure on the sphere (radius carries nothing after normalisation): interleaved great-circle arcs / spherical spirals / two linked rings in a random low-d subspace + ambient noise, new embedding per cell | manifold id | graph methods vs L1-L3 (C_B has no block structure) |
| P6 | density variation, no partition: circle with density (1 + eta cos theta)/2pi; paired view theta + c sin theta (IDEAS T6) | none; known density | which rules turn density into sets; reference-measure effects |
| P7 | unequal and rare groups (60/25/10/5 %), unequal spreads, persistent or cell-specific near-duplicate pairs (IDEAS T7) | group + duplicate provenance | balance bias of Sinkhorn/SEBA; pair-localised modes |
| P8 | noisy circle doubling phi = 2 theta + eta, vs independent pairing with the same marginals (IDEAS T8) | singular values exp(-m^2 sigma^2/2), harmonic pairs | the transfer estimators' correctness; linear CCA misses it |
| P9 | competing partitions u, v, uv in different cells (IDEAS T9) | span of three contrasts; no single partition | degenerate rotation, forced hard sets |
| P10 | hierarchy sharpening with depth: tree-structured means, early cells resolve the top split only | tree level per block | which level a block L'..L returns; rank nesting (S3) |
| P11 | multilayer SBM at the graph level, some cells assortative and some disassortative | community | mean of M vs mean of M^2 vs product |
| P12 | trained toy network with a known latent: reuse `explore/vit_coherent/mlp_planted.py` only if it plugs in without modification; else omitted | planted latent | end-to-end sanity |

## 9. Deliverables

1. `explore/cifar_baselines/` (untracked): `sources.py` (section 2), `nulls.py`, `variants.py`, `transfer.py`,
   `planted.py`, `stats.py`, tests (analytic controls of section 3 as assertions; each planted benchmark's easy regime
   as a regression test), Modal coordinator `measure_b.py` (the `modal` skill; re-extracts raw Grams R; reads stored operators; adds
   the pixel cell, seed-1 final-checkpoint spectra, the 10,000-image extraction), a written `SCHEMA.md`, output in
   `explore/cifar_coherent/data/r20s0c/`. Nothing in `explore/cifar_coherent/*.py` changes behaviour; imports only.
2. Explorer (`explore/cifar_coherent/site/`): "Sets from" control with status tags, the agreement line, null draws
   selectable; data budget: a typical interaction <= 2 MB, site <= 150 MB (if the full matrix does not fit: all
   sources for kernel rule 0 only; baselines for other rules in the tab's numbers only).
3. New tab "Baselines and nulls": spectra with null bands; source x source agreement matrices for the 12 blocks and
   summary distributions over all blocks; ablation panels (agreement with default, stability, null excess, per
   factor); stability; linear-content and CKA views; planted-benchmark curves; transfer pilot. Captions: what is
   plotted, the null value, the status tag. No interpretation.
4. `NOTES.md`: every number's source, what was cut, deviations from this plan, spend.
5. `results publish` to the laptop.

## 10. Phases, agents, review

- P0 this plan -> fresh GPT-6 Astra (high) critique, incl. section 6 mathematics (`REVIEW_1_plan.md`) -> revised
  with the same reviewer until no material objection.
- P1 (parallel; code and smoke tests local, sweeps on Modal CPU; the transfer pilot (c) is reserved first in the
  budget): (a) `stats.py` + `sources.py` + `nulls.py` + tests; (b) `planted.py` + benchmark runs;
  (c) `transfer.py` + P8/P2 validation. Builders: Opus 5.5. Each reviewed by `model-router:adversarial-code-reviewer`.
- P2 Modal run on real data (detached coordinator, smoke first, spend recorded; expected < $10).
- P3 site: explorer control + tab (Opus 5.5 designs and writes; GPT accuracy review against the data;
  `gpt-6.1-terra` fresh-eyes read; browser check at 1440 and 390 px, zero console errors).
- P4 publish; handback listing what was and was not done, with the plan's deviations.

Order if time or budget binds (the transfer pilot's budget, $5 and one builder, is reserved up front and is not
displaced): section 2 on the first two scope tiers, nulls N3, N6, N7, N2b, benchmarks P0/P1/P3/P4/P8 -> section 4 ->
section 5 A, C, E, G -> the remaining scope tiers, nulls and benchmarks.

## 11. Not in this round

ViT and Mamba explorers (port the informative subset afterwards); SAE / trained dictionaries; a broad clustering zoo
(HDBSCAN, GMM, Ward, sparse PCA: IDEAS §9); changes to `THEORY.md` / `EXPERIMENTS.md`; gh-pages deploy; any conclusion
about which variant is "right".
