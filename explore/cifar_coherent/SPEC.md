# Shareable coherent-sets mini-writeup (ResNet-20 / CIFAR-10): spec

Owner request (2026-11-11): turn the local explorer (`viewer_template.html`, `build_viewer.py`) into a public,
good-looking mini-writeup with slick interactivity, hosted on GitHub Pages (`gh-pages` branch of the public
`CharlesR-W/CoherentModes` repo), to be linked from a research Zulip. Audience: people *minimally* following the
owner's research; comfortable with linear algebra and ML, not with transfer-operator theory.

Status of the science: exploratory side study, one model, one seed. Not an `EXPERIMENTS.md` stage. The page must say so.

## 0. Fixed by the owner

- Title (joke, owner's): "Spectral Clustering of Tempered Cross-Hankel for Unsupervised Persistent Neural Ontology
  Identification (This name is a joke)", or a minimally modified version that is technically correct (see 4.3).
  Byline "Charles R-W", dated, exploratory banner, link to the repo.
- Overview tab content: coherent sets as "things the network treats similarly" (with only a tiny, unassuming note
  that this is the owner's step towards "finding the network's ontology"); brief relation to spectral clustering and
  to Hankel; static spectral clusters vs the dynamic-Laplacian ones and how to interpret the difference; how the
  plots are coloured (incl. exactly how SEBA colours are assigned); say out loud that the 2-d embedding is UMAP;
  explain the kappa selector (kNN-scale vs half-mass vs global). Appropriate amount of math, no heavy technicalities.
- Selecting an eigenfunction / set shows a handful of representative images without clicking points.
- Layout: the right sidebar must never cover the plots; auto-sizing must work at common widths.
- New analysis: "vertical" coherent sets = points at a fixed layer through training (time blocks).
- New plots: linear decodability per layer and per training time.
- Fewer checkpoints in the explorer grid.
- Neuron loadings: per layer, which spatially-averaged channels correlate with the selected eigenfunction/set, linear
  R^2 of the function on the layer's channels, and alignment with the classifier weight rows at the pool site.
- "Commit to the bit": nice figures, gifs incl. a very slow "how to read the plots" gif (in the manner of the
  "central flows" paper's animations) and figures showing how things are calculated.
- Design and writing by Opus 5.5; iterative critique by GPT (Codex) models.

Repo-level constraints that still apply (root `CLAUDE.md`): nothing is trained on top of the model inside the
construction (ridge probes are closed-form *fits*, used only to *validate*, and are labelled so); class labels never
define a kernel or basis, only colour and validate; no sparsity premise; do not use the coined vocabulary of
`THEOREMS.md`/`SPARSITY.md`; working language is sites, views, operators, blocks, measured regimes.
Do not touch `explore/vit_coherent/` (another session's work in progress).

## 1. Existing pipeline (math changes only as listed in section 2)

`coherent.py`: per cell (checkpoint t, site s): centre activations over the 2,000-image probe set, L2-normalise rows,
C = cosine Gram; vMF kernel K = exp(kappa (C - 1)); (old: alpha = 1 density normalisation, M = D^-1/2 Ktilde D^-1/2;
new: Sinkhorn, section 2). Block of cells: average the operators, take leading eigenvectors psi_0.., SEBA rotates psi_0..psi_{r-1} into r sparse non-negative vectors; a point's set is the argmax vector if
its value > 0, else "unassigned". kappa rules (`measure.py::_modes`), per cell, k = 15: kNN scale kappa = 1/d_k;
half-mass kappa = log(N/k)/(d_pair - d_k); global constants. UMAP: `umap.AlignedUMAP` across the 12 sites within a
checkpoint on the top-50 PCs of the centred normalised activations, then Procrustes across checkpoints.
Existing data: `data/r20s0/` (14 checkpoints; `umap_*.npy`, `pcs_*.npy`, `stats_*.json`, `probe.npz`, final-step
`decodability.json`). Model run on Modal volume `cm-cifar-coherent:/runs/resnet20-relu-seed0` (73 checkpoints).

## 2. New measurement (tag `r20s0b`, Modal, detached coordinator; follow the `modal` skill)

Explorer checkpoints `STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]` (T = 8). Sites S = 12 as before. N = 2000,
same probe (seed 0).

**Two changes to the math, both from the plan review (2026-11-11), both to be stated on the Method tab:**

- *Normalisation.* Replace alpha = 1 + `D^-1/2 K D^-1/2` by symmetric Sinkhorn scaling of the vMF kernel to a
  symmetric doubly stochastic operator M (THEORY.md Section 3.1; formula in `checks/cm_check.py`). Reason: with
  per-cell degree normalisation each cell has its own stationary vector (sqrt of degree), so the average over a block
  has no exact trivial mode and density mismatch leaks into the spectrum. With Sinkhorn every cell shares the uniform
  measure: psi_0 = 1/sqrt(N) exactly, eigenvalue exactly 1, for every block. Keep `diffusion_operator` (old variant)
  in the file, and report in `normalisation_compare.json` the subspace agreement (mean cos^2 of principal angles,
  r = 3 and 6, psi_0 excluded / constant projected out) between old and new for all layer blocks at the final step
  under each kappa mode, so the page can say how much the change mattered.
- *Self-loops* (2026-11-12, from first measurements + plan review). With the diagonal retained (K_ii = 1) and
  neighbours at cosine distance ~0.83, middle-layer operators under the half-mass rules are within ~1e-8 of the
  identity, so a block average is dominated by its least-lazy cells. Construction now: set K_ii = 0, symmetric
  Sinkhorn (float64 / log-domain; failing the row-sum tolerance is a failed measurement) to a zero-diagonal doubly
  stochastic P ("off-diagonal affinity diffusion", a stated *variant* of THEORY.md 3.1, which keeps the diagonal),
  then the fixed-lazy operator **M = (I + P)/2**, used everywhere (block averages, products, retention, SEBA input).
  M is PSD-shifted so that an eigenvalue of P near -1 (near-bipartite alternation) is not mistaken for a slow mode;
  every cell has self-mass exactly 1/2; lambda_M = (1 + lambda_P)/2 and eigenvectors are those of P. f^T (I - P) f =
  1/2 sum_ij P_ij (f_i - f_j)^2. This equalises self-loop laziness only; spectral gaps and mixing rates still differ
  between cells and are part of what is measured. `meta.json` gains `self_weight[m][t][s]` = median M_ii of the
  diagonal-retained Sinkhorn operator, and `normalisation_compare.json` compares, at the final step, the new
  construction against (i) the old alpha = 1 degree normalisation and (ii) diagonal-retained Sinkhorn.
  Extra tests: unequal-density cloud, two nearly disconnected groups, a nearly bipartite graph (Sinkhorn residuals,
  both spectral ends of P, product singular modes are slow rather than alternating under M).
- *SEBA.* Port the authors' reference algorithm faithfully (https://github.com/gfroyland/SEBA, Algorithm 3.1 of
  Froyland-Rock-Sakellariou 2019): soft-thresholded columns are normalised inside the iteration, the *thresholded*
  vectors are returned, sign fixed, scaled to max 1, ordered by decreasing reliability (minimum entry), small negative
  entries then clipped for display. Cross-check against the reference implementation on fixed inputs in the test file.

Kernel rules (M = 5), chosen 2026-11-12 from `sweep.json` (19 candidate rules x 15 real cells) using label-free
diagnostics only (participation number of psi_1, spectral gap, smallest SEBA set, Sinkhorn stability); class overlap
in the sweep is validation and did not enter the choice. Distances are cosine distances d = 1 - C; diagonal zero.
0. `kNN graph, k = 15` (**default**): K_ij = 1 if j is among i's 15 nearest neighbours or i among j's, else 1e-6
   (floor so the Sinkhorn scaling exists). Rank-based; the same neighbourhood size as the UMAP layouts.
1. `per-point perplexity 100`: for each i, kappa_i such that p(j|i) ~ exp(-kappa_i d_ij), j != i, has perplexity
   100; K = (p(j|i) + p(i|j))/2 (the t-SNE affinity).
2. `per-point perplexity 300`: same, coarser.
3. `kNN scale` (one kappa per cell, kappa = 1/d_15, about 1.2 in trained middle layers): nearly flat kernel,
   PCA-like spread modes with weak eigenvalues (1 - lambda_1 ~ 0.45). Kept as the flat reference.
4. `half-mass, calibrated` (one kappa per cell, solving median 15-NN share of non-self kernel mass = 1/2; kappa 26
   to 43 in trained middle layers): leading modes sit on near-duplicate pairs (participation 2). Kept as the
   cautionary case; it is the cleaned-up version of the rule the first explorer used by default.
Sweep summary at the final checkpoint: single global kappa gives spread modes only below ~8 (weak eigenvalues) and
pair-localised modes from ~12 up in b5/b8; no global-kappa window gives well-separated, non-localised modes in the
middle layers. Per-point perplexity 100/300 and the kNN graphs give spread psi_1 (participation ~900 to 1300),
1 - lambda_1 ~ 0.05 to 0.13, and smallest SEBA-8 set >= 86 images at every tested final cell; perplexity 15/30 still
produce a 2-image SEBA set somewhere. `meta.json`: `kappa[m][t][s]` is the per-cell kappa for rules 3 and 4, the
median kappa_i for rules 1 and 2 (with `kappa_q10`, `kappa_q90`), and null for rule 0. `nn_mass` as before.

Block families and index conventions (0-based; these arrays are the contract with the site builder):

- `layer_pairs`: all (a, b), 0 <= a <= b < S, row-major (P_L = 78). Block = sites a..b at checkpoint t.
- `time_pairs`: all (c, d), 0 <= c <= d < T, row-major (P_T = 36). Block = checkpoints STEPS[c..d] at site s, uniform
  weight per listed checkpoint.
- Block operator: Mbar = mean of the cells' M. Eigenpairs of Mbar, descending. psi_0 is constant and is not stored.

Output `data/r20s0b/` (schema_version 2):

| file | arrays |
|---|---|
| `meta.json` | `schema_version`, `construction_fingerprint` (shared rule/normalization/precision/SEBA contract hash), `steps`, `sites`, `modes` (names), `kappa[m][t][s]`, `kappa_q10[m][t][s]`, `kappa_q90[m][t][s]` (null for mode 0; per-point quantiles for modes 1–2; identical per-cell scalar for modes 3–4), `nn_mass[m][t][s]` (actual symmetric pre-Sinkhorn kernel), `self_weight[m][t][s]` (null for modes 0–2; diagonal-retained vMF Sinkhorn diagnostic for modes 3–4), `seba_r` = [2,3,4,6,8], `n_vecs` = 8, `layer_pairs`, `time_pairs`, `n`, `classes`, rule text |
| `layer_<step>.npz`, one per t | `vals (M,P_L,9) f32` (lambda_0..lambda_8); `vecs (M,P_L,8,N) i16` = psi_1..psi_8, psi_k = unit-norm vector, stored round(32767 psi_k / max\|psi_k\|); `scale (M,P_L,8) f32` = max\|psi_k\|; sign: largest-magnitude entry positive; `seba_label (M,P_L,5,N) i8`, -1 = unassigned else 0..r-1 in reliability order; `seba_rot (M,P_L,5,8,8) f32`: for r = seba_r[i] the top-left r x r block is the final rotation R acting on the columns [psi_0, psi_1..psi_{r-1}] (unit norm, psi_0 = 1/sqrt N), such that the SEBA vectors are recomputed as: Z = V R; soft-threshold at mu = 0.99/sqrt N; normalise columns; sign so column sum > 0; divide by column max; columns then permuted by `seba_perm (M,P_L,5,8) i8` into reliability order; `retention_psi (M,P_L,8,S) f32` = psi_k^T M_(t,s') psi_k for every site s' of checkpoint t (inside and outside the block); `retention_seba (M,P_L,5,8,S) f32` = v^T M v / v^T v for each SEBA vector v, NaN padded |
| `time_<site>.npz`, one per s | same names with P_T in place of P_L and the last axis of the retention arrays running over the T checkpoints at site s |
| `points_<step>.npz`, one per t | `loss (N) f32`, `pred (N) i8`, `correct (N) bool`; per site `chan_<site> (N,C) u8` = spatial mean of each channel (logits: the 10 logits; pool = b9's), linearly quantised per channel with `chan_<site>_lo`, `chan_<site>_hi (C) f32`; `fc_w (10,64) f32`, `fc_b (10) f32` |
| `avgprod.json` | modes 0 (default union-15NN graph) and 1 (per-point perplexity 100); for each t and layer pair a < b, r in {3, 6}: with chronological product Q = M_b ... M_a (column-vector convention, layer a applied first), mean cos^2 of principal angles between span(psi_1..psi_r of Mbar) and (i) the top-r non-trivial *right* singular vectors of Q, (ii) the top-r non-trivial *left* singular vectors of Q, (iii), (iv) the same for the two-end product M_b M_a |
| `normalisation_compare.json` | modes 3 and 4 only; all final-step layer blocks, r in {3,6}; `agreement` = old alpha=1 versus final construction, `retained_agreement` = diagonal-retained Sinkhorn versus final construction; discard stationary mode and project out constant before principal-angle comparison |
| `decod.json` | protocol of `decodability.py` (10k train fit, 10k test eval, centred + L2-normalised features, closed-form ridge on one-hot targets with GCV lambda, weighted cosine kNN) for all 14 checkpoints of `data/r20s0` x 12 sites x views {flat, gap}: `ridge_acc`, `knn_acc`. A validation fit, not part of the construction |
| `curves.json` | step, train/val acc and loss for all 73 checkpoints (run manifest) |
| `probe.npz`, `umap_<step>.npy` | copied from `data/r20s0` (same probe, same checkpoints; do not recompute UMAP) |

Per-point loss/correctness (`points_*.npz`) is an owner addition: eval mode, no augmentation, cross-entropy.

First deliverable of the pipeline agent, before any Modal run: `data/_fixture/` with exactly this layout produced by
the real code path on synthetic activations (N = 2000, any quick planted structure), so the site builder can be
developed against it. Then the real run.

Tests (`test_coherent.py`): Sinkhorn rows sum to 1 and psi_0 is constant on a heterogeneous-density input; SEBA matches
the reference on fixed inputs; planted time-block check (a partition coherent across the block is recovered; one
present in a single cell is not the top split); the documented recomputation of SEBA vectors from `vecs`, `scale`,
`seba_rot`, `seba_perm` reproduces `seba_label` on >= 99% of points (quantisation is the only allowed source of
disagreement; the int16 fallback is now required after int8 failed the high-dimensional regression).
Budget: <= $25 Modal (including the later additions). Record spend and the commands in `README.md`.

Loadings (computed in the browser from `points_*.npz`; definitions fixed here). Target f = the selected function:
psi_k, or SEBA vector j recomputed as above (the full vector, not the winner-masked one). For a cell (t, s) with
channel matrix X (N x C, columns centred): per-channel Pearson correlation corr(f, X_c); R^2 = in-sample ordinary
least squares of centred f on X (descriptive, not held out; chance level about C/N <= 3.2%, print it). "Alignment
with the classifier" = corr(f, logit_c - mean_c' logit_c') for each class c, i.e. the logits-site loadings with the
class-mean logit removed. Which cells: the R^2 is shown for every cell of the grid (T x S); the per-channel bars
for one cell chosen by the reader (default: last cell of the block).

### 2b. Owner additions (2026-11-11, later; "cute things", lower priority than everything above)

- `points_<step>.npz` gains, per probe image: `jac_fro (N) f32` = Frobenius norm of d logits / d input pixels (10 x 3072
  Jacobian, eval mode, normalised-input space); `grad_dot (N) f32` = g_i . gbar and `grad_cos (N) f32` = cosine, where
  g_i is the per-sample gradient of the cross-entropy loss w.r.t. all trainable parameters (eval-mode BatchNorm) and
  gbar is the mean of g_i over the 2,000 probe images; `grad_norm (N) f32`. Explorer colour-by gains "input-Jacobian
  norm" (log scale) and "gradient alignment with the mean gradient" (cosine; dot product as an option), each at the
  row's own checkpoint or the final one, like loss.
- **Second network** (layout only; the owner wants it just to re-UMAP and eyeball "universality"): train seed 1 with
  the identical recipe (`modal_train.py`); at the same STEPS and the same probe images compute only the activations'
  top-50 PCs and the AlignedUMAP + Procrustes layouts into `data/r20s1b/` (`umap_<step>.npy`, `pcs_<step>.npy`, plus a
  small `meta.json` with steps, sites and val accuracy per step). No spectra, points, loadings or decodability for
  seed 1. Explorer gains "layout from: seed 0 | seed 1": colours always come from seed 0, the layout from either;
  point index i is the same image in both.

### 2c. Question-led analyses (owner, 2026-11-12)

All plots other than the main explorer are organised under the question they answer (tab "Questions", which absorbs
"Decodability"). `site/analysis.py <tag>` (writer-owned) reads `data/<tag>/` and writes `site/assets/questions.json`
plus static figures; the site may render interactive charts from that JSON.

- **Set overlap.** For two blocks with SEBA sets {A_i}, {B_j} (same r, same kappa mode, hard assignments, unassigned
  points in no set): J_ij = |A_i n B_j| / |A_i u B_j|; match sets by the Hungarian assignment maximising total J;
  report mean matched Jaccard and the per-set values, next to a permutation null (same set sizes, random members).
  The explorer shows the matched Jaccard of the selected block's sets against classes and against a comparison block.
- **Q: when does the network learn its categories?** At each site, SEBA sets of the final checkpoint's single-cell
  block vs each earlier checkpoint's single-cell block (matched Jaccard, heatmap site x step and lines), and each
  checkpoint's sets vs the CIFAR classes (labels as validation only).
- **Q: does a block's set structure carry over to its successors?** At a fixed checkpoint, single-site SEBA sets of
  site a vs site a' (12 x 12 matched-Jaccard matrix per checkpoint; adjacent-site line as the summary), and block
  a..b vs the following block where informative.
- **Q: how do through-training coherent sets at a fixed layer relate to per-sample gradients?** New pipeline output
  `gradgram_<step>.npy (N,N) f16`: cosine Gram of the per-sample parameter gradients (same g_i as `grad_cos`). For
  the SEBA sets of time blocks at each site: mean gradient cosine within sets vs between sets at each checkpoint,
  against the same statistic for CIFAR classes and for random sets of the same sizes; and each set's mean alignment
  with the mean gradient. Gradients use the labels through the loss, so this relates the label-free sets to the
  training signal; it does not enter the construction.
- **Q: where and when is class linearly decodable?** The decodability plots (by site per checkpoint, by step per site).

## 3. Site

Source in `explore/cifar_coherent/site/` (tracked); `build_site.py <tag>` writes the static site to
`explore/cifar_coherent/viewer/site/` (untracked) which is deployed as the root of `gh-pages` with `.nojekyll`.
Plain HTML/CSS/JS, vendored libraries only (no CDN at runtime except optionally MathJax/KaTeX, prefer vendoring),
works from `https://charlesr-w.github.io/CoherentModes/` (relative URLs only). Data as lazily fetched binary/JS chunks;
a typical interaction should fetch <= ~1-2 MB; whole site <= ~150 MB; first paint of the overview needs no big chunk.

Tabs (hash-routed, one page):
1. **Overview**: the mini-writeup (section 4), with embedded live mini-figures and gifs.
2. **Explorer**: the grid (rows = checkpoints, columns = sites, each cell a UMAP of the same 2,000 images).
   A toggle selects the block family: *across layers* (pick checkpoint, L'..L) or *through training* (pick site,
   t'..t). The selected block's eigenfunctions / SEBA sets / nodal domains colour every cell. Controls: block family,
   kappa rule (with an inline explainer), colour-by, k or r. Side panel: block spectrum, legend with class make-up,
   **representative images** for the current selection (eigenfunction: most positive / most negative / near zero;
   set: highest-membership members, plus a few boundary members), hovered point, **channel loadings** for the current
   selection (per-site R^2 strip; top channels at a chosen site; alignment with classifier rows), site statistics.
   Responsive: cell size derives from the available width via ResizeObserver; below a breakpoint the side panel
   moves under the grid or into a drawer; nothing overlaps at 1280, 1440, 1920 px wide, and it is usable at 390 px.
3. **Decodability**: ridge and kNN accuracy by site (one line per checkpoint) and by training step (one line per
   site), flat vs GAP views; training curves; optionally how well SEBA sets agree with classes per block (purity/NMI
   computed at build time), clearly labelled as validation against labels.
4. **Method**: exact formulas, kappa rules with the measured kappa table, SEBA, UMAP settings, what is and is not
   canonical (individual eigenvectors inside a near-degenerate group are not; spans are), limitations, how to reproduce.

Colour: one grammar across the site; tokens in `site/palette.json` (class colours, set colours, diverging map),
used by both the JS and the matplotlib figure script. The designer may change the values, not the mechanism.

Figures/gifs (`site/make_figures.py`, matplotlib, real data, fixed axes): (a) pipeline diagram: activations ->
cosine Gram -> kernel -> operator -> average over a block -> eigenvectors -> SEBA sets; (b) a very slow annotated
"how to read the explorer" gif; (c) static-vs-dynamic illustration on real cells: a split that exists at one layer vs
one that persists through the block; (d) filmstrip + gif of a time block through training; (e) kappa: what the
kernel's neighbourhood looks like under kNN-scale vs half-mass (e.g. kernel weight vs neighbour rank). The designer
and writer may alter this list with reasons.

## 4. Overview copy: points that must be right

4.1 Coherent set, informally: a group of inputs the network keeps near one another, and away from the rest, across
    the cells of a block. Precisely what is computed: for a function f on the probe images, f^T (I - Mbar) f is the
    *average over the block's cells* of the per-cell diffusion penalty f^T (I - M_cell) f. Leading eigenvectors of Mbar
    minimise that average. It is an average, not a maximum: a split that is cheap in most cells and expensive in one
    can still rank high, so "persists through the block" is an interpretation to check, and the page gives the check:
    the per-cell retention strip (psi^T M_cell psi for each cell). Do not say members are "mutual neighbours in every
    layer" or that the boundary is small in every layer. A single cell's leading eigenvectors are ordinary spectral
    clustering / diffusion maps of that layer ("static"). Averaging over a block (in the spirit of Froyland's dynamic
    Laplacian and Banisch-Koltai space-time diffusion maps, here with Sinkhorn-normalised kernels and uniform weight on
    each image's trajectory) demotes a split that exists at one layer only. How to read: eigenvalue near 1 = low
    average penalty over the block; a wide block with a surviving near-1 eigenvalue and a flat retention strip = a
    distinction carried across those layers (or, for time blocks, kept at that layer through training).
4.2 Colouring. Eigenfunction: diverging colour of psi_k scaled by max |psi_k|. Nodal domains: sign of psi_k. SEBA
    (Froyland, Rock, Sakellariou 2019): the leading r eigenvectors (including the constant one) are rotated so that,
    after soft-thresholding, they are as sparse as possible; the result is r sparse, mostly non-negative vectors;
    each point takes the colour of the vector with the largest value at that point; points where all r are zero are
    grey "unassigned". Colours are computed from the *selected block only* and painted on every cell. Individual
    psi_k inside a group of close eigenvalues are not canonical; the span is, which is what SEBA uses.
4.3 Hankel, as an analogy only. The Hankel matrix of a process is the past x future cross matrix; its SVD gives a
    state that mediates the two. The cross-layer relative is the *product* of layer operators Q = M_b ... M_a
    (diffuse at layer a, then at the next, ...), whose singular vectors are Froyland-style coherent pairs; THEORY.md
    builds on that product. This explorer uses the *average* instead; note ((M_a + M_b)/2)^2 contains the symmetrised
    cross term M_a M_b + M_b M_a. Report the measured agreement from `avgprod.json`. Do not claim the two are the same
    object, do not call the average a Hankel estimator, and the title must not assert which one is computed falsely.
4.4 Kernel rule (the selector the first explorer called kappa). kappa is the kernel's inverse temperature on cosine.
    What the sweep showed: in the trained middle layers the cosine geometry is high-dimensional and nearly
    equidistant (15th neighbour at distance ~0.83), so one temperature per cell gives either a nearly flat kernel
    (kNN scale: kappa ~1.2, spread PCA-like modes with weak eigenvalues) or a kernel dominated by near-duplicate
    pairs (half-mass: kappa 26 to 43, modes on 2 images). Rules that adapt per image (a 15-nearest-neighbour graph;
    per-point perplexity) avoid both; the kNN graph is the default. State this as a finding about this construction on
    this network, with the sweep figure, and keep both failing rules selectable so the reader can see them fail.
    Quote numbers from `sweep.json` and `meta.json`. The participation number flags localised modes.
4.5 UMAP is display only: it is a nonlinear 2-d layout of each cell, aligned across cells; no quantity on the page is
    computed from the 2-d coordinates.
4.6 Honest limits: one model, one seed, 2,000 probe images, CIFAR classes used only for colouring and validation;
    cosine erases activation norm; kappa rule changes the answer and is exposed rather than hidden.

## 5. Review protocol

- This spec: `model-router:plan-reviewer` before work starts.
- Pipeline code: `model-router:adversarial-code-reviewer`.
- Overview and Method copy: accuracy review against the code by `model-router:gpt-6.1-sol`; fresh-eyes readability by
  `model-router:gpt-6.1-terra` (no context beyond "a reader who has seen one talk about this research"). Iterate with the
  same reviewer until no material item remains.
- Visual check: screenshots of every tab at 1280/1440/1920 and 390 px, zero console errors.
