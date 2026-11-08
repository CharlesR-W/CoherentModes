# Toy visualization plan: seeing the construction on a d = 3 residual stream

Status: v1.1 plan, 2026-11-07, approved for implementation by Codex re-check; revised after Codex review (planted weights instead of relying on training; directional contrast panel; constant mode and complement; temperature from the scan; compute budget; block rule; five sections with the rest deferred). Purpose: not an experiment, a *picture* of the method in `THEORY.md` so the owner can see sites, views, kernels, operators, modes, loadings, cross-layer coherence and the gradient decomposition at a glance, and see the assumptions where they bite. Everything must be computable in seconds on CPU and plotted in 3D where the objects live in R^3.

## The toy network

A residual network with stream dimension d = 3, so every activation is a point in R^3 and, after the cosine normalization of THEORY §2, a point on S^2.

- **Inputs.** 2D points from six planted Gaussian clusters on a ring (fixed centers at angles k·60°, radius 2, isotropic std 0.25; 200 points each; N = 1200 sites; seed 0). Cluster ids colour plots and validate; they never enter the construction.
- **Architecture.** `h0 = W_in x + b_in` (R^2 → R^3), two residual MLP blocks `h_{l+1} = h_l + W2_l relu(W1_l h_l + b_l)` with hidden width 16, output `logits = W_U h2` (4 classes). Layers L = 0, 1, 2 are the three residual states.
- **Planted geometry (the default model).** Weights are *constructed*, not trained, so that the intended merge and split are guaranteed and the page can label them as planted: block 1 maps clusters 0 and 1 onto a common direction (merge: two blobs at layer 0, one at layer 1) and leaves the others apart; block 2 cuts cluster 2 along a planted line into two directions (split: one blob at layer 1, two at layer 2; the two halves are distinct inputs, so this is realizable). The construction is written down in `toy.py` with a comment per block and the acceptance gate below is asserted.
- **Trained variant (optional panel).** The same architecture trained with Adam (lr 0.01, 300 steps, batch 128, cross-entropy) on labels that coarsen the clusters the same way (0,1 → A; 2 cut → B,C; 3,4 → D; 5 → B). Classification does not force cosine collapse, so the trained model may or may not realize the merge and split; the page shows the acceptance-gate result honestly ("realized" / "not realized") and never captions a merge or split the gate did not pass. Dense checkpoints every 10 steps; at steps {50, 150, 300}, n = 8 stochastic resamples (restore model *and Adam state*, one minibatch step, mean displacement removed) for β_noise.
- **Acceptance gate for a merge or split** (used on both models): for the planted contrast c (±1/√|S| on the two groups, zero elsewhere), report ‖P^0 c‖, ‖P^2 c‖, ‖P^2 P^0 c‖; a merge is "realized" when ‖P^0 c‖ > 0.9 and ‖P^2 P^0 c‖ < 0.2, a split when ‖P^0 c‖ < 0.2 and ‖P^2 c‖ > 0.9. Labels are validation only.

All quantities follow `THEORY.md` v1.5 and `checks/cm_check.py`: shifted sphere kernel exp(β(⟨ĥ(x), ĥ(y)⟩ − 1)) (same operator after Sinkhorn, no overflow); Sinkhorn to tolerance 1e-10 on row sums (not a fixed 5000 iterations); P symmetric doubly stochastic; centered C P C; eigenpairs by dense `eigh` (N = 1200 is fine: one eigh ≈ 0.3 s); products applied as matrices (N = 1200, trivial); loadings a_i = Hᵀψ_i *including the constant mode* a_0 = Hᵀ1/√N and the complement, so the reconstruction identity is exact; gradient block decomposition for block 2's W1 with u = h_1 and δ = ∂L/∂z under the same mean-over-sites loss reduction as autograd, with the constant piece G_0 and complement shown and an assertion that Δᵀ U equals the autograd gradient and that constant + blocks + complement sum to it.

**Compute budget.** A timed compute-only prototype comes first and must finish in under 60 s on this CPU for: the planted model, three illustrative temperatures, a β scan of 12 values for layer 2 only, and the gradient decomposition. Checkpoint scans (if the trained panel is kept) analyse 8 checkpoints, not 30. JSON exports hold plotted data only: heatmaps downsampled to 120 × 120 by cluster-sorted block averaging, spectra, mode values per site, loadings, gate numbers.

**Blocks and bases.** A block is a maximal run of eigenvalues whose consecutive gaps are below 0.1 × the gap to its neighbours (rule stated on the page); if no gap isolates a block of dimension ≤ 6 the page says "no isolated block" and shows the top-6 subspace with that label. Modes shown individually are the actual eigenfunctions with their eigenvalues; a varimax-rotated basis of a block is shown separately, labelled "rotated block basis (not eigenfunctions)", with loadings rotated consistently. Signs fixed by making the largest-|value| site positive. In d = 3 the loading matrix has rank ≤ 3; loadings are coefficients, not orthogonal directions, and the page says so.

**Temperature.** The scan is primary. β_noise (trained model only) and β_nn are reported as marks on the scan with their values; the illustrative temperature for the mode figures is the scan value with the largest isolated-block gap at layer 2, declared on the page as a display choice. The identity-limit caption carries its condition (distinct normalized vectors; repeated directions leave averaging blocks).

## Figures, first pass (five sections; each panel captioned with the THEORY section and the assumption it illustrates)

1. **Sites and the metric choice.** 3D scatter of h0, h1, h2 in R^3 (faint unit sphere) and the normalized clouds, coloured by planted cluster; point size encodes the dropped norm. Caption: [A1] the site is the atom; [A2] the dot product is the one geometric commitment.
2. **Kernel → operator → scale.** Layer 2: downsampled heatmaps of K_β and P at three scan temperatures, cluster-sorted; eigenvalue curves λ_1..λ_12 of C P C over the 12-value β scan (log axis) with β_nn (and β_noise if available) marked; for each β the principal angles between the top block and the span of the six planted cluster indicators (indicator-subspace recovery, THEORY §4.3) and the block dimension; a small inset comparing Sinkhorn with degree normalization (top singular value of the layer product: 1 vs < 1). A diffusion-time control: the same spectrum at t = 1, 4, 16. Caption: the two limits (centered Gram at β → 0; identity at β → ∞ for distinct vectors), clustering as a finite window, blocks delimited by gaps, and why the shared measure matters.
3. **Preservation, merge and split.** Q^{0→2}: the top three nontrivial singular pairs as pairs of spheres (v_i on the layer-0 cloud, u_i on the layer-2 cloud) with σ_i, as the *preserved-structure* panel; beside it the **directional contrast panel**: the planted merge contrast and the planted split contrast each shown on both clouds with ‖P^0 c‖, ‖P^2 c‖, ‖P^2 P^0 c‖ and the gate verdict. Caption: THEORY §4.1; merge = contrast slow at L and annihilated at L′; σ alone cannot give the direction because Q and Qᵀ share singular values; planted contrasts are validation, not discovery.
4. **Loadings and reconstruction.** At the illustrative temperature, layer 2: the constant mode and the top isolated block's eigenfunctions as colour on the sphere, their loadings a_i as arrows (raw activations, declared), and the reconstruction tail ‖H − H_r‖² vs r including the constant term, truncating only at block boundaries. The rotated block basis as a second row, labelled. Caption: THEORY §1.3, §5; exact expansion; tail sum; blocks are canonical, individual vectors inside are not.
5. **Gradient and units.** Block 2's W1 at the planted model (gradient of the cross-entropy at the planted weights) and, if kept, the trained model at step 150: block pressure ‖G_B‖_F for the constant piece and each block, the signed interaction heatmap ⟨G_B, G_B′⟩_F, the truncation residual ‖G − G_r‖_F vs r computed directly, input loadings b_i as arrows on the layer-1 sphere and error loadings g_i as bars over hidden neurons. Units: the 16 hidden neurons' block profiles (heatmap), plus a **constructed rectification inset**: one eigenfunction ψ and ReLU(ψ) expanded in the same complete basis, with constant and block energies, labelled as constructed. Learned-neuron pre/post profiles shown only where they exhibit the effect. Caption: THEORY §7.1–7.2; gradient, not update; exact at block granularity.

Deferred to a second pass (stretch, in this order): the training slider over 8 checkpoints with the gate numbers and the top-block projector matched across checkpoints; a second seed with principal angles and centered alignment (β → 0 value checked against linear CKA on normalized rows); the output view as a labelled probability tetrahedron with the operator computed from unprojected √p and a Q^{2→out} panel; the d = 3 attention context toy for carried vs read.

## Outputs

- `viz/toy.py`: builds the planted model (and optionally trains the variant), computes everything, asserts the identities and gates, times itself, writes `viz/data/toy.json` (plotted data only).
- `viz/build_page.py`: builds a self-contained `pages/toy.html` from the JSON using the report template in `viz/report/` (see `viz/TEMPLATE.md`), vendored Plotly, no network access at view time; embeds the diagrams below.
- `viz/report/verify_page.sh`: headless screenshots of each section to `viz/screenshots/`.

## Division of labour

- Plan review: GPT (Codex) before any code (done once; this is the revised plan).
- Template: a separate agent copies and de-crufts the CellsDevelopmental report machinery into `viz/report/`.
- Computation (`toy.py`): a coding agent; reuses the conventions of `checks/cm_check.py`, asserts identities on the toy, meets the 60 s budget.
- Visual design, diagrams and page: an Opus agent with the `dataviz` skill, after the data exists.
- Before handing over: a GPT read of the page captions against THEORY.md.

## Known limits of the toy (stated on the page)

- d = 3 cannot show superposition of many features; the toy shows the operator machinery, not the dictionary regime.
- The dense path only; no sparse-graph or Nyström estimand is exercised.
- The planted model's merge and split are constructed; the trained variant shows whether training produces them, and may not.

## Explanatory diagrams (added after owner request; Opus visual-design agent)

Separate from the data figures: schematic drawings of the *objects the theory introduces* (not of the owner's own motivating ideas), drawn as SVG so they can live in the same page and in future write-ups. Each is a single clean picture with a two-line caption pointing at a THEORY section.

D1. **Sites and views.** A short token sequence with one site highlighted; from that site, arrows to its activation at layers 0, 1, 2 (the same site at every layer, [A1]) and to its views: the geometric view (a point on a small sphere), the output view (a point on the positive orthant of the sphere, the square roots of probabilities; not the simplex itself), the context view (its past window), the action view (its local Jacobian). THEORY §1–2.

D2. **From a view to an operator.** A row of N sites; a kernel matrix K between them (drawn as an N × N grid with warm/cool cells); the Sinkhorn step to P (row and column sums shown equal to one); the eigenfunctions ψ_i drawn as colour patterns over the row of sites. THEORY §3.

D3. **Loadings: the two faces of the SVD.** Left: a mode ψ as a function on sites (a bar per site). Right: the same mode as a direction a = Hᵀψ in activation space (an arrow in R^3). Between them the identity H = Σ ψ_i a_iᵀ. THEORY §1.3, §5.

D4. **A mode lights up weights and neurons.** A small drawn network: residual stream at layer L (three units), an MLP with six hidden neurons, residual stream at layer L+1, with the weight matrices W1 and W2 drawn as edge bundles. For one chosen mode: the input loading b = Uᵀψ shades the layer-L units; the hidden neurons are shaded by their mode profile ⟨n, ψ⟩ (the pre-activation profile and the post-ReLU profile as two half-discs, to show rectification spreading it); the gradient piece g bᵀ shades the W1 edges by |g_k b_j|, labelled on the figure as a *basis-dependent gradient contribution*, not weight magnitude and not an identified mechanism (only block sums and block interactions are invariant); the error loading g shades the hidden pre-activations. A second copy for a second mode shows a different set of edges lit, with the overlap ⟨b_1, b_2⟩⟨g_1, g_2⟩ called out as the signed interaction. THEORY §7.1–7.2. This is the picture the owner asked for: "the loading would highlight the weights vs the neurons".

D5. **Cross-layer coherence.** Two spheres (layer L and L') with the same N sites; a blob at L that stays a blob at L' (coherent, σ ≈ 1), two blobs at L that become one at L' (merge), one blob at L that becomes two at L' (split), with the direction convention of THEORY §4.1 written on the figure.

D6. **Carried versus read.** A sequence with a site x; its past window; two targets drawn at different places: the stream's own future *at the same layer* (carried) and the update *at the next layer* at x (read); the canonical-correlation SVD between past and target drawn as a small bipartite "whitened cross-covariance" diagram; the predictive-risk ladder as a tiny descending staircase labelled as the idealized Bayes-risk schematic (fitted held-out risks need not decrease or reach zero) with the caution that it is predictive information, not architectural use. THEORY §6.3–6.4.

D7. **The gradient as a sum of rank-one pieces.** G = Σ_i g_i b_iᵀ drawn as a stack of thin outer-product rectangles, with the block grouping (brackets over near-degenerate modes) and the note that only block sums are invariant. THEORY §7.2.

Design constraints: one consistent palette and typography shared with the data figures (use the `dataviz` skill); no decoration; every symbol on a diagram appears in THEORY's notation; captions name the assumption tag where one applies. SVGs go in `viz/diagrams/` and are embedded in the page before the corresponding data figure.
