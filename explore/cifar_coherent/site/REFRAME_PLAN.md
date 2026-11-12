# Reframe, rename and harmonicity test (2026-11-12, owner's decisions)

## Owner's decisions (fixed)

- **Collaborator:** Brianna Grado-White. Stefan is *not* the collaborator. The activation-cells observation:
  credit is still to be confirmed with the owner. Draft with `<!-- CREDIT-TBD -->` beside the sentence and
  neutral wording ("an observation that …"). Name no one for it yet.
- **Activation cells:** interpolating between any three real-data activations shows Voronoi-cell-like
  structure, with sharp jumps rather than smooth transitions. The owner spent a long time trying to use this to
  decompile LLMs, and this work is a synthesis that grew out of that.
- **Rename "cell":** to the owner, "cell" means activation cells. In all user-visible text (prose, captions,
  explorer UI, tours, figures, chart labels), a (checkpoint, site) pair is now a **snapshot**. A
  single-cell block becomes a **single-snapshot block**. A picture in the explorer grid is a **panel**.
  "Cell" is used only for activation cells. Internal identifiers (`data-live="cells"`, fact names, JSON
  keys) stay as they are. Changed URL anchors (`glossary-cell` → `glossary-snapshot`,
  `glossary-single-cell-block` → `glossary-single-snapshot-block`) get aliases.
- **Rebrand:** the analysis is called **Neural Coherent Sets** (a working name). It is the owner's analysis
  for neural networks, inspired by spectral clustering, Froyland's coherent sets and SEBA, and Hankel methods.
  Present it as the owner's own construction. Method still credits the components it is built from (the
  dynamic Laplacian and SEBA) as building blocks, in the usual way.
- **Motivation reframe** (Overview `motivation`, and the lede where it helps), roughly in thirds:
  1. making activation-cell analysis tractable (cells are hard to measure or decompose directly);
  2. spectral clustering, plus the hypothesis that cells are interpretable as clusters on which the network
     does a limited, harmonic label propagation;
  3. tying this to principled objects: coherent-structure identification, and Hankel-based views of
     computation. Sketch WFA/WTA inference as program reconstruction, the ambition of doing that for richer
     programs (Turing machines), and dynamical systems' own Hankel and Koopman methods, which
     Pughe-Sanford et al. suggest are a good perspective.
  
  The owner prefers continuity of thought over technical precision in the visible text ("it's really a
  marvelous unification"). Be loose but never false. Precise statements stay in disclosures; the earlier
  review's corrections stay there.
- **The hypothesis, in our formulation:** a *cell* is a region of a layer's activation space where the rest
  of the network computes nearly the same thing. If most neighbourhood edges stay inside cells, any downstream
  function F satisfies F ≈ P F (harmonic, as in label propagation; Zhu, Ghahramani and Lafferty 2003) except
  across boundaries. Its penalty comes only from boundary-crossing edges, so cells are low-cut sets, and the
  slow eigenvectors approximately span the cell indicators. Persistence across layers or training makes them
  coherent sets, which SEBA reads off. This is a motivating hypothesis; Findings carries a first test
  (below).

## Harmonicity test (design v3, revised after two plan reviews)

The v1 review's verdict: energy, spectral concentration and set constancy do not tell cells apart from ordinary
smoothness on a kNN graph, and the linear reference was not a valid null. v2 splits the work into a labelled
**mechanism test**, which is primary and grounded in the owner's own activation-cells observation (sharp jumps when
interpolating between real activations), and a calibrated **descriptive** part, which is secondary.

### Part B (primary): interpolation test, a labelled mechanism-validation intervention

The hypothesis H, made concrete: real activations sit in cells (the jumps along interpolations). If the network
does a limited harmonic label propagation within coherent sets, then **crossing a cell boundary between two images
of the same coherent set changes the downstream computation less than crossing one between images of different
sets, at matched endpoint distance and matched endpoint outputs.** Coherent sets would then be groups of cells that
the rest of the network treats alike.

- **Setup.** Final checkpoint (step 64,000). **Predeclared sites: b2, b5, b8.** Each has a nonlinear suffix:
  b8's suffix still contains block b9. **b9 is an exact affine negative control**, since its suffix is global
  average pooling plus the linear head, so every b9 path is affine in τ. The sets are the hard SEBA sets of the
  **single-snapshot block at the same site** (default kNN rule, r = 4). Outputs are computed strictly downstream,
  so they are target-excluding.
- **Paths.** For a pair of probe images (i, j), take the raw activations h_ℓ(x_i) and h_ℓ(x_j) (re-extracted at
  full precision from the local checkpoint and stored probe), and form the straight line between them. Run the
  network's suffix in eval mode at **two resolutions, T = 64 and T = 256** equally spaced points, and record the
  logits z(τ).
- **Pairs.** For each site, 300 within-set and 300 crossing-set pairs (assigned images only), matched on endpoint
  cosine distance in the site's view **and** on ‖z(1) − z(0)‖ (bins). A secondary stratum keeps only pairs whose
  two endpoints get the same predicted class.
- **Statistics, defined to see sharp transitions, not curvature.** Let m_T = max over steps of ‖Δz‖ at resolution T.
  - **Sharpness index:** σ = log(m₆₄ / m₂₅₆) / log 4. Analytically, σ = 0 for a jump, whose step size does not
    shrink with resolution, and σ = 1 for any smooth path, whose largest step scales as 1/T. b9 must give σ = 1
    up to numerical error (control).
  - **Jump magnitude:** S = m₂₅₆ / ‖z(1) − z(0)‖₍median over all pairs₎, the size of the largest single
    transition in units of a typical endpoint output difference.
  - Descriptive only: excess variation EV (backtracking or curvature), the number of predicted-class changes, the
    location τ* of the largest step, and the transition width (the τ-span holding the middle 80% of Σ‖Δz‖).
- **Calibration.** Analytic, from planted paths (a step; logistic transitions of width w ∈ {1/256, 0.01, 0.05,
  0.2}; a linear path) passed through the same σ and S code. Empirical, from b9 (affine), the step-0 network, and
  a resolution check at T = 1024 on 50 paths, to confirm that σ ≈ 0 paths really are jumps.
- **Decision rule.**
  1. **Premise**, per site: the fraction of crossing-set paths with σ < 0.5. Below 10%, the site has no cell-like
     jumps at this resolution and the test is *inapplicable* there; that is reported as a finding in itself.
  2. **Primary**, at applicable predeclared sites: the within-minus-crossing difference in median S, with a
     bootstrap CI over pairs.
     - **H-consistent:** within < crossing at every applicable site, with CIs excluding 0.
     - **H-inconsistent:** within ≥ crossing at two or more applicable sites.
     - **Otherwise:** inconclusive.
  
  Everything else is secondary and descriptive.
- **Robustness:** perplexity-100 sets, and the step-0 network as a reference.

### Part A (secondary, descriptive): downstream smoothness, calibrated

- **Targets.** Re-extract full-precision logits, and the top 10 principal-component scores of the **centred,
  row-normalised** view of each later site (unit-variance scores; report the variance they retain). This needs
  re-extraction, because the local caches are quantised or float16. Rebuild the operators from full-precision
  Grams, and check the leading spectra against the stored ones.
- **Quantities.**
  - E_s(F) = tr(Fᵀ(I − M_s)F) / tr(FᵀF). This is equivalent to spectral concentration, so report only one of the
    two.
  - The local edge energy e_i = Σ_j P_ij‖F_i − F_j‖², split into its within-set and crossing-set share, using sets
    from blocks of **sites strictly before the target**.
  - Coverage, and variance explained over the whole probe, including unassigned images.
- **Calibration** on the same graphs and with the same pipeline:
  - planted piecewise-constant functions on the snapshot's own sets, at boundary leakage 0, 0.1 and 0.3;
  - affine functions of the raw activations (random W);
  - a smooth nonlinear function (random tanh features of the raw activations) with matched output variance.
  
  A real target's values are read **relative to these calibration curves**, never against permutation nulls
  alone.
- **Exact identity check:** pool → logits through the raw-activation affine head reproduces the stored logits.
  The normalised-view rank-50 fit, if reported at all, is descriptive only.
- **Neighbourhood scale:** for kNN k ∈ {15, 50}, report each site's 15-NN distance and Sinkhorn mass diagnostics.

### Outputs

questions.json keys `q_harm.interp.*` and `q_harm.smooth.*`, facts `q.harm.*`, and a short note for the writer
with the primary endpoint's verdict as computed. The interpolation test is labelled as an intervention everywhere.
CPU only (ResNet-20; on the order of 10⁵ suffix passes). If it needs more than about 30 minutes of CPU, use Modal.
