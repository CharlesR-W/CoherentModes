# Deviations, operationalisations and readings of PLAN.md (compute pipeline)

Each item: what PLAN says, what was done, why. Items marked *(pre-registered)* were written before any
model output was computed.

## Data

1. **Source resolution.** *(pre-registered)* PLAN §1 asks for the standard eval transform (resize 256, centre
   crop 224) on `clane9/imagenet-100`. That dataset stores images already resized to 160 px on the short side
   (checked: e.g. 160x213, 194x160). Kept the plan's dataset and transform (most faithful reading), so
   images are upsampled 160 -> 256 before cropping: inputs are blurrier than true ImageNet at 224 and
   absolute kNN/decodability numbers will be lower than published DINOv2 numbers. Not corrected by
   switching dataset, because PLAN names this one.
2. **Superclasses.** *(pre-registered)* Built from WordNet 3.0 (nltk) by a fixed rule (`superclasses.py`):
   follow each class's first hypernym path, split the largest group until no group exceeds 15 classes, pool
   groups of fewer than 3 classes into "other <parent>". Result: 18 superclasses (`classes.json`). WordNet
   quirks are kept (e.g. ambulance falls under container). Display/validation only.
3. **Class ids** are the rank of the WordNet id (sorted), not the dataset's own label order.
4. **Split halves** A/B are a random permutation of the probe set (seed 0), not class-stratified, so that
   no label touches the construction.

## Model and views

5. **Float64 manual forward.** *(pre-registered)* Views and components are computed by a float64
   re-implementation of each DINOv2 block (same weights) so that the §4 decomposition is exact to float64;
   the maximum deviation from the hub model's float32 `get_intermediate_layers(norm=True)` readout is
   reported in `acts.json` (`readout_vs_hub`).
6. **Held-out views** are centred with the probe-set mean and then L2-normalised (the view is defined by
   the probe set). The split-half halves are subsets of the probe-set view (not re-centred per half).

## Section 2

7. **Selection rule, operationalised.** *(pre-registered)* PLAN: "the downstream default is the candidate
   that is stable and non-localised at most layers". On the CLS view, per candidate and layer l = 1..12:
   - stable(l): mean principal cosine between the half-A and half-B top-16 extended subspaces on the
     held-out set >= 0.8 (if fewer than 16 modes of either half have |lambda| >= 0.05, not stable);
   - non-localised(l): effective number of sites of the top-16 projector (1/IPR, THEORY §4.3) >= 160
     (on average >= 10 sites per mode) **and** median single-mode effective number over the top 16 >= 10.
   Score = number of layers passing both. Default = highest score; tie-break by mean stability at r=16 over
   layers. If the runner-up's score is within 1 layer of the best, the choice is called not clear-cut and
   both are kept as defaults. If the best score is < 6 (not "most layers"), that is reported as the finding
   and the rule's winner is still used downstream (labelled so). No label-based number enters the rule.
8. **Extended subspaces are centred** on the held-out set (constant removed) before principal cosines,
   since the extension of a nonconstant mode is only approximately mean-zero off-sample.
9. **Alpha=1 comparison**: reported per layer are the top 65 eigenvalues of the symmetric alpha=1 operator
   and the effective number of sites of its eigenvectors 1..64 (index 0 is sqrt(degree), not constant).
   Split-half stability is not computed for alpha=1 (its Nyström extension is a different formula and it
   is not a downstream candidate).
10. **Label-based validation** (kept separate, `selection.json.validation`): weighted-kNN class accuracy on
    the held-out set using the Sinkhorn-consistent weights p(x, j) of the full-probe operator restricted to
    the 20 largest weights; class eta^2 of each of the top 64 modes (fraction of the mode's variance
    explained by class means).

## Section 3

11. **Cross-validated values** are indexed by rank: row j of `cv` is the j-th leading positive (or
    negative) mode of Delta_A, extended to B; it corresponds to, but is not the same function as, row j of
    the full-probe Delta modes. "Unsupported" = cross-validated value of the opposite sign to its
    half-A eigenvalue, or smaller in magnitude than 25% of it, or not computable: if more than 99% of the
    half-A mode's energy lies on endpoint eigenfunctions with |lambda| < 0.05 (`cv_discarded` > 0.99), its
    cross-validated value is recorded as NaN (added after the smoke run, where such modes occurred).
12. **Same-layer coarse-state baseline**: SEBA is run on the held-out set on the orthonormalised span of
    {constant, top r-1 modes of the half-A operator extended to held-out}, and likewise for half B; the
    baseline is H(c^B | c^A) and H(c^A | c^B) of the two hard labelings.

## Implementation / scope

13. **Weights source.** The official DINOv2 weight host (dl.fbaipublicfiles.com) had an expired TLS
    certificate on 2026-11-11 (notAfter Nov 11 23:59:59 2026 GMT), so `torch.hub` could not download the
    checkpoint. TLS verification was not disabled. The model is the official hub architecture
    (`torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)`) loaded with the same
    weights from Hugging Face `facebook/dinov2-small` (key-mapped; q/k/v concatenated into qkv). An
    asserted parity check against `transformers.Dinov2Model` on the same weights at 518 px (no positional
    interpolation) is recorded in `acts.json` (`weights.parity_vs_transformers_518px`).
14. **SEBA and coarse-state flows only for the default kernel(s).** SEBA at r = 64 does not converge
    within the reference 5,000 iterations (tolerance 1e-12) and costs minutes per call; SEBA memberships,
    diagnostics and flows are shipped only for the defaults (SCHEMA `[full]`), so they are computed only
    there (both views). Reference tolerances are kept; non-convergence is recorded per (layer, r) in
    `seba.json` (`seba_converged`, `seba_iters`).
15. **Storage formats** (viewer budget ~600 MB): neuron activations are stored as uint8 with a per-neuron
    affine range (`cls/neurons/range.f32`) instead of float16 (PLAN §8 says float16; all quantities computed
    from them on Modal, loadings, AUROC, attributions, use full precision); attention maps are raw uint8
    per layer instead of `attn_subset.npz` (static-JS friendly); Delta-mode attributions are binary
    (`delta_attr/`), not JSON.
16. **Mean-patch view scope.** PLAN calls it a variant run through the same pipeline. Computed for every
    kernel: Section 2 (operator, full spectrum, selection criteria, alpha=1, validation) and declared-pair
    Delta. Modes, smoothness profiles, SEBA and flows are shipped for the default kernel(s) only. Section 4
    (CLS-position decomposition) does not apply to mean-patch and is CLS only.
17. **Delta for non-default kernels**: declared pairs only (as PLAN says); for the defaults all 66 pairs.
    Attributions: each layer's top 16 modes, and every stored Delta mode (16 per pair and t) at its smooth
    endpoint, for the default kernel(s) only.
18. **Noise control**: isotropic Gaussian noise added to the unit-norm view with total power 1/SNR,
    SNR = 10, then re-centred and re-normalised; controls at layers 1, 6, 12 for the default kernel(s).
19. **Exemplar subset for attention maps**: 15 random probe images per class (seed 0), 1,500 total;
    display only.
20. **Uncentred sensitivity** (PLAN §1): for the default kernel(s), the CLS operator is rebuilt on the
    uncentred normalised view; reported per layer are principal cosines of its top-16 modes with the
    centred operator's, its top-16 projector effective number and top eigenvalues
    (`selection.json.uncentred_sensitivity`).
21. **Ridge GCV** grid: lambda in s_max^2 x logspace(-8, 1, 46); one lambda per target (Section 4) or per fit
    summed over outputs (Section 5).

## After the full run

22. **Two defaults and size.** The rule kept vmf5 and st7 (not clear-cut), so the shipped data is 778 MB,
    above the ~600 MB viewer target (one default would be ~600 MB). Nothing was dropped to fit.
23. **t = 16 and 64 are degenerate** for every non-degenerate kernel (all nonconstant |lambda| <= 0.83, so
    lambda^(2t) ~ 0). The grid is kept as PLAN declares it; values there are 0 up to rounding, and the
    eigenvectors of Delta at those t are numerical noise. Flagged in RESULTS_NOTES.md.
24. **PCA-overlap diagnostic not run.** `extras.py` (label-free: principal cosines between each kernel's
    top-r modes and the view's top-r PCs, written into `selection.json.pca_overlap`) was written after the
    selection, to check whether vmf5's stability comes from being near-PCA. It was not run, by instruction (no
    further compute after the full run). It is not part of the selection rule and would not change it.
25. **Delta-mode attributions** cover all 66 pairs x 4 t x 16 modes for both defaults, including the
    degenerate t = 16/64 modes, whose attributions are meaningless (their R^2 and total_cov are reported
    alongside).
