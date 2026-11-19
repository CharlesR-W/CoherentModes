# Verdict (exploratory; nothing here enters THEORY.md)

> **Superseded (2026-11-13).** The report for this study is replaced by the joint report in `explore/ncs_report/` (published as `CoherentModes/ncs-vit-mamba`). The data and code here are still read by that report's build; the page in this directory is kept for reference and is no longer updated.

DINOv2 ViT-S/14, CLS view at depths 1–12, 10,000 ImageNet-100 images, 10,000 held out. All objects are
defined in `PLAN.md`; numbers are in `RESULTS_NOTES.md`; departures in `DEVIATIONS.md`. Labels were used
only for the numbers marked "validation". This text was revised after a GPT referee checked a draft
against the stored data; its corrections are incorporated.

## What was established

1. **This configuration gives stable, non-localised modes at most depths.** With a fixed vMF kernel at
   κ = 5, or a self-tuned kernel (k = 7, 15, 30), the top-16 modes reproduce across disjoint halves of the
   images at 11–12 of 12 layers (mean principal cosine over layers 0.90–0.97) and are spread over many
   images. They overlap the top-16 principal components only partly (mean principal cosine 0.61–0.77 for
   κ = 5; 0.31–0.73 for self-tuned k = 7). The predecessor's failure is avoided, but the comparison is not
   like-for-like: architecture and view changed together with normalisation (Sinkhorn against α = 1),
   bandwidth rules and N (10,000 against 2,000). (The predecessor's near-PCA figure, top-3 principal
   cosines 0.98–0.999 at κ = 5, is from an ad hoc check in the session, not from a stored file.)
   For κ = 5 the choice to centre the view matters at early layers: centred and uncentred top-16
   subspaces overlap only 0.63–0.67 at layers 1–6.
2. **Importing the model's evaluation temperature into this construction localises the modes.**
   κ = 1/0.07 (the DINO kNN temperature) in the centred, dense, self-looped operator gives a near-identity
   operator at layers 1–11, each leading mode sitting on about one image; it is non-degenerate only at
   layer 12. Whether the temperature itself fails to transfer, as opposed to this construction differing
   from truncated kNN on uncentred features, was not controlled.
3. **Endpoint smoothness changes are reproducible measurements.** For κ = 5 at t = 1, the leading
   became-smooth and stopped-being-smooth modes of all 66 layer pairs keep a median 0.98 of their
   eigenvalue when extended to an independent half; identical-view and rotation controls give 0 to 1e-15.
   Against an isotropic perturbation at SNR 10 (noise power 0.1, i.e. RMS about 32% of the signal),
   tested at layers 1, 6 and 12 only, κ = 5 adjacent-layer changes are 2–30× larger.
4. **A long block's change is the sum of its sub-blocks' changes by definition; its leading directions
   are often, not always, those of the sub-blocks.** Δ(a→c) = Δ(a→b) + Δ(b→c) exactly. The leading
   subspaces often approximately overlap (for 1→12, "became smooth" aligns with 6→12 and "stopped being
   smooth" with 1→6), but the truncated sub-block spaces do not explain every leading long-block
   direction: for 6→12 the leading-4 positive space against the union of 6→9 and 9→12 has mean principal
   cosine 0.85, minimum 0.46. (An earlier 0.99–1.00 figure averaged the four largest cosines between
   8-dimensional spaces and overstated this.) Changes are spread over several layers (the largest single
   step is a median 43% of the path) and mostly monotone (net ≈ path), except into layer 12.
5. **Layer 12 differs sharply under these settings.** Its spectrum is flat (leading nonconstant eigenvalue
   0.24 against 0.58–0.83 elsewhere), and the largest single change in the network is a *loss* of
   smoothness at 11→12 (κ = 5: −0.65; self-tuned: −0.56, 2.4× its noise response, so not only a uniform
   change of scale). Class decodability jumps at the same place (kNN 0.77 → 0.91, validation).
6. **Leading modes align with class structure increasingly with depth** (validation): class η² of the top
   8 modes 0.13 at layer 1, 0.50 at layer 9, 0.81 at layer 12; linear decodability 0.16 → 0.90.

## What was not established, or failed

- **Coarse states (hard clusters) are not well supported at these settings.** Two operators built from
  disjoint halves of the *same* layer disagree on hard assignments by 0.4–1.8 bits (r = 16), comparable
  to the between-layer conditional entropies (1.3–2.7 bits). At layer 12 only 40% of images are assigned
  at r = 16. Modes and subspaces are the supported objects; partitions are not, here.
- **The diffusion-time grid resolves intermediate behaviour poorly.** With all nonconstant eigenvalues
  ≤ 0.83, higher times attenuate the operators strongly: at t = 16 the largest change is 0.002 (the leading
  modes still cross-validate, ratio about 1.0, while many weaker ones do not), and at t = 64 it is 2e-11.
  The planted tests show that whether a merge produces "became smooth" modes depends on resolution; on
  the ViT this was not examined usefully. A finer grid or a kernel-parameter sweep is needed.
- **The self-tuned kernel's adjacent-layer changes are not clearly above its noise response.** At the
  sampled early and middle layers (1, 6) they are smaller than the SNR-10 control; 11→12 exceeds it.
  Three sampled layers do not establish that it is unusable for adjacent layers in general.
- **The kernel choice changes qualitative conclusions.** In the trained planted MLP, the fixed-κ kernel
  reports a merge of 64 states into 8 at the last hidden layer while the self-tuned kernel reports none,
  and the inputs remain perfectly linearly decodable throughout. "Did the network merge these states" has
  no kernel-free answer: the fixed kernel measures absolute cosine scale, the self-tuned one locally
  scaled distances.
- **Component attributions are direct-path, correlational fits, with cancellation throughout.** Aggregated
  head and block contributions become less cancellation-dominated with depth, but neuron-level
  cancellation persists (summed absolute neuron fractions 5–25 for the top modes at layer 6), and some
  late self-tuned fits fail held-out validation (R² −0.78 for one layer-10 mode). No intervention was run,
  so no attribution here is a causal claim.
- **Nothing here shows that these objects capture an "operation".** Smoothness-change modes say which
  distinctions among images became or stopped being geometrically salient in the CLS view between two
  depths, at one resolution. CLS is a partial view, so such a change can be information arriving from
  patch tokens rather than computation on what CLS held.

## Reading for the owner's questions

- *How to interpret the eigenfunctions:* as reproducible axes of variation among images at a declared
  resolution, partly but not wholly principal components, increasingly class-aligned with depth.
- *How to interpret the sparse clusters:* with caution; at these settings they are an unstable rotation of
  a stable subspace.
- *How a 1→5 object relates to 1→3 and 3→5:* additively and exactly for the smoothness-change operator;
  the leading directions of the long block are mostly but not entirely those of the sub-blocks.
- *Usefulness for interpretability:* the descriptive layer is reproducible and controlled. Whether it is
  explanatory needs an intervention: remove the heads carrying a late became-smooth mode and test that
  this change, and not others, disappears.
