# Review 1: the plan (GPT-6.1 Sol, 2026-11-13), with the orchestrator's decisions

Verdict: **go with changes.** Every objection below is accepted by the orchestrator; build to the amended plan.
Paths are relative to the repository root.

## Material objections (reviewer's text, lightly condensed) and what to do

1. **The older studies are cut too far, and one justification is false.** Plan §5 says Pythia's sets "are sources
   at every step". In `explore/lm_coherent/data/p160t/blocks_0.npz`, `reading_src_jacc[:,1]` is only 0.174–0.236 at
   the first step; at step 512 it is 0.676–0.786. There is a training result here. Likewise
   `explore/mamba_coherent/diagnosis/DIAG_NOTES.md` D1 reports depth-block association with the current token's class
   at AMI 0.35–0.47 against source 0.12–0.15, so "both studies found the five sources" loses a real distinction.
   **Do:** keep a compact earlier-studies section (Findings, after the Mamba findings, short): Pythia's emergence of
   source alignment over training (one small chart), and Mamba v1's contrast between reading blocks (source) and
   depth blocks (token class). State 927 / 1,000 as agreement of the two studies' **T1 root binary cuts**, not of
   their sets in general. Put a four-study provenance strip in the Overview so Pythia and Mamba v1 are visible.

2. **M1 overstates how much of L4 is token identity.** The five token-dominated sets hold 1,135 of 8,000 contexts
   (14.2%); the two broad sets hold 5,135 and 1,465 (82.5%) (`sets_trained.json`, `point_L4["8"]`).
   **Do:** say that L4 isolates several frequent tokens while most contexts sit in broad mixtures; show sizes beside
   purity; keep the walk-level token-association result (excess share 0.380 at L4) as the separate, stronger evidence.

3. **M3 turns token rarity into an unsupported impossibility claim.** "Occur about once, so nothing can group them"
   does not follow: contexts with different tokens can share geometry. In `items.npz`, classes word_start / word_cont
   hold 3,536 contexts, of which 2,060 carry token types occurring once within those classes; that supports rarity,
   not the causal conclusion. The 2,798-member set also splits substantially at higher r.
   **Do:** write it as "at r = 16 many rare word tokens remain together; whether more items would give more
   readable structure is open", and show the stored higher-r breakdown as evidence about resolution.

4. **V2's curve does not measure the emergence of "kinds".** The count of classes with at least 16 of 20 images in
   one set measures fine classes staying together, whatever else is in the set. At L12 the r = 8 set sizes are
   899, 20, 403, 20, 62, 270, 172, 100 (+54 unassigned): a giant mixed set preserves many classes.
   **Do:** caption the curve as "fine classes increasingly stay together", show the largest set's share beside it,
   and argue "kinds" from the image grids and the make-up heatmaps.

5. **"Not pathological" is broader than the diagnostics, and the eigenfunctions stay peripheral.** The sanity table
   checks ψ₁..ψ₈ where the headline uses r = 16. Participation varies a lot: a stored whole-state PN is 61.8 of
   8,000; reconstructed ViT L9–L12 PN reaches 84.8 of 2,000 and L12 20.7 of 2,000. The table also puts subspace
   agreement, set Jaccard and pixel-statistic correlations side by side as if they answered one question. ViT stores
   only eight non-constant modes, so a λ₁..λ₁₆ display must be shortened there.
   **Do:** state exactly which checks passed for which objects and ranks; give PN relative to N; name each agreement
   metric. **Make the eigenfunctions prominent** (the owner: "the eigenstuff is the fundamental object and the
   clusters are just interpretive"): at least one eigenfunction tour per network with signed example items at both
   ends, reachable from the Overview, and Findings that show ψ's directly where they carry the point. Sets are the
   reading aid, not the only window.

6. **The finer-r "zoom" is not a hierarchy.** B-point's r = 16 catch-all meets r = 24 set 22 in 2,386 contexts, but
   that finer set has 2,554 in all; set 23 meets it in 373 of 586.
   **Do:** label it "selected members under another r", keep the intersection with every ancestor selection,
   recompute the summaries on that intersection, include unassigned members. Describe the stored trees as a
   different analysis, omitted for brevity (not as duplicated by this).

7. **Default Mamba set rows lack representative items.** Token chips and bars cannot show grammatical role or which
   current and next tokens occur together. Contexts and representative indices are in `items.npz:ctx` and
   `sets_trained.json:examples`.
   **Do:** show at least one marked example context per set by default, more on selection; keep chips as summaries.

8. **Export and verification need numerical and interaction checks.** `__ends` has shape (16, 2, 12): the positive
   and negative extremes, not nearest-zero examples. Correct SEBA reconstruction needs dequantisation, the constant
   column, thresholding, normalisation and `__seba_perm` (see `explore/mamba_contexts/mc_schema.py` lines 836–849);
   `__vecs` with `__seba_rot` alone is not enough. An int8 export adds a second quantisation that screenshots cannot
   validate.
   **Do:** follow the existing reconstruction convention; derive near-zero examples separately if wanted; validate
   exported labels, summaries, ordering and eigenfunction error against the source arrays in the build; exercise
   network and checkpoint switches, the other-r selection, breadcrumbs, reload and back navigation; give the 24 / 32
   set views a palette policy beyond 16 colours; put permutation bands in the same units as any plotted excess.

## Minor (accepted)

- Keep the initial Mamba controls compact; reveal references and lags progressively; make the difference between
  the layout's source and the colouring's source obvious.
- Extend the ViT analysis: `analysis_vit.py`'s `q_classmix.named` has L1–L12, L7–L12 and L1–L6, not L9–L12.
- Say early that an "all-layer block" averages operators, not activations or set labels.

## Checked by the reviewer and correct (no need to re-check)

ViT ≥16/20 curve 0, 0, 1, 5, 11, 23, 30, 38, 39, 50, 76, 94 (L9–L12: 72, of which 35 whole); ViT L12 λ 0.999, 0.995,
0.992; Mamba B-point r = 16 catch-all 2,798, 1,886 token types cover 90%; token excess 0.380 → 0.104 and
token-conditioned predictive excess 0.063 → 0.161 from L4 to L24; neighbour-count agreement 0.866 / 0.875; init
agreement 0.009; whole-state source AMI 0.735; whole / bag excess source 0.849 / 0.866, token class 0.491 / 0.058,
predictive 0.225 / 0.033; residual L16 retention 0.910 → 0.223 at lag 2; whole-state lag-16 set AMI 0.753; all 11
layouts and the listed spectral objects exist; the root-cut cross-tabulation supports 927 / 1,000.

## Orchestrator's answers to the plan's open questions

1. One Explorer tab with a network switch: yes.
2. Zoom as "selected members under another r" (objection 6), true zoom listed under checks: yes.
3. ViT explorer on the ResNet site's current code with the ViT changes merged, head-norm loadings dropped: yes.
4. Six Mamba panels plus toggles: yes.
5. Pythia over training: **one small chart** (objection 1), not just a sentence.
6. Split-half: "not measured" in the table, listed under checks. No Modal job in this work.
7. Title: keep the draft; the owner will be asked.
