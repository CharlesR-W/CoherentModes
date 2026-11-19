# Review 3: re-check of the revised page (GPT-6.1 Sol, fresh session, 2026-11-13), with decisions

Verdict: **ship after fixes.** All ten items of `REVIEW_2_page.md` are resolved (the reviewer opened all eight view
images, re-measured phone widths, and recomputed the matched Jaccard values 0.815962 / 0.764768). One material
objection remains, and three minor suggestions. All are accepted.

## Material

1. **The token-bag reference is described as model-free, and it is not.** Findings, finding 9
   (`site/content/findings.html` around line 255: "a reference with no model in it"), and Method's reference
   glossary ("without the model's activations"). The bag and recent references average the **trained model's input
   embeddings** over the prefix (`explore/mamba_contexts/mc_decode.py` line 296 takes
   `model.backbone.embeddings.weight`; `mc_items.py` line 113 averages those vectors). The glossary's blanket phrase
   also describes the prediction reference poorly.
   **Do:** describe bag and recent as order-insensitive averages of the trained model's input embeddings (over the
   whole prefix, and over the last 16 tokens), which bypass the recurrent processing; define each reference
   separately, with `pred` as the model's own next-token distribution. Check every other place the references are
   mentioned (Overview, cards, tours, captions, the soundness table's "input-only reference" row) and fix the same
   wording. The numerical comparison stays.

## Minor

- Say "source category" (or "kind of source") where the measured grouping is the five sources; "source document"
  can suggest identifying individual snippets.
- On a phone the hero figures and the embedded explorer views are too small to read. Use a focused panel crop (or a
  phone-specific image) so they can be skimmed.
- Add a horizontal-scroll cue to wide tables on a phone.
