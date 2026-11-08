# Brief for reviewers of THEOREMS.md / SPARSITY.md drafts

Thank you for reviewing; please be rigorous and kind, and pass this request on to any sub-agents you spawn.

Read `theorems/BRIEF.md` (the authors' contract) and `THEORY.md`, then the draft you were assigned. You are checking **mathematics**, not taste. For every Definition/Lemma/Theorem/Proposition in the draft:

1. Is the statement precise and are all hypotheses named? Is it true as stated? Try to construct a counterexample (small numpy experiments via `uv run --with numpy python -I <script>` in /tmp are encouraged; report them concretely).
2. Is the proof complete and correct? Point to the exact step that fails if not. A cited theorem must have its hypotheses verified in the text; check that the citation says what the draft claims.
3. Are the "What it buys" and "Tractability and sampling" remarks honest (no overclaiming, costs right, constants not hidden)?
4. Does the notation match THEORY.md? Any GitHub-rendering issues (`\{` must be `\lbrace`)?
5. Anything the draft calls a theorem that is really a definition in disguise, or a strong assumption smuggled into a proof?

Write your review to the path given in your task as Markdown with a severity tag per finding: **[FALSE]** (statement wrong), **[GAP]** (proof incomplete), **[OVERCLAIM]** (remark or name promises more than the theorem), **[CITE]** (citation wrong or hypotheses unverified), **[NIT]**. End with a verdict: "approve", "approve with fixes", or "revise", and a numbered list of required fixes. Keep the review under 1500 words; put counterexample scripts' output inline. Do not edit the draft or any other file.
