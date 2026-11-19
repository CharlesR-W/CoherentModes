# Review 4: fresh-eyes read (GPT-6.1 Terra, no context, rendered page only, 2026-11-13), with decisions

The reader's own summary of what the report found matched the intended content (ViT: appearance early, kinds of
object late, single classes at the last layer possibly an effect of sample and graph; Mamba: token just read early,
token families, punctuation and layout across layers; recurrent state groups by source and its groupings stay
smooth much longer along reading; exploratory, split-half not yet checked). So the page works. The items below are
where that reader slowed down. Decisions are the orchestrator's.

## Where the reader slowed down (ranked by cost)

1. **Overview, "The short answer for each network":** "the residual block's set vectors lose most of their retention
   within two tokens" read as a claim about the model forgetting; the distinction (smoothness of particular vectors
   on later walks, not information kept by the network) only arrived in Findings 8.
   **Do:** put the distinction next to the headline, in a clause, wherever retention along reading is first stated.
2. **Overview, "The sets, at a glance", Mamba figure:** hard to read alone. The reader had to reread the caption to
   see that all three panels use the same all-layer UMAP with different colourings, so changing shapes are not
   layer-specific layouts. The commonest tokens and arrows are hard to decode at figure scale.
   **Do:** make the figure say this itself (for example a title or subtitle "same positions, three colourings"),
   and make the token rows legible at the size they are shown.
3. **"source document"** made the reader expect individual documents; Method later says source is maths, code, wiki,
   forum or poetry. **Do:** "kind of source" / "source category" throughout (also in `REVIEW_3_recheck.md`).
4. **Overview, Mamba card:** "excess share 0.380" appears before a plain explanation of the quantity and its scale.
   Likewise "a share 0.92 of its variance lies between the ten class groups" needs a pause.
   **Do:** at first use say in plain words what an excess share is and its scale (0 = chance, 1 = always); write the
   variance statement as "92% of its variance lies between the ten class groups" (from the fact, formatted).
5. **Findings 1, heatmap:** "each row is mostly one or two groups" seemed at odds with a caption saying each
   **column** sums to 100%. **Do:** make the sentence a direct reading of the plotted normalisation (or change the
   normalisation to match the sentence), so text and figure agree.
6. **Overview, first figure:** "SEBA value" is used in the caption before SEBA is introduced; the reader needed the
   caption to learn the rows are the highest-membership examples, not random ones.
   **Do:** say it in plain words at first use ("the images that belong to the set most strongly").

## Explorer

1. Switching from ViT to Mamba reset the colouring to coherent sets after the reader had chosen an eigenfunction.
   **Do:** keep the colouring kind across the switch where both networks have it.
2. "Colour from" lists very different objects; the reader needed the expanded instructions to see that the choice
   colours **every** panel while each panel keeps its own stored layout. **Do:** one short always-visible line by
   the control saying that.
3. "Tour" suggested a multi-step guide; it is a preset plus a suggestion. **No change** (the ResNet report uses
   "Tours" the same way).
4. **Bug:** a ViT class-make-up tooltip stayed visible after switching to Mamba, over the Mamba side panel
   (`/tmp/ncs-fresh/mamba.png`). **Do:** hide tooltips on network switch.
5. "whole state read 1 tokens further" → "1 token".

## Visual

1. At 390 px the Overview figures and the "What was run on what" table are too small to read
   (`/tmp/ncs-fresh/mobile.png`). **Do:** as in `REVIEW_3_recheck.md` (phone-readable crops, scroll cue or stacked
   table).
2. The desktop full-page capture showed a blank region where the Mamba hero figure belongs
   (`/tmp/ncs-fresh/overview.png`); it appeared on the phone capture. Probably lazy loading in a full-page
   screenshot. **Do:** confirm it appears when a reader scrolls normally; if there is any doubt, load the hero
   figures eagerly.

## Length and missing information

1. The Overview's two long captions, detailed cards, five-step procedure and eigen-structure paragraph compete with
   the main result. **Do:** trim the Overview so that the result and the pictures come first and it does not look
   long: shorten the captions to what is plotted, collapse or cut the procedure to a line with a link (the reader
   knows the ResNet report), and keep the soundness note to a sentence with a link.
2. Findings repeats explorer views and "Plotted" descriptions per finding. **Keep** (owner's preference: captions
   say what is plotted, findings open on the explorer view), but tighten any caption that runs long.
3. The reader wanted a plain account of **why averaging walks across layers answers a different question from
   looking at one layer**, with a small concrete example. **Do:** two or three plain sentences, early.
4. The reader wanted to know how much the Mamba source result depends on the particular five source categories and
   snippets, and what the missing split-half check means for confidence. **Do:** one sentence each, where the result
   is stated; no more hedging than that.

## Wording

- "have more alike next-token distributions" → "have more similar next-token distributions".
