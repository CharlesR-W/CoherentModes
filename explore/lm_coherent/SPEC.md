# Coherent sets in a small language model (toy, exploratory): spec

Draft v2.1, 2026-11-12. This version is cut down to a toy after review. Changes from v1 are listed in section 14;
changes made while implementing the pipeline (v2.1: the fixture noise level, and implementation choices) in
section 14.2.

This is an exploratory side study in `explore/`, like `explore/cifar_coherent/` ("the ResNet study"). It is
**not** an `EXPERIMENTS.md` stage, and no claim enters `THEORY.md` from it. The scope is one model, one training
run and one probe of 1,000 snippets. The aim is a preliminary, readable picture of how far the ResNet construction
works on a language model out of the box.

**Vocabulary.**

- An **item** is a snippet. `THEORY.md` calls this a site.
- A **layer** is a place in the network: `L2`, `L4`, ..., `L12`.
- A **cell** is a (checkpoint, layer, position) triple.
- A **block** is a run of cells along one axis.

The working language is `THEORY.md`'s (sites, views, operators, blocks, measured regimes). Nothing here uses the
vocabulary of `THEOREMS.md` or `SPARSITY.md`.

## 1. Question, quantities, labels

### 1.1 Question

Does the ResNet study's construction give readable, non-trivial structure on a small causal language model when
the items are text snippets? The construction is, per cell, a cosine kNN graph, symmetric Sinkhorn scaling and a
lazy walk; the walks are averaged over a block; then come eigenfunctions with per-cell retention, and sets.

In particular:

- (a) Along reading (token position), depth and training, where does the walk separate kinds of text, for example
  maths from poetry?
- (b) How strongly is the walk tied to the token just read, and how strongly to the source?
- (c) Do the sets from the residual at position t agree with the sets from the whole layer state up to t (the prefix
  slab, which is all the KV cache derives from)?
- (d) What does recursively zooming into one block's 2-way cut show?

### 1.2 Measured quantities

**Spine (unchanged from the ResNet study).**

- Each cell c has a lazy walk $M_c = \tfrac12(I + P_c)$. A block B has the average $\bar M_B$.
- For any function f on the items, $f^\top(I - \bar P_B) f = \frac1{|B|}\sum_{c\in B} f^\top(I - P_c) f$.
- From each block come the eigenpairs $\lambda_k, \psi_k$ of $\bar M_B$, the retention $\rho_c(f) = f^\top M_c f /
  f^\top f$ along the block's axis, and the SEBA sets. The page shows r = 2 and r = 4.

**Walk shares.** These are linear in P, so a block's value is the mean of its cells' values **[thm]**.

- For a grouping $\mathcal G$ of the items, $S_c(\mathcal G) = \frac1N\sum_{g}\sum_{i\ne j\in g}(P_c)_{ij}$.
- Its chance level is $S^0 = \sum_g |g|(|g|-1)/(N(N-1))$. This is exact as the mean under permutations of group
  membership.
- The excess is $\tilde S = (S - S^0)/(1 - S^0)$.
- **$T$** uses the grouping by the token at position t. It is label-free, since tokens are inputs.
- **$W$** uses the grouping by source. It is validation.
- These are associations between the walk and a grouping. They do **not** split the walk into "token" and "context"
  contributions, and the page says so.

**Pair mixing ratio** (validation).

- $X_c(A,B) = \frac{N-1}{|A||B|}\sum_{i\in A,j\in B}(P_c)_{ij}$.
- It is 1 for a walk that ignores sources, and below 1 when the walk avoids crossing between A and B.

**Span agreement.**

- $a_r$ is the mean $\cos^2$ of the principal angles between the $\mathrm{span}(\psi_1..\psi_r)$ of two cells, with
  the constant projected out (`schema_v2.comparison`).
- Its null is $r/(N-1)$, which is 0.003 at r = 3.

**Matched Jaccard.** The ResNet definition: Hungarian matching, a 200-permutation null, and the size ceiling.

**Zoom tree** (section 7): conductance $\phi^*$, the clustering ratio $\Upsilon$, node spectra, and the retention of
each cut.

### 1.3 Labels

The method uses no labels. Snippet source is used only for these four things:

- assembling a balanced, varied probe, with per-source filters (section 3);
- colouring plots;
- the validation statistics $W$, $X$ and matched Jaccard against sources;
- choosing which snippets the "follow" figure shows, by source, for display only.

Source never enters a kernel, an operator, a basis, a cut or a block choice. The page states this once on Overview
and again in Method.

## 2. Substrate

- **Model.** `EleutherAI/pythia-160m`.
  - GPT-NeoX architecture, 12 layers, width 768, parallel attention+MLP residual.
  - Rotary embeddings on 25% of each head dimension; no positional embedding in the residual.
  - Trained on the Pile for 143,000 steps of about 2M tokens each.
  - Checkpoints are the HF revisions `step<k>`; their existence was verified on 2026-11-12.
  - The forward pass runs in float32 eval mode with `attn_implementation="eager"` and TF32 off, with the
    `transformers` version pinned.
- **Checkpoints (K = 5):** steps `[0, 512, 2000, 16000, 143000]`. Step 0 is the initialisation; the rest are
  roughly log-spaced.
- **Layers (S = 6):** `L2, L4, L6, L8, L10, L12`, the raw residual stream after block l.
  - Record them with a forward hook on `gpt_neox.layers[l-1]`, taking `output[0]`.
  - Do not use `output_hidden_states`, whose last entry is post-`final_layer_norm` in some versions.
  - The embedding output is not a layer here. At any position it is a function of the current token alone, so it
    would contain exact duplicates (section 5).
- **Positions.** The input is `[EOT] + 128 snippet tokens`. EOT is `<|endoftext|>` (id 0), the Pile document
  separator.
  - Position t is the sequence index, so t is the number of snippet tokens read. The residual at t predicts token
    t+1.
  - The sampled positions are $\mathcal T = \{4, 8, 16, 32, 64, 128\}$ (P = 6).
  - Positions 0 to 3 are excluded. Position 0 is identical for every item. At positions 1 to 3 the cells are
    functions of 1 to 3 tokens and would contain exact duplicates.
- **Cells (point view):** 5 x 6 x 6 = 180, all on the same N items.

## 3. Probe (`probe.py`, one CPU unit on Modal)

**Sources.** There are five, with 200 items each, so N = 1,000.

| source | where it comes from | selector |
|---|---|---|
| `maths` | Pile **test** split (`monology/pile-uncopyrighted`, `test.jsonl.zst`, 338 MB); never trained on | `meta.pile_set_name` = `ArXiv` |
| `code` | Pile test split | `Github` |
| `wiki` | Pile test split | `Wikipedia (en)` |
| `forum` | Pile test split | `StackExchange` |
| `poetry` | Gutenberg Poetry Corpus (`biglam/gutenberg-poetry-corpus`, 91 MB, one line per row, with `gutenberg_id`) | none |

Poetry is not guaranteed to be held out, since Pythia's training data include PG-19. The page states this.

**Documents.** Each snippet comes from a different document; for poetry, a document is one `gutenberg_id`. Take each
source's documents in file order, shuffle them with `RandomState(0)`, and accept snippets until 200 pass.

**Start at a natural boundary.**

- For Pile documents, a candidate start is offset 0, or the offset right after `"\n\n"` when the next character is not
  whitespace.
- For poetry, a candidate start is any line start, followed by consecutive rows of the same `gutenberg_id` joined
  with `"\n"`.
- Pick a candidate with `RandomState(1000 + doc_rank)`.
- Tokenise the substring from the boundary with the Pythia tokenizer, and keep the first 128 tokens.
- Skip the start if fewer than 128 tokens remain. Make at most 5 attempts per document.

**Filters.** These are written into `probe.json`.

- Every source:
  - at least 95% of characters are ASCII;
  - at most 30% of tokens are whitespace-only;
  - the text contains no EOT and no NUL.
- `maths`: at least 4 `$`, or one of `\begin{equation`, `\begin{align`, `\[`, `\frac`.
- `code`: at most 30% digit characters, and no line longer than 300 characters.

**De-duplication.** This is label-free and greedy, in draw order, across all sources.

- (i) For every $t \in \mathcal T$, no two snippets may have the same **multiset** of tokens 1..t. This implies
  distinct prefixes. So no cell of any view has two items with the same input, and the bag view (section 4) has no
  exact duplicates.
- (ii) The token 8-gram Jaccard with any accepted snippet must be at most 0.2.

**Order.** Items are permuted with `RandomState(0)`, so an item's index does not encode its source. The probe stores
tokens, per-token strings and text.

**Order-control probe S.** For each item, permute tokens 1..127 with `RandomState(item_id)`. Token 128 and the EOT
are kept. Probe S is read **only at t = 128**, where the readout token and the multiset of tokens 1..128 equal
those of the original. So "real vs S at t = 128" changes only the order of the earlier tokens.

## 4. Views

The **standard cell pipeline** is applied to any view $\phi$: centre over the N items, row-normalise, take the cosine
$C$, and set $d = 1 - C$. Everything is float64.

| view | $\phi(i)$ at (checkpoint, layer l, position t) | computed for |
|---|---|---|
| **point** (spine) | $h^l_t(i)$ | all 180 cells |
| **slab** (aligned prefix) | $t^{-1/2}[\hat h^l_1(i),\dots,\hat h^l_t(i)]$, where $\hat h^l_u$ is the standard pipeline applied at position u. Then the standard pipeline is applied to the concatenation. | all 180 cells |
| **bag** (reference) | $\frac1t\sum_{u\le t} E[x_u(i)]$: the raw input embedding rows of the checkpoint, summed in float64 in sorted token-id order | 5 checkpoints x 6 positions = 30 cells |

**Slab.**

- Its Gram is the mean over u = 1..t of the per-position cosine Grams, which is then double-centred and
  renormalised. All 128 positions enter, not only the sampled ones. A test checks this against explicit
  concatenation.
- It is aligned by position, and its per-position centring makes it depend on the probe's position-wise
  distributions. It is **not** permutation-invariant, and no claim depends on that.
- With uniform lag weights it is a window kernel in the sense of `THEORY.md` section 2, so [A2] covers it; the
  weights are a pro-tem choice [P3].

**Bag.**

- It is a function of the multiset of tokens 1..t alone, so it is permutation-invariant **[thm]**. Control C2 checks
  this.
- It is a reference for what the token counts alone give. It is not a layer of the network.

**Why the point view is legitimate, and what the slab adds.**

- **[thm]** The averaged operator needs no map between cells, and no cell has to be a sufficient statistic for
  another ([A1]; `THEORY.md` 6.1). So $h^l_t$ is a legitimate cell.
- **[thm]** In a causal decoder, the layer-l slab $(h^l_u)_{u\le t}$ determines all later layers at positions
  $\le t$, the next-token distributions there, and every KV-cache entry of later layers at those positions. So
  slabs form a Markov chain in depth. The point residual does not: $h^{l+1}_t$ also reads $h^l_{u<t}$.
- This Markov property matters for product, or composition, readings (`THEORY.md` 4.1). The averaged blocks used
  here do not need it.
- Being Markov is a property of the information. The slab's cosine geometry is one kernel on that information. So
  point-vs-slab agreement (figure F6) measures whether the cheap view tells the same coherent-set story as a view of
  the whole state. It is not an information measurement.

## 5. Operator pipeline

**Reused unchanged.** Local runs import from `explore/cifar_coherent/` via `sys.path`; Modal images add the files.

- `coherent.symmetric_balance`, which returns P: zero-diagonal, float64, log domain, row error < 2e-12.
  Nonconvergence is a failed measurement.
- `coherent.block_modes(shared_stationary=True)`.
- `seba.py`.
- `schema_v2.block_record`, which runs SEBA for all of `RANKS = [2,3,4,6,8]` and the retention over the cells
  passed as `ms`.
- `schema_v2.comparison`.

**Per cell.**

1. Apply the view and the standard pipeline.
2. Kernel: the ResNet default, unchanged. It is the union of each item's 15 nearest neighbours (`topk`, ties broken
   by index), with edge weight 1, a floor of $10^{-6}$ elsewhere, and $K_{ii} = 0$.
3. Scale with `symmetric_balance` to get P, then set $M = (I + P)/2$.

**Per block.** Average the cells' M, then run `block_record` with `ms` = all cells on the block's axis.

**Policies for degenerate items.** These are asserted per cell, and the counts go to `validation.json`.

- **Exact duplicates.** The probe design makes these impossible (section 3 (i)). The pipeline asserts that the
  smallest off-diagonal cosine distance in every cell is above $10^{-9}$.
  - If it is not, the unit fails and nothing is patched.
  - The pilot gate (section 8.1) catches this before the full run.
- **Zero centred vector.** An item whose centred vector has norm below $10^{-10}$ times the median norm triggers
  `THEORY.md` [P2]. The item is removed from every view and every cell of the run, and the removal is reported. No
  such item is expected.

**Storage.** Operators are dense float64 `.npy` files on the work volume. Only those needed later are kept: the
t = 128 cells for the training blocks, about 0.25 GB. They are never fetched. The page gets only derived arrays
(section 10).

**Deviations from the ResNet pipeline.**

1. Items are snippets, and cells have three indices.
2. There is one kernel rule (k = 15) and no kernel sweep.
3. N = 1,000.
4. The slab and bag views are new.
5. The walk shares and the zoom tree are new.
6. Eigensolves run on CPU in float64, with the same algorithm.
7. Layouts are aligned across positions (section 11).

## 6. Blocks and statistics

Each listed cell carries equal weight in a block. Positions and checkpoints are log-spaced, so early reading and
early training get more weight per token or step. The page states this.

**Blocks** (point view, chosen for the figures):

| family | blocks | count |
|---|---|---|
| reading | positions 4..128 at each layer and each checkpoint | 30 |
| depth | L2..L12 at each position, step 143000 | 6 |
| training | (layer, t = 128), steps 0..143000 and 2000..143000 | 12 |
| single cells | point view: all 180; slab view: all 180 (spans and sets, for F6); bag: 30 | 390 |

**Per cell and per block.**

- $T$, $W$, $X$ (5 x 5), each with its chance level and a 200-permutation band.
- $\mathrm{PN}(\psi_1..\psi_8)$, and the smallest SEBA set at r = 4.
- Matched Jaccard of the sets (r = 2, 4) against sources, with its null and ceiling.
- Per item: the own-source walk share (u8).
- For the 10 "follow" items (the first two of each source in probe order): the walk mass of their row of P by
  source.

**Comparisons.**

- point vs slab: $a_3$, $a_6$ and the r = 4 matched Jaccard with its null, for all 180 cells.
- real vs S at t = 128, step 143000, for each layer and the bag: $W$, and the r = 4 matched Jaccard with its null.
  The items are the same, so the sets are directly comparable.
- Each checkpoint's single-cell r = 4 sets against step 143000's, at (L8, 16) and (L8, 128).

## 7. Zoom recursion (`lm_tree.py`)

Take a block, cut its items in two using the block's own averaged walk, keep each side, redo the same analysis on
it, and repeat. SEBA is shown alongside but never used to make a cut.

### 7.1 Node analysis

For a node with item set S (n = |S|) and the fixed block B (the same cells at every depth):

**1. Rebuild on the subset.** For each $c \in B$:

- recompute the point view of the items in S from the saved activations;
- centre **over S**, then row-normalise;
- build the k = 15 union kNN graph **among S**;
- run Sinkhorn to get $P^S_c$, then form $M^S_c$.

Then average to get $\bar M^S$ and take its eigenpairs. Section 5's degenerate-item policies apply within S. There
are no exact duplicates by design, and the zero-norm check is repeated.

**2. Sweep cut** on the mean zero-diagonal walk $\bar P^S$.

- The conductance of a set A is $\phi(A) = \sum_{i\in A,\,j\in S\setminus A}\bar P^S_{ij} / \min(|A|, n-|A|)$.
- Every cell has the uniform stationary measure, because of Sinkhorn. So $\phi_B(A)$ is exactly the mean over cells
  of $\phi_c(A)$ **[thm]**. This is the discrete form of the averaged-boundary functional in Froyland's dynamic
  isoperimetry. It would not hold with per-cell degree normalisation.
- Candidate cuts are level sets $\{\psi > \theta\}$ along each of $\psi_1$, $\psi_2$ and $\psi_3$, with $\theta$
  between consecutive distinct values. Both sides must have at least $m_{\min} = \max(16, \lceil 0.05n\rceil)$
  items. The cut with the lowest $\phi$ is kept, and the code records which $\psi$ won (`axis_used`).
- Write $\mu_k = 1 - \lambda_k(\bar P^S) = 2(1-\lambda_k(\bar M^S))$.
  - For the unconstrained $\psi_1$ sweep, $\mu_1/2 \le h \le \phi_{\text{free}} \le \sqrt{2\mu_1}$ **[thm]**. Here
    h is the minimum conductance over all cuts, and $\phi_{\text{free}}$ is stored.
  - The constrained $\phi^*$ only satisfies $\phi^* \ge h \ge \mu_1/2$.
  - Both values and both bounds are stored.
- Child 0 is the side that contains the item with the largest value of the winning $\psi$.

**3. Clustering ratio** $\Upsilon = \mu_2/\phi^*$ (after Peng, Sun and Zanetti 2015).

- Since $\phi^* \ge \mu_1/2$, it follows that $\Upsilon \le 2\mu_2/\mu_1$.
- Reading rule: $\Upsilon \ge 3$ marks a "clear split"; anything lower is a "weak split", drawn faded.
- The threshold is fixed on the fixture (control P1) before any real tree is opened, and it is never revised on real
  data.
  - If planted splits and structureless splits do not separate at 3, the threshold becomes the geometric mean of the
    largest structureless $\Upsilon$ and the smallest planted $\Upsilon$.
  - If those two ranges overlap, the reading rule is dropped, every split is drawn the same way, and the page says
    so.

**4. Stored per node.**

- $n$; $\lambda_1..\lambda_4$; $\mu_1$, $\mu_2$; $\phi^*$; $\phi_{\text{free}}$; $\Upsilon$; `axis_used`;
  `cut_theta`; the members.
- $\psi_1$, $\psi_2$ and $\psi_3$ on S (int8 with a scale).
- The sweep profile, at 64 points.
- The retention of the centred cut indicator in each $M^S_c$.
- SEBA r = 2 on S: its matched Jaccard with the cut, and its unassigned count.
- The source make-up (validation).
- The 6 items at each end of the winning $\psi$.

### 7.2 Recursion and stopping

- A node is split if its depth is below 3 and $n \ge 64$; otherwise it is a leaf. That gives at most 7 splits and 8
  leaves.
- This stop uses size and depth only. $\Upsilon$ decides how a split is drawn and described, not whether children
  exist.
- A sweep cut is a full bipartition, so there are never unassigned items. SEBA's unassigned items appear only in the
  side-by-side comparison.

### 7.3 Alternative recorded, not run

**Restrict-and-rescale.** Restrict each $K_c$ to S without re-centring or rebuilding the graph, then rerun Sinkhorn.
It is not the default, for three reasons:

- Items near the cut lose most of their kNN edges, and Sinkhorn then inflates their $10^{-6}$ floor edges into
  uniform links.
- Without re-centring, the parent's split direction adds a common offset to every cosine within S.
- "The same analysis on those items" means a probe made of S, and a view is centred over its probe.

### 7.4 Trees

There are four trees, all on probe A and the point view:

| tree | block | step | role |
|---|---|---|---|
| **T1 (headline)** | reading block 4..128 at L8 | 143000 | headline |
| T2 | reading block at L2 | 143000 | contrast where the token just read dominates |
| T3 | single cell (L8, 128) | 143000 | one cell against a block |
| T4 | reading block 4..128 at L8 | 2000 | the same zoom earlier in training |

All four are precomputed. No recursion runs in the browser.

## 8. Controls and targets

| id | target | label | check |
|---|---|---|---|
| C1 | Causal indexing. With the model in **float64**, a forward pass on `[EOT] + tokens 1..64` gives the same residuals at t in {4, 8, 16, 32, 64}, at every layer, as the full 128-token pass (max relative error over items < $10^{-10}$). This verifies that position t means tokens 1..t were read. The float32 discrepancy of the same comparison is recorded per layer as a hazard (section 9), not gated. (v2.1 decision, 2026-11-12: the v2 float32 check at $10^{-4}$ failed at 8.6e-4 while the float64 error was exactly 0; see section 14.2.) | identity **[thm]** | pipeline assertion, step 143000 |
| C2 | The bag view at t = 128 is bitwise identical for probes A and S, because the multisets are equal and summation is in sorted order. So the operators and sets are identical. | identity **[thm]** | pipeline assertion |
| P1 | Planted hierarchy fixture (section 10.2, $\sigma = 1.3$), through the real operator. The 7 splits (depths 0 to 2) reproduce the planted tree exactly: every node equals its planted node, Jaccard 1, and the 8 depth-3 leaves are the planted leaves. Planted splits have $\Upsilon$ above the threshold. Structureless splits fall below it. The test asserts this on 3 independent draws of the fixture (seeds 0 to 2); the regime was checked on 20 (section 10.2). | planted | test |
| P2 | Planted block demotion (section 10.2). In the cell that carries a transient split, that split is the top cut. In the 6-cell block, the persistent split is the top cut. The SEBA r = 2 and sweep matched Jaccard with the planted labels is at least 0.95 in both cases. | planted | test |
| P3 | The block conductance is the mean of the cell conductances. Block $T$, $W$ and $X$ are the means of the cell values. The slab Gram equals the concatenation route. All to $10^{-12}$. | **[thm]** | test |
| E1 | At step 143000, $\tilde T$ is higher at L2 than at L12 at every sampled t: early layers are more tied to the token just read. | established (contextuality of LM representations; Ethayarajh 2019) | F3 |
| H1 | At step 143000, $\tilde W(t{=}128) - \tilde W(t{=}4) \ge 0.2$ at L6 to L12. | hypothesis | F3 |
| H2 | $X(\text{maths},\text{poetry}) < 0.5$ at (L8, t = 16), step 143000. | hypothesis | F4 |
| H3 | Point-vs-slab $a_3$ at t = 128, step 143000, is higher at L12 than at L2. | hypothesis | F6 |
| H4 | At (L8, 128), $\tilde W$ at step 0 is less than half its value at step 143000. | hypothesis | F7 |
| H5 | The first cut of T1 puts at least 80% of `code` and at least 80% of `poetry` on opposite sides. | hypothesis | F9 |

Note on P1: the depth limit makes the planted leaves leaves of the tree. "Structureless splits" therefore means two
things, both computed by the test only:

- the root split of one cell of isotropic Gaussian noise (`lm_schema.flat_cell`, N = 1,000, d = 32), the cell of
  `_fixture_flat`'s kind;
- one extra split of each planted leaf, which contains only noise.

P1 can only pass if the fixture's kNN graphs record the planted nesting: inside every planted node, edges must cross
between its children's sub-groups, denser between sibling leaves than between cousins. A regression test keeps the
v2 regime ($\sigma = 0.3$, compact leaves) as a negative example: its graphs have no edge between leaves, its cuts
never split a leaf, and its root cut is not the planted one.

H1 to H5 are numerical predictions written before the run. They are not tuned, and none of them claims a mechanism.
The verdict marks each as met, not met or not determinable. Step 0 and probe S are comparisons, not targets.

### 8.1 Pilot gate

Units U1 and U2 run first at step 143000. The full run proceeds only if all of these hold:

- C1, C2 and P3 pass;
- Sinkhorn converges in every cell;
- the duplicate and zero-norm assertions pass;
- no point cell has $\mathrm{PN}(\psi_1) < 10$ (0.01 N, the ResNet flag for a localised mode).

On failure, stop and report to the owner. Nothing is tuned on labels.

## 9. Hazards (reported in Method)

| hazard | design | reported as |
|---|---|---|
| Large first-token norm and attention sink | EOT at position 0, identical for every item and excluded; centring per cell | mean $\lVert h^l_u\rVert$ against u (all 129 positions) per layer at step 143000 |
| Rogue (dominant) coordinates | plain cosine after centring is kept; no metric is added ([A2]) | top-1 and top-5 shares of centred variance per cell (heatmap) |
| Exact duplicates and "one close pair takes over" | multiset de-duplication; positions $\ge 4$; no embedding row; rank-based kernel | smallest pairwise distance per cell; $\mathrm{PN}$ and the smallest r = 4 set per cell |
| Token-identity dominance early on | $T$ is shown next to $W$ everywhere | F3 |
| Log-spaced axes weighting blocks | stated | Method |
| Poetry possibly in training | stated | Method |
| Float32 rounding grows with depth | measurements use the float32 forward pass (section 2); C1 is gated in float64 | `C1.float32_by_layer` in `validation.json`: the max relative difference between two float32 passes that are identical in exact arithmetic (truncated vs full). At the pilot it grew with depth from about 1e-6 at L2 to about 1e-4 to 1e-3 at L6 to L12. This bounds how literally any bitwise or identity reading of float32 activations can be taken; it is far below the scale of the cosine geometry the kernel uses. |
| Selection by de-duplication | the label-free t = 4 multiset de-duplication (section 3) rejects snippets whose first four tokens repeat an accepted one | rejection counts per source in `probe.json`; at the pilot it rejected 68 `code` and 93 `forum` candidates (shared openings such as license headers and question templates), so those sources are somewhat biased toward unusual openings |
| kNN components disconnect | none in the construction: the kernel is rank-based with a uniform $10^{-6}$ floor (section 5); the fixture is built so that its graphs stay connected inside planted nodes (section 10.2) | When a node's kNN graph falls into separate components, the geometric proximity BETWEEN components is lost to the kernel: every cross-component pair gets the same floor. This does not make every grouping of components equal in conductance (component sizes, their internal structure and the Sinkhorn scaling still matter), but the kernel no longer records which components are closer, so with three or more the grouping chosen is not a statement about proximity, and $\Upsilon$ there involves floor-level terms. Reported per split node (label-free diagnostic; changes no cut): `n_components` and `component_sizes` of the union of the node's cells' pre-floor kNN masks, and `disconnected` = at least 2 components and some union of whole components leaves both sides $\ge m_{\min}$. |

## 10. Data

### 10.1 Schema

**Locations.**

- The run tag is `p160t`, on the Modal volume `cm-lm-coherent`.
- `/hf` holds the HF cache, `/work/p160t` the activations, cosines, operators and timings, and `/analysis/p160t`
  the public files.
- `fetch` copies `/analysis/p160t` to `explore/lm_coherent/data/p160t/` (about 60 MB, untracked).

**Conventions.**

- Arrays are float32 unless noted.
- Eigenvectors are int16/32767 with float32 scales, signed as in `schema_v2`.
- NaN and -1 are used for padding.

| file | contents |
|---|---|
| `meta.json` | `schema` = "lm-toy-1"; `construction_fingerprint` (a hash of section 5, as in `construction.py`); `model`; `model_config` (read from the HF config in U1: layers, width, heads, rotary fraction, parallel residual, vocabulary, ...); `transformers_version`; `steps`; `layers`; `positions`; `k`; `seba_r`; `sources`; `n`; `items`, `removed_items`; `follow_items` (10 indices); `threshold` |
| `probe.json` | for A: items [{`id`, `source`, `doc`, `char_start`, `tokens`, `token_str`, `text`}]; for S: the permutation per item; the filters, seeds, and rejection counts by reason and source |
| `cells_<step>.npz` | arrays [view (point, slab), layer, position], plus `bag_*` arrays [position]: `T`, `T0`, `T_band` (2), `W`, `W0`, `W_band`, `X` (5,5), `X_band` (5,5,2), `pn` (8), `min_set_r4`, `min_dist`, `rogue1`, `rogue5`, `sinkhorn_iters`, `row_err`. Point view only: `own_src` (u8, [layer, position, N]) and `follow_mass` (u8, [10, layer, position, 5]). Step 143000 only: `norm_mean` [layer, 129] |
| `single_<step>.npz` | single-cell records [view, layer, position], plus `bag_*` [position], in `block_record` arrays: `vals` (9), `vecs` (8,N) i16, `scale`, `seba_label` (5,N) i8, `seba_rot`, `seba_perm`, `retention_psi` and `retention_seba` along positions, `retention_*_depth` along layers, `top_members` (5,8,24) i16, `src_jacc` (+ null, ceiling) |
| `blocks_<step>.npz` | reading blocks [layer]; at step 143000 also depth blocks [position]; same arrays |
| `train.npz` | training blocks [layer, run], with retention over the 5 checkpoints, plus the single-cell retention along training at t = 128 [layer, step] |
| `compare.json` | rows {`kind` (`point_slab`, `real_vs_S`, `to_final`), `step`, `layer`, `t`, `a3`, `a6`, `jacc`, `jacc_null` (3), `W_a`, `W_b`} |
| `trees.json` | [{`id`, `block`, `step`, `threshold`, `nodes`: [{`id`, `parent`, `depth`, `members` (i16), `n`, `vals`, `mu1`, `mu2`, `phi`, `phi_free`, `upsilon`, `axis_used`, `cut_theta`, `cut_m`, `children`, `psi` (3, n) i8 + `psi_scale` (3), `profile` (64) at `profile_m`, `cheeger_lo`, `cheeger_hi`, `m_min`, `retention_cut`, `seba2` (n) i8 aligned with `members`, `seba2_vs_cut`, `seba2_unassigned`, `n_components`, `component_sizes`, `disconnected`, `makeup` (5), `ends` (2, 6)}]}], then the fixture tree `P1` (section 14.2). Leaves carry only `id`, `parent`, `depth`, `members`, `n`, `makeup`, `children` |
| `umap.npz` | `grid` [7 rows (bag, L2..L12), 6 positions, N, 2] at step 143000; `strip` [6 layers, 5 steps, N, 2] at t = 128 |
| `figdata.npz` | thumbnails for the pipeline figure (C, K, P of the cell (L8, 128), $\bar M$ of T1's block and `M_cells`, each of its 6 cells' $M_c$; sorted by the T1 root cut, at 250 x 250), float16 |
| `validation.json` | shapes, dtypes, finiteness; C1 (`max_rel_err`, `pass`), C2 per step, P1 and P2 (fixture numbers, `pass`), P3 (real-data linearity per step, `pass`); the duplicate/zero-norm assertions; Sinkhorn failures (must be 0) |

### 10.2 Fixtures (the pipeline implementer's first deliverable)

**`data/_fixture/`.** The full schema, produced by the real code path on synthetic cells (`lm_schema.fixture_inputs`
writes the U1 work files; `measure_step` and `finalize`, the U2 and U3 code, do the rest).

- N = 1,000 in d = 32. There are 5 checkpoints x 6 layers x 6 positions, plus synthetic cells at every position
  1..128 for the slab view.
- Fresh Gaussian noise $\sigma g_i$ ($g_i \sim N(0, I/32)$, so $\lVert \sigma g_i\rVert \approx \sigma$) is drawn in
  every cell, so no two cells are identical. **$\sigma = 1.3$** in every cell (v2 had 0.3; see "Why $\sigma = 1.3$").
- **Planted hierarchy.** Each item has a code $(b_1, b_2, b_3) \in \{\pm1\}^3$, giving 8 balanced leaves of 125.
  - The item's vector is $x_i = a_1 b_1 u_1 + a_2 b_2 u_{2,b_1} + a_3 b_3 u_{3,b_1 b_2} + \sigma g_i$, where the 7
    vectors $u$ are orthonormal and fixed across cells.
  - The amplitudes are $f \cdot (1.0, 0.7, 0.5)$.
  - The factor is $f = \min(1, \frac{i_{\text{step}}+1}{5}\cdot\frac{i_{\text{layer}}+2}{5})$, using 0-based
    indices. So f = 1 in all of T1's cells (L8 at step 143000), and structure "emerges" along training and depth.
- **Planted block demotion**, in the 6 cells of the reading block at L4, step 143000 (and at every slab position of
  that layer). In these cells the hierarchy's amplitudes are 0, so only the following and the noise are present.
  - A persistent split $b_p$ (balanced, independent of the hierarchy, on its own unit direction) has amplitude 0.6 in
    all 6 cells.
  - A transient split $b_q$ (balanced, independent of $b_p$, on another orthogonal direction) has amplitude 1.2 at
    t = 32 only, and 0 elsewhere.
- Probe S cells (t = 128, step 143000) are the same planted model with fresh noise.
- **Probe and tokens are fake.** Sources follow the planted leaves in blocks of 200 (leaves sorted, then cut into five
  runs of 200), so the validation colouring is partly aligned with the hierarchy; items are then in random order.
  Tokens come from a 512-word fake vocabulary, half from a 64-token band per source and half uniform; each checkpoint
  has a random embedding matrix for the bag view; probe S uses the real rule, so C2 is a real check. C1 is recorded
  as not run (no model). Any page built from the fixture carries "MOCKUP" in its title and file name.
- `probe.json` carries `planted` (`leaf` = 4[b_1>0] + 2[b_2>0] + [b_3>0], `persistent`, `transient`, $\sigma$,
  amplitudes); real probes have no such key.

**`data/_fixture_flat/`.** The same schema and code path with every cell (all checkpoints, layers, positions, and
probe S) fresh isotropic Gaussian noise $N(0, I_{32})$, N = 1,000: what "no structure" looks like everywhere on the
page. The structureless root used by P1 is one such cell (`lm_schema.flat_cell`).

**Why $\sigma = 1.3$.** The kernel is rank-based: each item's edges go to its 15 nearest neighbours (weight 1), and
every other pair gets the same $10^{-6}$ floor. A planted grouping can only steer the sweep if the graph records it:
inside each planted node, edges must cross between the children's sub-groups, denser between sibling leaves than
between cousins. With $\sigma = 0.3$ the leaves are compact (sibling centres $2a_3 = 1.0$ apart, within-leaf
distances about $0.42$), and the union-15NN graph has no edge between leaves in any cell. Every inter-leaf pair then
gets the same floor, so the graph no longer says which leaves are siblings. The root sweep picks four whole leaves
that are not a side of the planted root split (reproduced: per-leaf counts [0, 125, 0, 0, 125, 125, 0, 125]), and
$\Upsilon \approx 2$ at depths 0 and 1 is a ratio of floor terms. P2 fails the same way: at $\sigma = 0.3$ the
transient cell has four disconnected components, and its top cut is the persistent split.

Measured through the real code path (`node_walks`, `analyse_node`, `grow`) on independent draws of P1's block (6 cells
at f = 1; seeds 0 to 9, and 0 to 19 at $\sigma = 1.3$). Edges are union-15NN edges per cell, averaged over two cells:
"sibling" joins two leaves that differ in $b_3$ only, "cousin" two that differ in $b_2$ but not $b_1$, "across"
two that differ in $b_1$. "Structureless" here is the extra split of each planted leaf.

| $\sigma$ | edges: sibling / cousin / across | exact recovery | min planted $\Upsilon$ | max structureless $\Upsilon$ |
|---|---|---|---|---|
| 0.3 | 0 / 0 / 0 | 0 / 2 | 2.0 | 1.87 |
| 1.0 | 286 / 2 / 0 | 10 / 10 | 2.0 (some draws: cousins disconnected) | 1.89 |
| 1.1 | 490 / 9 / 0 | 10 / 10 | 14.6 | 1.89 |
| **1.3** | **1,000 / 74 / 0.2** | **20 / 20** | **7.6** | **1.89** |
| 1.5 | 1,516 / 224 / 2.5 | 10 / 10 | 5.0 | 1.89 |
| 1.8 | 2,110 / 622 / 31 | 9 / 10 | 3.4 | 1.90 |

$\sigma = 1.3$ is the middle of the window where recovery is exact and $\Upsilon$ separates at the fixed threshold 3
with a wide margin; the flat cell's root $\Upsilon$ is 1.34 to 1.41 (13 draws). P2 at $\sigma = 1.3$ with its v2
amplitudes passes on 13 of 13 draws (sweep and SEBA r = 2 matched Jaccard 1.0 in both the transient cell and the
block). At $\sigma = 1.3$ the two halves of the root split are themselves almost disconnected (0.2 edges across them
per cell on average, at most 1.5): two components force the root cut, so they do not make it ambiguous, but the
root's $\Upsilon$ (186 or more) involves floor-level terms and reads only as "clear".

## 11. The page

### 11.1 Form and style

- The output is one self-contained HTML file, `explore/lm_coherent/page/lm_coherent.html`, built by
  `page/build_page.py p160t`.
  - It has no runtime fetches. Data are embedded as gzip+base64 blobs and decoded with `DecompressionStream`, so it
    works from file://.
  - The target size is **under 8 MB**. The builder prints the actual size and fails above 10 MB.
- It mimics the ResNet site by **copying**, never editing, these files from `cifar_coherent/site/`:
  - `css/site.css` and `palette.json`;
  - the vendored fonts and KaTeX, inlined;
  - the dependency-free renderers `js/charts.js` and, from `js/explorer.js`, the canvas painter, `matchedJaccard`
    and `hungarianMax`.

  The copies are adapted in `page/js/` and bundled into one inline module.
- The look is the ResNet one: banner, masthead, tabs, `figure.fig` captions with **Plotted:** / **Values:**,
  `{{fact:...}}` substitution, callouts and the colour grammar.
  - Sources use `classes[0..4]`, as validation colours.
  - Sets use `sets`. Eigenfunctions use the diverging scale.
- This departs from the `notebook.py`/Plotly engine of the experiment-notebooks skill (decision D9). The skill's
  other rules hold:
  - the figure lines below are the checklist;
  - there is an architecture banner on every tab;
  - the decisions table appears on the tab it affects;
  - one builder, one figures script and one verify script;
  - screenshots of every tab, with zero console errors.

### 11.2 Three axes without overload

- No figure has three axes. Every figure is either layer (rows) x position (columns), or a line over one axis.
- Training enters in one of three forms only:
  - **small multiples**: the 5 checkpoints side by side, with one fixed colour range;
  - **one strip** under the explorer grid;
  - **lines** on a log-step axis, with step 0 at the left.
- There is no checkpoint slider in the explorer, and there are no gifs: five panels side by side read at a glance.
  This is the skill's stated exception to its gif default.

### 11.3 Tabs

**Overview.**

- The question.
- A boxed measured quantity: the averaged-walk penalty, plus $T$ and $W$.
- The labels sentence.
- The procedure in five steps, in the ResNet structure.
- F0, F1 and F10.
- The experiment map.
- The verdict box against section 8.

**Explorer.** See section 11.4.

**Zoom.** Figure F9.

**Measurements.** Organised by question:

- Q0, controls (F2);
- Q1, token or source (F3);
- Q2, maths vs poetry (F4, F5);
- Q3, position-t residual vs prefix slab (F6);
- Q4, through training (F7);
- Q5, order control (F8).

**Method.**

- Definitions and the glossary.
- The deviations listed in section 5.
- The decisions table (section 12).
- The hazards (F11).
- How to reproduce.
- References.

### 11.4 Explorer

```
[colour by v] [view: point | slab] [pin: follow items]
            t=4    8     16    32    64    128      | side panel: block; spectrum (lambda_1..8, PN);
  bag      [ ]   [ ]   [ ]   [ ]   [ ]   [ ]        |   retention strip along the block's axis; legend:
  L2       [ ]   ...                                |   set sizes + source make-up (validation); matched
  ...                                               |   Jaccard vs sources + null; representative snippets
  L12      ...                                      |   (tokens <= t dark, > t grey, token t underlined);
  -- layer L8 at t = 128 through training --        |   hovered and pinned snippets
  step 0   512   2k   16k   143k
  generated caption
```

- **Layouts.** The grid is at step 143000. Each row (the bag row and the six layers) is one `umap.AlignedUMAP` run
  across the 6 positions, with `n_neighbors=15`, `min_dist=0.1`, `alignment_regularisation=0.02`,
  `random_state=0`, on the top 50 PCs of the cell.
- The strip shows the selected layer at t = 128 across the 5 checkpoints, one AlignedUMAP run per layer.
- That is 13 runs in all. Layouts are for display only, and the page says so.
- **Selection.**
  - Click a cell for its single-cell block; that also sets the strip's layer.
  - Click a row label for that layer's reading block.
  - Click a column header for that position's depth block.
  - Use the strip's two buttons for the training blocks, "all steps" and "2k to 143k". The strip's cells at other
    checkpoints are single-cell blocks too.
- **Colour by.**
  - Label-free: sets r = 2 and r = 4; $\psi_1$; $\psi_2$; the nodal domains of $\psi_1$; the token at t (the 10 most
    frequent tokens at that position, the rest grey).
  - Validation: source; own-source walk share.
- **View: slab.** This paints the slab single-cell sets and eigenfunctions onto the same point-view layouts.

### 11.5 Figure lines

Each line gives what is plotted, the axes, the null or uncertainty, and the sentence the reader should be able to
say. Words that depend on the results are in [brackets]. F0, F1, F3 to F7, F9 and the Explorer are core.

- **F0 Architecture banner** (SVG, from `config.json`, on every tab).
  - A token row (EOT + 128, sampled positions marked) over a stack of 12 parallel-residual blocks, with the 6 layers
    tagged.
  - *"A cell is (checkpoint, layer, position); t = tokens read."*
- **F1 Hero** (PNG).
  - The explorer grid at step 143000, coloured by the r = 4 sets of T1's block (the outlined row); grey is
    unassigned.
  - Values: $\lambda_{1..3}$, the gap, set sizes, and the range of retention.
  - *"The L8 reading block's sets are [visible from t = .. at layers ..]."*
- **F2 Controls.**
  - A table of C1, C2 and P1 to P3: the measured deviation or agreement, and pass/fail.
  - The fixture tree as an icicle, planted against recovered.
  - *"The indexing and bag identities hold exactly; the recursion recovers the planted tree; structureless splits read
    as weak."*
- **F3 Token vs source.**
  - Two rows of five small-multiple heatmaps, one panel per checkpoint, each layer x position. Top row:
    $\tilde T$. Bottom row: $\tilde W$.
  - The range is fixed at [0, 1], and the bag row is included.
  - Hover gives the raw value, the chance level and the permutation band.
  - *"At the end of training the token just read [dominates at L2]; source separation [grows with t at layers ..];
    at step 0 [..]."*
- **F4 Maths vs poetry.**
  - Five small-multiple heatmaps of $\log_2 X(\text{maths},\text{poetry})$, diverging at 0.
  - A pair selector, with wiki vs forum as the prose-to-prose contrast.
  - Clicking a cell shows the full 5 x 5 $X$ there. Hover gives the band.
  - *"The walk stops linking maths and poetry from [t = ..] at [layers ..], [earlier/later] than wiki and forum."*
- **F5 Follow two snippets.**
  - Top: the two snippets' text as tokens, with the sampled positions underlined.
  - Below: three layer x position grids at the selected checkpoint (5 buttons):
    - one for each snippet, where each square is a stacked bar of that snippet's walk mass by source;
    - one for co-membership in the cell's own r = 4 sets (filled = same set; open = different; grey = unassigned).
  - Selectors offer the 10 follow items. The defaults are the first `maths` and the first `poetry` item in probe
    order, which is random, so the choice does not depend on results.
  - *"This maths snippet's walk goes mostly to maths from [t = ..]; the poetry one from [..]; they share a set only
    [..]."*
- **F6 Position-t residual vs prefix slab.**
  - Five small-multiple heatmaps of $a_3$ (point vs slab), range [0, 1], with the null 0.003 printed.
  - Hover gives $a_6$, and the r = 4 matched Jaccard with its null.
  - *"The cheap view and the slab agree [above ..] at [layers ..] from [t = ..], and disagree [at L2/early t]."*
- **F7 Through training.** Lines on a log-step axis.
  - Left: $\tilde W$ at t = 128, one line per layer, with the bag line dashed.
  - Right: matched Jaccard between each checkpoint's r = 4 sets and step 143000's, at (L8, 16) and (L8, 128), with
    the null band.
  - *"Source separation at L8 appears between steps [..]; the final sets are in place by [..]."*
- **F8 Order control** (t = 128, step 143000).
  - Left: $\tilde W$ by layer, real vs S, with the bag value as a reference line.
  - Right: matched Jaccard between the real and S sets by layer, with the null.
  - *"Reordering tokens 1..127 with token 128 kept changes the sets [little/much] at [layers ..]."* This is stated as
    an association only.
- **F9 Zoom.**
  - An icicle of the selected tree (T1 to T4). Width is node size; each node carries a source make-up bar; weak splits
    are faded.
  - For the clicked node:
    - (a) the items placed at ($\sqrt n$ times the winning $\psi$, $\sqrt n$ times the highest-ranked other of
      $\psi_1$, $\psi_2$), coloured by child, with the cut as a vertical line;
    - (b) the sweep profile with $m_{\min}$, $\phi^*$, $\phi_{\text{free}}$ and the Cheeger bounds;
    - (c) the retention strip of the cut;
    - (d) SEBA r = 2 against the cut;
    - (e) 6 representative snippets per child.
  - *"T1's first cut separates [..] ($\Upsilon$ = ..); [below depth .. the splits are weak]."*
- **F10 Three axes** (Overview schematic). A checkpoint x layer x position cube, highlighting one reading block, one
  depth block and one training block.
  - *"A block is a run of cells along one axis; its operator is their average."*
- **F11 Method diagnostics** (collapsible).
  - Residual norm against position, per layer.
  - Heatmap of the top-1 coordinate share of variance.
  - Heatmaps of $\mathrm{PN}(\psi_1)$, the smallest r = 4 set, and the smallest pairwise distance.
  - *"None of these [changes the reading], or here is where one does."*

Captions are definitional. Interpretation goes in the Measurements prose and the verdict, citing the targets of
section 8.

### 11.6 Payload (raw, before gzip and base64)

| item | MB |
|---|---|
| Layouts: 42 grid + 24 strip cells, N x 2 uint8 | 0.13 |
| Single cells (point 180, slab at step 143000 36, bag 30). Each has $\psi_1, \psi_2$ int8, labels for r = 2 and 4 as nibbles, $\lambda$, PN, retention and top members, about 3.4 KB | 0.84 |
| 48 blocks at about 3 KB each | 0.15 |
| Own-source share for the point cells | 0.18 |
| Tokens and the string table | 0.4 |
| Statistics, comparisons and nulls (JSON) | 0.3 |
| 4 trees plus the fixture tree | 0.15 |
| Static figures (hero, pipeline, schematic) | 1.0 |
| Fonts, KaTeX, CSS and JS | 1.3 |

The raw total is about 4.5 MB, so the expected HTML is **5 to 6 MB**.

## 12. Decisions made on the owner's behalf

| # | decision | default | alternative | test |
|---|---|---|---|---|
| D1 | Model | pythia-160m; the pilot is the final checkpoint | pythia-70m; the deduped variant | rerun F3 on 70m |
| D2 | Probe | 4 Pile-test sources plus Gutenberg poetry, 200 each | one corpus throughout; more sources | a source swap is a rerun |
| D3 | Positions | T = 128, EOT prefix, $t \in \{4..128\}$ log2 | include t = 1, 2 (needs a policy for duplicate items); linear spacing | none |
| D4 | Layers and steps | L2..L12 (even); steps 0, 512, 2k, 16k, 143k | all 12 layers; denser steps | units are independent |
| D5 | Kernel | the ResNet union kNN, k = 15 | perplexity 100; k = 8 or 30 | not run (toy) |
| D6 | Geometry | plain cosine of centred residuals | z-scored coordinates, a metric variant | not run; the rogue share is reported |
| D7 | Prefix view | aligned slab, uniform weights | lag-weighted window (`THEORY.md` 6.3); pooled | none |
| D8 | Zoom | rebuild on the subset; sweep over $\psi_{1..3}$ with sides $\ge \max(16, 5\%)$; depth 3; $\Upsilon$ read at 3, fixed on the fixture | restrict-and-rescale; $\psi_1$ only; SEBA cut | `axis_used`; SEBA vs cut per node |
| D9 | Page engine | one file with copied ResNet CSS and JS | `notebook.py` + Plotly | owner preference |
| D10 | Training format | small multiples, a strip and lines; no slider | an explorer with a checkpoint slider (needs 5x the layouts) | none |
| D11 | Publication | private page via `publish-results`; not `gh-pages` | public, which needs a licence review of the snippets | owner decides |

## 13. Units, compute, spend

**Prices.** These are the modal.com/pricing rates recorded on 2026-11-12 in `cifar_coherent/README.md`, to be
re-checked before launch:

- L40S: $0.000542/s
- CPU: $0.0000131 per core-second
- RAM: $0.00000222 per GiB-second

**Coordination.** A detached coordinator runs per the `modal` skill, with commands `deploy | launch <stage> |
status | fetch`. It keeps one part per unit, fingerprinted completion records, and launch records in `runs/`.

| unit | count | hardware | expected | timeout | worst-case cost |
|---|---|---|---|---|---|
| U0 probe | 1 | CPU, 4 cores, 16 GiB | 15 min | 30 min | $0.16 |
| U1 per step: load checkpoint (650 MB, cached on `/hf`); forward passes of A (and of S and the truncated pass at 143000); save point activations at the sampled positions (110 MB); cosines for point, slab and bag (GPU, float64) | 5 | L40S + 4 cores, 16 GiB | 4 min | 15 min | $2.85 |
| U2 per step: kernels, Sinkhorn, single cells, reading and depth blocks, statistics, comparisons, that step's trees | 5 | CPU, 8 cores, 32 GiB | 15 min | 40 min | $2.1 |
| U3: training blocks, the 13 layout runs, figdata, validate | 1 | CPU, 8 cores, 32 GiB | 25 min | 40 min | $0.42 |

**Spend.**

- Projected: about $1.9 (U1 about $0.76, U2 about $0.79, U0 and U3 about $0.34). The pilot (U1 and U2 at step 143000)
  is reused, not repeated.
- Budget control: before launching anything, the coordinator reserves the **worst case of every unit**, that is,
  timeout x rate, about $5.5 in total. It refuses to start if that exceeds the ceiling of **$6**. In-flight units
  therefore cannot overshoot.
- After the pilot, the timeouts are recalibrated to 2x the pilot's measured durations, and the reservation is
  recomputed.

**Wall-clock.** The pilot takes about 25 min. The rest takes about 30 min: U0, then U1 in parallel, then U2 in
parallel, then U3.

**The VPS orchestrates only.** It uses `TMPDIR=/var/tmp/cm-lm`, fetches about 60 MB, and builds the page and figures
with numpy and matplotlib.

Run ids, spend and status go in `explore/lm_coherent/README.md`. The page is published with `publish-results`.

**Files.** `explore/lm_coherent/` holds:

- `probe.py`
- `lm_construction.py`
- `lm_cells.py` (views, statistics, assertions)
- `lm_tree.py`
- `lm_schema.py` (fixtures, validation)
- `lm_modal.py`
- `test_lm.py`
- `page/`: `build_page.py`, `make_figures.py`, `verify.sh`, `content/`, `js/`, `css/`

The pipeline and the page can be built in parallel once `data/_fixture/` exists.

**Tests** (`test_lm.py`, CPU, before any Modal run):

- P1 to P3;
- that the sweep never separates equal values;
- the Cheeger bounds for $\phi_{\text{free}}$ on random graphs;
- the slab Gram identity;
- the bag view's invariance under permutation;
- the zero-norm policy;
- SEBA reconstruction (reusing the ResNet regression test);
- fingerprint and cache rejection.

**Review.**

- This spec: the same `model-router:plan-reviewer`.
- The code: `model-router:adversarial-code-reviewer`.
- The copy: `model-router:gpt-6.1-sol` for accuracy, then `model-router:gpt-6.1-terra` for fresh eyes.
- Screenshots at 1280, 1440, 1920 and 390 px, with zero console errors.

**Open points for the owner** (provisional defaults are in force):

- Publishing the snippet text publicly. Wikipedia and StackExchange are CC BY-SA; GitHub and arXiv licences vary.
  Default: private.
- No held-out poetry corpus has been identified.
- Disconnected kNN components in the zoom (hazard table, section 9). Adopted as a diagnostic only (v2.1, approved
  by the plan reviewer): it is stored per split node and changes no cut. Whether the tree should treat a node with
  three or more components differently (an unordered or multiway split) is open. Default: no change to the tree.

## 14. Changes

### 14.1 From v1 (review of 2026-11-12)

- **The pooled view was not permutation-invariant, so I3 was false.** The pooled view, I3 and the bag-of-tokens
  readings are removed. A true bag view (the mean of raw input embeddings, summed in sorted order) is added as a
  reference, and C2 checks its identity.
- **The fixture was impossible** (24 leaves against a limit of 16, and a non-binary top level). It is replaced by a
  binary 8-leaf hierarchy with explicit amplitudes and exact node matching up to the depth limit, a separate
  structureless fixture, and an explicit P2 with defined persistent and transient splits.
- **Duplicate-preserving cuts were not guaranteed.** Duplicates are removed from the design instead of being handled:
  multiset de-duplication at every sampled t, positions $\ge 4$, and no embedding row. A pipeline assertion fails
  the run if any exact duplicate appears, and there is an explicit zero-norm policy. Tie-inclusive graphs,
  canonicalisation, the duplicate eigenvector claim, and I1, I2 and I4 are all removed. A quotient operator was not
  adopted, because nothing needs it any more.
- **The compact operator format was flawed** (int16 row pointers; `symmetric_balance` returns P; the rebuild was not
  bit-identical). It is removed. Dense float64 P stays on the work volume, only where later units need it, and the
  page gets derived arrays.
- **The shuffled control did not isolate the current token.** Probe S now permutes tokens 1..127, keeps token 128,
  and is read at t = 128 only. $T$ and $W$ are stated to be associations, not a decomposition.
- **H4 and H6 overclaimed.** All hypotheses are rewritten as numerical predictions without claims about mechanism.
- **A $\psi_3$ cut could not be drawn on the $(\psi_1, \psi_2)$ plot.** All three $\psi$ are stored, and the node
  scatter uses the winning $\psi$ as its x axis.
- **The ceiling ignored in-flight units.** Every unit now has a timeout, the coordinator reserves the whole run's
  worst case before launch, and the timeouts are recalibrated from the pilot.
- **Scope cut to a toy:**

  | | v1 | v2 |
  |---|---|---|
  | checkpoints | 8 | 5 |
  | layers | 8 | 6 (plus the bag reference) |
  | positions | 8 | 6 |
  | sources | 6 | 5 |
  | N | 1,200 | 1,000 |
  | blocks shipped | 1,232 | 48 |
  | trees | 21 | 4 |
  | layout runs | 72 | 13 |
  | spend | $7 | $1.9 |
  | page | 13 MB | 5–6 MB |

  Dropped altogether: probe B, the pooled view, the `out` view, the k-sensitivity and z-score variants, the
  restrict-variant run, gifs, and the explorer's checkpoint slider.

### 14.2 v2.1: implementation (2026-11-12)

**The fixture (review finding, resolved empirically).** The reviewer showed, through the real operator, that the v2
P1 fixture could not pass: with compact leaves ($\sigma = 0.3$) the union-15NN graphs have no edge between leaves,
so every inter-leaf pair gets the same floor and the root cut is an arbitrary union of whole leaves. This was
reproduced, and P2 fails for the same reason. The fixture's noise is raised to $\sigma = 1.3$ in every cell, with
all amplitudes unchanged; the evidence (20 of 20 draws exact, planted $\Upsilon \ge 7.6$ against structureless
$\le 1.89$; P2 13 of 13) is in section 10.2. P1 and P2 keep their v2 wording; P1 now names the seeds the test runs.
The test suite keeps the $\sigma = 0.3$ regime as a negative regression. The hazard table gains the disconnected
components row. `_fixture_flat` is the full schema with every cell isotropic noise (v2: "one cell").

**Implementation choices** (smallest sensible reading of the spec; none touches a kernel, operator, basis or cut):

1. Ties in the kNN rule are broken by lower index through a stable argsort. Without ties this equals the ResNet
   `topk` rule exactly (tested against `sweep_math.final_kernel_bank`).
2. U1 computes the slab Grams on the GPU in float64. The point and bag Grams are computed in U2 (CPU, float64, the
   same `standard_gram`) from the saved float32 activations and float64 bag vectors, which is the same arithmetic.
3. Zero-norm policy: U1 scans every input (all 128 positions per layer) and, if it finds an item, records it and fails;
   the coordinator writes `removed.json`, and a relaunch reruns U1 without those items (`meta.removed_items`,
   `meta.items`). Inside a zoom node a zero-norm item fails the tree rather than being removed.
4. Permutation bands are the 2.5 and 97.5 percentiles of the 200 draws of the raw value (not the excess), from one
   bank (`default_rng(20261112)`) shared by every statistic. A block's value and each of its draws are the mean of
   its cells' (so P3 holds draw by draw). Matched-Jaccard nulls are stored as [mean, 2.5 %, 97.5 %], the ResNet
   convention; the ResNet `site/analysis.py` functions are imported, not copied.
5. De-duplication draws round-robin over the sources (maths, code, wiki, forum, poetry, maths, ...), each taking its
   next shuffled document. Snippets are tokenised on an 8,000-character window, widened when the window could cut
   the 128th token.
6. `top_members[r, j]` lists the members of hard set j by decreasing soft SEBA value, reconstructed from the
   exported transform. Slab cells have no coordinates, so their `rogue1`/`rogue5` are NaN.
7. Trees: split nodes also store `cut_m`, `profile_m` (the 64 prefix sizes of `profile`) and `m_min`; `vals` are
   $\lambda_1..\lambda_4$ of $\bar M^S$. Leaves store only `id`, `parent`, `depth`, `members`, `n`, `makeup`,
   `children`. `trees.json` ends with a fifth entry, `P1` (the planted tree, its node matching, the structureless
   $\Upsilon$ values and the threshold rule); every tree carries the calibrated threshold.
8. `compare.json`: `real_vs_S` has a bag row (`layer` = "bag") whose S side is A's bag cell, identical by C2;
   `to_final` $a_r$ uses the int16-decoded eigenvectors.
9. `figdata.npz` holds C, K and P of the cell (L8, t = 128) and $\bar M$ of T1's block, ordered by T1's root cut
   (child 0 first, each side by decreasing winning $\psi$), block-averaged to 250 x 250.
10. Volume `cm-lm-coherent` is mounted at `/vol` (`hf/step<k>`, `work/p160t`, `analysis/p160t`); the HF cache has
    one directory per revision so parallel U1 units never write the same files. `transformers` is pinned to 4.57.6
    and torch to 2.15.0. The pilot launch runs U0 first. The coordinator has two extra commands: `budget` (prints
    the reservation) and `calibrate` (writes the 2x-pilot timeouts to `runs/timeouts.json`, read at deploy).

### 14.3 The page (2026-11-12, built against the fixtures)

Simplifications and choices made while building `page/`, for intelligibility or because the data do not carry
something; none touches a measurement.

- **F11** is one row of five checkpoint panels (rows bag and L2..L12, columns $t$) with a quantity selector (top-1
  share, top-5 share, $\mathrm{PN}(\psi_1)$, smallest $r = 4$ set, smallest pairwise distance) instead of separate
  heatmap sets per quantity; the residual-norm lines are kept.
- **F2**: shows C1 (float64 gate, float32 error per layer), C2, P1, P2 and P3 from `validation.json`; the P1 icicle
  colours each recovered node by the planted leaves it contains.
- **F9 (d)**: SEBA $r = 2$ against the cut is shown as the matched Jaccard, the unassigned count and a 3 x 2 table of
  SEBA sets against the cut's sides; (a) can be coloured by the node's SEBA labels. (e) shows tokens 1..36 for the reading
  trees (no single $t$) and tokens 93..128 for T3. Υ labels say "clustering ratio Υ" (the glyph reads as Y in Inter).
- **F7 (right)**: step 143000 is drawn at 1 (the reference itself).
- **Explorer**: the bag row's panels are selectable single cells (bag view); the bag has no reading block, so its row
  label is not a button. Representative snippets use the block's $t$ (single cells and depth blocks) or $t = 128$
  (reading and training blocks). Matched Jaccard against T1's block is computed in the browser (the copied
  `matchedJaccard`, no browser null).
- **Payload**: only the blocks the explorer can select are shipped (6 reading and 6 depth blocks at the final step,
  12 training blocks: 24, not 48); reading blocks at earlier steps are not reachable from the step-143000 grid.
  Single cells: point at every step (180), slab and bag at the final step (36 + 6). Fixture page: 4.4 MB.
- **F0** is drawn from `meta.json` (layers, positions, steps, `model_config`).
- Verdict statuses (E1, H1..H5) and control statuses are computed by `build_page.py` from the data, with the
  measured values printed; interpretation stays in prose. Result-dependent wording is a striped placeholder until the
  real run.
11. **Stage-2 review fixes** (adversarial code review of `lm_modal.py`):
    - Spending: an append-only attempts ledger (`work/p160t/attempts/`, one file per dispatched attempt, the
      coordinator included) is written at the unit's full worst case before dispatch and reconciled on return:
      success at the worker's timed cost plus 90 s of container overhead, failure at the wall time since dispatch
      (capped at the timeout) plus the same overhead; an attempt that never returns stays at its full worst case.
      Every attempt counts, whatever its fingerprint. Before `launch` and before every batch, attempts + the worst
      case of every incomplete unit (+ the coordinator's, at launch) must stay within $6.
    - The pilot stage writes `pilot_gate.json` (section 8.1, with the numbers), stamped with the construction
      fingerprint, the probe's sha256 and the removal list; `launch full` refuses locally and the coordinator
      refuses remotely unless it passes and matches.
    - Every derived unit's completion record carries the same key (fingerprint, probe sha256, removal list), so a
      result made before a removal is never reused after it.
12. **Additions:** the per-node component diagnostic (hazard table); P3 is also checked on real data in every U2 (the
    L8 reading block: conductance of 5 random label-free sets, and T, W, X with their permutation draws, against the
    averaged walk, to $10^{-12}$; the slab identity stays a test, since U2 has no per-position activations);
    `validation.json` reports C1 as `max_rel_err` + `pass` and P1 to P3 with numbers; `meta.json` has `model_config`;
    tree nodes have `seba2`; `figdata.npz` has `M_cells`.
13. **Pilot 1 (2026-11-12): C1 failed at the gate; the run stopped.** (Resolved in item 14.) Float32 max relative error 8.6e-4 against the
    1e-4 tolerance (C2 passed bitwise). A ledgered diagnostic (`lm_modal.py diag`, 200 items) found TF32 off
    (`cuda.matmul.fp32_precision` = ieee), the same error with IEEE forced everywhere (3.9e-4), and an error of
    exactly 0.0 with the model in float64: the truncated and full passes are identical in exact arithmetic, and
    the float32 discrepancy grows with depth (about 1e-6 at L2, 1e-5 at L4, 1e-4 to 4e-4 at L6 to L12). U1 now also
    records `C1.float64_max_rel_err` (informational). The gate criterion is unchanged pending the owner's decision
    (options: C1 in float64 at a tolerance such as 1e-10, or a float32 tolerance of 1e-3).
14. **C1 is gated in float64** (decision taken by the orchestrator on the owner's behalf, 2026-11-12; the alternative
    was a float32 tolerance of 1e-3). C1 exists to verify that position t means tokens 1..t were read; float64
    agreement of exactly 0 establishes that more strictly than the original check, and the float32 discrepancy is
    rounding amplified through depth, not indexing. U1 now gates on the float64 model at 1e-10 and records the
    float32 error per layer (`C1.float32_by_layer`) as a reported hazard (section 9).
15. **Calibrated timeouts have a 120 s floor**: 2 x the pilot's durations would have been 67 s (U0), 113 s (U1) and
    100 s (U2), which leaves too little margin for a container cold start or a model download. U3 keeps 2,400 s,
    since it has no pilot.

Added after the real run (2026-11-12):

- **F3** uses a square-root colour scale on [0, 1], shared by both rows (on the real data $\tilde T \le 0.22$ was
  nearly invisible on a linear scale).
- **Two figures were added.** "Which blocks find sets that line up with sources" (Measurements, Findings 1) plots
  the matched Jaccard of reading blocks, single cells, slab cells and the bag against sources, by layer and through
  training; it needs the reading-block summary at every step, which the build now ships as JSON. "Clustering ratios"
  (Zoom, Findings 5) is a strip plot of $\Upsilon$ for every split of T1 to T4 beside the fixture's planted and
  structureless splits. A rogue-share-only version of F11 sits in Findings 9, where the hazard is discussed.
- **Step 3** of the Overview shows four of the L8 block's cell walks beside $\bar M$ (`figdata.M_cells`).
- **Explorer:** deep links (`#explorer?sel=reading:8&c=sets4`, `cell:<step>:<L|bag>:<t>`, `depth:<t>`,
  `train:<L>:<run>`, `v=s`, `pin=`) and `#zoom?tree=T1&node=0`, so each finding opens its view in one click. Panels
  are at least 108 px on desktop, because the real layouts have fine structure.
- Overview opens with a short answer to "how far does this work out of the box" and a "where to look" list. A
  Findings tab with nine tagged findings (observed / reading / speculation / not determinable) and a list of checks
  was added; the Method hazards table gained the rogue-coordinate, float32, de-duplication, positions 1 to 3,
  replicate-probe and threshold-calibration rows, and the decisions table gained D12 (C1 gated in float64).
