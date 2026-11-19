# Coherent sets in a small Mamba model, with the whole recurrent state (toy, exploratory): spec

Draft v1, 2026-11-12. This is a **delta spec** against `../lm_coherent/SPEC.md` v2.1 ("the Pythia study"). Anything
not changed here is inherited unchanged: the construction (section 5 there), the walk shares, mixing ratio, span
agreement and matched Jaccard (1.2), the label rules (1.3), the probe (3), the zoom recursion (7), the fixtures and
P1/P2/P3 (8, 10.2), the hazards (9), the page style (11.1), and the decisions there that this spec does not reopen.

Like the Pythia study, this is an exploratory side study in `explore/`, not an `EXPERIMENTS.md` stage, and no claim
enters `THEORY.md` from it. The working language is `THEORY.md`'s: no `THEOREMS.md`/`SPARSITY.md` vocabulary.

## 1. Why Mamba, and the question

In a causal transformer, the information carried forward from a prefix is the KV cache. It grows with t, and the
Pythia study could only reach it through a proxy, the prefix slab $(h^l_u)_{u\le t}$. A selective state-space model
carries a **fixed-size recurrent state** instead: after reading token t, layer l holds a conv buffer and an SSM
state, and the collection of these over all layers is *everything* the model keeps about the prefix. So the whole
state is directly measurable, as one vector per item per position.

Questions (a), (b) and (d) are the Pythia study's, asked of Mamba. (c) is replaced and (e) is new:

- (c') Do the sets from the residual at position t agree with the sets from layer l's own recurrent state, and with
  the whole model state (all layers' states)? Where along reading and depth do they part?
- (e) On the **same 1,000 snippets**, how far do Mamba's coherent sets agree with Pythia's at matched relative
  depth? (Both models were trained on the Pile, about 300B tokens, with the same tokenizer, so the Pythia probe is
  reused verbatim and items correspond one to one.)

## 2. Substrate

- **Model.** `state-spaces/mamba-130m-hf`, revision pinned to commit `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`
  (checked 2026-11-12). Mamba-1, 24 layers, width $D = 768$, inner width $E = 1536$, SSM state size $S = 16$, conv
  kernel 4, RMSNorm, `residual_in_fp32`. Vocabulary 50,280 (GPT-NeoX-20B tokenizer).
  - HF `MambaForCausalLM`, pinned `transformers` (same pin as the Pythia study unless it lacks the needed cache API;
    record the version). The **torch path only**: `mamba-ssm` and `causal-conv1d` must not be installed, so the
    scan runs in HF's reference implementation. Eval mode, TF32 off.
- **Checkpoints.** No intermediate training checkpoints of the official Mamba models are published (searched
  2026-11-12), so the training axis of the Pythia study is dropped. Two checkpoints:
  - `trained`: the pinned revision. This is the spine.
  - `init`: `MambaForCausalLM(config)` after `torch.manual_seed(0)`, i.e. HF's `_init_weights` (which follows the
    Mamba reference init: S4D-real $A$, the $\Delta t$ init range). A comparison, like Pythia's step 0; the page
    says it is HF's re-implementation of the init, not the authors' saved initial state.
- **Layers (6):** `L4, L8, L12, L16, L20, L24`, the raw residual stream after block l (forward hook on
  `backbone.layers[l-1]`, its output; not `output_hidden_states`, whose last entry is post-`norm_f`). These sit at
  the same relative depths as Pythia's L2..L12. `L24` is pre-`norm_f`.
- **Positions.** As in the Pythia study: input `[EOT] + 128 tokens`, $\mathcal T = \{4,8,16,32,64,128\}$.
- **Probe.** `../lm_coherent/data/p160t/probe.json` copied verbatim (items, order, probe S). U0 asserts that the
  Mamba tokenizer re-encodes every snippet text to the stored token ids and that every id is below 50,280. If that
  fails, stop and report: the item correspondence for (e) rests on it.

## 3. The recurrent state

For each layer m, after the step that reads token t (state indexed the same way as the residual: "t tokens read"):

- the **conv buffer** $c^m_t \in \mathbb R^{E\times 3}$: the last three inputs to the depthwise conv, the only part
  of HF's 4-wide `conv_states` that the next step reads (the oldest column is discarded at the next step; the
  implementer checks this against the HF code and records it);
- the **SSM state** $z^m_t \in \mathbb R^{E\times S}$.

**[thm]** The tuple $\Sigma_t = (c^m_t, z^m_t)_{m=1}^{24}$, with the tokens $t+1, t+2, \dots$, determines every
later residual, state and next-token distribution. (This is the recurrence itself. It holds exactly for the
inputs to the next step and is the content of control C1b.) $\Sigma^{\le l}_t = (c^m_t, z^m_t)_{m\le l}$ with the
later tokens determines everything at layers $\le l$ at later positions, so in depth these stacks form a Markov
chain, as the Pythia slabs did. The point residual $h^l_t$ is not a state: what layer l+1 does next depends on
its own state, which is not a function of $h^l_t$ alone.

**How the state is obtained.** Recurrent decoding, one token at a time, with HF's cache, batched over items, on the
GPU. After each step read every layer's `conv_states` and `ssm_states` and, through hooks, the residuals. No
re-implementation of the scan.

## 4. Views

The standard cell pipeline (centre over items, row-normalise, cosine, float64) and the Gram-averaging rule of the
Pythia slab (mean of standard-pipeline cosine Grams of the constituents, then double-centre and renormalise) are
reused unchanged.

| view | $\phi(i)$ at (checkpoint, layer l, position t) | cells |
|---|---|---|
| **point** (spine) | $h^l_t(i)$, $D = 768$ | 2 x 6 x 6 = 72 |
| **state** | layer l's own state: Gram average of the two constituents $c^l_t$ (flattened, 4,608) and $z^l_t$ (flattened, 24,576), equal weight | 72 |
| **stack** | the state of layers $1..l$: Gram average over the 2l constituents $\{c^m_t, z^m_t\}_{m\le l}$, equal weight, all layers $m\le l$ (not only the sampled ones). At l = 24 this is the **whole model state** $\Sigma_t$ | 72 |
| **bag** (reference) | as in the Pythia study, with Mamba's input embedding | 2 x 6 = 12 |

- **Weights (decision D-M1, pro-tem as [P3] was for the slab).** Each constituent enters with equal weight after its
  own standard pipeline. The alternative, raw concatenation, would let the parameterisation's arbitrary scales
  decide the weighting (conv inputs and accumulated SSM products have unrelated units). Diagnostic, not a
  target: at the trained checkpoint, $a_3$ between each constituent alone ($c$-only, $z$-only) and the state view,
  per (layer, t), in `compare.json`.
- **Degenerate items** (Pythia section 5 policies) apply to every constituent: the duplicate assertion on every
  view's cosine, the zero-norm scan on every constituent. Probe rule (i) makes all prefixes distinct, so distinct
  states are expected; the assertion, not the expectation, decides.
- Stored: the point activations at sampled cells; the state and stack Grams (float64, [6 layers, 6 positions, N, N]
  each). The per-layer constituent Grams for all 24 layers are formed on the GPU and not kept.

## 5. Blocks

| family | blocks | count |
|---|---|---|
| reading | positions 4..128 at each layer, trained; **for each of point, state and stack** | 18 |
| depth | L4..L24 at each position, trained, point view | 6 |
| single cells | point, state, stack: 72 each; bag: 12 | 228 |

There are no training blocks. Per cell and per block, the Pythia statistics (T, W, X with bands, PN, smallest
set, source Jaccard with null and ceiling, own-source shares, follow items).

**Comparisons** (`compare.json`, Pythia schema plus kinds):

- `point_state`, `point_stack`, `state_stack`: $a_3$, $a_6$, r = 4 matched Jaccard with null, every (layer, t),
  trained. `point_stack` at l = 24 is "the residual against the whole model state".
- `real_vs_S` at t = 128, trained: point, state and stack views at each layer, and the bag.
- `to_final`: init vs trained single cells at (L16, 16) and (L16, 128), all three views.
- `state_parts`: section 4 diagnostic.
- `cross_arch` (new, computed post hoc locally by `cross_arch.py` from both studies' public files): Pythia L2k
  against Mamba L4k, k = 1..6, every t, step 143000 vs trained: point vs point, and Pythia slab vs Mamba state and
  vs Mamba stack. $a_3$, $a_6$, r = 4 matched Jaccard with its null. Asserts identical item lists and no removed
  items in either.

## 6. Trees

| tree | block | role |
|---|---|---|
| T1 (headline) | point reading block at L16, trained | as Pythia T1 (L8 there) |
| T2 | point reading block at L4 | early-layer contrast |
| T3 | single cell (L16, 128), point | one cell against a block |
| T4 | **whole-state** reading block (stack, L24), trained | the zoom on everything the model keeps |

The tree rebuild on a subset (Pythia 7.1 step 1: re-centre over S, rebuild kNN) must repeat the view's own
construction on S. For the point view, as before (raw vectors). For the stack view (T4), the construction centres
and row-normalises **each constituent separately**, which cannot be recovered from an averaged Gram, nor from
constituent cosine Grams. So U1 stores, for T4's cells only, the **raw (uncentred) inner-product Gram of every
constituent**: 48 constituents x 6 positions x 8 MB = 2.3 GB, float64, on the work volume, never fetched. The rebuild
on S: restrict each raw Gram to S; double-centre over S; diagonal-normalise to a cosine Gram; average over
constituents; then the stack's final double-centring and renormalisation; then the usual kNN, Sinkhorn and walk.
`lm_tree`'s operator provider is generalised to accept a Gram-based cell (the point view keeps its vector route).
A fixture test compares the Gram route with a full rebuild from raw synthetic constituent vectors on non-root
subsets, with constituents of unequal item norms, to 1e-12. Restricting the parent's centred Gram (Pythia 7.3) is
not allowed.

## 7. Controls and targets

| id | target | label |
|---|---|---|
| C1a | Recurrent decoding and HF's parallel full-sequence (prefill) forward give the same residuals at every sampled (layer, t), stock HF arithmetic with parameters and cache in float64, max relative error < 1e-4 (see note) | identity [thm], mixed-precision check |
| C1b | The state is the state: resuming decoding at t = 64 from the stored three-column conv buffers and SSM states, placed in a fresh cache (the conv buffer rebuilt into HF's four-column layout with the retained columns in their slots, decode branch, explicit `cache_position`), reproduces the continuous decode's residuals and states at every t in 65..128 to relative 1e-12. Production precision (float32) | identity [thm] |
| C2 | Bag at t = 128 bitwise identical for probes A and S | identity [thm] |
| P1, P2 | Unchanged (fixture, through the real operator and tree code) | planted |
| P3 | As Pythia, plus: the state and stack Grams equal the explicit concatenation of standard-pipeline-normalised constituents, scaled by $1/\sqrt{\#\text{constituents}}$, to 1e-12 | [thm] |
| H1 | Trained: $\tilde T$ higher at L4 than at L24 at every sampled t (Pythia's E1, which is established for transformers; here untested, so a hypothesis) | hypothesis |
| H2 | Trained: $\tilde W(128) - \tilde W(4) \ge 0.2$ at L12..L24, point view | hypothesis |
| H3 | $X(\text{maths},\text{poetry}) < 0.5$ at (L16, t = 16), point view | hypothesis |
| H4 | At t = 128, the whole-state cell (stack L24) has $\tilde W$ within 0.05 of, or above, the largest point-view $\tilde W$ over layers | hypothesis |
| H5 | At (L16, 128), point view, $\tilde W$ at init is less than half its trained value | hypothesis |
| H6 | The first cut of T1 puts at least 80% of `code` and at least 80% of `poetry` on opposite sides | hypothesis |
| H7 | `cross_arch` point vs point at t = 128: $a_3 \ge 0.3$ (null 0.003) at every matched layer pair from relative depth 1/2 on (Pythia L6..L12 vs Mamba L12..L24) | hypothesis |

**Note on precision (C1a).** HF 4.57.6's Mamba is not a float64 reference after `.double()`: `MambaRMSNorm`, the
`residual_in_fp32` path and parts of `slow_forward` (`A_log`, B, conv outputs) cast to float32. So the Pythia
study's float64 gate at 1e-10 cannot be inherited. The decode and prefill branches also sum the conv in different
orders. C1a is therefore a mixed-precision agreement check at 1e-4, recorded per layer, and the exact identity
that the measured state carries everything forward is C1b, which compares the same arithmetic path with itself
across a cache hand-off. The tolerance 1e-4 is fixed here; the tiny-config CPU test reports the magnitude first,
and a pilot failure is reported, not relaxed. The production float32 C1a discrepancy is reported as a hazard.

H1 to H7 are written before the run and not tuned. The pilot gate is the Pythia one (C1a, C1b, C2, P3, Sinkhorn,
duplicates, zero norm, min $\mathrm{PN}(\psi_1) \ge 10$ over point, state and stack cells), applied to the trained
checkpoint. On failure, stop and report.

## 8. Implementation

- New directory `explore/mamba_coherent/`. Import unchanged modules from `../lm_coherent/` (`lm_cells`,
  `lm_tree`) and `../cifar_coherent/` as the Pythia study does. Fork what has to change (`mb_construction.py`,
  `mb_modal.py`, `mb_schema.py`); the forks say at the top what they change. Generalise rather than copy where a
  small parameter suffices (views list, checkpoint labels instead of integer steps).
- Fixture: the Pythia fixture code path, with the state and stack views fed by synthetic constituents (two per
  layer), so P3's new Gram identity and every page element run on synthetic data first. Pages built from it
  carry MOCKUP.
- Tests in `test_mb.py`, including C1a/C1b on a tiny randomly initialised Mamba config on CPU.
- Modal: app `cm-mamba-coherent`, volume `cm-mamba-coherent`, run tag `m130`. Units: U0 (probe copy and tokenizer
  check, CPU), U1 per checkpoint (forward, states, Grams; L40S), U2 per checkpoint (cells and blocks; CPU or GPU
  as in the Pythia study), U3 (trees, layouts, finalize). Pilot = U1 and U2 on `trained`, then the gate.
- **Cost.** Mamba-130m is smaller than Pythia-160m per token, but U1 decodes 129 steps instead of one parallel pass
  and forms 2 x 24 constituent Grams per position for 6 positions (and, for the slab-free design, nothing at
  other positions). U2 handles 3 views instead of 2 but 2 checkpoints instead of 5. Projected **about $1.5**, wall
  clock about 30 minutes after the code is ready; ceiling **$6** (the Pythia study's ceiling), enforced by the
  same attempts ledger.
- `cross_arch.py` runs locally on the fetched public files of both studies.

## 9. Page

`explore/mamba_coherent/page/mamba_coherent.html`, built by adapting (copying) the Pythia page builder. Changes:

- F0 banner: 24 Mamba blocks, each with a state box (conv buffer + SSM state), 6 layers tagged, and the whole
  state $\Sigma_t$ drawn as the column of boxes.
- Explorer: `view: point | state | stack`; the bag row and 6 layers; the training strip becomes a two-panel
  init/trained strip.
- Measurements: Q3 becomes "residual vs layer state vs whole state" (F6': $a_3$ heatmaps for the three
  comparisons, layer x position, with the null marked; and $\tilde W$ by view along t); Q4 becomes "init vs
  trained" (two small multiples, one colour range); new Q6 "Mamba vs Pythia on the same snippets" (F12: $a_3$ and
  matched Jaccard, matched layer x position, for point-point and slab-state/stack, with nulls; and the T1 root
  cuts of both studies cross-tabulated).
- Verdict box against section 7. Every other figure line as in the Pythia spec, with layers renamed.
- Published with the `publish-results` skill.

## 10. Decisions taken on the owner's behalf

| id | decision | alternative |
|---|---|---|
| D-M1 | Equal weight per constituent (conv, SSM; per layer in the stack) | raw concatenation (scale set by the parameterisation) |
| D-M2 | `mamba-130m` (Mamba-1): same data, tokenizer and probe as Pythia; state 29k numbers per layer | `mamba2-130m` (state 197k per layer, larger heads); `mamba-370m` (48 layers) |
| D-M3 | No training axis; HF-init as the step-0 comparison | train our own small Mamba with checkpoints (a different study) |
| D-M4 | Drop the residual prefix slab for Mamba; the state and stack replace it | keep it for symmetry with Pythia (one more Gram per cell; its role, a proxy for what is carried forward, is filled by the state itself) |
| D-M5 | Layers every 4th, matching Pythia's relative depths | all 24 |
