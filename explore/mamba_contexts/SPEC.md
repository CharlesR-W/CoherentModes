# Coherent sets over token contexts in Mamba (toy, exploratory): spec

Draft v2, 2026-11-12 (after the first review; section 11). A follow-up to `../mamba_coherent/` ("v1"), written after diagnosing why v1 was
uninformative (`../mamba_coherent/diagnosis/DIAG_NOTES.md`). It is a **delta spec** against
`../mamba_coherent/SPEC.md` and, through it, `../lm_coherent/SPEC.md` v2.1: the model, checkpoints, recurrent
decoding, the state $(c^m_t, z^m_t)$, the standard cell pipeline, the kernel (union kNN, k = 15, floor $10^{-6}$),
Sinkhorn, the lazy walk, block averaging, SEBA, span agreement $a_r$, degenerate-item policies and the page style
are inherited unchanged. Exploratory side study in `explore/`; no claim enters `THEORY.md`; the working language
is `THEORY.md`'s.

## 1. Why v1 was uninformative, and the question

v1's items were 1,000 snippets from five sources, and its headline blocks averaged walks along reading
(positions 4..128 of the same snippet). The three points below are the working diagnosis; the measurements behind
them are in `DIAG_NOTES.md`, and they are an account of v1, not something this study assumes.

- **What persists along a snippet is a document-level attribute.** Averaging over positions keeps the links that
  survive from token to token, which on that probe is the source's vocabulary. The bag-of-tokens reference finds
  the same partition. The probe had no other grouping, and every statistic ($W$, $X$, matched Jaccard, tree
  make-up) was scored against source.
- **The structure that was not along reading was not scored.** v1's depth blocks (the same position at six layers)
  give sets that read as token-context classes (function words, line ends, LaTeX, mid-word fragments), but they
  were scored against source and reported as failing to reproduce it.
- **Readouts that need clusters had little to read.** v1 read $\Upsilon = \mu_2/\phi^* \ge 3$ as a clear split and
  found one in 28. This is not an artefact of using two parts: at v1's roots the multi-way ratio $\hat\Upsilon(k)$
  of section 5 is 1.0 to 1.6 for k = 2..8, the label-derived source partition scores 1.2, and the spectral cap
  $2\mu_k/\mu_{k-1}$ is 2.0 to 3.6 over those k. 22 of the 28 zoom cuts are within 2% of balanced, as are the cuts
  of random subsets. The inspected root spectra do not support a strong gap-separated clustering interpretation.
  This motivates reporting mode- and walk-level associations (shares, retention, span agreement, $\eta^2$)
  whether or not clustering is certified.

**Question.** With items at the granularity the model computes on, a **token context** (a snippet prefix ending at
a sampled token), and N large enough to resolve classes of contexts:

- (a) What are the coherent sets of the residual stream along **depth**? Are they classes of contexts that
  predict alike, rather than sources or token identities?
- (b) What are the coherent sets of the **recurrent state** (per layer, and the whole state)? Do they follow
  the document, the recent tokens, or the same classes as the residual?
- (c) Along **reading** (the state after 1, 2, ..., 16 more tokens), how long does each kind of set stay together?
- (d) Is there clear multi-way structure by a statistic that is not tied to two clusters?

## 2. Items

- **Snippets.** The v1 probe verbatim (`../lm_coherent/data/p160t/probe.json`, 1,000 snippets, 128 tokens, U0 of v1
  applies).
- **Items.** N = 8,000: for each snippet, 8 distinct positions $t \in \{16, \dots, 112\}$, drawn uniformly without
  replacement with `RandomState(20261113 + snippet_id)`. Item (i, t) is the prefix `[EOT] + tokens 1..t` of
  snippet i. Items are stored in a seeded random order, so index encodes neither snippet nor position.
- **De-duplication (label-free).** No two items may share their last 16 tokens. On a clash the later-drawn
  position is redrawn (same generator); the count of redraws is recorded.
- **Lags.** $\mathcal L = \{0, 1, 2, 4, 8, 16\}$. Item (i, t) at lag $\tau$ is the same construction at
  $(i, t+\tau)$; $t + \tau \le 128$ by the position range.
- Positions are sampled uniformly, so the items carry the corpus's own token frequencies (many newlines, commas,
  `the`). That is the measured regime, not corrected for.

## 3. Views and cells

All at (checkpoint, lag 0) unless a lag is stated. Constituent weights are v1's decision D-M1 (Gram average).

| view | $\phi$ of item (i, t) | cells |
|---|---|---|
| **point** | $h^l_t(i)$, raw residual after block l | l = 1..24 |
| **state** | layer l's own state: Gram average of $c^l_t$ and $z^l_t$ | l = 1..24 |
| **whole** | the whole model state $\Sigma_t$: Gram average of the 48 constituents | 1 |
| *bag* (reference) | mean input embedding of tokens 1..t | 1 |
| *recent* (reference) | mean input embedding of tokens t-15..t | 1 |
| *pred* (reference) | $\sqrt{p_t(i)}$, the trained model's next-token distribution at t (Hellinger embedding), through the standard pipeline | 1 |

Lag cells (trained only): point, state at L8, L16, L24 and whole, at each $\tau \in \mathcal L$.

Reference cells never enter a block. They are exempt from the duplicate assertion (amended after pilot 1, section
11): two contexts with the same prediction, or the same token counts, are the same point of a reference by
definition. Their smallest distance and the number of pairs below $10^{-9}$ are reported as a hazard; kNN ties
break by index as elsewhere. **Tie sensitivity.** With more than k+1 identical points the index tie-break chooses
edges arbitrarily. So any reference cell with a pair below $10^{-9}$ is rebuilt under three seeded reorderings of
the items (seeds 1, 2, 3; the reordering is undone afterwards), and $a_8$, $a_{16}$ and the AMI of the r = 8 and
16 sets between each rebuild and the original are stored. If any $a_r$ is below 0.9 or any AMI below 0.8, the
spectral comparisons against that reference are marked tie-sensitive on the page and not read as evidence. The
walk-level statistics $\kappa$ and $A$ do not pass through a reference graph and are unaffected. `pred` is the trained network's own output; it is a reference for "contexts
that predict alike", in the way `bag` and `recent` are references for "contexts with the same token counts"
over the whole prefix and over the last 16 tokens.

## 4. Blocks

| block | cells | role |
|---|---|---|
| **B-point** (headline) | point, L1..L24 | residual along depth |
| B-point-early / -mid / -late | point, L1..8 / L9..16 / L17..24 | where along depth |
| **B-state** | state, L1..L24 | per-layer state along depth |
| **B-lag-whole** | whole at $\tau \in \mathcal L$ | whole state along reading |
| B-lag-point | point L16 at $\tau \in \mathcal L$ | residual along reading (v1's T1 at these items) |

Checkpoints: `trained` for everything; `init` for B-point, the whole cell and the point and state single cells
(lag 0 only).

## 5. Statistics

**Walk shares** (v1's definition, exact chance level, excess $\tilde S$, 50-permutation band), for every cell and
block, with these groupings of the items:

| grouping | definition | kind |
|---|---|---|
| $T$ | token id at t (the token just read) | label-free (input) |
| $N\!x$ | token id at t+1 (the true next token) | label-free (the training target) |
| $U$ | snippet id | label-free (input) |
| $A$ | the trained model's top-1 next-token prediction at t | the network's own output |
| $W$ | source | validation |
| $C_c$, $C_n$ | class of the token at t, at t+1 (`../mamba_coherent/diagnosis/token_classes.py`, 12 classes) | validation |

**Predictive coherence.** $\kappa = \frac1N\sum_{i\ne j} P_{ij}\,\mathrm{BC}(p_i, p_j)$ with BC the Bhattacharyya
coefficient of the trained model's next-token distributions; chance $\kappa^0$ is the mean BC over pairs; excess
$\tilde\kappa = (\kappa-\kappa^0)/(1-\kappa^0)$.

**Predictive coherence beyond the token just read.** The token at t alone predicts much of $p_t$, and the token
classes are functions of it, so $\tilde\kappa$, $\tilde C_c$ and agreement with `pred` can all be high for a
walk that only groups by current token. The conditional null: permute the prediction vectors **within groups of
items that share the token at t** (items whose token is unique in the probe keep their own; their share is
reported), 50 permutations, giving $\kappa^{T}$ (mean) and a band. $\tilde\kappa_{|T} =
(\kappa-\kappa^{T})/(1-\kappa^{T})$ is the predictive association the walk has beyond what token identity
gives. Pairs from the same snippet are excluded from $\kappa$, $\kappa^0$ and $\kappa^T$ throughout, so that
shared documents do not count as shared predictions. For sets: the share of each set's items that carry its
commonest token, and the number of distinct tokens covering 90% of it.

In lag cells the groupings $T$, $C_c$, $U$, $W$ are those of lag 0 (how long the state keeps together the
contexts that shared a token, a class, a snippet), plus $T_\tau$, the token at $t+\tau$.

**Spectra and sets.** For every block, and for the single cells point and state at L4, 8, .., 24, whole, bag, pred
and all lag cells: the top 33 eigenpairs; SEBA at $r \in R = \{2,3,4,6,8,12,16,24,32\}$; PN; set sizes.

**Multi-way clustering ratio.** With $\mu_j = 1 - \lambda_j(\bar P)$ the j-th smallest non-trivial value:

- gap ratios $\mu_k/\mu_{k-1}$, k = 2..32 (label-free, spectral only);
- $\hat\Upsilon(k) = \mu_k / \hat\rho(k)$ for $k \in R$, where $\hat\rho(k) = \max_j \phi(A_j)$ over the parts of a
  complete k-partition and $\phi(A) = \sum_{i\in A, j\notin A}\bar P_{ij}/\min(|A|, N-|A|)$.
- **Partitions (two label-free roundings; the larger $\hat\Upsilon$ is kept and both are stored).**
  - *SEBA*: every item goes to the SEBA vector (r = k) with the largest value, ties and all-zero rows to the
    lowest index. This is not `seba_labels`, which leaves items unassigned.
  - *k-means* (Peng, Sun and Zanetti's own rounding): k-means with k centres on the first k-1 non-trivial
    eigenvectors (the stationary measure is uniform, so no degree scaling), `n_init = 10`, seed 0.
  - A rounding with an empty part is **invalid** for that k and gives no value.
- **[thm]** Any complete k-partition has $\hat\rho(k) \ge \rho(k)$, so $\hat\Upsilon(k) \le \Upsilon(k)$ of Peng,
  Sun and Zanetti (2015); k = 2 with the sweep cut is v1's $\Upsilon$. By the higher-order Cheeger inequality
  $\rho(k) \ge \mu_{k-1}/2$, so $\Upsilon(k) \le 2\mu_k/\mu_{k-1}$ for every partition.
- **Reading rule, three outcomes per k.** *Certified*: $\hat\Upsilon(k) \ge \theta$. *Excluded*:
  $2\mu_k/\mu_{k-1} < \theta$, so no partition reaches $\theta$. *Undetermined* otherwise: the roundings did not
  certify it and the spectrum does not rule it out; this is not evidence of absence, and the page says so.
- $\theta$ is an empirical reading convention, not a recovery guarantee. It is fixed on the fixture (section 8) by
  v1's rule (3.0 if planted and structureless separate there; else their geometric mean; if they overlap the rule
  is dropped and the page says so) and is not revised on real data.

**Modes against groupings** (needs no clusters). For each of the leading 16 non-constant eigenfunctions $\psi_k$ of every block
and listed cell, and each grouping, the correlation ratio $\eta^2_k$ (between-group share of $\psi_k$'s variance),
with its exact chance level $(G-1)/(N-1)$ for G non-empty groups and a 50-permutation band (a null reference,
not a confidence interval). Reported per mode and as the
$\lambda$-weighted mean over the 16. For the page: the 12 items at each end of each $\psi_k$.

**Sets against groupings.** For the hard SEBA sets at r = 8 and 16 of each block and listed cell: adjusted mutual
information (assigned items only) with $T$, $N\!x$, $U$, $A$, $W$, $C_c$, $C_n$; matched Jaccard with null and
ceiling against the few-group ones ($W$, $C_c$, $C_n$). Per set, for the page: size, the 8 commonest tokens at t and
at t+1, class and source make-up, number of distinct snippets, 12 example contexts (the 24 tokens before t and
the 4 after).

**Comparisons.** $a_8$, $a_{16}$ and the AMI of the r = 8 and r = 16 hard sets between: B-point, B-state, whole,
bag, pred (all pairs); the three depth thirds (pairs); lag 0 against each lag, per lag view; trained against init
for B-point and whole.

**Lag retention.** For each SEBA r = 8 soft vector f (centred) of B-point: $f^\top P_c f / f^\top f$ in the point
lag cells (L8, L16, L24) at each $\tau$. The same for B-state's r = 8 vectors in the state lag cells (L8, L16,
L24), and for the whole cell's in the whole lag cells.
Chance is $-1/(N-1)$. Reported per vector and as the mean.

**Sensitivity (reported, not selected on).** B-point rebuilt with k = 8 and k = 30: $a_8$, $a_{16}$ against
k = 15, and $\hat\Upsilon(k)$.

## 6. Zoom trees

`lm_tree` as in v1 (sweep cut on $\psi_1..\psi_3$, depth below 3, $n \ge 64$, rebuild on the subset by the view's
own construction), with per-node additions: gap ratios, the 8 commonest tokens at t and t+1, class and source
make-up, distinct snippets. v1's $\Upsilon$ is stored and drawn as before, with the note from section 1 that it
is a two-way statistic.

| tree | block |
|---|---|
| T1 | B-point, trained |
| T2 | whole cell, trained |

## 7. Controls and targets

Written before the run; not tuned afterwards.

| id | target | label |
|---|---|---|
| C1a, C1b | v1's, rerun in the pilot on the trained checkpoint | identity |
| C3 | Every lag-0 vector read from the decode at (i, t) equals v1's stored value where v1 stored one (point, L4..L24, t = 16, 32, 64 where sampled), to relative $10^{-6}$ | identity |
| P1' | Fixture, check seeds (section 8): planted k-cluster blocks at separation $s \ge 1$ are certified at their k; structureless blocks are certified at no $k \in R$ | planted |
| P3 | Gram identity for state and whole, on a 500-item subsample, to $10^{-12}$ | [thm] |
| K1 | B-point, trained, r = 8 sets: AMI with $C_c$ exceeds AMI with $W$ | hypothesis |
| K1b | B-point, trained: $\kappa$ lies above the upper end of the within-token permutation band ($\tilde\kappa_{|T} > 0$ beyond the band). Its size is reported; K1 without K1b reads as "grouped by current token" | hypothesis |
| K2 | Whole cell, trained, r = 8 sets: AMI with $W$ exceeds AMI with $C_c$; and $\tilde U(\text{whole}) \ge \tilde U(\text{B-point}) + 0.2$ | hypothesis |
| K3 | Point cells, trained: $\tilde T$ is higher at L4 than at L24, and $\tilde\kappa$ is higher at L24 than at L4 by at least 0.1 | hypothesis |
| K4 | B-point's r = 8 set vectors lose at least half of their lag-0 retention by $\tau = 2$ in the point L16 lag cells; the whole cell's keep at least half at $\tau = 16$ in the whole lag cells | hypothesis |
| K5 | Some $k \in R$ is certified in B-point, trained | hypothesis |
| K6 | At init: $\tilde\kappa$ at point L24 and $\tilde W$ of the whole cell are each below half their trained values | hypothesis |

**Pilot gate** (trained): C1a, C1b, C3, P3; Sinkhorn row error below $2\times10^{-12}$ in every cell; smallest
off-diagonal cosine distance above $10^{-9}$ in every point, state and whole cell, lag cells included (constituents
may contain exact duplicates: $c^1$ is a function of the last three tokens; the three reference cells are exempt,
section 3); no zero-norm item in any view or constituent;
min PN$(\psi_1) \ge 10$ over blocks. On failure, stop and report; nothing is patched.

## 8. Implementation

- New directory `explore/mamba_contexts/`. Reuse by import what is unchanged (the HF decode step and cache
  handling in `mb_model`, the cell pipeline in `lm_cells`, `coherent.symmetric_balance`, `seba`, the sweep in
  `lm_tree`); the code that holds many N x N arrays at once (`mb_cells.StateStack`, `position_grams`,
  `block_modes` on a list, `lm_tree.node_operators`) is replaced by streaming versions here, tested against the
  originals at small N.

**Bounded-memory plan at N = 8,000** (one float64 N x N array is 512 MB).

1. **Snapshots.** Each snippet is decoded once per pass (1,000 sequences, 129 steps, batched). One pass per lag:
   at step u the pass copies the residuals and states of the items with $t + \tau = u$ into preallocated
   float32 buffers on the GPU (point 24 x 768, c 24 x 4,608, z 24 x 24,576 per item: 22.6 GB for all items).
   Predictions $p_t$ are kept as $\sqrt p$ in float32 (1.6 GB) at lag 0.
2. **Grams, one at a time.** After a pass, each constituent goes through the standard pipeline to a float64
   cosine Gram on the GPU, which is added into (i) its layer's state Gram and (ii) the whole-state running mean,
   then freed. Kept per lag: whole (1 array) and state at the lag layers (3); at lag 0: state at all 24 layers,
   written to local disk as they complete. Peak GPU: the 22.6 GB buffers plus about 3 GB.
3. **Raw vectors for the trees** (lag 0 only): point (0.6 GB) and the 48 constituents (22.6 GB) as float32
   `.npy` on the volume. A tree node loads the rows of its subset, one constituent at a time.
4. **Operators.** One cell at a time on the GPU: kNN, Sinkhorn, statistics; P is written to local disk
   (float64) for the retention pass; blocks accumulate $\bar M$ as a running mean and never stack cells. Eigenpairs
   by dense `eigh` on the GPU. The tree computes each node's $\bar P^S$ as a running mean too.
5. **Hardware.** U1, U2, U3 on one L40S (48 GB) with 64 GB host RAM and 100 GB local disk; an A100-80GB is the
   fallback if the benchmark shows a peak above 40 GB.
6. **Benchmark before the pilot** (`bench`, synthetic data at full N, on the same hardware): one constituent
   Gram at z's width; one cell (kNN, Sinkhorn, shares, $\kappa$ with its 50 conditional permutations); one
   `eigh` with SEBA at all $r \in R$ and both roundings; one tree-node rebuild at n = 8,000 and n = 2,000 for a
   24-cell block. The reservation and timeouts are set from these, and the run goes ahead only if the projected
   total is under the ceiling. If it is not, the first economies, in order: trees at depth 2; SEBA ranks capped at
   16 on single cells; `init` reduced to B-point and whole. Any economy taken is recorded.

**Fixture and calibration** (MOCKUP; through the real code path).

- Generator: N items, a block of 24 cells, dimension 64. Item i with planted label g has, in cell c, the vector
  $s\,e^c_g + \varepsilon$, with $e^c_1..e^c_k$ orthonormal and drawn independently per cell (the partition is
  shared across cells; the geometry is not) and $\varepsilon \sim \mathcal N(0, I/64)$. Structureless: s = 0.
- Cases: $k \in \{2, 4, 8, 16\}$; sizes balanced, and Zipf (proportional to 1/j); $s \in \{0.5, 0.75, 1, 2\}$.
  N = 1,200. Development seeds 0..2 (for getting the code right); **check seeds 100..104** decide $\theta$ and
  P1', using $s \in \{1, 2\}$ as "planted" and taking the structureless maximum over all $k \in R$. The two
  lower strengths are reported as a detection curve.
- One full-size check in `bench`: N = 8,000, 24 cells, k = 8 Zipf at s = 1, and structureless, seed 100. If these
  two fall on the wrong sides of $\theta$, the threshold is reported as not calibrated at this N and K5 is read
  from the gap ratios and the excluded/undetermined rule only.
- The page fixture: the k = 8 Zipf case with planted groupings for every statistic (tokens nested in classes,
  snippets, lags that decay), and its structureless twin.
- Tests (`test_mc.py`) include: $\hat\Upsilon(2)$ with the sweep partition equals `lm_tree`'s $\Upsilon$; the
  partition contract (ties, all-zero rows, empty parts); streaming Grams and the running-mean block against the
  v1 functions at small N; the snapshot and lag indexing against a direct decode on a tiny random Mamba; item
  de-duplication; the conditional permutation null on a planted case where predictions depend on the token only
  (so $\tilde\kappa_{|T}$ is within its band) and one where they depend on a planted context class.
- Modal: app and volume `cm-mamba-contexts`, run tag `m130c`. Units: `bench`; U1 per checkpoint (decode passes,
  Grams, raw vectors, predictions); U2 per checkpoint (cells, blocks, statistics); U3 (trees, layouts,
  finalize). Pilot = U1 and the gate on `trained`.
- **Cost.** Projected about $3 to $4 (to be replaced by the benchmark's figure), wall clock about 1.5 to 2 hours;
  ceiling **$8**, enforced by the attempts ledger as in v1.

## 9. Page

`explore/mamba_contexts/page/`, built with the `experiment-notebooks` skill; MOCKUP from the fixture first.

| figure | shows | the reader should be able to say |
|---|---|---|
| F1 | v1's depth-block and reading-block sets scored against source and against token class (from the diagnosis) | "v1 scored token-level sets against document-level labels" |
| F2 | excess shares $\tilde T, \tilde{N\!x}, \tilde A, \tilde\kappa, \tilde\kappa_{|T}, \tilde U, \tilde W, \tilde C_c, \tilde C_n$ by layer, point and state side by side, init faint | "up the stack the residual's neighbourhoods go from [..] to [..]; each layer's state is tied to [..]" |
| F3 | spectra, gap ratios, the cap $2\mu_k/\mu_{k-1}$ and $\hat\Upsilon(k)$ for B-point, B-state, whole, the references and the fixture, with $\theta$ | "k-way structure is certified at k = .. / excluded at k = .. / undetermined" |
| F4 | AMI of the r = 8 and 16 sets with each grouping, by block | "the residual's sets are [..]; the state's sets are [..]" |
| F5 | lag curves: retention and shares against $\tau$ for point, state and whole | "the residual's sets last [..] tokens; the state's last [..]" |
| F6 | the two trees with node descriptors | "the first cuts separate [..]" |
| F7 | explorer: layout per block coloured by set, class, source, snippet; set cards with tokens and example contexts | (reading the sets directly) |
| F8 | trained against init: F2's and F4's quantities | "training adds [..]" |

Verdict box against section 7. Published with `publish-results`.

## 10. Decisions taken on the owner's behalf

| id | decision | alternative |
|---|---|---|
| D-C1 | Items are token contexts, 8 per v1 snippet (N = 8,000) | new snippets, one position each (no same-snippet structure; a new probe build) |
| D-C2 | Headline block along depth, all 24 layers | every 4th layer as v1; reading blocks as headline |
| D-C3 | k = 15 unchanged at N = 8,000; k = 8 and 30 as reported sensitivity | scale k with N |
| D-C4 | $\hat\Upsilon(k)$ from SEBA partitions as the multi-way statistic; v1's $\Upsilon$ kept in the trees | k-means on the eigenvectors (PSZ's own algorithm); eigengap alone |
| D-C5 | Token classes by a fixed regular-expression taxonomy, validation only | a POS tagger (does not cover code or LaTeX) |
| D-C6 | Model outputs ($A$, $\kappa$, the `pred` reference) used for validation and as a reference cell, never in a block | no use of outputs |
| D-C7 | Reuse the v1 snippets, so sources stay as a contrast grouping | drop sources |

## 11. Changes

**From draft v1 (review by GPT-6.1 Sol, plan reviewer, 2026-11-12).**

- The validation could not tell "contexts that predict alike" from "contexts with the same current token". Added
  the within-token permutation null and $\tilde\kappa_{|T}$, same-snippet pairs excluded from $\kappa$, the
  `recent` reference, per-set token concentration, and target K1b.
- $\hat\Upsilon(k)$ is a one-sided certificate. Added the partition contract, a second rounding (k-means), the
  three-outcome reading rule with the spectral cap as the only route to "excluded", and the statement that
  $\theta$ is a convention.
- Feasibility at N = 8,000 was not established. Added the bounded-memory plan, hardware, and a full-size
  benchmark that sets the reservation before the pilot.
- Lag measurements did not cover the per-layer state. Added state lag cells at L8, L16, L24 and B-state's
  retention.
- The fixture was not reproducible. Fixed the generator, cases, development and check seeds, and a full-size
  calibration check.

**After the diagnosis finished (2026-11-12).** `DIAG_NOTES.md` supports the first two points of section 1 and
contradicts the draft's third (that v1's low $\Upsilon$ came from using two parts): the multi-way ratio is as low,
and the spectral cap shows why. Section 1's third point is rewritten accordingly, K5 stands as written (it may
well fail), and "modes against groupings" ($\eta^2$) is added to section 5 so that the readouts do not depend on
there being clusters. The reviewer's re-check tightened section 1's wording to what the
diagnosis measures.

**After pilot 1 (2026-11-13).** U1 passed every control. U2 stopped at the duplicate assertion on the `pred`
reference: two code contexts on which the trained model puts probability 1.000000 on the same next token have
the same $\sqrt p$ vector to float32 precision (distance $2.3\times10^{-10}$; 167 of the 8,000 items have top-1
probability above 0.9999). Nothing was patched in that run. The amendment: the three reference cells (bag, recent,
pred) are exempt from the duplicate assertion, as constituents already were; the assertion still applies to every
point, state and whole cell, including lag cells. This changes no view, block, statistic or target. It was made
after seeing the failure and before any measurement of the run was read (U2 wrote no outputs). The reviewer accepted it as a documented post-failure
amendment, asked that section 7 say the same, and added the tie-sensitivity check of section 3. The failed
attempt stays in the ledger and the runs table, and the construction fingerprint changes.
