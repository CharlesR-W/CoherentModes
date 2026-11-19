# Diagnosis of the v1 Mamba study (D1-D4)

Inputs: `../data/m130/*`, and for the recomputation `act_trained.npy` and `stack_trained.npy` from the Modal volume
(fetched to `/var/tmp/cm-mb/vol`). All numbers are in `diag_v1.json`, produced by `diag_v1.py` (about 25 s on this VPS).
The token classes (`token_classes.py`) and the source labels are used only for scoring. No construction uses them.

## D1. Depth-block sets are token-context classes

- Depth blocks (point, L4..L24 at one t), mean over the six t, AMI with the SEBA sets (assigned items), r = 2/3/4/6/8:
  current token's class 0.35/0.41/0.46/0.47/0.47; source 0.14/0.12/0.13/0.14/0.15; next token's class 0.19-0.22;
  current token id 0.10-0.18; next token id 0.05-0.09.
- The class signal does not come through source. Within one source, AMI(sets, cur_class) is 0.32-0.48. Within one
  class, AMI(sets, source) is 0.08-0.15. By chance, AMI(source, cur_class) is only 0.07.
- Matched Jaccard, as (J - null) / (ceiling - null), mean over t. At r >= 3, cur_class scores 0.39-0.43 and source
  0.16-0.21. At r = 2 source scores slightly higher (0.35 against 0.29).
- Per set (depth blocks, r = 8, 48 sets): the median share of the majority current-token class is 0.60. 16 sets
  are >= 70% one class, and 11 sets are >= 70% one source.
  - Clean sets: function words (' the', ' of', ' a'; seven sets at 0.73-0.89, every source); line ends ('\n'; six
    sets at 0.62-0.87, often one source); sentence punctuation (two, 0.76-0.82); LaTeX (three, 0.65-0.69, maths).
  - The rest are broad word_start / word_cont buckets (0.47-0.74), or mixtures of class and source. Examples: the
    "code layout" sets ('\n', indents, '{'; 80% code), and the "symbols in code" set ('.', '-', '_'; 68% code).
- Reading blocks (all 18; positions averaged at one layer) are source sets: AMI with source is 0.46-0.86. The best
  AMI with cur_class at any single t is 0.13, and at r = 4 it is 0.07-0.10, the chance level set by source.
- **Supported, with a qualification.** Depth-block sets follow the current token's class far more than source or
  the next token (so they are not mainly organised by prediction). About a third are clean classes; the rest are
  class-by-source mixtures (function words and line ends span sources; LaTeX and code layout are one source) or
  coarse word buckets.

## D2. Upsilon = mu2/phi is a two-way statistic; balanced cuts mean no bottleneck

- 22 of 28 internal tree nodes are within 2% of balanced, and 20 are balanced to within one item. No cut sits at the
  minimum side size m_min. Random item subsets of T1's block (n = 500/250/125, 9 subsets, k = 15) are also cut at
  exactly n/2 every time. **Supported:** the sweep has no bottleneck to find and falls to the midpoint.
- Root Upsilon: T1 1.77, T2 1.94, T3 0.72, T4 1.99. At T4's root the cut is source-clean ({maths, code, forum} vs
  {wiki, poetry}; 99% of items on their source's side) and phi = 0.10 is low. Upsilon is still only 1.99, because
  mu2 = 0.198 is also small. So the mechanism D2 names does occur at T4's root.
- Four cuts of mixed-source nodes are >= 90% source-respecting: T1 n1 2.08, T4 root 1.99, T4 n1 2.51, T4 n2 5.24.
  The other mixed-source cuts score 0.72-2.16.
- **Recomputation of the k-way statistic contradicts the remedy.** Root averaged walks were rebuilt with the study's
  functions. They reproduce the stored mu and the stored SEBA labels exactly.
  - Upsilon^(k) = mu_k / rho^(k), with SEBA partitions made hard by argmax, for k = 2..8:
    T1 1.34, 1.10, 1.36, 1.20, 1.09, 1.10, 1.12; T4 1.58, 1.21, 1.45, 1.31, 1.01, 1.11, 1.00.
  - The source partition at k = 5 (label-derived, an illustration only) scores T1 1.16 and T4 1.18. That is no
    better, even at T4, where the five sources are almost perfectly separated.
  - Label-free reason: higher-order Cheeger gives rho(k) >= mu_{k-1}/2, so Upsilon(k) <= 2 mu_k / mu_{k-1} for any
    partition, the optimal one included. Root mu_1..8: T1 0.31 0.44 0.48 0.59 0.67 0.73 0.74 0.76; T4 0.11 0.20 0.24
    0.38 0.48 0.55 0.61 0.64. Ceilings: T1 2.03-2.85; T4 2.08-3.58 (> 3 only at k = 2, 4). No large gap at any k.
- **Verdict.** It is true that mu2 is small when there are three or more groups. But choosing a different k does not
  rescue Upsilon, because the spectrum has no large gap at any k. Low Upsilon is a property of these operators
  (source groups that are separated but leaky), not an artefact of fixing k = 2.

## D3. Source labels are confounded

- Mixing ratio X (1 = the walk ignores the pair, < 1 = it avoids crossing; null 97.5% band tops out near 1.08):
  - Whole-state cell (stack, L24, t = 128): code-forum **1.36**, maths-forum 0.37, wiki-poetry 0.27, maths-wiki 0.16,
    every other pair <= 0.09.
  - Point cell (L16, t = 128): code-forum **1.33**, maths-forum 0.86, maths-wiki 0.62, maths-code 0.55.
- Code-forum is above 1: the walk crosses between code and forum more often than chance, so the two act as one
  group. Forum has the lowest self-mixing (diagonal) of the five sources in every cell checked. The reading blocks
  agree (stack L24: code-forum 1.08; point L16: code-forum 1.06, maths-forum 0.83).
- The cuts agree: T1's root splits maths 42/158 and forum 63/137; T4's node 1 puts 182 forum items with all of code.
- **Supported.** Forum overlaps code strongly and maths moderately. Wiki and poetry also mix at L24 (point 0.97).

## D4. Zoom resolution: k/n grows

- Trees by depth (n = 1000, ~500, ~250): k/n 0.015/0.031/0.062; mean mu1 0.25/0.31/0.47; mean Upsilon
  1.61/2.25/1.88; corr(mu1, k/n) = 0.42 over nodes. A random regular graph with 15-30 neighbours has mu1 about
  0.50-0.64; the depth-2 nodes of T1 and T2 reach 0.50-0.74.
- Random subsets of T1's block, with k = 15: mu1 rises from 0.31 (n = 1000) to 0.34-0.37 (500), 0.40-0.43 (250) and
  0.41-0.48 (125). T1's own depth-2 nodes (n ~250, mostly one or two sources) have mu1 = 0.50-0.58. So k/n
  accounts for roughly half of T1's rise from root to depth 2; the nodes' greater homogeneity for the rest.
- Scaling k to 15n/1000 brings mu1 back to about 0.25-0.33 on random subsets, and to 0.40-0.44 at T1's depth-2
  nodes. It does not raise Upsilon: 1.27-1.84 on random subsets, 1.67-2.00 at the depth-2 nodes, against 1.95-2.16
  with k = 15.
- **Partly supported.** k/n inflates mu1 as claimed. It does not explain the absence of clear splits.

## Other observations

- The brief says no zoom split reached Upsilon >= 3. One did: T4 node 2 (wiki 193 | poetry 198, Upsilon 5.24;
  the v1 page says "1 of 28"). Nine nodes have ceiling 2 mu2/mu1 >= 3; only that one's cut reaches 3.
- The reading-block SEBA sets at r = 2 leave many items unassigned (T1 349, T4 237).
- T3 (a single cell, L16, t = 128) has the lowest root Upsilon (0.72) and mu1 = 0.08, which is near-degenerate. Its
  first cut is unbalanced (580/420) and partly separates poetry (172/28).
