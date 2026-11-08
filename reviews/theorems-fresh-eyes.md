# Fresh-eyes readability review

Scope: Section 0 and the six cluster introductions of `THEOREMS.md`, selected labelled statements for the table check, and all of `SPARSITY.md`. I did not read `THEORY.md` or verify proofs.

## Three hardest passages

1. Section 0.1: “every measurable property of a site ... defines a kernel on sites, hence a symmetric doubly stochastic operator, hence a spectrum and an SVD.” The chain skips the choices of view, kernel and balancing. “Every measurable property” also reads as a general mathematical implication rather than a recipe applied to a chosen representation.
2. Section 0.3's sampling row packs four different estimands into one sentence. In particular, a reader cannot tell that the leading $e^{3\beta}$ rate needs a normalization condition rather than being the unconditional worst case. See the substantive table check below.
3. The Part B, cluster 1 “central positive result” combines conditional feature means, varying belief directions, zero-ridge canonical variates and reachable beliefs before the setup defines them. Explain first that it supplies coordinates for a hidden-state probability vector.

## Definitions, references and presentation

- Section 0.2 says coined terms are defined before use, but “belief-sufficient layers” occurs in 0.1 before B.1, and “sparse concept cluster” and “ideal dictionary regime” occur in 0.2 before their later definitions. Say where definitions can be found instead.
- In 0.5, $C$ is used in $CPC$ before being defined in the excerpt; specify $C=I-\mathbf1\mathbf1^\top/N$. CKA and SEBA first appear in 0.3 without expansions; SEBA is defined in `SPARSITY.md` S.5. Temperature $\beta$ and the requirement of nonzero view rows also deserve an early gloss. The symbol $\mathcal R$ in the dictionary row is undefined there; B.20 defines it.
- I checked every individual label covered by the table's ranges: all A and B labels exist in `THEOREMS.md`, and S.2–S.8 exist in `SPARSITY.md`, not `THEOREMS.md`. Make the external destination explicit in the table.
- B.14 precedes B.13, acknowledged in the cluster introduction, but this makes navigation less predictable. The heading “B.3.1 What comparison identifies” can be mistaken for theorem B.3. The first cluster heading uses a colon while the other five use em dashes.
- I found no odd-dollar lines or `\{`/`\}` in Section 0 or `SPARSITY.md`, and the headline table has consistent columns. No browser-rendering test was made.

## Headline table versus statements

- **Limits:** A.9–A.11 support the expansion and duplicate-class limit. A.10 requires nonzero centered Grams for CKA and a spectral gap for PCA; the headline should distinguish the operator expansion from those conditional interpretations.
- **Stability:** A.21 and A.23 support the advertised perturbation and normalization bounds, subject to their gap/small-error and kernel conditions. “Blocks are” identifiable without a gap is too broad: a block needs separation from its complement.
- **Sampling:** A.44 explicitly gives $O(S_{\max}^2e^{3\beta}h/\delta)$ and says the unconditional worst case is $O(e^{6\beta}h/\delta)$, not $O(e^{3\beta}h/\delta)$ without qualification. A.55 supports $N^{-1/4}$ and root-$N$ for finite support, under its stated conditions.
- **Cost:** A.51–A.53 support dense formation, the stated contraction factor, and certified Nyström/random-feature approximations,.
- **Dictionaries and superposition:** B.21–B.23 support exact indicator recovery under the named gap condition. B.28 additionally requires distinct unoriented lines for arbitrary signed codes, or nonnegative codes for distinct oriented rays. B.29 requires orthonormal directions, positive coefficients, and sufficiently controlled coefficient imbalance; “separated features” alone overstates it.
- **Sparsity:** S.2 supports the edge bound and S.3 gives qualified multiway bounds. S.7 provides a nearby-start basin and an exact counterexample plus a numerical default-parameter failure, not a theorem that SEBA “fails generically.”

## Epistemic impression

Section 0.2 largely reads as honest: it names idealizations, gives counterexamples a role, and mentions the exponential price. “Mandatory and honest,” however, asks for trust rather than earning it, while the claim that terms are defined before use is false and the headline sampling rate omits a condition stressed in the theorem. Fixing these would make its confidence feel earned.

## Suggested edits

1. Replace “Coined terms (\"sparse concept cluster\", \"ideal dictionary regime\", \"belief-sufficient layer\") are defined before use.” with “Coined terms (\"sparse concept cluster\", \"ideal dictionary regime\", \"belief-sufficient layer\") are defined in the relevant Part B clusters.”
2. Replace “centered $\tilde P = CPC$” with “centering matrix $C=I-\mathbf1\mathbf1^\top/N$; centered $\tilde P = CPC$”.
3. Replace “at rate $O(e^{3\beta}\sqrt{\log(1/\eta)/N}/\delta)$” with “at rate $O(S_{\max}^2e^{3\beta}\sqrt{\log(1/\eta)/N}/\delta)$; the unconditional worst-case bound has $e^{6\beta}$”.
4. Replace “without a gap, vectors are not identifiable and blocks are” with “individual vectors within a degenerate block are not identifiable; a block separated from its complement is stable”.
5. Replace “Sparse sums of separated features form angular clusters by active set” with “Under orthonormal features and the stated coefficient-balance and separation conditions, sparse sums form angular clusters by active set”.
6. Replace “fails generically at the published default” with “can fail at the published default, including in the reported numerical example”.
7. Replace “S.2–S.8” in the headline table with “`SPARSITY.md`, S.2–S.8”.
