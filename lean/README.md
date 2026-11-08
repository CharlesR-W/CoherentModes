# Finite-dimensional exact facts in Lean

This Lake library formalizes the requested algebraic portions of
[`THEORY.md`](../THEORY.md), together with contraction and the PSD stochastic
spectral bound. Its draft references are to A.3, A.5, A.6, and A.7 of
[`theorems/drafts/A1-exactness.md`](../theorems/drafts/A1-exactness.md).
**It does not claim to formalize those draft statements in their entirety.**

## Build

Install [elan](https://github.com/leanprover/elan) if necessary, then run from the
repository root:

```sh
export PATH="$HOME/.elan/bin:$PATH"
cd lean && lake exe cache get && lake build
```

The committed toolchain and dependency manifest pin:

- **Lean 4.35.0**, `leanprover/lean4:v4.35.0`.
- **Mathlib v4.35.0**, commit `56a005855ea17e0b0958f73ec801ae140b0a1679`.
- Transitive dependencies at the revisions in `lake-manifest.json`.

Use the prebuilt Mathlib cache; do not build Mathlib from source on this VPS.
The default build target imports all three proof modules through
`CoherentModes.lean`. A successful build has no errors or warnings. To use the
results in another Lean file, write `import CoherentModes`.

The setup was validated on 2026-11-07 with a trivial matrix `#check`,
`lake exe cache get`, and `lake build`. All 22 theorem declarations were also
checked with `#print axioms`: their only dependencies are `propext`,
`Classical.choice`, and `Quot.sound`; none depends on `sorryAx` or a custom axiom.
There are no proof placeholders or admitted statements in this library.

## Conventions and hypotheses

All declarations are in namespace `CoherentModes`.

- Operators are `Matrix (Fin n) (Fin n) ℝ`. Activation and gradient-factor
  matrices may be rectangular, with independent feature widths.
- `DoublyStochastic P` means entrywise nonnegativity, every row summing to one,
  and every column summing to one. It does **not** assume symmetry or PSD.
- `Orthogonal Ψ` means only `Ψᵀ * Ψ = 1`; the reverse identity is proved using
  finite-dimensional square-matrix invertibility, not assumed separately.
- `loading Ψ F i` is `Fᵀ *ᵥ ψᵢ`, where `ψᵢ` is column `i` of `Ψ`.
  `modeSum Ψ F S` is the finite sum of outer products `ψᵢ aᵢᵀ` over `S`.
- `vectorNormSq v = ∑ i, v i ^ 2` and
  `frobeniusSq F = ∑ x, ∑ j, F x j ^ 2` use counting measure.
  `frobeniusNorm F = Real.sqrt (frobeniusSq F)`. We deliberately do **not** use
  the default function-space or matrix norm, which need not be Euclidean or
  Frobenius. The spectral proof explicitly connects `vectorNormSq` to
  `EuclideanSpace.real_norm_sq_eq`.
- `P.PosSemidef` is Mathlib's condition and includes `P.IsHermitian`, which is
  symmetry over the reals. No separate diagonalization assumption is added.
- The identities allow zero dimensions. The fixed-vector statements alone
  therefore do not assert the existence of a nonzero unit constant vector.

## Formalized statements

Names below omit the common `CoherentModes.` prefix.

### `CoherentModes/Stochastic.lean`

| Theorem | Statement | Source |
| --- | --- | --- |
| `DoublyStochastic.mul` | A product of two doubly stochastic matrices is doubly stochastic. | THEORY 4.1; A.3, composition portion |
| `DoublyStochastic.transpose` | Transposition preserves double stochasticity. | A.3, supporting fact |
| `DoublyStochastic.fixed_one` | `P 1 = 1`. | THEORY 4.1; A.3, constant-vector portion |
| `DoublyStochastic.transpose_fixed_one` | `Pᵀ 1 = 1`. | THEORY 4.1; A.3, constant-vector portion |
| `two_stage_dissipation` | `1 − ‖P'Pv‖² = (1 − ‖Pv‖²) + (‖Pv‖² − ‖P'Pv‖²)`. | THEORY 4.1; A.5, equality only; no assumptions needed |

### `CoherentModes/Exactness.lean`

| Theorem | Statement | Source |
| --- | --- | --- |
| `orthogonal_complete` | `ΨᵀΨ = I` implies `ΨΨᵀ = I`. | A.6, completeness |
| `exact_reconstruction` | `F = ΨΨᵀF`. | THEORY 1.3; A.6 |
| `exact_mode_expansion` | `F` is the sum of all `ψᵢ aᵢᵀ`. | THEORY 1.3; A.6 |
| `exact_tail_identity` | Squared Frobenius residual after retaining any subset `S` equals the omitted loading energy. | THEORY 1.3; A.6 |
| `exact_tail_norm_identity` | The same identity written as `frobeniusNorm(residual) ^ 2`. | THEORY 1.3; A.6 |
| `exact_gradient_decomposition` | `ΔᵀU = ∑ i, (Δᵀψᵢ)(Uᵀψᵢ)ᵀ`. | THEORY 7.2; A.7, single-slot algebra |
| `block_projector_rotation` | `(BR)(BR)ᵀ = BBᵀ` for an orthogonal square `R`. | THEORY 7.2; A.7 |
| `block_pressure_rotation` | The Frobenius norm of `ΔᵀBBᵀU` is unchanged by `B ↦ BR`. | THEORY 7.2; A.7 |
| `coefficient_eq_loading` | Entry `(i,j)` of `ΨᵀF` is component `j` of loading `i`. | A.6, supporting fact |
| `frobeniusNorm_sq` | The defined Frobenius norm squared equals the entrywise sum of squares. | Norm bridge |
| `frobeniusSq_eq_gram` | Squared Frobenius norm equals the sum of diagonal entries of `FᵀF`. | Parseval proof helper |
| `frobeniusSq_orthogonal_mul` | Left multiplication by orthogonal `Ψ` preserves squared Frobenius norm. | Parseval proof helper |
| `modeSum_eq_mul_mask` | A retained outer-product sum equals multiplication by the row-masked coefficient matrix. | Tail proof helper |

The block-rotation identities are stronger than the projector case: they hold
for any rectangular `B`. When its columns are orthonormal, `BBᵀ` is the block
projector intended by the theory; idempotence of that projector is not a separate
formalized result. The gradient decomposition is an identity for arbitrary
supplied matrices `Δ` and `U`, not a formal verification of automatic
differentiation or of the identification with the gradient of a loss.

### `CoherentModes/Spectral.lean`

| Theorem | Statement | Source |
| --- | --- | --- |
| `DoublyStochastic.contracts` | `‖Pv‖² ≤ ‖v‖²` in the counting-measure Euclidean norm. | A.3/A.5 supporting contraction fact |
| `two_stage_dissipation_nonneg` | For unit input and doubly stochastic factors, both dissipation summands are nonnegative. | THEORY 4.1; A.5 |
| `DoublyStochastic.rayleigh_le` | `vᵀPv ≤ ‖v‖²`. | Supporting Rayleigh bound |
| `stochastic_psd_eigenvalues_mem_Icc` | Every eigenvalue of a PSD doubly stochastic real matrix belongs to `[0,1]`. | THEORY 3.1; optional requested result |

Contraction uses weighted Cauchy–Schwarz and both stochastic sum identities.
The spectral result uses Mathlib's `Matrix.IsHermitian` orthonormal eigenbasis,
its Rayleigh identity, and the PSD eigenvalue-nonnegativity theorem.

## Scope not formalized

All explicitly requested core identities and the optional eigenvalue bound are
finished. The following additional claims from the broader draft are **not**:

- A.3: an explicitly packaged arbitrary-length product, the normalized singular
  pair, singular-value/operator-norm statements, centered block decomposition,
  or simplicity from strict positivity.
- A.5: quadratic-form restatements, the additional PSD geometric bounds,
  balanced-set mass/retention identities, or singular-value min–max formulas.
- A.6: indicator-centroid re-embedding and the PCA/SVD loading specialization.
- A.7: the reverse-mode chain rule, shared-use summation, rank bounds, gradient
  residual bounds, or optimizer/frozen-input update statements. The proved
  `exact_reconstruction` applies to any supplied activation-difference matrix,
  but does not identify that difference with a particular update formula.
- Sinkhorn existence/uniqueness, kernel PSD, sampling, stability, and semantic
  interpretations are outside this finite algebraic formalization.

## Files and resources

`lakefile.toml`, `lean-toolchain`, and `lake-manifest.json` specify a reproducible
library build. The repository's `.gitignore` excludes `lean/.lake/`, including
Mathlib checkouts and generated artifacts.

On this VPS the setup occupied approximately **10.64 GiB** after validation:
2.98 GiB in `~/.elan`, 7.22 GiB in `lean/` (almost entirely `.lake/`), and
0.44 GiB in `~/.cache/mathlib`. Approximately **25 GB** remained free. Mathlib
oleans were downloaded rather than built from source; the small cache utility
and this project's proof modules were compiled locally.
