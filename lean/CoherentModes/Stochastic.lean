module

public import Mathlib.Data.Matrix.Mul
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic

/-! Finite counting-measure stochastic identities (A.3, exact algebra in A.5). -/
@[expose] public section

namespace CoherentModes
open Matrix
open scoped BigOperators

/-- Nonnegative entries and unit row and column sums; no symmetry is assumed. -/
structure DoublyStochastic {n : ℕ} (P : Matrix (Fin n) (Fin n) ℝ) : Prop where
  nonneg : ∀ i j, 0 ≤ P i j
  row_sum : ∀ i, ∑ j, P i j = 1
  col_sum : ∀ j, ∑ i, P i j = 1

namespace DoublyStochastic
variable {n : ℕ} {P Q : Matrix (Fin n) (Fin n) ℝ}

theorem mul (hP : DoublyStochastic P) (hQ : DoublyStochastic Q) :
    DoublyStochastic (P * Q) := by
  constructor
  · intro i j
    simp only [Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (hP.nonneg i k) (hQ.nonneg k j)
  · intro i
    simp only [Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, hQ.row_sum, mul_one]
    exact hP.row_sum i
  · intro j
    simp only [Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp_rw [← Finset.sum_mul, hP.col_sum, one_mul]
    exact hQ.col_sum j

theorem transpose (hP : DoublyStochastic P) : DoublyStochastic Pᵀ :=
  ⟨fun i j => hP.nonneg j i, hP.col_sum, hP.row_sum⟩

/-- The unnormalized constant vector is a right fixed vector. -/
theorem fixed_one (hP : DoublyStochastic P) : P *ᵥ (fun _ => 1) = (fun _ => 1) := by
  ext i
  simpa [Matrix.mulVec, dotProduct] using hP.row_sum i

/-- It is also a left fixed vector, expressed as a right fixed vector of the transpose. -/
theorem transpose_fixed_one (hP : DoublyStochastic P) :
    Pᵀ *ᵥ (fun _ => 1) = (fun _ => 1) := hP.transpose.fixed_one
end DoublyStochastic

/-- Squared Euclidean norm on the finite site space (not the function-space sup norm). -/
def vectorNormSq {n : ℕ} (v : Fin n → ℝ) : ℝ := ∑ i, v i ^ 2

/-- A.5's telescoping equality. No stochasticity or unit-vector assumption is needed
for the equality itself; nonnegativity of its summands is a separate assertion. -/
theorem two_stage_dissipation {n : ℕ} (P P' : Matrix (Fin n) (Fin n) ℝ)
    (v : Fin n → ℝ) :
    1 - vectorNormSq ((P' * P) *ᵥ v) =
      (1 - vectorNormSq (P *ᵥ v)) +
        (vectorNormSq (P *ᵥ v) - vectorNormSq (P' *ᵥ (P *ᵥ v))) := by
  rw [Matrix.mulVec_mulVec]
  ring
end CoherentModes
