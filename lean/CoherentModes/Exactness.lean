module

public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Tactic

/-! Complete orthogonal coordinates, Parseval tails, and gradient block invariance.
All sums use counting measure. Rectangular matrices are allowed for activations. -/
@[expose] public section

namespace CoherentModes
open Matrix
open scoped BigOperators

/-- Column orthonormality of a square real matrix. -/
def Orthogonal {n : ℕ} (Ψ : Matrix (Fin n) (Fin n) ℝ) : Prop := Ψᵀ * Ψ = 1

/-- Squared Frobenius norm, explicitly avoiding the default matrix norm. -/
def frobeniusSq {n p : ℕ} (F : Matrix (Fin n) (Fin p) ℝ) : ℝ :=
  ∑ x, ∑ j, F x j ^ 2

/-- The Frobenius norm is the square root of the entrywise sum of squares. -/
noncomputable def frobeniusNorm {n p : ℕ} (F : Matrix (Fin n) (Fin p) ℝ) : ℝ :=
  Real.sqrt (frobeniusSq F)

theorem frobeniusNorm_sq {n p : ℕ} (F : Matrix (Fin n) (Fin p) ℝ) :
    frobeniusNorm F ^ 2 = frobeniusSq F := by
  apply Real.sq_sqrt
  exact Finset.sum_nonneg fun x _ => Finset.sum_nonneg fun j _ => sq_nonneg (F x j)

/-- The loading of column i of Ψ, as a vector in feature space. -/
def loading {n p : ℕ} (Ψ : Matrix (Fin n) (Fin n) ℝ)
    (F : Matrix (Fin n) (Fin p) ℝ) (i : Fin n) : Fin p → ℝ :=
  Fᵀ *ᵥ (fun x => Ψ x i)

/-- Sum of the retained outer products ψᵢ aᵢᵀ. -/
def modeSum {n p : ℕ} (Ψ : Matrix (Fin n) (Fin n) ℝ)
    (F : Matrix (Fin n) (Fin p) ℝ) (S : Finset (Fin n)) : Matrix (Fin n) (Fin p) ℝ :=
  ∑ i ∈ S, Matrix.vecMulVec (fun x => Ψ x i) (loading Ψ F i)

theorem orthogonal_complete {n : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) : Ψ * Ψᵀ = 1 := by
  exact mul_eq_one_comm.mp hΨ

/-- A.6: completeness reconstructs every activation matrix. -/
theorem exact_reconstruction {n p : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) (F : Matrix (Fin n) (Fin p) ℝ) :
    F = Ψ * Ψᵀ * F := by
  rw [orthogonal_complete hΨ, Matrix.one_mul]

theorem coefficient_eq_loading {n p : ℕ} (Ψ : Matrix (Fin n) (Fin n) ℝ)
    (F : Matrix (Fin n) (Fin p) ℝ) (i : Fin n) (j : Fin p) :
    (Ψᵀ * F) i j = loading Ψ F i j := by
  simp [loading, Matrix.mul_apply, Matrix.mulVec, dotProduct, mul_comm]

/-- Frobenius squared norm is the sum of the diagonal of FᵀF. -/
theorem frobeniusSq_eq_gram {n p : ℕ} (F : Matrix (Fin n) (Fin p) ℝ) :
    frobeniusSq F = ∑ j, (Fᵀ * F) j j := by
  simp only [frobeniusSq, Matrix.mul_apply, Matrix.transpose_apply, ← pow_two]
  exact Finset.sum_comm

/-- Parseval for any coefficient matrix under left multiplication by Ψ. -/
theorem frobeniusSq_orthogonal_mul {n p : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) (A : Matrix (Fin n) (Fin p) ℝ) :
    frobeniusSq (Ψ * A) = frobeniusSq A := by
  rw [frobeniusSq_eq_gram, frobeniusSq_eq_gram]
  have hgram : (Ψ * A)ᵀ * (Ψ * A) = Aᵀ * A := by
    rw [Matrix.transpose_mul]
    calc
      Aᵀ * Ψᵀ * (Ψ * A) = Aᵀ * (Ψᵀ * Ψ) * A := by simp only [Matrix.mul_assoc]
      _ = Aᵀ * A := by rw [hΨ, Matrix.mul_one]
  rw [hgram]

/-- Row masking in coefficient space is precisely a sum of retained outer products. -/
theorem modeSum_eq_mul_mask {n p : ℕ} (Ψ : Matrix (Fin n) (Fin n) ℝ)
    (F : Matrix (Fin n) (Fin p) ℝ) (S : Finset (Fin n)) :
    modeSum Ψ F S = Ψ * (Matrix.of fun i j => if i ∈ S then (Ψᵀ * F) i j else 0) := by
  classical
  ext x j
  simp [modeSum, Matrix.mul_apply, Matrix.vecMulVec, Matrix.sum_apply,
    loading, Matrix.mulVec, dotProduct, mul_comm, mul_ite]

/-- A.6: the complete outer-product expansion. -/
theorem exact_mode_expansion {n p : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) (F : Matrix (Fin n) (Fin p) ℝ) :
    F = modeSum Ψ F Finset.univ := by
  rw [modeSum_eq_mul_mask]
  simp only [Finset.mem_univ, ite_true]
  change F = Ψ * (Ψᵀ * F)
  simpa only [Matrix.mul_assoc] using exact_reconstruction hΨ F

/-- A.6: an arbitrary subset's squared reconstruction residual is the omitted
loading energy, not a claim of optimal rank-r approximation. -/
theorem exact_tail_identity {n p : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) (F : Matrix (Fin n) (Fin p) ℝ) (S : Finset (Fin n)) :
    frobeniusSq (F - modeSum Ψ F S) =
      ∑ i ∈ Sᶜ, ∑ j, loading Ψ F i j ^ 2 := by
  classical
  have hres : F - modeSum Ψ F S =
      Ψ * (Matrix.of fun i j => if i ∈ S then 0 else (Ψᵀ * F) i j) := by
    rw [modeSum_eq_mul_mask]
    calc
      F - Ψ * (Matrix.of fun i j => if i ∈ S then (Ψᵀ * F) i j else 0) =
          Ψ * (Ψᵀ * F) - Ψ * (Matrix.of fun i j => if i ∈ S then (Ψᵀ * F) i j else 0) := by
            rw [← Matrix.mul_assoc, ← exact_reconstruction hΨ F]
      _ = Ψ * ((Ψᵀ * F) - (Matrix.of fun i j => if i ∈ S then (Ψᵀ * F) i j else 0)) :=
        (Matrix.mul_sub _ _ _).symm
      _ = _ := by congr 1; ext i j; by_cases hi : i ∈ S <;> simp [hi]
  rw [hres, frobeniusSq_orthogonal_mul hΨ]
  simp [frobeniusSq, coefficient_eq_loading, Finset.sum_ite,
    Finset.filter_notMem_eq_sdiff, Finset.compl_eq_univ_sdiff]

/-- The same tail identity written with the explicitly defined Frobenius norm. -/
theorem exact_tail_norm_identity {n p : ℕ} {Ψ : Matrix (Fin n) (Fin n) ℝ}
    (hΨ : Orthogonal Ψ) (F : Matrix (Fin n) (Fin p) ℝ) (S : Finset (Fin n)) :
    frobeniusNorm (F - modeSum Ψ F S) ^ 2 =
      ∑ i ∈ Sᶜ, ∑ j, loading Ψ F i j ^ 2 := by
  rw [frobeniusNorm_sq, exact_tail_identity hΨ]

/-- A.7: insert the complete mode basis into the supplied gradient matrix ΔᵀU.
This is algebra only; the identification of Δ with a reverse-mode derivative is not formalized. -/
theorem exact_gradient_decomposition {n p q : ℕ}
    {Ψ : Matrix (Fin n) (Fin n) ℝ} (hΨ : Orthogonal Ψ)
    (Δ : Matrix (Fin n) (Fin q) ℝ) (U : Matrix (Fin n) (Fin p) ℝ) :
    Δᵀ * U = ∑ i, Matrix.vecMulVec (loading Ψ Δ i) (loading Ψ U i) := by
  have h : Δᵀ * U = (Δᵀ * Ψ) * (Ψᵀ * U) := by
    calc
      Δᵀ * U = Δᵀ * (Ψ * Ψᵀ * U) := by rw [← exact_reconstruction hΨ U]
      _ = _ := by simp only [Matrix.mul_assoc]
  rw [h]
  ext x j
  simp [Matrix.mul_apply, Matrix.sum_apply, Matrix.vecMulVec,
    loading, Matrix.mulVec, dotProduct, mul_comm]

/-- A.7: even a rectangular block factor has an unchanged Gram projector after
an orthogonal change of its block coordinates. -/
theorem block_projector_rotation {n b : ℕ} (B : Matrix (Fin n) (Fin b) ℝ)
    {R : Matrix (Fin b) (Fin b) ℝ} (hR : Orthogonal R) :
    (B * R) * (B * R)ᵀ = B * Bᵀ := by
  rw [Matrix.transpose_mul]
  calc
    B * R * (Rᵀ * Bᵀ) = B * (R * Rᵀ) * Bᵀ := by simp only [Matrix.mul_assoc]
    _ = B * Bᵀ := by rw [orthogonal_complete hR, Matrix.mul_one]

/-- Block pressure itself (not merely its square) is rotation invariant. -/
theorem block_pressure_rotation {n b p q : ℕ} (B : Matrix (Fin n) (Fin b) ℝ)
    {R : Matrix (Fin b) (Fin b) ℝ} (hR : Orthogonal R)
    (Δ : Matrix (Fin n) (Fin q) ℝ) (U : Matrix (Fin n) (Fin p) ℝ) :
    frobeniusNorm (Δᵀ * ((B * R) * (B * R)ᵀ) * U) =
      frobeniusNorm (Δᵀ * (B * Bᵀ) * U) := by
  rw [block_projector_rotation B hR]
end CoherentModes
