module

public import CoherentModes.Stochastic
public import Mathlib.Analysis.Matrix.PosDef

/-! Counting-measure contraction, dissipation nonnegativity, and spectral bounds. -/
@[expose] public section
namespace CoherentModes
open Matrix
open scoped BigOperators

/-- Weighted Cauchy–Schwarz proves contraction for every bistochastic matrix,
without symmetry or positive-semidefiniteness. -/
theorem DoublyStochastic.contracts {n : ℕ} {P : Matrix (Fin n) (Fin n) ℝ}
    (hP : DoublyStochastic P) (v : Fin n → ℝ) :
    vectorNormSq (P *ᵥ v) ≤ vectorNormSq v := by
  have hrow (i : Fin n) : (P *ᵥ v) i ^ 2 ≤ ∑ j, P i j * v j ^ 2 := by
    have h := Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul (s := Finset.univ)
      (r := fun j => P i j * v j) (f := fun j => P i j)
      (g := fun j => P i j * v j ^ 2)
      (fun j _ => hP.nonneg i j)
      (fun j _ => mul_nonneg (hP.nonneg i j) (sq_nonneg (v j)))
      (fun j _ => by nlinarith only [sq_nonneg (P i j * v j)])
    simpa only [hP.row_sum, one_mul, Matrix.mulVec, dotProduct] using h
  calc
    vectorNormSq (P *ᵥ v) ≤ ∑ i, ∑ j, P i j * v j ^ 2 :=
      Finset.sum_le_sum fun i _ => hrow i
    _ = vectorNormSq v := by
      rw [Finset.sum_comm]
      simp_rw [← Finset.sum_mul, hP.col_sum, one_mul]
      rfl

/-- A.5: for a unit input, each stage's dissipation is nonnegative. -/
theorem two_stage_dissipation_nonneg {n : ℕ} {P P' : Matrix (Fin n) (Fin n) ℝ}
    (hP : DoublyStochastic P) (hP' : DoublyStochastic P')
    (v : Fin n → ℝ) (hv : vectorNormSq v = 1) :
    0 ≤ 1 - vectorNormSq (P *ᵥ v) ∧
      0 ≤ vectorNormSq (P *ᵥ v) - vectorNormSq (P' *ᵥ (P *ᵥ v)) := by
  constructor
  · have h := hP.contracts v
    rw [hv] at h
    exact sub_nonneg.mpr h
  · exact sub_nonneg.mpr (hP'.contracts (P *ᵥ v))

/-- Upper Rayleigh bound for a bistochastic matrix (symmetry not required). -/
theorem DoublyStochastic.rayleigh_le {n : ℕ} {P : Matrix (Fin n) (Fin n) ℝ}
    (hP : DoublyStochastic P) (v : Fin n → ℝ) :
    v ⬝ᵥ (P *ᵥ v) ≤ vectorNormSq v := by
  have hd : 0 ≤ ∑ i, (v i - (P *ᵥ v) i) ^ 2 :=
    Finset.sum_nonneg fun i _ => sq_nonneg _
  have he : (∑ i, (v i - (P *ᵥ v) i) ^ 2) =
      vectorNormSq v + vectorNormSq (P *ᵥ v) - 2 * (v ⬝ᵥ (P *ᵥ v)) := by
    simp only [vectorNormSq, dotProduct, Finset.mul_sum, ← Finset.sum_add_distrib,
      ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [he] at hd
  have hc := hP.contracts v
  linarith

/-- Section 3.1: every eigenvalue of a PSD bistochastic real matrix is in [0,1].
Mathlib's PSD hypothesis includes the Hermitian (real symmetric) condition. -/
theorem stochastic_psd_eigenvalues_mem_Icc {n : ℕ} {P : Matrix (Fin n) (Fin n) ℝ}
    (hP : DoublyStochastic P) (hPSD : P.PosSemidef) (i : Fin n) :
    hPSD.isHermitian.eigenvalues i ∈ Set.Icc (0 : ℝ) 1 := by
  constructor
  · exact hPSD.eigenvalues_nonneg i
  · have h := hP.rayleigh_le (hPSD.isHermitian.eigenvectorBasis i)
    have hv : vectorNormSq ⇑(hPSD.isHermitian.eigenvectorBasis i) = 1 := by
      rw [vectorNormSq, ← EuclideanSpace.real_norm_sq_eq]
      rw [hPSD.isHermitian.eigenvectorBasis.orthonormal.1 i, one_pow]
    rw [hv] at h
    simpa only [hPSD.isHermitian.eigenvalues_eq, RCLike.re_to_real,
      star_trivial] using h
end CoherentModes
