# Review 1 of PLAN.md draft 2 (GPT-6 Astra, high effort, 2026-11-15)

Recorded by the orchestrator from the reviewer's hand-back (the reviewer's session could not write files).

Material objections, with the response in draft 3:

1. Cached cosine Grams discard activation norms: cannot give fit-only centring, N2a's pre-normalisation covariance, or
   uncentred views. -> Accepted. New extraction stores the raw uncentred Gram X X^T per cell (everything else derives
   from it). PLAN §2 preamble, §9.
2. N2b invalid as written (concatenated row-normalised Z need not be centred, can have rank N). Exact Gram-only form:
   one orthogonal O fixing the constant, C*_l = D_l O C_l O^T D_l with D_l restoring unit diagonals; preserves means and
   all joint second moments before renormalisation; report the mismatch after. -> Accepted verbatim. §3.
3. Transfer orientation inconsistent (joint table and W are source-by-destination; Q and the composition are
   destination-by-source). -> Accepted: one convention stated. §6.
4. Chapman-Kolmogorov equality is not expected (normalised views are not sufficient Markov states; projection, ridge,
   smoothing break it; counterexample X = Z, Y constant). -> Accepted: "composition discrepancy", with a known-Markov
   calibration. §6.
5. Fit-only permutation is not a held-out independence null. -> Accepted: permute pairing within fit, tuning and test,
   repeating selection; coherent permutations across candidate blocks/ranks; N7 does not calibrate selected spectral
   excess. §3, §6.
6. Retention null (m-1)/(N-1) holds only for zero-diagonal stochastic P; with diagonal mass t = tr(P)/N it is
   t + (1-t)(m-1)/(N-1); degree-normalised symmetric matrices need conversion to Markov/function coordinates.
   -> Accepted. §7.
7. Nested shells are erased by cosine normalisation; P2's Gaussian latent span differs from its normalised
   observations. -> Accepted: angular structure on the sphere; every benchmark verifies its planted information
   survives the preprocessing. §8.
8. Pilot needs a finite-resolution success criterion; 64-128 landmarks at 1,000 fit images do not support fine
   transition tables. -> Accepted: preregistered scoring and gates. §6.
9. Scope and delivery budgets unsupported (N3 on all blocks = 21,120 eigensolves; cached bank ~18.4 GB; one all-grid
   hard-label source = 10.56 MB). -> Accepted: transfer pilot reserved first, expensive items on representative blocks,
   iterative solvers, streamed data, benchmark before expansion, synthetic sweeps on Modal. §2, §3, §9, §10.

Checked and found sound: PCA/MDS equivalence; soft-Ulam normalisation; circle-doubling harmonics; fixed-laziness
invariance; centred MAXVAR-GCCA projector formulation; repository mathematical checks pass.
