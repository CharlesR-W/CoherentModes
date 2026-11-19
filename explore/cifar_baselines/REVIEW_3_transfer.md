# Review 3: mathematics of the transfer-operator pilot (GPT-6 Astra; responses by builder C, 2026-11-15)

Recorded by builder C from the orchestrator's relay of the reviewer's hand-back. Reviewed: `TRANSFER.md` (first
version), `transfer.py`, `transfer_pilot.py`, `test_transfer.py`.

Found sound by the reviewer: orientation conventions; whitening; fit-only preprocessing; held-out correlations;
VAMP-2; the within-split permutation control (tuning repeated inside each permutation; no leakage); 14 tests passing.

## Material items

1. **The composition sweep was not an intermediate-basis-only experiment.** `fit_chain(mb_scale=...)` retuned
   bandwidth and ridge at each size, and the soft-Ulam gate shrank the endpoint bases too (saved results had
   endpoint sizes 32 -> 16 -> 8 with the middle at 32) while the plot labelled them middle sizes 32 / 64 / 128.
   Requested: freeze endpoints, ridge convention and absolute kernel bandwidths; mark inadmissible middle sizes
   unavailable; plot actual dimensions; regenerate outputs; keep the exact identity; remove any claim that the
   sweep or the calibration separates finite-basis error, view insufficiency and sampling; report the held-out
   residual cross-moment as a separate diagnostic.
   **Response: accepted.** `fit_chain` lost `mb_scale`. New `transfer.composition_sweep`: base chain selected once;
   then only the b-features change (m_b = f * m_b0, f in {1/4, 1/2, 1, 2, 4}, nested landmarks / eigenfunctions /
   PCs) with endpoint feature objects, eta (relative-ridge convention) and the absolute kappa of every view frozen
   (feature specs accept an absolute kappa). Soft-Ulam sizes below the gate are rows marked `unavailable`. Every row
   records actual dimensions (m_a, m_b, m_c), rho_b, kappa_b, and the numerical rank / condition of G_b. The
   identity is kept and checked relative to the cancelling product (`additivity_error_rel` < 1e-10; the absolute
   residual reached 3e-4 when G_b was ill-conditioned, which is floating-point cancellation of G_b^+). The key
   `partial_test` became `heldout_residual_crossmoment`, documented as a separate diagnostic; the fit-split term is
   `partial`. TRANSFER.md section 5 now states that nothing separates the three sources. `synthetic.json['chain']` and
   `chain_composition.png` regenerated (x axis = actual m_b; endpoints annotated). Observed in passing: at a frozen
   bandwidth Nystrom's numerical rank saturates (64 and 128 landmarks give m_b = 35 on the Markov chain); larger
   soft-Ulam middle tables (64, 128) fail the gate at N_fit = 1000. Tests: `test_composition_sweep_freezes_endpoints`.

2. **Destination sets bypassed the stability requirement** (only source stability was checked; reproduced with
   source overlap 1, destination 0.00007). Requested: gate each side separately; show a coherent pair only when both
   are stable; label the consensus / SEBA sets and joint table "transductive exploratory summaries"; any held-out
   stability statement must compare functions on shared held-out images.
   **Response: accepted.** `heldout_stability` compares each pair of splits on the images held out from both fits
   (about 500 per pair), per side. `transfer_sets` takes a side only if that side is stable; the joint table is
   produced only when both sides have sets. `sets.status`, the top-level `sets_status` and `agreement.scope` carry
   the transductive label. Tests: `test_sets_gated_per_side_and_labelled_transductive`,
   `test_heldout_stability_uses_shared_heldout_images_only`.

3. **The default 10k runner iterated time pairs** that need steps 1024 / 6400, but 10,000-image Grams exist only at
   the final checkpoint. Requested: default to the eight final-checkpoint pairs; preflight all files, shapes and
   image ids before computing; a final-only fixture test.
   **Response: accepted.** `run_real10k` defaults to `FINAL_PAIRS` (8) and raises on a time pair. `preflight`
   (both runners) checks existence, shape, finite positive diagonal, sampled symmetry and, for 10k, the
   `raw10k_index.npz` ids, and raises with the full list before any compute. File templates follow builder A's
   SCHEMA.md (`raw_flat_{step}_{si}.npy`, `raw10k_flat_{step}_{si}.npy`). Test:
   `test_real10k_final_checkpoint_only_fixture`.

## Minor items (folded into TRANSFER.md)

- Soft-Ulam claim: holds for moments under the same table / distribution; a training sigma does not lower-bound a
  held-out correlation; equality needs zero conditional variance of the relevant category functions, not globally
  hard memberships. **Accepted**, section 3.2 rewritten.
- Occupancy: dropping landmarks by a prespecified rule is not vacuous; soft count and Kish ESS are different
  criteria. **Accepted**: section 3.2 states the gate uses the soft count, Kish is reported only, and dropping is a
  legitimate alternative not chosen.
- GCCA: ridge hat matrices are not the identity at positive ridge; centred full-rank unregularised projectors
  approach H, not I. **Accepted**, section 8.
- Hermite-product spectrum: only for jointly Gaussian raw views, not cosine-normalised P2. **Accepted**, section 9.
- m selection: the code reported two fixed-m variants rather than selecting m. **Accepted**: the real runner now
  also reports `<est>_mjoint`, with m selected by tuning VAMP-2 and the selection repeated inside each permutation
  (shared permutation sequences across m); the fixed-m variants stay as additional characterisation, declared in
  section 4.4. Test: `test_joint_m_selection_repeated_inside_permutations`.

All 19 tests pass:
`uv run --with torch --with numpy --with scipy --with scikit-learn --with deeptime --with pytest pytest explore/cifar_baselines/test_transfer.py`.
