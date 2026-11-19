# NOTES_C: transfer-operator pilot, builder C (2026-11-15)

Design: `TRANSFER.md`. Scope of this phase: synthetic only; real data after builder A's raw-Gram extraction lands.
Nothing committed, pushed or deployed.

## Files

- `transfer.py`: views from raw Grams (fit-only centring, block-wise), feature maps (linear / Nystrom / diffusion /
  soft-Ulam memberships), `fit_cca`, `fit_ulam`, `soft_ulam`, `whitened_cca`, `heldout_correlations`, `vamp2`,
  `tune`, `permutation_control`, `fit_chain`, `composition_discrepancy`, `composed_model`, `diffusion_reference`.
  Public API used by builder B's `bench.py` (`FeatureBank`, `View.from_X`, `pair`, `tune`, `evaluate`,
  `permutation_control`): kept stable.
- `test_transfer.py`: 14 asserted tests (TRANSFER.md section 7).
- `transfer_pilot.py`: `synthetic`, `curve-local`, `real`, `real10k`, `smoke-real`, `plots`.
- `transfer_modal.py` (not in the ownership list; added as builder C's own Modal CPU fan-out, adapted from the
  modal skill template; app `crw-cm-transfer`, volume `crw-cm-transfer-results`).
- Outputs `out/transfer/`: `synthetic.json`, `curve_p8.json` (120 units), `p8_curve_test_corr.png`,
  `p8_curve_train_sv.png`, `p8_occupancy_gate.png`, `chain_composition.png`, `smoke_real/` (stand-in smoke),
  `modal_runs/`, `modal/<run_id>/` (raw parts).

## Test command and result

    uv run --with torch --with numpy --with scipy --with scikit-learn --with deeptime --with pytest \
        pytest explore/cifar_baselines/test_transfer.py        # 14 passed (~15 s), deeptime 0.5.0 installed

The command as briefed (without `--with pytest`) fails with "Failed to spawn: pytest": pytest is not in the
environment otherwise. Threads: OMP/OpenBLAS/MKL = 2 and torch 2 threads, set in the test and runner modules.

## Synthetic numbers

P8, sigma = 0.35; truth (two per harmonic) 0.941, 0.941, 0.783, 0.783, 0.576, 0.576. Curve: tune 1,000, test 2,000
fresh images per cell; 3 seeds; control = 20 within-split permutations with tuning repeated (seed 0). Mean held-out
correlations of functions 1-6 (sd over seeds <= 0.02 at N_fit >= 1000):

| N_fit | estimator, m | held-out correlations 1-6 | training singular values 1-6 | control 95% (max over j) |
|---|---|---|---|---|
| 1000 | Nystrom, 64 | .94 .93 .76 .77 .55 .55 | .95 .94 .82 .80 .64 .60 | .04 |
| 1000 | diffusion, 64 | .93 .93 .75 .75 .51 .51 | .95 .95 .83 .81 .67 .64 | .04 |
| 1000 | soft Ulam (gate: m = 32) | .93 .92 .75 .70 .49 .42 | .81 .78 .49 .38 .19 .15 | .04 |
| 1000 | Nystrom, 128 | .93 .93 .74 .75 .51 .51 | .95 .94 .83 .81 .67 .64 | .05 |
| 1000 | diffusion, 128 | .93 .92 .71 .72 .44 .46 | .94 .94 .83 .82 .70 .67 | .04 |
| 6000 | Nystrom, 64 | .94 .94 .78 .78 .57 .58 | .94 .94 .79 .78 .59 .58 | .04 |
| 6000 | diffusion, 64 | .94 .94 .78 .78 .56 .57 | .94 .94 .79 .78 .59 .58 | .06 |
| 6000 | soft Ulam, 64 | .94 .94 .77 .78 .56 .55 | .89 .88 .63 .61 .37 .34 | .05 |
| 6000 | soft Ulam, 128 | .94 .94 .78 .77 .56 .57 | .92 .92 .72 .71 .48 .46 | .04 |
| any | linear CCA (2 functions) | .00 to .03 | .04 to .12 | .03 to .04 |

- Control (i) at N_fit = 1000 (`synthetic.json`, 50 permutations, n_test = 2000): test-correlation 95% points
  0.024-0.053 for every estimator and function. Under the same control the *training* singular values have median
  0.41-0.48 (Nystrom / diffusion, m = 64) and 0.60-0.65 (m = 128): the in-sample floor at this n and m.
- P8 with independent pairing (same marginals; 5 draws, N_fit = 1000): max |held-out correlation| 0.027 (linear),
  0.048 (Ulam), 0.063 (Nystrom), 0.052 (diffusion).
- Soft-Ulam training singular values sit below its held-out correlations (TRANSFER.md 3.2: sigma_j / (sd f sd g)).
- Occupancy gate (soft count >= 10): landmarks used 16 (N_fit 250), 32 (500, 1000), 64 / 128 as requested (3000,
  6000). Below N_fit = 1000 the gate leaves 8-32 landmarks and soft-Ulam functions 3-6 are unstable across seeds
  (sd up to 0.27).
- Sample size, P8 only: at N_fit = 1000 all six functions of Nystrom and diffusion (both m) and of soft Ulam are
  above the control band by > 0.35; the held-out correlations of functions 5-6 are below truth by 0.02-0.03
  (Nystrom m = 64), 0.07-0.13 (diffusion m = 128) and 0.08-0.15 (Ulam at m = 32); by N_fit = 3000 every
  nonlinear estimator is within 0.03 of truth for all six. m = 128 has more downward bias than m = 64 below
  N_fit = 3000. P8 is a one-dimensional circle; these numbers say nothing about the sample size a CIFAR view needs.
- P2 (h = 3, a = 1; oracle = population linear CCA after preprocessing, 40,000 images: 0.893, 0.865, 0.818; raw
  latent 0.918, 0.893, 0.843): linear held-out 0.875, 0.844, 0.825 then chance; Nystrom 0.861, 0.826, 0.807 then
  0.674, 0.619, 0.611 (degree-2 Hermite dependence, rho_i rho_j); diffusion 0.81, 0.81, 0.75, 0.60, 0.54, 0.48;
  soft Ulam 0.72, 0.75, 0.77, 0.44, 0.47, 0.44. Control 95% points 0.02-0.07.
- Composition, calibration chains (N_fit = 1000, base m = 32, intermediate-basis sweep with endpoints, eta and
  every kappa frozen; regenerated after review 3; `synthetic.json['chain']`, `chain_composition.png`). Relative to
  the top-6 scale of T_ac, by actual m_b:
  Markov chain: Nystrom (a 25, c 30) m_b 8 / 16 / 28 / 35 / 35: partial 0.41 / 0.18 / 0.17 / 0.17 / 0.17,
  regulariser 0.00-0.01, held-out residual cross-moment 0.36 / 0.07 / 0.06 / 0.07 / 0.07; diffusion (a 32, c 32)
  m_b 8 / 16 / 32 / 64 / 128: partial 0.31 / 0.17 / 0.16 / 0.15 / 0.14, regulariser 0.00, held-out 0.13 / 0.06 /
  0.06 / 0.06 / 0.09; soft Ulam (a 32, c 32) m_b 8 / 16 / 32: partial 0.62 / 0.34 / 0.12, regulariser 0.06 / 0.07 /
  0.11, held-out 0.61 / 0.33 / 0.05; m_b 64, 128 unavailable (gate).
  Broken chain: partial 0.88-1.00 and held-out 0.92-1.12 for all available rows; regulariser <= 0.03.
  Fit-split identity: relative residual < 1e-10 (absolute up to 3e-4 when G_b is ill-conditioned).

## Deviations and builder's choices (all also in TRANSFER.md)

- Support / sets operationalisation (signed correlations, j-th vs j-th control function, leading-supported count,
  stability threshold 0.8 for sets, consensus-projector span for SEBA) is mine; the 0.8 is provisional.
- Occupancy gate on the soft count, m halved, no landmark dropped.
- Four composition triples added to the real runner (`TRIPLES`), with the frozen intermediate-basis sweep
  (descriptive; identifies no error component).
- Real runner reports fixed m = 64, m = 128 and the jointly selected m (`mjoint`, selection repeated inside the
  permutations); PLAN names only the selected m.
- Sets: gated per side on held-out stability (images held out from both fits); labelled transductive exploratory
  summaries; joint table only when both sides have sets.
- 10k set: one predeclared split (PLAN names one); the N curve uses nested subsets of its fit split.
- Bandwidth reference kappa_ref = 1 / median nearest-landmark cosine distance (not the 15-NN scale of the existing
  code, which gave kernels too narrow for 64-128 landmarks on P8).
- `real` smoke used stand-in Grams at N = 2,000; the `real10k` smoke ran the same code at a reduced N = 2,000
  (1,200 / 400 / 400) because 12 stand-in 10,000 x 10,000 float64 Grams are 9.6 GB.
- Diffusion features at N_fit = 6000 ran on Modal only (locally the dense Sinkhorn + eigensolve did not finish in
  25 min under contention with the other builders; on Modal 26-55 s per unit).

## Smoke test of the real runner

`python transfer_pilot.py smoke-real`: stand-in raw Grams (N = 2,000, 64 dims, nonzero mean, per-image gain; in
/tmp, deleted after) for every (site, step) the 12 pairs and 4 triples need; `real` on 2 pairs (stem-b1,
b5:1024-64000) x 2 splits x 4 estimators x m = 64 x (3 + 2 permutations) + 4 triples x 2 splits x m_b in
{64, 128, 256}: 123 s; `real10k` code path at a reduced N (curve 300, 1200): 5 s. Output `out/transfer/smoke_real/`
contains every schema field, including SEBA sets where the stability rule passed. Stand-in numbers mean nothing.
Observation: the held-out partial part of the composition rises with m_b on stand-ins and on the calibration chain
(TRANSFER.md section 11, item 6).

## Compute and spend

- Modal (the owner's workspace, app crw-cm-transfer): run 20261115-083829, 120 P8-curve units, CPU 2 physical
  cores / 4 GiB each, 452 container-seconds -> about $0.016 at $0.0000131 per core-second and $0.00000222 per
  GiB-second (modal.com/pricing, read 2026-11-15), plus the coordinator container and the image build; total
  estimated < $0.05 of the $1 cap. The coordinator's final manifest write failed (stale volume view; fixed with
  `vol.reload()` for the next run); all 120 parts were written and fetched.
- Local: tests ~15 s; `synthetic --no-6000` ~10 min at 2 threads; smoke-real see below.

## Review 3 (mathematics, GPT-6 Astra)

Record and responses: `REVIEW_3_transfer.md`. Three material items fixed (composition sweep frozen; sets gated per
side, transductive label, held-out stability on shared held-out images; 10k runner final-checkpoint pairs only with
preflight), minor items folded into TRANSFER.md; 19 tests pass.

## Real-run estimate (written before the extraction landed)

12 pairs x 10 splits x 4 estimators x 2 m x (1 + 200 + 50 permutations) tunings, about 2 million fits; as 120
(pair, split) Modal CPU units roughly 2-4 CPU-hours, well under $1. Needs: the raw Gram file naming (template flag),
steps 1024, 6400, 64000 at b5 / b9 and all 12 sites at 64000.

## Real pilot (Modal run 20261115-091057, 2026-11-15)

Inputs: builder A's raw Grams on volume `cm-cifar-coherent`, `/work/r20s0c` (`raw_flat_<step>_<si>.npy`,
`raw10k_flat_64000_<si>.npy`, `raw10k_index.npz`); preflight passed in the coordinator. Smoke run 20261115-090443
(10 units, $0.006) first. Full run: 204 units, 0 failed: 120 (pair, split) units, 12 reference units, 40
(triple, split) units at 2,000 images (n_perm 200, fit-only 50, m in {64, 128} + joint), 32 10k units (8 final
pairs x N_fit {500, 1000, 2000, 6000}, n_perm 50). Unit container-seconds 11,285 (2 cores / 4 GiB, est. $0.40) +
2,208 (2 cores / 8 GiB, est. $0.10); with coordinators, image builds, the P8 curve and the smoke, total builder-C
Modal spend estimated at about $0.6 of the $5 cap. Outputs: `out/transfer/real/transfer_pilot.json` (4.9 MB, schema
transfer-pilot-v2), `out/transfer/real/transfer_pilot_10k.json` (0.4 MB); raw parts in `out/transfer/modal/20261115-091057/`.
