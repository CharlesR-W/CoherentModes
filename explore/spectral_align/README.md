# Spectral alignment measurements

Exploratory, single-site image-graph comparison; protocol in `SPEC.md`. No page is built here.

## Files and commands

- `align.py`: device-agnostic eigenspectra, projector/eigenvalue/heat/CKA distances, streamed pairwise accumulation, DTW, architecture reference and permutation null.
- `test_align.py`: small CPU controls, including repeated versus separated planted spectra and high-time diffusion underflow.
- `measure_modal.py`: detached Modal app `cm-spectral-align`; `deploy`, `launch debug|pilot|all`, `status`, `fetch`.
- `sanity.py`: checks fetched tensors and exact probe-index/label agreement.

```bash
uv run --with torch --with numpy --with dtw-python --with pytest pytest test_align.py -q
uv run --with modal python measure_modal.py deploy
uv run --with modal python measure_modal.py launch pilot
# Inspect cost gate; redeploy with SALIGN_TIMEOUT set to suggested_timeout.
uv run --with modal python measure_modal.py launch all
uv run --with modal python measure_modal.py status
uv run --with modal python measure_modal.py fetch
uv run --with numpy python -I sanity.py
```

Rules are numbered `0` (union 15-NN) and `1` (per-point perplexity 100). Eigenvalue files include the constant eigenvalue at index zero; exported U contains the next 64 modes. Iso-accuracy cells that coincide with standard checkpoints are deduplicated and flagged in `cells.json`.

The scratch Volume footprint is approximately 34 GB of full float32 eigenvectors and 17 GB of float32 centred-cosine Grams for 1,056 cells, plus small spectra/span/metadata files. Operators and eigensolves are float64. All-pairs workers materialize only one rule and diffusion time in host memory, then accumulate the flattened-kernel Gram in 32-row chunks on GPU; reconstructed kernels are not persisted. Heat eigenweights are analytically unit-Frobenius-normalized in float64 log space before casting, avoiding high-time float32 underflow without changing cosine distances.

## Status

Complete, 2026-11-18. **27 local data files, 469,458,991 bytes**, under `data/`: 978 cells, both rules' pair-distance tensors, CKA, full spectra, cut reproducibility, probe labels/indices, availability masks, timings and 13 selected eigenvector banks. Six CPU tests and the post-fetch sanity checks pass. No page was built and nothing was committed or pushed.

| Stage | Call ID | Outcome |
|---|---|---|
| Debug/control | `fc-01M7RZE14BGV2A6DMMSVC8GKQ0` | Control 6 exact; R20-final extraction 17.15 s |
| Largest pilot | `fc-01M7S0C84SWRHJRHH9WS3GTEBV` | R56-final 32.81 s; projected total **$6.60** |
| Initial fan-out | `fc-01M7S0EHZBVHY26VEKKQY4DF53` | Stopped before assemble on three units' undefined perplexity rule; successful units retained |
| Resumed full run | `fc-01M7S0SN80Q6SR83FEQPM4YCN3` | Complete; only failed units repaired, no operator substitution |

Measured timings: 49 completed extraction units, 1,085.47 worker-seconds total, 13.94–33.25 s per unit. Largest pilot included all 30 sites × both rules × two subset rebuilds. Extraction timeout was set to 128 s from that pilot, with at most eight concurrent extraction workers and zero retries. All-pairs rule 0 took **1,049.14 s**, rule 1 **607.71 s**, primarily Volume reads. Their maximum random-pair discrepancies against float64 dense matrix powers were **4.35120e-6** and **2.09845e-5**, both below 1e-4. Control 6 matched both stored operators with **maximum absolute error 0** on both runs (10.47 s initially, 6.51 s on repeat).

The resumed coordinator took 22.11 min. From initial debug launch through final coordinator completion was 45.9 min, including the wait for existing training; through completed local fetch it was **46.9 min**. The actual scratch matrix footprint is approximately **46.26 GB** (30.61 GB full eigenvectors plus 15.65 GB Grams), excluding filesystem/metadata overhead.

**Estimated measurement spend: about $3.25**, not a billing statement. `timings.json` records $1.9488 from completed worker runtimes. Three failed initial attempts add at most $0.2476 at their 128 s timeouts; the repeated control adds $0.00675 not retained in the latest per-unit record. A further $1 allowance covers image/cold-start/idle/coordinator/assembly costs, giving $3.203 before rounding. Already-running training is excluded. This remains below both the $6.60 pilot projection and the hard $30 ceiling; no budget-driven measurement reductions were needed.

### Final test accuracy

| Network | Seed 0 | Seed 1 |
|---|---:|---:|
| ResNet-20 | 91.56% | 91.51% |
| ResNet-32 | 92.51% | 92.64% |
| ResNet-56 | 92.81% | 93.03% |

The four pre-existing training runs completed without relaunch: R32 seeds 0/1 in 1,146.6 / 1,207.5 s; R56 seeds 0/1 in 1,715.5 / 1,598.8 s.

### Iso-accuracy selection

All four earliest target crossings were at step 33,280. The required acceptance tolerance is +0.5 percentage points, not nearest-checkpoint selection.

| Network | Same-seed R20 target | Crossing accuracy | Overshoot | Accepted? |
|---|---:|---:|---:|---|
| R32 seed 0 | 91.56% | 92.03% | +0.47 pp | Yes |
| R32 seed 1 | 91.51% | 92.16% | +0.65 pp | Unavailable |
| R56 seed 0 | 91.56% | 92.33% | +0.77 pp | Unavailable |
| R56 seed 1 | 91.51% | 92.66% | +1.15 pp | Unavailable |

Thus 960 standard cells plus 18 accepted iso-accuracy cells gives 978, rather than the protocol's approximate 1,056.

### Explicit deviations and unavailable measurements

The original perplexity-100 builder cannot bracket its entropy target for **42 early ResNet-56 cells**. The construction was not changed and the tie-aware alternative was not substituted. After stopping and consulting the coordinator, these rule-1 cells were explicitly marked unavailable: NaN eigenvalue/span rows and NaN pair-distance rows/columns, with complete cell/error records in `rule1_unavailable.json`.

- R56 seed 0, step 64: b15–b27, pool, logits (15 cells).
- R56 seed 1, step 64: b15–b27, pool, logits (15 cells).
- R56 seed 1, step 256: b18–b27, pool, logits (12 cells).

**Every rule-0 tensor and CKA is finite.** Rule 1 is finite on its 936 available cells and all pairs among them. There are zero subset-only failures (`subset_failures.json` is empty). The sanity check permits NaNs only at documented unavailable indices; diagonals are exactly zero on available cells, and distances are symmetric. Probe indices and labels exactly match the existing explorer probe.

Prices checked at <https://modal.com/pricing> on 2026-11-18: A100-40GB $0.000583/s, CPU $0.0000131/physical-core-s (two vCPUs per physical core), memory $0.00000222/GiB-s. Estimates conservatively charge each requested vCPU as a physical core and exclude image-build/cold-start overhead. This measurement task uses a hard $30 gate (stricter than the original SPEC's $35 extraction gate).

Protocol correction: `dtw-python` symmetric2 charges the initial cell once, so its path-independent total weight is L_A + L_B - 1. The coordinator corrected SPEC accordingly; code and the constant-cost test use this denominator. Documented subset-rebuild failures, if any, retain full-probe measurements but have NaN span, explicitly listed in `subset_failures.json` and treated as unresolved; none occurred in the completed run. The full-probe rule-1 exceptions above are separately masked. The requested `~/.claude/compute/README.md` was absent; existing detached Modal conventions and the Modal skill document were used.

## Verdict (2026-11-19; default rule, final checkpoint unless stated)

Page: the "Alignment" tab of the `cifar_coherent` site (`../cifar_coherent/site/content/alignment.html`; build with
`--alignment ../spectral_align/data`), published as `CoherentModes/cifar-coherent`. Not committed, not pushed.

- **Across depth (hypothesis → supported):** DTW-matched projector distance between ResNet-20 and ResNet-56 is
  0.19–0.22 at k=9 (0.10–0.14 at k=3; diffusion kernel t=16: 0.10–0.12), against 0.18 / 0.10 / 0.09–0.11 for two seeds of
  one depth and 0.30–0.38 / 0.25–0.32 / 0.21–0.26 for permuted blocks. Unconstrained argmins are monotone (no
  inversions); the path passes all three stage ends; within a stage it is not proportional to depth.
- **Learned, not architectural:** DTW cost equals the permutation null through step 256 and separates between steps
  1,024 and 6,400. Same-seed networks of different depth share initial draws for the stem and early blocks, so
  cross-depth summaries use different-seed pairings only.
- **Within ResNet-20:** gradual change with depth; adjacent sites are no farther apart than two seeds at the same site
  (0.11–0.34 against a floor of 0.10–0.35), except b9 → pool (0.48). Each site approaches its final operator
  monotonically and reaches the seed floor by step 25,600.
- **k:** pair orderings agree across k (Spearman ≥ 0.83); cuts are reproducible in ≥ 86% of final snapshots for k ≤ 4,
  about half for k = 6–9, rarely beyond 10. Small k and t = 16 separate best from the null.
- **Baseline:** CKA on the centred-cosine Grams gives nearly the same alignment (Spearman 0.86–0.97). Eigenvalue-only
  distance does not (0.45–0.57, inversions up to 0.25).
- **Iso-accuracy:** only ResNet-32 seed 0 has an accepted checkpoint (step 33,280); accuracy jumps past the target at
  the first learning-rate drop, so the saved schedule cannot match accuracy more finely.
- Total Modal spend ≈ $6.5 (training ≈ $3.1 at L40S rates for 5,665 s, measurement ≈ $3.25). Scratch on
  `cm-cifar-coherent:/work/salign` is ≈ 46 GB and can be deleted once the results are accepted.
