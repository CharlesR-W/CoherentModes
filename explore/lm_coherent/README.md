# Coherent sets in a small language model (toy, exploratory)

> **Superseded (2026-11-13).** The report for this study is replaced by the joint report in `explore/ncs_report/` (published as `CoherentModes/ncs-vit-mamba`). The data and code here are still read by that report's build; the page in this directory is kept for reference and is no longer updated.

Spec: `SPEC.md` (v2.1). Pipeline: `probe.py`, `lm_construction.py`, `lm_cells.py`, `lm_tree.py`, `lm_schema.py`,
`lm_modal.py`; tests `test_lm.py`. Page: `page/` (separate). Data (untracked): `data/_fixture/`,
`data/_fixture_flat/` (synthetic, MOCKUP only) and `data/p160t/` (the run).

## Status

**Complete, 2026-11-12.** `data/p160t/` (13 MB) is fetched and validates (`lm_schema.py validate`: 438 SEBA
records, worst label reconstruction 0.999; P1 exact; threshold 3.0 by the fixed rule). N = 1,000, no items removed,
transformers 4.57.6.

- **Probe (U0):** 1,000 snippets, 200 per source. Rejections by reason (documents drawn):
  - maths (222): maths_markup 135, too_short 69, ascii 7, whitespace 2, dup_multiset_t4 2, dup_8gram 1;
  - code (274): too_short 291, dup_multiset_t4 68, code_digits 20, whitespace 13, ascii 10, code_long_line 5;
  - wiki (272): too_short 486, dup_multiset_t4 2, ascii 1;
  - forum (220): too_short 154, dup_multiset_t4 93, ascii 15, whitespace 2;
  - poetry (202): too_short 14.
  Selection effect (hazard): the t = 4 multiset de-duplication rejected 68 `code` and 93 `forum` candidates with
  shared openings (license headers, question templates), so those sources lean toward unusual openings.
- **Pilot 1 stopped at C1** (float32 error 8.6e-4 against 1e-4). Diagnosis: TF32 off, the same error with IEEE fp32
  forced everywhere, exactly 0.0 in float64. Decision (orchestrator, on the owner's behalf): gate C1 in float64 at
  1e-10, and report the float32 error per layer as a hazard (SPEC 8, 9, 14.2 item 14).
- **Pilot 2 passed the gate:**
  - C1 float64 error 0.0. The float32 error by layer (hazard) was L2 1.9e-6, L4 9.0e-5, L6 5.7e-4, L8 3.0e-4,
    L10 8.6e-4, L12 5.0e-4.
  - C2 bitwise identical. P3 on real data: largest deviation 1.3e-15.
  - Sinkhorn max row error 5.1e-13; min pairwise distance 3.9e-6; no zero-norm items; min PN(psi_1) 27.4.
- **Full run:** U1 and U2 for the other 4 steps, then U3. All units succeeded on the first attempt.
- **Controls:** C1, C2, P1, P2 and P3 pass. P1: planted Upsilon min 8.26, structureless max 1.88. P2: Jaccard 1.0
  in both cases. P3: largest deviation 2.2e-15 over all steps.

## Runs

| stage | coordinator call | launched (VPS time) | outcome |
|---|---|---|---|
| pilot 1 | `fc-01M78HXATF01YWEGDFDC5HKSHA` | 2026-11-12 10:56 | U0 ok (34 s); U1 stopped at C1 (37 s) |
| diagnosis | 3 calls of `diag_c1` (`lm_modal.py diag`) | 2026-11-12 11:0x | 2 failed on a bug in the diagnostic itself (one ran stale code from a warm container); 1 ok |
| pilot 2 | `fc-01M78KFDJTPNG2G954ZNP5QVGT` | 2026-11-12 11:24 | gate passed; U0 cached, U1 56 s, U2 50 s; 2.9 min wall |
| full | `fc-01M78KPY4GZVF0V8CS2DXC6H9F` | 2026-11-12 11:28 | U1 x4 ok (2.1 min), U2 x4 ok (1.0 min), U3 ok (11.1 min); 14.5 min wall |

Timeouts after the pilot (`runs/timeouts.json`): 2 x the pilot's durations, with a 120 s floor: U0/U1/U2 120 s,
U3 2,400 s (U3 has no pilot).

## Spend

Prices re-checked at <https://modal.com/pricing> on 2026-11-12: L40S $0.000542/s; CPU $0.0000131 per
physical core-second; memory $0.00000222 per GiB-second (preemptible functions; no multiplier). Spend figures come
from the attempts ledger on the volume (`work/p160t/attempts/`), a timing/rate estimate, not an invoice.

Final ledger: **$1.054** against the $6 ceiling (projected $1.9).

| kind | attempts | counted |
|---|---|---|
| U0 | 2 | $0.019 |
| U1 | 6 | $0.457 |
| U2 | 5 | $0.112 |
| U3 | 1 | $0.130 |
| diagnostics | 3 | $0.330 |
| coordinators | 3 | $0.006 |

These are upper bounds: each attempt adds 90 s of container overhead, and failures count the wall time since
dispatch.

The largest reservation was $5.866, at the pilot relaunch. Wall-clock: about 30 minutes of diagnosis, then
2.9 minutes of pilot and 14.5 minutes of full run.

## Reproduce

```bash
export TMPDIR=/var/tmp/cm-lm
uv run --with torch --with numpy --with scipy --with umap-learn python -I test_lm.py
uv run --with modal python lm_modal.py deploy
uv run --with modal python lm_modal.py launch pilot      # then: status pilot
uv run --with modal python lm_modal.py calibrate && uv run --with modal python lm_modal.py deploy
uv run --with modal python lm_modal.py launch full       # then: status full; fetch
uv run --with torch --with numpy --with scipy python -I lm_schema.py validate data/p160t
```
