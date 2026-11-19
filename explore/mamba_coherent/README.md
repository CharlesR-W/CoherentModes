# Coherent sets in a small Mamba model, with the whole recurrent state (toy, exploratory)

> **Superseded (2026-11-13).** The report for this study is replaced by the joint report in `explore/ncs_report/` (published as `CoherentModes/ncs-vit-mamba`). The data and code here are still read by that report's build; the page in this directory is kept for reference and is no longer updated.

Spec: `SPEC.md` (delta against `../lm_coherent/SPEC.md` v2.1). Pipeline: `mb_construction.py`, `mb_cells.py`,
`mb_model.py` (U1), `mb_schema.py` (U0, U2, U3, fixtures, validation), `mb_modal.py`, `cross_arch.py`; tests
`test_mb.py`. Reused by import, unchanged: `../lm_coherent/lm_cells.py`, `lm_schema.py` (helpers), `probe.py` (not
needed), and the ResNet modules in `../cifar_coherent/`. Generalised: `../lm_coherent/lm_tree.py` gained
`cell_gram`, an operator provider that accepts a Gram-based cell (vector cells take the previous code path; the
Pythia suite `test_lm.py` still passes, 16/16). Data (untracked): `data/_fixture/`, `data/_fixture_flat/`
(synthetic, MOCKUP only) and `data/m130/` (the run, 9.7 MB).

## Status

**Complete, 2026-11-12.** `data/m130/` is fetched and validates (252 SEBA records, worst label reconstruction
0.998; P1 exact; threshold 3.0); `cross_arch.json` (108 rows) is computed against `../lm_coherent/data/p160t`.
N = 1,000, no items removed.

- **Review before launch** (GPT-6.1 Sol, adversarial): one finding, fixed: C1b's error accumulation used
  `max(0.0, nan) == 0.0`, so a NaN in the resumed trajectory would have read as zero error. `_rel` now fails closed
  (returns inf), with a negative regression test (`test_c1b_fails_closed_on_nan`). The first pilot ran the
  unfixed code, so its trained U1/U2 were invalidated and the pilot rerun with the fix.
- **Pilot gate (rerun) passed:** U0 checks; C1a 4.6e-7 (float64 params and cache, stock HF casts; tol 1e-4),
  float32 hazard 4.2e-5 (max at L24); C1b exactly 0.0 over 64 resumed steps; C2 identical; P3 Gram identity
  4.2e-15; Sinkhorn max row error 1.7e-12; min distance 1.1e-4; no zero-norm items; min PN(psi_1) 68 (point),
  232 (state), 262 (stack); T4 Gram route at the root 4.7e-15.
- **Hypotheses (SPEC 7):** H1 met (T~ L4 > L24 at every t); H2 **not met** (W~(128) - W~(4) at L12..L24 is 0.00,
  0.08, 0.14, 0.17); H3 met (X(maths, poetry) = 0.29 at L16, t = 16); H4 met (whole state W~ = 0.74 at t = 128
  against the best point layer's 0.53); H5 met (init 0.0001 vs trained 0.44); H6 met (T1 root cut: code 1/199,
  poetry 200/0); H7 met (point-point a3 at t = 128: 0.55, 0.37, 0.42, 0.33 at Pythia L6..L12 vs Mamba L12..L24).
- **Cross-architecture:** the two studies' T1 root cuts agree on 927 of 1,000 snippets (matched Jaccard 0.86,
  null about 0.35).

## Runs

| stage | coordinator call | outcome |
|---|---|---|
| pilot 1 | `fc-01M79AZ03K2KE103MNR9JXCXD9` | gate passed, but on the unfixed C1b; invalidated |
| pilot 2 | `fc-01M79BCC64JP9B6DDB13E6G314` | gate passed; U1 2.5 min, U2 2.1 min |
| full | `fc-01M79BPVVQMD3Y6JQPDBXAC84J` | U1/U2 init ok, U3 ok (8.1 min); 15 min wall |

Final ledger: **$0.613** total from the attempts ledger (an upper-bound estimate, not an invoice), against the $6 ceiling.

## Pre-launch record

- **U0 (run locally on CPU, tokenizer only):** passes as implemented (see DEVIATIONS 1). The Mamba tokenizer at the
  pinned revision has the same backend as Pythia's (vocabulary, merges, normaliser, pre-tokeniser, decoder, added
  tokens; only a stored `truncation` setting differs). Max stored id 50,276 < 50,280. Decoding the stored ids
  with clean-up off reproduces all 1,000 texts and all token strings exactly. The SPEC's literal check (re-encoding
  each text gives the stored ids) **fails on 16 items, and fails on the same 16 items under Pythia's own
  tokenizer**: each such snippet ends inside a merge (for example `'\n', '\n'` stored, `'\n\n'` re-encoded). The
  check is about the probe, not about Mamba.
- **Tests (`test_mb.py`, 11/11)**, tiny random Mamba-1 (24 layers, width 16, E = 32, S = 4) on CPU,
  transformers 4.57.6:
  - C1a: float64 parameters and cache, max relative error 1.5e-16 (tolerance 1e-4); float32 4.2e-7 (hazard).
  - C1b: float32, 0.0 (exactly) over the 64 resumed steps, for residuals, conv buffers and SSM states, with column
    0 of the resumed conv buffer set to NaN.
  - The conv-buffer claim holds column by column; a perturbed resumed state does not reproduce the decode; the HF
    init is deterministic under the seed; the whole of U1 runs on 48 items.
  - P3 Gram identity: 6.7e-16. T4 Gram route vs a full rebuild from raw constituent vectors on non-root subsets:
    6.3e-15.
- **Fixtures (MOCKUP), built through the real U2/U3 code path and valid** (`mb_schema.py validate`; 41 min for
  both on this VPS with 3 threads, most of it the 25 AlignedUMAP runs per fixture):
  - `data/_fixture`: 252 SEBA records, worst label reconstruction 0.999; P1 exact (planted Upsilon min 8.26,
    structureless max 1.88, threshold 3.0 by the fixed rule); P2 passes; P3 passes (Gram identity 1.1e-15,
    linearity <= 2.2e-16); T4 Gram route at the root 1.3e-14; U2 `trained` took 136 s.
  - `data/_fixture_flat`: valid, 252 records, worst 0.999; P3 Gram identity 1.4e-15; T4 1.9e-14.
  - `cross_arch.py` on the two studies' planted fixtures: 108 rows, valid; T1 root cuts cross-tabulate as
    [[500, 0], [0, 500]] (both fixtures plant the same seeded leaves). It refuses the fixtures without `--fixture`.
- **Modal:** app `cm-mamba-coherent` deployed (image builds; 5 functions); `budget` reads an empty ledger: $0 spent,
  reservation at launch $3.58.

## The HF cache (transformers 4.57.6, checked against the installed source)

`MambaMixer.slow_forward` takes the decode branch when `cache_position` has one element (`shape[0] !=
conv_kernel`). `MambaCache.update_conv_state` then rolls the (B, E, 4) buffer left by one column and writes the new
conv input at column `clamp(cache_position, 0, 3)`; the conv output is `sum(buffer * w)`. So after reading sequence
index u, the buffer holds [x_{u-3}, x_{u-2}, x_{u-1}, x_u], zeros before the start. The next step's roll discards
column 0, so **the carried state is `conv_states[..., 1:]`**, as the SPEC says. Two consequences:

- The true index u < 3 as `cache_position` would write the input into column u, not the last column: a wrong conv.
  HF's own `generate` never does this (it prefills with `arange(0, 4)`, then decodes with 4, 5, ...). We decode
  every token, starting from an empty cache, with `cache_position = max(u, 3)` (`MB.cache_position`). Every value
  >= 3 is equivalent.
- C1b fills column 0 of the resumed buffer with NaN; NaN would poison every later value if column 0 were read.

The SSM state is updated in place (`ssm_states[layer].copy_(...)`). `mb_model.assert_slow_path` refuses to run
if `mamba-ssm`, `causal-conv1d` or the `kernels` hub package is importable (the Modal image installs none of them).

## Projected spend

Rates: modal.com/pricing as recorded on 2026-11-12 (L40S $0.000542/s; CPU $0.0000131 per core-second; memory
$0.00000222 per GiB-second). The figures are estimates, with 90 s container overhead per attempt.

| unit | hardware | estimate | worst case (timeout x rate) |
|---|---|---|---|
| U0 | 2 cores, 4 GiB | ~$0.01 | $0.03 (900 s) |
| U1 `trained` | L40S, 4 cores, 32 GiB | ~$0.25 (decode A and S, 288 float64 Grams per position, 2.3 GB raw Grams, C1a float64 decode + prefill) | $1.00 (1,500 s) |
| U1 `init` | same | ~$0.12 | $1.00 |
| U2 `trained` | 8 cores, 32 GiB | ~$0.06 | $0.42 (2,400 s) |
| U2 `init` | same | ~$0.03 | $0.42 |
| U3 | 8 cores, 32 GiB | ~$0.25 (25 AlignedUMAP runs) | $0.63 (3,600 s) |
| coordinators | 0.25 core, 1 GiB | ~$0.02 | $0.08 each |

Projected total **about $0.75** (SPEC: about $1.5); worst-case reservation at launch $3.58, under the $6 ceiling
with room for a full U1 retry. After the pilot, `calibrate` sets timeouts to 2x the pilot durations.

## Reproduce

```bash
export TMPDIR=/var/tmp/cm-mb
uv run --with torch --with numpy --with scipy --with transformers==4.57.6 python -I test_mb.py
uv run --with torch --with numpy --with scipy --with umap-learn python -I mb_schema.py fixture data/_fixture --threads=3
uv run --with torch --with numpy --with scipy --with umap-learn python -I mb_schema.py fixture data/_fixture_flat --flat --threads=3
uv run --with torch --with numpy --with scipy python -I mb_schema.py validate data/_fixture
uv run --with torch --with numpy --with scipy --with transformers==4.57.6 python -I mb_schema.py u0 \
    ../lm_coherent/data/p160t/probe.json /var/tmp/cm-mb/u0          # U0 locally (tokenizers only)
uv run --with modal python mb_modal.py deploy
uv run --with modal python mb_modal.py budget                 # spends nothing
uv run --with modal python mb_modal.py launch pilot           # then: status pilot
uv run --with modal python mb_modal.py calibrate && uv run --with modal python mb_modal.py deploy
uv run --with modal python mb_modal.py launch full            # then: status full; fetch
uv run --with torch --with numpy --with scipy python -I mb_schema.py validate data/m130
uv run --with torch --with numpy --with scipy python -I cross_arch.py ../lm_coherent/data/p160t data/m130
```

## DEVIATIONS (from SPEC.md, smallest sensible choices)

1. **U0's check is reformulated.** The literal check (Mamba's tokenizer re-encodes every stored text to the stored
   ids) is ill-posed: it fails on 16 of 1,000 items under the Pythia tokenizer that made the probe, because the
   stored text is `decode(ids)` and a snippet can end inside a merge. What the item correspondence needs is that
   both models read the same ids with the same meaning. U0 therefore asserts: identical tokenizer backends (only
   truncation/padding settings may differ), every id < 50,280 and < len(tokenizer), exact decode of every text and
   token string with clean-up off, and that Mamba's re-encoding failures equal Pythia's. The literal result is
   recorded too (`u0.json`: 16 failures, identical under both tokenizers). Also noted: the Mamba tokenizer's config
   has `clean_up_tokenization_spaces=True` (Pythia's: False); with it, default decoding changes 96 texts. Anything
   that decodes ids for display must pass `clean_up_tokenization_spaces=False`, or use `probe.json`'s stored text.
2. **`cache_position` is `max(u, 3)`, not u**, for the reason in "The HF cache" above.
3. **C1b puts NaN in conv column 0** of the fresh cache (the SPEC only says the retained columns go in their slots).
   C1b runs, and is asserted, for both checkpoints; the gate reads `trained`. Residuals are compared at all 24
   layers.
4. **C1a's observed magnitude is far below the SPEC's expectation.** On the tiny model, C1a is 1.5e-16 in float64
   parameters and cache: decode and prefill apply the same float32 casts to the same values, and differ only in
   the conv's summation order. The 1e-4 gate stands as written. The float32 hazard is reported per layer.
5. **Zero-norm scan:** the point view at every position u >= 1 at the sampled layers (as in the Pythia study);
   constituents (all 24 layers, both parts) at the sampled positions, where they enter a view.
6. **P3 on real data:** the Gram identity is checked in U1 at t = 128: state at all 6 layers, stack at L4 and L24
   (L24 is the explicit concatenation of all 48 constituents, 1,000 x 700,416 in float64, 5.6 GB on the GPU). The
   stacks at L8..L20 are covered by the tests and the fixtures, which check all six. Pythia's linearity check runs
   on the point reading block at L16 and on the stack reading block at L24.
7. **New gate checks:** U0 passes, and **the T4 Gram route at the root** equals the stored stack Grams (the raw-Gram
   route vs the vector route, to 1e-8; a check that the 2.3 GB of raw Grams are right before trees use them).
8. **`cross_arch` goes in its own file** (`data/m130/cross_arch.json`, same row fields plus `pair`,
   `pythia_layer`, `jacc_ceiling`), not inside `compare.json`, because it is made locally after the run from both
   studies' public files. It also holds the T1 root-cut cross-tabulation for F12. `mb_schema.validate` checks it when present.
9. **`compare.json` row schema:** `checkpoint` replaces `step`; rows carry `view` (real_vs_S, to_final) and `part`
   (state_parts). `to_final` includes the trained-vs-trained reference rows (the Pythia convention). `state_parts`
   rows also carry a_6 and the matched Jaccard. The pairwise kinds and real_vs_S are computed on `trained` only.
10. **Stored beyond the SPEC's list:** the constituent cosine Grams at the 6 sampled layers (`parts_trained.npy`,
    576 MB, work volume only) for the `state_parts` diagnostic.
11. **Layouts:** `umap.npz` adds `grid_state` and `grid_stack` (AlignedUMAP of the Grams' principal coordinates) for
    the explorer's view switch; `strip` is [layer, 2 checkpoints].
12. **Fixture:** the demotion (P2's page illustration) sits at L8, index-matched to the Pythia fixture's L4.
    Synthetic constituents have dimensions 16 (c) and 48 (z), log-normal item norms and large non-zero means.
    The flat fixture's constituents are noise.
13. **Pythia files reused by import, not forked:** `lm_schema.Probe`, the cell and block helpers, `p2_check`,
    `calibrate` (P1), the planted generators and the record validators. `p3_check`'s block label is overwritten.
