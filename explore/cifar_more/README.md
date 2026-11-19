# Coherent-set explorers: VGG-16 / CIFAR-10 and ResNet-18 / CIFAR-100 (exploratory side study)

Owner request (2026-11-13): the ResNet-20 / CIFAR-10 explorer (`../cifar_coherent`, live at
<https://charlesr-w.github.io/CoherentModes/>) for more networks, pared down, one seed, and with more eigenmodes plus
a way to tell whether the higher ones mean anything. Not an `EXPERIMENTS.md` stage. Spec: `SPEC.md` (delta against
`../cifar_coherent/SPEC.md`; GPT plan review approved after one revision). ImageNet was skipped (no affordable
training trajectory). Labels only colour and validate.

## Status — complete, 2026-11-13. Both explorers built, checked in headless Chrome (1440 and 390 px, no console errors) and published

| | A: `r18c100s0` | B: `v16s0` |
|---|---|---|
| Network | ResNet-18, CIFAR variant (kuangliu definition, `nets.py`), own training loop | VGG-16-BN, ModelZoo `vgg16/relu-seed0` recipe, retrained with `zoo.train.train_run` |
| Data | CIFAR-100 (Hugging Face mirror `uoft-cs/cifar100`, pinned revision; see below) | CIFAR-10 (zoo Volume) |
| Training | SGD .1 Nesterov, wd 5e-4, batch 128, 60 epochs cosine, bf16; 419 s on L40S | 64,000 steps, ResNet-20 recipe; 1,003 s on L40S |
| Final test accuracy | **76.5%** | **93.1%** (zoo original 93.17%) |
| Checkpoints (explorer) | 0, 256, 2,340, 9,360, 23,400 | 0, 256, 2,560, 25,600, 64,000 |
| Probe | 5,000 test images, 50 per fine class | the reference 2,000-image probe (asserted equal) |
| Sites (11) | stem, b1..b8 (BasicBlock outputs), pool, logits | c1, c2, c4, c6, c7, c9, c10, c12, c13 (ReLU outputs; all convs except the first of stages 2–5), pool (512-d classifier input), logits |
| Kernel rules | kNN graph k = 15 (default), perplexity 100 | same |
| Eigenfunctions stored | K = 64 | K = 48 |
| SEBA ranks | 2, 4, 10, 20, 30, 40 | 2, 4, 6, 8, 10, 12, 16, 20, 24 |
| Blocks | 990 (all 66 layer pairs × 5 checkpoints, 15 time pairs × 11 sites, × 2 rules) | 990 |
| Validation (`data/<tag>/validation.json`) | 990 blocks pass; minimum SEBA label reconstruction agreement 99.96% | 990 blocks pass; minimum SEBA label reconstruction agreement 99.95% |
| Site | `viewer/r18c100s0/` (310.0 MB) | `viewer/v16s0/` (125.5 MB) |

Published to the laptop with `publish-results` (see the end). Not committed, not pushed.

## Stability horizon (the "is it nonsensical?" answer)

Each block is rebuilt from two overlapping label-free random subsets of the probe (2/3 of the images each, sharing
1/3); `span_k` is the minimum squared cosine of the principal angles between ψ1..ψk of the two rebuilds on the
shared images, computed up to K + 16. Horizon h = the largest k with `span_k` ≥ 0.5. Cuts inside a group of
near-equal eigenvalues are expected to fail; only group boundaries are tested (synthetic checks in `test_cm3.py`).

| | A (CIFAR-100) | B (CIFAR-10) |
|---|---|---|
| median h, all layer blocks (kNN graph / perplexity 100) | 12 / 13 (range 4–31 / 6–80) | 12.5 / 13 (range 4–36 / 6–37) |
| median h, all time blocks | 12 / 12 (range 4–31 / 6–35) | 10 / 11 (range 1–34 / 3–37) |
| full stem→logits block, final checkpoint | 10 / 11 | 10 / 10 |
| single snapshots, final checkpoint (kNN graph) | stem 10, b1 7, b2 6, b3 12, b4 11, b5 12, b6 9, b7 12, b8 4, pool 4, logits 5 | c1 8, c2 8, c4 12, c6 10, c7 13, c9 12, c10 11, c12 23, c13 19, pool 21, logits 13 |

So for B, "more than ten" is mostly not reproducible: the trained network's full block has ten reproducible modes
(nine class-contrast directions plus one), then a slowly decaying continuum of within-class directions. For A the horizons are similarly short (about 10) and shortest in the late layers (4–6, kNN graph): with 100 classes and 50 probe images per class the late spectrum is a smooth continuum (b8→logits: λ1..λ25 from 0.983 to 0.947) with no gap to cut at, although the r = 40 SEBA sets still match fine classes (matched Jaccard 0.52; superclasses 0.35). The one horizon of 80 (A, pool→logits, perplexity 100, final) is a single isolated reproducible cut at the last guard mode after ~65 failures: a large near-degenerate group whose whole span reproduces, not 80 meaningful modes.
The highest horizons (25–34) occur in B's collapsed step-256 deep layers, where most images coincide; they describe
tie structure, not a rich representation.

## Deviations from the brief / spec, and why

- **CIFAR-100 source:** the cs.toronto.edu tarball stalled (~0.5 kB/s from both the VPS and Modal); the pinned
  HF mirror keeps the original order (checked: first test labels 49, 33, 72, 51).
- **Coincident images (new edge case):** VGG's step-256 deep layers (and the diverged 60-step smoke) map hundreds to
  thousands of images to identical activations, so perplexity 100 is unattainable for them. `cm3.perplexity_logkernel`
  gives those rows the β → ∞ limit (uniform over the tied nearest images, distance ≤ 1e-9) and allows 60 bracket
  doublings instead of 40; every other row is bit-identical to the reference (tested). Counts are in
  `meta.perplexity_unattainable`.
- **Stability measure** was redesigned after plan review: the horizon uses leading-span reproducibility with a
  16-mode guard band; the windowed per-index score is kept as descriptive only (it under-reads inside wide groups).
- **Browser precision:** ψ ships as int8 for display (max error 0.0078 of max|ψ|); SEBA labels, amplitudes and
  representative images come from double precision; the browser never reconstructs SEBA.
- **Budget guard:** blocks forecast used 1.25× (not 1.5×) of the pilot-measured unit time, because the remaining
  units were identical in shape to the pilot; the 1.5× forecast would have exceeded the $18.5 admission cap although
  the projected actual total was ~$16.6.
- **Site:** a generalised copy of `../cifar_coherent/site/` (that directory is untouched); tabs Overview, Explorer,
  First look, Measurements, Method. Dropped: second seed, gifs/figures, pixel statistics, loadings, derivatives.

## Spend

Ceiling $20 for both studies. Estimates from per-unit wall time × Modal list prices (A100-40GB $0.000583/s,
L40S $0.000542/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s, as checked for the reference on 2026-11-12), plus
60 s of startup per unit; failed attempts included. Not an invoice.

| item | $ |
|---|---:|
| Training: VGG-16 (1,003 s L40S) + 60-step smoke | 0.94 |
| Training: ResNet-18 / CIFAR-100 (419 s L40S) + 60-step smoke | 0.53 |
| Two CIFAR-100 smoke trainings stalled on the Toronto download, cancelled (~28 min L40S; no logs) | 1.21 |
| VGG smoke measurement (three attempts; records deleted with its scratch) | 1.58 |
| `v16s0` measurement (extraction, 16 block units, UMAP, decodability, validation) | 2.19 |
| `r18c100s0` measurement (incl. pilot; layer units ~1,230 s, time units ~430 s on A100) | 6.88 |
| Aborted one-block profiling run (≤ 15 min A100) | 0.64 |
| **Total** | **13.97** |

`measure.py spend` prints the live ledger (it no longer includes the deleted smoke records). Scratch Grams were
deleted from `cm-cifar-coherent:/work/` after validation; checkpoints remain under `/runs/`.

## Reproduce

```bash
# From this directory; scratch in /var/tmp (the VPS /tmp is a small tmpfs).
export TMPDIR=/var/tmp/cm-more; mkdir -p $TMPDIR
OMP_NUM_THREADS=3 uv run --with torch --with numpy --with scipy python -I test_cm3.py      # add --big for the N=5,000 r=40 test
uv run --with modal python modal_train.py deploy
uv run --with modal python modal_train.py launch vgg        # and: launch c100   (status vgg|c100)
uv run --with modal python measure.py deploy
uv run --with modal python measure.py launch r18c100s0 pilot   # one tag at a time is enforced (max_containers=1)
uv run --with modal python measure.py launch r18c100s0 rest
uv run --with modal python measure.py launch v16s0             # all stages
uv run --with modal python measure.py status <tag> [stage]; uv run --with modal python measure.py spend
uv run --with modal python measure.py fetch <tag>
uv run --with torch --with numpy --with scipy python -I validate3.py data/<tag>
python3 site/content/make_content.py                            # per-tag Overview/Measurements/Method from templates
uv run --with numpy --with pillow python -I site/build_site.py <tag>   # -> viewer/<tag>/
python3 -m http.server 8765 --bind 127.0.0.1 -d viewer/<tag>
```

If a coordinator stage fails, its budget reservation is kept until `measure.py unreserve <tag>` (after checking
its workers have finished). After redeploying, stop the app first (`modal app stop -y cm-cifar-more`): a warm
coordinator container of the previous deployment otherwise receives the next launch with old code (this happened
twice; the smoke's `meta.json` fingerprint was corrected locally because of it).

## Files

`cm3.py` (generalised block record, tie-safe perplexity, split-core stability), `measure.py` (Modal pipeline,
budget ledger), `validate3.py`, `decod.py`, `nets.py`, `c100.py`, `modal_train.py`, `test_cm3.py`, `site/`
(builder, JS, content templates in `site/content/`), `runs/` (launch records). The construction is imported from
`../cifar_coherent` (`coherent.py`, `seba.py`, `sweep_math.py`, `construction.py`, `schema_v2.py`). `seba.py` there
is GPL-3.0-or-later (port of gfroyland/SEBA); see that README's licence note.
