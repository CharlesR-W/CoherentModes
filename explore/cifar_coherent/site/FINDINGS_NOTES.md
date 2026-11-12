# Findings notes (ANA, phase 1a), raw material for the Findings writer

All numbers are facts in `assets/questions.json["facts"]`, cited by build name (`{{fact:q.<name>}}`). Definitions:
docstrings of `site/analysis.py` and `site/analysis_findings.py`. Default kernel rule (kNN graph, k = 15; `m=0`)
unless noted. "Final" = step 64,000 (`t=7`). Labels, losses and gradients only describe the sets here.

Live-chart objects (`data-src="assets/questions.json#..."`):
- `q_va.by_mode.0.lines` (r = 2 vs vehicles/animals, null band), `q_va.by_mode.0.lines_norm_excess`,
  `q_va.by_mode.0.score` / `.norm_excess` / `.agree` (site x checkpoint), `q_va.class_placement.in_matched` / `.unassigned`
  (site x class, checkpoint slider), `q_va.best_bipartition.va_rank`, `q_va.pixel_baseline.score`.
- `q_classmix.cells.<r>.<site>` (set x class counts, checkpoint slider); `q_classmix.named.<key>` with keys
  `layer_t7_b7-pool_r{2,3,4}`, `time_b9_4-7_r{2,3,4}`, `time_b9_0-7_r{2,3,4}`, `layer_t{0..7}_b9_r2`.
- `q_imgstats.heat_maxabs_psi` (+ `_wc`), `.heat_r2_psi`, `.heat_psi.<k>`, `.heat_seba.<r>`, `.heat_blocks_psi`,
  `.heat_blocks_seba.<r>`, `.lines_maxabs_psi.<stat>`, `.class_contrast`.

## Finding 1: four late-layer sets at the end of training (b7 to pool). Holds.

**Claim (observed).** b7..pool, final, r = 4: four sets plus `q.f1.b7_pool.final.r4.n_unassigned` unassigned.
Class make-up (validation): set J size `q.f1.b7_pool.final.r4.setJ.size`, classes `...setJ.top_classes`, vehicle
fraction `...setJ.vehicle_frac` (J = 1..4; now frog/cat/dog/bird; automobile+truck; ship+airplane; horse+most deer).
Spectrum `q.f1.b7_pool.final.r4.lambda1`..`lambda4`; gaps λ3−λ4 `...gap_after_lambda3`, λ4−λ5 `...gap_after_lambda4`.
SEBA retention over the block `q.f1.b7_pool.final.r4.retention_min`..`retention_max` (per set `...setJ.retention_*`).
vs classes `q.f1.b7_pool.final.r4.vs_classes` against ceiling `...vs_classes_ceiling`: as class-aligned as sizes allow.
**Robustness.** Same block under perplexity 100 / 300: `q.f1.b7_pool.final.r4.vs_rule1` / `vs_rule2`; kNN scale
`vs_rule3`; half-mass `vs_rule4` (the failing rule). vs b9 late time block `...vs_b9_time_late`; vs b9 alone
`...vs_b9_alone`. The b9 time block's sets have the same make-up (`q.f2.b9_time.step2560-64000.r4.setJ.top_classes`).
**Links.** `q.f1.b7_pool.final.r4.hash` (= `#explorer?fam=layer&t=7&a=7&b=10&m=0&c=seba&r=4`), and with `c=class`.
Table: `q_classmix.named.layer_t7_b7-pool_r4`.
**Alternatives.** (a) No spectral gap singles out r = 4 (λ4−λ5 is the smallest gap); set 4 (horse+deer) is the least
canonical; see r = 3 (`q.f1.b7_pool.final.r3.*`). (b) The block includes pool, one linear map from the logits:
the sets may be the class geometry propagated back; agreement with b9 alone is lower than with the b9 time block.
(c) Class make-up is description; within-class pixel correlations at b9..logits are near the null (below).
**Checks.** Split-half resampling; second seed's sets; r = 3 / 5; `images` grids per set; per-set agreement across rules.

## Finding 2: when the 2-set split becomes vehicles vs animals. Holds, with two corrections.

**Correction 1: animals vs automobile, truck, ship; airplanes left out.** b9, final: airplanes in the vehicle set
`q.va.b9.step64000.airplane_in_vehicle_set`, unassigned `q.va.b9.step64000.airplane_unassigned`; every other class
≥ `q.va.b9.step64000.min_class_in_matched_excl_airplane` in its super-class set. Worst-placed class from b7 on:
`q.va.b7.final.worst_class`, `q.va.b9.final.worst_class`. Earlier: `q.va.b9.step1024.airplane_in_vehicle_set`.
Same in `q_classmix.named.layer_t7_b7-pool_r2` (53/200 airplanes in the vehicle set, 143 unassigned) and
`q_classmix.named.time_b9_4-7_r2`.
**Correction 2: "beats the null" is met at step 0 everywhere, so it cannot date the split.** b9 step 0:
`q.va.b9.step0.score` vs null max `q.va.b9.step0.null_max` (onset_step = 0 at every site), but vehicles/animals ranks
only `q.va.b9.step0.va_rank` of 511 class splits, norm excess `q.va.b9.step0.norm_excess`, and a same-size split by
one pixel statistic does better: `q.va.b9.step0.pixel_best` (`q.va.b9.step0.pixel_best_stat`; vehicles are
`q.img.vehicle_contrast.mean_b` SD bluer). Step-0 ψ1 is mostly linear in pixel statistics (R² `q.img.step0.b9.r2_psi1`).
**Claim (observed + reading).** b9: half the size-allowed excess, kept to the end, from step `q.va.b9.half_step`
(score `q.va.b9.step64.score`, vehicles/animals already rank `q.va.b9.step64.va_rank`; val acc `q.train.step64.val_acc`);
80 % of final excess from `q.va.b9.most_step` (score `q.va.b9.step1024.score`); final `q.va.b9.final.score`, ceiling
`q.va.b9.final.ceiling`, agreement on assigned `q.va.b9.final.agree`. b7..logits: half_step 64; b3..b5
`q.va.b3.half_step`; stem, b1, b2 never (`q.va.stem.half_step`), yet at the end vehicles/animals is the
best-matching class split even at stem (`q.va.stem.final.va_rank`, score `q.va.stem.final.score`, agreement
`q.va.stem.final.agree`). most_step per site `q.va.<site>.most_step`. Time block b9 r = 2: all
`q.va.time.b9.step0-64000.score`, late `q.va.time.b9.step2560-64000.score` (null max `...null_max`). Perplexity-100
gives the same b9 milestones (`q_va.by_mode.1.half_step` / `.most_step`). Reading: deep layers split vehicle-like
from step 64, mostly in place by 1,024; early layers move toward it but stay weak.
**Links.** `#explorer?fam=layer&t=0&a=9&b=9&m=0&c=seba&r=2` (step 0), `t=1` (64), `t=3` (1,024), `t=7`; same with
`c=class`; `#explorer?fam=time&t=9&a=4&b=7&m=0&c=seba&r=2`. Charts: `q_va.by_mode.0.lines_norm_excess`,
`q_classmix.cells.2.b9`, `q_va.class_placement.in_matched`.
**Alternatives.** (a) Colour/background (blue sky/water) carries the split early: pixel baseline covers step 0 but not
step 64 at b9. (b) Matched Jaccard depends on sizes and unassigned counts: read norm_excess and agree too.
(c) Eight checkpoints cannot locate onsets within 0–64 or 256–1,024.
**Checks.** Denser checkpoints (73 exist); split-half; are airplanes unassigned because they sit between sets?

## Finding 4: persistence across depth. Holds descriptively; persistence is not by itself learned.

**Claim (observed).** Final, r = 4: adjacent sites agree `q.carry.r4.final.adjacent_min`–`q.carry.r4.final.adjacent_max`
(null upper edge ≤ `q.carry.r4.final.adjacent_null_hi_max`). Redrawn most at `q.carry.r4.final.adjacent_min_pair`
(`q.carry.r4.final.b2_b3`), then stem→b1 `q.carry.r4.final.stem_b1`, b6→b7 `q.carry.r4.final.b6_b7`, b8→b9
`q.carry.r4.final.b8_b9`; kept most at `q.carry.r4.final.adjacent_max_pair`. Long range: stem–logits
`q.carry.r4.final.stem_logits`, b9–logits `q.carry.r4.final.b9_logits`. The old "runs" depend on the cut
(four lowest cuts give {stem}, {b1,b2}, {b3–b6}, {b7,b8}, {b9,pool,logits}); say "where the grouping changes most".
**Does not hold as learned persistence.** Mean adjacent agreement `q.carry.r4.step0.mean_adjacent` at step 0 vs
`q.carry.r4.step64000.mean_adjacent` final (minimum `q.carry.r4.mean_adjacent_min` at `q.carry.r4.mean_adjacent_min_step`).
At step 0 every site's ψ1 is mostly luminance (R² `q.img.step0.stem.r2_psi1` stem), passed through an untrained net.
**Links.** `#explorer?fam=layer&t=7&a=2&b=2&m=0&c=seba&r=4`, `a=3&b=3`, step 0 `#explorer?fam=layer&t=0&a=5&b=5&m=0&c=seba&r=4`.
Chart `q_carry.0_4.matrix`.
**Alternatives.** Residual shortcuts copy inputs; stage boundaries (projection shortcuts) are sharper tests:
b6→b7 is a low point, b3→b4 (`q.carry.r4.final.b3_b4`) is not.
**Checks.** r = 2 (`q.carry.r2.*`), perplexity rules; the step-0 network as the reference null.

## Finding 5: single-cell vs block split. True, but half of it holds by construction.

**By construction.** Block ψ1 maximises block-mean retention; single-cell ψ1 maximises own-cell retention, so the
block ψ1 always retains less at the start site (`q.f5.final.n_sites_block_ret_own_below_static` of `q.f5.final.n_sites`).
**Informative (observed).** b1 vs b1..logits: static ψ1 retention `q.f5.b1_vs_b1-logits.final.static_ret_own` at b1,
`...static_ret_b9` at b9 (block mean `...static_ret_block_mean`); block ψ1 λ1 `...block_lambda1`, retention
`...block_ret_own` at b1, `...block_ret_pool` at pool. |cos| `...abs_cos_psi1`; ψ1..ψ3 span overlap `...span3_mean_cos2`.
The block ψ1 is closest to the static ψ1 of `...block_psi1_best_static_site` (|cos| `...block_psi1_best_static_abs_cos`),
a middle layer. |cos| rises with the start site (b5 `q.f5.b5_vs_b5-logits.final.abs_cos_psi1`, b7
`q.f5.b7_vs_b7-logits.final.abs_cos_psi1`, median `q.f5.final.median_abs_cos_psi1`). Detail: `findings_support.f5`.
Suggested phrasing: averaging over b1..logits trades retention at b1 for the split b6 to pool keep, which is b6's own.
**Links.** `q.f5.b1_vs_b1-logits.final.static_hash`, `...block_hash`, b6: `#explorer?fam=layer&t=7&a=6&b=6&m=0&c=eig&r=4&k=1`.
**Alternatives.** Near-degenerate λ1/λ2 make "ψ1" non-canonical (span overlap guards partly).
**Checks.** Product operator (`avgprod` data); equal-length blocks from each start site.

## Finding 6: gradient coherence. Does not hold as a negative result.

**Old reading.** b9, time block all, r = 4: sets' within−between gradient cosine `q.grad.r4.b9.all.first.sets_diff` →
`q.grad.r4.b9.all.final.sets_diff`, above null at `q.grad.r4.b9.all.n_above_null` checkpoints (null edge ≤
`q.grad.r4.b9.all.null_hi_max`); classes `q.grad.classes.first.diff` → `q.grad.classes.final.diff`; sets at most
`q.grad.r4.all.final.ratio_to_classes_max` of classes (r = 2 `q.grad.r2.all.final.ratio_to_classes_max`).
**Fair reference.** Cross-entropy gradients carry the one-hot label, so any class-merging partition loses coherence.
Class-union partition (each class wholly to the set holding most of it; r = 2 `q.grad.r2.b9.all.class_union_groups`,
r = 4 `q.grad.r4.b9.all.class_union_groups`), final: r = 2 union `q.grad.r2.b9.all.final.class_union_diff`, sets
`q.grad.r2.b9.all.final.sets_diff`, ratio `q.grad.r2.b9.all.final.sets_over_class_union` (late
`q.grad.r2.b9.late.final.sets_over_class_union`); r = 4 union `q.grad.r4.b9.all.final.class_union_diff`, ratio
`q.grad.r4.b9.all.final.sets_over_class_union` (late `q.grad.r4.b9.late.final.sets_over_class_union`).
Claim (reading): the sets are about as gradient-coherent as the class groupings they resemble; the gap to the 10
classes is mostly the cost of merging classes under a one-hot loss.
**Links.** `#explorer?fam=time&t=9&a=0&b=7&m=0&c=seba&r=4`, `...&c=gcos`. Chart `q_grad.0_4_9_all.lines`.
**Checks.** Class-union reference at all sites; a within-class null; a label-free loss's gradients.

## Image statistics: raw material (no readings)

Thresholds: block familywise median `q.img.thr_family_median`; whole table `q.img.thr_global`; within-class
`q.img.thr_family_wc_median`; R² null ≤ `q.img.r2_null_hi_max`. Mean R/G/luminance inter-correlate 0.86–0.99, the
border−centre family 0.85–0.99: a hit on one is a hit on the family.
- Step 0, all sites: ψ1 vs luminance up to `q.img.luminance.step0.max_abs_corr_psi` (wc
  `q.img.luminance.step0.max_abs_corr_wc_psi`); R² ψ1 `q.img.step0.stem.r2_psi1` (stem) to 0.66 (pool). Step-0
  layer blocks (`q_imgstats.blocks`, "… @ step 0"): ψ3 vs border−centre luminance ≈ −0.73; r = 2 SEBA vs luminance ≈ ±0.7.
- Final, stem–b3: mean B strongest, `q.img.mean_b.final.max_abs_corr_psi` at `q.img.mean_b.final.max_abs_corr_psi.where`
  (wc `q.img.mean_b.final.max_abs_corr_wc_psi`); R² ψ1 `q.img.final.stem.r2_psi1` (stem), `q.img.final.b5.r2_psi1`
  (b5); border−centre ≈ 0.5 at b1/b3; saturation ≈ 0.5 at stem.
- Time blocks stem / b3, steps 2,560–64,000: ψ1 vs mean B ≈ 0.83 / 0.74; stem ψ2 vs border−centre G ≈ 0.50;
  r = 4 SEBA vectors split along mean G, B, R, saturation (wc ≈ 0.3–0.6).
- Final, b7/b8: high-frequency share is the largest wc correlation (`q.img.hf_frac.final.max_abs_corr_wc_psi` at
  `q.img.hf_frac.final.max_abs_corr_wc_psi.where`); b7..pool ψ4 vs hf ≈ −0.40 (wc −0.27); r = 4 set 1 ≈ +0.20 wc.
- Final, b9–logits: mean-B correlations ≈ 0.3–0.4 match class make-up alone (`q.img.vehicle_pointbiserial.mean_b`);
  within class ≈ 0.1, near null; R² ψ1 `q.img.final.b9.r2_psi1` (b9), `q.img.final.logits.r2_psi1` (logits).
- `q_imgstats.class_contrast`: airplane/ship bluer and brighter; frog/deer less blue, more high-frequency;
  truck/automobile higher contrast. Check any colour reading within class (`_wc`) first.
Charts: `q_imgstats.heat_maxabs_psi` (+`_wc`), `.heat_r2_psi`, `.lines_maxabs_psi.{mean_b,luminance,hf_frac}`,
`.heat_blocks_psi`. Per-image values: `q_imgstats.per_image.<stat>`.
