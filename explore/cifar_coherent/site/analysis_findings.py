"""Class make-up, vehicle/animal split, image statistics and facts (called by site/analysis.py).

Nothing here enters the construction. Every quantity is a closed-form or descriptive statistic of already-computed
hard SEBA sets, stored eigenfunctions psi_k, reconstructed SEBA vectors, CIFAR class labels (validation only) and
probe-image pixels (description only). Notation as in the docstring of site/analysis.py; in addition:

Blocks and functions
--------------------
psi_k (k = 1..8) of a block = `vecs[m, pair, k-1] * scale[m, pair, k-1] / 32767` (unit vectors; psi_0 = 1/sqrt(N) is
implicit). SEBA vectors of rank r: v = [psi_0, psi_1, .., psi_{r-1}] (N x r), z = v @ seba_rot[m, pair, ri, :r, :r],
s = sign(z) max(|z| - 0.99/sqrt(N), 0), columns scaled to unit norm, sign flipped so each column sums >= 0, scaled to
max 1, columns permuted by seba_perm, negatives clipped to 0 (seba.py `seba_reconstruct`). The hard sets are the
stored `seba_label` (argmax of s, -1 where all are 0); `run` checks that argmax of the reconstruction agrees with
them on >= 99 % of images in every block it uses.

1. Class make-up (q_classmix)
-----------------------------
For the default kernel rule m = meta.default_mode (other rules: `--classmix-modes`), every stored layer block (every
checkpoint t, every pair (a, b) of meta.layer_pairs) and every stored time block (every site s, every pair (c, d) of
meta.time_pairs), and r in {2, 3, 4}:
    counts[j][c] = #{images in SEBA set j with CIFAR class c}, j = 0..r-1 (reliability order), and row r =
                   #{unassigned images with class c}.
Stored as nested integer lists: layer[str(r)][t][pair][row][class], time[str(r)][s][pair][row][class]
(pair indices as in meta.layer_pairs / meta.time_pairs, which are copied in). For live charts: cells[str(r)][site] =
heatmap {rows: "set 1".."set r", "unassigned"; cols: classes; z[t][row][class]; slider over checkpoints;
zrange [0, 200] (200 = images per class)} for the single-cell blocks, and named[...] heatmaps for the blocks
cited in FINDINGS_NOTES.md.

2. Vehicle/animal split at r = 2 (q_va)
---------------------------------------
V = {airplane, automobile, ship, truck} (800 images), A = the other six classes (1,200). For every kernel rule m,
site s, checkpoint t, take the r = 2 SEBA sets of the single-cell block (t, s) and the 2-set labeling (V, A):
    score(t, s)   = matched Jaccard (Statistic 1 of analysis.py) of the two SEBA sets against (V, A). For 2 x 2 the
                    Hungarian match is the better of the two pairings; score in [0, 1].
    null          = the same score after permuting the SEBA labeling (the shared bank of 200 permutations, seed
                    20261112; set sizes and the unassigned count kept). Stored: mean mu0, 2.5 / 97.5 percentiles,
                    and the maximum over the 200 draws.
    ceiling       = size ceiling (Statistic 1): the largest matched Jaccard any 2 sets of these sizes could reach.
    excess        = score - mu0;  norm_excess = (score - mu0) / (ceiling - mu0)  (1 = as aligned as sizes allow).
    agree         = under the matching, the fraction of *assigned* images whose set is matched to their own
                    super-class (V or A); unassigned images are excluded (their number is stored as n_unassigned).
    veh_frac[j]   = fraction of vehicles among the members of set j.
Milestones per site (kernel rule m, single-cell blocks only):
    onset_step    = the first checkpoint t at which score(t, s) exceeds the maximum of its 200 null draws
                    (one-sided permutation p < 1/201 at that checkpoint) **and** does so at every later checkpoint.
                    null if the final checkpoint does not exceed its null maximum.
    most_step     = the first checkpoint t such that excess(t', s) >= 0.8 * excess(T-1, s) at every t' >= t
                    ("reaches 80 % of its final excess over the null mean and keeps it"). null if the final
                    checkpoint is not above its null maximum.
    first_beat_step = the first checkpoint exceeding its null maximum, without the "stays" requirement.
    half_step     = the first checkpoint t such that norm_excess(t', s) >= 0.5 at every t' >= t (an effect-size
                    milestone: at least half of the excess over the null mean that the set sizes allow).
    On this network the null-maximum criterion is already met at step 0 at every site, so onset_step is 0
    everywhere; half_step and most_step are the informative milestones.
Per-class placement: for every cell and class c, in_matched[c] = fraction of class c's 200 images in the SEBA set
matched to c's super-class, unassigned[c] = fraction unassigned (stored as heatmaps with a checkpoint slider).
Pixel baseline (a closed-form comparison, not part of the method): for each cell, each image statistic of
section 3 and each sign, the split "top n_1 images by the signed statistic -> set 1, bottom n_2 -> set 2, the rest
unassigned" (n_j = the SEBA set sizes of that cell) is scored against (V, A) in the same way; pixel_best = the
largest of these 24 scores (chosen with the labels, so it is an upper reference for single-statistic splits of
these sizes), pixel_best_stat names it.
Best class bipartition (rule m = default only): over all 511 splits {C, not C} of the 10 classes, the one whose
2-set labeling has the largest matched Jaccard with the r = 2 SEBA sets ("best"), its score, and the rank of V/A
among the 511 (1 = V/A is the best-matching split of the classes). The best split is stored as the class names on
the side matched to SEBA set 1.
The same score, null band and ceiling are stored for every time block at r = 2 (rule m only).

3. Image statistics (q_imgstats)
--------------------------------
Pixels: `probe.npz['images']` (N x 32 x 32 x 3, uint8, unnormalised CIFAR-10 test images), divided by 255. Per image:
    mean_r, mean_g, mean_b  mean of each channel over the 1,024 pixels;
    luminance               mean of Y = 0.299 R + 0.587 G + 0.114 B (Rec. 601 luma of the gamma-encoded values);
    contrast                standard deviation of Y over the pixels (RMS contrast);
    saturation              mean over pixels of HSV saturation (max - min) / max of (R, G, B) (0 where max = 0);
    bc_r, bc_g, bc_b, bc_lum  border-ring mean minus centre mean of R, G, B and Y; the border ring is the 448
                            pixels within 4 px of the edge, the centre the central 16 x 16 pixels (a background
                            proxy: positive bc_b = border bluer than centre);
    bc_dist                 Euclidean norm of (bc_r, bc_g, bc_b);
    hf_frac                 high-frequency energy: with Y0 = Y - mean(Y) and F its 2-D DFT, the share of
                            sum |F|^2 over frequencies of radius > 8 cycles per image (half the Nyquist radius 16)
                            in sum |F|^2 over all non-zero frequencies (0 if Y is constant).
Correlations: for each block in the list below, functions f = psi_1..psi_8, then the SEBA vectors of r = 2, 3, 4
(set 1..r each; 17 functions). corr[f][stat] = Pearson correlation over the N probe images (signs of psi_k are the
stored convention, largest-magnitude entry positive; only |corr| is sign-free).
Set contrasts: for the hard sets j of r = 2, 3, 4, contrast[j][stat] = mean of the standardised statistic over set j
minus its mean over all other images (including unassigned), in units of the statistic's SD over the N images.
Linear fit (a closed-form least-squares fit, called a fit, never part of the construction): r2_psi[k] = R^2 of
psi_k regressed on the 12 standardised statistics plus an intercept, i.e. ||Q^T psi_k||^2 / ||psi_k - mean||^2 with
Q an orthonormal basis of the column span of the centred statistics. The 12 statistics have rank 10 (luminance is
linear in mean R, G, B; bc_lum in bc_r, bc_g, bc_b), so Q is taken from a rank-revealing SVD: the left singular
vectors whose singular value exceeds 1e-8 times the largest (stored as img_rank). Its null uses the same permutation
bank (rows of Q permuted, same retained rank); stored: r2_null_hi = 95th percentile, pooled over k = 1..8, per block.
Expected null mean ~ img_rank / N = 0.005. Every block's R^2 is asserted equal (to 1e-8) to numpy lstsq's.
Class contrasts (labels describe the statistics only): class_contrast[c][stat] = mean of the standardised statistic
over class c minus over the other classes; vehicle_contrast[stat] = mean over V minus mean over A.
Within-class correlation (labels describe only): corr_wc[f][stat] = Pearson correlation after subtracting, from
both the function and the statistic, their mean over the image's CIFAR class; it asks whether f tracks the
statistic among images of the same class, which class make-up alone cannot produce. Its null permutes images
within each class (a second bank of 1,000, numpy default_rng(20261112 + 2)); stored per block: thr_family_wc
(95th percentile of the max |null corr_wc| over the 17 x 12 entries) and thr_family_psi_wc[stat].
Null: one bank of 1,000 permutations of the images (numpy default_rng(20261112 + 1)) is applied to the statistics,
the same draws for every block (exact for each block on its own). Stored per block: thr_point = 95th percentile of
|null corr| pooled over all 17 x 12 entries (a pointwise two-sided 5 % level), thr_family = 95th percentile over
draws of the maximum |null corr| over the block's 17 x 12 entries (a familywise 5 % level within the block),
thr_family_psi[stat] = 95th percentile of max_k |null corr(stat, psi_k)| over k = 1..8, and the same pair of
thresholds for the contrasts (cthr_point, cthr_family). thr_global = 95th percentile over draws of the maximum
|null corr| over every block and entry (familywise over the whole table).
Blocks: every single-cell block (t, s) of rule m; layer blocks stem..logits, b1..logits, b5..pool and b7..pool at
every checkpoint; time blocks over all checkpoints (0..T-1) and over the late half (T//2..T-1) at every site.
JSON (per_image to 4 decimals, everything else in q_imgstats to 3):
    stats, stat_labels, definitions; per_image {stat: [N values]};
    cells {"corr": [t][s][f][stat], "contrast": {r: [t][s][j][stat]}, thr_point, thr_family: [t][s],
           r2_psi: [t][s][k], r2_null_hi: [t][s], corr_wc: [t][s][f][stat], thr_family_wc: [t][s],
           thr_family_psi_wc: [t][s][stat],
           thr_family_psi: [t][s][stat], cthr_point, cthr_family: [t][s]};
    blocks [{name, fam, t, a, b, hash, corr [f][stat], contrast {r: [j][stat]}, thr_*}], thr_global;
    heat_maxabs_psi_wc (as heat_maxabs_psi, within-class correlations);
    heat_r2_psi (rows psi_1..psi_8, cols sites, slider checkpoints, z = R^2 of the fit);
    class_contrast, vehicle_contrast;
    heatmaps for live charts: heat_maxabs_psi (rows stats, cols sites, slider checkpoints, z = max_k |corr|),
        heat_psi[k] (k = 1..8; signed, diverging), heat_seba[r] (cols "site set j"; signed, diverging),
        heat_blocks_psi / heat_blocks_seba[r] (slider over the multi-cell blocks);
    lines_maxabs_psi[stat]: x = steps (symlog), one series per site, y = max_k |corr(stat, psi_k)|; one shared
        refline at the largest thr_family_psi[stat] over all cells (the per-cell values differ by < 0.01; they are
        in cells.thr_family_psi).

4. Facts (questions.json["facts"])
----------------------------------
findings_support.f5[site a] (last checkpoint, rule m): the single-cell psi_1 of site a against psi_1 of the layer
block a..logits: |cos| of the two unit vectors, mean cos^2 of the principal angles between their psi_1..psi_3 spans,
retention of each across all 12 sites (`retention_psi`), and |cos| of the block psi_1 with every site's own
single-cell psi_1 (block_psi1_vs_static_psi1).
Step-0 facts: img.step0.<site>.corr2_psi1_luminance = corr(psi_1, luminance)^2 (luminance alone, unlike r2_psi1,
which is the fit on all 12 statistics); step0.psi1_abs_cos_with_stem.<site> = |cos| between stem's and the site's
single-cell psi_1 at step 0.
Flat {name: formatted string}, named <topic>.<block or site>.<quantity>; build_site.py exposes them as q.<name>.
Steps are formatted with thousands separators, scores and correlations with 2 decimals, eigenvalues with 3.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

RS = (2, 3, 4)
VEHICLES = ('airplane', 'automobile', 'ship', 'truck')
N_PERM_IMG = 1000
STATS = ['mean_r', 'mean_g', 'mean_b', 'luminance', 'contrast', 'saturation', 'bc_r', 'bc_g', 'bc_b', 'bc_lum',
         'bc_dist', 'hf_frac']
STAT_LABELS = {'mean_r': 'mean R', 'mean_g': 'mean G', 'mean_b': 'mean B', 'luminance': 'luminance Y',
               'contrast': 'RMS contrast of Y', 'saturation': 'HSV saturation', 'bc_r': 'border − centre R',
               'bc_g': 'border − centre G', 'bc_b': 'border − centre B', 'bc_lum': 'border − centre Y',
               'bc_dist': '|border − centre| RGB', 'hf_frac': 'high-frequency share of Y'}
MAIN_LAYER = [('stem', 'logits'), ('b1', 'logits'), ('b5', 'pool'), ('b7', 'pool')]


# ---------------------------------------------------------------- helpers

def step_fmt(x):
    return None if x is None else f'{int(x):,}'


def f2(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.2f}'


def f3(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{x:.3f}'


def explorer_hash(fam, t, a, b, m=0, c='seba', r=4, **kw):
    q = f'#explorer?fam={fam}&t={t}&a={a}&b={b}&m={m}&c={c}&r={r}'
    return q + ''.join(f'&{k}={v}' for k, v in kw.items())


def class_counts(lab, cls, r):
    """(r + 1, 10) counts: rows = sets 0..r-1, then unassigned."""
    row = np.where(lab >= 0, lab, r)
    return np.bincount(row * 10 + cls, minlength=(r + 1) * 10).reshape(r + 1, 10)


def psi_of(npz_arrays, m, pi):
    vecs, scale = npz_arrays
    return vecs[m, pi].T.astype(np.float64) * (scale[m, pi].astype(np.float64) / 32767.0)  # (N, 8)


def seba_vectors(psi, rot, perm, r):
    """numpy port of seba.seba_reconstruct for rank r (clip=True). psi (N, 8) = psi_1..psi_8."""
    N = psi.shape[0]
    v = np.concatenate([np.full((N, 1), N ** -0.5), psi[:, :r - 1]], 1)
    z = v @ rot[:r, :r].astype(np.float64)
    s = np.sign(z) * np.maximum(np.abs(z) - 0.99 / np.sqrt(N), 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0))
    sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    s = s[:, perm[:r].astype(np.int64)]
    return np.maximum(s, 0)


def hard_labels(s):
    a = s.argmax(1)
    a[s.max(1) <= 0] = -1
    return a


# ---------------------------------------------------------------- image statistics

def image_stats(images):
    x = images.astype(np.float64) / 255.0  # (N, 32, 32, 3)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0.0)
    H, W = Y.shape[1:]
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ring = np.minimum(np.minimum(ii, jj), np.minimum(H - 1 - ii, W - 1 - jj)) < 4
    centre = (ii >= 8) & (ii < 24) & (jj >= 8) & (jj < 24)
    assert ring.sum() == 448 and centre.sum() == 256
    bc = lambda A: A[:, ring].mean(1) - A[:, centre].mean(1)
    Y0 = Y - Y.mean((1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(Y0)) ** 2
    f = np.fft.fftfreq(H) * H
    rad = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    tot = P[:, rad > 0].sum(1)
    hf = np.where(tot > 0, P[:, rad > 8].sum(1) / np.where(tot > 0, tot, 1), 0.0)
    out = dict(mean_r=R.mean((1, 2)), mean_g=G.mean((1, 2)), mean_b=B.mean((1, 2)), luminance=Y.mean((1, 2)),
               contrast=Y.std((1, 2)), saturation=sat.mean((1, 2)), bc_r=bc(R), bc_g=bc(G), bc_b=bc(B), bc_lum=bc(Y),
               hf_frac=hf)
    out['bc_dist'] = np.sqrt(out['bc_r'] ** 2 + out['bc_g'] ** 2 + out['bc_b'] ** 2)
    return np.stack([out[k] for k in STATS], 1)  # (N, n_stats)


def round_floats(x, nd=3):
    """Round every float in a nested structure (numpy arrays included) to nd decimals (JSON size)."""
    if isinstance(x, dict):
        return {k: round_floats(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_floats(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x, nd) if x.dtype.kind == 'f' else x
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd) if np.isfinite(x) else x
    return x


def zscore(X):
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


# ---------------------------------------------------------------- main entry

def run(D, data_dir, A, classmix_modes=None):
    """D: the analysis.py data dict (with 'perms'); A: the analysis module (helpers). Returns (sections, facts)."""
    meta, steps, sites, T, S, R = D['meta'], D['steps'], D['sites'], D['T'], D['S'], D['R']
    LL, TL, lp, tp, cls, perms = D['LL'], D['TL'], D['lp'], D['tp'], D['classes'], D['perms']
    classes = meta['classes']
    N = len(cls)
    dm = int(meta['default_mode']) if meta.get('default_mode') is not None else 0
    M = len(meta['modes'])
    cm_modes = [dm] if classmix_modes is None else classmix_modes
    rix = {r: R.index(r) for r in RS}
    sidx = {s: i for i, s in enumerate(sites)}
    step_lbl = [f'step {s}' for s in steps]
    facts = {}

    # ------------------------------------------------ 1. class make-up
    q_cm = dict(mode=dm, modes=cm_modes, classes=classes, row_note='rows: set 1..set r (reliability order), then '
                'unassigned', layer_pairs=meta['layer_pairs'], time_pairs=meta['time_pairs'], by_mode={})
    for m in cm_modes:
        lay = {str(r): [[class_counts(LL[t, m, pi, rix[r]], cls, r).tolist() for pi in range(len(meta['layer_pairs']))]
                        for t in range(T)] for r in RS}
        tim = {str(r): [[class_counts(TL[s, m, pi, rix[r]].astype(np.int64), cls, r).tolist()
                         for pi in range(len(meta['time_pairs']))] for s in range(S)] for r in RS}
        q_cm['by_mode'][str(m)] = dict(layer=lay, time=tim)
    lay0 = q_cm['by_mode'][str(dm)]['layer']
    tim0 = q_cm['by_mode'][str(dm)]['time']
    rowsr = lambda r: [f'set {j + 1}' for j in range(r)] + ['unassigned']
    q_cm['cells'] = {str(r): {sites[s]: dict(
        rows=rowsr(r), cols=classes, z=[lay0[str(r)][t][lp[(s, s)]] for t in range(T)], zlabel='images',
        zrange=[0, 200], rowlabel='SEBA set', collabel='CIFAR class (validation)',
        slider=dict(label='checkpoint (step)', values=steps)) for s in range(S)} for r in RS}

    def cm_layer(t, a, b, r):
        return np.array(lay0[str(r)][t][lp[(a, b)]])

    def cm_time(s, c, d, r):
        return np.array(tim0[str(r)][s][tp[(c, d)]])

    named = {}

    def name_block(key, counts, r, title):
        named[key] = dict(rows=rowsr(r), cols=classes, z=counts.tolist(), zlabel='images', zrange=[0, 200],
                          rowlabel='SEBA set', collabel='CIFAR class (validation)', title=title)

    # ------------------------------------------------ 2. vehicle / animal at r = 2
    vmask = np.array([c in VEHICLES for c in classes])
    va = vmask[cls].astype(np.int64)  # 1 = vehicle, 0 = animal -> labeling with 2 sets
    va_lab = np.where(va == 1, 0, 1)  # set 0 = vehicles, set 1 = animals
    ri2 = rix[2]
    q_va = dict(vehicles=list(VEHICLES), definitions='see analysis_findings.py docstring, section 2', by_mode={})
    VA = {}
    for m in range(M):
        arr = {k: np.full((S, T), np.nan) for k in ('score', 'mu', 'lo', 'hi', 'max', 'ceil', 'agree', 'nun',
                                                     'vf0', 'vf1', 'n0', 'n1')}
        for s in range(S):
            for t in range(T):
                lab = LL[t, m, lp[(s, s)], ri2]
                sc, pairs, nl = A.matched_jaccard(lab, va_lab, 2, 2, perms)
                arr['score'][s, t] = sc
                arr['mu'][s, t], arr['lo'][s, t], arr['hi'][s, t] = A.null_summary(nl)
                arr['max'][s, t] = nl.max()
                arr['ceil'][s, t] = A.size_ceiling(lab, va_lab, 2, 2)
                assigned = lab >= 0
                match = {i: j for i, j, _ in pairs}
                pred = np.array([match.get(int(x), -9) for x in lab[assigned]])
                arr['agree'][s, t] = float((pred == va_lab[assigned]).mean()) if assigned.any() else np.nan
                arr['nun'][s, t] = int((~assigned).sum())
                for j in (0, 1):
                    mem = lab == j
                    arr[f'n{j}'][s, t] = mem.sum()
                    arr[f'vf{j}'][s, t] = va[mem].mean() if mem.any() else np.nan
        exc = arr['score'] - arr['mu']
        arr['excess'] = exc
        den = arr['ceil'] - arr['mu']
        arr['nexc'] = np.where(den > 0, exc / np.where(den > 0, den, 1), np.nan)
        beat = arr['score'] > arr['max']
        on, most, first, half = {}, {}, {}, {}
        for s in range(S):
            b = beat[s]
            first[sites[s]] = next((steps[t] for t in range(T) if b[t]), None)
            hk = arr['nexc'][s] >= 0.5
            half[sites[s]] = next((steps[t] for t in range(T) if hk[t:].all()), None)
            on[sites[s]] = next((steps[t] for t in range(T) if b[t:].all()), None) if b[-1] else None
            if b[-1] and exc[s, -1] > 0:
                ok = exc[s] >= 0.8 * exc[s, -1]
                most[sites[s]] = next((steps[t] for t in range(T) if ok[t:].all()), None)
            else:
                most[sites[s]] = None
        VA[m] = dict(arr=arr, onset=on, most=most, first=first, half=half)
        H = lambda z, zl, zr=(0, 1), **kw: dict(rows=sites, cols=steps, z=z, zlabel=zl, zrange=list(zr),
                                               rowlabel='site', collabel='checkpoint (step)', **kw)
        q_va['by_mode'][str(m)] = dict(
            score=H(arr['score'], 'matched Jaccard, r = 2 sets vs vehicles/animals'),
            null_mean=H(arr['mu'], 'permutation-null mean'),
            null_hi=H(arr['hi'], 'permutation null, 97.5th percentile'),
            null_max=H(arr['max'], 'permutation null, maximum of 200 draws'),
            ceiling=H(arr['ceil'], 'size ceiling'),
            norm_excess=H(arr['nexc'], '(score − null mean) / (ceiling − null mean)', (-0.2, 1), diverging=False),
            agree=H(arr['agree'], 'assigned images in the matched super-class'),
            n_unassigned=arr['nun'], n_set1=arr['n0'], n_set2=arr['n1'],
            vehicle_frac_set1=arr['vf0'], vehicle_frac_set2=arr['vf1'],
            lines=A.lines(steps, [dict(name=sites[s], y=arr['score'][s], band=[arr['lo'][s], arr['hi'][s]])
                                  for s in range(S)], 'training step', 'matched Jaccard vs vehicles/animals (r = 2)',
                          'symlog', colors='ordinal'),
            lines_norm_excess=A.lines(steps, [dict(name=sites[s], y=arr['nexc'][s]) for s in range(S)],
                                      'training step', '(score − null mean) / (ceiling − null mean)', 'symlog',
                                      (-0.2, 1), colors='ordinal', refline=dict(y=0, label='null mean')),
            onset_step=on, most_step=most, first_beat_step=first, half_step=half)

    # best class bipartition (default mode)
    masks = [mk for mk in range(1, 1023) if mk & 1]  # class 0 on side C: 511 unordered splits
    Cm = np.array([[(mk >> c) & 1 for c in range(10)] for mk in masks], bool)  # (511, 10)
    va_mask_idx = masks.index(int(sum(1 << c for c in range(10) if vmask[c])))
    csize = np.bincount(cls, minlength=10)
    best = {}
    rank_z = np.full((S, T), np.nan)
    best_sc = np.full((S, T), np.nan)
    for s in range(S):
        best[sites[s]] = []
        for t in range(T):
            cnt = cm_layer(t, s, s, 2)  # (3, 10)
            n = cnt[:2].sum(1).astype(float)
            ic = cnt[:2] @ Cm.T.astype(float)  # (2, 511) members of set j in C
            inc = n[:, None] - ic
            sC = Cm.astype(float) @ csize
            snC = N - sC
            J = lambda inter, ns, nc: np.where(ns + nc - inter > 0, inter / np.maximum(ns + nc - inter, 1), 0)
            p1 = (J(ic[0], n[0], sC) + J(inc[1], n[1], snC)) / 2  # set1<->C, set2<->notC
            p2 = (J(inc[0], n[0], snC) + J(ic[1], n[1], sC)) / 2
            sc = np.maximum(p1, p2)
            k = int(np.argmax(sc))
            side1 = Cm[k] if p1[k] >= p2[k] else ~Cm[k]
            best[sites[s]].append(dict(score=sc[k], set1_side=[classes[c] for c in range(10) if side1[c]]))
            rank_z[s, t] = 1 + int((sc > sc[va_mask_idx] + 1e-12).sum())
            best_sc[s, t] = sc[k]
    q_va['best_bipartition'] = dict(mode=dm, per_site=best,
                                     va_rank=dict(rows=sites, cols=steps, z=rank_z, zlabel='rank of vehicles/animals '
                                                  'among 511 class splits (1 = best)', zrange=[1, 511],
                                                  rowlabel='site', collabel='checkpoint (step)'),
                                     best_score=dict(rows=sites, cols=steps, z=best_sc, zlabel='matched Jaccard of the '
                                                     'best-matching class split', zrange=[0, 1], rowlabel='site',
                                                     collabel='checkpoint (step)'))
    # per-class placement and pixel baseline (default mode)
    X = None
    with np.load(data_dir / 'probe.npz') as p:
        images = p['images']
    X = image_stats(images)
    inm = np.full((T, S, 10), np.nan)
    una = np.full((T, S, 10), np.nan)
    pix = np.full((S, T), np.nan)
    pix_stat = [[None] * T for _ in range(S)]
    for s in range(S):
        for t in range(T):
            lab = LL[t, dm, lp[(s, s)], ri2]
            _, pairs, _ = A.matched_jaccard(lab, va_lab, 2, 2)
            match = {i: j for i, j, _ in pairs}
            for c in range(10):
                mem = cls == c
                own = 0 if vmask[c] else 1
                inm[t, s, c] = np.mean([match.get(int(x), -9) == own for x in lab[mem]])
                una[t, s, c] = np.mean(lab[mem] < 0)
            n1, n2 = int((lab == 0).sum()), int((lab == 1).sum())
            bestv, bests = -1, None
            for i, st in enumerate(STATS):
                for sg in (1, -1):
                    order = np.argsort(-sg * X[:, i], kind='stable')
                    pl = np.full(N, -1)
                    pl[order[:n1]] = 0
                    pl[order[N - n2:]] = 1
                    v = A.matched_jaccard(pl, va_lab, 2, 2)[0]
                    if v > bestv:
                        bestv, bests = v, ('+' if sg > 0 else '−') + st
            pix[s, t], pix_stat[s][t] = bestv, bests
    slider_t = dict(label='checkpoint (step)', values=steps)
    q_va['class_placement'] = dict(
        mode=dm,
        in_matched=dict(rows=sites, cols=classes, z=inm, zlabel="fraction of the class in the set matched to its "
                        "super-class (r = 2)", zrange=[0, 1], rowlabel='site', collabel='CIFAR class', slider=slider_t),
        unassigned=dict(rows=sites, cols=classes, z=una, zlabel='fraction of the class unassigned (r = 2)',
                        zrange=[0, 1], rowlabel='site', collabel='CIFAR class', slider=slider_t))
    q_va['pixel_baseline'] = dict(mode=dm, score=dict(rows=sites, cols=steps, z=pix, zlabel='best single-statistic '
                                                      'split, matched Jaccard vs vehicles/animals', zrange=[0, 1],
                                                      rowlabel='site', collabel='checkpoint (step)'),
                                  stat=pix_stat)
    # time blocks, r = 2, default mode
    tb = {k: np.full((S, len(meta['time_pairs'])), np.nan) for k in ('score', 'mu', 'hi', 'max', 'ceil')}
    for s in range(S):
        for pi in range(len(meta['time_pairs'])):
            lab = TL[s, dm, pi, ri2].astype(np.int64)
            sc, _, nl = A.matched_jaccard(lab, va_lab, 2, 2, perms)
            tb['score'][s, pi], tb['mu'][s, pi], tb['hi'][s, pi], tb['max'][s, pi] = (
                sc, nl.mean(), np.percentile(nl, 97.5), nl.max())
            tb['ceil'][s, pi] = A.size_ceiling(lab, va_lab, 2, 2)
    q_va['time_blocks'] = dict(mode=dm, rows=sites, pairs=meta['time_pairs'], score=tb['score'],
                               null_mean=tb['mu'], null_hi=tb['hi'], null_max=tb['max'], ceiling=tb['ceil'])

    # ------------------------------------------------ 3. image statistics
    Zx = zscore(X)
    ns = len(STATS)
    rng = np.random.default_rng(A.SEED + 1)
    pbank = np.stack([rng.permutation(N) for _ in range(N_PERM_IMG)])  # (P, N)
    Zp = Zx[pbank].transpose(1, 0, 2).reshape(N, N_PERM_IMG * ns)  # column p*ns + stat
    def within(Mx):
        Mx = Mx.astype(np.float64).copy()
        for c in range(10):
            Mx[cls == c] -= Mx[cls == c].mean(0)
        return zscore(Mx)
    Zxw = within(X)
    rng2 = np.random.default_rng(A.SEED + 2)
    idx_c = [np.flatnonzero(cls == c) for c in range(10)]
    wbank = np.empty((N_PERM_IMG, N), np.int64)
    for q_ in range(N_PERM_IMG):
        for ic in idx_c:
            wbank[q_, ic] = ic[rng2.permutation(len(ic))]
    Zpw = Zxw[wbank].transpose(1, 0, 2).reshape(N, N_PERM_IMG * ns)
    U_, sv_, _ = np.linalg.svd(Zx, full_matrices=False)  # rank-revealing: the statistics are collinear
    img_rank = int((sv_ > 1e-8 * sv_[0]).sum())
    Q = U_[:, :img_rank]  # orthonormal basis of the span of the centred statistics (N, rank)
    Qp = Q[pbank].transpose(1, 0, 2).reshape(N, N_PERM_IMG * img_rank)
    design = np.concatenate([np.ones((N, 1)), Zx], 1)
    npz_cache = {}

    def arrays(fam, key):
        k = (fam, key)
        if k not in npz_cache:
            fn = data_dir / (f'layer_{key}.npz' if fam == 'layer' else f'time_{key}.npz')
            with np.load(fn) as z:
                npz_cache.clear()  # keep one file in memory at a time
                npz_cache[k] = (z['vecs'][dm], z['scale'][dm], z['seba_rot'][dm], z['seba_perm'][dm])
        return npz_cache[k]

    min_agree = [1.0]

    def block_functions(fam, key, pi, stored_labels):
        vecs, scale, rot, perm = arrays(fam, key)
        psi = vecs[pi].T.astype(np.float64) * (scale[pi].astype(np.float64) / 32767.0)
        F = [psi]
        hard = {}
        for r in RS:
            sv = seba_vectors(psi, rot[pi, rix[r]], perm[pi, rix[r]], r)
            agree = float((hard_labels(sv) == stored_labels[r]).mean())
            min_agree[0] = min(min_agree[0], agree)
            F.append(sv)
            hard[r] = stored_labels[r]
        return np.concatenate(F, 1), hard  # (N, 17)

    nf = 8 + sum(RS)
    gmax = np.zeros(N_PERM_IMG)

    def correlate(Fm, hard):
        Zf = zscore(Fm)
        corr = Zf.T @ Zx / N  # (17, ns)
        nul = (Zf.T @ Zp / N).reshape(nf, N_PERM_IMG, ns)
        an = np.abs(nul)
        thr_point = float(np.percentile(an, 95))
        mx = an.max((0, 2))
        gmax[:] = np.maximum(gmax, mx)
        thr_family = float(np.percentile(mx, 95))
        thr_family_psi = np.percentile(an[:8].max(0), 95, axis=0)  # (ns,)
        con, cnul = {}, []
        for r in RS:
            Hm = np.stack([(hard[r] == j) for j in range(r)], 1).astype(np.float64)  # (N, r)
            nj = Hm.sum(0)
            w = np.where((nj > 0) & (nj < N), 1 / np.maximum(nj, 1) + 1 / np.maximum(N - nj, 1), 0)
            con[str(r)] = (Hm.T @ Zx) * w[:, None]
            cnul.append(((Hm.T @ Zp) * w[:, None]).reshape(r, N_PERM_IMG, ns))
        cn = np.abs(np.concatenate(cnul, 0))
        Zfw = within(Fm)
        cw = Zfw.T @ Zxw / N
        anw = np.abs((Zfw.T @ Zpw / N).reshape(nf, N_PERM_IMG, ns))
        pc = Fm[:, :8] - Fm[:, :8].mean(0)
        den = (pc ** 2).sum(0)
        r2 = ((Q.T @ pc) ** 2).sum(0) / den
        coef = np.linalg.lstsq(design, Fm[:, :8], rcond=None)[0]  # rank-aware reference fit
        r2_ls = 1 - ((Fm[:, :8] - design @ coef) ** 2).sum(0) / den
        assert np.allclose(r2, r2_ls, rtol=0, atol=1e-8), np.abs(r2 - r2_ls).max()
        r2n = ((Qp.T @ pc) ** 2).reshape(N_PERM_IMG, img_rank, 8).sum(1) / den  # (P, 8)
        return dict(corr=corr, contrast=con, thr_point=thr_point, thr_family=thr_family, r2_psi=r2,
                    r2_null_hi=float(np.percentile(r2n, 95)), corr_wc=cw,
                    thr_family_wc=float(np.percentile(anw.max((0, 2)), 95)),
                    thr_family_psi_wc=np.percentile(anw[:8].max(0), 95, axis=0),
                    thr_family_psi=thr_family_psi, cthr_point=float(np.percentile(cn, 95)),
                    cthr_family=float(np.percentile(cn.max((0, 2)), 95)))

    cells = {}
    for t in range(T):
        for s in range(S):
            pi = lp[(s, s)]
            Fm, hard = block_functions('layer', steps[t], pi, {r: LL[t, dm, pi, rix[r]] for r in RS})
            cells[(t, s)] = correlate(Fm, hard)
        print(f'  image stats: checkpoint {steps[t]} cells done', flush=True)
    blocks = []
    for t in range(T):
        for an_, bn_ in MAIN_LAYER:
            a, b = sidx[an_], sidx[bn_]
            pi = lp[(a, b)]
            Fm, hard = block_functions('layer', steps[t], pi, {r: LL[t, dm, pi, rix[r]] for r in RS})
            res = correlate(Fm, hard)
            res.update(name=f'{an_}–{bn_} @ step {steps[t]}', fam='layer', t=t, a=a, b=b,
                       hash=explorer_hash('layer', t, a, b, dm))
            blocks.append(res)
    for s in range(S):
        for (c, d), lbl in (((0, T - 1), 'all checkpoints'), ((T // 2, T - 1), f'steps {steps[T // 2]}–{steps[-1]}')):
            pi = tp[(c, d)]
            Fm, hard = block_functions('time', sites[s], pi, {r: TL[s, dm, pi, rix[r]].astype(np.int64) for r in RS})
            res = correlate(Fm, hard)
            res.update(name=f'{sites[s]}, {lbl}', fam='time', t=s, a=c, b=d,
                       hash=explorer_hash('time', s, c, d, dm))
            blocks.append(res)
    print(f'  image stats: blocks done; min SEBA reconstruction agreement {min_agree[0]:.4f}', flush=True)
    assert min_agree[0] >= 0.99, min_agree[0]
    thr_global = float(np.percentile(gmax, 95))

    fnames = [f'ψ{k}' for k in range(1, 9)] + [f'SEBA r={r} set {j + 1}' for r in RS for j in range(r)]
    C = np.array([[cells[(t, s)]['corr'] for s in range(S)] for t in range(T)])  # (T, S, 17, ns)
    TFP = np.array([[cells[(t, s)]['thr_family_psi'] for s in range(S)] for t in range(T)])  # (T, S, ns)
    maxabs = np.abs(C[:, :, :8]).max(2)  # (T, S, ns)
    slider = dict(label='checkpoint (step)', values=steps)
    zr_max = [0, float(np.ceil(maxabs.max() * 10) / 10)]
    zr_sig = float(np.ceil(np.abs(C).max() * 10) / 10)
    heat_psi = {str(k): dict(rows=[STAT_LABELS[x] for x in STATS], cols=sites,
                             z=C[:, :, k - 1].transpose(0, 2, 1), zlabel=f'Pearson corr with ψ{k}',
                             zrange=[-zr_sig, zr_sig], diverging=True, rowlabel='image statistic',
                             collabel='site (single-snapshot block)', slider=slider) for k in range(1, 9)}
    off = {2: 8, 3: 10, 4: 13}
    heat_seba = {}
    for r in RS:
        z = C[:, :, off[r]:off[r] + r]  # (T, S, r, ns)
        heat_seba[str(r)] = dict(rows=[STAT_LABELS[x] for x in STATS],
                                 cols=[f'{sites[s]} set {j + 1}' for s in range(S) for j in range(r)],
                                 z=z.reshape(T, S * r, ns).transpose(0, 2, 1), zlabel=f'Pearson corr with SEBA vector '
                                 f'(r = {r})', zrange=[-zr_sig, zr_sig], diverging=True, rowlabel='image statistic',
                                 collabel='site, SEBA set (single-snapshot block)', slider=slider)
    BC = np.array([b['corr'] for b in blocks])  # (nb, 17, ns)
    bnames = [b['name'] for b in blocks]
    q_img = dict(
        stats=STATS, stat_labels=STAT_LABELS, functions=fnames, mode=dm, n_perm=N_PERM_IMG, img_rank=img_rank,
        definitions='see analysis_findings.py docstring, section 3',
        per_image={k: np.round(X[:, i], 4) for i, k in enumerate(STATS)},
        cells=dict(corr=C, contrast={str(r): [[cells[(t, s)]['contrast'][str(r)] for s in range(S)]
                                              for t in range(T)] for r in RS},
                   thr_point=[[cells[(t, s)]['thr_point'] for s in range(S)] for t in range(T)],
                   thr_family=[[cells[(t, s)]['thr_family'] for s in range(S)] for t in range(T)],
                   thr_family_psi=TFP,
                   cthr_point=[[cells[(t, s)]['cthr_point'] for s in range(S)] for t in range(T)],
                   cthr_family=[[cells[(t, s)]['cthr_family'] for s in range(S)] for t in range(T)],
                   r2_psi=[[cells[(t, s)]['r2_psi'] for s in range(S)] for t in range(T)],
                   r2_null_hi=[[cells[(t, s)]['r2_null_hi'] for s in range(S)] for t in range(T)],
                   corr_wc=[[cells[(t, s)]['corr_wc'] for s in range(S)] for t in range(T)],
                   thr_family_wc=[[cells[(t, s)]['thr_family_wc'] for s in range(S)] for t in range(T)],
                   thr_family_psi_wc=[[cells[(t, s)]['thr_family_psi_wc'] for s in range(S)] for t in range(T)]),
        heat_maxabs_psi_wc=dict(rows=[STAT_LABELS[x] for x in STATS], cols=sites,
                                z=np.abs(np.array([[cells[(t, s)]['corr_wc'][:8] for s in range(S)]
                                                   for t in range(T)])).max(2).transpose(0, 2, 1),
                                zlabel='max over k = 1..8 of |within-class corr(statistic, ψk)|', zrange=zr_max,
                                rowlabel='image statistic', collabel='site (single-snapshot block)', slider=slider),
        heat_r2_psi=dict(rows=fnames[:8], cols=sites, z=np.array([[cells[(t, s)]['r2_psi'] for s in range(S)]
                                                                   for t in range(T)]).transpose(0, 2, 1),
                         zlabel='R² of ψk on the 12 image statistics (least-squares fit)', zrange=[0, 1],
                         rowlabel='eigenfunction', collabel='site (single-snapshot block)', slider=slider),
        class_contrast=dict(rows=classes, cols=[STAT_LABELS[x] for x in STATS],
                            z=np.array([Zx[cls == c].mean(0) - Zx[cls != c].mean(0) for c in range(10)]),
                            zlabel='class mean − other classes, in SDs of the statistic', zrange=[-1.5, 1.5],
                            diverging=True, rowlabel='CIFAR class (validation)', collabel='image statistic'),
        vehicle_contrast={st: Zx[va == 1, i].mean() - Zx[va == 0, i].mean() for i, st in enumerate(STATS)},
        blocks=[{k: v for k, v in b.items()} for b in blocks],
        thr_global=thr_global,
        heat_maxabs_psi=dict(rows=[STAT_LABELS[x] for x in STATS], cols=sites, z=maxabs.transpose(0, 2, 1),
                             zlabel='max over k = 1..8 of |corr(statistic, ψk)|', zrange=zr_max,
                             rowlabel='image statistic', collabel='site (single-snapshot block)', slider=slider),
        heat_psi=heat_psi, heat_seba=heat_seba,
        heat_blocks_psi=dict(rows=[STAT_LABELS[x] for x in STATS], cols=fnames[:8], z=BC[:, :8].transpose(0, 2, 1),
                             zlabel='Pearson corr', zrange=[-zr_sig, zr_sig], diverging=True,
                             rowlabel='image statistic', collabel='eigenfunction',
                             slider=dict(label='block', values=bnames)),
        heat_blocks_seba={str(r): dict(rows=[STAT_LABELS[x] for x in STATS], cols=[f'set {j + 1}' for j in range(r)],
                                       z=BC[:, off[r]:off[r] + r].transpose(0, 2, 1), zlabel='Pearson corr',
                                       zrange=[-zr_sig, zr_sig], diverging=True, rowlabel='image statistic',
                                       collabel=f'SEBA vector (r = {r})', slider=dict(label='block', values=bnames))
                          for r in RS},
        lines_maxabs_psi={st: A.lines(steps, [dict(name=sites[s], y=maxabs[:, s, i]) for s in range(S)],
                                      'training step', f'max over k of |corr({STAT_LABELS[st]}, ψk)|', 'symlog',
                                      zr_max, colors='ordinal',
                                      refline=dict(y=float(TFP[:, :, i].max()),
                                                   label='permutation null 95 % (max over snapshots)'))
                          for i, st in enumerate(STATS)})

    # ------------------------------------------------ 4. facts and findings support
    T1 = T - 1
    a0 = VA[dm]['arr']
    for s in range(S):
        sn = sites[s]
        facts[f'va.{sn}.final.score'] = f2(a0['score'][s, T1])
        facts[f'va.{sn}.final.null_mean'] = f2(a0['mu'][s, T1])
        facts[f'va.{sn}.final.null_max'] = f2(a0['max'][s, T1])
        facts[f'va.{sn}.final.ceiling'] = f2(a0['ceil'][s, T1])
        facts[f'va.{sn}.final.norm_excess'] = f2(a0['nexc'][s, T1])
        facts[f'va.{sn}.final.agree'] = f2(a0['agree'][s, T1])
        facts[f'va.{sn}.final.n_unassigned'] = str(int(a0['nun'][s, T1]))
        facts[f'va.{sn}.onset_step'] = step_fmt(VA[dm]['onset'][sn]) or 'none'
        facts[f'va.{sn}.most_step'] = step_fmt(VA[dm]['most'][sn]) or 'none'
        facts[f'va.{sn}.first_beat_step'] = step_fmt(VA[dm]['first'][sn]) or 'none'
        facts[f'va.{sn}.half_step'] = step_fmt(VA[dm]['half'][sn]) or 'none'
        facts[f'va.{sn}.final.pixel_best'] = f2(pix[s, T1])
        facts[f'va.{sn}.final.pixel_best_stat'] = pix_stat[s][T1]
        facts[f'va.{sn}.step0.pixel_best'] = f2(pix[s, 0])
        facts[f'va.{sn}.step0.pixel_best_stat'] = pix_stat[s][0]
        worst = int(np.argmin(inm[T1, s]))
        facts[f'va.{sn}.final.worst_class'] = classes[worst]
        facts[f'va.{sn}.final.worst_class_in_matched'] = f2(inm[T1, s, worst])
        facts[f'va.{sn}.final.worst_class_unassigned'] = f2(una[T1, s, worst])
        facts[f'va.{sn}.final.va_rank'] = str(int(rank_z[s, T1]))
        facts[f'va.{sn}.final.best_split'] = ', '.join(best[sn][T1]['set1_side'])
        for t in range(T):
            facts[f'va.{sn}.step{steps[t]}.score'] = f2(a0['score'][s, t])
            facts[f'va.{sn}.step{steps[t]}.norm_excess'] = f2(a0['nexc'][s, t])
            facts[f'va.{sn}.step{steps[t]}.agree'] = f2(a0['agree'][s, t])
            facts[f'va.{sn}.step{steps[t]}.va_rank'] = str(int(rank_z[s, t]))
            facts[f'va.{sn}.step{steps[t]}.null_max'] = f2(a0['max'][s, t])
            facts[f'va.{sn}.step{steps[t]}.pixel_best'] = f2(pix[s, t])
            facts[f'va.{sn}.step{steps[t]}.pixel_best_stat'] = pix_stat[s][t]
            ia = classes.index('airplane')
            facts[f'va.{sn}.step{steps[t]}.airplane_in_vehicle_set'] = f2(inm[t, s, ia])
            facts[f'va.{sn}.step{steps[t]}.airplane_unassigned'] = f2(una[t, s, ia])
            facts[f'va.{sn}.step{steps[t]}.min_class_in_matched_excl_airplane'] = f2(
                np.delete(inm[t, s], ia).min())
    facts['va.null_max.max_over_cells'] = f2(np.nanmax(a0['max']))
    on_vals = [VA[dm]['onset'][sn] for sn in sites]
    facts['va.onset_step.all_sites'] = (step_fmt(on_vals[0]) if len(set(on_vals)) == 1 and on_vals[0] is not None
                                        else 'varies by site')
    facts['va.onset_step.n_sites_at_first_checkpoint'] = str(sum(v == steps[0] for v in on_vals))
    facts['va.n_sites'] = str(S)
    for s in range(S):  # first checkpoint from which the SEBA split beats the pixel baseline at every later one
        above = a0['score'][s] > pix[s]
        facts[f'va.{sites[s]}.above_pixel_from_step'] = step_fmt(
            next((steps[t] for t in range(T) if above[t:].all()), None)) or 'none'
    for s in range(S):  # time blocks at r = 2
        for (c, d) in ((0, T - 1), (T // 2, T - 1)):
            pi = tp[(c, d)]
            facts[f'va.time.{sites[s]}.step{steps[c]}-{steps[d]}.score'] = f2(tb['score'][s, pi])
            facts[f'va.time.{sites[s]}.step{steps[c]}-{steps[d]}.null_max'] = f2(tb['max'][s, pi])

    def block_facts(prefix, fam, t, a, b, r, counts, vals, ret_seba, cells_in_block):
        lam = vals[1:r + 1]
        for k in range(r):
            facts[f'{prefix}.lambda{k + 1}'] = f3(lam[k])
        facts[f'{prefix}.gap_after_lambda{r - 1}'] = f3(vals[r - 1] - vals[r])
        facts[f'{prefix}.gap_after_lambda{r}'] = f3(vals[r] - vals[r + 1])
        facts[f'{prefix}.n_unassigned'] = f'{int(counts[r].sum()):,}'
        for j in range(r):
            facts[f'{prefix}.set{j + 1}.size'] = f'{int(counts[j].sum()):,}'
            order = np.argsort(-counts[j], kind='stable')
            tot = counts[j].sum()
            facts[f'{prefix}.set{j + 1}.top_classes'] = ', '.join(
                f'{classes[c]} {int(counts[j, c])}' for c in order[:4] if counts[j, c] > 0)
            facts[f'{prefix}.set{j + 1}.vehicle_frac'] = f2(counts[j, vmask].sum() / tot if tot else np.nan)
            rj = ret_seba[j, cells_in_block]
            facts[f'{prefix}.set{j + 1}.retention_min'] = f2(rj.min())
            facts[f'{prefix}.set{j + 1}.retention_max'] = f2(rj.max())
        rr = ret_seba[:r][:, cells_in_block]
        facts[f'{prefix}.retention_min'] = f2(rr.min())
        facts[f'{prefix}.retention_max'] = f2(rr.max())
        facts[f'{prefix}.hash'] = explorer_hash(fam, t, a, b, dm, r=r)

    # finding 1: b7..pool at the last checkpoint, r = 2..4
    with np.load(data_dir / f'layer_{steps[T1]}.npz') as z:
        Lvals, Lret, Lrpsi, Lvec, Lsc = z['vals'][dm], z['retention_seba'][dm], z['retention_psi'][dm], \
            z['vecs'][dm], z['scale'][dm]
    for r in RS:
        a, b = sidx['b7'], sidx['pool']
        counts = cm_layer(T1, a, b, r)
        block_facts(f'f1.b7_pool.final.r{r}', 'layer', T1, a, b, r, counts, Lvals[lp[(a, b)]],
                    Lret[lp[(a, b)], rix[r]], list(range(a, b + 1)))
        name_block(f'layer_t{T1}_b7-pool_r{r}', counts, r, f'b7–pool, step {steps[T1]}, r = {r}')
    # finding 1 consistency: the same block under other kernel rules, and against the b9 late time block
    a7, b10 = sidx['b7'], sidx['pool']
    for r in RS:
        ref = LL[T1, dm, lp[(a7, b10)], rix[r]]
        for m in range(M):
            if m != dm:
                facts[f'f1.b7_pool.final.r{r}.vs_rule{m}'] = f2(A.matched_jaccard(ref, LL[T1, m, lp[(a7, b10)], rix[r]],
                                                                               r, r)[0])
        facts[f'f1.b7_pool.final.r{r}.vs_b9_time_late'] = f2(A.matched_jaccard(
            ref, TL[sidx['b9'], dm, tp[(T // 2, T1)], rix[r]].astype(np.int64), r, r)[0])
        facts[f'f1.b7_pool.final.r{r}.vs_b9_alone'] = f2(A.matched_jaccard(
            ref, LL[T1, dm, lp[(sidx['b9'], sidx['b9'])], rix[r]], r, r)[0])
        facts[f'f1.b7_pool.final.r{r}.vs_classes'] = f2(A.matched_jaccard(ref, cls, r, 10)[0])
        facts[f'f1.b7_pool.final.r{r}.vs_classes_ceiling'] = f2(A.size_ceiling(ref, cls, r, 10))
    # finding 2: time block at b9 (late half and all), r = 2 and 4
    sb9 = sidx['b9']
    for (c, d) in ((T // 2, T - 1), (0, T - 1)):
        with np.load(data_dir / 'time_b9.npz') as z:
            tv, tr = z['vals'][dm][tp[(c, d)]], z['retention_seba'][dm][tp[(c, d)]]
        for r in RS:
            counts = cm_time(sb9, c, d, r)
            pref = f'f2.b9_time.step{steps[c]}-{steps[d]}.r{r}'
            block_facts(pref, 'time', sb9, c, d, r, counts, tv, tr[rix[r]], list(range(c, d + 1)))
            facts[f'{pref}.retention_step0_max'] = f2(tr[rix[r], :r, 0].max())
            facts[f'{pref}.retention_step0_min'] = f2(tr[rix[r], :r, 0].min())
            name_block(f'time_b9_{c}-{d}_r{r}', counts, r, f'b9, steps {steps[c]}–{steps[d]}, r = {r}')
    for t in range(T):
        name_block(f'layer_t{t}_b9_r2', cm_layer(t, sb9, sb9, 2), 2, f'b9 alone, step {steps[t]}, r = 2')
    q_cm['named'] = named

    # finding 4: carry (default mode, r = 4): adjacent and segments; read back from Q12 via A-provided summary
    # (filled by the caller, which has Q12; see analysis.py)

    # finding 5: single-cell psi_1 at site a vs block psi_1 of a..logits, last checkpoint
    f5 = {}
    for a in range(S - 1):
        pa, pb = lp[(a, a)], lp[(a, S - 1)]
        u = Lvec[pa, 0].astype(np.float64) * Lsc[pa, 0] / 32767
        w = Lvec[pb, 0].astype(np.float64) * Lsc[pb, 0] / 32767
        cosv = float(abs(u @ w) / np.linalg.norm(u) / np.linalg.norm(w))
        rs_, rb_ = Lrpsi[pa, 0], Lrpsi[pb, 0]  # retention over all 12 sites
        blk = list(range(a, S))
        # mean cos^2 between the leading 3-d spans
        U = Lvec[pa, :3].T.astype(np.float64) * Lsc[pa, :3] / 32767
        W = Lvec[pb, :3].T.astype(np.float64) * Lsc[pb, :3] / 32767
        U /= np.linalg.norm(U, axis=0)
        W /= np.linalg.norm(W, axis=0)
        sv = np.linalg.svd(U.T @ W, compute_uv=False)
        f5[sites[a]] = dict(abs_cos_psi1=cosv, span3_mean_cos2=float((sv ** 2).mean()),
                            static_ret_own=float(rs_[a]), static_ret_mean_block=float(rs_[blk].mean()),
                            static_ret_last=float(rs_[S - 2]), block_ret_own=float(rb_[a]),
                            block_lambda1=float(Lvals[pb, 1]), block_ret_last=float(rb_[S - 2]),
                            static_ret=rs_.tolist(), block_ret=rb_.tolist(),
                            block_psi1_vs_static_psi1=[float(abs(w @ (Lvec[lp[(c, c)], 0].astype(np.float64)
                                                                      * Lsc[lp[(c, c)], 0] / 32767)))
                                                       / np.linalg.norm(w) for c in range(S)])
        pre = f'f5.{sites[a]}_vs_{sites[a]}-logits.final'
        facts[f'{pre}.abs_cos_psi1'] = f2(cosv)
        facts[f'{pre}.span3_mean_cos2'] = f2(f5[sites[a]]['span3_mean_cos2'])
        facts[f'{pre}.static_ret_own'] = f2(rs_[a])
        facts[f'{pre}.static_ret_b9'] = f2(rs_[sidx['b9']])
        facts[f'{pre}.static_ret_pool'] = f2(rs_[sidx['pool']])
        facts[f'{pre}.static_ret_block_mean'] = f2(rs_[blk].mean())
        facts[f'{pre}.block_ret_own'] = f2(rb_[a])
        facts[f'{pre}.block_ret_pool'] = f2(rb_[sidx['pool']])
        facts[f'{pre}.block_lambda1'] = f2(Lvals[pb, 1])
        bv = f5[sites[a]]['block_psi1_vs_static_psi1']
        cbest = int(np.argmax(bv[a:])) + a
        facts[f'{pre}.block_psi1_best_static_site'] = sites[cbest]
        facts[f'{pre}.block_psi1_best_static_abs_cos'] = f2(bv[cbest])
        facts[f'{pre}.static_hash'] = explorer_hash('layer', T1, a, a, dm, c='eig', r=4, k=1)
        facts[f'{pre}.block_hash'] = explorer_hash('layer', T1, a, S - 1, dm, c='eig', r=4, k=1)
    f5_rows = list(f5.values())
    facts['f5.final.n_sites_block_ret_own_below_static'] = str(sum(r_['block_ret_own'] < r_['static_ret_own']
                                                                   for r_ in f5_rows))
    facts['f5.final.n_sites'] = str(len(f5_rows))
    facts['f5.final.median_abs_cos_psi1'] = f2(np.median([r_['abs_cos_psi1'] for r_ in f5_rows]))

    # image statistics: strongest cells, for the notes (facts only, no interpretation)
    def top_entries(Carr, thr, labels, k=12):
        flat = []
        for idx in np.ndindex(Carr.shape):
            flat.append((abs(Carr[idx]), idx))
        flat.sort(key=lambda x: -x[0])
        return [(v, idx) for v, idx in flat[:k]]
    facts['img.thr_global'] = f2(thr_global)
    # step 0: luminance alone (squared Pearson correlation with psi_1) and |cos| of each site's psi_1 with stem's
    il = STATS.index('luminance')
    with np.load(data_dir / f'layer_{steps[0]}.npz') as z:
        v0, s0 = z['vecs'][dm], z['scale'][dm]
    p1 = lambda si: v0[lp[(si, si)], 0].astype(np.float64) * s0[lp[(si, si)], 0] / 32767
    u0 = p1(0)
    for si in range(S):
        facts[f'img.step0.{sites[si]}.corr2_psi1_luminance'] = f2(C[0, si, 0, il] ** 2)
        w0 = p1(si)
        facts[f'step0.psi1_abs_cos_with_stem.{sites[si]}'] = f3(abs(u0 @ w0) / np.linalg.norm(u0) / np.linalg.norm(w0))
    R2 = np.array([[cells[(t, s)]['r2_psi'] for s in range(S)] for t in range(T)])  # (T, S, 8)
    R2n = np.array([[cells[(t, s)]['r2_null_hi'] for s in range(S)] for t in range(T)])
    facts['img.r2_null_hi_max'] = f3(R2n.max())
    facts['img.rank'] = str(img_rank)
    for t in (0, T1):
        tag = 'step0' if t == 0 else 'final'
        for s in range(S):
            facts[f'img.{tag}.{sites[s]}.r2_psi1'] = f2(R2[t, s, 0])
            facts[f'img.{tag}.{sites[s]}.r2_psi_max'] = f2(R2[t, s].max())
            facts[f'img.{tag}.{sites[s]}.r2_psi_max_k'] = str(int(R2[t, s].argmax()) + 1)
    CW = np.array([[cells[(t, s)]['corr_wc'] for s in range(S)] for t in range(T)])
    MW = np.abs(CW[:, :, :8]).max(2)
    for i, st in enumerate(STATS):
        facts[f'img.{st}.step0.max_abs_corr_wc_psi'] = f2(MW[0, :, i].max())
        facts[f'img.{st}.final.max_abs_corr_wc_psi'] = f2(MW[T1, :, i].max())
        facts[f'img.{st}.final.max_abs_corr_wc_psi.where'] = sites[int(MW[T1, :, i].argmax())]
        facts[f'img.vehicle_pointbiserial.{st}'] = f2(np.corrcoef(va, X[:, i])[0, 1])
    facts['img.thr_family_wc_median'] = f2(np.median([cells[k]['thr_family_wc'] for k in cells]))
    for i, st in enumerate(STATS):
        facts[f'img.vehicle_contrast.{st}'] = f2(Zx[va == 1, i].mean() - Zx[va == 0, i].mean())
    facts['img.thr_point_median'] = f2(np.median([cells[k]['thr_point'] for k in cells]))
    facts['img.thr_family_median'] = f2(np.median([cells[k]['thr_family'] for k in cells]))
    for i, st in enumerate(STATS):
        v = maxabs[:, :, i]
        t_, s_ = np.unravel_index(np.argmax(v), v.shape)
        k_ = int(np.argmax(np.abs(C[t_, s_, :8, i])))
        facts[f'img.{st}.max_abs_corr_psi'] = f2(v[t_, s_])
        facts[f'img.{st}.max_abs_corr_psi.where'] = f'{sites[s_]}, step {steps[t_]:,}, ψ{k_ + 1}'
        facts[f'img.{st}.max_abs_corr_psi.signed'] = f2(C[t_, s_, k_, i])
        facts[f'img.{st}.step0.max_abs_corr_psi'] = f2(v[0].max())
        facts[f'img.{st}.final.max_abs_corr_psi'] = f2(v[T1].max())
        facts[f'img.{st}.final.max_abs_corr_psi.where'] = sites[int(v[T1].argmax())]

    # context: validation accuracy at the explorer checkpoints (curves.json; a training log, description only)
    cp = data_dir / 'curves.json'
    if cp.exists():
        va_acc = {c_['step']: c_.get('val_acc') for c_ in json.loads(cp.read_text()).get('checkpoints', [])}
        for st in steps:
            if va_acc.get(st) is not None:
                facts[f'train.step{st}.val_acc'] = f2(va_acc[st])

    q_img = {k: (v if k == 'per_image' else round_floats(v)) for k, v in q_img.items()}
    support = dict(f5=f5)
    sections = dict(q_classmix=q_cm, q_va=q_va, q_imgstats=q_img, findings_support=support)
    return sections, facts
