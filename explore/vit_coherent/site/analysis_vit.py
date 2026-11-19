"""Measurements for the ViT site (DINOv2 ViT-S/14, one checkpoint, sites L1..L12): writes site/assets/questions.json.

    uv run --with numpy --with scipy python site/analysis_vit.py [vits_b2k]

The ResNet site's analysis.py / analysis_findings.py, reduced to what applies with one checkpoint, on the same
definitions (matched Jaccard, permutation null, size ceiling, normalised excess, class make-up, image statistics).
Labels (10 display groups, 100 fine classes), pixel statistics and the earlier run's decodability only validate and
describe; none of them enters a kernel, a block operator or a SEBA rotation.

Sections (questions.json keys):
  q_carry     site x site matched Jaccard of single-cell-block SEBA sets (slider: r), its permutation null, and the
              adjacent-site score per r with null bands; per kernel rule under by_mode.
  q_classmix  n_jc / n_c (fraction of class group c's images in set j; last row unassigned) for single-cell blocks
              (slider: site) at r = 2, 4, 8, and for the layer blocks L1..L12 and L7..L12 at r = 4, 8.
  q_va        r = 2 sets against the fixed animal / non-animal partition (groups 0-4 vs 5-9): score, null band,
              size ceiling, normalised excess, by site and kernel rule.
  q_imgstats  per-image pixel statistics on the 64 x 64 thumbnails (also read by the explorer's px: colourings) and
              their correlations with psi_1..psi_8 and SEBA vectors of each single-cell block; permutation thresholds.
  q_decod     ridge / kNN decodability by layer from data/dinov2s/decodability.json (the earlier run's 10,000 / 10,000
              protocol; not the 2,000 images here).
  q_kernel    kappa and median 15-NN kernel mass per kernel rule and site (meta.json).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SITE_DIR = Path(__file__).resolve().parent
ROOT = SITE_DIR.parent
SEED = 20261112
N_PERM = 200
DEFAULT_R = 4
ANIMAL_GROUPS = (0, 1, 2, 3, 4)
STATS = ['mean_r', 'mean_g', 'mean_b', 'luminance', 'contrast', 'saturation', 'bc_r', 'bc_g', 'bc_b', 'bc_lum',
         'bc_dist', 'hf_frac']
STAT_LABELS = {'mean_r': 'mean R', 'mean_g': 'mean G', 'mean_b': 'mean B', 'luminance': 'luminance Y',
               'contrast': 'RMS contrast of Y', 'saturation': 'HSV saturation', 'bc_r': 'border − centre R',
               'bc_g': 'border − centre G', 'bc_b': 'border − centre B', 'bc_lum': 'border − centre Y',
               'bc_dist': '|border − centre| RGB', 'hf_frac': 'high-frequency share of Y'}
# 64 x 64 thumbnails (centre crop 224 of the model input, resized to 64): every ResNet (32 px) length doubled
PIX = dict(size=64, ring=8, centre=(16, 48), hf_radius=16)
STAT_DEFS = {
    'mean_r': 'mean of the red channel over the 4,096 pixels (values in [0, 1])',
    'mean_g': 'mean of the green channel over the 4,096 pixels',
    'mean_b': 'mean of the blue channel over the 4,096 pixels',
    'luminance': 'mean of Y = 0.299 R + 0.587 G + 0.114 B over the pixels',
    'contrast': 'standard deviation of Y over the pixels',
    'saturation': 'mean over pixels of (max − min) / max of (R, G, B)',
    'bc_r': 'border-ring mean minus centre mean of R; border = the 1,792 pixels within 8 px of the edge, centre = the central 32 x 32',
    'bc_g': 'border-ring mean minus centre mean of G',
    'bc_b': 'border-ring mean minus centre mean of B',
    'bc_lum': 'border-ring mean minus centre mean of Y',
    'bc_dist': 'Euclidean norm of the border − centre differences of (R, G, B)',
    'hf_frac': 'share of the DFT energy of Y − mean(Y) at radius > 16 cycles per image among all non-zero frequencies',
}


# ---------------------------------------------------------------- statistics (as analysis.py)
def _jaccard(A, b, ra, rb):
    P = A.shape[0]
    va = A >= 0
    mask = va & (b >= 0)[None]
    key = (np.arange(P)[:, None] * (ra * rb) + A * rb + b[None])[mask]
    inter = np.bincount(key, minlength=P * ra * rb).reshape(P, ra, rb)
    ka = (np.arange(P)[:, None] * ra + A)[va]
    sa = np.bincount(ka, minlength=P * ra).reshape(P, ra)
    sb = np.bincount(b[b >= 0], minlength=rb)
    union = sa[:, :, None] + sb[None, None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _hmean(J):
    r, c = linear_sum_assignment(J, maximize=True)
    return J[r, c].sum() / min(J.shape)


def matched(a, b, ra, rb, perms=None):
    s = float(_hmean(_jaccard(a[None], b, ra, rb)[0]))
    if perms is None:
        return s, None
    Jn = _jaccard(a[perms], b, ra, rb)
    return s, np.array([_hmean(Jn[p]) for p in range(len(perms))])


def ceiling(a, b, ra, rb):
    sa = np.bincount(a[a >= 0], minlength=ra).astype(float)
    sb = np.bincount(b[b >= 0], minlength=rb).astype(float)
    hi = np.maximum(sa[:, None], sb[None])
    return float(_hmean(np.where(hi > 0, np.minimum(sa[:, None], sb[None]) / np.maximum(hi, 1), 0.0)))


def seba_vectors(psi, rot, perm, r):
    N = psi.shape[0]
    v = np.concatenate([np.full((N, 1), N ** -0.5), psi[:, :r - 1]], 1)
    z = v @ rot[:r, :r].astype(np.float64)
    s = np.sign(z) * np.maximum(np.abs(z) - 0.99 / np.sqrt(N), 0)
    s /= np.linalg.norm(s, axis=0)
    sg = np.sign(s.sum(0)); sg[sg == 0] = 1
    s *= sg
    s /= s.max(0)
    return np.maximum(s[:, perm[:r].astype(np.int64)], 0)


def image_stats(images):
    x = images.astype(np.float64) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0.0)
    H, W = Y.shape[1:]
    assert H == W == PIX['size']
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ring = np.minimum(np.minimum(ii, jj), np.minimum(H - 1 - ii, W - 1 - jj)) < PIX['ring']
    c0, c1 = PIX['centre']
    centre = (ii >= c0) & (ii < c1) & (jj >= c0) & (jj < c1)
    assert ring.sum() == 1792 and centre.sum() == 1024
    bc = lambda A: A[:, ring].mean(1) - A[:, centre].mean(1)
    Y0 = Y - Y.mean((1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(Y0)) ** 2
    f = np.fft.fftfreq(H) * H
    rad = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    tot = P[:, rad > 0].sum(1)
    hf = np.where(tot > 0, P[:, rad > PIX['hf_radius']].sum(1) / np.where(tot > 0, tot, 1), 0.0)
    out = dict(mean_r=R.mean((1, 2)), mean_g=G.mean((1, 2)), mean_b=B.mean((1, 2)), luminance=Y.mean((1, 2)),
               contrast=Y.std((1, 2)), saturation=sat.mean((1, 2)), bc_r=bc(R), bc_g=bc(G), bc_b=bc(B), bc_lum=bc(Y),
               hf_frac=hf)
    out['bc_dist'] = np.sqrt(out['bc_r'] ** 2 + out['bc_g'] ** 2 + out['bc_b'] ** 2)
    return np.stack([out[k] for k in STATS], 1)


def zcols(X):
    X = X - X.mean(0)
    n = np.linalg.norm(X, axis=0)
    return X / np.where(n > 0, n, 1)


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, np.ndarray):
        return r3(x.tolist())
    if isinstance(x, (float, np.floating)):
        return round(float(x), 4) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


f2 = lambda v: f"{v:.2f}"


# ---------------------------------------------------------------- main
def main(tag='vits_b2k'):
    t0 = time.time()
    src = ROOT / 'data' / tag
    meta = json.loads((src / 'meta.json').read_text())
    sites, modes, R = meta['sites'], meta['modes'], meta['seba_r']
    SETC = json.loads((SITE_DIR / 'palette.json').read_text())['sets']   # one colour per kernel rule / readout
    S, M = len(sites), len(modes)
    lp = {tuple(p): i for i, p in enumerate(meta['layer_pairs'])}
    z = np.load(src / 'layer_0.npz')
    LAB = z['seba_label'].astype(np.int64)                      # (M, P, R, N)
    vecs, scale, rot, perm = z['vecs'], z['scale'], z['seba_rot'], z['seba_perm']
    vals = z['vals']
    with np.load(src / 'probe.npz') as p:
        grp, fine, images = p['labels'].astype(np.int64), p['fine_labels'].astype(np.int64), p['images']
    N = len(grp)
    groups = meta['classes']
    G = len(groups)
    gsize = np.bincount(grp, minlength=G)
    rng = np.random.default_rng(SEED)
    perms = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    cell = lambda s: lp[(s, s)]
    ri = {r: i for i, r in enumerate(R)}
    facts, out = {}, {}

    # ---------------- q_carry: agreement between sites
    carry = {}
    for mi in range(M):
        mat = np.zeros((len(R), S, S)); mat_null_hi = np.zeros((len(R), S, S))
        adj = {r: dict(score=[], lo=[], hi=[], mean=[]) for r in R}
        for k, r in enumerate(R):
            for a in range(S):
                mat[k, a, a] = 1.0
                for b in range(a + 1, S):
                    want_null = (mi == 0) or (b == a + 1)
                    sc, nl = matched(LAB[mi, cell(a), k], LAB[mi, cell(b), k], r, r, perms if want_null else None)
                    mat[k, a, b] = mat[k, b, a] = sc
                    if nl is not None:
                        mat_null_hi[k, a, b] = mat_null_hi[k, b, a] = np.percentile(nl, 97.5)
                    if b == a + 1:
                        adj[r]['score'].append(sc); adj[r]['mean'].append(nl.mean())
                        adj[r]['lo'].append(np.percentile(nl, 2.5)); adj[r]['hi'].append(np.percentile(nl, 97.5))
        pairs = [f"{sites[a]}→{sites[a + 1]}" for a in range(S - 1)]
        carry[str(mi)] = dict(
            matrix=dict(rows=sites, cols=sites, z=mat, zlabel='matched Jaccard between single-site sets', zrange=[0, 1],
                        rowlabel='site', collabel='site', slider=dict(label='r (number of SEBA sets)', values=R, default=R.index(DEFAULT_R))),
            null_hi=mat_null_hi if mi == 0 else None,
            adjacent=dict(x=list(range(S - 1)), xticklabels=pairs, xlabel='adjacent site pair',
                          ylabel='matched Jaccard, site vs next site', yrange=[0, 1], colors='ordinal',
                          series=[dict(name=f'r = {r}', y=adj[r]['score'], band=[adj[r]['lo'], adj[r]['hi']]) for r in R]),
            adjacent_null_mean={str(r): adj[r]['mean'] for r in R})
        if mi == 0:
            A4 = np.array(adj[4]['score']); off = mat[ri[4]][np.triu_indices(S, 1)]
            facts.update({'carry.r4.adjacent_min': f2(A4.min()), 'carry.r4.adjacent_min_pair': pairs[int(A4.argmin())],
                          'carry.r4.adjacent_max': f2(A4.max()), 'carry.r4.adjacent_max_pair': pairs[int(A4.argmax())],
                          'carry.r4.adjacent_null_hi_max': f2(max(adj[4]['hi'])),
                          'carry.r4.L1_L12': f2(mat[ri[4], 0, S - 1]),
                          'carry.r4.offdiag_min': f2(off.min()), 'carry.r4.offdiag_max': f2(off.max()),
                          'carry.r4.null_hi_max': f2(mat_null_hi[ri[4]][np.triu_indices(S, 1)].max())})
            for r in R:
                facts[f'carry.r{r}.adjacent_mean'] = f2(np.mean(adj[r]['score']))
    out['q_carry'] = carry
    print('carry', f'{time.time() - t0:.0f}s', flush=True)

    # ---------------- q_classmix
    def makeup(lab, r):
        row = np.where(lab >= 0, lab, r)
        cnt = np.bincount(row * G + grp, minlength=(r + 1) * G).reshape(r + 1, G)
        return cnt, cnt / gsize[None]
    rows = lambda r: [f'set {j + 1}' for j in range(r)] + ['unassigned']
    cm = dict(group_sizes=gsize.tolist(), cells={}, named={}, counts={})
    for r in (2, 3, 4, 6, 8):
        fr, ct = [], []
        for s in range(S):
            c, f = makeup(LAB[0, cell(s), ri[r]], r)
            fr.append(f); ct.append(c)
        cm['cells'][str(r)] = dict(rows=rows(r), cols=groups, z=np.array(fr), fmt='pct', zrange=[0, 1],
                                   zlabel='share of the group\'s images in the set', rowlabel='SEBA set',
                                   collabel='class group (validation)', slider=dict(label='site', values=sites), ml=82)
        cm['counts'][str(r)] = np.array(ct)
    for a, b in ((0, S - 1), (6, S - 1), (0, 5)):
        for r in (4, 8):
            lab = LAB[0, lp[(a, b)], ri[r]]
            c, f = makeup(lab, r)
            name = f'{sites[a]}-{sites[b]}_r{r}'
            cm['named'][name] = dict(rows=rows(r), cols=groups, z=f, fmt='pct', zrange=[0, 1], ml=82,
                                     zlabel='share of the group\'s images in the set', rowlabel='SEBA set',
                                     collabel='class group (validation)', counts=c)
            v = vals[0, lp[(a, b)]]
            key = f'cm.{sites[a]}_{sites[b]}.r{r}'
            facts[f'{key}.sizes'] = ', '.join(f'{int(x):,}' for x in c[:r].sum(1))
            facts[f'{key}.n_unassigned'] = f'{int(c[r].sum()):,}'
            facts[f'{key}.lambda_r_minus_1'] = f'{v[r - 1]:.3f}'
            facts[f'{key}.gap'] = f'{v[r - 1] - v[r]:.3f}'
    out['q_classmix'] = cm

    # ---------------- q_va: r = 2 sets vs animal / non-animal
    ref = np.isin(grp, ANIMAL_GROUPS).astype(np.int64)        # 1 = animal
    va = dict(definition=f'animal = class groups {", ".join(groups[g] for g in ANIMAL_GROUPS)} '
                         f'({int(ref.sum())} images); non-animal = the rest ({int(N - ref.sum())} images)',
              by_mode={})
    ne_series = []
    for mi in range(M):
        sc, mu, lo, hi, ce, ne, af1 = [], [], [], [], [], [], []
        for s in range(S):
            lab = LAB[mi, cell(s), ri[2]]
            x, nl = matched(lab, ref, 2, 2, perms)
            c = ceiling(lab, ref, 2, 2)
            sc.append(x); mu.append(nl.mean()); lo.append(np.percentile(nl, 2.5)); hi.append(np.percentile(nl, 97.5))
            ce.append(c); ne.append((x - nl.mean()) / (c - nl.mean()) if c > nl.mean() else np.nan)
            af1.append([float(ref[lab == j].mean()) if (lab == j).any() else np.nan for j in range(2)])
        va['by_mode'][str(mi)] = dict(score=sc, null_mean=mu, null_lo=lo, null_hi=hi, ceiling=ce, norm_excess=ne,
                                      animal_frac=af1)
        ne_series.append(dict(name=modes[mi], y=ne, color=SETC[mi]))
        if mi == 0:
            va['lines'] = dict(x=list(range(S)), xticklabels=sites, xlabel='site (single-cell block)',
                               ylabel='matched Jaccard vs animal / non-animal (r = 2)', yrange=[0, 1],
                               series=[dict(name='score', y=sc, band=[lo, hi]),
                                       dict(name='size ceiling', y=ce, dash=True, color='#8d8c86')])
            k = int(np.argmax(ne))
            facts.update({'va.r2.max_score': f2(max(sc)), 'va.r2.max_score_site': sites[int(np.argmax(sc))],
                          'va.r2.max_norm_excess': f2(ne[k]), 'va.r2.max_norm_excess_site': sites[k],
                          'va.r2.L1_score': f2(sc[0]), 'va.r2.L12_score': f2(sc[-1]),
                          'va.r2.L12_norm_excess': f2(ne[-1]), 'va.r2.null_hi_max': f2(max(hi)),
                          'va.r2.ceiling_min': f2(min(ce)), 'va.r2.ceiling_max': f2(max(ce)),
                          'va.n_animal': f'{int(ref.sum()):,}', 'va.n_other': f'{int(N - ref.sum()):,}'})
    va['lines_norm_excess'] = dict(x=list(range(S)), xticklabels=sites, xlabel='site (single-cell block)',
                                   ylabel='(score − null mean) / (ceiling − null mean)',
                                   yrange=[float(min(-0.2, np.floor(np.nanmin([x['y'] for x in ne_series]) * 5) / 5)), 1],
                                   series=ne_series, refline=dict(y=0, label='null mean'))
    out['q_va'] = va
    print('va', f'{time.time() - t0:.0f}s', flush=True)

    # ---------------- q_imgstats
    X = image_stats(images)                                     # (N, 12)
    Xz = zcols(X)
    # within-class: subtract the fine-class mean (100 classes x 20 images) from both sides
    def wc(A):
        A = A.astype(np.float64).copy()
        for c in np.unique(fine):
            A[fine == c] -= A[fine == c].mean(0)
        return A
    Xw = zcols(wc(X))
    rngp = np.random.default_rng(SEED + 1)
    pidx = np.stack([rngp.permutation(N) for _ in range(N_PERM)])
    corr_psi, corr_psi_wc, r2_psi, corr_seba, nullmax = [], [], [], [], []
    for s in range(S):
        pi = cell(s)
        psi = vecs[0, pi].T.astype(np.float64) * (scale[0, pi].astype(np.float64) / 32767.0)   # (N, 8)
        P = zcols(psi)
        corr_psi.append(Xz.T @ P)                                                             # (12, 8)
        corr_psi_wc.append(Xw.T @ zcols(wc(psi)))
        # rank-aware basis of the centred statistics (luminance and border-centre Y are linear in the RGB ones)
        U, sv_, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
        U = U[:, sv_ > sv_[0] * 1e-10]
        pc = psi - psi.mean(0)
        r2 = ((U.T @ pc) ** 2).sum(0) / (pc ** 2).sum(0)
        ref = 1 - ((pc - (X - X.mean(0)) @ np.linalg.lstsq(X - X.mean(0), pc, rcond=None)[0]) ** 2).sum(0) / (pc ** 2).sum(0)
        assert np.allclose(r2, ref, atol=1e-8), (r2, ref)
        r2_psi.append(r2)
        sv = seba_vectors(psi, rot[0, pi, ri[4]], perm[0, pi, ri[4]], 4)
        corr_seba.append(Xz.T @ zcols(sv))                                                    # (12, 4)
        nullmax.append(np.abs(np.einsum('pnk,nl->pkl', Xz[pidx], P)).max((1, 2)))            # (N_PERM,)
    corr_psi, corr_psi_wc, corr_seba = np.array(corr_psi), np.array(corr_psi_wc), np.array(corr_seba)
    thr = float(np.percentile(np.max(nullmax, 0), 95))        # family-wise over the 12 cells x 12 stats x 8 psi
    statrows = [STAT_LABELS[k] for k in STATS]
    heat = lambda Z, zl, rr=statrows, rl='image statistic', zr=(0, 1): dict(
        rows=rr, cols=sites, z=Z, zlabel=zl, zrange=list(zr), rowlabel=rl, collabel='site (single-cell block)', ml=180)
    mx = np.abs(corr_psi).max(2).T
    zmax = float(np.ceil(max(mx.max(), np.abs(corr_seba).max()) * 10) / 10)
    out['q_imgstats'] = dict(
        stats=STATS, stat_labels=STAT_LABELS, definitions=STAT_DEFS, pixel_geometry=PIX,
        per_image={k: np.round(X[:, i], 4).tolist() for i, k in enumerate(STATS)},
        heat_maxabs_psi=heat(mx, 'max over k = 1..8 of |corr(statistic, ψk)|', zr=(0, zmax)),
        heat_maxabs_psi_wc=heat(np.abs(corr_psi_wc).max(2).T, 'max over k = 1..8 of |within-class corr(statistic, ψk)|', zr=(0, zmax)),
        heat_maxabs_seba=heat(np.abs(corr_seba).max(2).T, 'max over the 4 sets of |corr(statistic, SEBA vector)| (r = 4)', zr=(0, zmax)),
        heat_r2_psi=heat(np.array(r2_psi).T, 'R² of ψk on the 12 image statistics (least-squares fit)',
                         rr=[f'ψ{k}' for k in range(1, 9)], rl='eigenfunction'),
        corr_psi=corr_psi, corr_seba_r4=corr_seba, thr_global=thr,
        lines_maxabs_psi=dict(x=list(range(S)), xticklabels=sites, xlabel='site (single-cell block)',
                              ylabel='max over k of |corr(statistic, ψk)|', yrange=[0, zmax], colors='categorical',
                              series=[dict(name=STAT_LABELS[k], y=mx[STATS.index(k)]) for k in
                                      ('luminance', 'saturation', 'contrast', 'bc_lum', 'hf_frac', 'mean_b')],
                              refline=dict(y=round(thr, 3), label='permutation null 95 % (family-wise)')))
    top = np.unravel_index(np.argmax(mx), mx.shape)
    facts.update({'img.thr_global': f2(thr), 'img.max': f2(mx.max()), 'img.max_stat': statrows[top[0]],
                  'img.max_site': sites[top[1]], 'img.L12_max': f2(mx[:, -1].max()),
                  'img.L12_max_stat': statrows[int(np.argmax(mx[:, -1]))],
                  'img.wc_max': f2(np.abs(corr_psi_wc).max())})
    print('imgstats', f'{time.time() - t0:.0f}s', flush=True)

    # ---------------- q_decod (earlier run's protocol)
    dec = json.loads((ROOT / 'data' / 'dinov2s' / 'decodability.json').read_text())
    c = dec['cls']
    out['q_decod'] = dict(protocol=dec['protocol'], lines=dict(
        x=c['layers'], xticklabels=[f'L{l}' for l in c['layers']], xlabel='site (CLS after block l)',
        ylabel='held-out accuracy', yrange=[0, 1],
        series=[dict(name='ridge, 100 classes', y=c['ridge']['fine'], color=SETC[0]),
                dict(name='kNN, 100 classes', y=c['knn']['fine'], color=SETC[1]),
                dict(name='ridge, 18 superclasses', y=c['ridge']['super'], dash=True, color=SETC[0]),
                dict(name='kNN, 18 superclasses', y=c['knn']['super'], dash=True, color=SETC[1])]))
    facts.update({'decod.ridge_fine.L1': f"{100 * c['ridge']['fine'][0]:.0f}%", 'decod.ridge_fine.L12': f"{100 * c['ridge']['fine'][-1]:.0f}%",
                  'decod.knn_fine.L1': f"{100 * c['knn']['fine'][0]:.0f}%", 'decod.knn_fine.L12': f"{100 * c['knn']['fine'][-1]:.0f}%"})

    # ---------------- q_kernel
    kap = np.array(meta['kappa'], dtype=float)[:, 0]        # (M, S); None -> nan
    mass = np.array(meta['nn_mass'], dtype=float)[:, 0]
    out['q_kernel'] = dict(
        kappa=dict(x=list(range(S)), xticklabels=sites, xlabel='site', ylabel='κ (median over images for perplexity rules)',
                   yrange=[0, float(np.ceil(np.nanmax(kap) / 5) * 5)],
                   series=[dict(name=modes[mi], y=kap[mi], color=SETC[mi]) for mi in range(1, M)]),
        mass=dict(x=list(range(S)), xticklabels=sites, xlabel='site', ylabel='median 15-NN kernel mass', yrange=[0, 1],
                  series=[dict(name=modes[mi], y=mass[mi], color=SETC[mi]) for mi in range(M)]))
    facts.update({'kernel.kappa_scale_min': f"{np.nanmin(kap[3]):.2g}", 'kernel.kappa_scale_max': f"{np.nanmax(kap[3]):.2g}",
                  'kernel.kappa_half_min': f"{np.nanmin(kap[4]):.3g}", 'kernel.kappa_half_max': f"{np.nanmax(kap[4]):.3g}",
                  'kernel.mass_knn_min': f"{100 * np.nanmin(mass[0]):.0f}%", 'kernel.mass_knn_max': f"{100 * np.nanmax(mass[0]):.0f}%"})

    facts.update({'n_perm': str(N_PERM), 'group_sizes': ', '.join(f'{g} {n}' for g, n in zip(groups, gsize))})
    q = {k: (v if k != 'q_imgstats' else v) for k, v in out.items()}
    q = r3(q)
    q['q_imgstats']['per_image'] = out['q_imgstats']['per_image']
    q['meta'] = dict(tag=tag, n=N, sites=sites, modes=modes, seba_r=R, n_perm=N_PERM, seed=SEED)
    q['facts'] = facts
    text = json.dumps(q, separators=(',', ':'), allow_nan=False, ensure_ascii=False)
    (SITE_DIR / 'assets').mkdir(exist_ok=True)
    (SITE_DIR / 'assets' / 'questions.json').write_text(text)
    print(f'wrote questions.json ({len(text) / 1e6:.2f} MB) in {time.time() - t0:.0f}s')
    print(json.dumps(facts, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(*sys.argv[1:])
