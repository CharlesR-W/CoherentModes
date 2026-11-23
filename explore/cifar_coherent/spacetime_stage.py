"""Everything one (checkpoint, rule) unit of the through-depth stage computes (SPEC_SPACETIME.md section 5).

`compute(ms, labels)` takes the L walks (L, N, N) float64 and returns a dict of arrays; it has no Modal
or file dependencies, so the toy in test_spacetime.py and the Modal unit run the same code.
Labels are used for validation columns only (class_share, nmi).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

import spacetime as S

SCALES = ('own', 'none')              # 'relative site units', 'native walk units'
FACTORS = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0, 16.0)
K = 16                                # modes stored and the low space that is read
KC = 24                               # modes computed (gaps past the cutoff)
R_OWN = 8                             # a site's own leading directions (own_share) and the scale rank
R_SETS = 8                            # rank of the post-hoc set readout
KSETS = (5, 10)
KTRADE = (4, 8, 16)
R_FRAME = 4


def quant(v):
    """Unit slices (K, L, N) -> int16 with full range per slice."""
    mx = np.abs(v).max(-1, keepdims=True).clip(1e-300)
    return np.round(32767 * v / mx).astype(np.int16), mx[..., 0].astype(np.float32)


def nmi(a, b):
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    j = np.zeros((ka, kb))
    np.add.at(j, (a, b), 1)
    j /= j.sum()
    pa, pb = j.sum(1), j.sum(0)
    nz = j > 0
    mi = (j[nz] * np.log(j[nz] / np.outer(pa, pb)[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(mi / max(np.sqrt(h(pa) * h(pb)), 1e-300))


def info_nonlazy(p, lab):
    """chi^2 information a partition keeps about one step of the non-lazy walk P = 2M - I."""
    k = int(lab.max()) + 1
    h = np.eye(k)[lab]
    sz = h.sum(0)
    keep = sz > 0
    s = h[:, keep].T @ p
    return float(((s ** 2).sum(1) / sz[keep]).sum() - 1)


def relabel_chain(labs, k):
    """Compose optimal (Hungarian) relabellings along the chain of sites."""
    out = [labs[0]]
    for l in range(1, len(labs)):
        c = np.zeros((k, k))
        np.add.at(c, (out[-1], labs[l]), 1)
        r, cc = linear_sum_assignment(-c)
        perm = np.empty(k, int)
        perm[cc] = r
        out.append(perm[labs[l]])
    return np.stack(out)


def set_scores(lab, ps, ceil, labels, pfro, k):
    """Per-site scores of one (L, N) labelling. Sites with label -1 get NaN."""
    nl = len(lab)
    out = dict(info=np.full(nl, np.nan), null=np.full(nl, np.nan), groups=np.zeros(nl), nmi=np.full(nl, np.nan))
    for l in range(nl):
        if lab[l].min() < 0:
            continue
        out['info'][l] = info_nonlazy(ps[l], lab[l])
        q = len(np.unique(lab[l]))
        out['null'][l] = (q - 1) / (len(lab[l]) - 1) * pfro[l]       # exact mean over same-sizes random relabellings
        out['groups'][l] = q
        out['nmi'][l] = nmi(lab[l], labels)
    ok = (lab[1:].min(1) >= 0) & (lab[:-1].min(1) >= 0)
    out['churn'] = np.where(ok, S.churn(lab), np.nan)
    re = relabel_chain(np.where(lab < 0, 0, lab), k)
    out['churn_matched'] = np.where(ok, S.churn(re), np.nan)
    out['ceiling'] = ceil
    return out


def frame_cos2(gk):
    """(L-1,) mean cos^2 of principal angles between the leading R_FRAME local directions of neighbouring sites."""
    fr = S.local_frames(gk, R_FRAME)
    return np.array([(np.linalg.svd(fr[l].T @ fr[l + 1], compute_uv=False) ** 2).mean() for l in range(len(gk) - 1)], 'f4')


def set_block(g, ps, ceil, labels, pfro, seed, prefix):
    """Both set readouts (shape-aligned 'a', shared coordinates 's') with scores, and the alignment links."""
    out = {prefix + 'link': S.aligned_rows(g[:, :, :R_SETS])[2].astype('f4')}
    for ki, k in enumerate(KSETS):
        for tag, align in (('a', True), ('s', False)):
            lab = S.depth_sets(g[:, :, :R_SETS], k, seed=seed, align=align)
            out[f'{prefix}labels_{tag}{ki}'] = lab.astype(np.int8)
            for key, v in set_scores(lab, ps, ceil[k], labels, pfro, k).items():
                out[f'{prefix}set_{tag}{ki}_{key}'] = np.asarray(v, 'f4')
    return out


def space_readings(ms, c, t, vals, g, lam, vec, labels_q, gaps):
    """Readings of one solved setting. g (L, N, KC) raw slices; returns dict of arrays."""
    nl = len(ms)
    gk = g[:, :, :K]
    w = S.depth_weight(gk)
    out = dict(vals=vals, resid=S.residual(ms, c, t, vals[:K], gk), weight=w.astype('f4'),
               adj_cos=S.adjacent_cos(gk).astype('f4'), retention=S.slice_retention(ms, gk).astype('f4'),
               own_share=S.span_share(gk, vec[:, :, 1:1 + R_OWN]).astype('f4'),
               class_share=S.span_share(gk, labels_q).astype('f4'),
               occupancy=S.occupancy(gk), strengths=S.local_strengths(gk).astype('f4'),
               recurrent=S.recurrent(gk)[0].astype('f4'),
               tradeoff=np.array([S.tradeoff(ms, c, g[:, :, :kk]) for kk in KTRADE]))
    out['frame_cos2'] = frame_cos2(gk)
    e, sw = S.scalar_control(gaps, t)
    out['scalar_energy'], out['scalar_weight'] = e, sw.astype('f4')
    return out


def compute(ms: np.ndarray, labels: np.ndarray, controls: bool = False, seed: int = 0, log=print) -> dict:
    nl, n, _ = ms.shape
    rng = np.random.default_rng(seed)
    lam, vec = S.site_eigs(ms)
    y = np.eye(int(labels.max()) + 1)[labels]
    y -= y.mean(0)
    labels_q = np.linalg.svd(y, full_matrices=False)[0][:, :y.shape[1] - 1]
    ps = 2 * ms - np.eye(n)
    plam = 2 * lam - 1                                             # non-lazy spectra, constant first
    pfro = (plam[:, 1:] ** 2).sum(1)                               # |P|_F^2 - 1
    res = dict(site_lam=lam[:, :33], factors=np.array(FACTORS), ksets=np.array(KSETS), ktrade=np.array(KTRADE))
    ceil = {k: np.sort(plam[:, 1:] ** 2, axis=1)[:, ::-1][:, :k - 1].sum(1) for k in KSETS}

    per = {}
    for si, mode in enumerate(SCALES):
        c = np.array([np.mean(1 - lam[l, 1:1 + R_OWN]) for l in range(nl)]) if mode == 'own' else np.ones(nl)
        gaps = (1 - lam[:, 1]) / c
        tstar = S.balance_direct(ms, c, eigs=(lam, vec))
        log(mode, 'c', np.round(c, 4), 't*', tstar)
        for fi, f in enumerate(FACTORS):
            t = tstar * f
            vals, g = S.modes(ms, c, t, k=KC)
            r = space_readings(ms, c, t, vals, g, lam, vec, labels_q, gaps)
            u = S.unit_slices(g[:, :, :K]).transpose(2, 0, 1)        # (K, L, N)
            r['vecs'], r['vmax'] = quant(u)
            r.update(set_block(g, ps, ceil, labels, pfro, seed, ''))
            r['t'] = t
            per[(si, fi)] = r
            log(mode, f'x{f:g}', 'vals', np.round(vals[:4], 4), 'resid', f'{r["resid"].max():.1e}')
        res[f'scales_{mode}'] = c
        res[f'tstar_{mode}'] = tstar
        res[f'balance_check_{mode}'] = abs(per[(si, FACTORS.index(1.0))]['vals'][0] / S.temporal_eigs(nl, tstar)[1] - 1)
    keys = [k for k in per[(0, 0)]]
    for key in keys:
        res[key] = np.stack([np.stack([per[(si, fi)][key] for fi in range(len(FACTORS))]) for si in range(len(SCALES))])

    # the paper's matrix, as is
    pv, pf = S.paper_modes(ms, k=KC)
    b = S.PAPER_B(nl)
    gp = pf * np.sqrt(b)[:, None, None]                               # orthonormal columns
    # residual of the generalized problem A f = lambda B f, in the B^{-1/2} metric
    af = np.zeros_like(pf)
    af[:-1] += np.matmul(ms[:-1], pf[1:])
    af[1:] += np.matmul(ms[:-1], pf[:-1])
    rr = (af - pf * b[:, None, None] * pv) / np.sqrt(b)[:, None, None]
    res.update(paper_vals=pv, paper_resid=np.sqrt((rr ** 2).sum((0, 1)))[:K],
               paper_weight=S.depth_weight(gp[:, :, :K]).astype('f4'), paper_adj_cos=S.adjacent_cos(pf[:, :, :K]).astype('f4'),
               paper_retention=S.slice_retention(ms, pf[:, :, :K]).astype('f4'),
               paper_own_share=S.span_share(pf[:, :, :K], vec[:, :, 1:1 + R_OWN]).astype('f4'),
               paper_class_share=S.span_share(pf[:, :, :K], labels_q).astype('f4'),
               paper_occupancy=S.occupancy(gp[:, :, :K]), paper_strengths=S.local_strengths(gp[:, :, :K]).astype('f4'),
               paper_recurrent=S.recurrent(gp[:, :, :K])[0].astype('f4'),
               paper_frame_cos2=frame_cos2(gp[:, :, :K]))
    res['paper_vecs'], res['paper_vmax'] = quant(S.unit_slices(pf[:, :, :K]).transpose(2, 0, 1))
    res.update(set_block(gp, ps, ceil, labels, pfro, seed, 'paper_'))

    # reference partitions: per-site spectral k-means (chain-matched), block averages (all sites, thirds)
    def spectral_labels(m_lam, m_vec, k):
        return S.kmeans(m_vec[:, 1:1 + R_SETS] * np.sqrt(n / R_SETS), k, seed=seed)[0]   # unweighted, as depth_sets

    def avg_labels(sites, k, c=None):
        gen = np.mean([(np.eye(n) - ms[l]) / (c[l] if c is not None else 1.0) for l in sites], 0)
        w, v = np.linalg.eigh(gen)
        return S.kmeans(v[:, 1:1 + R_SETS] * np.sqrt(n / R_SETS), k, seed=seed)[0]

    thirds = [list(range(i, min(i + 3, nl))) for i in range(0, nl, 3)]
    for ki, k in enumerate(KSETS):
        refs = {
            'site': relabel_chain([spectral_labels(lam[l], vec[l], k) for l in range(nl)], k),
            'avg': np.stack([avg_labels(range(nl), k)] * nl),
            'avg_rel': np.stack([avg_labels(range(nl), k, res['scales_own'])] * nl),
            'thirds': relabel_chain([lab for blk in thirds for lab in [avg_labels(blk, k)] * len(blk)], k),
        }
        for name, lab in refs.items():
            res[f'ref_{name}_labels{ki}'] = lab.astype(np.int8)
            for key, v in set_scores(lab, ps, ceil[k], labels, pfro, k).items():
                res[f'ref_{name}_set{ki}_{key}'] = np.asarray(v, 'f4')

    if controls:
        res.update(control_runs(ms, lam, vec, rng, log))
    return res


def control_runs(ms, lam, vec, rng, log=print):
    """Pilot-only checks of what is the network and what is the construction (relative units)."""
    nl, n, _ = ms.shape
    out = {}

    def run(mm, c, t=None, k=K):
        ts = S.balance_direct(mm, c)
        vals, g = S.modes(mm, c, ts if t is None else t, k=k)
        return ts, vals, g

    c8 = np.array([np.mean(1 - lam[l, 1:1 + R_OWN]) for l in range(nl)])
    ts, vals, g = run(ms, c8)
    proj_share = lambda ga, gb: float((np.einsum('lni,lnj->ij', ga, gb) ** 2).sum() / ga.shape[2])
    # 1. correspondence shuffle: same spectra at every site, images relabelled independently per site
    mp = S.permute_sites(ms, rng)
    tp, vp, gpm = run(mp, c8, t=ts)
    out.update(shuffle_tstar=S.balance_direct(mp, c8), shuffle_vals=vp, base_vals=vals,
               shuffle_adj_cos=S.adjacent_cos(gpm).astype('f4'), shuffle_occupancy=S.occupancy(gpm),
               shuffle_tradeoff=np.array(S.tradeoff(mp, c8, gpm)), base_tradeoff=np.array(S.tradeoff(ms, c8, g)))
    out['shuffle_scalar_energy'] = S.scalar_control((1 - lam[:, 1]) / c8, ts)[0]     # same bound for both
    log('shuffle first vals', vp[:3], 'base', vals[:3], 'scalar bound', out['shuffle_scalar_energy'])
    # 2. scale rank 4 and 16: same low space?
    for r in (4, 16):
        cr = np.array([np.mean(1 - lam[l, 1:1 + r]) for l in range(nl)])
        tr, vr, gr = run(ms, cr)
        out[f'rank{r}_tstar'], out[f'rank{r}_scales'], out[f'rank{r}_space_share'] = tr, cr, proj_share(g, gr)
        out[f'rank{r}_occupancy'] = S.occupancy(gr)
    # 3. leave one readout out / duplicate one: occupancy of the remaining sites and the shared space
    for name, sites in (('drop_pool', [l for l in range(nl) if l != nl - 2]), ('drop_logits', list(range(nl - 1))),
                        ('dup_logits', list(range(nl)) + [nl - 1])):
        td, vd, gd = run(ms[sites], c8[sites])
        keep = [i for i, l in enumerate(sites) if l < nl - 2][:nl - 2]
        ga, gb = g[:nl - 2], gd[keep]
        # compare the spaces on stem..b9 after renormalising each to an orthonormal basis there
        qa = np.linalg.qr(ga.reshape(-1, K))[0]
        qb = np.linalg.qr(gb.reshape(-1, K))[0]
        out[f'{name}_tstar'], out[f'{name}_occupancy'] = td, S.occupancy(gd)
        out[f'{name}_space_share'] = float(((qa.T @ qb) ** 2).sum() / K)
        out[f'{name}_restricted_smin'] = float(min(np.linalg.svd(ga.reshape(-1, K), compute_uv=False)[-1], np.linalg.svd(gb.reshape(-1, K), compute_uv=False)[-1]))
        log(name, 't*', td, 'share', out[f'{name}_space_share'])
    return out


def accept(res: dict) -> list[str]:
    """Numerical gates a unit must pass before it is marked complete (empty list = pass)."""
    bad = []
    K = res['weight'].shape[2]
    chk = lambda ok, msg: None if ok else bad.append(msg)
    chk(res['resid'].max() < 1e-8, f"residual {res['resid'].max():.2e}")
    chk(res['paper_resid'].max() < 1e-8, f"paper residual {res['paper_resid'].max():.2e}")
    for m in SCALES:
        chk(res[f'balance_check_{m}'] < 1e-6, f"balance {m} {res[f'balance_check_{m}']:.2e}")
    chk(np.allclose(res['occupancy'].sum(-1), K, atol=1e-8), 'occupancy does not sum to K')
    chk(np.allclose(res['recurrent'].sum(-1), K, atol=1e-3), 'recurrent strengths do not sum to K')
    chk((res['scalar_energy'] <= res['vals'][..., 0] + 1e-10).all(), 'scalar bound violated')
    a, b, t = res['tradeoff'][..., 0], res['tradeoff'][..., 1], res['t'][..., None]
    ksum = np.stack([res['vals'][..., :k].sum(-1) for k in res['ktrade']], -1)
    chk(np.allclose(a + t * b, ksum, rtol=1e-8, atol=1e-8), 'A + tB != sum of eigenvalues')
    chk((np.diff(a, axis=1) >= -1e-7).all() and (np.diff(b, axis=1) <= 1e-7).all(), 'trade-off not monotone')
    for key in ('vals', 'weight', 'adj_cos', 'retention', 'strengths', 'paper_vals', 'paper_weight'):
        chk(np.isfinite(res[key]).all(), f'non-finite {key}')
    return bad
