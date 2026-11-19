"""Transfer-operator pilot runner (PLAN section 6; design note TRANSFER.md).

  python transfer_pilot.py synthetic [--out DIR]          P8 at N_fit 1000 / 6000 (truth, estimates, control bands),
                                                          P8 independent pairing, P2, composition calibration chain
  python transfer_pilot.py curve --units out.json ...     one unit of the P8 sample-size curve (Modal or local)
  python transfer_pilot.py curve-local                    the whole curve locally (small grids only)
  python transfer_pilot.py real --grams DIR [...]         the real pilot (2,000 probe images, 12 pairs, 10 splits)
  python transfer_pilot.py real10k --grams DIR [...]      the 10,000-image set, 6,000 / 2,000 / 2,000, N curve
  python transfer_pilot.py smoke-real                     stand-in raw Grams shaped like the real inputs -> real
  python transfer_pilot.py plots                          diagnostic PNGs from the synthetic JSONs

Everything is label-free. Characterisation only: outputs state what was computed and the control value.
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer as tr  # noqa: E402

OUT = HERE / 'out' / 'transfer'
ESTIMATORS = ('linear', 'soft_ulam', 'nystrom', 'diffusion')
M_VALUES = (64, 128)
SIGMA = 0.35

# ------------------------------------------------------------------------------------------------ predeclared real design

FINAL = 64000
SITES = ['stem', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'pool', 'logits']
PAIRS = [  # (id, kind, (src_site, src_step), (dst_site, dst_step)) -- PLAN section 6, fixed in advance
    ('stem-b1', 'adjacent', ('stem', FINAL), ('b1', FINAL)),
    ('b4-b5', 'adjacent', ('b4', FINAL), ('b5', FINAL)),
    ('b8-b9', 'adjacent', ('b8', FINAL), ('b9', FINAL)),
    ('pool-logits', 'adjacent', ('pool', FINAL), ('logits', FINAL)),
    ('stem-b9', 'distant', ('stem', FINAL), ('b9', FINAL)),
    ('b3-b9', 'distant', ('b3', FINAL), ('b9', FINAL)),
    ('b6-logits', 'distant', ('b6', FINAL), ('logits', FINAL)),
    ('b1-b5', 'distant', ('b1', FINAL), ('b5', FINAL)),
    ('b5:1024-64000', 'time', ('b5', 1024), ('b5', FINAL)),
    ('b5:6400-64000', 'time', ('b5', 6400), ('b5', FINAL)),
    ('b9:1024-64000', 'time', ('b9', 1024), ('b9', FINAL)),
    ('b9:6400-64000', 'time', ('b9', 6400), ('b9', FINAL)),
]
TRIPLES = [  # composition discrepancy (builder's predeclared addition; TRANSFER.md section 5)
    ('stem-b4-b9', (('stem', FINAL), ('b4', FINAL), ('b9', FINAL))),
    ('b1-b5-b9', (('b1', FINAL), ('b5', FINAL), ('b9', FINAL))),
    ('b5:1024-6400-64000', (('b5', 1024), ('b5', 6400), ('b5', FINAL))),
    ('b9:1024-6400-64000', (('b9', 1024), ('b9', 6400), ('b9', FINAL))),
]
SPLIT_SEED = 20261115
SUPPORT_SPLITS = 8          # PLAN: supported if test corr > control 95% point in >= 8 of 10 splits
STABILITY_MIN = 0.8         # builder's provisional threshold for computing sets (TRANSFER.md section 6)
SEBA_R = (2, 3, 4, 6, 8)


def predeclared_splits(n, sizes, n_splits, seed=SPLIT_SEED):
    out = []
    for s in range(n_splits):
        o = np.random.default_rng([seed, n, s]).permutation(n)
        a, b = sizes[0], sizes[0] + sizes[1]
        out.append(dict(fit=np.sort(o[:a]), tune=np.sort(o[a:b]), test=np.sort(o[b:b + sizes[2]])))
    return out


# ------------------------------------------------------------------------------------------------ synthetic generators


def p8_truth(sigma, k=6):
    v = []
    for m in range(1, 10):
        v += [math.exp(-m * m * sigma * sigma / 2)] * 2
    return np.array(v[:k])


def make_chain(seed, N, s1=0.25, s2=0.25, widths=(48, 64, 96)):
    """Known-Markov chain theta -> phi = 2 theta + eta1 -> chi = 2 phi + eta2 (independent wrapped noise), each
    angle observed on a circle embedded by an independent isometry. Truth: T_ab sv exp(-m^2 s1^2/2),
    T_bc exp(-m^2 s2^2/2), T_ac exp(-m^2 (4 s1^2 + s2^2)/2), two per m; T_ac = T_bc T_ab in population."""
    import planted
    rng = np.random.default_rng([int(seed), 88])
    th = rng.uniform(0, 2 * np.pi, N)
    ph = np.mod(2 * th + s1 * rng.standard_normal(N), 2 * np.pi)
    ch = np.mod(2 * ph + s2 * rng.standard_normal(N), 2 * np.pi)
    Xs = []
    for ang, d in zip((th, ph, ch), widths):
        Q = planted.isometry(rng, 2, d)
        Xs.append(np.stack([np.cos(ang), np.sin(ang)], 1) @ Q.T)
    sv = lambda s: [math.exp(-m * m * s / 2) for m in (1, 2, 3)]  # noqa: E731
    return Xs, dict(angles=(th, ph, ch), sv_ab=sv(s1 ** 2), sv_bc=sv(s2 ** 2), sv_ac=sv(4 * s1 ** 2 + s2 ** 2))


def _bank_X(Xs, fit, labels, seed=0):
    return tr.FeatureBank({lab: tr.View.from_X(X, fit, lab) for lab, X in zip(labels, Xs)}, seed)


def _js(x):
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return _js(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _band(a):
    a = np.asarray(a, float)
    return dict(q05=np.nanquantile(a, .05, axis=0).tolist(), q50=np.nanquantile(a, .5, axis=0).tolist(),
                q95=np.nanquantile(a, .95, axis=0).tolist())


# ------------------------------------------------------------------------------------------------ P8 unit (curve + synthetic)


def p8_unit(n_fit, seed, m, estimator, n_perm=0, n_tune=1000, n_test=2000, sigma=SIGMA, level=None):
    """One P8 cell: fit / tune / test = n_fit / n_tune / n_test fresh images; label-free tuning; permutation control."""
    import planted
    N = n_fit + n_tune + n_test
    Xs, truth = planted.make_P8(seed, N=N, level=sigma if level is None else level)
    o = np.random.default_rng([seed, 31]).permutation(N)
    fit, tun, te = np.sort(o[:n_fit]), np.sort(o[n_fit:n_fit + n_tune]), np.sort(o[n_fit + n_tune:])
    pair = _bank_X(Xs, fit, 'ab', seed).pair('a', 'b')
    t0 = time.time()
    model, rows = tr.tune(pair, estimator, m, tun)
    out = dict(n_fit=n_fit, seed=seed, m=m, estimator=estimator, sigma=sigma, level=str(level),
               truth=p8_truth(sigma).tolist())
    if model is None:
        out.update(error='no table passes the occupancy gate')
        return out
    out.update(tr.evaluate(model, tun, te))
    out['grid'] = [dict(spec=list(r['spec']) if r['spec'] else None, eta=r['eta'], m_used=r['m_used'],
                        vamp2_tune=r['vamp2_tune']) for r in rows]
    if estimator in ('nystrom', 'diffusion', 'soft_ulam'):
        th, ph = truth['angle'][te], truth['angle2'][te]
        f, g = model.source_functions(te)[:, :2], model.dest_functions(te)[:, :2]
        out['src2_overlap_harmonic2'] = tr._overlap(f, np.stack([np.cos(2 * th), np.sin(2 * th)], 1))
        out['dst2_overlap_harmonic1'] = tr._overlap(g, np.stack([np.cos(ph), np.sin(ph)], 1))
    out['seconds_fit'] = time.time() - t0
    if n_perm:
        pc = tr.permutation_control(pair, estimator, m, tun, te, n_perm=n_perm, seed=seed)
        out['control'] = dict(n_perm=n_perm, test_corr=_band(pc['test_corr']), train_sv=_band(pc['train_sv']),
                              vamp2_test=_band(pc['vamp2_test'][:, None]))
    out['seconds'] = time.time() - t0
    return _js(out)


CURVE_N = (250, 500, 1000, 3000, 6000)


def curve_units(seeds=(0, 1, 2), n_perm=20):
    units = []
    for n in CURVE_N:
        for m in M_VALUES:
            for est in ESTIMATORS:
                for s in seeds:
                    units.append(dict(n_fit=n, seed=s, m=m, estimator=est, n_perm=n_perm if s == seeds[0] else 0))
    return units


# ------------------------------------------------------------------------------------------------ synthetic summary


def run_synthetic(out_dir, n_perm=50, include_6000=True):
    import planted
    res = dict(schema='transfer-synthetic-v1', note='characterisation; truth from the generators, no labels used',
               sigma=SIGMA, truth=p8_truth(SIGMA).tolist(), p8={}, p8_independent={}, p2={}, chain={})
    sizes = (1000, 6000) if include_6000 else (1000,)
    for n in sizes:
        for m in M_VALUES:
            for est in ESTIMATORS:
                if n >= 6000 and est == 'diffusion':
                    continue          # 6,000-image dense Sinkhorn + eigensolve: Modal curve only
                key = f'N{n}_m{m}_{est}'
                res['p8'][key] = p8_unit(n, 0, m, est, n_perm=n_perm if n <= 1000 else 20)
                r = res['p8'][key]
                f = lambda v: np.round(np.array([np.nan if x is None else x for x in v], float), 3)  # noqa: E731
                print(key, f(r.get('test_corr', [])), 'ctl q95', f(r.get('control', {}).get('test_corr', {}).get('q95', [])),
                      f"{r.get('seconds', 0):.0f}s", flush=True)
    for est in ESTIMATORS:
        for s in range(5):
            key = f'N1000_m64_{est}_seed{s}'
            res['p8_independent'][key] = p8_unit(1000, s, 64, est, level='independent')
    # P2: shared linear factors; oracle = population linear CCA after preprocessing (40,000-image draw)
    big, _ = planted.make_P2(0, N=40000, level=1.0, L=2)
    Z = [planted.preprocess(X) for X in big]
    Z = [z - z.mean(0) for z in Z]
    n = len(Z[0])
    W = tr._psd_inv_sqrt(Z[1].T @ Z[1] / n) @ (Z[1].T @ Z[0] / n) @ tr._psd_inv_sqrt(Z[0].T @ Z[0] / n)
    raw = [X - X.mean(0) for X in big]
    Wr = tr._psd_inv_sqrt(raw[1].T @ raw[1] / n) @ (raw[1].T @ raw[0] / n) @ tr._psd_inv_sqrt(raw[0].T @ raw[0] / n)
    res['p2']['oracle_normalised'] = np.linalg.svd(W, compute_uv=False)[:6].tolist()
    res['p2']['raw_latent'] = np.linalg.svd(Wr, compute_uv=False)[:6].tolist()
    N = 3000
    Xs, truth = planted.make_P2(1, N=N, level=1.0, L=2)
    o = np.random.default_rng(1).permutation(N)
    fit, tun, te = np.sort(o[:1000]), np.sort(o[1000:2000]), np.sort(o[2000:])
    pair = _bank_X(Xs, fit, 'ab').pair('a', 'b')
    import stats
    for est in ESTIMATORS:
        mod, _ = tr.tune(pair, est, 64, tun)
        ev = tr.evaluate(mod, tun, te)
        ev['src3_captures_u'] = stats.subspace_overlap(mod.source_functions(np.arange(N))[:, :3],
                                                       truth['contrasts'])['captured_v']
        pc = tr.permutation_control(pair, est, 64, tun, te, n_perm=20, seed=1)
        ev['control_q95'] = pc['q95'].tolist()
        res['p2'][est] = _js(ev)
        print('P2', est, np.round(ev['test_corr'], 3), flush=True)
    res['chain'] = run_chain()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'synthetic.json').write_text(json.dumps(_js(res), indent=1))
    return res


def run_chain(m=32, factors=(0.25, 0.5, 1, 2, 4)):
    """Composition diagnostic on the known-Markov chain and on the broken chain (middle view's pairing destroyed):
    base chain selected at m, then only the intermediate basis varied with endpoints, eta and absolute bandwidths
    frozen (transfer.composition_sweep). Characterisation only; the sweep does not identify error components."""
    N = 3000
    Xs, ct = make_chain(0, N)
    o = np.random.default_rng(1).permutation(N)
    fit, tun, te = np.sort(o[:1000]), np.sort(o[1000:2000]), np.sort(o[2000:])
    out = dict(truth=dict(sv_ab=ct['sv_ab'], sv_bc=ct['sv_bc'], sv_ac=ct['sv_ac']), m=m, factors=list(factors),
               n_fit=1000, n_tune=1000, n_test=1000)
    for case in ('markov', 'broken'):
        Ys = list(Xs)
        if case == 'broken':
            Ys[1] = Xs[1][np.random.default_rng(5).permutation(N)]
        bank = _bank_X(Ys, fit, 'abc')
        for est in ('nystrom', 'diffusion', 'soft_ulam'):
            sw = tr.composition_sweep(bank, 'abc', est, m, tun, te, factors)
            out[f'{case}_{est}'] = _js(sw)
            for r in sw['rows']:
                if 'unavailable' in r:
                    print('chain', case, est, 'm_b', r['m_b_requested'], 'unavailable:', r['unavailable'], flush=True)
                else:
                    print('chain', case, est, 'dims', r['dims'], 'partial', round(r['partial']['topk_rel'], 3),
                          'reg', round(r['regulariser']['topk_rel'], 3), 'heldout resid',
                          round(r['heldout_residual_crossmoment']['topk_rel'], 3), flush=True)
    return out


# ------------------------------------------------------------------------------------------------ real pilot

TEMPLATE_2000 = 'raw_flat_{step}_{si}.npy'          # SCHEMA.md section 6 (builder A): float64, probe order
TEMPLATE_10K = 'raw10k_flat_{step}_{si}.npy'        # float32, CIFAR-10 test-set order, final checkpoint only
INDEX_10K = 'raw10k_index.npz'                      # test_index (0..9999), probe_positions, labels (labels unused)
FINAL_PAIRS = [p[0] for p in PAIRS if p[2][1] == FINAL and p[3][1] == FINAL]


def gram_path(grams, template, site, step):
    return Path(grams) / template.format(site=site, si=SITES.index(site), step=step)


def load_R(grams, template, site, step, mmap=True):
    return np.load(gram_path(grams, template, site, step), mmap_mode='r' if mmap else None)


def preflight(grams, template, cells, n_expected, index_file=None):
    """Check every input before computing: file exists, square (n_expected x n_expected), finite positive diagonal,
    symmetric on a sample; for the 10k set the image-id file. Raises with the full list of problems."""
    problems = []
    for site, step in cells:
        p = gram_path(grams, template, site, step)
        if not p.exists():
            problems.append(f'missing {p}')
            continue
        R = np.load(p, mmap_mode='r')
        if R.ndim != 2 or R.shape != (n_expected, n_expected):
            problems.append(f'{p}: shape {R.shape}, expected ({n_expected}, {n_expected})')
            continue
        idx = np.linspace(0, n_expected - 1, min(n_expected, 50)).astype(int)
        d = np.asarray(np.diagonal(R))
        if not np.all(np.isfinite(d)) or d.min() <= 0:
            problems.append(f'{p}: diagonal not finite positive')
        blk = np.asarray(R[np.ix_(idx, idx)], dtype=np.float64)
        if np.abs(blk - blk.T).max() > 1e-5 * max(1.0, np.abs(blk).max()):
            problems.append(f'{p}: not symmetric')
    if index_file is not None:
        ip = Path(grams) / index_file
        if not ip.exists():
            problems.append(f'missing {ip}')
        else:
            z = np.load(ip)
            ti = np.asarray(z['test_index'])
            if len(ti) != n_expected or len(np.unique(ti)) != n_expected:
                problems.append(f'{ip}: test_index has {len(ti)} entries ({len(np.unique(ti))} unique), '
                                f'expected {n_expected}')
    if problems:
        raise RuntimeError('preflight failed:\n  ' + '\n  '.join(problems))
    return dict(cells=len(cells), n=n_expected)


def _pair_cells(pair_ids, triples=()):
    cells = set()
    for pid, _, a, b in PAIRS:
        if pid in pair_ids:
            cells |= {a, b}
    for tid, vs in TRIPLES:
        if tid in triples:
            cells |= set(vs)
    return sorted(cells, key=lambda c: (c[1], SITES.index(c[0])))


def real_unit(Ra, Rb, split, split_id, estimators=ESTIMATORS, m_values=M_VALUES, n_perm=200, n_perm_fit=50,
              k=tr.K_FUNCS):
    """One (pair, split): tuned fit at each fixed m, evaluation, permutation control (all splits) and
    training-pairing disruption (fit only); plus the jointly selected m (`<est>_mjoint`: the m with the larger
    tuning VAMP-2, the selection repeated inside every permutation, which share their permutation sequence across
    m). Source / destination functions on all images are kept for stability (on shared held-out images) and for
    the transductive summaries."""
    fit, tun, te = split['fit'], split['tune'], split['test']
    bank = tr.FeatureBank.from_R({'a': Ra, 'b': Rb}, fit, seed=split_id)
    pair = bank.pair('a', 'b')
    allidx = np.arange(len(Ra))
    res = {}
    for est in estimators:
        ctl = {}
        for m in m_values:
            t0 = time.time()
            model, _ = tr.tune(pair, est, m, tun)
            key = f'{est}_m{m}'
            if model is None:
                res[key] = dict(error='no table passes the occupancy gate')
                continue
            ev = tr.evaluate(model, tun, te, k)
            ev['src_funcs'] = model.source_functions(allidx)[:, :k].astype(np.float32)
            ev['dst_funcs'] = model.dest_functions(allidx)[:, :k].astype(np.float32)
            if n_perm:
                pc = tr.permutation_control(pair, est, m, tun, te, n_perm=n_perm, seed=split_id)
                ev['control_q95'] = pc['q95']
                ev['control_vamp2_q95'] = float(np.nanquantile(pc['vamp2_test'], .95))
                ev['control_train_sv_q50'] = np.nanquantile(pc['train_sv'], .5, axis=0)
                ctl[(m, 'all')] = pc
            if n_perm_fit:
                pf = tr.permutation_control(pair, est, m, tun, te, n_perm=n_perm_fit, seed=split_id, mode='fit_only')
                ev['fitonly_q95'] = pf['q95']
                ctl[(m, 'fit_only')] = pf
            ev['seconds'] = time.time() - t0
            res[key] = ev
        done = [m for m in m_values if 'error' not in res.get(f'{est}_m{m}', {'error': 1})]
        if len(done) > 1:
            best = max(done, key=lambda m: res[f'{est}_m{m}']['selected']['vamp2_tune'])
            ev = dict(res[f'{est}_m{best}'], m_selected=best)
            for mode, field in (('all', 'control_q95'), ('fit_only', 'fitonly_q95')):
                if all((m, mode) in ctl for m in done):
                    t = np.stack([ctl[(m, mode)]['vamp2_tune'] for m in done])           # (len(done), n_perm)
                    pick = np.nanargmax(np.where(np.isnan(t), -np.inf, t), axis=0)
                    corr = np.stack([ctl[(m, mode)]['test_corr'] for m in done])[pick, np.arange(t.shape[1])]
                    ev[field] = np.nanquantile(corr, .95, axis=0)
                    if mode == 'all':
                        v2 = np.stack([ctl[(m, mode)]['vamp2_test'] for m in done])[pick, np.arange(t.shape[1])]
                        ev['control_vamp2_q95'] = float(np.nanquantile(v2, .95))
                        ev['control_m_selected'] = [done[i] for i in pick]
            res[f'{est}_mjoint'] = ev
    return res


def real_triple_unit(Rs, split, split_id, estimators=('nystrom', 'soft_ulam', 'diffusion'), m=64,
                     factors=(0.25, 0.5, 1, 2, 4)):
    fit, tun, te = split['fit'], split['tune'], split['test']
    bank = tr.FeatureBank.from_R(dict(zip('abc', Rs)), fit, seed=split_id)
    return {est: tr.composition_sweep(bank, 'abc', est, m, tun, te, factors) for est in estimators}


def _stack(units, key, field, k):
    rows = []
    for u in units:
        r = u.get(key, {})
        v = r.get(field)
        rows.append(np.full(k, np.nan) if v is None else np.pad(np.asarray(v, float), (0, k - len(v)),
                                                               constant_values=np.nan)[:k])
    return np.array(rows)


def heldout_stability(Fs, splits, q):
    """Mean over split pairs (i, j) of the subspace overlap (mean cos^2, constant removed) of the two splits'
    leading q functions, compared only on the images held out from both fits (not in fit_i or fit_j)."""
    import stats
    vals = []
    for i in range(len(Fs)):
        for j in range(i + 1, len(Fs)):
            shared = np.setdiff1d(np.arange(Fs[i].shape[0]), np.union1d(splits[i]['fit'], splits[j]['fit']))
            if len(shared) > q + 1:
                vals.append(stats.subspace_overlap(Fs[i][shared, :q], Fs[j][shared, :q])['mean_cos2'])
    return dict(q=q, mean_pairwise_cos2=float(np.mean(vals)) if vals else None, n_pairs=len(vals),
                images='held out from both fits of each split pair')


TRANSDUCTIVE = ('transductive exploratory summary: consensus span and SEBA sets computed on all images after '
                'fitting; held-out correlations validate the functions, not these partitions')


def aggregate_pair(units, ref, linear_key, splits, k=tr.K_FUNCS):
    """Across the predeclared splits: support (8 of 10 rule), held-out stability of each side's leading supported
    span, agreement with M_b M_a, Mbar and linear CCA (all images; transductive), and per-side SEBA sets where that
    side is stable (a source-destination joint table only when both sides are)."""
    import stats
    keys = sorted({kk for u in units for kk in u})
    out = {}
    n_splits = len(units)
    need = math.ceil(SUPPORT_SPLITS * n_splits / 10)
    for key in keys:
        tc = _stack(units, key, 'test_corr', k)
        q95 = _stack(units, key, 'control_q95', k)
        ok = tc > q95
        n_sup = np.nansum(ok, 0).astype(int)
        supported = (n_sup >= need).tolist()
        q = 0
        while q < k and supported[q]:
            q += 1
        r = dict(train_sv=_stack(units, key, 'train_sv', k), test_corr=tc, control_q95=q95,
                 fitonly_q95=_stack(units, key, 'fitonly_q95', k),
                 vamp2_test=[u.get(key, {}).get('vamp2_test') for u in units],
                 control_vamp2_q95=[u.get(key, {}).get('control_vamp2_q95') for u in units],
                 n_support_splits=n_sup, supported=supported, leading_supported=q,
                 support_rule=f'>= {need} of {n_splits}',
                 selected=[u.get(key, {}).get('selected') for u in units],
                 m_selected=[u.get(key, {}).get('m_selected') for u in units],
                 occupancy=[u.get(key, {}).get('occupancy') for u in units])
        F = [u[key]['src_funcs'] for u in units if 'src_funcs' in u.get(key, {})]
        G = [u[key]['dst_funcs'] for u in units if 'dst_funcs' in u.get(key, {})]
        if F and len(F) == n_splits:
            qq = max(q, 1)
            r['stability'] = dict(src=heldout_stability(F, splits, qq), dst=heldout_stability(G, splits, qq))
            agree = {}
            for nm, ref_span, funcs in (('Q_src', ref['Q_src'], F), ('Q_dst', ref['Q_dst'], G),
                                        ('Mbar_src', ref['Mbar'], F), ('Mbar_dst', ref['Mbar'], G)):
                ov = [stats.subspace_overlap(f[:, :k], ref_span[:, :k]) for f in funcs]
                agree[nm] = dict(mean_cos2_median=float(np.median([o['mean_cos2'] for o in ov])),
                                 chance=ov[0]['chance'])
            if linear_key in keys and not key.startswith('linear'):
                L = [u[linear_key]['src_funcs'] for u in units if 'src_funcs' in u.get(linear_key, {})]
                ov = [stats.subspace_overlap(f[:, :k], l[:, :k])['mean_cos2'] for f, l in zip(F, L)]
                agree['linear_src'] = dict(mean_cos2_median=float(np.median(ov)))
            agree['scope'] = 'all images (transductive)'
            r['agreement'] = agree
            stable = {side: q >= 1 and r['stability'][side]['mean_pairwise_cos2'] is not None and
                      r['stability'][side]['mean_pairwise_cos2'] >= STABILITY_MIN for side in ('src', 'dst')}
            r['sets'] = transfer_sets(F if stable['src'] else None, G if stable['dst'] else None, q) \
                if any(stable.values()) else None
        out[key] = r
    return out


def transfer_sets(F, G, q):
    """Transductive exploratory summary. Per stable side: SEBA on the consensus span (top-q eigenvectors of the
    mean over splits of the projectors onto each split's leading q functions, constant removed; all images), with
    the constant prepended. A side passed as None (not stable) gets no sets. The source x destination joint table
    (matching sets through the images, not by column index) is given only when both sides have sets."""
    import torch
    from seba import seba, seba_labels
    import stats

    def consensus(Fs):
        n = Fs[0].shape[0]
        Pm = np.zeros((n, n))
        for f in Fs:
            Qf = stats._const_free_orthonormal(f[:, :q])
            Pm += Qf @ Qf.T
        Pm /= len(Fs)
        lam, U = np.linalg.eigh(Pm)
        return np.concatenate([np.ones((n, 1)) / math.sqrt(n), U[:, ::-1][:, :q]], 1), lam[::-1][:q]
    out = dict(status=TRANSDUCTIVE, sides=[])
    for side, Fs in (('src', F), ('dst', G)):
        if Fs is None:
            continue
        out['sides'].append(side)
        V, lam = consensus(Fs)
        out[f'{side}_consensus_eig'] = lam.tolist()
        for r in SEBA_R:
            if r - 1 > q:
                continue
            lab = seba_labels(seba(torch.from_numpy(V), r)).numpy().astype(np.int8)
            out.setdefault(f'{side}_labels', {})[str(r)] = lab
    if 'src' in out['sides'] and 'dst' in out['sides']:
        for r in SEBA_R:
            if str(r) in out.get('src_labels', {}):
                a, b = out['src_labels'][str(r)], out['dst_labels'][str(r)]
                out.setdefault('joint_table', {})[str(r)] = [[int(((a == i) & (b == j)).sum())
                                                              for j in range(-1, r)] for i in range(-1, r)]
    return out


def real_units_list(n_splits=10, pairs=None, triples=True):
    """The (kind, id, split) work units of the real pilot, plus one reference unit per pair."""
    pair_ids = [p[0] for p in PAIRS if not pairs or p[0] in pairs]
    triple_ids = [t[0] for t in TRIPLES] if triples else []
    units = [dict(kind='pair', id=p, split=i) for p in pair_ids for i in range(n_splits)]
    units += [dict(kind='reference', id=p, split=-1) for p in pair_ids]
    units += [dict(kind='triple', id=t, split=i) for t in triple_ids for i in range(n_splits)]
    return units, pair_ids, triple_ids


def run_real_unit(unit, grams, template, n_images=2000, sizes=(1000, 500, 500), n_splits=10, n_perm=200,
                  n_perm_fit=50, estimators=ESTIMATORS, m_values=M_VALUES):
    """Execute one unit of `real_units_list` (the same code locally and on Modal)."""
    splits = predeclared_splits(n_images, sizes, n_splits)
    t0 = time.time()
    if unit['kind'] == 'pair':
        _, _, (sa, ta), (sb, tb) = next(p for p in PAIRS if p[0] == unit['id'])
        Ra = np.asarray(load_R(grams, template, sa, ta)); Rb = np.asarray(load_R(grams, template, sb, tb))
        out = real_unit(Ra, Rb, splits[unit['split']], unit['split'], estimators, m_values, n_perm, n_perm_fit)
    elif unit['kind'] == 'reference':
        _, _, (sa, ta), (sb, tb) = next(p for p in PAIRS if p[0] == unit['id'])
        allidx = np.arange(n_images)
        out = tr.diffusion_reference(tr.View.from_R(np.asarray(load_R(grams, template, sa, ta)), allidx),
                                     tr.View.from_R(np.asarray(load_R(grams, template, sb, tb)), allidx))
    else:
        views = next(v for t, v in TRIPLES if t == unit['id'])
        Rs = [np.asarray(load_R(grams, template, s_, t_)) for s_, t_ in views]
        out = real_triple_unit(Rs, splits[unit['split']], unit['split'])
    return dict(unit=unit, result=out, seconds=time.time() - t0)


def assemble_real(parts, n_images=2000, sizes=(1000, 500, 500), n_splits=10, n_perm=200, n_perm_fit=50,
                  estimators=ESTIMATORS, m_values=M_VALUES, tag='real'):
    """Combine unit outputs (list of run_real_unit dicts) into the site JSON (schema transfer-pilot-v2)."""
    splits = predeclared_splits(n_images, sizes, n_splits)
    result = real_header(tag, n_images, sizes, n_splits, n_perm, n_perm_fit, estimators, m_values)
    by = {}
    for p in parts:
        by[(p['unit']['kind'], p['unit']['id'], p['unit']['split'])] = p
    result['unit_seconds'] = float(sum(p['seconds'] for p in parts))
    for pid, kind, (sa, ta), (sb, tb) in PAIRS:
        units = [by.get(('pair', pid, i)) for i in range(n_splits)]
        ref = by.get(('reference', pid, -1))
        if ref is None or any(u is None for u in units):
            if ref is not None or any(u is not None for u in units):
                result.setdefault('incomplete', []).append(pid)
            continue
        agg = aggregate_pair([u['result'] for u in units], ref['result'], f'linear_m{m_values[0]}', splits)
        result['pairs'].append(dict(id=pid, kind=kind, src=dict(site=sa, step=ta), dst=dict(site=sb, step=tb),
                                    reference=dict(Q_sv=ref['result']['Q_sv'], Mbar_eig=ref['result']['Mbar_eig']),
                                    results=agg))
    for tid, views in TRIPLES:
        per = [by.get(('triple', tid, i)) for i in range(n_splits)]
        if all(p is not None for p in per):
            result['composition'].append(dict(id=tid, views=[dict(site=s_, step=t_) for s_, t_ in views],
                                              per_split=[p['result'] for p in per]))
    return _js(result)


def run_real(grams, template, out_dir, n_images=2000, sizes=(1000, 500, 500), n_splits=10, n_perm=200,
             n_perm_fit=50, pairs=None, triples=True, estimators=ESTIMATORS, m_values=M_VALUES, tag='real'):
    units, pair_ids, triple_ids = real_units_list(n_splits, pairs, triples)
    preflight(grams, template, _pair_cells(pair_ids, triple_ids), n_images)
    splits = predeclared_splits(n_images, sizes, n_splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'splits_{n_images}.json').write_text(json.dumps(
        dict(seed=SPLIT_SEED, sizes=sizes, splits=[{k: v.tolist() for k, v in s_.items()} for s_ in splits])))
    parts = []
    for u in units:
        parts.append(run_real_unit(u, grams, template, n_images, sizes, n_splits, n_perm, n_perm_fit, estimators,
                                   m_values))
        print(u, f"{parts[-1]['seconds']:.0f}s", flush=True)
    result = assemble_real(parts, n_images, sizes, n_splits, n_perm, n_perm_fit, estimators, m_values, tag)
    (out_dir / 'transfer_pilot.json').write_text(json.dumps(result))
    return result


def real_header(tag, n_images, sizes, n_splits, n_perm, n_perm_fit, estimators, m_values):
    return dict(schema='transfer-pilot-v2', status='pilot', tag=tag, n_images=n_images,
                split_sizes=list(sizes), n_splits=n_splits, n_perm=n_perm, n_perm_fit_only=n_perm_fit,
                estimators=list(estimators), m_values=list(m_values), k=tr.K_FUNCS,
                eta_grid=list(tr.ETA_GRID), c_grid=list(tr.C_GRID), occupancy_min=tr.OCCUPANCY_MIN,
                stability_min=STABILITY_MIN, sets_status=TRANSDUCTIVE, pairs=[], composition=[])


def real10k_units_list(pairs=None, curve=(500, 1000, 2000, 6000)):
    pair_ids = list(pairs or FINAL_PAIRS)
    bad = [p for p in pair_ids if p not in FINAL_PAIRS]
    if bad:
        raise ValueError(f'10k set has the final checkpoint only; not available: {bad}')
    return [dict(kind='pair10k', id=p, n_fit=int(n)) for p in pair_ids for n in curve], pair_ids


def run_real10k_unit(unit, grams, template, n_images=10000, sizes=(6000, 2000, 2000), n_perm=50,
                     estimators=ESTIMATORS, m_values=M_VALUES):
    """One (pair, N_fit) cell of the 10k curve: nested fit subset of the predeclared fit split; fixed tune / test."""
    split = predeclared_splits(n_images, sizes, 1)[0]
    order = np.random.default_rng([SPLIT_SEED, 1]).permutation(split['fit'])
    _, _, (sa, ta), (sb, tb) = next(p for p in PAIRS if p[0] == unit['id'])
    t0 = time.time()
    Ra = load_R(grams, template, sa, ta); Rb = load_R(grams, template, sb, tb)
    sp = dict(fit=np.sort(order[:unit['n_fit']]), tune=split['tune'], test=split['test'])
    u = real_unit(Ra, Rb, sp, 0, estimators, m_values, n_perm, 0)
    res = {kk: {f: v for f, v in vv.items() if f not in ('src_funcs', 'dst_funcs')} for kk, vv in u.items()}
    return dict(unit=unit, result=res, seconds=time.time() - t0)


def assemble_real10k(parts, n_images=10000, sizes=(6000, 2000, 2000), curve=(500, 1000, 2000, 6000), n_perm=50):
    result = dict(schema='transfer-pilot-10k-v2', status='pilot', n_images=n_images, split_sizes=list(sizes),
                  curve=list(curve), n_perm=n_perm, checkpoint=FINAL, pairs=[],
                  unit_seconds=float(sum(p['seconds'] for p in parts)))
    by = {(p['unit']['id'], p['unit']['n_fit']): p['result'] for p in parts}
    for pid, kind, (sa, ta), (sb, tb) in PAIRS:
        rows = {str(n): by[(pid, n)] for n in curve if (pid, n) in by}
        if rows:
            result['pairs'].append(dict(id=pid, kind=kind, src=dict(site=sa, step=ta), dst=dict(site=sb, step=tb),
                                        curve=rows))
    return _js(result)


def run_real10k(grams, template, out_dir, n_images=10000, sizes=(6000, 2000, 2000), curve=(500, 1000, 2000, 6000),
                n_perm=50, estimators=ESTIMATORS, m_values=M_VALUES, pairs=None, index_file=INDEX_10K):
    """The 10,000-image set (final checkpoint only): one predeclared 6,000 / 2,000 / 2,000 split; N_fit curve on
    nested fit subsets. Default pairs: the eight final-checkpoint pairs (no time pairs exist at 10,000 images)."""
    units, pair_ids = real10k_units_list(pairs, curve)
    preflight(grams, template, _pair_cells(pair_ids), n_images, index_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for u in units:
        parts.append(run_real10k_unit(u, grams, template, n_images, sizes, n_perm, estimators, m_values))
        print(u, f"{parts[-1]['seconds']:.0f}s", flush=True)
    result = assemble_real10k(parts, n_images, sizes, curve, n_perm)
    (out_dir / 'transfer_pilot_10k.json').write_text(json.dumps(result))
    return result


# ------------------------------------------------------------------------------------------------ stand-ins


def make_standins(dir_, n=2000, seed=0, cells=None, template=TEMPLATE_2000, dtype=np.float64, index=False):
    """Raw uncentred Grams shaped like the real inputs (N x N per (site, step); builder A's file naming), from a
    synthetic chain: a shared circle angle multiplied site to site, Gaussian factors and per-site noise, a nonzero
    mean (so fit-only centring matters) and per-image gain (so normalisation matters). `cells` defaults to every
    cell the PAIRS and TRIPLES need; `index=True` also writes a 10k-style image-id file."""
    import planted
    rng = np.random.default_rng([seed, 4242])
    dir_.mkdir(parents=True, exist_ok=True)
    need = cells or sorted({v for _, _, a, b in PAIRS for v in (a, b)} | {v for _, vs in TRIPLES for v in vs})
    th = rng.uniform(0, 2 * np.pi, n)
    u = rng.standard_normal((n, 3))
    gain = np.exp(0.3 * rng.standard_normal(n))
    for site, step in need:
        depth = SITES.index(site) + (0 if step == FINAL else -3 if step == 6400 else -6)
        mult = 1 + max(depth, 0) % 3
        noise = 0.3 + 0.05 * abs(depth)
        Y = np.concatenate([np.cos(mult * th)[:, None], np.sin(mult * th)[:, None], 0.5 * u,
                            noise * rng.standard_normal((n, 11))], 1)
        Q = planted.isometry(rng, Y.shape[1], 64)
        X = gain[:, None] * (Y @ Q.T + 0.4)
        np.save(gram_path(dir_, template, site, step), (X @ X.T).astype(dtype))
    if index:
        np.savez(dir_ / INDEX_10K, test_index=np.arange(n), probe_positions=np.arange(min(n, 2000)))
    return need


# ------------------------------------------------------------------------------------------------ plots


def plots(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(linear='#2a78d6', soft_ulam='#eb6834', nystrom='#1baf7a', diffusion='#eda100')
    ink, muted = '#0b0b0b', '#8a8984'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': muted, 'axes.labelcolor': ink, 'xtick.color': muted,
                         'ytick.color': muted, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    curve_p = out_dir / 'curve_p8.json'
    if curve_p.exists():
        units = json.loads(curve_p.read_text())['units']
        truth = p8_truth(SIGMA)
        for metric, label in (('test_corr', 'held-out correlation'), ('train_sv', 'training singular value')):
            fig, axes = plt.subplots(2, 6, figsize=(15, 5.2), sharey=True, sharex=True)
            for row, m in enumerate(M_VALUES):
                for j in range(6):
                    ax = axes[row, j]
                    ax.axhline(truth[j], color=ink, lw=1, ls='--')
                    for est in ESTIMATORS:
                        xs, mu, sd = [], [], []
                        for n in CURVE_N:
                            v = [u[metric][j] for u in units if u.get('n_fit') == n and u.get('m') == m and
                                 u.get('estimator') == est and metric in u and len(u[metric]) > j and
                                 u[metric][j] is not None]
                            if v:
                                xs.append(n); mu.append(np.mean(v)); sd.append(np.std(v))
                        if xs:
                            ax.errorbar(xs, mu, yerr=sd, color=colors[est], lw=2, marker='o', ms=4, capsize=2,
                                        label=est)
                    if metric == 'test_corr':
                        cx, cq = [], []
                        for n in CURVE_N:
                            v = [u['control']['test_corr']['q95'][j] for u in units if u.get('n_fit') == n and
                                 u.get('m') == m and u.get('estimator') == 'nystrom' and 'control' in u]
                            if v:
                                cx.append(n); cq.append(v[0])
                        ax.plot(cx, cq, color=muted, lw=1.5, label='control 95% (Nystrom)')
                    ax.set_xscale('log'); ax.set_ylim(-0.1, 1.02)
                    ax.set_xticks(CURVE_N, [str(n) for n in CURVE_N], fontsize=7); ax.minorticks_off()
                    ax.set_title(f'function {j + 1}, m = {m}', fontsize=9, color=ink)
                    if j == 0:
                        ax.set_ylabel(label)
                    if row == 1:
                        ax.set_xlabel('N fit images')
            h, l = axes[0, 0].get_legend_handles_labels()
            fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, frameon=False)
            fig.suptitle(f'P8 sample-size curve ({label}; mean +- sd over seeds; dashed = exp(-m^2 sigma^2/2), '
                         f'sigma = {SIGMA})', fontsize=10, color=ink)
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(out_dir / f'p8_curve_{metric}.png', dpi=110)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for m, ls in zip(M_VALUES, ('-', ':')):
            xs, ys = [], []
            for n in CURVE_N:
                v = [u['selected']['m_used'] for u in units if u.get('n_fit') == n and u.get('m') == m and
                     u.get('estimator') == 'soft_ulam' and u.get('selected')]
                if v:
                    xs.append(n); ys.append(np.median(v))
            ax.plot(xs, ys, color=colors['soft_ulam'], lw=2, ls=ls, marker='o', label=f'requested m = {m}')
        ax.set_xscale('log'); ax.set_yscale('log', base=2)
        ax.set_xticks(CURVE_N, [str(n) for n in CURVE_N]); ax.minorticks_off()
        ax.set_yticks([8, 16, 32, 64, 128], ['8', '16', '32', '64', '128'])
        ax.set_xlabel('N fit images'); ax.set_ylabel('landmarks used (median over seeds)')
        ax.set_title('Soft-Ulam occupancy gate: every landmark >= 10 fit images (soft count)', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / 'p8_occupancy_gate.png', dpi=110); plt.close(fig)
    syn_p = out_dir / 'synthetic.json'
    if syn_p.exists():
        syn = json.loads(syn_p.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
        ests = ('nystrom', 'diffusion', 'soft_ulam')
        for ax, (case, title) in zip(axes, (('markov', 'Markov chain theta -> 2 theta -> 4 theta'),
                                            ('broken', 'middle view re-paired at random'))):
            ticks, labels = [], []
            for i, est in enumerate(ests):
                sw = syn['chain'].get(f'{case}_{est}')
                if not sw:
                    continue
                bd = sw['base_dims']
                ax.text(i * 6 + 2, 1.17, f"{est}\na {bd['m_a']}, c {bd['m_c']}", ha='center', va='top', fontsize=7,
                        color=ink)
                for jr, row in enumerate(sw['rows']):
                    x = i * 6 + jr
                    ticks.append(x)
                    if 'unavailable' in row:
                        labels.append(f"{row['m_b_requested']}\nn/a")
                        continue
                    dm = row['dims']
                    labels.append(f"{dm['m_b']}")
                    ax.bar(x - 0.2, row['partial']['topk_rel'], 0.4, color=colors[est])
                    ax.bar(x + 0.2, row['regulariser']['topk_rel'], 0.4, color=colors[est], alpha=0.45)
                    ax.plot([x - 0.4, x + 0.4], [row['heldout_residual_crossmoment']['topk_rel']] * 2, color=ink, lw=1.5)
            ax.set_xticks(ticks, labels, fontsize=7)
            ax.set_ylim(0, 1.2)
            ax.set_xlabel('actual intermediate dimension m_b (n/a: below the occupancy gate); endpoints fixed', fontsize=8)
            ax.set_title(title, fontsize=9)
        axes[0].set_ylabel('relative to top-6 scale of T_ac')
        fig.suptitle('Composition discrepancy, fit split: partial term (solid) + regulariser term (light) = total; '
                     'black tick: held-out residual cross-moment (separate diagnostic)', fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / 'chain_composition.png', dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('synthetic'); s.add_argument('--out', default=str(OUT)); s.add_argument('--n-perm', type=int, default=50)
    s.add_argument('--no-6000', action='store_true')
    c = sub.add_parser('curve-local'); c.add_argument('--out', default=str(OUT)); c.add_argument('--n-perm', type=int, default=10)
    c.add_argument('--max-n', type=int, default=1000)
    for name in ('real', 'real10k'):
        r = sub.add_parser(name)
        r.add_argument('--grams', required=True)
        r.add_argument('--template', default=TEMPLATE_2000 if name == 'real' else TEMPLATE_10K)
        r.add_argument('--out', default=str(OUT / name))
        r.add_argument('--n-perm', type=int, default=200 if name == 'real' else 50)
        r.add_argument('--n-perm-fit', type=int, default=50)
        r.add_argument('--splits', type=int, default=10)
        r.add_argument('--n-images', type=int, default=2000 if name == 'real' else 10000)
        r.add_argument('--sizes', type=int, nargs=3, default=None)
        r.add_argument('--curve', type=int, nargs='+', default=[500, 1000, 2000, 6000])
        r.add_argument('--pairs', nargs='*', default=None)
        r.add_argument('--no-triples', action='store_true')
        r.add_argument('--estimators', nargs='+', default=list(ESTIMATORS))
        r.add_argument('--m', type=int, nargs='+', default=list(M_VALUES))
    sm = sub.add_parser('smoke-real'); sm.add_argument('--dir', default='/tmp/transfer_standin')
    sm.add_argument('--out', default=str(OUT / 'smoke_real'))
    sub.add_parser('plots').add_argument('--out', default=str(OUT))
    sub.add_parser('chain').add_argument('--out', default=str(OUT))
    a = ap.parse_args(argv)
    if a.cmd == 'synthetic':
        run_synthetic(Path(a.out), a.n_perm, not a.no_6000)
    elif a.cmd == 'curve-local':
        units = [u for u in curve_units(n_perm=a.n_perm) if u['n_fit'] <= a.max_n]
        res = [p8_unit(**u) for u in units]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / 'curve_p8.json').write_text(json.dumps(dict(sigma=SIGMA, units=res)))
    elif a.cmd == 'real':
        run_real(a.grams, a.template, Path(a.out), a.n_images, tuple(a.sizes or (1000, 500, 500)), a.splits, a.n_perm,
                 a.n_perm_fit, a.pairs, not a.no_triples, a.estimators, tuple(a.m))
    elif a.cmd == 'real10k':
        run_real10k(a.grams, a.template, Path(a.out), a.n_images, tuple(a.sizes or (6000, 2000, 2000)),
                    tuple(a.curve), a.n_perm, a.estimators, tuple(a.m), a.pairs)
    elif a.cmd == 'smoke-real':
        d = Path(a.dir)
        make_standins(d)
        t0 = time.time()
        run_real(str(d), TEMPLATE_2000, Path(a.out), 2000, (1000, 500, 500), 2, 3, 2,
                 pairs=['stem-b1', 'b5:1024-64000'], triples=True, m_values=(64, 128))
        print('real smoke', f'{time.time() - t0:.0f}s', flush=True)
        d10 = d / 'final10k'
        make_standins(d10, n=2000, cells=[(s_, FINAL) for s_ in SITES], template=TEMPLATE_10K, dtype=np.float32,
                      index=True)
        t0 = time.time()
        run_real10k(str(d10), TEMPLATE_10K, Path(a.out), 2000, (1200, 400, 400), (300, 1200), 2,
                    m_values=(64,), pairs=['b4-b5'])
        print('10k-shaped smoke (reduced N, final-only stand-ins)', f'{time.time() - t0:.0f}s', flush=True)
    elif a.cmd == 'plots':
        plots(Path(a.out))
    elif a.cmd == 'chain':
        p = Path(a.out) / 'synthetic.json'
        syn = json.loads(p.read_text())
        syn['chain'] = run_chain()
        p.write_text(json.dumps(_js(syn), indent=1))


if __name__ == '__main__':
    main()
