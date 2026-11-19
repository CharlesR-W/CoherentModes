"""Asserted tests for transfer.py (PLAN section 6; TRANSFER.md section 7 lists what each test pins down).

Run: uv run --with torch --with numpy --with scipy --with scikit-learn --with deeptime --with pytest \
        pytest explore/cifar_baselines/test_transfer.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planted  # noqa: E402
import transfer as tr  # noqa: E402
import transfer_pilot as tp  # noqa: E402
from transfer_pilot import make_chain, p8_truth  # noqa: E402

try:
    import torch
    torch.set_num_threads(2)
except ImportError:  # pragma: no cover
    pass


def _splits(N, n_fit, n_tune, seed=1):
    o = np.random.default_rng(seed).permutation(N)
    return o[:n_fit], o[n_fit:n_fit + n_tune], o[n_fit + n_tune:]


def _bank(Xs, fit, labels='abc'):
    return tr.FeatureBank({lab: tr.View.from_X(X, fit, lab) for lab, X in zip(labels, Xs)})


@pytest.fixture(scope='module')
def p8():
    N = 3000
    Xs, truth = planted.make_P8(0, N=N, level=0.35)
    fit, tun, te = _splits(N, 1000, 1000)
    return Xs, truth, fit, tun, te, _bank(Xs, fit, 'ab').pair('a', 'b')


# ------------------------------------------------------------------------------------------------ views


def test_view_from_R_matches_stats_and_features():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 12)) + 0.7
    R = X @ X.T
    fit = np.sort(rng.choice(80, 50, replace=False))
    vR, vX = tr.View.from_R(R, fit), tr.View.from_X(X, fit)
    allidx = np.arange(80)
    C = vR.cos(allidx, allidx)
    assert np.abs(C - vX.cos(allidx, allidx)).max() < 1e-10
    assert np.abs(C - tr.view_via_stats(R, fit)).max() < 1e-10        # builder A's raw_to_views
    Z = X - X[fit].mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    assert np.abs(C - Z @ Z.T).max() < 1e-10                           # direct feature-space computation


# ------------------------------------------------------------------------------------------------ conventions


def test_orientation_and_shapes(p8):
    Xs, truth, fit, tun, te, pair = p8
    rng = np.random.default_rng(3)
    # different source / destination feature dimensions: T is destination x source
    Xa = np.concatenate([Xs[0], 0.3 * rng.standard_normal((len(Xs[0]), 8))], 1)
    bank = tr.FeatureBank({'a': tr.View.from_X(Xa, fit, 'a'), 'b': tr.View.from_X(Xs[1], fit, 'b')})
    mod = tr.fit_point(bank.pair('a', 'b'), 'linear', ('linear', 64), 1e-3)
    ma, mb = mod.src.feat.m, mod.dst.feat.m
    assert (ma, mb) == (10, 2) and mod.T.shape == (mb, ma)
    U, s, Vt = np.linalg.svd(mod.T)
    assert np.allclose(mod.T @ Vt[0], s[0] * U[:, 0])                  # right singular vectors are source
    # semantic orientation on P8: source functions are harmonic 2 in theta, destination harmonic 1 in phi
    mod = tr.fit_point(pair, 'nystrom', ('nystrom', 64, 1 / 64), 1e-2)
    th, ph = truth['angle'], truth['angle2']
    f, g = mod.source_functions(te)[:, :2], mod.dest_functions(te)[:, :2]
    ov = tr._overlap
    assert ov(f, np.stack([np.cos(2 * th[te]), np.sin(2 * th[te])], 1)) > 0.85
    assert ov(f, np.stack([np.cos(th[te]), np.sin(th[te])], 1)) < 0.05
    assert ov(g, np.stack([np.cos(ph[te]), np.sin(ph[te])], 1)) > 0.85


def test_forward_composition_orientation():
    N = 3000
    Xs, _ = make_chain(0, N, 0.25, 0.25)
    fit, tun, te = _splits(N, 1000, 1000)
    bank = _bank(Xs, fit)
    tab, tbc, tac = tr.fit_chain(bank, 'abc', 'nystrom', 64, tun)
    comp = tbc.T @ tab.T
    assert comp.shape == tac.T.shape == (tac.dst.feat.m, tac.src.feat.m)
    with pytest.raises(ValueError):
        tr.composition_discrepancy(tbc, tab, tac)                    # wrong order is rejected


def test_soft_ulam_trivial_pair(p8):
    Xs, truth, fit, tun, te, pair = p8
    mod = tr.fit_point(pair, 'soft_ulam', ('ulam', 32, 0.25), None)
    B = mod.S
    p, q = B.sum(0), B.sum(1)
    assert abs(B.sum() - 1) < 1e-12                                    # a joint probability table
    assert np.allclose(mod.T @ np.sqrt(p), np.sqrt(q), atol=1e-12)
    assert np.allclose(mod.T.T @ np.sqrt(q), np.sqrt(p), atol=1e-12)
    assert abs(mod.trivial['sigma'] - 1) < 1e-10
    assert mod.trivial['src_overlap'] > 1 - 1e-10 and mod.trivial['dst_overlap'] > 1 - 1e-10
    assert mod.singular_values[0] < 1 - 1e-6
    # non-trivial source functions have zero mean under the source marginal (orthogonal to the constant)
    f = mod.source_functions(fit)
    assert np.abs(f.mean(0)[:6]).max() < 1e-10


# ------------------------------------------------------------------------------------------------ memorisation


def test_memorisation_pathology(p8):
    """Unregularised full-rank fits memorise the pairing: in-sample singular values ~ 1 even under the
    permutation control, held-out correlations at chance. Hence held-out validation and declared function spaces."""
    Xs, truth, _, _, _, _ = p8
    N = 600
    fit, tun, te = _splits(N, 150, 150)
    bank = _bank([X[:N] for X in Xs], fit, 'ab')
    pair = bank.pair('a', 'b')
    rng = np.random.default_rng(11)
    for spec in (('nystrom', 150, 4.0), ('ulam', 150, 64.0)):
        est = 'soft_ulam' if spec[0] == 'ulam' else 'nystrom'
        hcs = []
        for t in range(20):
            perm = tr.split_permutation(N, [fit, tun, te], rng)
            mod = tr.fit_point(pair, est, spec, None, perm)
            assert mod.src.feat.m == 150
            assert mod.singular_values[:6].min() > 0.99, (est, mod.singular_values[:6])
            hcs.append(tr.heldout_correlations(mod, te)['corr'])
        hcs = np.array(hcs)
        # chance held-out: centred at zero. Memorised functions are heavy-tailed (a few images carry them), so a
        # single held-out Pearson correlation can be large; the statement is about the distribution.
        assert abs(np.nanmean(hcs)) < 0.05 and np.nanmedian(np.abs(hcs)) < 0.15, (est, hcs.round(2))
        if est == 'soft_ulam':
            assert not mod.occupancy['src']['passes']                 # the occupancy gate refuses this table


def test_split_permutation_structure():
    rng = np.random.default_rng(0)
    splits = [np.arange(0, 10), np.arange(10, 15), np.arange(15, 20)]
    p = tr.split_permutation(20, splits, rng)
    for s in splits:
        assert set(p[s]) == set(s)
    q = tr.split_permutation(20, splits, rng, mode='fit_only')
    assert np.array_equal(q[10:], np.arange(10, 20))


# ------------------------------------------------------------------------------------------------ P8


def test_p8_recovery_and_linear_miss(p8):
    """Truth: sigma = exp(-m^2 s^2/2), m = 1, 2, 3, two each. N_fit = 1000. Tolerances (stated in TRANSFER.md):
    Nystrom and diffusion: training singular values of all 6 within 0.10; held-out correlations of functions 1-4
    within 0.10 and of 5-6 within 0.15 (held-out correlations of estimated functions are biased low);
    soft Ulam (gate-limited m): first 4 held-out correlations within 0.10, functions 5-6 reported only.
    Linear CCA on the 2-d coordinates: <= 2 functions, test correlation inside the permutation band."""
    Xs, truth, fit, tun, te, pair = p8
    sv = p8_truth(0.35)
    for est in ('nystrom', 'diffusion', 'soft_ulam'):
        mod, _ = tr.tune(pair, est, 64, tun)
        ev = tr.evaluate(mod, tun, te)
        hc = np.array(ev['test_corr'])
        if est == 'soft_ulam':
            assert np.abs(hc[:4] - sv[:4]).max() < 0.10, (est, hc)
        else:
            assert np.abs(hc[:4] - sv[:4]).max() < 0.10, (est, hc)
            assert np.abs(hc[4:] - sv[4:]).max() < 0.15, (est, hc)
            assert np.abs(np.array(ev['train_sv']) - sv).max() < 0.10, (est, ev['train_sv'])
    lin, _ = tr.tune(pair, 'linear', 64, tun)
    assert lin.src.feat.m == 2 and lin.dst.feat.m == 2
    ctl = tr.permutation_control(pair, 'linear', 64, tun, te, n_perm=40, seed=2)
    hc = tr.heldout_correlations(lin, te)['corr']
    assert hc[0] < np.nanquantile(ctl['test_corr'][:, 0], 0.995) + 0.02
    assert tr.vamp2(lin, te) < 0.05


def test_p8_independent_pairing_gives_control_band():
    """Independent pairing with the same marginals: held-out correlations fall in the permutation-control band."""
    N = 2000
    Xs, _ = planted.make_P8(0, N=N, level='independent')
    fit, tun, te = _splits(N, 1000, 500)
    pair = _bank(Xs, fit, 'ab').pair('a', 'b')
    mod, _ = tr.tune(pair, 'nystrom', 64, tun)
    hc = tr.heldout_correlations(mod, te)['corr']
    ctl = tr.permutation_control(pair, 'nystrom', 64, tun, te, n_perm=20, seed=5)
    hi = np.nanmax(ctl['test_corr'], axis=0)
    assert np.all(hc <= hi + 0.05), (hc, hi)
    assert np.abs(hc).max() < 0.2


# ------------------------------------------------------------------------------------------------ P2


def test_p2_shared_factor_linear_cca():
    """P2 (shared linear factors, h = 3). Oracle: population linear CCA after the pipeline's preprocessing, from
    a 40,000-image draw. Linear whitened CCA held-out correlations of the 3 shared functions within 0.06 of the
    oracle; functions 4-6 at chance for linear features."""
    big, _ = planted.make_P2(0, N=40000, level=1.0, L=2)
    Z = [planted.preprocess(X) for X in big]
    n = len(Z[0]); Z = [z - z.mean(0) for z in Z]
    W = tr._psd_inv_sqrt(Z[1].T @ Z[1] / n) @ (Z[1].T @ Z[0] / n) @ tr._psd_inv_sqrt(Z[0].T @ Z[0] / n)
    oracle = np.linalg.svd(W, compute_uv=False)[:3]
    N = 3000
    Xs, _ = planted.make_P2(1, N=N, level=1.0, L=2)
    fit, tun, te = _splits(N, 1000, 1000)
    pair = _bank(Xs, fit, 'ab').pair('a', 'b')
    mod, _ = tr.tune(pair, 'linear', 64, tun)
    hc = tr.heldout_correlations(mod, te)['corr']
    assert np.abs(hc[:3] - oracle).max() < 0.06, (hc, oracle)
    assert np.abs(hc[3:]).max() < 0.12


# ------------------------------------------------------------------------------------------------ composition


def test_composition_discrepancy_calibration():
    """Known-Markov chain theta -> 2 theta -> 4 theta: the held-out residual cross-moment (a separate diagnostic) is
    small; with the middle view's pairing destroyed it is of the order of the operator itself. The regulariser and
    partial terms add up exactly to the fit-split discrepancy."""
    N = 3000
    Xs, _ = make_chain(0, N, 0.25, 0.25)
    fit, tun, te = _splits(N, 1000, 1000)
    rel = {}
    for case in ('markov', 'broken'):
        Ys = list(Xs)
        if case == 'broken':
            Ys[1] = Xs[1][np.random.default_rng(5).permutation(N)]
        tab, tbc, tac = tr.fit_chain(_bank(Ys, fit), 'abc', 'nystrom', 64, tun)
        d = tr.composition_discrepancy(tab, tbc, tac, te)
        assert d['additivity_error'] < 1e-6
        rel[case] = d['heldout_residual_crossmoment']['topk_rel']
    assert rel['markov'] < 0.15 and rel['broken'] > 0.6, rel


def test_soft_ulam_composition_preserves_trivial_pair():
    N = 2000
    Xs, _ = make_chain(1, N, 0.25, 0.25)
    fit, tun, te = _splits(N, 1000, 500)
    tab, tbc, tac = tr.fit_chain(_bank(Xs, fit), 'abc', 'soft_ulam', 32, tun)
    comp = tr.composed_model(tab, tbc)
    assert abs(comp.trivial['sigma'] - 1) < 1e-10
    d = tr.composition_discrepancy(tab, tbc, tac, te)
    assert d['additivity_error'] < 1e-10


# ------------------------------------------------------------------------------------------------ diffusion extension


def test_diffusion_extension_rows():
    N = 400
    Xs, _ = planted.make_P8(2, N=N, level=0.35)
    fit, _, te = _splits(N, 300, 50)
    v = tr.View.from_X(Xs[0], fit)
    F = tr.diffusion_features(v, 8)
    P, u, dk = tr.rule0_operator(v.cos(fit, fit))
    assert np.abs(P.sum(1) - 1).max() < 1e-10
    # manual extension of one held-out image
    i = te[0]
    d = 1 - v.cos([i], fit)[0]
    edge = np.zeros(len(fit), bool); edge[np.argsort(d)[:15]] = True; edge |= d < dk
    w = np.where(edge, 1.0, 1e-6) * np.exp(u); w /= w.sum()
    Psi = F.arr[fit] / np.array(F.meta['lambda_P'][:8])[None, :]    # fit rows are lambda_P psi exactly
    assert np.abs(P @ Psi - F.arr[fit]).max() < 1e-10
    assert np.abs(w @ Psi - F.arr[i]).max() < 1e-10


# ------------------------------------------------------------------------------------------------ occupancy


def test_occupancy_gate_reduces_m():
    N = 1250
    Xs, _ = planted.make_P8(3, N=N, level=0.35)
    fit, tun, te = _splits(N, 250, 500)
    pair = _bank(Xs, fit, 'ab').pair('a', 'b')
    mod, rows = tr.tune(pair, 'soft_ulam', 128, tun)
    m_used = mod.params['selected']['m_used']
    assert m_used <= 25
    assert mod.occupancy['src']['min_soft'] >= 10 and mod.occupancy['dst']['min_soft'] >= 10


# ------------------------------------------------------------------------------------------------ deeptime


def test_deeptime_crosscheck():
    """Whitened CCA singular values and held-out VAMP-2 against deeptime's VAMP, where normalisations agree:
    biased covariances (1/n), mean removed, VAMP-2 on the evaluation split with its own mean removed,
    deeptime's score includes +1 for the constant. Ridge: deeptime has none (epsilon truncation), so the ridge
    path is checked by handing deeptime the ridge-regularised covariances."""
    deeptime = pytest.importorskip('deeptime')
    from deeptime.covariance import CovarianceModel
    from deeptime.decomposition import VAMP, vamp_score
    N = 3000
    Xs, _ = planted.make_P2(4, N=N, level=0.7, L=2, h=3)
    fit, tun, te = _splits(N, 1500, 500)
    pair = _bank(Xs, fit, 'ab').pair('a', 'b')
    for eta in (None, 1e-2):
        mod = tr.fit_point(pair, 'linear', ('linear', 64), eta)
        Fx = mod.src.feat.arr; Fy = mod.dst.feat.arr
        X, Y = Fx[fit] - mod.src.mu, Fy[fit] - mod.dst.mu
        n = len(fit)
        cov = CovarianceModel(cov_00=X.T @ X / n + mod.src.rho * np.eye(X.shape[1]), cov_0t=X.T @ Y / n,
                              cov_tt=Y.T @ Y / n + mod.dst.rho * np.eye(Y.shape[1]),
                              mean_0=mod.src.mu, mean_t=mod.dst.mu, bessels_correction=False,
                              data_mean_removed=True)
        dm = VAMP(epsilon=1e-12).fit_from_covariances(cov).fetch_model()
        k = 6
        assert np.allclose(dm.singular_values[:k], mod.singular_values[:k], atol=1e-8)
        Xt, Yt = Fx[te] - Fx[te].mean(0), Fy[te] - Fy[te].mean(0)
        nt = len(te)
        ct = CovarianceModel(cov_00=Xt.T @ Xt / nt + mod.src.rho * np.eye(X.shape[1]), cov_0t=Xt.T @ Yt / nt,
                             cov_tt=Yt.T @ Yt / nt + mod.dst.rho * np.eye(Y.shape[1]),
                             bessels_correction=False, data_mean_removed=True)
        ds = vamp_score(dm, 2, covariances_test=ct, dim=k, epsilon=1e-12) - 1
        assert abs(ds - tr.vamp2(mod, te, k)) < 1e-8, (ds, tr.vamp2(mod, te, k))


# ------------------------------------------------------------------------------------------------ review-3 fixes


def test_composition_sweep_freezes_endpoints():
    """Only the intermediate basis varies: endpoint features (objects), eta and every kappa are those of the base
    chain; inadmissible soft-Ulam middle sizes are reported unavailable, never compensated by shrinking endpoints."""
    N = 2000
    Xs, _ = make_chain(0, N, 0.25, 0.25)
    fit, tun, te = _splits(N, 1000, 500)
    bank = _bank(Xs, fit)
    for est in ('nystrom', 'soft_ulam'):
        sw = tr.composition_sweep(bank, 'abc', est, 32, tun, te, (0.5, 1, 2, 4))
        base = sw['base_dims']
        avail = [r for r in sw['rows'] if 'unavailable' not in r]
        assert avail
        for r in avail:
            assert (r['dims']['m_a'], r['dims']['m_c']) == (base['m_a'], base['m_c'])
            assert r['eta'] == sw['base_selected']['eta']
            assert r['kappa_b'] == pytest.approx(sw['kappa']['b'])
            assert r['additivity_error_rel'] < 1e-10, r['additivity_error_rel']
        if est == 'nystrom':
            # landmarks are added at a frozen bandwidth: the numerical rank saturates (actual dims recorded)
            assert [r['dims']['m_b'] for r in avail] == sorted(r['dims']['m_b'] for r in avail)
            assert len(avail) == 4 and avail[0]['dims']['m_b'] < avail[-1]['dims']['m_b']
        else:
            assert any('unavailable' in r for r in sw['rows'])                   # larger middle tables fail the gate
            for r in sw['rows']:
                if 'unavailable' in r:
                    assert r['occupancy_b_min_soft'] < tr.OCCUPANCY_MIN


def _fake_units(n, splits, F_same, G_same, k=6, seed=0):
    rng = np.random.default_rng(seed)
    base_f, base_g = rng.standard_normal((n, k)), rng.standard_normal((n, k))
    units = []
    for i in range(len(splits)):
        f = base_f if F_same else rng.standard_normal((n, k))
        g = base_g if G_same else rng.standard_normal((n, k))
        units.append({'nystrom_m64': dict(test_corr=[.9] * k, control_q95=[.05] * k, fitonly_q95=[.05] * k,
                                          train_sv=[.9] * k, src_funcs=f, dst_funcs=g, vamp2_test=3.0,
                                          control_vamp2_q95=.1, selected={}, occupancy=None)})
    ref = dict(Q_src=rng.standard_normal((n, k)), Q_dst=rng.standard_normal((n, k)), Mbar=rng.standard_normal((n, k)))
    return units, ref


def test_sets_gated_per_side_and_labelled_transductive():
    n = 400
    splits = tp.predeclared_splits(n, (200, 100, 100), 10)
    units, ref = _fake_units(n, splits, F_same=True, G_same=False)
    r = tp.aggregate_pair(units, ref, 'linear_m64', splits)['nystrom_m64']
    assert r['stability']['src']['mean_pairwise_cos2'] > 0.99
    assert r['stability']['dst']['mean_pairwise_cos2'] < 0.5
    assert r['sets']['sides'] == ['src'] and 'dst_labels' not in r['sets'] and 'joint_table' not in r['sets']
    assert 'transductive' in r['sets']['status']
    units, ref = _fake_units(n, splits, F_same=True, G_same=True)
    r = tp.aggregate_pair(units, ref, 'linear_m64', splits)['nystrom_m64']
    assert r['sets']['sides'] == ['src', 'dst'] and 'joint_table' in r['sets']


def test_heldout_stability_uses_shared_heldout_images_only():
    n, k = 400, 3
    splits = tp.predeclared_splits(n, (200, 100, 100), 4)
    rng = np.random.default_rng(1)
    base = rng.standard_normal((n, k))
    Fs = []
    for s in splits:                                   # functions differ arbitrarily on each split's own fit images
        f = base.copy(); f[s['fit']] = rng.standard_normal((len(s['fit']), k)); Fs.append(f)
    st = tp.heldout_stability(Fs, splits, k)
    assert st['mean_pairwise_cos2'] > 1 - 1e-10


def test_joint_m_selection_repeated_inside_permutations():
    N = 1500
    Xs, _ = planted.make_P8(5, N=N, level=0.35)
    R = [X @ X.T for X in Xs]
    split = tp.predeclared_splits(N, (700, 400, 400), 1)[0]
    u = tp.real_unit(R[0], R[1], split, 0, estimators=('nystrom',), m_values=(16, 32), n_perm=3, n_perm_fit=2)
    j = u['nystrom_mjoint']
    assert j['m_selected'] in (16, 32)
    assert j['selected'] == u[f"nystrom_m{j['m_selected']}"]['selected']
    assert len(j['control_q95']) == 6 and len(j['control_m_selected']) == 3 and len(j['fitonly_q95']) == 6


def test_real10k_final_checkpoint_only_fixture(tmp_path):
    """10k runner: final-checkpoint stand-ins only (as builder A extracts); default pairs are the eight
    final-checkpoint pairs; time pairs are refused; preflight fails before any compute when an input is missing."""
    n = 300
    d = tmp_path / 'final'
    tp.make_standins(d, n=n, cells=[(s, tp.FINAL) for s in tp.SITES], template=tp.TEMPLATE_10K,
                     dtype=np.float32, index=True)
    out = tp.run_real10k(str(d), tp.TEMPLATE_10K, tmp_path / 'o', n, (150, 75, 75), (60, 150), n_perm=1,
                         estimators=('nystrom',), m_values=(16,))
    assert [p['id'] for p in out['pairs']] == tp.FINAL_PAIRS and len(tp.FINAL_PAIRS) == 8
    assert all(p['src']['step'] == p['dst']['step'] == tp.FINAL for p in out['pairs'])
    with pytest.raises(ValueError):
        tp.run_real10k(str(d), tp.TEMPLATE_10K, tmp_path / 'o', n, (150, 75, 75), (60,), pairs=['b5:1024-64000'])
    tp.gram_path(d, tp.TEMPLATE_10K, 'b9', tp.FINAL).unlink()
    with pytest.raises(RuntimeError, match='preflight'):
        tp.run_real10k(str(d), tp.TEMPLATE_10K, tmp_path / 'o', n, (150, 75, 75), (60,))
