"""Regression tests for planted.py / bench.py: each benchmark's easy regime and its negative control at N = 300.

Thresholds describe the behaviour measured when the benchmarks were built (2026-11-15, seed 0); they are regression
guards, not statements that any method should behave this way. Run:
    uv run --with torch --with numpy --with scipy --with scikit-learn pytest explore/cifar_baselines/test_planted.py
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'cifar_coherent'))
torch.set_num_threads(2)

import bench  # noqa: E402
import planted  # noqa: E402
from coherent import centred_cosine_gram  # noqa: E402

N = 300


@functools.lru_cache(maxsize=None)
def unit(b, lv):
    res = bench.run_unit(b, lv, 0, Ns=(N,), stability=False, transfer_perms=0)
    out = {}
    for r in res['rows']:
        out[(r['method'], r['block'], r['r'], r['stat'])] = r['value']
    return out


def val(b, lv, method, stat, r=None, block='all'):
    u = unit(b, lv)
    if r is None:
        rs = [k[2] for k in u if k[0] == method and k[1] == block and k[3] == stat]
        assert rs, (b, lv, method, stat)
        r = rs[0]
    return u[(method, block, r, stat)]


def orc(b, lv, stat):
    return unit(b, lv)[('oracle', 'all', None, stat)]


# ------------------------------------------------------------------------------------------------ generators


@pytest.mark.parametrize('name', [b for b in planted.BENCHMARKS if b != 'P12'])
def test_generators_shapes_and_determinism(name):
    make, levels = planted.BENCHMARKS[name]
    for lv in (levels[0], levels[-1]):
        Xs, t = make(3, N=200, level=lv)
        Xs2, _ = make(3, N=200, level=lv)
        assert t['id'] == name and t['level'] == lv and t['ranks'] and t['blocks']
        if Xs is None:
            assert name == 'P11' and len(t['adjacency']) == t['params']['L']
            for A in t['adjacency']:
                assert A.shape == (200, 200) and (A == A.T).all() and A.diagonal().sum() == 0
            continue
        for X, X2 in zip(Xs, Xs2):
            assert X.shape[0] == 200 and np.isfinite(X).all()
            np.testing.assert_array_equal(X, X2)
        if t.get('labels') is not None:
            assert len(t['labels']) == 200


def test_isometric_embedding_changes_no_gram():
    rng = np.random.default_rng(0)
    Ys = [rng.standard_normal((150, 6)) for _ in range(2)]
    a, _ = planted._cells(Ys, np.random.default_rng(1), widths=[8, 9])
    b, _ = planted._cells(Ys, np.random.default_rng(2), widths=[300, 41])
    for x, y, Y in zip(a, b, Ys):
        ga = centred_cosine_gram(torch.from_numpy(x))
        assert float((ga - centred_cosine_gram(torch.from_numpy(y))).abs().max()) < 1e-10
        assert float((ga - centred_cosine_gram(torch.from_numpy(Y))).abs().max()) < 1e-10


def test_simplex_and_balance():
    mu = planted.simplex(4)
    G = mu @ mu.T
    np.testing.assert_allclose(np.diag(G), 1, atol=1e-12)
    np.testing.assert_allclose(G[~np.eye(4, dtype=bool)], -1 / 3, atol=1e-12)
    g = planted.balanced_labels(np.random.default_rng(0), 1001, 4, (.6, .25, .1, .05))
    assert abs(np.bincount(g)[3] - 50) <= 1


def test_density_sampler_moment():
    th = planted.circle_density_sample(np.random.default_rng(0), 20000, .6)
    assert abs(np.cos(th).mean() - .3) < .02


# ------------------------------------------------------------------------------------------------ benchmarks


def test_P0_null():
    assert abs(orc('P0', 0.0, 'pseudo_kappa_cell0')) < .1
    assert orc('P0', 0.0, 'cka_cells_01') < .3
    assert orc('P0', 1.0, 'cka_cells_01') > .99
    assert abs(val('P0', 0.0, 'M0', 'ami_pseudo')) < .05


def test_P1_easy_and_null():
    assert orc('P1', 1.0, 'pooled_kappa') > .95
    for m in ('M0', 'L1', 'L3', 'G2', 'C3'):
        assert val('P1', 1.0, m, 'ami') > .95, m
    assert abs(orc('P1', 0.0, 'pooled_kappa')) < .1
    for m in ('M0', 'L1', 'L3'):
        assert abs(val('P1', 0.0, m, 'ami')) < .05, m


def test_P2_shared_factor_span():
    assert orc('P2', 2.0, 'mean_cell_r2') > .7
    assert orc('P2', 2.0, 'raw_mean_cell_r2') >= orc('P2', 2.0, 'mean_cell_r2') - .01
    assert val('P2', 2.0, 'L1', 'contrast_captured') > .8
    assert orc('P2', 0.0, 'mean_cell_r2') < .1
    assert val('P2', 0.0, 'M0', 'contrast_captured') < .1


def test_P3_shared_vs_nuisance():
    assert val('P3', 0.0, 'M0', 'ami') > .9
    assert orc('P3', 4.0, 'nuisance_cell_kappa_mean') > .9
    assert val('P3', 4.0, 'C1first', 'ami_nuisance_max') > .9
    assert abs(orc('P3', 0.0, 'nuisance_cell_kappa_mean')) < .1


def test_P4_bottleneck():
    assert val('P4', 0, 'M0', 'ami') > .95 and orc('P4', 0, 'min_cell_kappa') > .9
    assert orc('P4', 3, 'min_cell_kappa') < .1 and orc('P4', 3, 'pooled_kappa') > .9
    # retention of the block's sets in a bottleneck cell (position 3) sits at its size null
    assert abs(val('P4', 3, 'M0', 'ret_excess_cell3')) < .05


def test_P5_nonlinear_groups():
    assert orc('P5', 0.05, 'pooled_kappa') > .95
    assert val('P5', 0.05, 'M0', 'ami', r=2) > .9
    assert orc('P5', 1.2, 'min_cell_kappa') < .1


def test_P6_density_oracle():
    assert orc('P6', 0.9, 'cell0_r2_first_harmonic') > .9
    assert orc('P6', 0.0, 'cell0_eta_hat') < .15
    assert orc('P6', 0.9, 'cell0_eta_hat') > orc('P6', 0.0, 'cell0_eta_hat') + .2


def test_P7_unequal_groups():
    assert orc('P7', 'none', 'pooled_kappa') > .9
    assert val('P7', 'none', 'M0', 'ami') > .9
    assert orc('P7', '2%-persistent', 'pooled_kappa') > .9


def test_P8_circle_doubling():
    assert orc('P8', 0.0, 'harmonic1_corr') > .95
    assert orc('P8', 'independent', 'harmonic1_corr') < .2
    assert val('P8', 0.0, 'M0', 'span_theta2_captured', r=3) > .9
    assert val('P8', 0.0, 'C1first', 'span_theta1_captured', r=3) > .9
    if bench.load_transfer() is not None:
        assert val('P8', 0.0, 'T_nystrom', 'test_corr0') > .8
        assert abs(val('P8', 'independent', 'T_nystrom', 'test_corr0')) < .3


def test_P9_competing_partitions():
    for j in range(3):
        assert orc('P9', 2.0, f'cell{j}_own_kappa') > .9
        assert abs(orc('P9', 0.0, f'cell{j}_own_kappa')) < .15
    assert val('P9', 2.0, 'L1', 'contrast_captured', r=4) > .9
    assert val('P9', 0.0, 'M0', 'contrast_captured', r=4) < .1


def test_P10_hierarchy():
    assert orc('P10', 2.0, 'leaf_last_cell_kappa') > .9 and orc('P10', 2.0, 'leaf_first_cell_kappa') < .5
    assert val('P10', 2.0, 'M0', 'ami_top', r=2) > .95
    assert val('P10', 2.0, 'M0', 'ami', r=4, block='late') > .9


def test_P11_operator_level_sbm():
    assert orc('P11', 0, 'min_cell_kappa') > .9 and orc('P11', 4, 'min_cell_kappa') > .9
    assert val('P11', 0, 'M0', 'ami') > .95
    # all-disassortative cells: the lazy average's leading span carries no community contrast at this size
    assert abs(val('P11', 4, 'M0', 'ami')) < .05


def test_P12_untrained_runs():
    Xs, t = planted.make_P12(0, N=256, level=0)
    assert len(Xs) == 5 and Xs[0].shape == (256, 256)
    o = planted.oracle(Xs, t)
    assert o['pooled_kappa'] < .2


def test_summary_format(tmp_path):
    res = bench.run_unit('P1', 1.0, 0, Ns=(N,), stability=False)
    res2 = bench.run_unit('P1', 1.0, 1, Ns=(N,), stability=False)
    import json
    (tmp_path / 'a.json').write_text(json.dumps(res))
    (tmp_path / 'b.json').write_text(json.dumps(res2))
    nrows, nsum = bench.merge_parts(tmp_path, tmp_path / 'merged')
    s = json.loads((tmp_path / 'merged' / 'summary.json').read_text())
    row = next(r for r in s['rows'] if r['method'] == 'M0' and r['stat'] == 'ami')
    assert row['n'] == 2 and row['q10'] <= row['median'] <= row['q90']


def test_mj_max_counts_trailing_empty_set():
    """Review finding (P2): a trailing empty predicted set must stay in the padded matched-Jaccard denominator,
    so the score is invariant to which slot of the declared r sets is empty."""
    y = np.repeat([0, 1], 50)
    lab_trailing = y.copy()                 # r = 3 declared, sets {0, 1} used, set 2 empty (trailing)
    lab_leading = y + 1                     # same partition, set 0 empty
    truth = dict(id='TEST', labels=y, n_groups=2)
    Ps = [torch.eye(100, dtype=torch.float64)]
    a = bench.score(dict(labels=lab_trailing, basis=None), truth, Ps, [0], 3)
    b = bench.score(dict(labels=lab_leading, basis=None), truth, Ps, [0], 3)
    assert a['mj_max'] == pytest.approx(2 / 3) and b['mj_max'] == pytest.approx(2 / 3)
    assert a['ami'] == pytest.approx(b['ami'])
