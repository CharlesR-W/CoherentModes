"""Local smoke of measure_b's block-level code paths on a small synthetic block (no Modal
calls; the volume loaders are replaced). Skipped when the modal package is absent."""
import numpy as np
import pytest
import torch

torch.set_num_threads(2)
modal = pytest.importorskip('modal')

import measure_b as MB  # noqa: E402
import sources as S  # noqa: E402
from test_sources import planted_block, grams_of  # noqa: E402


def _block(n=150, L=3):
    xs, truth = planted_block(n=n, L=L, sep=2.0)
    grams = grams_of(xs)
    ops = S.default_ops(grams)
    return xs, truth, grams, ops


def test_block_sources_rows(monkeypatch):
    xs, truth, grams, ops = _block()
    keys = ['k0', 'k1', 'k2']
    vals, vecs = S.mbar_modes(ops, 9)
    m0 = np.stack([S.seba_sets(vecs, r)[0] for r in MB.RANKS])
    cache = {}
    for i, k in enumerate(keys):
        v_, U_ = S.mbar_modes([ops[i]], 9)
        cache[('cell', k, 'modes')] = (v_, U_)
        for r in MB.RANKS:
            cache[('cell', k, ('seba', r))] = S.seba_sets(U_, r)[0]
    tm = {}
    labels, agree = MB.block_sources(grams, ops, keys, m0, vecs, vals, truth, cache,
                                     S.CHEAP + ['L3', 'N3'], n_perm=20, timing=tm,
                                     extra_labels={'C1first': cache[('cell', 'k0', ('seba', 2))][None].repeat(5, 0)})
    assert set(labels) >= set(S.CHEAP) | {'L3', 'N3', 'M0'}
    for k, a in agree.items():
        assert a.shape == (5, len(MB.AGREE_FIELDS)), k
    assert np.allclose(agree['M0'][:, MB.AGREE_FIELDS.index('ami_all')], 1)
    bands = MB.max_bands({}, ['L1'])
    assert bands['maxnull_L1'].shape == (5, len(MB.MAX_FIELDS), 2)
    assert agree['L1'][2, MB.AGREE_FIELDS.index('class_ami')] > .8


def test_ablate_block_runs(monkeypatch):
    xs, truth, grams, ops = _block(n=120, L=2)
    Rs = {('s', i): (x @ x.T).numpy() for i, x in enumerate(xs)}
    monkeypatch.setattr(MB, 'load_raw', lambda s, si, view='flat', run='s0': Rs[(s, si)])
    monkeypatch.setattr(MB, 'load_op', lambda s, si, mode=0: S.lazy(__import__('coherent').symmetric_balance(
        -3.0 * (1 - grams[si]).fill_diagonal_(float('inf'))), .5))
    monkeypatch.setattr(MB, 'load_kappa', lambda s, si, mode: 3.0)
    cells = [('s', 0), ('s', 1)]
    vals, vecs = S.mbar_modes(ops, 9)
    real = {r: S.seba_sets(vecs, r)[0] for r in MB.RANKS}
    out = MB.ablate_block(grams, ops, cells, ['a', 'b'], real, vecs, truth, 'layer', 0, 0, 1, n_sub=2, n_n3=3)
    import json
    json.loads(MB._dumps(out), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    lv = out['A']['knn5']
    assert len(lv['S1']) == 2 and len(lv['n3']['excess_over_q95']) == 8
    ag = lv['agreement_with_M0'][3]
    for f in ('ami_joint_null_q95', 'mj_null_mean', 'mj_null_q95', 'ami_null_q95'):
        assert f in ag
    assert abs(lv['overlap_with_M0'][3]['chance'] - 2 / 119) < 1e-12
    # the stored-kappa level reproduces its stored operator on the full set (kappa 3 here)
    assert 'error' not in out['A']['vmf_stored_kNN_scale']
    assert out['D']['retained_diagonal']['laziness'] == 'none' and 'S1' in out['D']['retained_diagonal']
    assert set(out) == set('ABCDEFGHI') | {'conventions'}
    ok = [k for k, v in out['A'].items() if isinstance(v, dict) and 'error' not in v]
    assert 'knn15' in ok and 'knn5' in ok
    # the default level reproduces the default exactly
    assert out['A']['knn15']['agreement_with_M0'][4]['ami_all'] > 1 - 1e-9


@pytest.mark.parametrize('task', ['nulls', 'stability', 'linear'])
def test_run_rep_tasks(monkeypatch, task):
    import json
    xs, truth, grams, ops = _block(n=120, L=2)
    Rs = {('s', i): (x @ x.T).numpy() for i, x in enumerate(xs)}
    monkeypatch.setattr(MB, 'load_raw', lambda s, si, view='flat', run='s0': Rs[(s, si)])
    monkeypatch.setattr(MB, 'load_op', lambda s, si, mode=0: ops[0])
    cells = [('s', 0), ('s', 1)]
    res, draws = MB.run_rep('t', task, 'layer', 's', 0, 1, cells, grams, ops, truth, n_draws=2)
    # REVIEW_2 item 1: public JSON must parse in a browser (no NaN / Infinity tokens)
    text = MB._dumps(res)
    assert 'NaN' not in text and 'Infinity' not in text
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    if task == 'stability':
        # this configuration (n = 120, two cells, separation 2) has empty held-out sets,
        # which used to put bare NaN into reps.json; they must now be null
        assert 'null' in text
    if task == 'linear':
        assert res['cka_debiased_perm_null'] is not None and res['cka_perm_null_biased'] == res['cka_perm_null']
        assert 'method_dirichlet_under_cells_P' in res
    if task == 'stability':
        assert 'mj_null_q95' in res['S1'][0]['per_r'][3] and 'overlap_chance' in res['S1'][0]['per_r'][3]
        assert 'ami_joint_null_q95' in res['N6_rotation'][3][0]
    if task == 'nulls':
        assert set(draws) == {'N2a', 'N2b', 'N3', 'N4', 'N5'}
        assert len(res['bands']['N3']['lambda_P']) == 8
