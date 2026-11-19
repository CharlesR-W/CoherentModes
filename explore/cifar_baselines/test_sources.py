"""Tests for sources.py: the contract, M0 = the stored pipeline, analytic controls."""
import os

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

import sources as S  # noqa: E402
import stats as T  # noqa: E402
from coherent import centred_cosine_gram  # noqa: E402


def planted_block(n=240, L=3, k=3, d=40, sep=2.5, seed=0):
    """L cells sharing a k-group partition; each cell its own random embedding + noise."""
    g = torch.Generator().manual_seed(seed)
    truth = torch.arange(n) % k
    xs = []
    for l in range(L):
        centres = torch.randn(k, d, generator=g, dtype=torch.float64) * sep
        xs.append(centres[truth] + torch.randn(n, d, generator=g, dtype=torch.float64))
    return xs, truth.numpy()


def grams_of(xs):
    return [centred_cosine_gram(x) for x in xs]


def test_rule0_kernel_is_the_pipeline_kernel():
    from sweep_math import final_kernel_bank
    xs, _ = planted_block(n=450, L=1, sep=.5)   # perplexity 300 needs N well above 300
    c = centred_cosine_gram(xs[0])
    ref = final_kernel_bank(c)[0][0]
    assert torch.equal(S.knn_logkernel(c), ref)


def test_M0_reproduces_block_record():
    from schema_v2 import block_record, RANKS
    xs, _ = planted_block(n=300, L=3, sep=1.0, seed=3)
    grams = grams_of(xs)
    ops = S.default_ops(grams)
    rec, _ = block_record(ops, 0, len(ops) - 1)
    cache = {}
    for ri, r in enumerate(RANKS):
        out = S.SOURCES['M0'](grams, r, ops=ops, cache=cache, cell_keys=[0, 1, 2])
        assert np.array_equal(out['labels'], rec['seba_label'][ri].astype(np.int64)), r
    vals = S.SOURCES['M0'](grams, 3, ops=ops)['values']
    assert np.allclose(vals, rec['vals'], atol=1e-6)


@pytest.mark.parametrize('name', sorted(S.SOURCES))
def test_source_contract(name):
    xs, truth = planted_block(n=180, L=3)
    grams = grams_of(xs)
    cache = {}
    for r in (2, 4):
        out = S.SOURCES[name](grams, r, seed=0, cache=cache, cell_keys=['a', 'b', 'c'])
        lab = out['labels']
        assert lab.shape == (180,) and lab.dtype == np.int64
        assert lab.min() >= -1 and lab.max() < r
        assert set(out['status'].split('+')) <= set('SPBV')
        if out['basis'] is not None:
            B = out['basis']
            assert B.shape == (180, r - 1)
            assert np.abs(B.T @ B - np.eye(r - 1)).max() < 1e-6
            assert np.abs(B.sum(0)).max() < 1e-6
        assert isinstance(out['info'], dict)


def test_easy_planted_recovered_by_main_sources():
    xs, truth = planted_block(n=240, L=3, sep=3.0)
    grams = grams_of(xs)
    cache = {}
    for name in ('M0', 'L1', 'L2', 'L3', 'G1', 'G1n', 'G2', 'C2', 'C3'):
        lab = S.SOURCES[name](grams, 3, cache=cache, cell_keys=[0, 1, 2])['labels']
        assert T.ami(lab, truth) > .9, name


def test_top_eigs_iterative_matches_dense():
    xs, _ = planted_block(n=500, L=1, sep=1.0, d=60)
    M = S.default_operator(centred_cosine_gram(xs[0])).numpy()
    v1, U1, i1 = S.top_eigs(M, 8, dense=False)
    v2, U2, i2 = S.top_eigs(M, 8, dense=True)
    assert np.allclose(v1, v2, atol=1e-9)
    assert T.subspace_overlap(U1, U2)['min_cos'] > 1 - 1e-6
    vals, vecs = S.mbar_modes([M], 9)
    assert np.allclose(vals[1:], v2, atol=1e-9)


def test_spherical_kmeans_gram_matches_feature_objective():
    xs, truth = planted_block(n=150, L=1, sep=3.0)
    C = centred_cosine_gram(xs[0]).numpy()
    lab, info = S.spherical_kmeans_gram(C, 3)
    assert T.ami(lab, truth) > .95
    # objective computed in feature space with unit-normalised set sums
    Z = xs[0] - xs[0].mean(0); Z = (Z / Z.norm(dim=1, keepdim=True)).numpy()
    obj = sum(np.linalg.norm(Z[lab == j].sum(0)) for j in range(3))
    assert abs(obj - info['objective']) < 1e-8


# ---------------------------------------------------------------- analytic controls

def test_common_permutation_only_relabels():
    xs, _ = planted_block(n=200, L=3, sep=1.0, seed=5)
    grams = grams_of(xs)
    ops = S.default_ops(grams)
    p = np.random.default_rng(0).permutation(200)
    pops = [m[p][:, p] for m in ops]
    for r in (3, 6):
        a = S.SOURCES['M0'](grams, r, ops=ops)
        b = S.SOURCES['M0'](grams, r, ops=pops)
        assert T.ami(a['labels'][p], b['labels']) > 1 - 1e-9, r
        assert np.allclose(a['values'], b['values'], atol=1e-10)


def test_orthogonal_feature_rotation_changes_nothing():
    xs, _ = planted_block(n=200, L=2, sep=1.0, seed=6)
    Q = torch.linalg.qr(torch.randn(40, 40, dtype=torch.float64))[0]
    g1 = grams_of(xs)
    g2 = grams_of([x @ Q for x in xs])
    assert max(float((a - b).abs().max()) for a, b in zip(g1, g2)) < 1e-12
    for r in (3, 6):
        assert np.array_equal(S.SOURCES['M0'](g1, r)['labels'], S.SOURCES['M0'](g2, r)['labels'])


def test_common_laziness_leaves_averaged_eigenspaces_unchanged():
    from coherent import symmetric_balance
    xs, _ = planted_block(n=200, L=3, sep=1.0, seed=7)
    Ps = [symmetric_balance(S.knn_logkernel(c)) for c in grams_of(xs)]
    ref = None
    for gamma in (0.0, .5, .9):
        ms = [S.lazy(p.clone(), gamma) for p in Ps]
        A = np.mean([m.numpy() for m in ms], 0)
        vals, U, _ = S.top_eigs(A, 6, dense=True)
        if ref is None:
            ref = (vals, U)
        else:
            assert T.subspace_overlap(ref[1], U)['min_cos'] > 1 - 1e-8
            assert np.allclose(gamma + (1 - gamma) * ref[0], vals, atol=1e-10)


def test_product_reversal_gives_transpose():
    import variants as V
    xs, _ = planted_block(n=150, L=3, sep=1.0, seed=8)
    ops = S.default_ops(grams_of(xs))
    fwd = V.combine(ops, 'product')
    rev = V.combine(ops[::-1], 'product')
    assert np.allclose(fwd['vals'], rev['vals'], atol=1e-12)
    assert T.subspace_overlap(fwd['left'][:, :4], rev['right'][:, :4])['min_cos'] > 1 - 1e-8
    assert T.subspace_overlap(fwd['right'][:, :4], rev['left'][:, :4])['min_cos'] > 1 - 1e-8


def test_C3_identical_cells_returns_cell_span():
    xs, _ = planted_block(n=150, L=1, sep=1.0)
    g = grams_of(xs) * 3
    m0 = S.SOURCES['M0'](g, 4)
    c3 = S.SOURCES['C3'](g, 4)
    assert T.subspace_overlap(m0['basis'], c3['basis'])['min_cos'] > 1 - 1e-8


@pytest.mark.skipif(not os.environ.get('CM_SLOW'), reason='N = 2000 fixture; run on Modal or with CM_SLOW=1')
def test_M0_reproduces_fixture_labels():
    import measure_b
    report = measure_b.fixture_check_local()
    assert report['all_equal'], report
