"""Tests for stats.py (local: N <= 500, <= 2 threads)."""
import numpy as np
import torch

torch.set_num_threads(2)

import stats as T  # noqa: E402
from coherent import centred_cosine_gram  # noqa: E402


def feats(n=120, d=30, seed=0, shift=3.0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) + shift   # a large mean, so centring matters


def test_raw_to_views_full_matches_centred_cosine_gram():
    X = feats()
    R = X @ X.T
    C = centred_cosine_gram(torch.as_tensor(X)).numpy()
    v = T.raw_to_views(R)
    assert np.abs(v['C'] - C).max() < 1e-12
    Xu = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.abs(v['C_unc'] - Xu @ Xu.T).max() < 1e-12
    assert len(v['test_idx']) == 0


def test_raw_to_views_split_matches_feature_space():
    X = feats(n=150)
    R = X @ X.T
    rng = np.random.default_rng(1)
    fit = np.sort(rng.choice(150, 100, replace=False))
    test = np.setdiff1d(np.arange(150), fit)
    mu = X[fit].mean(0)
    Z = X - mu
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    v = T.raw_to_views(R, fit)
    assert np.array_equal(v['test_idx'], test)
    assert np.abs(v['C'] - Z[fit] @ Z[fit].T).max() < 1e-12
    assert np.abs(v['C_tf'] - Z[test] @ Z[fit].T).max() < 1e-12
    assert np.abs(v['C_tt'] - Z[test] @ Z[test].T).max() < 1e-12
    # fit-set view equals the full-set construction applied to the fit rows only
    C_fit_only = centred_cosine_gram(torch.as_tensor(X[fit])).numpy()
    assert np.abs(v['C'] - C_fit_only).max() < 1e-12


def test_covariance_spectrum():
    X = feats(n=80, d=10)
    lam = T.covariance_spectrum(X @ X.T)
    ref = np.linalg.eigvalsh(np.cov(X.T, bias=True))[::-1]
    assert np.allclose(lam[:10], ref, atol=1e-10) and np.abs(lam[10:]).max() < 1e-9


def test_agreement_relabel_and_unassigned():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 300)
    b = np.array([2, 0, 3, 1])[a]
    assert abs(T.ami(a, b) - 1) < 1e-12 and abs(T.ari(a, b) - 1) < 1e-12
    assert abs(T.matched_jaccard(a, b) - 1) < 1e-12
    a2 = a.copy(); a2[:30] = -1
    ag = T.agreement(a2, b)
    assert abs(ag['joint_coverage'] - .9) < 1e-12 and abs(ag['ami_joint'] - 1) < 1e-12
    assert ag['ami_all'] < 1
    # Two unassigned points are not a shared set: all-unassigned vs anything gives J = 0
    assert T.matched_jaccard(-np.ones(300, int), b) == 0


def test_matched_jaccard_denominators():
    a = np.repeat([0, 1], 50)
    b = np.repeat([0, 1, 2, 3], 25)
    d = T.matched_jaccard_detail(a, b)
    # two matched pairs, each J = 25/50; legacy = mean over matched = .5, max = 1/4 = .25
    assert abs(d['legacy'] - .5) < 1e-12 and abs(d['max'] - .25) < 1e-12


def test_conditional_entropy_refinement():
    fine = np.repeat(np.arange(6), 20)
    coarse = fine // 2
    assert abs(T.conditional_entropy(coarse, fine)) < 1e-12
    assert T.conditional_entropy(fine, coarse) > .5


def test_retention_size_null_is_exact_expectation():
    rng = np.random.default_rng(0)
    n = 30
    A = rng.random((n, n)); A = A + A.T; np.fill_diagonal(A, 0)
    # make a symmetric doubly stochastic P by Sinkhorn
    for _ in range(2000):
        d = A.sum(1); A = A / np.sqrt(d[:, None] * d[None, :])
    M = .5 * (np.eye(n) + A)
    for P in (A, M):
        m = 8
        vals = []
        for _ in range(20000):
            lab = -np.ones(n, int); lab[rng.choice(n, m, replace=False)] = 0
            vals.append(T.retention(P, lab)['values'][0])
        null = T.retention(P, lab)['null'][0]
        assert abs(np.mean(vals) - null) < 4 * np.std(vals) / np.sqrt(len(vals)) + 1e-3


def test_subspace_overlap():
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((200, 4)))[0]
    O = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    s = T.subspace_overlap(U, U @ O)
    assert abs(s['mean_cos2'] - 1) < 1e-10 and abs(s['min_cos'] - 1) < 1e-8
    V = np.linalg.qr(rng.standard_normal((200, 4)))[0]
    assert T.subspace_overlap(U, V)['mean_cos2'] < .2
    # the constant is projected out of both sides
    W = np.hstack([np.ones((200, 1)), U[:, :2]])
    assert T.subspace_overlap(W, U[:, :2])['q'] == (2, 2)


def test_linear_content_pca_is_optimum():
    X = feats(n=100, d=20)
    C = centred_cosine_gram(torch.as_tensor(X)).numpy()
    G = C - C.mean(0) - C.mean(1)[:, None] + C.mean()
    lam, U = np.linalg.eigh(G)
    lc = T.linear_content(U[:, ::-1][:, :3], G)
    assert abs(lc['value'] - lc['optimum']) < 1e-10


def test_cka_high_dimensional_isotropic_bias():
    rng = np.random.default_rng(0)
    n = 100
    C1 = centred_cosine_gram(torch.as_tensor(rng.standard_normal((n, 4000)))).numpy()
    C2 = centred_cosine_gram(torch.as_tensor(rng.standard_normal((n, 4000)))).numpy()
    assert abs(T.cka(C1, C1) - 1) < 1e-12
    assert T.cka(C1, C2) > .9           # independent clouds: biased CKA near 1 (IDEAS 5.1)
    assert abs(T.cka(C1, C2, debiased=True)) < .1


def test_perm_null_selection_inside_permutation():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 3, 200)
    cands = [rng.integers(0, 3, 200) for _ in range(5)]
    res = T.perm_null(T.ami, cands, b, n_perm=50, seed=0)
    assert abs(res['observed'] - max(T.ami(c, b) for c in cands)) < 1e-12
    single = T.perm_null(T.ami, cands[0], b, n_perm=50, seed=0)
    assert res['null']['mean'] >= single['null']['mean']   # max over 5 candidates is larger


def test_fast_ami_null_is_exact_per_draw():
    rng = np.random.default_rng(1)
    a = rng.integers(-1, 4, 300); b = rng.integers(0, 6, 300); b[:20] = -1
    perms = np.array([rng.permutation(300) for _ in range(15)])
    out = T.fast_ami_null(a, b, perms=perms, return_draws=True)
    for p, am, mj in zip(perms, out['ami_draws'], out['mj_draws']):
        assert abs(am - T.ami(a, b[p])) < 1e-10
        assert abs(mj - T.matched_jaccard(a, b[p])) < 1e-12


# ---------------------------------------------------------------- REVIEW_2 item 3: declared set counts

def test_matched_jaccard_empty_set_position_does_not_matter():
    rng = np.random.default_rng(3)
    part = rng.integers(0, 2, 200)                 # a 2-set partition declared as r = 3
    trailing = part.copy()                         # labels {0, 1}, empty set 2
    internal = np.where(part == 1, 2, 0)           # labels {0, 2}, empty set 1
    b = rng.integers(0, 3, 200)
    for denom in ('max', 'legacy', 'weighted'):
        x = T.matched_jaccard(trailing, b, denom, ka=3, kb=3)
        y = T.matched_jaccard(internal, b, denom, ka=3, kb=3)
        assert abs(x - y) < 1e-12, denom
    # the reviewer's case: the same partition against itself, numbered two ways
    assert abs(T.matched_jaccard(trailing, internal, ka=3, kb=3) - 2 / 3) < 1e-12
    assert abs(T.matched_jaccard(trailing, trailing, ka=3, kb=3) - 2 / 3) < 1e-12
    # arbitrary relabelling of both sides leaves every summary unchanged
    pa, pb = np.array([2, 0, 1]), np.array([1, 2, 0])
    for denom in ('max', 'legacy', 'weighted'):
        assert abs(T.matched_jaccard(pa[trailing], pb[b], denom, 3, 3) - T.matched_jaccard(trailing, b, denom, 3, 3)) < 1e-12
    import pytest
    with pytest.raises(ValueError):
        T.matched_jaccard(np.array([0, 3]), np.array([0, 1]), ka=3, kb=3)


def test_fast_null_declared_counts_and_relabelling():
    rng = np.random.default_rng(4)
    part = rng.integers(0, 2, 150)
    trailing, internal = part.copy(), np.where(part == 1, 2, 0)
    b = rng.integers(-1, 4, 150)
    perms = np.array([rng.permutation(150) for _ in range(30)])
    x = T.fast_ami_null(trailing, b, perms=perms, return_draws=True, ka=3, kb=4)
    y = T.fast_ami_null(internal, b, perms=perms, return_draws=True, ka=3, kb=4)
    assert np.allclose(x['mj_draws'], y['mj_draws']) and np.allclose(x['ami_draws'], y['ami_draws'])
    for p, mj in zip(perms, x['mj_draws']):
        assert abs(mj - T.matched_jaccard(trailing, b[p], ka=3, kb=4)) < 1e-12
    assert abs(x['observed_mj_max'] - T.matched_jaccard(trailing, b, ka=3, kb=4)) < 1e-12


def test_retention_internal_empty_set_and_finite_json():
    import json
    P = np.full((6, 6), 1 / 5); np.fill_diagonal(P, 0)
    lab = np.array([0, 0, 2, 2, 2, -1])            # set 1 declared but empty
    ret = T.retention(P, lab, 3)
    assert np.isnan(ret['values'][1]) and np.isnan(ret['null'][1]) and ret['sizes'][1] == 0
    s = T.dumps(dict(ret=ret, x=[float('inf'), np.float32(1.5)], n=np.int64(3)))
    d = json.loads(s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert d['ret']['values'][1] is None and d['x'] == [None, 1.5] and d['n'] == 3
    assert d['ret']['sizes'] == [2, 0, 3]
