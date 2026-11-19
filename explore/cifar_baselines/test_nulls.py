"""Tests for nulls.py and variants.py (local: N <= 500, <= 2 threads)."""
import numpy as np
import torch

torch.set_num_threads(2)

import nulls as N  # noqa: E402
import sources as S  # noqa: E402
import stats as T  # noqa: E402
import variants as V  # noqa: E402
from coherent import centred_cosine_gram, symmetric_balance  # noqa: E402
from test_sources import planted_block, grams_of  # noqa: E402


def unit_rows(x):
    z = x - x.mean(0)
    return z / z.norm(dim=1, keepdim=True)


def test_haar_fixing_constant():
    O = N.haar_fixing_constant(50, seed=1)
    assert np.abs(O @ O.T - np.eye(50)).max() < 1e-12
    assert np.abs(O @ np.ones(50) - np.ones(50)).max() < 1e-12


def test_n2b_preserves_joint_second_moments_before_renormalisation():
    xs, _ = planted_block(n=120, L=3, d=20, seed=2)
    Z = [unit_rows(x).numpy() for x in xs]
    O = N.haar_fixing_constant(120, seed=3)
    Zs = [O @ z for z in Z]
    for a in range(3):
        assert np.abs(Zs[a].sum(0) - Z[a].sum(0)).max() < 1e-10           # means
        for b in range(3):
            assert np.abs(Zs[a].T @ Zs[b] - Z[a].T @ Z[b]).max() < 1e-10   # within/cross
        C = Z[a] @ Z[a].T
        assert np.abs(O @ C @ O.T - Zs[a] @ Zs[a].T).max() < 1e-10       # Gram-only form
    out, info = N.n2b_grams([torch.as_tensor(z @ z.T) for z in Z], O=O)
    for c in out:
        assert np.abs(np.diag(c.numpy()) - 1).max() < 1e-12
    assert np.isfinite(info['second_moment_rel_change_max'])
    assert info['diag_before_min'] > 0


def test_n2a_matches_covariance_spectrum_in_expectation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30)) * np.linspace(3, .1, 30)
    R = X @ X.T
    g = N.n2a_grams([R], seed=0)[0].numpy()
    assert g.shape == (200, 200) and np.abs(np.diag(g) - 1).max() < 1e-12


def test_n3_fixes_first_cell_and_preserves_cells():
    xs, _ = planted_block(n=150, L=3, seed=4)
    ops = S.default_ops(grams_of(xs))
    pops = N.n3_ops(ops, ['t0s0', 't0s1', 't0s2'], seed=7)
    assert torch.equal(pops[0], ops[0])
    for a, b in zip(ops, pops):
        assert np.allclose(np.linalg.eigvalsh(a.numpy()), np.linalg.eigvalsh(b.numpy()), atol=1e-10)
    # coherence: the same (seed, key) gives the same permutation in another block
    p1 = N.cell_permutations(['x', 'y'], 50, seed=3); p2 = N.cell_permutations(['y'], 50, seed=3)
    assert np.array_equal(p1['y'], p2['y'])
    # N4 permutes within strata only
    strata = np.arange(50) % 5
    p = N.cell_permutations(['y'], 50, seed=3, strata=strata)['y']
    assert np.array_equal(strata[p], strata)
    out = N.n3_source(grams_of(xs), 3, ops=ops, cell_keys=['a', 'b', 'c'])
    assert out['labels'].shape == (150,)


def test_n3_destroys_cross_cell_alignment():
    xs, truth = planted_block(n=240, L=3, sep=3.0)
    grams = grams_of(xs)
    ops = S.default_ops(grams)
    lab = N.n3_source(grams, 3, ops=ops, cell_keys=list('abc'), seed=1)['labels']
    assert T.ami(S.SOURCES['M0'](grams, 3, ops=ops)['labels'], truth) > .9
    assert T.ami(lab, truth) < .9


def test_n5_rewire_preserves_degrees():
    xs, _ = planted_block(n=150, L=1, seed=5)
    mask = S.knn_mask(grams_of(xs)[0]).numpy()
    B, info = N.rewire(mask, seed=0)
    assert np.array_equal(B.sum(1), mask.sum(1)) and not np.diag(B).any() and (B == B.T).all()
    assert info['original_edges_kept'] < .5


def test_n6():
    U = N.haar_subspace(300, 3, seed=0)
    assert np.abs(U.sum(0)).max() < 1e-10 and np.abs(U.T @ U - np.eye(3)).max() < 1e-12
    res = N.n6_haar(300, 4, n_draws=5)
    assert res['draws'] == 5
    xs, _ = planted_block(n=150, L=1, sep=3.0)
    _, v = S.mbar_modes(S.default_ops(grams_of(xs)), 9)
    rows = N.n6_rotation(v, 3, n_draws=3)
    assert all(r['ami_all'] > .9 for r in rows)


def test_spectral_summary():
    xs, truth = planted_block(n=150, L=2)
    ops = S.default_ops(grams_of(xs))
    out, labels, vecs = N.spectral_summary(ops, ranks=[2, 3], classes=truth)
    assert len(out['lambda_P']) == 8 and out['seba'][3]['class_ami'] > .8


# ---------------------------------------------------------------- variants

def test_seba_mu_default_reproduces_seba():
    xs, _ = planted_block(n=200, L=2, sep=1.0)
    _, v = S.mbar_modes(S.default_ops(grams_of(xs)), 9)
    for r in (3, 6):
        assert np.array_equal(V.seba_mu(v, r, .99)[0], S.seba_sets(v, r)[0])


def test_scaling_vector_recovered():
    xs, _ = planted_block(n=150, L=1)
    lk = S.knn_logkernel(grams_of(xs)[0])
    P = symmetric_balance(lk)
    u = V._scaling_vector(P, lk)
    rebuilt = np.exp(lk.numpy() + u[:, None] + u[None, :])
    assert np.abs(rebuilt - P.numpy()).max() < 1e-12


def test_fit_extend_recovers_planted_on_test_images():
    xs, truth = planted_block(n=300, L=2, sep=3.0)
    Rs = [(x @ x.T).numpy() for x in xs]
    fit = np.sort(np.random.default_rng(0).choice(300, 225, replace=False))
    out = V.fit_extend(Rs, fit, ranks=[3])
    test = np.setdiff1d(np.arange(300), fit)
    # extended sets on test images vs the truth; and retained above the size null
    for name in ('M0', 'L1', 'L3'):
        d = out[name][3]
        assert np.all(np.array(d['test_retention']) > np.array(d['test_null'])), name


def test_variant_operators_and_combinations():
    xs, _ = planted_block(n=150, L=3, sep=1.0)
    grams = grams_of(xs)
    for fam, p in [('knn', dict(k=5)), ('knn', dict(mutual=True)), ('perplexity', dict(perplexity=30)),
                   ('vmf', dict(kappa=5.)), ('local', {}), ('eps', {})]:
        M, info = V.operator(V.logkernel(grams[0], fam, **p))
        assert np.abs(M.sum(1).numpy() - 1).max() < 1e-9, fam
    Sm, info = V.operator(V.logkernel(grams[0], 'knn'), norm='degree', alpha=.5)
    assert np.allclose(Sm.numpy(), Sm.numpy().T)
    ops = S.default_ops(grams)
    mean = V.combine(ops, 'mean'); es = V.combine(ops, 'expsum')
    assert T.subspace_overlap(mean['vecs'][:, :5], es['vecs'][:, :5])['min_cos'] > 1 - 1e-6
    for how in ('meansq', 'twoend', 'expprod'):
        assert V.combine(ops, how)['vals'].shape[0] == 8
    for how in ('concat', 'union', 'intersection'):
        assert V.combine(ops, how, grams=grams)['vecs'].shape == (150, 8)
    w = V.time_weights([1024, 2560, 6400, 25600, 64000], 'interval')
    assert abs(w.sum() - 1) < 1e-12 and w[-1] > w[0]
    assert V.commutators(ops).shape == (3, 3)
    loo = V.leave_one_out(ops, 3, next_op=ops[0])
    assert len(loo['rows']) == 3 and 'next' in loo


def test_effective_neighbours_match():
    xs, _ = planted_block(n=150, L=1, sep=1.0)
    c = grams_of(xs)[0]
    target = V.effective_neighbours(symmetric_balance(S.knn_logkernel(c)))
    k, val = V.match_neighbours(c, 'vmf', target, 0.1, 200.)
    assert abs(val - target) / target < .1


# ---------------------------------------------------------------- REVIEW_2 item 2: laziness convention

def test_spectral_summary_laziness_matches_direct_eigendecomposition():
    xs, _ = planted_block(n=150, L=2, sep=1.0)
    grams = grams_of(xs)
    # lazy default: reported lambda_P / bottom are those of P = 2M - I
    ops = S.default_ops(grams)
    out, _, _ = N.spectral_summary(ops, ranks=[3])
    Pbar = 2 * np.mean([o.numpy() for o in ops], 0) - np.eye(150)
    ev = np.sort(np.linalg.eigvalsh(Pbar))
    assert out['laziness'] == 'half'
    assert np.allclose(out['lambda_P'], ev[::-1][1:9], atol=1e-8)
    assert np.allclose(out['lambda_P_bottom'], ev[:3], atol=1e-8)
    # retained diagonal: the operator is the balanced kernel itself, no lazy step
    rops = [V.operator(V.logkernel(g, 'knn'), diagonal='retained')[0] for g in grams]
    out2, _, _ = N.spectral_summary(rops, ranks=[3], laziness='none')
    A = np.mean([o.numpy() for o in rops], 0)
    ev2 = np.sort(np.linalg.eigvalsh(A))
    assert out2['laziness'] == 'none'
    assert np.allclose(out2['lambda_P'], ev2[::-1][1:9], atol=1e-8)
    assert np.allclose(out2['lambda_P_bottom'], ev2[:3], atol=1e-8)
    import pytest
    with pytest.raises(ValueError):
        N.spectral_summary(ops, laziness='quarter')
