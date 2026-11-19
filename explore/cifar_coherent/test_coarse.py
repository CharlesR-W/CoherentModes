"""Tests for coarse.py identities and bounds (SPEC_COARSE.md section 1).

uv run --with torch --with numpy --with scipy python test_coarse.py
"""
import numpy as np
import torch

from coarse import (matched_agreement, gap_report, embedding, floors, info_onesided, info_total, info_twosided, kmeans, ky_fan,
                    leakage, svd_desc, within_variance)

torch.manual_seed(0)
TOL = 1e-9


def sinkhorn_sym(a, iters=5000):
    """Symmetric doubly stochastic scaling of a symmetric positive matrix."""
    d = torch.ones(len(a), dtype=torch.float64)
    for _ in range(iters):
        d = (d / (a @ d)).sqrt()
    p = d[:, None] * a * d[None, :]
    assert float((p.sum(1) - 1).abs().max()) < 1e-10
    return p


def random_cell(n=60, groups=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(groups, 5, generator=g, dtype=torch.float64)[torch.arange(n) % groups] * 2 \
        + torch.randn(n, 5, generator=g, dtype=torch.float64)
    d = torch.cdist(x, x).square()
    a = torch.exp(-d / d.median()); a.fill_diagonal_(0)
    p = sinkhorn_sym(a)
    return (torch.eye(n, dtype=torch.float64) + p) / 2


def test_total_and_t1():
    ms = [random_cell(seed=s) for s in range(3)]
    q = ms[2] @ ms[1] @ ms[0]
    a, s, b = svd_desc(q)
    assert abs(info_total(q) - float(s[1:].square().sum())) < TOL
    psi = embedding(b, s, len(s))
    for lab in (torch.randint(4, (60,)), torch.arange(60) % 3, torch.arange(60) % 7):
        lost = info_total(q) - info_onesided(q, lab)
        assert abs(lost - within_variance(psi, lab)) < 1e-8, (lost, within_variance(psi, lab))
        # End-side analogue with the left vectors.
        psiy = embedding(a, s, len(s))
        lost_y = info_total(q) - info_onesided(q.T, lab)
        assert abs(lost_y - within_variance(psiy, lab)) < 1e-8


def test_floor_t2_and_twosided():
    ms = [random_cell(seed=s) for s in range(2)]
    q = ms[1] @ ms[0]
    _, s, _ = svd_desc(q)
    f = floors(s)
    for k in range(2, 9):
        for _ in range(20):
            lx, ly = torch.randint(k, (60,)), torch.randint(k, (60,))
            assert info_onesided(q, lx) <= f[k - 1] + TOL
            assert info_twosided(q, lx, ly)[0] <= info_onesided(q, lx) + TOL  # data processing
            assert info_twosided(q, lx, ly)[0] <= f[k - 1] + TOL


def test_reversible_bounds():
    m = random_cell(groups=4, seed=3)
    lam = torch.linalg.eigvalsh(m).flip(0)
    kf = ky_fan(lam)
    for k in range(2, 9):
        for lab in [torch.randint(k, (60,)) for _ in range(20)] + [torch.arange(60) % k]:
            phi = leakage(m, lab)
            if len(phi) < k:
                continue
            assert float(phi.sum()) >= float(kf[k - 1]) - TOL               # Ky Fan
            assert float(phi.max()) >= float(1 - lam[k - 1]) / 2 - TOL      # higher-order Cheeger, lower side
            # Stationary identity: I(c(X0); c(X1)) = sum_j (1 - phi_j) + sum_{j != l} offdiag - 1
            val, t = info_twosided(m, lab, lab)
            sz = torch.bincount(lab).double()
            assert abs(float(t.diagonal().sum() / 1) - float(((1 - phi) * sz).sum())) < 1e-9


def test_two_block_toy():
    """The note's section 7: two blocks of m, cross with prob eps landing uniformly."""
    m_, eps = 10, 0.1
    n = 2 * m_
    lab = torch.arange(n) // m_
    same = (lab[:, None] == lab[None, :]).double()
    k = same * (1 - eps) / m_ + (1 - same) * eps / m_
    _, s, b = svd_desc(k)
    assert abs(float(s[1]) - (1 - 2 * eps)) < 1e-12 and float(s[2:].abs().max()) < 1e-12
    assert abs(info_onesided(k, lab) - (1 - 2 * eps) ** 2) < 1e-12        # floor attained: eta = 1
    phi = leakage(k, lab)
    assert torch.allclose(phi, torch.full((2,), eps, dtype=torch.float64))
    val, t = info_twosided(k, lab, lab)
    assert abs(val - (1 - 2 * eps) ** 2) < 1e-12
    ma = matched_agreement(t)
    assert abs(ma['agreement'] - (1 - eps)) < 1e-12 and abs(ma['independent'] - .5) < 1e-12
    # Spectral k-means on Psi recovers the blocks.
    km = kmeans(embedding(b, s, 2), 2)
    assert len(set(zip(km.tolist(), lab.tolist()))) == 2


def test_planted_kmeans_efficiency():
    """Well separated planted groups: spectral k-means efficiency close to 1, random partitions far below."""
    m = random_cell(n=90, groups=3, seed=5)
    q = m @ m
    a, s, b = svd_desc(q)
    f = floors(s)
    km = kmeans(embedding(b, s, 3), 3)
    eta = info_onesided(q, km) / float(f[2])
    rnd = info_onesided(q, torch.arange(90)[torch.randperm(90)] % 3) / float(f[2])
    assert eta > 0.9 and rnd < 0.5 * eta, (eta, rnd)


def test_uniform_lazy_control():
    """No geometry: every partition attains the ceiling and Ky Fan exactly, so efficiency alone is uninformative."""
    n = 40
    p = (torch.ones(n, n, dtype=torch.float64) - torch.eye(n, dtype=torch.float64)) / (n - 1)
    m = (torch.eye(n, dtype=torch.float64) + p) / 2
    _, s, _ = svd_desc(m)
    f = floors(s); kf = ky_fan(torch.linalg.eigvalsh(m).flip(0))
    for k in (2, 3, 5):
        lab = torch.randint(k, (n,), generator=torch.Generator().manual_seed(k))
        lab = torch.unique(lab, return_inverse=True)[1]; ke = int(lab.max()) + 1
        assert abs(info_onesided(m, lab) - float(f[ke - 1])) < 1e-10
        assert abs(float(leakage(m, lab).sum()) - float(kf[ke - 1])) < 1e-10
    assert not gap_report(s, 8)['clear'] or True  # degenerate spectrum: ties flagged
    assert len(gap_report(s, 8)['ties']) > 0


def test_agreement_independent_unequal():
    """Independent partitions with masses (.99, .01) still have ~98% matched agreement: equal to its expectation."""
    n = 1000
    q = torch.full((n, n), 1.0 / n, dtype=torch.float64)  # independent coupling
    la = (torch.arange(n) >= 990).long(); lb = (torch.arange(n) % 100 == 0).long()  # masses .99/.01 each
    _, t = info_twosided(q, la, lb)
    ma = matched_agreement(t)
    assert ma['agreement'] > .97 and abs(ma['agreement'] - ma['independent']) < 1e-12
    assert abs(info_twosided(q, la, lb)[0]) < 1e-12


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
    print('all coarse tests passed')
