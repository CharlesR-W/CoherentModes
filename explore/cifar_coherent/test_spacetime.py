"""Checks for spacetime.py. Run: uv run --with numpy --with scipy python test_spacetime.py"""
import numpy as np

import spacetime as S


def walk(x, kappa=1.5, iters=4000):
    """Symmetric doubly stochastic lazy walk from points x (n, d): zero-diagonal Gaussian kernel, scaled."""
    d2 = ((x[:, None] - x[None]) ** 2).sum(-1)
    k = np.exp(-kappa * d2)
    np.fill_diagonal(k, 0)
    v = np.ones(len(x))
    for _ in range(iters):
        v = np.sqrt(v / (k @ v))
    p = k * v[:, None] * v[None]
    assert np.abs(p.sum(1) - 1).max() < 1e-10
    return (np.eye(len(x)) + p) / 2


def clouds(rng, centres, n_per, noise=.3):
    return np.concatenate([c + noise * rng.standard_normal((n_per, len(c))) for c in centres])


def dense_L(ms, c, a2):
    nl, n, _ = ms.shape
    out = np.zeros((nl * n, nl * n))
    for l in range(nl):
        out[l * n:(l + 1) * n, l * n:(l + 1) * n] = (np.eye(n) - ms[l]) / c[l]
    path = np.zeros((nl, nl))
    for l in range(nl - 1):
        path[l, l] += 1; path[l + 1, l + 1] += 1; path[l, l + 1] -= 1; path[l + 1, l] -= 1
    return out + a2 * np.kron(path, np.eye(n))


def make(rng, nl=5, n_per=20):
    """Three groups; groups 0 and 1 approach each other with depth and coincide at the last two sites."""
    ms = []
    for l in range(nl):
        gap = max(0.0, 1.0 - l / (nl - 2))
        x = clouds(rng, [np.array([0., 0.]), np.array([gap, 0.]), np.array([.5, 2.])], n_per)
        ms.append(walk(x))
    return np.stack(ms)


def test_matches_dense_and_temporal_sector():
    rng = np.random.default_rng(0)
    ms = make(rng)
    nl, n, _ = ms.shape
    for mode in ('none', 'own'):
        c = S.site_scales(ms, mode)
        for a2 in (.01, .3, 5.):
            dl = dense_L(ms, c, a2)
            const = np.kron(np.eye(nl), np.ones((n, 1)) / np.sqrt(n))        # slice-constant functions
            pc = const @ const.T
            assert np.abs(dl @ pc - pc @ dl).max() < 1e-12, 'temporal sector does not decouple'
            assert np.allclose(np.linalg.eigvalsh(const.T @ dl @ const), np.sort(S.temporal_eigs(nl, a2)), atol=1e-9)
            comp = np.linalg.svd(np.eye(nl * n) - pc)[0][:, :nl * (n - 1)]   # basis of the complement
            w = np.linalg.eigvalsh(comp.T @ dl @ comp)
            vals, vecs = S.modes(ms, c, a2, k=8, tol=1e-11)
            assert np.allclose(vals, w[:8], atol=1e-8)
            assert S.residual(ms, c, a2, vals, vecs).max() < 1e-8
            assert np.abs(vecs.mean(1)).max() < 1e-12
            flat = vecs.reshape(-1, 8)
            assert np.allclose(flat.T @ flat, np.eye(8), atol=1e-8)
            rough, change = S.energy_terms(ms, c, vecs)
            assert np.allclose(rough.sum(0) + a2 * change.sum(0), vals, atol=1e-8)


def test_identical_sites_factorise():
    rng = np.random.default_rng(1)
    m = walk(clouds(rng, [np.array([0., 0.]), np.array([1., 0.]), np.array([.5, 2.])], 20))
    nl, n = 6, len(m)
    ms = np.stack([m] * nl)
    lam, phi = np.linalg.eigh(m - 1.0 / n)
    lam, phi = lam[::-1], phi[:, ::-1]
    a2 = .02
    want = np.sort(((1 - lam[:n - 1])[:, None] + S.temporal_eigs(nl, a2)[None]).ravel())[:10]
    vals, vecs = S.modes(ms, np.ones(nl), a2, k=10, tol=1e-11)
    assert np.allclose(vals, want, atol=1e-8)
    # first mode: phi_1 (x) constant; then phi_1 (x) cos(pi (l + 1/2) / L) if cheaper than phi_2
    assert np.allclose(np.abs(S.adjacent_cos(vecs)[0]), 1, atol=1e-8)
    prof = np.cos(np.pi * (np.arange(nl) + .5) / nl)
    j = int(np.argmin(np.abs(vals - (1 - lam[0] + S.temporal_eigs(nl, a2)[1]))))
    amp = vecs[:, :, j] @ phi[:, 0]
    assert abs(abs(amp @ prof) / (np.linalg.norm(amp) * np.linalg.norm(prof)) - 1) < 1e-6


def test_limits():
    rng = np.random.default_rng(2)
    ms = make(rng)
    nl, n, _ = ms.shape
    c = np.ones(nl)
    # weak coupling: the spectrum is the union of the sites' own spectra, slices are own eigenvectors
    own = np.sort(np.concatenate([1 - S.site_spectrum(m, n - 1) for m in ms]))[:6]
    vals, vecs = S.modes(ms, c, 1e-7, k=6, tol=1e-12)
    assert np.allclose(vals, own, atol=1e-5)
    # strong coupling: slice-independent eigenvectors of the block average
    mbar = ms.mean(0)
    lam, phi = np.linalg.eigh(mbar - 1.0 / n)
    vals, vecs = S.modes(ms, c, 1e4, k=3, tol=1e-12)
    assert np.allclose(vals, (1 - lam[::-1][:3]), atol=1e-3)
    assert S.adjacent_cos(vecs).min() > 1 - 1e-6
    assert np.allclose(S.depth_weight(vecs), 1 / nl, atol=1e-3)
    assert np.allclose(S.span_share(vecs, phi[:, ::-1][:, :3]), 1, atol=1e-4)
    # with site scales the strong-coupling limit is the average of the scaled generators
    c = S.site_scales(ms, 'own')
    gen = np.mean([(np.eye(n) - m) / ci for m, ci in zip(ms, c)], 0)
    w = np.linalg.eigvalsh(gen)
    assert np.allclose(S.modes(ms, c, 1e4, k=3, tol=1e-12)[0], w[1:4], atol=1e-3)


def test_paper_matrix():
    rng = np.random.default_rng(3)
    ms = make(rng)
    nl, n, _ = ms.shape
    b = S.PAPER_B(nl)
    a = np.zeros((nl * n, nl * n))
    for l in range(nl - 1):
        a[l * n:(l + 1) * n, (l + 1) * n:(l + 2) * n] = ms[l]
        a[(l + 1) * n:(l + 2) * n, l * n:(l + 1) * n] = ms[l]
    cmat = a / np.repeat(b, n)[:, None]
    assert np.allclose(cmat.sum(1), 1)                                   # row-stochastic (Prop. III.4)
    w = np.sort(np.linalg.eigvals(cmat).real)
    assert np.allclose(w, -w[::-1], atol=1e-9) and w.max() < 1 + 1e-9    # symmetric about 0, inside [-1, 1]
    vals, f = S.paper_modes(ms, k=6, tol=1e-12)
    flat = f.reshape(-1, 6)
    assert np.allclose(cmat @ flat, flat * vals, atol=1e-7)              # eigenvectors of C itself
    # slice-constant ("temporal") eigenvalues cos(j pi / (L-1)) are exactly the ones left out
    temporal = np.cos(np.arange(nl) * np.pi / (nl - 1))
    rest = np.sort(np.concatenate([vals, temporal[temporal > vals.min()]]))
    assert np.allclose(rest, w[-len(rest):], atol=1e-7)
    lhs, own, nxt, change = S.paper_terms(ms, f)
    assert np.allclose(lhs, own + nxt + change, atol=1e-10)
    assert np.allclose(lhs, 1 - vals, atol=1e-7)                         # = 1 - lambda, since sum_l b_l |f_l|^2 = 1
    # two sites: ordinary CCA; lambda are the eigenvalues of M_0 (forward-backward operator M_0^2)
    vals2, _ = S.paper_modes(ms[:2], k=4, tol=1e-12)
    assert np.allclose(vals2, S.site_spectrum(ms[0], 4), atol=1e-8)


def test_balance_and_tradeoff():
    rng = np.random.default_rng(4)
    ms = make(rng, nl=6, n_per=25)
    nl = len(ms)
    for mode in ('none', 'own'):
        c = S.site_scales(ms, mode)
        t = S.balance_direct(ms, c)
        assert abs(t / S.balance(ms, c, iters=40, tol=1e-12) - 1) < 1e-5
        assert abs(S.modes(ms, c, t, k=1)[0][0] / S.temporal_eigs(nl, t)[1] - 1) < 1e-6
        # scalar control: a lower bound on the first eigenvalue at every coupling
        gaps = np.array([(1 - S.site_spectrum(m, 1)[0]) / ci for m, ci in zip(ms, c)])
        prev = None
        for f in (1 / 16, 1 / 4, 1, 4, 16):
            vals, g = S.modes(ms, c, t * f, k=6)
            assert S.scalar_control(gaps, t * f)[0] <= vals[0] + 1e-10
            a, b = S.tradeoff(ms, c, g)
            assert abs(a + t * f * b - vals.sum()) < 1e-8
            if prev:
                assert a >= prev[0] - 1e-8 and b <= prev[1] + 1e-8   # exact monotone trade-off
            prev = (a, b)
            assert abs(S.occupancy(g).sum() - 6) < 1e-9 and abs(S.local_strengths(g).sum() - 6) < 1e-9


def test_identical_sites_count_one_direction():
    """Control: nothing changes with depth. Harmonics of a cheap split fill the leading modes, and the
    rotation-invariant readings must say 'one image direction, everywhere'."""
    rng = np.random.default_rng(5)
    m = walk(clouds(rng, [np.array([0., 0.]), np.array([3., 0.])], 30))      # one very cheap split
    nl, n = 6, len(m)
    ms = np.stack([m] * nl)
    c = np.ones(nl)
    t = S.balance_direct(ms, c)
    vals, g = S.modes(ms, c, t, k=4)
    phi = np.linalg.eigh(m - 1 / n)[1][:, -1]
    live = S.depth_weight(g) > 1e-6          # a harmonic has nodes: a slice of zero weight has no direction
    assert not live.all()
    assert np.allclose(S.span_share(g, phi[:, None])[live], 1, atol=1e-6)    # all four modes: the same image function
    strength, direction = S.recurrent(g)
    assert strength[0] > 4 - 1e-6 and abs(abs(direction[:, 0] @ phi) - 1) < 1e-6
    assert np.allclose(S.local_strengths(g)[:, 1:], 0, atol=1e-8)            # one direction at every site
    # equivariance: one common relabelling of the images at every site permutes the modes' entries
    p = rng.permutation(n)
    ms2 = make(rng)
    c2 = S.site_scales(ms2, 'own')
    v1, g1 = S.modes(ms2, c2, .3, k=3)
    v2, g2 = S.modes(ms2[:, p][:, :, p], c2, .3, k=3)
    assert np.allclose(v1, v2, atol=1e-9) and np.allclose(np.abs(np.einsum('lnk,lnk->k', g1[:, p], g2)), 1, atol=1e-7)


def test_planted_merge_recovery():
    """Groups 0 and 1 are apart early and coincide late; group 2 is apart throughout."""
    rng = np.random.default_rng(4)
    ms = make(rng, nl=6, n_per=25)
    g3 = np.repeat(np.arange(3), 25)
    early = (g3 == 0) * 1. - (g3 == 1)
    persistent = (g3 == 2) * 1.
    c = S.site_scales(ms, 'own')
    t = S.balance_direct(ms, c)
    vals, g = S.modes(ms, c, t, k=12)
    # the lasting split is in the low space at every site from the first mode on
    assert min(S.captured(g, persistent, l)[0] for l in range(6)) > .08      # about 1/L of a unit vector per site
    assert sum(S.captured(g, persistent, l)[3] for l in range(6)) > .95 * min(4, 6)  # 4 modes, all this direction
    # the early split enters only at higher rank, at the early sites, and never at the late ones
    cap = np.stack([S.captured(g, early, l) for l in range(6)])              # (L, K)
    assert cap[0, 3] < .01 and cap[0, 7] > .5, cap[0]
    assert cap[-1].max() < .05, cap[-1]
    # correspondence control: relabelling images independently per site keeps every site's spectrum
    # but the low space can no longer hold one function across sites
    vs, gs = S.modes(S.permute_sites(ms, rng), c, t, k=12)
    assert vs[0] > 3 * vals[0], (vs[0], vals[0])


def test_set_readout_controls():
    """The post-hoc sets (spacetime.depth_sets): no invented change, and the planted merge at rank 8."""
    rng = np.random.default_rng(5)
    m = walk(clouds(rng, [np.array([0., 0.]), np.array([3., 0.])], 30))
    ms = np.stack([m] * 6)
    c = np.ones(6)
    g = S.modes(ms, c, S.balance_direct(ms, c), k=8)[1]
    for k in (2, 3):
        assert S.churn(S.depth_sets(g, k)).max() == 0                         # harmonics do not move anything
    truth = np.repeat([0, 1], 30)
    lab = S.depth_sets(g, 2)[0]
    assert (lab == truth).all() or (lab != truth).all()

    rng = np.random.default_rng(4)
    ms = make(rng, nl=6, n_per=25)
    g3 = np.repeat(np.arange(3), 25)
    c = S.site_scales(ms, 'own')
    lab = S.depth_sets(S.modes(ms, c, S.balance_direct(ms, c), k=8)[1], 3)
    differ = [float(np.mean(lab[l][g3 == 0][:, None] != lab[l][g3 == 1][None])) for l in range(6)]
    assert differ[0] > .7 and max(differ[3:]) == 0, differ                    # apart early, one set late
    assert all((lab[l][g3 == 2][:, None] != lab[l][g3 != 2][None]).all() for l in range(6))   # group 2 apart throughout
    # shared-coordinates rows do not depend on the basis of the space
    gg = S.modes(ms, c, .3, k=8)[1]
    q = np.linalg.qr(rng.standard_normal((8, 8)))[0]
    x1, x2 = S.shared_rows(gg)[0], S.shared_rows(gg @ q)[0]
    d = lambda x: ((x.reshape(-1, 1, 8)[::7] - x.reshape(1, -1, 8)[:, ::11]) ** 2).sum(-1)
    assert np.allclose(d(x1), d(x2), atol=1e-9)
    # known limit (SPEC_SPACETIME.md section 4): at rank 12-16 the late sites' noise directions are
    # rescaled to full size and the merged group is cut at random; the rank is part of the readout.


def test_stage_runs_and_is_consistent():
    import spacetime_stage as G
    rng = np.random.default_rng(6)
    ms = make(rng, nl=5, n_per=30)
    labels = np.repeat(np.arange(3), 30)
    old = (G.K, G.KC, G.KSETS, G.KTRADE)
    G.K, G.KC, G.KSETS, G.KTRADE = 8, 12, (2, 3), (4, 8)
    try:
        r = G.compute(ms, labels, controls=True, log=lambda *a: None)
    finally:
        G.K, G.KC, G.KSETS, G.KTRADE = old
    nl, n = 5, 90
    assert r['vecs'].shape == (2, 6, 8, nl, n) and r['vecs'].dtype == np.int16
    assert G.accept(r) == [], G.accept(r)
    assert np.allclose(r['weight'].sum(-1), 1, atol=1e-5)
    a, b = r['tradeoff'][..., 0], r['tradeoff'][..., 1]                      # (scale, coupling, K)
    assert (np.diff(a, axis=1) >= -1e-8).all() and (np.diff(b, axis=1) <= 1e-8).all()
    assert (r['scalar_energy'] <= r['vals'][..., 0] + 1e-10).all()
    for ki in range(2):
        for tag in 'as':
            assert np.nanmax(r[f'set_{tag}{ki}_info'] - r[f'set_{tag}{ki}_ceiling']) <= 1e-6
        assert (r[f'ref_site_set{ki}_info'] <= r[f'ref_site_set{ki}_ceiling'] + 1e-6).all()
        assert np.nanmax(r[f'ref_avg_set{ki}_churn']) == 0
    assert r['shuffle_vals'][0] > 2 * r['base_vals'][0]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
    print('all spacetime tests passed')
