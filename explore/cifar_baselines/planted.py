"""Planted-structure and null benchmarks for the coherent-sets study (PLAN.md section 8; IDEAS_gpt.md section 7).

Every generator is `make_<id>(seed, N=..., level=..., **params) -> (Xs, truth)`:

* `Xs` is a list of raw activation arrays X_l (N x d_l, float64), one per cell, run through the production
  preprocessing (column centring, row normalisation, Gram) by the caller. P11 is operator-level (graphs, not
  activations): `Xs` is None and `truth['adjacency']` holds the cells' 0/1 graphs.
* `truth` holds the planted information for **scoring only** (status V): it never feeds a kernel, basis, rank or
  selection. Keys: id, axis, level, kind, labels, contrasts, ranks, blocks, plus generator-specific extras and the
  generative description in `params`.

Latent coordinates Y_l (N x D) are embedded per cell by an independent random isometry Q_l (d_l x D, orthonormal
columns) into ambient width d_l (WIDTHS, cycled). An isometry changes no dot product, so this checks only that
nothing depends on the ambient width (asserted in test_planted.py). `oracle(Xs, truth)` measures how much of the
planted information is still available after centring + row normalisation, knowing the generative form: it projects
each preprocessed cell onto the planted latent subspace (known to the generator) and runs a 2-fold kNN classifier /
regressor there. Oracles are fits with labels (status V) and are reported, never used by a method.
"""
from __future__ import annotations

import math

import numpy as np

WIDTHS = [48, 64, 96, 128, 160, 192, 256, 320]

# ------------------------------------------------------------------------------------------------ helpers


def _rng(seed, salt):
    return np.random.default_rng([int(seed), int(salt)])


def isometry(rng, D, d):
    """d x D with orthonormal columns (Haar on the Stiefel manifold)."""
    q, r = np.linalg.qr(rng.standard_normal((d, D)))
    return q * np.sign(np.diag(r))[None, :]


def embed(Y, Q):
    return Y @ Q.T


def simplex(k):
    """k unit vectors in R^(k-1), pairwise inner product -1/(k-1)."""
    v = np.eye(k) - 1.0 / k
    _, _, wt = np.linalg.svd(v)
    c = v @ wt[: k - 1].T
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def balanced_labels(rng, N, k, masses=None):
    if masses is None:
        masses = np.full(k, 1.0 / k)
    counts = np.floor(np.asarray(masses) * N).astype(int)
    counts[: N - counts.sum()] += 1
    g = np.repeat(np.arange(k), counts)
    return rng.permutation(g)


def preprocess(X):
    """The production view: column-centre over the probe set, then row-normalise (coherent.centred_cosine_gram)."""
    Z = X - X.mean(0, keepdims=True)
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def _cells(Ys, rng, widths=None):
    """Embed latent cells by independent isometries; return Xs and the Q_l."""
    Xs, Qs = [], []
    for l, Y in enumerate(Ys):
        d = (widths or WIDTHS)[l % len(widths or WIDTHS)]
        d = max(d, Y.shape[1])
        Q = isometry(rng, Y.shape[1], d)
        Xs.append(embed(Y, Q))
        Qs.append(Q)
    return Xs, Qs


def _truth(id_, axis, level, kind, **kw):
    t = dict(id=id_, axis=axis, level=level, kind=kind, labels=None, contrasts=None, ranks=[2], blocks=None)
    t.update(kw)
    if t['labels'] is not None and 'n_groups' not in t:
        raise ValueError(f'{id_}: partition truth needs a declared n_groups')
    return t


# ------------------------------------------------------------------------------------------------ P0


def make_P0(seed, N=1000, level=0.0, d=64, L=4):
    """Isotropic Gaussian cells, G_l = sqrt(rho) G_0 + sqrt(1-rho) E_l; axis rho (persistence). No groups."""
    rho = float(level)
    rng = _rng(seed, 0)
    G0 = rng.standard_normal((N, d))
    Xs = [math.sqrt(rho) * G0 + math.sqrt(1 - rho) * rng.standard_normal((N, d)) for _ in range(L)]
    pseudo = balanced_labels(rng, N, 10)
    return Xs, _truth('P0', 'rho', rho, 'none', ranks=[2, 4], blocks={'all': (0, L - 1)}, pseudo_labels=pseudo,
                      params=dict(d=d, L=L, rho=rho, formula='X_l = sqrt(rho) G0 + sqrt(1-rho) E_l, entries N(0,1)'))


def make_P0d(seed, N=1000, level=64, rho=0.0, L=4):
    """P0 with axis = ambient dimension d at rho = 0."""
    Xs, t = make_P0(seed, N=N, level=rho, d=int(level), L=L)
    t.update(id='P0d', axis='d', level=int(level))
    return Xs, t


# ------------------------------------------------------------------------------------------------ P1


def _mixture(rng, g, means, a, s, D, spreads=None):
    N = len(g)
    Y = np.zeros((N, D))
    Y[:, : means.shape[1]] = a * means[g]
    sd = s if spreads is None else s * np.asarray(spreads)[g][:, None]
    return Y + sd * rng.standard_normal((N, D))


def make_P1(seed, N=1000, level=1.0, k=4, D=16, s=0.25, L=4):
    """Persistent separated mixture: Y_l = a mu_g + s eps_l (eps_l iid N(0, I_D), independent across cells),
    mu_g regular-simplex unit vectors in the first k-1 latent dims; axis a (separation)."""
    a = float(level)
    rng = _rng(seed, 1)
    g = balanced_labels(rng, N, k)
    mu = simplex(k)
    Ys = [_mixture(rng, g, mu, a, s, D) for _ in range(L)]
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P1', 'a', a, 'partition', labels=g, n_groups=k, ranks=[k], blocks={'all': (0, L - 1)},
                      planted=[Q[:, : k - 1] for Q in Qs], oracle_mode='classify',
                      params=dict(k=k, D=D, s=s, L=L, a=a, formula='Y_l = a mu_g + s eps_l; X_l = Y_l Q_l^T'))


# ------------------------------------------------------------------------------------------------ P2


def make_P2(seed, N=1000, level=1.0, h=3, D=16, L=4, decay=0.5):
    """Shared linear factors, no groups: Y_l = a A_l u + B_l v_l, u ~ N(0, I_h) shared, v_l ~ N(0, I_D) per cell,
    A_l D x h orthonormal columns (random per cell), B_l = O_l diag(j^-decay) (O_l random rotation); axis a."""
    a = float(level)
    rng = _rng(seed, 2)
    u = rng.standard_normal((N, h))
    Ys, As = [], []
    spec = np.arange(1, D + 1) ** (-decay)
    for _ in range(L):
        A = isometry(rng, h, D)
        O = isometry(rng, D, D)
        Ys.append(a * u @ A.T + (rng.standard_normal((N, D)) * spec) @ O.T)
        As.append(A)
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P2', 'a', a, 'span', contrasts=u, ranks=[h + 1], blocks={'all': (0, L - 1)},
                      planted=[Q @ A for Q, A in zip(Qs, As)], oracle_mode='linear',
                      params=dict(h=h, D=D, L=L, a=a, decay=decay,
                                  formula='Y_l = a A_l u + O_l diag(j^-decay) v_l; X_l = Y_l Q_l^T'))


# ------------------------------------------------------------------------------------------------ P3


def make_P3(seed, N=1000, level=2.0, L=6, a=1.0, D=8, s=0.5):
    """Weak persistent g in {-1,1} on e1, strong cell-specific h_l in {-1,1} on e2:
    Y_l = a g e1 + b h_l e2 + s eps_l; axis b/a."""
    b = float(level) * a
    rng = _rng(seed, 3)
    g = balanced_labels(rng, N, 2)
    hs = [balanced_labels(rng, N, 2) for _ in range(L)]
    Ys = []
    for hl in hs:
        Y = s * rng.standard_normal((N, D))
        Y[:, 0] += a * (2 * g - 1)
        Y[:, 1] += b * (2 * hl - 1)
        Ys.append(Y)
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P3', 'b/a', float(level), 'partition', labels=g, n_groups=2, ranks=[2], blocks={'all': (0, L - 1)},
                      cell_labels=hs, planted=[Q[:, :1] for Q in Qs], oracle_mode='classify',
                      nuisance_planted=[Q[:, 1:2] for Q in Qs],
                      params=dict(L=L, a=a, b=b, D=D, s=s, formula='Y_l = a g e1 + b h_l e2 + s eps_l'))


def make_P3L(seed, N=1000, level=4, ratio=2.0, **kw):
    """P3 with axis = number of cells L at b/a = ratio."""
    Xs, t = make_P3(seed, N=N, level=ratio, L=int(level), **kw)
    t.update(id='P3L', axis='L', level=int(level))
    return Xs, t


# ------------------------------------------------------------------------------------------------ P4

BOTTLENECK_POS = {0: [], 1: [3], 2: [2, 4], 3: [1, 3, 4]}


def make_P4(seed, N=1000, level=1, L=6, k=4, a=1.0, D=16, s=0.25, bottleneck='noise'):
    """P1 at a = 1 in L - n_b cells; n_b interior cells replaced by isotropic noise s * eps (bottleneck='noise')
    or by the same mixture with independent labels (bottleneck='indep_labels'); axis n_b."""
    nb = int(level)
    rng = _rng(seed, 4)
    g = balanced_labels(rng, N, k)
    mu = simplex(k)
    pos = BOTTLENECK_POS[nb]
    Ys = []
    for l in range(L):
        if l in pos:
            if bottleneck == 'noise':
                Ys.append(s * rng.standard_normal((N, D)))
            else:
                Ys.append(_mixture(rng, rng.permutation(g), mu, a, s, D))
        else:
            Ys.append(_mixture(rng, g, mu, a, s, D))
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P4', 'n_bottleneck', nb, 'partition', labels=g, n_groups=k, ranks=[k], blocks={'all': (0, L - 1)},
                      informative=[l not in pos for l in range(L)], planted=[Q[:, : k - 1] for Q in Qs],
                      oracle_mode='classify',
                      params=dict(L=L, k=k, a=a, D=D, s=s, bottleneck=bottleneck, positions=pos,
                                  formula='P1 cells, interior cells at positions replaced'))


# ------------------------------------------------------------------------------------------------ P5


def make_P5(seed, N=1000, level=0.15, D=8):
    """Two Hopf-linked great circles on S^3: group 0 at (cos t, sin t, 0, 0), group 1 at (0, 0, cos t, sin t), t
    uniform, same t and group in every cell; Y_l = circle + s eps_l in D latent dims; axis s. Both groups have mean
    zero, so no linear functional separates them (the indicator is quadratic). L = 4 cells."""
    s = float(level)
    L = 4
    rng = _rng(seed, 5)
    g = balanced_labels(rng, N, 2)
    t = rng.uniform(0, 2 * np.pi, N)
    base = np.zeros((N, D))
    base[g == 0, 0], base[g == 0, 1] = np.cos(t[g == 0]), np.sin(t[g == 0])
    base[g == 1, 2], base[g == 1, 3] = np.cos(t[g == 1]), np.sin(t[g == 1])
    Ys = [base + s * rng.standard_normal((N, D)) for _ in range(L)]
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P5', 's', s, 'partition', labels=g, n_groups=2, ranks=[2, 3], blocks={'all': (0, L - 1)},
                      planted=[Q[:, :4] for Q in Qs], oracle_mode='classify', angle=t,
                      params=dict(D=D, L=L, s=s, formula='linked great circles in R^4 + s eps_l'))


# ------------------------------------------------------------------------------------------------ P6


def circle_density_sample(rng, N, eta):
    out = []
    while len(out) < N:
        th = rng.uniform(0, 2 * np.pi, 2 * N)
        keep = rng.uniform(0, 1 + eta, 2 * N) < 1 + eta * np.cos(th)
        out.extend(th[keep].tolist())
    return np.array(out[:N])


def make_P6(seed, N=1000, level=0.6, c=0.5, D=4, s=0.05):
    """Density on a circle, no partition: theta ~ (1 + eta cos theta) / 2 pi; cell 0 = (cos theta, sin theta),
    cell 1 = (cos phi, sin phi) with phi = theta + c sin theta; Y_l = circle + s eps_l in D dims; axis eta."""
    eta = float(level)
    rng = _rng(seed, 6)
    th = circle_density_sample(rng, N, eta)
    ph = th + c * np.sin(th)
    Ys = []
    for ang in (th, ph):
        Y = s * rng.standard_normal((N, D))
        Y[:, 0] += np.cos(ang)
        Y[:, 1] += np.sin(ang)
        Ys.append(Y)
    Xs, Qs = _cells(Ys, rng)
    dens = (1 + eta * np.cos(th)) / (2 * np.pi)
    return Xs, _truth('P6', 'eta', eta, 'none', ranks=[2, 3, 5], blocks={'all': (0, 1), 'theta': (0, 0)},
                      contrasts=np.stack([np.cos(th), np.sin(th)], 1), density=dens, angle=th,
                      planted=[Q[:, :2] for Q in Qs], oracle_mode='angle_density', pseudo_labels=balanced_labels(rng, N, 10),
                      params=dict(eta=eta, c=c, D=D, s=s, formula='theta ~ (1+eta cos)/2pi; phi = theta + c sin theta'))


# ------------------------------------------------------------------------------------------------ P7

P7_LEVELS = ['none', '0.5%-persistent', '2%-persistent', '0.5%-cell', '2%-cell']


def make_P7(seed, N=1000, level='none', k=4, a=0.8, D=16, s=0.25, L=4, delta=1e-3,
            masses=(.60, .25, .10, .05), spreads=(1.0, .8, .6, .5)):
    """Unequal groups (masses), unequal spreads (s * spreads[g]); near-duplicate pairs x_j = x_i + delta xi in the
    latent space, persistent (same pairs in every cell) or cell-specific (new pairs per cell); axis contamination."""
    rng = _rng(seed, 7)
    g = balanced_labels(rng, N, k, masses)
    mu = simplex(k)
    Ys = [_mixture(rng, g, mu, a, s, D, spreads) for _ in range(L)]
    frac = {'none': 0.0, '0.5%': .005, '2%': .02}[level.split('-')[0]]
    persistent = level.endswith('persistent')
    m = int(round(frac * N))
    dup = []
    g_true = g.copy()
    for l in range(L):
        if m == 0:
            dup.append(np.zeros((0, 2), int))
            continue
        if persistent and l > 0:
            pairs = dup[0]
        else:
            idx = rng.choice(N, 2 * m, replace=False)
            pairs = idx.reshape(m, 2)
        Ys[l][pairs[:, 1]] = Ys[l][pairs[:, 0]] + delta * rng.standard_normal((m, D))
        dup.append(pairs)
    if persistent and m:
        g_true[dup[0][:, 1]] = g[dup[0][:, 0]]
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P7', 'contamination', level, 'partition', labels=g_true, n_groups=k, ranks=[k], blocks={'all': (0, L - 1)},
                      duplicates=dup, persistent=persistent, planted=[Q[:, : k - 1] for Q in Qs], oracle_mode='classify',
                      params=dict(k=k, a=a, D=D, s=s, L=L, delta=delta, masses=list(masses), spreads=list(spreads),
                                  frac=frac, persistent=persistent))


# ------------------------------------------------------------------------------------------------ P8

P8_LEVELS = [0.0, 0.25, 0.5, 1.0, 'independent']


def make_P8(seed, N=1000, level=0.5, D=2):
    """Circle doubling: x = (cos theta, sin theta), y = (cos phi, sin phi), phi = 2 theta + eta, eta wrapped
    N(0, sigma^2); level 'independent' draws phi uniform independently (same marginals). No ambient noise.
    Transfer singular values exp(-m^2 sigma^2 / 2), two per m."""
    rng = _rng(seed, 8)
    th = rng.uniform(0, 2 * np.pi, N)
    if level == 'independent':
        ph = rng.uniform(0, 2 * np.pi, N)
        sig = None
    else:
        sig = float(level)
        ph = np.mod(2 * th + sig * rng.standard_normal(N), 2 * np.pi)
    Ys = []
    for ang in (th, ph):
        Y = np.zeros((N, D))
        Y[:, 0], Y[:, 1] = np.cos(ang), np.sin(ang)
        Ys.append(Y)
    Xs, Qs = _cells(Ys, rng)
    sv = [] if sig is None else [math.exp(-(m ** 2) * sig ** 2 / 2) for m in (1, 2, 3)]
    return Xs, _truth('P8', 'sigma', level, 'span', ranks=[3, 5], blocks={'all': (0, 1)},
                      contrasts=np.stack([np.cos(2 * th), np.sin(2 * th)], 1),
                      spans={'theta1': np.stack([np.cos(th), np.sin(th)], 1),
                             'theta2': np.stack([np.cos(2 * th), np.sin(2 * th)], 1),
                             'phi1': np.stack([np.cos(ph), np.sin(ph)], 1)},
                      angle=th, angle2=ph, singular_values=sv, planted=[Q[:, :2] for Q in Qs],
                      oracle_mode='transfer', pseudo_labels=balanced_labels(rng, N, 10),
                      params=dict(sigma=level, D=D, formula='phi = 2 theta + N(0, sigma^2) mod 2pi'))


# ------------------------------------------------------------------------------------------------ P9


def make_P9(seed, N=1000, level=1.0, D=8, s=0.5, weights=(1.0, 1.0, 1.0)):
    """Competing partitions: u, v independent balanced +-1, w = u v; cell 0 encodes u, cell 1 v, cell 2 w:
    Y_l = a w_l c_l e1 + s eps_l; axis a. Truth: span of the three contrasts; no single partition."""
    a = float(level)
    rng = _rng(seed, 9)
    u = 2 * balanced_labels(rng, N, 2) - 1
    v = 2 * balanced_labels(rng, N, 2) - 1
    w = u * v
    Ys = []
    for c, wt in zip((u, v, w), weights):
        Y = s * rng.standard_normal((N, D))
        Y[:, 0] += a * wt * c
        Ys.append(Y)
    Xs, Qs = _cells(Ys, rng)
    joint = (u > 0).astype(int) * 2 + (v > 0).astype(int)
    return Xs, _truth('P9', 'a', a, 'span', contrasts=np.stack([u, v, w], 1).astype(float), ranks=[4, 2],
                      blocks={'all': (0, 2)}, labels=joint, n_groups=4, cell_labels=[(u > 0).astype(int), (v > 0).astype(int),
                                                                       (w > 0).astype(int)],
                      planted=[Q[:, :1] for Q in Qs], oracle_mode='classify_cells',
                      params=dict(D=D, s=s, a=a, weights=list(weights), formula='Y_l = a c_l e1 + s eps, c=(u,v,uv)'))


# ------------------------------------------------------------------------------------------------ P10


def make_P10(seed, N=1000, level=1.0, L=6, D=16, s=0.3):
    """Hierarchy sharpening with depth: top split t in {0,1} on e1 (amplitude 1), leaves b in {0,1} within each top
    on e2 (top 0) or e3 (top 1) with amplitude beta_l = beta_max * l / (L-1); Y_l = mean + s eps_l; axis beta_max."""
    bmax = float(level)
    rng = _rng(seed, 10)
    leaf = balanced_labels(rng, N, 4)
    top = leaf // 2
    sub = leaf % 2
    Ys = []
    for l in range(L):
        beta = bmax * l / (L - 1)
        Y = s * rng.standard_normal((N, D))
        Y[:, 0] += 2 * top - 1
        Y[top == 0, 1] += beta * (2 * sub[top == 0] - 1)
        Y[top == 1, 2] += beta * (2 * sub[top == 1] - 1)
        Ys.append(Y)
    Xs, Qs = _cells(Ys, rng)
    return Xs, _truth('P10', 'beta_max', bmax, 'hierarchy', labels=leaf, n_groups=4, top_labels=top, ranks=[2, 4],
                      blocks={'early': (0, 1), 'late': (L - 2, L - 1), 'all': (0, L - 1)},
                      planted=[Q[:, :3] for Q in Qs], oracle_mode='classify',
                      params=dict(L=L, D=D, s=s, beta_max=bmax, formula='top on e1; leaves on e2/e3 with beta_l'))


# ------------------------------------------------------------------------------------------------ P11


def make_P11(seed, N=1000, level=1, L=4, k=2, degree=20.0, ratio=4.0):
    """Operator-level multilayer SBM (graphs, not activations): k balanced communities, L cells; the first
    n_d cells disassortative (p_out = ratio p_in), the rest assortative (p_in = ratio p_out); expected degree
    `degree`; axis n_d. Xs is None; truth['adjacency'] holds symmetric 0/1 zero-diagonal graphs."""
    nd = int(level)
    rng = _rng(seed, 11)
    g = balanced_labels(rng, N, k)
    same = g[:, None] == g[None, :]
    n_in, n_out = N / k, N - N / k
    adj = []
    for l in range(L):
        dis = l < nd
        # p_in n_in + p_out n_out = degree with the stated ratio
        if dis:
            p_in = degree / (n_in + ratio * n_out)
            p_out = ratio * p_in
        else:
            p_out = degree / (ratio * n_in + n_out)
            p_in = ratio * p_out
        p = np.where(same, p_in, p_out)
        A = np.triu(rng.uniform(size=(N, N)) < p, 1)
        adj.append((A | A.T).astype(np.uint8))
    return None, _truth('P11', 'n_disassortative', nd, 'partition', labels=g, n_groups=k, ranks=[k], blocks={'all': (0, L - 1)},
                        adjacency=adj, disassortative=[l < nd for l in range(L)], oracle_mode='sbm',
                        params=dict(L=L, k=k, degree=degree, ratio=ratio, operator_level=True))


# ------------------------------------------------------------------------------------------------ P12


_P12_CACHE = {}


def _p12_net(seed, steps):
    """Train once per (seed, steps); reused across N within a process."""
    import torch
    import mlp_planted as mp
    if (seed, steps) not in _P12_CACHE:
        gE = torch.Generator().manual_seed(1000 + seed)
        E_A = torch.randn(8, mp.D_IN, generator=gE) / math.sqrt(mp.D_IN)
        E_B = torch.randn(8, mp.D_IN, generator=gE) / math.sqrt(mp.D_IN)
        if steps > 0:
            net, curve = mp.train(E_A, E_B, steps=steps, seed=seed)
        else:
            torch.manual_seed(seed)
            net, curve = mp.ResMLP(), []
        net.eval()
        _P12_CACHE[(seed, steps)] = (E_A, E_B, net, curve)
    return _P12_CACHE[(seed, steps)]


def make_P12(seed, N=1000, level=4000):
    """Trained toy network (explore/vit_coherent/mlp_planted.py, imported unmodified): residual MLP trained on
    y = (A + B) mod 8 for `level` steps (0 = untrained); cells = residual states h0..h4; truth y (and A, B)."""
    import os
    import sys
    import torch
    here = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vit_coherent'))
    if here not in sys.path:
        sys.path.insert(0, here)
    import mlp_planted as mp
    steps = int(level)
    E_A, E_B, net, curve = _p12_net(int(seed), steps)
    x, A, B, y = mp.sample(N, E_A, E_B, torch.Generator().manual_seed(2000 + int(seed)))
    with torch.no_grad():
        logits, hs = net(x, return_states=True)
    Xs = [h.double().numpy() for h in hs]
    acc = float((logits.argmax(1) == y).float().mean())
    return Xs, _truth('P12', 'train_steps', steps, 'partition', labels=y.numpy(), n_groups=8, ranks=[8],
                      blocks={'all': (0, len(Xs) - 1)}, extra_labels={'A': A.numpy(), 'B': B.numpy()},
                      oracle_mode='classify_full',
                      params=dict(steps=steps, probe_accuracy=acc, final_loss=curve[-1][1] if curve else None,
                                  source='explore/vit_coherent/mlp_planted.py (unmodified: ResMLP, train, sample)'))


# ------------------------------------------------------------------------------------------------ registry

BENCHMARKS = {
    'P0': (make_P0, [0.0, 0.5, 0.9, 1.0]),
    'P0d': (make_P0d, [16, 64, 256, 1024]),
    'P1': (make_P1, [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]),
    'P2': (make_P2, [0.0, 0.25, 0.5, 1.0, 2.0]),
    'P3': (make_P3, [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]),
    'P3L': (make_P3L, [1, 2, 4, 8]),
    'P4': (make_P4, [0, 1, 2, 3]),
    'P5': (make_P5, [0.05, 0.3, 0.5, 0.8, 1.2]),
    'P6': (make_P6, [0.0, 0.3, 0.6, 0.9]),
    'P7': (make_P7, P7_LEVELS),
    'P8': (make_P8, P8_LEVELS),
    'P9': (make_P9, [0.0, 0.25, 0.5, 1.0, 2.0]),
    'P10': (make_P10, [0.0, 0.25, 0.5, 1.0, 2.0]),
    'P11': (make_P11, [0, 1, 2, 3, 4]),
    'P12': (make_P12, [0, 250, 1000, 4000]),
}


# ------------------------------------------------------------------------------------------------ oracles


def _folds(N, seed=0):
    idx = np.random.default_rng(seed).permutation(N)
    return idx[: N // 2], idx[N // 2:]


def _knn_predict(Ftr, ytr, Fte, k=15, classify=True):
    d = (Fte ** 2).sum(1)[:, None] + (Ftr ** 2).sum(1)[None, :] - 2 * Fte @ Ftr.T
    nn = np.argpartition(d, min(k, len(Ftr) - 1), axis=1)[:, :k]
    if classify:
        votes = ytr[nn]
        K = int(ytr.max()) + 1
        counts = np.stack([(votes == c).sum(1) for c in range(K)], 1)
        return counts.argmax(1)
    return ytr[nn].mean(1)


def classify_score(F, y, seed=0):
    """2-fold kNN (k=15) classification; returns chance-normalised accuracy (acc - chance) / (1 - chance)
    with chance = largest class share, and the raw accuracy."""
    a, b = _folds(len(y), seed)
    pred = np.empty_like(y)
    pred[b] = _knn_predict(F[a], y[a], F[b])
    pred[a] = _knn_predict(F[b], y[b], F[a])
    acc = float((pred == y).mean())
    chance = float(np.bincount(y).max() / len(y))
    return dict(acc=acc, chance=chance, kappa=(acc - chance) / (1 - chance))


def regress_r2(F, T, mode='knn', seed=0, ridge=1e-3):
    """2-fold held-out R^2 (mean over target columns) of targets T from features F (kNN or ridge)."""
    T = np.asarray(T, float).reshape(len(T), -1)
    a, b = _folds(len(T), seed)
    pred = np.empty_like(T)
    for tr, te in ((a, b), (b, a)):
        if mode == 'knn':
            pred[te] = _knn_predict(F[tr], T[tr], F[te], classify=False)
        else:
            mu, mt = F[tr].mean(0), T[tr].mean(0)
            A = F[tr] - mu
            W = np.linalg.solve(A.T @ A + ridge * len(tr) * np.eye(A.shape[1]), A.T @ (T[tr] - mt))
            pred[te] = (F[te] - mu) @ W + mt
    ss = ((T - pred) ** 2).sum(0)
    st = ((T - T.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss / st))


def _proj(X, Q, raw=False):
    Z = X - X.mean(0, keepdims=True) if raw else preprocess(X)
    return Z @ Q


def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    hsic = np.linalg.norm(A.T @ B) ** 2
    return float(hsic / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def oracle(Xs, truth, seed=0):
    """Planted information available after centring + row normalisation (and, where informative, before
    normalisation: `raw_*`), per cell and pooled over the block's cells. Returns a flat dict of floats."""
    mode = truth.get('oracle_mode')
    out = {}
    if truth['id'] in ('P0', 'P0d'):
        y = truth['pseudo_labels']
        Zs = [preprocess(X) for X in Xs]
        out['pseudo_kappa_cell0'] = classify_score(Zs[0], y, seed)['kappa']
        out['cka_cells_01'] = linear_cka(Zs[0], Zs[1])
        return out
    if mode == 'sbm':
        g = truth['labels']
        k = int(g.max()) + 1
        vals = []
        for A, dis in zip(truth['adjacency'], truth['disassortative']):
            counts = np.stack([A[:, g == c].sum(1) for c in range(k)], 1).astype(float)
            pred = counts.argmin(1) if dis else counts.argmax(1)
            acc = float((pred == g).mean())
            chance = float(np.bincount(g).max() / len(g))
            vals.append((acc - chance) / (1 - chance))
        out.update({f'cell{l}_kappa': v for l, v in enumerate(vals)})
        out['min_cell_kappa'] = float(min(vals))
        return out
    if mode == 'classify_full':
        y = truth['labels']
        Fs = [preprocess(X) for X in Xs]
        vals = [classify_score(F, y, seed)['kappa'] for F in Fs]
        out.update({f'cell{l}_kappa': v for l, v in enumerate(vals)})
        out['pooled_kappa'] = classify_score(np.concatenate(Fs, 1), y, seed)['kappa']
        out['min_cell_kappa'] = float(min(vals))
        return out
    Q = truth['planted']
    Fs = [_proj(X, q) for X, q in zip(Xs, Q)]
    if mode == 'classify':
        y = truth['labels']
        vals = [classify_score(F, y, seed)['kappa'] for F in Fs]
        out.update({f'cell{l}_kappa': v for l, v in enumerate(vals)})
        out['pooled_kappa'] = classify_score(np.concatenate(Fs, 1), y, seed)['kappa']
        out['min_cell_kappa'] = float(min(vals))
        if truth['id'] in ('P3', 'P3L'):
            hv = [classify_score(_proj(X, q), h, seed)['kappa']
                  for X, q, h in zip(Xs, truth['nuisance_planted'], truth['cell_labels'])]
            out['nuisance_cell_kappa_mean'] = float(np.mean(hv))
        if truth['id'] == 'P10':
            out['top_pooled_kappa'] = classify_score(np.concatenate(Fs, 1), truth['top_labels'], seed)['kappa']
            out['leaf_first_cell_kappa'] = vals[0]
            out['leaf_last_cell_kappa'] = vals[-1]
        return out
    if mode == 'classify_cells':
        vals = [classify_score(F, y, seed)['kappa'] for F, y in zip(Fs, truth['cell_labels'])]
        out.update({f'cell{l}_own_kappa': v for l, v in enumerate(vals)})
        return out
    if mode == 'linear':
        u = truth['contrasts']
        vals = [regress_r2(F, u, 'linear', seed) for F in Fs]
        raw = [regress_r2(_proj(X, q, raw=True), u, 'linear', seed) for X, q in zip(Xs, Q)]
        out.update({f'cell{l}_r2': v for l, v in enumerate(vals)})
        out['mean_cell_r2'] = float(np.mean(vals))
        out['raw_mean_cell_r2'] = float(np.mean(raw))
        out['pooled_r2'] = regress_r2(np.concatenate(Fs, 1), u, 'linear', seed)
        return out
    if mode == 'angle_density':
        th = truth['angle']
        eta = truth['params']['eta']
        for name, F, Fr in (('cell0', Fs[0], _proj(Xs[0], Q[0], raw=True)),):
            out[f'{name}_r2_first_harmonic'] = regress_r2(F, truth['contrasts'], 'knn', seed)
            ang = np.arctan2(F[:, 1], F[:, 0])
            angr = np.arctan2(Fr[:, 1], Fr[:, 0])
            # rotation of the latent plane by Q is undone by matching the phase of the sample's first moment
            out[f'{name}_eta_hat'] = float(2 * np.abs(np.exp(1j * ang).mean()))
            out[f'{name}_raw_eta_hat'] = float(2 * np.abs(np.exp(1j * angr).mean()))
        out['eta'] = eta
        return out
    if mode == 'transfer':
        ax = np.arctan2(Fs[0][:, 1], Fs[0][:, 0])
        ay = np.arctan2(Fs[1][:, 1], Fs[1][:, 0])
        # Q_l is an arbitrary isometry: recovered angles are known up to a reflection and offset; the oracle
        # takes the best of the two orientations and the phase-free harmonic correlation |E exp(i(2ax - ay))|.
        for m in (1, 2):
            c = max(abs(np.exp(1j * (2 * m * ax - m * ay)).mean()), abs(np.exp(1j * (2 * m * ax + m * ay)).mean()))
            out[f'harmonic{m}_corr'] = float(c)
        sv = truth['singular_values']
        for m, v in enumerate(sv[:2], 1):
            out[f'harmonic{m}_theory'] = v
        out['odd_source_corr'] = float(max(abs(np.exp(1j * (ax - ay)).mean()), abs(np.exp(1j * (ax + ay)).mean())))
        return out
    raise ValueError(mode)
