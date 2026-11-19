"""Markov-closure measures in coarse.py (SPEC_COARSE.md section 6). uv run --with torch --with numpy --with scipy python test_markov.py"""
import itertools
import math

import torch

from coarse import closure_step, closure_triple, coarse_transition, ck_error, onehot, uniform_lazy, nmi
from test_coarse import random_cell

TOL = 1e-10


def brute_joint(ms, labs):
    """Exact p(c_0, ..., c_L) and p(z_0, c_1, ...) by enumeration on a tiny chain (Z_0 uniform)."""
    n = len(ms[0])
    p = {}
    for path in itertools.product(range(n), repeat=len(ms) + 1):
        w = 1.0 / n
        for l, m in enumerate(ms):
            w *= float(m[path[l + 1], path[l]])
        key = tuple(int(labs[l][z]) for l, z in enumerate(path))
        p[key] = p.get(key, 0) + w
        fkey = (path[0],) + key[1:]
        p[('f',) + fkey] = p.get(('f',) + fkey, 0) + w
    return p


def mi(p2):
    pa, pb = {}, {}
    for (a, b), v in p2.items():
        pa[a] = pa.get(a, 0) + v; pb[b] = pb.get(b, 0) + v
    return sum(v * math.log(v / (pa[a] * pb[b])) for (a, b), v in p2.items() if v > 0)


def cmi(p3):
    pb, pab, pbc = {}, {}, {}
    for (a, b, c), v in p3.items():
        pb[b] = pb.get(b, 0) + v; pab[a, b] = pab.get((a, b), 0) + v; pbc[b, c] = pbc.get((b, c), 0) + v
    return sum(v * math.log(v * pb[b] / (pab[a, b] * pbc[b, c])) for (a, b, c), v in p3.items() if v > 0)


def test_brute_force():
    n = 6
    ms = [random_cell(n=n, groups=2, seed=s) for s in (1, 2)]
    labs = [torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([0, 0, 0, 1, 1, 1])]
    p = brute_joint(ms, labs)
    coarse3 = {k: v for k, v in p.items() if k[0] != 'f'}
    fine3 = {k[1:]: v for k, v in p.items() if k[0] == 'f'}
    # one-step defect = I(Z0;C1) - I(C0;C1)
    zc1, c0c1 = {}, {}
    for (z, c1, c2), v in fine3.items():
        zc1[z, c1] = zc1.get((z, c1), 0) + v
    for (c0, c1, c2), v in coarse3.items():
        c0c1[c0, c1] = c0c1.get((c0, c1), 0) + v
    st = closure_step(ms[0], labs[0], labs[1])
    assert abs(st['I_fine'] - mi(zc1)) < 1e-9 and abs(st['I_coarse'] - mi(c0c1)) < 1e-9
    tr = closure_triple(ms[0], ms[1], *labs)
    assert abs(tr['cmi_coarse'] - cmi(coarse3)) < 1e-9, (tr, cmi(coarse3))
    assert abs(tr['cmi_fine'] - cmi(fine3)) < 1e-9
    assert tr['cmi_fine'] >= tr['cmi_coarse'] - TOL


def lumpable(n=12, k=3, eps=.2, seed=0):
    """Block-constant transitions: P(y | x) depends on x only through its block -> exactly lumpable."""
    g = torch.Generator().manual_seed(seed)
    lab = torch.arange(n) % k
    t = torch.rand(k, k, generator=g, dtype=torch.float64) + torch.eye(k, dtype=torch.float64) * 3
    t = t + t.T
    # symmetric doubly stochastic with block structure: m[y,x] = T[c(y), c(x)] / size, then Sinkhorn on k x k
    for _ in range(2000):
        t = t / t.sum(0, keepdim=True); t = (t + t.T) / 2
    m = t[lab][:, lab] / (n // k)
    return m, lab


def test_lumpable_zero():
    m, lab = lumpable()
    assert float((m.sum(0) - 1).abs().max()) < 1e-9
    st = closure_step(m, lab, lab)
    assert abs(st['defect']) < TOL and abs(st['defect_chi2']) < TOL
    tr = closure_triple(m, m, lab, lab, lab)
    assert abs(tr['cmi_coarse']) < TOL and abs(tr['cmi_fine']) < TOL
    t1, pa = coarse_transition(m, lab, lab)
    t2, _ = coarse_transition(m @ m, lab, lab)
    err, ind = ck_error(t2, t1 @ t1, pa)
    assert err < TOL and ind > 1e-3


def test_nonlumpable_positive_and_singletons():
    m = random_cell(n=30, groups=3, seed=4)
    lab = torch.arange(30) % 3                      # not aligned with the planted groups
    st = closure_step(m, lab, lab); tr = closure_triple(m, m, lab, lab, lab)
    assert st['defect'] > 1e-4 and tr['cmi_coarse'] > 1e-6
    t1, pa = coarse_transition(m, lab, lab); t2, _ = coarse_transition(m @ m, lab, lab)
    assert ck_error(t2, t1 @ t1, pa)[0] > 1e-6
    sing = torch.arange(30)
    st = closure_step(m, sing, sing); tr = closure_triple(m, m, sing, sing, sing)
    assert abs(st['defect']) < TOL and abs(tr['cmi_coarse']) < 1e-9
    t1, pa = coarse_transition(m, sing, sing); t2, _ = coarse_transition(m @ m, sing, sing)
    assert ck_error(t2, t1 @ t1, pa)[0] < 1e-9


def test_reviewer_counterexamples():
    """Lazy uniform walk, 8 equal groups = 3 independent binary attributes A, B, D read at successive sites."""
    n = 48
    u = uniform_lazy(n)
    g = torch.arange(n) % 8
    A, B, D = g % 2, (g // 2) % 2, (g // 4) % 2
    tr = closure_triple(u, u, A, B, D)
    assert abs(tr['cmi_coarse']) < 1e-10 and tr['cmi_fine'] > 1e-3, tr      # coarse Markov, fine memory > 0
    tr = closure_triple(u, u, A, B, A)
    assert tr['cmi_coarse'] > 1e-3 and abs(tr['I_next']) < 1e-10           # memory with zero adjacent information
    assert tr['H_cond'] > 0
    t1, pa = coarse_transition(u, A, B); t2, _ = coarse_transition(u, B, A); tt, _ = coarse_transition(u @ u, A, A)
    err, _ = ck_error(tt, t2 @ t1, pa)
    assert err > .1, err                                                    # about .125 for large N
    shared = closure_triple(u, u, A, A, A)
    assert abs(shared['cmi_coarse']) < 1e-10                                # a shared partition is closed under U
    assert abs(nmi(A, A) - 1) < 1e-12 and abs(nmi(A, B)) < 1e-12


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
    print('all markov tests passed')
