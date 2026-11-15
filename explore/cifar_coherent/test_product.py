"""Checks for product (composition) blocks, product.py.

1. Single cell: product modes equal the cell's eigenmodes (sigma = lambda).
2. Gains multiply to the singular values; record path round-trips through SEBA.
3. Reversal transposes the product: same singular values, sides swap.
   A non-reversal permutation changes them (order sensitivity).
4. Planted, commuting spectra: a split that is very strong in all but one cell and
   destroyed in that one beats a uniformly moderate split under the average, and
   loses to it under the product (the product is multiplicative: one destroying
   cell is enough).
5. Planted point clouds: three groups separated at sites 0-1, B and C merged at
   site 2; the product's 2 SEBA sets are A and (B, C), and the B/C mode's singular
   value collapses when site 2 joins the block.
"""
import numpy as np
import torch

from coherent import block_modes, centred_cosine_gram, seba, seba_labels, symmetric_balance
from product import centred, chain, gains, product_modes, product_record, allocate_product, validate_product
from schema_v2 import comparison, pairs, put_record


def lazy_ops(xs):
    ms = []
    for x in xs:
        c = centred_cosine_gram(x)          # rule 0 of the final bank: union 15-NN graph
        d = 1 - c; d.diagonal().fill_(float('inf'))
        idx = d.topk(15, largest=False, dim=1).indices
        mask = torch.zeros_like(c, dtype=torch.bool); mask.scatter_(1, idx, True); mask = mask | mask.T
        w = torch.where(mask, torch.ones_like(c), torch.full_like(c, 1e-6)); w.diagonal().fill_(0)
        logk = torch.log(w)
        m = symmetric_balance(logk) * .5
        m.diagonal().add_(.5)
        ms.append(m)
    return ms


def clouds(n=80, d=30, seed=0, merge=False):
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(3, d, generator=g) * 3
    if merge:
        centres[2] = centres[1]
    lab = torch.arange(3).repeat_interleave(n)
    return centres[lab] + .6 * torch.randn(3 * n, d, generator=g), lab


def test_single_cell():
    x, _ = clouds(seed=3)
    m = lazy_ops([x])[0]
    lam, psi = block_modes([m], 7, shared_stationary=True)
    for side in 'RL':
        sv, v = product_modes(centred(m), 7, side)
        assert torch.allclose(sv, lam, atol=1e-10), (sv, lam)
        assert comparison(psi[:, 1:], v[:, 1:], 6) > 1 - 1e-8


def test_gains_and_record():
    xs = [clouds(seed=s)[0] for s in range(4)]
    ms = lazy_ops(xs); cs = [centred(m) for m in ms]
    n = len(ms[0])
    arrays = allocate_product(1, len(pairs(4)), n, 4)
    for pi, (a, b) in enumerate(pairs(4)):
        for side in 'RL':
            rec = product_record(ms, cs, a, b, side)
            if side == 'R':
                put_record(arrays, 0, pi, rec)
            q = chain(cs, a, b)
            sv = torch.linalg.svdvals(q)[:8].numpy()
            assert np.allclose(rec['vals'][1:], sv, atol=1e-6), (a, b, side)
            assert np.allclose(np.prod(rec['gain_psi'][:, a:b + 1], -1), rec['vals'][1:], atol=1e-6)
            assert np.isnan(np.delete(rec['gain_psi'], range(a, b + 1), axis=-1)).all()
    for key in arrays:
        arrays[key] = np.repeat(arrays[key], 5, axis=0)
    assert validate_product(arrays, 4, len(pairs(4)), n=n) < 1e-6


def test_order():
    # Unclustered, independent geometries: the cells do not nearly commute.
    xs = [torch.randn(240, 8, generator=torch.Generator().manual_seed(s), dtype=torch.float64) for s in (5, 6, 7)]
    ms = lazy_ops(xs); cs = [centred(m) for m in ms]
    fwd = chain(cs, 0, 2)
    rev = cs[0] @ cs[1] @ cs[2]
    assert torch.allclose(rev, fwd.T, atol=1e-12)
    sv_f, v_r = product_modes(fwd, 5, 'R')
    sv_b, u_l = product_modes(rev, 5, 'L')
    assert torch.allclose(sv_f, sv_b, atol=1e-10)
    assert comparison(v_r[:, 1:], u_l[:, 1:], 4) > 1 - 1e-8
    other = cs[1] @ cs[0] @ cs[2]          # not a reversal
    sv_o, _ = product_modes(other, 5, 'R')
    assert float((sv_o - sv_f).abs().max()) > 1e-3, (sv_o, sv_f)


def test_planted_spectra():
    torch.manual_seed(0)
    n, L = 200, 6
    basis = torch.linalg.qr(torch.cat([torch.ones(n, 1), torch.randn(n, 3, dtype=torch.float64)], 1)).Q
    basis[:, 0] = n ** -.5
    fa, fb = basis[:, 1], basis[:, 2]
    ms = []
    for l in range(L):
        lam_a = .895                               # moderate, every cell
        lam_b = .5 if l == 3 else .99              # very strong, destroyed in cell 3
        rest = .3
        m = rest * torch.eye(n, dtype=torch.float64) + (1 - rest) * torch.outer(basis[:, 0], basis[:, 0])
        m += (lam_a - rest) * torch.outer(fa, fa) + (lam_b - rest) * torch.outer(fb, fb)
        ms.append(m)
    _, psi = block_modes(ms, 3, shared_stationary=True)
    assert abs(float(psi[:, 1] @ fb)) > .999       # average prefers the split destroyed in one cell
    sv, v = product_modes(chain([centred(m) for m in ms], 0, L - 1), 3, 'R')
    assert abs(float(v[:, 1] @ fa)) > .999          # product prefers the persistent one
    assert abs(float(sv[1]) - .895 ** L) < 1e-10


def test_planted_clouds():
    x0, lab = clouds(seed=0); x1, _ = clouds(seed=1); x2, _ = clouds(seed=2, merge=True)
    ms = lazy_ops([x0, x1, x2]); cs = [centred(m) for m in ms]
    merged = (lab > 0).long()
    sv01, v01 = product_modes(chain(cs, 0, 1), 4, 'R')
    sv02, v02 = product_modes(chain(cs, 0, 2), 4, 'R')
    l3 = seba_labels(seba(v01, 3))
    ok = sum(torch.bincount(lab[l3 == s], minlength=3).max().item() for s in l3.unique() if s >= 0)
    assert ok / len(lab) > .95
    for side in 'RL':
        _, v = product_modes(chain(cs, 0, 2), 4, side)
        l2 = seba_labels(seba(v, 2))
        agree = max(float((l2 == merged).float().mean()), float((l2 == 1 - merged).float().mean()))
        assert agree > .95, (side, agree)
    assert float(sv02[2]) < .75 * float(sv01[2]), (sv01, sv02)
    g = gains(cs, 0, 2, v01[:, 2:3], 'R')[0]
    assert float(g[2]) < float(g[:2].min()), g     # the merging cell is where the B/C mode loses norm
    print('sigma [0,1]:', sv01.numpy().round(3), ' sigma [0,2]:', sv02.numpy().round(3), ' gains of v2:', g.numpy().round(3))


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_single_cell(); test_gains_and_record(); test_order(); test_planted_spectra(); test_planted_clouds()
    print('all product checks passed')
