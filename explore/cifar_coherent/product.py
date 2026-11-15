"""Product (composition) blocks: coherent pairs of the ordered product of cell operators.

For a block of cells a..b (sites at one checkpoint, or checkpoints at one site)
the average block uses Mbar = mean_l M_l. The product block instead uses the
chronological composition

    Q = C_b ... C_{a+1} C_a,   C_l = M_l - J/N   (J = all-ones),

column-vector convention: cell a acts first. Every M_l is symmetric and doubly
stochastic, so the constant is a shared eigenvector with eigenvalue 1 and
C_l = M_l (I - J/N); the centred product equals the product with the constant
projected out on both sides. Its singular value decomposition Q = U S V^T gives
Froyland-style coherent pairs (v_k, u_k): v_k is the function on the probe
images that keeps the largest norm after being diffused by every cell of the
block *in order*; u_k = Q v_k / s_k is what it becomes after the last cell.

  side 'R' (start of block): vectors are v_k, eigenvectors of Q^T Q.
  side 'L' (end of block):   vectors are u_k, eigenvectors of Q Q^T.

Both sides share the singular values s_k <= 1. The trivial pair (the constant,
singular value 1 of the uncentred product) is restored as mode 0 so that the
downstream SEBA/export path is exactly the one used for average blocks.

Per-cell gains make the sequential reading explicit. For a right vector v,
w_a = C_a v, w_l = C_l w_{l-1}; gain_l = |w_l| / |w_{l-1}|. For a left vector
u the adjoint chain runs backwards from b to a. Within the block the gains
multiply to s_k exactly; outside the block they are NaN. Each M_l is symmetric,
so reversing the order of the cells transposes Q: singular values are unchanged
and the two sides swap. Products are therefore order-sensitive only up to
reversal; averages are not order-sensitive at all.

Eigenvectors are computed with float64 eigh of the centred Q^T Q (or Q Q^T),
whose null space contains the constant, so the nontrivial top modes are found
without deflation. Left vectors for side 'R' are not stored.
"""
from __future__ import annotations

import numpy as np
import torch

from schema_v2 import RANKS, allocate, block_record

SIDES = ('R', 'L')


def centred(m: torch.Tensor) -> torch.Tensor:
    return m - m.mean(0, keepdim=True) - m.mean(1, keepdim=True) + m.mean()


def chain(cs: list[torch.Tensor], a: int, b: int) -> torch.Tensor:
    q = cs[a]
    for l in range(a + 1, b + 1):
        q = cs[l] @ q
    return q


def sign_fix(v: torch.Tensor) -> torch.Tensor:
    idx = v.abs().argmax(0)
    s = torch.sign(v[idx, torch.arange(v.shape[1], device=v.device)])
    s[s == 0] = 1
    return v * s


def product_modes(q: torch.Tensor, n_modes: int, side: str):
    """Trivial mode + top n_modes-1 nontrivial singular pairs of a centred product q."""
    n = len(q)
    g = q.T @ q if side == 'R' else q @ q.T
    g = (g + g.T) / 2
    lam, w = torch.linalg.eigh(g)
    lam, w = lam.flip(0)[:n_modes - 1], w.flip(1)[:, :n_modes - 1]
    sv = lam.clamp_min(0).sqrt()
    # Exactly orthogonal to the constant (it is in the null space up to rounding).
    w = w - w.mean(0, keepdim=True)
    w = sign_fix(w / w.norm(dim=0, keepdim=True))
    const = torch.full((n, 1), n ** -.5, dtype=q.dtype, device=q.device)
    vals = torch.cat([torch.ones(1, dtype=q.dtype, device=q.device), sv])
    return vals, torch.cat([const, w], 1)


def gains(cs: list[torch.Tensor], a: int, b: int, f: torch.Tensor, side: str) -> torch.Tensor:
    """(k, n_cells) per-cell norm ratios of f pushed through the block; NaN outside."""
    f = f - f.mean(0, keepdim=True)
    out = torch.full((f.shape[1], len(cs)), float('nan'), dtype=f.dtype, device=f.device)
    order = range(a, b + 1) if side == 'R' else range(b, a - 1, -1)
    w = f
    for l in order:
        nxt = cs[l] @ w
        out[:, l] = nxt.norm(dim=0) / w.norm(dim=0).clamp_min(1e-300)
        w = nxt
    return out


def allocate_product(nm, npair, n, nc):
    arrays = allocate(nm, npair, n, nc)
    arrays['gain_psi'] = np.full((nm, npair, 8, nc), np.nan, np.float32)
    arrays['gain_seba'] = np.full((nm, npair, 5, 8, nc), np.nan, np.float32)
    return arrays


def product_record(ms, cs, a, b, side, q=None):
    """Same record as schema_v2.block_record, from the product's singular vectors, plus gains."""
    if q is None:
        q = chain(cs, a, b)
    vals, vecs = product_modes(q, 9, side)
    rec, vectors = block_record(ms, a, b, eigen=(vals, vecs))
    v = vectors.to(cs[0].device)
    rec['gain_psi'] = gains(cs, a, b, v[:, 1:], side).cpu().numpy().astype('f4')
    rec['gain_seba'] = np.full((5, 8, len(cs)), np.nan, np.float32)
    for ri, (r, s) in enumerate(zip(RANKS, rec.pop('_seba'))):
        rec['gain_seba'][ri, :r] = gains(cs, a, b, s.to(v.device), side).cpu().numpy()
    return rec


def validate_product(arrays, nc, npair, n=2000, tol=2e-4):
    """Shape/dtype/padding checks and the exact product-of-gains identity."""
    expected = allocate_product(5, npair, n, nc)
    assert set(arrays.files if hasattr(arrays, 'files') else arrays) == set(expected)
    for key, t in expected.items():
        a = arrays[key]
        assert a.shape == t.shape and a.dtype == t.dtype, (key, a.shape, a.dtype)
    vals = arrays['vals']
    assert np.max(np.abs(vals[..., 0] - 1)) < 1e-6
    assert np.all(np.diff(vals, axis=-1) <= 1e-6) and np.all(vals >= 0)
    g = arrays['gain_psi'].astype('f8')
    from schema_v2 import pairs
    worst = 0.
    for pi, (a, b) in enumerate(pairs(nc)):
        inside = g[:, pi, :, a:b + 1]
        assert np.isfinite(inside).all() and np.isnan(np.delete(g[:, pi], range(a, b + 1), axis=-1)).all()
        prod = inside.prod(-1)
        worst = max(worst, float(np.max(np.abs(prod - vals[:, pi, 1:]))))
    assert worst < tol, worst
    return worst
