"""Colour sources for the coherent-sets explorer (PLAN section 2).

Contract (builders B and C code against it):
    SOURCES[name](grams, r, *, seed=0, ops=None, cache=None, cell_keys=None) -> dict
      grams  list of the block's cells' centred-cosine Grams C (N x N, float64; torch or numpy)
      r      SEBA rank (counts the constant); q = r - 1 non-constant directions; k_c = r sets
      ops    the cells' default operators M (torch or numpy), else built with default_ops
      cache  optional dict shared across calls (ranks, blocks) to reuse eigensolves;
             cell_keys (hashable per cell) let single-cell results be shared across blocks
    result keys: labels (N,) int64 (-1 = unassigned); basis (N, q) orthonormal,
      constant-free span or None; values (eigen/singular values or None); status
      ('S', 'P', 'B', 'V' or '+'-joined); info (dict).
    default_ops(grams) -> list of M (torch float64) with the default rule 0.

Status tags (PLAN section 0): S label-free dot-product spectral object; P post-processing
or null generator; B fitted baseline outside the construction (k-means is a fit and is
called one); V validation with labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CC = Path(__file__).resolve().parent.parent / 'cifar_coherent'
if _CC.is_dir() and str(_CC) not in sys.path:
    sys.path.insert(0, str(_CC))

from coherent import block_modes, symmetric_balance  # noqa: E402
from seba import seba, seba_labels  # noqa: E402

N_VECS = 8            # psi_1..psi_8 as stored by the pipeline
RANKS = [2, 3, 4, 6, 8]
KMEANS_STARTS = 20


# ---------------------------------------------------------------- default construction

def _t(x):
    if isinstance(x, torch.Tensor):
        return x.double()
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def _n(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def knn_mask(c, k=15, mutual=False):
    """Boolean kNN adjacency on cosine distance 1 - C (diagonal excluded), union or mutual.
    Same top-k call as sweep_math.final_kernel_bank rule 0."""
    c = _t(c)
    d = 1 - c
    d.diagonal().fill_(float('inf'))
    idx = d.topk(k, largest=False, dim=1).indices
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return (mask & mask.T) if mutual else (mask | mask.T)


def knn_logkernel(c, k=15, floor=1e-6, mutual=False):
    """Rule 0 (default): K_ij = 1 on union-kNN edges, `floor` elsewhere, zero diagonal.
    With k=15, floor=1e-6 this is bit-identical to final_kernel_bank(c)[0][0]."""
    c = _t(c)
    mask = knn_mask(c, k, mutual)
    if floor > 0:
        w = torch.where(mask, torch.ones_like(c), torch.full_like(c, floor))
    else:
        w = mask.double()
    w.diagonal().fill_(0)
    return torch.log(w)


def lazy(p, gamma=.5):
    m = p * (1 - gamma)
    m.diagonal().add_(gamma)
    return m


def default_operator(c, k=15, floor=1e-6):
    """Zero-diagonal symmetric Sinkhorn P of the union-kNN kernel, M = (I + P)/2."""
    p = symmetric_balance(knn_logkernel(c, k, floor))
    return lazy(p, .5)


def default_ops(grams):
    return [default_operator(c) for c in grams]


# ---------------------------------------------------------------- eigen-solvers

def sign_fix(v):
    """Deterministic sign: largest-magnitude entry of each column positive (block_modes)."""
    idx = np.abs(v).argmax(0)
    s = np.sign(v[idx, np.arange(v.shape[1])])
    s[s == 0] = 1
    return v * s


def top_eigs(A, k, project_constant=True, seed=0, dense=None, tol=1e-11, res_tol=1e-8,
             gap_min=1e-6):
    """Top-k eigenpairs (descending) of symmetric A, restricted to 1-perp when
    project_constant (the constant then never appears among them).

    Iterative Lanczos (scipy eigsh on a projected LinearOperator) with a residual check
    ||A u - lambda u|| <= res_tol and a check that the k-th eigenvalue is separated from
    the (k+1)-th by gap_min; otherwise (or for N <= 800, or dense=True) a dense fallback
    (Householder deflation of the constant as in coherent.block_modes, then eigh)."""
    A = _n(A) if not hasattr(A, 'tocsr') else A
    n = A.shape[0]
    info = {}
    if dense is None:
        dense = n <= 800
    if not dense:
        from scipy.sparse.linalg import LinearOperator, eigsh
        def mv(x):
            x = np.asarray(x, dtype=np.float64)
            x1 = x.ndim == 1
            x = x[:, None] if x1 else x
            if project_constant:
                x = x - x.mean(0, keepdims=True)
            y = A @ x
            if project_constant:
                y = y - y.mean(0, keepdims=True)
            return y[:, 0] if x1 else y
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(n)
        if project_constant:
            v0 -= v0.mean()
        op = LinearOperator((n, n), matvec=mv, matmat=mv, dtype=np.float64)
        try:
            vals, vecs = eigsh(op, k=k + 1, which='LA', v0=v0, tol=tol,
                               ncv=min(n - 1, max(4 * (k + 1), 40)), maxiter=20000)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            res = np.linalg.norm(mv(vecs[:, :k]) - vecs[:, :k] * vals[:k], axis=0)
            gap = float(vals[k - 1] - vals[k])
            info = dict(method='eigsh', residual=float(res.max()), gap_k=gap)
            if res.max() <= res_tol and gap > gap_min:
                return vals[:k], sign_fix(vecs[:, :k]), info
            info['fallback'] = 'residual or gap'
        except Exception as e:  # ArpackNoConvergence etc.
            info = dict(method='eigsh', fallback=type(e).__name__)
    # dense fallback
    A = A.toarray() if hasattr(A, 'toarray') else np.asarray(A, dtype=np.float64)
    if project_constant:
        h = np.ones(n) / np.sqrt(n); h[0] -= 1; h /= np.linalg.norm(h)
        ah = A @ h
        T = A - 2 * np.outer(h, ah) - 2 * np.outer(ah, h) + 4 * (h @ ah) * np.outer(h, h)
        lam, w = np.linalg.eigh(T[1:, 1:])
        lam, w = lam[::-1][:k], w[:, ::-1][:, :k]
        lifted = np.vstack([np.zeros((1, k)), w])
        lifted -= 2 * np.outer(h, h @ lifted)
        vals, vecs = lam, lifted
    else:
        lam, w = np.linalg.eigh(A)
        vals, vecs = lam[::-1][:k], w[:, ::-1][:, :k]
    Av = A @ vecs
    if project_constant:
        Av = Av - Av.mean(0, keepdims=True)
    info = dict(info, method=('eigsh->dense' if 'method' in info else 'dense'),
                residual=float(np.linalg.norm(Av - vecs * vals, axis=0).max()))
    return vals, sign_fix(vecs), info


def mbar_modes(ops, n_modes=9):
    """The method's block eigenpairs exactly as schema_v2.block_record computes them:
    coherent.block_modes(shared_stationary=True), constant column then fixed exactly."""
    ms = [_t(m) for m in ops]
    vals, vecs = block_modes(ms, n_modes, shared_stationary=True)
    assert abs(float(vals[0]) - 1) < 1e-6
    assert float((vecs[:, 0] - 1 / len(vecs) ** .5).abs().max()) < 1e-5
    vecs[:, 0] = 1 / len(vecs) ** .5
    return vals.cpu().numpy(), vecs.cpu().double().numpy()


def with_constant(U):
    n = U.shape[0]
    return np.hstack([np.full((n, 1), 1 / np.sqrt(n)), U])


def seba_sets(V, r):
    """Unchanged SEBA (coherent/seba.py) on the first r columns of V = [const, ...].
    Returns labels (int64), the transform, perm, and a diagnostics dict."""
    try:
        s, rot, perm = seba(torch.as_tensor(np.ascontiguousarray(V[:, :r])), r, return_transform=True)
    except ValueError as e:
        return -np.ones(len(V), np.int64), None, None, dict(seba_error=str(e))
    lab = seba_labels(s).numpy().astype(np.int64)
    sizes = np.bincount(lab[lab >= 0], minlength=r).tolist()
    return lab, rot.numpy(), perm.numpy(), dict(sizes=sizes, unassigned=float((lab < 0).mean()))


def kmeans(X, k, seed=0, n_init=KMEANS_STARTS):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(np.ascontiguousarray(X))
    return km.labels_.astype(np.int64), km


def spherical_kmeans_gram(C, k, seed=0, n_init=KMEANS_STARTS, max_iter=300):
    """Exact spherical k-means in the Gram (IDEAS B2): point i scores set j by
    (C h_j)_i / sqrt(h_j^T C h_j) (its cosine to the unit-normalised set sum); the
    objective sum_j sqrt(h_j^T C h_j) is maximised; k-means++ seeding on chordal
    distance^2 = 2 - 2 C_ij (rows of a block Gram are unit length: diag(C_B) = 1).
    An empty set is re-seeded with the worst-fitting point. Best of n_init starts."""
    C = np.asarray(C, dtype=np.float64)
    n = len(C)
    rng = np.random.default_rng(seed)
    best = None
    diag = np.diag(C)
    for start in range(n_init):
        # k-means++ seeding
        centres = [int(rng.integers(n))]
        d2 = np.clip(diag + diag[centres[0]] - 2 * C[centres[0]], 0, None)
        for _ in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            c = int(rng.choice(n, p=p))
            centres.append(c)
            d2 = np.minimum(d2, np.clip(diag + diag[c] - 2 * C[c], 0, None))
        lab = np.argmax(C[:, centres], axis=1)
        for it in range(max_iter):
            H = np.zeros((n, k)); H[np.arange(n), lab] = 1
            CH = C @ H
            norms = np.sqrt(np.clip((H * CH).sum(0), 1e-300, None))
            score = CH / norms
            empty = H.sum(0) == 0
            score[:, empty] = -np.inf
            new = np.argmax(score, axis=1)
            for j in np.where(np.bincount(new, minlength=k) == 0)[0]:
                worst = int(np.argmin(score[np.arange(n), new]))
                new[worst] = j
            if np.array_equal(new, lab):
                break
            lab = new
        H = np.zeros((n, k)); H[np.arange(n), lab] = 1
        obj = float(np.sqrt(np.clip((H * (C @ H)).sum(0), 0, None)).sum())
        if best is None or obj > best[0]:
            best = (obj, lab.copy(), it + 1)
    return best[1].astype(np.int64), dict(objective=best[0], iterations=best[2], starts=n_init)


# ---------------------------------------------------------------- block context (cache)

class Block:
    """Lazily computed per-block objects shared by all sources and ranks."""

    def __init__(self, grams, ops=None, cache=None, cell_keys=None):
        self.grams = grams
        self._ops = ops
        self.cache = cache if cache is not None else {}
        self.keys = list(cell_keys) if cell_keys is not None else None
        self.L = len(grams)
        self.n = grams[0].shape[0]
        self.local = {}

    def _global(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def _mine(self, key, fn):
        if key not in self.local:
            self.local[key] = fn()
        return self.local[key]

    def ckey(self, i, what):
        return ('cell', self.keys[i], what) if self.keys is not None else None

    def cell(self, i, what, fn):
        k = self.ckey(i, what)
        if k is None:
            return self._mine(('cell', i, what), fn)
        return self._global(k, fn)

    @property
    def ops(self):
        if self._ops is None:
            self._ops = [self.cell(i, 'M', lambda i=i: default_operator(self.grams[i]))
                         for i in range(self.L)]
        return self._ops

    def mbar(self):
        return self._mine('mbar', lambda: mbar_modes(self.ops, N_VECS + 1))

    def cell_modes(self, i):
        return self.cell(i, 'modes', lambda: mbar_modes([self.ops[i]], N_VECS + 1))

    def cell_seba(self, i, r):
        return self.cell(i, ('seba', r), lambda: seba_sets(self.cell_modes(i)[1], r)[0])

    def CB(self):
        return self._mine('CB', lambda: np.mean([_n(c).astype(np.float64) for c in self.grams], 0))

    def pca(self, k=20):
        def f():
            CB = self.CB()
            G = CB - CB.mean(0, keepdims=True) - CB.mean(1, keepdims=True) + CB.mean()
            vals, vecs, info = top_eigs(G, k, project_constant=True)
            return vals, vecs, info
        return self._mine('pca', f)

    def g2(self):
        def f():
            import scipy.sparse as sp
            S = None
            for i in range(self.L):
                def cellS(i=i):
                    A = knn_mask(self.grams[i]).numpy().astype(np.float64)
                    np.fill_diagonal(A, 0)
                    d = A.sum(1)
                    Ad = sp.csr_matrix(A)
                    Dm = sp.diags(1 / np.sqrt(d))
                    return (Dm @ Ad @ Dm).tocsr()
                Si = self.cell(i, 'textbookS', cellS)
                S = Si if S is None else S + Si
            S = S / self.L
            vals, vecs, info = top_eigs(S, N_VECS, project_constant=False)
            return vals, vecs, info
        return self._mine('g2', f)


def _block(grams, ops, cache, cell_keys):
    if cache is not None and cell_keys is not None:
        key = ('block', tuple(cell_keys))
        if key not in cache:
            cache[key] = Block(grams, ops, cache, cell_keys)
        b = cache[key]
        if ops is not None and b._ops is None:
            b._ops = ops
        return b
    return Block(grams, ops, cache, cell_keys)


def _result(labels, basis, values, status, **info):
    return dict(labels=np.asarray(labels, np.int64), basis=basis,
                values=None if values is None else np.asarray(values), status=status, info=info)


# ---------------------------------------------------------------- the sources

def src_M0(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """The method: Mbar of the cells' default M, psi_0..psi_8, SEBA at r."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs = b.mbar()
    lab, rot, perm, info = seba_sets(vecs, r)
    return _result(lab, vecs[:, 1:r], vals, 'S+P', **info)


def src_L1(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """Cosine PCA (= classical MDS of chordal distances) of H C_B H, then the same SEBA."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs, einfo = b.pca()
    U = vecs[:, :r - 1]
    lab, rot, perm, info = seba_sets(with_constant(U), r)
    return _result(lab, U, vals[:N_VECS], 'S+P', eig=einfo, **info)


def src_L2(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """k-means (k_c = r, 20 starts) on PCA scores U_p Lambda_p^(1/2), p = max(q, 20). A fit."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs, einfo = b.pca()
    p = max(r - 1, 20)
    X = vecs[:, :p] * np.sqrt(np.clip(vals[:p], 0, None))
    lab, km = kmeans(X, r, seed)
    return _result(lab, None, None, 'B', p=p, inertia=float(km.inertia_),
                   sizes=np.bincount(lab, minlength=r).tolist())


def src_L3(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """Exact spherical k-means in the block Gram C_B (k_c = r, 20 k-means++ starts). A fit."""
    b = _block(grams, ops, cache, cell_keys)
    lab, info = spherical_kmeans_gram(b.CB(), r, seed)
    return _result(lab, None, None, 'B', sizes=np.bincount(lab, minlength=r).tolist(), **info)


def src_G1(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """k-means on the method's psi_1..psi_q (same span as M0, different rounding). A fit."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs = b.mbar()
    lab, km = kmeans(vecs[:, 1:r], r, seed)
    return _result(lab, vecs[:, 1:r], vals, 'S+B', rounding='kmeans', sizes=np.bincount(lab, minlength=r).tolist())


def src_G1n(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """NJW rounding: rows of [psi_0..psi_{r-1}] normalised to unit length, then k-means."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs = b.mbar()
    X = vecs[:, :r] / np.linalg.norm(vecs[:, :r], axis=1, keepdims=True)
    lab, km = kmeans(X, r, seed)
    return _result(lab, vecs[:, 1:r], vals, 'S+B', rounding='row-normalised kmeans (NJW)',
                   sizes=np.bincount(lab, minlength=r).tolist())


def src_G2(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """Textbook spectral clustering: same union-15NN 0/1 graph (no floor, no Sinkhorn, no
    lazy step), S = D^-1/2 A D^-1/2 per cell, block = mean of S; top r eigenvectors
    (the leading one is not constant), rows normalised, k-means with k_c = r."""
    b = _block(grams, ops, cache, cell_keys)
    vals, vecs, einfo = b.g2()
    X = vecs[:, :r] / np.linalg.norm(vecs[:, :r], axis=1, keepdims=True)
    lab, km = kmeans(X, r, seed)
    n = len(vecs)
    const_overlap = float(abs(vecs[:, 0].sum()) / np.sqrt(n))
    # The mean of degree-normalised matrices has no shared stationary vector: the basis
    # reported for span comparisons is psi_2..psi_r projected off the constant, re-orthonormalised.
    B = vecs[:, 1:r] - vecs[:, 1:r].mean(0, keepdims=True)
    B = np.linalg.qr(B)[0] if r > 1 else B
    return _result(lab, B, vals, 'S+B', eig=einfo, top_vector_constant_cosine=const_overlap,
                   sizes=np.bincount(lab, minlength=r).tolist())


def _src_C1(which):
    def f(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
        b = _block(grams, ops, cache, cell_keys)
        i = {'first': 0, 'mid': (b.L - 1) // 2, 'last': b.L - 1}[which]
        vals, vecs = b.cell_modes(i)
        lab, rot, perm, info = seba_sets(vecs, r)
        return _result(lab, vecs[:, 1:r], vals, 'S+P', cell=i, **info)
    f.__doc__ = f'The method on the {which} cell of the block alone (a pointer to stored data).'
    return f


def src_C2(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """Co-assignment consensus: A_B(i,j) = share of cells where i, j share a single-cell
    SEBA set (unassigned never co-assigned); top q eigenvectors of H A_B H; SEBA.
    A_B = F F^T / L with F the stacked one-hot single-cell labels, so H A_B H's
    eigenvectors are the left singular vectors of H F / sqrt(L) (exact, low rank)."""
    b = _block(grams, ops, cache, cell_keys)
    cols = []
    for i in range(b.L):
        lab = b.cell_seba(i, r)
        E = np.zeros((b.n, r)); m = lab >= 0
        E[np.where(m)[0], lab[m]] = 1
        cols.append(E)
    F = np.hstack(cols) / np.sqrt(b.L)
    F -= F.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    q = r - 1
    U = sign_fix(U[:, :q])
    lab, rot, perm, info = seba_sets(with_constant(U), r)
    return _result(lab, U, s[:N_VECS] ** 2, 'P', **info)


def src_C3(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None):
    """Mean-projector consensus: top q eigenvectors of mean_l U_l U_l^T (each cell's
    psi_1..psi_q); computed exactly as the left singular vectors of [U_1..U_L]/sqrt(L)."""
    b = _block(grams, ops, cache, cell_keys)
    q = r - 1
    F = np.hstack([b.cell_modes(i)[1][:, 1:r] for i in range(b.L)]) / np.sqrt(b.L)
    U, s, _ = np.linalg.svd(F, full_matrices=False)
    U = sign_fix(U[:, :q])
    lab, rot, perm, info = seba_sets(with_constant(U), r)
    return _result(lab, U, s[:min(len(s), N_VECS)] ** 2, 'S+P', **info)


SOURCES = {
    'M0': src_M0, 'L1': src_L1, 'L2': src_L2, 'L3': src_L3,
    'G1': src_G1, 'G1n': src_G1n, 'G2': src_G2,
    'C1first': _src_C1('first'), 'C1mid': _src_C1('mid'), 'C1last': _src_C1('last'),
    'C2': src_C2, 'C3': src_C3,
}
STATUS = {'M0': 'S+P', 'L1': 'S+P', 'L2': 'B', 'L3': 'B', 'G1': 'S+B', 'G1n': 'S+B', 'G2': 'S+B',
          'C1first': 'S+P', 'C1mid': 'S+P', 'C1last': 'S+P', 'C2': 'P', 'C3': 'S+P',
          'I1': 'S+P', 'I2': 'S+P', 'I3': 'S+P', 'Y1': 'V', 'Y2': 'V', 'N3': 'P'}
CHEAP = ['L1', 'L2', 'G1', 'G1n', 'G2', 'C2', 'C3']   # explorer-grid sources (+ L3 by timing)
