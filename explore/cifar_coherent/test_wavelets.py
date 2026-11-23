"""Checks for wavelets.py on a planted lazy walk: orthonormal completeness, nesting, and dyadic compression."""
import numpy as np
from wavelets import diffusion_wavelets


def lazy_knn_walk(N=300, k=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, 3)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    d = 1 - x @ x.T; np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, 1)[:, :k]
    A = np.full((N, N), 1e-6); np.fill_diagonal(A, 0)
    for i in range(N):
        A[i, nn[i]] = 1
    A = np.maximum(A, A.T)
    # symmetric Sinkhorn to doubly stochastic
    s = np.ones(N)
    for _ in range(5000):
        s = np.sqrt(s / (A @ s))
    P = s[:, None] * A * s[None, :]
    assert np.abs(P.sum(1) - 1).max() < 1e-9
    return (np.eye(N) + P) / 2


def test_basis():
    M = lazy_knn_walk()
    N = len(M)
    L = diffusion_wavelets(M, eps=1e-6)
    F = np.concatenate([l['psi'] for l in L] + [L[-1]['phi']], 1)
    assert F.shape == (N, N), F.shape
    assert np.abs(F.T @ F - np.eye(N)).max() < 1e-8          # orthonormal basis of R^N
    for a, b in zip(L[:-1], L[1:]):                          # nesting: Phi_{j+1} subset of Phi_j
        if a['j'] == 0:
            continue
        proj = a['phi'] @ (a['phi'].T @ b['phi'])
        assert np.abs(proj - b['phi']).max() < 1e-8
    assert L[-1]['n'] == 1
    c = L[-1]['phi'][:, 0]
    assert np.abs(np.abs(c) - 1 / np.sqrt(N)).max() < 1e-3   # coarsest scaling function is the constant
    # rank of level j tracks #{lambda^(2^j) > eps}
    lam = np.linalg.eigvalsh(M)
    for l in L[:-1]:
        expect = int((lam ** (2 ** l['j']) > 1e-6).sum())
        assert abs(l['r'] - expect) <= max(3, 0.05 * expect), (l['j'], l['r'], expect)
    # fine-scale functions are localized: median participation of level-1 scaling functions is small
    ph = L[1]['phi']
    pr = 1 / (ph ** 4).sum(0)
    assert np.median(pr) < N / 10, np.median(pr)


if __name__ == '__main__':
    test_basis()
    print('wavelets: all checks passed')
