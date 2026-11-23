"""Diffusion wavelets (Coifman & Maggioni 2006, "Diffusion wavelets", ACHA 21:53-94) for one symmetric walk operator.

Exploratory, for the explorer's "diffusion wavelet" colouring. Input: a single snapshot's fixed-lazy walk
M = (I + P)/2 (symmetric doubly stochastic, spectrum in [0, 1]). Nothing is fitted: every step is a pivoted QR of a
dyadic power of M, represented on the previous level's scaling functions.

Level j represents T_j = M^(2^j) on the orthonormal scaling basis Phi_j (columns of B_j, in image coordinates);
Phi_0 = the N delta functions. Then
  T_j[:, piv] = Q R            (Householder QR with column pivoting, LAPACK geqp3)
  r_j = #{k : |R_kk| > eps}    (geqp3's |R_kk| are non-increasing)
  Phi_{j+1} = Phi_j Q[:, :r_j]  -- Gram-Schmidt of the pivoted columns of T_j (pivoting is by residual norm, not localization)
  T_{j+1} = Q_r^T T_j^2 Q_r = A A^T with A = Q_r^T T_j  (T_j symmetric)
  Psi_j  = Phi_j W, W = the first (n_j - r_j) pivoted-QR columns of the projector I - Q_r Q_r^T
so Phi_j = Phi_{j+1} (+) Psi_j and the wavelets of all levels plus the last scaling functions are an orthonormal
basis of R^N. Each function carries a centre: the image of the delta it descends from through the pivots.
"""
import numpy as np
from scipy.linalg import qr


def diffusion_wavelets(M, eps=1e-4, max_levels=24):
    """Returns a list of levels; level j is a dict with
    n (= n_j), r (= n_{j+1}), phi (N, n_j) scaling functions Phi_j (omitted for j = 0: deltas), phi_c (n_j,) centres,
    psi (N, n_j - r) wavelets Psi_j, psi_c centres, rdiag (n_j,) |R_kk| of the pivoted QR of T_j.
    The last entry has psi of width 0 and is the coarsest scaling level (n_J == 1 or max_levels reached)."""
    M = np.asarray(M, dtype=np.float64)
    N = len(M)
    T = (M + M.T) / 2
    B = None                                  # Phi_j in image coordinates; None means identity (j = 0)
    cen = np.arange(N)
    out = []
    for j in range(max_levels):
        n = len(T)
        Q, R, piv = qr(T, pivoting=True, mode='economic')
        rd = np.abs(np.diag(R))
        r = max(1, int((rd > eps).sum()))
        Qr = Q[:, :r]
        # wavelets: localized orthonormal basis of the complement of span(Qr) in R^{n}
        if r < n:
            Pc = np.eye(n) - Qr @ Qr.T
            W, Rw, pw = qr(Pc, pivoting=True, mode='economic')
            W = W[:, :n - r]
            wc = cen[pw[:n - r]]
        else:
            W, wc = np.zeros((n, 0)), np.zeros(0, int)
        lev = dict(j=j, n=n, r=r, rdiag=rd, phi_c=cen.copy(), psi_c=wc,
                   phi=(np.eye(N) if B is None else B), psi=(W if B is None else B @ W))
        out.append(lev)
        if n == 1:
            lev['r'] = 1
            lev['psi'] = np.zeros((N, 0)); lev['psi_c'] = np.zeros(0, int)
            break
        A = Qr.T @ T
        T = A @ A.T
        T = (T + T.T) / 2
        B = Qr if B is None else B @ Qr
        cen = cen[piv[:r]]
        if j == max_levels - 1:               # record the last scaling level without further compression
            out.append(dict(j=j + 1, n=r, r=r, rdiag=np.zeros(r), phi_c=cen.copy(), psi_c=np.zeros(0, int),
                            phi=B, psi=np.zeros((N, 0))))
    return out


def sign_fix(F):
    """Each column's largest-|.| entry positive."""
    if F.shape[1] == 0:
        return F
    s = np.sign(F[np.abs(F).argmax(0), np.arange(F.shape[1])]); s[s == 0] = 1
    return F * s


def sparsify(f, mass=0.99, cap=None):
    """Smallest set of entries holding `mass` of the squared norm (largest |f| first); returns (idx, vals)."""
    o = np.argsort(-np.abs(f), kind='stable')
    c = np.cumsum(f[o] ** 2)
    k = int(np.searchsorted(c, mass * c[-1]) + 1)
    if cap:
        k = min(k, cap)
    return o[:k], f[o[:k]]
