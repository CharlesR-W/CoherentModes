"""Density-aware layer-to-layer transfer estimators (PLAN.md section 6; design note TRANSFER.md).

Estimand: for paired views (x_i, y_i) = (z_{a,i}, z_{b,i}) of the same image, (T f)(y) = E[f(X) | Y = y] from
L2(mu_X) (source, view a) to L2(mu_Y) (destination, view b), with empirical marginals. Every object here is closed
form and ends in an SVD. Ridge terms are fits and are called fits. No labels enter anywhere.

Conventions (PLAN section 6, REVIEW_1 item 3):
* Every operator matrix `.T` is destination-by-source and acts on column vectors of source coefficients, as
  Q = M_b M_a does. Right singular vectors are source functions, left singular vectors are destination functions,
  and forward composition a -> b -> c is T_bc @ T_ab.
* Geometry: the only within-view geometry is the cosine of fit-centred, row-normalised activations, derived from the
  raw uncentred Gram R = X X^T (fit-only centring; held-out images use their own norm after the fit mean is removed).
* Image indices index the N images of R. Fit / tune / test index sets are disjoint. Features are built from fit
  images only (landmarks, PCA axes, graphs), then evaluated on any image.

Estimators:
  soft_ulam      soft-Ulam joint table of two coarse observations (partition-of-unity landmark memberships).
  whitened_cca   ridge-regularised whitened cross-covariance SVD in a declared finite feature space:
                 'linear' (cosine PCA scores), 'nystrom' (exponential dot-product kernel), 'diffusion'
                 (row-extended leading eigenfunctions of the view's own rule-0 Sinkhorn operator P).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import linalg as sla

_HERE = Path(__file__).resolve().parent
_CC = _HERE.parent / 'cifar_coherent'
for _p in (str(_HERE), str(_CC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ETA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)       # ridge fit: rho = eta * tr(C) / m (IDEAS 6.3)
C_GRID = (1 / 64, 1 / 16, 1 / 4, 1.0, 4.0)      # bandwidth: kappa = c * kappa_ref (kappa_ref from landmark spacing)
OCCUPANCY_MIN = 10.0                            # PLAN section 6 gate (soft count of fit images per landmark)
K_FUNCS = 6                                     # PLAN: the report targets the first <= 6 functions
KNN_K = 15                                      # rule 0 of the existing construction
KNN_FLOOR = 1e-6
EIG_REL = 1e-12                                 # relative eigenvalue cutoff for pseudo-inverses (rho = 0 only)


# ============================================================================================ views


class View:
    """Cosine view of one cell under fit-only centring.

    From the raw uncentred Gram R (N x N): with w uniform on the fit images, G = (I - 1 w^T) R (I - w 1^T), and the
    cosine of images i, j is G_ij / sqrt(G_ii G_jj) (norm floor 1e-12, clipped to [-1, 1], as
    coherent.centred_cosine_gram and stats.raw_to_views). Blocks are computed on demand, so an N = 10,000 R (or a
    memory-mapped one) never needs an N x N cosine copy. `from_X` gives the identical object from features (used
    only by synthetic benchmarks; asserted equal in the tests).
    """

    def __init__(self, fit, n, block, label=''):
        self.fit = np.asarray(fit)
        self.n = int(n)
        self._block = block
        self.label = label

    @classmethod
    def from_R(cls, R, fit, label=''):
        R = np.asarray(R)
        fit = np.asarray(fit)
        n = len(R)
        rw = np.asarray(R[:, fit], dtype=np.float64).mean(1)          # (R w)_i
        c = float(rw[fit].mean())                                     # w^T R w
        g = np.asarray(np.diagonal(R), dtype=np.float64) - 2 * rw + c
        inv = 1.0 / np.maximum(np.sqrt(np.clip(g, 0, None)), 1e-12)

        def block(rows, cols):
            rows = np.asarray(rows); cols = np.asarray(cols)
            Rb = np.asarray(R[np.ix_(rows, cols)], dtype=np.float64)
            G = Rb - rw[rows][:, None] - rw[cols][None, :] + c
            return np.clip(G * inv[rows][:, None] * inv[cols][None, :], -1.0, 1.0)
        return cls(fit, n, block, label)

    @classmethod
    def from_X(cls, X, fit, label=''):
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        fit = np.asarray(fit)
        Z = X - X[fit].mean(0, keepdims=True)
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)

        def block(rows, cols):
            return np.clip(Z[np.asarray(rows)] @ Z[np.asarray(cols)].T, -1.0, 1.0)
        return cls(fit, len(X), block, label)

    def cos(self, rows, cols, chunk=2048):
        rows = np.asarray(rows)
        if len(rows) <= chunk:
            return self._block(rows, cols)
        return np.concatenate([self._block(rows[s:s + chunk], cols) for s in range(0, len(rows), chunk)])


def view_via_stats(R, fit):
    """Full N x N fit-centred cosine matrix assembled from builder A's stats.raw_to_views (small N only)."""
    import stats
    out = stats.raw_to_views(R, fit)
    n = len(R)
    C = np.empty((n, n))
    f, t = out['fit_idx'], out['test_idx']
    C[np.ix_(f, f)] = out['C']
    if len(t):
        C[np.ix_(t, f)] = out['C_tf']; C[np.ix_(f, t)] = out['C_tf'].T
        C[np.ix_(t, t)] = out['C_tt']
    return C


# ============================================================================================ features


@dataclass
class Features:
    """A declared finite function space for one view, evaluated on all N images.

    arr: (N, m) values. kind: 'linear' | 'nystrom' | 'diffusion' | 'ulam' (partition-of-unity memberships).
    Built from fit images only; `key` identifies the construction (used to check composition compatibility).
    """
    arr: np.ndarray
    kind: str
    key: tuple
    meta: dict = field(default_factory=dict)

    @property
    def m(self):
        return self.arr.shape[1]


def _all(view):
    return np.arange(view.n)


def _landmarks(view, m, seed, salt):
    rng = np.random.default_rng([int(seed), int(salt)])
    order = rng.permutation(len(view.fit))
    return view.fit[order[:m]]


def _kappa_ref(view, L):
    """1 / median over landmarks of the cosine distance to the nearest other landmark (label-free scale)."""
    d = 1.0 - view.cos(L, L)
    np.fill_diagonal(d, np.inf)
    dn = d.min(1)
    med = float(np.median(dn))
    return 1.0 / max(med, 1e-12)


def linear_features(view, p):
    """Cosine PCA scores (kernel PCA of the linear cosine kernel, centred on the fit images), top p, evaluated on
    every image by the centred cross-kernel: (z_i - zbar)^T (z_j - zbar) over fit j, times U Lambda^-1/2."""
    fit = view.fit
    Cff = view.cos(fit, fit)
    col = Cff.mean(0); tot = float(col.mean())
    Hc = Cff - col[None, :] - col[:, None] + tot
    n = len(Hc)
    lam, U = sla.eigh(Hc, subset_by_index=[max(n - p, 0), n - 1], driver='evr')
    lam, U = lam[::-1], U[:, ::-1]
    keep = lam > 1e-10 * max(lam[0], 1e-300)
    r = int(min(p, keep.sum()))
    U, lam = U[:, :r], lam[:r]
    proj = U / np.sqrt(lam)[None, :]
    K = view.cos(_all(view), fit)
    Kc = K - K.mean(1, keepdims=True) - col[None, :] + tot
    return Features(Kc @ proj, 'linear', ('linear', view.label, int(p)),
                    dict(p=r, eig=lam.tolist()[:16]))


def nystrom_features(view, m, c, seed, salt=0, kappa_abs=None):
    """Nystrom features of the PSD kernel k(x, x') = exp(kappa (x.x' - 1)) on m random fit landmarks:
    F = K(., L) U Lambda^-1/2 (eigenvalues below 1e-10 of the largest dropped). kappa = c * kappa_ref, or the
    absolute kappa_abs when given (used to freeze a bandwidth while the landmark count changes)."""
    L = _landmarks(view, m, seed, salt)
    kref = _kappa_ref(view, L)
    kappa = c * kref if kappa_abs is None else float(kappa_abs)
    KLL = np.exp(kappa * (view.cos(L, L) - 1.0))
    lam, U = np.linalg.eigh(KLL)
    keep = lam > 1e-10 * lam.max()
    U, lam = U[:, keep], lam[keep]
    F = np.exp(kappa * (view.cos(_all(view), L) - 1.0)) @ (U / np.sqrt(lam)[None, :])
    return Features(F, 'nystrom', ('nystrom', view.label, int(m), float(c), int(seed), int(salt), kappa_abs),
                    dict(kappa=kappa, kappa_ref=kref, c=c, rank=int(keep.sum()), landmarks=L))


def ulam_memberships(view, m, c, seed, salt=0, kappa_abs=None):
    """Partition-of-unity memberships phi_j(x) = softmax_j(kappa (x.l_j - 1)) on m random fit landmarks."""
    L = _landmarks(view, m, seed, salt)
    kref = _kappa_ref(view, L)
    kappa = c * kref if kappa_abs is None else float(kappa_abs)
    S = kappa * (view.cos(_all(view), L) - 1.0)
    S -= S.max(1, keepdims=True)
    E = np.exp(S)
    Phi = E / E.sum(1, keepdims=True)
    return Features(Phi, 'ulam', ('ulam', view.label, int(m), float(c), int(seed), int(salt), kappa_abs),
                    dict(kappa=kappa, kappa_ref=kref, c=c, landmarks=L))


def rule0_operator(C, k=KNN_K, floor=KNN_FLOOR):
    """Rule 0 of the existing construction on a cosine block: union k-NN 0/1 graph, floor, zero diagonal,
    symmetric Sinkhorn (coherent.symmetric_balance). Returns P (n x n) and the log-potentials u with
    P_ij = K_ij exp(u_i + u_j)."""
    import torch
    from coherent import symmetric_balance
    n = len(C)
    d = 1.0 - C.copy()
    np.fill_diagonal(d, np.inf)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), bool)
    mask[np.arange(n)[:, None], idx] = True
    mask |= mask.T
    W = np.where(mask, 1.0, floor)
    np.fill_diagonal(W, 0.0)
    with np.errstate(divide='ignore'):
        logk = np.log(W)
    P = symmetric_balance(torch.from_numpy(logk)).numpy()
    S = np.log(np.where(np.eye(n, dtype=bool), 1.0, P)) - np.where(np.eye(n, dtype=bool), 0.0, logk)
    np.fill_diagonal(S, 0.0)
    rm = S.sum(1) / (n - 1)
    U = n * (S.sum() / (n * (n - 1))) / 2
    u = (rm - U / (n - 1)) * (n - 1) / (n - 2)
    recon = np.exp(np.where(np.eye(n, dtype=bool), -np.inf, logk) + u[:, None] + u[None, :])
    err = float(np.abs(recon - P).max())
    if err > 1e-8:
        raise RuntimeError(f'Sinkhorn potential recovery error {err}')
    dk = np.sort(d, axis=1)[:, k - 1]
    return P, u, dk


def diffusion_features(view, m, k=KNN_K, floor=KNN_FLOOR):
    """Diffusion coordinates: the m leading non-constant eigenvectors psi of the fit images' rule-0 operator P,
    read after one step of P: F(x) = sum_j P(x, j) psi(j). For fit images this is lambda_P psi (exact); a held-out
    image's row is the declared extension: union k-NN with the reverse-neighbour rule (x joins j's neighbourhood iff
    d(x, j) < d_k(j)), floor elsewhere, training Sinkhorn factors exp(u_j), renormalised to sum 1."""
    fit = view.fit
    n = len(fit)
    Cff = view.cos(fit, fit)
    P, u, dk = rule0_operator(Cff, k, floor)
    lam, Psi = sla.eigh(P, subset_by_index=[n - m - 1, n - 1])
    lam, Psi = lam[::-1], Psi[:, ::-1]
    const = np.ones(n) / math.sqrt(n)
    j0 = int(np.argmax(np.abs(const @ Psi)))
    keep = [j for j in range(Psi.shape[1]) if j != j0][:m]
    Psi, lam = Psi[:, keep], lam[keep]
    Psi = Psi - const[:, None] * (const @ Psi)[None, :]
    F = np.empty((view.n, Psi.shape[1]))
    F[fit] = P @ Psi
    out = np.setdiff1d(_all(view), fit)
    eu = np.exp(u - u.max())
    for s in range(0, len(out), 1024):
        rows = out[s:s + 1024]
        d = 1.0 - view.cos(rows, fit)
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        edge = np.zeros_like(d, dtype=bool)
        edge[np.arange(len(rows))[:, None], nn] = True
        edge |= d < dk[None, :]
        w = np.where(edge, 1.0, floor) * eu[None, :]
        w /= w.sum(1, keepdims=True)
        F[rows] = w @ Psi
    return Features(F, 'diffusion', ('diffusion', view.label, int(m), int(k), float(floor)),
                    dict(lambda_P=lam.tolist()[:16], k=k, floor=floor))


def make_features(view, spec, seed=0, salt=0):
    kind = spec[0]
    if kind == 'linear':
        return linear_features(view, spec[1])
    if kind == 'nystrom':
        return nystrom_features(view, spec[1], spec[2], seed, salt, spec[3] if len(spec) > 3 else None)
    if kind == 'ulam':
        return ulam_memberships(view, spec[1], spec[2], seed, salt, spec[3] if len(spec) > 3 else None)
    if kind == 'diffusion':
        return diffusion_features(view, spec[1])
    raise ValueError(kind)


# ============================================================================================ models


def _psd_inv_sqrt(A, rel=EIG_REL):
    """Symmetric (pseudo-)inverse square root, eigenvalues <= rel * max treated as zero."""
    lam, U = np.linalg.eigh(0.5 * (A + A.T))
    keep = lam > rel * max(lam.max(), 1e-300)
    return (U[:, keep] / np.sqrt(lam[keep])[None, :]) @ U[:, keep].T


def _psd_pinv(A, rel=EIG_REL):
    lam, U = np.linalg.eigh(0.5 * (A + A.T))
    keep = lam > rel * max(lam.max(), 1e-300)
    return (U[:, keep] / lam[keep][None, :]) @ U[:, keep].T


@dataclass
class SideStats:
    """Fit-split marginal moments of one side's features. For CCA: mean mu, covariance G of centred features;
    A = G + rho I (ridge fit). For soft Ulam: mu = 0, G = Phi^T W Phi (membership Gram), A = diag(marginal)."""
    feat: Features
    fit: np.ndarray
    mu: np.ndarray
    G: np.ndarray
    A: np.ndarray
    A_is: np.ndarray        # A^-1/2
    rho: float
    marginal: np.ndarray | None = None


def side_stats(feat, fit, eta=None):
    X = feat.arr[fit]
    n = len(fit)
    if feat.kind == 'ulam':
        mu = np.zeros(feat.m)
        G = X.T @ X / n
        p = X.mean(0)
        A = np.diag(p)
        A_is = np.diag(1.0 / np.sqrt(np.maximum(p, 1e-300)))
        return SideStats(feat, fit, mu, G, A, A_is, 0.0, p)
    mu = X.mean(0)
    Xc = X - mu
    G = Xc.T @ Xc / n
    rho = 0.0 if not eta else float(eta) * float(np.trace(G)) / feat.m
    A = G + rho * np.eye(feat.m)
    return SideStats(feat, fit, mu, G, A, _psd_inv_sqrt(A), rho)


@dataclass
class TransferModel:
    """Fitted transfer operator. `.T` is destination-by-source (whitened / marginal-normalised coordinates);
    `.singular_values` are the non-trivial ones (soft Ulam: the trivial pair (sqrt q, sqrt p), sigma = 1, is kept
    separately in `.trivial`). Source functions are evaluated on the source view, destination functions on the
    destination view of the same images (`dst_perm` re-pairs images for the permutation control)."""
    kind: str
    T: np.ndarray
    singular_values: np.ndarray
    src: SideStats
    dst: SideStats
    S: np.ndarray                 # fit cross-moment, destination x source (unwhitened; centred for CCA)
    src_coef: np.ndarray          # (m_src, k)  source function coefficients on centred features
    dst_coef: np.ndarray          # (m_dst, k)
    params: dict
    dst_perm: np.ndarray | None = None
    trivial: dict | None = None
    occupancy: dict | None = None

    def _dst_rows(self, idx):
        idx = np.asarray(idx)
        return idx if self.dst_perm is None else self.dst_perm[idx]

    def source_features(self, idx):
        return self.src.feat.arr[np.asarray(idx)] - self.src.mu

    def dest_features(self, idx):
        return self.dst.feat.arr[self._dst_rows(idx)] - self.dst.mu

    def source_functions(self, idx):
        return self.source_features(idx) @ self.src_coef

    def dest_functions(self, idx):
        return self.dest_features(idx) @ self.dst_coef

    @property
    def source_side(self):
        return self.src.feat.key

    @property
    def dest_side(self):
        return self.dst.feat.key


def _cross_moment(src, dst, fit, perm):
    X = src.feat.arr[fit] - src.mu
    Y = dst.feat.arr[fit if perm is None else perm[fit]] - dst.mu
    return Y.T @ X / len(fit)


def fit_cca(src, dst, perm=None, kind=None, eta=None):
    """Whitened cross-covariance SVD: T = (C_YY + rho_Y)^-1/2 C_YX (C_XX + rho_X)^-1/2 (destination x source)."""
    fit = src.fit
    S = _cross_moment(src, dst, fit, perm)
    T = dst.A_is @ S @ src.A_is
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    return TransferModel(kind or f'cca_{src.feat.kind}', T, s, src, dst, S, src.A_is @ Vt.T, dst.A_is @ U,
                         dict(eta=eta, rho_src=src.rho, rho_dst=dst.rho, m_src=src.feat.m,
                              m_dst=dst.feat.m, **{f'src_{k}': v for k, v in src.feat.meta.items()
                                                    if k in ('kappa', 'c', 'p', 'rank')}),
                         dst_perm=perm)


def occupancy(st):
    """Per-landmark soft count n p_j (fit images' worth of membership) and Kish effective sample size."""
    X = st.feat.arr[st.fit]
    soft = X.sum(0)
    kish = soft ** 2 / np.maximum((X ** 2).sum(0), 1e-300)
    return dict(soft_count=soft, kish=kish, min_soft=float(soft.min()), min_kish=float(kish.min()),
                passes=bool(soft.min() >= OCCUPANCY_MIN))


def fit_ulam(src, dst, perm=None):
    """Soft-Ulam joint table B = sum_i w_i psi(y_i) phi(x_i)^T (destination x source); q = row sums, p = column sums;
    T = D_q^-1/2 B D_p^-1/2. The top singular pair is the trivial one (sqrt q, sqrt p) with sigma = 1."""
    fit = src.fit
    B = _cross_moment(src, dst, fit, perm)
    p = B.sum(0); q = B.sum(1)
    T = (B / np.sqrt(np.maximum(q, 1e-300))[:, None]) / np.sqrt(np.maximum(p, 1e-300))[None, :]
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    sp, sq = np.sqrt(p), np.sqrt(q)
    triv = dict(sigma=float(s[0]), src_overlap=float(abs(Vt[0] @ sp) / np.linalg.norm(sp)),
                dst_overlap=float(abs(U[:, 0] @ sq) / np.linalg.norm(sq)))
    src_coef = Vt[1:].T / np.sqrt(np.maximum(p, 1e-300))[:, None]
    dst_coef = U[:, 1:] / np.sqrt(np.maximum(q, 1e-300))[:, None]
    occ = dict(src=occupancy(src), dst=occupancy(dst))
    return TransferModel('soft_ulam', T, s[1:], src, dst, B, src_coef, dst_coef,
                         dict(m_src=src.feat.m, m_dst=dst.feat.m, kappa_src=src.feat.meta['kappa'],
                              kappa_dst=dst.feat.meta['kappa'], c=src.feat.meta['c']),
                         dst_perm=perm, trivial=triv, occupancy=occ)


# ============================================================================================ evaluation


def heldout_correlations(model, idx, k=K_FUNCS):
    """Pearson correlations (evaluation-split means and variances) of the j-th source / destination functions on
    the images idx, and the full k x k cross-correlation matrix. Not equal to sigma_j in general."""
    f = model.source_functions(idx)[:, :k]
    g = model.dest_functions(idx)[:, :k]
    f = f - f.mean(0); g = g - g.mean(0)
    sf = np.sqrt((f ** 2).mean(0)); sg = np.sqrt((g ** 2).mean(0))
    cross = (f.T @ g / len(f)) / np.maximum(sf[:, None] * sg[None, :], 1e-300)
    return dict(corr=np.diag(cross).copy(), cross=cross)


def vamp2(model, idx, k=K_FUNCS, regularise=True):
    """PLAN VAMP-2: sum of squared singular values of the evaluation-split whitened cross-covariance of the fitted
    top-k non-constant source and destination functions. Constant removed (evaluation-split means); covariances
    regularised with the tuned rho (CCA: a^T (C_eval + rho I) a; soft Ulam: no ridge). Basis-invariant within the
    two k-dim spans. Equals deeptime's vamp_score(r=2, dim=k) - 1 at rho = 0 (see test)."""
    Xf = model.source_features(idx); Yf = model.dest_features(idx)
    Xf = Xf - Xf.mean(0); Yf = Yf - Yf.mean(0)
    a = model.src_coef[:, :k]; b = model.dst_coef[:, :k]
    n = len(Xf)
    Cxx = Xf.T @ Xf / n; Cyy = Yf.T @ Yf / n; Cxy = Xf.T @ Yf / n
    if regularise:
        Cxx = Cxx + model.src.rho * np.eye(len(Cxx)); Cyy = Cyy + model.dst.rho * np.eye(len(Cyy))
    Sf = a.T @ Cxx @ a; Sg = b.T @ Cyy @ b; Sfg = a.T @ Cxy @ b
    W = _psd_inv_sqrt(Sf, 1e-10) @ Sfg @ _psd_inv_sqrt(Sg, 1e-10)
    return float((W ** 2).sum())


# ============================================================================================ pairs, grids, tuning


class FeatureBank:
    """Views of the same N images (one per cell, shared fit indices) with a cache of features and fit moments.
    Features are keyed by view label, so a view has the same feature object whether it is a source or a destination
    (required for composition). Landmark draws are independent across views (salt = the view's position)."""

    def __init__(self, views, seed=0):
        views = dict(views)
        fits = [v.fit for v in views.values()]
        assert all(np.array_equal(fits[0], f) for f in fits), 'views must share fit indices'
        self.views, self.seed = views, seed
        self.labels = list(views)
        self.fit = fits[0]
        self.n = next(iter(views.values())).n
        self._feat, self._stats = {}, {}

    @classmethod
    def from_R(cls, Rs, fit, seed=0):
        return cls({lab: View.from_R(R, fit, lab) for lab, R in dict(Rs).items()}, seed)

    def features(self, label, spec):
        key = (label, spec)
        if key not in self._feat:
            self._feat[key] = make_features(self.views[label], spec, self.seed, salt=self.labels.index(label))
        return self._feat[key]

    def stats(self, label, spec, eta=None):
        key = (label, spec, eta)
        if key not in self._stats:
            self._stats[key] = side_stats(self.features(label, spec), self.fit, eta)
        return self._stats[key]

    def pair(self, a, b):
        return Pair(self, a, b)


class Pair:
    """Source view a and destination view b of one FeatureBank."""

    def __init__(self, bank, a, b):
        self.bank, self.a, self.b = bank, a, b
        self.fit, self.n = bank.fit, bank.n

    @classmethod
    def from_R(cls, Ra, Rb, fit, seed=0):
        return FeatureBank.from_R({'a': Ra, 'b': Rb}, fit, seed).pair('a', 'b')

    def stats(self, side, spec, eta=None):
        return self.bank.stats(self.a if side == 'a' else self.b, spec, eta)


def grid(estimator, m):
    """Label-free grid per estimator. Each entry: (feature spec, eta). Soft Ulam has no ridge."""
    if estimator == 'soft_ulam':
        return [(('ulam', m, c), None) for c in C_GRID]
    if estimator == 'linear':
        return [(('linear', m), e) for e in ETA_GRID]
    if estimator == 'nystrom':
        return [(('nystrom', m, c), e) for c in C_GRID for e in ETA_GRID]
    if estimator == 'diffusion':
        return [(('diffusion', m), e) for e in ETA_GRID]
    raise ValueError(estimator)


def fit_point(pair, estimator, spec, eta, perm=None, spec_dst=None):
    src = pair.stats('a', spec, eta); dst = pair.stats('b', spec_dst or spec, eta)
    if estimator == 'soft_ulam':
        return fit_ulam(src, dst, perm)
    return fit_cca(src, dst, perm, kind=f'cca_{estimator}', eta=eta)


def _gated_ulam_spec(pair, spec):
    """Occupancy gate: halve m (nested landmark prefix) until every landmark of both views has soft count >= 10."""
    m = spec[1]
    while m >= 2:
        s = ('ulam', m, spec[2])
        if occupancy(pair.stats('a', s))['passes'] and occupancy(pair.stats('b', s))['passes']:
            return s, m
        m //= 2
    return None, 0


def tune(pair, estimator, m, tune_idx, perm=None, k=K_FUNCS):
    """Label-free selection of (m, rho, bandwidth) by tuning-split VAMP-2. Returns (best model, table).
    For soft Ulam the occupancy gate is applied per bandwidth before scoring (m used is recorded)."""
    rows, best = [], None
    for spec, eta in grid(estimator, m):
        m_used = m
        if estimator == 'soft_ulam':
            spec, m_used = _gated_ulam_spec(pair, spec)
            if spec is None:
                rows.append(dict(spec=None, eta=None, m_used=0, vamp2_tune=float('nan')))
                continue
        model = fit_point(pair, estimator, spec, eta, perm)
        score = vamp2(model, tune_idx, k)
        rows.append(dict(spec=spec, eta=eta, m_used=m_used, vamp2_tune=score))
        if best is None or score > best[0]:
            best = (score, model, rows[-1])
    if best is None:
        return None, rows
    best[1].params.update(selected=dict(spec=best[2]['spec'], eta=best[2]['eta'], m_used=best[2]['m_used'],
                                         vamp2_tune=best[0]))
    return best[1], rows


def split_permutation(n, splits, rng, mode='all'):
    """Pairing permutation within each split (mode 'all'), or within the fit split only ('fit_only')."""
    perm = np.arange(n)
    for j, s in enumerate(splits):
        if mode == 'all' or j == 0:
            s = np.asarray(s)
            perm[s] = s[rng.permutation(len(s))]
    return perm


def evaluate(model, tune_idx, test_idx, k=K_FUNCS):
    hc = heldout_correlations(model, test_idx, k)
    ntr = min(k, len(model.singular_values))
    return dict(train_sv=model.singular_values[:k].tolist(), test_corr=hc['corr'].tolist(),
                vamp2_test=vamp2(model, test_idx, k), vamp2_tune=vamp2(model, tune_idx, k),
                n_funcs=ntr, selected=model.params.get('selected'),
                occupancy=None if model.occupancy is None else
                dict(min_soft_src=model.occupancy['src']['min_soft'], min_soft_dst=model.occupancy['dst']['min_soft'],
                     min_kish_src=model.occupancy['src']['min_kish'], min_kish_dst=model.occupancy['dst']['min_kish'],
                     passes=model.occupancy['src']['passes'] and model.occupancy['dst']['passes']))


def permutation_control(pair, estimator, m, tune_idx, test_idx, n_perm=200, seed=0, mode='all', k=K_FUNCS):
    """Independence control (i): pairing permuted within each of fit, tune and test, and the whole procedure
    (fit + tuning-split selection) repeated per permutation. mode='fit_only' gives the separately reported
    'training-pairing disruption'. Returns arrays (n_perm, k) of test correlations and training singular values,
    and test / tuning VAMP-2 values. The permutation sequence depends only on (seed, splits, mode), so runs with
    different m share permutations, and joint selection of m can be repeated inside each permutation post hoc."""
    rng = np.random.default_rng([int(seed), 7])
    n = pair.n
    corr = np.full((n_perm, k), np.nan); sv = np.full((n_perm, k), np.nan); v2 = np.full(n_perm, np.nan)
    v2t = np.full(n_perm, np.nan)
    for t in range(n_perm):
        perm = split_permutation(n, [pair.fit, tune_idx, test_idx], rng, mode)
        model, _ = tune(pair, estimator, m, tune_idx, perm, k)
        if model is None:
            continue
        hc = heldout_correlations(model, test_idx, k)['corr']
        corr[t, :len(hc)] = hc
        s = model.singular_values[:k]; sv[t, :len(s)] = s
        v2[t] = vamp2(model, test_idx, k)
        v2t[t] = model.params['selected']['vamp2_tune']
    return dict(test_corr=corr, train_sv=sv, vamp2_test=v2, vamp2_tune=v2t,
                q95=np.nanquantile(corr, .95, axis=0) if n_perm else None, mode=mode)


# ============================================================================================ R-level API


def _split_rest(n, fit):
    return np.setdiff1d(np.arange(n), fit)


def soft_ulam(Ra, Rb, fit, m, kappa, seed=0, pair=None):
    """Soft-Ulam table at m landmarks per view and bandwidth multiplier kappa (kappa_X = kappa * kappa_ref,X)."""
    pair = pair or Pair.from_R(Ra, Rb, fit, seed)
    return fit_point(pair, 'soft_ulam', ('ulam', int(m), float(kappa)), None)


def whitened_cca(Ra, Rb, fit, features='linear', m=64, rho=1e-2, c=1.0, seed=0, pair=None):
    """Ridge-regularised whitened cross-covariance SVD (a fit). `rho` is the relative ridge eta
    (rho_X = eta tr(C_XX)/m); `c` the Nystrom bandwidth multiplier."""
    pair = pair or Pair.from_R(Ra, Rb, fit, seed)
    spec = {'linear': ('linear', m), 'nystrom': ('nystrom', m, float(c)), 'diffusion': ('diffusion', m)}[features]
    return fit_point(pair, features, spec, rho)


# ============================================================================================ composition


def _compatible(T_ab, T_bc, T_ac):
    if T_ab.dst.feat is not T_bc.src.feat:
        raise ValueError('T_ab destination features and T_bc source features differ: composition undefined')
    if T_ab.src.feat is not T_ac.src.feat or T_bc.dst.feat is not T_ac.dst.feat:
        raise ValueError('endpoint features differ from the chain endpoints')
    if not (T_ab.dst.rho == T_bc.src.rho):
        raise ValueError('the intermediate view must carry the same ridge in both fits')
    for t in (T_ab, T_bc, T_ac):
        if t.dst_perm is not None:
            raise ValueError('composition is defined for the observed pairing only')


def fit_chain(bank, labels, estimator, m, tune_idx, k=K_FUNCS):
    """T_ab, T_bc, T_ac on one common feature specification (compatible intermediate features), selected by the
    sum of the three tuning-split VAMP-2 scores (label-free). Soft Ulam: occupancy gate on all three views (m halved
    for all three until every landmark passes)."""
    a, b, c = labels
    best = None
    for spec, eta in grid(estimator, m):
        if estimator == 'soft_ulam':
            mm = spec[1]
            while mm >= 2 and not all(occupancy(bank.stats(v, ('ulam', mm, spec[2])))['passes'] for v in labels):
                mm //= 2
            if mm < 2:
                continue
            spec = ('ulam', mm, spec[2])
        models = [fit_point(bank.pair(x, y), estimator, spec, eta) for x, y in ((a, b), (b, c), (a, c))]
        score = sum(vamp2(t, tune_idx, k) for t in models)
        if best is None or score > best[0]:
            best = (score, models, dict(spec=spec, eta=eta))
    for t in best[1]:
        t.params.update(selected=dict(best[2], vamp2_tune_sum=best[0]))
    return best[1]


def _dims(t_ab, t_bc):
    return dict(m_a=t_ab.src.feat.m, m_b=t_ab.dst.feat.m, m_c=t_bc.dst.feat.m)


def composition_sweep(bank, labels, estimator, m, tune_idx, test_idx, mb_factors=(0.25, 0.5, 1, 2, 4),
                      k=K_FUNCS):
    """Composition diagnostic with the intermediate basis varied and everything else frozen.

    The base chain is selected once (fit_chain at m; base intermediate size m_b0 = its actual b-feature count before
    rank truncation). Then for each m_b = round(factor * m_b0): the endpoint
    feature objects (a, c), the relative-ridge convention and eta, and the absolute kernel bandwidth kappa of every
    view (kappa_b frozen at the base value, landmarks nested) are held fixed; only the b-features change (Nystrom /
    Ulam: first m_b landmarks of the same order; diffusion: first m_b eigenfunctions; linear: first m_b PCs). A size
    at which the b-features are inadmissible (soft Ulam: a landmark below the occupancy gate) is reported as
    unavailable; endpoints are never shrunk. Actual feature dimensions are recorded for every row."""
    a, b, c = labels
    tab0, tbc0, tac = fit_chain(bank, labels, estimator, m, tune_idx, k)
    sel = tac.params['selected']
    spec, eta = sel['spec'], sel['eta']
    mb0 = int(spec[1])
    rows = []
    for fac in mb_factors:
        mb = max(2, int(round(fac * mb0)))
        if mb > len(bank.fit):
            rows.append(dict(m_b_requested=mb, factor=fac, unavailable='m_b exceeds the fit set'))
            continue
        if spec[0] in ('nystrom', 'ulam'):
            spec_b = (spec[0], int(mb), spec[2], tab0.dst.feat.meta['kappa'])
        else:
            spec_b = (spec[0], int(mb))
        row = dict(m_b_requested=int(mb), factor=fac, spec_b=list(spec_b), eta=eta)
        if estimator == 'soft_ulam':
            occ = occupancy(bank.stats(b, spec_b))
            row['occupancy_b_min_soft'] = occ['min_soft']
            if not occ['passes']:
                row['unavailable'] = 'intermediate landmarks below the occupancy gate'
                rows.append(row)
                continue
        tab = fit_point(bank.pair(a, b), estimator, spec, eta, spec_dst=spec_b)
        tbc = fit_point(bank.pair(b, c), estimator, spec_b, eta, spec_dst=spec)
        d = composition_discrepancy(tab, tbc, tac, test_idx, k)
        row.update(dims=_dims(tab, tbc), rho_b=tab.dst.rho, kappa_b=tab.dst.feat.meta.get('kappa'), **d)
        rows.append(row)
    return dict(base_selected=sel, base_dims=_dims(tab0, tbc0),
                kappa=dict(a=tac.src.feat.meta.get('kappa'), b=tab0.dst.feat.meta.get('kappa'),
                           c=tac.dst.feat.meta.get('kappa')), rows=rows)


def composed_model(T_ab, T_bc):
    """The composition T_bc @ T_ab as a TransferModel on a-features -> c-features (functions from its own SVD)."""
    T = T_bc.T @ T_ab.T
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    trivial = None
    if T_ab.kind == 'soft_ulam':
        trivial = dict(sigma=float(s[0])); U, s, Vt = U[:, 1:], s[1:], Vt[1:]
    return TransferModel(f'{T_ab.kind}_composed', T, s, T_ab.src, T_bc.dst, None,
                         T_ab.src.A_is @ Vt.T, T_bc.dst.A_is @ U, dict(composed=True), trivial=trivial)


def composition_discrepancy(T_ab, T_bc, T_ac, test_idx=None, k=K_FUNCS):
    """Composition discrepancy D = T_bc T_ab - T_ac (whitened coordinates) and an exact algebraic identity:

        D = A_c^-1/2 S_cb (A_b^-1 - G_b^+) S_ba A_a^-1/2       regulariser term (ridge for CCA; soft-membership
                                                                 overlap, D_p versus Phi^T W Phi, for soft Ulam)
          + A_c^-1/2 (S_cb G_b^+ S_ba - S_ca) A_a^-1/2           partial term = -(whitened partial cross-moment of
                                                                 a- and c-features given the b-features, fit split)

    The identity is exact on the fit split. The partial term vanishes in population when a -> b -> c is Markov and
    the b-features span L2(view b); otherwise finite basis, non-Markov views and sampling all contribute to it, and
    nothing here separates them. `heldout_residual_crossmoment` is a separate diagnostic, not a component of D:
    regressions on the b-features fitted on fit images, residual cross-moment of a- and c-features on test images,
    fit whitening (sign as the partial term). Norms: Frobenius, full and projected on the top-k non-trivial
    singular pairs of T_ac; relative = / ||Sigma_k||."""
    _compatible(T_ab, T_bc, T_ac)
    Aa, Ab, Ac = T_ab.src, T_ab.dst, T_bc.dst
    S_ba, S_cb, S_ca = T_ab.S, T_bc.S, T_ac.S
    Gb_p = _psd_pinv(Ab.G)
    Ab_inv = _psd_pinv(Ab.A)
    comp = T_bc.T @ T_ab.T
    total = comp - T_ac.T
    reg = Ac.A_is @ S_cb @ (Ab_inv - Gb_p) @ S_ba @ Aa.A_is
    partial = Ac.A_is @ (S_cb @ Gb_p @ S_ba - S_ca) @ Aa.A_is
    U, s, Vt = np.linalg.svd(T_ac.T)
    off = 1 if T_ac.kind == 'soft_ulam' else 0
    Uk, Vk, sk = U[:, off:off + k], Vt[off:off + k].T, s[off:off + k]
    scale = float(np.sqrt((sk ** 2).sum()))

    def norms(D):
        return dict(fro=float(np.linalg.norm(D)), topk=float(np.linalg.norm(Uk.T @ D @ Vk)),
                    topk_rel=float(np.linalg.norm(Uk.T @ D @ Vk) / max(scale, 1e-300)))
    lam_b = np.linalg.eigvalsh(0.5 * (Ab.G + Ab.G.T))
    keep_b = lam_b > EIG_REL * max(lam_b.max(), 1e-300)
    # G_b^+ cancels exactly in reg + partial; the floating-point residual scales with the cancelling product
    canc = float(np.linalg.norm(Ac.A_is @ S_cb, 2) * np.linalg.norm(Gb_p, 2) * np.linalg.norm(S_ba @ Aa.A_is, 2))
    err = float(np.abs(total - reg - partial).max())
    out = dict(total=norms(total), regulariser=norms(reg), partial=norms(partial),
               additivity_error=err, additivity_error_rel=err / max(canc, 1e-300),
               G_b_rank=int(keep_b.sum()), G_b_cond=float(lam_b.max() / lam_b[keep_b].min()),
               sv_endpoint=sk.tolist(), sv_composed=np.linalg.svd(comp, compute_uv=False)[off:off + k].tolist())
    if test_idx is not None:
        test_idx = np.asarray(test_idx)
        Ba = Gb_p @ S_ba                                          # (m_b, m_a): regression of a-features on b
        Bc = Gb_p @ S_cb.T                                        # (m_b, m_c)
        xa = T_ab.source_features(test_idx); xb = T_ab.dest_features(test_idx); xc = T_bc.dest_features(test_idx)
        ra = xa - xb @ Ba; rc = xc - xb @ Bc
        resid = -(Ac.A_is @ (rc.T @ ra / len(test_idx)) @ Aa.A_is)
        out['heldout_residual_crossmoment'] = norms(resid)
        comp_model = composed_model(T_ab, T_bc)
        out['test_corr_endpoint'] = heldout_correlations(T_ac, test_idx, k)['corr'].tolist()
        out['test_corr_composed'] = heldout_correlations(comp_model, test_idx, k)['corr'].tolist()
        fa = T_ac.source_functions(test_idx)[:, :k]; fc = comp_model.source_functions(test_idx)[:, :k]
        out['source_span_overlap_test'] = _overlap(fa, fc)
    return out


def _overlap(F1, F2):
    def orth(F):
        F = F - F.mean(0)
        Q, Rr = np.linalg.qr(F)
        keep = np.abs(np.diag(Rr)) > 1e-10 * max(1e-300, np.abs(np.diag(Rr)).max())
        return Q[:, keep]
    Q1, Q2 = orth(F1), orth(F2)
    s = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), 0, 1)
    return float((s ** 2).sum() / max(Q1.shape[1], Q2.shape[1]))


# ============================================================================================ references on images


def diffusion_reference(va_all, vb_all, k=K_FUNCS):
    """The existing construction's operators on all N images (views centred on all images, rule 0, M = (I+P)/2):
    leading non-constant right (source, view a) and left (destination, view b) singular vectors of Q = M_b M_a, and
    leading non-constant eigenvectors of the two-cell average Mbar = (M_a + M_b)/2."""
    allidx = np.arange(va_all.n)
    Ms = []
    for v in (va_all, vb_all):
        P, _, _ = rule0_operator(v.cos(allidx, allidx))
        Ms.append(0.5 * (np.eye(len(P)) + P))
    Q = Ms[1] @ Ms[0]
    U, s, Vt = np.linalg.svd(Q)
    lam, W = np.linalg.eigh(0.5 * (Ms[0] + Ms[1]))
    lam, W = lam[::-1], W[:, ::-1]
    return dict(Q_src=Vt[1:k + 1].T, Q_dst=U[:, 1:k + 1], Q_sv=s[1:k + 1], Mbar=W[:, 1:k + 1], Mbar_eig=lam[1:k + 1])
