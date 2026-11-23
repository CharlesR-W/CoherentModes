"""Through-depth (image x site) operator: sets that are allowed to change from site to site.

A block average gives one function on the probe images for a whole run of sites. Here a function
has one slice per site, f = (f_0, ..., f_{L-1}), each f_l a function on the N probe images, and the
operator charges for two things: roughness of each slice on its own site's walk, and change of the
slice from one site to the next (the network map is the identity on image indices):

    E_a(f) = sum_l  f_l^T (I - M_l) f_l / c_l   +   a^2 sum_{l<L-1} |f_{l+1} - f_l|^2 ,
    L_a    = blockdiag((I - M_l) / c_l)  +  a^2 (D_path (x) I_N)          (symmetric, PSD),

with D_path the path-graph Laplacian on the L sites (free ends). M_l are the symmetric doubly
stochastic lazy walks of README.md, so the slice-constant functions g (x) 1 form an invariant
subspace with eigenvalues a^2 * 4 sin^2(j pi / 2L) ("temporal" modes; `temporal_eigs`). Everything
here works in its orthogonal complement, slices of mean zero, where the constants never mix in.

c_l is a per-site scale: 1 ('none'), or the mean of the site's own first R nontrivial 1 - lambda_k
('own', label-free): by Ky Fan the least mean roughness of an orthonormal R-frame at that site,
set to 1 at every site. It does not equalise best cuts or gaps.

Limits (tests): a -> 0 gives every site's own eigenvectors (union of spectra); a -> infinity gives
slice-independent functions that are eigenvectors of mean_l (I - M_l)/c_l, the block average of
README.md when c = 1; L identical sites give phi_k (x) cos(j pi (l + 1/2) / L).

`paper_modes` is the matrix C of Trower, Djurdjevac Conrad & Klus (Chaos 35, 013126, 2025) for the
same walks (uniform density, so K_t = T_t = M_t). `paper_terms` is the exact identity

    sum_l b_l |f_l|^2 - 2 sum_{l<L-1} f_l^T M_l f_{l+1}
        = sum_{l<L-1} [ f_l^T (I - M_l) f_l + f_{l+1}^T (I - M_l) f_{l+1} + (f_{l+1}-f_l)^T M_l (f_{l+1}-f_l) ],

b = (1, 2, ..., 2, 1): the paper's Laplacian is the same two charges with a fixed exchange rate
(the change between sites is charged through M_l, at most 1 per unit), and site l's walk charged
on slices l and l+1. The last site's walk is never used.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

SCALE_R = 8


def centre(f: np.ndarray) -> np.ndarray:
    """f (L, N, k): remove each slice's mean."""
    return f - f.mean(1, keepdims=True)


def site_spectrum(m: np.ndarray, k: int = SCALE_R) -> np.ndarray:
    """First k nontrivial eigenvalues of one walk, descending."""
    n = len(m)
    w = np.linalg.eigvalsh(m - 1.0 / n)
    return w[::-1][:k]


def site_scales(ms: np.ndarray, mode: str, r: int = SCALE_R) -> np.ndarray:
    if mode == 'none':
        return np.ones(len(ms))
    if mode == 'own':
        return np.array([np.mean(1 - site_spectrum(m, r)) for m in ms])
    raise ValueError(mode)


def temporal_eigs(nsite: int, a2: float) -> np.ndarray:
    return a2 * 4 * np.sin(np.arange(nsite) * np.pi / (2 * nsite)) ** 2


def path_lap(f: np.ndarray) -> np.ndarray:
    """D_path f along the first axis (free ends)."""
    out = np.zeros_like(f)
    d = f[1:] - f[:-1]
    out[1:] += d
    out[:-1] -= d
    return out


def apply_L(ms: np.ndarray, c: np.ndarray, a2: float, f: np.ndarray) -> np.ndarray:
    """L_a f for f (L, N, k)."""
    return (f - np.matmul(ms, f)) / c[:, None, None] + a2 * path_lap(f)


def energy_terms(ms: np.ndarray, c: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-site roughness (L, k) (already divided by c) and per-edge squared change (L-1, k) of f (L, N, k)."""
    rough = np.einsum('lnk,lnk->lk', f, f - np.matmul(ms, f)) / c[:, None]
    change = ((f[1:] - f[:-1]) ** 2).sum(1)
    return rough, change


def _eig_top(mv, dim, k, tol, ncv=None, v0=None):
    op = LinearOperator((dim, dim), matvec=lambda v: mv(v[:, None])[:, 0], matmat=mv, dtype=np.float64)
    w, v = eigsh(op, k=k, which='LA', tol=tol, ncv=ncv or max(4 * k, 40), v0=v0)
    o = np.argsort(-w)
    return w[o], v[:, o]


def sign_fix(v: np.ndarray) -> np.ndarray:
    """v (L, N, k): make the largest-magnitude entry of each mode positive."""
    flat = v.reshape(-1, v.shape[-1])
    idx = np.abs(flat).argmax(0)
    s = np.sign(flat[idx, np.arange(flat.shape[1])])
    s[s == 0] = 1
    return v * s


class Solver:
    """Exact (L_a + eps)^{-1} by block elimination along the path of sites.

    L_a + eps is block tridiagonal with diagonal blocks D_l = (I - M_l)/c_l + (deg_l a^2 + eps) I and
    off-diagonal blocks -a^2 I, so twelve N x N inversions factor it. Lanczos on the inverse converges
    in a few dozen steps, where Lanczos on L_a itself needs thousands when the scaled walks have very
    different top eigenvalues (1/c_l up to ~100).
    """

    def __init__(self, ms: np.ndarray, c: np.ndarray, a2: float, eps: float | None = None):
        nl, n, _ = ms.shape
        self.ms, self.c, self.a2, self.shape = ms, c, a2, (nl, n)
        self.eps = eps = float(eps if eps is not None else 1e-3 * min(1.0, a2 if a2 > 0 else 1.0, (1.0 / c).min()))
        deg = np.r_[1.0, np.full(nl - 2, 2.0), 1.0] if nl > 1 else np.zeros(1)
        self.sinv = []
        eye = np.eye(n)
        for l in range(nl):
            s = (eye - ms[l]) / c[l] + (deg[l] * a2 + eps) * eye
            if l:
                s -= a2 * a2 * self.sinv[-1]
            self.sinv.append(np.linalg.inv((s + s.T) / 2))

    def solve(self, b: np.ndarray) -> np.ndarray:
        """(L_a + eps)^{-1} b for b (L, N, k)."""
        nl, a2 = self.shape[0], self.a2
        y = [b[0]]
        for l in range(1, nl):
            y.append(b[l] + a2 * (self.sinv[l - 1] @ y[-1]))
        x = [None] * nl
        x[-1] = self.sinv[-1] @ y[-1]
        for l in range(nl - 2, -1, -1):
            x[l] = self.sinv[l] @ (y[l] + a2 * x[l + 1])
        return np.stack(x)


def modes(ms: np.ndarray, c: np.ndarray, a2: float, k: int = 16, tol: float = 1e-10, solver: Solver | None = None):
    """Smallest k eigenpairs of L_a on slice-mean-zero functions.

    Returns vals (k,) ascending and vecs (L, N, k), orthonormal over all L*N entries.
    """
    nl, n, _ = ms.shape
    sv = solver or Solver(ms, c, a2)

    def mv(x):
        return centre(sv.solve(centre(x.reshape(nl, n, -1)))).reshape(nl * n, -1)

    rng = np.random.default_rng(0)
    w, v = _eig_top(mv, nl * n, k, tol, v0=centre(rng.standard_normal((nl, n, 1))).ravel())
    v = centre(v.reshape(nl, n, k))
    q, _ = np.linalg.qr(v.reshape(nl * n, k))                 # Rayleigh-Ritz on L_a itself for the values
    q = q.reshape(nl, n, k)
    h = np.einsum('lni,lnj->ij', q, apply_L(ms, c, a2, q))
    vals, r = np.linalg.eigh((h + h.T) / 2)
    return vals, sign_fix(q @ r)


def residual(ms, c, a2, vals, vecs) -> np.ndarray:
    """(k,) unprojected 2-norm residuals |L_a f - lambda f| of unit vectors."""
    r = apply_L(ms, c, a2, vecs) - vecs * vals
    return np.sqrt((r ** 2).sum((0, 1)))


def site_eigs(ms: np.ndarray):
    """Full eigendecomposition of every walk: lam (L, N) descending with the constant first, vec (L, N, N)."""
    lam, vec = [], []
    for m in ms:
        w, v = np.linalg.eigh((m + m.T) / 2)
        lam.append(w[::-1]); vec.append(v[:, ::-1])
    return np.stack(lam), np.stack(vec)


def balance_direct(ms: np.ndarray, c: np.ndarray, eigs=None, tol: float = 1e-10) -> float:
    """The balance coupling from one extremal eigensolve.

    On slice-mean-zero functions with every H_l = (I - M_l)/c_l positive definite, H + t (T - tau I) first
    becomes singular at 1/t* = lambda_max(H^{-1/2} (tau I - T) H^{-1/2}), tau = 4 sin^2(pi / 2L).
    Needs every walk connected (second eigenvalue below 1).
    """
    nl, n, _ = ms.shape
    lam, vec = eigs if eigs is not None else site_eigs(ms)
    h = (1 - lam[:, 1:]) / c[:, None]
    assert h.min() > 0, 'a walk is disconnected: no balance coupling'
    v = vec[:, :, 1:]
    hi = h ** -.5
    tau = 4 * np.sin(np.pi / (2 * nl)) ** 2

    def mv(x):
        f = x.reshape(nl, n - 1, -1) * hi[:, :, None]
        g = np.matmul(v, f)
        g = tau * g - path_lap(g)
        return (np.matmul(v.transpose(0, 2, 1), g) * hi[:, :, None]).reshape(nl * (n - 1), -1)

    w, _ = _eig_top(mv, nl * (n - 1), 1, tol)
    return float(1.0 / w[0])


def balance(ms: np.ndarray, c: np.ndarray, lo: float = 1e-4, hi: float = 1e4, iters: int = 18, tol: float = 1e-6):
    """Coupling a^2 at which the first temporal eigenvalue equals the first slice-mean-zero eigenvalue.

    lambda_1(a2) is a minimum of functions linear in a2, so it is concave, and it is bounded by its
    a = infinity value; g(a2) = a2 * 4 sin^2(pi / 2L) - lambda_1(a2) is therefore convex, negative
    near 0 and eventually positive: one crossing, found by bisection in log a2. This is the smallest
    coupling at which the first temporal eigenvalue is no longer below the first spatial one, the rule
    of Froyland, Kalia & Koltai (arXiv:2409.11984) for the supra-Laplacian; see SPEC_SPACETIME.md.
    """
    nl = len(ms)
    t1 = 4 * np.sin(np.pi / (2 * nl)) ** 2

    def g(a2):
        return a2 * t1 - modes(ms, c, a2, k=1, tol=tol)[0][0]

    while g(lo) >= 0 and lo > 1e-14:      # nearly disconnected walks put the crossing very low
        lo /= 100
    while g(hi) <= 0 and hi < 1e12:
        hi *= 100
    assert g(lo) < 0 < g(hi), (lo, hi)
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(iters):
        mid = (llo + lhi) / 2
        if g(np.exp(mid)) < 0:
            llo = mid
        else:
            lhi = mid
    return float(np.exp((llo + lhi) / 2))


# ---------------------------------------------------------------- the paper's matrix

PAPER_B = lambda nl: np.r_[1.0, np.full(nl - 2, 2.0), 1.0]


def paper_modes(ms: np.ndarray, k: int = 16, tol: float = 1e-9):
    """Largest k eigenpairs of C = B^{-1} A of the paper, on slice-mean-zero functions.

    A has blocks A[l, l+1] = A[l+1, l] = M_l (l < L-1), B = diag(1, 2, ..., 2, 1) (x) I.
    Solved as the symmetric B^{-1/2} A B^{-1/2}; returned vectors are the paper's f = B^{-1/2} g,
    scaled so that sum_l b_l |f_l|^2 = 1. vals descending (the positive half of a symmetric spectrum).
    """
    nl, n, _ = ms.shape
    bi = PAPER_B(nl)[:, None, None] ** -.5

    def mv(x):
        f = centre(x.reshape(nl, n, -1)) * bi
        out = np.zeros_like(f)
        out[:-1] += np.matmul(ms[:-1], f[1:])
        out[1:] += np.matmul(ms[:-1], f[:-1])
        return centre(out * bi).reshape(nl * n, -1)

    rng = np.random.default_rng(0)
    w, v = _eig_top(mv, nl * n, k, tol, v0=centre(rng.standard_normal((nl, n, 1))).ravel())
    return w, sign_fix(centre(v.reshape(nl, n, k)) * bi)


def paper_terms(ms: np.ndarray, f: np.ndarray):
    """(lhs, rough_own, rough_next, change) of the identity in the module docstring, each (k,)."""
    b = PAPER_B(len(ms))[:, None]
    lhs = (b * (f ** 2).sum(1)).sum(0) - 2 * np.einsum('lnk,lnk->k', f[:-1], np.matmul(ms[:-1], f[1:]))
    own = np.einsum('lnk,lnk->k', f[:-1], f[:-1] - np.matmul(ms[:-1], f[:-1]))
    nxt = np.einsum('lnk,lnk->k', f[1:], f[1:] - np.matmul(ms[:-1], f[1:]))
    d = f[1:] - f[:-1]
    change = np.einsum('lnk,lnk->k', d, np.matmul(ms[:-1], d))
    return lhs, own, nxt, change


# ---------------------------------------------------------------- reading a mode

def depth_weight(vecs: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """(k, L): share of each mode's squared norm at each site (rows sum to 1)."""
    w = (vecs ** 2).sum(1).T
    if b is not None:
        w = w * b
    return w / w.sum(1, keepdims=True)


def unit_slices(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.sqrt((vecs ** 2).sum(1, keepdims=True)).clip(1e-300)


def adjacent_cos(vecs: np.ndarray) -> np.ndarray:
    """(k, L-1): cosine between a mode's slices at neighbouring sites."""
    u = unit_slices(vecs)
    return (u[1:] * u[:-1]).sum(1).T


def slice_retention(ms: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """(k, L): f_l^T M_l f_l / f_l^T f_l, the explorer's retention for each slice on its own site."""
    u = unit_slices(vecs)
    return np.einsum('lnk,lnk->lk', u, np.matmul(ms, u)).T


def span_share(vecs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """(k, L): share of each unit slice inside span(basis); basis (N, r) or (L, N, r), orthonormal columns."""
    u = unit_slices(vecs)
    if basis.ndim == 2:
        basis = np.broadcast_to(basis, (len(u),) + basis.shape)
    return (np.einsum('lnr,lnk->lrk', basis, u) ** 2).sum(1).T


# ---------------------------------------------------------------- reading a low space (rotation-invariant)
# G (L, N, K): the first K modes' slices. Additive operator: orthonormal columns over all L*N entries.
# Paper's matrix: pass sqrt(b_l) f_l so that the columns are orthonormal in the same sense.

def occupancy(g: np.ndarray) -> np.ndarray:
    """(L,) trace of the projector's diagonal block at each site; sums to K."""
    return (g ** 2).sum((1, 2))


def local_strengths(g: np.ndarray) -> np.ndarray:
    """(L, K) eigenvalues of G_l^T G_l, descending: how strongly each distinct image direction is present at site l."""
    return np.stack([np.linalg.eigvalsh(x.T @ x)[::-1] for x in g]).clip(0)


def local_frames(g: np.ndarray, r: int) -> np.ndarray:
    """(L, N, r) leading left singular vectors of each G_l."""
    return np.stack([np.linalg.svd(x, full_matrices=False)[0][:, :r] for x in g])


def recurrent(g: np.ndarray):
    """Eigen-decomposition of R = sum_l G_l G_l^T (trace K): image directions counted once however many
    modes or depth envelopes reuse them. Returns strengths (K',) descending and directions (N, K')."""
    nl, n, k = g.shape
    u, s, _ = np.linalg.svd(g.transpose(1, 0, 2).reshape(n, nl * k), full_matrices=False)
    return s ** 2, u


def tradeoff(ms: np.ndarray, c: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """(A, B): total within-site roughness and total between-site change of the K-frame. The first K modes
    minimise A + a^2 B over orthonormal K-frames, so along increasing a^2, B cannot rise and A cannot fall."""
    rough, change = energy_terms(ms, c, g)
    return float(rough.sum()), float(change.sum())


def scalar_control(gaps: np.ndarray, a2: float) -> tuple[float, np.ndarray]:
    """Ground state of diag(g_l) + a^2 D_path, g_l = site l's smallest scaled 1 - lambda.

    E_a(f) >= sum_l g_l r_l^2 + a^2 sum_l (r_{l+1} - r_l)^2 with r_l = |f_l|, so the ground energy is a lower
    bound on the first mode's, and the squared ground vector is the depth weight the gaps alone would give
    if the cheapest image directions of all sites were the same function. Returns (energy, weight (L,))."""
    nl = len(gaps)
    d = np.diag(np.r_[1.0, np.full(nl - 2, 2.0), 1.0]) - np.eye(nl, k=1) - np.eye(nl, k=-1)
    w, v = np.linalg.eigh(np.diag(gaps) + a2 * d)
    return float(w[0]), v[:, 0] ** 2


def captured(g: np.ndarray, e: np.ndarray, site: int) -> np.ndarray:
    """(K,) cumulative squared norm of the unit image function e, placed at one site, inside the first 1..K modes."""
    e = e - e.mean()
    e = e / np.linalg.norm(e)
    return np.cumsum((g[site].T @ e) ** 2)


def permute_sites(ms: np.ndarray, rng) -> np.ndarray:
    """Control: relabel the images independently at every site. Every site's spectrum is unchanged;
    the correspondence between sites is destroyed."""
    out = []
    for m in ms:
        p = rng.permutation(len(m))
        out.append(m[np.ix_(p, p)])
    return np.stack(out)


# ---------------------------------------------------------------- post-hoc readout: sets through depth
# Not part of the construction: k-means is a fitted readout of the low space, labelled as such.

def shared_rows(g: np.ndarray, floor: float = 1e-3):
    """Rows of the low space in its own shared coordinates: X_l = G_l / |G_l|_F * sqrt(N), one scalar per
    site, so pooled row distances do not depend on the basis chosen for the space. live (L,) is False where
    a site holds under `floor` of an even share of the space (rows left at zero)."""
    nl, n, k = g.shape
    live = occupancy(g) / k > floor / nl
    x = np.zeros_like(g)
    for l in np.flatnonzero(live):
        x[l] = g[l] / np.linalg.norm(g[l]) * np.sqrt(n)
    return x, live


def aligned_rows(g: np.ndarray, floor: float = 1e-3):
    """`shared_rows`, then each site rotated onto the previous one by orthogonal Procrustes (an SVD).

    A different observable from the shared rows: within-site distances are untouched, rigid motion of the
    whole cloud between neighbouring sites is removed, so it asks whether the cloud changed shape. It
    minimises squared displacement, not label churn. The rotation is determined only as far as the
    cross matrix X_l^T Y_{l-1} is nonsingular; `link` (L-1,) returns sigma_min / sigma_max of that matrix
    (NaN where either site is not live; the chain restarts after a gap rather than bridging it).
    """
    x, live = shared_rows(g, floor)
    nl = len(g)
    y = np.zeros_like(g)
    link = np.full(nl - 1, np.nan)
    for l in range(nl):
        if not live[l]:
            continue
        y[l] = x[l]
        if l and live[l - 1]:
            u, sv, vt = np.linalg.svd(x[l].T @ y[l - 1])
            y[l] = x[l] @ (u @ vt)
            link[l - 1] = sv[-1] / sv[0]
    return y, live, link


def kmeans(x: np.ndarray, k: int, restarts: int = 8, iters: int = 200, seed: int = 0):
    """Plain Lloyd with k-means++ starts; returns labels and the within-cluster sum of squares."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(restarts):
        cen = x[[rng.integers(len(x))]]
        for _ in range(k - 1):
            d = ((x[:, None] - cen[None]) ** 2).sum(-1).min(1)
            cen = np.vstack([cen, x[rng.choice(len(x), p=d / d.sum())]])
        for _ in range(iters):
            d = (x ** 2).sum(1)[:, None] - 2 * x @ cen.T + (cen ** 2).sum(1)[None]
            lab = d.argmin(1)
            new = np.stack([x[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
            if np.linalg.norm(new - cen) <= 1e-10 * np.linalg.norm(cen):
                break
            cen = new
        cost = float(d[np.arange(len(x)), lab].sum())
        if cost < best[0]:
            best = (cost, lab.copy())
    return best[1], best[0]


def depth_sets(g: np.ndarray, k: int, seed: int = 0, align: bool = True) -> np.ndarray:
    """(L, N) int labels, -1 at sites that hold almost none of the low space. One k-means with centroids
    shared across sites on `aligned_rows` (align=True, 'shape-aligned') or `shared_rows` ('shared coordinates')."""
    y, live = (aligned_rows(g)[:2] if align else shared_rows(g))
    nl, n, _ = g.shape
    lab = np.full((nl, n), -1)
    lab[live] = kmeans(y[live].reshape(-1, g.shape[2]), k, seed=seed)[0].reshape(-1, n)
    return lab


def churn(lab: np.ndarray) -> np.ndarray:
    """(L-1,) share of images whose label differs between neighbouring sites (labels as assigned)."""
    return (lab[1:] != lab[:-1]).mean(1)
