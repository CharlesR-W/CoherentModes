"""Construction ablations (PLAN section 5) and stability / out-of-sample checks (section 4).

Every function works on ONE block and returns operators, a spectrum or labels; the
coordinator (measure_b.py) scores them with stats.py. Factors (PLAN section 5 table):
  A graph family/scale   logkernel(), effective_neighbours(), match_neighbours()
  B floor                logkernel(..., floor=), off_edge_mass(), components()
  C normalisation        operator(..., norm='sinkhorn'|'degree'|'raw'), stationary_drift()
  D diagonal             operator(..., diagonal='zero'|'retained'), product gamma
  E block combination    combine()
  F block weights        time_weights(), weighted mean in combine()
  G rounding             seba_mu(), kmeans on the same span, (q, k_c) grid in round_grid()
  H view preprocessing   view_grams() (uncentred cosine, GAP raw Gram, PCA of C vs HCH)
  I sample size          subsample_grams()
Stability: S1 subsample_block(), S2 fit_extend(), S3 stats.conditional_entropy,
leave-one-cell-out leave_one_out().
"""
from __future__ import annotations

import math

import numpy as np
import torch

import sources as S
from stats import raw_to_views, retention, agreement, subspace_overlap, agreement_with_null
from coherent import symmetric_balance  # noqa: E402
from sweep_math import perplexity_logkernel  # noqa: E402


# ---------------------------------------------------------------- A, B: kernels

def _dist(c):
    d = 1 - S._t(c)
    d.diagonal().fill_(float('inf'))
    return d


def logkernel(c, family='knn', **p):
    """Log of a zero-diagonal affinity on cosine distance d = 1 - C.
    knn: k (15), floor (1e-6; 0 = none), mutual (False)
    perplexity: perplexity (symmetrised per-point conditional, sweep_math)
    vmf: kappa (global)            -> -kappa d
    local: k (15)                  -> -d_ij / sqrt(delta_i delta_j), delta = k-th NN distance
    eps: k (15), floor             -> 1 if d_ij <= eps else floor, eps set so the edge count
                                      equals that of the union-kNN graph (matched mean degree)
    """
    c = S._t(c)
    if family == 'knn':
        return S.knn_logkernel(c, p.get('k', 15), p.get('floor', 1e-6), p.get('mutual', False))
    if family == 'perplexity':
        return perplexity_logkernel(c, p['perplexity'])[0]
    d = _dist(c)
    if family == 'vmf':
        return -float(p['kappa']) * d
    if family == 'local':
        delta = d.kthvalue(p.get('k', 15), dim=1).values
        return -d / torch.sqrt(delta[:, None] * delta[None, :])
    if family == 'eps':
        edges = int(S.knn_mask(c, p.get('k', 15)).sum())
        finite = d[torch.isfinite(d)]
        eps = float(finite.kthvalue(edges).values)
        w = torch.where(d <= eps, torch.ones_like(c), torch.full_like(c, p.get('floor', 1e-6)))
        w.diagonal().fill_(0)
        return torch.log(w)
    raise ValueError(family)


def effective_neighbours(P):
    """Median over rows of 1 / sum_{j != i} P_ij^2 (after normalisation; IDEAS A1)."""
    P = S._n(P).copy()
    np.fill_diagonal(P, 0)
    return float(np.median(1 / (P ** 2).sum(1)))


def off_edge_mass(P, mask):
    """Mean over rows of the balanced mass on non-edges (the floor's contribution)."""
    P = S._n(P); m = S._n(mask).astype(bool).copy()
    np.fill_diagonal(m, True)
    return float((P * ~m).sum() / len(P))


def components(mask):
    """Connected components of the pre-floor graph: count and largest share."""
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp
    k, lab = connected_components(sp.csr_matrix(S._n(mask).astype(float)), directed=False)
    return dict(components=int(k), largest_share=float(np.bincount(lab).max() / len(lab)))


def match_neighbours(c, family, target, lo, hi, iters=18, integer=False):
    """Bisection on the family parameter (k, perplexity or kappa) so that the median
    effective neighbour count of the zero-diagonal Sinkhorn P equals `target`
    (monotone decreasing in kappa, increasing in k / perplexity)."""
    key = dict(knn='k', perplexity='perplexity', vmf='kappa')[family]
    increasing = family != 'vmf'
    def val(x):
        P = symmetric_balance(logkernel(c, family, **{key: int(round(x)) if integer else x}))
        return effective_neighbours(P)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if integer and hi - lo <= 1:
            break
        v = val(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    x = int(round((lo + hi) / 2)) if integer else (lo + hi) / 2
    return x, val(x)


# ---------------------------------------------------------------- C, D: normalisation

def operator(logk, norm='sinkhorn', diagonal='zero', gamma=.5, alpha=0.0, self_logk=0.0):
    """One cell's operator from a zero-diagonal log-affinity.
    sinkhorn + zero      symmetric doubly stochastic P (zero diagonal), M = gamma I + (1-gamma) P
    sinkhorn + retained  the kernel diagonal K_ii = exp(self_logk) (1 for kNN 0/1 and vMF)
                         is kept before scaling; the scaled matrix is returned as is
    degree (alpha)       S = D^-1/2 K^(alpha) D^-1/2, K^(alpha) = K / (q_i q_j)^alpha; symmetric,
                         stationary measure pi ~ d (not uniform); returned with laziness gamma
    raw                  K / ||K||_2 (not a Markov operator; no retention is reported)
    Returns (operator torch float64, info)."""
    logk = S._t(logk).clone()
    if norm == 'sinkhorn':
        if diagonal == 'retained':
            logk.diagonal().fill_(float(self_logk))
            P = symmetric_balance(logk)
            return P, dict(kind='markov', stationary='uniform', diag_median=float(P.diag().median()))
        P = symmetric_balance(logk)
        return S.lazy(P, gamma), dict(kind='markov', stationary='uniform')
    K = torch.exp(logk)
    if diagonal == 'retained':
        K.diagonal().fill_(math.exp(self_logk))
    if norm == 'degree':
        q = K.sum(1)
        Ka = K / (q[:, None] * q[None, :]) ** alpha if alpha else K
        d = Ka.sum(1)
        Sm = Ka / torch.sqrt(d[:, None] * d[None, :])
        return S.lazy(Sm, gamma), dict(kind='symmetric-degree', alpha=alpha, stationary=(d / d.sum()).numpy())
    if norm == 'raw':
        return K / torch.linalg.matrix_norm(K, 2), dict(kind='raw')
    raise ValueError(norm)


def stationary_drift(infos):
    """Max total-variation distance of each cell's stationary vector from their mean, and
    from uniform (degree-normalised variants only; Sinkhorn is exactly uniform)."""
    pis = [i['stationary'] for i in infos if isinstance(i.get('stationary'), np.ndarray)]
    if not pis:
        return dict(tv_between=0.0, tv_uniform=0.0)
    pis = np.array(pis); m = pis.mean(0); n = pis.shape[1]
    return dict(tv_between=float(.5 * np.abs(pis - m).sum(1).max()),
                tv_uniform=float(.5 * np.abs(pis - 1 / n).sum(1).max()))


def top_unprojected(ops, k=S.N_VECS + 1):
    """Eigenpairs of a block mean WITHOUT assuming a shared constant mode (degree / raw
    variants): returns vals, vecs and the cosine of the top vector with the constant."""
    A = np.mean([S._n(m) for m in ops], 0)
    vals, vecs, info = S.top_eigs(A, k, project_constant=False)
    return vals, vecs, dict(info, top_constant_cosine=float(abs(vecs[:, 0].sum()) / math.sqrt(len(A))))


# ---------------------------------------------------------------- E, F: combination

def _centre2(Q):
    return Q - Q.mean(0, keepdims=True) - Q.mean(1, keepdims=True) + Q.mean()


def combine(ops, how='mean', weights=None, k=S.N_VECS, grams=None):
    """Block combination variants (column-vector convention; cell a applied first).
    mean       weighted mean of M (weights default equal) -> eigenpairs
    meansq     mean of M^2 -> eigenpairs
    product    Q = M_b ... M_a, constant projected both sides -> singular values,
               right (source) and left (destination) singular spans
    twoend     M_b M_a (same outputs)
    expprod    prod_l exp(-(tau/L) L_l), L_l = I - P_l, tau = L/2 (so tau/L = 1/2 matches
               M_l = I - L_l/2 to first order) -> singular spans
    expsum     exp(-(tau/L) sum_l L_l): its eigenvectors are exactly those of the mean
               (an identity; computed as a check) -> eigenpairs
    concat     one kNN graph on the concatenated view (default rule on C_B; needs grams)
    union / intersection of the cells' kNN graphs (needs grams), then floor, Sinkhorn, lazy
    Returns dict(kind, vals, vecs | left, right, info)."""
    ms = [S._n(m).astype(np.float64) for m in ops]
    L = len(ms)
    n = len(ms[0])
    if how in ('mean', 'meansq'):
        w = np.ones(L) / L if weights is None else np.asarray(weights, float) / np.sum(weights)
        A = sum(wi * (m @ m if how == 'meansq' else m) for wi, m in zip(w, ms))
        vals, vecs, info = S.top_eigs(A, k, project_constant=True)
        return dict(kind='eig', vals=vals, vecs=vecs, info=info)
    if how in ('product', 'twoend', 'expprod'):
        if how == 'expprod':
            fs = []
            for m in ms:
                lam, U = np.linalg.eigh(2 * m - np.eye(n))     # P_l
                fs.append((U * np.exp(-.5 * (1 - lam))) @ U.T)
        else:
            fs = ms if how == 'product' else [ms[0], ms[-1]]
        Q = fs[0]
        for f in fs[1:]:
            Q = f @ Q
        U, s, Vt = np.linalg.svd(_centre2(Q))
        return dict(kind='svd', vals=s[:k], left=S.sign_fix(U[:, :k]), right=S.sign_fix(Vt[:k].T),
                    info=dict(how=how))
    if how == 'expsum':
        Lsum = sum(np.eye(n) - (2 * m - np.eye(n)) for m in ms)
        lam, U = np.linalg.eigh(-.5 * Lsum)
        E = (U * np.exp(lam)) @ U.T
        vals, vecs, info = S.top_eigs(E, k, project_constant=True)
        return dict(kind='eig', vals=vals, vecs=vecs, info=info)
    if how in ('concat', 'union', 'intersection'):
        if grams is None:
            raise ValueError('needs grams')
        if how == 'concat':
            CB = np.mean([S._n(c) for c in grams], 0)
            M = S.default_operator(CB)
        else:
            masks = [S.knn_mask(c) for c in grams]
            mk = masks[0]
            for m in masks[1:]:
                mk = (mk | m) if how == 'union' else (mk & m)
            w = torch.where(mk, torch.ones(n, n, dtype=torch.float64), torch.full((n, n), 1e-6, dtype=torch.float64))
            w.diagonal().fill_(0)
            M = S.lazy(symmetric_balance(torch.log(w)), .5)
        vals, vecs, info = S.top_eigs(S._n(M), k, project_constant=True)
        return dict(kind='eig', vals=vals, vecs=vecs, info=info, op=M)
    raise ValueError(how)


def commutators(ops):
    """||M_i M_j - M_j M_i||_F for all cell pairs of the block (descriptive)."""
    ms = [S._n(m) for m in ops]
    L = len(ms)
    out = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            out[i, j] = out[j, i] = np.linalg.norm(ms[i] @ ms[j] - ms[j] @ ms[i])
    return out


def time_weights(steps, scheme='equal'):
    """Block weights for a through-training block over listed checkpoints `steps`.
    equal: one per listed checkpoint; interval: the training-step span each checkpoint
    represents (half-way to its neighbours); log: the same on log(1 + step)."""
    s = np.asarray(steps, float)
    if scheme == 'equal':
        return np.ones(len(s)) / len(s)
    x = s if scheme == 'interval' else np.log1p(s)
    if len(x) == 1:
        return np.ones(1)
    mids = (x[1:] + x[:-1]) / 2
    edges = np.concatenate([[x[0]], mids, [x[-1]]])
    w = np.diff(edges)
    w[w <= 0] = 1e-12
    return w / w.sum()


# ---------------------------------------------------------------- G: rounding

def seba_mu(V, r, c=.99, max_iter=5000, tol=1e-14):
    """The reference SEBA iteration (as in coherent/seba.py, Froyland-Rock-Sakellariou
    2019) with threshold mu = c / sqrt(N) as a parameter; c = .99 reproduces seba().
    Returns (labels, info)."""
    v0 = torch.as_tensor(np.ascontiguousarray(V[:, :r]), dtype=torch.float64)
    n = len(v0)
    mu = c / n ** .5
    v, tri = torch.linalg.qr(v0)
    g = torch.Generator().manual_seed(0)
    for j in range(r):
        if float(v[:, j].max() - v[:, j].min()) < 1e-14:
            v[:, j] += (torch.rand(n, generator=g, dtype=v.dtype) - .5) * 1e-12
    rot = torch.eye(r, dtype=v.dtype)
    for it in range(max_iter):
        z = v @ rot
        s = z.sign() * (z.abs() - mu).clamp_min(0)
        nr = s.norm(dim=0)
        if bool((nr == 0).any()):
            return -np.ones(n, np.int64), dict(error='empty column', c=c)
        s /= nr
        u, _, wt = torch.linalg.svd(v.T @ s)
        new = u @ wt
        if float((new - rot).norm()) <= tol:
            break
        if it != max_iter - 1:
            rot = new
    transform = torch.linalg.solve_triangular(tri, rot, upper=True)
    z = v0 @ transform
    s = z.sign() * (z.abs() - mu).clamp_min(0)
    nr = s.norm(dim=0)
    if bool((nr == 0).any()):
        return -np.ones(n, np.int64), dict(error='empty column', c=c)
    s /= nr
    sg = s.sum(0).sign(); sg[sg == 0] = 1
    s *= sg
    s /= s.max(0).values
    perm = s.min(0).values.argsort(descending=True, stable=True)
    s = s[:, perm].clamp_min(0)
    m, a = s.max(1)
    a[m <= 0] = -1
    lab = a.numpy().astype(np.int64)
    return lab, dict(c=c, iterations=it + 1, unassigned=float((lab < 0).mean()),
                     sizes=np.bincount(lab[lab >= 0], minlength=r).tolist())


def round_grid(vecs, qs=(1, 2, 3, 5, 7), kcs=(2, 3, 4, 6, 8), seed=0):
    """k-means on psi_1..psi_q with k_c sets over a small (q, k_c) grid (separates rank
    from the number of sets). vecs excludes the constant."""
    out = {}
    for q in qs:
        for kc in kcs:
            out[(q, kc)] = S.kmeans(vecs[:, :q], kc, seed)[0]
    return out


# ---------------------------------------------------------------- H, I: views, sample size

def view_grams(R, which='centred'):
    """Cosine views from the raw Gram: 'centred' (default), 'uncentred'."""
    v = raw_to_views(R)
    return torch.as_tensor(v['C'] if which == 'centred' else v['C_unc'])


def pca_c_vs_hch(C, k=S.N_VECS):
    """Top eigenvectors of C (what the stored 'PCs' diagonalise) and of H C H, and the
    overlap of their constant-free spans (IDEAS 1.1)."""
    C = S._n(C)
    lam, U = np.linalg.eigh(C)
    Uc = U[:, ::-1][:, :k]
    vals, Uh, _ = S.top_eigs(C - C.mean(0) - C.mean(1)[:, None] + C.mean(), k, project_constant=True)
    return dict(overlap=subspace_overlap(Uc, Uh), top_C_constant_cosine=float(abs(Uc[:, 0].sum()) / math.sqrt(len(C))))


def subsample_grams(Rs, idx, k_rule='fixed'):
    """Recompute every cell's centred cosine Gram on a subset (centring on the subset) and
    the default operator with k fixed at 15 or k proportional to N (k = 15 N / 2000)."""
    grams = [torch.as_tensor(raw_to_views(S._n(R)[np.ix_(idx, idx)])['C']) for R in Rs]
    k = 15 if k_rule == 'fixed' else max(3, int(round(15 * len(idx) / 2000)))
    ops = [S.lazy(symmetric_balance(S.knn_logkernel(c, k)), .5) for c in grams]
    return grams, ops, k


# ---------------------------------------------------------------- S1 subsampling

def subsample_block(Rs, frac=.8, n_sub=10, seed=0, ranks=S.RANKS, full=None):
    """S1: n_sub subsets of frac*N images without replacement; everything (centring,
    normalisation, graph, balancing) is recomputed; compared with the full-set M0 on the
    shared images (agreement per r and subspace overlap of psi_1..q)."""
    n = len(S._n(Rs[0]))
    if full is None:
        grams = [torch.as_tensor(raw_to_views(R)['C']) for R in Rs]
        fv = S.mbar_modes(S.default_ops(grams), S.N_VECS + 1)
    else:
        fv = full
    full_labels = {r: S.seba_sets(fv[1], r)[0] for r in ranks}
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_sub):
        idx = np.sort(rng.choice(n, int(round(frac * n)), replace=False))
        grams, ops, _ = subsample_grams(Rs, idx)
        vals, vecs = S.mbar_modes(ops, S.N_VECS + 1)
        row = dict(subset=t, lambda_M=vals.tolist(), per_r={})
        for r in ranks:
            lab = S.seba_sets(vecs, r)[0]
            row['per_r'][r] = dict(agreement_with_null(full_labels[r][idx], lab, r, r, seed=31 * t + r),
                                   overlap=subspace_overlap(fv[1][idx, 1:r], vecs[:, 1:r])['mean_cos2'],
                                   overlap_chance=(r - 1) / (len(idx) - 1))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- S2 fit / test extension

def _scaling_vector(P, logk):
    """Recover u (P = diag(e^u) K diag(e^u)) from the balanced P and its log-kernel:
    Y_ij = log P_ij - log K_ij = u_i + u_j on the off-diagonal (all finite with the floor)."""
    P = S._n(P); lk = S._n(logk)
    n = len(P)
    Y = np.log(np.where(np.eye(n, dtype=bool), 1.0, P)) - np.where(np.eye(n, dtype=bool), 0.0, lk)
    np.fill_diagonal(Y, 0)
    s = Y.sum() / (2 * n - 2)
    u = (Y.sum(1) - s) / (n - 2)
    return u


def extension_rows(C_ff, C_tf, P, logk, k=15, floor=1e-6):
    """Row extension of one cell's zero-diagonal Sinkhorn operator to held-out images
    (IDEAS 4.4): p(x, j) = K(x, z_j) a_j / sum_k K(x, z_k) a_k with the TRAINING scaling
    a = e^u. Union-kNN rule with a declared reverse-neighbour convention: K(x, z_j) = 1 if
    z_j is among x's k nearest fit images, or if d(x, z_j) < delta_j (z_j's stored k-th
    fit-neighbour distance, i.e. x would displace one of z_j's neighbours); floor otherwise.
    Rows sum to one (constants are preserved); the enlarged operator is NOT jointly
    rebalanced to be doubly stochastic (a stated limitation)."""
    C_ff = S._n(C_ff); C_tf = S._n(C_tf)
    a = np.exp(_scaling_vector(P, logk))
    d_ff = 1 - C_ff.copy(); np.fill_diagonal(d_ff, np.inf)
    delta = np.partition(d_ff, k - 1, axis=1)[:, k - 1]
    d_tf = 1 - C_tf
    near = np.argpartition(d_tf, k - 1, axis=1)[:, :k]
    edge = np.zeros_like(d_tf, bool)
    edge[np.arange(len(d_tf))[:, None], near] = True
    edge |= d_tf < delta[None, :]
    K = np.where(edge, 1.0, floor) * a[None, :]
    return K / K.sum(1, keepdims=True), dict(mean_degree=float(edge.sum(1).mean()))


def _frozen_seba_apply(V_fit, V_test, rot, perm):
    """Apply a fit-set SEBA solution to extended vectors: the training rotation, threshold
    mu = .99/sqrt(N_fit), column norms, signs and max scaling are all frozen."""
    n = len(V_fit)
    mu = .99 / n ** .5
    def thr(z):
        return np.sign(z) * np.clip(np.abs(z) - mu, 0, None)
    s = thr(V_fit @ rot)
    norms = np.linalg.norm(s, axis=0)
    s /= norms
    sg = np.sign(s.sum(0)); sg[sg == 0] = 1
    s *= sg
    mx = s.max(0)
    st = thr(V_test @ rot) / norms * sg / mx
    st = np.clip(st[:, perm], 0, None)
    lab = st.argmax(1)
    lab[st.max(1) <= 0] = -1
    return lab.astype(np.int64)


def fit_extend(Rs, fit_idx, ranks=S.RANKS, refit=None, lam_min=.1, seed=0):
    """S2: disjoint fit/test with a frozen extension, for M0, L1 and L3.
    Centring from the fit images only (stats.raw_to_views); fit construction on fit images;
    extension of M0 by averaged extended rows and u_hat = pbar(x,.) u / lambda_P
    (lambda_P = 2 lambda_M - 1; directions with lambda_P < lam_min are flagged), frozen SEBA;
    L1 by fit-centred kernel-PCA projection, frozen SEBA; L3 by frozen spherical centroids.
    Compared with `refit` (labels per r of the same source refit on all N, test rows) and
    scored on a graph built from the test images alone (default rule, fit-only centring):
    retention per set against its size null, block mean over cells."""
    n = len(S._n(Rs[0]))
    fit_idx = np.sort(np.asarray(fit_idx))
    views = [raw_to_views(S._n(R), fit_idx) for R in Rs]
    test_idx = views[0]['test_idx']
    Cf = [v['C'] for v in views]; Ctf = [v['C_tf'] for v in views]; Ctt = [v['C_tt'] for v in views]
    logks = [S.knn_logkernel(c) for c in Cf]
    Ps = [symmetric_balance(lk) for lk in logks]
    ops = [S.lazy(p.clone(), .5) for p in Ps]
    vals, vecs = S.mbar_modes(ops, S.N_VECS + 1)
    rows = [extension_rows(cf, ctf, p, lk)[0] for cf, ctf, p, lk in zip(Cf, Ctf, Ps, logks)]
    pbar = np.mean(rows, 0)
    lamP = 2 * vals - 1
    ext = np.empty((len(test_idx), vecs.shape[1]))
    ext[:, 0] = 1 / np.sqrt(len(fit_idx))
    ext[:, 1:] = pbar @ vecs[:, 1:] / lamP[None, 1:]
    test_ops = [S.default_operator(torch.as_tensor(c)) for c in Ctt]
    out = dict(n_fit=len(fit_idx), n_test=len(test_idx), lambda_P=lamP.tolist(),
               flagged_small_lambda=[int(i) for i in np.where(lamP[1:] < lam_min)[0] + 1], M0={}, L1={}, L3={})

    def score(name, r, lab_test):
        ret = retention([S._n(m) for m in test_ops], lab_test, r)
        d = dict(test_retention=np.nanmean(ret['values'], 0).tolist(), test_null=np.nanmean(ret['null'], 0).tolist(),
                 sizes=ret['sizes'].tolist(), unassigned=float((lab_test < 0).mean()))
        if refit is not None and name in refit:
            d.update(agreement_with_null(refit[name][r][test_idx], lab_test, r, r, seed=r))
        out[name][r] = d

    for r in ranks:
        lab, rot, perm, _ = S.seba_sets(vecs, r)
        if rot is not None:
            score('M0', r, _frozen_seba_apply(vecs[:, :r], ext[:, :r], rot, perm))
    # L1: kernel PCA of the block cosine Gram with fit-set centring
    Kff = np.mean(Cf, 0); Ktf = np.mean(Ctf, 0)
    nf = len(fit_idx)
    cm = Kff.mean(0)
    Gff = Kff - cm[None, :] - cm[:, None] + Kff.mean()
    lv, U, _ = S.top_eigs(Gff, S.N_VECS, project_constant=True)
    Gtf = Ktf - cm[None, :] - Ktf.mean(1, keepdims=True) + Kff.mean()
    Ut = Gtf @ U / lv[None, :]
    for r in ranks:
        Vf = S.with_constant(U[:, :r - 1])
        Vt = np.hstack([np.full((len(test_idx), 1), 1 / np.sqrt(nf)), Ut[:, :r - 1]])
        lab, rot, perm, _ = S.seba_sets(Vf, r)
        if rot is not None:
            score('L1', r, _frozen_seba_apply(Vf, Vt, rot, perm))
    # L3: frozen spherical centroids in the block Gram
    for r in ranks:
        lab, _ = S.spherical_kmeans_gram(Kff, r, seed)
        H = np.zeros((nf, r)); H[np.arange(nf), lab] = 1
        nrm = np.sqrt(np.clip((H * (Kff @ H)).sum(0), 1e-300, None))
        score('L3', r, np.argmax(Ktf @ H / nrm, 1).astype(np.int64))
    return out


# ---------------------------------------------------------------- leave one cell out

def leave_one_out(ops, r, next_op=None):
    """Block sets from all cells but one, scored on the omitted cell's operator (retention
    per set with size null); the full block's sets scored on the first cell after the
    block (`next_op`), which tests generalisation across views, not images."""
    rows = []
    L = len(ops)
    for i in range(L):
        rest = [m for j, m in enumerate(ops) if j != i]
        if not rest:
            break
        _, v = S.mbar_modes(rest, S.N_VECS + 1)
        lab = S.seba_sets(v, r)[0]
        ret = retention(S._n(ops[i]), lab, r)
        rows.append(dict(omitted=i, retention=ret['values'].tolist(), null=ret['null'].tolist(),
                         sizes=ret['sizes'].tolist()))
    out = dict(rows=rows)
    if next_op is not None:
        _, v = S.mbar_modes(ops, S.N_VECS + 1)
        lab = S.seba_sets(v, r)[0]
        ret = retention(S._n(next_op), lab, r)
        out['next'] = dict(retention=ret['values'].tolist(), null=ret['null'].tolist())
    return out
