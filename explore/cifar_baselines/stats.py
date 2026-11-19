"""Comparison statistics for the baselines study (PLAN section 7, IDEAS section 4).

Everything here is descriptive: agreement numbers, overlaps, retention, linear content,
CKA, and permutation nulls. Nothing here selects a method or a rank.

Unassigned convention (PLAN section 7): a hard labelling is an int array with -1 for
"unassigned". Agreement scores are given (a) on all points with "unassigned" treated as
one more category, and (b) on the jointly assigned points, always reported together with
that coverage. Two unassigned points are never "in the same set" for Jaccard or
co-assignment: unassigned points belong to no set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CC = Path(__file__).resolve().parent.parent / 'cifar_coherent'
if _CC.is_dir() and str(_CC) not in sys.path:
    sys.path.insert(0, str(_CC))


def _np(x):
    """Torch tensor or array -> float64 numpy array (no copy when already numpy)."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- views from raw Grams

def raw_to_views(R, fit_idx=None):
    """Derive every cosine view from the raw uncentred Gram R = X X^T (N x N).

    Fit-only centring: with w uniform on the fit images, the centred dot products of ALL
    images are G = (I - 1 w^T) R (I - w 1^T) (each row of X minus the fit mean). Cosines
    divide by each image's own centred norm sqrt(G_ii) (a held-out image uses its own
    norm after subtracting the fit mean). Returned keys:
      C       centred cosine Gram on the fit set (= centred_cosine_gram of the fit rows)
      norms   centred norms of all N images (fit-mean centring)
      C_unc   uncentred cosine Gram of all N images (R normalised by sqrt(R_ii))
      fit_idx, test_idx  index arrays (test_idx empty when fit_idx is None / everything)
      C_tf    test x fit cross-cosines under fit-only centring (only for a strict subset)
      C_tt    test x test cosines under fit-only centring (only for a strict subset)
      G_fit   centred (fit-mean) dot-product Gram on the fit set; its nonzero eigenvalues
              divided by n_fit are the pre-normalisation covariance spectrum.
    Clamping to [-1, 1] and the 1e-12 norm floor follow coherent.centred_cosine_gram.
    """
    R = _np(R).astype(np.float64, copy=False)
    n = len(R)
    if fit_idx is None:
        fit_idx = np.arange(n)
    fit_idx = np.asarray(fit_idx)
    mask = np.zeros(n, bool); mask[fit_idx] = True
    test_idx = np.where(~mask)[0]
    w = np.zeros(n); w[fit_idx] = 1.0 / len(fit_idx)
    Rw = R @ w                      # R w  (N,)
    wRw = float(w @ Rw)
    # G = R - 1 (w^T R) - (R w) 1^T + (w^T R w) 1 1^T
    G = R - Rw[None, :] - Rw[:, None] + wRw
    G = 0.5 * (G + G.T)
    norms = np.sqrt(np.clip(np.diag(G), 0, None))
    inv = 1.0 / np.maximum(norms, 1e-12)
    Cn = np.clip(G * inv[:, None] * inv[None, :], -1.0, 1.0)
    du = np.sqrt(np.clip(np.diag(R), 0, None))
    iu = 1.0 / np.maximum(du, 1e-12)
    out = dict(C=Cn[np.ix_(fit_idx, fit_idx)], norms=norms,
               C_unc=np.clip(R * iu[:, None] * iu[None, :], -1.0, 1.0),
               fit_idx=fit_idx, test_idx=test_idx, G_fit=G[np.ix_(fit_idx, fit_idx)])
    if len(test_idx):
        out['C_tf'] = Cn[np.ix_(test_idx, fit_idx)]
        out['C_tt'] = Cn[np.ix_(test_idx, test_idx)]
    return out


def covariance_spectrum(R):
    """Eigenvalues (descending) of the pre-normalisation covariance (H R H)/N of the
    centred activations; at most N-1 are nonzero. Gram-only, so no feature matrix."""
    G = raw_to_views(R)['G_fit']
    lam = np.linalg.eigvalsh(G)[::-1] / len(G)
    return np.clip(lam, 0, None)


# ---------------------------------------------------------------- hard-set agreement

def _joint(a, b):
    a = np.asarray(a); b = np.asarray(b)
    m = (a >= 0) & (b >= 0)
    return a[m], b[m], float(m.mean())


def ami(a, b, unassigned='category'):
    """Adjusted mutual information (sklearn, arithmetic normalisation).
    unassigned='category': -1 is one more label; 'joint': only jointly assigned points."""
    from sklearn.metrics import adjusted_mutual_info_score
    if unassigned == 'joint':
        a, b, cov = _joint(a, b)
        if len(a) < 2:
            return float('nan')
    return float(adjusted_mutual_info_score(np.asarray(a), np.asarray(b)))


def ari(a, b, unassigned='category'):
    from sklearn.metrics import adjusted_rand_score
    if unassigned == 'joint':
        a, b, cov = _joint(a, b)
        if len(a) < 2:
            return float('nan')
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def _count(lab, k):
    """Declared number of sets of a labelling. With k given, labels must lie in -1..k-1 and
    sets that receive no point are EMPTY SETS that still count (size 0, Jaccard 0 with
    everything). k=None falls back to max(label)+1, which makes the result depend on how
    empty sets are numbered (REVIEW_2 item 3); every caller in this package passes k."""
    lab = np.asarray(lab)
    if k is None:
        return int(lab.max()) + 1 if (lab >= 0).any() else 0
    if lab.size and lab.max() >= k:
        raise ValueError(f'label {int(lab.max())} >= declared set count {k}')
    return int(k)


def contingency(a, b, ka=None, kb=None):
    """Rectangular contingency table, ka sets of a (rows) x kb sets of b (columns).
    Unassigned points (-1) are excluded from every set (they form no row/column)."""
    a = np.asarray(a); b = np.asarray(b)
    ka, kb = _count(a, ka), _count(b, kb)
    m = (a >= 0) & (b >= 0)
    t = np.zeros((ka, kb), np.int64)
    np.add.at(t, (a[m], b[m]), 1)
    return t


def matched_jaccard_detail(a, b, ka=None, kb=None):
    """Hungarian matching of sets maximising total Jaccard; unassigned in no set.
    ka, kb: declared set counts (r for a source, 10 for classes); empty sets count.
    Returns the matched pairs and per-pair J, with three summaries:
      max     sum of matched J / max(k_a, k_b)  (unmatched and empty sets count as 0)
      legacy  mean of matched J over the min(k_a, k_b) matched pairs (old explorer number)
      weighted  matched J weighted by (|A|+|B|) over the total size of all sets of both
    All three are invariant to renumbering the sets (including where an empty set sits)."""
    from scipy.optimize import linear_sum_assignment
    a = np.asarray(a); b = np.asarray(b)
    t = contingency(a, b, ka, kb)
    ka, kb = t.shape
    if ka == 0 or kb == 0:
        return dict(max=0.0, legacy=0.0, weighted=0.0, pairs=[], jaccard=[], k=(ka, kb))
    sa = np.bincount(a[a >= 0], minlength=ka)[:ka]
    sb = np.bincount(b[b >= 0], minlength=kb)[:kb]
    union = sa[:, None] + sb[None, :] - t
    J = t / np.maximum(union, 1)
    ri, ci = linear_sum_assignment(-J)
    jm = J[ri, ci]
    tot = sa.sum() + sb.sum()
    w = (sa[ri] + sb[ci]) / max(tot, 1)
    return dict(max=float(jm.sum() / max(ka, kb)), legacy=float(jm.mean()),
                weighted=float((w * jm).sum()), pairs=list(zip(ri.tolist(), ci.tolist())),
                jaccard=jm.tolist(), k=(ka, kb))


def matched_jaccard(a, b, denom='max', ka=None, kb=None):
    """Matched Jaccard summary; denom='max' (padded, divide by max(k_a,k_b)), 'legacy' or
    'weighted'. Pass the declared set counts ka, kb (see _count)."""
    if denom not in ('max', 'legacy', 'weighted'):
        raise ValueError(denom)
    return matched_jaccard_detail(a, b, ka, kb)[denom]


def agreement(a, b, ka=None, kb=None):
    """All hard-set agreement numbers of PLAN section 7 for one pair of labellings.
    Headline: ami_joint with joint_coverage (SCHEMA.md); ami_all treats unassigned as a
    category and so also rewards agreeing on which points are unassigned."""
    _, _, cov = _joint(a, b)
    d = matched_jaccard_detail(a, b, ka, kb)
    return dict(ami_all=ami(a, b), ami_joint=ami(a, b, 'joint'), ari_all=ari(a, b),
                ari_joint=ari(a, b, 'joint'), joint_coverage=cov,
                unassigned_a=float((np.asarray(a) < 0).mean()),
                unassigned_b=float((np.asarray(b) < 0).mean()),
                mj_max=d['max'], mj_legacy=d['legacy'], mj_weighted=d['weighted'])


def conditional_entropy(coarse, fine):
    """H(coarse | fine) in nats over all points, unassigned as its own category (S3).
    Zero exactly when every fine set lies inside one coarse set (a refinement)."""
    coarse = np.asarray(coarse) + 1; fine = np.asarray(fine) + 1
    t = np.zeros((coarse.max() + 1, fine.max() + 1))
    np.add.at(t, (coarse, fine), 1)
    p = t / t.sum()
    pf = p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.nansum(np.where(p > 0, p * np.log(p / pf), 0.0))
    return float(h)


# ---------------------------------------------------------------- subspaces

def _const_free_orthonormal(U):
    U = _np(U).astype(np.float64)
    if U.ndim == 1:
        U = U[:, None]
    U = U - U.mean(0, keepdims=True)
    Q, Rr = np.linalg.qr(U)
    keep = np.abs(np.diag(Rr)) > 1e-10 * max(1.0, np.abs(np.diag(Rr)).max())
    return Q[:, keep]


def subspace_overlap(U, V):
    """Principal-angle comparison of span(U) and span(V), constant projected out of both.
    cosines: full principal-cosine spectrum (descending); mean_cos2 = sum cos^2 / max(q_u, q_v)
    (= ||U^T V||_F^2 / q for equal dimensions); captured_u / captured_v use each side's
    dimension as denominator; min_cos = smallest of the min(q_u, q_v) cosines;
    chance = expected mean_cos2 for independent Haar subspaces of 1-perp, min(q)/(N-1)."""
    Qu = _const_free_orthonormal(U); Qv = _const_free_orthonormal(V)
    qu, qv = Qu.shape[1], Qv.shape[1]
    n = Qu.shape[0]
    if qu == 0 or qv == 0:
        return dict(mean_cos2=float('nan'), min_cos=float('nan'), cosines=[], q=(qu, qv))
    s = np.clip(np.linalg.svd(Qu.T @ Qv, compute_uv=False), 0, 1)
    s2 = float((s ** 2).sum())
    return dict(mean_cos2=s2 / max(qu, qv), captured_u=s2 / qu, captured_v=s2 / qv,
                min_cos=float(s.min()), cosines=s.tolist(), q=(qu, qv),
                chance=min(qu, qv) / (n - 1))


def leverage(U):
    """Span leverage h_i = (U U^T)_ii of the constant-free span, and its concentration
    (sum h)^2 / sum h^2 (an effective number of points; span-invariant)."""
    Q = _const_free_orthonormal(U)
    h = (Q ** 2).sum(1)
    return dict(h=h, effective_points=float(h.sum() ** 2 / (h ** 2).sum()),
                max_h=float(h.max()))


def participation(v):
    """Per-column participation number 1 / sum v^4 of unit columns (basis-dependent)."""
    v = _np(v)
    v = v / np.linalg.norm(v, axis=0, keepdims=True)
    return (1.0 / (v ** 4).sum(0)).tolist()


# ---------------------------------------------------------------- retention / smoothness

def retention(P, labels, k=None):
    """R_P(A) = 1_A^T P 1_A / |A| for every set A of a hard labelling, with its size null.

    P is any symmetric stochastic operator (a cell's P or M, or a block Mbar). For a
    uniformly random subset of size m the expectation is exactly t + (1-t)(m-1)/(N-1) with
    t = tr(P)/N (t = 0 for the default zero-diagonal P, 1/2 for M = (I+P)/2), for any
    stochastic P (only sum_ij P_ij = N is used; the diagonal need not be constant).
    Unassigned points belong to no set; an empty declared set has value and null None
    (NaN internally, written as JSON null by finite_json).
    Accepts a list of operators (one per cell); then values has shape (cells, k)."""
    if isinstance(P, (list, tuple)):
        rows = [retention(p, labels, k) for p in P]
        return dict(values=np.array([r['values'] for r in rows]),
                    null=np.array([r['null'] for r in rows]), sizes=rows[0]['sizes'])
    P = _np(P)
    labels = np.asarray(labels)
    n = len(P)
    k = _count(labels, k)
    E = np.zeros((n, k))
    E[np.where(labels >= 0)[0], labels[labels >= 0]] = 1
    sizes = E.sum(0)
    vals = np.einsum('ij,ij->j', E, P @ E) / np.maximum(sizes, 1)
    t = float(np.trace(P)) / n
    null = t + (1 - t) * (sizes - 1) / (n - 1)
    vals[sizes == 0] = np.nan
    null[sizes == 0] = np.nan
    return dict(values=vals, null=null, sizes=sizes.astype(int), t=t)


def dirichlet(P, f):
    """Per-column f^T (I - P) f / f^T f (a Rayleigh quotient; smaller = smoother)."""
    P = _np(P); f = _np(f)
    if f.ndim == 1:
        f = f[:, None]
    return (1 - np.einsum('ij,ij->j', f, P @ f) / np.einsum('ij,ij->j', f, f))


def dirichlet_profile(ops, f):
    """Mean, max and 90th percentile over the block's cells of the per-cell energy."""
    e = np.array([dirichlet(p, f) for p in ops])  # cells x cols
    return dict(per_cell=e, mean=e.mean(0), max=e.max(0), q90=np.quantile(e, .9, axis=0))


# ---------------------------------------------------------------- linear content, CKA

def linear_content(U, G):
    """||Pi_U G Pi_U||_F^2 / ||G||_F^2 for the constant-free span U and G = H C H (a
    centred Gram is formed if G is not centred), with the rank-q optimum
    sum_{j<=q} lambda_j^2 / sum lambda_j^2. Descriptive variance capture only."""
    G = _np(G).astype(np.float64)
    n = len(G)
    G = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    Q = _const_free_orthonormal(U)
    q = Q.shape[1]
    A = Q.T @ G @ Q
    lam = np.linalg.eigvalsh(G)[::-1]
    tot = float((lam ** 2).sum())
    return dict(value=float((A ** 2).sum() / tot), optimum=float((lam[:q] ** 2).sum() / tot), q=q)


def _centre(A):
    return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()


def _ucentre(A):
    A = np.array(A, dtype=np.float64, copy=True)
    n = len(A)
    np.fill_diagonal(A, 0)
    r = A.sum(1)
    out = A - r[:, None] / (n - 2) - r[None, :] / (n - 2) + A.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(out, 0)
    return out


def cka(C1, C2, debiased=False):
    """Linear CKA between Grams; debiased=True uses U-centring (IDEAS 5.1), which can be
    negative and is unstable when a corrected norm is near zero (returned as nan)."""
    C1 = _np(C1).astype(np.float64); C2 = _np(C2).astype(np.float64)
    if debiased:
        A, B = _ucentre(C1), _ucentre(C2)
    else:
        A, B = _centre(C1), _centre(C2)
    na, nb = np.sqrt((A * A).sum()), np.sqrt((B * B).sum())
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float((A * B).sum() / (na * nb))


def cka_perm_null(C1, C2, n_perm=100, seed=0, debiased=False):
    """CKA with C2's rows/columns jointly permuted (breaks image correspondence)."""
    rng = np.random.default_rng(seed)
    C2 = _np(C2)
    vals = []
    for _ in range(n_perm):
        p = rng.permutation(len(C2))
        vals.append(cka(C1, C2[np.ix_(p, p)], debiased))
    return band(vals)


# ---------------------------------------------------------------- permutation nulls (N7)

def band(samples, qs=(.05, .5, .95)):
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return dict(mean=float('nan'), n=0, **{f'q{int(q*100):02d}': float('nan') for q in qs})
    return dict(mean=float(s.mean()), sd=float(s.std()), n=int(len(s)),
                **{f'q{int(q*100):02d}': float(np.quantile(s, q)) for q in qs})


def perm_null(stat, a, b, n_perm=1000, seed=0, strata=None):
    """Size-preserving permutation null for an agreement statistic (N7).

    stat(a, b) -> float. `a` may be ONE labelling or a LIST of candidate labellings; for a
    list the statistic is max over candidates and that maximisation (with its whole
    Hungarian matching) is repeated inside every permutation with the same permuted b,
    so selection is calibrated. b is permuted as a whole vector, which preserves every set
    size and the unassigned count; strata (e.g. nothing, or a grouping) restricts the
    permutation to within strata. Returns observed, the null band and an upper-tail
    p-value with the +1 correction."""
    rng = np.random.default_rng(seed)
    cands = a if isinstance(a, (list, tuple)) else [a]
    b = np.asarray(b)
    def score(bb):
        return max(stat(c, bb) for c in cands)
    obs = score(b)
    null = []
    idx = np.arange(len(b))
    for _ in range(n_perm):
        if strata is None:
            p = rng.permutation(len(b))
        else:
            p = idx.copy()
            for g in np.unique(strata):
                m = np.where(strata == g)[0]
                p[m] = m[rng.permutation(len(m))]
        null.append(score(b[p]))
    null = np.asarray(null)
    return dict(observed=float(obs), null=band(null),
                p_upper=float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null))))


def fast_ami_null(a, b, n_perm=1000, seed=0, perms=None, return_draws=False, ka=None, kb=None):
    """N7 bands for AMI (unassigned as a category) and padded matched Jaccard together.

    Vectorised and exact: a permutation of b preserves both marginals, so the entropies and
    the expected mutual information (sklearn's EMI, arithmetic normalisation) are computed
    once; each permutation needs only its contingency table (one bincount for all draws)
    and a small Hungarian matching. Agrees with stats.ami / matched_jaccard per draw."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.cluster._expected_mutual_info_fast import expected_mutual_information
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    kma, kmb = _count(a, ka), _count(b, kb)   # declared set counts: empty sets count
    ua, ai = np.unique(a, return_inverse=True); ub, bi = np.unique(b, return_inverse=True)
    ua_n, ub_n = len(ua), len(ub)             # distinct observed labels (incl. -1)
    obs_t = np.zeros((ua_n, ub_n), np.int64); np.add.at(obs_t, (ai, bi), 1)
    ra, cb = obs_t.sum(1), obs_t.sum(0)
    def H(c):
        p = c[c > 0] / n
        return float(-(p * np.log(p)).sum())
    Ha, Hb = H(ra), H(cb)
    emi = float(expected_mutual_information(obs_t, n)) if ua_n > 1 and ub_n > 1 else 0.0
    if perms is None:
        perms = np.argsort(rng.random((n_perm, n)), axis=1)
    perms = np.asarray(perms); n_perm = len(perms)
    bp = bi[perms]                                   # (n_perm, n)
    flat = (np.arange(n_perm)[:, None] * (ua_n * ub_n) + ai[None, :] * ub_n + bp).ravel()
    T = np.bincount(flat, minlength=n_perm * ua_n * ub_n).reshape(n_perm, ua_n, ub_n).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        outer = ra[None, :, None] * cb[None, None, :]
        mi = np.where(T > 0, T / n * np.log(T * n / outer), 0.0).sum((1, 2))
    denom = .5 * (Ha + Hb) - emi
    am = (mi - emi) / denom if abs(denom) > 1e-15 else np.ones(n_perm)
    # matched Jaccard over assigned sets only (unassigned never in a set)
    sa = ua >= 0; sb = ub >= 0
    mj = np.zeros(n_perm)
    if sa.any() and sb.any():
        Ts = T[:, sa][:, :, sb]
        sz_a, sz_b = ra[sa], cb[sb]
        U = sz_a[None, :, None] + sz_b[None, None, :] - Ts
        J = Ts / np.maximum(U, 1)
        for t in range(n_perm):
            r_, c_ = linear_sum_assignment(-J[t])
            mj[t] = J[t][r_, c_].sum() / max(kma, kmb)
    out = dict(ami=band(am), mj_max=band(mj), observed_ami=ami(a, b),
               observed_mj_max=matched_jaccard(a, b, ka=kma, kb=kmb), n_perm=n_perm)
    if return_draws:
        out.update(ami_draws=am, mj_draws=mj)
    return out


def agreement_with_null(a, b, ka, kb, n_perm=1000, seed=0):
    """agreement() plus its own N7 references, same conventions as the grid rows:
    ami_all and mj_max (declared counts) under size-preserving permutations of b with the
    whole matching repeated; ami_joint under permutations of b within the jointly assigned
    points (conditional on coverage). Fields added: ami_null_mean, ami_null_q95,
    mj_null_mean, mj_null_q95, ami_joint_null_mean, ami_joint_null_q95, n_perm."""
    a = np.asarray(a); b = np.asarray(b)
    ag = agreement(a, b, ka, kb)
    nm = fast_ami_null(a, b, n_perm, seed, ka=ka, kb=kb)
    m = (a >= 0) & (b >= 0)
    nj = fast_ami_null(a[m], b[m], n_perm, seed + 1) if m.sum() > 1 else None
    ag.update(ami_null_mean=nm['ami']['mean'], ami_null_q95=nm['ami']['q95'],
              mj_null_mean=nm['mj_max']['mean'], mj_null_q95=nm['mj_max']['q95'],
              ami_joint_null_mean=nj['ami']['mean'] if nj else float('nan'),
              ami_joint_null_q95=nj['ami']['q95'] if nj else float('nan'), n_perm=n_perm)
    return ag


def overlap_with_chance(U, V):
    """subspace_overlap with the Haar chance value min(q)/(N-1) of mean_cos2 alongside."""
    so = subspace_overlap(U, V)
    return dict(mean_cos2=so['mean_cos2'], min_cos=so['min_cos'], chance=so.get('chance'))


# ---------------------------------------------------------------- JSON

def finite_json(o):
    """Recursively convert numpy types to Python and every non-finite float to None, so the
    output can be written with json.dumps(..., allow_nan=False) and parsed by a browser
    (REVIEW_2 item 1). Sizes, counts and error strings are kept unchanged."""
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, str) else k: finite_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return finite_json(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def dumps(o, **kw):
    import json
    return json.dumps(finite_json(o), allow_nan=False, **kw)
