"""Null generators (PLAN section 3) and the per-block summary they are compared on.

Each null preserves something stated and destroys something stated; none is "the" null.
All are status P (null generators); N4 uses class labels to define its strata and is
therefore tagged V as well. N1 (isotropic synthetic) lives with the planted benchmarks.

  N2a  per-cell covariance-spectrum-matched Gaussian, independent across cells (Gram-only:
       X = G diag(sqrt(lambda)) with lambda the nonzero eigenvalues of (H R H)/N)
  N2b  joint second-order surrogate: one Haar orthogonal O with O 1 = 1 shared by all cells,
       C*_l = D_l O C_l O^T D_l, D_l restoring unit diagonal
  N3   independent permutations of the cells' operators, one cell fixed (exact, no rebuild)
  N4   N3 within CIFAR class
  N5   degree-preserving double-edge-swap rewiring of each cell's pre-floor kNN graph
  N6   SEBA on Haar random q-subspaces orthogonal to the constant; SEBA on a randomly
       rotated real basis (same span, different optimiser start)
  N7   size-preserving permutation of labellings: stats.perm_null / stats.fast_ami_null
"""
from __future__ import annotations

import numpy as np
import torch

import sources as S
from stats import band, covariance_spectrum, retention, leverage, participation
from coherent import centred_cosine_gram, symmetric_balance  # noqa: E402  (path set by sources)
from seba import seba_reconstruct  # noqa: E402


# ---------------------------------------------------------------- shared summary

def spectral_summary(ops, ranks=S.RANKS, classes=None, n_low=3, eig=None, laziness='half'):
    """What every null and variant is compared on, for one block (PLAN section 3):
    the non-trivial spectrum of the block's non-lazy operator Pbar for psi_1..psi_8, gaps,
    the bottom of Pbar's spectrum (near -1 = near-bipartite alternation), participation of
    psi_1..8, span leverage concentration per q, SEBA sizes, unassigned share and clipped
    negative mass per r, and (validation only) class AMI.

    laziness declares what `ops` are (REVIEW_2 item 2): 'half' = M = (I + P)/2 (the default
    construction and every lazy variant), so lambda_P = 2 lambda_M - 1 and Pbar = 2 Mbar - I;
    'none' = the operators are already the non-lazy balanced P (e.g. the retained-diagonal
    Sinkhorn of factor D), reported natively. The convention is recorded in the output.
    Returns (summary dict, labels per r, eigvecs)."""
    from stats import ami
    if laziness not in ('half', 'none'):
        raise ValueError(laziness)
    ms = [S._n(m) for m in ops]
    mbar = np.mean(ms, 0) if len(ms) > 1 else ms[0]
    if eig is None:
        vals, vecs, einfo = S.top_eigs(mbar, S.N_VECS, project_constant=True)
    else:
        vals, vecs = eig[0][1:S.N_VECS + 1], eig[1][:, 1:S.N_VECS + 1]
        einfo = dict(method='given')
    if laziness == 'half':
        lamP = 2 * np.asarray(vals) - 1
        pbar = 2 * mbar - np.eye(len(mbar))
    else:
        lamP = np.asarray(vals, dtype=np.float64)
        pbar = mbar
    low = np.sort(np.linalg.eigvalsh(pbar))[:n_low] if len(mbar) <= 800 else _bottom(pbar, n_low)
    V = S.with_constant(vecs)
    out = dict(lambda_P=lamP.tolist(), gaps=(-np.diff(np.concatenate([[1.0], lamP]))).tolist(),
               lambda_P_bottom=np.asarray(low).tolist(), participation=participation(vecs),
               eig=einfo, laziness=laziness,
               spectrum_of='P = 2M - I (M lazy)' if laziness == 'half' else 'the operator itself (non-lazy)',
               seba={})
    labels = {}
    for r in ranks:
        lab, rot, perm, info = S.seba_sets(V, r)
        labels[r] = lab
        d = dict(info, effective_points=leverage(vecs[:, :r - 1])['effective_points'])
        if rot is not None:
            raw = seba_reconstruct(torch.as_tensor(V[:, :r]), torch.as_tensor(rot), clip=False).numpy()
            d['clipped_negative_mass'] = float(np.clip(-raw, 0, None).sum() / np.abs(raw).sum())
            d['raw_min'] = float(raw.min())
        if classes is not None:
            d['class_ami'] = ami(lab, classes)
        out['seba'][r] = d
    return out, labels, vecs


def _bottom(A, k):
    from scipy.sparse.linalg import eigsh
    vals = eigsh(A, k=k, which='SA', tol=1e-8, return_eigenvectors=False)
    return np.sort(vals)


def ops_from_grams(grams):
    return S.default_ops(grams)


# ---------------------------------------------------------------- N2a

def n2a_grams(Rs, seed=0, spectra=None):
    """Per-cell covariance-spectrum-matched Gaussian cells, independent across cells.
    X_l = G_l diag(sqrt(lambda_l)), G_l iid N(0,1) (N x rank), lambda_l the nonzero
    eigenvalues of the cell's pre-normalisation covariance (H R_l H)/N; then the real
    preprocessing (centre, row-normalise). Right feature rotations do not affect the Gram,
    so this Gram-only form is exact. Matches the estimated covariance spectrum in
    expectation only; higher moments and cross-cell dependence are destroyed."""
    g = torch.Generator().manual_seed(int(seed))
    out = []
    for i, R in enumerate(Rs):
        lam = spectra[i] if spectra is not None else covariance_spectrum(R)
        lam = lam[lam > 1e-12 * lam.max()]
        X = torch.randn(len(R), len(lam), generator=g, dtype=torch.float64) * torch.as_tensor(np.sqrt(lam))
        out.append(centred_cosine_gram(X))
    return out


# ---------------------------------------------------------------- N2b

def haar_fixing_constant(n, seed=0):
    """Haar-distributed orthogonal O on R^n with O 1 = 1: O = Hh diag(1, W) Hh, Hh the
    Householder reflection swapping e_0 and 1/sqrt(n), W Haar on O(n-1) (QR with the
    sign correction of Mezzadri 2007)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n - 1, n - 1))
    Q, Rr = np.linalg.qr(Z)
    W = Q * np.sign(np.diag(Rr))
    h = np.ones(n) / np.sqrt(n); h[0] -= 1; h /= np.linalg.norm(h)
    B = np.zeros((n, n)); B[0, 0] = 1; B[1:, 1:] = W
    # Hh B Hh with Hh = I - 2 h h^T
    BH = B - 2 * np.outer(B @ h, h)
    return BH - 2 * np.outer(h, h @ BH)


def n2b_grams(grams, seed=0, O=None):
    """Joint second-order surrogate (Gram-only, exact form of REVIEW_1 item 2).
    One O (Haar, O 1 = 1) shared by all cells: before renormalisation, O C_l O^T is the
    Gram of the rotated rows O Z_l, so the column means 1^T O Z_l = 1^T Z_l and every
    within- and cross-cell second moment Z_l^T O^T O Z_m = Z_l^T Z_m are preserved exactly.
    D_l then restores unit diagonals (row renormalisation); the mismatch it causes is
    reported: diagonal spread before D_l, and tr(C*_l C*_m)/tr(C_l C_m) - 1 (= the change
    in ||Z_l^T Z_m||_F^2) and 1^T C* 1 / 1^T C 1 - 1 (the change in ||mean row||^2) after."""
    Cs = [S._n(c).astype(np.float64) for c in grams]
    n = len(Cs[0])
    if O is None:
        O = haar_fixing_constant(n, seed)
    out, diag_lo, diag_hi = [], [], []
    for C in Cs:
        T = O @ C @ O.T
        T = .5 * (T + T.T)
        d = np.sqrt(np.clip(np.diag(T), 1e-300, None))
        diag_lo.append(float(d.min() ** 2)); diag_hi.append(float(d.max() ** 2))
        out.append(np.clip(T / d[:, None] / d[None, :], -1, 1))
    L = len(Cs)
    cross = np.array([[np.sum(out[a] * out[b]) / np.sum(Cs[a] * Cs[b]) - 1 for b in range(L)] for a in range(L)])
    mean = [float(out[a].sum() / Cs[a].sum() - 1) for a in range(L)]
    info = dict(diag_before_min=min(diag_lo), diag_before_max=max(diag_hi),
                second_moment_rel_change_max=float(np.abs(cross).max()),
                second_moment_rel_change=cross.tolist(), mean_norm2_rel_change=mean)
    return [torch.as_tensor(c) for c in out], info


# ---------------------------------------------------------------- N3 / N4

def cell_permutations(cell_keys, n, seed=0, strata=None):
    """One permutation per cell key for one null draw. The same (seed, key) always gives
    the same permutation, so draws are coherent across every block and rank that contains
    the cell (PLAN section 3, N3). strata: permute only within strata (N4)."""
    import zlib
    perms = {}
    for key in cell_keys:
        h = zlib.crc32(str(key).encode())   # stable across processes (unlike hash(str))
        rng = np.random.default_rng([int(seed), h])
        if strata is None:
            perms[key] = rng.permutation(n)
        else:
            p = np.arange(n)
            for g in np.unique(strata):
                m = np.where(strata == g)[0]
                p[m] = m[rng.permutation(len(m))]
            perms[key] = p
    return perms


def _stable_key(key):
    return str(key)


def permute_ops(ops, perms):
    """M_l -> M_l[pi_l][:, pi_l], with every pi_l re-expressed relative to the first cell
    (so the first cell is fixed and labels are in its frame). A common permutation of all
    cells only relabels the result (asserted in the tests)."""
    sigma = np.argsort(perms[0])
    out = []
    for M, p in zip(ops, perms):
        idx = p[sigma]
        out.append(S._t(M)[idx][:, idx])
    return out


def n3_ops(ops, cell_keys, seed=0, strata=None):
    keys = [_stable_key(k) for k in cell_keys]
    pd = cell_permutations(keys, ops[0].shape[0], seed, strata)
    return permute_ops(ops, [pd[k] for k in keys])


def n3_source(grams, r, *, seed=0, ops=None, cache=None, cell_keys=None, strata=None):
    """One stored N3 draw as a colour source (labels in the block's first-cell frame)."""
    ops = ops if ops is not None else S.default_ops(grams)
    keys = cell_keys if cell_keys is not None else list(range(len(ops)))
    pops = n3_ops(ops, keys, seed, strata)
    ck = ('n3', seed, tuple(map(str, keys)), None if strata is None else 'strata')
    if cache is not None and ck in cache:
        vals, vecs = cache[ck]
    else:
        ms = [S._n(m) for m in pops]
        vals, vecs, _ = S.top_eigs(np.mean(ms, 0), S.N_VECS, project_constant=True)
        if cache is not None:
            cache[ck] = (vals, vecs)
    lab, rot, perm, info = S.seba_sets(S.with_constant(vecs), r)
    return S._result(lab, vecs[:, :r - 1], vals, 'P' if strata is None else 'P+V', draw=seed, **info)


# ---------------------------------------------------------------- N5

def rewire(mask, n_swaps_per_edge=10, seed=0):
    """Degree-preserving double-edge swaps on a symmetric boolean adjacency (no loops, no
    multi-edges; rejected proposals are counted). Returns the new adjacency and the share
    of original edges that survive (a decorrelation diagnostic)."""
    rng = np.random.default_rng(seed)
    A = S._n(mask).astype(bool)
    iu = np.argwhere(np.triu(A, 1))
    edges = [tuple(e) for e in iu]
    eset = set(edges)
    m = len(edges)
    accepted = 0
    for _ in range(n_swaps_per_edge * m):
        x, y = rng.integers(m, size=2)
        if x == y:
            continue
        a, b = edges[x]; c, d = edges[y]
        if rng.random() < .5:
            c, d = d, c
        # propose (a, d), (c, b)
        if len({a, b, c, d}) < 4:
            continue
        e1 = (min(a, d), max(a, d)); e2 = (min(c, b), max(c, b))
        if e1 in eset or e2 in eset:
            continue
        eset.discard(edges[x]); eset.discard(edges[y])
        eset.add(e1); eset.add(e2)
        edges[x], edges[y] = e1, e2
        accepted += 1
    B = np.zeros_like(A)
    e = np.array(edges)
    B[e[:, 0], e[:, 1]] = True; B[e[:, 1], e[:, 0]] = True
    kept = float(len(eset & set(tuple(t) for t in iu)) / m)
    return B, dict(edges=m, accepted_swaps=accepted, original_edges_kept=kept)


def n5_ops(grams, seed=0, k=15, floor=1e-6):
    out, infos = [], []
    for i, c in enumerate(grams):
        mask = S.knn_mask(c, k)
        B, info = rewire(mask.numpy(), seed=seed * 1000 + i)
        assert (B.sum(1) == mask.numpy().sum(1)).all()
        w = np.where(B, 1.0, floor); np.fill_diagonal(w, 0)
        p = symmetric_balance(torch.log(torch.as_tensor(w)))
        out.append(S.lazy(p, .5)); infos.append(info)
    return out, infos


# ---------------------------------------------------------------- N6

def haar_subspace(n, q, seed=0):
    """Haar-random orthonormal q-frame orthogonal to the constant (QR of H G)."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, q))
    G -= G.mean(0, keepdims=True)
    return np.linalg.qr(G)[0]


def n6_haar(n, r, n_draws=1000, seed=0):
    """SEBA on [1/sqrt N, U] for Haar U: set sizes, unassigned share, clipped mass."""
    rows = []
    for t in range(n_draws):
        U = haar_subspace(n, r - 1, seed * 100003 + t)
        V = S.with_constant(U)
        lab, rot, perm, info = S.seba_sets(V, r)
        row = dict(info)
        if rot is not None:
            raw = seba_reconstruct(torch.as_tensor(V), torch.as_tensor(rot), clip=False).numpy()
            row['clipped_negative_mass'] = float(np.clip(-raw, 0, None).sum() / np.abs(raw).sum())
            row['min_size'] = int(min(info['sizes']))
        rows.append(row)
    ok = [r_ for r_ in rows if 'sizes' in r_]
    return dict(r=r, n=n, draws=n_draws, failures=n_draws - len(ok),
                unassigned=band([x['unassigned'] for x in ok]),
                min_size=band([x['min_size'] for x in ok]),
                clipped_negative_mass=band([x['clipped_negative_mass'] for x in ok]))


def n6_rotation(V, r, n_draws=20, seed=0):
    """SEBA on the real span after a random rotation O in O(r) of [psi_0..psi_{r-1}]
    (span unchanged, the optimiser's start changes). Returns per-draw labels and the
    agreement of each with the unrotated SEBA labels."""
    from stats import agreement_with_null
    base = S.seba_sets(V, r)[0]
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_draws):
        Q, Rr = np.linalg.qr(rng.standard_normal((r, r)))
        O = Q * np.sign(np.diag(Rr))
        lab = S.seba_sets(V[:, :r] @ O, r)[0]
        out.append(dict(agreement_with_null(base, lab, r, r, seed=1000 * seed + t), labels=lab))
    return out
