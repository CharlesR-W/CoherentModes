"""Views, the per-cell operator, statistics and degenerate-item assertions (SPEC sections 1.2, 4, 5, 6).

Everything that defines a kernel, an operator or a basis here is label-free. Source labels enter only the
functions whose names say so (`walk_shares`, `mixing`, `own_source`, `source_jaccard`), and only after the
operator exists. Token identities (inputs) enter `walk_shares` for T.

Conventions: torch float64 for every Gram, kernel and operator; numpy for exported statistics.
"""
from __future__ import annotations

import numpy as np
import torch

import lm_construction as LC
from coherent import symmetric_balance
from schema_v2 import RANKS, block_record, comparison

ZERO_NORM_REL = 1e-10
DUPLICATE_TOL = 1e-9


class DegenerateItems(RuntimeError):
    """A cell violates the duplicate or zero-norm policy (SPEC 5). The unit fails; nothing is patched."""

    def __init__(self, kind, items, detail=''):
        self.kind, self.items = kind, list(map(int, items))
        super().__init__(f'{kind}: items {self.items[:20]} {detail}')


# ------------------------------------------------------------------------------------------------- views
def _t(x):
    return x.double() if torch.is_tensor(x) else torch.from_numpy(np.array(x, dtype=np.float64))


def zero_norm_items(x):
    """Items whose centred vector has norm below 1e-10 times the median centred norm (THEORY [P2])."""
    x = _t(x).reshape(len(x), -1)
    norms = (x - x.mean(0, keepdim=True)).norm(dim=1)
    return np.flatnonzero((norms < ZERO_NORM_REL * norms.median()).cpu().numpy())


def unit_rows(x):
    """Standard pipeline, first half: centre over the items, row-normalise. Raises on a zero centred vector."""
    x = _t(x).reshape(len(x), -1)
    x = x - x.mean(0, keepdim=True)
    norms = x.norm(dim=1)
    bad = np.flatnonzero((norms < ZERO_NORM_REL * norms.median()).cpu().numpy())
    if bad.size:
        raise DegenerateItems('zero-norm', bad, '(remove run-wide per SPEC 5 and rerun)')
    return x / norms[:, None]


def standard_gram(x):
    """Standard pipeline: the centred cosine Gram C (float64, clamped to [-1, 1])."""
    u = unit_rows(x)
    return (u @ u.T).clamp(-1.0, 1.0)


def renormalise(g):
    """Double-centre a Gram and rescale it to unit diagonal: the standard pipeline applied in Gram form."""
    g = _t(g)
    g = g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()
    d = g.diagonal().clamp_min(0).sqrt()
    bad = np.flatnonzero((d < ZERO_NORM_REL * d.median()).cpu().numpy())
    if bad.size:
        raise DegenerateItems('zero-norm', bad, '(slab)')
    return (g / d[:, None] / d[None, :]).clamp(-1.0, 1.0)


class SlabAccumulator:
    """Slab view (SPEC 4): Gram of t^{-1/2}[U_1..U_t] = mean_u U_u U_u^T, then the standard pipeline.

    Feed positions u = 1, 2, ... in order; `gram(t)` is valid after position t was added. All positions enter,
    not only the sampled ones. `device` lets U1 run it on the GPU.
    """

    def __init__(self, n, device='cpu'):
        self.sum = torch.zeros((n, n), dtype=torch.float64, device=device)
        self.count = 0

    def add(self, x_u):
        u = unit_rows(x_u.to(self.sum.device))
        self.sum += u @ u.T
        self.count += 1

    def gram(self):
        return renormalise(self.sum / self.count)


def slab_gram(xs):
    """Slab Gram from the per-position activations xs[0..t-1] (positions 1..t)."""
    acc = SlabAccumulator(len(xs[0]))
    for x in xs:
        acc.add(_t(x))
    return acc.gram()


def bag_vectors(tokens, embedding, positions=LC.POSITIONS):
    """Bag view inputs: (1/t) sum_{u<=t} E[x_u], float64, summed sequentially in sorted token-id order.

    tokens: (N, SEQ) snippet tokens 1..SEQ (no EOT). Returns (len(positions), N, D) float64. The value depends on
    the multiset of tokens 1..t only, bit for bit [thm]; C2 checks this on probes A and S.
    """
    tokens = np.asarray(tokens)
    e = np.asarray(embedding, dtype=np.float64)
    out = np.empty((len(positions), len(tokens), e.shape[1]))
    for i, row in enumerate(tokens):
        for pi, t in enumerate(positions):
            out[pi, i] = np.cumsum(e[np.sort(row[:t], kind='stable')], axis=0)[-1] / t
    return out


def rogue_shares(x):
    """Top-1 and top-5 coordinate shares of the centred variance (hazard table)."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    var = ((x - x.mean(0)) ** 2).sum(0)
    var = np.sort(var)[::-1] / var.sum()
    return float(var[0]), float(var[:5].sum())


# --------------------------------------------------------------------------------------------- operator
def min_offdiag_distance(c):
    d = 1.0 - _t(c)
    d.diagonal().fill_(float('inf'))
    return float(d.min())


def knn_logkernel(c, k=LC.K, floor=LC.FLOOR):
    """The ResNet default rule: union of each item's k nearest (ties by lower index), weight 1, floor elsewhere."""
    d = 1.0 - _t(c)
    d.diagonal().fill_(float('inf'))
    idx = torch.argsort(d, dim=1, stable=True)[:, :k]
    mask = torch.zeros_like(d, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    mask = mask | mask.T
    w = torch.where(mask, torch.ones_like(d), torch.full_like(d, floor))
    w.diagonal().fill_(0.0)
    return torch.log(w)


def walk(c, k=LC.K, return_mask=False):
    """Cell operator: duplicate assertion, kNN kernel, zero-diagonal symmetric scaling. Returns (P, info)."""
    c = _t(c)
    md = min_offdiag_distance(c)
    if not md > DUPLICATE_TOL:
        d = 1.0 - c
        d.diagonal().fill_(float('inf'))
        i, j = np.unravel_index(int(torch.argmin(d)), d.shape)
        raise DegenerateItems('duplicate', [i, j], f'(min cosine distance {md:.3g})')
    logk = knn_logkernel(c, k)
    p, info = symmetric_balance(logk, return_info=True)
    info = dict(info, min_dist=md)
    if return_mask:
        info['mask'] = (logk == 0).cpu().numpy()        # the pre-floor kNN edges (weight 1), symmetric
    return p, info


def lazy(p):
    m = p * 0.5
    m.diagonal().add_(0.5)
    return m


# ------------------------------------------------------------------------------------------- statistics
def permutation_bank(n, n_perm=LC.N_PERM, seed=LC.PERM_SEED):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(n_perm)])


def same_group_pairs(labels):
    labels = np.asarray(labels)
    i, j = [], []
    for g in np.unique(labels):
        idx = np.flatnonzero(labels == g)
        if len(idx) > 1:
            a, b = np.meshgrid(idx, idx, indexing='ij')
            keep = a != b
            i.append(a[keep]); j.append(b[keep])
    if not i:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(i), np.concatenate(j)


def _np(p):
    return np.asarray(p.cpu() if torch.is_tensor(p) else p, dtype=np.float64)


def walk_share(p, labels, perms):
    """S(G) = (1/N) sum_g sum_{i != j in g} P_ij, its exact chance S0, and its value under each permutation of
    group membership. Permuting membership by pi equals reading P at (pi(i), pi(j)) over the fixed pairs."""
    p = _np(p)
    n = len(p)
    labels = np.asarray(labels)
    sizes = np.unique(labels, return_counts=True)[1].astype(float)
    s0 = float((sizes * (sizes - 1)).sum() / (n * (n - 1)))
    i, j = same_group_pairs(labels)
    s = float(p[i, j].sum() / n)
    null = np.array([p[q[i], q[j]].sum() / n for q in perms])
    return s, s0, null


def excess(s, s0):
    return (s - s0) / (1 - s0)


def onehot(labels, g):
    out = np.zeros((len(labels), g))
    out[np.arange(len(labels)), labels] = 1
    return out


def mixing(p, sources, perms, g=len(LC.SOURCES)):
    """Pair mixing X(A,B) = (N-1)/(|A||B|) sum_{A x B} P and the source share W = (1/N) sum_g (H^T P H)_gg,
    with their values under each permutation of membership. Returns X, X_null, W, W0, W_null."""
    p = _np(p)
    n = len(p)
    sources = np.asarray(sources)
    h = onehot(sources, g)
    size = h.sum(0)
    scale = (n - 1) / np.outer(size, size)
    block = h.T @ p @ h
    hs = np.concatenate([onehot(sources[q], g) for q in perms], 1)
    ph = p @ hs
    blocks = np.einsum('nag,nah->agh', hs.reshape(n, len(perms), g), ph.reshape(n, len(perms), g))
    w0 = float((size * (size - 1)).sum() / (n * (n - 1)))
    return block * scale, blocks * scale, float(np.trace(block) / n), w0, np.trace(blocks, axis1=1, axis2=2) / n


def own_source(p, sources, g=len(LC.SOURCES)):
    """Per item: walk mass by source (rows of P H); the own-source column is the own-source share."""
    by = _np(p) @ onehot(np.asarray(sources), g)
    return by[np.arange(len(by)), sources], by


def u8(v):
    return np.rint(np.clip(np.asarray(v), 0, 1) * 255).astype(np.uint8)


def raw_statistics(p, tokens_t, sources, perms, follow):
    """Per-cell walk statistics with their permutation draws (one shared bank). Linear in P."""
    t, t0, t_null = walk_share(p, tokens_t, perms)
    x, x_null, w, w0, w_null = mixing(p, sources, perms)
    own, by = own_source(p, sources)
    return dict(T=t, T0=t0, T_null=t_null, W=w, W0=w0, W_null=w_null, X=x, X_null=x_null,
                own=own, follow=by[np.asarray(follow)])


def export_statistics(raws):
    """Block (or cell, for one raw) statistics: every value and every permutation draw is the equal-weight mean
    over the block's cells (SPEC 1.2: linear in P), then 2.5/97.5 % bands of the averaged draws."""
    mean = {k: np.mean([r[k] for r in raws], axis=0) for k in raws[0]}
    band = lambda a: np.percentile(a, [2.5, 97.5], axis=0)
    return dict(T=mean['T'], T0=mean['T0'], T_band=band(mean['T_null']),
                W=mean['W'], W0=mean['W0'], W_band=band(mean['W_null']),
                X=mean['X'], X_band=np.moveaxis(band(mean['X_null']), 0, -1),
                own_src=u8(mean['own']), follow_mass=u8(mean['follow']))


def participation(vectors):
    """PN(psi_1..psi_8) = 1 / sum psi^4 of the unit eigenvectors."""
    v = vectors[:, 1:9].double()
    return (1.0 / v.pow(4).sum(0)).cpu().numpy()


# ----------------------------------------------------------------------------------------- set records
def reliable_soft_sets(vectors, rec):
    """Soft SEBA sets from the exported transform (as the page will reconstruct them)."""
    from coherent import seba_reconstruct
    out = []
    for ri, r in enumerate(RANKS):
        out.append(seba_reconstruct(vectors[:, :r].cpu(), torch.from_numpy(rec['seba_rot'][ri, :r, :r]).double(),
                                    torch.from_numpy(rec['seba_perm'][ri, :r].astype('i8'))).numpy())
    return out


def top_members(vectors, rec, m=LC.TOP_MEMBERS):
    """(5, 8, m) int16: per rank and set, members of the hard set by decreasing soft value; -1 padded."""
    out = np.full((len(RANKS), 8, m), -1, np.int16)
    for ri, (r, s) in enumerate(zip(RANKS, reliable_soft_sets(vectors, rec))):
        lab = rec['seba_label'][ri]
        for j in range(r):
            idx = np.flatnonzero(lab == j)
            idx = idx[np.argsort(-s[idx, j], kind='stable')][:m]
            out[ri, j, :len(idx)] = idx
    return out


def matched_jaccard(a, b, ra, rb, perms):
    """The ResNet Statistic 1 (cifar_coherent/site/analysis.py): Hungarian mean, null over permutations of a,
    reported as (score, (null mean, 2.5 %, 97.5 %), size ceiling)."""
    A = LC.resnet_site_analysis()
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    score, _, null = A.matched_jaccard(a, b, ra, rb, perms)
    return score, np.array(A.null_summary(null)), A.size_ceiling(a, b, ra, rb)


def source_jaccard(rec, sources, perms):
    """Matched Jaccard of the r = 2 and r = 4 sets against sources (validation)."""
    out = dict(src_jacc=np.zeros(2, np.float32), src_jacc_null=np.zeros((2, 3), np.float32),
               src_jacc_ceiling=np.zeros(2, np.float32))
    for k, r in enumerate(LC.SEBA_R_SHOWN):
        s, null, ceil = matched_jaccard(rec['seba_label'][RANKS.index(r)], sources, r, len(LC.SOURCES), perms)
        out['src_jacc'][k], out['src_jacc_null'][k], out['src_jacc_ceiling'][k] = s, null, ceil
    return out


def spectral_record(ms_axis, index_range, depth_axis=None, depth_index=None, sources=None, perms=None):
    """block_record over ms_axis[a..b] with retention along that axis; optional retention along a second axis
    (same functions, `_depth` keys; NaN when absent); plus PN, the smallest r = 4 set, top members and the
    source Jaccard. Returns (record, float64 vectors)."""
    a, b = index_range
    rec, vectors = block_record(ms_axis, a, b)
    vectors = vectors.clone()
    nc = len(ms_axis)
    if depth_axis is not None:
        drec, _ = block_record(depth_axis, depth_index, depth_index, eigen=(torch.as_tensor(rec['vals']).double(),
                                                                              vectors.clone()))
        assert np.array_equal(drec['seba_label'], rec['seba_label'])
        rec['retention_psi_depth'] = drec['retention_psi']
        rec['retention_seba_depth'] = drec['retention_seba']
    else:
        rec['retention_psi_depth'] = np.full((8, nc), np.nan, np.float32)
        rec['retention_seba_depth'] = np.full((len(RANKS), 8, nc), np.nan, np.float32)
    rec['pn'] = participation(vectors).astype(np.float32)
    lab4 = rec['seba_label'][RANKS.index(4)]
    rec['min_set_r4'] = np.int16(min(int((lab4 == j).sum()) for j in range(4)))
    rec['top_members'] = top_members(vectors, rec)
    if sources is not None:
        rec.update(source_jaccard(rec, sources, perms))
    return rec, vectors


def span_agreement(va, vb, r):
    """a_r: mean cos^2 of principal angles between span(psi_1..psi_r) of two cells, constant projected out."""
    return comparison(va[:, 1:].cpu(), vb[:, 1:].cpu(), r)
