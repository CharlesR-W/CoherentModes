"""Streaming Grams, cell operators, running-mean blocks, spectra and every statistic of SPEC section 5; the streaming
zoom tree of section 6.

New module. Reused by import, unchanged: the standard pipeline and the cell operator (`lm_cells.unit_rows`,
`renormalise`, `SlabAccumulator`, `walk`, `lazy`), `coherent.block_modes`, `seba`, `schema_v2.comparison`, and the
sweep/m_min/components/threshold rule of `lm_tree`. What is replaced here (SPEC 8) are the functions that hold many
N x N arrays at once: `mb_cells.StateStack`/`position_grams` (-> `ViewGrams`), `block_modes` on a list of cells
(-> `BlockMean`, a running mean), and `lm_tree.node_operators`/`analyse_node`/`grow` (-> `node_operators_streaming`,
`analyse_node_streaming`, `grow_streaming`, which keep only the running mean of P in memory and spill each cell's P
to local disk for the retention pass). `test_mc.py` checks each against the original at small N.

Label discipline: kernels, operators, eigenvectors, SEBA, the two roundings and the tree cuts are functions of the
views alone. Groupings (tokens, snippets, predictions, sources, classes) enter only the functions that score an
operator or a basis after it exists (`share_values`, `kappa_stats`, `eta2`, `ami`, `jaccard`, set descriptors).
Torch float64 for every Gram, kernel, operator and eigensolve; numpy for exported statistics.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

import mc_construction as K
import lm_cells as C
import lm_tree as Z
from coherent import block_modes, symmetric_balance
from seba import seba, seba_labels
from schema_v2 import comparison


# ------------------------------------------------------------------------------------------- Grams
def unit_rows_(x, device=None):
    """`lm_cells.unit_rows` with in-place centring and scaling (identical arithmetic, one N x D float64 buffer)."""
    x = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    dev = x.device if device is None else torch.device(device)
    if x.dtype == torch.float64 and x.device == dev:
        x = x.clone()
    else:
        x = x.to(device=dev, dtype=torch.float64)
    x = x.reshape(len(x), -1)
    x -= x.mean(0, keepdim=True)
    norms = x.norm(dim=1)
    bad = np.flatnonzero((norms < C.ZERO_NORM_REL * norms.median()).cpu().numpy())
    if bad.size:
        raise C.DegenerateItems('zero-norm', bad, '(constituent or view)')
    x /= norms[:, None]
    return x


def view_gram(x, device=None):
    """`lm_cells.standard_gram` (the centred cosine Gram, clamped), memory-lean."""
    u = unit_rows_(x, device)
    g = u @ u.T
    del u
    return g.clamp_(-1.0, 1.0)


class ViewGrams:
    """The state and whole views of one snapshot, streamed (SPEC 8 item 2). Feed layers m = 1..24 in order with
    their two constituents; `layer` returns layer m's state Gram (renormalised running mean of the two constituent
    cosine Grams) and adds both into the whole-state running mean; `whole()` after the last layer. Arithmetic as
    `mb_cells.StateStack` (unclamped U U^T summed, then `lm_cells.renormalise` of the mean)."""

    def __init__(self, n, device='cpu'):
        self.n, self.device = n, device
        self.acc = C.SlabAccumulator(n, device=device)

    def layer(self, c, z):
        st = C.SlabAccumulator(self.n, device=self.device)
        for x in (c, z):
            u = unit_rows_(x, self.device)
            g = u @ u.T
            del u
            st.sum += g
            st.count += 1
            self.acc.sum += g
            self.acc.count += 1
            del g
        return st.gram()

    def whole(self):
        return self.acc.gram()


def whole_gram(constituents, n, device='cpu', members=None):
    """Whole-state Gram from an iterable of (c, z) pairs for layers 1..24 (optionally restricted to `members`, i.e.
    the view's own construction repeated on the subset)."""
    vg = ViewGrams(n if members is None else len(members), device)
    for c, z in constituents:
        if members is not None:
            c, z = c[members], z[members]
        vg.layer(c, z)
    return vg.whole()


# ------------------------------------------------------------------------------------------- operators
def duplicate_pairs(gram, tol=K.DUPLICATE_TOL):
    """(smallest off-diagonal cosine distance, number of unordered pairs with distance <= tol)."""
    d = 1.0 - C._t(gram)
    d.diagonal().fill_(float('inf'))
    m = d <= tol
    m |= m.T.clone()                                     # a pair counts once even if the Gram is not bit-symmetric
    return float(d.min()), int(m.sum()) // 2


def cell_walk(gram, k=K.K, return_mask=False, assert_distinct=True):
    """The v1 cell operator (`lm_cells.walk`: duplicate assertion, union kNN, symmetric_balance).
    `assert_distinct=False` (the reference cells, SPEC 3 amendment after pilot 1): the same kNN kernel and scaling
    without the duplicate assertion (ties break by index as everywhere); info adds `n_dup_pairs`. Zero-norm items
    (raised by the view's standard pipeline) and Sinkhorn non-convergence still fail."""
    if assert_distinct:
        return C.walk(gram, k=k, return_mask=return_mask)
    c = C._t(gram)
    md, n_dup = duplicate_pairs(c)
    logk = C.knn_logkernel(c, k)
    p, info = symmetric_balance(logk, return_info=True)
    info = dict(info, min_dist=md, n_dup_pairs=n_dup)
    if return_mask:
        info['mask'] = (logk == 0).cpu().numpy()
    return p, info


def tie_check(gram, vecs, labels, k=K.K, seeds=K.TIE_SEEDS, ranks=K.SET_R):
    """SPEC 3 tie sensitivity of a reference cell with duplicate pairs: rebuild the walk under seeded reorderings of
    the items (so the index tie-break picks other edges), undo the reordering, and compare with the original:
    a_8, a_16 and the AMI of the r = 8, 16 hard SEBA sets (items assigned in both). `labels`: {r: original hard set
    labels}. Returns (rows, tie_sensitive)."""
    n = len(gram)
    rows = []
    for seed in seeds:
        perm = np.random.RandomState(seed).permutation(n)
        inv = np.argsort(perm)
        pt = torch.as_tensor(perm, device=gram.device)
        p, info = cell_walk(gram.index_select(0, pt).index_select(1, pt), k, assert_distinct=False)
        it = torch.as_tensor(inv, device=gram.device)
        p = p.index_select(0, it).index_select(1, it)                       # back to the original item order
        _, v = spectrum(C.lazy(p))
        del p
        row = dict(seed=seed, row_err=info['row_error'])
        for r in ranks:
            row[f'a{r}'] = span_agreement(vecs, v, r)
            lab = seba_labels(seba(torch.from_numpy(v), r)).numpy().astype(np.int64)
            a = np.asarray(labels[r], dtype=np.int64)
            row[f'ami{r}'] = ami(np.where((a >= 0) & (lab >= 0), a, -1), lab)
        rows.append(row)
    vals_a = [row[f'a{r}'] for row in rows for r in ranks]
    vals_m = [row[f'ami{r}'] for row in rows for r in ranks]
    sensitive = not (all(np.isfinite(vals_a)) and all(np.isfinite(vals_m)) and min(vals_a) >= K.TIE_MIN_A and
                     min(vals_m) >= K.TIE_MIN_AMI)
    return rows, bool(sensitive)


class BlockMean:
    """Running mean of the cells' lazy walks M_c = (I + P_c)/2 (SPEC 8 item 4: blocks never stack cells)."""

    def __init__(self):
        self.sum, self.count, self.names = None, 0, []

    def add(self, p, name=None):
        m = C.lazy(p)
        if self.sum is None:
            self.sum = m
        else:
            self.sum += m
            del m
        self.count += 1
        self.names.append(name)

    def mbar(self):
        return self.sum / self.count

    def free(self):
        self.sum = None


def pbar_of(mbar):
    p = 2 * mbar
    p.diagonal().sub_(1.0)
    return p


def spectrum(mbar, n_eig=K.N_EIG):
    """Leading eigenpairs of a block's Mbar (`coherent.block_modes`, shared stationary). Returns float64 numpy
    (vals, vecs) with vecs[:, 0] the exact constant."""
    n_eig = min(n_eig, len(mbar))
    vals, vecs = block_modes([mbar], n_eig, shared_stationary=True)
    vals, vecs = vals.cpu().numpy(), vecs.cpu().numpy()
    assert abs(vals[0] - 1) < 1e-6
    assert np.abs(vecs[:, 0] - 1 / math.sqrt(len(vecs))).max() < 1e-5
    vecs[:, 0] = 1 / math.sqrt(len(vecs))
    return vals, vecs


def mus(vals):
    """mu_j = 1 - lambda_j(Pbar) = 2 (1 - lambda_j(Mbar)); mu[0] = 0 (the constant), mu[j] the j-th non-trivial."""
    return 2.0 * (1.0 - np.asarray(vals, dtype=np.float64))


def gap_ratios(mu):
    """mu_k / mu_{k-1} for k = 2..len(mu)-1 (label-free)."""
    return mu[2:] / mu[1:-1]


# ------------------------------------------------------------------------------- partitions and Upsilon
def seba_partition(s):
    """Complete partition from SEBA vectors: every item to the column with the largest value; ties and all-zero
    rows to the lowest index (numpy argmax returns the first maximum). Not `seba_labels` (which leaves items
    unassigned)."""
    return np.argmax(np.asarray(s), axis=1).astype(np.int64)


def kmeans_partition(vecs, k, seed=K.KMEANS_SEED, n_init=K.KMEANS_N_INIT):
    """Peng-Sun-Zanetti rounding: k-means with k centres on psi_1..psi_{k-1} (uniform stationary measure, so no
    degree scaling)."""
    from sklearn.cluster import KMeans
    x = np.ascontiguousarray(np.asarray(vecs)[:, 1:k])
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(x).astype(np.int64)


def part_conductance(pbar, labels, k):
    """phi(A_j) = sum_{i in A_j, l not in A_j} Pbar_il / min(|A_j|, N - |A_j|) for each part j (NaN for an empty
    part). cut(A_j) is computed directly as (H^T Pbar (1 - H))_jj, without cancellation."""
    n = len(labels)
    lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=pbar.device)
    h = torch.zeros((n, k), dtype=torch.float64, device=pbar.device)
    h[torch.arange(n, device=pbar.device), lab] = 1.0
    cut = (h * (pbar @ (1.0 - h))).sum(0).cpu().numpy()
    size = np.bincount(np.asarray(labels), minlength=k).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        phi = cut / np.minimum(size, n - size)
    phi[size == 0] = np.nan
    return phi


def upsilon_hat(mu, k, pbar, labels):
    """Upsilon_hat(k) = mu_k / max_j phi(A_j) of one complete k-partition; None if a part is empty (invalid)."""
    sizes = np.bincount(np.asarray(labels), minlength=k)
    if len(sizes) != k or sizes.min() == 0:
        return None, np.full(k, np.nan)
    phi = part_conductance(pbar, labels, k)
    rho = float(np.max(phi))
    return (float(mu[k] / rho) if rho > 0 else math.inf), phi


def outcome(ups, cap, theta, certify=True):
    """SPEC 5 reading rule: certified / excluded / undetermined; None when the rule is dropped (theta None).
    `certify` False (SPEC 8: theta not calibrated at this N, or the full-size check missing): nothing is certified,
    and k is read from the cap only (excluded / undetermined). NaN comparisons fail closed (an invalid rounding never
    certifies; a NaN cap never excludes)."""
    if theta is None:
        return None
    if certify and ups is not None and np.isfinite(ups) and ups >= theta:
        return 'certified'
    if cap is not None and np.isfinite(cap) and cap < theta:
        return 'excluded'
    return 'undetermined'


# ------------------------------------------------------------------------------------- walk statistics
def _codes(labels):
    return np.unique(np.asarray(labels), return_inverse=True)[1].astype(np.int64)


def permutation_bank(n, n_perm=K.N_PERM, seed=K.PERM_SEED):
    return C.permutation_bank(n, n_perm, seed)


def share_values(p, labels, perms):
    """S = (1/N) sum_g sum_{i != j in g} P_ij (P has a zero diagonal), then its value with group membership
    permuted: draw q gives the labels c[q]. Returns (S, S0, null (n_perm,)). Device-agnostic (dense index_add)."""
    n = len(p)
    codes = _codes(labels)
    g = int(codes.max()) + 1
    sizes = np.bincount(codes).astype(float)
    s0 = float((sizes * (sizes - 1)).sum() / (n * (n - 1)))
    dev = p.device

    def one(c):
        ct = torch.as_tensor(c, device=dev)
        b = torch.zeros((n, g), dtype=torch.float64, device=dev).index_add_(1, ct, p)
        return float(b.gather(1, ct[:, None]).sum()) / n
    s = one(codes)
    null = np.array([one(codes[q]) for q in perms])
    return s, s0, null


def excess(s, s0):
    return (s - s0) / (1 - s0)


def band(draws):
    return np.percentile(np.asarray(draws), [2.5, 97.5], axis=0)


def within_group_perms(labels, n_perm=K.N_PERM, seed=K.WITHIN_SEED):
    """Permutations q of the items that only move items within groups of equal label (items alone in their group
    keep their own). Item i receives the prediction of item q[i]. Returns (perms (n_perm, N), unique share)."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    groups = [g for g in np.split(order, bounds) if len(g) > 1]
    unique = 1.0 - sum(len(g) for g in groups) / len(labels)
    rng = np.random.default_rng(seed)
    out = np.empty((n_perm, len(labels)), np.int64)
    for k in range(n_perm):
        q = np.arange(len(labels))
        for g in groups:
            q[g] = g[rng.permutation(len(g))]
        out[k] = q
    return out, float(unique)


def kappa_stats(p, bc, snippet, within):
    """Predictive coherence (SPEC 5). Same-snippet pairs (and i = j) are excluded throughout.
    kappa = sum_{diff} P_ij BC_ij / sum_{diff} P_ij (the walk's mean BC over the pairs it crosses that are not in one
    snippet; equal to SPEC's (1/N) sum form when no same-snippet pair carries mass); kappa_raw = (1/N) sum_{diff}
    P_ij BC_ij; kappa0 = mean BC over different-snippet pairs.
    Null draws (`within`: item i receives the prediction of item q[i], q permuting within token-at-t groups): a pair
    (i, j) counts only if i, j are in different snippets AND their prediction sources q[i], q[j] are in different
    snippets, so that no draw scores shared-document predictive similarity; each draw is normalised by its own
    surviving walk mass. A draw (or the observed value) with no surviving mass is NaN, which fails every downstream
    check (fail closed)."""
    n = len(p)
    sn_np = np.asarray(snippet)
    sn = torch.as_tensor(sn_np, device=p.device)
    zero = torch.zeros((), dtype=p.dtype, device=p.device)
    diff = sn[:, None] != sn[None, :]
    w = torch.where(diff, p, zero)
    mass = float(w.sum())
    mass = mass if mass > 0 else math.nan          # no cross-snippet walk mass: kappa undefined (fails validation)
    num = float((w * bc).sum())
    k0 = float(torch.where(diff, bc, torch.zeros((), dtype=bc.dtype, device=bc.device)).sum() / diff.sum())
    draws = []
    for q in within:
        qt = torch.as_tensor(q, device=p.device)
        sq = sn[qt]
        wq = torch.where(sq[:, None] != sq[None, :], w, zero)
        mq = float(wq.sum())
        draws.append(float((wq * bc.index_select(0, qt).index_select(1, qt)).sum()) / mq if mq > 0 else math.nan)
        del wq
    del w, diff
    return dict(kappa=num / mass, kappa_raw=num / n, mass_diff=0.0 if math.isnan(mass) else mass / n, kappa0=k0,
                kappaT_draws=np.array(draws))


def kappa_summary(ks, unique_share):
    """Mean and 2.5/97.5 % band of the null draws; any non-finite draw makes kappaT and its band NaN (fail closed)."""
    d = np.asarray(ks['kappaT_draws'], dtype=np.float64)
    if len(d) == 0 or not np.isfinite(d).all():
        kt, bd = math.nan, np.array([math.nan, math.nan])
    else:
        kt, bd = float(d.mean()), band(d)
    return dict(kappa=ks['kappa'], kappa_raw=ks['kappa_raw'], mass_diff=ks['mass_diff'], kappa0=ks['kappa0'],
                kappa_ex=excess(ks['kappa'], ks['kappa0']), kappaT=kt, kappaT_band=bd,
                kappa_ex_T=excess(ks['kappa'], kt), unique_share=unique_share)


def cell_statistics(p, groupings, perms, bc=None, snippet=None, within=None, unique_share=None):
    """Walk shares for every grouping (and kappa when `bc` is given). Linear in P."""
    out = {}
    for name, lab in groupings.items():
        s, s0, null = share_values(p, lab, perms)
        out[name] = dict(S=s, S0=s0, ex=excess(s, s0), band=band(null), null=null)
    if bc is not None:
        out['kappa'] = kappa_summary(kappa_stats(p, bc, snippet, within), unique_share)
    return out


def retention(f, op):
    """f^T op f / f^T f per column of f (float64 torch on op's device)."""
    f = torch.as_tensor(np.asarray(f), dtype=torch.float64, device=op.device)
    return ((f * (op @ f)).sum(0) / f.square().sum(0)).cpu().numpy()


def centred(f):
    f = np.asarray(f, dtype=np.float64)
    return f - f.mean(0, keepdims=True)


# --------------------------------------------------------------------------------------- modes and sets
def eta2(psi, labels, perms, weights):
    """Correlation ratio of each column of psi against a grouping: between-group share of psi's variance.
    Chance (G - 1)/(N - 1) for G non-empty groups (exact under permutation of membership); draws with the labels
    permuted by each row of perms; the `weights`-weighted mean over columns (and of each draw)."""
    psi = centred(psi)
    n, m = psi.shape
    codes = _codes(labels)
    g = int(codes.max()) + 1
    tot = (psi ** 2).sum(0)
    x = torch.from_numpy(psi)

    def between(c):
        ct = torch.from_numpy(c)
        sums = torch.zeros((g, m), dtype=torch.float64).index_add_(0, ct, x).numpy()
        cnt = np.bincount(c, minlength=g).astype(float)
        keep = cnt > 0
        return (sums[keep] ** 2 / cnt[keep, None]).sum(0) / tot
    val = between(codes)
    draws = np.stack([between(codes[q]) for q in perms])
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return dict(eta2=val, chance=(g - 1) / (n - 1), band=band(draws), weighted=float(val @ w),
                weighted_band=band(draws @ w), weighted_chance=(g - 1) / (n - 1))


def mode_ends(psi, m=K.N_ENDS):
    """(n_modes, 2, m): the m items with the largest and the m with the smallest value of each column."""
    order = np.argsort(np.asarray(psi), axis=0, kind='stable')
    return np.stack([order[::-1][:m].T, order[:m].T], 1)


def ami(a, b):
    """Adjusted mutual information on the items with a >= 0 (assigned)."""
    from sklearn.metrics import adjusted_mutual_info_score
    a, b = np.asarray(a), np.asarray(b)
    keep = a >= 0
    if keep.sum() < 2:
        return math.nan
    return float(adjusted_mutual_info_score(b[keep], a[keep]))


def jaccard(a, ra, b, perms):
    """Matched Jaccard (ResNet Statistic 1) of a labelling with r_a sets against a grouping; (score, null mean/2.5/
    97.5, ceiling)."""
    codes = _codes(b)
    return C.matched_jaccard(np.asarray(a, np.int64), codes, ra, int(codes.max()) + 1, perms)


def set_descriptors(labels, soft, r, g, n_top=K.N_TOP_TOKENS, n_examples=K.N_EXAMPLES):
    """Per hard set (validation and display): size, the commonest tokens at t and t+1 (ids, counts), class make-up
    at t and t+1, source make-up, distinct snippets, the share of the commonest token at t and the number of
    distinct tokens covering 90 % of the set, and the members with the largest soft values."""
    out = []
    labels = np.asarray(labels)
    for j in range(r):
        idx = np.flatnonzero(labels == j)
        d = dict(set=j, size=int(len(idx)))
        if len(idx):
            for key, name in (('T', 'tok_t'), ('Nx', 'tok_next')):
                ids, cnt = np.unique(g[key][idx], return_counts=True)
                o = np.argsort(-cnt, kind='stable')
                d[name] = [[int(ids[k]), int(cnt[k])] for k in o[:n_top]]
                if key == 'T':
                    cs = np.cumsum(cnt[o])
                    d['top_token_share'] = float(cnt[o[0]] / len(idx))
                    d['tokens_for_90'] = int(np.searchsorted(cs, 0.9 * len(idx) - 1e-9) + 1)
            d['class_t'] = np.bincount(g['Cc'][idx], minlength=len(K.CLASSES)).tolist()
            d['class_next'] = np.bincount(g['Cn'][idx], minlength=len(K.CLASSES)).tolist()
            d['source'] = np.bincount(g['W'][idx], minlength=len(K.SOURCES)).tolist()
            d['snippets'] = int(len(np.unique(g['U'][idx])))
            d['examples'] = idx[np.argsort(-np.asarray(soft)[idx, j], kind='stable')][:n_examples].tolist()
        out.append(d)
    return out


def quantize(v):
    """int16/32767 columns with float32 scales (schema_v2 convention); v (N, m)."""
    scale = np.abs(v).max(0)
    scale[scale == 0] = 1.0
    return np.rint(32767 * v / scale).astype(np.int16).T, scale.astype(np.float32)


def dequantize(q, scale):
    return q.T.astype(np.float64) * scale / 32767


def spectral_object(mbar, groupings, perms, perms_jacc, ranks=K.R, desc_groupings=None, log=None):
    """Everything SPEC 5 asks of a block or listed cell from its Mbar: 33 eigenpairs, mu, gap ratios, the cap,
    SEBA at every r (soft, hard, rotation, reliability order), both complete roundings and Upsilon_hat(k), PN,
    eta^2 of psi_1..16 against each grouping, the ends of each psi_k, AMI and matched Jaccard of the r = 8, 16 hard
    sets, and per-set descriptors. Returns (record of numpy arrays, set descriptors {r: list}, float64 vecs)."""
    n = len(mbar)
    vals, vecs = spectrum(mbar)
    mu = mus(vals)
    pbar = pbar_of(mbar)
    ne = len(vals)
    rec = dict(vals=vals, mu=mu[1:], gap=gap_ratios(mu))
    q, scale = quantize(vecs[:, 1:])
    rec['vecs'], rec['scale'] = q, scale
    nr, rmax = len(ranks), max(ranks)
    rec.update(seba_label=np.full((nr, n), -1, np.int8), seba_part=np.full((nr, n), -1, np.int8),
               kmeans_part=np.full((nr, n), -1, np.int8), seba_rot=np.zeros((nr, rmax, rmax), np.float32),
               seba_perm=np.full((nr, rmax), -1, np.int8), seba_sizes=np.full((nr, rmax), -1, np.int32),
               phi_seba=np.full((nr, rmax), np.nan), phi_kmeans=np.full((nr, rmax), np.nan),
               ups_seba=np.full(nr, np.nan), ups_kmeans=np.full(nr, np.nan), ups=np.full(nr, np.nan),
               rounding=np.full(nr, -1, np.int8), cap=np.full(nr, np.nan), unassigned=np.zeros(nr, np.int32))
    soft = {}
    vt = torch.from_numpy(vecs)
    todo = [r for r in ranks if r <= ne - 1]
    roundings = roundings_parallel(vt, vecs, todo)
    for ri, r in enumerate(ranks):
        if r > ne - 1:
            continue
        s, rot, perm, km = roundings[r]
        s = s.numpy()
        soft[r] = s
        lab = seba_labels(torch.from_numpy(s)).numpy()
        rec['seba_label'][ri] = lab
        rec['seba_rot'][ri, :r, :r] = rot.numpy()
        rec['seba_perm'][ri, :r] = perm.numpy()
        rec['seba_sizes'][ri, :r] = np.bincount(lab[lab >= 0], minlength=r)[:r]
        rec['unassigned'][ri] = int((lab < 0).sum())
        rec['cap'][ri] = 2 * mu[r] / mu[r - 1]
        for name, part in (('seba', seba_partition(s)), ('kmeans', km)):
            u, phi = upsilon_hat(mu, r, pbar, part)
            rec[f'{name}_part'][ri] = part
            rec[f'phi_{name}'][ri, :r] = phi
            rec[f'ups_{name}'][ri] = np.nan if u is None else u
        both = [rec['ups_seba'][ri], rec['ups_kmeans'][ri]]
        if np.isfinite(both).any():
            k = int(np.nanargmax(both))
            rec['ups'][ri], rec['rounding'][ri] = both[k], k
    del pbar
    rec['pn'] = 1.0 / (vecs[:, 1:9] ** 4).sum(0)
    m = min(K.N_ETA, ne - 1)
    psi = vecs[:, 1:1 + m]
    rec['ends'] = mode_ends(psi).astype(np.int16)
    for name, lab in groupings.items():
        e = eta2(psi, lab, perms, vals[1:1 + m])
        for key, val in e.items():
            rec[f'eta2_{name}' + ('' if key == 'eta2' else '_' + key)] = np.asarray(val)
    set_desc = {}
    for r in K.SET_R:
        if r not in soft:
            continue
        lab = rec['seba_label'][ranks.index(r)].astype(np.int64)
        for name, g in groupings.items():
            rec.setdefault(f'ami_{name}', np.full(len(K.SET_R), np.nan))[K.SET_R.index(r)] = ami(lab, g)
            if name in K.FEW_GROUPS:
                score, null, ceil = jaccard(lab, r, g, perms_jacc)
                ji = K.SET_R.index(r)
                rec.setdefault(f'jacc_{name}', np.full(len(K.SET_R), np.nan))[ji] = score
                rec.setdefault(f'jacc_{name}_null', np.full((len(K.SET_R), 3), np.nan))[ji] = null
                rec.setdefault(f'jacc_{name}_ceiling', np.full(len(K.SET_R), np.nan))[ji] = ceil
        dg = groupings if desc_groupings is None else desc_groupings
        if all(k in dg for k in ('T', 'Nx', 'Cc', 'Cn', 'W', 'U')):
            set_desc[r] = set_descriptors(lab, soft[r], r, dg)
    rec['soft8'] = soft[8].astype(np.float32) if 8 in soft else None
    return rec, set_desc, vecs


class _single_threaded:
    """torch intra-op threads and every OpenMP/BLAS pool (threadpoolctl) set to 1 inside the block."""

    def __enter__(self):
        from threadpoolctl import threadpool_limits
        self.torch = torch.get_num_threads()
        torch.set_num_threads(1)
        self.limits = threadpool_limits(limits=1)
        return self

    def __exit__(self, *exc):
        self.limits.restore_original_limits()
        torch.set_num_threads(self.torch)


def roundings_parallel(vt, vecs, ranks, workers=None):
    """SEBA (`seba.seba`, return_transform) and the k-means rounding for every r, run concurrently in threads, each
    single-threaded (torch and scikit-learn release the GIL in their kernels). The same computations as a serial
    loop; only the scheduling changes (Stage A bench: SEBA runs to max_iter on unclustered spectra at N = 8,000).
    Returns {r: (s, rot, perm, kmeans labels)}."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = workers or max(1, min(len(ranks), len(os.sched_getaffinity(0))))

    def one(r):
        s, rot, perm = seba(vt, r, return_transform=True)
        return r, (s, rot, perm, kmeans_partition(vecs, r))
    with _single_threaded():
        with ThreadPoolExecutor(workers) as ex:
            return dict(ex.map(one, sorted(ranks, reverse=True)))


def span_agreement(va, vb, r):
    """a_r between two eigenbases (constant column 0 dropped; `schema_v2.comparison`)."""
    return comparison(torch.from_numpy(np.asarray(va)[:, 1:]), torch.from_numpy(np.asarray(vb)[:, 1:]), r)


# ------------------------------------------------------------------------------------------ P spill
class Spill:
    """Each cell's P (float64) on local disk, for retention passes after the block's eigenvectors exist."""

    def __init__(self, root=None):
        self._tmp = None if root is not None else tempfile.TemporaryDirectory(prefix='mc-p-')
        self.root = Path(root or self._tmp.name)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def put(self, name, p):
        path = self.root / f'{name}.npy'
        np.save(path, p.cpu().numpy() if torch.is_tensor(p) else p)
        self.files[name] = path

    def get(self, name, device='cpu'):
        return torch.from_numpy(np.load(self.files[name])).to(device)

    def drop(self, name):
        path = self.files.pop(name, None)
        if path is not None and path.exists():
            path.unlink()

    def close(self):
        for name in list(self.files):
            self.drop(name)
        if self._tmp is not None:
            self._tmp.cleanup()


# ------------------------------------------------------------------------------------- streaming tree
class VectorCell:
    """Tree cell of a vector view (the point view): the standard pipeline re-run on the rows S."""

    def __init__(self, x, device='cpu'):
        self.x, self.device = x, device

    def __len__(self):
        return len(self.x)

    def gram(self, members):
        rows = self.x[torch.as_tensor(members, device=self.x.device)] if torch.is_tensor(self.x) else \
            np.asarray(self.x)[members]
        return view_gram(rows, self.device)


class WholeCell:
    """Tree cell of the whole state: each constituent's standard pipeline re-run on S, running mean, renormalise
    (the view's own construction on S, SPEC 6). `layers` yields (c, z) for m = 1..24 (row-indexable arrays)."""

    def __init__(self, layers, n, device='cpu'):
        self.layers, self.n, self.device = layers, n, device

    def __len__(self):
        return self.n

    def gram(self, members):
        idx = np.asarray(members)
        def rows(x):
            return x[torch.as_tensor(idx, device=x.device)] if torch.is_tensor(x) else np.asarray(x[idx])
        return whole_gram(((rows(c), rows(z)) for c, z in self.layers()), len(idx), self.device)


def node_operators_streaming(cells, members, spill, device='cpu'):
    """`lm_tree.node_operators`, streamed: each cell's walk on S is added into a running mean of P and spilled to
    disk; also the union of the pre-floor kNN masks. Returns (pbar on device, mask, names)."""
    acc, union, names = None, None, []
    for ci, cell in enumerate(cells):
        g = cell.gram(members)
        p, info = cell_walk(g, return_mask=True)
        del g
        name = f'node_c{ci}'
        spill.put(name, p)
        names.append(name)
        acc = p if acc is None else acc.add_(p)
        union = info['mask'] if union is None else union | info['mask']
    return acc / len(cells), union, names


def analyse_node_streaming(pbar, members, spill, names, sources=None, mask=None, n_eig=K.N_EIG):
    """`lm_tree.analyse_node` on a running-mean Pbar (cells' P read back from disk for the cut's retention), plus the
    node's gap ratios mu_k / mu_{k-1}, k = 2..min(32, n-2). Record keys are v1's."""
    members = np.asarray(members)
    n = len(members)
    dev = pbar.device
    mbar = C.lazy(pbar)
    vals, vecs = spectrum(mbar, n_eig)
    del mbar
    psi = vecs[:, 1:4]
    pb = pbar.cpu().numpy()
    mu = mus(vals)
    mu1, mu2 = float(mu[1]), float(mu[2])
    lo = Z.m_min(n)
    best = None
    for axis in range(3):
        phi, m, theta, order, curve = Z.sweep(pb, psi[:, axis], lo)
        if m is not None and (best is None or phi < best[0]):
            best = (phi, m, theta, order, curve, axis)
    if best is None:
        raise RuntimeError(f'no admissible sweep cut for n={n}')
    phi, m, theta, order, curve, axis = best
    phi_free, *_ = Z.sweep(pb, psi[:, 0], 1)
    del pb
    side = np.zeros(n, bool)
    side[order[:m]] = True
    f = (side - side.mean())[:, None]
    ret = [float(retention(f, C.lazy(spill.get(nm, dev)))[0]) for nm in names]
    lab2 = seba_labels(seba(torch.from_numpy(vecs[:, :5]), 2)).numpy().astype(np.int64)
    j2 = K.resnet_site_analysis().matched_jaccard(lab2, np.where(side, 0, 1), 2, 2)[0]
    grid = np.rint(1 + (n - 2) * np.arange(Z.PROFILE_POINTS) / (Z.PROFILE_POINTS - 1)).astype(int)
    scale = np.abs(psi).max(0)
    rec = dict(
        n=n, vals=vals[1:5].tolist(), mu1=mu1, mu2=mu2, phi=phi, phi_free=float(phi_free),
        upsilon=float(mu2 / phi) if phi > 0 else math.inf, axis_used=axis + 1, cut_theta=theta, cut_m=m,
        psi=np.rint(127 * psi / scale).astype(np.int8).T, psi_scale=scale.astype(np.float32),
        profile=curve[grid - 1].astype(np.float32), profile_m=grid,
        cheeger_lo=mu1 / 2, cheeger_hi=math.sqrt(2 * max(mu1, 0)), m_min=lo,
        retention_cut=ret, seba2_vs_cut=float(j2), seba2_unassigned=int((lab2 < 0).sum()),
        seba2=lab2.astype(np.int8), ends=np.stack([members[order[:6]], members[order[::-1][:6]]]),
        gap=gap_ratios(mu).tolist(), mu=mu[1:].tolist(),
        children_members=(members[side], members[~side]))
    if sources is not None:
        rec['makeup'] = np.bincount(np.asarray(sources)[members], minlength=len(K.SOURCES)).tolist()
    if mask is not None:
        rec.update(Z.components(mask, lo))
    return rec


def grow_streaming(cells, device='cpu', sources=None, describe=None, items=None, max_depth=Z.MAX_DEPTH,
                   min_n=Z.MIN_SPLIT_N, spill_root=None, log=None):
    """`lm_tree.grow` with streaming node operators. `describe(members) -> dict` adds per-node descriptors (SPEC 6
    additions) to every node, leaves included. Nodes breadth-first; split iff depth < max_depth and n >= min_n."""
    n_all = len(cells[0])
    queue = [(np.arange(n_all) if items is None else np.asarray(items), -1, 0)]
    nodes = []
    spill = Spill(spill_root)
    try:
        while queue:
            members, parent, depth = queue.pop(0)
            node = dict(id=len(nodes), parent=parent, depth=depth, members=np.sort(members), n=len(members),
                        children=[])
            if sources is not None:
                node['makeup'] = np.bincount(np.asarray(sources)[members], minlength=len(K.SOURCES)).tolist()
            if describe is not None:
                node.update(describe(node['members']))
            if parent >= 0:
                nodes[parent]['children'].append(node['id'])
            if depth < max_depth and len(members) >= min_n:
                pbar, mask, names = node_operators_streaming(cells, node['members'], spill, device)
                rec = analyse_node_streaming(pbar, node['members'], spill, names, sources, mask)
                del pbar
                for nm in names:
                    spill.drop(nm)
                kids = rec.pop('children_members')
                node.update(rec)
                for kid in kids:
                    queue.append((kid, node['id'], depth + 1))
                if log:
                    log(f"  node {node['id']} n={node['n']} upsilon={rec['upsilon']:.2f}")
            nodes.append(node)
    finally:
        spill.close()
    return nodes
