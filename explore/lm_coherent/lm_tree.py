"""Zoom recursion (SPEC section 7): rebuild on the subset, sweep cut on the block's averaged walk, repeat.

Label-free: the cut is chosen from the node's own averaged operator. Source labels enter only `makeup`, and
SEBA only the side-by-side comparison `seba2_vs_cut`; neither is used to make or keep a cut.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import lm_construction as LC
from coherent import block_modes, seba, seba_labels
from lm_cells import lazy, standard_gram, walk

MAX_DEPTH = 3
MIN_SPLIT_N = 64
PROFILE_POINTS = 64
READING_THRESHOLD = 3.0


def m_min(n):
    return max(16, math.ceil(0.05 * n))


def cell_gram(x, members):
    """Operator provider: the cell's cosine Gram rebuilt on the items S. A vector cell (an array: the point view) is
    re-centred over S by the standard pipeline. A Gram-based cell (any object with `subset_gram(members)`, e.g.
    the Mamba study's stack view) repeats its own construction on S. Added 2026-11-12 for explore/mamba_coherent;
    vector cells take exactly the previous code path."""
    if hasattr(x, 'subset_gram'):
        return x.subset_gram(members)
    return standard_gram(np.asarray(x)[members])


def node_operators(cells, members):
    """Rebuild each cell's zero-diagonal walk on the items S: re-centre over S, kNN among S, Sinkhorn. Also returns
    the union over cells of the PRE-FLOOR kNN masks (the graph the walk is built on, before the 1e-6 floor)."""
    out, union = [], None
    for x in cells:
        p, info = walk(cell_gram(x, members), return_mask=True)
        out.append(p)
        union = info['mask'] if union is None else union | info['mask']
    return out, union


def node_walks(cells, members):
    return node_operators(cells, members)[0]


def components(mask, lo):
    """Connected components of the node's union kNN graph (label-free diagnostic, SPEC 9). `flag` is set when
    there are >= 2 components and some union of whole components leaves both sides >= lo, i.e. a valid sweep cut
    exists that crosses no kNN edge. Disconnection means geometric proximity BETWEEN components is lost to the
    kernel; it does not make every grouping of components equal in conductance (sizes, internal structure and
    Sinkhorn scaling still matter). It changes no cut."""
    from scipy.sparse.csgraph import connected_components
    k, lab = connected_components(mask, directed=False)
    sizes = sorted(np.bincount(lab).tolist(), reverse=True)
    n = len(lab)
    reach = np.zeros(n + 1, bool); reach[0] = True
    for s in sizes:                                     # subset sums of component sizes
        reach[s:] |= reach[:n + 1 - s].copy()
    room = bool(reach[lo:n - lo + 1].any()) if n - lo >= lo else False
    return dict(n_components=int(k), component_sizes=sizes, disconnected=bool(k >= 2 and room))


def cut_profile(pbar, psi):
    """Cut sizes of every prefix {psi > theta} in decreasing-psi order. Returns (order, cut[m-1], valid[m-1])
    for m = 1..n-1; valid marks m where psi strictly drops (the sweep never separates equal values).

    Uses cut(m) = sum_{k<m} (upper_k - lower_k) with upper_k = sum_{j>k} P_kj, lower_k = sum_{j<k} P_kj in sweep
    order, which needs only symmetry of P (no reliance on exact row sums)."""
    psi = np.asarray(psi, dtype=np.float64)
    n = len(psi)
    order = np.argsort(-psi, kind='stable')
    pp = pbar[np.ix_(order, order)]
    upper = np.triu(pp, 1).sum(1)
    lower = np.tril(pp, -1).sum(1)
    cut = np.cumsum(upper - lower)[:n - 1]
    s = psi[order]
    valid = s[:-1] > s[1:]
    return order, cut, valid


def sweep(pbar, psi, lo):
    """Lowest-conductance level-set cut with both sides >= lo. Returns (phi, m, theta, order, phi_curve)."""
    order, cut, valid = cut_profile(pbar, psi)
    n = len(psi)
    m = np.arange(1, n)
    phi = cut / np.minimum(m, n - m)
    ok = valid & (m >= lo) & (n - m >= lo)
    if not ok.any():
        return math.inf, None, None, order, phi
    k = int(np.flatnonzero(ok)[np.argmin(phi[ok])])
    s = np.asarray(psi)[order]
    return float(phi[k]), int(m[k]), float((s[k] + s[k + 1]) / 2), order, phi


def analyse_node(ps, members, sources=None, mask=None):
    """Full node record (SPEC 7.1) from the rebuilt walks `ps` on the items `members` (global ids); `mask` is the
    union pre-floor kNN graph from `node_operators` (adds the component diagnostic)."""
    members = np.asarray(members)
    n = len(members)
    ms = [lazy(p) for p in ps]
    vals, vecs = block_modes(ms, 5, shared_stationary=True)
    vals = vals.cpu().numpy()
    psi = vecs[:, 1:4].cpu().numpy()
    pbar = torch.stack(ps).mean(0).cpu().numpy()
    mu = 2 * (1 - vals)
    mu1, mu2 = float(mu[1]), float(mu[2])
    lo = m_min(n)
    best = None
    for axis in range(3):
        phi, m, theta, order, curve = sweep(pbar, psi[:, axis], lo)
        if m is not None and (best is None or phi < best[0]):
            best = (phi, m, theta, order, curve, axis)
    if best is None:
        raise RuntimeError(f'no admissible sweep cut for n={n}')
    phi, m, theta, order, curve, axis = best
    phi_free, *_ = sweep(pbar, psi[:, 0], 1)
    side = np.zeros(n, bool)
    side[order[:m]] = True                  # child 0: contains the item with the largest winning psi
    f = side - side.mean()
    retention = [float(f @ (mm.cpu().numpy() @ f) / (f @ f)) for mm in ms]
    lab2 = seba_labels(seba(vecs.cpu(), 2)).numpy().astype(np.int64)
    j2 = LC.resnet_site_analysis().matched_jaccard(lab2, np.where(side, 0, 1), 2, 2)[0]
    grid = np.rint(1 + (n - 2) * np.arange(PROFILE_POINTS) / (PROFILE_POINTS - 1)).astype(int)
    scale = np.abs(psi).max(0)
    rec = dict(
        n=n, vals=vals[1:5].tolist(), mu1=mu1, mu2=mu2, phi=phi, phi_free=float(phi_free),
        upsilon=float(mu2 / phi) if phi > 0 else math.inf, axis_used=axis + 1, cut_theta=theta, cut_m=m,
        psi=np.rint(127 * psi / scale).astype(np.int8).T, psi_scale=scale.astype(np.float32),
        profile=curve[grid - 1].astype(np.float32), profile_m=grid,
        cheeger_lo=mu1 / 2, cheeger_hi=math.sqrt(2 * max(mu1, 0)), m_min=lo,
        retention_cut=retention, seba2_vs_cut=float(j2), seba2_unassigned=int((lab2 < 0).sum()),
        seba2=lab2.astype(np.int8),
        ends=np.stack([members[order[:6]], members[order[::-1][:6]]]),
        children_members=(members[side], members[~side]))
    if sources is not None:
        rec['makeup'] = np.bincount(np.asarray(sources)[members], minlength=len(LC.SOURCES)).tolist()
    if mask is not None:
        rec.update(components(mask, lo))
    return rec


def grow(cells, sources=None, items=None, max_depth=MAX_DEPTH, min_n=MIN_SPLIT_N):
    """The recursion of SPEC 7.2 on the block given by `cells` (each (N, D), the point view of one cell; or a
    Gram-based cell with `__len__` and `subset_gram`, see `cell_gram`).

    Nodes are numbered breadth-first; split iff depth < max_depth and n >= min_n (size and depth only)."""
    n_all = len(cells[0])
    queue = [(np.arange(n_all) if items is None else np.asarray(items), -1, 0)]
    nodes = []
    while queue:
        members, parent, depth = queue.pop(0)
        node = dict(id=len(nodes), parent=parent, depth=depth, members=np.sort(members), n=len(members),
                    children=[])
        if sources is not None:
            node['makeup'] = np.bincount(np.asarray(sources)[members], minlength=len(LC.SOURCES)).tolist()
        if parent >= 0:
            nodes[parent]['children'].append(node['id'])
        if depth < max_depth and len(members) >= min_n:
            ps, mask = node_operators(cells, node['members'])
            rec = analyse_node(ps, node['members'], sources, mask)
            kids = rec.pop('children_members')
            node.update(rec)
            for kid in kids:
                queue.append((kid, node['id'], depth + 1))
        nodes.append(node)
    return nodes


def threshold_rule(planted, structureless, fixed=READING_THRESHOLD):
    """SPEC 7.1 step 3: 3 if it separates; else the geometric mean of the two extremes; else None (rule dropped)."""
    lo, hi = min(planted), max(structureless)
    if lo >= fixed > hi:
        return fixed, 'fixed'
    if lo > hi:
        return float(math.sqrt(lo * hi)), 'geometric-mean'
    return None, 'dropped'


def to_json(nodes):
    """trees.json node list (SPEC 10.1). Leaves carry only id/parent/depth/members/n/makeup/children."""
    out = []
    for node in nodes:
        row = {}
        for key, value in node.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                value = value.item()
            row[key] = value
        out.append(row)
    return out
