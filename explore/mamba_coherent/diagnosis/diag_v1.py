"""Diagnosis of the v1 Mamba study (D1-D4). Read-only on `../data/m130`; writes `diag_v1.json` beside this file.

D1  SEBA sets of the depth and reading blocks scored against source, token-context class (token_classes.py) and
    token identity: AMI over assigned items, and the study's matched Jaccard with its permutation null and ceiling.
D3  Source mixing ratios X at two cells (cells_trained.npz) and the matching reading blocks.
D2/D4  Every internal node of trees T1-T4 (trees.json): balance, Upsilon, mu1, mu2, phi, k/n, children's make-up.
D2 (optional)  Root averaged walks of T1 and T4 rebuilt from the Modal-volume inputs in $CM_MB_VOL (default
    /var/tmp/cm-mb/vol: act_trained.npy, stack_trained.npy) with the study's own functions; k-way
    Upsilon^(k) = mu_k / rho^(k) for SEBA partitions (k = 2..8) and, for k = 5, the source partition (label-derived:
    an illustration, not a construction output). Skipped, with the reason recorded, if the inputs are absent.

Token classes and source labels are validation labels only; no operator, kernel, basis or cut here uses them.
Run: TMPDIR=/var/tmp/cm-mb uv run -q --with numpy --with scipy --with torch --with scikit-learn python -I diag_v1.py
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # leave no __pycache__ beside the reused study modules
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '3')

import collections  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (str(HERE), str(STUDY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import mb_construction as MB  # noqa: E402  (puts ../lm_coherent and ../cifar_coherent on sys.path)
import lm_cells as C  # noqa: E402
from coherent import block_modes, seba, seba_labels, seba_reconstruct  # noqa: E402
from token_classes import CLASSES, token_class  # noqa: E402

torch.set_num_threads(3)
DATA = STUDY / 'data' / 'm130'
VOL = Path(os.environ.get('CM_MB_VOL', '/var/tmp/cm-mb/vol'))
OUT = HERE / 'diag_v1.json'
RANKS = [2, 3, 4, 6, 8]
POS = MB.POSITIONS
LAYERS = MB.LAYERS
SOURCES = MB.LC.SOURCES
K = MB.LC.K


def rnd(x, d=4):
    """JSON-friendly rounding of nested numpy/python values."""
    if isinstance(x, dict):
        return {str(k): rnd(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), d)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if not math.isfinite(x) else round(x, d)
    return x


# ------------------------------------------------------------------------------------------------- inputs
probe = json.loads((DATA / 'probe.json').read_text())
items = probe['items']
N = len(items)
src = np.array([SOURCES.index(it['source']) for it in items])
tok_id = np.array([it['tokens'] for it in items], dtype=np.int64)            # (N, 128); position t -> [:, t-1]
tok_str = [it['token_str'] for it in items]
assert tok_id.shape == (N, 128) and all(len(s) == 128 for s in tok_str)
cls_idx = {c: i for i, c in enumerate(CLASSES)}
tok_cls = np.array([[cls_idx[token_class(s)] for s in row] for row in tok_str])  # (N, 128)
blocks = np.load(DATA / 'blocks_trained.npz')
cells = np.load(DATA / 'cells_trained.npz')
trees = {t['id']: t for t in json.loads((DATA / 'trees.json').read_text()) if t['id'] in ('T1', 'T2', 'T3', 'T4')}
perms = C.permutation_bank(N)


def cur(t):
    return tok_cls[:, t - 1], tok_id[:, t - 1]


def nxt(t):
    return (tok_cls[:, t], tok_id[:, t]) if t < 128 else (None, None)


def compact(g):
    """Relabel a grouping to 0..r-1 over its present values (-1 kept)."""
    vals = np.unique(g[g >= 0])
    lut = {v: i for i, v in enumerate(vals)}
    return np.array([lut[v] if v >= 0 else -1 for v in g], dtype=np.int64), len(vals)


def score(lab, r, g, jaccard=True):
    """AMI over items assigned in both, and the matched Jaccard (score, null mean/2.5/97.5 %, ceiling)."""
    lab = np.asarray(lab, dtype=np.int64)
    ok = (lab >= 0) & (g >= 0)
    out = dict(ami=float(ami(g[ok], lab[ok])), n=int(ok.sum()))
    if jaccard:
        gc, rb = compact(g)
        j, null, ceil = C.matched_jaccard(lab, gc, r, rb, perms)
        out.update(jacc=float(j), null=[float(v) for v in null], ceil=float(ceil),
                   jacc_rel=float((j - null[0]) / (ceil - null[0])) if ceil > null[0] else float('nan'))
    return out


def within_source_ami(lab, g):
    """Mean over sources (weighted by assigned count) of AMI(g, lab) restricted to one source: token-class
    information in the sets that is not carried by source."""
    vals, w = [], []
    for s in range(len(SOURCES)):
        ok = (lab >= 0) & (g >= 0) & (src == s)
        if ok.sum() > 1 and len(np.unique(lab[ok])) > 1 and len(np.unique(g[ok])) > 1:
            vals.append(ami(g[ok], lab[ok])); w.append(ok.sum())
        elif ok.sum() > 1:
            vals.append(0.0); w.append(ok.sum())
    return float(np.average(vals, weights=w)) if w else float('nan')


def within_class_ami(lab, g):
    """Mean over token classes (weighted) of AMI(source, lab) restricted to one class."""
    vals, w = [], []
    for c in np.unique(g):
        ok = (lab >= 0) & (g == c)
        if ok.sum() > 1 and len(np.unique(lab[ok])) > 1 and len(np.unique(src[ok])) > 1:
            vals.append(ami(src[ok], lab[ok])); w.append(ok.sum())
        elif ok.sum() > 1:
            vals.append(0.0); w.append(ok.sum())
    return float(np.average(vals, weights=w)) if w else float('nan')


# ------------------------------------------------------------------------------------------------- D1
def d1():
    t0 = time.time()
    out = dict(classes=CLASSES, reference={}, depth={}, reading={}, depth_sets_r8={})
    for t in POS:                       # how much source and class already share (no sets involved)
        c, i = cur(t)
        ref = dict(ami_source_cur_class=float(ami(src, c)),
                   cur_class_counts={CLASSES[k]: int(v) for k, v in zip(*np.unique(c, return_counts=True))})
        nc, _ = nxt(t)
        if nc is not None:
            ref['ami_source_next_class'] = float(ami(src, nc))
        out['reference'][t] = ref
    for pi, t in enumerate(POS):
        c, i = cur(t)
        nc, ni = nxt(t)
        per_r = {}
        for ri, r in enumerate(RANKS):
            lab = blocks['depth_point_seba_label'][pi, ri].astype(np.int64)
            row = dict(n_assigned=int((lab >= 0).sum()), source=score(lab, r, src), cur_class=score(lab, r, c),
                       cur_id=score(lab, r, i, jaccard=False),
                       cur_class_within_source_ami=within_source_ami(lab, c),
                       source_within_cur_class_ami=within_class_ami(lab, c))
            if nc is not None:
                row.update(next_class=score(lab, r, nc), next_id=score(lab, r, ni, jaccard=False),
                           next_class_within_source_ami=within_source_ami(lab, nc))
            per_r[r] = row
        out['depth'][t] = per_r
        out['depth_sets_r8'][t] = set_table(blocks['depth_point_seba_label'][pi, RANKS.index(8)].astype(np.int64), t)
    for view in MB.VIEWS:
        for li, layer in enumerate(LAYERS):
            per_r = {}
            for ri, r in enumerate(RANKS):
                lab = blocks[f'reading_{view}_seba_label'][li, ri].astype(np.int64)
                row = dict(n_assigned=int((lab >= 0).sum()), source=score(lab, r, src), cur_class_by_t={},
                           next_class_by_t={}, cur_id_ami_by_t={})
                for t in POS:
                    c, i = cur(t)
                    row['cur_class_by_t'][t] = score(lab, r, c)
                    row['cur_id_ami_by_t'][t] = score(lab, r, i, jaccard=False)['ami']
                    nc, _ = nxt(t)
                    if nc is not None:
                        row['next_class_by_t'][t] = score(lab, r, nc)
                per_r[r] = row
            out['reading'][f'{view}_L{layer}'] = per_r
    out['seconds'] = time.time() - t0
    return out


def set_table(lab, t):
    rows = []
    for j in range(8):
        m = np.flatnonzero(lab == j)
        if not len(m):
            rows.append(dict(set=j, size=0)); continue
        cc = collections.Counter(tok_cls[m, t - 1].tolist())
        maj, cnt = cc.most_common(1)[0]
        row = dict(set=j, size=int(len(m)), makeup=np.bincount(src[m], minlength=5).tolist(),
                   top_cur=collections.Counter(tok_str[k][t - 1] for k in m).most_common(6),
                   majority_cur_class=CLASSES[maj], majority_cur_share=cnt / len(m))
        if t < 128:
            nc = collections.Counter(tok_cls[m, t].tolist()).most_common(1)[0]
            row.update(top_next=collections.Counter(tok_str[k][t] for k in m).most_common(6),
                       majority_next_class=CLASSES[nc[0]], majority_next_share=nc[1] / len(m))
        rows.append(row)
    rows.append(dict(set=-1, size=int((lab < 0).sum())))
    return rows


# ------------------------------------------------------------------------------------------------- D3
def offdiag_ranked(x):
    pairs = [(SOURCES[a], SOURCES[b], float(x[a, b])) for a in range(5) for b in range(a + 1, 5)]
    return sorted(pairs, key=lambda p: -p[2])


def d3():
    vi = {v: i for i, v in enumerate(MB.VIEWS)}
    li, pi = LAYERS.index, POS.index
    out = dict(sources=SOURCES, note='X = 1: walk ignores the pair; < 1: walk avoids crossing')
    for name, view, layer, t in (('stack_L24_t128', 'stack', 24, 128), ('point_L16_t128', 'point', 16, 128),
                                 ('point_L16_t16', 'point', 16, 16), ('point_L24_t128', 'point', 24, 128)):
        x = cells['X'][vi[view], li(layer), pi(t)]
        band = cells['X_band'][vi[view], li(layer), pi(t)]
        out[name] = dict(X=x, null_band_hi=band[..., 1], ranked_pairs=offdiag_ranked(x))
    for name, key, layer in (('reading_stack_L24', 'reading_stack_X', 24), ('reading_point_L16', 'reading_point_X', 16)):
        x = blocks[key][li(layer)]
        out[name] = dict(X=x, ranked_pairs=offdiag_ranked(x))
    return out


# ---------------------------------------------------------------------------------------------- D2 / D4
def d2_trees():
    out, nodes_all = {}, []
    for tid, tree in trees.items():
        nodes = tree['nodes']
        rows = []
        for nd in nodes:
            if 'upsilon' not in nd:
                continue
            kids = [nodes[c] for c in nd['children']]
            n = nd['n']
            a, b = kids[0]['n'], kids[1]['n']
            mk = [np.array(k['makeup'], float) for k in kids]
            top1 = [float(np.sort(m)[-1] / m.sum()) for m in mk]
            top2 = [float(np.sort(m)[-2:].sum() / m.sum()) for m in mk]
            # source-respecting purity: share of node items on the side holding the majority of their source
            tot = mk[0] + mk[1]
            respect = float(np.maximum(mk[0], mk[1]).sum() / tot.sum())
            parent_top1 = float(tot.max() / tot.sum())
            rows.append(dict(tree=tid, id=nd['id'], depth=nd['depth'], n=n, sides=[a, b], imbalance=abs(a - b) / n,
                             upsilon=nd['upsilon'], spectral_ceiling=2 * nd['mu2'] / nd['mu1'],
                             mu1=nd['mu1'], mu2=nd['mu2'], phi=nd['phi'],
                             phi_free=nd['phi_free'], k_over_n=K / n, m_min=nd['m_min'],
                             cut_at_m_min=bool(min(a, b) == nd['m_min']), axis_used=nd['axis_used'],
                             makeup=nd['makeup'], child_makeup=[k['makeup'] for k in kids],
                             child_top1_share=top1, child_top2_share=top2, source_respect=respect,
                             parent_top1_share=parent_top1,
                             n_components=nd.get('n_components'), seba2_vs_cut=nd.get('seba2_vs_cut')))
        out[tid] = dict(block=tree['block'], nodes=rows)
        nodes_all += rows
    up = lambda rs: [round(r['upsilon'], 3) for r in rs]
    roots = [r for r in nodes_all if r['depth'] == 0]
    mixed = [r for r in nodes_all if r['parent_top1_share'] < 0.9]      # cuts of single-source nodes excluded
    clean1 = [r for r in mixed if min(r['child_top1_share']) >= 0.9]
    clean2 = [r for r in mixed if min(r['child_top2_share']) >= 0.9]
    resp = [r for r in mixed if r['source_respect'] >= 0.9]
    by_depth = {}
    for d in range(3):
        rs = [r for r in nodes_all if r['depth'] == d]
        by_depth[d] = dict(count=len(rs), n_mean=float(np.mean([r['n'] for r in rs])),
                           k_over_n_mean=float(np.mean([r['k_over_n'] for r in rs])),
                           mu1_mean=float(np.mean([r['mu1'] for r in rs])), mu2_mean=float(np.mean([r['mu2'] for r in rs])),
                           phi_mean=float(np.mean([r['phi'] for r in rs])),
                           upsilon_mean=float(np.mean([r['upsilon'] for r in rs])))
    # random d-regular (Ramanujan) reference for the walk's spectral gap, at d = k and d = 2k (the union kNN graph
    # has degree between k and 2k): mu1 ~ 1 - 2 sqrt(d - 1) / d
    expander = {d: 1 - 2 * math.sqrt(d - 1) / d for d in (K, 2 * K)}
    summary = dict(
        internal_nodes=len(nodes_all),
        within_2pct_balanced=sum(r['imbalance'] <= 0.02 for r in nodes_all),
        exactly_balanced=sum(r['imbalance'] <= 1 / r['n'] + 1e-12 for r in nodes_all),
        cut_at_m_min=sum(r['cut_at_m_min'] for r in nodes_all),
        upsilon_max=max(r['upsilon'] for r in nodes_all), upsilon_ge_3=sum(r['upsilon'] >= 3 for r in nodes_all),
        # Cheeger: phi >= mu1 / 2 for every cut, so Upsilon = mu2 / phi <= 2 mu2 / mu1 whatever the cut
        spectral_ceiling_ge_3=sum(r['spectral_ceiling'] >= 3 for r in nodes_all),
        spectral_ceiling_by_node=[(r['tree'], r['id'], round(r['spectral_ceiling'], 3)) for r in nodes_all],
        root_upsilon={r['tree']: r['upsilon'] for r in roots},
        clean_top1_children=dict(definition='parent < 90% one source; each child >= 90% one source', count=len(clean1),
                                 nodes=[(r['tree'], r['id']) for r in clean1], upsilon=up(clean1)),
        clean_top2_children=dict(definition='parent < 90% one source; each child >= 90% from its top two sources', count=len(clean2),
                                 nodes=[(r['tree'], r['id']) for r in clean2], upsilon=up(clean2)),
        source_respecting_cut=dict(definition='parent < 90% one source; >= 90% of node items on the side holding most of their source',
                                   count=len(resp), nodes=[(r['tree'], r['id']) for r in resp], upsilon=up(resp)),
        by_depth=by_depth, expander_mu1=expander,
        corr_mu1_k_over_n=float(np.corrcoef([r['k_over_n'] for r in nodes_all], [r['mu1'] for r in nodes_all])[0, 1]))
    return dict(trees=out, summary=summary)


# ------------------------------------------------------------------------------------- D2 recompute (optional)
def phi_parts(pbar, labels):
    n = len(labels)
    phis, sizes = [], []
    for a in np.unique(labels):
        m = labels == a
        s = int(m.sum())
        leave = float(s - pbar[np.ix_(m, m)].sum())          # rows of pbar sum to 1
        phis.append(leave / min(s, n - s)); sizes.append(s)
    return phis, sizes


def d2_recompute():
    need = [VOL / 'act_trained.npy', VOL / 'stack_trained.npy']
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        return dict(skipped=f'inputs absent: {missing} (fetch with `modal volume get cm-mamba-coherent '
                            f'work/m130/<file>` into {VOL})')
    t0 = time.time()
    act = np.load(VOL / 'act_trained.npy', mmap_mode='r')        # [layer, position, n, D]
    stack = np.load(VOL / 'stack_trained.npy', mmap_mode='r')    # [layer, position, n, n]
    assert act.shape[:3] == (len(LAYERS), len(POS), N) and stack.shape == (len(LAYERS), len(POS), N, N)
    out = {}
    for tid, li, view in (('T1', LAYERS.index(16), 'point'), ('T4', LAYERS.index(24), 'stack')):
        ps = []
        for pi in range(len(POS)):
            g = C.standard_gram(np.asarray(act[li, pi])) if view == 'point' else torch.from_numpy(np.array(stack[li, pi]))
            p, _ = C.walk(g)
            ps.append(p)
        vals, vecs = block_modes([C.lazy(p) for p in ps], 9, shared_stationary=True)
        vecs = vecs.clone(); vecs[:, 0] = 1 / math.sqrt(N)
        pbar = torch.stack(ps).mean(0).numpy()
        mu = (2 * (1 - vals.numpy()))                            # mu[j] = 1 - lambda_j(Pbar), j = 1..8 nontrivial
        stored = trees[tid]['nodes'][0]
        check = dict(vals_dev=float(np.abs(vals.numpy()[1:5] - np.array(stored['vals'])).max()),
                     mu1=float(mu[1]), mu2=float(mu[2]), stored_mu1=stored['mu1'], stored_mu2=stored['mu2'])
        rows = {}
        for k in range(2, 9):
            s, rot, perm = seba(vecs.double(), k, return_transform=True)
            raw = seba_reconstruct(vecs[:, :k].double(), rot, perm, clip=False).numpy()
            hard = raw.argmax(1)
            phis, sizes = phi_parts(pbar, hard)
            thr = seba_labels(s).numpy()
            rows[k] = dict(mu_k=float(mu[k]), ceiling=float(2 * mu[k] / mu[k - 1]), rho=max(phis), upsilon=float(mu[k] / max(phis)), part_phi=phis,
                           part_sizes=sizes, n_parts=len(sizes), seba_unassigned=int((thr < 0).sum()))
            if k in RANKS:   # the stored reading-block labels are the same block's SEBA: check we reproduce them
                key = 'reading_point_seba_label' if view == 'point' else 'reading_stack_seba_label'
                rows[k]['matches_stored_labels'] = bool(np.array_equal(thr, blocks[key][li, RANKS.index(k)]))
            rows[k]['makeup_by_part'] = [np.bincount(src[hard == a], minlength=5).tolist() for a in np.unique(hard)]
        phis, sizes = phi_parts(pbar, src)
        rows['source_k5'] = dict(note='label-derived partition: illustration only, not a construction output',
                                 mu_k=float(mu[5]), rho=max(phis), upsilon=float(mu[5] / max(phis)),
                                 part_phi=dict(zip(SOURCES, phis)))
        out[tid] = dict(check=check, mu=mu[1:].tolist(), by_k=rows, v1_root_upsilon=stored['upsilon'],
                        v1_root_phi=stored['phi'])
    out['d4_subsample'] = d4_subsample(act)
    out['seconds'] = time.time() - t0
    return out


def d4_subsample(act, sizes=(500, 250, 125), reps=3, seed=0):
    """D4: the T1 block (point L16, positions 4..128) rebuilt by the tree's own node code on RANDOM item subsets
    (all sources kept, so cluster structure is not removed) with k = 15, and with k scaled to 15 n / 1000; and the
    T1 depth-2 nodes rebuilt with k scaled. If mu1 rises on random subsets with k = 15, k/n drives it; if not, the
    rise in the tree comes from the nodes being more homogeneous."""
    import lm_tree as Z
    li = LAYERS.index(16)
    cells_t1 = [np.asarray(act[li, pi]) for pi in range(len(POS))]
    rng = np.random.default_rng(seed)

    def node(members, k):
        ps = [C.walk(Z.cell_gram(x, members), k=k)[0] for x in cells_t1]
        rec = Z.analyse_node(ps, members, src)
        return dict(n=len(members), k=k, mu1=rec['mu1'], mu2=rec['mu2'], phi=rec['phi'], upsilon=rec['upsilon'],
                    cut_m=rec['cut_m'])

    out = dict(random={}, t1_depth2_scaled_k=[])
    for n in sizes:
        rows = []
        for _ in range(reps):
            members = np.sort(rng.choice(N, n, replace=False))
            rows.append(dict(k15=node(members, K), k_scaled=node(members, max(2, round(K * n / N)))))
        out['random'][n] = rows
    for nd in trees['T1']['nodes']:
        if nd['depth'] == 2:
            members = np.asarray(nd['members'])
            out['t1_depth2_scaled_k'].append(dict(id=nd['id'], stored_mu1=nd['mu1'], stored_upsilon=nd['upsilon'],
                                                  **node(members, max(2, round(K * len(members) / N)))))
    return out


def main():
    res = dict(d1=d1())
    print(f"D1 done ({res['d1']['seconds']:.0f}s)")
    res['d3'] = d3()
    res['d2_trees'] = d2_trees()
    res['d2_recompute'] = d2_recompute()
    print('D2 recompute:', 'skipped' if 'skipped' in res['d2_recompute'] else f"{res['d2_recompute']['seconds']:.0f}s")
    OUT.write_text(json.dumps(rnd(res), indent=1))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
