"""Measurement drivers, fixtures and validation for the LM toy (SPEC sections 6, 7, 8, 10).

One code path serves the Modal units and the fixtures:

    U1 (model, GPU; lm_modal.py) or `fixture_inputs` (synthetic) writes the WORK files of a step:
        act_<step>.npy    float32 [layer, position, n, D]  point activations of probe A at the sampled positions
        slab_<step>.npy   float64 [layer, position, n, n]  slab Grams (lm_cells.SlabAccumulator over u = 1..t)
        bag_<step>.npy    float64 [position, n, Dv]        bag vectors of probe A
        actS_<FINAL>.npy  float32 [layer, n, D]            probe S point activations at t = 128 (final step only)
        u1_<step>.json    C1, C2, zero-norm scan, norm_mean [layer, 129] (final step), timings
    U2 `measure_step`   -> public cells_/single_/blocks_<step>.npz (+ figdata.npz at the final step);
                           work op128_<step>.npy, compare_<step>.json, trees_<step>.json
    U3 `finalize`       -> public train.npz, compare.json, trees.json, umap.npz, meta.json, validation.json

Public array layout (SPEC 10.1; documented here because the table is terse):
    cells_<step>.npz: key [view(point, slab), layer, position, ...] and bag_<key> [position, ...] for
        T T0 T_band(2) W W0 W_band(2) X(5,5) X_band(5,5,2) pn(8) min_set_r4 min_dist rogue1 rogue5
        sinkhorn_iters row_err; point only: own_src u8 [layer, position, n], follow_mass u8
        [10, layer, position, 5]; final step only: norm_mean [layer, 129]. Slab rogue1/rogue5 are NaN (no
        coordinates). Bands are the 2.5/97.5 % of the permutation draws of the raw (not excess) value.
    single_<step>.npz: key [view, layer, position, ...] and bag_<key> [position, ...]: schema_v2.block_record
        arrays (vals 9, vecs 8xn i16, scale 8, seba_label 5xn i8, seba_rot 5x8x8, seba_perm 5x8,
        retention_psi 8x6 and retention_seba 5x8x6 along POSITIONS) plus retention_psi_depth /
        retention_seba_depth along LAYERS (NaN for bag), top_members 5x8x24 i16, src_jacc (r=2,4),
        src_jacc_null (2, [mean, 2.5 %, 97.5 %]), src_jacc_ceiling (2).
    blocks_<step>.npz: reading_<key> [layer, ...] (retention along positions); final step also depth_<key>
        [position, ...] (retention along layers). Keys: block_record arrays, top_members, src_jacc*, and the
        block statistics T.. X_band, pn, min_set_r4, own_src u8 [n], follow_mass u8 [10, 5].
    train.npz: train_<key> [layer, run] (runs = LC.TRAIN_RUNS; retention over all 5 checkpoints) with the same
        keys as blocks; single_retention_psi [layer, step, 8, 5], single_retention_seba [layer, step, 5, 8, 5].
    trees.json: list of T1..T4 then P1 (planted fixture: planted_leaf, match, planted_upsilon, structureless, rule).
        Split nodes: n, vals (lambda_1..4 of Mbar^S), mu1, mu2, phi, phi_free, upsilon, axis_used, cut_theta, cut_m,
        psi (3, n) int8 + psi_scale, profile (64) at prefix sizes profile_m, cheeger_lo/hi, m_min, retention_cut
        (per cell), seba2 (int8 [n], -1 unassigned) and seba2_vs_cut, seba2_unassigned, makeup, ends (2, 6),
        n_components / component_sizes / disconnected (union of the cells' pre-floor kNN masks; diagnostic only).
        `members`, `psi` and `seba2` are aligned (members sorted ascending). Leaves: id, parent, depth, members, n,
        makeup, children = [].
    figdata.npz: C, K, P (cell L8, t = 128), Mbar (T1 block), M_cells [6 positions] (each cell's M), all 250 x 250
        float16, items ordered by T1's root cut (`order`); cells_t, cell.
    meta.json also: model_config (read from the HF config in U1), items, removed_items, follow_items, threshold.
    validation.json: schema checks; C1 {max_rel_err, pass}; C2 per step; P1, P2 (fixture numbers) and P3 (real-data
        linearity per step) with numbers and pass; zero_norm, removed_items.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))      # `python -I lm_schema.py ...` drops the script dir
import lm_construction as LC  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_tree as Z  # noqa: E402
from schema_v2 import RANKS  # noqa: E402

L, P, NS = len(LC.LAYERS), len(LC.POSITIONS), len(LC.STEPS)
STAT_KEYS = ['T', 'T0', 'T_band', 'W', 'W0', 'W_band', 'X', 'X_band']
RECORD_KEYS = ['vals', 'vecs', 'scale', 'seba_label', 'seba_rot', 'seba_perm', 'retention_psi', 'retention_seba',
               'top_members', 'src_jacc', 'src_jacc_null', 'src_jacc_ceiling']
DEPTH_KEYS = ['retention_psi_depth', 'retention_seba_depth']


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False, default=_default))


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _stack(dicts, keys):
    return {k: np.stack([np.asarray(d[k]) for d in dicts]) for k in keys}


# --------------------------------------------------------------------------------------------- the probe
class Probe:
    """Items, sources (validation only), tokens 1..SEQ, follow items; respects a run-wide removal list."""

    def __init__(self, probe_json, removed=()):
        d = json.loads(Path(probe_json).read_text()) if not isinstance(probe_json, dict) else probe_json
        items = d['items']
        keep = np.setdiff1d(np.arange(len(items)), np.asarray(removed, int))
        self.items = keep
        self.removed = sorted(map(int, removed))
        self.n = len(keep)
        self.sources = np.array([LC.SOURCES.index(items[i]['source']) for i in keep])
        self.tokens = np.array([items[i]['tokens'] for i in keep])          # (n, SEQ), snippet tokens 1..SEQ
        self.follow = np.concatenate([np.flatnonzero(self.sources == s)[:2] for s in range(len(LC.SOURCES))])
        self.raw = d


# -------------------------------------------------------------------------------------------- U2: a step
class _Cell:
    __slots__ = ('m', 'raw', 'info', 'rogue')


def _cell(c, tokens_t, probe, perms, rogue=(math.nan, math.nan), keep_p=False):
    p, info = C.walk(c)
    cell = _Cell()
    cell.raw = C.raw_statistics(p, tokens_t, probe.sources, perms, probe.follow)
    cell.info, cell.rogue = info, rogue
    cell.m = C.lazy(p)
    return (cell, p) if keep_p else cell


def _cell_row(cell, rec):
    out = C.export_statistics([cell.raw])
    out.update(pn=rec['pn'], min_set_r4=rec['min_set_r4'], min_dist=cell.info['min_dist'],
               rogue1=cell.rogue[0], rogue5=cell.rogue[1], sinkhorn_iters=cell.info['iterations'],
               row_err=cell.info['row_error'])
    return out


CELL_KEYS = STAT_KEYS + ['pn', 'min_set_r4', 'min_dist', 'rogue1', 'rogue5', 'sinkhorn_iters', 'row_err']
CELL_DTYPES = dict(min_set_r4=np.int16, sinkhorn_iters=np.int16)


def _block(cells, a, b, probe, perms, depth_axis=None, depth_index=None):
    rec, v = C.spectral_record([c.m for c in cells], (a, b), depth_axis, depth_index, probe.sources, perms)
    stats = C.export_statistics([c.raw for c in cells[a:b + 1]])
    return rec, v, stats


def measure_step(work, public, step, probe, log=print):
    """U2: every cell, single-cell record, block, comparison and tree of one checkpoint (SPEC 5-7)."""
    work, public = Path(work), Path(public)
    start = time.time()
    final = step == LC.FINAL
    act = np.load(work / f'act_{step}.npy', mmap_mode='r')
    slab = np.load(work / f'slab_{step}.npy', mmap_mode='r')
    bag = np.load(work / f'bag_{step}.npy', mmap_mode='r')
    n = probe.n
    assert act.shape[:3] == (L, P, n) and slab.shape == (L, P, n, n) and bag.shape[:2] == (P, n)
    perms = C.permutation_bank(n)
    tok = {t: probe.tokens[:, t - 1] for t in LC.POSITIONS}       # the token just read at position t
    grid = {}
    op128 = np.empty((L, n, n))
    for vi, view in enumerate(LC.VIEWS):
        for li in range(L):
            for pi, t in enumerate(LC.POSITIONS):
                if view == 'point':
                    x = act[li, pi]
                    cell, p = _cell(C.standard_gram(x), tok[t], probe, perms, C.rogue_shares(x), keep_p=True)
                    if t == LC.SEQ:
                        op128[li] = p.numpy()
                else:
                    cell = _cell(torch.from_numpy(np.array(slab[li, pi])), tok[t], probe, perms)
                grid[view, li, pi] = cell
        log(f'step {step}: {view} cells done, {time.time() - start:.0f}s')
    bags = [_cell(C.standard_gram(bag[pi]), tok[t], probe, perms, C.rogue_shares(bag[pi]))
            for pi, t in enumerate(LC.POSITIONS)]
    np.save(work / f'op128_{step}.npy', op128)
    del op128
    _json(work / f'p3_{step}.json', p3_check([grid['point', LC.LAYERS.index(LC.COMPARE_LAYER), pj]
                                               for pj in range(P)], probe, perms))

    # single cells: retention along positions (rows) and along layers (columns)
    single, vecs = {}, {}
    for view in LC.VIEWS:
        for li in range(L):
            row = [grid[view, li, pj] for pj in range(P)]
            for pi in range(P):
                col = [grid[view, lj, pi].m for lj in range(L)]
                rec, v, _ = _block(row, pi, pi, probe, perms, col, li)
                single[view, li, pi], vecs[view, li, pi] = rec, v
    for pi in range(P):
        rec, v, _ = _block(bags, pi, pi, probe, perms)
        single['bag', pi], vecs['bag', pi] = rec, v
    log(f'step {step}: single cells done, {time.time() - start:.0f}s')

    cells_out, single_out = {}, {}
    rows = {vk: [[_cell_row(grid[view, li, pi], single[view, li, pi]) for pi in range(P)] for li in range(L)]
            for vk, view in enumerate(LC.VIEWS)}
    for key in CELL_KEYS:
        arr = np.array([[[rows[v][li][pi][key] for pi in range(P)] for li in range(L)] for v in range(2)])
        cells_out[key] = arr.astype(CELL_DTYPES.get(key, np.float32))
        barr = np.array([_cell_row(bags[pi], single['bag', pi])[key] for pi in range(P)])
        cells_out['bag_' + key] = barr.astype(CELL_DTYPES.get(key, np.float32))
    pr = [[C.export_statistics([grid['point', li, pi].raw]) for pi in range(P)] for li in range(L)]
    cells_out['own_src'] = np.array([[pr[li][pi]['own_src'] for pi in range(P)] for li in range(L)], np.uint8)
    cells_out['follow_mass'] = np.array([[pr[li][pi]['follow_mass'] for pi in range(P)] for li in range(L)],
                                        np.uint8).transpose(2, 0, 1, 3)
    if final:
        u1 = json.loads((work / f'u1_{step}.json').read_text())
        cells_out['norm_mean'] = np.asarray(u1['norm_mean'], np.float32)
    np.savez_compressed(public / f'cells_{step}.npz', **cells_out)

    for key in RECORD_KEYS + DEPTH_KEYS:
        single_out[key] = np.stack([np.stack([np.stack([single[view, li, pi][key] for pi in range(P)])
                                              for li in range(L)]) for view in LC.VIEWS])
        single_out['bag_' + key] = np.stack([single['bag', pi][key] for pi in range(P)])
    np.savez_compressed(public / f'single_{step}.npz', **single_out)

    # blocks
    blocks_out = {}
    def put(prefix, recs):
        for key in RECORD_KEYS + ['pn', 'min_set_r4']:
            blocks_out[f'{prefix}_{key}'] = np.stack([r[0][key] for r in recs])
        for key in STAT_KEYS + ['own_src', 'follow_mass']:
            blocks_out[f'{prefix}_{key}'] = np.stack([r[2][key] for r in recs]).astype(
                np.uint8 if key in ('own_src', 'follow_mass') else np.float32)
    reading = [_block([grid['point', li, pj] for pj in range(P)], 0, P - 1, probe, perms) for li in range(L)]
    put('reading', reading)
    if final:
        depth = [_block([grid['point', lj, pi] for lj in range(L)], 0, L - 1, probe, perms) for pi in range(P)]
        put('depth', depth)
    np.savez_compressed(public / f'blocks_{step}.npz', **blocks_out)
    log(f'step {step}: blocks done, {time.time() - start:.0f}s')

    # comparisons: point vs slab (all cells); real vs S (final step)
    rows_c = []
    for li, layer in enumerate(LC.LAYERS):
        for pi, t in enumerate(LC.POSITIONS):
            a, b = single['point', li, pi], single['slab', li, pi]
            j, null, _ = C.matched_jaccard(a['seba_label'][RANKS.index(4)], b['seba_label'][RANKS.index(4)], 4, 4,
                                           perms)
            rows_c.append(dict(kind='point_slab', step=step, layer=layer, t=t,
                               a3=C.span_agreement(vecs['point', li, pi], vecs['slab', li, pi], 3),
                               a6=C.span_agreement(vecs['point', li, pi], vecs['slab', li, pi], 6),
                               jacc=j, jacc_null=null, W_a=None, W_b=None))
    if final:
        rows_c += real_vs_shuffled(work, step, probe, perms, grid, single, vecs, bags)
    _json(work / f'compare_{step}.json', rows_c)

    # trees of this step (SPEC 7.4)
    trees = []
    for spec in LC.TREES:
        if spec['step'] != step:
            continue
        li = LC.LAYERS.index(spec['layer'])
        if spec['kind'] == 'reading':
            cells = [np.asarray(act[li, pj]) for pj in range(P)]
            positions = LC.POSITIONS
        else:
            pi = LC.POSITIONS.index(spec['position'])
            cells, positions = [np.asarray(act[li, pi])], [spec['position']]
        nodes = Z.grow(cells, probe.sources)
        trees.append(dict(id=spec['id'], block=dict(kind=spec['kind'], layer=spec['layer'], positions=positions),
                          step=step, nodes=Z.to_json(nodes)))
        if spec['id'] == 'T1':
            figdata(public, grid, li, nodes, cells)
    _json(work / f'trees_{step}.json', trees)
    seconds = time.time() - start
    log(f'step {step}: complete, {seconds:.0f}s')
    return dict(step=step, seconds=seconds, sinkhorn_max_row_err=float(
        max(c.info['row_error'] for c in list(grid.values()) + bags)))


def p3_check(cells, probe, perms, tol=1e-12, n_sets=5):
    """P3 on real cells (one reading block): block conductance of random label-free sets = mean of cell
    conductances; W, X (fixed grouping) and T (one fixed token grouping) on the averaged walk = means of the cell
    values; all to 1e-12. The slab identity needs every position's activations and is checked by the tests."""
    ps = [(2 * c.m - torch.eye(len(c.m), dtype=torch.float64)).numpy() for c in cells]
    pbar = np.mean(ps, 0)
    rng = np.random.default_rng(0)
    dev_phi = 0.0
    for _ in range(n_sets):
        a = rng.random(len(pbar)) < rng.uniform(.1, .5)
        phi = lambda q: q[np.ix_(a, ~a)].sum() / min(a.sum(), (~a).sum())
        dev_phi = max(dev_phi, abs(phi(pbar) - np.mean([phi(q) for q in ps])))
    tok = probe.tokens[:, LC.SEQ - 1]
    few = perms[:20]
    cell_vals = [C.raw_statistics(q, tok, probe.sources, few, probe.follow) for q in ps]
    direct = C.raw_statistics(pbar, tok, probe.sources, few, probe.follow)
    dev = {k: float(np.max(np.abs(np.mean([r[k] for r in cell_vals], 0) - direct[k])))
           for k in ('T', 'W', 'X', 'T_null', 'W_null', 'X_null')}
    out = dict(block=f'reading L{LC.COMPARE_LAYER}', dev_phi=float(dev_phi), **{'dev_' + k: v for k, v in dev.items()},
               tolerance=tol)
    out['pass'] = bool(max(v for k, v in out.items() if k.startswith('dev_')) < tol)
    return out


def p2_check(seed=7):
    """P2 (SPEC 8) on the fixture's demotion block: the transient split tops its cell, the persistent split
    tops the 6-cell block; sweep and SEBA r = 2 matched Jaccard with the planted labels >= 0.95."""
    from coherent import block_modes, seba, seba_labels
    rng = np.random.default_rng(seed)
    u = _directions(rng)
    bp = rng.permutation(np.arange(FIX_N) % 2)
    bq = rng.permutation(np.arange(FIX_N) % 2)
    cells = []
    for t in LC.POSITIONS:
        x = noise(rng) + FIX_PERSISTENT * (2 * bp[:, None] - 1) * u[7]
        if t == FIX_TRANSIENT_T:
            x = x + FIX_TRANSIENT * (2 * bq[:, None] - 1) * u[8]
        cells.append(x)
    everyone = np.arange(FIX_N)
    jac = lambda a, b: LC.resnet_site_analysis().matched_jaccard(np.asarray(a, np.int64), np.asarray(b, np.int64),
                                                                  2, 2)[0]
    out = {}
    k = LC.POSITIONS.index(FIX_TRANSIENT_T)
    for name, block, truth in (('transient_cell', [cells[k]], bq), ('block', cells, bp)):
        rec = Z.analyse_node(Z.node_walks(block, everyone), everyone)
        side = np.isin(everyone, rec['children_members'][0]).astype(int)
        _, v = block_modes([C.lazy(C.walk(C.standard_gram(x))[0]) for x in block], 3, shared_stationary=True)
        lab = seba_labels(seba(v, 2)).numpy()
        out[name] = dict(target='transient' if truth is bq else 'persistent', sweep_jaccard=float(jac(side, truth)),
                         seba2_jaccard=float(jac(lab, truth)), upsilon=float(rec['upsilon']))
    out['pass'] = all(min(out[k]['sweep_jaccard'], out[k]['seba2_jaccard']) >= .95 for k in ('transient_cell', 'block'))
    out['seed'] = seed
    return out


def real_vs_shuffled(work, step, probe, perms, grid, single, vecs, bags):
    """Probe S (tokens 1..127 permuted, token 128 kept) against probe A at t = 128: W and r = 4 sets."""
    act_s = np.load(work / f'actS_{step}.npy', mmap_mode='r')
    pi = LC.POSITIONS.index(LC.SEQ)
    rows = []
    tok = probe.tokens[:, LC.SEQ - 1]                      # token 128 is kept in S
    r4 = RANKS.index(4)
    for li, layer in enumerate(LC.LAYERS):
        cell = _cell(C.standard_gram(act_s[li]), tok, probe, perms)
        rec, v = C.spectral_record([cell.m], (0, 0))
        a = single['point', li, pi]
        j, null, _ = C.matched_jaccard(rec['seba_label'][r4], a['seba_label'][r4], 4, 4, perms)
        rows.append(dict(kind='real_vs_S', step=step, layer=layer, t=LC.SEQ,
                         a3=C.span_agreement(vecs['point', li, pi], v, 3),
                         a6=C.span_agreement(vecs['point', li, pi], v, 6), jacc=j, jacc_null=null,
                         W_a=float(grid['point', li, pi].raw['W']), W_b=float(cell.raw['W'])))
    # bag: identical by C2 (asserted bitwise in U1), so S's bag cell IS A's bag cell
    a = single['bag', pi]
    j, null, _ = C.matched_jaccard(a['seba_label'][r4], a['seba_label'][r4], 4, 4, perms)
    rows.append(dict(kind='real_vs_S', step=step, layer='bag', t=LC.SEQ, a3=1.0, a6=1.0, jacc=j, jacc_null=null,
                     W_a=float(bags[pi].raw['W']), W_b=float(bags[pi].raw['W'])))
    return rows


def figdata(public, grid, li, nodes, cells, size=250):
    """Pipeline thumbnails (SPEC 10.1): C, K, P of the cell (layer, t = 128) and Mbar of T1's block, items
    ordered by T1's root cut (child 0 first, each side by decreasing winning psi), block-averaged to size^2."""
    root = nodes[0]
    axis = root['axis_used'] - 1
    psi = root['psi'][axis].astype(float)
    side0 = set(nodes[root['children'][0]]['members'].tolist())
    members = root['members']
    order = sorted(range(len(members)), key=lambda k: (members[k] not in side0, -psi[k]))
    order = members[np.array(order)]
    x = cells[-1]
    c = C.standard_gram(x)
    k = torch.exp(C.knn_logkernel(c))
    p = 2 * grid['point', li, len(LC.POSITIONS) - 1].m - torch.eye(len(c), dtype=torch.float64)
    mbar = torch.stack([grid['point', li, pj].m for pj in range(len(LC.POSITIONS))]).mean(0)
    def thumb(a):
        a = np.asarray(a)[np.ix_(order, order)]
        parts = np.array_split(np.arange(len(a)), size)
        return np.array([[a[np.ix_(r, q)].mean() for q in parts] for r in parts]).astype(np.float16)
    m_cells = np.stack([thumb(grid['point', li, pj].m) for pj in range(len(LC.POSITIONS))])
    np.savez_compressed(Path(public) / 'figdata.npz', C=thumb(c), K=thumb(k), P=thumb(p), Mbar=thumb(mbar),
                        M_cells=m_cells, cells_t=np.array(LC.POSITIONS, np.int16),
                        order=order.astype(np.int16), cell=np.array([LC.LAYERS[li], LC.SEQ], np.int16))


# ------------------------------------------------------------------------------------------- U3: finalize
def training(work, public, probe, log=print):
    """Training blocks [layer, run] at t = 128 and single-cell retention along training."""
    perms = C.permutation_bank(probe.n)
    tok = probe.tokens[:, LC.SEQ - 1]
    out = {}
    recs = {}
    single_psi = np.full((L, NS, 8, NS), np.nan, np.float32)
    single_seba = np.full((L, NS, len(RANKS), 8, NS), np.nan, np.float32)
    ops = [np.load(Path(work) / f'op128_{s}.npy', mmap_mode='r') for s in LC.STEPS]
    for li in range(L):
        cells = []
        for si in range(NS):
            p = torch.from_numpy(np.array(ops[si][li]))
            cell = _Cell()
            cell.raw = C.raw_statistics(p, tok, probe.sources, perms, probe.follow)
            cell.m = C.lazy(p)
            cells.append(cell)
        for ri, run in enumerate(LC.TRAIN_RUNS):
            a, b = LC.STEPS.index(run[0]), LC.STEPS.index(run[-1])
            recs[li, ri] = _block(cells, a, b, probe, perms)
        for si in range(NS):
            rec, _ = C.spectral_record([c.m for c in cells], (si, si))
            single_psi[li, si] = rec['retention_psi']
            single_seba[li, si] = rec['retention_seba']
        log(f'training blocks layer {LC.LAYERS[li]} done')
    for key in RECORD_KEYS + ['pn', 'min_set_r4']:
        out['train_' + key] = np.stack([np.stack([recs[li, ri][0][key] for ri in range(2)]) for li in range(L)])
    for key in STAT_KEYS + ['own_src', 'follow_mass']:
        out['train_' + key] = np.stack([np.stack([recs[li, ri][2][key] for ri in range(2)]) for li in range(L)]
                                       ).astype(np.uint8 if key in ('own_src', 'follow_mass') else np.float32)
    out['single_retention_psi'], out['single_retention_seba'] = single_psi, single_seba
    np.savez_compressed(Path(public) / 'train.npz', **out)


def _decode(arrs, idx, n):
    return np.concatenate([np.full((n, 1), 1 / n ** .5), arrs['vecs'][idx].T.astype('f8') * arrs['scale'][idx] / 32767], 1)


def to_final_rows(public, probe):
    perms = C.permutation_bank(probe.n)
    r4 = RANKS.index(4)
    li = LC.LAYERS.index(LC.COMPARE_LAYER)
    final = np.load(Path(public) / f'single_{LC.FINAL}.npz')
    rows = []
    for step in LC.STEPS:
        s = np.load(Path(public) / f'single_{step}.npz')
        for t in LC.TO_FINAL_POSITIONS:
            pi = LC.POSITIONS.index(t)
            idx = (0, li, pi)
            j, null, _ = C.matched_jaccard(s['seba_label'][idx][r4], final['seba_label'][idx][r4], 4, 4, perms)
            va, vb = (torch.from_numpy(_decode(z, idx, probe.n)) for z in (s, final))
            rows.append(dict(kind='to_final', step=step, layer=LC.COMPARE_LAYER, t=t,
                             a3=C.span_agreement(va, vb, 3), a6=C.span_agreement(va, vb, 6), jacc=j,
                             jacc_null=null, W_a=None, W_b=None))
    return rows


def calibrate(log=print):
    """P1 (SPEC 8): the planted-hierarchy tree, the structureless splits, and the Upsilon reading threshold."""
    cells, codes = planted_hierarchy_block()
    nodes = Z.grow(cells)
    match = planted_match(nodes, codes)
    structureless = []
    for node in nodes:
        if node['depth'] == Z.MAX_DEPTH:
            rec = Z.analyse_node(Z.node_walks(cells, node['members']), node['members'])
            structureless.append(dict(kind='planted leaf', node=node['id'], upsilon=rec['upsilon']))
    flat = flat_cell()
    rec = Z.analyse_node(Z.node_walks([flat], np.arange(len(flat))), np.arange(len(flat)))
    structureless.append(dict(kind='flat root', node=None, upsilon=rec['upsilon']))
    planted = [nd['upsilon'] for nd in nodes if 'upsilon' in nd]
    threshold, rule = Z.threshold_rule(planted, [s['upsilon'] for s in structureless])
    log(f'P1: exact={match["exact"]} planted min {min(planted):.2f} structureless max '
        f'{max(s["upsilon"] for s in structureless):.2f} -> threshold {threshold} ({rule})')
    return dict(id='P1', block=dict(kind='planted fixture', layer=None, positions=list(range(6))), step=None,
                threshold=threshold, rule=rule, planted_leaf=codes.tolist(), match=match,
                planted_upsilon=planted, structureless=structureless, nodes=Z.to_json(nodes))


def finalize(work, public, probe, fixture=False, layouts=True, log=print, extra_meta=None):
    """U3: training blocks, merged comparisons and trees (with P1 and the threshold), layouts, meta, validate."""
    work, public = Path(work), Path(public)
    training(work, public, probe, log)
    rows = sum((json.loads((work / f'compare_{s}.json').read_text()) for s in LC.STEPS), [])
    rows += to_final_rows(public, probe)
    _json(public / 'compare.json', dict(rows=rows))
    p1 = calibrate(log)
    p2 = p2_check()
    log(f'P2: pass={p2["pass"]}')
    p3 = {s: json.loads((work / f'p3_{s}.json').read_text()) for s in LC.STEPS}
    trees = []
    for s in LC.STEPS:
        path = work / f'trees_{s}.json'
        if path.exists():
            trees += json.loads(path.read_text())
    order = [t['id'] for t in LC.TREES]
    trees.sort(key=lambda t: order.index(t['id']))
    for t in trees:
        t['threshold'] = p1['threshold']
    _json(public / 'trees.json', trees + [p1])
    if layouts:
        umap_layouts(work, public)
    u1 = {s: json.loads((work / f'u1_{s}.json').read_text()) for s in LC.STEPS}
    meta = dict(schema='lm-toy-1', construction_fingerprint=LC.FINGERPRINT, model=LC.MODEL,
                transformers_version=u1[LC.FINAL].get('transformers_version'), steps=LC.STEPS, layers=LC.LAYERS,
                positions=LC.POSITIONS, k=LC.K, seba_r=RANKS, seba_r_shown=LC.SEBA_R_SHOWN, sources=LC.SOURCES,
                n=probe.n, items=probe.items.tolist(), removed_items=probe.removed,
                follow_items=probe.follow.tolist(), views=LC.VIEWS, train_runs=LC.TRAIN_RUNS, trees=LC.TREES,
                n_perm=LC.N_PERM, perm_seed=LC.PERM_SEED, fixture=fixture, threshold=p1['threshold'],
                threshold_rule=p1['rule'], layout=u1[LC.FINAL].get('layout', 'AlignedUMAP'))
    meta['model_config'] = u1[LC.FINAL].get('model_config')
    meta.update(extra_meta or {})
    _json(public / 'meta.json', meta)
    report = validate(public)
    report.update(C1=u1[LC.FINAL].get('C1'), C2={s: u1[s].get('C2') for s in LC.STEPS}, P2=p2,
                  P3=dict(slab_identity='test_lm.test_slab_gram_identity (needs all positions; not in U2)',
                          per_step=p3, **{'pass': all(r['pass'] for r in p3.values())}),
                  zero_norm={s: u1[s].get('zero_norm', []) for s in LC.STEPS}, removed_items=probe.removed,
                  duplicates='asserted per cell (min_dist > 1e-9) in U2; a violation fails the unit',
                  sinkhorn_failures=0, P1=dict(exact=p1['match']['exact'], threshold=p1['threshold'],
                                               rule=p1['rule'], planted_upsilon_min=min(p1['planted_upsilon']),
                                               structureless_upsilon_max=max(x['upsilon'] for x in p1['structureless']),
                                               **{'pass': bool(p1['match']['exact'] and p1['rule'] == 'fixed')}))
    _json(public / 'validation.json', report)
    return report


def pcs(c, k=50):
    val, vec = torch.linalg.eigh(torch.as_tensor(c))
    return (vec[:, -k:] * val[-k:].clamp_min(0).sqrt()).flip(1).numpy().astype(np.float32)


def umap_layouts(work, public):
    """13 AlignedUMAP runs (SPEC 11.4): grid rows (bag, L2..L12) across positions at the final step; strip per
    layer across checkpoints at t = 128. Display only."""
    import umap
    work = Path(work)
    act = np.load(work / f'act_{LC.FINAL}.npy', mmap_mode='r')
    bag = np.load(work / f'bag_{LC.FINAL}.npy', mmap_mode='r')
    n = act.shape[2]
    def run(slices):
        model = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, alignment_regularisation=0.02, random_state=0)
        model.fit(slices, relations=[{i: i for i in range(n)} for _ in range(len(slices) - 1)])
        return np.stack(model.embeddings_).astype(np.float32)
    grid = [run([pcs(C.standard_gram(bag[pi])) for pi in range(P)])]
    grid += [run([pcs(C.standard_gram(act[li, pi])) for pi in range(P)]) for li in range(L)]
    acts = [np.load(work / f'act_{s}.npy', mmap_mode='r') for s in LC.STEPS]
    strip = [run([pcs(C.standard_gram(a[li, P - 1])) for a in acts]) for li in range(L)]
    np.savez_compressed(Path(public) / 'umap.npz', grid=np.stack(grid), strip=np.stack(strip))


# ----------------------------------------------------------------------------------------------- fixtures
FIX_N, FIX_D, FIX_V = 1000, 32, 512
FIXTURE_MODEL_CONFIG = dict(model_type='gpt_neox', num_hidden_layers=12, hidden_size=768, num_attention_heads=12,
                            intermediate_size=3072, rotary_pct=0.25, rotary_emb_base=10000, use_parallel_residual=True,
                            vocab_size=50304, max_position_embeddings=2048, hidden_act='gelu', layer_norm_eps=1e-05,
                            tie_word_embeddings=False, note='fixture: copied from the HF config of pythia-160m')
FIX_SIGMA = 1.3
FIX_AMPS = (1.0, 0.7, 0.5)
FIX_PERSISTENT, FIX_TRANSIENT = 0.6, 1.2
FIX_DEMOTION_LAYER, FIX_TRANSIENT_T = 4, 32


def _directions(rng, d=FIX_D):
    return np.linalg.qr(rng.normal(size=(d, 9)))[0].T        # 7 hierarchy + persistent + transient, orthonormal


def hierarchy_codes(rng, n=FIX_N):
    leaf = rng.permutation(np.arange(n) % 8)                    # leaf = 4*[b1>0] + 2*[b2>0] + [b3>0]
    return leaf, np.stack([(leaf >> 2) & 1, (leaf >> 1) & 1, leaf & 1], 1) * 2 - 1


def hierarchy_signal(b, u, amps):
    """a1 b1 u1 + a2 b2 u_{2,b1} + a3 b3 u_{3,b1 b2} with 7 orthonormal u (rows 0..6 of u)."""
    i2 = 1 + (b[:, 0] > 0)
    i3 = 3 + 2 * (b[:, 0] > 0) + (b[:, 1] > 0)
    return amps[0] * b[:, :1] * u[0] + amps[1] * b[:, 1:2] * u[i2] + amps[2] * b[:, 2:3] * u[i3]


def noise(rng, n=FIX_N, d=FIX_D, sigma=FIX_SIGMA):
    return sigma * rng.normal(size=(n, d)) / np.sqrt(d)


def planted_hierarchy_block(seed=0, sigma=FIX_SIGMA, amps=FIX_AMPS, cells=6):
    """P1's block: 6 cells of the planted hierarchy at full amplitude, fresh noise in each."""
    rng = np.random.default_rng(seed)
    leaf, b = hierarchy_codes(rng)
    u = _directions(rng)
    sig = hierarchy_signal(b, u, amps)
    return [sig + noise(rng, sigma=sigma) for _ in range(cells)], leaf


def flat_cell(seed=1):
    return np.random.default_rng(seed).normal(size=(FIX_N, FIX_D))


def planted_match(nodes, leaf):
    """Exact node matching: every node at depth d is one planted node (a code prefix of length d, all of its
    2^(3-d) leaves, nothing else); returns per-node Jaccard with its best planted node."""
    per = []
    for nd in nodes:
        m = np.asarray(nd['members'])
        d = nd['depth']
        prefix = leaf[m] >> (3 - d)
        best = np.bincount(prefix).argmax()
        planted = np.flatnonzero((leaf >> (3 - d)) == best)
        jac = len(np.intersect1d(m, planted)) / len(np.union1d(m, planted))
        per.append(dict(node=nd['id'], depth=d, jaccard=jac))
    depths = [nd['depth'] for nd in nodes]
    exact = all(p['jaccard'] == 1 for p in per) and len(nodes) == 15 and depths.count(Z.MAX_DEPTH) == 8
    return dict(exact=exact, nodes=per)


def _fake_vocab(rng, v=FIX_V):
    syll = [a + b for a in 'bdfgklmnprstvz' for b in 'aeiou']
    return [(' ' if rng.random() < .6 else '') + ''.join(rng.choice(syll, size=rng.integers(1, 3))) for _ in range(v)]


def fixture_probe(rng, leaf):
    """Fake probe.json: sources follow the planted leaves in blocks of 200 (validation colouring only), tokens are
    drawn half from a source-specific band, half uniformly; text is fake. Probe S per SPEC 3."""
    n = len(leaf)
    order = np.argsort(leaf, kind='stable')
    source = np.empty(n, int)
    source[order] = np.arange(n) // LC.PER_SOURCE
    vocab = _fake_vocab(rng)
    items = []
    for i in range(n):
        band = 1 + 64 * source[i] + rng.integers(0, 64, LC.SEQ)
        uniform = rng.integers(1, FIX_V, LC.SEQ)
        tokens = np.where(rng.random(LC.SEQ) < .5, band, uniform)
        strs = [vocab[t] for t in tokens]
        items.append(dict(id=i, source=LC.SOURCES[source[i]], doc=f'fixture-{i}', char_start=0,
                          tokens=tokens.tolist(), token_str=strs, text=''.join(strs)))
    perms_s = [np.random.RandomState(i).permutation(LC.SEQ - 1).tolist() for i in range(n)]
    return dict(fixture=True, note='MOCKUP: synthetic probe; tokens and text are fake', items=items,
                S=dict(rule='tokens 1..127 permuted with RandomState(item_id); token 128 and EOT kept',
                       permutation=perms_s),
                filters={}, seeds=dict(fixture=True), rejections={}, vocab=vocab)


def shuffled_tokens(probe_json):
    out = []
    for item, perm in zip(probe_json['items'], probe_json['S']['permutation']):
        t = np.asarray(item['tokens'])
        out.append(np.concatenate([t[:-1][perm], t[-1:]]))
    return np.array(out)


def fixture_inputs(root_work, public, flat=False, seed=0, log=print):
    """Synthetic U1: writes the WORK files and probe.json for `_fixture` (planted) or `_fixture_flat`."""
    work, public = Path(root_work), Path(public)
    work.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    leaf, b = hierarchy_codes(rng)
    u = _directions(rng)
    bp = rng.permutation(np.arange(FIX_N) % 2) * 2 - 1
    bq = rng.permutation(np.arange(FIX_N) % 2) * 2 - 1
    probe_json = fixture_probe(rng, leaf)
    probe_json['planted'] = dict(leaf=leaf.tolist(), persistent=bp.tolist(), transient=bq.tolist(),
                                 sigma=FIX_SIGMA, amps=FIX_AMPS, flat=flat)
    if flat:
        probe_json['note'] += '; every cell is isotropic Gaussian noise (no planted structure)'
    _json(public / 'probe.json', probe_json)
    tokens = np.array([it['tokens'] for it in probe_json['items']])
    tokens_s = shuffled_tokens(probe_json)
    for si, step in enumerate(LC.STEPS):
        final = step == LC.FINAL
        def cell(li, u_pos):
            layer = LC.LAYERS[li]
            if flat:
                return rng.normal(size=(FIX_N, FIX_D))
            f = min(1.0, (si + 1) / 5 * (li + 2) / 5)
            x = noise(rng)
            if final and layer == FIX_DEMOTION_LAYER:
                x += FIX_PERSISTENT * bp[:, None] * u[7]
                if u_pos == FIX_TRANSIENT_T:
                    x += FIX_TRANSIENT * bq[:, None] * u[8]
            else:
                x += hierarchy_signal(b, u, tuple(f * a for a in FIX_AMPS))
            return x
        act = np.empty((L, P, FIX_N, FIX_D), np.float32)
        slab = np.empty((L, P, FIX_N, FIX_N))
        norm = np.zeros((L, LC.SEQ + 1))
        for li in range(L):
            acc = C.SlabAccumulator(FIX_N)
            for u_pos in range(1, LC.SEQ + 1):
                x = cell(li, u_pos).astype(np.float32)
                acc.add(torch.from_numpy(x))
                norm[li, u_pos] = np.linalg.norm(x, axis=1).mean()
                if u_pos in LC.POSITIONS:
                    pi = LC.POSITIONS.index(u_pos)
                    act[li, pi] = x
                    slab[li, pi] = acc.gram().numpy()
            norm[li, 0] = 30.0                                  # mock attention-sink norm at EOT
        np.save(work / f'act_{step}.npy', act)
        np.save(work / f'slab_{step}.npy', slab)
        emb = np.random.default_rng(1000 + si).normal(size=(FIX_V, FIX_D)).astype(np.float32)
        bag = C.bag_vectors(tokens, emb)
        bag_s = C.bag_vectors(tokens_s, emb, [LC.SEQ])
        c2 = bool(np.array_equal(bag[-1], bag_s[0]))
        assert c2, 'C2 failed: bag view not identical for A and S at t = 128'
        np.save(work / f'bag_{step}.npy', bag)
        u1 = dict(step=step, C2=dict(identical=c2, max_abs_diff=float(np.abs(bag[-1] - bag_s[0]).max())),
                  C1=dict(status='not run: fixture has no model', max_rel_err=None, **{'pass': None}),
                  zero_norm=[], transformers_version=None, model_config=FIXTURE_MODEL_CONFIG,
                  layout='AlignedUMAP', fixture=True, construction_fingerprint=LC.FINGERPRINT)
        if final:
            u1['norm_mean'] = norm.tolist()
            act_s = np.empty((L, FIX_N, FIX_D), np.float32)
            for li in range(L):
                act_s[li] = cell(li, LC.SEQ)
            np.save(work / f'actS_{step}.npy', act_s)
        _json(work / f'u1_{step}.json', u1)
        log(f'fixture inputs step {step} written')
    return probe_json


def build_fixture(root, flat=False, workers=1, layouts=True, log=print):
    """`data/_fixture/` or `data/_fixture_flat/` through the real U2/U3 code path."""
    import os
    import shutil
    root = Path(root)
    work = Path(os.environ.get('TMPDIR', '/tmp')) / f'cm-lm-work{root.name}'
    fixture_inputs(work, root, flat=flat, log=log)
    probe = Probe(root / 'probe.json')
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_measure_step_job, [(str(work), str(root), s, str(root / 'probe.json')) for s in LC.STEPS]))
    else:
        for s in LC.STEPS:
            measure_step(work, root, s, probe, log)
    report = finalize(work, root, probe, fixture='flat' if flat else 'planted', layouts=layouts, log=log)
    shutil.rmtree(work)
    return report


def _measure_step_job(args):
    torch.set_num_threads(1)
    work, root, step, probe_path = args
    return measure_step(work, root, step, Probe(probe_path))


# --------------------------------------------------------------------------------------------- validation
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _check_records(z, prefix, lead, n, nc, depth=False):
    """Shapes, dtypes, finiteness, padding and SEBA label reconstruction (as schema_v2.validate)."""
    from coherent import seba_labels, seba_reconstruct
    shapes = dict(vals=((9,), 'f4'), vecs=((8, n), 'i2'), scale=((8,), 'f4'), seba_label=((5, n), 'i1'),
                  seba_rot=((5, 8, 8), 'f4'), seba_perm=((5, 8), 'i1'), retention_psi=((8, nc), 'f4'),
                  retention_seba=((5, 8, nc), 'f4'), top_members=((5, 8, LC.TOP_MEMBERS), 'i2'),
                  src_jacc=((2,), 'f4'), src_jacc_null=((2, 3), 'f4'), src_jacc_ceiling=((2,), 'f4'))
    if depth:
        shapes.update(retention_psi_depth=((8, L), 'f4'), retention_seba_depth=((5, 8, L), 'f4'))
    worst = 1.0
    for key, (shape, dtype) in shapes.items():
        a = z[prefix + key]
        _check(a.shape == tuple(lead) + shape, (prefix, key, a.shape))
        _check(a.dtype == np.dtype(dtype), (prefix, key, a.dtype))
        if key not in ('retention_seba', 'retention_seba_depth', 'retention_psi_depth'):
            _check(np.isfinite(a).all(), (prefix, key, 'nonfinite'))
    vals, vecs, scale = z[prefix + 'vals'], z[prefix + 'vecs'], z[prefix + 'scale']
    rot, perm, lab, ret = (z[prefix + k] for k in ('seba_rot', 'seba_perm', 'seba_label', 'retention_seba'))
    _check(np.max(np.abs(vals[..., 0] - 1)) < 1e-6, (prefix, 'lambda0'))
    _check(np.all(np.diff(vals, axis=-1) <= 1e-6), (prefix, 'vals order'))
    for idx in np.ndindex(*lead):
        v = np.concatenate([np.full((n, 1), 1 / n ** .5), vecs[idx].T.astype('f8') * scale[idx] / 32767], 1)
        _check(np.max(np.abs(np.linalg.norm(v[:, 1:], axis=0) - 1)) < 2e-5, (prefix, idx, 'norm'))
        for ri, r in enumerate(RANKS):
            _check(((lab[idx][ri] >= -1) & (lab[idx][ri] < r)).all(), (prefix, idx, r, 'labels'))
            _check(np.all(rot[idx][ri, r:, :] == 0) and np.all(rot[idx][ri, :, r:] == 0), (prefix, idx, 'rot pad'))
            p = perm[idx][ri, :r]
            _check(sorted(p) == list(range(r)) and (perm[idx][ri, r:] == -1).all(), (prefix, idx, 'perm'))
            _check(np.isfinite(ret[idx][ri, :r]).all() and np.isnan(ret[idx][ri, r:]).all(), (prefix, idx, 'ret pad'))
            s = seba_reconstruct(torch.from_numpy(v[:, :r]), torch.from_numpy(rot[idx][ri, :r, :r]).double(),
                                 torch.from_numpy(p.astype('i8')))
            agree = float(np.mean(seba_labels(s).numpy() == lab[idx][ri]))
            worst = min(worst, agree)
            _check(agree >= .99, (prefix, idx, r, agree))
    return worst, int(np.prod(lead))


def _check_stats(z, prefix, lead, n):
    shapes = dict(T=(), T0=(), T_band=(2,), W=(), W0=(), W_band=(2,), X=(5, 5), X_band=(5, 5, 2), pn=(8,),
                  min_set_r4=())
    for key, shape in shapes.items():
        a = z[prefix + key]
        _check(a.shape == tuple(lead) + shape, (prefix, key, a.shape))
        _check(np.isfinite(a).all(), (prefix, key))
    for key in ('T', 'W'):
        _check(np.all((z[prefix + key] >= -1e-6) & (z[prefix + key] <= 1 + 1e-6)), (prefix, key, 'range'))
        band = z[prefix + key + '_band']
        _check(np.all(band[..., 0] <= band[..., 1] + 1e-7), (prefix, key, 'band order'))


def validate(root):
    """Schema check of a data directory (SPEC 10.1). Returns a report; raises AssertionError on any violation."""
    root = Path(root)
    meta = json.loads((root / 'meta.json').read_text())
    n = meta['n']
    _check(meta['schema'] == 'lm-toy-1' and meta['construction_fingerprint'] == LC.FINGERPRINT, 'meta schema')
    _check(meta['steps'] == LC.STEPS and meta['layers'] == LC.LAYERS and meta['positions'] == LC.POSITIONS, 'axes')
    _check(meta['seba_r'] == RANKS and meta['sources'] == LC.SOURCES and len(meta['follow_items']) == LC.N_FOLLOW,
           'meta lists')
    probe = json.loads((root / 'probe.json').read_text())
    _check(len(probe['items']) == n + len(meta['removed_items']), 'probe size')
    for it in probe['items']:
        _check(len(it['tokens']) == LC.SEQ and len(it['token_str']) == LC.SEQ and it['source'] in LC.SOURCES,
               ('probe item', it['id']))
    _check(len(probe['S']['permutation']) == len(probe['items']), 'probe S')
    worst, records = 1.0, 0
    for step in LC.STEPS:
        final = step == LC.FINAL
        with np.load(root / f'cells_{step}.npz') as z:
            _check_stats(z, '', (2, L, P), n)
            _check_stats(z, 'bag_', (P,), n)
            for key in ('min_dist', 'rogue1', 'rogue5', 'sinkhorn_iters', 'row_err'):
                _check(z[key].shape == (2, L, P) and z['bag_' + key].shape == (P,), (step, key))
            _check(np.all(z['min_dist'] > C.DUPLICATE_TOL) and np.all(z['bag_min_dist'] > C.DUPLICATE_TOL),
                   (step, 'duplicates'))
            _check(np.all(z['row_err'] < 2e-12) and np.all(z['bag_row_err'] < 2e-12), (step, 'sinkhorn'))
            _check(np.isfinite(z['rogue1'][0]).all() and np.isnan(z['rogue1'][1]).all(), (step, 'rogue'))
            _check(z['own_src'].shape == (L, P, n) and z['own_src'].dtype == np.uint8, (step, 'own_src'))
            _check(z['follow_mass'].shape == (LC.N_FOLLOW, L, P, 5) and z['follow_mass'].dtype == np.uint8,
                   (step, 'follow'))
            _check(np.all(np.abs(z['follow_mass'].astype(int).sum(-1) - 255) <= 3), (step, 'follow sums'))
            if final:
                _check(z['norm_mean'].shape == (L, LC.SEQ + 1), (step, 'norm_mean'))
        with np.load(root / f'single_{step}.npz') as z:
            w, k = _check_records(z, '', (2, L, P), n, P, depth=True)
            w2, k2 = _check_records(z, 'bag_', (P,), n, P, depth=True)
            _check(np.isfinite(z['retention_psi_depth']).all() and np.isnan(z['bag_retention_psi_depth']).all(),
                   (step, 'depth retention'))
            worst, records = min(worst, w, w2), records + k + k2
        with np.load(root / f'blocks_{step}.npz') as z:
            groups = [('reading_', L)] + ([('depth_', P)] if final else [])
            _check(set(z.files) == {g + key for g, _ in groups for key in RECORD_KEYS + STAT_KEYS +
                                    ['pn', 'min_set_r4', 'own_src', 'follow_mass']}, (step, 'block keys'))
            for g, m in groups:
                w, k = _check_records(z, g, (m,), n, L if g == 'depth_' else P)
                _check_stats(z, g, (m,), n)
                worst, records = min(worst, w), records + k
    with np.load(root / 'train.npz') as z:
        w, k = _check_records(z, 'train_', (L, 2), n, NS)
        _check_stats(z, 'train_', (L, 2), n)
        _check(z['single_retention_psi'].shape == (L, NS, 8, NS) and np.isfinite(z['single_retention_psi']).all(),
               'train single retention')
        worst, records = min(worst, w), records + k
    rows = json.loads((root / 'compare.json').read_text())['rows']
    kinds = {k: [r for r in rows if r['kind'] == k] for k in ('point_slab', 'real_vs_S', 'to_final')}
    _check(len(kinds['point_slab']) == NS * L * P and len(kinds['real_vs_S']) == L + 1 and
           len(kinds['to_final']) == NS * len(LC.TO_FINAL_POSITIONS), 'compare rows')
    for r in rows:
        _check(-1e-9 <= r['a3'] <= 1 + 1e-9 and 0 <= r['jacc'] <= 1 and len(r['jacc_null']) == 3, ('compare', r))
    trees = json.loads((root / 'trees.json').read_text())
    _check([t['id'] for t in trees] == [t['id'] for t in LC.TREES] + ['P1'], 'tree ids')
    for t in trees:
        nodes = t['nodes']
        for nd in nodes:
            if nd['children']:
                a, b = (set(nodes[c]['members']) for c in nd['children'])
                _check(not (a & b) and a | b == set(nd['members']), (t['id'], nd['id'], 'partition'))
                _check(min(len(a), len(b)) >= Z.m_min(nd['n']), (t['id'], nd['id'], 'm_min'))
                _check(len(nd['psi']) == 3 and len(nd['psi'][0]) == nd['n'] and len(nd['profile']) == 64,
                       (t['id'], nd['id'], 'node arrays'))
                _check(len(nd['seba2']) == nd['n'] and set(nd['seba2']) <= {-1, 0, 1}, (t['id'], nd['id'], 'seba2'))
                if t['id'] != 'P1' or 'n_components' in nd:
                    _check(nd['n_components'] >= 1 and sum(nd['component_sizes']) == nd['n'] and
                           isinstance(nd['disconnected'], bool), (t['id'], nd['id'], 'components'))
                _check(nd['cheeger_lo'] <= nd['phi_free'] * (1 + 1e-9) + 1e-12 and
                       nd['phi_free'] <= nd['cheeger_hi'] * (1 + 1e-9) + 1e-12, (t['id'], nd['id'], 'cheeger'))
            _check(nd['members'] == sorted(set(nd['members'])), (t['id'], nd['id'], 'members'))
    with np.load(root / 'umap.npz') as z:
        _check(z['grid'].shape == (L + 1, P, n, 2) and z['strip'].shape == (L, NS, n, 2), 'umap shapes')
        _check(np.isfinite(z['grid']).all() and np.isfinite(z['strip']).all(), 'umap finite')
    with np.load(root / 'figdata.npz') as z:
        for key in ('C', 'K', 'P', 'Mbar'):
            _check(z[key].shape == (250, 250) and z[key].dtype == np.float16 and np.isfinite(z[key]).all(), key)
        _check(z['M_cells'].shape == (P, 250, 250) and np.isfinite(z['M_cells']).all(), 'M_cells')
    _check(isinstance(meta.get('model_config'), dict) and meta['model_config'].get('num_hidden_layers'), 'model_config')
    report = dict(complete=True, records=records, minimum_seba_label_agreement=worst, n=n,
                  p1_exact=trees[-1]['match']['exact'], threshold=trees[-1]['threshold'])
    return report


if __name__ == '__main__':
    import sys
    torch.set_num_threads(1)
    if sys.argv[1] == 'fixture':
        print(json.dumps(build_fixture(sys.argv[2], flat='--flat' in sys.argv,
                                       workers=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--workers=')), 1)),
                                       layouts='--no-layouts' not in sys.argv), default=_default))
    elif sys.argv[1] == 'validate':
        print(json.dumps(validate(sys.argv[2]), default=_default))
    else:
        raise SystemExit(__doc__)
