"""Units U2 (cells, blocks, statistics) and U3 (trees, comparisons, layouts, finalize), the page fixture, and
validation of the public data directory (SPEC sections 3-9).

New module. One code path serves Modal and the fixture: U1 (`mc_decode.run_u1`) or `fixture_inputs` (synthetic)
writes the WORK files of a checkpoint (documented in `mc_decode`); `measure_checkpoint` (U2) and `finalize` (U3) read
them. Reused by import: `lm_schema.Probe`, `lm_cells` (walk, permutation bank, matched Jaccard), `lm_tree`
(sweep, threshold rule, to_json), `mc_stats` (everything streamed), `seba.seba_reconstruct` (validation).

Public files (data/<tag>/; derived arrays only):
    meta.json           axes, object and cell lists, theta and its rule, checkpoints, fingerprints, model config
    items.npz           per item: snippet, t, T, Nx, A, U, W, Cc, Cn, ctx (24 tokens before t and 4 after; -1 pad),
                        top5, top5_p (trained model's next-token predictions at t)
    vocab.json          {id: decoded string} for every id the page can show (clean-up off)
    cells_<c>.json      per cell (and per block, on Pbar): {grouping: S, S0, ex, band}, kappa (kappa, kappa_raw,
                        mass_diff, kappa0, kappa_ex, kappaT, kappaT_band, kappa_ex_T, unique_share), row_err,
                        min_dist, iters, k
    spectra_<c>.npz     per object o (blocks and listed cells), keys "o__<key>": vals (33), mu (32), gap (31), vecs
                        int16 (32, N) + scale (32), seba_label/seba_part/kmeans_part (9, N) int8, seba_rot (9, 32, 32),
                        seba_perm (9, 32), seba_sizes (9, 32), unassigned (9), phi_seba/phi_kmeans (9, 32),
                        ups_seba/ups_kmeans/ups (9), rounding (9; 0 SEBA, 1 k-means, -1 none valid), cap (9), pn (8),
                        ends (16, 2, 12), eta2_<g> (16) with _chance, _band (2, 16), _weighted, _weighted_band (2),
                        ami_<g> (2: r = 8, 16), jacc_<g> (2), jacc_<g>_null (2, 3), jacc_<g>_ceiling (2); blocks also
                        retention_psi (8, cells), retention_seba8 (8, cells) along the block's axis (v1, M-form)
    sets_<c>.json       {object: {r: [per-set descriptor]}} (SPEC 5 "per set, for the page")
    lags.json           lag retention (P-form, centred SEBA r = 8 vectors) per view, layer, tau, vector; chance
    compare.json        rows: kind, a, b, a8, a16, ami8, ami16
    upsilon.json        per checkpoint and object: k, ups (both roundings), cap, outcome under theta
    trees.json          T1 (B-point), T2 (whole): v1 node records plus gap, mu, tok_t, tok_next, class_t, class_next,
                        source, snippets; threshold = v1's 3.0-rule value from the fixture calibration
    layouts.npz         <object>: (N, 2) float32 UMAP of psi_1..psi_16 (display only)
    targets.json        SPEC 7 hypotheses K1..K6 evaluated (verdict inputs; labels as written)
    validation.json, calibration.json (copy of the fixture calibration and the full-size check)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_construction as K  # noqa: E402
import mc_items as I  # noqa: E402
import mc_stats as ST  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_schema as LS  # noqa: E402
import lm_tree as Z  # noqa: E402

SCHEMA = 'mamba-contexts-1'
BLOCKS = ['Bpoint', 'Bearly', 'Bmid', 'Blate', 'Bstate', 'Bk8', 'Bk30', 'Blagpoint', 'Blagwhole']
REFS = ['bag', 'recent', 'pred']
LAYOUT_OBJECTS = ['Bpoint', 'Bearly', 'Bmid', 'Blate', 'Bstate', 'whole', 'Blagpoint', 'Blagwhole'] + REFS
LAG_VIEWS = [('point', l) for l in K.LAG_LAYERS] + [('state', l) for l in K.LAG_LAYERS] + [('whole', None)]


def groupings_of(obj):
    """Names of the groupings an object is scored against (SPEC 5)."""
    if obj in ('Blagpoint', 'Blagwhole'):
        return ['T', 'Cc', 'U', 'W']
    return K.LAG_GROUPINGS if '_lag' in obj else K.GROUPINGS


def cell_name(view, layer=None, lag=0):
    base = view if layer is None else f'{view}_L{layer}'
    return base if lag == 0 else f'{base}_lag{lag}'


def objects_for(checkpoint):
    """Spectral objects (blocks and listed cells) of a checkpoint, in computation order."""
    if checkpoint != K.FINAL:
        return ['Bpoint', 'whole'] + [cell_name('point', l) for l in K.SINGLE_LAYERS] + \
            [cell_name('state', l) for l in K.SINGLE_LAYERS]
    out = [cell_name('point', l) for l in K.SINGLE_LAYERS] + ['Bpoint', 'Bearly', 'Bmid', 'Blate', 'Bk8', 'Bk30']
    out += [cell_name('state', l) for l in K.SINGLE_LAYERS] + ['Bstate', 'whole'] + REFS
    for tau in K.LAGS[1:]:
        out += [cell_name(v, l, tau) for v, l in LAG_VIEWS]
    return out + ['Blagpoint', 'Blagwhole']


# --------------------------------------------------------------------------------------------- inputs
def load_probe(path):
    probe = LS.Probe(path)
    probe.token_str = [it['token_str'] for it in probe.raw['items']]
    return probe


def load_items(work):
    with np.load(Path(work) / 'items.npz') as z:
        return {k: z[k] for k in z.files}


def item_groupings(work_root, probe):
    items = load_items(Path(work_root) / K.FINAL)
    top = np.load(Path(work_root) / K.FINAL / 'top.npz')
    g = I.groupings(items, probe.tokens, probe.token_str, probe.sources, top['top1'], lags=K.LAGS[1:])
    return items, g, top


def _npy(path, device):
    return torch.from_numpy(np.load(path)).to(device)


# ----------------------------------------------------------------------------------------------- U2
class Measure:
    """The state of one U2 run: statistics per cell, spectral objects, sets, spill of the cells' P."""

    def __init__(self, n, groupings, bc, device, spill_root=None, log=print):
        self.device, self.log, self.n = device, log, n
        self.g = groupings
        self.lag0 = {k: groupings[k] for k in K.GROUPINGS}
        self.perms = ST.permutation_bank(n)
        self.perms_jacc = C.permutation_bank(n, K.N_PERM_JACC, K.JACC_SEED)
        self.within, self.unique = ST.within_group_perms(groupings['T'])
        self.bc = bc
        self.cells, self.objects, self.sets, self.vecs = {}, {}, {}, {}
        self.spill = ST.Spill(spill_root)
        self.t0 = time.time()

    def stats(self, p, groupings, kappa=True):
        return ST.cell_statistics(p, groupings, self.perms, self.bc if kappa else None, self.g['U'], self.within,
                                  self.unique)

    def cell(self, name, gram, groupings=None, kappa=True, blocks=(), spill=False, spectral=False, k=K.K):
        reference = name in K.REFERENCES
        assert spectral or not reference, 'reference cells carry a spectral object (the tie check needs it)'
        p, info = ST.cell_walk(gram, k=k, assert_distinct=not reference)
        if not (reference and info['n_dup_pairs']):
            del gram
            gram = None
        rec = dict(row_err=info['row_error'], min_dist=info['min_dist'], iters=info['iterations'], k=k)
        if reference:
            rec.update(reference=True, n_dup_pairs=info['n_dup_pairs'])
        if groupings is not False:
            rec.update(self.stats(p, self.lag0 if groupings is None else groupings, kappa))
        self.cells[name] = rec
        for b in blocks:
            b.add(p, name)
        if spill:
            self.spill.put(name, p)
        if spectral:
            m = C.lazy(p)
            self.spectral(name, m, self.lag0 if groupings in (None, False) else groupings)
            del m
        if reference:
            rec['tie_check'], rec['tie_sensitive'] = [], False
            if gram is not None:
                labels = {r: self.objects[name]['seba_label'][K.R.index(r)] for r in K.SET_R}
                rec['tie_check'], rec['tie_sensitive'] = ST.tie_check(gram, self.vecs[name], labels, k)
                self.log(f"  {name}: {info['n_dup_pairs']} duplicate pairs; tie-sensitive {rec['tie_sensitive']}")
                del gram
        return p

    def spectral(self, name, mbar, groupings=None):
        rec, sets, vecs = ST.spectral_object(mbar, self.lag0 if groupings is None else groupings, self.perms,
                                             self.perms_jacc, desc_groupings=self.g)
        self.objects[name], self.sets[name], self.vecs[name] = rec, sets, vecs
        self.log(f'  {name}: spectrum done ({time.time() - self.t0:.0f}s)')
        return rec

    def block(self, name, bm, groupings=None, retention=True, kappa=True):
        """A block's spectral object from its running mean, its statistics on Pbar, and (optionally) the v1
        retention of psi_1..8 and the r = 8 SEBA vectors in each of its cells' M (read back from the spill)."""
        mbar = bm.mbar()
        bm.free()
        g = self.lag0 if groupings is None else groupings
        rec = self.spectral(name, mbar, g)
        pbar = ST.pbar_of(mbar)
        del mbar
        self.cells[name] = dict(block=True, cells=list(bm.names), **self.stats(pbar, g, kappa))
        del pbar
        if retention:
            f = np.concatenate([self.vecs[name][:, 1:9], rec['soft8']], 1)
            ret = np.stack([ST.retention(f, C.lazy(self.spill.get(c, self.device))) for c in bm.names], 1)
            rec['retention_psi'], rec['retention_seba8'] = ret[:8].astype(np.float32), ret[8:].astype(np.float32)
        return rec


def measure_checkpoint(work_root, public, checkpoint, probe, device='cpu', log=print, spill_root=None):
    """U2 (SPEC 5, 8 item 4): every cell, block and statistic of one checkpoint. Writes public cells_/spectra_/
    sets_<c> and, for the trained checkpoint, lags.json; work vecs_<c>.npz (float64 eigenvectors for U3)."""
    work_root, public = Path(work_root), Path(public)
    work = work_root / checkpoint
    public.mkdir(parents=True, exist_ok=True)
    final = checkpoint == K.FINAL
    items, g, _ = item_groupings(work_root, probe)
    items_ck = load_items(work)
    assert all(np.array_equal(items[k], items_ck[k]) for k in ('snippet', 't')), 'item tables differ'
    n = len(items['t'])
    bc = _npy(work_root / K.FINAL / 'bc.npy', device)
    M = Measure(n, g, bc, device, spill_root, log)
    start = time.time()
    point = np.load(work / 'point_lag0.npy', mmap_mode='r')
    assert point.shape[:2] == (n, K.N_LAYERS)
    lagret = {}
    try:
        # ---------------------------------------------------------------- point cells, lag 0
        bp, thirds = ST.BlockMean(), {t: ST.BlockMean() for t in K.THIRDS}
        sens = {k: ST.BlockMean() for k in K.K_SENS} if final else {}
        blp, blw = ST.BlockMean(), ST.BlockMean()
        for l in K.LAYERS:
            name = cell_name('point', l)
            blocks = [bp] + ([thirds[t] for t in K.THIRDS if l in K.THIRDS[t]] if final else [])
            if final and l == K.LAG_POINT_LAYER:
                blocks.append(blp)
            gram = ST.view_gram(torch.from_numpy(np.array(point[:, l - 1])), device)
            if final:
                for kk, bm in sens.items():
                    M.cell(f'{name}_k{kk}', gram.clone(), groupings=False, blocks=[bm], k=kk)
            M.cell(name, gram, blocks=blocks, spill=True, spectral=l in K.SINGLE_LAYERS)
        log(f'{checkpoint}: point cells done ({time.time() - start:.0f}s)')
        M.block('Bpoint', bp)
        if final:
            for t, bm in thirds.items():
                M.block('B' + t, bm)
            for kk, bm in sens.items():
                M.block(f'Bk{kk}', bm, retention=False)
            for l in K.LAG_LAYERS:
                lagret.setdefault('point', {}).setdefault(l, {})[0] = ST.retention(
                    ST.centred(M.objects['Bpoint']['soft8']), M.spill.get(cell_name('point', l), device))
        for l in K.LAYERS:
            if not (final and l == K.LAG_POINT_LAYER):
                M.spill.drop(cell_name('point', l))
        # ---------------------------------------------------------------- state cells, lag 0
        bs = ST.BlockMean()
        for l in K.LAYERS:
            name = cell_name('state', l)
            M.cell(name, _npy(work / f'state_lag0_L{l}.npy', device), blocks=[bs] if final else [],
                   spill=final, spectral=l in K.SINGLE_LAYERS)
        log(f'{checkpoint}: state cells done ({time.time() - start:.0f}s)')
        if final:
            M.block('Bstate', bs)
            for l in K.LAG_LAYERS:
                lagret.setdefault('state', {}).setdefault(l, {})[0] = ST.retention(
                    ST.centred(M.objects['Bstate']['soft8']), M.spill.get(cell_name('state', l), device))
            for l in K.LAYERS:
                M.spill.drop(cell_name('state', l))
        # ---------------------------------------------------------------- whole and references
        M.cell('whole', _npy(work / 'whole_lag0.npy', device), blocks=[blw] if final else [], spill=final,
               spectral=True)
        if final:
            lagret['whole'] = {None: {0: ST.retention(ST.centred(M.objects['whole']['soft8']),
                                                       M.spill.get('whole', device))}}
            for ref in ('bag', 'recent'):
                M.cell(ref, ST.view_gram(torch.from_numpy(np.load(work / f'{ref}.npy')), device), spectral=True)
            M.cell('pred', _npy(work / 'pred.npy', device), spectral=True)
            log(f'{checkpoint}: whole and references done ({time.time() - start:.0f}s)')
            # ------------------------------------------------------------ lag cells
            src = dict(point='Bpoint', state='Bstate', whole='whole')
            for tau in K.LAGS[1:]:
                lg = I.lag_groupings(g, tau)
                pts = np.load(work / f'point_lag{tau}.npy')
                for view, l in LAG_VIEWS:
                    name = cell_name(view, l, tau)
                    if view == 'point':
                        gram = ST.view_gram(torch.from_numpy(pts[:, K.LAG_LAYERS.index(l)]), device)
                    elif view == 'state':
                        gram = _npy(work / f'state_lag{tau}_L{l}.npy', device)
                    else:
                        gram = _npy(work / f'whole_lag{tau}.npy', device)
                    blocks = [blp] if (view, l) == ('point', K.LAG_POINT_LAYER) else \
                        [blw] if view == 'whole' else []
                    p = M.cell(name, gram, groupings=lg, kappa=False, blocks=blocks, spill=bool(blocks),
                               spectral=True)
                    lagret.setdefault(view, {}).setdefault(l, {})[tau] = ST.retention(
                        ST.centred(M.objects[src[view]]['soft8']), p)
                    del p
                log(f'{checkpoint}: lag {tau} done ({time.time() - start:.0f}s)')
            lag_g = dict(T=g['T'], Cc=g['Cc'], U=g['U'], W=g['W'])
            M.block('Blagpoint', blp, groupings=lag_g, kappa=False)
            M.block('Blagwhole', blw, groupings=lag_g, kappa=False)
    except BaseException as error:
        # keep the records of every cell completed so far, so a late failure does not lose the gate numbers
        K.write_json(work / f'u2_partial_{checkpoint}.json',
                     dict(error=f'{type(error).__name__}: {error}'[:2000], cells=strip_null(M.cells),
                          objects_done=sorted(M.objects), seconds=time.time() - start))
        raise
    finally:
        M.spill.close()
    assert set(M.objects) == set(objects_for(checkpoint)), set(M.objects) ^ set(objects_for(checkpoint))
    K.write_json(public / f'cells_{checkpoint}.json', strip_null(M.cells))
    arrays = {}
    for o, rec in M.objects.items():
        for key, val in rec.items():
            if key != 'soft8':
                arrays[f'{o}__{key}'] = np.asarray(val)
    np.savez_compressed(public / f'spectra_{checkpoint}.npz', **arrays)
    K.write_json(public / f'sets_{checkpoint}.json', M.sets)
    np.savez(work / f'vecs_{checkpoint}.npz', **{o: v for o, v in M.vecs.items()},
             **{f'{o}__soft8': r['soft8'] for o, r in M.objects.items() if r.get('soft8') is not None})
    if final:
        K.write_json(public / 'lags.json', lag_json(lagret, n))
    seconds = time.time() - start
    row_err = K.worst_of(c['row_err'] for c in M.cells.values() if 'row_err' in c)        # fail closed
    min_dist = K.least_of(c['min_dist'] for c in M.cells.values() if 'min_dist' in c and not c.get('reference'))
    references = {r: dict(min_dist=M.cells[r]['min_dist'], n_dup_pairs=M.cells[r]['n_dup_pairs'],
                          tie_sensitive=M.cells[r]['tie_sensitive']) for r in K.REFERENCES if r in M.cells}
    pn1 = K.least_of(float(M.objects[b]['pn'][0]) for b in M.objects if b.startswith('B'))
    summary = dict(checkpoint=checkpoint, seconds=seconds, n_cells=len(M.cells), n_objects=len(M.objects),
                   sinkhorn_max_row_err=row_err, min_dist=min_dist, min_pn_psi1_blocks=pn1,
                   references=references,
                   min_dist_definition='over point, state and whole cells (lag cells included); references exempt')
    K.write_json(work / f'u2_{checkpoint}.json', summary)
    log(f'{checkpoint}: U2 complete ({seconds:.0f}s)')
    return summary


def strip_null(cells):
    """Drop the raw permutation draws from the exported cell statistics (bands are kept)."""
    out = {}
    for name, rec in cells.items():
        out[name] = {k: ({kk: vv for kk, vv in v.items() if kk not in ('null',)} if isinstance(v, dict) else v)
                     for k, v in rec.items()}
    return out


def lag_json(lagret, n):
    out = dict(chance=-1.0 / (n - 1), lags=K.LAGS, source=dict(point='Bpoint', state='Bstate', whole='whole'),
               definition='f^T P_c f / f^T f, f = centred SEBA r = 8 soft vectors of the source object; P_c the '
                          'lag cell\'s zero-diagonal walk', views={})
    for view, by_layer in lagret.items():
        for l, by_tau in by_layer.items():
            key = view if l is None else f'{view}_L{l}'
            vals = np.array([by_tau[t] for t in K.LAGS])
            out['views'][key] = dict(per_vector=vals.T.tolist(), mean=vals.mean(1).tolist())
    return out


# ----------------------------------------------------------------------------------------------- U3
def describe_factory(g):
    def describe(members):
        members = np.asarray(members)
        d = {}
        for key, name in (('T', 'tok_t'), ('Nx', 'tok_next')):
            ids, cnt = np.unique(g[key][members], return_counts=True)
            o = np.argsort(-cnt, kind='stable')[:K.N_TOP_TOKENS]
            d[name] = [[int(ids[k]), int(cnt[k])] for k in o]
        d['class_t'] = np.bincount(g['Cc'][members], minlength=len(K.CLASSES)).tolist()
        d['class_next'] = np.bincount(g['Cn'][members], minlength=len(K.CLASSES)).tolist()
        d['source'] = np.bincount(g['W'][members], minlength=len(K.SOURCES)).tolist()
        d['snippets'] = int(len(np.unique(g['U'][members])))
        return d
    return describe


def grow_trees(work_root, probe, g, device='cpu', log=print, spill_root=None):
    """SPEC 6: T1 on B-point (24 point cells, vectors re-centred on each node) and T2 on the whole cell (48 raw
    constituents re-run through the standard pipeline on each node). Streaming (mc_stats.grow_streaming)."""
    work = Path(work_root) / K.FINAL
    describe = describe_factory(g)
    trees = []
    point = torch.from_numpy(np.load(work / 'point_lag0.npy')).to(device)
    t0 = time.time()
    nodes = ST.grow_streaming([ST.VectorCell(point[:, l - 1], device) for l in K.LAYERS], device, g['W'], describe,
                              spill_root=spill_root, log=log)
    trees.append(dict(id='T1', block='B-point', checkpoint=K.FINAL, cells=[cell_name('point', l) for l in K.LAYERS],
                      nodes=Z.to_json(nodes), seconds=time.time() - t0))
    del point
    log(f'T1: {len(nodes)} nodes ({time.time() - t0:.0f}s)')
    t0 = time.time()
    cons = {}
    for m in K.LAYERS:
        for part in K.PARTS:
            cons[part, m] = torch.from_numpy(np.load(work / f'cons_{part}{m}.npy')).to(device)

    def layers():
        for m in K.LAYERS:
            yield cons['c', m], cons['z', m]
    n = len(cons['c', 1])
    whole = ST.WholeCell(layers, n, device)
    root = whole.gram(np.arange(n))
    stored = _npy(work / 'whole_lag0.npy', device)
    root_dev = float((root - stored).abs().max())
    del root, stored
    nodes = ST.grow_streaming([whole], device, g['W'], describe, spill_root=spill_root, log=log)
    trees.append(dict(id='T2', block='whole', checkpoint=K.FINAL, cells=['whole'], nodes=Z.to_json(nodes),
                      root_gram_dev=root_dev, seconds=time.time() - t0))
    log(f'T2: {len(nodes)} nodes, root Gram route deviation {root_dev:.2e} ({time.time() - t0:.0f}s)')
    del cons
    return trees


def _ami_pair(sa, sb, oa, ob, r):
    ri = K.R.index(r)
    a = sa[f'{oa}__seba_label'][ri].astype(np.int64)
    b = sb[f'{ob}__seba_label'][ri].astype(np.int64)
    keep = (a >= 0) & (b >= 0)
    return ST.ami(np.where(keep, a, -1), b)


def comparisons(work_root, public, checkpoints):
    """SPEC 5 comparisons: a_8, a_16 and AMI of the r = 8, 16 hard sets (items assigned in both)."""
    vec = {c: np.load(Path(work_root) / c / f'vecs_{c}.npz') for c in checkpoints}
    spec = {c: np.load(Path(public) / f'spectra_{c}.npz') for c in checkpoints}
    rows = []

    def row(kind, ca, oa, cb, ob):
        va, vb = vec[ca][oa], vec[cb][ob]
        rows.append(dict(kind=kind, a=oa, b=ob, checkpoint_a=ca, checkpoint_b=cb,
                         a8=ST.span_agreement(va, vb, 8), a16=ST.span_agreement(va, vb, 16),
                         ami8=_ami_pair(spec[ca], spec[cb], oa, ob, 8), ami16=_ami_pair(spec[ca], spec[cb], oa, ob, 16)))
    f = K.FINAL
    main = ['Bpoint', 'Bstate', 'whole', 'bag', 'pred']
    for i, a in enumerate(main):
        for b in main[i + 1:]:
            row('objects', f, a, f, b)
    thirds = ['Bearly', 'Bmid', 'Blate']
    for i, a in enumerate(thirds):
        for b in thirds[i + 1:]:
            row('thirds', f, a, f, b)
    for view, l in LAG_VIEWS:
        for tau in K.LAGS[1:]:
            row('lag', f, cell_name(view, l), f, cell_name(view, l, tau))
    for kk in K.K_SENS:
        row('sensitivity', f, 'Bpoint', f, f'Bk{kk}')
    if 'init' in checkpoints:
        for o in ('Bpoint', 'whole'):
            row('init_trained', 'init', o, f, o)
    return rows


def reading_rule(calibration):
    """The effective SPEC 5/8 reading rule from both calibration stages: theta from the N = 1,200 fixture, and the
    full-size check (N = 8,000, from the bench). Certification is enabled only if theta exists and the full-size
    check is present and `calibrated` is True; otherwise k is read from the cap only, with the reason recorded."""
    theta = calibration.get('theta')
    fs = calibration.get('full_size_check')
    if theta is None:
        return dict(theta=None, certify=False, reason='rule dropped: planted and structureless overlap on the fixture')
    if not fs or 'calibrated' not in fs:
        return dict(theta=theta, certify=False,
                    reason='full-size check missing: theta not calibrated at N = 8,000; read from the cap only')
    if fs['calibrated'] is not True:
        return dict(theta=theta, certify=False,
                    reason=f"full-size check failed (planted {fs.get('planted_ups')}, structureless max "
                           f"{fs.get('structureless_max')}, theta {theta}): read from the cap only")
    return dict(theta=theta, certify=True, reason=None)


def upsilon_table(public, checkpoints, rule):
    out = {}
    for c in checkpoints:
        with np.load(Path(public) / f'spectra_{c}.npz') as z:
            objs = sorted({k.split('__')[0] for k in z.files})
            out[c] = {}
            for o in objs:
                rows = []
                for ri, r in enumerate(K.R):
                    ups = float(z[f'{o}__ups'][ri])
                    cap = float(z[f'{o}__cap'][ri])
                    rows.append(dict(k=r, ups=ups if math.isfinite(ups) else None,
                                     ups_seba=_f(z[f'{o}__ups_seba'][ri]), ups_kmeans=_f(z[f'{o}__ups_kmeans'][ri]),
                                     cap=cap, outcome=ST.outcome(ups if math.isfinite(ups) else None, cap,
                                                                 rule['theta'], rule['certify'])))
                out[c][o] = rows
    return out


def _num(x):
    """A json number that may have been written as 'nan'/'inf' (K.jsonable) back to float."""
    return float(x)


def _f(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _umap_one(x, seed=0):
    import umap
    return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(x).astype(np.float32)


def layouts(work_root, public, seed=0, workers=None):
    """F7: plain UMAP of each listed object's psi_1..psi_16 (display only). With a fixed random_state each UMAP is
    single-threaded, so the objects run in parallel processes (same results as a serial loop)."""
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor
    vec = np.load(Path(work_root) / K.FINAL / f'vecs_{K.FINAL}.npz')
    xs = [np.ascontiguousarray(vec[o][:, 1:1 + K.N_ETA]) for o in LAYOUT_OBJECTS]
    workers = workers or max(1, min(len(xs), len(os.sched_getaffinity(0))))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn')) as ex:
        out = dict(zip(LAYOUT_OBJECTS, ex.map(_umap_one, xs, [seed] * len(xs))))
    np.savez_compressed(Path(public) / 'layouts.npz', **out)


def targets(public, checkpoints, ups, lags, rule):
    """SPEC 7 hypotheses K1..K6 (labels as written; evaluated, not tuned)."""
    cells = {c: json.loads((Path(public) / f'cells_{c}.json').read_text()) for c in checkpoints}
    f = K.FINAL
    with np.load(Path(public) / f'spectra_{f}.npz') as z:
        ami = lambda o, gname: float(z[f'{o}__ami_{gname}'][0])
        k1 = dict(ami_Cc=ami('Bpoint', 'Cc'), ami_W=ami('Bpoint', 'W'))
        k2a = dict(ami_W=ami('whole', 'W'), ami_Cc=ami('whole', 'Cc'))
    k1['met'] = k1['ami_Cc'] > k1['ami_W']
    kb = cells[f]['Bpoint']['kappa']
    hi = _num(kb['kappaT_band'][1])
    k1b = dict(kappa=kb['kappa'], kappaT=kb['kappaT'], band_hi=hi, kappa_ex_T=kb['kappa_ex_T'],
               met=bool(math.isfinite(_num(kb['kappa'])) and math.isfinite(hi) and _num(kb['kappa']) > hi))
    uw, ub = cells[f]['whole']['U']['ex'], cells[f]['Bpoint']['U']['ex']
    k2 = dict(**k2a, U_whole=uw, U_Bpoint=ub, met=bool(k2a['ami_W'] > k2a['ami_Cc'] and uw >= ub + 0.2))
    p4, p24 = cells[f][cell_name('point', 4)], cells[f][cell_name('point', 24)]
    k3 = dict(T_L4=p4['T']['ex'], T_L24=p24['T']['ex'], kappa_L4=p4['kappa']['kappa_ex'],
              kappa_L24=p24['kappa']['kappa_ex'])
    k3['met'] = bool(k3['T_L4'] > k3['T_L24'] and k3['kappa_L24'] - k3['kappa_L4'] >= 0.1)
    rp, rw = lags['views'][f'point_L{K.LAG_POINT_LAYER}']['mean'], lags['views']['whole']['mean']
    i2, i16 = K.LAGS.index(2), K.LAGS.index(16)
    k4 = dict(point_L16_ret0=rp[0], point_L16_ret2=rp[i2], whole_ret0=rw[0], whole_ret16=rw[i16],
              met=bool(rp[i2] <= 0.5 * rp[0] and rw[i16] >= 0.5 * rw[0]),
              note='mean over the 8 vectors; per-vector values in lags.json')
    cert = [r['k'] for r in ups[f]['Bpoint'] if r['outcome'] == 'certified']
    with np.load(Path(public) / f'spectra_{f}.npz') as z:
        gaps = z['Bpoint__gap'].tolist()
    k5 = dict(certified=cert, outcomes={r['k']: r['outcome'] for r in ups[f]['Bpoint']}, gap_ratios=gaps,
              certification_enabled=rule['certify'], met=bool(rule['certify'] and cert))
    if not rule['certify']:
        k5['status'] = f"not certifiable: {rule['reason']}; K5 is read from the gap ratios and the excluded/" \
                       "undetermined outcomes only"
    out = dict(K1=k1, K1b=k1b, K2=k2, K3=k3, K4=k4, K5=k5)
    if 'init' in checkpoints:
        ki = cells['init']
        k6 = dict(kappa_L24_init=ki[cell_name('point', 24)]['kappa']['kappa_ex'], kappa_L24_trained=k3['kappa_L24'],
                  W_whole_init=ki['whole']['W']['ex'], W_whole_trained=cells[f]['whole']['W']['ex'])
        k6['met'] = bool(k6['kappa_L24_init'] < 0.5 * k6['kappa_L24_trained'] and
                         k6['W_whole_init'] < 0.5 * k6['W_whole_trained'])
        out['K6'] = k6
    return out


def public_items(work_root, public, probe, g, top, vocab_extra=None, tokenizer=None):
    items = load_items(Path(work_root) / K.FINAL)
    ctx = I.contexts(items, probe.tokens)
    np.savez_compressed(Path(public) / 'items.npz', snippet=items['snippet'].astype(np.int16),
                        t=items['t'].astype(np.int16), ctx=ctx, top5=top['top5'].astype(np.int32),
                        top5_p=top['top5_p'].astype(np.float32),
                        **{k: np.asarray(g[k]).astype(np.int32) for k in K.GROUPINGS})
    vocab = {}
    for row, strs in zip(probe.tokens, probe.token_str):
        for tid, s in zip(row, strs):
            vocab[int(tid)] = s
    vocab.update({int(k): v for k, v in (vocab_extra or {}).items()})
    missing = sorted({int(x) for x in np.unique(top['top5'])} - set(vocab))
    if missing and tokenizer is not None:
        for tid in missing:
            vocab[tid] = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    missing = sorted({int(x) for x in np.unique(top['top5'])} - set(vocab))
    K.write_json(Path(public) / 'vocab.json', dict(tokens={str(k): v for k, v in sorted(vocab.items())},
                                                    missing=missing))


def finalize(work_root, public, probe, calibration, checkpoints=tuple(K.CHECKPOINTS), device='cpu', log=print,
             fixture=False, layouts_on=True, tokenizer=None, extra_meta=None, spill_root=None):
    """U3: trees, comparisons, Upsilon outcomes under theta, layouts, targets, items/vocab, meta, validation."""
    work_root, public = Path(work_root), Path(public)
    items, g, top = item_groupings(work_root, probe)
    rule = reading_rule(calibration)
    theta = rule['theta']
    trees = grow_trees(work_root, probe, g, device, log, spill_root)
    for t in trees:
        t['threshold'] = Z.READING_THRESHOLD if calibration.get('tree_threshold') is None else \
            calibration['tree_threshold']
    K.write_json(public / 'trees.json', trees)
    K.write_json(public / 'compare.json', dict(rows=comparisons(work_root, public, list(checkpoints))))
    ups = upsilon_table(public, list(checkpoints), rule)
    K.write_json(public / 'upsilon.json', dict(theta=theta, rule=calibration.get('rule'),
                                               certification=dict(enabled=rule['certify'], reason=rule['reason']),
                                               by_checkpoint=ups))
    if layouts_on:
        t0 = time.time()
        layouts(work_root, public)
        log(f'layouts done ({time.time() - t0:.0f}s)')
    lags = json.loads((public / 'lags.json').read_text())
    K.write_json(public / 'targets.json', targets(public, list(checkpoints), ups, lags, rule))
    public_items(work_root, public, probe, g, top, probe.raw.get('vocab_extra'), tokenizer)
    K.write_json(public / 'calibration.json', calibration)
    u1 = {c: json.loads((work_root / c / f'u1_{c}.json').read_text()) for c in checkpoints}
    meta = dict(schema=SCHEMA, construction_fingerprint=K.FINGERPRINT, model=K.MODEL, revision=K.REVISION,
                fixture=fixture, checkpoints=list(checkpoints), n=len(items['t']), lags=K.LAGS, layers=K.LAYERS,
                single_layers=K.SINGLE_LAYERS, lag_layers=K.LAG_LAYERS, ranks=K.R, set_r=K.SET_R, n_eig=K.N_EIG,
                n_eta=K.N_ETA, groupings=K.GROUPINGS, lag_groupings=K.LAG_GROUPINGS, classes=K.CLASSES,
                sources=K.SOURCES, k=K.K, k_sens=K.K_SENS, n_perm=K.N_PERM, theta=theta,
                certification=dict(enabled=rule['certify'], reason=rule['reason'],
                                   full_size_check=(calibration.get('full_size_check') or {}).get('calibrated')),
                theta_rule=calibration.get('rule'), objects={c: objects_for(c) for c in checkpoints},
                blocks=BLOCKS, layout_objects=LAYOUT_OBJECTS, trees=K.TREES, context=[K.CONTEXT_BEFORE,
                                                                                       K.CONTEXT_AFTER],
                dedup=u1[K.FINAL].get('dedup'), model_config=u1[K.FINAL].get('model_config'),
                transformers_version=u1[K.FINAL].get('transformers_version'),
                parameter_sha={c: u1[c].get('parameter_sha') for c in checkpoints}, probe_sha=K.PROBE_SHA)
    ref_cells = json.loads((public / f'cells_{K.FINAL}.json').read_text())
    meta['references'] = {r: dict(min_dist=ref_cells[r]['min_dist'], n_dup_pairs=ref_cells[r]['n_dup_pairs'],
                                  tie_sensitive=ref_cells[r]['tie_sensitive']) for r in K.REFERENCES}
    meta['decode_fingerprint'] = K.DECODE_FINGERPRINT
    meta['amendments'] = K.AMENDMENTS
    meta.update(extra_meta or {})
    K.write_json(public / 'meta.json', meta)
    report = validate(public, require_layouts=layouts_on)
    report.update(controls={c: {k: u1[c].get(k) for k in ('C1a', 'C1b', 'C3', 'P3', 'zero_norm')}
                            for c in checkpoints}, references=meta['references'],
                  T2_root_gram_dev=trees[1]['root_gram_dev'])
    K.write_json(public / 'validation.json', report)
    return report


# ----------------------------------------------------------------------------------------------- fixture
FIX_SNIPPETS = K.FIX_N // K.PER_SNIPPET                     # 150
CLASS_TOKENS = {  # 8 planted classes (Zipf over this order), token strings chosen so that token_class agrees
    'word_start': [' cat', ' dog', ' house', ' tree', ' river', ' stone'],
    'function_word': [' the', ' of', ' a', ' and', ' in', ' to'],
    'sentence_punct': ['.', ',', ';', ':', '!', '?'],
    'newline': ['\n', '\n\n', ' \n', '\n\n\n'],
    'word_cont': ['ing', 'tion', 'ed', 'ly', 'ment', 'ness'],
    'number': [' 1', ' 2', ' 3', ' 10', ' 42', '7'],
    'latex': ['\\alpha', '\\beta', ' $x', '\\frac', ' \\cdot', '$'],
    'open': [' (', '(', '[', ' {', '_{', ' <'],
}
FIX_C_DIM, FIX_Z_DIM, FIX_POINT_DIM, FIX_EMB_DIM = 16, 48, 64, 32
FIX_AMP = {'trained': 1.0, 'init': 0.2}


def fixture_vocab():
    """Fake vocabulary: id 0 = EOT, then the class tokens. Returns (strings by id, class index by id)."""
    strs, cls = ['<|endoftext|>'], [-1]
    for name, toks in CLASS_TOKENS.items():
        for s in toks:
            assert K.TCL.token_class(s) == name, (s, K.TCL.token_class(s), name)
            strs.append(s)
            cls.append(K.CLASSES.index(name))
    return strs, np.array(cls)


def _unit(rng, k, d):
    return np.linalg.qr(rng.normal(size=(d, max(k, 1))))[0].T[:k] if k <= d else rng.normal(size=(k, d)) / np.sqrt(d)


def fixture_inputs(work_root, public, flat=False, seed=0, log=print):
    """Synthetic U1 (MOCKUP): the WORK files of both checkpoints and probe.json, with planted groupings for every
    statistic: 8 token classes with Zipf frequencies (tokens nested in classes), 150 snippets of 5 sources (each
    source favours one class), point cells carrying token identity early and class late, with a memory of the class
    at t that decays with the lag; layer states carrying the conv buffer's recent class (c) and the snippet and source
    (z, persistent across lags); predictions that follow the class at t. `flat`: every view is noise (the
    structureless twin; tokens and groupings unchanged). The state and whole views go through mc_stats.ViewGrams."""
    import mc_decode as D
    work_root, public = Path(work_root), Path(public)
    public.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    strs, tcls = fixture_vocab()
    v = len(strs)
    ncls = len(CLASS_TOKENS)
    class_ids = [np.flatnonzero(tcls == K.CLASSES.index(name)) for name in CLASS_TOKENS]
    zipf = 1.0 / np.arange(1, ncls + 1)
    zipf /= zipf.sum()
    sources = np.arange(FIX_SNIPPETS) % len(K.SOURCES)
    rng.shuffle(sources)
    tokens = np.empty((FIX_SNIPPETS, K.SEQ), np.int64)
    for i in range(FIX_SNIPPETS):
        pc = 0.75 * zipf + 0.25 * np.eye(ncls)[sources[i] + 1]
        cl = rng.choice(ncls, K.SEQ, p=pc / pc.sum())
        tokens[i] = [rng.choice(class_ids[c]) for c in cl]
    probe_json = dict(items=[dict(id=i, source=K.SOURCES[sources[i]], doc=i, tokens=tokens[i].tolist(),
                                  token_str=[strs[t] for t in tokens[i]], text=''.join(strs[t] for t in tokens[i]))
                             for i in range(FIX_SNIPPETS)],
                      S=dict(permutation=[]), fixture=True, flat=flat,
                      vocab_extra={str(i): s for i, s in enumerate(strs)},
                      note='MOCKUP: synthetic token-context fixture (mc_schema.fixture_inputs)')
    K.write_json(public / 'probe.json', probe_json)
    probe = load_probe(public / 'probe.json')
    items = I.sample_items(tokens)
    I.check_items(items, tokens)
    snip, t = items['snippet'], items['t']
    n = len(t)
    cls_at = lambda u: tcls[tokens[snip, u - 1]] - 0                       # class index into K.CLASSES
    remap = {K.CLASSES.index(name): j for j, name in enumerate(CLASS_TOKENS)}
    pc_at = lambda u: np.array([remap[c] for c in cls_at(u)])               # planted class 0..7
    tok_at = lambda u: tokens[snip, u - 1]
    geo = np.random.default_rng(seed + 1)
    e_cls = [_unit(geo, ncls, FIX_POINT_DIM) for _ in K.LAYERS]
    e_mem = [_unit(geo, ncls, FIX_POINT_DIM) for _ in K.LAYERS]
    e_tok = [geo.normal(size=(v, FIX_POINT_DIM)) / np.sqrt(FIX_POINT_DIM) for _ in K.LAYERS]
    c_cls = [_unit(geo, ncls, FIX_C_DIM) for _ in K.LAYERS]
    z_src = [_unit(geo, len(K.SOURCES), FIX_Z_DIM) for _ in K.LAYERS]
    z_snip = [geo.normal(size=(FIX_SNIPPETS, FIX_Z_DIM)) / np.sqrt(FIX_Z_DIM) for _ in K.LAYERS]
    means = [(geo.normal(size=FIX_C_DIM) * 2, geo.normal(size=FIX_Z_DIM) * 2) for _ in K.LAYERS]
    emb = geo.normal(size=(v, FIX_EMB_DIM))
    w_pred = geo.normal(size=(ncls, v)) * 2.5
    w_tok = geo.normal(size=(v, v)) * 0.4
    for ck in K.CHECKPOINTS:
        work = work_root / ck
        work.mkdir(parents=True, exist_ok=True)
        np.savez(work / 'items.npz', snippet=snip, t=t, order=items['order'])
        a = 0.0 if flat else FIX_AMP[ck]
        final = ck == K.FINAL
        crng = np.random.default_rng([seed, K.CHECKPOINTS.index(ck)])
        info = dict(checkpoint=ck, construction_fingerprint=K.FINGERPRINT, parameter_sha=None, fixture=True,
                    model_config=dict(num_hidden_layers=24, note='fixture'), n=n, zero_norm={},
                    dedup=dict(redraws=items['redraws'], redraw_items=items['redraw_items'][:200]),
                    C1a=dict(status='not run: fixture has no model'), C1b=dict(status='not run: fixture'),
                    C3=dict(status='not run: fixture'))
        for lag in (K.LAGS if final else [0]):
            u = t + lag
            layers = K.LAYERS if lag == 0 else K.LAG_LAYERS
            point = np.empty((n, len(layers), FIX_POINT_DIM), np.float32)
            for li, l in enumerate(layers):
                s_cls, s_tok = a * min(1.0, l / 12), a * 0.9 * max(0.0, 1 - l / 16)
                x = crng.normal(size=(n, FIX_POINT_DIM)) / np.sqrt(FIX_POINT_DIM)
                x += s_cls * e_cls[l - 1][pc_at(u)] + s_tok * e_tok[l - 1][tok_at(u)]
                x += s_cls * 0.6 ** lag * e_mem[l - 1][pc_at(t)]
                point[:, li] = x
            np.save(work / f'point_lag{lag}.npy', point)
            cbuf = np.empty((n, K.N_LAYERS, FIX_C_DIM), np.float32)
            zbuf = np.empty((n, K.N_LAYERS, FIX_Z_DIM), np.float32)
            for m in K.LAYERS:
                c = crng.normal(size=(n, FIX_C_DIM)) / np.sqrt(FIX_C_DIM) + a * 0.9 * c_cls[m - 1][pc_at(u)]
                z = crng.normal(size=(n, FIX_Z_DIM)) / np.sqrt(FIX_Z_DIM) + a * (0.6 * z_src[m - 1][probe.sources[snip]]
                                                                                   + 0.7 * z_snip[m - 1][snip])
                cbuf[:, m - 1] = c * np.exp(0.3 * crng.normal(size=(n, 1))) + means[m - 1][0]
                zbuf[:, m - 1] = z * np.exp(0.3 * crng.normal(size=(n, 1))) + means[m - 1][1]
            snap = D.Snapshot.__new__(D.Snapshot)
            snap.c, snap.z = torch.from_numpy(cbuf), torch.from_numpy(zbuf)
            if lag == 0 and final:
                info['P3'] = D.p3_check(snap, K.N_LAYERS, 'cpu')
                for m in K.LAYERS:
                    np.save(work / f'cons_c{m}.npy', cbuf[:, m - 1])
                    np.save(work / f'cons_z{m}.npy', zbuf[:, m - 1])
            vg = ST.ViewGrams(n, 'cpu')
            for m in K.LAYERS:
                gm = vg.layer(snap.c[:, m - 1], snap.z[:, m - 1])
                if lag == 0 or m in K.LAG_LAYERS:
                    np.save(work / f'state_lag{lag}_L{m}.npy', gm.numpy())
            np.save(work / f'whole_lag{lag}.npy', vg.whole().numpy())
            if final and lag == 0:
                logits = (0 if flat else 1) * (w_pred[pc_at(t)] + w_tok[tok_at(t)]) + crng.normal(size=(n, v))
                logp = torch.log_softmax(torch.from_numpy(logits), -1)
                sq = torch.exp(0.5 * logp).float()
                q = sq.double()
                np.save(work / 'bc.npy', (q @ q.T).numpy())
                np.save(work / 'pred.npy', ST.view_gram(sq).numpy())
                tp, ti = torch.topk(logp, 5, dim=-1)
                np.savez(work / 'top.npz', top1=ti[:, 0].numpy(), top5=ti.numpy(), top5_p=tp.exp().float().numpy())
        if final:
            np.save(work / 'bag.npy', I.bag_vectors(items, tokens, emb))
            np.save(work / 'recent.npy', I.recent_vectors(items, tokens, emb))
        K.write_json(work / f'u1_{ck}.json', info)
        log(f'fixture inputs {ck} written')
    return probe


def build_fixture(root, calibration, flat=False, layouts_on=True, device='cpu', log=print):
    """data/_fixture or data/_fixture_flat through the real U2/U3 code path (MOCKUP)."""
    import os
    root = Path(root)
    work = Path(os.environ.get('TMPDIR', '/tmp')) / f'cm-mc-work{root.name}'
    try:
        probe = fixture_inputs(work, root, flat=flat, log=log)
        timings = {}
        for ck in K.CHECKPOINTS:
            timings[ck] = measure_checkpoint(work, root, ck, probe, device, log)
        t0 = time.time()
        report = finalize(work, root, probe, calibration, device=device, log=log,
                          fixture='flat' if flat else 'planted', layouts_on=layouts_on)
        report['timings'] = dict(u2=timings, u3=time.time() - t0)
        K.write_json(root / 'validation.json', report)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return report


# --------------------------------------------------------------------------------------------- validation
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def validate(root, require_layouts=True):
    """Schema check of a public data directory. Returns a report; raises AssertionError on any violation."""
    from seba import seba_reconstruct, seba_labels
    root = Path(root)
    meta = json.loads((root / 'meta.json').read_text())
    n = meta['n']
    _check(meta['schema'] == SCHEMA and meta['construction_fingerprint'] == K.FINGERPRINT, 'meta schema')
    _check(meta['ranks'] == K.R and meta['lags'] == K.LAGS and meta['groupings'] == K.GROUPINGS, 'meta axes')
    theta = meta['theta']
    certify = meta['certification']['enabled']
    _check(set(meta['references']) == set(K.REFERENCES), 'meta references')
    _check(isinstance(certify, bool) and (certify or meta['certification']['reason']), 'certification record')
    _check(not certify or (theta is not None and meta['certification']['full_size_check'] is True),
           'certification enabled without a passing full-size check')
    worst, records = 1.0, 0
    for ck in meta['checkpoints']:
        cells = json.loads((root / f'cells_{ck}.json').read_text())
        for name, rec in cells.items():
            if 'row_err' in rec:
                _check(rec['row_err'] < K.SINKHORN_TOL, (ck, name, 'sinkhorn'))
                if name in K.REFERENCES:                 # exempt (SPEC 3, after pilot 1): hazard and tie check
                    _check(rec.get('reference') is True and isinstance(rec['n_dup_pairs'], int), (ck, name, 'ref'))
                    _check((rec['n_dup_pairs'] > 0) == (rec['min_dist'] <= K.DUPLICATE_TOL), (ck, name, 'ref count'))
                    tc = rec['tie_check']
                    _check(len(tc) == (len(K.TIE_SEEDS) if rec['n_dup_pairs'] else 0), (ck, name, 'tie check'))
                    if tc:
                        a = [row[f'a{r}'] for row in tc for r in K.SET_R]
                        m = [_num(row[f'ami{r}']) for row in tc for r in K.SET_R]
                        sens = not (all(map(math.isfinite, a + m)) and min(a) >= K.TIE_MIN_A and
                                    min(m) >= K.TIE_MIN_AMI)
                        _check(rec['tie_sensitive'] == sens, (ck, name, 'tie flag'))
                    else:
                        _check(rec['tie_sensitive'] is False, (ck, name, 'tie flag'))
                else:
                    _check(rec['min_dist'] > K.DUPLICATE_TOL, (ck, name, 'duplicates'))
            for gname, s in rec.items():
                if isinstance(s, dict) and 'S' in s:
                    _check(-1e-12 <= s['S'] <= 1 + 1e-12 and 0 <= s['S0'] <= 1 and s['band'][0] <= s['band'][1],
                           (ck, name, gname))
            if 'kappa' in rec:
                kp = {k: _num(rec['kappa'][k]) for k in ('kappa', 'kappa0', 'kappaT', 'kappa_ex', 'kappa_ex_T')}
                kp['kappaT_band'] = rec['kappa']['kappaT_band']
                _check(all(math.isfinite(kp[k]) for k in ('kappa', 'kappa0', 'kappaT', 'kappa_ex', 'kappa_ex_T')),
                       (ck, name, 'kappa finite'))
                lo, hi = (_num(x) for x in kp['kappaT_band'])
                _check(lo <= kp['kappaT'] <= hi + 1e-12, (ck, name, 'kappa band'))
        sets = json.loads((root / f'sets_{ck}.json').read_text())
        with np.load(root / f'spectra_{ck}.npz') as z:
            objs = sorted({k.split('__')[0] for k in z.files})
            _check(set(objs) == set(meta['objects'][ck]), (ck, 'objects'))
            for o in objs:
                g = lambda key: z[f'{o}__{key}']
                vals = g('vals')
                _check(abs(vals[0] - 1) < 1e-6 and np.all(np.diff(vals) <= 1e-9) and len(vals) == K.N_EIG, (ck, o, 'vals'))
                mu = g('mu')
                _check(np.allclose(mu, 2 * (1 - vals[1:])), (ck, o, 'mu'))
                _check(np.allclose(g('gap'), mu[1:] / mu[:-1]), (ck, o, 'gap'))
                q, scale = g('vecs'), g('scale')
                _check(q.shape == (K.N_EIG - 1, n) and q.dtype == np.int16, (ck, o, 'vecs'))
                v = np.concatenate([np.full((n, 1), 1 / n ** .5), ST.dequantize(q, scale)], 1)
                _check(np.max(np.abs(np.linalg.norm(v[:, 1:], axis=0) - 1)) < 2e-4, (ck, o, 'vec norms'))
                for ri, r in enumerate(K.R):
                    lab = g('seba_label')[ri]
                    _check(((lab >= -1) & (lab < r)).all(), (ck, o, r, 'labels'))
                    perm = g('seba_perm')[ri]
                    _check(sorted(perm[:r]) == list(range(r)) and (perm[r:] == -1).all(), (ck, o, r, 'perm'))
                    s = seba_reconstruct(torch.from_numpy(v[:, :r]), torch.from_numpy(g('seba_rot')[ri, :r, :r]).double(),
                                         torch.from_numpy(perm[:r].astype('i8')))
                    agree = float(np.mean(seba_labels(s).numpy() == lab))
                    worst = min(worst, agree)
                    _check(agree >= .99, (ck, o, r, 'seba reconstruct', agree))
                    sizes = g('seba_sizes')[ri]
                    _check(np.array_equal(sizes[:r], np.bincount(lab[lab >= 0], minlength=r)[:r]), (ck, o, r, 'sizes'))
                    _check(int(g('unassigned')[ri]) == int((lab < 0).sum()), (ck, o, r, 'unassigned'))
                    for part in ('seba_part', 'kmeans_part'):
                        pl = g(part)[ri]
                        _check(((pl >= 0) & (pl < r)).all(), (ck, o, r, part, 'complete'))
                    for name, part in (('seba', 'seba_part'), ('kmeans', 'kmeans_part')):
                        u = float(g(f'ups_{name}')[ri])
                        empty = np.bincount(g(part)[ri], minlength=r).min() == 0
                        _check(math.isnan(u) == bool(empty), (ck, o, r, name, 'invalid iff empty'))
                    best = np.nanmax([g('ups_seba')[ri], g('ups_kmeans')[ri]]) if np.isfinite(
                        [g('ups_seba')[ri], g('ups_kmeans')[ri]]).any() else np.nan
                    _check((math.isnan(best) and math.isnan(g('ups')[ri])) or best == g('ups')[ri], (ck, o, r, 'ups'))
                    _check(np.isclose(g('cap')[ri], 2 * mu[r - 1] / mu[r - 2]), (ck, o, r, 'cap'))
                    records += 1
                for gname in groupings_of(o):
                    e = g(f'eta2_{gname}')
                    _check(e.shape == (K.N_ETA,) and np.all((e >= -1e-12) & (e <= 1 + 1e-12)), (ck, o, gname, 'eta2'))
                    _check(0 < float(g(f'eta2_{gname}_chance')) < 1, (ck, o, gname, 'eta2 chance'))
                    _check(g(f'eta2_{gname}_band').shape == (2, K.N_ETA), (ck, o, gname, 'eta2 band'))
                    _check(g(f'ami_{gname}').shape == (2,), (ck, o, gname, 'ami'))
                _check(g('ends').shape == (K.N_ETA, 2, K.N_ENDS), (ck, o, 'ends'))
                _check(g('pn').shape == (8,) and np.all(g('pn') >= 1), (ck, o, 'pn'))
                if o in sets:
                    for r, desc in sets[o].items():
                        _check(len(desc) == int(r) and sum(d['size'] for d in desc) ==
                               n - int(g('unassigned')[K.R.index(int(r))]), (ck, o, r, 'set descriptors'))
    rows = json.loads((root / 'compare.json').read_text())['rows']
    for r in rows:
        _check(-1e-9 <= r['a8'] <= 1 + 1e-9 and -1e-9 <= r['a16'] <= 1 + 1e-9, ('compare', r))
    ups = json.loads((root / 'upsilon.json').read_text())
    _check(ups['certification']['enabled'] == certify and ups['theta'] == theta, 'upsilon.json certification')
    for ck, by in ups['by_checkpoint'].items():
        for o, rows_o in by.items():
            for r in rows_o:
                _check(r['outcome'] == ST.outcome(r['ups'], r['cap'], theta, certify), (ck, o, r['k'], 'outcome'))
                _check(certify or r['outcome'] != 'certified', (ck, o, r['k'], 'certified while disabled'))
    trees = json.loads((root / 'trees.json').read_text())
    _check([t['id'] for t in trees] == [t['id'] for t in K.TREES], 'tree ids')
    for t in trees:
        nodes = t['nodes']
        _check(nodes[0]['n'] == n, (t['id'], 'root size'))
        for nd in nodes:
            if nd['children']:
                a, b = (set(nodes[c]['members']) for c in nd['children'])
                _check(not (a & b) and a | b == set(nd['members']), (t['id'], nd['id'], 'partition'))
                _check(min(len(a), len(b)) >= Z.m_min(nd['n']), (t['id'], nd['id'], 'm_min'))
                _check(nd['cheeger_lo'] <= nd['phi_free'] * (1 + 1e-9) + 1e-12 and
                       nd['phi_free'] <= nd['cheeger_hi'] * (1 + 1e-9) + 1e-12, (t['id'], nd['id'], 'cheeger'))
                _check(len(nd['gap']) >= 1 and len(nd['psi']) == 3, (t['id'], nd['id'], 'node arrays'))
            _check(sum(nd['class_t']) == nd['n'] and sum(nd['source']) == nd['n'], (t['id'], nd['id'], 'makeup'))
    _check(trees[1]['root_gram_dev'] < 1e-8, 'T2 Gram route at the root')
    lags = json.loads((root / 'lags.json').read_text())
    _check(set(lags['views']) == {f'{v}_L{l}' if l else v for v, l in LAG_VIEWS}, 'lag views')
    for key, rec in lags['views'].items():
        _check(np.asarray(rec['per_vector']).shape == (8, len(K.LAGS)), ('lags', key))
    if require_layouts:
        with np.load(root / 'layouts.npz') as z:
            _check(set(z.files) == set(LAYOUT_OBJECTS), 'layout objects')
            _check(all(z[k].shape == (n, 2) and np.isfinite(z[k]).all() for k in z.files), 'layouts')
    with np.load(root / 'items.npz') as z:
        _check(z['ctx'].shape == (n, sum(meta['context'])) and z['top5'].shape == (n, 5), 'items')
        _check(all(z[k].shape == (n,) for k in K.GROUPINGS), 'item groupings')
    vocab = json.loads((root / 'vocab.json').read_text())
    _check(not vocab['missing'], ('vocab missing', vocab['missing'][:10]))
    targets_ = json.loads((root / 'targets.json').read_text())
    _check(all(isinstance(v.get('met'), bool) for v in targets_.values()), 'targets')
    _check(targets_['K5']['certification_enabled'] == certify and (certify or (not targets_['K5']['met'] and
                                                                                'status' in targets_['K5'])),
           'K5 certification')
    return dict(complete=True, n=n, spectral_records=records, minimum_seba_label_agreement=worst, theta=theta)


if __name__ == '__main__':
    torch.set_num_threads(int(next((a.split('=')[1] for a in sys.argv if a.startswith('--threads=')), 1)))
    if sys.argv[1] == 'fixture':
        cal = json.loads(Path(sys.argv[3]).read_text())
        bench = Path(sys.argv[3]).parent / 'bench.json'          # the full-size check (SPEC 8) comes from the bench
        if 'full_size_check' not in cal and bench.is_file():
            cal['full_size_check'] = json.loads(bench.read_text()).get('calib_full')
        print(json.dumps(K.jsonable(build_fixture(sys.argv[2], cal, flat='--flat' in sys.argv,
                                                  layouts_on='--no-layouts' not in sys.argv))))
    elif sys.argv[1] == 'validate':
        print(json.dumps(K.jsonable(validate(sys.argv[2], require_layouts='--no-layouts' not in sys.argv))))
    else:
        raise SystemExit(__doc__)
