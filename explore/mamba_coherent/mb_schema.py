"""Measurement drivers, fixtures and validation for the Mamba study (SPEC sections 4-7 and the Pythia SPEC 8, 10).

FORK of `../lm_coherent/lm_schema.py`. What changes:
  - checkpoints are labels (`init`, `trained`) instead of training steps; no training axis (no train.npz, no
    op128 files); views are point/state/stack (+ bag); the state and stack cells come from Grams like the Pythia
    slab did;
  - blocks (`blocks_trained.npz`): reading blocks for every view at each layer, depth blocks (point), trained only;
  - comparisons: point_state, point_stack, state_stack, real_vs_S (three views and the bag), to_final (three
    views), state_parts (the D-M1 diagnostic); `cross_arch.py` adds cross_arch.json post hoc;
  - trees T1..T4 of SPEC 6: T4 is a Gram-based cell (mb_cells.RawGramCell) through lm_tree's generalised provider;
  - the fixture feeds the state and stack views from synthetic constituents (two per layer, 24 layers, unequal item
    norms and non-zero means) through `mb_cells.position_grams`, the code path of U1;
  - P3 adds the Gram identity (U1 record), and the Pythia linearity check runs on the point AND stack blocks;
  - validation for this schema ('mamba-toy-1').
Reused from lm_schema unchanged by import: Probe, the cell/block helpers, p2_check, calibrate (P1), the planted
fixture generators, the record/statistics validators and pcs.

WORK files of a checkpoint (U1 = mb_model.run_u1, or `fixture_inputs`):
    act_<c>.npy     float32 [layer, position, n, D]   point activations (probe A)
    state_<c>.npy   float64 [layer, position, n, n]   state Grams;   stack_<c>.npy likewise (stack Grams)
    bag_<c>.npy     float64 [position, n, Dv]         bag vectors
    trained only:   parts_trained.npy float64 [layer, position, 2 (c, z), n, n] constituent cosine Grams;
                    raw_trained.npy float64 [position, 48, n, n] raw constituent Grams (c^1, z^1, c^2, ...; T4);
                    actS_ / stateS_ / stackS_trained.npy at t = 128 for probe S ([layer, n, D] / [layer, n, n])
    u1_<c>.json     C1a (trained), C1b, C2, P3_gram_identity (trained), zero-norm scan, norm_mean (trained)
Public files (`data/m130/`):
    cells_<c>.npz   as lm_coherent with view axis (point, state, stack); rogue NaN for state/stack
    single_<c>.npz  as lm_coherent with view axis (point, state, stack)
    blocks_trained.npz  reading_<view>_<key> [layer, ...] for view in point/state/stack; depth_point_<key>
                    [position, ...]; keys as lm_coherent
    compare.json    rows: kind, checkpoint, view, part, layer, t, a3, a6, jacc, jacc_null, W_a, W_b
    trees.json      T1..T4 then P1 (as lm_coherent; each tree also has `view`; T4 has `root_gram_dev`)
    figdata.npz     as lm_coherent at (L16, t = 128) and T1
    umap.npz        grid [bag + 6 layers, position, n, 2] (point), grid_state / grid_stack [layer, position, n, 2],
                    strip [layer, checkpoint, n, 2] (point, t = 128)
    meta.json, validation.json, probe.json (copied verbatim from the Pythia run by U0)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mb_construction as MB  # noqa: E402
import mb_cells as MC  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_schema as S  # noqa: E402
import lm_tree as Z  # noqa: E402
from schema_v2 import RANKS  # noqa: E402

L, P, NC, V = len(MB.LAYERS), len(MB.POSITIONS), len(MB.CHECKPOINTS), len(MB.VIEWS)
STAT_KEYS, RECORD_KEYS, DEPTH_KEYS = S.STAT_KEYS, S.RECORD_KEYS, S.DEPTH_KEYS
CELL_KEYS, CELL_DTYPES = S.CELL_KEYS, S.CELL_DTYPES
Probe, _json, _cell, _cell_row, _block = S.Probe, S._json, S._cell, S._cell_row, S._block
R4 = RANKS.index(4)
PAIR_KINDS = [('point_state', 'point', 'state'), ('point_stack', 'point', 'stack'), ('state_stack', 'state', 'stack')]


def _row(kind, checkpoint, layer, t, a3, a6, jacc, null, view=None, part=None, W_a=None, W_b=None, **extra):
    return dict(kind=kind, checkpoint=checkpoint, view=view, part=part, layer=layer, t=t, a3=float(a3), a6=float(a6),
                jacc=float(jacc), jacc_null=[float(x) for x in null], W_a=W_a, W_b=W_b, **extra)


def _compare(rec_a, va, rec_b, vb, perms):
    j, null, _ = C.matched_jaccard(rec_a['seba_label'][R4], rec_b['seba_label'][R4], 4, 4, perms)
    return C.span_agreement(va, vb, 3), C.span_agreement(va, vb, 6), j, null


def _gram_cell(g, tokens_t, probe, perms):
    return _cell(torch.from_numpy(np.array(g)), tokens_t, probe, perms)


# ------------------------------------------------------------------------------------------------ U0: probe
def tokenizer_check(probe_json, tok_m, tok_p, vocab=MB.VOCAB):
    """U0 (SPEC 2, as implemented; see README DEVIATIONS 1). The item correspondence with the Pythia study rests on
    feeding both models the SAME token ids with the SAME meaning. Checks: (a) the Mamba and Pythia tokenizers'
    backends are identical (vocabulary, merges, normaliser, pre-tokeniser, decoder, added tokens; only the stored
    truncation/padding settings may differ); (b) every stored id < vocab (50,280) and < len(tokenizer); (c) decoding
    the stored ids with clean-up off reproduces every stored text and token string exactly. Also recorded: the
    SPEC's literal re-encoding check (text -> ids), which also fails under Pythia's own tokenizer wherever a
    snippet ends inside a merge (e.g. '\\n', '\\n' -> '\\n\\n'); we require Mamba's failures to equal Pythia's."""
    jm, jp = json.loads(tok_m.backend_tokenizer.to_str()), json.loads(tok_p.backend_tokenizer.to_str())
    differ = sorted(k for k in set(jm) | set(jp) if k not in ('truncation', 'padding') and jm.get(k) != jp.get(k))
    items = probe_json['items']
    max_id = max(max(it['tokens']) for it in items)
    dec = lambda ids: tok_m.decode(ids, clean_up_tokenization_spaces=False)
    decode_bad = [it['id'] for it in items if dec(it['tokens']) != it['text']]
    token_str_bad = [it['id'] for it in items if [dec([t]) for t in it['tokens']] != it['token_str']]
    reencode = lambda tok: [it['id'] for it in items if tok(it['text'], add_special_tokens=False)['input_ids'] !=
                            it['tokens']]
    rm, rp = reencode(tok_m), reencode(tok_p)
    default_decode_differs = sum(tok_m.decode(it['tokens']) != it['text'] for it in items)
    checks = dict(backend_identical=not differ, vocab_identical=tok_m.get_vocab() == tok_p.get_vocab(),
                  ids_below_vocab=bool(max_id < vocab and max_id < len(tok_m)), decode_exact=not decode_bad,
                  token_str_exact=not token_str_bad, reencode_failures_match_pythia=rm == rp)
    return dict(checks=checks, **{'pass': all(checks.values())}, backend_keys_differing=differ, max_id=int(max_id),
                vocab=vocab, len_tokenizer=len(tok_m), decode_mismatch_items=decode_bad[:50],
                literal_spec_check=dict(definition='Mamba tokenizer re-encodes every snippet text to the stored ids',
                                        **{'pass': not rm}, mismatches=len(rm), items=rm,
                                        pythia_tokenizer_mismatches=len(rp)),
                clean_up_tokenization_spaces=dict(mamba=bool(tok_m.clean_up_tokenization_spaces),
                                                  pythia=bool(tok_p.clean_up_tokenization_spaces),
                                                  items_whose_default_decode_differs=int(default_decode_differs)),
                n_items=len(items))


def u0_main(probe_src, public, cache_dir=None):
    """U0: copy the Pythia probe verbatim (sha256 pinned), then `tokenizer_check`. Raises on failure."""
    import shutil
    from transformers import AutoTokenizer
    public = Path(public)
    public.mkdir(parents=True, exist_ok=True)
    sha = MB.probe_identity(Path(probe_src).read_bytes())
    assert sha == MB.PROBE_SHA, f'probe sha {sha} is not the Pythia probe {MB.PROBE_SHA}'
    shutil.copyfile(probe_src, public / 'probe.json')
    assert MB.probe_identity((public / 'probe.json').read_bytes()) == MB.PROBE_SHA
    tok_m = AutoTokenizer.from_pretrained(MB.MODEL, revision=MB.REVISION, cache_dir=cache_dir)
    tok_p = AutoTokenizer.from_pretrained(MB.PYTHIA_TOKENIZER[0], revision=MB.PYTHIA_TOKENIZER[1], cache_dir=cache_dir)
    out = tokenizer_check(json.loads((public / 'probe.json').read_text()), tok_m, tok_p)
    out['probe_sha'] = sha
    return out


# -------------------------------------------------------------------------------------- U2: a checkpoint
def measure_checkpoint(work, public, checkpoint, probe, log=print, raw_in_memory=False):
    """U2: every cell, single-cell record, block, comparison and tree of one checkpoint."""
    work, public = Path(work), Path(public)
    start = time.time()
    final = checkpoint == MB.FINAL
    act = np.load(work / f'act_{checkpoint}.npy', mmap_mode='r')
    grams = {v: np.load(work / f'{v}_{checkpoint}.npy', mmap_mode='r') for v in ('state', 'stack')}
    bag = np.load(work / f'bag_{checkpoint}.npy', mmap_mode='r')
    n = probe.n
    assert act.shape[:3] == (L, P, n) and bag.shape[:2] == (P, n)
    assert all(g.shape == (L, P, n, n) for g in grams.values())
    perms = C.permutation_bank(n)
    tok = {t: probe.tokens[:, t - 1] for t in MB.POSITIONS}       # the token just read at position t
    grid = {}
    for view in MB.VIEWS:
        for li in range(L):
            for pi, t in enumerate(MB.POSITIONS):
                if view == 'point':
                    x = act[li, pi]
                    grid[view, li, pi] = _cell(C.standard_gram(x), tok[t], probe, perms, C.rogue_shares(x))
                else:
                    grid[view, li, pi] = _gram_cell(grams[view][li, pi], tok[t], probe, perms)
        log(f'{checkpoint}: {view} cells done, {time.time() - start:.0f}s')
    bags = [_cell(C.standard_gram(bag[pi]), tok[t], probe, perms, C.rogue_shares(bag[pi]))
            for pi, t in enumerate(MB.POSITIONS)]
    lc = MB.LAYERS.index(MB.COMPARE_LAYER)
    p3 = {}
    for view, li in (('point', lc), ('stack', L - 1)):
        rec = S.p3_check([grid[view, li, pj] for pj in range(P)], probe, perms)
        rec['block'] = f'reading L{MB.LAYERS[li]} ({view})'
        p3[view] = rec
    p3['pass'] = all(r['pass'] for r in p3.values())
    _json(work / f'p3_{checkpoint}.json', p3)

    # single cells: retention along positions (rows) and along layers (columns)
    single, vecs = {}, {}
    for view in MB.VIEWS:
        for li in range(L):
            row = [grid[view, li, pj] for pj in range(P)]
            for pi in range(P):
                col = [grid[view, lj, pi].m for lj in range(L)]
                rec, v, _ = _block(row, pi, pi, probe, perms, col, li)
                single[view, li, pi], vecs[view, li, pi] = rec, v
    for pi in range(P):
        rec, v, _ = _block(bags, pi, pi, probe, perms)
        single['bag', pi], vecs['bag', pi] = rec, v
    log(f'{checkpoint}: single cells done, {time.time() - start:.0f}s')

    cells_out, single_out = {}, {}
    rows = {view: [[_cell_row(grid[view, li, pi], single[view, li, pi]) for pi in range(P)] for li in range(L)]
            for view in MB.VIEWS}
    for key in CELL_KEYS:
        arr = np.array([[[rows[v][li][pi][key] for pi in range(P)] for li in range(L)] for v in MB.VIEWS])
        cells_out[key] = arr.astype(CELL_DTYPES.get(key, np.float32))
        barr = np.array([_cell_row(bags[pi], single['bag', pi])[key] for pi in range(P)])
        cells_out['bag_' + key] = barr.astype(CELL_DTYPES.get(key, np.float32))
    pr = [[C.export_statistics([grid['point', li, pi].raw]) for pi in range(P)] for li in range(L)]
    cells_out['own_src'] = np.array([[pr[li][pi]['own_src'] for pi in range(P)] for li in range(L)], np.uint8)
    cells_out['follow_mass'] = np.array([[pr[li][pi]['follow_mass'] for pi in range(P)] for li in range(L)],
                                        np.uint8).transpose(2, 0, 1, 3)
    if final:
        u1 = json.loads((work / f'u1_{checkpoint}.json').read_text())
        cells_out['norm_mean'] = np.asarray(u1['norm_mean'], np.float32)
    np.savez_compressed(public / f'cells_{checkpoint}.npz', **cells_out)
    for key in RECORD_KEYS + DEPTH_KEYS:
        single_out[key] = np.stack([np.stack([np.stack([single[view, li, pi][key] for pi in range(P)])
                                              for li in range(L)]) for view in MB.VIEWS])
        single_out['bag_' + key] = np.stack([single['bag', pi][key] for pi in range(P)])
    np.savez_compressed(public / f'single_{checkpoint}.npz', **single_out)

    rows_c, trees = [], []
    if final:
        blocks(public, grid, probe, perms)
        log(f'{checkpoint}: blocks done, {time.time() - start:.0f}s')
        for kind, a, b in PAIR_KINDS:
            for li, layer in enumerate(MB.LAYERS):
                for pi, t in enumerate(MB.POSITIONS):
                    rows_c.append(_row(kind, checkpoint, layer, t, *_compare(single[a, li, pi], vecs[a, li, pi],
                                                                             single[b, li, pi], vecs[b, li, pi],
                                                                             perms)))
        rows_c += real_vs_shuffled(work, checkpoint, probe, perms, grid, single, vecs, bags)
        rows_c += state_parts(work, checkpoint, probe, perms, single, vecs)
        log(f'{checkpoint}: comparisons done, {time.time() - start:.0f}s')
        trees, t4 = grow_trees(work, public, checkpoint, probe, grid, act, grams, raw_in_memory, log)
        _json(work / f't4check_{checkpoint}.json', t4)
    _json(work / f'compare_{checkpoint}.json', rows_c)
    _json(work / f'trees_{checkpoint}.json', trees)
    seconds = time.time() - start
    log(f'{checkpoint}: complete, {seconds:.0f}s')
    return dict(checkpoint=checkpoint, seconds=seconds, sinkhorn_max_row_err=float(
        max(c.info['row_error'] for c in list(grid.values()) + bags)))


def blocks(public, grid, probe, perms):
    """blocks_trained.npz: reading blocks (every view, each layer) and depth blocks (point, each position)."""
    out = {}

    def put(prefix, recs):
        for key in RECORD_KEYS + ['pn', 'min_set_r4']:
            out[f'{prefix}_{key}'] = np.stack([r[0][key] for r in recs])
        for key in STAT_KEYS + ['own_src', 'follow_mass']:
            out[f'{prefix}_{key}'] = np.stack([r[2][key] for r in recs]).astype(
                np.uint8 if key in ('own_src', 'follow_mass') else np.float32)
    for view in MB.VIEWS:
        put(f'reading_{view}', [_block([grid[view, li, pj] for pj in range(P)], 0, P - 1, probe, perms)
                                for li in range(L)])
    put('depth_point', [_block([grid['point', lj, pi] for lj in range(L)], 0, L - 1, probe, perms) for pi in range(P)])
    np.savez_compressed(Path(public) / f'blocks_{MB.FINAL}.npz', **out)


def real_vs_shuffled(work, checkpoint, probe, perms, grid, single, vecs, bags):
    """Probe S (tokens 1..127 permuted, token 128 kept) against probe A at t = 128: every view and the bag."""
    pi = MB.POSITIONS.index(MB.SEQ)
    tok = probe.tokens[:, MB.SEQ - 1]
    src = dict(point=np.load(work / f'actS_{checkpoint}.npy', mmap_mode='r'),
               state=np.load(work / f'stateS_{checkpoint}.npy', mmap_mode='r'),
               stack=np.load(work / f'stackS_{checkpoint}.npy', mmap_mode='r'))
    rows = []
    for view in MB.VIEWS:
        for li, layer in enumerate(MB.LAYERS):
            if view == 'point':
                cell = _cell(C.standard_gram(src[view][li]), tok, probe, perms)
            else:
                cell = _gram_cell(src[view][li], tok, probe, perms)
            rec, v = C.spectral_record([cell.m], (0, 0))
            rows.append(_row('real_vs_S', checkpoint, layer, MB.SEQ,
                             *_compare(single[view, li, pi], vecs[view, li, pi], rec, v, perms), view=view,
                             W_a=float(grid[view, li, pi].raw['W']), W_b=float(cell.raw['W'])))
    # bag: identical by C2 (asserted bitwise in U1), so S's bag cell IS A's bag cell
    a = single['bag', pi]
    j, null, _ = C.matched_jaccard(a['seba_label'][R4], a['seba_label'][R4], 4, 4, perms)
    rows.append(_row('real_vs_S', checkpoint, 'bag', MB.SEQ, 1.0, 1.0, j, null, view='bag',
                     W_a=float(bags[pi].raw['W']), W_b=float(bags[pi].raw['W'])))
    return rows


def state_parts(work, checkpoint, probe, perms, single, vecs):
    """D-M1 diagnostic (SPEC 4): each constituent alone (c-only, z-only) against the state view, every (layer, t)."""
    parts = np.load(Path(work) / f'parts_{checkpoint}.npy', mmap_mode='r')
    rows = []
    for li, layer in enumerate(MB.LAYERS):
        for pi, t in enumerate(MB.POSITIONS):
            for k, part in enumerate(MB.PARTS):
                p, _ = C.walk(torch.from_numpy(np.array(parts[li, pi, k])))
                rec, v = C.spectral_record([C.lazy(p)], (0, 0))
                rows.append(_row('state_parts', checkpoint, layer, t,
                                 *_compare(single['state', li, pi], vecs['state', li, pi], rec, v, perms),
                                 view='state', part=part))
    return rows


def grow_trees(work, public, checkpoint, probe, grid, act, grams, raw_in_memory=False, log=print):
    """SPEC 6. Point trees use vectors (re-centred over S); T4 (stack) uses the raw constituent Grams."""
    trees, t4 = [], {}
    raw = None
    for spec in MB.TREES:
        if spec['checkpoint'] != checkpoint:
            continue
        li = MB.LAYERS.index(spec['layer'])
        pis = list(range(P)) if spec['kind'] == 'reading' else [MB.POSITIONS.index(spec['position'])]
        positions = [MB.POSITIONS[pi] for pi in pis]
        if spec['view'] == 'point':
            cells = [np.asarray(act[li, pi]) for pi in pis]
        elif spec['view'] == 'stack':
            if raw is None:
                raw = np.load(Path(work) / f'raw_{checkpoint}.npy', mmap_mode=None if raw_in_memory else 'r')
            k = 2 * spec['layer']                                  # constituents of layers 1..l
            cells = [MC.RawGramCell(raw[pi, :k]) for pi in pis]
            everyone = np.arange(probe.n)
            dev = [float((cell.subset_gram(everyone) - torch.from_numpy(np.array(grams['stack'][li, pi]))).abs().max())
                   for cell, pi in zip(cells, pis)]
            t4 = dict(tree=spec['id'], root_gram_dev=max(dev), by_position=dev, tolerance=MB.T4_ROOT_TOL,
                      definition='max |Gram route on all items - stored stack Gram| (raw Grams vs vector route)',
                      **{'pass': bool(max(dev) < MB.T4_ROOT_TOL)})
            log(f"{spec['id']}: Gram route at the root deviates {max(dev):.2e} from the stored stack Grams")
            assert t4['pass'], t4
        else:
            raise ValueError(spec)
        nodes = Z.grow(cells, probe.sources)
        tree = dict(id=spec['id'], block=dict(kind=spec['kind'], view=spec['view'], layer=spec['layer'],
                                              positions=positions), checkpoint=checkpoint, nodes=Z.to_json(nodes))
        if spec['view'] == 'stack':
            tree['root_gram_dev'] = t4['root_gram_dev']
        trees.append(tree)
        log(f"{spec['id']}: {len(nodes)} nodes")
        if spec['id'] == 'T1':
            figdata(public, grid, li, nodes, cells)
    return trees, t4


def figdata(public, grid, li, nodes, cells, size=250):
    """As lm_schema.figdata, with this study's layer label (C, K, P of the point cell (L16, 128); Mbar of T1)."""
    root = nodes[0]
    psi = root['psi'][root['axis_used'] - 1].astype(float)
    side0 = set(nodes[root['children'][0]]['members'].tolist())
    members = root['members']
    order = members[np.array(sorted(range(len(members)), key=lambda k: (members[k] not in side0, -psi[k])))]
    c = C.standard_gram(cells[-1])
    k = torch.exp(C.knn_logkernel(c))
    p = 2 * grid['point', li, P - 1].m - torch.eye(len(c), dtype=torch.float64)
    mbar = torch.stack([grid['point', li, pj].m for pj in range(P)]).mean(0)

    def thumb(a):
        a = np.asarray(a)[np.ix_(order, order)]
        parts = np.array_split(np.arange(len(a)), size)
        return np.array([[a[np.ix_(r, q)].mean() for q in parts] for r in parts]).astype(np.float16)
    np.savez_compressed(Path(public) / 'figdata.npz', C=thumb(c), K=thumb(k), P=thumb(p), Mbar=thumb(mbar),
                        M_cells=np.stack([thumb(grid['point', li, pj].m) for pj in range(P)]),
                        cells_t=np.array(MB.POSITIONS, np.int16), order=order.astype(np.int16),
                        cell=np.array([MB.LAYERS[li], MB.SEQ], np.int16))


# ------------------------------------------------------------------------------------------- U3: finalize
def _decode(arrs, idx, n):
    return np.concatenate([np.full((n, 1), 1 / n ** .5), arrs['vecs'][idx].T.astype('f8') * arrs['scale'][idx] / 32767], 1)


def to_final_rows(public, probe):
    """Each checkpoint's single cells at (L16, t) against the trained ones, every view (trained vs itself is the
    reference row, as in the Pythia study)."""
    perms = C.permutation_bank(probe.n)
    li = MB.LAYERS.index(MB.COMPARE_LAYER)
    final = np.load(Path(public) / f'single_{MB.FINAL}.npz')
    rows = []
    for ck in MB.CHECKPOINTS:
        s = np.load(Path(public) / f'single_{ck}.npz')
        for vi, view in enumerate(MB.VIEWS):
            for t in MB.TO_FINAL_POSITIONS:
                idx = (vi, li, MB.POSITIONS.index(t))
                j, null, _ = C.matched_jaccard(s['seba_label'][idx][R4], final['seba_label'][idx][R4], 4, 4, perms)
                va, vb = (torch.from_numpy(_decode(z, idx, probe.n)) for z in (s, final))
                rows.append(_row('to_final', ck, MB.COMPARE_LAYER, t, C.span_agreement(va, vb, 3),
                                 C.span_agreement(va, vb, 6), j, null, view=view))
    return rows


def finalize(work, public, probe, fixture=False, layouts=True, log=print, extra_meta=None):
    """U3: merged comparisons, trees (with P1 and the threshold), P2, layouts, meta, validation."""
    work, public = Path(work), Path(public)
    rows = sum((json.loads((work / f'compare_{c}.json').read_text()) for c in MB.CHECKPOINTS), [])
    rows += to_final_rows(public, probe)
    _json(public / 'compare.json', dict(rows=rows))
    p1 = S.calibrate(log)
    p2 = S.p2_check()
    log(f'P2: pass={p2["pass"]}')
    p3 = {c: json.loads((work / f'p3_{c}.json').read_text()) for c in MB.CHECKPOINTS}
    trees = sum((json.loads((work / f'trees_{c}.json').read_text()) for c in MB.CHECKPOINTS), [])
    order = [t['id'] for t in MB.TREES]
    trees.sort(key=lambda t: order.index(t['id']))
    for t in trees:
        t['threshold'] = p1['threshold']
    _json(public / 'trees.json', trees + [p1])
    if layouts:
        umap_layouts(work, public)
    u1 = {c: json.loads((work / f'u1_{c}.json').read_text()) for c in MB.CHECKPOINTS}
    t4 = json.loads((work / f't4check_{MB.FINAL}.json').read_text())
    meta = dict(schema='mamba-toy-1', construction_fingerprint=MB.FINGERPRINT, model=MB.MODEL, revision=MB.REVISION,
                transformers_version=u1[MB.FINAL].get('transformers_version'),
                torch_version=u1[MB.FINAL].get('torch_version'), checkpoints=MB.CHECKPOINTS, layers=MB.LAYERS,
                n_layers=MB.N_LAYERS, positions=MB.POSITIONS, views=MB.VIEWS, parts=MB.PARTS, k=MB.K, seba_r=RANKS,
                seba_r_shown=MB.SEBA_R_SHOWN, sources=MB.SOURCES, n=probe.n, items=probe.items.tolist(),
                removed_items=probe.removed, follow_items=probe.follow.tolist(), trees=MB.TREES, n_perm=MB.N_PERM,
                perm_seed=MB.PERM_SEED, fixture=fixture, threshold=p1['threshold'], threshold_rule=p1['rule'],
                layout=u1[MB.FINAL].get('layout', 'AlignedUMAP'), model_config=u1[MB.FINAL].get('model_config'),
                parameter_sha={c: u1[c].get('parameter_sha') for c in MB.CHECKPOINTS}, init_seed=MB.INIT_SEED,
                cross_arch_pairs=MB.CROSS_ARCH_PAIRS, pythia_probe_sha=MB.PROBE_SHA)
    meta.update(extra_meta or {})
    _json(public / 'meta.json', meta)
    report = validate(public, require_layouts=layouts)
    gid = u1[MB.FINAL].get('P3_gram_identity')
    report.update(C1a=u1[MB.FINAL].get('C1a'), C1b={c: u1[c].get('C1b') for c in MB.CHECKPOINTS},
                  C2={c: u1[c].get('C2') for c in MB.CHECKPOINTS}, P2=p2,
                  P3=dict(gram_identity=gid, per_checkpoint=p3,
                          **{'pass': bool(all(r['pass'] for r in p3.values()) and gid and gid['pass'])}),
                  T4_gram_route=t4, zero_norm={c: u1[c].get('zero_norm', []) for c in MB.CHECKPOINTS},
                  removed_items=probe.removed,
                  duplicates='asserted per cell (min_dist > 1e-9) in U2; a violation fails the unit',
                  sinkhorn_failures=0,
                  P1=dict(exact=p1['match']['exact'], threshold=p1['threshold'], rule=p1['rule'],
                          planted_upsilon_min=min(p1['planted_upsilon']),
                          structureless_upsilon_max=max(x['upsilon'] for x in p1['structureless']),
                          **{'pass': bool(p1['match']['exact'] and p1['rule'] == 'fixed')}))
    _json(public / 'validation.json', report)
    return report


def umap_layouts(work, public):
    """AlignedUMAP (display only): point grid rows (bag, L4..L24) across positions, trained; state and stack grids
    from their Grams' principal coordinates; strip per layer across checkpoints at t = 128 (point)."""
    import umap
    work = Path(work)
    act = np.load(work / f'act_{MB.FINAL}.npy', mmap_mode='r')
    bag = np.load(work / f'bag_{MB.FINAL}.npy', mmap_mode='r')
    n = act.shape[2]

    def run(slices):
        model = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, alignment_regularisation=0.02, random_state=0)
        model.fit(slices, relations=[{i: i for i in range(n)} for _ in range(len(slices) - 1)])
        return np.stack(model.embeddings_).astype(np.float32)
    out = {}
    grid = [run([S.pcs(C.standard_gram(bag[pi])) for pi in range(P)])]
    grid += [run([S.pcs(C.standard_gram(act[li, pi])) for pi in range(P)]) for li in range(L)]
    out['grid'] = np.stack(grid)
    for view in ('state', 'stack'):
        g = np.load(work / f'{view}_{MB.FINAL}.npy', mmap_mode='r')
        out['grid_' + view] = np.stack([run([S.pcs(np.array(g[li, pi])) for pi in range(P)]) for li in range(L)])
    acts = [np.load(work / f'act_{c}.npy', mmap_mode='r') for c in MB.CHECKPOINTS]
    out['strip'] = np.stack([run([S.pcs(C.standard_gram(a[li, P - 1])) for a in acts]) for li in range(L)])
    np.savez_compressed(Path(public) / 'umap.npz', **out)


# ----------------------------------------------------------------------------------------------- fixtures
FIX_N, FIX_D = S.FIX_N, S.FIX_D
FIX_PART_DIMS = (16, 48)                  # c, z
FIX_DEMOTION_LAYER = 8                    # the Pythia fixture's L4, at the same index
FIX_AMP = {'init': 0.2, 'trained': 1.0}
FIXTURE_MODEL_CONFIG = dict(model_type='mamba', num_hidden_layers=24, hidden_size=768, intermediate_size=1536,
                            state_size=16, conv_kernel=4, time_step_rank=48, vocab_size=50280, residual_in_fp32=True,
                            rms_norm=True, layer_norm_epsilon=1e-05, use_conv_bias=True, use_bias=False,
                            tie_word_embeddings=True, hidden_act='silu', use_mambapy=False,
                            note='fixture: copied from the HF config of mamba-130m-hf at the pinned revision')


def synthetic_constituents(rng, b, n_layers=MB.N_LAYERS, amp=1.0, flat=False, dims=FIX_PART_DIMS):
    """Two synthetic constituents per layer (c, z) with the planted hierarchy `b` growing with depth, unequal item
    norms (log-normal per item and constituent) and a non-zero mean (so centring over S matters). Returns
    {m: (c, z)} for m = 1..n_layers, float32."""
    out = {}
    for m in range(1, n_layers + 1):
        pair = []
        for d in dims:
            x = rng.normal(size=(len(b), d)) * S.FIX_SIGMA / math.sqrt(d)
            if not flat:
                u = S._directions(rng, d)
                f = amp * min(1.0, 0.3 + m / n_layers)
                x = x + S.hierarchy_signal(b, u, tuple(f * a for a in S.FIX_AMPS))
            x = x * np.exp(0.3 * rng.normal(size=(len(b), 1))) + 2.0 * rng.normal(size=(1, d))
            pair.append(x.astype(np.float32))
        out[m] = tuple(pair)
    return out


def fixture_inputs(root_work, public, flat=False, seed=0, log=print):
    """Synthetic U1: the WORK files and probe.json for `_fixture` (planted) or `_fixture_flat`. The state and stack
    Grams go through `mb_cells.position_grams`, U1's code path."""
    work, public = Path(root_work), Path(public)
    work.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    leaf, b = S.hierarchy_codes(rng)
    u = S._directions(rng)
    bp = rng.permutation(np.arange(FIX_N) % 2) * 2 - 1
    bq = rng.permutation(np.arange(FIX_N) % 2) * 2 - 1
    probe_json = S.fixture_probe(rng, leaf)
    probe_json['planted'] = dict(leaf=leaf.tolist(), persistent=bp.tolist(), transient=bq.tolist(),
                                 sigma=S.FIX_SIGMA, amps=S.FIX_AMPS, flat=flat)
    if flat:
        probe_json['note'] += '; every cell and constituent is isotropic Gaussian noise (no planted structure)'
    _json(public / 'probe.json', probe_json)
    tokens = np.array([it['tokens'] for it in probe_json['items']])
    tokens_s = S.shuffled_tokens(probe_json)
    for ci, ck in enumerate(MB.CHECKPOINTS):
        final = ck == MB.FINAL

        def cell(li, t):
            if flat:
                return rng.normal(size=(FIX_N, FIX_D))
            x = S.noise(rng)
            if final and MB.LAYERS[li] == FIX_DEMOTION_LAYER:
                x += S.FIX_PERSISTENT * bp[:, None] * u[7]
                if t == S.FIX_TRANSIENT_T:
                    x += S.FIX_TRANSIENT * bq[:, None] * u[8]
            else:
                x += S.hierarchy_signal(b, u, tuple(FIX_AMP[ck] * min(1.0, (li + 2) / 5) * a for a in S.FIX_AMPS))
            return x
        act = np.empty((L, P, FIX_N, FIX_D), np.float32)
        state = np.empty((L, P, FIX_N, FIX_N)); stack = np.empty((L, P, FIX_N, FIX_N))
        parts = np.empty((L, P, 2, FIX_N, FIX_N)) if final else None
        raw = (np.lib.format.open_memmap(work / f'raw_{ck}.npy', mode='w+', dtype=np.float64,
                                         shape=(P, 2 * MB.N_LAYERS, FIX_N, FIX_N)) if final else None)
        norm = np.zeros((L, MB.SEQ + 1))
        gid = None
        for pi, t in enumerate(MB.POSITIONS):
            for li in range(L):
                act[li, pi] = cell(li, t).astype(np.float32)
            cons = synthetic_constituents(rng, b, amp=FIX_AMP[ck], flat=flat)
            state_of = lambda m: tuple(torch.from_numpy(x) for x in cons[m])
            g_state, g_stack, g_parts, g_raw = MC.position_grams(state_of, FIX_N, keep_parts=final, keep_raw=final)
            state[:, pi], stack[:, pi] = g_state, g_stack
            if final:
                parts[:, pi] = g_parts
                for k, g in enumerate(g_raw):
                    raw[pi, k] = g
                del g_raw
                if t == MB.SEQ:
                    gid = MC.gram_identity(state_of, g_state, g_stack)
                    cons_s = synthetic_constituents(rng, b, amp=FIX_AMP[ck], flat=flat)
                    s_state, s_stack, _, _ = MC.position_grams(lambda m: tuple(torch.from_numpy(x) for x in cons_s[m]),
                                                               FIX_N)
                    np.save(work / f'stateS_{ck}.npy', s_state)
                    np.save(work / f'stackS_{ck}.npy', s_stack)
        for li in range(L):
            norm[li] = np.linalg.norm(act[li, -1], axis=1).mean() * (1 + 0.1 * np.sin(np.arange(MB.SEQ + 1) / 9))
            norm[li, 0] = 30.0                                  # mock EOT norm
        np.save(work / f'act_{ck}.npy', act)
        np.save(work / f'state_{ck}.npy', state)
        np.save(work / f'stack_{ck}.npy', stack)
        emb = np.random.default_rng(1000 + ci).normal(size=(S.FIX_V, FIX_D)).astype(np.float32)
        bag = C.bag_vectors(tokens, emb)
        bag_s = C.bag_vectors(tokens_s, emb, [MB.SEQ])
        c2 = bool(np.array_equal(bag[-1], bag_s[0]))
        assert c2, 'C2 failed: bag view not identical for A and S at t = 128'
        np.save(work / f'bag_{ck}.npy', bag)
        not_run = dict(status='not run: fixture has no model', max_rel_err=None, **{'pass': None})
        u1 = dict(checkpoint=ck, C2=dict(identical=c2, max_abs_diff=float(np.abs(bag[-1] - bag_s[0]).max())),
                  C1b=not_run, zero_norm=[], transformers_version=None, model_config=FIXTURE_MODEL_CONFIG,
                  parameter_sha=None, layout='AlignedUMAP', fixture=True, construction_fingerprint=MB.FINGERPRINT)
        if final:
            raw.flush(); del raw
            np.save(work / f'parts_{ck}.npy', parts)
            u1.update(norm_mean=norm.tolist(), C1a=not_run, P3_gram_identity=gid)
            assert gid['pass'], gid
            np.save(work / f'actS_{ck}.npy', np.stack([cell(li, MB.SEQ) for li in range(L)]).astype(np.float32))
        _json(work / f'u1_{ck}.json', u1)
        log(f'fixture inputs {ck} written')
    return probe_json


def build_fixture(root, flat=False, layouts=True, log=print):
    """`data/_fixture/` or `data/_fixture_flat/` through the real U2/U3 code path (MOCKUP data)."""
    import os
    import shutil
    root = Path(root)
    work = Path(os.environ.get('TMPDIR', '/tmp')) / f'cm-mb-work{root.name}'
    try:
        fixture_inputs(work, root, flat=flat, log=log)
        probe = Probe(root / 'probe.json')
        for ck in MB.CHECKPOINTS:
            measure_checkpoint(work, root, ck, probe, log)
        report = finalize(work, root, probe, fixture='flat' if flat else 'planted', layouts=layouts, log=log)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return report


# --------------------------------------------------------------------------------------------- validation
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def validate(root, require_layouts=True):
    """Schema check of a data directory. Returns a report; raises AssertionError on any violation."""
    root = Path(root)
    meta = json.loads((root / 'meta.json').read_text())
    n = meta['n']
    _check(meta['schema'] == 'mamba-toy-1' and meta['construction_fingerprint'] == MB.FINGERPRINT, 'meta schema')
    _check(meta['checkpoints'] == MB.CHECKPOINTS and meta['layers'] == MB.LAYERS and
           meta['positions'] == MB.POSITIONS and meta['views'] == MB.VIEWS, 'axes')
    _check(meta['seba_r'] == RANKS and meta['sources'] == MB.SOURCES and len(meta['follow_items']) == MB.N_FOLLOW,
           'meta lists')
    _check(meta['revision'] == MB.REVISION and meta['model'] == MB.MODEL, 'model pin')
    probe = json.loads((root / 'probe.json').read_text())
    _check(len(probe['items']) == n + len(meta['removed_items']), 'probe size')
    for it in probe['items']:
        _check(len(it['tokens']) == MB.SEQ and len(it['token_str']) == MB.SEQ and it['source'] in MB.SOURCES,
               ('probe item', it['id']))
    _check(len(probe['S']['permutation']) == len(probe['items']), 'probe S')
    if not meta['fixture']:
        _check(MB.probe_identity((root / 'probe.json').read_bytes()) == MB.PROBE_SHA, 'probe is not the Pythia probe')
    worst, records = 1.0, 0
    for ck in MB.CHECKPOINTS:
        final = ck == MB.FINAL
        with np.load(root / f'cells_{ck}.npz') as z:
            S._check_stats(z, '', (V, L, P), n)
            S._check_stats(z, 'bag_', (P,), n)
            for key in ('min_dist', 'rogue1', 'rogue5', 'sinkhorn_iters', 'row_err'):
                _check(z[key].shape == (V, L, P) and z['bag_' + key].shape == (P,), (ck, key))
            _check(np.all(z['min_dist'] > C.DUPLICATE_TOL) and np.all(z['bag_min_dist'] > C.DUPLICATE_TOL),
                   (ck, 'duplicates'))
            _check(np.all(z['row_err'] < 2e-12) and np.all(z['bag_row_err'] < 2e-12), (ck, 'sinkhorn'))
            _check(np.isfinite(z['rogue1'][0]).all() and np.isnan(z['rogue1'][1:]).all(), (ck, 'rogue'))
            _check(z['own_src'].shape == (L, P, n) and z['own_src'].dtype == np.uint8, (ck, 'own_src'))
            _check(z['follow_mass'].shape == (MB.N_FOLLOW, L, P, 5) and z['follow_mass'].dtype == np.uint8,
                   (ck, 'follow'))
            _check(np.all(np.abs(z['follow_mass'].astype(int).sum(-1) - 255) <= 3), (ck, 'follow sums'))
            if final:
                _check(z['norm_mean'].shape == (L, MB.SEQ + 1), (ck, 'norm_mean'))
        with np.load(root / f'single_{ck}.npz') as z:
            w, k = S._check_records(z, '', (V, L, P), n, P, depth=True)
            w2, k2 = S._check_records(z, 'bag_', (P,), n, P, depth=True)
            _check(np.isfinite(z['retention_psi_depth']).all() and np.isnan(z['bag_retention_psi_depth']).all(),
                   (ck, 'depth retention'))
            worst, records = min(worst, w, w2), records + k + k2
        _check((root / f'blocks_{ck}.npz').exists() == final, (ck, 'blocks file only for trained'))
    with np.load(root / f'blocks_{MB.FINAL}.npz') as z:
        groups = [(f'reading_{v}_', L, P) for v in MB.VIEWS] + [('depth_point_', P, L)]
        _check(set(z.files) == {g + key for g, _, _ in groups for key in RECORD_KEYS + STAT_KEYS +
                                ['pn', 'min_set_r4', 'own_src', 'follow_mass']}, 'block keys')
        for g, m, nc in groups:
            w, k = S._check_records(z, g, (m,), n, nc)
            S._check_stats(z, g, (m,), n)
            worst, records = min(worst, w), records + k
    rows = json.loads((root / 'compare.json').read_text())['rows']
    count = lambda kind: len([r for r in rows if r['kind'] == kind])
    for kind, _, _ in PAIR_KINDS:
        _check(count(kind) == L * P, (kind, count(kind)))
    _check(count('real_vs_S') == V * L + 1, 'real_vs_S rows')
    _check(count('to_final') == NC * V * len(MB.TO_FINAL_POSITIONS), 'to_final rows')
    _check(count('state_parts') == len(MB.PARTS) * L * P, 'state_parts rows')
    _check(len(rows) == 3 * L * P + V * L + 1 + NC * V * len(MB.TO_FINAL_POSITIONS) + len(MB.PARTS) * L * P,
           'no other compare rows')
    for r in rows:
        _check(-1e-9 <= r['a3'] <= 1 + 1e-9 and -1e-9 <= r['a6'] <= 1 + 1e-9 and 0 <= r['jacc'] <= 1 and
               len(r['jacc_null']) == 3, ('compare', r))
    trees = json.loads((root / 'trees.json').read_text())
    _check([t['id'] for t in trees] == [t['id'] for t in MB.TREES] + ['P1'], 'tree ids')
    for t in trees:
        nodes = t['nodes']
        _check(nodes[0]['n'] == n if t['id'] != 'P1' else True, (t['id'], 'root size'))
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
    t4 = next(t for t in trees if t['id'] == 'T4')
    _check(t4['block']['view'] == 'stack' and t4['root_gram_dev'] < MB.T4_ROOT_TOL, 'T4 Gram route')
    if require_layouts:
        with np.load(root / 'umap.npz') as z:
            _check(z['grid'].shape == (L + 1, P, n, 2) and z['strip'].shape == (L, NC, n, 2), 'umap shapes')
            _check(z['grid_state'].shape == (L, P, n, 2) and z['grid_stack'].shape == (L, P, n, 2), 'umap views')
            _check(all(np.isfinite(z[k]).all() for k in z.files), 'umap finite')
    with np.load(root / 'figdata.npz') as z:
        for key in ('C', 'K', 'P', 'Mbar'):
            _check(z[key].shape == (250, 250) and z[key].dtype == np.float16 and np.isfinite(z[key]).all(), key)
        _check(z['M_cells'].shape == (P, 250, 250) and np.isfinite(z['M_cells']).all(), 'M_cells')
        _check(int(z['cell'][0]) == MB.COMPARE_LAYER, 'figdata cell')
    _check(isinstance(meta.get('model_config'), dict) and meta['model_config'].get('num_hidden_layers') == 24,
           'model_config')
    report = dict(complete=True, records=records, minimum_seba_label_agreement=worst, n=n,
                  p1_exact=trees[-1]['match']['exact'], threshold=trees[-1]['threshold'])
    if (root / 'cross_arch.json').exists():
        import cross_arch
        report['cross_arch'] = cross_arch.validate(root / 'cross_arch.json')
    return report


if __name__ == '__main__':
    torch.set_num_threads(int(next((a.split('=')[1] for a in sys.argv if a.startswith('--threads=')), 1)))
    if sys.argv[1] == 'fixture':
        print(json.dumps(build_fixture(sys.argv[2], flat='--flat' in sys.argv,
                                       layouts='--no-layouts' not in sys.argv), default=S._default))
    elif sys.argv[1] == 'u0':                       # u0 <probe.json> <out dir>: local CPU run of U0
        out = u0_main(sys.argv[2], sys.argv[3])
        print(json.dumps(out, indent=1, default=S._default))
        sys.exit(0 if out['pass'] else 1)
    elif sys.argv[1] == 'validate':
        print(json.dumps(validate(sys.argv[2], require_layouts='--no-layouts' not in sys.argv), default=S._default))
    else:
        raise SystemExit(__doc__)
