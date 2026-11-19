"""CPU tests of the token-context study (SPEC 8), run before any Modal unit.

    export TMPDIR=/var/tmp/cm-mc
    uv run -q --with torch --with numpy --with scipy --with scikit-learn --with transformers==4.57.6 \
        python -I test_mc.py

Covers: Upsilon_hat(2) with the sweep partition = lm_tree's Upsilon; the partition contract (ties, all-zero rows,
empty parts) and the fail-closed reading rule; streaming Grams and the running-mean block against the v1 functions;
walk shares against lm_cells.walk_share; the streaming tree against lm_tree; item de-duplication; the within-token
permutation null (predictions that depend on the token only, and on a planted context class); eta^2 on a planted
case; the snapshot and lag indexing against a direct decode on a tiny random Mamba (and the whole of U1, C3 and
P3 on it, then U2 and U3 on U1's own output); provenance and budget keys; the fixtures when built.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mc_construction as K  # noqa: E402
import mc_items as I  # noqa: E402
import mc_stats as ST  # noqa: E402
import mc_calib as CB  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_tree as Z  # noqa: E402
import mb_cells as MC  # noqa: E402
from coherent import block_modes  # noqa: E402

RESULTS = {}


def _clusters(n=300, k=4, d=12, s=1.2, seed=0, cells=3):
    rng = np.random.default_rng(seed)
    lab = rng.permutation(np.arange(n) % k)
    out = []
    for _ in range(cells):
        e = rng.normal(size=(k, d))
        out.append(s * e[lab] + rng.normal(size=(n, d)))
    return out, lab


# ------------------------------------------------------------------------------------- Upsilon, partitions
def test_upsilon2_equals_lm_tree():
    """Upsilon_hat(2) of the sweep cut = lm_tree's Upsilon = mu_2 / phi* (SPEC 8)."""
    cells, _ = _clusters(seed=1, k=2)
    members = np.arange(len(cells[0]))
    ps = Z.node_walks(cells, members)
    rec = Z.analyse_node(ps, members)
    side = np.isin(members, rec['children_members'][0]).astype(np.int64)
    pbar = torch.stack(ps).mean(0)
    vals, _ = block_modes([C.lazy(p) for p in ps], 5, shared_stationary=True)
    mu = ST.mus(vals.numpy())
    u, phi = ST.upsilon_hat(mu, 2, pbar, side)
    assert abs(u - rec['upsilon']) < 1e-10 * rec['upsilon'], (u, rec['upsilon'])
    assert abs(phi[0] - phi[1]) < 1e-14 and abs(phi[0] - rec['phi']) < 1e-12
    RESULTS['upsilon2_vs_lm_tree'] = abs(u - rec['upsilon'])


def test_partition_contract():
    s = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.1], [0.1, 0.7, 0.7], [0.0, 0.2, 0.9], [0.0, 0.0, 0.0]])
    assert ST.seba_partition(s).tolist() == [0, 0, 1, 2, 0]           # zero rows and ties -> lowest index
    pbar = torch.full((5, 5), 0.25, dtype=torch.float64)
    pbar.diagonal().fill_(0.0)
    mu = np.array([0, .5, .6, .7])
    u, phi = ST.upsilon_hat(mu, 3, pbar, np.array([0, 0, 1, 1, 1]))  # part 2 empty: invalid
    assert u is None and np.isnan(phi).all()
    u, phi = ST.upsilon_hat(mu, 2, pbar, np.array([0, 0, 1, 1, 1]))
    assert u is not None and abs(phi[0] - 6 * 0.25 / 2) < 1e-15        # cut = 2 * 3 * 0.25, min side 2
    # reading rule: three outcomes; NaN never certifies or excludes; dropped rule reads nothing
    assert ST.outcome(3.5, 9.0, 3.0) == 'certified'
    assert ST.outcome(1.0, 2.5, 3.0) == 'excluded'
    assert ST.outcome(1.0, 4.0, 3.0) == 'undetermined'
    assert ST.outcome(None, 4.0, 3.0) == 'undetermined' and ST.outcome(float('nan'), float('nan'), 3.0) == 'undetermined'
    assert ST.outcome(5.0, 9.0, None) is None
    # the cap bounds every partition's Upsilon (higher-order Cheeger) on a real walk
    cells, _ = _clusters(seed=2, k=4)
    bm = ST.BlockMean()
    for x in cells:
        bm.add(ST.cell_walk(ST.view_gram(x))[0])
    rec, _, vecs = ST.spectral_object(bm.mbar(), {'X': np.arange(300) % 3}, ST.permutation_bank(300, 5),
                                      C.permutation_bank(300, 5, 1))
    ok = np.isfinite(rec['ups'])
    assert np.all(rec['ups'][ok] <= rec['cap'][ok] * (1 + 1e-9)), (rec['ups'], rec['cap'])
    assert rec['ups'][K.R.index(4)] > 3, rec['ups']


# ------------------------------------------------------------------------------------- streaming vs v1
def _cons(seed, n=60, dims=(5, 13)):
    rng = np.random.default_rng(seed)
    out = {}
    for m in range(1, 25):
        out[m] = tuple(torch.from_numpy(rng.normal(size=(n, d)) * np.exp(0.8 * rng.normal(size=(n, 1)))
                                        + 3 * rng.normal(size=(1, d))) for d in dims)
    return out


def test_streaming_grams_and_block():
    cons = _cons(3)
    state, stack, _, _ = MC.position_grams(cons.__getitem__, 60)
    vg = ST.ViewGrams(60)
    mine = {m: vg.layer(*cons[m]) for m in range(1, 25)}
    dev_state = max(float((mine[l] - torch.from_numpy(state[li])).abs().max()) for li, l in enumerate(K.SINGLE_LAYERS))
    dev_whole = float((vg.whole() - torch.from_numpy(stack[-1])).abs().max())
    assert dev_state == 0.0 and dev_whole == 0.0, (dev_state, dev_whole)        # identical arithmetic
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(50, 7)) + 4).float()
    assert torch.equal(ST.unit_rows_(x), C.unit_rows(x)) and torch.equal(ST.view_gram(x), C.standard_gram(x))
    # running-mean block vs block_modes on a stacked list (v1)
    cells, _ = _clusters(seed=4)
    ps = [C.walk(C.standard_gram(x))[0] for x in cells]
    bm = ST.BlockMean()
    for p in ps:
        bm.add(p)
    va, wa = block_modes([C.lazy(p) for p in ps], 9, shared_stationary=True)
    vb, wb = ST.spectrum(bm.mbar(), 9)
    dv = float(np.abs(va.numpy() - vb).max())
    dw = float(np.abs(np.abs(wa.numpy()[:, 1:]) - np.abs(wb[:, 1:])).max())
    assert dv < 1e-13 and dw < 1e-8, (dv, dw)
    RESULTS['block_running_mean_vs_v1'] = dict(vals=dv, vecs=dw)


def test_shares_vs_v1():
    cells, lab = _clusters(seed=5, n=200)
    p = C.walk(C.standard_gram(cells[0]))[0]
    perms = C.permutation_bank(200, 7, 3)
    for labels in (lab, np.random.default_rng(0).integers(0, 40, 200)):
        s, s0, null = C.walk_share(p, labels, perms)
        # v1 permutes membership by reading P at (q(i), q(j)); the same draw is the labels c[argsort(q)]
        inv = np.argsort(perms, axis=1)
        s2, s02, null2 = ST.share_values(p, labels, inv)
        assert abs(s - s2) < 1e-14 and s0 == s02 and np.max(np.abs(null - null2)) < 1e-14


def test_streaming_tree_vs_lm_tree():
    cells, lab = _clusters(seed=6, n=240, k=3)
    members = np.arange(240)
    ps, mask = Z.node_operators(cells, members)
    ref = Z.analyse_node(ps, members, lab % 5, mask)
    spill = ST.Spill()
    try:
        pbar, mask2, names = ST.node_operators_streaming([ST.VectorCell(x) for x in cells], members, spill)
        rec = ST.analyse_node_streaming(pbar, members, spill, names, lab % 5, mask2)
    finally:
        spill.close()
    assert np.array_equal(mask, mask2)
    for key in ('n', 'axis_used', 'cut_m', 'm_min', 'seba2_unassigned', 'makeup', 'n_components'):
        assert rec[key] == ref[key], key
    for key in ('mu1', 'mu2', 'phi', 'phi_free', 'upsilon', 'cheeger_lo', 'seba2_vs_cut'):
        assert abs(rec[key] - ref[key]) < 1e-10 * max(1, abs(ref[key])), (key, rec[key], ref[key])
    assert np.allclose(rec['retention_cut'], ref['retention_cut'], atol=1e-12)
    assert all(np.array_equal(a, b) for a, b in zip(rec['children_members'], ref['children_members']))
    nodes = ST.grow_streaming([ST.VectorCell(x) for x in cells], sources=lab % 5, max_depth=2, min_n=40,
                              describe=lambda m: dict(size_check=len(m)))
    ref_nodes = Z.grow(cells, lab % 5, max_depth=2, min_n=40)
    assert [nd['members'].tolist() for nd in nodes] == [nd['members'].tolist() for nd in ref_nodes]
    assert all(nd['size_check'] == nd['n'] for nd in nodes)


def test_whole_cell_subset_rebuild():
    """T2's node rebuild (each constituent re-run on S) = the explicit concatenation on S (v1 P3 reference)."""
    cons = _cons(7, n=80)
    cell = ST.WholeCell(lambda: (cons[m] for m in range(1, 25)), 80)
    s = np.sort(np.random.default_rng(1).choice(80, 50, replace=False))
    ref = MC.explicit_concat_gram([x[s] for m in range(1, 25) for x in cons[m]])
    dev = float((cell.gram(s) - ref).abs().max())
    assert dev < 1e-12, dev
    RESULTS['whole_subset_rebuild'] = dev


# ------------------------------------------------------------------------------------------ items
def test_item_dedup():
    rng = np.random.default_rng(0)
    toks = rng.integers(1, 50, (40, K.SEQ))
    toks[4:12] = toks[3]                                # 9 identical snippets: 72 positions out of 97 windows
    toks[20, 60:] = toks[20, 20:88]                     # an internal repeat
    items = I.sample_items(toks)
    I.check_items(items, toks)
    assert items['redraws'] >= 20 and len(items['t']) == 320 and items['redraws'] == len(items['redraw_items'])
    for i, t, j, u in items['redraw_items']:            # every redraw was a real clash with an earlier item
        assert I.window_key(toks[i, t - 16:t]) == I.window_key(toks[j, u - 16:u]) and (j < i or (j == i and u != t))
    assert sorted(items['order'].tolist()) == list(range(320))
    again = I.sample_items(toks)
    assert all(np.array_equal(items[k], again[k]) for k in ('snippet', 't'))
    used = [set(items['t'][items['snippet'] == s].tolist()) for s in range(3, 12)]
    assert sum(len(u) for u in used) == len(set().union(*used)) == 72
    # a permuted window (same multiset, different order) also clashes: `recent` would be an exact duplicate
    toks2 = rng.integers(1, 50, (2, K.SEQ))
    toks2[1] = toks2[0][::-1]
    it2 = I.sample_items(toks2)
    I.check_items(it2, toks2)
    rec = I.recent_vectors(it2, toks2, rng.normal(size=(50, 4)))
    d = np.abs(rec[:, None] - rec[None]).sum(-1) + np.eye(16)
    assert d.min() > 0
    try:
        I.sample_items(np.ones((2, K.SEQ), int))
        raise AssertionError('constant snippets accepted')
    except I.DedupFailed:
        pass
    # contexts: 24 tokens before (ending at t) and 4 after
    ctx = I.contexts(items, toks)
    k = int(np.flatnonzero(items['t'] >= 30)[0])
    s, t = items['snippet'][k], items['t'][k]
    assert ctx[k].tolist() == toks[s, t - 24:t + 4].tolist()
    RESULTS['dedup_redraws_test'] = items['redraws']


# ------------------------------------------------------------------------------------- kappa null
def _kappa_case(depends_on_context, seed=0, n=600):
    rng = np.random.default_rng(seed)
    tok = rng.integers(0, 12, n)                        # token at t
    ctx = rng.integers(0, 2, n)                         # a planted context class, independent of the token
    snip = np.arange(n) // 4
    v = 40
    wt, wc = rng.normal(size=(12, v)) * 2, rng.normal(size=(2, v)) * 3
    logits = wt[tok] + (wc[ctx] if depends_on_context else 0) + 0.3 * rng.normal(size=(n, v))
    q = torch.exp(0.5 * torch.log_softmax(torch.from_numpy(logits), -1))
    bc = q @ q.T
    # the walk groups by token AND by the context class (as a residual that tracks the context would)
    x = np.concatenate([np.eye(12)[tok] * 2, np.eye(2)[ctx][:, [0, 1] * 3] * 1.5], 1) + 0.4 * rng.normal(size=(n, 18))
    p = C.walk(C.standard_gram(x))[0]
    within, unique = ST.within_group_perms(tok)
    ks = ST.kappa_summary(ST.kappa_stats(p, bc, snip, within), unique)
    return ks, within, tok


def test_kappa_conditional_null():
    token_only, within, tok = _kappa_case(False)
    assert all((tok[q] == tok).all() for q in within)
    lo, hi = token_only['kappaT_band']
    assert lo <= token_only['kappa'] <= hi or abs(token_only['kappa_ex_T']) < 0.01, token_only
    assert token_only['kappa_ex'] > 0.1                                # high predictive coherence via the token
    context, _, _ = _kappa_case(True)
    assert context['kappa'] > context['kappaT_band'][1] and context['kappa_ex_T'] > 0.05, context
    RESULTS['kappa_null'] = dict(token_only=dict(kappa=token_only['kappa'], band=list(token_only['kappaT_band']),
                                                 ex_T=token_only['kappa_ex_T']),
                                 context=dict(kappa=context['kappa'], band=list(context['kappaT_band']),
                                              ex_T=context['kappa_ex_T']))
    # same-snippet pairs are excluded: a walk inside snippets only has no mass to score
    n = 40
    snip = np.arange(n) // 4
    p = torch.zeros((n, n), dtype=torch.float64)
    for s in range(10):
        b = torch.full((4, 4), 1 / 3, dtype=torch.float64); b.fill_diagonal_(0)
        p[4 * s:4 * s + 4, 4 * s:4 * s + 4] = b
    ks = ST.kappa_stats(p, torch.ones((n, n), dtype=torch.float64), snip, [np.arange(n)])
    assert ks['mass_diff'] == 0.0 and math.isnan(ks['kappa'])


def test_eta2_planted():
    """eta^2 of an eigenfunction that is (nearly) a function of the planted grouping is near 1; against an
    unrelated grouping it sits in its permutation band around (G-1)/(N-1)."""
    cells, lab = _clusters(seed=8, n=400, k=2, s=2.0)
    bm = ST.BlockMean()
    for x in cells:
        bm.add(ST.cell_walk(ST.view_gram(x))[0])
    other = np.random.default_rng(3).integers(0, 10, 400)
    rec, _, vecs = ST.spectral_object(bm.mbar(), {'P': lab, 'O': other}, ST.permutation_bank(400),
                                      C.permutation_bank(400, 20, 1))
    assert rec['eta2_P'][0] > 0.95, rec['eta2_P'][:3]
    assert abs(rec['eta2_P_chance'] - 1 / 399) < 1e-15 and abs(rec['eta2_O_chance'] - 9 / 399) < 1e-15
    inside = (rec['eta2_O'] >= rec['eta2_O_band'][0] - 1e-12) & (rec['eta2_O'] <= rec['eta2_O_band'][1] + 1e-12)
    assert inside.mean() >= 0.75, inside
    assert rec['ends'].shape == (16, 2, 12)
    assert set(lab[rec['ends'][0, 0]]) != set(lab[rec['ends'][0, 1]])
    RESULTS['eta2_planted'] = float(rec['eta2_P'][0])


# ---------------------------------------------------------------------------------- tiny Mamba: U1
def _tiny():
    import mb_model as MM
    return MM, MM.init_model(MM.tiny_config())


def _tiny_probe(n_snip=30, seed=2):
    import lm_schema as LS
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, 512, (n_snip, K.SEQ))
    strs = [f' w{i}' if i % 3 else f'{i}' for i in range(512)]
    probe = LS.Probe(dict(items=[dict(id=i, source=K.SOURCES[i % 5], tokens=t.tolist(),
                                      token_str=[strs[x] for x in t]) for i, t in enumerate(toks)],
                          vocab_extra={str(i): s for i, s in enumerate(strs)}))
    probe.token_str = [it['token_str'] for it in probe.raw['items']]
    return probe


def test_snapshot_indexing_tiny():
    """Every buffer row of a pass at lag tau equals a direct (batch 1) decode of that item's snippet up to t + tau
    (float32, relative 1e-5: batched and unbatched matmuls differ in rounding), and equals bitwise the same
    quantities read from a batched decode that keeps everything."""
    import mc_decode as D
    MM, m = _tiny()
    probe = _tiny_probe(12)
    items = I.sample_items(probe.tokens)
    worst = 0.0
    for lag in (0, 4, 16):
        snap = D.snapshot_pass(m, probe.tokens, items, lag, 'cpu', K.LAYERS, predictions=lag == 0, batch=5,
                               log=lambda s: None)
        for k in np.random.default_rng(lag).choice(len(items['t']), 6, replace=False):
            ref = D.direct_snapshot(m, probe.tokens, items['snippet'][k], items['t'][k] + lag, 'cpu')
            for li, l in enumerate(K.LAYERS):
                worst = max(worst, MM._rel(ref['point'][l][None], snap.point[k, li][None]))
            for mm in range(24):
                worst = max(worst, MM._rel(ref['c'][mm][None], snap.c[k, mm][None]),
                            MM._rel(ref['z'][mm][None], snap.z[k, mm][None]))
            if lag == 0:
                with torch.no_grad():
                    logits = m.lm_head(ref["last"][None]).double()
                sq = torch.exp(0.5 * torch.log_softmax(logits, -1)).float()
                worst = max(worst, MM._rel(sq, snap.sqrtp[k][None]))
                assert abs(float(snap.sqrtp[k].double().square().sum()) - 1) < 1e-5
    assert worst < 1e-5, worst
    # exact: a batched decode of all snippets (same batch) read at the item's step
    snap = D.snapshot_pass(m, probe.tokens, items, 2, 'cpu', [8], batch=12, log=lambda s: None)
    seq = MM.sequence(probe.tokens)
    hooks = MM.Hooks(m)
    try:
        dec = D.SnapDecoder(m, hooks, len(seq), torch.float32, 'cpu')
        with torch.no_grad():
            for u in range(K.SEQ + 1):
                h = dec.step(seq[:, u], u)
                sel = np.flatnonzero(items['t'] + 2 == u)
                for k in sel:
                    assert torch.equal(snap.point[k, 0], h[8][items['snippet'][k]])
                    assert torch.equal(snap.z[k, 23], dec.state(24)[1][items['snippet'][k]].reshape(-1))
    finally:
        hooks.close()
    RESULTS['snapshot_vs_direct_decode'] = worst


def test_run_u1_u2_u3_tiny():
    """The whole of U1 on a tiny random Mamba (30 snippets, N = 240): files, C1a, C1b, C3 against a v1-format act
    array made by v1's own decode, P3; then U2 (both checkpoints) and U3 on U1's output, and validation."""
    import mc_decode as D
    import mc_schema as SC
    import mb_construction as MB
    MM, m = _tiny()
    probe = _tiny_probe(30)
    res = MM.decode_residuals(m, probe.tokens, 'cpu', torch.float32)
    v1_act = np.stack([res[l].permute(1, 0, 2).numpy() for l in MB.LAYERS])
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        info = D.run_u1(m, 'trained', probe, root / 'trained', 'cpu', log=lambda s: None, v1_act=v1_act,
                        batch=16, chunk32=16, chunk64=16)
        assert info['C3']['pass'] and info['C3']['max_rel_err'] == 0.0, info['C3']
        assert info['P3']['pass'] and info['C1a']['pass'] and info['C1b']['pass']
        n = info['n']
        assert np.load(root / 'trained' / 'point_lag0.npy').shape == (n, 24, 16)
        assert np.load(root / 'trained' / 'point_lag16.npy').shape == (n, 3, 16)
        assert np.load(root / 'trained' / 'cons_z24.npy').shape == (n, 128)
        bc = np.load(root / 'trained' / 'bc.npy')
        assert np.abs(np.diag(bc) - 1).max() < 1e-5 and (bc <= 1 + 1e-6).all()
        D.run_u1(MM.init_model(MM.tiny_config()), 'init', probe, root / 'init', 'cpu', log=lambda s: None)
        assert not (root / 'init' / 'whole_lag1.npy').exists()
        # a C3 failure is reported, not hidden (fail closed on a perturbed reference)
        items = I.sample_items(probe.tokens)
        k = int(np.flatnonzero(np.isin(items['t'], K.C3_POSITIONS))[0])
        bad = v1_act.copy()
        bad[2, MB.POSITIONS.index(int(items['t'][k])), items['snippet'][k], 0] = np.nan
        snap = D.snapshot_pass(m, probe.tokens, items, 0, 'cpu', K.LAYERS, constituents=False, log=lambda s: None)
        assert D.c3_check(snap, items, v1_act)['pass']
        rec = D.c3_check(snap, items, bad)
        assert not rec['pass'] and rec['max_rel_err'] == math.inf, rec['max_rel_err']
        RESULTS['tiny_U1'] = dict(C3=info['C3']['max_rel_err'], P3=info['P3']['max_dev'],
                                  C1a=info['C1a']['max_rel_err'], C1b=info['C1b']['max_rel_err'],
                                  redraws=info['dedup']['redraws'])
        public = root / 'public'
        t0 = time.time()
        for ck in K.CHECKPOINTS:
            SC.measure_checkpoint(root, public, ck, probe, 'cpu', log=lambda s: None)
        cal = dict(theta=3.0, rule='test')                         # no full-size check: certification disabled
        report = SC.finalize(root, public, probe, cal, layouts_on=False, log=lambda s: None)
        assert report['complete'] and report['minimum_seba_label_agreement'] >= 0.99
        meta = json.loads((public / 'meta.json').read_text())
        k5 = json.loads((public / 'targets.json').read_text())['K5']
        ups = json.loads((public / 'upsilon.json').read_text())
        assert not meta['certification']['enabled'] and 'missing' in meta['certification']['reason']
        assert not k5['met'] and 'not certifiable' in k5['status']
        assert all(r['outcome'] != 'certified' for by in ups['by_checkpoint'].values() for rows in by.values()
                   for r in rows)
        # validate refuses a record that claims certification without a passing full-size check
        bad = dict(meta, certification=dict(meta['certification'], enabled=True))
        (public / 'meta.json').write_text(json.dumps(bad))
        try:
            SC.validate(public, require_layouts=False)
            raise RuntimeError('validate accepted certification without a full-size check')
        except AssertionError:
            pass
        for calibrated in (False, True):
            cal = dict(theta=3.0, rule='test', full_size_check=dict(calibrated=calibrated))
            report = SC.finalize(root, public, probe, cal, layouts_on=False, log=lambda s: None)
            meta = json.loads((public / 'meta.json').read_text())
            assert report['complete'] and meta['certification']['enabled'] is calibrated
            assert json.loads((public / 'targets.json').read_text())['K5']['certification_enabled'] is calibrated
        RESULTS['tiny_U2_U3_seconds'] = time.time() - t0
        print(f"  tiny U1/U2/U3: N = {n}, C3 {info['C3']['max_rel_err']:.1e}, P3 {info['P3']['max_dev']:.1e}, "
              f"C1a {info['C1a']['max_rel_err']:.1e}, C1b {info['C1b']['max_rel_err']:.1e}; U2+U3 "
              f"{time.time() - t0:.0f}s, valid ({report['spectral_records']} records)")


# ----------------------------------------------------------------------------------- provenance, budget
def test_keys_and_budget():
    assert K.FINGERPRINT != K.MB.FINGERPRINT
    # U1's key is the original construction (pilot 1's U1 outputs stay valid); U2/U3 carry the amendment
    assert K.DECODE_FINGERPRINT == '78a62a3c5489f7870fe03c1982b3172f12db4b7fc4fab8509033c5d3e9a3b79e'
    assert K.FINGERPRINT != K.DECODE_FINGERPRINT and K.unit_key(decode=True)['construction_fingerprint'] == \
        K.DECODE_FINGERPRINT
    key = K.unit_key('abc', [])
    assert K.gate_ok(dict(key, **{'pass': True}), key)
    assert not K.gate_ok(dict(K.MB.unit_key('abc', []), **{'pass': True}), key)
    assert not K.gate_ok(dict(key, **{'pass': False}), key)
    with tempfile.TemporaryDirectory() as d:
        marker, out = Path(d) / 't.json', Path(d) / 'o.npy'
        out.write_bytes(b'x')
        marker.write_text(json.dumps(dict(seconds=1, **key)))
        assert K.cached_unit(marker, [out], key) is not None
        marker.write_text(json.dumps(dict(seconds=1, **K.MB.unit_key('abc', []))))
        assert K.cached_unit(marker, [out], key) is None
    assert K.jsonable({'a': float('inf'), 'b': np.float32(1.5), 'c': np.arange(2)}) == {'a': 'inf', 'b': 1.5, 'c': [0, 1]}


def test_calibration_generator():
    gen = CB.generate(120, 4, 'zipf', 1.0, 100)
    lab = next(gen)
    assert np.bincount(lab).tolist() == CB.sizes_of(120, 4, 'zipf').tolist() == [58, 29, 19, 14]
    x = next(gen)
    assert x.shape == (120, 64)
    gen2 = CB.generate(120, 4, 'zipf', 1.0, 100)
    assert np.array_equal(next(gen2), lab) and np.array_equal(next(gen2), x)
    flat = next(CB.generate(120, 0, None, 0.0, 100))
    assert (flat == 0).all()


def test_fixtures_validate():
    import mc_schema as SC
    for name in ('_fixture', '_fixture_flat'):
        root = HERE / 'data' / name
        if not (root / 'validation.json').exists():
            print(f'  {name}: not built')
            continue
        fp = json.loads((root / 'meta.json').read_text()).get('construction_fingerprint')
        if fp != K.FINGERPRINT:
            print(f'  {name}: built under an earlier fingerprint ({fp[:12]}); not validated (the page builder reads '
                  f'it; rebuild when it allows)')
            continue
        report = SC.validate(root, require_layouts=(root / 'layouts.npz').exists())
        print(f"  {name}: valid ({report['spectral_records']} records, worst SEBA label agreement "
              f"{report['minimum_seba_label_agreement']:.3f})")


# ------------------------------------------------------------------------------- review fixes (2026-11-12)
def test_ledger_reserves_every_executing_unit():
    """Finding 1: a unit drops out of the reservation only if its function will skip it (cached kind, matching
    record, every output present); U3 and diag always run; the worst case includes the per-attempt overhead."""
    import mc_ledger as LG
    key = K.unit_key('sha', [])
    units = [('U1_trained', 'U1_trained', None), ('U2_trained', 'U2_trained', None), ('U3', 'U3', None),
             ('bench', 'bench', None), ('diag', 'diag', None)]
    records = {'U1_trained': K.unit_key('sha', [], decode=True), 'U2_trained': dict(key), 'U3': dict(key),
               'bench': K.unit_key(None, []),
               'diag': K.unit_key(None, [])}
    outputs = {'U1_trained': ['a', 'b'], 'U2_trained': ['c', 'missing'], 'bench': ['d']}
    present = {'a', 'b', 'c', 'd'}
    done = LG.complete_units(units, records.get, key, lambda n: outputs.get(n, []), present.__contains__)
    assert done == {'U1_trained', 'bench'}, done                 # U2: output missing; U3, diag: never cached
    rate = lambda kind: 1.0
    timeouts = {'U1_trained': 100, 'U2_trained': 200, 'U3': 300, 'bench': 50, 'diag': 10}
    res = LG.reservation(units, timeouts, rate, done)
    assert res == {'U2_trained': 290, 'U3': 390, 'diag': 100}, res
    row = LG.completion_row('U2_trained', 5.0, 0.1, 1.0, 6.0, key, dict(seconds=700.0, checkpoint='trained'))
    assert row['seconds'] == 5.0 and row['unit_seconds'] == 700.0 and K.key_matches(row, key)
    # a record under another key (an older fingerprint, or U1 under the amended one) does not count
    stale = dict(records, U1_trained=K.MB.unit_key('sha', []))
    assert 'U1_trained' not in LG.complete_units(units, dict(records, U1_trained=dict(key)).get, key,
                                                 lambda n: outputs.get(n, []), present.__contains__)
    assert 'U1_trained' not in LG.complete_units(units, stale.get, key, lambda n: outputs.get(n, []),
                                                 present.__contains__)


def _kappa_reference(p, bc, snip, q):
    """Brute-force reference of one null draw: pairs (i, j) with snip_i != snip_j and snip_q(i) != snip_q(j)."""
    p, bc = np.asarray(p), np.asarray(bc)
    num = den = 0.0
    for i in range(len(p)):
        for j in range(len(p)):
            if i != j and snip[i] != snip[j] and snip[q[i]] != snip[q[j]]:
                num += p[i, j] * bc[q[i], q[j]]
                den += p[i, j]
    return num / den if den > 0 else math.nan


def test_kappa_null_excludes_shared_prediction_sources():
    """Finding 2: a null draw never scores two predictions taken from one snippet."""
    # reviewer's case: items 0, 1 in snippet 0 and 2, 3 in snippet 1; tokens X, Y, X, Y; predictions identical
    # within a snippet (BC 1) and orthogonal across (BC 0); the walk only crosses snippets
    snip = np.array([0, 0, 1, 1])
    tok = np.array([0, 1, 0, 1])
    bc = torch.tensor([[1., 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float64)
    p = torch.tensor([[0, 0, .5, .5], [0, 0, .5, .5], [.5, .5, 0, 0], [.5, .5, 0, 0]], dtype=torch.float64)
    within, _ = ST.within_group_perms(tok, 20)
    ks = ST.kappa_stats(p, bc, snip, within)
    assert ks['kappa'] == 0.0
    d = ks['kappaT_draws']
    assert np.all((d == 0.0) | np.isnan(d)), d                     # old code: 0.5 on draws that swap one pair
    q = np.array([2, 1, 0, 3])                                      # swap only items 0 and 2
    one = ST.kappa_stats(p, bc, snip, [q])['kappaT_draws'][0]
    assert one == 0.0 and _kappa_reference(p, bc, snip, q) == 0.0
    # a draw with no surviving mass is NaN and poisons the summary (fail closed)
    q0 = np.array([0, 1, 2, 3])
    p_in = torch.tensor([[0, 1., 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float64)
    ks0 = ST.kappa_stats(p_in, bc, snip, [q0])
    assert math.isnan(ks0['kappaT_draws'][0]) and math.isnan(ST.kappa_summary(ks0, 0.0)['kappaT'])
    # random case with repeated token groups and several items per snippet: every draw equals the brute force
    rng = np.random.default_rng(4)
    n = 48
    snip = np.arange(n) // 4
    tok = rng.integers(0, 5, n)
    x = rng.normal(size=(n, 6))
    p = C.walk(C.standard_gram(x), k=6)[0]
    lg = torch.log_softmax(torch.from_numpy(rng.normal(size=(n, 9))), -1)
    qv = torch.exp(0.5 * lg)
    bc = qv @ qv.T
    within, _ = ST.within_group_perms(tok, 8)
    d = ST.kappa_stats(p, bc, snip, within)['kappaT_draws']
    ref = np.array([_kappa_reference(p, bc, snip, q) for q in within])
    assert np.allclose(d, ref, rtol=1e-12, atol=0), (d, ref)
    assert any(len(set(snip[q[np.flatnonzero(snip == s)]])) < 4 for q in within for s in range(12))


def test_controls_fail_closed():
    """Finding 3: NaN anywhere in a control's comparisons fails it, also after a finite first element."""
    import mc_decode as D
    assert K.worst_of([1e-15, float('nan'), 2e-15]) == math.inf and K.worst_of([1e-15, 3e-15]) == 3e-15
    assert math.isnan(K.least_of([0.3, float('nan')])) and K.least_of([0.3, 0.1]) == 0.1
    assert K.worst_of([]) == math.inf and math.isnan(K.least_of([]))
    rng = np.random.default_rng(0)
    snap = D.Snapshot.__new__(D.Snapshot)
    snap.c = torch.from_numpy(rng.normal(size=(40, 24, 5)) + 2)
    snap.z = torch.from_numpy(rng.normal(size=(40, 24, 7)) - 1)
    assert D.p3_check(snap, 24, 'cpu', size=40)['pass']
    orig, calls = MC.explicit_concat_gram, [0]

    def poisoned(cons):
        calls[0] += 1
        g = orig(cons)
        if calls[0] == 3:                                            # state_L3: not the first comparison
            g = g.clone(); g[0, 1] = float('nan')
        return g
    MC.explicit_concat_gram = poisoned
    try:
        rec = D.p3_check(snap, 24, 'cpu', size=40)
    finally:
        MC.explicit_concat_gram = orig
    assert not rec['pass'] and rec['max_dev'] == math.inf, rec['max_dev']
    # C1a from v1's code is re-decided here: a NaN after a finite layer value fails it
    MM, m = _tiny()
    toks = np.random.default_rng(1).integers(1, 512, (6, K.SEQ))
    real = MM.c1a
    MM.c1a = lambda *a, **k: dict(real(*a, **k), float64_by_layer={'L4': 1e-16, 'L8': float('nan')},
                                  **{'pass': True})
    try:
        c1b, c1a = D.c1_checks(m, toks, 'cpu', log=lambda s: None, chunk32=6, chunk64=6)
    finally:
        MM.c1a = real
    assert c1b['pass'] and not c1a['pass'] and c1a['max_rel_err'] == math.inf


def test_certification_rule():
    """Finding 4: certification needs theta AND a passing full-size check; otherwise outcomes come from the cap only."""
    import mc_schema as SC
    ok = SC.reading_rule(dict(theta=3.0, full_size_check=dict(calibrated=True)))
    assert ok['certify'] and ok['reason'] is None
    for cal in (dict(theta=3.0), dict(theta=3.0, full_size_check=None), dict(theta=3.0, full_size_check={}),
                dict(theta=3.0, full_size_check=dict(calibrated=False)),
                dict(theta=3.0, full_size_check=dict(calibrated='true'))):
        r = SC.reading_rule(cal)
        assert r['theta'] == 3.0 and not r['certify'] and r['reason'], cal
    dropped = SC.reading_rule(dict(theta=None, full_size_check=dict(calibrated=True)))
    assert dropped['theta'] is None and not dropped['certify']
    assert ST.outcome(500.0, 900.0, 3.0, certify=False) == 'undetermined'
    assert ST.outcome(500.0, 2.0, 3.0, certify=False) == 'excluded'
    assert ST.outcome(500.0, 900.0, 3.0, certify=True) == 'certified'


def test_reference_duplicates_exempt():
    """SPEC 3 amendment after pilot 1: duplicates in a reference cell pass and are counted (with the tie check);
    duplicates in a point cell still raise."""
    import mc_schema as SC
    rng = np.random.default_rng(9)
    n = 160
    x = rng.normal(size=(n, 10)) + np.eye(4)[np.arange(n) % 4] @ rng.normal(size=(4, 10)) * 2
    x[5] = x[17]                                         # one exact duplicate pair
    x[40:60] = x[40]                                     # 20 identical items (more than k + 1): a tie-break cluster
    gram = ST.view_gram(torch.from_numpy(x))
    try:
        ST.cell_walk(gram.clone())
        raise AssertionError('duplicate accepted in a view cell')
    except C.DegenerateItems:
        pass
    p, info = ST.cell_walk(gram.clone(), assert_distinct=False)
    assert info['n_dup_pairs'] == 1 + 20 * 19 // 2 and info['min_dist'] <= K.DUPLICATE_TOL
    assert info['row_error'] < K.SINKHORN_TOL
    snip = np.arange(n) // 4
    g = dict(T=rng.integers(0, 9, n), Nx=rng.integers(0, 9, n), U=snip, A=rng.integers(0, 5, n), W=snip % 5,
             Cc=rng.integers(0, 12, n), Cn=rng.integers(0, 12, n))
    q = torch.exp(0.5 * torch.log_softmax(torch.from_numpy(rng.normal(size=(n, 7))), -1))
    M = SC.Measure(n, g, q @ q.T, 'cpu', log=lambda s: None)
    try:
        M.cell('pred', gram.clone(), spectral=True)
        rec = M.cells['pred']
        assert rec['reference'] and rec['n_dup_pairs'] == 191 and len(rec['tie_check']) == 3
        a = [row[f'a{r}'] for row in rec['tie_check'] for r in K.SET_R]
        m = [row[f'ami{r}'] for row in rec['tie_check'] for r in K.SET_R]
        assert rec['tie_sensitive'] == (min(a) < K.TIE_MIN_A or min(m) < K.TIE_MIN_AMI)
        assert all(row['row_err'] < K.SINKHORN_TOL for row in rec['tie_check'])
        x2 = rng.normal(size=(n, 10)); x2[3] = x2[9]
        M.cell('bag', ST.view_gram(torch.from_numpy(x2)), spectral=True)
        assert M.cells['bag']['n_dup_pairs'] == 1 and len(M.cells['bag']['tie_check']) == 3
        try:
            M.cell('point_L1', gram.clone())
            raise AssertionError('duplicate accepted in a point cell')
        except C.DegenerateItems:
            pass
    finally:
        M.spill.close()
    RESULTS['reference_tie_check'] = dict(pairs=rec['n_dup_pairs'], min_a=min(a), min_ami=min(m),
                                          sensitive=rec['tie_sensitive'])


TESTS = [test_keys_and_budget, test_partition_contract, test_upsilon2_equals_lm_tree, test_streaming_grams_and_block,
         test_shares_vs_v1, test_streaming_tree_vs_lm_tree, test_whole_cell_subset_rebuild, test_item_dedup,
         test_kappa_conditional_null, test_eta2_planted, test_calibration_generator, test_snapshot_indexing_tiny,
         test_run_u1_u2_u3_tiny, test_fixtures_validate, test_ledger_reserves_every_executing_unit,
         test_kappa_null_excludes_shared_prediction_sources, test_controls_fail_closed, test_certification_rule,
         test_reference_duplicates_exempt]

if __name__ == '__main__':
    torch.set_num_threads(2)
    only = sys.argv[1:]
    for test in TESTS:
        if only and test.__name__ not in only:
            continue
        start = time.time()
        test()
        print(f'ok  {test.__name__} ({time.time() - start:.1f}s)', flush=True)
    print(json.dumps(K.jsonable(RESULTS)))
    print(f'all {len(TESTS) if not only else len(only)} token-context checks passed')
