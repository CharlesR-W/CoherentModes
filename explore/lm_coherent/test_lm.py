"""CPU tests of SPEC section 13, run before any Modal unit.

    uv run --with torch --with numpy --with scipy python -I test_lm.py        (or: pytest test_lm.py)

P1-P3; the sweep never separates equal values; Cheeger bounds for phi_free on random graphs; the slab Gram
identity; bag permutation invariance (C2); the zero-norm and duplicate policies; SEBA reconstruction (the ResNet
regression tests, reused); fingerprint and cache rejection; the kNN rule equals the ResNet rule; walk shares; the
probe builder on a toy corpus.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lm_construction as LC  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_schema as S  # noqa: E402
import lm_tree as Z  # noqa: E402
from coherent import block_modes, seba, seba_labels  # noqa: E402

P1_SEEDS = (0, 1, 2)


# ------------------------------------------------------------------------------------------------- P1, P2
def test_p1_planted_hierarchy():
    """The 7 splits reproduce the planted tree exactly; planted Upsilon above the threshold, structureless below.
    Several seeds of the same fixture regime (SPEC 10.2), not one draw."""
    planted_all, structureless_all = [], []
    for seed in P1_SEEDS:
        cells, leaf = S.planted_hierarchy_block(seed)
        nodes = Z.grow(cells)
        match = S.planted_match(nodes, leaf)
        assert match['exact'], (seed, match)
        assert [nd['depth'] for nd in nodes] == [0, 1, 1] + [2] * 4 + [3] * 8
        planted_all += [nd['upsilon'] for nd in nodes if 'upsilon' in nd]
        for nd in nodes:
            if nd['depth'] == 3:
                rec = Z.analyse_node(Z.node_walks(cells, nd['members']), nd['members'])
                structureless_all.append(rec['upsilon'])
    flat = S.flat_cell()
    rec = Z.analyse_node(Z.node_walks([flat], np.arange(len(flat))), np.arange(len(flat)))
    structureless_all.append(rec['upsilon'])
    threshold, rule = Z.threshold_rule(planted_all, structureless_all)
    assert rule == 'fixed' and threshold == 3.0, (rule, min(planted_all), max(structureless_all))
    assert min(planted_all) > threshold > max(structureless_all)
    print(f'  P1: {len(P1_SEEDS)} seeds exact; planted Upsilon min {min(planted_all):.2f}, '
          f'structureless max {max(structureless_all):.2f}; threshold {threshold} ({rule})')


def test_p1_old_regime_is_uninformative():
    """Regression for the review finding: with compact leaves (sigma = 0.3) the union-kNN graph has no edge
    between leaves, so the root cut is a union of whole leaves but not the planted root split."""
    cells, leaf = S.planted_hierarchy_block(0, sigma=0.3)
    lk = C.knn_logkernel(C.standard_gram(cells[0])).numpy() > -1
    np.fill_diagonal(lk, False)
    i, j = np.nonzero(lk)
    assert np.all(leaf[i] == leaf[j]), 'expected no cross-leaf kNN edges at sigma = 0.3'
    nodes = Z.grow(cells)
    root0 = leaf[nodes[nodes[0]['children'][0]]['members']]
    counts = np.bincount(root0, minlength=8)
    assert set(counts) <= {0, 125}                      # never splits a leaf ...
    assert not S.planted_match(nodes, leaf)['exact']     # ... but the grouping is arbitrary


def _jac2(lab, truth):
    return LC.resnet_site_analysis().matched_jaccard(np.asarray(lab, np.int64), np.asarray(truth, np.int64), 2, 2)[0]


def test_p2_block_demotion():
    """The fixture's L4 reading block at step 143000 (lm_schema.p2_check, also run by U3 for validation.json)."""
    out = S.p2_check()
    assert out['pass'], out
    print(f"  P2: transient cell sweep/SEBA Jaccard {out['transient_cell']['sweep_jaccard']:.3f}/"
          f"{out['transient_cell']['seba2_jaccard']:.3f}; block {out['block']['sweep_jaccard']:.3f}/"
          f"{out['block']['seba2_jaccard']:.3f}")


def test_component_flag():
    """Union pre-floor kNN components per node: the v2 regime (sigma 0.3) is flagged at the root with 8
    components; the fixture regime (sigma 1.3) has <= 2 at every split; the flag changes no cut."""
    cells, leaf = S.planted_hierarchy_block(0, sigma=0.3)
    ps, mask = Z.node_operators(cells, np.arange(1000))
    comp = Z.components(mask, Z.m_min(1000))
    assert comp['n_components'] == 8 and comp['component_sizes'] == [125] * 8 and comp['disconnected']
    cells, leaf = S.planted_hierarchy_block(0)
    nodes = Z.grow(cells)
    assert all(nd['n_components'] <= 2 for nd in nodes if nd['children'])
    assert all(len(nd['seba2']) == nd['n'] for nd in nodes if nd['children'])
    # room for a valid cut: one big component plus crumbs below m_min is not flagged
    m = np.zeros((40, 40), bool); m[:37, :37] = True; m[37:, 37:] = True
    assert Z.components(m, 16) == dict(n_components=2, component_sizes=[37, 3], disconnected=False)
    m = np.zeros((40, 40), bool); m[:20, :20] = True; m[20:, 20:] = True
    assert Z.components(m, 16)['disconnected']


def test_budget_ledger_gate_and_cache_keys():
    """Reviewer findings A1-A3 (lm_modal relies on these pure functions)."""
    worst = {'U0': .158, **{f'U1_{s}': .567 for s in LC.STEPS}, **{f'U2_{s}': .422 for s in LC.STEPS}, 'U3': .422}
    assert abs(LC.budget([], worst, set())['total'] - 5.525) < 1e-9
    # A1: a timed-out U1 attempt (never reconciled) keeps its full worst case, so a retry cannot pass $6
    lost = dict(unit='U1_143000', reserved_usd=.567, actual_usd=None)
    b = LC.budget([lost], worst, set(), extra=.079)
    assert b['total'] > 6.0
    # a reconciled attempt counts at its actual cost; attempts count whatever their fingerprint
    ok = dict(unit='U0', reserved_usd=.158, actual_usd=.02, fingerprint='stale')
    assert abs(LC.budget([ok], worst, {'U0'})['total'] - (5.525 - .158 + .02)) < 1e-9
    # A2: the gate needs a passing record with the current construction, probe and removal list
    key = LC.unit_key('abc', [])
    good = dict(key, **{'pass': True})
    assert LC.gate_ok(good, key)
    assert not LC.gate_ok(dict(good, **{'pass': False}), key)
    assert not LC.gate_ok(good, LC.unit_key('abc', [17]))
    assert not LC.gate_ok(good, LC.unit_key('other', []))
    assert not LC.gate_ok(dict(good, construction_fingerprint='old'), key)
    assert not LC.gate_ok(None, key)
    # A3: a U2 record made before a removal (N = 1,000) is not reused after it (N = 999)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker, output = root / 'timing_U2_143000.json', root / 'cells_143000.npz'
        output.write_bytes(b'x')
        marker.write_text(json.dumps(dict(seconds=1, **LC.unit_key('abc', []))))
        assert LC.cached_unit(marker, [output], LC.unit_key('abc', []))['seconds'] == 1
        assert LC.cached_unit(marker, [output], LC.unit_key('abc', [17])) is None
        assert LC.cached_unit(marker, [output], LC.unit_key('new-probe', [])) is None
        assert LC.probe_identity(b'a') != LC.probe_identity(b'b')


# ------------------------------------------------------------------------------------------------------ P3
def _cells(seed=3, n=120, d=10, k=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(4, d)) * 1.5
    lab = np.arange(n) % 4
    return [centres[lab] + rng.normal(size=(n, d)) for _ in range(k)], lab


def test_p3_linearity():
    """Block conductance = mean of cell conductances; block T, W, X = means of cell values (and equal to the value
    on the averaged walk for a fixed grouping); to 1e-12."""
    xs, lab = _cells()
    n = len(lab)
    ps = [C.walk(C.standard_gram(x))[0].numpy() for x in xs]
    pbar = np.mean(ps, 0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.random(n) < .4
        phi = lambda p: p[np.ix_(a, ~a)].sum() / min(a.sum(), (~a).sum())
        assert abs(phi(pbar) - np.mean([phi(p) for p in ps])) < 1e-12
    perms = C.permutation_bank(n, 20)
    sources = np.arange(n) % len(LC.SOURCES)
    tokens = rng.integers(0, 15, n)
    follow = np.arange(10)
    raws = [C.raw_statistics(p, tokens, sources, perms, follow) for p in ps]
    block = C.export_statistics(raws)
    direct = C.raw_statistics(pbar, tokens, sources, perms, follow)
    for key in ('T', 'W', 'X'):
        assert np.max(np.abs(block[key] - direct[key])) < 1e-12, key
        assert np.max(np.abs(block[key] - np.mean([r[key] for r in raws], 0))) < 1e-12, key
    for key in ('T_null', 'W_null', 'X_null'):
        assert np.max(np.abs(np.mean([r[key] for r in raws], 0) - direct[key])) < 1e-12, key
    # exact chance level: S0 is the mean of S over ALL relabelings (exhaustive on a tiny walk)
    small = C.walk(C.standard_gram(np.random.default_rng(5).normal(size=(7, 3))), k=3)[0].numpy()
    g = np.array([0, 0, 0, 1, 1, 2, 2])
    vals = [C.walk_share(small, g[list(q)], np.arange(7)[None])[0] for q in itertools.permutations(range(7))]
    s0 = C.walk_share(small, g, np.arange(7)[None])[1]
    assert abs(np.mean(vals) - s0) < 1e-12
    # X = 1 for a walk that ignores sources: the uniform walk off the diagonal
    uni = (np.ones((n, n)) - np.eye(n)) / (n - 1)
    x, *_ = C.mixing(uni, sources, perms)
    off = ~np.eye(len(LC.SOURCES), dtype=bool)
    assert np.max(np.abs(x[off] - 1)) < 1e-12
    print('  P3: conductance, T, W, X and their permutation draws are linear to 1e-12; S0 exact')


def test_slab_gram_identity():
    """Slab Gram via the mean of per-position Grams equals the standard pipeline on the explicit concatenation."""
    rng = np.random.default_rng(11)
    xs = [rng.normal(size=(40, 6)) + rng.normal(size=6) * 3 for _ in range(9)]
    via_gram = C.slab_gram(xs)
    concat = np.concatenate([C.unit_rows(x).numpy() for x in xs], 1) / math.sqrt(len(xs))
    explicit = C.standard_gram(concat)
    assert torch.max(torch.abs(via_gram - explicit)) < 1e-12
    # the accumulator gives every intermediate t
    acc = C.SlabAccumulator(40)
    for t, x in enumerate(xs, 1):
        acc.add(torch.from_numpy(x))
        if t in (1, 4):
            assert torch.max(torch.abs(acc.gram() - C.slab_gram(xs[:t]))) < 1e-12
    print('  slab Gram identity holds to 1e-12')


# ------------------------------------------------------------------------------------------- the sweep cut
def test_sweep_never_separates_equal_values():
    rng = np.random.default_rng(2)
    n = 80
    p = rng.random((n, n)); p = (p + p.T) / 2; np.fill_diagonal(p, 0); p /= p.sum(1).mean()
    for trial in range(20):
        psi = np.round(rng.normal(size=n), 1)              # many ties
        phi, m, theta, order, _ = Z.sweep(p, psi, 5)
        if m is None:
            continue
        a, b = psi[order[:m]], psi[order[m:]]
        assert a.min() > b.max() and a.min() > theta > b.max()
    psi = np.repeat([1.0, 0.0], [10, n - 10])               # only one admissible cut, below m_min
    assert Z.sweep(p, psi, 20)[1] is None and Z.sweep(p, psi, 10)[1] == 10


def test_cheeger_bounds_random_graphs():
    """mu1/2 <= h <= phi_free <= sqrt(2 mu1); h by brute force on small graphs."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 12 if seed < 3 else 200
        x = rng.normal(size=(n, 4)) + (rng.normal(size=(1, 4)) * 2 if seed % 2 else 0)
        p, _ = C.walk(C.standard_gram(x), k=3 if n == 12 else 15)
        vals, vecs = block_modes([C.lazy(p)], 3, shared_stationary=True)
        mu1 = 2 * (1 - float(vals[1]))
        phi_free = Z.sweep(p.numpy(), vecs[:, 1].numpy(), 1)[0]
        assert mu1 / 2 - 1e-12 <= phi_free <= math.sqrt(2 * mu1) + 1e-12, (seed, mu1, phi_free)
        if n == 12:
            pp = p.numpy()
            h = min(pp[np.ix_(a, ~a)].sum() / min(a.sum(), (~a).sum())
                    for bits in range(1, 2 ** (n - 1)) for a in [np.array([(bits >> i) & 1 for i in range(n)], bool)])
            assert mu1 / 2 - 1e-12 <= h <= phi_free + 1e-12, (seed, mu1, h, phi_free)
    print('  Cheeger bounds hold (brute-force h on 12-node graphs)')


def test_node_record_fields():
    cells, leaf = S.planted_hierarchy_block(4)
    members = np.flatnonzero(leaf < 4)
    rec = Z.analyse_node(Z.node_walks(cells, members), members, np.arange(1000) % 5)
    a, b = rec['children_members']
    assert len(a) + len(b) == len(members) and min(len(a), len(b)) >= Z.m_min(len(members))
    assert rec['psi'].shape == (3, len(members)) and rec['psi'].dtype == np.int8 and len(rec['profile']) == 64
    assert rec['cheeger_lo'] <= rec['phi_free'] <= rec['cheeger_hi'] and rec['phi'] >= rec['cheeger_lo'] - 1e-12
    assert sum(rec['makeup']) == len(members) and len(rec['retention_cut']) == len(cells)
    top = rec['ends'][0]
    k = rec['axis_used'] - 1
    pos = {m: i for i, m in enumerate(members)}
    assert all(rec['psi'][k][pos[i]] >= rec['psi'][k][pos[j]] for i in top for j in rec['ends'][1])


# ------------------------------------------------------------------------------------- views and policies
def test_bag_permutation_invariance():
    rng = np.random.default_rng(3)
    tokens = rng.integers(1, 300, (30, LC.SEQ))
    emb = rng.normal(size=(300, 16)).astype(np.float32)
    bag = C.bag_vectors(tokens, emb)
    probe = dict(items=[dict(tokens=t.tolist()) for t in tokens],
                 S=dict(permutation=[np.random.RandomState(i).permutation(LC.SEQ - 1).tolist() for i in range(30)]))
    shuffled = S.shuffled_tokens(probe)
    assert np.array_equal(shuffled[:, -1], tokens[:, -1])
    bag_s = C.bag_vectors(shuffled, emb, [LC.SEQ])
    assert np.array_equal(bag[-1], bag_s[0])                     # C2: bitwise
    for pi, t in enumerate(LC.POSITIONS):                          # any reordering of tokens 1..t
        perm = rng.permutation(t)
        other = tokens.copy(); other[:, :t] = tokens[:, :t][:, perm]
        assert np.array_equal(C.bag_vectors(other, emb, [t])[0], bag[pi])
    assert torch.equal(C.standard_gram(bag[-1]), C.standard_gram(bag_s[0]))
    print('  bag view is bitwise permutation-invariant (C2)')


def test_zero_norm_policy():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 8))
    x[17] = np.delete(x, 17, 0).mean(0)                           # centred vector exactly zero
    assert list(C.zero_norm_items(x)) == [17]
    try:
        C.standard_gram(x)
        raise AssertionError('zero-norm item was not caught')
    except C.DegenerateItems as e:
        assert e.kind == 'zero-norm' and e.items == [17]
    probe = dict(items=[dict(source=LC.SOURCES[i % 5], tokens=list(range(1, LC.SEQ + 1))) for i in range(50)])
    pr = S.Probe(probe, removed=[17])
    assert pr.n == 49 and 17 not in pr.items and pr.removed == [17]
    c = C.standard_gram(x[pr.items])
    assert torch.allclose(c.diagonal(), torch.ones(49, dtype=torch.float64))
    # exact duplicates fail the cell
    y = rng.normal(size=(40, 5)); y[9] = y[3]
    try:
        C.walk(C.standard_gram(y))
        raise AssertionError('duplicate was not caught')
    except C.DegenerateItems as e:
        assert e.kind == 'duplicate' and sorted(e.items) == [3, 9]


def test_knn_rule_is_the_resnet_rule():
    from sweep_math import final_kernel_bank
    x = np.random.default_rng(8).normal(size=(400, 24))
    c = C.standard_gram(x)
    resnet = final_kernel_bank(c)[0][0]
    assert torch.equal(C.knn_logkernel(c), resnet)
    p, info = C.walk(c)
    assert info['row_error'] < 2e-12 and torch.equal(p.diagonal(), torch.zeros(400, dtype=torch.float64))
    # ties broken by lower index
    tied = torch.tensor([[1., .5, .5, .5], [.5, 1., .9, .1], [.5, .9, 1., .8], [.5, .1, .8, 1.]], dtype=torch.float64)
    w = C.knn_logkernel(tied, k=1).exp()
    assert w[0, 1] == 1 and w[0, 2] < 1e-5 and w[0, 3] < 1e-5


# ------------------------------------------------------------------------------------- SEBA, provenance
def test_seba_reconstruction():
    """The ResNet SEBA regressions, reused, plus our export path (spectral_record -> int16 -> reconstruct)."""
    from test_seba import test_reference
    from test_coherent import test_reconstruction, test_near_identity_saved_operator
    test_reference()
    test_reconstruction()
    test_near_identity_saved_operator()
    cells, _ = S.planted_hierarchy_block(5)
    ms = [C.lazy(C.walk(C.standard_gram(x))[0]) for x in cells[:3]]
    rec, v = C.spectral_record(ms, (0, 2), sources=np.arange(1000) % 5, perms=C.permutation_bank(1000, 10))
    z = {'x_' + k: np.asarray(val)[None] for k, val in rec.items() if k in S.RECORD_KEYS}
    worst, count = S._check_records(z, 'x_', (1,), 1000, 3)
    assert worst >= .99 and count == 1
    tm = rec['top_members']
    lab = rec['seba_label'][2]
    assert all(lab[i] == j for j in range(4) for i in tm[2, j] if i >= 0)


def test_fingerprint_and_cache_rejection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        marker, output = root / 'timing_U2_0.json', root / 'cells_0.npz'
        output.write_bytes(b'placeholder')
        marker.write_text(json.dumps({'construction_fingerprint': LC.RESNET_FINGERPRINT, 'seconds': 2}))
        assert LC.cached_completion(marker, [output]) is None          # the ResNet fingerprint is not ours
        marker.write_text(json.dumps({'seconds': 2}))
        assert LC.cached_completion(marker, [output]) is None
        marker.write_text(json.dumps({'construction_fingerprint': LC.FINGERPRINT, 'seconds': 2}))
        assert LC.cached_completion(marker, [output])['seconds'] == 2
        output.unlink()
        assert LC.cached_completion(marker, [output]) is None
    changed = dict(LC.CONSTRUCTION, k=16)
    import hashlib
    other = hashlib.sha256(json.dumps(changed, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert other != LC.FINGERPRINT and LC.CONSTRUCTION['resnet_fingerprint'] == LC.RESNET_FINGERPRINT


# --------------------------------------------------------------------------------------------- the probe
class ToyTokenizer:
    """Word-level stand-in for the Pythia tokenizer: tokens are ' word' pieces, ids from a fixed vocabulary."""

    def __init__(self, words):
        self.vocab = {w: i + 1 for i, w in enumerate(words)}
        self.inv = {i: w for w, i in self.vocab.items()}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        ids, offs, i = [], [], 0
        import re
        for m in re.finditer(r'\s*\S+', text):
            ids.append(self.vocab.get(m.group().strip(), 1)); offs.append((m.start(), m.end()))
        return dict(input_ids=ids, offset_mapping=offs)

    def decode(self, ids):
        return ''.join(' ' + self.inv.get(i, '?') for i in ids)


def test_probe_builder():
    import probe as PB
    rng = np.random.default_rng(0)
    words = [a + b + c for a in 'bcdfg' for b in 'aeiou' for c in 'klmnprstvz'][:240] + ['$x$']
    tok = ToyTokenizer(words)
    def doc(extra=''):
        paras = [' '.join(rng.choice(words[:240], 50)) + extra + ' '.join(rng.choice(words[:240], 150))
                 for _ in range(3)]
        return '\n\n'.join(paras)
    docs = {s: [(f'{s}-{i}', doc(' $x$ $x$ $x$ $x$ ' if s == 'maths' else ' ')) for i in range(14)]
            for s in LC.SOURCES}
    wrap = lambda text: '\n\n'.join('\n'.join(' '.join(para.split(' ')[k:k + 10]) for k in range(0, 200, 10))
                                       for para in text.split('\n\n'))
    docs['code'] = [(d, wrap(text)) for d, text in docs['code']]
    docs['code'].insert(0, ('code-long', 'x' * 400 + '\n\n' + docs['code'][0][1]))
    probe = PB.build_probe(docs, tok, per_source=8)
    items = probe['items']
    assert len(items) == 40 and [it['id'] for it in items] == list(range(40))
    from collections import Counter
    assert Counter(it['source'] for it in items) == {s: 8 for s in LC.SOURCES}
    for t in LC.POSITIONS:
        assert len({tuple(sorted(it['tokens'][:t])) for it in items}) == 40
    for it, perm in zip(items, probe['S']['permutation']):
        s = PB.shuffled(it['tokens'], perm)
        assert s[-1] == it['tokens'][-1] and sorted(s) == sorted(it['tokens'])
    assert PB.pile_starts('ab\n\ncd\n\n ef\n\ngh') == [0, 4, 13]
    again = PB.build_probe(docs, tok, per_source=8)
    assert json.dumps(again, sort_keys=True) == json.dumps(probe, sort_keys=True)


TESTS = [test_fingerprint_and_cache_rejection, test_knn_rule_is_the_resnet_rule, test_zero_norm_policy,
         test_bag_permutation_invariance, test_slab_gram_identity, test_p3_linearity,
         test_sweep_never_separates_equal_values, test_cheeger_bounds_random_graphs, test_node_record_fields,
         test_seba_reconstruction, test_probe_builder, test_p1_old_regime_is_uninformative,
         test_p1_planted_hierarchy, test_p2_block_demotion,
         test_component_flag, test_budget_ledger_gate_and_cache_keys]

if __name__ == '__main__':
    torch.set_num_threads(2)
    for test in TESTS:
        start = time.time()
        test()
        print(f'ok  {test.__name__} ({time.time() - start:.1f}s)', flush=True)
    print(f'all {len(TESTS)} lm checks passed')
