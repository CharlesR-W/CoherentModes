"""CPU tests of the Mamba study, run before any Modal unit.

    export TMPDIR=/var/tmp/cm-mb
    uv run --with torch --with numpy --with scipy --with transformers==4.57.6 python -I test_mb.py

On a tiny randomly initialised Mamba-1 (24 layers, width 16) with HF transformers 4.57.6, torch path: the conv
buffer claim (columns of conv_states after each step), C1a (decode vs prefill, float64 parameters and cache; the
magnitude is printed), C1b (resume from the three-column conv buffer and SSM state with column 0 = NaN), the HF init
is deterministic under the seed, and the whole of U1 (`mb_model.run_u1`) on 48 items. Without a model: the P3 Gram
identity, the Gram-based subset rebuild of T4 against a full rebuild from raw constituent vectors (non-root subsets,
unequal item norms, non-zero means) to 1e-12, lm_tree's provider unchanged for vector cells, budget/gate/cache keys,
and (when present) the fixtures and the cross-architecture comparison on the fixtures.
The Pythia suite (`../lm_coherent/test_lm.py`) must also still pass (lm_tree was generalised).
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

import mb_construction as MB  # noqa: E402
import mb_cells as MC  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_tree as Z  # noqa: E402

RESULTS = {}


def _model():
    import mb_model as MM
    return MM, MM.init_model(MM.tiny_config())


def _tokens(n, seed=0, vocab=512):
    return np.random.default_rng(seed).integers(1, vocab, (n, MB.SEQ))


# ------------------------------------------------------------------------------------------ the HF cache
def test_slow_path_and_init_determinism():
    MM, m = _model()
    MM.assert_slow_path()
    other = MM.init_model(MM.tiny_config())
    assert MM.parameter_sha(m) == MM.parameter_sha(other)
    torch.manual_seed(123)                                   # a different global RNG state before the call
    assert MM.parameter_sha(MM.init_model(MM.tiny_config())) == MM.parameter_sha(m)
    assert MM.parameter_sha(MM.init_model(MM.tiny_config(), seed=1)) != MM.parameter_sha(m)
    # HF's _init_weights ran: S4D-real A, D = 1
    mixer = m.backbone.layers[0].mixer
    assert torch.allclose(mixer.A_log.exp(), torch.arange(1, 5, dtype=torch.float32).expand(32, 4))
    assert torch.equal(mixer.D, torch.ones(32))


def test_conv_buffer_claim():
    """After the step that reads sequence index u, conv_states[m] = [x_{u-3}, x_{u-2}, x_{u-1}, x_u] (zeros before
    the start), x = the conv input (first half of in_proj's output); the next step reads only columns 1..3."""
    MM, m = _model()
    seq = MM.sequence(_tokens(5))
    hooks = MM.Hooks(m)
    xs = {1: [], 7: []}
    grabs = [m.backbone.layers[l - 1].mixer.in_proj.register_forward_hook(
        lambda mod, i, o, l=l: xs[l].append(o[:, -1, :32].detach().clone())) for l in xs]
    try:
        dec = MM.Decoder(m, hooks, len(seq), torch.float32, 'cpu')
        with torch.no_grad():
            for u in range(10):
                dec.step(seq[:, u], u)
                for l in xs:
                    buf = dec.cache.conv_states[l - 1]
                    for col in range(4):
                        k = u - 3 + col
                        want = xs[l][k] if k >= 0 else torch.zeros_like(xs[l][0])
                        assert torch.equal(buf[..., col], want), (l, u, col)
                    c, _ = dec.state(l)
                    assert torch.equal(c, buf[..., 1:])
    finally:
        hooks.close()
        for g in grabs:
            g.remove()
    # with the true index as cache_position the first steps would misplace the input (why MB.cache_position)
    assert [MB.cache_position(u) for u in range(6)] == [3, 3, 3, 3, 4, 5]


def test_c1b_fails_closed_on_nan():
    """Negative control (review finding): NaN in a RETAINED conv column must fail C1b, not read as zero error."""
    MM, m = _model()
    toks = _tokens(8, 1)
    orig = MM.Decoder.resume_from

    def poisoned(self, other):
        orig(self, other)
        self.cache.conv_states[0][..., 1] = float('nan')
    MM.Decoder.resume_from = poisoned
    try:
        out = MM.decode_views(m, toks, 'cpu', c1b=True, log=lambda s: None)
    finally:
        MM.Decoder.resume_from = orig
    assert not out['C1b']['pass'] and out['C1b']['max_rel_err'] == float('inf'), out['C1b']


def test_c1a_c1b_tiny():
    """C1a in float64 parameters and cache (stock HF arithmetic) and the float32 hazard; C1b in float32."""
    MM, m = _model()
    toks = _tokens(32, 1)
    out = MM.decode_views(m, toks, 'cpu', c1b=True, log=lambda s: None)
    c1b = out['C1b']
    assert c1b['pass'] and c1b['steps'] == 64, c1b
    rec = MM.c1a(m, toks, out['act'], 'cpu', log=lambda s: None)
    assert rec['pass'], rec
    assert next(m.parameters()).dtype == torch.float32
    RESULTS['C1a_tiny_float64'] = rec['max_rel_err']
    RESULTS['C1a_tiny_float32'] = rec['float32_max_rel_err']
    RESULTS['C1b_tiny_float32'] = c1b['max_rel_err']
    print(f"  C1a (tiny, float64 params+cache): max rel err {rec['max_rel_err']:.3e} (tol 1e-4); by layer "
          f"{ {k: f'{v:.1e}' for k, v in rec['float64_by_layer'].items()} }")
    print(f"  C1a float32 hazard: {rec['float32_max_rel_err']:.3e}; C1b (float32): max rel err "
          f"{c1b['max_rel_err']:.3e} over {c1b['steps']} resumed steps (resid {c1b['resid']:.1e}, conv "
          f"{c1b['conv']:.1e}, ssm {c1b['ssm']:.1e})")
    # the resume really uses the state: a resume from a WRONG state does not reproduce the decode
    seq = MM.sequence(toks)
    hooks = MM.Hooks(m)
    try:
        a = MM.Decoder(m, hooks, len(seq), torch.float32, 'cpu')
        b = MM.Decoder(m, hooks, len(seq), torch.float32, 'cpu')
        with torch.no_grad():
            for u in range(MB.RESUME_T + 1):
                a.step(seq[:, u], u)
            b.resume_from(a)
            b.cache.ssm_states[3].mul_(1.01)
            ha = a.step(seq[:, MB.RESUME_T + 1], MB.RESUME_T + 1)
            hb = b.step(seq[:, MB.RESUME_T + 1], MB.RESUME_T + 1)
        assert MM._rel(ha[24], hb[24]) > 1e-6
    finally:
        hooks.close()


def test_run_u1_tiny():
    """The whole U1 (both checkpoints' code paths) on 48 items: files, shapes, controls, the P3 Gram identity."""
    import lm_schema as S
    MM, m = _model()
    n = 48
    toks = _tokens(n, 2)
    probe_json = dict(items=[dict(id=i, source=MB.SOURCES[i % 5], tokens=t.tolist()) for i, t in enumerate(toks)],
                      S=dict(permutation=[np.random.RandomState(i).permutation(MB.SEQ - 1).tolist()
                                          for i in range(n)]))
    probe = S.Probe(probe_json)
    with tempfile.TemporaryDirectory(dir=None) as d:
        work, scratch = Path(d) / 'work', Path(d) / 'scratch'
        scratch.mkdir()
        info = MM.run_u1(m, 'trained', probe, work, 'cpu', log=lambda s: None, chunk32=16, chunk64=16,
                         scratch=scratch)
        assert not any(scratch.iterdir())                    # the raw Grams were moved to work
        info0 = MM.run_u1(m, 'init', probe, work, 'cpu', log=lambda s: None)
        assert info['C1a']['pass'] and info['C1b']['pass'] and info['C2']['identical'] and info0['C1b']['pass']
        assert info['P3_gram_identity']['pass'], info['P3_gram_identity']
        L, P = len(MB.LAYERS), len(MB.POSITIONS)
        shapes = dict(act=(L, P, n, 16), state=(L, P, n, n), stack=(L, P, n, n), bag=(P, n, 16),
                      parts=(L, P, 2, n, n), raw=(P, 48, n, n), actS=(L, n, 16), stateS=(L, n, n), stackS=(L, n, n))
        for k, shape in shapes.items():
            assert np.load(work / f'{k}_trained.npy', mmap_mode='r').shape == shape, k
        assert not (work / 'parts_init.npy').exists() and not (work / 'raw_init.npy').exists()
        u1 = json.loads((work / 'u1_trained.json').read_text())
        assert len(u1['norm_mean']) == L and len(u1['norm_mean'][0]) == MB.SEQ + 1
        # the stored state/stack Grams are the views of the stored raw Grams (vector route vs Gram route)
        raw = np.load(work / 'raw_trained.npy')
        stack = np.load(work / 'stack_trained.npy')
        state = np.load(work / 'state_trained.npy')
        everyone = np.arange(n)
        for pi in range(P):
            assert torch.max(torch.abs(MC.RawGramCell(raw[pi]).subset_gram(everyone) - torch.from_numpy(stack[-1, pi]))) < 1e-10
            assert torch.max(torch.abs(MC.RawGramCell(raw[pi, 14:16]).subset_gram(everyone) -
                                       torch.from_numpy(state[1, pi]))) < 1e-10           # layer 8 = constituents 14, 15
        # probe S keeps token 128, so the S state at 128 differs from A's (different prefixes) but shapes agree
        assert np.abs(np.load(work / 'stateS_trained.npy') - state[:, -1]).max() > 1e-6
    print(f"  U1 on the tiny model: C1a {info['C1a']['max_rel_err']:.1e}, C1b {info['C1b']['max_rel_err']:.1e}, "
          f"Gram identity {info['P3_gram_identity']['max_dev']:.1e}")


# ----------------------------------------------------------------------------------------- Grams and trees
def _constituents(seed, n=120, dims=(6, 20, 9, 30, 5, 12), offset=3.0):
    """Synthetic constituents with unequal item norms (log-normal per item) and large non-zero means."""
    rng = np.random.default_rng(seed)
    lab = np.arange(n) % 4
    out = []
    for d in dims:
        centres = rng.normal(size=(4, d)) * 1.2
        x = centres[lab] + rng.normal(size=(n, d))
        x = x * np.exp(0.8 * rng.normal(size=(n, 1))) + offset * rng.normal(size=(1, d))
        out.append(x)
    return out, lab


def test_p3_gram_identity():
    """State and stack Grams (the U1/fixture code path) equal the standard pipeline on the explicit concatenation
    of the standard-pipeline-normalised constituents scaled by 1/sqrt(k), to 1e-12."""
    cons, _ = _constituents(3, n=60, dims=(5, 13) * 24)
    by_layer = {m: (torch.from_numpy(cons[2 * m - 2]), torch.from_numpy(cons[2 * m - 1])) for m in range(1, 25)}
    state, stack, parts, raw = MC.position_grams(by_layer.__getitem__, 60, keep_parts=True, keep_raw=True)
    rec = MC.gram_identity(by_layer.__getitem__, state, stack)
    assert rec['pass'], rec
    for li, l in enumerate(MB.LAYERS):
        for k in range(2):
            assert torch.max(torch.abs(torch.from_numpy(parts[li, k]) - C.standard_gram(by_layer[l][k]))) < 1e-15
    assert len(raw) == 48 and np.allclose(raw[0], cons[0] @ cons[0].T)
    RESULTS['P3_gram_identity_synthetic'] = rec['max_dev']
    print(f"  P3 Gram identity (24 layers x 2 synthetic constituents): max deviation {rec['max_dev']:.1e}")


def test_tree_gram_route():
    """SPEC 6: the Gram route (raw Grams restricted to S, double-centred over S, normalised, averaged,
    renormalised) equals a full rebuild from the raw constituent vectors on NON-ROOT subsets, to 1e-12; the walks
    built on it agree; restricting the parent's centred Gram does not (so the test can fail)."""
    cons, lab = _constituents(5)
    cell = MC.RawGramCell([x @ x.T for x in cons])
    rng = np.random.default_rng(0)
    worst = 0.0
    subsets = [np.flatnonzero(lab < 2), np.flatnonzero(lab == 3)] + [np.sort(rng.choice(120, k, replace=False))
                                                                     for k in (90, 64, 40)]
    for s in subsets:
        a = cell.subset_gram(s)
        b = MC.vector_subset_gram(cons, s)
        worst = max(worst, float(torch.max(torch.abs(a - b))))
        pa, ia = C.walk(a, k=10, return_mask=True)
        pb, ib = C.walk(b, k=10, return_mask=True)
        assert np.array_equal(ia['mask'], ib['mask']) and float(torch.max(torch.abs(pa - pb))) < 1e-10
    assert worst < 1e-12, worst
    parent = MC.vector_subset_gram(cons, np.arange(120))
    s = subsets[0]
    assert float(torch.max(torch.abs(parent[np.ix_(s, s)] - MC.vector_subset_gram(cons, s)))) > 1e-3
    # through lm_tree: node operators of the Gram cell = those of an explicit rebuild
    ps, mask = Z.node_operators([cell], s)
    pb, ib = C.walk(MC.vector_subset_gram(cons, s), return_mask=True)
    assert float(torch.max(torch.abs(ps[0] - pb))) < 1e-10 and np.array_equal(mask, ib['mask'])
    # a whole tree grows on Gram cells, and its non-root nodes match explicit rebuilds
    cells = [MC.RawGramCell([x @ x.T for x in _constituents(seed)[0]]) for seed in (5, 6)]
    nodes = Z.grow(cells, max_depth=2, min_n=40)
    assert len(nodes) >= 3 and nodes[0]['n'] == 120
    cons6 = _constituents(6)[0]
    for nd in nodes[1:]:
        for c, cs in zip(cells, (cons, cons6)):
            g = c.subset_gram(nd['members'])
            worst = max(worst, float(torch.max(torch.abs(g - MC.vector_subset_gram(cs, nd['members'])))))
    assert worst < 1e-12, worst
    RESULTS['T4_gram_route_synthetic'] = worst
    print(f'  T4 Gram route vs full rebuild from raw vectors (non-root subsets): max deviation {worst:.1e}')


def test_lm_tree_provider_unchanged_for_vectors():
    x = np.random.default_rng(1).normal(size=(80, 7)) + 5
    s = np.arange(10, 70)
    assert torch.equal(Z.cell_gram(x, s), C.standard_gram(x[s]))


# ------------------------------------------------------------------------------------- provenance, budget
def test_budget_gate_and_keys():
    assert MB.FINGERPRINT != MB.LC.FINGERPRINT
    key = MB.unit_key('abc', [])
    assert key['construction_fingerprint'] == MB.FINGERPRINT
    assert MB.gate_ok(dict(key, **{'pass': True}), key)
    assert not MB.gate_ok(dict(MB.LC.unit_key('abc', []), **{'pass': True}), key)     # a Pythia record
    assert not MB.gate_ok(dict(key, **{'pass': True}), MB.unit_key('abc', [3]))
    with tempfile.TemporaryDirectory() as d:
        marker, out = Path(d) / 't.json', Path(d) / 'o.npy'
        out.write_bytes(b'x')
        marker.write_text(json.dumps(dict(seconds=1, **key)))
        assert MB.cached_unit(marker, [out], key) is not None
        marker.write_text(json.dumps(dict(seconds=1, **MB.LC.unit_key('abc', []))))
        assert MB.cached_unit(marker, [out], key) is None
    # the default worst case fits the ceiling with room for one full retry of U1_trained
    rates = dict(U0=(0, 2, 4, 900), U1=(1, 4, 32, 1500), U2=(0, 8, 32, 2400), U3=(0, 8, 32, 3600))
    rate = lambda g, c, m: g * .000542 + c * .0000131 + m * .00000222
    worst = {'U0': 900 * rate(*rates['U0'][:3])}
    for ck in MB.CHECKPOINTS:
        worst[f'U1_{ck}'] = 1500 * rate(*rates['U1'][:3])
        worst[f'U2_{ck}'] = 2400 * rate(*rates['U2'][:3])
    worst['U3'] = 3600 * rate(*rates['U3'][:3])
    coord = 4 * 3600 * (.25 * .0000131 + 1 * .00000222)
    total = MB.budget([], worst, set(), coord)['total']
    assert total + worst['U1_trained'] < 6.0, total
    RESULTS['worst_case_reservation'] = total
    print(f'  worst-case reservation at launch ${total:.2f} (ceiling $6)')


def test_cache_position_rule():
    assert MB.cache_position(0) == 3 and MB.cache_position(128) == 128


# --------------------------------------------------------------------------------------- fixture outputs
def test_fixtures_validate():
    import mb_schema as S
    for name in ('_fixture', '_fixture_flat'):
        root = HERE / 'data' / name
        if not (root / 'validation.json').exists():
            print(f'  {name}: not built (python -I mb_schema.py fixture data/{name}{" --flat" if "flat" in name else ""})')
            continue
        report = S.validate(root, require_layouts=(root / 'umap.npz').exists())
        val = json.loads((root / 'validation.json').read_text())
        assert val['P1']['pass'] and val['P2']['pass'] and val['P3']['pass'], name
        assert val['T4_gram_route']['pass']
        print(f"  {name}: valid ({report['records']} SEBA records, worst label agreement "
              f"{report['minimum_seba_label_agreement']:.3f}); P3 Gram identity "
              f"{val['P3']['gram_identity']['max_dev']:.1e}; T4 root Gram dev {val['T4_gram_route']['root_gram_dev']:.1e}")


def test_cross_arch_on_fixtures():
    import cross_arch
    pythia, mamba = HERE.parent / 'lm_coherent' / 'data' / '_fixture', HERE / 'data' / '_fixture'
    if not (pythia / 'meta.json').exists() or not (mamba / 'meta.json').exists():
        print('  cross_arch: fixtures absent, skipped')
        return
    out = cross_arch.compute(pythia, mamba, fixture=True)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'cross_arch.json'
        path.write_text(json.dumps(out))
        assert cross_arch.validate(path)['rows'] == 108
    try:
        cross_arch.compute(pythia, mamba, fixture=False)
        raise AssertionError('fixture probes accepted as the real probe')
    except AssertionError as e:
        assert 'probe' in str(e) or 'fixture' in str(e)


TESTS = [test_cache_position_rule, test_budget_gate_and_keys, test_lm_tree_provider_unchanged_for_vectors,
         test_p3_gram_identity, test_tree_gram_route, test_slow_path_and_init_determinism, test_conv_buffer_claim,
         test_c1b_fails_closed_on_nan, test_c1a_c1b_tiny, test_run_u1_tiny, test_fixtures_validate, test_cross_arch_on_fixtures]

if __name__ == '__main__':
    torch.set_num_threads(2)
    for test in TESTS:
        start = time.time()
        test()
        print(f'ok  {test.__name__} ({time.time() - start:.1f}s)', flush=True)
    print(json.dumps(RESULTS))
    print(f'all {len(TESTS)} mamba checks passed')
