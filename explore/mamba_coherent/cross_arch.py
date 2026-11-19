"""Mamba vs Pythia on the same snippets (SPEC section 5, comparison `cross_arch`), computed locally, post hoc, from
both studies' PUBLIC files. Writes `<mamba dir>/cross_arch.json`.

    uv run --with torch --with numpy --with scipy python -I cross_arch.py ../lm_coherent/data/p160t data/m130
    (fixtures: ... ../lm_coherent/data/_fixture data/_fixture --fixture)

Pythia L(2k) against Mamba L(4k), k = 1..6, every t, Pythia step 143000 vs Mamba `trained`:
point vs point, Pythia slab vs Mamba state, Pythia slab vs Mamba stack. a_3, a_6 (span agreement of psi_1..r with
the constant projected out) and the r = 4 matched Jaccard with its permutation null (one shared bank, as both
studies). Asserts identical item lists and no removed items in either; for real runs also the same probe file.
Also cross-tabulates the two T1 root cuts (Pythia T1 = L8 reading block, Mamba T1 = L16 reading block).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mb_construction as MB  # noqa: E402
import lm_cells as C  # noqa: E402
import lm_construction as LC  # noqa: E402
import lm_schema as S  # noqa: E402
from schema_v2 import RANKS  # noqa: E402

PAIRS = [('point_point', 0, 0), ('slab_state', 1, 1), ('slab_stack', 1, 2)]     # (name, Pythia view, Mamba view)
R4 = RANKS.index(4)


def _root_sides(trees, tree_id, n):
    t = next(t for t in trees if t['id'] == tree_id)
    nodes = t['nodes']
    side = np.full(n, -1)
    for k, child in enumerate(nodes[0]['children']):
        side[nodes[child]['members']] = k
    return side


def compute(pythia_dir, mamba_dir, fixture=False):
    pythia_dir, mamba_dir = Path(pythia_dir), Path(mamba_dir)
    pm = json.loads((pythia_dir / 'meta.json').read_text())
    mm = json.loads((mamba_dir / 'meta.json').read_text())
    assert pm['items'] == mm['items'], 'item lists differ'
    assert not pm['removed_items'] and not mm['removed_items'], 'removed items present'
    assert pm['layers'] == LC.LAYERS and mm['layers'] == MB.LAYERS and pm['positions'] == mm['positions']
    psha = MB.probe_identity((pythia_dir / 'probe.json').read_bytes())
    msha = MB.probe_identity((mamba_dir / 'probe.json').read_bytes())
    if not fixture:
        assert psha == msha == MB.PROBE_SHA, 'the two studies did not use the same probe file'
        assert not pm['fixture'] and not mm['fixture']
    n = mm['n']
    perms = C.permutation_bank(n)
    ps = np.load(pythia_dir / f'single_{MB.PYTHIA_FINAL}.npz')
    ms = np.load(mamba_dir / f'single_{MB.FINAL}.npz')
    rows = []
    for name, pv, mv in PAIRS:
        for pl, ml in MB.CROSS_ARCH_PAIRS:
            pli, mli = LC.LAYERS.index(pl), MB.LAYERS.index(ml)
            for pi, t in enumerate(MB.POSITIONS):
                a, b = (pv, pli, pi), (mv, mli, pi)
                va = torch.from_numpy(S._decode(ps, a, n))
                vb = torch.from_numpy(S._decode(ms, b, n))
                j, null, ceil = C.matched_jaccard(ps['seba_label'][a][R4], ms['seba_label'][b][R4], 4, 4, perms)
                rows.append(dict(kind='cross_arch', pair=name, pythia_layer=pl, layer=ml, t=t,
                                 a3=C.span_agreement(va, vb, 3), a6=C.span_agreement(va, vb, 6), jacc=float(j),
                                 jacc_null=[float(x) for x in null], jacc_ceiling=float(ceil)))
    pt = json.loads((pythia_dir / 'trees.json').read_text())
    mt = json.loads((mamba_dir / 'trees.json').read_text())
    sp, sm = _root_sides(pt, 'T1', n), _root_sides(mt, 'T1', n)
    cross = [[int(((sp == i) & (sm == j)).sum()) for j in range(2)] for i in range(2)]
    jt, nt, _ = C.matched_jaccard(sp, sm, 2, 2, perms)
    return dict(rows=rows, t1_root_crosstab=dict(rows='Pythia T1 root child 0/1', cols='Mamba T1 root child 0/1',
                                                 counts=cross, jacc=float(jt), jacc_null=[float(x) for x in nt]),
                pythia=dict(dir=str(pythia_dir), fingerprint=pm['construction_fingerprint'], step=MB.PYTHIA_FINAL),
                mamba=dict(dir=str(mamba_dir), fingerprint=mm['construction_fingerprint'], checkpoint=MB.FINAL),
                probe_sha=dict(pythia=psha, mamba=msha), fixture=bool(fixture), n=n,
                definition='a_r = mean cos^2 of principal angles between span(psi_1..psi_r) of the two cells; jacc = '
                           'r = 4 matched Jaccard (Hungarian), null over 200 permutations [mean, 2.5 %, 97.5 %]')


def validate(path):
    d = json.loads(Path(path).read_text())
    rows = d['rows']
    assert len(rows) == len(PAIRS) * len(MB.CROSS_ARCH_PAIRS) * len(MB.POSITIONS), len(rows)
    for r in rows:
        assert r['kind'] == 'cross_arch' and -1e-9 <= r['a3'] <= 1 + 1e-9 and -1e-9 <= r['a6'] <= 1 + 1e-9
        assert 0 <= r['jacc'] <= 1 and len(r['jacc_null']) == 3
    assert sum(map(sum, d['t1_root_crosstab']['counts'])) == d['n']
    return dict(rows=len(rows), pass_=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = compute(args[0], args[1], fixture='--fixture' in sys.argv)
    path = Path(args[1]) / 'cross_arch.json'
    path.write_text(json.dumps(out, allow_nan=False))
    print(json.dumps(validate(path)), json.dumps(out['t1_root_crosstab']))
