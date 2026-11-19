"""Aggregate coarse_<step>_<rule>.npz (SPEC_COARSE.md) into data/r20s0b/coarse_summary.json.

uv run --with numpy python coarse_summary.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / 'data' / 'r20s0b' / 'coarse'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
RULES = ['kNN graph, k = 15', 'perplexity 100', 'perplexity 300', 'kNN scale', 'half-mass, calibrated']
K = list(range(2, 17)) + [20, 24, 32]
METHODS = ['spectral', 'seba', 'pca', 'random', 'labels', 'pred']
SCORES = ['I_start', 'I_end', 'I_two', 'keff_start', 'keff_end', 'agree', 'agree_indep', 'agree_min_fwd',
          'agree_min_rev', 'ncut', 'maxleak', 'I_start_P', 'I_two_P', 'ncut_P', 'maxleak_P', 'I_start_sd']
PAIRS = [(a, b) for a in range(12) for b in range(a, 12)]
S = {s: i for i, s in enumerate(SCORES)}


def ceiling(sv, k):
    return float((np.asarray(sv[1:k], float) ** 2).sum())


def gap_scale(x, kmax=32, clear_ratio=1.15, margin=1.1):
    """Largest ratio x[k]/x[k-1] (x increasing, x[k-1] = k-th value), k in [2, kmax]: a gap after k modes."""
    x = np.asarray(x, float)
    ks = np.arange(2, kmax + 1)
    lo, hi = x[ks - 1], x[ks]
    ok = lo > 1e-8
    r = np.where(ok, hi / np.maximum(lo, 1e-300), np.nan)
    o = np.argsort(-np.nan_to_num(r, nan=-1))
    top, sec = r[o[0]], r[o[1]]
    return dict(k=int(ks[o[0]]), ratio=float(top), k2=int(ks[o[1]]), ratio2=float(sec),
                clear=bool(np.isfinite(top) and top >= clear_ratio and top > margin * sec))


def block_gap(sv, cell):
    """Cells: Laplacian scale 1 - sigma (laziness compresses sigma ratios); chains: sigma_k / sigma_{k+1}."""
    sv = np.asarray(sv, float)
    if cell:
        return gap_scale(1 - sv)
    inv = 1 / np.maximum(sv, 1e-300)       # increasing; ratio inv[k]/inv[k-1] = sigma_k / sigma_{k+1}
    return gap_scale(inv)


def main():
    out = dict(K=K, methods=METHODS, scores=SCORES, steps=STEPS, sites=SITES, rules=RULES, units={}, missing=[])
    for step in STEPS:
        for rule in range(5):
            f = D / f'coarse_{step}_{rule}.npz'
            if not f.exists():
                out['missing'].append([step, rule]); continue
            z = np.load(f)
            sv, sc = z['sv'].astype(float), z['scores'].astype(float)
            blocks = []
            for bi, (a, b) in enumerate(PAIRS):
                cell = a == b
                F = [ceiling(sv[bi], k) for k in K]
                rec = dict(a=a, b=b, total=float(z['total'][bi]), sv=np.round(sv[bi, :65], 5).tolist(), F=np.round(F, 5).tolist(),
                           gap=block_gap(sv[bi], cell), check=np.round(z['check'][bi, :, 1], 5).tolist(),
                           part=np.round(z['part_start'][bi], 3).tolist())
                for mi, m in enumerate(METHODS):
                    rec[m] = {s: [None if not np.isfinite(v) else round(float(v), 6) for v in sc[bi, mi, :, S[s]]]
                              for s in (SCORES if cell else SCORES[:9] + ['I_start_sd'])}
                if cell:
                    svp = z['sv_P'][bi].astype(float)
                    rec['sv_P'] = np.round(svp[:65], 5).tolist(); rec['total_P'] = float(z['total_P'][bi])
                    rec['F_P'] = np.round([ceiling(svp, k) for k in K], 5).tolist()
                    rec['kyfan'] = np.round([float((1 - sv[bi, :k]).sum()) for k in K], 5).tolist()
                    rec['cheeger_lo'] = np.round([float(1 - sv[bi, k - 1]) / 2 for k in K], 5).tolist()
                blocks.append(rec)
            out['units'][f'{step}_{rule}'] = blocks
    (HERE / 'data' / 'r20s0b' / 'coarse_summary.json').write_text(json.dumps(out, allow_nan=False))
    print('units', len(out['units']), 'missing', out['missing'])




def facts():
    """Numbers quoted on the site ({{fact:c.<name>}}), written to data/r20s0b/coarse_facts.json."""
    d = json.loads((HERE / 'data' / 'r20s0b' / 'coarse_summary.json').read_text())
    U = d['units']; k10 = K.index(10); k32 = K.index(32)
    blk = lambda st, r, a, b: next(x for x in U[f'{st}_{r}'] if x['a'] == a and x['b'] == b)
    eff = lambda x, m, s='I_start', ki=k10: x[m][s][ki] / x['F'][ki]
    f = {}
    fin = STEPS[-1]
    for si in (0, 6, 9, 11):
        cs = [blk(fin, r, si, si) for r in range(5)]
        for m in ('spectral', 'seba', 'pca', 'random', 'labels', 'pred'):
            f[f'cell.{SITES[si]}.{m}'] = f"{np.mean([eff(x, m) for x in cs]):.2f}"
    cells = [blk(st, r, s, s) for st in STEPS for r in range(5) for s in range(12)]
    f['cell.total_min'] = f"{min(x['total'] for x in cells):.0f}"
    f['cell.total_max'] = f"{max(x['total'] for x in cells):.0f}"
    f['cell.F32_max'] = f"{max(x['F'][k32] for x in cells):.1f}"
    ratios = [x['spectral']['ncut'][k10] / x['kyfan'][k10] for x in cells]
    f['cell.ncut_kf_min'] = f"{min(ratios):.2f}"; f['cell.ncut_kf_max'] = f"{max(ratios):.2f}"
    f['cell.ncut_kf_final_med'] = f"{np.median([blk(fin, r, s, s)['spectral']['ncut'][k10] / blk(fin, r, s, s)['kyfan'][k10] for r in range(5) for s in range(12)]):.2f}"
    finc = [blk(fin, r, s, s) for r in range(5) for s in range(12)]
    f['cell.n_final'] = len(finc)
    f['cell.gap2_clear'] = sum(1 for x in finc if x['gap']['clear'] and x['gap']['k'] == 2)
    late = [blk(fin, r, s, s) for r in range(5) for s in (9, 10, 11)]
    f['cell.late_n'] = len(late)
    f['cell.late_second10'] = sum(1 for x in late if x['gap']['k2'] in (9, 10, 11, 12) or x['gap']['k'] in (9, 10, 11, 12))
    for si in (3, 9):
        f[f'cell.{SITES[si]}.labels_first'] = f"{np.mean([eff(blk(STEPS[0], r, si, si), 'labels') for r in range(5)]):.2f}"
        f[f'cell.{SITES[si]}.labels_last'] = f"{np.mean([eff(blk(fin, r, si, si), 'labels') for r in range(5)]):.2f}"
    x = blk(fin, 0, 9, 11)
    f['chain.b9_logits.s10'] = f"{x['sv'][9]:.3f}"; f['chain.b9_logits.s11'] = f"{x['sv'][10]:.3f}"
    f['chain.b9_logits.clear_rules'] = sum(1 for r in range(5) if blk(fin, r, 9, 11)['gap']['clear'] and blk(fin, r, 9, 11)['gap']['k'] == 10)
    f['chain.b9_logits.clear_steps_rule0'] = ', '.join(f'{st:,}' for st in STEPS if blk(st, 0, 9, 11)['gap']['clear'] and blk(st, 0, 9, 11)['gap']['k'] == 10) or 'none'
    for L in (1, 11):
        rs = [blk(fin, r, a, a + L) for r in range(5) for a in range(12 - L)]
        f[f'chain.L{L}.F10'] = f"{np.median([y['F'][k10] for y in rs]):.3g}"
        for m in ('spectral', 'pca', 'labels', 'random'):
            f[f'chain.L{L}.two.{m}'] = f"{np.median([eff(y, m, 'I_two') for y in rs]):.2f}"
            f[f'chain.L{L}.start.{m}'] = f"{np.median([eff(y, m) for y in rs]):.2f}"
    (HERE / 'data' / 'r20s0b' / 'coarse_facts.json').write_text(json.dumps(f, indent=1))
    return f


if __name__ == '__main__':
    import sys
    if '--facts-only' not in sys.argv:
        main()
    print(json.dumps(facts(), indent=1))
