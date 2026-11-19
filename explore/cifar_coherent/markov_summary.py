"""Aggregate markov_<step>_<rule>.npz (SPEC_COARSE.md section 6) into data/r20s0b/markov_summary.json and
data/r20s0b/markov_facts.json (quoted on the site as {{fact:c.mk.<name>}} via coarse_facts.json merge).

uv run --with numpy python markov_summary.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / 'data' / 'r20s0b' / 'coarse'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
MK = list(range(2, 17))
METHODS = ['spectral', 'pca', 'shared', 'random', 'labels']
STEP_F = ['defect', 'I_fine', 'I_coarse', 'defect_chi2', 'X_fine', 'X_coarse']
TRIPLE_F = ['cmi_coarse', 'cmi_fine', 'I_next', 'H_cond']
WORLDS = ['real', 'uniform']


def per_unit(z):
    """Per (world, method, k): per-site defect fraction, memory fraction, CK error (horizon >= 2) and its scale."""
    st, tr, ck, pairs = z['step'], z['triple'], z['ck'], z['ck_pairs']
    fine = st[..., 1]
    frac = np.where(fine > 1e-12, st[..., 0] / np.where(fine > 1e-12, fine, 1), np.nan)
    h = tr[..., 3]
    mem = np.where(h > 1e-12, tr[..., 0] / np.where(h > 1e-12, h, 1), np.nan)
    memf = np.where(h > 1e-12, tr[..., 1] / np.where(h > 1e-12, h, 1), np.nan)
    horizon = pairs[:, 1] - pairs[:, 0]
    keep = horizon >= 2
    return dict(defect_frac=frac, mem_frac=mem, memfine_frac=memf, ck=ck[..., keep, 0], ck_ind=ck[..., keep, 1],
                horizon=horizon[keep], nmi=z['nmi'])


def main():
    units = {}
    for step in STEPS:
        for rule in range(5):
            f = D / f'markov_{step}_{rule}.npz'
            if f.exists():
                units[step, rule] = per_unit(np.load(f))
    out = dict(steps=STEPS, sites=SITES, k=MK, methods=METHODS, worlds=WORLDS, units={})
    for (step, rule), u in units.items():
        rec = {}
        for key in ('defect_frac', 'mem_frac', 'memfine_frac'):
            rec[key] = np.round(np.nan_to_num(u[key], nan=-1), 5).tolist()
        rec['ck_by_h'] = {}
        for hz in range(2, 12):
            sel = u['horizon'] == hz
            rec['ck_by_h'][hz] = [np.round(np.nanmean(u['ck'][..., sel], -1), 5).tolist(),
                                  np.round(np.nanmean(u['ck_ind'][..., sel], -1), 5).tolist()]
        rec['nmi'] = np.round(np.nan_to_num(u['nmi'], nan=-1), 4).tolist()
        out['units'][f'{step}_{rule}'] = rec
    (HERE / 'data' / 'r20s0b' / 'markov_summary.json').write_text(json.dumps(out))
    return units


def facts(units):
    fin = STEPS[-1]
    f = {}
    rules = [r for r in range(5) if (fin, r) in units]
    for k in (4, 10):
        ki = MK.index(k)
        for mi, m in enumerate(METHODS):
            if m == 'labels' and k != 10:
                continue
            for w, wn in enumerate(WORLDS):
                d = np.nanmedian([units[fin, r]['defect_frac'][w, mi, ki] for r in rules])
                mm = np.nanmedian([units[fin, r]['mem_frac'][w, mi, ki] for r in rules])
                ck = np.nanmedian([units[fin, r]['ck'][w, mi, ki] for r in rules])
                ind = np.nanmedian([units[fin, r]['ck_ind'][w, mi, ki] for r in rules])
                f[f'mk.k{k}.{m}.{wn}.defect'] = f'{d:.2f}'
                f[f'mk.k{k}.{m}.{wn}.mem'] = f'{mm:.3f}'
                f[f'mk.k{k}.{m}.{wn}.ck'] = f'{ck:.3f}'
                f[f'mk.k{k}.{m}.{wn}.ckind'] = f'{ind:.3f}'
            f[f'mk.k{k}.{m}.nmi'] = f"{np.nanmedian([units[fin, r]['nmi'][mi, ki] for r in rules]):.2f}"
    f['mk.rules'] = len(rules)
    ki = MK.index(10)
    med = lambda st, key, w, mi, site=None: np.nanmedian([np.nanmedian(units[st, r][key][w, mi, ki]) if site is None
                                                          else units[st, r][key][w, mi, ki, site] for r in range(5)])
    f['mk.late.spectral.defect'] = f"{np.nanmedian([med(fin, 'defect_frac', 0, 0, s) for s in (9, 10)]):.2f}"
    f['mk.mid.spectral.defect_max'] = f"{max(med(fin, 'defect_frac', 0, 0, s) for s in range(9)):.2f}"
    f['mk.first.spectral.mem'] = f"{med(STEPS[0], 'mem_frac', 0, 0):.3f}"
    f['mk.first.spectral.mem_unif'] = f"{med(STEPS[0], 'mem_frac', 1, 0):.3f}"
    f['mk.final.spectral.mem_logits'] = f"{med(fin, 'mem_frac', 0, 0, 9):.3f}"
    for hz in (2, 8):
        for mi, m in ((0, 'spectral'), (2, 'shared')):
            for w, wn in enumerate(WORLDS):
                sel = lambda r: units[fin, r]['horizon'] == hz
                f[f'mk.h{hz}.{m}.{wn}.ck'] = f"{np.nanmedian([units[fin, r]['ck'][w, mi, ki][sel(r)].mean() for r in range(5)]):.3f}"
            f[f'mk.h{hz}.{m}.ckind'] = f"{np.nanmedian([units[fin, r]['ck_ind'][0, mi, ki][sel(r)].mean() for r in range(5)]):.3f}"
    return f


if __name__ == '__main__':
    u = main()
    f = facts(u)
    p = HERE / 'data' / 'r20s0b' / 'coarse_facts.json'
    cf = json.loads(p.read_text())
    cf = {k: v for k, v in cf.items() if not k.startswith('mk.')}
    cf.update(f)
    p.write_text(json.dumps(cf, indent=1))
    print(json.dumps(f, indent=1))
