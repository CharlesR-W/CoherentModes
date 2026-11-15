"""Average versus product blocks, from the exported files (no recomputation of spectra).

    uv run --with numpy --with scipy python product_analysis.py [tag]   -> data/<tag>/product_summary.json

For every block a < b, kernel rule and r: matched Jaccard (Hungarian, hard SEBA labels,
unassigned images belong to no set) between the average block's sets and the product's
start / end sets; the same against CIFAR classes (validation only); mean cos^2 of principal
angles between span(psi_1..psi_r-1) of the average and of each product side; and, for each
product mode, the snapshot with the smallest step gain.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent


def jaccard(a, ra, b, rb):
    A = (a[:, None] == np.arange(ra)).astype('f8'); B = (b[:, None] == np.arange(rb)).astype('f8')
    inter = A.T @ B; union = A.sum(0)[:, None] + B.sum(0)[None] - inter
    m = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / ra)


def span_agree(u, v):
    qa = np.linalg.qr(u - u.mean(0))[0]; qb = np.linalg.qr(v - v.mean(0))[0]
    return float(np.square(qa.T @ qb).sum() / u.shape[1])


def main(tag='r20s0b'):
    root = HERE / 'data' / tag
    meta = json.loads((root / 'meta.json').read_text())
    labels = np.load(root / 'probe.npz')['labels']
    ranks = meta['seba_r']
    out = dict(tag=tag, ranks=ranks, rows=[])
    for fam, names, prs in (('layer', meta['steps'], meta['layer_pairs']), ('time', meta['sites'], meta['time_pairs'])):
        for idx, name in enumerate(names):
            z = {op: np.load(root / f'{fam}{suf}_{name}.npz') for op, suf in (('avg', ''), ('start', 'prodR'), ('end', 'prodL'))}
            arr = {op: {k: z[op][k] for k in ('vals', 'vecs', 'seba_label', *(('gain_psi',) if op != 'avg' else ()))} for op in z}
            for mi in range(len(meta['modes'])):
                for pi, (a, b) in enumerate(prs):
                    if a == b:
                        continue
                    row = dict(fam=fam, anchor=idx, mode=mi, a=a, b=b,
                               lam=arr['avg']['vals'][mi, pi, 1:4].round(5).tolist(),
                               sigma=arr['start']['vals'][mi, pi, 1:4].round(5).tolist())
                    for op in ('start', 'end'):
                        g = arr[op]['gain_psi'][mi, pi, :3, a:b + 1]
                        row[f'min_gain_at_{op}'] = (a + g.argmin(-1)).tolist()
                    for ri, r in enumerate(ranks):
                        A = arr['avg']['seba_label'][mi, pi, ri]
                        row[f'cls_avg_r{r}'] = round(jaccard(A, r, labels, 10), 4)
                        for op in ('start', 'end'):
                            B = arr[op]['seba_label'][mi, pi, ri]
                            row[f'avg_{op}_r{r}'] = round(jaccard(A, r, B, r), 4)
                            row[f'cls_{op}_r{r}'] = round(jaccard(B, r, labels, 10), 4)
                            if r > 1:
                                row[f'span_{op}_r{r}'] = round(span_agree(arr['avg']['vecs'][mi, pi, :r - 1].T.astype('f8'),
                                                                          arr[op]['vecs'][mi, pi, :r - 1].T.astype('f8')), 4)
                    out['rows'].append(row)
            print(fam, name, flush=True)
    (root / 'product_summary.json').write_text(json.dumps(out, separators=(',', ':')))


if __name__ == '__main__':
    main(*sys.argv[1:])
