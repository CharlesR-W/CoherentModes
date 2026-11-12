"""Question-led statistics for the Questions tab (SPEC.md section 2c).

Usage:
    uv run --with numpy --with scipy --with matplotlib python -I site/analysis.py <tag> [--data-dir PATH] [--out-dir PATH]

Reads `data/<tag>/` (or PATH) and writes `site/assets/questions.json` and `site/assets/fig/q_*.png`.
Nothing here enters the construction: every statistic compares already-computed hard SEBA sets with each
other, with CIFAR class labels (validation only) or with per-sample loss gradients (validation only).

Notation
--------
T checkpoints `meta.steps`, S sites `meta.sites`, M kappa modes `meta.modes`, ranks `meta.seba_r`, N = 2000.
Hard sets of a block: `seba_label[m, pair, ri, :]` (-1 = unassigned, else 0..r-1 in reliability order).
"Single-cell block (t, s)" = layer pair (s, s) of `layer_<steps[t]>.npz` (pair index = row-major index of
(a, b) over 0 <= a <= b < S). "Layer block a..b at t" = layer pair (a, b) of `layer_<steps[t]>.npz`.
"Time block (c, d) at s" = time pair (c, d) of `time_<site>.npz` (indexed by `meta.time_pairs`).
Classes = `probe.npz['labels']` (10 sets, every point assigned).

Statistic 1: matched Jaccard (SPEC 2c "Set overlap")
------------------------------------------------------
For hard labelings A (sets A_0..A_{rA-1}) and B (sets B_0..B_{rB-1}); unassigned points are in no set; rA and
rB are the nominal ranks (r for SEBA, 10 for classes), so an empty set is a row/column of zeros:
    J_ij = |A_i ∩ B_j| / |A_i ∪ B_j|  (0 if the union is empty).
Hungarian assignment (scipy linear_sum_assignment, maximise sum J) on the rA x rB matrix matches min(rA, rB)
pairs; score = mean of J over the matched pairs (so in [0, 1]; 1 iff identical partitions of the assigned
points with no empty set). Per-matched-pair values are kept where noted ("pairs" fields).
Null: the label vector of A (length N, including its -1 entries) is permuted uniformly at random, which keeps
A's set sizes and unassigned count and B fixed; the score is recomputed. One bank of n_perm = 200 permutations
of 0..N-1 is drawn once from `numpy.random.default_rng(meta.seed)` and reused for every comparison (each null
is individually exact; nulls of different comparisons share draws). Stored: null mean and the 2.5 / 97.5
percentiles (numpy "linear" interpolation) of the null scores.
Size ceiling (vs classes only, "vs_class_ceiling"): the same Hungarian mean over the matrix
min(|A_i|, |B_j|) / max(|A_i|, |B_j|), the largest matched Jaccard any partition with these set sizes could
reach. With r < 10 sets against 10 classes of ~200, it is far below 1; read vs_class against it.

Q1 "when" (q_when)
------------------
For every mode m, rank r, site s, checkpoint t:
  vs_final[s][t] = score(sets of single-cell block (t, s), sets of single-cell block (T-1, s)), with null
                   (A = the checkpoint-t sets). At t = T-1 it is a self-comparison (1 unless a set is empty).
  vs_class[s][t] = score(sets of (t, s), classes), with null (A = the SEBA sets).

Q2 "carry-over" (q_carry)
-------------------------
For every m, r, checkpoint t:
  matrix[t][a][a'] = score(sets of single-cell block (t, a), sets of (t, a')); symmetric; no null.
  adjacent: score((t, a), (t, a+1)) for a = 0..S-2 (A = site a's sets for the null).
  w2, w3: score(layer block a..a+w-1 at t, layer block a+w..a+2w-1 at t) for a = 0..S-2w, with null
          (A = the first block's sets).

Q3 "gradients" (q_grad)
-----------------------
G_t = `gradgram_<steps[t]>.npy` (cosine Gram of per-sample loss gradients, read as float32). For every m, r,
site s and time block "all" = time pair (0, T-1) and "late" = (T//2, T-1) at site s, take that block's hard
sets (one labeling, fixed across t) and at every checkpoint t that has G_t:
  within_t  = sum_{i != j, same set} G_t[i,j] / #{ordered pairs i != j in the same set}   (pair-weighted;
              unassigned points excluded; big sets weigh more)
  between_t = sum_{i, j assigned, different sets} G_t[i,j] / #{such ordered pairs}
  diff_t    = within_t - between_t.
The same three numbers for the 10 classes. Null: the set labels of the assigned points are permuted among the
assigned points (sizes kept; unassigned stay unassigned), n_perm_grad = 100 permutations (bank drawn once from
the same seeded generator, restricted to the assigned points by argsort of shared random keys); stored: null
mean and 2.5 / 97.5 percentiles of diff_t. (The null mean of diff is 0 in expectation exactly; its spread is
the informative part.)
Set alignment ("setalign"): mean of `points_<steps[t]>.npz['grad_cos']` (cosine of g_i with the mean probe
gradient) over each set's members, at each checkpoint t, plus the mean over all N points.
If no gradgram file exists, q_grad = {"available": false} and no figure is made.

Q4 decodability (q_decod)
-------------------------
From `decod.json` (results[i].views[view][site].ridge_acc / knn_acc; held-out test accuracy of closed-form
ridge and weighted cosine kNN, a validation fit). "by_site": x = site index (xticklabels = site names), one
series per decod checkpoint. "by_step": x = decod steps (symlog), one series per site. The gap view has no
pool/logits entry (they have no spatial axis); the gap charts reuse the flat value there and say so in "note".

JSON layout (site/assets/questions.json; floats rounded to 4 decimals; NaN -> null)
----------------------------------------------------------------------------------
{
 "meta": {tag, fixture, steps, sites, modes, seba_r, default_mode, default_r, n_perm, n_perm_grad, seed,
          generated, kept: {"<m>": [r, ...]} (which (mode, r) variants are stored), definitions: str},
 "q_when": {"<m>_<r>": {
     "vs_final", "vs_final_null", "vs_class", "vs_class_null", "vs_class_ceiling":
         heatmap {rows: sites, cols: steps, z[site][t], zrange [0, 1], ...}; *_null = null mean,
     "lines_vs_final", "lines_vs_class": lines {x: steps, xscale "symlog", series: one per site (name = site,
         y = score, band = [null 2.5 %, null 97.5 %]), colors "ordinal", yrange [0, 1]},
     "pairs_vs_class_final": {site: [[set index, class index, J], ...]} (Hungarian pairs at the last checkpoint)}},
 "q_carry": {"<m>_<r>": {
     "matrix": heatmap {rows = cols = sites, z[t][a][a'], slider {label, values: steps}, zrange [0, 1]},
     "adjacent": lines {x: 0..S-2, xticklabels "stem→b1"..., one series per checkpoint (name "step N"), no band},
     "adjacent_null": lines, same x and series names, y = null mean, band = null [2.5, 97.5] %,
     "w2", "w3": lines {x: 0..S-2w, xticklabels "stem–b1→b2–b3"..., one series per checkpoint, y = score,
         band = null [2.5, 97.5] %}}},
 "q_grad": {"available": bool, "yrange_diff": [lo, hi] (shared by every "lines"), "yrange_align": [lo, hi]
     (shared by every "setalign"), "steps": steps with a gradgram,
     "<m>_<r>_<s>_<block>" (s = site index, block in {"all", "late"}): {
        "lines": lines {x: steps, series: "SEBA sets" (diff), "CIFAR classes" (diff), "random sets (null)" (y =
                 null mean, band = null [2.5, 97.5] %, dash), refline y = 0,
                 "detail": {sets_within, sets_between, classes_within, classes_between} per x},
        "setalign": lines {x: steps, one series per set ("set 1".."set r", reliability order; colours = palette
                 sets) + "all points" (dash)}}},
 "q_decod": {"ridge_by_site_flat", "ridge_by_site_gap", "knn_by_site_flat", "knn_by_site_gap",
             "ridge_by_step_flat", "ridge_by_step_gap", "knn_by_step_flat", "knn_by_step_gap"}: lines objects
             (yrange [0, 1], refline 0.1 = chance, "note" on gap charts),
 "summary": {mode, r, and
     "when": per site {first_step: first checkpoint whose vs_final > 0.5 and > its null 97.5 % (the last
              checkpoint qualifies trivially), settled_step: first checkpoint from which this holds at every
              later checkpoint, vs_class_final, vs_class_null_hi_final, vs_class_ceiling_final},
     "carry": {mean_adjacent[t] (mean over a of adjacent score), mean_adjacent_null[t]},
     "grad": per site, block "all", last gradgram checkpoint {sets_diff, classes_diff, null_mean, null_hi}}
}
"""
import argparse
import datetime
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SITE_DIR = Path(__file__).resolve().parent
ROOT = SITE_DIR.parent
SEED = 20261112
N_PERM = 200
N_PERM_GRAD = 100
DEFAULT_R = 4

# ---------------------------------------------------------------- statistics


def _jaccard(A, b, ra, rb):
    """A (P, N) labels, b (N,) labels, -1 = unassigned. Returns J (P, ra, rb) float64."""
    P = A.shape[0]
    va = A >= 0
    mask = va & (b >= 0)[None]
    key = (np.arange(P)[:, None] * (ra * rb) + A * rb + b[None])[mask]
    inter = np.bincount(key, minlength=P * ra * rb).reshape(P, ra, rb)
    ka = (np.arange(P)[:, None] * ra + A)[va]
    sa = np.bincount(ka, minlength=P * ra).reshape(P, ra)
    sb = np.bincount(b[b >= 0], minlength=rb)
    union = sa[:, :, None] + sb[None, None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def _hungarian_mean(J):
    rows, cols = linear_sum_assignment(J, maximize=True)
    return J[rows, cols].sum() / min(J.shape), rows, cols


def matched_jaccard(a, b, ra, rb, perms=None):
    """Matched Jaccard of labelings a, b. Returns (score, [(i, j, J_ij)], null scores or None)."""
    J = _jaccard(a[None], b, ra, rb)[0]
    score, rows, cols = _hungarian_mean(J)
    pairs = [(int(i), int(j), float(J[i, j])) for i, j in zip(rows, cols)]
    null = None
    if perms is not None:
        Jn = _jaccard(a[perms], b, ra, rb)
        null = np.array([_hungarian_mean(Jn[p])[0] for p in range(len(perms))])
    return float(score), pairs, null


def size_ceiling(a, b, ra, rb):
    sa = np.bincount(a[a >= 0], minlength=ra).astype(float)
    sb = np.bincount(b[b >= 0], minlength=rb).astype(float)
    hi = np.maximum(sa[:, None], sb[None])
    C = np.where(hi > 0, np.minimum(sa[:, None], sb[None]) / np.maximum(hi, 1), 0.0)
    return float(_hungarian_mean(C)[0])


def null_summary(null):
    return float(null.mean()), float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))


def grad_within_between(G, l, r, order=None):
    """Pair-weighted mean gradient cosine within / between sets of labeling l (-1 excluded).

    `order` (P, N): rows are uniform random permutations of 0..N-1 (argsort of random keys); restricting a row
    to the assigned points gives a uniform random arrangement of them, which receives the observed labels
    (sizes kept, unassigned points stay unassigned). Returns arrays (1 + P,) within, between: observed first,
    then the nulls."""
    assigned = l >= 0
    I = np.flatnonzero(assigned)
    la = l[I].astype(np.int64)
    na = len(I)
    a = assigned.astype(np.float32)
    diag = float(np.diagonal(G)[I].sum(dtype=np.float64))
    total = float(a @ (G @ a))
    sq = float((np.bincount(la, minlength=r).astype(np.float64) ** 2).sum())
    wc, bc = sq - na, na * na - sq
    Q = I[None]
    if order is not None:
        Q = np.vstack([Q, order[assigned[order]].reshape(len(order), na)])
    P = Q.shape[0]
    H = np.zeros((len(l), P * r), np.float32)
    H[Q.ravel(), (np.arange(P)[:, None] * r + la[None]).ravel()] = 1
    win = ((H * (G @ H)).sum(0, dtype=np.float64)).reshape(P, r).sum(1)
    within = (win - diag) / wc if wc > 0 else np.full(P, np.nan)
    between = (total - win) / bc if bc > 0 else np.full(P, np.nan)
    return within, between


# ---------------------------------------------------------------- data

D = {}  # filled in main, inherited by forked workers


def load(data_dir):
    meta = json.loads((data_dir / 'meta.json').read_text())
    steps, sites = meta['steps'], meta['sites']
    lp = {tuple(p): i for i, p in enumerate(meta['layer_pairs'])}
    tp = {tuple(p): i for i, p in enumerate(meta['time_pairs'])}
    LL = np.stack([np.load(data_dir / f'layer_{st}.npz')['seba_label'] for st in steps])  # (T,M,P_L,R,N)
    TL = np.stack([np.load(data_dir / f'time_{s}.npz')['seba_label'] for s in sites])  # (S,M,P_T,R,N)
    with np.load(data_dir / 'probe.npz') as p:
        classes = p['labels'].astype(np.int64)
    D.update(meta=meta, steps=steps, sites=sites, lp=lp, tp=tp, LL=LL.astype(np.int64, copy=False),
             TL=TL, classes=classes, T=len(steps), S=len(sites), R=meta['seba_r'])


def _q12_job(job):
    m, ri = job
    T, S, LL, lp, cls, perms = D['T'], D['S'], D['LL'], D['lp'], D['classes'], D['perms']
    r = D['R'][ri]
    lab = lambda t, a, b: LL[t, m, lp[(a, b)], ri]
    out = {k: np.full((S, T), np.nan) for k in
           ('vf', 'vf_m', 'vf_lo', 'vf_hi', 'vc', 'vc_m', 'vc_lo', 'vc_hi', 'vc_ceil')}
    pairs_final = {}
    for s in range(S):
        fin = lab(T - 1, s, s)
        for t in range(T):
            a = lab(t, s, s)
            sc, _, nl = matched_jaccard(a, fin, r, r, perms)
            out['vf'][s, t] = sc
            out['vf_m'][s, t], out['vf_lo'][s, t], out['vf_hi'][s, t] = null_summary(nl)
            sc, pr, nl = matched_jaccard(a, cls, r, 10, perms)
            out['vc'][s, t] = sc
            out['vc_m'][s, t], out['vc_lo'][s, t], out['vc_hi'][s, t] = null_summary(nl)
            out['vc_ceil'][s, t] = size_ceiling(a, cls, r, 10)
            if t == T - 1:
                pairs_final[D['sites'][s]] = [[i, j, round(v, 4)] for i, j, v in pr]
    out['pairs_final'] = pairs_final
    mat = np.zeros((T, S, S))
    adj = np.full((T, S - 1, 4), np.nan)
    wl = {w: np.full((T, S - 2 * w + 1, 4), np.nan) for w in (2, 3)}
    for t in range(T):
        for a in range(S):
            for b in range(a, S):
                sc = matched_jaccard(lab(t, a, a), lab(t, b, b), r, r)[0]
                mat[t, a, b] = mat[t, b, a] = sc
        for a in range(S - 1):
            sc, _, nl = matched_jaccard(lab(t, a, a), lab(t, a + 1, a + 1), r, r, perms)
            adj[t, a] = (sc, *null_summary(nl))
        for w in (2, 3):
            for a in range(S - 2 * w + 1):
                sc, _, nl = matched_jaccard(lab(t, a, a + w - 1), lab(t, a + w, a + 2 * w - 1), r, r, perms)
                wl[w][t, a] = (sc, *null_summary(nl))
    out.update(mat=mat, adj=adj, w2=wl[2], w3=wl[3])
    return (m, ri), out


def q3(data_dir, variants, keys):
    """Returns (available, grad steps, results dict keyed (m, ri, s, block) -> arrays, class arrays, align)."""
    T, S, steps, TL, tp = D['T'], D['S'], D['steps'], D['TL'], D['tp']
    tg = [t for t in range(T) if (data_dir / f'gradgram_{steps[t]}.npy').exists()]
    if not tg:
        return False, [], {}, None, None
    blocks = {'all': tp[(0, T - 1)], 'late': tp[(T // 2, T - 1)]}
    res = {}
    cls_w, cls_b = np.full(len(tg), np.nan), np.full(len(tg), np.nan)
    gcos = {}
    for k, t in enumerate(tg):
        G = np.load(data_dir / f'gradgram_{steps[t]}.npy').astype(np.float32)
        w, b = grad_within_between(G, D['classes'], 10)
        cls_w[k], cls_b[k] = w[0], b[0]
        with np.load(data_dir / f'points_{steps[t]}.npz') as p:
            gcos[t] = p['grad_cos'].astype(np.float64) if 'grad_cos' in p.files else None
        for (m, ri) in variants:
            r = D['R'][ri]
            for s in range(S):
                for bn, pi in blocks.items():
                    l = TL[s, m, pi, ri].astype(np.int64)
                    rec = res.setdefault((m, ri, s, bn), dict(
                        w=np.full(len(tg), np.nan), b=np.full(len(tg), np.nan),
                        nm=np.full(len(tg), np.nan), nlo=np.full(len(tg), np.nan), nhi=np.full(len(tg), np.nan),
                        align=np.full((r + 1, len(tg)), np.nan)))
                    w, b = grad_within_between(G, l, r, keys)
                    rec['w'][k], rec['b'][k] = w[0], b[0]
                    d = w[1:] - b[1:]
                    rec['nm'][k], rec['nlo'][k], rec['nhi'][k] = null_summary(d)
                    if gcos[t] is not None:
                        for j in range(r):
                            mem = l == j
                            rec['align'][j, k] = gcos[t][mem].mean() if mem.any() else np.nan
                        rec['align'][r, k] = gcos[t].mean()
        print(f'  q3 checkpoint {steps[t]} done', flush=True)
    return True, tg, res, (cls_w, cls_b), gcos


# ---------------------------------------------------------------- JSON helpers


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not np.isfinite(x) else round(x, 4) + 0.0
    return x


def heat(z, rows, cols, zlabel, rowlabel, collabel, **kw):
    return dict(rows=rows, cols=cols, z=z, zlabel=zlabel, zrange=[0, 1], rowlabel=rowlabel, collabel=collabel, **kw)


def lines(x, series, xlabel, ylabel, xscale='linear', yrange=(0, 1), colors='categorical', **kw):
    return dict(x=x, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yrange=list(yrange), series=series,
                colors=colors, **kw)


def nice_range(vals, pad=0.08, step=0.01):
    v = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in vals])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0, 1]
    lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
    span = max(hi - lo, step)
    return [float(np.floor((lo - pad * span) / step) * step), float(np.ceil((hi + pad * span) / step) * step)]


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', default=None, help='default: site/assets (testing only)')
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data' / args.tag
    out_dir = Path(args.out_dir) if args.out_dir else SITE_DIR / 'assets'
    (out_dir / 'fig').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load(data_dir)
    meta, steps, sites, T, S, R = D['meta'], D['steps'], D['sites'], D['T'], D['S'], D['R']
    modes = meta['modes']
    N = len(D['classes'])
    rng = np.random.default_rng(SEED)
    D['perms'] = np.stack([rng.permutation(N) for _ in range(N_PERM)])
    grad_keys = np.argsort(rng.random((N_PERM_GRAD, N)), axis=1)  # uniform permutations of 0..N-1
    # SPEC 2 'Kernel rules': rule 0 (kNN graph, k = 15) is the default; meta.json may override.
    dm = int(meta['default_mode']) if meta.get('default_mode') is not None else 0
    dr = DEFAULT_R if DEFAULT_R in R else R[len(R) // 2]
    dri = R.index(dr)

    jobs = [(m, ri) for m in range(len(modes)) for ri in range(len(R))]
    print(f'loaded {data_dir} in {time.time() - t0:.1f}s; Q1/Q2 over {len(jobs)} (mode, r) variants', flush=True)
    with mp.get_context('fork').Pool(args.workers) as pool:
        Q12 = dict(pool.map(_q12_job, jobs))
    print(f'Q1/Q2 done at {time.time() - t0:.1f}s', flush=True)

    step_lbl = [f'step {s}' for s in steps]
    q_when, q_carry = {}, {}
    for (m, ri), o in Q12.items():
        key = f'{m}_{R[ri]}'
        H = lambda z, zl: heat(z, sites, steps, zl, 'site', 'checkpoint (step)')
        band = lambda lo, hi, s: [lo[s], hi[s]]
        q_when[key] = dict(
            vs_final=H(o['vf'], 'matched Jaccard vs final-checkpoint sets'),
            vs_final_null=H(o['vf_m'], 'permutation-null mean'),
            vs_class=H(o['vc'], 'matched Jaccard vs CIFAR classes'),
            vs_class_null=H(o['vc_m'], 'permutation-null mean'),
            vs_class_ceiling=H(o['vc_ceil'], 'size ceiling of matched Jaccard vs classes'),
            lines_vs_final=lines(steps, [dict(name=sites[s], y=o['vf'][s], band=band(o['vf_lo'], o['vf_hi'], s))
                                         for s in range(S)], 'training step', 'matched Jaccard vs final sets',
                                 'symlog', colors='ordinal'),
            lines_vs_class=lines(steps, [dict(name=sites[s], y=o['vc'][s], band=band(o['vc_lo'], o['vc_hi'], s))
                                         for s in range(S)], 'training step', 'matched Jaccard vs classes',
                                 'symlog', colors='ordinal'),
            pairs_vs_class_final=o['pairs_final'])
        adj_x = list(range(S - 1))
        adj_tl = [f'{sites[a]}→{sites[a + 1]}' for a in adj_x]
        wobj = {}
        for w in (2, 3):
            xs = list(range(S - 2 * w + 1))
            tl = [f'{sites[a]}–{sites[a + w - 1]}→{sites[a + w]}–{sites[a + 2 * w - 1]}' for a in xs]
            arr = o[f'w{w}']
            wobj[f'w{w}'] = lines(xs, [dict(name=step_lbl[t], y=arr[t, :, 0], band=[arr[t, :, 2], arr[t, :, 3]])
                                       for t in range(T)], f'first block (width {w})',
                                  'matched Jaccard, block vs next block', colors='ordinal', xticklabels=tl)
        q_carry[key] = dict(
            matrix=heat(o['mat'], sites, sites, 'matched Jaccard between single-site sets', 'site', 'site',
                        slider=dict(label='checkpoint (step)', values=steps)),
            adjacent=lines(adj_x, [dict(name=step_lbl[t], y=o['adj'][t, :, 0]) for t in range(T)],
                           'adjacent site pair', 'matched Jaccard, site vs next site', colors='ordinal',
                           xticklabels=adj_tl),
            adjacent_null=lines(adj_x, [dict(name=step_lbl[t], y=o['adj'][t, :, 1],
                                             band=[o['adj'][t, :, 2], o['adj'][t, :, 3]]) for t in range(T)],
                                'adjacent site pair', 'permutation null of matched Jaccard', colors='ordinal',
                                xticklabels=adj_tl),
            **wobj)

    # Q3
    variants = [(m, ri) for m in range(len(modes)) for ri in range(len(R))]
    avail, tg, G3, clsWB, gcos = q3(data_dir, variants, grad_keys)
    print(f'Q3 done at {time.time() - t0:.1f}s (available={avail})', flush=True)
    q_grad = dict(available=avail)
    if avail:
        gsteps = [steps[t] for t in tg]
        cw, cb = clsWB
        diffs, aligns = [cw - cb], []
        for rec in G3.values():
            diffs += [rec['w'] - rec['b'], rec['nlo'], rec['nhi']]
            aligns.append(rec['align'])
        yd, ya = nice_range(diffs), nice_range(aligns)
        q_grad.update(yrange_diff=yd, yrange_align=ya, steps=gsteps)
        for (m, ri, s, bn), rec in G3.items():
            r = R[ri]
            q_grad[f'{m}_{r}_{s}_{bn}'] = dict(
                lines=lines(gsteps, [dict(name='SEBA sets', y=rec['w'] - rec['b']),
                                     dict(name='CIFAR classes', y=cw - cb),
                                     dict(name='random sets (null)', y=rec['nm'], band=[rec['nlo'], rec['nhi']],
                                          dash=True)],
                            'training step', 'gradient cosine: within − between', 'symlog', yd,
                            refline=dict(y=0, label='no difference'),
                            detail=dict(sets_within=rec['w'], sets_between=rec['b'], classes_within=cw,
                                        classes_between=cb)),
                setalign=lines(gsteps, [dict(name=f'set {j + 1}', y=rec['align'][j]) for j in range(r)]
                               + [dict(name='all points', y=rec['align'][r], dash=True)],
                               'training step', 'mean gradient alignment (cosine with mean gradient)',
                               'symlog', ya))

    # Q4
    dec = json.loads((data_dir / 'decod.json').read_text())
    dsteps = [res['step'] for res in dec['results']]
    dsites = dec.get('sites', sites)

    def acc(res, view, site, metric):
        v = res['views'].get(view, {}).get(site)
        if v is None and view == 'gap':
            v = res['views'].get('flat', {}).get(site)
        return None if v is None else v.get(metric)

    q_decod = {}
    gap_note = ('pool and logits have no spatial axis, so they have no gap view; the gap chart shows their flat '
                'value at those sites')
    for metric, mname in (('ridge_acc', 'ridge'), ('knn_acc', 'knn')):
        ylab = ('ridge-probe' if mname == 'ridge' else 'cosine kNN') + ' test accuracy'
        for view in ('flat', 'gap'):
            kw = dict(refline=dict(y=0.1, label='chance'))
            if view == 'gap':
                kw['note'] = gap_note
            q_decod[f'{mname}_by_site_{view}'] = lines(
                list(range(len(dsites))),
                [dict(name=f'step {res["step"]}', y=[acc(res, view, s, metric) for s in dsites])
                 for res in dec['results']], 'site', ylab, colors='ordinal', xticklabels=dsites, **kw)
            q_decod[f'{mname}_by_step_{view}'] = lines(
                dsteps, [dict(name=s, y=[acc(res, view, s, metric) for res in dec['results']]) for s in dsites],
                'training step', ylab, 'symlog', colors='ordinal', **kw)

    # summary (default mode and r)
    o = Q12[(dm, dri)]
    when = {}
    for s in range(S):
        ok = (o['vf'][s] > 0.5) & (o['vf'][s] > o['vf_hi'][s])
        first = next((steps[t] for t in range(T) if ok[t]), None)
        settled = next((steps[t] for t in range(T) if ok[t:].all()), None)
        when[sites[s]] = dict(first_step=first, settled_step=settled, vs_class_final=o['vc'][s, -1],
                              vs_class_null_hi_final=o['vc_hi'][s, -1], vs_class_ceiling_final=o['vc_ceil'][s, -1])
    summary = dict(mode=dm, mode_name=modes[dm], r=dr, when=when,
                   carry=dict(steps=steps, mean_adjacent=o['adj'][:, :, 0].mean(1),
                              mean_adjacent_null=o['adj'][:, :, 1].mean(1)))
    if avail:
        k = len(tg) - 1
        summary['grad'] = dict(step=steps[tg[k]], block='all', per_site={
            sites[s]: dict(sets_diff=G3[(dm, dri, s, 'all')]['w'][k] - G3[(dm, dri, s, 'all')]['b'][k],
                           classes_diff=clsWB[0][k] - clsWB[1][k],
                           null_mean=G3[(dm, dri, s, 'all')]['nm'][k], null_hi=G3[(dm, dri, s, 'all')]['nhi'][k])
            for s in range(S)})

    out = dict(
        meta=dict(tag=args.tag, fixture=bool(meta.get('fixture', False)), steps=steps, sites=sites, modes=modes,
                  seba_r=R, default_mode=dm, default_r=dr, n_perm=N_PERM, n_perm_grad=N_PERM_GRAD, seed=SEED,
                  generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                  kept={str(m): list(R) for m in range(len(modes))},
                  definitions='See the docstring of site/analysis.py: matched Jaccard = mean J over Hungarian-'
                              'matched set pairs; nulls permute the first labeling (sizes kept); gradient '
                              'within/between = pair-weighted mean cosine of per-sample loss gradients.'),
        q_when=q_when, q_carry=q_carry, q_grad=q_grad, q_decod=q_decod, summary=summary)
    out = clean(out)
    text = json.dumps(out, separators=(',', ':'), allow_nan=False, ensure_ascii=False)
    if len(text) > 3_000_000:  # keep all r for the default mode, r in {3, 4, 6} for the others
        keep = {str(m): (list(R) if m == dm else [r for r in R if r in (3, 4, 6)]) for m in range(len(modes))}
        ok = lambda k: int(k.split('_')[1]) in keep[k.split('_')[0]]
        for sec in ('q_when', 'q_carry'):
            out[sec] = {k: v for k, v in out[sec].items() if ok(k)}
        out['q_grad'] = {k: v for k, v in out['q_grad'].items() if '_' not in k or k.startswith('yrange')
                         or ok(k)}
        out['meta']['kept'] = keep
        text = json.dumps(out, separators=(',', ':'), allow_nan=False, ensure_ascii=False)
    (out_dir / 'questions.json').write_text(text)
    print(f'wrote {out_dir / "questions.json"} ({len(text) / 1e6:.2f} MB) at {time.time() - t0:.1f}s', flush=True)

    figures(out, out_dir / 'fig', dm, dr)
    print(f'figures done at {time.time() - t0:.1f}s', flush=True)
    print(json.dumps(out['summary'], indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- figures


def figures(out, fig_dir, dm, dr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    pal = json.loads((SITE_DIR / 'palette.json').read_text())
    fg, mute, line, bg = pal['fg'], pal['mute'], pal['line'], pal['surface']
    plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': mute, 'axes.labelcolor': fg, 'text.color': fg,
                         'xtick.color': mute, 'ytick.color': mute, 'xtick.labelcolor': fg, 'ytick.labelcolor': fg,
                         'axes.titlesize': 9.5, 'axes.titlecolor': fg, 'legend.fontsize': 7, 'legend.frameon': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': bg,
                         'axes.facecolor': bg, 'savefig.facecolor': bg, 'axes.grid': True, 'grid.color': line,
                         'grid.linewidth': 0.6, 'lines.linewidth': 1.4})
    seq = LinearSegmentedColormap.from_list('seq', ['#ffffff'] + pal['sequential'])
    ordc = LinearSegmentedColormap.from_list('ord', pal['ordinal'])
    ordinal = lambda n: [ordc(i / max(n - 1, 1)) for i in range(n)]
    meta = out['meta']
    steps, sites = meta['steps'], meta['sites']
    key = f'{dm}_{dr}'
    sub = f'kernel rule: {meta["modes"][dm]}, r = {dr}' + ('   [SYNTHETIC FIXTURE]' if meta['fixture'] else '')
    DPI = 200

    def symlog(ax, xs):
        ax.set_xscale('symlog', linthresh=max(min([x for x in xs if x > 0] or [1]), 1))
        ticks = [x for x in xs]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{x // 1000}k' if x >= 1000 else str(x) for x in ticks], rotation=45, fontsize=7)
        ax.minorticks_off()
        ax.grid(axis='x', visible=False)

    def heatmap(ax, z, rows, cols, title, fig, clabel, annot=False):
        z = np.array([[np.nan if v is None else v for v in r] for r in z])
        im = ax.imshow(z, cmap=seq, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=7.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc='left')
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(clabel, color=fg)
        cb.outline.set_visible(False)
        if annot:
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if np.isfinite(z[i, j]):
                        ax.text(j, i, f'{z[i, j]:.2f}'[1:] if z[i, j] < 1 else '1', ha='center', va='center',
                                fontsize=5.5, color=bg if z[i, j] > 0.55 else fg)

    # q_when
    w = out['q_when'][key]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    stl = [f'{x // 1000}k' if x >= 1000 else str(x) for x in steps]
    heatmap(axs[0], w['vs_final']['z'], sites, stl, 'Sets at each step vs final sets', fig, 'matched Jaccard',
            annot=True)
    axs[0].set_xlabel('training step')
    ax = axs[1]
    lv = w['lines_vs_class']
    cols = ordinal(len(sites))
    lo = np.nanmin(np.array([s['band'][0] for s in lv['series']], dtype=float), 0)
    hi = np.nanmax(np.array([s['band'][1] for s in lv['series']], dtype=float), 0)
    ax.fill_between(steps, lo, hi, color=pal['unassigned'], alpha=0.7, lw=0, label='null 95% (all sites)')
    ceil = np.nanmean(np.array(w['vs_class_ceiling']['z'], dtype=float), 0)
    ax.plot(steps, ceil, color=mute, ls='--', lw=1, label='size ceiling (mean)')
    for s, c in zip(lv['series'], cols):
        ax.plot(steps, np.array(s['y'], dtype=float), color=c, label=s['name'])
    symlog(ax, steps)
    ax.set_ylim(0, 1)
    ax.set_xlabel('training step')
    ax.set_ylabel('matched Jaccard')
    ax.set_title('Sets vs CIFAR classes (validation)', loc='left')
    ax.legend(ncol=2, loc='upper left', bbox_to_anchor=(1.0, 1.02), handlelength=1.2, columnspacing=0.8)
    fig.suptitle(sub, x=0.01, ha='left', fontsize=7.5, color=mute)
    fig.tight_layout()
    fig.savefig(fig_dir / 'q_when.png', dpi=DPI)
    plt.close(fig)

    # q_carry
    c = out['q_carry'][key]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    heatmap(axs[0], c['matrix']['z'][-1], sites, sites, f'Site vs site sets, step {steps[-1]}', fig,
            'matched Jaccard')
    ax = axs[1]
    adj, adjn = c['adjacent'], c['adjacent_null']
    xs = adj['x']
    nb = adjn['series'][-1]['band']
    ax.fill_between(xs, np.array(nb[0], dtype=float), np.array(nb[1], dtype=float), color=pal['unassigned'],
                    alpha=0.7, lw=0, label=f'null 95% (step {steps[-1]})')
    for s, col in zip(adj['series'], ordinal(len(adj['series']))):
        ax.plot(xs, np.array(s['y'], dtype=float), color=col, marker='o', ms=2.5, label=s['name'].replace('step ', ''))
    ax.set_xticks(xs)
    ax.set_xticklabels(adj['xticklabels'], rotation=55, ha='right', fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel('matched Jaccard')
    ax.set_title('Each site vs the next, per checkpoint', loc='left')
    ax.legend(title='step', title_fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.02), handlelength=1.2)
    fig.suptitle(sub, x=0.01, ha='left', fontsize=7.5, color=mute)
    fig.tight_layout()
    fig.savefig(fig_dir / 'q_carry.png', dpi=DPI)
    plt.close(fig)

    # q_grad
    g = out['q_grad']
    if g.get('available'):
        gs = g['steps']
        nr = int(np.ceil(len(sites) / 4))
        fig, axs = plt.subplots(nr, 4, figsize=(7, 1.75 * nr + 0.6), sharex=True, sharey=True)
        for s, ax in enumerate(axs.ravel()):
            if s >= len(sites):
                ax.axis('off')
                continue
            ser = g[f'{dm}_{dr}_{s}_all']['lines']['series']
            nl = ser[2]
            ax.fill_between(gs, np.array(nl['band'][0], dtype=float), np.array(nl['band'][1], dtype=float),
                            color=pal['unassigned'], alpha=0.8, lw=0, label='random sets, 95%')
            ax.axhline(0, color=mute, lw=0.6)
            ax.plot(gs, np.array(ser[0]['y'], dtype=float), color=pal['sets'][0], marker='o', ms=2, label='SEBA sets')
            ax.plot(gs, np.array(ser[1]['y'], dtype=float), color=pal['sets'][1], ls='--', label='CIFAR classes')
            ax.set_title(sites[s], loc='left', fontsize=8.5)
            ax.set_ylim(*g['yrange_diff'])
            symlog(ax, gs)
            ax.tick_params(labelsize=6.5)
            ax.set_xticklabels([f'{x // 1000}k' if x >= 1000 else str(x) for x in gs], rotation=45, fontsize=6)
        for ax in axs[:, 0]:
            ax.set_ylabel('within − between', fontsize=7.5)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5)
        fig.suptitle('Gradient cosine within minus between sets (time block over all checkpoints)\n' + sub,
                     x=0.01, ha='left', fontsize=8, color=fg)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(fig_dir / 'q_grad.png', dpi=DPI)
        plt.close(fig)

    # q_decod
    d = out['q_decod']
    fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)
    for i, mname in enumerate(('ridge', 'knn')):
        for j, view in enumerate(('flat', 'gap')):
            ax = axs[i, j]
            obj = d[f'{mname}_by_site_{view}']
            cols = ordinal(len(obj['series']))
            for s, col in zip(obj['series'], cols):
                y = np.array([np.nan if v is None else v for v in s['y']], dtype=float)
                ax.plot(obj['x'], y, color=col, marker='o', ms=2, lw=1.1, label=s['name'].replace('step ', ''))
            if view == 'gap':
                n = len(obj['x'])
                ax.axvspan(n - 2.5, n - 0.5, color=line, alpha=0.6, lw=0)
                ax.text(n - 1.5, 0.03, 'flat\nvalue', ha='center', va='bottom', fontsize=6, color=mute)
            ax.axhline(0.1, color=mute, lw=0.7, ls=':')
            ax.set_ylim(0, 1)
            ax.set_title(f'{"ridge probe" if mname == "ridge" else "cosine kNN"}, '
                         f'{"flattened map" if view == "flat" else "channel means (GAP)"}', loc='left')
            ax.set_xticks(obj['x'])
            ax.set_xticklabels(obj['xticklabels'], rotation=45, fontsize=7)
            if j == 0:
                ax.set_ylabel('test accuracy')
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, title='checkpoint (step)', title_fontsize=7, loc='center right', fontsize=6.5,
               handlelength=1.2)
    fig.suptitle('Linear and kNN decodability of class by site (validation fits; dotted = chance)'
                 + ('   [SYNTHETIC FIXTURE]' if meta['fixture'] else ''), x=0.01, ha='left', fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    fig.savefig(fig_dir / 'q_decod.png', dpi=DPI)
    plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
