"""Planted / null benchmark runner (PLAN.md section 8). Descriptive output only.

One *unit* = (benchmark, level, seed); it runs every N in `Ns`. For each N the generator's cells go through the
production preprocessing (coherent.centred_cosine_gram), the default operators (sources.default_ops), every colour
source in sources.SOURCES, the single-cell pointers C1 (first / middle / last cell via M0), the block-combination
variants of variants.py, and (P8, P2) the paired transfer estimators of transfer.py when present. Each output is
scored against the generator's truth. Rows are tidy dicts
    {bench, level, seed, N, block, r, method, stat, value}
and `summarise(rows)` gives medians and 10-90% bands over seeds.

CLI:
    python bench.py smoke                 # local smoke run (N=300, seeds 0-1, threads <= 2)
    python bench.py unit P1 0.5 0 --N 500 # one unit to stdout-summary + out/bench/parts/
    python bench.py summarise DIR         # merge parts in DIR into rows.jsonl.gz and summary.json
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_CC = HERE.parent / 'cifar_coherent'
for p in (str(HERE), str(_CC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import planted  # noqa: E402
import stats  # noqa: E402
from coherent import centred_cosine_gram, symmetric_balance, block_modes, seba, seba_labels  # noqa: E402

OUT = HERE / 'out' / 'bench'


# ------------------------------------------------------------------------------------------------ backends


def load_backend():
    """Builder A's sources.py. Returns (SOURCES, default_ops, name)."""
    import sources
    return sources.SOURCES, sources.default_ops, 'sources.py'


# PLAN 5 E block-combination variants from builder A's variants.combine: name -> (how, side, needs_grams)
VARIANT_SPECS = {
    'V_meanM2': ('meansq', None, False), 'V_product': ('product', 'right', False),
    'V_product_left': ('product', 'left', False), 'V_two_end': ('twoend', 'right', False),
    'V_expprod': ('expprod', 'right', False), 'V_concat_graph': ('concat', None, True),
    'V_union_graph': ('union', None, True), 'V_intersection_graph': ('intersection', None, True),
}


def load_variants():
    import variants
    return variants, 'variants.py'


def load_transfer():
    try:
        import transfer  # noqa: F401
    except ImportError:
        return None
    return transfer


def sub_ops(ops, idx):
    """Restrict default_ops output to the cells idx (dict of per-cell lists, or a list)."""
    if ops is None:
        return None
    if isinstance(ops, dict):
        n = max((len(v) for v in ops.values() if isinstance(v, (list, tuple))), default=0)
        return {k: ([v[i] for i in idx] if isinstance(v, (list, tuple)) and len(v) == n else v)
                for k, v in ops.items()}
    if isinstance(ops, (list, tuple)):
        return [ops[i] for i in idx]
    raise TypeError(type(ops))


def cell_P(ops):
    """Per-cell zero-diagonal P from ops (dict with 'P' or 'M', or list of M)."""
    if isinstance(ops, dict):
        if 'P' in ops:
            return list(ops['P'])
        return [2 * m - torch.eye(len(m), dtype=m.dtype) for m in ops['M']]
    return [2 * m - torch.eye(len(m), dtype=m.dtype) for m in ops]


# ------------------------------------------------------------------------------------------------ local operator-level methods
# Used for P11 (graphs, no activations) and as the local block-combination variants when variants.py is absent.


def _lazy(Ps):
    out = []
    for p in Ps:
        m = p * .5
        m.diagonal().add_(.5)
        out.append(m)
    return out


def op_textbook(As, r):
    """G2 at the operator level: per-cell D^-1/2 A D^-1/2 (A with floor), mean, row-normalised k-means."""
    from sklearn.cluster import KMeans
    Ns = []
    for a in As:
        d = a.sum(1).rsqrt()
        Ns.append(d[:, None] * a * d[None, :])
    nb = torch.stack(Ns).mean(0)
    lam, u = torch.linalg.eigh(nb)
    u = u.flip(1)[:, :r].numpy()
    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    lab = KMeans(r, n_init=20, random_state=0).fit_predict(un).astype(np.int64)
    return dict(labels=lab, basis=u[:, 1:], values=lam.flip(0)[:9].numpy(), status='S+B', info={})


# ------------------------------------------------------------------------------------------------ scoring


def _labels(out):
    lab = out['labels']
    return np.asarray(lab.numpy() if hasattr(lab, 'numpy') else lab).astype(np.int64)


def _basis(out):
    b = out.get('basis')
    if b is None:
        return None
    b = np.asarray(b.numpy() if hasattr(b, 'numpy') else b, dtype=np.float64)
    return b if b.ndim == 2 and b.shape[1] > 0 else None


def onehot(y):
    y = np.asarray(y)
    k = int(y.max()) + 1
    E = np.zeros((len(y), k))
    E[np.arange(len(y)), y] = 1
    return E


def set_stats(lab):
    n = len(lab)
    a = lab[lab >= 0]
    sizes = np.bincount(a) if len(a) else np.zeros(0, int)
    sizes = sizes[sizes > 0]
    p = sizes / max(sizes.sum(), 1)
    return dict(unassigned=float((lab < 0).mean()), n_sets=float(len(sizes)),
                min_set_share=float(sizes.min() / n) if len(sizes) else 0.0,
                eff_sets=float(np.exp(-(p * np.log(p)).sum())) if len(sizes) else 0.0)


def retention_stats(Ps, lab, per_cell=False):
    out = {}
    if not (lab >= 0).any():
        return out
    ex = []
    for P in Ps:
        r = stats.retention(P, lab)
        v = r['values'] - r['null']
        ex.append(float(np.nanmean(v)))
    out['ret_excess_mean'] = float(np.mean(ex))
    out['ret_excess_min'] = float(np.min(ex))
    if per_cell:
        out.update({f'ret_excess_cell{j}': e for j, e in enumerate(ex)})
    return out


def truth_groups(truth):
    """Declared number of planted groups of truth['labels'] (generator parameter, not read off the labels)."""
    return int(truth['n_groups'])


def score(out, truth, Ps, block_cells, rank):
    lab = _labels(out)
    U = _basis(out)
    res = set_stats(lab)
    tid = truth['id']
    res.update(retention_stats(Ps, lab, per_cell=tid in ('P3', 'P3L', 'P4', 'P9', 'P11')))
    if U is not None:
        lev = stats.leverage(U)
        res['span_effective_points'] = lev['effective_points']
    y = truth.get('labels')
    if y is not None:
        # declared set counts: the method's rank r (k_c = r sets, empty ones included) and the planted group count
        ag = stats.agreement(lab, y, ka=rank, kb=truth_groups(truth))
        res.update(ami=ag['ami_all'], ari=ag['ari_all'], ami_joint=ag['ami_joint'], mj_max=ag['mj_max'])
        if U is not None:
            so = stats.subspace_overlap(U, onehot(y))
            res.update(truth_span_captured=so['captured_v'], truth_span_mean_cos2=so['mean_cos2'],
                       truth_span_chance=so['chance'])
    if tid in ('P3', 'P3L'):
        res['ami_nuisance_max'] = max(stats.ami(lab, h) for h in (truth['cell_labels'][i] for i in block_cells))
    if tid == 'P10':
        res['ami_top'] = stats.ami(lab, truth['top_labels'])
        if U is not None:
            res['top_span_captured'] = stats.subspace_overlap(U, onehot(truth['top_labels']))['captured_v']
    if tid == 'P9':
        for nm, c in zip('uvw', truth['cell_labels']):
            res[f'ami_{nm}'] = stats.ami(lab, c)
    if tid == 'P12':
        for nm, c in truth['extra_labels'].items():
            res[f'ami_{nm}'] = stats.ami(lab, c)
    if tid == 'P7':
        y = truth['labels']
        k = int(y.max()) + 1
        for g in range(k):
            A = y == g
            best = 0.0
            for s in np.unique(lab[lab >= 0]):
                B = lab == s
                best = max(best, float((A & B).sum() / (A | B).sum()))
            res[f'best_jaccard_group{g}'] = best
        dup = np.unique(np.concatenate([d.ravel() for d in truth['duplicates']]))
        if len(dup) and U is not None:
            h = stats.leverage(U)['h']
            res['dup_leverage_ratio'] = float(h[dup].mean() / h.mean())
    if truth.get('contrasts') is not None and U is not None:
        so = stats.subspace_overlap(U, truth['contrasts'])
        res.update(contrast_captured=so['captured_v'], contrast_mean_cos2=so['mean_cos2'], contrast_chance=so['chance'])
    for nm, S in (truth.get('spans') or {}).items():
        if U is not None:
            res[f'span_{nm}_captured'] = stats.subspace_overlap(U, S)['captured_v']
    if tid == 'P6' and U is not None:
        dens = truth['density']
        if dens.std() > 0:
            Uc = U - U.mean(0)
            cor = (Uc.T @ (dens - dens.mean())) / (np.linalg.norm(Uc, axis=0) * np.linalg.norm(dens - dens.mean()) + 1e-300)
            res['density_corr_max'] = float(np.abs(cor).max())
        sizes = np.array([np.mean(dens[lab == s]) for s in np.unique(lab[lab >= 0])])
        if len(sizes) > 1:
            res['set_mean_density_spread'] = float(sizes.max() / sizes.min())
    if truth.get('pseudo_labels') is not None:
        res['ami_pseudo'] = stats.ami(lab, truth['pseudo_labels'])
    return res


# ------------------------------------------------------------------------------------------------ unit


NO_TRUTH = ('P0', 'P0d', 'P2', 'P6', 'P8')
TRANSFER_ESTIMATORS = ('linear', 'nystrom', 'diffusion', 'soft_ulam')
TRANSFER_M = 64
TRANSFER_PERMS = 50
# variants skipped at large N for cost: the union-graph Sinkhorn took ~1150 s at N = 2000 (vs ~2 s at N = 1000)
SKIP_AT_N = {2000: ('V_union_graph',)}



def _block_slices(truth, L):
    return {nm: list(range(a, b + 1)) for nm, (a, b) in truth['blocks'].items() if b < L}


# sources whose `values` start with the constant / leading mode (gap index r-1 -> r); the others list
# non-constant eigenvalues only (gap index r-2 -> r-1)
VALUES_WITH_LEADING = {'M0', 'G1', 'G1n', 'G2', 'C1first', 'C1mid', 'C1last', 'G2op'}
OPERATOR_ONLY = ('M0', 'G1', 'G1n', 'C1first', 'C1mid', 'C1last', 'C2', 'C3')


def _run_activation_methods(grams, ops, sources, variants, r, seed, idx, cache, operator_level=False, skip=()):
    """Yield (method, output, seconds) for the block of cells idx. `cache` is shared across ranks and blocks
    (sources.py contract), keyed by cell index."""
    g = [grams[i] for i in idx]
    o = sub_ops(ops, idx)
    for name, fn in sources.items():
        if len(idx) == 1 and name.startswith('C') and name != 'C3' and name != 'C2':
            continue  # C1 pointers equal M0 on a one-cell block
        if operator_level and name not in OPERATOR_ONLY:
            continue
        t = time.time()
        try:
            out = fn(g, r, seed=seed, ops=o, cache=cache, cell_keys=tuple(idx))
        except Exception as e:  # recorded, never silently dropped
            yield name, dict(error=repr(e)), time.time() - t
            continue
        yield name, out, time.time() - t
    if len(idx) > 1:
        import sources as S
        for name, (how, side, needs_grams) in VARIANT_SPECS.items():
            if name in skip:
                continue
            if operator_level and needs_grams:
                continue
            t = time.time()
            try:
                key = ('variant', how, tuple(idx))
                if key not in cache:
                    cache[key] = variants.combine(o, how, grams=g if needs_grams else None)
                res = cache[key]
                vecs = res['vecs'] if res['kind'] == 'eig' else res[side]
                U = vecs[:, :r - 1]
                lab, _, _, info = S.seba_sets(S.with_constant(U), r)
                out = dict(labels=lab, basis=U, values=res['vals'], status='S+P', info=info)
            except Exception as e:
                yield name, dict(error=repr(e)), time.time() - t
                continue
            yield name, out, time.time() - t


def _sbm_ops(adj):
    Ps, As = [], []
    for A in adj:
        a = torch.from_numpy(A.astype(np.float64)) + 1e-6
        a.diagonal().fill_(0)
        Ps.append(symmetric_balance(torch.log(a)))
        As.append(a)
    return Ps, As


def _gap(name, vals, r):
    """Gap bordering the method's q = r-1 dimensional span, in the source's own value convention."""
    off = 0 if name in VALUES_WITH_LEADING else 1
    i = r - 1 - off
    if vals is None or i + 1 >= len(vals) or i < 0:
        return None
    return float(vals[i] - vals[i + 1])


def run_unit(bench, level, seed, Ns=(500, 1000, 2000), stability=True, verbose=False, transfer_perms=TRANSFER_PERMS):
    sources, default_ops, backend = load_backend()
    variants, vbackend = load_variants()
    make, _ = planted.BENCHMARKS[bench]
    rows = []
    meta = dict(bench=bench, level=level, seed=seed, backend=backend, variants=vbackend,
                methods=sorted(sources) + sorted(VARIANT_SPECS))

    def emit(N, block, r, method, d):
        for k, v in d.items():
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            rows.append(dict(bench=bench, level=level, seed=seed, N=N, block=block, r=r, method=method,
                             stat=k, value=float(v)))

    for N in Ns:
        t0 = time.time()
        Xs, truth = make(seed, N=N, level=level)
        emit(N, 'all', None, 'oracle', planted.oracle(Xs, truth))
        extra = {}
        if Xs is None:  # operator-level fixture (P11): graphs given, operator-only sources + operator-level G2
            Ps, As = _sbm_ops(truth['adjacency'])
            ops = _lazy(Ps)
            grams = ops  # placeholders: operator-only sources read only shapes from grams
            L = len(ops)
            operator_level = True
        else:
            grams = [centred_cosine_gram(torch.from_numpy(X)) for X in Xs]
            ops = default_ops(grams)
            L = len(Xs)
            operator_level = False
        emit(N, 'all', None, 'pipeline', dict(seconds_ops=time.time() - t0))
        cache = {}
        for bname, idx in _block_slices(truth, L).items():
            Ps = cell_P(sub_ops(ops, idx))
            labs = {}
            for r in truth['ranks']:
                outs = list(_run_activation_methods(grams, ops, sources, variants, r, seed, idx, cache, operator_level,
                                                   skip=SKIP_AT_N.get(N, ())))
                if operator_level:
                    t = time.time()
                    outs.append(('G2op', op_textbook([As[i] for i in idx], r), time.time() - t))
                for name, out, sec in outs:
                    if 'error' in out:
                        emit(N, bname, r, name, dict(error=1.0))
                        if verbose:
                            print('ERROR', bench, level, N, bname, r, name, out['error'], flush=True)
                        continue
                    sc = score(out, truth, Ps, idx, r)
                    sc['seconds'] = sec
                    if out.get('info', {}).get('seba_error'):
                        sc['seba_error'] = 1.0
                    vals = out.get('values')
                    if vals is not None:
                        vals = np.asarray(vals, dtype=np.float64).ravel()
                        sc.update({f'val{j}': float(v) for j, v in enumerate(vals[:9])})
                        sc['gap_after_span'] = _gap(name, vals, r)
                    labs[(name, r)] = _labels(out)
                    emit(N, bname, r, name, sc)
            ranks = sorted(truth['ranks'])
            for (name, r), lab in labs.items():
                for r2 in ranks:
                    if r2 > r and (name, r2) in labs:
                        emit(N, bname, r, name, {f'nest_H_given_r{r2}': stats.conditional_entropy(lab, labs[(name, r2)])})
            if stability and bench in NO_TRUTH and bname == 'all' and not operator_level:
                rng = np.random.default_rng([seed, 77])
                r = truth['ranks'][0]
                for d in range(2):
                    keep = np.sort(rng.choice(N, int(.8 * N), replace=False))
                    g2 = [centred_cosine_gram(torch.from_numpy(X[keep])) for X in Xs]
                    o2 = default_ops(g2)
                    for name, out, _ in _run_activation_methods(g2, o2, sources, variants, r, seed, idx, {},
                                                                skip=SKIP_AT_N.get(N, ())):
                        if 'error' in out or (name, r) not in labs:
                            continue
                        emit(N, bname, r, name, {f'stab_ami_{d}': stats.ami(_labels(out), labs[(name, r)][keep])})
        tr = load_transfer()
        if tr is not None and bench in ('P8', 'P2') and not operator_level:
            emit_transfer(tr, Xs, truth, seed, lambda m, d: emit(N, 'all', None, m, d), n_perm=transfer_perms)
        if verbose:
            print(f'{bench} level={level} seed={seed} N={N} done in {time.time() - t0:.1f}s', flush=True)
    return dict(meta=meta, rows=rows)




def emit_transfer(tr, Xs, truth, seed, emit, n_perm=TRANSFER_PERMS):
    """Adapter for builder C's transfer.py: source = cell 0, destination = cell 1 (P8: theta -> phi; P2: the first
    two cells). Split 50 / 25 / 25 % fit / tune / test (one split per seed); m = 64; label-free tuning by tuning-split
    VAMP-2 (transfer.tune); independence control = pairing permuted within each split with the whole tuning repeated
    (transfer.permutation_control), n_perm draws. Scored: training singular values, held-out correlations, test
    VAMP-2, the control's 95% point, and the source functions' overlap with the planted spans."""
    N = len(Xs[0])
    rng = np.random.default_rng([int(seed), 99])
    idx = rng.permutation(N)
    fit, tune_idx, test_idx = np.sort(idx[: N // 2]), np.sort(idx[N // 2: 3 * N // 4]), np.sort(idx[3 * N // 4:])
    bank = tr.FeatureBank({'a': tr.View.from_X(Xs[0], fit, 'a'), 'b': tr.View.from_X(Xs[1], fit, 'b')}, seed)
    pair = bank.pair('a', 'b')
    theory = []
    for v in truth.get('singular_values') or []:
        theory += [v, v]
    for est in TRANSFER_ESTIMATORS:
        name = f'T_{est}'
        t = time.time()
        try:
            model, _ = tr.tune(pair, est, TRANSFER_M, tune_idx)
            if model is None:
                emit(name, dict(error=1.0))
                continue
            ev = tr.evaluate(model, tune_idx, test_idx)
            d = {f'train_sv{j}': v for j, v in enumerate(ev['train_sv'])}
            d.update({f'test_corr{j}': v for j, v in enumerate(ev['test_corr'])})
            d['vamp2_test'] = ev['vamp2_test']
            for j, v in enumerate(theory[:len(ev['test_corr'])]):
                d[f'theory_sv{j}'] = v
            F = model.source_functions(np.arange(N))
            G = model.dest_functions(np.arange(N))
            for nm, S in (truth.get('spans') or {}).items():
                if nm.startswith('theta'):
                    d[f'src2_span_{nm}_captured'] = stats.subspace_overlap(F[:, :2], S)['captured_v']
                else:
                    d[f'dst2_span_{nm}_captured'] = stats.subspace_overlap(G[:, :2], S)['captured_v']
            if truth['id'] == 'P2':
                h = truth['contrasts'].shape[1]
                d['src_span_u_captured'] = stats.subspace_overlap(F[:, :h], truth['contrasts'])['captured_v']
            if n_perm:
                pc = tr.permutation_control(pair, est, TRANSFER_M, tune_idx, test_idx, n_perm=n_perm, seed=seed)
                d.update({f'perm_q95_corr{j}': float(v) for j, v in enumerate(pc['q95'])})
                d['perm_q95_vamp2'] = float(np.nanquantile(pc['vamp2_test'], .95))
            d['seconds'] = time.time() - t
            emit(name, d)
        except Exception as e:
            emit(name, dict(error=1.0))
            print('TRANSFER ERROR', truth['id'], est, repr(e), flush=True)


# ------------------------------------------------------------------------------------------------ summary


def summarise(rows):
    groups = {}
    for r in rows:
        key = (r['bench'], str(r['level']), r['N'], r['block'], r['r'], r['method'], r['stat'])
        groups.setdefault(key, []).append(r['value'])
    out = []
    for (b, lv, N, blk, rk, m, st), v in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        a = np.asarray(v)
        out.append(dict(bench=b, level=lv, N=N, block=blk, r=rk, method=m, stat=st, n=len(a),
                        median=float(np.median(a)), q10=float(np.quantile(a, .1)), q90=float(np.quantile(a, .9))))
    return out


def merge_parts(parts_dir, out_dir=OUT):
    parts_dir = Path(parts_dir)
    rows, metas = [], []
    for p in sorted(parts_dir.rglob('*.json')):
        if p.name in ('MANIFEST.json', 'summary.json'):
            continue
        d = json.loads(p.read_text())
        if 'rows' not in d:
            continue
        rows.extend(d['rows'])
        metas.append(d['meta'])
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / 'rows.jsonl.gz', 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    summ = summarise(rows)
    backends = sorted({m['backend'] for m in metas})
    (out_dir / 'summary.json').write_text(json.dumps(dict(
        description='Planted/null benchmark summary: median and 10/90% quantiles over seeds per '
                    '(bench, level, N, block, r, method, stat). Descriptive; see NOTES_B.md for definitions.',
        backends=backends, variants=sorted({m['variants'] for m in metas}), n_units=len(metas),
        rows=summ)))
    return len(rows), len(summ)


def main(argv):
    if argv and argv[0] == 'smoke':
        torch.set_num_threads(2)
        parts = OUT / 'smoke'
        parts.mkdir(parents=True, exist_ok=True)
        benches = argv[1:] or list(planted.BENCHMARKS)
        for b in benches:
            levels = planted.BENCHMARKS[b][1]
            for lv in (levels[0], levels[-1]):
                for seed in (0, 1):
                    if b == 'P12' and lv != 0:
                        lv = 250
                    t = time.time()
                    res = run_unit(b, lv, seed, Ns=(300,), verbose=False)
                    res['meta']['seconds'] = time.time() - t
                    (parts / f'{b}_{lv}_{seed}.json').write_text(json.dumps(res))
                    print(b, lv, seed, f'{time.time() - t:.1f}s', len(res['rows']), 'rows', flush=True)
        print(merge_parts(parts, OUT / 'smoke_merged'))
    elif argv and argv[0] == 'unit':
        torch.set_num_threads(int(os.environ.get('BENCH_THREADS', '2')))
        b, lv, seed = argv[1], argv[2], int(argv[3])
        levels = planted.BENCHMARKS[b][1]
        lv = next(x for x in levels if str(x) == lv)
        Ns = tuple(int(x) for x in argv[argv.index('--N') + 1:]) if '--N' in argv else (500,)
        res = run_unit(b, lv, seed, Ns=Ns, verbose=True)
        p = OUT / 'parts' / f'{b}_{lv}_{seed}.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res))
        print(p, len(res['rows']))
    elif argv and argv[0] == 'summarise':
        print(merge_parts(argv[1], Path(argv[2]) if len(argv) > 2 else OUT))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
