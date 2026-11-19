"""Fixture calibration of the multi-way reading threshold theta and control P1' (SPEC 5, 7, 8), through the real code
path (`mc_stats`: standard pipeline, cell walk, running-mean block, spectrum, both roundings, Upsilon_hat).

Generator (SPEC 8, written before any run): N items, a block of 24 cells, dimension 64. Item i with planted label
g has, in cell c, the vector s e^c_g + eps with e^c_1..e^c_k orthonormal and drawn independently per cell (the
partition is shared across cells; the geometry is not) and eps ~ N(0, I/64). Structureless: s = 0.
Cases: k in {2, 4, 8, 16}; sizes balanced or Zipf (proportional to 1/j); s in {0.5, 0.75, 1, 2}; one structureless
block per seed. Development seeds 0..2; check seeds 100..104 decide theta and P1' with s in {1, 2} as planted and the
structureless maximum over all k in R. The labels only generate data and score the result.

Seeding (an implementation choice, recorded in the README): every block draws from
`numpy.random.default_rng([seed, k, sizes_index, round(100 s)])`; the structureless block of a seed uses
[seed, 0, 0, 0]. Within a block: sizes, then a permutation of the labels, then per cell (QR of a 64 x k Gaussian,
then the noise).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

import mc_construction as K
import mc_stats as ST
import lm_tree as Z


def sizes_of(n, k, kind):
    w = np.ones(k) if kind == 'balanced' else 1.0 / np.arange(1, k + 1)
    raw = n * w / w.sum()
    out = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - out), kind='stable')[:n - out.sum()]:
        out[j] += 1
    return out


def generate(n, k, kind, s, seed, cells=K.FIX_CELLS, dim=K.FIX_DIM):
    """Yields the block's cells (n, dim) float64 one at a time, and the planted labels (first item yielded)."""
    rng = np.random.default_rng([seed, 0, 0, 0] if s == 0 else [seed, k, K.FIX_SIZES.index(kind), round(100 * s)])
    sizes = sizes_of(n, k, kind) if s > 0 else np.array([n])
    labels = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    yield labels
    for _ in range(cells):
        if s > 0:
            e = np.linalg.qr(rng.normal(size=(dim, k)))[0].T            # (k, dim) orthonormal rows
            x = s * e[labels] + rng.normal(size=(n, dim)) / math.sqrt(dim)
        else:
            x = rng.normal(size=(n, dim)) / math.sqrt(dim)
        yield x


def block_upsilon(cells, n, device='cpu', ranks=K.R):
    """Upsilon_hat(k) for every k in R (both roundings, the larger kept), the cap and mu, from cells through the
    real code path. `cells`: iterable of (n, d) arrays."""
    bm = ST.BlockMean()
    row_err = 0.0
    for x in cells:
        p, info = ST.cell_walk(ST.view_gram(torch.from_numpy(np.asarray(x)), device))
        row_err = K.worst_of([row_err, info['row_error']])
        bm.add(p)
        del p
    mbar = bm.mbar()
    bm.free()
    vals, vecs = ST.spectrum(mbar)
    mu = ST.mus(vals)
    pbar = ST.pbar_of(mbar)
    del mbar
    out = dict(mu=mu[1:].tolist(), gap=ST.gap_ratios(mu).tolist(), row_err=row_err, k={})
    vt = torch.from_numpy(vecs)
    from seba import seba
    for r in ranks:
        s = seba(vt, r).numpy()
        rec = dict(cap=float(2 * mu[r] / mu[r - 1]))
        for name, part in (('seba', ST.seba_partition(s)), ('kmeans', ST.kmeans_partition(vecs, r))):
            u, _ = ST.upsilon_hat(mu, r, pbar, part)
            rec[name] = u
        valid = [v for v in (rec['seba'], rec['kmeans']) if v is not None]
        rec['ups'] = (math.nan if any(math.isnan(v) for v in valid) else max(valid)) if valid else None
        out['k'][r] = rec
    return out


def case(n, k, kind, s, seed, device='cpu'):
    gen = generate(n, k, kind, s, seed)
    labels = next(gen)
    t0 = time.time()
    res = block_upsilon(gen, n, device)
    res.update(n=n, k_planted=k if s > 0 else None, sizes=kind if s > 0 else None, s=s, seed=seed,
               seconds=time.time() - t0, planted_sizes=np.bincount(labels).tolist())
    return res


def cases_for_seed(seed, n=K.FIX_N, device='cpu', log=print):
    out = [case(n, 0, None, 0.0, seed, device)]
    log(f'seed {seed}: structureless max Upsilon {max_ups(out[0]):.2f} ({out[0]["seconds"]:.0f}s)')
    for k in K.FIX_K:
        for kind in K.FIX_SIZES:
            for s in K.FIX_S:
                r = case(n, k, kind, s, seed, device)
                out.append(r)
                log(f'seed {seed} k={k} {kind} s={s}: Upsilon({k}) = {planted_ups(r)} ({r["seconds"]:.0f}s)')
    return out


def planted_ups(rec):
    """Upsilon_hat at the planted k (works on records before and after a json round trip, whose keys are str)."""
    k = rec['k_planted']
    row = rec['k'].get(k, rec['k'].get(str(k)))
    v = None if row is None else row['ups']
    return -math.inf if v is None else v


def max_ups(rec):
    vals = [r['ups'] for r in rec['k'].values() if r['ups'] is not None]
    return K.worst_of(vals) if vals else -math.inf          # a NaN reads as +inf: fails the structureless side


def calibrate(records):
    """theta by v1's rule on the check-seed records (planted: s in {1, 2} at their k; structureless: the maximum over
    all k in R), P1', and the detection curve (share certified at its k, by s, k and sizes)."""
    check = [r for r in records if r['seed'] in K.CHECK_SEEDS]
    planted = [planted_ups(r) for r in check if r['s'] in K.PLANTED_S]
    flat = [max_ups(r) for r in check if r['s'] == 0]
    assert len(planted) == len(K.CHECK_SEEDS) * len(K.FIX_K) * len(K.FIX_SIZES) * len(K.PLANTED_S), len(planted)
    assert len(flat) == len(K.CHECK_SEEDS)
    lo, hi = K.least_of(planted), K.worst_of(flat)
    theta, rule = Z.threshold_rule([lo], [hi])          # NaN anywhere -> no separation -> rule dropped
    p1 = None
    curve = {}
    if theta is not None:
        cert_planted = all(math.isfinite(v) and v >= theta for v in planted)
        cert_flat = [r['seed'] for r in check if r['s'] == 0 and any(
            (x['ups'] is not None and x['ups'] >= theta) for x in r['k'].values())]
        p1 = bool(cert_planted and not cert_flat)
        for s in K.FIX_S:
            for k in K.FIX_K:
                for kind in K.FIX_SIZES:
                    rs = [r for r in check if r['s'] == s and r['k_planted'] == k and r['sizes'] == kind]
                    curve[f's={s},k={k},{kind}'] = float(np.mean([planted_ups(r) >= theta for r in rs]))
    return dict(theta=theta, rule=rule, P1_prime=p1, planted_min=lo, structureless_max=hi, n_planted=len(planted), n_structureless=len(flat),
                detection=curve,
                detection_by_s={str(s): float(np.mean([planted_ups(r) >= theta for r in check
                                                        if r['s'] == s and r['k_planted']]))
                                for s in K.FIX_S} if theta is not None else None)


def full_size_check(theta, device='cpu', n=K.BENCH_N, log=print):
    """SPEC 8: N = 8,000, 24 cells, k = 8 Zipf at s = 1, and structureless, seed 100. Fails (calibrated = False) when
    the two fall on the wrong sides of theta."""
    c = K.BENCH_CHECK
    planted = case(n, c['k'], c['sizes'], c['s'], c['seed'], device)
    flat = case(n, 0, None, 0.0, c['seed'], device)
    pv, fv = planted_ups(planted), max_ups(flat)
    ok = theta is not None and pv >= theta and fv < theta
    log(f'full-size check: planted Upsilon(8) {pv:.2f}, structureless max {fv:.2f}, theta {theta} -> {ok}')
    return dict(planted=planted, structureless=flat, planted_ups=pv, structureless_max=fv, theta=theta,
                calibrated=bool(ok))
