"""Projected cost and timeouts of the pilot and full run from the bench (SPEC 8 item 6). Laptop side, no Modal.

    uv run python -I mc_project.py runs/bench.json runs/diag.json [--spent=USD]     -> runs/projection.json

Counts of each operation per unit follow the U1/U2/U3 code (mc_decode, mc_schema). The decode itself was not
benchmarked (Stage A spends nothing on the model); its time per pass is v1's measured U1 decode (42 s for one
pass over the 1,000 snippets on an L40S, including v1's Grams at 6 positions), taken at 1.5x as a margin.
Timeouts: 2x the projection, at least 900 s. A stage may launch only if spent + (timeout + 90 s overhead) x rate
of every unit it will run (+ the coordinator) is under the $8 ceiling (mc_ledger).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_construction as K  # noqa: E402

L40S_RATE, CPU_RATE, MEM_RATE = .000542, .0000131, .00000222
RATE = L40S_RATE + 8 * CPU_RATE + 64 * MEM_RATE           # U1, U2, U3: L40S, 8 cores, 64 GiB
COORD = (6 * 3600 + 90) * (.25 * CPU_RATE + 1 * MEM_RATE)
OVERHEAD_S = 90
V1_DECODE_S = 42.2                                         # v1 U1 `decode_seconds` (work/m130/u1_trained.json)
DECODE_MARGIN = 1.5
CEILING = 8.0
GB = 2 ** 30


def project(bench, diag, spent=0.0):
    """`diag`: SEBA run to max_iter on structureless vectors at N = 8,000 (the bench's spectral op had clustered
    synthetic data, on which SEBA converges early, so it understates real spectra). With `roundings_parallel` the
    critical path of a spectral object is the r = 32 SEBA single-threaded (x 1.15 for 1 vs 8 threads) plus one
    k-means; the rest of the bench's spectral op (eigh, eta^2, AMI, sets) is kept."""
    op = {k: v['seconds'] for k, v in bench['ops'].items()}
    ex = bench['ops']
    io = ex['disk_io']
    nbytes = io['bytes']
    vol_w = io['volume_write_s'] / nbytes                  # seconds per byte
    vol_r = max(io['volume_read_s'], io['volume_write_s']) / nbytes   # the bench read was cached: take the write rate
    loc_w, loc_r = io['local_write_s'] / nbytes, io['local_read_s'] / nbytes
    n2 = 8 * K.N ** 2                                      # one N x N float64 array
    decode = V1_DECODE_S * DECODE_MARGIN
    grams_lag = K.N_LAYERS * (op['gram_z'] + op['gram_c']) * 1.15     # + renormalise / accumulate
    point_bytes = K.N * 24 * 768 * 4
    cons_bytes = K.N * 24 * (4608 + 24576) * 4
    lags = len(K.LAGS)
    u1 = {}
    u1['trained'] = dict(
        decode=lags * (decode + op['buffers']),
        grams=lags * grams_lag,
        writes=vol_w * ((K.N_LAYERS + 1) * n2 + point_bytes + 2 * n2 + cons_bytes +
                        (lags - 1) * (4 * n2 + point_bytes / 8)),
        predictions=op['bc_pred'], controls=4 * decode + 120, load=60)
    u1['init'] = dict(decode=decode + op['buffers'], grams=grams_lag,
                      writes=vol_w * ((K.N_LAYERS + 1) * n2 + point_bytes), load=30)
    cell, walk = op['cell'], ex['cell']['walk_s']
    stats = cell - walk
    seba32 = diag['seba_full_by_r'][str(max(K.R))] if str(max(K.R)) in diag['seba_full_by_r'] else \
        diag['seba_full_by_r'][max(K.R)]
    spec = ex['spectral']['eigh_s'] + 1.15 * seba32 + 1.0 + (op['spectral'] - ex['spectral']['eigh_s']) * 0.5
    n_lag_cells = (lags - 1) * 7
    u2 = {}
    u2['trained'] = dict(
        cells=(24 + 24 + 1 + 3) * cell + n_lag_cells * cell + 2 * 24 * walk,
        spectra=(len(K.SINGLE_LAYERS) * 2 + 4 + 9 + n_lag_cells) * spec,
        block_stats=9 * stats,
        reads=vol_r * (K.N_LAYERS + 3 + (lags - 1) * 4) * n2 + vol_r * 6 * point_bytes / 8,
        spill=(loc_w + 2 * loc_r) * (24 + 24 + 2 * lags) * n2)
    u2['init'] = dict(cells=49 * cell, spectra=14 * spec, block_stats=stats, reads=vol_r * (K.N_LAYERS + 2) * n2,
                      spill=(loc_w + 2 * loc_r) * 24 * n2)
    t8, t2 = op['tree_8000'], op['tree_2000']
    node_whole = lambda frac: (K.N_LAYERS * (op['gram_z'] + op['gram_c']) * frac ** 2 + walk * frac ** 3 +
                               (spec / 3) * frac ** 3)
    u3 = dict(T1=t8 + 2 * (t8 / 2) + 4 * t2, T2=node_whole(1) + 2 * node_whole(.5) + 4 * node_whole(.25),
              T2_load=vol_r * cons_bytes + vol_r * point_bytes,
              layouts=2 * op['umap'] + 60, compare_finalize=180)
    units = {'U1_trained': u1['trained'], 'U2_trained': u2['trained'], 'U1_init': u1['init'],
             'U2_init': u2['init'], 'U3': u3}
    seconds = {u: sum(parts.values()) for u, parts in units.items()}
    cost = {u: (s + OVERHEAD_S) * RATE for u, s in seconds.items()}
    timeouts = {u: int(max(900, 2 * s)) for u, s in seconds.items()}
    pilot = ['U1_trained', 'U2_trained']
    full = ['U1_init', 'U2_init', 'U3']
    reserve = lambda us: sum((timeouts[u] + OVERHEAD_S) * RATE for u in us) + COORD   # as mc_ledger.worst_case
    pilot_usd = sum(cost[u] for u in pilot)
    worst_pilot = spent + reserve(pilot)
    worst_full_after_projected_pilot = spent + pilot_usd + reserve(full)
    worst_full_after_worst_pilot = spent + reserve(pilot) + reserve(full)
    out = dict(rule='timeout = max(900 s, 2 x projected seconds), per unit',
               parts={u: {k: round(v, 1) for k, v in p.items()} for u, p in units.items()},
               seconds={u: round(s) for u, s in seconds.items()}, usd={u: round(c, 3) for u, c in cost.items()},
               pilot_usd=round(pilot_usd, 3), pilot_seconds=round(sum(seconds[u] for u in pilot)),
               full_usd=round(sum(cost[u] for u in units if u not in pilot), 3),
               full_seconds=round(sum(seconds[u] for u in units if u not in pilot)),
               total_usd=round(sum(cost.values()) + 2 * COORD / 6, 3), timeouts=timeouts,
               spent_usd=spent, projected_study_total_usd=round(spent + sum(cost.values()) + 2 * COORD / 6, 3),
               reservation_at_pilot_launch_usd=round(worst_pilot, 3),
               reservation_at_full_launch_if_pilot_as_projected_usd=round(worst_full_after_projected_pilot, 3),
               reservation_everything_at_timeout_usd=round(worst_full_after_worst_pilot, 3),
               fits_ceiling=bool(worst_pilot <= CEILING and worst_full_after_projected_pilot <= CEILING), ceiling=CEILING,
               rates=dict(L40S=L40S_RATE, cpu_core=CPU_RATE, mem_gib=MEM_RATE, unit_rate=RATE),
               assumptions=dict(decode_per_pass_s=decode, v1_decode_s=V1_DECODE_S, margin=DECODE_MARGIN))
    return out


if __name__ == '__main__':
    bench = json.loads(Path(sys.argv[1]).read_text())
    diag = json.loads(Path(sys.argv[2]).read_text())
    spent = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--spent=')), 0.0))
    out = project(bench, diag, spent)
    (HERE / 'runs' / 'projection.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('seconds', 'usd', 'pilot_usd', 'full_usd', 'total_usd', 'timeouts',
                                         'projected_study_total_usd', 'reservation_at_pilot_launch_usd',
                                         'reservation_at_full_launch_if_pilot_as_projected_usd',
                                         'reservation_everything_at_timeout_usd', 'fits_ceiling')}, indent=1))
