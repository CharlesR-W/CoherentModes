"""Modal CPU fan-out of the planted / null benchmark sweep (bench.run_unit), detached coordinator, results on the
Volume `cm-cifar-baselines-bench`. Pattern of ~/.claude/skills/modal/modal_job.py.

    uv run --with modal python explore/cifar_baselines/bench_modal.py deploy
    uv run --with modal python explore/cifar_baselines/bench_modal.py launch pilot|full
    uv run --with modal python explore/cifar_baselines/bench_modal.py status <run_id>
    uv run --with modal python explore/cifar_baselines/bench_modal.py fetch <run_id>     # -> out/bench/parts/<run_id>/

Unit = (bench, level, seed, N); P12 units hold all N (one training per seed and level). Each part records its own
wall-clock so spend = sum(seconds) x (CPU + memory price) can be reported (bench_modal.py cost <run_id>).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

APP_NAME = 'cm-cifar-baselines-bench'
CPU = 2.0                 # physical cores per container
MEM_MB = 4096
TIMEOUT_S = 60 * 60
MAX_CONTAINERS = 60
PRICE_CORE_S = 0.0000131  # modal.com/pricing, fetched 2026-11-15
PRICE_GIB_S = 0.00000222

HERE = Path(__file__).resolve().parent
EXPLORE = HERE.parent
REMOTE = '/root/explore'

_files = {
    'cifar_baselines': ['bench.py', 'planted.py', 'sources.py', 'stats.py', 'variants.py', 'transfer.py'],
    'cifar_coherent': ['coherent.py', 'seba.py', 'sweep_math.py', 'construction.py'],
    'vit_coherent': ['mlp_planted.py', 'ops.py'],
}
IMAGE = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy', 'scipy', 'scikit-learn')
         .env({'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}))
for d, names in _files.items():
    for n in names:
        IMAGE = IMAGE.add_local_file(EXPLORE / d / n, f'{REMOTE}/{d}/{n}')

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(APP_NAME, create_if_missing=True)
RESULTS = '/results'


def unit_name(u):
    ns = '-'.join(str(n) for n in u['Ns'])
    return f"{u['bench']}_{u['level']}_s{u['seed']}_N{ns}"


@app.function(image=IMAGE, cpu=CPU, memory=MEM_MB, timeout=TIMEOUT_S, volumes={RESULTS: vol},
              max_containers=MAX_CONTAINERS)
def run_unit(run_id: str, unit: dict) -> str:
    sys.path.insert(0, f'{REMOTE}/cifar_baselines')
    import torch
    torch.set_num_threads(2)
    import bench
    import planted
    t0 = time.time()
    lv = next(x for x in planted.BENCHMARKS[unit['bench']][1] if str(x) == str(unit['level']))
    res = bench.run_unit(unit['bench'], lv, unit['seed'], Ns=tuple(unit['Ns']))
    res['meta']['seconds'] = time.time() - t0
    res['meta']['unit'] = unit
    part = Path(RESULTS) / run_id / f'{unit_name(unit)}.json'
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(res))
    vol.commit()
    return str(part)


@app.function(image=IMAGE, timeout=24 * 60 * 60, volumes={RESULTS: vol})
def coordinator(run_id: str, units: list) -> dict:
    done, failed = [], []
    for unit, res in zip(units, run_unit.map([run_id] * len(units), units, return_exceptions=True)):
        if isinstance(res, Exception):
            failed.append({'unit': unit, 'error': repr(res)})
        else:
            done.append(res)
    manifest = {'run_id': run_id, 'n_units': len(units), 'done': done, 'failed': failed}
    p = Path(RESULTS) / run_id / 'MANIFEST.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1))
    vol.commit()
    return manifest


# ------------------------------------------------------------------------------------------------ unit lists


def all_units(seeds=range(10), Ns=(500, 1000, 2000)):
    sys.path.insert(0, str(HERE))
    import planted
    units = []
    for b, (_, levels) in planted.BENCHMARKS.items():
        for lv in levels:
            for s in seeds:
                if b == 'P12':
                    units.append(dict(bench=b, level=str(lv), seed=s, Ns=list(Ns)))
                else:
                    units += [dict(bench=b, level=str(lv), seed=s, Ns=[n]) for n in Ns]
    # largest first, so the long tail starts early
    return sorted(units, key=lambda u: -max(u['Ns']) - (10 ** 4 if u['bench'] == 'P12' else 0))


def pilot_units():
    sys.path.insert(0, str(HERE))
    import planted
    units = []
    for b, (_, levels) in planted.BENCHMARKS.items():
        lv = levels[len(levels) // 2]
        units.append(dict(bench=b, level=str(lv), seed=0, Ns=[500, 1000, 2000]) if b == 'P12'
                     else dict(bench=b, level=str(lv), seed=0, Ns=[2000]))
    return units


# ------------------------------------------------------------------------------------------------ CLI


def _runs_dir():
    d = HERE / 'out' / 'bench' / 'runs'
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_launch(which):
    if which == 'pilot':
        units = pilot_units()
    elif which == 'partition':   # rerun after the mj_max declared-set-count fix (review P2)
        aff = {'P1', 'P3', 'P3L', 'P4', 'P5', 'P7', 'P9', 'P10', 'P11', 'P12'}
        units = [u for u in all_units() if u['bench'] in aff]
    elif which == 'full':
        units = all_units()
    else:
        units = json.loads(Path(which).read_text())
    run_id = f'{which if which in ("pilot", "full", "partition") else "custom"}-' + time.strftime('%Y%m%d-%H%M%S')
    call = modal.Function.from_name(APP_NAME, 'coordinator').spawn(run_id, units)
    rec = dict(run_id=run_id, call_id=call.object_id, n_units=len(units), cpu=CPU, mem_mb=MEM_MB,
               launched=time.ctime())
    (_runs_dir() / f'{run_id}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def cmd_status(run_id):
    rec = json.loads((_runs_dir() / f'{run_id}.json').read_text())
    call = modal.FunctionCall.from_id(rec['call_id'])
    try:
        m = call.get(timeout=0)
    except TimeoutError:
        try:
            n = sum(1 for e in vol.listdir(f'/{run_id}') if e.path.endswith('.json'))
        except Exception:
            n = 0
        print(f"running: {n}/{rec['n_units']} parts written")
        return
    print(f"finished: {len(m['done'])} done, {len(m['failed'])} failed")
    for f in m['failed'][:20]:
        print('  FAILED', f)


def cmd_fetch(run_id):
    dest = HERE / 'out' / 'bench' / 'parts' / run_id
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in vol.listdir(f'/{run_id}'):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        t = dest / Path(e.path).name
        if not t.exists():
            t.write_bytes(b''.join(vol.read_file(e.path)))
            n += 1
    print(f'fetched {n} new files to {dest}')


def cmd_cost(run_id):
    dest = HERE / 'out' / 'bench' / 'parts' / run_id
    secs = [json.loads(p.read_text())['meta'].get('seconds', 0) for p in dest.glob('*.json') if p.name != 'MANIFEST.json']
    tot = sum(secs)
    usd = tot * (CPU * PRICE_CORE_S + MEM_MB / 1024 * PRICE_GIB_S)
    print(f'{len(secs)} parts, {tot:.0f} container-s (unit compute only, excludes start-up), ~${usd:.2f}')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'deploy':
        import subprocess
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif a[0] == 'launch':
        cmd_launch(a[1])
    elif a[0] == 'status':
        cmd_status(a[1])
    elif a[0] == 'fetch':
        cmd_fetch(a[1])
    elif a[0] == 'cost':
        cmd_cost(a[1])
    else:
        sys.exit(__doc__)
