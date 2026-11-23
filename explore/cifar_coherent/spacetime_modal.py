"""Through-depth stage (SPEC_SPACETIME.md) on Modal, reading the r20s0b operators on the volume.

Commands:  deploy | pilot | launch | status [pilot|launch] | fetch
Unit = (checkpoint, rule): rules 0 (kNN graph) and 1 (perplexity 100). CPU only.
Writes /analysis/r20s0b/spacetime/st_<step>_<rule>.npz and a timing record under /work/r20s0b/spacetime/.
The pilot unit (64000, 0) also runs the controls of spacetime_stage.control_runs.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
APP = 'cm-cifar-coherent-spacetime'
TAG = 'r20s0b'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
RULES = [0, 1]
NSITE = 12
CORES, MEM_GB = 8, 24
CPU_RATE, MEM_RATE = .0000131, .00000222
VERSION = 2

image = (modal.Image.debian_slim(python_version='3.12').pip_install('numpy', 'scipy')
         .env({'PYTHONPATH': '/root/cc', 'OMP_NUM_THREADS': str(CORES), 'OPENBLAS_NUM_THREADS': str(CORES)}))
for name in ('spacetime.py', 'spacetime_stage.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/cc/' + name)
app = modal.App(APP)
vol = modal.Volume.from_name('cm-cifar-coherent')
VOLS = {'/out': vol}


@app.function(image=image, cpu=CORES, memory=MEM_GB * 1024, timeout=4 * 3600, max_containers=16, retries=0, volumes=VOLS)
def unit(step: int, rule: int, controls: bool = False, tag: str = TAG):
    import numpy as np
    import spacetime_stage as G
    vol.reload()
    root, work = Path('/out/analysis') / tag, Path('/out/work') / tag
    outdir, wdir = root / 'spacetime', work / 'spacetime'
    outdir.mkdir(exist_ok=True); wdir.mkdir(exist_ok=True)
    out, timing = outdir / f'st_{step}_{rule}.npz', wdir / f'timing_{step}_{rule}.json'
    if out.exists() and timing.exists():
        row = json.loads(timing.read_text())
        if row.get('version') == VERSION and (row.get('controls') or not controls):
            return row
    start = time.time()
    probe = np.load(root / 'probe.npz')
    ms = np.stack([np.array(np.load(work / f'operators_{step}_{si}.npy', mmap_mode='r')[rule], dtype=np.float64)
                   for si in range(NSITE)])
    assert np.abs(ms.sum(1) - 1).max() < 1e-9 and np.abs(ms - ms.transpose(0, 2, 1)).max() < 1e-12
    res = G.compute(ms, probe['labels'].astype(int), controls=controls, seed=1000 * rule + STEPS.index(step),
                    log=lambda *a: print(step, rule, *a, f'{time.time() - start:.0f}s', flush=True))
    res['probe_idx'] = probe['idx']
    bad = G.accept(res)
    if bad:
        raise RuntimeError(f'unit {step},{rule} failed acceptance: {bad}')
    np.savez_compressed(out, **res)
    seconds = time.time() - start
    row = dict(step=step, rule=rule, seconds=seconds, controls=controls, version=VERSION,
               estimated_usd=seconds * (CORES * CPU_RATE + MEM_GB * MEM_RATE),
               worst_resid=float(max(res['resid'].max(), res['paper_resid'].max())),
               balance_check=float(max(res['balance_check_own'], res['balance_check_none'])), tstar_own=float(res['tstar_own']), tstar_none=float(res['tstar_none']))
    timing.write_text(json.dumps(row)); vol.commit()
    return row


@app.function(image=image, cpu=.25, memory=1024, timeout=12 * 3600, volumes=VOLS)
def coordinate(units: list):
    rows, spent = [], 0.0
    for row in unit.starmap([tuple(u) for u in units], return_exceptions=True):
        if isinstance(row, Exception):
            print('unit failed:', repr(row), flush=True); rows.append(dict(error=repr(row))); continue
        rows.append(row); spent += row.get('estimated_usd', 0)
        print('done', row, f'spent ${spent:.2f}', flush=True)
    vol.reload(); (Path('/out/work') / TAG / 'spacetime').mkdir(parents=True, exist_ok=True)
    (Path('/out/work') / TAG / 'spacetime' / f'coordinator_{len(units)}.json').write_text(json.dumps(rows))
    vol.commit()
    return dict(spent=spent, n=len(rows), rows=rows)


if __name__ == '__main__':
    cmd = sys.argv[1]
    rec = HERE / 'runs'
    if cmd == 'deploy':
        subprocess.run(['modal', 'deploy', __file__], check=True)
    elif cmd in ('pilot', 'launch'):
        units = [[64000, 0, True]] if cmd == 'pilot' else [[s, r, False] for s in STEPS for r in RULES if [s, r] != [64000, 0]]
        call = modal.Function.from_name(APP, 'coordinate').spawn(units)
        r = dict(stage=f'spacetime-{cmd}', call_id=call.object_id, launched=time.ctime(), units=len(units))
        (rec / f'spacetime-{cmd}.json').write_text(json.dumps(r, indent=1)); print(r)
    elif cmd == 'status':
        arg = sys.argv[2] if len(sys.argv) > 2 else 'launch'
        r = json.loads((rec / f'spacetime-{arg}.json').read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(r['call_id']).get(timeout=0), indent=1))
        except TimeoutError:
            print('running', r)
    elif cmd == 'fetch':
        dest = HERE / 'data' / TAG
        subprocess.run(['modal', 'volume', 'get', '--force', 'cm-cifar-coherent', f'/analysis/{TAG}/spacetime/', str(dest) + '/'], check=True)
        print(len(list((dest / 'spacetime').glob('st_*.npz'))), 'files')
