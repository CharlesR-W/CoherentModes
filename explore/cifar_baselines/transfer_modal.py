"""Modal CPU fan-out for the transfer pilot (builder C). Adapted from ~/.claude/skills/modal/modal_job.py.

    uv run --with modal --with numpy --with scipy python transfer_modal.py deploy
    ... transfer_modal.py launch-curve [--seeds 0 1 2] [--n-perm 20]      P8 sample-size curve (synthetic)
    ... transfer_modal.py launch-real [--smoke]                           real pilot: 2,000-image units + 10k units
    ... transfer_modal.py status <run_id>
    ... transfer_modal.py fetch <run_id>     -> out/transfer/modal/<run_id>/ ; curve_p8.json or real/*.json assembled

CPU only (no GPU). P8 and 2,000-image units: 2 physical cores, 4 GiB. 10k units: 2 cores, 8 GiB. Inputs: builder A's
raw Grams on volume `cm-cifar-coherent` (read-only, mounted at /in; files under /in/work/r20s0c). Results: volume
`crw-cm-transfer-results`. Every launch writes out/transfer/modal_runs/<run_id>.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = 'crw-cm-transfer'
CPU, MEM_MB, MEM_10K_MB = 2.0, 4096, 8192
TIMEOUT_S = 50 * 60
RUNS = HERE / 'out' / 'transfer' / 'modal_runs'
GRAMS = '/in/work/r20s0c'

IMAGE = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy', 'scipy')
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cpu')
    .env({'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    .add_local_file(str(HERE / 'transfer.py'), '/opt/explore/cifar_baselines/transfer.py')
    .add_local_file(str(HERE / 'transfer_pilot.py'), '/opt/explore/cifar_baselines/transfer_pilot.py')
    .add_local_file(str(HERE / 'planted.py'), '/opt/explore/cifar_baselines/planted.py')
    .add_local_file(str(HERE / 'stats.py'), '/opt/explore/cifar_baselines/stats.py')
    .add_local_file(str(HERE.parent / 'cifar_coherent' / 'coherent.py'), '/opt/explore/cifar_coherent/coherent.py')
    .add_local_file(str(HERE.parent / 'cifar_coherent' / 'seba.py'), '/opt/explore/cifar_coherent/seba.py')
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(f'{APP_NAME}-results', create_if_missing=True)
inputs = modal.Volume.from_name('cm-cifar-coherent')
RESULTS = '/results'


def unit_name(unit):
    return '-'.join(f'{k}{v}' for k, v in sorted(unit.items()) if k != 'cfg').replace(':', '_')


def _work(run_id, unit):
    sys.path.insert(0, '/opt/explore/cifar_baselines')
    import torch
    torch.set_num_threads(2)
    import transfer_pilot as tp
    t0 = time.time()
    kind = unit.get('kind', 'p8')
    cfg = unit.get('cfg', {})
    if kind == 'p8':
        res = tp.p8_unit(**{k: v for k, v in unit.items() if k not in ('kind', 'cfg')})
        part = Path(RESULTS) / run_id / f'{unit_name(unit)}.json'
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps(dict(unit=unit, result=res, seconds=time.time() - t0)))
    else:
        if kind == 'pair10k':
            out = tp.run_real10k_unit(unit, GRAMS, tp.TEMPLATE_10K, n_perm=cfg.get('n_perm_10k', 50),
                                      m_values=tuple(cfg.get('m', tp.M_VALUES)))
        else:
            out = tp.run_real_unit(unit, GRAMS, tp.TEMPLATE_2000, n_splits=cfg.get('n_splits', 10),
                                   n_perm=cfg.get('n_perm', 200), n_perm_fit=cfg.get('n_perm_fit', 50),
                                   m_values=tuple(cfg.get('m', tp.M_VALUES)))
        out['unit'] = {k: v for k, v in unit.items() if k != 'cfg'}
        part = Path(RESULTS) / run_id / f'{unit_name(unit)}.pkl'
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(pickle.dumps(out))
    vol.commit()
    return str(part)


@app.function(image=IMAGE, cpu=CPU, memory=MEM_MB, timeout=TIMEOUT_S, volumes={RESULTS: vol, '/in': inputs.read_only()},
              max_containers=80)
def run_unit(run_id: str, unit: dict) -> str:
    return _work(run_id, unit)


@app.function(image=IMAGE, cpu=CPU, memory=MEM_10K_MB, timeout=3 * TIMEOUT_S,
              volumes={RESULTS: vol, '/in': inputs.read_only()}, max_containers=40)
def run_unit_10k(run_id: str, unit: dict) -> str:
    return _work(run_id, unit)


@app.function(image=IMAGE, timeout=8 * 60 * 60, volumes={RESULTS: vol, '/in': inputs.read_only()})
def coordinator(run_id: str, cfg: dict) -> dict:
    sys.path.insert(0, '/opt/explore/cifar_baselines')
    units = cfg['units']
    if cfg.get('preflight'):
        import transfer_pilot as tp
        pre = cfg['preflight']
        if 'n2000' in pre:
            tp.preflight(GRAMS, tp.TEMPLATE_2000, [tuple(c) for c in pre['n2000']], 2000)
        if 'n10k' in pre:
            tp.preflight(GRAMS, tp.TEMPLATE_10K, [tuple(c) for c in pre['n10k']], 10000, tp.INDEX_10K)
    small = [u for u in units if u.get('kind') != 'pair10k']
    big = [u for u in units if u.get('kind') == 'pair10k']
    done, failed = [], []
    calls = []
    if big:
        calls.append((big, run_unit_10k.map([run_id] * len(big), big, return_exceptions=True)))
    if small:
        calls.append((small, run_unit.map([run_id] * len(small), small, return_exceptions=True)))
    for us, it in calls:
        for unit, res in zip(us, it):
            (failed if isinstance(res, Exception) else done).append(
                dict(unit={k: v for k, v in unit.items() if k != 'cfg'}, error=repr(res))
                if isinstance(res, Exception) else res)
    manifest = dict(run_id=run_id, n_units=len(units), done=done, failed=failed)
    vol.reload()
    p = Path(RESULTS) / run_id / 'MANIFEST.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1))
    vol.commit()
    return manifest


def _launch(units, kind, extra=None):
    run_id = time.strftime('%Y%m%d-%H%M%S')
    cfg = dict(units=units, **(extra or {}))
    call = modal.Function.from_name(APP_NAME, 'coordinator').spawn(run_id, cfg)
    rec = dict(run_id=run_id, call_id=call.object_id, n_units=len(units), cpu=CPU, mem_mb=MEM_MB,
               mem_10k_mb=MEM_10K_MB, launched=time.ctime(), kind=kind, cfg={k: v for k, v in (extra or {}).items()})
    (RUNS / f'{run_id}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def main():
    a = sys.argv[1:]
    RUNS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    if not a or a[0] == 'deploy':
        import subprocess
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif a[0] == 'launch-curve':
        import transfer_pilot as tp
        seeds = (0, 1, 2); n_perm = 20
        if '--seeds' in a:
            i = a.index('--seeds'); seeds = tuple(int(x) for x in a[i + 1:] if x.isdigit())
        if '--n-perm' in a:
            n_perm = int(a[a.index('--n-perm') + 1])
        units = tp.curve_units(seeds, n_perm)
        units.sort(key=lambda u: -(u['n_fit'] * (8 if u['estimator'] == 'diffusion' else 1) * (1 + u['n_perm'])))
        _launch(units, 'p8 curve')
    elif a[0] == 'launch-real':
        import transfer_pilot as tp
        smoke = '--smoke' in a
        cfg = dict(n_splits=10, n_perm=200, n_perm_fit=50, n_perm_10k=50, m=list(tp.M_VALUES))
        if smoke:
            cfg.update(n_splits=2, n_perm=2, n_perm_fit=1, n_perm_10k=1)
        units, pair_ids, triple_ids = tp.real_units_list(cfg['n_splits'])
        u10, p10 = tp.real10k_units_list()
        if smoke:
            units = [u for u in units if u['id'] in ('stem-b1', 'b5:1024-64000', 'b5:1024-6400-64000')]
            u10 = [u for u in u10 if u['id'] == 'b4-b5' and u['n_fit'] in (500, 6000)]
        all_units = [dict(u, cfg=cfg) for u in u10 + units]
        pre = dict(n2000=[list(c) for c in tp._pair_cells(pair_ids, triple_ids)],
                   n10k=[list(c) for c in tp._pair_cells(p10)])
        _launch(all_units, 'real smoke' if smoke else 'real', dict(preflight=pre, run_cfg=cfg))
    elif a[0] == 'status':
        rec = json.loads((RUNS / f'{a[1]}.json').read_text())
        call = modal.FunctionCall.from_id(rec['call_id'])
        try:
            man = call.get(timeout=0)
        except TimeoutError:
            n = sum(1 for e in vol.listdir(f'/{a[1]}', recursive=True) if e.path.endswith(('.json', '.pkl')))
            print(f"running: {n}/{rec['n_units']} parts written")
            return
        except Exception as error:
            print(f'coordinator FAILED: {error!r}')
            return
        print(f"finished: {len(man['done'])} done, {len(man['failed'])} failed")
        for f in man['failed']:
            print('  FAILED', f)
    elif a[0] == 'fetch':
        run_id = a[1]
        rec = json.loads((RUNS / f'{run_id}.json').read_text())
        dest = HERE / 'out' / 'transfer' / 'modal' / run_id
        dest.mkdir(parents=True, exist_ok=True)
        jsons, pkls = [], []
        for e in vol.listdir(f'/{run_id}', recursive=True):
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            data = b''.join(vol.read_file(e.path))
            (dest / Path(e.path).name).write_bytes(data)
            if e.path.endswith('.pkl'):
                pkls.append(pickle.loads(data))
            elif not e.path.endswith('MANIFEST.json'):
                jsons.append(json.loads(data))
        cost_s = CPU * 0.0000131 + MEM_MB / 1024 * 0.00000222
        cost_b = CPU * 0.0000131 + MEM_10K_MB / 1024 * 0.00000222
        if jsons:
            secs = sum(d['seconds'] for d in jsons)
            out = HERE / 'out' / 'transfer' / 'curve_p8.json'
            out.write_text(json.dumps(dict(sigma=0.35, run_id=run_id, container_seconds=secs,
                                           units=[d['result'] for d in jsons])))
            print(f'fetched {len(jsons)} P8 units; unit seconds {secs:.0f}; est. ${secs * cost_s:.3f}')
        if pkls:
            import transfer_pilot as tp
            cfg = rec['cfg']['run_cfg']
            small = [p for p in pkls if p['unit']['kind'] != 'pair10k']
            big = [p for p in pkls if p['unit']['kind'] == 'pair10k']
            s_small = sum(p['seconds'] for p in small); s_big = sum(p['seconds'] for p in big)
            od = HERE / 'out' / 'transfer' / ('real' if rec['kind'] == 'real' else 'real_smoke_modal')
            od.mkdir(parents=True, exist_ok=True)
            if small:
                res = tp.assemble_real(small, n_splits=cfg['n_splits'], n_perm=cfg['n_perm'],
                                       n_perm_fit=cfg['n_perm_fit'], m_values=tuple(cfg['m']))
                res['modal_run'] = run_id
                (od / 'transfer_pilot.json').write_text(json.dumps(res))
            if big:
                res = tp.assemble_real10k(big, n_perm=cfg['n_perm_10k'])
                res['modal_run'] = run_id
                (od / 'transfer_pilot_10k.json').write_text(json.dumps(res))
            print(f'fetched {len(small)} 2,000-image units ({s_small:.0f} s, est. ${s_small * cost_s:.3f}) and '
                  f'{len(big)} 10k units ({s_big:.0f} s, est. ${s_big * cost_b:.3f}) -> {od}')


if __name__ == '__main__':
    main()
