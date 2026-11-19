"""Detached Modal coordinator for the Mamba study. FORK of `../lm_coherent/lm_modal.py` (the `modal` skill's
pattern). What changes: app/volume `cm-mamba-coherent`, tag `m130`; units are U0 (copy the Pythia probe verbatim,
tokenizer check; CPU), U1/U2 per checkpoint LABEL (`init`, `trained`), U3; U1 calls `mb_model.run_u1` (recurrent
decoding with HF's MambaCache, torch path only); the pilot gate adds C1a/C1b, the P3 Gram identity, the T4 Gram
route and the state/stack PN; the `diag` command is dropped. The ledger, guard, cache keys and gate refusal are
the Pythia ones, unchanged in logic.

    uv run --with modal python mb_modal.py deploy            # once per code change
    uv run --with modal python mb_modal.py budget            # print the reservation; spends nothing
    uv run --with modal python mb_modal.py launch pilot      # U0, then U1 + U2 on `trained`, then the gate
    uv run --with modal python mb_modal.py calibrate         # timeouts := 2 x pilot durations -> runs/timeouts.json
    uv run --with modal python mb_modal.py launch full       # U1 + U2 on `init`, then U3
    uv run --with modal python mb_modal.py status [stage]
    uv run --with modal python mb_modal.py fetch             # /analysis/m130 -> data/m130

Volume `cm-mamba-coherent` (mounted at /vol): /vol/hf (HF caches), /vol/work/m130 (activations, Grams incl. the
2.3 GB raw constituent Grams for T4, timings, attempts ledger, coordinator log), /vol/analysis/m130 (public files).

Budget (MB.budget = LC.budget): an append-only attempts ledger, work/m130/attempts/<unit>-<ns>.json, written by the
coordinator BEFORE dispatch at the unit's full worst case (timeout x rate) and reconciled on return (success: timed
cost + OVERHEAD_S; failure: wall time since dispatch, capped at the timeout, + OVERHEAD_S; lost: stays at the worst
case). Before every batch and before `launch`: attempts + worst case of every incomplete unit (+ the coordinator's
own worst case, at launch) <= CEILING_USD ($6).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LM = HERE.parent / 'lm_coherent'
CIFAR = HERE.parent / 'cifar_coherent'
sys.path.insert(0, str(HERE))
import mb_construction as MB  # noqa: E402

APP = 'cm-mamba-coherent'
VOLUME = 'cm-mamba-coherent'
TAG = 'm130'
CEILING_USD = 6.0
# modal.com/pricing as recorded on 2026-11-12 (lm_coherent/README.md); re-check before launch.
L40S_RATE, CPU_RATE, MEM_RATE = .000542, .0000131, .00000222
TRANSFORMERS = '4.57.6'
TORCH = '2.15.0'
COORD_CORES, COORD_GIB, COORD_TIMEOUT = .25, 1, 4 * 3600
OVERHEAD_S = 90            # per attempt: container start and image load, not timed inside the function
HW = {  # unit: (gpu, cores, GiB, default timeout s)
    'U0': (None, 2, 4, 900),
    'U1': ('L40S', 4, 32, 1500),
    'U2': (None, 8, 32, 2400),
    'U3': (None, 8, 32, 3600),
}
_tfile = HERE / 'runs' / 'timeouts.json'
TIMEOUTS = {u: hw[3] for u, hw in HW.items()}
if _tfile.is_file():
    TIMEOUTS.update(json.loads(_tfile.read_text())['timeouts'])
elif 'MB_TIMEOUTS' in os.environ:                        # inside a container: the deployed values
    TIMEOUTS.update(json.loads(os.environ['MB_TIMEOUTS']))


def rate(unit):
    gpu, cores, gib, _ = HW[unit]
    return (L40S_RATE if gpu else 0) + cores * CPU_RATE + gib * MEM_RATE


def units():
    """Every unit of the run: (name, kind, checkpoint)."""
    out = [('U0', 'U0', None)]
    out += [(f'U1_{c}', 'U1', c) for c in MB.CHECKPOINTS]
    out += [(f'U2_{c}', 'U2', c) for c in MB.CHECKPOINTS]
    return out + [('U3', 'U3', None)]


# torch path only: mamba-ssm, causal-conv1d and `kernels` are NOT installed (mb_model.assert_slow_path checks).
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install(f'torch=={TORCH}', 'numpy', 'scipy', f'transformers=={TRANSFORMERS}', 'umap-learn',
                      'huggingface_hub', 'safetensors')
         .env({'PYTHONPATH': '/root/mb:/root/lm:/root/cc', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'MB_TIMEOUTS': json.dumps(TIMEOUTS)}))
for name in ('coherent.py', 'seba.py', 'schema_v2.py', 'construction.py', 'sweep_math.py'):
    image = image.add_local_file(str(CIFAR / name), remote_path='/root/cc/' + name)
image = image.add_local_file(str(CIFAR / 'site' / 'analysis.py'), remote_path='/root/cc/site/analysis.py')
for name in ('lm_construction.py', 'lm_cells.py', 'lm_tree.py', 'lm_schema.py'):
    image = image.add_local_file(str(LM / name), remote_path='/root/lm/' + name)
for name in ('mb_construction.py', 'mb_cells.py', 'mb_model.py', 'mb_schema.py', 'cross_arch.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/mb/' + name)
image = image.add_local_file(str(LM / 'data' / 'p160t' / 'probe.json'), remote_path='/root/mb/probe.json')
app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
VOLS = {'/vol': vol}


def paths(tag=TAG):
    work, public = Path('/vol/work') / tag, Path('/vol/analysis') / tag
    work.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    return work, public


def removed_items(work):
    path = Path(work) / 'removed.json'
    return json.loads(path.read_text())['items'] if path.is_file() else []


def run_key(work, public):
    probe = Path(public) / 'probe.json'
    sha = MB.probe_identity(probe.read_bytes()) if probe.is_file() else None
    return MB.unit_key(sha, removed_items(work))


def done(work, name, start, unit, key, **extra):
    seconds = time.time() - start
    row = dict(unit=name, seconds=seconds, estimated_usd=seconds * rate(unit), started_at=start,
               completed_at=time.time(), **key, **extra)
    (work / f'timing_{name}.json').write_text(json.dumps(row, allow_nan=False))
    vol.commit()
    return row


def reused(row):
    return dict(row, cached=True)


def u1_outputs(work, checkpoint):
    names = [f'{k}_{checkpoint}.npy' for k in ('act', 'state', 'stack', 'bag')] + [f'u1_{checkpoint}.json']
    if checkpoint == MB.FINAL:
        names += [f'{k}_{checkpoint}.npy' for k in ('parts', 'raw', 'actS', 'stateS', 'stackS')]
    return [work / n for n in names]


def u2_outputs(work, public, checkpoint):
    out = [public / f'{k}_{checkpoint}.npz' for k in ('cells', 'single')]
    out += [work / f'{k}_{checkpoint}.json' for k in ('compare', 'trees', 'p3')]
    if checkpoint == MB.FINAL:
        out += [public / f'blocks_{checkpoint}.npz', public / 'figdata.npz', work / f't4check_{checkpoint}.json']
    return out


# ---------------------------------------------------------------------------------------------------- U0
@app.function(image=image, cpu=HW['U0'][1], memory=HW['U0'][2] * 1024, timeout=TIMEOUTS['U0'], volumes=VOLS)
def u0_probe(tag: str = TAG):
    import mb_schema as S
    vol.reload(); work, public = paths(tag); start = time.time()
    key = MB.unit_key(None, [])                      # depends on nothing but the construction
    cached = MB.cached_unit(work / 'timing_U0.json', [public / 'probe.json', work / 'u0.json'], key)
    if cached:
        return reused(cached)
    out = S.u0_main('/root/mb/probe.json', public, '/vol/hf/tok')
    (work / 'u0.json').write_text(json.dumps(out, allow_nan=False)); vol.commit()
    assert out['pass'], f"U0 failed: {out['checks']}"
    return done(work, 'U0', start, 'U0', key, checks=out['checks'])


# ---------------------------------------------------------------------------------------------------- U1
@app.function(image=image, gpu='L40S', cpu=HW['U1'][1], memory=HW['U1'][2] * 1024, timeout=TIMEOUTS['U1'],
              volumes=VOLS, max_containers=2)
def u1_forward(checkpoint: str, tag: str = TAG):
    """Recurrent decode, point activations, state/stack Grams (GPU float64), bag, C1a (trained), C1b, C2, P3 Gram
    identity (trained), zero-norm scan (mb_model.run_u1)."""
    import torch
    import mb_model as MM
    import mb_schema as S
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    MM.assert_slow_path()
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U1_{checkpoint}'
    key = run_key(work, public)
    cached = MB.cached_unit(work / f'timing_{name}.json', u1_outputs(work, checkpoint), key)
    if cached:
        return reused(cached)
    probe = S.Probe(public / 'probe.json', removed_items(work))
    model = MM.load_model(checkpoint, cache_dir='/vol/hf/mamba')
    vol.commit()
    log = lambda s: print(s, flush=True)
    info = MM.run_u1(model, checkpoint, probe, work, 'cuda', log=log, scratch='/tmp')
    vol.commit()
    return done(work, name, start, 'U1', key, C1b=info['C1b']['max_rel_err'],
                C1a=(info.get('C1a') or {}).get('max_rel_err'))


# ---------------------------------------------------------------------------------------------------- U2
@app.function(image=image, cpu=HW['U2'][1], memory=HW['U2'][2] * 1024, timeout=TIMEOUTS['U2'], volumes=VOLS,
              max_containers=2)
def u2_measure(checkpoint: str, tag: str = TAG):
    import torch
    import mb_schema as S
    torch.set_num_threads(HW['U2'][1])
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U2_{checkpoint}'
    key = run_key(work, public)
    cached = MB.cached_unit(work / f'timing_{name}.json', u2_outputs(work, public, checkpoint), key)
    if cached:
        return reused(cached)
    result = S.measure_checkpoint(work, public, checkpoint, S.Probe(public / 'probe.json', removed_items(work)),
                                  raw_in_memory=True)
    return done(work, name, start, 'U2', key, sinkhorn_max_row_err=result['sinkhorn_max_row_err'])


# ---------------------------------------------------------------------------------------------------- U3
@app.function(image=image, cpu=HW['U3'][1], memory=HW['U3'][2] * 1024, timeout=TIMEOUTS['U3'], volumes=VOLS)
def u3_finalize(tag: str = TAG):
    import torch
    import mb_schema as S
    torch.set_num_threads(HW['U3'][1])
    vol.reload(); work, public = paths(tag); start = time.time()
    key = run_key(work, public)
    u0 = json.loads((work / 'u0.json').read_text())
    report = S.finalize(work, public, S.Probe(public / 'probe.json', removed_items(work)),
                        extra_meta=dict(probe_sha=key['probe_sha'], U0=u0))
    return done(work, 'U3', start, 'U3', key, validation=report)


# ------------------------------------------------------------------------------------------- budget/ledger
class Store:
    """Read access to the run's work directory: the mounted volume inside Modal, the volume API on the laptop."""

    def __init__(self, tag=TAG, mounted=False):
        self.mounted, self.tag = mounted, tag
        self.root = Path('/vol/work') / tag

    def read(self, rel, base='work'):
        try:
            if self.mounted:
                return (Path('/vol') / base / self.tag / rel).read_bytes()
            return b''.join(vol.read_file(f'{base}/{self.tag}/{rel}'))
        except Exception:
            return None

    def json(self, rel, base='work'):
        raw = self.read(rel, base)
        return json.loads(raw) if raw is not None else None

    def attempts(self):
        if self.mounted:
            return [json.loads(p.read_text()) for p in sorted((self.root / 'attempts').glob('*.json'))]
        try:
            entries = vol.listdir(f'work/{self.tag}/attempts')
        except Exception:
            return []
        return [json.loads(b''.join(vol.read_file(e.path))) for e in entries if e.path.endswith('.json')]

    def key(self):
        probe = self.read('probe.json', base='analysis')
        removed = (self.json('removed.json') or {}).get('items', [])
        return MB.unit_key(MB.probe_identity(probe) if probe is not None else None, removed)


def coordinator_worst():
    return COORD_TIMEOUT * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE)


def status_of(store):
    key = store.key()
    worst = {name: TIMEOUTS[kind] * rate(kind) for name, kind, _ in units()}
    complete = set()
    for name, kind, _ in units():
        row = store.json(f'timing_{name}.json')
        if MB.key_matches(row, MB.unit_key(None, []) if kind == 'U0' else key):
            complete.add(name)
    return key, worst, complete


def guard(store, log=print, extra=0.0):
    key, worst, complete = status_of(store)
    b = MB.budget(store.attempts(), worst, complete, extra)
    log(f"budget: attempts ${b['spent']:.3f} + incomplete units ${b['reserve']:.3f} + coordinator "
        f"${b['extra']:.3f} = ${b['total']:.3f} (ceiling ${CEILING_USD})")
    if b['total'] > CEILING_USD:
        raise RuntimeError(f"reservation ${b['total']:.2f} exceeds ceiling ${CEILING_USD}; nothing dispatched")
    return b


def open_attempt(work, name, kind, reserved):
    folder = work / 'attempts'; folder.mkdir(exist_ok=True)
    path = folder / f'{name}-{time.time_ns()}.json'
    path.write_text(json.dumps(dict(unit=name, kind=kind, reserved_usd=reserved, actual_usd=None,
                                    dispatched_at=time.time(), fingerprint=MB.FINGERPRINT)))
    vol.commit()
    return path


def close_attempt(path, actual, outcome):
    row = json.loads(path.read_text())
    row.update(actual_usd=actual, outcome=outcome, closed_at=time.time())
    path.write_text(json.dumps(row)); vol.commit()


def gate_record(work, public, key):
    """The pilot gate (SPEC 7) from the pilot's own outputs, stamped with the run key: U0, C1a (float64 parameters
    and cache, < 1e-4), C1b (< 1e-12), C2, P3 (linearity on real blocks + the Gram identity), Sinkhorn, duplicates,
    zero norm, min PN(psi_1) >= 10 over point, state and stack cells, and the T4 Gram route at the root."""
    import numpy as np
    u0 = json.loads((work / 'u0.json').read_text())
    u1 = json.loads((work / f'u1_{MB.FINAL}.json').read_text())
    p3 = json.loads((work / f'p3_{MB.FINAL}.json').read_text())
    t4 = json.loads((work / f't4check_{MB.FINAL}.json').read_text())
    with np.load(public / f'cells_{MB.FINAL}.npz') as z:
        pn_by_view = {v: float(z['pn'][vi, ..., 0].min()) for vi, v in enumerate(MB.VIEWS)}
        row_err = float(max(z['row_err'].max(), z['bag_row_err'].max()))
        min_dist = float(min(z['min_dist'].min(), z['bag_min_dist'].min()))
    pn1 = min(pn_by_view.values())
    checks = dict(U0=bool(u0['pass']), C1a=bool(u1['C1a']['pass']), C1b=bool(u1['C1b']['pass']),
                  C2=bool(u1['C2']['identical']), P3=bool(p3['pass'] and u1['P3_gram_identity']['pass']),
                  sinkhorn=row_err < 2e-12, duplicates=min_dist > 1e-9, zero_norm=not u1['zero_norm'],
                  pn_psi1=pn1 >= 10, T4_gram_route=bool(t4['pass']))
    record = dict(**key, checks=checks, C1a=u1['C1a'], C1b=u1['C1b'], C2=u1['C2'], P3=p3,
                  P3_gram_identity=u1['P3_gram_identity'], T4=t4, min_pn_psi1=pn1, min_pn_psi1_by_view=pn_by_view,
                  max_row_err=row_err, min_dist=min_dist, zero_norm=u1['zero_norm'], U0_checks=u0['checks'],
                  **{'pass': all(checks.values())})
    (work / 'pilot_gate.json').write_text(json.dumps(record, allow_nan=False)); vol.commit()
    return record


@app.function(image=image, cpu=COORD_CORES, memory=COORD_GIB * 1024, timeout=COORD_TIMEOUT, volumes=VOLS)
def coordinate(stage: str, tag: str = TAG):
    work, public = paths(tag)
    store = Store(tag, mounted=True)
    log_path = work / 'coordinator.log'
    started = time.time()

    def say(text):
        with log_path.open('a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')
        print(text, flush=True); vol.commit()

    def batch(kind, fn, args):
        """Guard, record an attempt per unit (full worst case), dispatch, reconcile what returns."""
        vol.reload()
        guard(store, say)
        names = [kind if a is None else f'{kind}_{a}' for a in args]
        files = [open_attempt(work, n, kind, TIMEOUTS[kind] * rate(kind)) for n in names]
        t0 = time.time()
        if args == [None]:
            try:
                results = [fn.remote(tag)]
            except Exception as error:
                results = [error]
        else:
            results = list(fn.map(args, kwargs={'tag': tag}, return_exceptions=True))
        failed = []
        for name, path, res in zip(names, files, results):
            if isinstance(res, BaseException):
                bound = (min(time.time() - t0, TIMEOUTS[kind]) + OVERHEAD_S) * rate(kind)
                close_attempt(path, bound, f'failed: {res!r}'[:500]); failed.append((name, repr(res)[:300]))
            else:
                worker = 0.0 if res.get('cached') else res['estimated_usd']
                close_attempt(path, worker + OVERHEAD_S * rate(kind), 'cached' if res.get('cached') else 'ok')
        say(f'{kind} {args}: {len(results) - len(failed)} ok, failed {failed}')
        if failed:
            raise RuntimeError(f'{kind} failures: {failed}')

    def u1(args):
        try:
            batch('U1', u1_forward, args)
        finally:
            vol.reload()
            zero = sorted({i for c in MB.CHECKPOINTS for i in (store.json(f'u1_{c}.json') or {}).get('zero_norm', [])})
            if zero:
                (work / 'removed.json').write_text(json.dumps(dict(items=zero, reason='zero centred norm')))
                vol.commit()
                say(f'zero-norm items {zero} recorded in removed.json: the pilot must be rerun')

    me = open_attempt(work, f'coordinator_{stage}', 'COORD', coordinator_worst())
    outcome = 'failed'
    try:
        if stage in ('pilot', 'probe'):
            batch('U0', u0_probe, [None])
        if stage == 'pilot':
            u1([MB.FINAL])
            batch('U2', u2_measure, [MB.FINAL])
            vol.reload()
            gate = gate_record(work, public, run_key(work, public))
            say(f"pilot gate: pass={gate['pass']} {json.dumps(gate['checks'])}")
        if stage in ('full', 'u3'):
            vol.reload()
            gate = json.loads((work / 'pilot_gate.json').read_text()) if (work / 'pilot_gate.json').is_file() else None
            if not MB.gate_ok(gate, run_key(work, public)):
                raise RuntimeError('no passing pilot-gate record for this construction, probe and removal list')
        if stage == 'full':
            rest = [c for c in MB.CHECKPOINTS if c != MB.FINAL]
            u1(rest)
            batch('U2', u2_measure, rest)
        if stage in ('full', 'u3'):
            batch('U3', u3_finalize, [None])
        outcome = 'ok'
    finally:
        close_attempt(me, (time.time() - started) * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE), outcome)
        b = MB.budget(store.attempts(), *status_of(store)[1:])
        say(f"{stage} {outcome}; attempts total ${b['spent']:.3f}")
    return dict(stage=stage, spent=b['spent'])


# ------------------------------------------------------------------------------------------- laptop side
if __name__ == '__main__':
    args = sys.argv[1:]
    command = args[0] if args else 'deploy'
    rec_dir = HERE / 'runs'; rec_dir.mkdir(exist_ok=True)
    if command == 'deploy':
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif command == 'budget':
        store = Store()
        key, worst, complete = status_of(store)
        b = guard(store, extra=coordinator_worst())
        print(json.dumps(dict(**b, ceiling=CEILING_USD, timeouts=TIMEOUTS, worst_by_unit=worst,
                              complete=sorted(complete), attempts=store.attempts()), indent=1))
    elif command == 'launch':
        stage = args[1]
        assert stage in ('probe', 'pilot', 'full', 'u3'), stage
        store = Store()
        guard(store, extra=coordinator_worst())
        if stage in ('full', 'u3') and not MB.gate_ok(store.json('pilot_gate.json'), store.key()):
            raise SystemExit('refused: no passing pilot-gate record for this construction, probe and removals')
        call = modal.Function.from_name(APP, 'coordinate').spawn(stage, TAG)
        rec = dict(stage=stage, call_id=call.object_id, launched=time.ctime(), fingerprint=MB.FINGERPRINT,
                   timeouts=TIMEOUTS)
        (rec_dir / f'{stage}.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))
    elif command == 'status':
        stage = args[1] if len(args) > 1 else 'pilot'
        rec = json.loads((rec_dir / f'{stage}.json').read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec['call_id']).get(timeout=0)))
        except TimeoutError:
            print('running')
        except Exception as error:
            print(f'coordinator FAILED: {error!r} (`modal app logs {APP}`)')
        store = Store()
        log = store.read('coordinator.log')
        print(log.decode()[-3000:] if log else 'no log yet')
        u0 = store.json('u0.json')
        if u0:
            print('U0', json.dumps(dict(checks=u0['checks'], literal=u0['literal_spec_check']['mismatches'])))
        u1 = store.json(f'u1_{MB.FINAL}.json')
        if u1:
            print('U1', json.dumps({k: u1.get(k) for k in ('C1a', 'C1b', 'C2', 'P3_gram_identity', 'zero_norm')},
                                   indent=1))
        if stage == 'pilot':
            print(json.dumps(store.json('pilot_gate.json'), indent=1))
    elif command == 'calibrate':
        store = Store()
        secs = {k: (store.json(f'timing_{u}.json') or {}).get('seconds') for k, u in
                (('U0', 'U0'), ('U1', f'U1_{MB.FINAL}'), ('U2', f'U2_{MB.FINAL}'))}
        assert all(secs.values()), f'pilot incomplete: {secs}'
        timeouts = {k: int(max(120, 2 * v)) for k, v in secs.items()}
        timeouts['U3'] = TIMEOUTS['U3']
        _tfile.write_text(json.dumps(dict(timeouts=timeouts, pilot_seconds=secs, rule='2 x pilot duration'),
                                     indent=1))
        print(json.dumps(timeouts), '-> redeploy before launching full')
    elif command == 'fetch':
        dest = HERE / 'data'; dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force', VOLUME, f'analysis/{TAG}',
                        str(dest)], check=True)
    else:
        raise SystemExit(__doc__)
