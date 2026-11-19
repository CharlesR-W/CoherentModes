"""Detached Modal coordinator for the token-context study. FORK of `../mamba_coherent/mb_modal.py` (the `modal`
skill's pattern). What changes: app/volume `cm-mamba-contexts`, tag `m130c`; units are `bench` (synthetic, full N, no
model), `calib_<seed>` (the fixture calibration of SPEC 8, CPU, one per development and check seed), U1/U2 per
checkpoint and U3; the probe is copied by U1 (v1's U0 already established the tokenizer facts, and the probe sha is
asserted); U1 mounts v1's volume read-only in practice (never written) for C3; the guard reserves the units of the
launched stage (bench; calib; pilot = U1/U2 trained; full = U1/U2 init + U3), each at its own timeout (one Modal
function per unit and checkpoint), on top of everything already spent.
The ledger, cache keys and gate refusal are v1's, unchanged in logic.

    uv run --with modal python mc_modal.py deploy             # once per code change
    uv run --with modal python mc_modal.py budget [stage]     # print the reservation; spends nothing
    uv run --with modal python mc_modal.py launch calib       # fixture calibration on Modal CPU -> runs/calibration.json
    uv run --with modal python mc_modal.py launch bench       # synthetic full-size benchmark (+ full-size calibration)
    uv run --with modal python mc_modal.py status <stage>
    uv run --with modal python mc_modal.py pull <stage>       # calib/bench records -> runs/
    uv run --with modal python mc_modal.py timeouts           # runs/timeouts.json from the bench projection
    uv run --with modal python mc_modal.py launch pilot       # U1 + U2 on `trained`, then the gate     (Stage B)
    uv run --with modal python mc_modal.py launch full        # U1 + U2 on `init`, then U3             (Stage B)
    uv run --with modal python mc_modal.py fetch              # /analysis/m130c -> data/m130c

Volume `cm-mamba-contexts` (mounted at /vol): /vol/hf (HF caches), /vol/work/m130c/<checkpoint> (WORK files, see
mc_decode), /vol/work/m130c (attempts ledger, timings, records, coordinator log), /vol/analysis/m130c (public files).
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
MBDIR = HERE.parent / 'mamba_coherent'
LM = HERE.parent / 'lm_coherent'
CIFAR = HERE.parent / 'cifar_coherent'
sys.path.insert(0, str(HERE))
import mc_construction as K  # noqa: E402
import mc_ledger as LG  # noqa: E402

APP = 'cm-mamba-contexts'
VOLUME = 'cm-mamba-contexts'
V1_VOLUME = 'cm-mamba-coherent'
TAG = 'm130c'
CEILING_USD = 8.0
# modal.com/pricing, checked 2026-11-12 (L40S $0.000542/s; CPU $0.0000131 per physical core-second; memory
# $0.00000222 per GiB-second).
L40S_RATE, CPU_RATE, MEM_RATE = .000542, .0000131, .00000222
TRANSFORMERS, TORCH = '4.57.6', '2.15.0'
COORD_CORES, COORD_GIB, COORD_TIMEOUT = .25, 1, 6 * 3600
OVERHEAD_S = LG.OVERHEAD_S
# local disk: the default per-container quota is 512 GiB (modal.com/docs/guide/resources); no request needed
HW = {  # unit kind: (gpu, cores, GiB, default timeout s)
    'bench': ('L40S', 8, 64, 2400),
    'calib': (None, 4, 8, 2400),
    'diag': (None, 8, 16, 900),
    'U1_trained': ('L40S', 8, 64, 3600),
    'U2_trained': ('L40S', 8, 64, 3600),
    'U1_init': ('L40S', 8, 64, 1800),
    'U2_init': ('L40S', 8, 64, 1800),
    'U3': ('L40S', 8, 64, 3600),
}
_tfile = HERE / 'runs' / 'timeouts.json'
TIMEOUTS = {u: hw[3] for u, hw in HW.items()}
if _tfile.is_file():
    TIMEOUTS.update(json.loads(_tfile.read_text())['timeouts'])
elif 'MC_TIMEOUTS' in os.environ:
    TIMEOUTS.update(json.loads(os.environ['MC_TIMEOUTS']))
CALIB_SEEDS = K.DEV_SEEDS + K.CHECK_SEEDS


def rate(kind):
    gpu, cores, gib, _ = HW[kind]
    return (L40S_RATE if gpu else 0) + cores * CPU_RATE + gib * MEM_RATE


def units(stage=None):
    """(name, kind, arg) of every unit, or of those a stage's guard reserves."""
    pilot = [('U1_trained', 'U1_trained', None), ('U2_trained', 'U2_trained', None)]
    full = [('U1_init', 'U1_init', None), ('U2_init', 'U2_init', None), ('U3', 'U3', None)]
    if stage == 'pilot':
        return pilot
    if stage in ('full', 'u3'):
        return full
    run = pilot + full
    if stage == 'bench':
        return [('bench', 'bench', None)]
    if stage == 'calib':
        return [(f'calib_{s}', 'calib', s) for s in CALIB_SEEDS]
    if stage == 'diag':
        return [('diag', 'diag', None)]
    return [('bench', 'bench', None), ('diag', 'diag', None)] + [(f'calib_{s}', 'calib', s) for s in CALIB_SEEDS] + run


# torch path only: mamba-ssm, causal-conv1d and `kernels` are NOT installed (mb_model.assert_slow_path checks).
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install(f'torch=={TORCH}', 'numpy', 'scipy', 'scikit-learn', f'transformers=={TRANSFORMERS}',
                      'umap-learn', 'huggingface_hub', 'safetensors')
         .env({'PYTHONPATH': '/root/mc:/root/mb:/root/mb/diagnosis:/root/lm:/root/cc', 'OMP_NUM_THREADS': '8',
               'TOKENIZERS_PARALLELISM': 'false', 'NUMBA_NUM_THREADS': '1', 'MC_TIMEOUTS': json.dumps(TIMEOUTS)}))
for name in ('coherent.py', 'seba.py', 'schema_v2.py', 'construction.py', 'sweep_math.py'):
    image = image.add_local_file(str(CIFAR / name), remote_path='/root/cc/' + name)
image = image.add_local_file(str(CIFAR / 'site' / 'analysis.py'), remote_path='/root/cc/site/analysis.py')
for name in ('lm_construction.py', 'lm_cells.py', 'lm_tree.py', 'lm_schema.py'):
    image = image.add_local_file(str(LM / name), remote_path='/root/lm/' + name)
for name in ('mb_construction.py', 'mb_cells.py', 'mb_model.py'):
    image = image.add_local_file(str(MBDIR / name), remote_path='/root/mb/' + name)
image = image.add_local_file(str(MBDIR / 'diagnosis' / 'token_classes.py'),
                             remote_path='/root/mb/diagnosis/token_classes.py')
for name in ('mc_construction.py', 'mc_items.py', 'mc_stats.py', 'mc_decode.py', 'mc_schema.py', 'mc_calib.py',
             'mc_bench.py', 'mc_ledger.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/mc/' + name)
image = image.add_local_file(str(LM / 'data' / 'p160t' / 'probe.json'), remote_path='/root/mc/probe.json')
app = modal.App(APP)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
v1vol = modal.Volume.from_name(V1_VOLUME)
VOLS = {'/vol': vol}


def paths(tag=TAG):
    work, public = Path('/vol/work') / tag, Path('/vol/analysis') / tag
    work.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    return work, public


def run_key(work, public):
    probe = Path(public) / 'probe.json'
    return K.unit_key(K.probe_identity(probe.read_bytes()) if probe.is_file() else None, [])


def done(work, name, start, kind, key, **extra):
    seconds = time.time() - start
    row = LG.completion_row(name, seconds, seconds * rate(kind), start, time.time(), key, extra)
    K.write_json(work / f'timing_{name}.json', row)
    vol.commit()
    return row


def reused(row):
    return dict(row, cached=True)


def _gpu_setup():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(HW['U1_trained'][1])


def _probe(public):
    """The v1 probe, copied verbatim (sha256 asserted), then loaded."""
    import shutil
    import mc_schema as SC
    dest = Path(public) / 'probe.json'
    if not dest.is_file():
        assert K.probe_identity(Path('/root/mc/probe.json').read_bytes()) == K.PROBE_SHA
        shutil.copyfile('/root/mc/probe.json', dest)
        vol.commit()
    assert K.probe_identity(dest.read_bytes()) == K.PROBE_SHA, 'probe is not the v1 probe'
    return SC.load_probe(dest)


def u1_outputs(work, ck):
    w = work / ck
    names = ['items.npz', 'point_lag0.npy', 'whole_lag0.npy'] + [f'state_lag0_L{l}.npy' for l in K.LAYERS]
    if ck == K.FINAL:
        for tau in K.LAGS[1:]:
            names += [f'point_lag{tau}.npy', f'whole_lag{tau}.npy'] + [f'state_lag{tau}_L{l}.npy' for l in K.LAG_LAYERS]
        names += ['bc.npy', 'pred.npy', 'top.npz', 'bag.npy', 'recent.npy']
        names += [f'cons_{p}{m}.npy' for p in K.PARTS for m in K.LAYERS]
    return [w / n for n in names] + [w / f'u1_{ck}.json']


def u2_outputs(work, public, ck):
    out = [public / f'{k}_{ck}.{e}' for k, e in (('cells', 'json'), ('spectra', 'npz'), ('sets', 'json'))]
    out += [work / ck / f'vecs_{ck}.npz', work / ck / f'u2_{ck}.json']
    return out + ([public / 'lags.json'] if ck == K.FINAL else [])


# ------------------------------------------------------------------------------------------- bench, calib
@app.function(image=image, gpu='L40S', cpu=HW['bench'][1], memory=HW['bench'][2] * 1024,
              timeout=TIMEOUTS['bench'], volumes=VOLS)
def bench(tag: str = TAG):
    import mc_bench as B
    _gpu_setup()
    vol.reload(); work, _ = paths(tag); start = time.time()
    key = K.unit_key(None, [])
    cached = K.cached_unit(work / 'timing_bench.json', [work / 'bench.json'], key)
    if cached:
        return reused(cached)
    cal = work / 'calibration.json'
    theta = json.loads(cal.read_text())['theta'] if cal.is_file() else None
    out = B.run_bench('cuda', work / 'bench_io', theta=theta, log=lambda s: print(s, flush=True),
                      volume_commit=vol.commit, local='/tmp')
    out['theta'] = theta
    K.write_json(work / 'bench.json', out, indent=1); vol.commit()
    return done(work, 'bench', start, 'bench', key, theta=theta,
                calibrated=(out.get('calib_full') or {}).get('calibrated'))


@app.function(image=image, cpu=HW['calib'][1], memory=HW['calib'][2] * 1024, timeout=TIMEOUTS['calib'],
              volumes=VOLS)
def calib(seed: int, tag: str = TAG):
    import torch
    import mc_calib as CB
    torch.set_num_threads(HW['calib'][1])
    vol.reload(); work, _ = paths(tag); start = time.time(); name = f'calib_{seed}'
    key = K.unit_key(None, [])
    out_path = work / f'{name}.json'
    cached = K.cached_unit(work / f'timing_{name}.json', [out_path], key)
    if cached:
        return reused(cached)
    recs = CB.cases_for_seed(seed, log=lambda s: print(s, flush=True))
    K.write_json(out_path, recs); vol.commit()
    return done(work, name, start, 'calib', key)


@app.function(image=image, cpu=HW['diag'][1], memory=HW['diag'][2] * 1024, timeout=TIMEOUTS['diag'], volumes=VOLS)
def diag(tag: str = TAG):
    """Stage A diagnostic (CPU, synthetic): SEBA's cost when it runs to max_iter (structureless eigenvectors) at
    N = 8,000 for every r in R, with torch at 8 threads and at 1 thread, before and after a k-means call (thread
    contention between torch's and scikit-learn's OpenMP pools under the container's CPU quota)."""
    import numpy as np
    import torch
    import seba as S
    import mc_stats as ST
    vol.reload(); work, _ = paths(tag); start = time.time()
    rng = np.random.default_rng(0)
    v = np.linalg.qr(rng.normal(size=(K.N, K.N_EIG)))[0]
    vt = torch.from_numpy(v)
    out = dict(cpu_count=os.cpu_count(), rows=[])
    for threads, kmeans_first in ((8, False), (8, True), (1, True), (1, False)):
        torch.set_num_threads(threads)
        if kmeans_first:
            t0 = time.time(); ST.kmeans_partition(v, 16); tk = time.time() - t0
        else:
            tk = None
        t0 = time.time(); S.seba(vt, 16, max_iter=500); t16 = time.time() - t0
        out['rows'].append(dict(threads=threads, kmeans_first=kmeans_first, kmeans_s=tk, seba16_500it_s=t16))
        print(out['rows'][-1], flush=True)
    best = min(out['rows'], key=lambda r: r['seba16_500it_s'])
    torch.set_num_threads(best['threads'])
    per_r = {}
    for r in K.R:
        t0 = time.time(); S.seba(vt, r); per_r[r] = time.time() - t0
        print(r, per_r[r], flush=True)
    out.update(best_threads=best['threads'], seba_full_by_r=per_r, seba_full_total_s=sum(per_r.values()))
    K.write_json(work / 'diag.json', out, indent=1); vol.commit()
    return done(work, 'diag', start, 'diag', K.unit_key(None, []))


# ---------------------------------------------------------------------------------------------- U1, U2, U3
def _plain_errors(fn):
    """Re-raise any exception from a unit as a RuntimeError carrying its type and message, so the coordinator's log
    shows it (Modal cannot unpickle exceptions such as lm_cells.DegenerateItems whose __init__ takes extra
    arguments)."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as error:
            raise RuntimeError(f'{type(error).__name__}: {error}'[:2000]) from None
    return run


@_plain_errors
def _u1(checkpoint, tag):
    import numpy as np
    import mb_model as MM
    import mc_decode as D
    _gpu_setup()
    MM.assert_slow_path()
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U1_{checkpoint}'
    probe = _probe(public)
    key = dict(run_key(work, public), construction_fingerprint=K.DECODE_FINGERPRINT)     # U1 keyed on the decode
    cached = K.cached_unit(work / f'timing_{name}.json', u1_outputs(work, checkpoint), key)
    if cached:
        return reused(cached)
    model = MM.load_model(checkpoint, cache_dir='/vol/hf/mamba')
    vol.commit()
    v1_act = np.load('/v1/work/m130/act_trained.npy', mmap_mode='r') if checkpoint == K.FINAL else None
    info = D.run_u1(model, checkpoint, probe, work / checkpoint, 'cuda', log=lambda s: print(s, flush=True),
                    v1_act=v1_act)
    vol.commit()
    return done(work, name, start, name, key, timings=info['timings'], peak_gpu_gb=info.get('peak_gpu_gb'))


@_plain_errors
def _u2(checkpoint, tag):
    import mc_schema as SC
    _gpu_setup()
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U2_{checkpoint}'
    key = run_key(work, public)
    cached = K.cached_unit(work / f'timing_{name}.json', u2_outputs(work, public, checkpoint), key)
    if cached:
        return reused(cached)
    summary = SC.measure_checkpoint(work, public, checkpoint, SC.load_probe(public / 'probe.json'), 'cuda',
                                    log=lambda s: print(s, flush=True), spill_root='/tmp/spill')
    vol.commit()
    return done(work, name, start, name, key, **summary)


# One Modal function per (unit, checkpoint) so that each has its own timeout (the guard reserves timeout x rate).
_GPU = dict(image=image, gpu='L40S', cpu=8, memory=64 * 1024, max_containers=1)


@app.function(**_GPU, timeout=TIMEOUTS['U1_trained'], volumes={'/vol': vol, '/v1': v1vol})
def u1_trained(tag: str = TAG):
    return _u1('trained', tag)


@app.function(**_GPU, timeout=TIMEOUTS['U1_init'], volumes={'/vol': vol, '/v1': v1vol})
def u1_init(tag: str = TAG):
    return _u1('init', tag)


@app.function(**_GPU, timeout=TIMEOUTS['U2_trained'], volumes=VOLS)
def u2_trained(tag: str = TAG):
    return _u2('trained', tag)


@app.function(**_GPU, timeout=TIMEOUTS['U2_init'], volumes=VOLS)
def u2_init(tag: str = TAG):
    return _u2('init', tag)


@app.function(image=image, gpu='L40S', cpu=HW['U3'][1], memory=HW['U3'][2] * 1024, timeout=TIMEOUTS['U3'],
              volumes=VOLS)
def u3_finalize(tag: str = TAG):
    return _u3(tag)


@_plain_errors
def _u3(tag):
    import mc_schema as SC
    from transformers import AutoTokenizer
    _gpu_setup()
    vol.reload(); work, public = paths(tag); start = time.time()
    key = run_key(work, public)
    calibration = json.loads((work / 'calibration.json').read_text())
    b = work / 'bench.json'
    if b.is_file():
        calibration['full_size_check'] = json.loads(b.read_text()).get('calib_full')
    tok = AutoTokenizer.from_pretrained(K.MODEL, revision=K.REVISION, cache_dir='/vol/hf/tok')
    report = SC.finalize(work, public, SC.load_probe(public / 'probe.json'), calibration, device='cuda',
                         log=lambda s: print(s, flush=True), tokenizer=tok, spill_root='/tmp/spill')
    vol.commit()
    return done(work, 'U3', start, 'U3', key, validation=dict(complete=report['complete']))


# ------------------------------------------------------------------------------------------- budget/ledger
class Store:
    """Read access to the run's work directory: the mounted volume inside Modal, the volume API on the laptop."""

    def __init__(self, tag=TAG, mounted=False):
        self.mounted, self.tag = mounted, tag
        self.root = Path('/vol/work') / tag
        self._listing = {}

    def read(self, rel, base='work'):
        try:
            if self.mounted:
                return (Path('/vol') / base / self.tag / rel).read_bytes()
            return b''.join(vol.read_file(f'{base}/{self.tag}/{rel}'))
        except Exception:
            return None

    def exists(self, rel):
        """Whether a path (relative to the volume root) exists; directory listings are cached per call site."""
        if self.mounted:
            return (Path('/vol') / rel).exists()
        parent, name = rel.rsplit('/', 1)
        if parent not in self._listing:
            try:
                self._listing[parent] = {e.path.rsplit('/', 1)[-1] for e in vol.listdir(parent)}
            except Exception:
                self._listing[parent] = set()
        return name in self._listing[parent]

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
        return K.unit_key(K.probe_identity(probe) if probe is not None else None, [])


def coordinator_worst():
    return (COORD_TIMEOUT + OVERHEAD_S) * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE)


def outputs_of(name, tag=TAG):
    """The outputs each cached unit's function requires before it skips (the same lists the functions use),
    as paths relative to the volume root."""
    work, public = Path('/vol/work') / tag, Path('/vol/analysis') / tag
    if name == 'bench':
        paths_ = [work / 'bench.json']
    elif name.startswith('calib_'):
        paths_ = [work / f'{name}.json']
    elif name.startswith('U1_'):
        paths_ = u1_outputs(work, name[3:])
    elif name.startswith('U2_'):
        paths_ = u2_outputs(work, public, name[3:])
    else:
        paths_ = []
    return [str(p.relative_to('/vol')) for p in paths_] + [str((work / f'timing_{name}.json').relative_to('/vol'))]


def status_of(store, stage=None):
    """(key, worst case of every unit that will execute, units that will be skipped). See mc_ledger."""
    key = store.key()
    us = units(stage)
    complete = LG.complete_units(us, lambda n: store.json(f'timing_{n}.json'), key, outputs_of, store.exists)
    return key, LG.reservation(us, TIMEOUTS, rate, complete), complete


def guard(store, stage, log=print, extra=0.0):
    key, worst, complete = status_of(store, stage)
    b = K.budget(store.attempts(), worst, set(), extra)        # `worst` already excludes skipped units
    log(f"budget ({stage}): attempts ${b['spent']:.3f} + incomplete units ${b['reserve']:.3f} + coordinator "
        f"${b['extra']:.3f} = ${b['total']:.3f} (ceiling ${CEILING_USD})")
    if b['total'] > CEILING_USD:
        raise RuntimeError(f"reservation ${b['total']:.2f} exceeds ceiling ${CEILING_USD}; nothing dispatched")
    return b


def open_attempt(work, name, kind, reserved):
    folder = work / 'attempts'; folder.mkdir(exist_ok=True)
    path = folder / f'{name}-{time.time_ns()}.json'
    path.write_text(json.dumps(dict(unit=name, kind=kind, reserved_usd=reserved, actual_usd=None,
                                    dispatched_at=time.time(), fingerprint=K.FINGERPRINT)))
    vol.commit()
    return path


def close_attempt(path, actual, outcome):
    row = json.loads(path.read_text())
    row.update(actual_usd=actual, outcome=outcome, closed_at=time.time())
    path.write_text(json.dumps(row)); vol.commit()


def gate_record(work, public, key):
    """The pilot gate (SPEC 7) from the pilot's own outputs, stamped with the run key: C1a, C1b, C3, P3; Sinkhorn
    row error < 2e-12 in every cell; smallest off-diagonal cosine distance > 1e-9 in every view cell; no zero-norm
    item in any view or constituent; min PN(psi_1) >= 10 over blocks. A NaN in any compared number fails."""
    import math
    u1 = json.loads((work / K.FINAL / f'u1_{K.FINAL}.json').read_text())
    u2 = json.loads((work / K.FINAL / f'u2_{K.FINAL}.json').read_text())
    fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    checks = dict(C1a=bool(u1['C1a']['pass']), C1b=bool(u1['C1b']['pass']), C3=bool(u1['C3']['pass']),
                  P3=bool(u1['P3']['pass']),
                  sinkhorn=fin(u2['sinkhorn_max_row_err']) and u2['sinkhorn_max_row_err'] < K.SINKHORN_TOL,
                  duplicates=fin(u2['min_dist']) and u2['min_dist'] > K.DUPLICATE_TOL,
                  zero_norm=not u1['zero_norm'],
                  pn_psi1=fin(u2['min_pn_psi1_blocks']) and u2['min_pn_psi1_blocks'] >= K.PN_MIN)
    record = dict(**key, checks=checks, C1a=u1['C1a'], C1b=u1['C1b'], C3=u1['C3'], P3=u1['P3'], U2=u2,
                  zero_norm=u1['zero_norm'], **{'pass': all(checks.values())})
    K.write_json(work / 'pilot_gate.json', record); vol.commit()
    return record


def merge_calibration(work, log=print):
    import mc_calib as CB
    recs = []
    for s in CALIB_SEEDS:
        recs += json.loads((work / f'calib_{s}.json').read_text())
    cal = CB.calibrate(recs)
    dev = [r for r in recs if r['seed'] in K.DEV_SEEDS]
    cal.update(seeds=dict(development=K.DEV_SEEDS, check=K.CHECK_SEEDS), records=recs,
               development=dict(note='development seeds: reported, not used for theta',
                                planted_min=min(CB.planted_ups(r) for r in dev if r['s'] in K.PLANTED_S),
                                structureless_max=max(CB.max_ups(r) for r in dev if r['s'] == 0)))
    K.write_json(work / 'calibration.json', cal); vol.commit()
    log(f"calibration: theta {cal['theta']} ({cal['rule']}), P1' {cal['P1_prime']}, planted min "
        f"{cal['planted_min']:.2f}, structureless max {cal['structureless_max']:.2f}")
    return cal



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
        vol.reload()
        guard(store, stage, say)
        names = [kind if a is None else f'{kind}_{a}' for a in args]
        files = [open_attempt(work, n, kind, LG.worst_case(kind, TIMEOUTS, rate)) for n in names]
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

    me = open_attempt(work, f'coordinator_{stage}', 'COORD', coordinator_worst())
    outcome = 'failed'
    try:
        if stage == 'calib':
            batch('calib', calib, CALIB_SEEDS)
            vol.reload()
            merge_calibration(work, say)
        if stage == 'bench':
            batch('bench', bench, [None])
        if stage == 'diag':
            batch('diag', diag, [None])
        if stage == 'pilot':
            assert (work / 'calibration.json').is_file() and (work / 'bench.json').is_file(), 'calib and bench first'
            batch('U1_trained', u1_trained, [None])
            batch('U2_trained', u2_trained, [None])
            vol.reload()
            gate = gate_record(work, public, run_key(work, public))
            say(f"pilot gate: pass={gate['pass']} {json.dumps(gate['checks'])}")
        if stage in ('full', 'u3'):
            vol.reload()
            gate = json.loads((work / 'pilot_gate.json').read_text()) if (work / 'pilot_gate.json').is_file() else None
            if not K.gate_ok(gate, run_key(work, public)):
                raise RuntimeError('no passing pilot-gate record for this construction and probe')
        if stage == 'full':
            batch('U1_init', u1_init, [None])
            batch('U2_init', u2_init, [None])
        if stage in ('full', 'u3'):
            batch('U3', u3_finalize, [None])
        outcome = 'ok'
    finally:
        close_attempt(me, (time.time() - started) * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE), outcome)
        b = K.budget(store.attempts(), {}, set())
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
        stage = args[1] if len(args) > 1 else 'pilot'
        store = Store()
        key, worst, complete = status_of(store, stage)
        b = guard(store, stage, extra=coordinator_worst())
        print(json.dumps(dict(**b, ceiling=CEILING_USD, timeouts=TIMEOUTS, worst_by_unit=worst,
                              complete=sorted(complete), attempts=store.attempts()), indent=1))
    elif command == 'launch':
        stage = args[1]
        assert stage in ('calib', 'bench', 'diag', 'pilot', 'full', 'u3'), stage
        store = Store()
        guard(store, stage, extra=coordinator_worst())
        if stage in ('full', 'u3') and not K.gate_ok(store.json('pilot_gate.json'), store.key()):
            raise SystemExit('refused: no passing pilot-gate record for this construction and probe')
        call = modal.Function.from_name(APP, 'coordinate').spawn(stage, TAG)
        rec = dict(stage=stage, call_id=call.object_id, launched=time.ctime(), fingerprint=K.FINGERPRINT,
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
        log = Store().read('coordinator.log')
        print(log.decode()[-3000:] if log else 'no log yet')
    elif command == 'pull':
        stage = args[1]
        name = dict(calib='calibration.json', bench='bench.json', diag='diag.json', pilot='pilot_gate.json')[stage]
        data = Store().read(name)
        assert data is not None, f'{name} not on the volume'
        (rec_dir / name).write_bytes(data)
        print(f'runs/{name} written')
    elif command == 'timeouts':
        proj = json.loads((rec_dir / 'projection.json').read_text())
        _tfile.write_text(json.dumps(dict(timeouts=dict(TIMEOUTS, **proj['timeouts']), rule=proj['rule']), indent=1))
        print(json.dumps(proj['timeouts']), '-> redeploy before launching pilot')
    elif command == 'calibrate':
        # v1's rule after the pilot: timeouts from measured durations (2x, at least 900 s), scaled to the init units
        store = Store()
        u1 = store.json(f'{K.FINAL}/u1_{K.FINAL}.json')
        u2 = store.json(f'{K.FINAL}/u2_{K.FINAL}.json')
        assert u1 and u2, 'pilot incomplete'
        t = u1['timings']
        u1_init = t['pass_0'] + t['grams_0'] + 60
        u2_init = u2['seconds'] * (49 + 14) / (u2['n_cells'] + u2['n_objects'])
        timeouts = dict(TIMEOUTS, U1_init=int(max(900, 2 * u1_init)), U2_init=int(max(900, 2 * u2_init)))
        _tfile.write_text(json.dumps(dict(timeouts=timeouts, rule='after the pilot: U1_init = 2 x (pass_0 + grams_0 + '
                                          '60 s load), U2_init = 2 x U2_trained x 63/204 (cells + objects), at least '
                                          '900 s; U3 from the bench projection',
                                          pilot=dict(u1_timings=t, u2_seconds=u2['seconds'])), indent=1))
        print(json.dumps(timeouts), '-> redeploy before launching full')
    elif command == 'fetch':
        dest = HERE / 'data'; dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force', VOLUME, f'analysis/{TAG}',
                        str(dest)], check=True)
    else:
        raise SystemExit(__doc__)
