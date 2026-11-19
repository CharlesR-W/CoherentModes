"""Detached Modal coordinator for the LM toy (SPEC section 13; the `modal` skill's pattern).

    uv run --with modal python lm_modal.py deploy            # once per code change
    uv run --with modal python lm_modal.py budget [stage]    # print the reservation; spends nothing
    uv run --with modal python lm_modal.py launch pilot      # U0 + U1/U2 at step 143000 (the pilot gate)
    uv run --with modal python lm_modal.py calibrate         # timeouts := 2 x pilot durations -> runs/timeouts.json
    uv run --with modal python lm_modal.py launch full       # remaining U1/U2 in parallel, then U3
    uv run --with modal python lm_modal.py status [stage]
    uv run --with modal python lm_modal.py fetch             # /analysis/p160t -> data/p160t

Volume `cm-lm-coherent` (mounted at /vol): /vol/hf (HF cache, one directory per revision), /vol/work/p160t
(activations, slab Grams, operators, timings, coordinator log), /vol/analysis/p160t (the public files).

Budget (LC.budget): an append-only attempts ledger, work/p160t/attempts/<unit>-<ns>.json, one file per dispatched
attempt, written by the coordinator BEFORE dispatch at the unit's full worst case (timeout x rate). On return it is
reconciled: success -> the worker's timed cost + OVERHEAD_S; failure -> wall time since dispatch (capped at the
timeout) + OVERHEAD_S; no return (coordinator lost) -> stays at the full worst case. Every attempt counts, whatever
its fingerprint; the coordinator records itself the same way. Before every batch (and before `launch`), the
guard requires attempts + the worst case of every unit without a matching completion record (+ the coordinator's
own worst case, at launch) <= CEILING_USD.

Cache keys: every derived unit's completion record carries the construction fingerprint, the probe's sha256 and
the removal list (LC.unit_key); a record that differs in any of them is not reused.

Pilot gate: the pilot stage ends by writing work/p160t/pilot_gate.json (SPEC 8.1, stamped with the same key).
`launch full`/`launch u3` refuse locally, and the coordinator refuses remotely, unless that record passes and its
key matches the current construction, probe and removal list.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CIFAR = HERE.parent / 'cifar_coherent'
sys.path.insert(0, str(HERE))
import lm_construction as LC  # noqa: E402

APP = 'cm-lm-coherent'
TAG = 'p160t'
CEILING_USD = 6.0
# modal.com/pricing as recorded on 2026-11-12 (cifar_coherent/README.md); re-check before launch.
L40S_RATE, CPU_RATE, MEM_RATE = .000542, .0000131, .00000222
TRANSFORMERS = '4.57.6'
COORD_CORES, COORD_GIB, COORD_TIMEOUT = .25, 1, 4 * 3600
DIAG_TIMEOUT = 240
OVERHEAD_S = 90            # per attempt: container start and image load, not timed inside the function
MODEL_CONFIG_KEYS = ['model_type', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size',
                     'rotary_pct', 'rotary_emb_base', 'use_parallel_residual', 'vocab_size',
                     'max_position_embeddings', 'hidden_act', 'layer_norm_eps', 'tie_word_embeddings']
HW = {  # unit: (gpu, cores, GiB, default timeout s)
    'U0': (None, 4, 16, 1800),
    'U1': ('L40S', 4, 16, 900),
    'U2': (None, 8, 32, 2400),
    'U3': (None, 8, 32, 2400),
}
_tfile = HERE / 'runs' / 'timeouts.json'
TIMEOUTS = {u: hw[3] for u, hw in HW.items()}
if _tfile.is_file():
    TIMEOUTS.update(json.loads(_tfile.read_text())['timeouts'])
elif 'LM_TIMEOUTS' in __import__('os').environ:          # inside a container: the deployed values
    TIMEOUTS.update(json.loads(__import__('os').environ['LM_TIMEOUTS']))


def rate(unit):
    gpu, cores, gib, _ = HW[unit]
    return (L40S_RATE if gpu else 0) + cores * CPU_RATE + gib * MEM_RATE


def units():
    """Every unit of the run: (name, kind, step)."""
    out = [('U0', 'U0', None)]
    out += [(f'U1_{s}', 'U1', s) for s in LC.STEPS]
    out += [(f'U2_{s}', 'U2', s) for s in LC.STEPS]
    return out + [('U3', 'U3', None)]


image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.15.0', 'numpy', 'scipy', f'transformers=={TRANSFORMERS}', 'accelerate',
                      'umap-learn', 'zstandard', 'huggingface_hub', 'pyarrow')
         .env({'PYTHONPATH': '/root/lm:/root/cc', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'LM_TIMEOUTS': json.dumps(TIMEOUTS)}))
for name in ('coherent.py', 'seba.py', 'schema_v2.py', 'construction.py', 'sweep_math.py'):
    image = image.add_local_file(str(CIFAR / name), remote_path='/root/cc/' + name)
image = image.add_local_file(str(CIFAR / 'site' / 'analysis.py'), remote_path='/root/cc/site/analysis.py')
for name in ('lm_construction.py', 'lm_cells.py', 'lm_tree.py', 'lm_schema.py', 'probe.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/lm/' + name)
app = modal.App(APP)
vol = modal.Volume.from_name('cm-lm-coherent', create_if_missing=True)
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
    """Construction + probe identity + removal list: the cache key of every derived unit (U1, U2) and the gate."""
    probe = Path(public) / 'probe.json'
    sha = LC.probe_identity(probe.read_bytes()) if probe.is_file() else None
    return LC.unit_key(sha, removed_items(work))


def done(work, name, start, unit, key, **extra):
    seconds = time.time() - start
    row = dict(unit=name, seconds=seconds, estimated_usd=seconds * rate(unit), started_at=start,
               completed_at=time.time(), **key, **extra)
    (work / f'timing_{name}.json').write_text(json.dumps(row, allow_nan=False))
    vol.commit()
    return row


def reused(row):
    return dict(row, cached=True)


# ---------------------------------------------------------------------------------------------------- U0
@app.function(image=image, cpu=HW['U0'][1], memory=HW['U0'][2] * 1024, timeout=TIMEOUTS['U0'], volumes=VOLS)
def u0_probe(tag: str = TAG):
    import probe
    vol.reload(); work, public = paths(tag); start = time.time()
    key = LC.unit_key(None, [])                    # the probe does not depend on the probe or on removals
    cached = LC.cached_unit(work / 'timing_U0.json', [public / 'probe.json'], key)
    if cached:
        return reused(cached)
    summary = probe.main(str(public / 'probe.json'), '/vol/hf/datasets')
    return done(work, 'U0', start, 'U0', key, **summary)


# ---------------------------------------------------------------------------------------------------- U1
@app.function(image=image, gpu='L40S', cpu=HW['U1'][1], memory=HW['U1'][2] * 1024, timeout=TIMEOUTS['U1'],
              volumes=VOLS, max_containers=5)
def u1_forward(step: int, tag: str = TAG):
    """Forward passes, point activations, slab Grams (GPU float64), bag vectors, C1, C2, zero-norm scan."""
    import numpy as np
    import torch
    import transformers
    import lm_cells as C
    import lm_schema as S
    from transformers import AutoModelForCausalLM
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U1_{step}'
    removed = removed_items(work)
    key = run_key(work, public)
    outputs = [work / f'{k}_{step}.npy' for k in ('act', 'slab', 'bag')] + [work / f'u1_{step}.json']
    cached = LC.cached_unit(work / f'timing_{name}.json', outputs, key)
    if cached:
        return reused(cached)
    probe = S.Probe(public / 'probe.json', removed)
    raw = probe.raw
    tokens = np.asarray(probe.tokens)
    tokens_s = S.shuffled_tokens(raw)[probe.items]
    model = AutoModelForCausalLM.from_pretrained(LC.MODEL, revision=f'step{step}', cache_dir=f'/vol/hf/step{step}',
                                                 torch_dtype=torch.float32, attn_implementation='eager')
    model.eval().cuda()
    vol.commit()
    blocks = model.gpt_neox.layers

    def forward(tok, positions=None):
        """Residuals after the sampled layers, [layer][n, positions, D] on the GPU (float32)."""
        seq = torch.cat([torch.full((len(tok), 1), LC.EOT), torch.as_tensor(tok)], 1)
        grabbed = {l: [] for l in LC.LAYERS}
        hooks = [blocks[l - 1].register_forward_hook(
            lambda m, i, o, l=l: grabbed[l].append(o[0] if isinstance(o, tuple) else o)) for l in LC.LAYERS]
        try:
            with torch.inference_mode():
                for at in range(0, len(seq), 50):
                    model(seq[at:at + 50].cuda(), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        return {l: torch.cat(grabbed[l]) for l in LC.LAYERS}

    hid = forward(tokens)
    n = probe.n
    zero = sorted({int(i) for l in LC.LAYERS for u in range(1, LC.SEQ + 1) for i in C.zero_norm_items(hid[l][:, u])})
    cfg = model.config
    info = dict(step=step, transformers_version=transformers.__version__, zero_norm=zero, removed=removed,
                layout='AlignedUMAP', construction_fingerprint=LC.FINGERPRINT,
                model_config={k: getattr(cfg, k, None) for k in MODEL_CONFIG_KEYS})
    if zero:
        (work / f'u1_{step}.json').write_text(json.dumps(info)); vol.commit()
        raise RuntimeError(f'zero-norm items {zero}: coordinator must record them in removed.json and rerun U1')
    act = np.empty((len(LC.LAYERS), len(LC.POSITIONS), n, hid[LC.LAYERS[0]].shape[-1]), np.float32)
    slab = np.empty((len(LC.LAYERS), len(LC.POSITIONS), n, n))
    norm = np.empty((len(LC.LAYERS), LC.SEQ + 1))
    for li, l in enumerate(LC.LAYERS):
        h = hid[l]
        norm[li] = h.norm(dim=-1).mean(0).double().cpu().numpy()
        acc = C.SlabAccumulator(n, device='cuda')
        for u in range(1, LC.SEQ + 1):
            acc.add(h[:, u])
            if u in LC.POSITIONS:
                pi = LC.POSITIONS.index(u)
                act[li, pi] = h[:, u].cpu().numpy()
                slab[li, pi] = acc.gram().cpu().numpy()
    np.save(work / f'act_{step}.npy', act)
    np.save(work / f'slab_{step}.npy', slab)
    emb = model.gpt_neox.embed_in.weight.detach().float().cpu().numpy()
    bag = C.bag_vectors(tokens, emb)
    bag_s = C.bag_vectors(tokens_s, emb, [LC.SEQ])
    info['C2'] = dict(identical=bool(np.array_equal(bag[-1], bag_s[0])),
                      max_abs_diff=float(np.abs(bag[-1] - bag_s[0]).max()))
    np.save(work / f'bag_{step}.npy', bag)
    if step == LC.FINAL:
        info['norm_mean'] = norm.tolist()
        hs = forward(tokens_s)
        np.save(work / f'actS_{step}.npy',
                np.stack([hs[l][:, LC.SEQ].cpu().numpy() for l in LC.LAYERS]).astype(np.float32))
        # C1 (SPEC 8, decision of 2026-11-12): gate on the float64 model, truncated vs full pass, tolerance 1e-10.
        # The float32 discrepancy (rounding amplified through depth) is reported per layer as a hazard, not gated.
        def rel_errors(full, short):
            return {l: max(float(((short[l][:, t].double() - full[l][:, t].double()).norm(dim=1)
                                  / full[l][:, t].double().norm(dim=1)).max()) for t in LC.POSITIONS if t <= 64)
                    for l in LC.LAYERS}
        f32 = rel_errors(hid, forward(tokens[:, :64]))
        del hs
        model.double()
        f64 = rel_errors(forward(tokens), forward(tokens[:, :64]))
        model.float()
        rel = max(f64.values())
        info['C1'] = dict(max_rel_err=rel, tolerance=1e-10, precision='float64', **{'pass': rel < 1e-10},
                          float64_by_layer={f'L{l}': v for l, v in f64.items()},
                          float32_max_rel_err=max(f32.values()),
                          float32_by_layer={f'L{l}': v for l, v in f32.items()},
                          definition='max over items and sampled t <= 64 of |h_trunc - h_full| / |h_full|; '
                                     'gate: float64 model; float32 reported as a hazard')
    (work / f'u1_{step}.json').write_text(json.dumps(info, allow_nan=False)); vol.commit()
    assert info['C2']['identical'], 'C2 failed'
    assert step != LC.FINAL or info['C1']['pass'], f"C1 failed: {info['C1']['max_rel_err']}"
    return done(work, name, start, 'U1', key)


# ------------------------------------------------------------------------------------------- diagnostics
@app.function(image=image, gpu='L40S', cpu=HW['U1'][1], memory=HW['U1'][2] * 1024, timeout=DIAG_TIMEOUT, volumes=VOLS)
def diag_c1(step: int = LC.FINAL, n: int = 100, tag: str = TAG):
    """C1 diagnosis on n probe items: truncated-vs-full relative error per (layer, t) under (a) the U1 settings,
    (b) IEEE fp32 forced through every precision switch, (c) the model in float64. Spends; ledgered by the CLI."""
    import numpy as np
    import torch
    import lm_schema as S
    from transformers import AutoModelForCausalLM
    vol.reload(); work, public = paths(tag)
    tokens = np.asarray(S.Probe(public / 'probe.json', removed_items(work)).tokens)[:n]
    model = AutoModelForCausalLM.from_pretrained(LC.MODEL, revision=f'step{step}', cache_dir=f'/vol/hf/step{step}',
                                                 torch_dtype=torch.float32, attn_implementation='eager').eval().cuda()
    def flags():
        out = dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32, matmul_precision=torch.get_float32_matmul_precision())
        for label, mod in (('cuda.matmul', torch.backends.cuda.matmul), ('cudnn', torch.backends.cudnn),
                           ('backends', torch.backends)):
            try:
                out[f'{label}.fp32_precision'] = str(getattr(mod, 'fp32_precision'))
            except Exception as error:
                out[f'{label}.fp32_precision'] = f'n/a ({type(error).__name__})'
        return out
    def run(tok, dtype):
        seq = torch.cat([torch.full((len(tok), 1), LC.EOT), torch.as_tensor(tok)], 1)
        grabbed = {l: [] for l in LC.LAYERS}
        hooks = [model.gpt_neox.layers[l - 1].register_forward_hook(
            lambda m, i, o, l=l: grabbed[l].append((o[0] if isinstance(o, tuple) else o).to(dtype))) for l in LC.LAYERS]
        with torch.inference_mode():
            for at in range(0, len(seq), 50):
                model(seq[at:at + 50].cuda(), use_cache=False)
        for h in hooks:
            h.remove()
        return {l: torch.cat(grabbed[l]) for l in LC.LAYERS}
    def errors(dtype):
        full, short = run(tokens, dtype), run(tokens[:, :64], dtype)
        table = {}
        for l in LC.LAYERS:
            for t in [p for p in LC.POSITIONS if p <= 64]:
                a, b = short[l][:, t].double(), full[l][:, t].double()
                r = ((a - b).norm(dim=1) / b.norm(dim=1)).cpu().numpy()
                table[f'L{l}_t{t}'] = dict(max=float(r.max()), median=float(np.median(r)), argmax=int(r.argmax()),
                                          norm=float(b.norm(dim=1).median()))
        worst = max(table.values(), key=lambda v: v['max'])
        return dict(flags=flags(), max=worst['max'], table=table)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = dict(u1_settings=errors(torch.float32))
    torch.set_float32_matmul_precision('highest')
    for mod in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(mod, 'fp32_precision'):
            try:
                mod.fp32_precision = 'ieee'
            except Exception as error:
                out.setdefault('notes', []).append(repr(error))
    out['ieee_forced'] = errors(torch.float32)
    model.double()
    out['float64'] = errors(torch.float64)
    return out


# ---------------------------------------------------------------------------------------------------- U2
@app.function(image=image, cpu=HW['U2'][1], memory=HW['U2'][2] * 1024, timeout=TIMEOUTS['U2'], volumes=VOLS,
              max_containers=5)
def u2_measure(step: int, tag: str = TAG):
    import lm_schema as S
    import torch
    torch.set_num_threads(HW['U2'][1])
    vol.reload(); work, public = paths(tag); start = time.time(); name = f'U2_{step}'
    key = run_key(work, public)
    required = [public / f'{k}_{step}.npz' for k in ('cells', 'single', 'blocks')] + \
               [work / f'{k}_{step}.{e}' for k, e in (('op128', 'npy'), ('compare', 'json'), ('trees', 'json'),
                                                       ('p3', 'json'))]
    cached = LC.cached_unit(work / f'timing_{name}.json', required, key)
    if cached:
        return reused(cached)
    result = S.measure_step(work, public, step, S.Probe(public / 'probe.json', removed_items(work)))
    return done(work, name, start, 'U2', key, sinkhorn_max_row_err=result['sinkhorn_max_row_err'])


# ---------------------------------------------------------------------------------------------------- U3
@app.function(image=image, cpu=HW['U3'][1], memory=HW['U3'][2] * 1024, timeout=TIMEOUTS['U3'], volumes=VOLS)
def u3_finalize(tag: str = TAG):
    import lm_schema as S
    import torch
    torch.set_num_threads(HW['U3'][1])
    vol.reload(); work, public = paths(tag); start = time.time()
    key = run_key(work, public)
    model_config = json.loads((work / f'u1_{LC.FINAL}.json').read_text()).get('model_config')
    report = S.finalize(work, public, S.Probe(public / 'probe.json', removed_items(work)),
                        extra_meta=dict(model_config=model_config, probe_sha=key['probe_sha']))
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
        return LC.unit_key(LC.probe_identity(probe) if probe is not None else None, removed)


def coordinator_worst():
    return COORD_TIMEOUT * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE)


def status_of(store):
    key = store.key()
    worst = {name: TIMEOUTS[kind] * rate(kind) for name, kind, _ in units()}
    complete = set()
    for name, kind, _ in units():
        row = store.json(f'timing_{name}.json')
        if LC.key_matches(row, LC.unit_key(None, []) if kind == 'U0' else key):
            complete.add(name)
    return key, worst, complete


def guard(store, log=print, extra=0.0):
    key, worst, complete = status_of(store)
    b = LC.budget(store.attempts(), worst, complete, extra)
    log(f"budget: attempts ${b['spent']:.3f} + incomplete units ${b['reserve']:.3f} + coordinator "
        f"${b['extra']:.3f} = ${b['total']:.3f} (ceiling ${CEILING_USD})")
    if b['total'] > CEILING_USD:
        raise RuntimeError(f"reservation ${b['total']:.2f} exceeds ceiling ${CEILING_USD}; nothing dispatched")
    return b


def open_attempt(work, name, kind, reserved):
    folder = work / 'attempts'; folder.mkdir(exist_ok=True)
    path = folder / f'{name}-{time.time_ns()}.json'
    path.write_text(json.dumps(dict(unit=name, kind=kind, reserved_usd=reserved, actual_usd=None,
                                    dispatched_at=time.time(), fingerprint=LC.FINGERPRINT)))
    vol.commit()
    return path


def close_attempt(path, actual, outcome):
    row = json.loads(path.read_text())
    row.update(actual_usd=actual, outcome=outcome, closed_at=time.time())
    path.write_text(json.dumps(row)); vol.commit()


def gate_record(work, public, key):
    """SPEC 8.1 from the pilot's own outputs (C1, C2, the real-data P3 check, Sinkhorn, duplicates, zero-norm,
    PN(psi_1) >= 10 in every point cell), stamped with the run key."""
    import numpy as np
    u1 = json.loads((work / f'u1_{LC.FINAL}.json').read_text())
    p3 = json.loads((work / f'p3_{LC.FINAL}.json').read_text())
    with np.load(public / f'cells_{LC.FINAL}.npz') as z:
        pn1 = float(z['pn'][0, ..., 0].min())
        row_err = float(max(z['row_err'].max(), z['bag_row_err'].max()))
        min_dist = float(min(z['min_dist'].min(), z['bag_min_dist'].min()))
    checks = dict(C1=bool(u1['C1']['pass']), C2=bool(u1['C2']['identical']), P3=bool(p3['pass']),
                  sinkhorn=row_err < 2e-12, duplicates=min_dist > 1e-9, zero_norm=not u1['zero_norm'],
                  pn_psi1=pn1 >= 10)
    record = dict(**key, checks=checks, C1=u1['C1'], C2=u1['C2'], P3=p3, min_pn_psi1_point=pn1,
                  max_row_err=row_err, min_dist=min_dist, zero_norm=u1['zero_norm'],
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

    def batch(kind, fn, steps):
        """Guard, record an attempt per unit (full worst case), dispatch, reconcile what returns."""
        vol.reload()
        guard(store, say)
        names = [kind if s is None else f'{kind}_{s}' for s in steps]
        files = [open_attempt(work, n, kind, TIMEOUTS[kind] * rate(kind)) for n in names]
        t0 = time.time()
        if steps == [None]:
            try:
                results = [fn.remote(tag)]
            except Exception as error:
                results = [error]
        else:
            results = list(fn.map(steps, kwargs={'tag': tag}, return_exceptions=True))
        failed = []
        for name, path, res in zip(names, files, results):
            if isinstance(res, BaseException):
                # billed time <= wall time since dispatch, and <= the timeout: a bounded, known outcome
                bound = (min(time.time() - t0, TIMEOUTS[kind]) + OVERHEAD_S) * rate(kind)
                close_attempt(path, bound, f'failed: {res!r}'[:500]); failed.append((name, repr(res)[:300]))
            else:
                worker = 0.0 if res.get('cached') else res['estimated_usd']
                close_attempt(path, worker + OVERHEAD_S * rate(kind), 'cached' if res.get('cached') else 'ok')
        say(f'{kind} {steps}: {len(results) - len(failed)} ok, failed {failed}')
        if failed:
            raise RuntimeError(f'{kind} failures: {failed}')

    me = open_attempt(work, f'coordinator_{stage}', 'COORD', coordinator_worst())
    outcome = 'failed'
    try:
        if stage in ('pilot', 'probe'):
            batch('U0', u0_probe, [None])
        if stage == 'pilot':
            batch('U1', u1_forward, [LC.FINAL])
            batch('U2', u2_measure, [LC.FINAL])
            vol.reload()
            gate = gate_record(work, public, run_key(work, public))
            say(f"pilot gate: pass={gate['pass']} {json.dumps(gate['checks'])}")
        if stage in ('full', 'u3'):
            vol.reload()
            gate = json.loads((work / 'pilot_gate.json').read_text()) if (work / 'pilot_gate.json').is_file() else None
            if not LC.gate_ok(gate, run_key(work, public)):
                raise RuntimeError('no passing pilot-gate record for this construction, probe and removal list')
        if stage == 'full':
            rest = [s for s in LC.STEPS if s != LC.FINAL]
            try:
                batch('U1', u1_forward, rest)
            finally:
                zero = sorted({i for s in LC.STEPS for i in (store.json(f'u1_{s}.json') or {}).get('zero_norm', [])})
                if zero:
                    (work / 'removed.json').write_text(json.dumps(dict(items=zero, reason='zero centred norm')))
                    vol.commit()
                    say(f'zero-norm items {zero} recorded in removed.json: the pilot must be rerun')
            batch('U2', u2_measure, rest)
        if stage in ('full', 'u3'):
            batch('U3', u3_finalize, [None])
        outcome = 'ok'
    finally:
        close_attempt(me, (time.time() - started) * (COORD_CORES * CPU_RATE + COORD_GIB * MEM_RATE), outcome)
        b = LC.budget(store.attempts(), *status_of(store)[1:])
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
        print(json.dumps(dict(**b, ceiling=CEILING_USD, timeouts=TIMEOUTS, complete=sorted(complete),
                              attempts=store.attempts()), indent=1))
    elif command == 'launch':
        stage = args[1]
        assert stage in ('probe', 'pilot', 'full', 'u3'), stage
        store = Store()
        guard(store, extra=coordinator_worst())
        if stage in ('full', 'u3') and not LC.gate_ok(store.json('pilot_gate.json'), store.key()):
            raise SystemExit('refused: no passing pilot-gate record for this construction, probe and removals')
        call = modal.Function.from_name(APP, 'coordinate').spawn(stage, TAG)
        rec = dict(stage=stage, call_id=call.object_id, launched=time.ctime(), fingerprint=LC.FINGERPRINT,
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
        if stage == 'pilot':
            print(json.dumps(Store().json('pilot_gate.json'), indent=1))
    elif command == 'calibrate':
        store = Store()
        secs = {k: (store.json(f'timing_{u}.json') or {}).get('seconds') for k, u in
                (('U0', 'U0'), ('U1', f'U1_{LC.FINAL}'), ('U2', f'U2_{LC.FINAL}'))}
        assert all(secs.values()), f'pilot incomplete: {secs}'
        timeouts = {k: int(max(120, 2 * v)) for k, v in secs.items()}
        timeouts['U3'] = TIMEOUTS['U3']
        _tfile.write_text(json.dumps(dict(timeouts=timeouts, pilot_seconds=secs, rule='2 x pilot duration'),
                                     indent=1))
        print(json.dumps(timeouts), '-> redeploy before launching full')
    elif command == 'diag':
        import tempfile
        store = Store()
        reserved = DIAG_TIMEOUT * rate('U1')
        guard(store, extra=reserved)
        name = f'diag_c1-{time.time_ns()}.json'
        row = dict(unit='diag_c1', kind='diag', reserved_usd=reserved, actual_usd=None, dispatched_at=time.time())
        def upload(r):
            with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                json.dump(r, f)
            with vol.batch_upload(force=True) as b:
                b.put_file(f.name, f'work/{TAG}/attempts/{name}')
        upload(row)
        t0 = time.time()
        try:
            result = modal.Function.from_name(APP, 'diag_c1').remote()
            outcome = 'ok'
        except Exception as error:
            result, outcome = dict(error=repr(error)), 'failed'
        row.update(actual_usd=(min(time.time() - t0, DIAG_TIMEOUT) + OVERHEAD_S) * rate('U1'), outcome=outcome)
        upload(row)
        (rec_dir / 'diag_c1.json').write_text(json.dumps(result, indent=1))
        print(json.dumps({k: (v['max'], v['flags']) if isinstance(v, dict) and 'max' in v else v
                          for k, v in result.items()}, indent=1))
    elif command == 'fetch':
        dest = HERE / 'data'; dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force', 'cm-lm-coherent',
                        f'analysis/{TAG}', str(dest)], check=True)
    else:
        raise SystemExit(__doc__)
