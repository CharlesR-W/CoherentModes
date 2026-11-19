"""Baselines / nulls measurement on Modal (detached coordinator; the `modal` skill).

Commands (run from this directory with the cifar_coherent environment):
    python measure_b.py deploy
    python measure_b.py launch <stage> [cap_usd] [pause_usd]   stages: smoke | extract | sources | reps | assemble | all
    python measure_b.py status <stage>
    python measure_b.py fetch                       analysis/r20s0c -> ../cifar_coherent/data/r20s0c
    python measure_b.py fixture-local               (N = 2000; not for this VPS)

Inputs (volume cm-cifar-coherent, read only here): /work/r20s0b/gram_<step>_<si>.npy (stored centred
cosine Grams), /work/r20s0b/operators_<step>_<si>.npy (5 rules x N x N; rule 0 = default),
/analysis/r20s0b/{layer_<step>,time_<site>}.npz (stored M0), probe.npz, points_<step>.npz,
checkpoints /runs/<run>/checkpoints/step_<step>.pt.
Outputs: /work/r20s0c (raw Grams, per-unit parts, timing_*.json spend records) and
/analysis/r20s0c (the public files described in SCHEMA.md).

Units (each cached by its timing record + outputs, so a failed unit reruns alone):
  extract_raw(step)   GPU   raw uncentred Grams R = X X^T (float64) for 12 sites, flat and GAP
                            views; asserted against the stored cosine Grams; pixel cell at step 0
  seed1()             GPU   seed-1 final checkpoint: R, default operators, all 78 layer blocks (rule 0)
  extract_10k()       GPU   all 10,000 CIFAR-10 test images, final checkpoint, R float32, 12 sites
  pixels()            CPU   M0 and L1 on the pixel cell (I1)
  sources_unit(f, i)  CPU   cheap sources + one N3 draw + agreement numbers for one layer step / time site
  reps_unit(b, task)  CPU   nulls / stability / ablations / linear content on one representative block
  n6_unit(r)          CPU   SEBA on 1,000 Haar subspaces at rank r
  timing()            CPU   per-source cost on 5 blocks (printed and saved)
  fixture_check()     CPU   M0 reproduces data/_fixture labels (regenerated synthetic activations)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CC = HERE.parent / 'cifar_coherent'
APP = 'cm-cifar-baselines-a'
TAG_IN = 'r20s0b'
TAG = 'r20s0c'
RUN = 'resnet20-relu-seed0'
RUN1 = 'resnet20-relu-seed1'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
FINAL = 64000
N = 2000
RANKS = [2, 3, 4, 6, 8]
N_PERM = 1000

# Same base layers as measure_v2 (so Modal reuses the cached torch build that produced the
# stored Grams); scikit-learn is a separate, later layer.
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy', 'umap-learn')
         .pip_install('scikit-learn')
         .env({'PYTHONPATH': '/root/src:/root/cc:/root/cb', 'OMP_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src'))
for name in ('coherent.py', 'seba.py', 'schema_v2.py', 'sweep_math.py', 'construction.py'):
    image = image.add_local_file(str(CC / name), remote_path='/root/cc/' + name)
for name in ('layer_0.npz', 'umap_0.npy'):
    image = image.add_local_file(str(CC / 'data/_fixture' / name), remote_path='/root/fixture/' + name)
for name in ('stats.py', 'sources.py', 'nulls.py', 'variants.py', 'measure_b.py', 'NOTES_A.md'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/cb/' + name)

app = modal.App(APP)
vol = modal.Volume.from_name('cm-cifar-coherent')
zoo = modal.Volume.from_name('crw-modelzoo-train').read_only()
VOLS = {'/out': vol, '/zoo': zoo}
# https://modal.com/pricing, checked 2026-11-15 (unchanged from measure_v2's 2026-11-12 values).
# Conservative: each requested CPU unit is charged as a full physical core; cold starts excluded.
GPU_RATE = .000583   # A100-40GB
CPU_RATE = .0000131
MEM_RATE = .00000222
CPU_UNIT = dict(cpu=4, memory=16384)

# Representative blocks, fixed before looking at class outcomes (PLAN section 3).
# (id, family, step-or-site, a, b): layer blocks are sites a..b at a step; time blocks are
# STEPS[a..b] at a site.
REPS = [('cell_stem', 'layer', FINAL, 0, 0), ('cell_b5', 'layer', FINAL, 5, 5),
        ('cell_b9', 'layer', FINAL, 9, 9), ('cell_logits', 'layer', FINAL, 11, 11),
        ('stem-b3', 'layer', FINAL, 0, 3), ('b4-b6', 'layer', FINAL, 4, 6),
        ('b7-b9', 'layer', FINAL, 7, 9), ('stem-logits', 'layer', FINAL, 0, 11),
        ('b4-b6@0', 'layer', 0, 4, 6), ('b4-b6@1024', 'layer', 1024, 4, 6),
        ('time-b5', 'time', 'b5', 3, 7), ('time-b9', 'time', 'b9', 3, 7)]
TIMING_BLOCKS = [('cell_b5', FINAL, 5, 5), ('b4-b6', FINAL, 4, 6), ('stem-b3', FINAL, 0, 3),
                 ('b7-b9', FINAL, 7, 9), ('stem-logits', FINAL, 0, 11)]
# Scope tiers for the explorer-grid sources (PLAN section 2), in order.
TIERS = [[('layer', STEPS.index(FINAL))],
         [('time', SITES.index(s)) for s in ('b5', 'b9', 'logits')],
         [('layer', STEPS.index(s)) for s in (0, 1024, 6400)],
         [('layer', i) for i, s in enumerate(STEPS) if s not in (0, 1024, 6400, FINAL)]
         + [('time', i) for i, s in enumerate(SITES) if s not in ('b5', 'b9', 'logits')]]
SCHEMA_VERSION = '1.1'   # 1.1 (2026-11-15): additive null references, per-level S1 / N3 excess, conventions
PART_VERSION = '_v2'     # parts recomputed for schema 1.1 (stability, linear, ablate, s1_cells, n10k)
REPS_V2 = ('stability', 'linear', 'ablate')


def part_name(rid, task):
    return f'rep_{rid}_{task}' + (PART_VERSION if task in REPS_V2 else '')


# Projected cost per coordinator step (USD), from the smoke's timing.json scaled by unit counts,
# with about 50% margin; used only to refuse a step that would pass the hard cap.
PROJECTED = {'extract': 0.8, 'sources_tier0': 0.4, 'sources_tier1': 0.3, 'sources_tier2': 1.0,
             'sources_tier3': 1.8, 'reps': 3.5, 'followup': 1.5}
AGREE_FIELDS = ['ami_all', 'ami_joint', 'joint_coverage', 'ari_all', 'mj_max', 'mj_legacy', 'mj_weighted',
                'unassigned', 'ami_null_q95', 'mj_null_mean', 'mj_null_q95', 'overlap_mean_cos2',
                'overlap_min_cos', 'class_ami', 'class_ami_null_q95', 'class_mj_max', 'class_mj_null_mean',
                'class_mj_null_q95', 'ami_joint_null_q95', 'class_ami_joint', 'class_ami_joint_null_q95']
# max-over-blocks null bands (family = all blocks of one file, fixed source and rank)
MAX_FIELDS = ['ami_all', 'ami_joint', 'mj_max', 'class_ami', 'class_ami_joint', 'class_mj_max']
N_CLASSES = 10

def pairs(n):
    return [(a, b) for a in range(n) for b in range(a, n)]


# ---------------------------------------------------------------- volume helpers

def paths():
    pub = Path('/out/analysis') / TAG; work = Path('/out/work') / TAG
    pub.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    return pub, work, Path('/out/analysis') / TAG_IN, Path('/out/work') / TAG_IN


def finished(work, name, start, cpu=4, mem=16, gpu=False, **extra):
    seconds = time.time() - start
    row = dict(unit=name, seconds=seconds, gpu=gpu, cpu=cpu, mem_gib=mem,
               estimated_usd=seconds * ((GPU_RATE if gpu else 0) + cpu * CPU_RATE + mem * MEM_RATE),
               started_at=start, completed_at=time.time(), **extra)
    (work / f'timing_{name}.json').write_text(_dumps(row))
    vol.commit()
    return row


def cached(work, name, outputs=()):
    p = work / f'timing_{name}.json'
    if p.is_file() and all(Path(o).is_file() for o in outputs):
        return json.loads(p.read_text())
    return None


def load_gram(step, si):
    import numpy as np
    return np.load(Path('/out/work') / TAG_IN / f'gram_{step}_{si}.npy')


def load_op(step, si, mode=0):
    import numpy as np
    return np.array(np.load(Path('/out/work') / TAG_IN / f'operators_{step}_{si}.npy', mmap_mode='r')[mode])


def load_raw(step, si, view='flat', run='s0'):
    import numpy as np
    pre = '' if run == 's0' else f'{run}_'
    return np.load(Path('/out/work') / TAG / f'{pre}raw_{view}_{step}_{si}.npy')


def block_cells(family, key, a, b):
    """(step, site index) of each cell of a block, in block order."""
    if family == 'layer':
        return [(key, s) for s in range(a, b + 1)]
    si = SITES.index(key) if isinstance(key, str) else key
    return [(STEPS[t], si) for t in range(a, b + 1)]


def stored_m0(family, key):
    """Stored rule-0 M0 for every block of one file: labels (P,5,N), vecs (P,N,9) with the
    constant, vals (P,9)."""
    import numpy as np
    name = f'layer_{key}' if family == 'layer' else f'time_{SITES[key] if isinstance(key, int) else key}'
    with np.load(Path('/out/analysis') / TAG_IN / f'{name}.npz') as z:
        lab = z['seba_label'][0].astype(np.int64)
        v = z['vecs'][0].astype(np.float64) * z['scale'][0][:, :, None] / 32767
        vals = z['vals'][0].astype(np.float64)
    vecs = np.concatenate([np.full((v.shape[0], N, 1), 1 / N ** .5), v.transpose(0, 2, 1)], 2)
    return lab, vecs, vals


# ---------------------------------------------------------------- extraction (GPU)

def environment():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_data(run, step):
    """Exactly measure_v2.model_data (same probe: 200 per class, RandomState(0), sorted)."""
    import numpy as np
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model
    ck = load_checkpoint(Path(f'/out/runs/{run}/checkpoints/step_{step}.pt'))
    model = build_model(ck['config']); model.load_state_dict(ck['model_state']); model.eval().cuda()
    data = build_data(ck['config'], Path('/zoo/data'), 'cpu')
    y = data.test_y.numpy(); rng = np.random.RandomState(0)
    idx = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], 200, replace=False) for c in range(10)]))
    return model, data, idx


def activations(model, x):
    h = model.act(model.bn1(model.conv1(x))); acts = [h]
    for block in model.blocks:
        h = block(h); acts.append(h)
    pool = h.mean((2, 3)); return acts + [pool, model.fc(pool)]


def _acts(model, data, idx, batch=128):
    import torch
    storage = [[] for _ in SITES]
    with torch.inference_mode():
        for at in range(0, len(idx), batch):
            aa = activations(model, data.normalize(data.test_x[idx[at:at + batch]]).cuda())
            for dst, a in zip(storage, aa):
                dst.append(a.cpu())
    return [torch.cat(a) for a in storage]


def _gram_from_raw_torch(R):
    """centred cosine Gram from R on the GPU (formula of stats.raw_to_views, full set)."""
    import torch
    m = R.mean(1)
    G = R - m[None, :] - m[:, None] + R.mean()
    G = .5 * (G + G.T)
    d = G.diagonal().clamp_min(0).sqrt().clamp_min(1e-12)
    return (G / d[:, None] / d[None, :]).clamp(-1, 1)


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=16384, timeout=1200, max_containers=8, retries=0, volumes=VOLS)
def extract_raw(step: int, run: str = RUN):
    import numpy as np
    import torch
    from coherent import centred_cosine_gram
    environment(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    rt = 's0' if run == RUN else 's1'
    name = f'extract_raw_{rt}_{step}'
    pre = '' if rt == 's0' else f'{rt}_'
    outs = [work / f'{pre}raw_{v}_{step}_{si}.npy' for si in range(12) for v in ('flat', 'gap')]
    c = cached(work, name, outs)
    if c:
        return c
    model, data, idx = model_data(run, step)
    with np.load(apub / 'probe.npz') as p:
        assert np.array_equal(p['idx'], idx)
    acts = _acts(model, data, idx)
    checks = []
    for si, a in enumerate(acts):
        x = a.reshape(len(a), -1).cuda().double()
        R = x @ x.T
        C = _gram_from_raw_torch(R)
        internal = float((C - centred_cosine_gram(x)).abs().max())
        assert internal < 1e-10, (si, internal)
        row = dict(site=SITES[si], dim=int(x.shape[1]), internal_max_abs=internal)
        if rt == 's0':
            stored = torch.as_tensor(load_gram(step, si), device='cuda')
            row['stored_max_abs'] = float((C - stored).abs().max())
        np.save(work / f'{pre}raw_flat_{step}_{si}.npy', R.cpu().numpy())
        if a.ndim == 4:
            g = a.mean((2, 3)).cuda().double(); Rg = g @ g.T
        else:
            Rg = R
        np.save(work / f'{pre}raw_gap_{step}_{si}.npy', Rg.cpu().numpy())
        checks.append(row)
        print(name, row, flush=True)
    if step == 0 and rt == 's0':
        x = data.normalize(data.test_x[idx]).reshape(len(idx), -1).double()
        np.save(work / 'raw_pixels_input.npy', (x @ x.T).numpy())
        u = data.test_x[idx].reshape(len(idx), -1).double() / 255
        np.save(work / 'raw_pixels_u8.npy', (u @ u.T).numpy())
    worst = max(r.get('stored_max_abs', 0) for r in checks)
    (work / f'checks_{name}.json').write_text(_dumps(dict(rows=checks, stored_max_abs=worst,
        torch=torch.__version__, cudnn=torch.backends.cudnn.version())))
    vol.commit()
    if rt == 's0' and worst > 1e-10:
        raise AssertionError(f'raw-derived cosine Gram differs from stored by {worst:.3e} (> 1e-10); see checks_{name}.json')
    return finished(work, name, start, cpu=2, mem=16, gpu=True, stored_max_abs=worst)


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=16384, timeout=2400, retries=0, volumes=VOLS)
def seed1():
    """I3: seed-1 final checkpoint, full default method (rule 0) for all 78 layer blocks."""
    import numpy as np
    import torch
    from schema_v2 import block_record
    from coherent import symmetric_balance
    import sources as S
    environment(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = 'seed1_layer_64000'
    out = pub / f'I3_{name}.npz'
    c = cached(work, name, [out])
    if c:
        return c
    extract_raw.local(FINAL, RUN1)
    ops = []
    for si in range(12):
        R = torch.as_tensor(load_raw(FINAL, si, 'flat', 's1'), device='cuda')
        C = _gram_from_raw_torch(R)
        ops.append(S.lazy(symmetric_balance(S.knn_logkernel(C)), .5))
    ps = pairs(12)
    labels = np.empty((len(ps), 5, N), np.int8); vals = np.empty((len(ps), 9), np.float32)
    for pi, (a, b) in enumerate(ps):
        rec, _ = block_record(ops, a, b)
        labels[pi] = rec['seba_label']; vals[pi] = rec['vals']
    np.savez_compressed(out, seba_label=labels, vals=vals)
    return finished(work, name, start, cpu=2, mem=16, gpu=True)


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=32768, timeout=2400, retries=0, volumes=VOLS)
def extract_10k():
    """All 10,000 CIFAR-10 test images (test-set order), final checkpoint, raw Grams float32."""
    import numpy as np
    import torch
    environment(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = 'extract_10k_64000'
    outs = [work / f'raw10k_flat_{FINAL}_{si}.npy' for si in range(12)]
    c = cached(work, name, outs)
    if c:
        return c
    model, data, idx = model_data(RUN, FINAL)
    allidx = np.arange(len(data.test_y))
    assert len(allidx) == 10000
    acts = _acts(model, data, allidx, batch=500)
    rows = []
    for si, a in enumerate(acts):
        x = a.reshape(len(a), -1).cuda().float()
        R = (x @ x.T)
        np.save(work / f'raw10k_flat_{FINAL}_{si}.npy', R.cpu().numpy())
        r2 = load_raw(FINAL, si) if (work / f'raw_flat_{FINAL}_{si}.npy').is_file() else None
        if r2 is not None:
            sub = R[idx][:, idx].double().cpu().numpy()
            rows.append(dict(site=SITES[si], probe_rel_max=float(np.abs(sub - r2).max() / np.abs(r2).max())))
    np.savez(work / 'raw10k_index.npz', test_index=allidx, probe_positions=idx, labels=data.test_y.numpy())
    (work / f'checks_{name}.json').write_text(_dumps(dict(rows=rows, dtype='float32')))
    return finished(work, name, start, cpu=4, mem=32, gpu=True)


# ---------------------------------------------------------------- CPU units

def _threads():
    import torch
    torch.set_num_threads(4)


@app.function(image=image, **CPU_UNIT, timeout=1800, retries=0, volumes=VOLS)
def pixels():
    """I1: the method and L1 on centred, row-normalised network-input pixels (one cell)."""
    import numpy as np
    import torch
    import sources as S
    from stats import raw_to_views
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = 'pixels'; out = pub / 'I1_pixels.npz'
    c = cached(work, name, [out])
    if c:
        return c
    res = {}
    for which in ('input', 'u8'):
        C = torch.as_tensor(raw_to_views(np.load(work / f'raw_pixels_{which}.npy'))['C'])
        cache = {}
        for src in ('M0', 'L1'):
            res[f'{src}_{which}'] = np.stack([S.SOURCES[src]([C], r, cache=cache, cell_keys=['px'])['labels']
                                              for r in RANKS]).astype(np.int8)
    np.savez_compressed(out, **res)
    return finished(work, name, start)


def _fill_block(cache, keys, fam_lab, fam_vecs, fam_vals, pindex, cells_pairs):
    """Pre-populate a sources.Block cache with STORED results: the block's Mbar modes and
    its cells' single-cell modes / SEBA labels (pairs (a, a) of the same file)."""
    for key, pi in cells_pairs:
        ck = ('cell', key, 'modes')
        if ck not in cache:
            cache[ck] = (fam_vals[pi], fam_vecs[pi])
            for ri, r in enumerate(RANKS):
                cache[('cell', key, ('seba', r))] = fam_lab[pi, ri]


def block_sources(grams, ops, keys, m0_lab, m0_vecs, m0_vals, classes, cache, names, n_perm=N_PERM,
                  extra_labels=None, timing=None, perms=None, maxacc=None, block_seed=0):
    """All requested sources for one block and their agreement rows (AGREE_FIELDS).

    Set counts are declared (r for every source at rank r, N_CLASSES for classes), so empty
    sets count and numbering does not matter (REVIEW_2 item 3). N7 bands: for statistics on
    all points (ami_all, mj_max, class_*) the permutations are `perms`, shared by every block
    of the file (coherent), and each draw's value is folded into the running max in `maxacc`
    (max-over-blocks band). The jointly-assigned AMIs use a permutation within the jointly
    assigned points (conditional on coverage, marginals fixed), drawn independently per block;
    their max-over-blocks band is therefore that of independent blocks (no smaller than the
    coherent one for positively dependent blocks; stated in SCHEMA.md)."""
    import numpy as np
    import sources as S
    import nulls as NL
    from stats import agreement, fast_ami_null, subspace_overlap, ami, matched_jaccard
    b = S._block(grams, ops, cache, keys)
    b.local['mbar'] = (m0_vals, m0_vecs)
    labels, agree = {}, {}
    def fn(name):
        return NL.n3_source if name == 'N3' else S.SOURCES[name]
    def fold(name, ri, field, draws):
        if maxacc is not None:
            k = (name, ri, field)
            maxacc[k] = draws.copy() if k not in maxacc else np.maximum(maxacc[k], draws)
    for name in ['M0'] + names:
        lab_r, ag_r = [], []
        for ri, r in enumerate(RANKS):
            t0 = time.time()
            if name == 'M0':
                lab, basis = m0_lab[ri], m0_vecs[:, 1:r]
            elif extra_labels is not None and name in extra_labels:
                lab, basis = extra_labels[name][ri], None
            else:
                out = fn(name)(grams, r, seed=0, ops=ops, cache=cache, cell_keys=keys)
                lab, basis = out['labels'], out['basis']
            t1 = time.time()
            ref = m0_lab[ri]
            ag = agreement(ref, lab, r, r)
            nm = fast_ami_null(ref, lab, n_perm, seed=ri, perms=perms, return_draws=True, ka=r, kb=r)
            nc = fast_ami_null(lab, classes, n_perm, seed=100 + ri, perms=perms, return_draws=True, ka=r, kb=N_CLASSES)
            jm = (ref >= 0) & (lab >= 0)
            sm = lab >= 0
            seed_j = 1000003 * block_seed + 17 * ri
            nj = fast_ami_null(ref[jm], lab[jm], n_perm, seed=seed_j, return_draws=True) if jm.sum() > 1 else None
            ncj = fast_ami_null(lab[sm], classes[sm], n_perm, seed=seed_j + 1, return_draws=True) if sm.sum() > 1 else None
            row = dict(ag, unassigned=ag['unassigned_b'], ami_null_q95=nm['ami']['q95'],
                       mj_null_mean=nm['mj_max']['mean'], mj_null_q95=nm['mj_max']['q95'],
                       overlap_mean_cos2=np.nan, overlap_min_cos=np.nan,
                       class_ami=ami(lab, classes), class_ami_null_q95=nc['ami']['q95'],
                       class_mj_max=matched_jaccard(lab, classes, ka=r, kb=N_CLASSES),
                       class_mj_null_mean=nc['mj_max']['mean'], class_mj_null_q95=nc['mj_max']['q95'],
                       ami_joint_null_q95=nj['ami']['q95'] if nj else np.nan,
                       class_ami_joint=ami(lab, classes, 'joint'),
                       class_ami_joint_null_q95=ncj['ami']['q95'] if ncj else np.nan)
            fold(name, ri, 'ami_all', nm['ami_draws']); fold(name, ri, 'mj_max', nm['mj_draws'])
            fold(name, ri, 'class_ami', nc['ami_draws']); fold(name, ri, 'class_mj_max', nc['mj_draws'])
            if nj:
                fold(name, ri, 'ami_joint', nj['ami_draws'])
            if ncj:
                fold(name, ri, 'class_ami_joint', ncj['ami_draws'])
            if basis is not None and r > 1:
                so = subspace_overlap(m0_vecs[:, 1:r], basis)
                row['overlap_mean_cos2'], row['overlap_min_cos'] = so['mean_cos2'], so['min_cos']
            if timing is not None:
                timing.setdefault(name, []).append(t1 - t0)
                timing.setdefault('agreement+nulls', []).append(time.time() - t1)
            lab_r.append(lab); ag_r.append([row[f] for f in AGREE_FIELDS])
        labels[name] = np.stack(lab_r).astype(np.int8)
        agree[name] = np.array(ag_r, np.float32)
    return labels, agree


def max_bands(maxacc, names):
    """maxnull_<SRC> arrays (5 ranks, len(MAX_FIELDS), 2) = [mean, q95] of the max-over-blocks
    null draws; NaN where a field had no draws (written as such in npz; see SCHEMA.md)."""
    import numpy as np
    out = {}
    for name in names:
        a = np.full((len(RANKS), len(MAX_FIELDS), 2), np.nan, np.float32)
        for ri in range(len(RANKS)):
            for fi, f in enumerate(MAX_FIELDS):
                d = maxacc.get((name, ri, f))
                if d is not None:
                    a[ri, fi] = (d.mean(), np.quantile(d, .95))
        out[f'maxnull_{name}'] = a
    return out


@app.function(image=image, **CPU_UNIT, timeout=4 * 3600, max_containers=20, retries=0, volumes=VOLS)
def sources_unit(family: str, index: int, names: list, with_l3: bool = False):
    """Explorer-grid sources for every block of one layer step (78) or time site (36)."""
    import numpy as np
    import torch
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    key = STEPS[index] if family == 'layer' else SITES[index]
    fname = f'{family}_{key}'; name = f'sources_{fname}'
    out = pub / f'sources_{fname}.npz'
    c = cached(work, name, [out])
    if c:
        return c
    names = list(names) + (['L3'] if with_l3 else [])
    ncell = 12 if family == 'layer' else 8
    cells = block_cells(family, key, 0, ncell - 1)
    grams = [torch.as_tensor(load_gram(*c)) for c in cells]
    ops = [torch.as_tensor(load_op(*c)) for c in cells]
    keys = [f'{s}_{si}' for s, si in cells]
    fam_lab, fam_vecs, fam_vals = stored_m0(family, key)
    with np.load(apub / 'probe.npz') as p:
        classes = p['labels'].astype(np.int64)
    ps = pairs(ncell)
    single = {a: ps.index((a, a)) for a in range(ncell)}
    extra_files = {}
    if family == 'layer':
        extra_files['I2'] = stored_m0('layer', 0)[0]
        if key == FINAL and (pub / 'I3_seed1_layer_64000.npz').is_file():
            with np.load(pub / 'I3_seed1_layer_64000.npz') as z:
                extra_files['I3'] = z['seba_label'].astype(np.int64)
    pix = None
    if (pub / 'I1_pixels.npz').is_file():
        with np.load(pub / 'I1_pixels.npz') as z:
            pix = z['M0_input'].astype(np.int64)
    cache = {}
    all_labels, all_agree = {}, {}
    perms = np.argsort(np.random.default_rng([7, index, 0 if family == 'layer' else 1]).random((N_PERM, N)), axis=1)
    maxacc = {}
    for pi, (a, b) in enumerate(ps):
        bk = keys[a:b + 1]
        _fill_block(cache, keys, fam_lab, fam_vecs, fam_vals, pi, [(keys[s], single[s]) for s in range(a, b + 1)])
        extra = {}
        m = (a + b) // 2
        for nm_, s in (('C1first', a), ('C1mid', m), ('C1last', b)):
            extra[nm_] = fam_lab[single[s]]
        for k_, arr in extra_files.items():
            extra[k_] = arr[pi]
        if pix is not None:
            extra['I1'] = pix
        labels, agree = block_sources(grams[a:b + 1], ops[a:b + 1], bk, fam_lab[pi], fam_vecs[pi], fam_vals[pi],
                                      classes, cache, names + ['N3'] + list(extra), extra_labels=extra,
                                      perms=perms, maxacc=maxacc, block_seed=pi)
        for k_ in agree:
            all_agree.setdefault(k_, []).append(agree[k_])
            if k_ not in extra and k_ != 'M0':
                all_labels.setdefault(k_, []).append(labels[k_])
        # drop block-level cache entries (cell entries are reused by later blocks)
        cache.pop(('block', tuple(bk)), None)
        for ck in [k for k in cache if k[0] == 'n3']:
            cache.pop(ck)
        print(fname, pi, round(time.time() - start), flush=True)
    arrays = {k: np.stack(v) for k, v in all_labels.items()}
    arrays.update({f'agree_{k}': np.stack(v) for k, v in all_agree.items()})
    arrays.update(max_bands(maxacc, list(all_agree)))
    np.savez_compressed(out, **arrays)
    return finished(work, name, start)


def _rep_block(rid):
    import torch
    _, family, key, a, b = next(r for r in REPS if r[0] == rid)
    cells = block_cells(family, key, a, b)
    return family, key, a, b, cells


@app.function(image=image, **CPU_UNIT, timeout=6 * 3600, max_containers=36, retries=0, volumes=VOLS)
def reps_unit(rid: str, task: str, n_draws: int = 20):
    """One representative block x one task: 'nulls' | 'stability' | 'ablate' | 'linear'."""
    import numpy as np
    import torch
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = part_name(rid, task); out = work / f'{name}.json'
    c = cached(work, name, [out])
    if c:
        return c
    family, key, a, b, cells = _rep_block(rid)
    grams = [torch.as_tensor(load_gram(*c_)) for c_ in cells]
    ops = [torch.as_tensor(load_op(*c_)) for c_ in cells]
    with np.load(apub / 'probe.npz') as p:
        classes = p['labels'].astype(np.int64)
    res, draws = run_rep(rid, task, family, key, a, b, cells, grams, ops, classes, n_draws)
    if draws:
        np.savez_compressed(pub / f'rep_{rid}_{task}_draws.npz', **draws)
    out.write_text(_dumps(res))
    return finished(work, name, start)


def run_rep(rid, task, family, key, a, b, cells, grams, ops, classes, n_draws=20):
    """Body of reps_unit, free of the volume (loaders are module functions, patchable)."""
    import numpy as np
    import torch
    import sources as S
    import nulls as NL
    import variants as V
    from stats import (agreement, subspace_overlap, conditional_entropy, band, cka, cka_perm_null,
                       linear_content)
    n = len(S._n(ops[0]))
    keys = [f'{s}_{si}' for s, si in cells]
    vals, vecs = S.mbar_modes(ops, S.N_VECS + 1)
    real, real_labels, _ = NL.spectral_summary(ops, classes=classes, eig=(vals, vecs))
    res = dict(block=rid, cells=keys, real=real)
    draws = {}

    def summ(o, tag, t):
        s_, lab_, _ = NL.spectral_summary(o, classes=classes)
        s_['agreement_with_M0'] = {r: agreement(real_labels[r], lab_[r], r, r) for r in RANKS}
        if t == 0:
            draws[tag] = np.stack([lab_[r] for r in RANKS]).astype(np.int8)
        return s_

    if task == 'nulls':
        Rs = [load_raw(s, si) for s, si in cells]
        spectra = [NL.covariance_spectrum(R) for R in Rs]
        res['N2a'] = [summ(NL.ops_from_grams(NL.n2a_grams(Rs, seed=t, spectra=spectra)), 'N2a', t) for t in range(n_draws)]
        n2b = []
        for t in range(n_draws):
            g, info = NL.n2b_grams(grams, seed=t)
            s_ = summ(NL.ops_from_grams(g), 'N2b', t); s_['moment_mismatch'] = info; n2b.append(s_)
        res['N2b'] = n2b
        res['N3'] = [summ(NL.n3_ops(ops, keys, seed=t), 'N3', t) for t in range(n_draws)]
        res['N4'] = [summ(NL.n3_ops(ops, keys, seed=t, strata=classes), 'N4', t) for t in range(n_draws)]
        n5 = []
        for t in range(min(10, n_draws)):
            o, info = NL.n5_ops(grams, seed=t)
            s_ = summ(o, 'N5', t); s_['rewire'] = info; n5.append(s_)
        res['N5'] = n5
        res['bands'] = {nl: dict(lambda_P=[band([d['lambda_P'][j] for d in res[nl]]) for j in range(8)])
                        for nl in ('N2a', 'N2b', 'N3', 'N4', 'N5')}
    elif task == 'stability':
        Rs = [load_raw(s, si) for s, si in cells]
        res['S1'] = V.subsample_block(Rs, full=(vals, vecs))
        fit = np.sort(np.random.default_rng(0).choice(n, int(round(.75 * n)), replace=False))
        refit = {'M0': real_labels}
        cache = {}
        refit['L1'] = {r: S.SOURCES['L1'](grams, r, cache=cache, cell_keys=keys)['labels'] for r in RANKS}
        refit['L3'] = {r: S.SOURCES['L3'](grams, r, cache=cache, cell_keys=keys)['labels'] for r in RANKS}
        res['S2'] = V.fit_extend(Rs, fit, refit=refit)
        res['S3'] = {f'{r}|{r2}': conditional_entropy(real_labels[r], real_labels[r2])
                     for r in RANKS for r2 in RANKS if r2 > r}
        nxt = None
        if family == 'layer' and b + 1 < 12:
            nxt = torch.as_tensor(load_op(key, b + 1))
        elif family == 'time' and b + 1 < len(STEPS):
            nxt = torch.as_tensor(load_op(STEPS[b + 1], cells[0][1]))
        res['LOCO'] = {r: V.leave_one_out(ops, r, next_op=nxt) for r in RANKS} if len(ops) > 1 or nxt is not None else {}
        res['N6_rotation'] = {r: [{k_: v for k_, v in d.items() if k_ != 'labels'} for d in NL.n6_rotation(vecs, r, n_draws)]
                              for r in RANKS}
    elif task == 'linear':
        G = [S._n(g) for g in grams]
        CB = np.mean(G, 0)
        res['linear_content'] = {r: dict(block=linear_content(vecs[:, 1:r], CB),
                                         per_cell=[linear_content(vecs[:, 1:r], g) for g in G]) for r in RANKS}
        L = len(G)
        res['cka'] = [[cka(G[i], G[j]) for j in range(L)] for i in range(L)]
        res['cka_debiased'] = [[cka(G[i], G[j], True) for j in range(L)] for i in range(L)]
        res['cka_perm_null'] = cka_perm_null(G[0], G[-1], 100) if L > 1 else None   # biased CKA (kept)
        res['cka_perm_null_biased'] = res['cka_perm_null']
        res['cka_debiased_perm_null'] = cka_perm_null(G[0], G[-1], 100, debiased=True) if L > 1 else None
        res['cka_perm_null_pair'] = [0, L - 1]
        res['pca_C_vs_HCH'] = [V.pca_c_vs_hch(g) for g in G]
        lv, U, _ = S.top_eigs(CB - CB.mean(0) - CB.mean(1)[:, None] + CB.mean(), S.N_VECS)
        from stats import dirichlet_profile
        res['pca_dirichlet_under_cells'] = {r: {k_: np.asarray(v).tolist() for k_, v in dirichlet_profile(ops, U[:, :r - 1]).items() if k_ != 'per_cell'} for r in RANKS}
        res['method_dirichlet_under_cells'] = {r: {k_: np.asarray(v).tolist() for k_, v in dirichlet_profile(ops, vecs[:, 1:r]).items() if k_ != 'per_cell'} for r in RANKS}
        # the two profiles above use the lazy M_c: f^T (I - M_c) f / f^T f = half the P_c energy; also store the P_c form
        for k_ in ('pca_dirichlet_under_cells', 'method_dirichlet_under_cells'):
            res[k_ + '_P'] = {r: {q_: (2 * np.asarray(v)).tolist() for q_, v in d.items()} for r, d in res[k_].items()}
        res['dirichlet_convention'] = dict(
            pca_dirichlet_under_cells='f^T (I - M_c) f / f^T f with the lazy M_c = (I + P_c)/2 (= half the P_c energy)',
            pca_dirichlet_under_cells_P='f^T (I - P_c) f / f^T f (twice the M_c value)')
        cm = [S.mbar_modes([o], 9)[1] for o in ops]
        res['subspace_overlap_cells'] = {r: [[subspace_overlap(cm[i][:, 1:r], cm[j][:, 1:r])['mean_cos2'] for j in range(L)] for i in range(L)] for r in (3, 6)}
    elif task == 'ablate':
        res['ablations'] = ablate_block(grams, ops, cells, keys, real_labels, vecs, classes, family, key, a, b)
    else:
        raise ValueError(task)
    return res, draws


def _dumps(o, **kw):
    """Every JSON this module writes: numpy -> Python, non-finite -> null, allow_nan=False."""
    from stats import dumps
    return dumps(o, **kw)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def load_kappa(step, si, mode):
    """Stored per-cell kappa of kernel rule `mode` (3 kNN scale, 4 half-mass) from r20s0b meta."""
    meta = json.loads((Path('/out/analysis') / TAG_IN / 'meta.json').read_text())
    return float(meta['kappa'][mode][STEPS.index(step)][si])


def ablate_block(grams, ops, cells, keys, real_labels, vecs, classes, family, key, a, b, n_sub=5, n_n3=10):
    """One factor at a time around the default (PLAN section 5, factors A-I).

    Markov levels (A, B, D retained diagonal, H) are built by a constructor of an image
    subset, so each gets (PLAN section 5: "agreement with the default, stability, null
    excess"): a spectral summary; agreement with the default M0 with its own N7 references
    (stats.agreement_with_null, declared counts r); span overlap with M0 and its chance value;
    `n3`: lambda_P of the level's own operators under n_n3 N3 draws, the band and the excess
    of the real lambda_P over the band's 95th percentile (undefined for one cell); `S1`:
    n_sub 80% subsets (the same subsets for every level, seed 0), the level rebuilt on each
    subset and compared with the level's own full-set labels on the shared images.
    Subset rebuilds keep per-cell parameters found on the full set (matched levels' k /
    perplexity / kappa and the stored rule-3/4 kappas) fixed."""
    import numpy as np
    import torch
    import sources as S
    import nulls as NL
    import variants as V
    from stats import agreement_with_null, raw_to_views, band, ami
    out = {}
    n = len(S._n(ops[0]))
    Rs = [load_raw(s_, si) for s_, si in cells]
    Rgap = {}
    rng = np.random.default_rng(0)
    subsets = [np.sort(rng.choice(n, int(round(.8 * n)), replace=False)) for _ in range(n_sub)]

    def views(view, idx):
        if view == 'centred' and idx is None:
            return grams
        if view == 'gap':
            if not Rgap:
                Rgap.update({i: load_raw(s_, si, 'gap') for i, (s_, si) in enumerate(cells)})
            src = [Rgap[i] for i in range(len(cells))]
        else:
            src = Rs
        res = []
        for R in src:
            v = raw_to_views(R if idx is None else R[np.ix_(idx, idx)])
            res.append(torch.as_tensor(v['C_unc'] if view == 'uncentred' else v['C']))
        return res

    def ov(U_ref, U, r):
        from stats import subspace_overlap
        return dict(mean_cos2=subspace_overlap(U_ref, U)['mean_cos2'], chance=(r - 1) / (len(U) - 1))

    def agree_m0(labs, idx=None):
        return {r: agreement_with_null(real_labels[r] if idx is None else real_labels[r][idx], labs[r], r, r, seed=r)
                for r in RANKS}

    def eig_labels(o):
        mb = np.mean([S._n(m) for m in o], 0)
        vals, vv, _ = S.top_eigs(mb, S.N_VECS, project_constant=True)
        return vals, vv, {r: S.seba_sets(S.with_constant(vv), r)[0] for r in RANKS}

    def to_lamP(vals, laziness):
        vals = np.asarray(vals)
        return 2 * vals - 1 if laziness == 'half' else vals

    def level(ctor, laziness='half', full_ops=None):
        o = ctor(None) if full_ops is None else full_ops
        s_, lab_, v_ = NL.spectral_summary(o, classes=classes, laziness=laziness)
        s_['agreement_with_M0'] = agree_m0(lab_)
        s_['overlap_with_M0'] = {r: ov(vecs[:, 1:r], v_[:, :r - 1], r) for r in RANKS}
        if len(o) > 1:
            draws = []
            for t in range(n_n3):
                po = NL.n3_ops(o, keys, seed=t)
                vals, _, _ = S.top_eigs(np.mean([S._n(m) for m in po], 0), S.N_VECS, project_constant=True)
                draws.append(to_lamP(vals, laziness))
            draws = np.array(draws)
            bands = [band(draws[:, j]) for j in range(S.N_VECS)]
            s_['n3'] = dict(draws=n_n3, lambda_P_band=bands,
                            excess_over_q95=[float(s_['lambda_P'][j] - bands[j]['q95']) for j in range(S.N_VECS)])
        else:
            s_['n3'] = dict(undefined='single-cell block: N3 fixes the only cell')
        rows = []
        for t, idx in enumerate(subsets):
            vals, vv, labs = eig_labels(ctor(idx))
            rows.append(dict(subset=t, lambda_P=to_lamP(vals, laziness),
                             per_r={r: dict(agreement_with_null(lab_[r][idx], labs[r], r, r, seed=97 * t + r),
                                            overlap=ov(v_[idx, :r - 1], vv[:, :r - 1], r)) for r in RANKS}))
        s_['S1'] = rows
        return s_

    def attempt(f):
        try:
            return f()
        except Exception as e:   # a failed level (e.g. Sinkhorn non-convergence) is a result
            return dict(error=f'{type(e).__name__}: {str(e)[:300]}')

    def kern(fam, view='centred', per_cell=None, **kw):
        """Constructor of an image subset: the level's operator for every cell."""
        kparams = {k_: v for k_, v in kw.items() if k_ in ('k', 'floor', 'mutual', 'perplexity', 'kappa')}
        oparams = {k_: v for k_, v in kw.items() if k_ not in kparams}
        def ctor(idx):
            gs = views(view, idx)
            if per_cell is not None:
                return [V.operator(V.logkernel(g, fam, **kparams, **{per_cell[0]: p}), **oparams)[0]
                        for g, p in zip(gs, per_cell[1])]
            return [V.operator(V.logkernel(g, fam, **kparams), **oparams)[0] for g in gs]
        return ctor

    target = float(np.median([V.effective_neighbours(2 * S._n(m) - np.eye(n)) for m in ops]))
    A = {}
    for k in (5, 15, 50, 150):
        A[f'knn{k}'] = attempt(lambda: level(kern('knn', k=k)))
    A['mutual15'] = attempt(lambda: level(kern('knn', mutual=True)))
    for pp in (30, 100, 300):
        A[f'perplexity{pp}'] = attempt(lambda: level(kern('perplexity', perplexity=pp)))
    A['local15'] = attempt(lambda: level(kern('local')))
    A['eps_matched_degree'] = attempt(lambda: level(kern('eps')))
    # global vMF: the two stored per-cell kappas (rules 3 'kNN scale' and 4 'half-mass'; the stored operators
    # are exactly this construction, used for the full set; subsets reuse the stored kappas) + 2 globals
    for mode, nm in ((3, 'vmf_stored_kNN_scale'), (4, 'vmf_stored_half_mass')):
        def run_stored(mode=mode):
            kaps = [load_kappa(s_, si_, mode) for s_, si_ in cells]
            full = [torch.as_tensor(load_op(s_, si_, mode)) for s_, si_ in cells]
            return dict(level(kern('vmf', per_cell=('kappa', kaps)), full_ops=full), kappa=kaps)
        A[nm] = attempt(run_stored)
    for kap in (2., 8.):
        A[f'vmf{kap:g}'] = attempt(lambda: level(kern('vmf', kappa=kap)))
    matched = {}
    for fam, lo, hi, integer in (('knn', 2, 400, True), ('perplexity', 5., 1500., False), ('vmf', .1, 300., False)):
        def run_matched():
            params = [V.match_neighbours(g, fam, target, lo, hi, integer=integer)[0] for g in grams]
            kk = dict(knn='k', perplexity='perplexity', vmf='kappa')[fam]
            return dict(params=params, summary=level(kern(fam, per_cell=(kk, params))))
        matched[fam] = attempt(run_matched)
    A['matched_effective_neighbours'] = dict(target=target, **matched)
    out['A'] = A
    B = {}
    for fl in (1e-8, 1e-6, 1e-4, 0.0):
        def run_floor(fl=fl):
            s_ = level(kern('knn', floor=fl))
            o = kern('knn', floor=fl)(None)
            s_['off_edge_mass'] = [V.off_edge_mass(2 * S._n(m) - np.eye(n), S.knn_mask(g)) for m, g in zip(o, grams)]
            s_['components'] = [V.components(S.knn_mask(g)) for g in grams]
            return s_
        B[f'floor{fl:g}'] = attempt(run_floor)
    out['B'] = B
    C = {}
    for al in (0., .5, 1.):
        o = kern('knn', norm='degree', alpha=al)(None)
        infos = [V.operator(V.logkernel(g, 'knn'), norm='degree', alpha=al)[1] for g in grams]
        vals_, vecs_, info_ = V.top_unprojected(o)
        U_ = np.linalg.qr(vecs_[:, 1:] - vecs_[:, 1:].mean(0))[0]
        lab = {r: S.seba_sets(S.with_constant(U_[:, :r - 1]), r)[0] for r in RANKS}
        C[f'degree_alpha{al:g}'] = dict(stationary=V.stationary_drift(infos), eig=info_, vals=vals_,
                                        vals_includes_leading_vector=True,
                                        spectrum_of='eigenvalues of the mean over cells of (I + S_alpha)/2; vals[0] is the leading (non-constant) vector, kept',
                                        agreement_with_M0=agree_m0(lab),
                                        overlap_with_M0={r: ov(vecs[:, 1:r], vecs_[:, 1:r], r) for r in RANKS})
    o = kern('knn', norm='raw')(None)
    vals_, vecs_, info_ = V.top_unprojected(o)
    C['raw_adjacency'] = dict(eig=info_, vals=vals_, vals_includes_leading_vector=True,
                              note='not a Markov operator; no retention',
                              spectrum_of='eigenvalues of the mean over cells of K / ||K||_2; vals[0] is the leading vector, kept',
                              overlap_with_M0={r: ov(vecs[:, 1:r], vecs_[:, 1:r], r) for r in RANKS})
    out['C'] = C
    D = {}
    # V.operator(diagonal='retained') returns the balanced kernel with NO lazy step: native spectrum
    def run_retained():
        ctor = kern('knn', diagonal='retained')
        infos = [V.operator(V.logkernel(g, 'knn'), diagonal='retained')[1] for g in grams]
        return dict(level(ctor, laziness='none'), diag_median=[i['diag_median'] for i in infos])
    D['retained_diagonal'] = attempt(run_retained)
    for gm in (0., .5, .9):
        pr = V.combine([S.lazy(2 * S._t(m) - torch.eye(n, dtype=torch.float64), gm) if gm != .5 else m for m in ops], 'product')
        D[f'product_gamma{gm:g}'] = dict(vals=pr['vals'], spectrum_of='singular values of the constant-projected product of gamma I + (1-gamma) P_l',
                                         right_overlap={r: ov(vecs[:, 1:r], pr['right'][:, :r - 1], r) for r in RANKS},
                                         left_overlap={r: ov(vecs[:, 1:r], pr['left'][:, :r - 1], r) for r in RANKS})
    out['D'] = D
    E = {}
    if len(ops) > 1:
        for how in ('meansq', 'expsum', 'concat', 'union', 'intersection'):
            r_ = V.combine(ops, how, grams=grams)
            E[how] = dict(vals=r_['vals'], spectrum_of=dict(meansq='eigenvalues of the mean of M^2 (constant removed)', expsum='eigenvalues of exp(-(1/2) sum_l L_l), L_l = I - P_l (constant removed)', concat='lambda_M of the kNN graph on C_B (constant removed)', union='lambda_M of the union graph (constant removed)', intersection='lambda_M of the intersection graph (constant removed)')[how],
                          overlap_with_M0={r: ov(vecs[:, 1:r], r_['vecs'][:, :r - 1], r) for r in RANKS},
                          agreement_with_M0=agree_m0({r: S.seba_sets(S.with_constant(r_['vecs']), r)[0] for r in RANKS}))
        for how in ('product', 'twoend', 'expprod'):
            r_ = V.combine(ops, how)
            E[how] = dict(vals=r_['vals'], spectrum_of='singular values of the constant-projected product',
                          right_overlap={r: ov(vecs[:, 1:r], r_['right'][:, :r - 1], r) for r in RANKS},
                          left_overlap={r: ov(vecs[:, 1:r], r_['left'][:, :r - 1], r) for r in RANKS})
        E['commutator_fro'] = V.commutators(ops)
    out['E'] = E
    F = {}
    if family == 'time':
        steps = [s_ for s_, _ in cells]
        def fl_(r_):
            return dict(overlap_with_M0={r: ov(vecs[:, 1:r], r_['vecs'][:, :r - 1], r) for r in RANKS},
                        agreement_with_M0=agree_m0({r: S.seba_sets(S.with_constant(r_['vecs']), r)[0] for r in RANKS}))
        for sch in ('interval', 'log'):
            r_ = V.combine(ops, 'mean', weights=V.time_weights(steps, sch))
            F[sch] = dict(fl_(r_), vals=r_['vals'], vals_convention='lambda_M (lazy M)', lambda_P=2 * np.asarray(r_['vals']) - 1)
        F['drop_every_other'] = fl_(V.combine(ops[::2], 'mean'))
        F['repeat_last'] = fl_(V.combine(ops + [ops[-1]], 'mean'))
    out['F'] = F
    G = {}
    V0 = vecs
    for cc in (.5, .99, 1.5):
        G[f'seba_c{cc:g}'] = {}
        for r in RANKS:
            lab, info = V.seba_mu(V0, r, cc)
            G[f'seba_c{cc:g}'][r] = dict(info, **agreement_with_null(real_labels[r], lab, r, r, seed=r))
    grid = V.round_grid(V0[:, 1:])
    G['kmeans_grid'] = {f'q{q}_k{kc}': dict(class_ami=float(ami(l, classes))) for (q, kc), l in grid.items()}
    out['G'] = G
    H = {}
    H['uncentred'] = attempt(lambda: level(lambda idx: S.default_ops(views('uncentred', idx))))
    H['gap'] = attempt(lambda: level(lambda idx: S.default_ops(views('gap', idx))))
    out['H'] = H
    I = {}
    for nn in [m for m in (500, 1000) if m < n]:
        idx = np.sort(np.random.default_rng(nn).choice(n, nn, replace=False))
        for rule in ('fixed', 'proportional'):
            g_, o_, k_ = V.subsample_grams(Rs, idx, rule)
            v_ = S.mbar_modes(o_, 9)[1]
            I[f'N{nn}_{rule}'] = dict(k=k_, agreement_with_M0=agree_m0({r: S.seba_sets(v_, r)[0] for r in RANKS}, idx),
                                      overlap_with_M0={r: ov(vecs[idx, 1:r], v_[:, 1:r], r) for r in RANKS})
    out['I'] = I
    out['conventions'] = dict(
        lambda_P='non-trivial spectrum of the non-lazy Pbar (2 lambda_M - 1 for lazy levels; native for laziness none)',
        overlap='mean cos^2 of principal angles, constant removed; chance = (r-1)/(N-1) for the rows compared',
        agreement='stats.agreement_with_null: declared counts r; N7 permutation bands, whole matching repeated',
        S1=f'{n_sub} subsets of 80%, the same for every level; compared with the level\'s own full-set labels',
        n3=f'{n_n3} N3 draws of the level\'s own operators; excess = real lambda_P - band q95')
    return out


@app.function(image=image, **CPU_UNIT, timeout=3 * 3600, retries=0, volumes=VOLS)
def s1_cells():
    """S1 for every single cell at the final checkpoint (PLAN section 4)."""
    import numpy as np
    import sources as S
    import variants as V
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = 's1_cells_64000' + PART_VERSION; out = work / f'{name}.json'
    c = cached(work, name, [out])
    if c:
        return c
    fam_lab, fam_vecs, fam_vals = stored_m0('layer', FINAL)
    ps = pairs(12)
    res = {SITES[si]: V.subsample_block([load_raw(FINAL, si)], full=(fam_vals[ps.index((si, si))], fam_vecs[ps.index((si, si))]))
           for si in range(12)}
    out.write_text(_dumps(res))
    return finished(work, name, start)


@app.function(image=image, cpu=8, memory=65536, timeout=6 * 3600, max_containers=5, retries=0, volumes=VOLS)
def n10k_unit(rid: str):
    """Factor I at N = 10,000 (all CIFAR-10 test images, final checkpoint, float32 raw Grams
    promoted to float64): the default construction with k = 15 and k = 75 (proportional),
    compared with the probe-set M0 on the 2,000 probe images."""
    import numpy as np
    import torch
    import sources as S
    from stats import raw_to_views, agreement_with_null, subspace_overlap
    from coherent import symmetric_balance
    torch.set_num_threads(8); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = f'n10k_{rid}' + PART_VERSION; out = work / f'{name}.json'
    c = cached(work, name, [out])
    if c:
        return c
    family, key, a, b, cells = _rep_block(rid)
    assert family == 'layer' and key == FINAL
    with np.load(work / 'raw10k_index.npz') as z:
        pos = z['probe_positions']
    fam_lab, fam_vecs, fam_vals = stored_m0('layer', FINAL)
    pi = pairs(12).index((a, b))
    res = {}
    for k in (15, 75):
        ms = []
        for s_, si in cells:
            C = torch.as_tensor(raw_to_views(np.load(work / f'raw10k_flat_{FINAL}_{si}.npy').astype(np.float64))['C'])
            ms.append(S._n(S.lazy(symmetric_balance(S.knn_logkernel(C, k)), .5)))
            del C
        mb = np.mean(ms, 0); del ms
        vals, vecs, info = S.top_eigs(mb, S.N_VECS, project_constant=True, dense=False)
        V_ = S.with_constant(vecs)
        res[f'k{k}'] = dict(lambda_M=vals, eig=info, per_r={
            r: dict(agreement_with_null(fam_lab[pi][RANKS.index(r)], S.seba_sets(V_, r)[0][pos], r, r, seed=r),
                    overlap_on_probe=subspace_overlap(fam_vecs[pi][:, 1:r], vecs[pos, :r - 1])['mean_cos2'],
                    overlap_chance=(r - 1) / (len(pos) - 1))
            for r in RANKS})
    out.write_text(_dumps(res))
    return finished(work, name, start, cpu=8, mem=64)


@app.function(image=image, **CPU_UNIT, timeout=4 * 3600, max_containers=5, retries=0, volumes=VOLS)
def n6_unit(r: int, n_draws: int = 1000):
    import nulls as NL
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = f'n6_haar_r{r}'; out = work / f'{name}.json'
    c = cached(work, name, [out])
    if c:
        return c
    out.write_text(_dumps(NL.n6_haar(N, r, n_draws, seed=r)))
    return finished(work, name, start)


# ---------------------------------------------------------------- timing and fixture

@app.function(image=image, **CPU_UNIT, timeout=3 * 3600, retries=0, volumes=VOLS)
def timing(n_perm: int = N_PERM):
    """Per-source cost on 5 blocks of the final checkpoint (all inputs from the volume)."""
    import numpy as np
    import torch
    import sources as S
    import nulls as NL
    import variants as V
    from schema_v2 import block_record
    _threads(); vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    name = 'timing'
    fam_lab, fam_vecs, fam_vals = stored_m0('layer', FINAL)
    with np.load(apub / 'probe.npz') as p:
        classes = p['labels'].astype(np.int64)
    ps = pairs(12)
    rows = []
    for bid, step, a, b in TIMING_BLOCKS:
        cells = block_cells('layer', step, a, b)
        keys = [f'{s}_{si}' for s, si in cells]
        t0 = time.time()
        grams = [torch.as_tensor(load_gram(*c_)) for c_ in cells]
        ops = [torch.as_tensor(load_op(*c_)) for c_ in cells]
        row = dict(block=bid, cells=len(cells), load_s=time.time() - t0)
        pi = ps.index((a, b))
        t0 = time.time(); vals, vecs = S.mbar_modes(ops, 9); row['M0_dense_eig_s'] = time.time() - t0
        t0 = time.time(); rec, _ = block_record(ops, a - a, b - a); row['block_record_s'] = time.time() - t0
        row['M0_matches_stored_labels'] = float((rec['seba_label'] == fam_lab[pi]).mean())
        t0 = time.time()
        mb = np.mean([S._n(m) for m in ops], 0)
        v2, U2, info = S.top_eigs(mb, 8, dense=False); row['M0_iterative_eig_s'] = time.time() - t0
        row['iterative_info'] = info
        row['iterative_vs_dense_min_cos'] = float(__import__('stats').subspace_overlap(vecs[:, 1:9], U2)['min_cos'])
        cache = {}
        single = {s: ps.index((s, s)) for s in range(a, b + 1)}
        _fill_block(cache, keys, fam_lab, fam_vecs, fam_vals, pi, [(keys[s - a], single[s]) for s in range(a, b + 1)])
        tm = {}
        t0 = time.time()
        block_sources(grams, ops, keys, fam_lab[pi], fam_vecs[pi], fam_vals[pi], classes, cache,
                      S.CHEAP + ['L3', 'N3'], n_perm=n_perm, timing=tm)
        row['all_sources_s'] = time.time() - t0
        row['per_source_s_5ranks'] = {k: float(np.sum(v)) for k, v in tm.items()}
        Rs = [load_raw(s, si) for s, si in cells]
        t0 = time.time(); g, _ = NL.n2b_grams(grams, seed=0); row['N2b_grams_s'] = time.time() - t0
        t0 = time.time(); o = S.default_ops(g); row['default_ops_s'] = time.time() - t0
        t0 = time.time(); NL.spectral_summary(o, classes=classes); row['null_summary_s'] = time.time() - t0
        t0 = time.time(); NL.n2a_grams(Rs, seed=0); row['N2a_grams_s'] = time.time() - t0
        t0 = time.time(); V.subsample_block(Rs, n_sub=1, full=(vals, vecs)); row['S1_one_subset_s'] = time.time() - t0
        t0 = time.time(); V.fit_extend(Rs, np.arange(1500), ranks=[3, 8]); row['S2_one_split_2ranks_s'] = time.time() - t0
        t0 = time.time(); NL.n5_ops(grams[:1], seed=0); row['N5_one_cell_s'] = time.time() - t0
        rows.append(row)
        print(json.dumps(row, default=_json_default), flush=True)
    t0 = time.time(); NL.n6_haar(N, 8, 10); n6 = (time.time() - t0) / 10
    res = dict(rows=rows, n6_seba_r8_per_draw_s=n6, n_perm=n_perm, threads=4)
    (work / 'timing.json').write_text(_dumps(res))
    (pub / 'timing.json').write_text(_dumps(res))
    return finished(work, name, start)


def _fixture_check(layer0, umap0):
    """Regenerate data/_fixture's synthetic activations (schema_v2.fixture: seed 41) and check
    that M0 reproduces its stored labels. The regenerated x is first verified against the
    fixture's stored layout (umap = x[:, :2]), so a torch RNG change cannot pass silently."""
    import numpy as np
    import torch
    import sources as S
    from coherent import centred_cosine_gram
    from schema_v2 import RANKS as RK
    torch.manual_seed(41)
    n = 2000
    planted = torch.arange(n) % 10
    x = torch.randn(10, 64)[planted] * 1.3 + .8 * torch.randn(n, 64)
    same_x = bool(np.array_equal(x[:, :2].numpy().astype('f4'), umap0[0]))
    c = centred_cosine_gram(x)
    ops = S.default_ops([c])
    rep = dict(regenerated_activations_match=same_x, ranks={})
    for ri, r in enumerate(RK):
        lab = S.SOURCES['M0']([c], r, ops=ops, cache={}, cell_keys=['fx'])['labels']
        rep['ranks'][r] = float((lab == layer0[ri]).mean())
    rep['all_equal'] = same_x and all(v == 1.0 for v in rep['ranks'].values())
    return rep


def fixture_check_local():
    import numpy as np
    with np.load(CC / 'data/_fixture/layer_0.npz') as z:
        layer0 = z['seba_label'][0, 0].astype(np.int64)
    return _fixture_check(layer0, np.load(CC / 'data/_fixture/umap_0.npy'))


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def fixture_check():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    vol.reload(); pub, work, apub, awork = paths(); start = time.time()
    with np.load('/root/fixture/layer_0.npz') as z:
        layer0 = z['seba_label'][0, 0].astype(np.int64)
    rep = _fixture_check(layer0, np.load('/root/fixture/umap_0.npy'))
    (work / 'fixture_check.json').write_text(_dumps(rep))
    return finished(work, 'fixture_check', start, cpu=2, mem=8, **{'all_equal': rep['all_equal']})


# ---------------------------------------------------------------- coordinator

def assemble(pub, work):
    """Public summary files from the per-unit parts (cheap; idempotent)."""
    import sources as S
    reps = {}
    for rid, family, key, a, b in REPS:
        reps[rid] = {t: json.loads((work / f'{part_name(rid, t)}.json').read_text())
                     for t in ('nulls', 'stability', 'linear', 'ablate') if (work / f'{part_name(rid, t)}.json').is_file()}
    (pub / 'reps.json').write_text(_dumps(reps))
    n6 = {r: json.loads((work / f'n6_haar_r{r}.json').read_text()) for r in RANKS if (work / f'n6_haar_r{r}.json').is_file()}
    (pub / 'n6_haar.json').write_text(_dumps(n6))
    extra = {}
    if (work / f's1_cells_64000{PART_VERSION}.json').is_file():
        extra['s1_cells_64000'] = json.loads((work / f's1_cells_64000{PART_VERSION}.json').read_text())
    extra['n10k'] = {rid: json.loads((work / f'n10k_{rid}{PART_VERSION}.json').read_text()) for rid, *_ in REPS
                     if (work / f'n10k_{rid}{PART_VERSION}.json').is_file()}
    (pub / 'stability_extra.json').write_text(_dumps(extra))
    notes = (Path('/root/cb/NOTES_A.md').read_text() if Path('/root/cb/NOTES_A.md').is_file() else '')
    meta = dict(tag=TAG, schema_version=SCHEMA_VERSION, based_on=TAG_IN, kernel_rule=0, seba_r=RANKS, n=N, sources=S.STATUS,
                agree_fields=AGREE_FIELDS, max_fields=MAX_FIELDS, n_classes=N_CLASSES, n_perm=N_PERM,
                tiers_done=sorted(p.stem.replace('sources_', '') for p in pub.glob('sources_*.npz')),
                reps={rid: dict(family=f, key=k, a=a, b=b) for rid, f, k, a, b in REPS},
                timing='timing.json', notes=notes)
    (pub / 'meta_c.json').write_text(_dumps(meta, indent=1))
    vol.commit()


@app.function(image=image, cpu=.25, memory=1024, timeout=20 * 3600, volumes=VOLS)
def coordinate(stage: str = 'smoke', cap_usd: float = 1.5, soft_usd: float = None):
    """Runs the stages in order. Spend control: before each step, the completed-work estimate
    plus that step's projected cost (PROJECTED, from timing.json) must stay under cap_usd;
    and once the estimate passes soft_usd with steps still outstanding, the run pauses."""
    import numpy as np
    pub, work, apub, awork = paths()
    log = work / 'coordinator.log'
    soft = cap_usd if soft_usd is None else soft_usd

    def say(text):
        with log.open('a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + text + '\n')
        print(text, flush=True); vol.commit()

    def spent():
        vol.reload()
        return sum(json.loads(p.read_text()).get('estimated_usd', 0) for p in work.glob('timing_*.json'))

    def guard(next_step=None):
        usd = spent(); say(f'estimated completed compute ${usd:.3f} (hard cap ${cap_usd}, pause above ${soft})')
        if usd > cap_usd:
            raise RuntimeError(f'Stop: estimated spend ${usd:.2f} exceeds the hard cap ${cap_usd}')
        if next_step is not None:
            if usd > soft:
                raise RuntimeError(f'Pause: estimated spend ${usd:.2f} passed ${soft} with {next_step} outstanding')
            if usd + PROJECTED[next_step] > cap_usd:
                raise RuntimeError(f'Pause: ${usd:.2f} + projected ${PROJECTED[next_step]} for {next_step} would pass the hard cap ${cap_usd}')
            say(f'starting {next_step} (projected ${PROJECTED[next_step]})')

    def safe(calls):
        out = []
        for c_ in calls:
            try:
                out.append(c_.get())
            except Exception as e:
                say(f'unit failed: {type(e).__name__}: {str(e)[:400]}'); out.append(None)
        return out

    if stage == 'followup':   # schema 1.1: cap_usd is the budget for THIS stage, added to what is already spent
        cap_usd = spent() + cap_usd
        soft = cap_usd
    guard()
    if stage == 'followup':
        guard('followup')
        say('followup (schema 1.1): reps stability/linear/ablate, s1_cells, n10k')
        calls = [reps_unit.spawn(r[0], t) for r in REPS for t in REPS_V2] + [s1_cells.spawn()]
        calls += [n10k_unit.spawn(r[0]) for r in REPS if r[1] == 'layer' and r[2] == FINAL and r[4] - r[3] < 3]
        safe(calls); guard()
    if stage == 'smoke':
        say('smoke: extract_raw(64000) + fixture check, then timing')
        calls = [extract_raw.spawn(FINAL), fixture_check.spawn()]
        res = safe(calls); say(f'smoke extraction/fixture: {json.dumps(res, default=str)[:1500]}'); guard()
        if res[0] is not None:
            say(f'timing: {json.dumps(timing.remote(), default=str)[:600]}')
        guard()
    if stage in ('extract', 'all'):
        guard('extract')
        safe([extract_raw.spawn(s) for s in STEPS] + [seed1.spawn(), extract_10k.spawn()])
        safe([pixels.spawn()])
        say('EXTRACTION DONE: raw_flat/raw_gap_<step>_<si>.npy, s1_raw_*_64000_<si>.npy, raw10k_flat_64000_<si>.npy, raw10k_index.npz under /work/r20s0c')
        guard()
    if stage in ('sources', 'all'):
        import sources as S
        for ti, tier in enumerate(TIERS):
            guard(f'sources_tier{ti}')
            say(f'sources tier {ti}: {tier}')
            safe([sources_unit.spawn(f, i, S.CHEAP, True) for f, i in tier])   # L3 everywhere: ~2 s/block (timing.json)
            vol.reload()   # see the units' writes before measuring the budget
            size = sum(p.stat().st_size for p in pub.glob('sources_*.npz')) / 1e6
            say(f'sources files total {size:.1f} MB')
            if size > 60:
                say('explorer data budget (60 MB) reached; later tiers skipped'); break
        guard()
    if stage in ('reps', 'all'):
        guard('reps')
        say('representative blocks: nulls, stability, linear, ablations; N6 Haar; S1 cells; N = 10,000')
        calls = [reps_unit.spawn(r[0], t) for r in REPS for t in ('nulls', 'stability', 'linear', 'ablate')]
        calls += [n6_unit.spawn(r) for r in RANKS] + [s1_cells.spawn()]
        calls += [n10k_unit.spawn(r[0]) for r in REPS if r[1] == 'layer' and r[2] == FINAL and r[4] - r[3] < 3]
        safe(calls); guard()
    if stage in ('assemble', 'sources', 'reps', 'all', 'followup'):
        assemble(pub, work)
        say('assembled meta_c.json, reps.json, n6_haar.json, stability_extra.json')
    say(f'{stage} complete; estimated spend ${spent():.3f}')
    return dict(stage=stage, estimated_usd=spent())


def validate_local(root=None):
    """Check a fetched data/r20s0c against SCHEMA.md; returns a report (raises nothing)."""
    import numpy as np
    root = Path(root) if root else CC / 'data' / TAG
    problems, notes = [], []
    def strict(path):
        # strict parse: a bare NaN / Infinity / -Infinity token raises (the words inside strings are fine)
        return json.loads(path.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    for jf in sorted(root.glob('*.json')):
        try:
            strict(jf)
        except Exception as e:
            problems.append(f'{jf.name}: {type(e).__name__} {e}')
    meta = strict(root / 'meta_c.json') if (root / 'meta_c.json').is_file() else None
    if meta is None:
        problems.append('meta_c.json missing')
    else:
        if meta['agree_fields'] != AGREE_FIELDS or meta['max_fields'] != MAX_FIELDS:
            problems.append('meta_c field lists differ from measure_b')
    grid = ['L1', 'L2', 'L3', 'G1', 'G1n', 'G2', 'C2', 'C3', 'N3']
    tiers = {}
    for f in sorted(root.glob('sources_*.npz')):
        fam, key = f.stem.split('_')[1], f.stem.split('_')[2]
        P = 78 if fam == 'layer' else 36
        with np.load(f) as z:
            for src in grid:
                if src not in z.files:
                    problems.append(f'{f.name}: {src} missing'); continue
                a = z[src]
                if a.shape != (P, 5, N) or a.dtype != np.int8:
                    problems.append(f'{f.name}: {src} {a.shape} {a.dtype}')
                for ri, r in enumerate(RANKS):
                    if a[:, ri].min() < -1 or a[:, ri].max() >= r:
                        problems.append(f'{f.name}: {src} r={r} labels out of range')
            agree = [k for k in z.files if k.startswith('agree_')]
            want = {'agree_' + s for s in grid + ['M0', 'C1first', 'C1mid', 'C1last', 'I1']}
            if fam == 'layer':
                want.add('agree_I2')
                if key == str(FINAL):
                    want.add('agree_I3')
            if not want <= set(agree):
                problems.append(f'{f.name}: missing {sorted(want - set(agree))}')
            for k in agree:
                if z[k].shape != (P, 5, len(AGREE_FIELDS)):
                    problems.append(f'{f.name}: {k} shape {z[k].shape}')
                mk = 'maxnull_' + k[6:]
                if mk not in z.files or z[mk].shape != (5, len(MAX_FIELDS), 2):
                    problems.append(f'{f.name}: {mk} missing or wrong shape')
            m0 = z['agree_M0']
            if not np.allclose(m0[:, :, AGREE_FIELDS.index('ami_all')], 1, atol=1e-5):
                problems.append(f'{f.name}: M0 self-agreement not 1')
            ov = z['agree_L1'][:, :, AGREE_FIELDS.index('overlap_mean_cos2')]
            if not (np.isfinite(ov).all() and (ov >= -1e-6).all() and (ov <= 1 + 1e-6).all()):
                problems.append(f'{f.name}: L1 overlap outside [0, 1]')
            tiers[f.stem] = dict(blocks=P, mb=round(f.stat().st_size / 1e6, 2))
    for name, shape in (('I1_pixels.npz', (5, N)), ('I3_seed1_layer_64000.npz', (78, 5, N))):
        if not (root / name).is_file():
            problems.append(f'{name} missing'); continue
        with np.load(root / name) as z:
            k = 'M0_input' if name.startswith('I1') else 'seba_label'
            if z[k].shape != shape:
                problems.append(f'{name}: {k} shape {z[k].shape}')
    if meta is not None and meta.get('schema_version') != SCHEMA_VERSION:
        problems.append(f'meta_c schema_version {meta.get("schema_version")} != {SCHEMA_VERSION}')
    if (root / 'reps.json').is_file():
        reps = strict(root / 'reps.json')
        for rid, *_ in REPS:
            st = reps.get(rid, {}).get('stability', {})
            if st and 'mj_null_q95' not in st['S1'][0]['per_r']['3']:
                problems.append(f'reps.json {rid}: S1 rows lack nulls')
            if st and 'mj_null_q95' not in st['S2']['M0'].get('3', {}):
                problems.append(f'reps.json {rid}: S2 rows lack nulls')
            lin = reps.get(rid, {}).get('linear', {})
            if lin and 'method_dirichlet_under_cells_P' not in lin:
                problems.append(f'reps.json {rid}: linear lacks _P Dirichlet')
            if lin and lin.get('cka_perm_null') is not None and lin.get('cka_debiased_perm_null') is None:
                problems.append(f'reps.json {rid}: no debiased CKA null')
            ab = reps.get(rid, {}).get('ablate', {}).get('ablations', {})
            for lv in ('knn5', 'knn15', 'perplexity100'):
                d = ab.get('A', {}).get(lv, {})
                if 'error' in d:
                    continue
                if 'S1' not in d or 'n3' not in d or 'ami_joint_null_q95' not in d['agreement_with_M0']['3']:
                    problems.append(f'reps.json {rid}: ablation {lv} lacks S1 / n3 / nulls')
        for rid, *_ in REPS:
            got = set(reps.get(rid, {}))
            if got != {'nulls', 'stability', 'linear', 'ablate'}:
                problems.append(f'reps.json {rid}: tasks {sorted(got)}')
            nl = reps.get(rid, {}).get('nulls')
            if nl:
                for k in ('N2a', 'N2b', 'N3', 'N4'):
                    if len(nl.get(k, [])) != 20:
                        problems.append(f'reps.json {rid}: {k} has {len(nl.get(k, []))} draws')
                if nl['real'].get('laziness') != 'half':
                    problems.append(f'reps.json {rid}: real laziness {nl["real"].get("laziness")}')
            ab = reps.get(rid, {}).get('ablate', {}).get('ablations', {})
            rd = ab.get('D', {}).get('retained_diagonal', {})
            if rd and 'error' not in rd and rd.get('laziness') != 'none':
                problems.append(f'reps.json {rid}: retained_diagonal laziness {rd.get("laziness")}')
            errs = [f'{fac}.{lv}' for fac, d in ab.items() if isinstance(d, dict)
                    for lv, v in d.items() if isinstance(v, dict) and 'error' in v]
            if errs:
                notes.append(f'{rid}: ablation levels recorded as error: {errs}')
            if (root / f'rep_{rid}_nulls_draws.npz').is_file():
                with np.load(root / f'rep_{rid}_nulls_draws.npz') as z:
                    if set(z.files) != {'N2a', 'N2b', 'N3', 'N4', 'N5'} or any(z[k].shape != (5, N) for k in z.files):
                        problems.append(f'rep_{rid}_nulls_draws.npz: keys/shapes')
            else:
                problems.append(f'rep_{rid}_nulls_draws.npz missing')
    else:
        problems.append('reps.json missing')
    for name in ('n6_haar.json', 'stability_extra.json', 'timing.json'):
        if not (root / name).is_file():
            problems.append(f'{name} missing')
    if (root / 'n6_haar.json').is_file() and len(strict(root / 'n6_haar.json')) != 5:
        problems.append('n6_haar.json: not 5 ranks')
    size = sum(p.stat().st_size for p in root.glob('sources_*.npz')) / 1e6
    return dict(ok=not problems, problems=problems, notes=notes, sources_mb=round(size, 1), files=tiers)


if __name__ == '__main__':
    args = sys.argv[1:]; recdir = HERE / 'runs'; recdir.mkdir(exist_ok=True)
    cmd = args[0] if args else 'deploy'
    if cmd == 'deploy':
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif cmd == 'launch':
        stage = args[1] if len(args) > 1 else 'smoke'; cap = float(args[2]) if len(args) > 2 else 1.5
        soft = float(args[3]) if len(args) > 3 else None
        call = modal.Function.from_name(APP, 'coordinate').spawn(stage, cap, soft)
        rec = dict(stage=stage, cap_usd=cap, soft_usd=soft, call_id=call.object_id, launched=time.ctime())
        (recdir / f'b-{stage}.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))
    elif cmd == 'status':
        stage = args[1] if len(args) > 1 else 'smoke'
        rec = json.loads((recdir / f'b-{stage}.json').read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec['call_id']).get(timeout=0), default=str))
        except TimeoutError:
            print('running')
        try:
            print(b''.join(vol.read_file(f'work/{TAG}/coordinator.log')).decode()[-3000:])
        except Exception as e:
            print(type(e).__name__)
    elif cmd == 'fetch':
        subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force', 'cm-cifar-coherent',
                        f'analysis/{TAG}', str(CC / 'data')], check=True)
    elif cmd == 'validate':
        print(json.dumps(validate_local(args[1] if len(args) > 1 else None), indent=1))
    elif cmd == 'fixture-local':
        print(json.dumps(fixture_check_local()))
    else:
        raise SystemExit(__doc__)
