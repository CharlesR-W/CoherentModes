"""Detached gradient-kernel and float64 comparison jobs (SPEC_DOT.md).

Deploy: modal deploy kernels_modal.py
Commands: launch pilot|kernels|analysis [stage budget USD]; status STAGE; fetch
"""
from pathlib import Path
import json
import math
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
APP = 'cm-cifar-coherent-kernels'
STEPS = (0, 64, 256, 1024, 2560, 6400, 25600, 64000)
SITES = ('stem', *(f'b{i}' for i in range(1, 10)), 'pool', 'logits')
REFS = ('GGN-cos', 'GGN-dot', 'NTK-cos', 'NTK-dot', 'loss-gradient-cos')
TAGS = dict(cos='r20s0b', dot='r20s0d')
OUT = Path('/out/work/kernels')
PUBLIC = Path('/out/analysis/r20s0b')
RECORD = HERE / 'runs/kernels-modal.json'
CPU, MEM = .0000131, .00000222
# USD/s: A100-40GB per measure_v2, conservative reservation for A100-80GB.
GPU40, GPU80 = .000583, .00075
KERNEL_TIMEOUT, ANALYSIS_TIMEOUT = 600, 450
KERNEL_RATE = GPU80 + 2 * CPU + 16 * MEM
ANALYSIS_RATE = GPU40 + 2 * CPU + 12 * MEM
vol = modal.Volume.from_name('cm-cifar-coherent')
zoo = modal.Volume.from_name('crw-modelzoo-train').read_only()
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy', 'umap-learn')
         .env({'PYTHONPATH': '/root/src:/root/cc', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src'))
for module in ('coherent', 'seba', 'sweep_math', 'schema_v2', 'construction'):
    image = image.add_local_file(str(HERE / f'{module}.py'), remote_path=f'/root/cc/{module}.py')
app = modal.App(APP)


def centre_gram(g):
    """Feature centring without access to features: HKH with H=I-11ᵀ/N."""
    g = g.double()
    return g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()


def row_cosine(g):
    norm = g.diagonal().clamp_min(0).sqrt()
    median = norm.median()
    if not bool(median > 0) or bool((norm < 1e-8 * median).any()):
        raise ValueError('Gradient view has near-zero centred feature norm')
    return (g / norm[:, None] / norm[None, :]).clamp(-1, 1)


def hessian_sqrt(logits):
    """Symmetric PSD square root of the softmax cross-entropy Hessian."""
    import torch
    p = logits.double().softmax(-1)
    lam, u = torch.linalg.eigh(torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2))
    return (u * lam.clamp_min(0).sqrt().unsqueeze(-2)) @ u.mT


def jacobians(model, x, batch=16):
    import torch
    from torch.func import functional_call, jacrev, vmap
    params = {k: p.detach() for k, p in model.named_parameters() if p.requires_grad}
    buffers = {k: b.detach() for k, b in model.named_buffers()}

    def logits(p, b, im):
        z = functional_call(model, (p, b), (im.unsqueeze(0),)).squeeze(0)
        return z, z

    differentiate = vmap(jacrev(logits, has_aux=True), in_dims=(None, None, 0))
    j = z_all = None
    for at in range(0, len(x), batch):
        derivatives, z = differentiate(params, buffers, x[at:at+batch])
        n, c = z.shape
        if j is None:
            j = x.new_empty((len(x), c, sum(p.numel() for p in params.values())))
            z_all = z.new_empty((len(x), c))
        # Copy each parameter directly into the contiguous buffer, no extra full J.
        offset = 0
        for k, p in params.items():
            j[at:at+n, :, offset:offset+p.numel()] = derivatives[k].reshape(n, c, -1)
            offset += p.numel()
        z_all[at:at+n] = z
    return j, z_all


def jacobian_grams(j, logits, tile=4096):
    """Float64 Gram accumulation in parameter tiles, keeping only one full J."""
    import torch
    n = len(j)
    sqrt_h = hessian_sqrt(logits)
    ntk = torch.zeros((n, n), device=j.device, dtype=torch.float64)
    ggn = torch.zeros_like(ntk)
    for at in range(0, j.shape[-1], tile):
        t = j[:, :, at:at+tile].to(torch.float64, memory_format=torch.contiguous_format)
        v = t.reshape(n, -1)
        ntk.addmm_(v, v.T)
        w = (sqrt_h @ t).reshape(n, -1)
        ggn.addmm_(w, w.T)
        del t, v, w
    return ggn, ntk


def setup_gpu():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensor_file(path, index=None):
    import numpy as np
    import torch
    a = np.load(path, mmap_mode='r')
    return torch.tensor(a if index is None else a[index], device='cuda', dtype=torch.float64)


def load_probe():
    import numpy as np
    with np.load(PUBLIC / 'probe.npz') as p:
        return p['idx'], p['labels']


def loss_reference(step):
    """The archived cosine Gram is the starting view for this comparison."""
    return tensor_file(PUBLIC / f'gradgram_{step}.npy')


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False, separators=(',', ':')) + '\n')
    tmp.replace(path)


def timing(unit, start, rate, **extra):
    seconds = time.monotonic() - start
    row = dict(unit=unit, seconds=seconds, estimated_usd=seconds * rate, **extra)
    save_json(OUT / f'timing_{unit}.json', row)
    vol.commit()
    print(json.dumps(row), flush=True)
    return row


def rule_zero(score, k=15, *, raw=False):
    """Union graph on the highest scores; return the directed neighbour indices too."""
    import torch
    d = -score if raw else 1 - score
    d.diagonal().fill_(torch.inf)
    ix = d.topk(k, largest=False, dim=1).indices
    mask = torch.zeros_like(score, dtype=torch.bool).scatter_(1, ix, True)
    weights = torch.full_like(score, 1e-6).masked_fill_(mask | mask.T, 1.)
    weights.diagonal().zero_()
    return weights.log(), ix


def lazy_walk(log_k):
    from coherent import symmetric_balance
    m = symmetric_balance(log_k) * .5
    m.diagonal().add_(.5)
    return m


def mode_pairs(m):
    from coherent import block_modes
    return block_modes([m], 33, shared_stationary=True)


def reference_inputs(step):
    ggn = centre_gram(tensor_file(OUT / f'ggn_{step}.npy'))
    ntk = centre_gram(tensor_file(OUT / f'ntk_{step}.npy'))
    loss = centre_gram(loss_reference(step))
    return dict(zip(REFS, (row_cosine(ggn), ggn, row_cosine(ntk), ntk, row_cosine(loss))))


@app.function(image=image, gpu='A100-80GB', cpu=2, memory=16384, timeout=KERNEL_TIMEOUT,
              max_containers=7, retries=0, volumes={'/out': vol, '/zoo': zoo})
def checkpoint(step: int):
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model
    from coherent import dot_effective_cosine
    from sweep_math import perplexity_logkernel

    start = time.monotonic()
    setup_gpu()
    vol.reload()
    OUT.mkdir(parents=True, exist_ok=True)
    unit = f'kernel_{step}'
    products = [OUT / f'{k}_{step}.npy' for k in ('ggn', 'ntk', 'lossgram')]
    modes_path = OUT / f'reference_modes_{step}.npz'
    completion = OUT / f'timing_{unit}.json'
    if completion.is_file() and all(p.is_file() for p in (*products, modes_path)):
        return json.loads(completion.read_text())
    ck = load_checkpoint(Path(f'/out/runs/resnet20-relu-seed0/checkpoints/step_{step}.pt'))
    model = build_model(ck['config'])
    model.load_state_dict(ck['model_state'])
    model = model.eval().cuda()
    data = build_data(ck['config'], Path('/zoo/data'), 'cpu')
    idx, _ = load_probe()
    x = data.normalize(data.test_x[idx]).cuda()
    torch.cuda.reset_peak_memory_stats()
    t = time.monotonic()
    j, logits = jacobians(model, x)
    torch.cuda.synchronize()
    jac_seconds = time.monotonic() - t
    print(step, 'Jacobian', tuple(j.shape), jac_seconds, flush=True)
    t = time.monotonic()
    ggn, ntk = jacobian_grams(j, logits)
    shape = list(j.shape)
    del j
    torch.cuda.synchronize()
    gram_seconds = time.monotonic() - t
    # Persist the original loss-gradient cosine (the stored f16 approximant),
    # not a second label-dependent derivative calculation.
    loss = tensor_file(PUBLIC / f'gradgram_{step}.npy')
    for p, g in zip(products, (ggn, ntk, loss)):
        if not bool(g.isfinite().all()):
            raise ValueError(f'Non-finite Gram for {p}')
        np.save(p, g.float().cpu().numpy())
    del ggn, ntk, loss
    modes = {}
    for name, gram in reference_inputs(step).items():
        score = dot_effective_cosine(gram / gram.diagonal().mean()) if name.endswith('-dot') else gram
        for rule in (0, 1):
            try:
                log_k = rule_zero(score)[0] if rule == 0 else perplexity_logkernel(score, 100)[0]
                lam, vec = mode_pairs(lazy_walk(log_k))
            except (ValueError, RuntimeError, AssertionError) as e:
                raise ValueError(f'{step}, {name}, rule {rule}: {e}') from e
            modes[f'{name}_{rule}_vals'] = lam.cpu().numpy()
            modes[f'{name}_{rule}_vecs'] = vec.cpu().numpy()
    np.savez_compressed(modes_path, **modes)
    return timing(unit, start, KERNEL_RATE, jacobian_shape=shape, jacobian_seconds=jac_seconds,
                  gram_seconds=gram_seconds, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())


def work_path(form):
    return OUT.parent / TAGS[form]


def site_file(form, step, si, kind, rule=None):
    return tensor_file(work_path(form) / f'{kind}_{step}_{si}.npy', rule)


def span_pair(a, b, r):
    from schema_v2 import comparison
    av, aw = a
    bv, bw = b
    return dict(agreement=comparison(aw[:, 1:], bw[:, 1:], r),
                gap_a=float(av[r] - av[r+1]), gap_b=float(bv[r] - bv[r+1]))


def cka(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()))


def class_span(labels, device):
    import torch
    # The first nine centred indicators span the entire ten-class contrast space.
    x = torch.nn.functional.one_hot(torch.as_tensor(labels, device=device).long(), 10).double()[:, :-1]
    return torch.linalg.qr(x - x.mean(0)).Q


def partition(eig, rank):
    from seba import seba, seba_labels
    return seba_labels(seba(eig[1][:, :rank].cpu(), rank)).numpy()


def matched_jaccard(a, size_a, b, size_b):
    """Disjoint hard sets, ignoring -1 (unassigned), with empty-union scores zero."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    sa = np.bincount(a[a >= 0], minlength=size_a)
    sb = np.bincount(b[b >= 0], minlength=size_b)
    valid = (a >= 0) & (b >= 0)
    inter = np.bincount(a[valid] * size_b + b[valid], minlength=size_a * size_b).reshape(size_a, size_b)
    jac = inter / np.maximum(sa[:, None] + sb[None, :] - inter, 1)
    i, j = linear_sum_assignment(-jac)
    return dict(mean=float(jac[i, j].mean()), matched=jac[i, j].tolist(), sets_a=i.tolist(), sets_b=j.tolist(),
                set_sizes_a=sa.tolist(), set_sizes_b=sb.tolist(),
                unassigned_a=int((a < 0).sum()), unassigned_b=int((b < 0).sum()))


def class_validation(eig, labels, basis):
    lam, v = eig
    spans = {}
    for r in (4, 10):
        overlap = float((basis.T @ v[:, 1:r+1]).square().sum())
        spans[r] = dict(agreement=overlap / min(r, basis.shape[1]), class_coverage=overlap / basis.shape[1],
                        mode_coverage=overlap / r, eigengap=float(lam[r] - lam[r+1]))
    return dict(span=spans, class_jaccard=matched_jaccard(partition(eig, 10), 10, labels, 10),
                seba_gap=float(lam[9] - lam[10]))


def hubness(ix, log_k):
    import torch
    n = len(ix)
    degree = torch.bincount(ix.flatten(), minlength=n).double()
    centered = degree - degree.mean()
    std = centered.square().mean().sqrt()
    # Stable index tie-break for equal indegrees at the top-one-percent boundary.
    hubs = degree.argsort(descending=True, stable=True)[:math.ceil(n * .01)]
    weights = log_k.exp()
    return dict(max_indegree=int(degree.max()), skew=float((centered / std).pow(3).mean()) if std > 0 else 0.,
                outgoing_normalised_mass=float((weights / weights.sum(1, keepdim=True))[:, hubs].sum() / n),
                total_affinity_mass=float(weights[:, hubs].sum() / weights.sum()))


def neighbour_overlap(a, b):
    return float((a[:, :, None] == b[:, None, :]).any(2).double().mean())


def block_comparison(ms, family, name, indexes, all_pairs=True, single=None):
    from schema_v2 import pairs
    rows = []
    full = (0, len(indexes)-1)
    ps = pairs(len(indexes)) if all_pairs else [full]
    spectra = {}
    for a, b in ps:
        for rule in (0, 1):
            ev = {}
            for form in TAGS:
                cells = ms[form][rule][a:b+1]
                if a == b and single is not None:
                    ev[form] = single[a][rule][form]
                else:
                    # Same arithmetic definition and stationary isolation as the v2 average blocks.
                    import torch
                    ev[form] = mode_pairs(torch.stack(cells).mean(0) if len(cells) > 1 else cells[0])
            if (a, b) == full:
                spectra[rule] = ev
            left, right = ev['cos'], ev['dot']
            span, sets = {}, {}
            for r in (4, 8):
                span[r] = span_pair(left, right, r)
                sets[r] = dict(**matched_jaccard(partition(left, r), r, partition(right, r), r),
                               gap_a=float(left[0][r-1] - left[0][r]), gap_b=float(right[0][r-1] - right[0][r]))
            rows.append(dict(family=family, name=name, a=a, b=b, first=indexes[a], last=indexes[b],
                             rule=rule, span=span, seba=sets))
    return rows, spectra


def operator_grid(cells):
    return {form: {rule: [site_file(form, step, si, 'operators', rule) for step, si in cells]
                   for rule in (0, 1)} for form in TAGS}


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=12288, timeout=ANALYSIS_TIMEOUT,
              max_containers=8, retries=0, volumes={'/out': vol})
def compare_layer(step: int, all_pairs: bool):
    import numpy as np
    import torch
    from sweep_math import perplexity_logkernel

    start = time.monotonic()
    setup_gpu()
    vol.reload()
    unit = f'layer_{step}_{"all" if all_pairs else "full"}'
    path = OUT / f'{unit}.json'
    done = OUT / f'timing_{unit}.json'
    if path.is_file() and done.is_file():
        return json.loads(done.read_text())
    _, labels = load_probe()
    basis = class_span(labels, 'cuda')
    with np.load(OUT / f'reference_modes_{step}.npz') as f:
        ref_ev = {rule: {ref: tuple(torch.tensor(f[f'{ref}_{rule}_{suffix}'], device='cuda')
                                  for suffix in ('vals', 'vecs')) for ref in REFS} for rule in (0, 1)}
    gram_ref = {name: centre_gram(g) for name, g in reference_inputs(step).items()}
    validation_ref = [dict(step=step, rule=rule, reference=ref, validation=class_validation(ev, labels, basis))
                      for rule in (0, 1) for ref, ev in ref_ev[rule].items()]
    ms = operator_grid([(step, si) for si in range(len(SITES))])
    scales = json.loads((work_path('dot') / f'kappa_{step}.json').read_text())['dot_scale_mean_sq_norm']
    single, site_ref, mips, site_ev = [], [], [], []
    for si, site in enumerate(SITES):
        scores = {form: site_file(form, step, si, 'gram') for form in TAGS}
        # For a centred site Gram G, HK_eff H = G since K_eff = 1 - D.
        grams = {form: centre_gram(g) for form, g in scores.items()}
        site_cka = {ref: {form: cka(gram, g) for form, gram in grams.items()} for ref, g in gram_ref.items()}
        nearest = {form: rule_zero(g) for form, g in scores.items()}
        ev = {rule: {form: mode_pairs(ms[form][rule][si]) for form in TAGS} for rule in (0, 1)}
        site_ev.append(ev)
        for rule in (0, 1):
            eig = ev[rule]
            diag = {form: hubness(ix, log_k if rule == 0 else perplexity_logkernel(scores[form], 100)[0])
                    for form, (log_k, ix) in nearest.items()}
            single.append(dict(step=step, site=site, rule=rule, dot_scale=scales[si], hubness=diag,
                               neighbour_overlap=neighbour_overlap(nearest['cos'][1], nearest['dot'][1]),
                               eigenvalues={form: e[0][1:11].tolist() for form, e in eig.items()},
                               span={r: span_pair(eig['cos'], eig['dot'], r) for r in (4, 10)},
                               validation={form: class_validation(e, labels, basis) for form, e in eig.items()}))
            for ref, e in ref_ev[rule].items():
                cos, dot = ({r: span_pair(eig[form], e, r) for r in (4, 10)} for form in TAGS)
                ck = site_cka[ref]
                site_ref.append(dict(step=step, site=site, rule=rule, reference=ref,
                                     cos=cos, dot=dot, delta={r: dot[r]['agreement']-cos[r]['agreement'] for r in (4, 10)},
                                     cka=dict(**ck, delta=ck['dot']-ck['cos'])))
        if step in (2560, 64000):
            log_k, ix = rule_zero(grams['dot'], raw=True)
            eig = mode_pairs(lazy_walk(log_k))
            mips.append(dict(step=step, site=site, rule=0, hubness=hubness(ix, log_k),
                             neighbour_overlap={form: neighbour_overlap(ix, v[1]) for form, v in nearest.items()},
                             span={form: {r: span_pair(eig, e, r) for r in (4, 10)} for form, e in ev[0].items()}))
    blocks, full = block_comparison(ms, 'layer', step, SITES, all_pairs=all_pairs, single=site_ev)
    block_ref = []
    for rule, modes in full.items():
        for ref, ev in ref_ev[rule].items():
            cos, dot = ({r: span_pair(modes[form], ev, r) for r in (4, 10)} for form in TAGS)
            block_ref.append(dict(step=step, first=SITES[0], last=SITES[-1], rule=rule, reference=ref,
                                  cos=cos, dot=dot, delta={r: dot[r]['agreement']-cos[r]['agreement'] for r in (4, 10)}))
    save_json(path, dict(single=single, site_reference=site_ref, blocks=blocks, mips=mips,
                         reference_validation=validation_ref, block_reference=block_ref))
    return timing(unit, start, ANALYSIS_RATE, layer_blocks=len(blocks), single_rows=len(single))


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=12288, timeout=ANALYSIS_TIMEOUT,
              max_containers=8, retries=0, volumes={'/out': vol})
def compare_time(si: int):
    start = time.monotonic()
    setup_gpu()
    vol.reload()
    unit = f'time_{SITES[si]}'
    path = OUT / f'{unit}.json'
    done = OUT / f'timing_{unit}.json'
    if path.is_file() and done.is_file():
        return json.loads(done.read_text())
    ms = operator_grid([(step, si) for step in STEPS])
    blocks, _ = block_comparison(ms, 'time', SITES[si], STEPS)
    save_json(path, blocks)
    return timing(unit, start, ANALYSIS_RATE, time_blocks=len(blocks))


def run_analysis(stage, budget):
    if stage == 'analysis-pilot':
        # Measure both unit types before fan-out; the layer pilot covers one headline checkpoint.
        layer = compare_layer.spawn(64000, True)
        temporal = compare_time.spawn(0)
        return [layer.get(), temporal.get()]
    layer = json.loads((OUT / 'timing_layer_64000_all.json').read_text())
    temporal = json.loads((OUT / 'timing_time_stem.json').read_text())
    margin = 30 * ANALYSIS_RATE  # conservative per-worker startup allowance
    layer_cost = 2 * layer['estimated_usd'] + margin
    time_cost = 2 * temporal['estimated_usd'] + margin
    # Reserve the full job estimate for the mandatory second headline checkpoint,
    # and 65% of it for rows with no other layer block eigensolves.
    minimum = layer_cost * (1 + 6*.65) + 11 * time_cost
    all_pairs = 7 * layer_cost + 11 * time_cost <= budget
    if minimum > budget and not all_pairs:
        raise RuntimeError(f'Projected analysis minimum ${minimum:.3f} exceeds budget ${budget:.3f}')
    jobs = [(s, all_pairs or s in (2560, 64000)) for s in STEPS]
    times = [compare_layer.spawn(*job) for job in jobs]
    times += [compare_time.spawn(i) for i in range(len(SITES))]
    records = [call.get() for call in times]
    vol.reload()
    rows = [json.loads((OUT / f'layer_{s}_{"all" if full else "full"}.json').read_text()) for s, full in jobs]
    temporal_rows = [json.loads((OUT / f'time_{site}.json').read_text()) for site in SITES]
    def collect(key):
        return [row for group in rows for row in group[key]]
    common = dict(n=2000, steps=STEPS, sites=SITES, rules=[0, 1], spectrum_precision='float64',
                  probe='Existing label-stratified probe, 200 images/class. Labels used only for validation, except inside the pre-existing loss-gradient reference.',
                  gaps='Nontrivial rank r: lambda[r]-lambda[r+1], indexing constant at zero. SEBA rank r: lambda[r-1]-lambda[r].')
    save_json(OUT / 'compare_single.json', dict(**common, rows=collect('single'), mips=collect('mips'),
              class_span='Squared Frobenius overlap divided by the smaller rank; also report fractions of class and mode spans captured.',
              hub_mass='Top 1% of directed-15NN indegrees, with sample-index tie-break; use either row-normalised pre-Sinkhorn affinity or total pre-Sinkhorn affinity.'))
    save_json(OUT / 'compare_blocks.json', dict(**common, rows=collect('blocks') + sum(temporal_rows, []),
                                               all_layer_blocks=all_pairs, block_operator='arithmetic mean'))
    save_json(OUT / 'compare_ggn.json', dict(**common, references=REFS, rows=collect('site_reference'),
              block_rows=collect('block_reference'), reference_validation=collect('reference_validation'),
              reference_convention='Feature-centre before cosine normalisation; linear CKA double-centres the resulting site and reference Grams.',
              loss_gradient='The archived f16 cosine Gram is treated as a Gram of unit loss-gradient features; centre these features and renormalise. This is not centring the raw unnormalised loss gradients.'))
    vol.commit()
    return dict(units=records, estimated_usd=sum(r['estimated_usd'] for r in records), all_layer_blocks=all_pairs)


@app.function(image=image, cpu=.25, memory=1024, timeout=3600, volumes={'/out': vol})
def coordinate(stage, budget):
    vol.reload()
    if stage == 'pilot':
        return checkpoint.remote(64000)
    if stage == 'kernels':
        pilot = json.loads((OUT / 'timing_kernel_64000.json').read_text())
        if 7 * (2 * pilot['estimated_usd'] + 30 * KERNEL_RATE) > budget:
            raise RuntimeError('Projected kernel fan-out exceeds the allotted budget')
        return list(checkpoint.map(STEPS[:-1]))
    if stage in ('analysis-pilot', 'analysis'):
        return run_analysis(stage, budget)
    raise ValueError(stage)


def main():
    args = sys.argv[1:]
    command = args[0]
    if command == 'deploy':
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif command == 'launch':
        stage = args[1]
        budget = float(args[2]) if len(args) > 2 else 0.
        records = json.loads(RECORD.read_text()) if RECORD.exists() else {}
        call = modal.Function.from_name(APP, 'coordinate').spawn(stage, budget)
        row = dict(stage=stage, call_id=call.object_id, launched=time.ctime(), budget_usd=budget)
        records.setdefault(stage, []).append(row)
        save_json(RECORD, records)
        print(json.dumps(row))
    elif command == 'status':
        rec = json.loads(RECORD.read_text())[args[1]][-1]
        call = modal.FunctionCall.from_id(rec['call_id'])
        try:
            print(json.dumps(call.get(timeout=0)))
        except TimeoutError:
            print('running')
    elif command == 'fetch':
        dst = HERE / 'data/kernels'
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('single', 'blocks', 'ggn'):
            filename = f'compare_{name}.json'
            subprocess.run([sys.executable, '-m', 'modal', 'volume', 'get', '--force',
                            'cm-cifar-coherent', f'work/kernels/{filename}', str(dst / filename)], check=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
