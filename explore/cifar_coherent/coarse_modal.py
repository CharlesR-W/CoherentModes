"""Spectral coarse-graining stage (SPEC_COARSE.md) on Modal, reading the r20s0b operators on the volume.

Commands:  deploy | pilot | launch | status | fetch
Unit = (checkpoint, rule): every single cell, every layer average and every layer chain a < b at that checkpoint.
Writes /analysis/r20s0b/coarse/coarse_<step>_<rule>.npz and a timing record under /work/r20s0b/coarse/.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
APP = 'cm-cifar-coherent-coarse'
TAG = 'r20s0b'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
NSITE = 12
K = list(range(2, 17)) + [20, 24, 32]
SEBA_R = [2, 3, 4, 6, 8, 10, 12, 16]
KLAB = list(range(2, 17))            # spectral k-means labels stored for the explorer
NSV = 129                            # singular values kept (incl. trivial)
NV = 33                              # vectors kept (trivial + 32)
METHODS = ['spectral', 'seba', 'pca', 'random', 'labels', 'pred']
SCORES = ['I_start', 'I_end', 'I_two', 'keff_start', 'keff_end', 'agree', 'agree_indep', 'agree_min_fwd',
          'agree_min_rev', 'ncut', 'maxleak', 'I_start_P', 'I_two_P', 'ncut_P', 'maxleak_P', 'I_start_sd']
GPU_RATE, CPU_RATE, MEM_RATE = .000583, .0000131, .00000222

image = (modal.Image.debian_slim(python_version='3.12').pip_install('torch', 'numpy', 'scipy')
         .env({'PYTHONPATH': '/root/cc', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}))
for name in ('coarse.py', 'seba.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/cc/' + name)
app = modal.App(APP)
vol = modal.Volume.from_name('cm-cifar-coherent')
VOLS = {'/out': vol}


def pairs(n):
    return [(a, b) for a in range(n) for b in range(a, n)]


def quant(v):
    import numpy as np
    scale = np.abs(v).max(0).clip(1e-300)
    return np.round(32767 * v / scale).astype(np.int16), scale.astype(np.float32)


def sign_fix(v):
    import torch
    idx = v.abs().argmax(0)
    s = torch.sign(v[idx, torch.arange(v.shape[1], device=v.device)]); s[s == 0] = 1
    return v * s


def avg_modes(ms, n):
    """Top-n eigenpairs of a mean of symmetric doubly stochastic cells, constant placed first exactly
    (Householder deflation, as coherent.block_modes(shared_stationary=True))."""
    import torch
    m = torch.stack(ms).mean(0) if len(ms) > 1 else ms[0]
    N = len(m)
    const = torch.ones(N, dtype=m.dtype, device=m.device) / N ** .5
    h = const.clone(); h[0] -= 1; h /= h.norm()
    mh = m @ h
    t = m - 2 * h[:, None] * mh[None, :] - 2 * mh[:, None] * h[None, :] + 4 * torch.dot(h, mh) * h[:, None] * h[None, :]
    lam, w = torch.linalg.eigh(t[1:, 1:])
    lam, w = lam.flip(0)[:n - 1], w.flip(1)[:, :n - 1]
    lifted = torch.cat([torch.zeros((1, n - 1), dtype=w.dtype, device=w.device), w])
    lifted -= 2 * h[:, None] * (h @ lifted)[None, :]
    return torch.cat([torch.ones(1, dtype=lam.dtype, device=lam.device), lam]), torch.cat([const[:, None], lifted], 1)


def subspace_check(new, stored_int16, scale, gaps):
    """Worst |cos| between our vectors 1..8 and the stored ones, over modes whose gap to neighbours > 1e-6."""
    import numpy as np
    old = stored_int16.astype(np.float64).T * scale / 32767.0          # (N, 8)
    old /= np.linalg.norm(old, axis=0, keepdims=True)
    c = np.abs((new / np.linalg.norm(new, axis=0, keepdims=True) * old).sum(0))
    sel = gaps > 1e-6
    worst = float(c[sel].min()) if sel.any() else 1.0
    qa, _ = np.linalg.qr(new); qb, _ = np.linalg.qr(old)
    sub = float(np.square(qa.T @ qb).sum() / 8)
    return worst, sub


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=16384, timeout=3 * 3600, max_containers=10, retries=0, volumes=VOLS)
def unit(step: int, rule: int, tag: str = TAG):
    import numpy as np
    import torch
    import coarse as C
    from seba import seba
    torch.backends.cuda.matmul.allow_tf32 = False
    vol.reload()
    root, work = Path('/out/analysis') / tag, Path('/out/work') / tag
    outdir, wdir = root / 'coarse', work / 'coarse'
    outdir.mkdir(exist_ok=True); wdir.mkdir(exist_ok=True)
    out = outdir / f'coarse_{step}_{rule}.npz'
    if out.exists():
        return json.loads((wdir / f'timing_{step}_{rule}.json').read_text())
    start = time.time(); dev = 'cuda'
    labels = torch.as_tensor(np.load(root / 'probe.npz')['labels']).long()
    pred = torch.as_tensor(np.load(root / f'points_{step}.npz')['pred']).long()
    pred = torch.unique(pred, return_inverse=True)[1]
    ms = [torch.tensor(np.load(work / f'operators_{step}_{si}.npy', mmap_mode='r')[rule], device=dev, dtype=torch.float64)
          for si in range(NSITE)]
    N = len(ms[0])
    stored = {}
    for f in ('layer', 'layerprodR', 'layerprodL'):
        with np.load(root / f'{f}_{step}.npz') as z:
            stored[f] = dict(vecs=z['vecs'][rule], scale=z['scale'][rule])

    # PCA baseline labels per site (rule independent; deterministic seeds).
    pca_lab = {}
    for si in range(NSITE):
        g = torch.tensor(np.load(work / f'gram_{step}_{si}.npy'), device=dev, dtype=torch.float64)
        g = g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()
        lam, w = torch.linalg.eigh((g + g.T) / 2)
        sc = w[:, -50:] * lam[-50:].clamp_min(0).sqrt()
        pca_lab[si] = {k: C.kmeans(sc, k, seed=1000 + k) for k in K}
        del g
    gen = torch.Generator().manual_seed(7 + step + rule)
    rand = {k: [torch.randperm(N, generator=gen) % k for _ in range(8)] for k in K}   # 4 start + 4 end draws

    ps = pairs(NSITE); nb = len(ps)
    res = dict(sv=np.full((nb, NSV), np.nan, np.float32), sv_P=np.full((nb, NSV), np.nan, np.float32),
               total=np.zeros(nb, np.float64), total_P=np.full(nb, np.nan),
               scores=np.full((nb, len(METHODS), len(K), len(SCORES)), np.nan, np.float32),
               lab_start=np.full((nb, len(KLAB), N), -1, np.int8), lab_end=np.full((nb, len(KLAB), N), -1, np.int8),
               part_start=np.zeros((nb, NV - 1), np.float32), part_end=np.zeros((nb, NV - 1), np.float32),
               avg_vals=np.zeros((nb, NSV), np.float32), avg_vecs=np.zeros((nb, NV - 9, N), np.int16),
               avg_scale=np.zeros((nb, NV - 9), np.float32),
               prodR_vecs=np.zeros((nb, NV - 9, N), np.int16), prodR_scale=np.zeros((nb, NV - 9), np.float32),
               prodL_vecs=np.zeros((nb, NV - 9, N), np.int16), prodL_scale=np.zeros((nb, NV - 9), np.float32),
               check=np.zeros((nb, 3, 2), np.float32))
    gaps_json = []

    def score(q, qP, cell, ls, le, row):
        row[:] = np.nan
        row[0] = C.info_onesided(q, ls); row[1] = C.info_onesided(q.T, le)
        val, t = C.info_twosided(q, ls, le); row[2] = val
        row[3] = len(torch.unique(ls)); row[4] = len(torch.unique(le))
        ma = C.matched_agreement(t)
        row[5], row[6] = ma['agreement'], ma['independent']
        row[7], row[8] = ma['forward'].min(), ma['reverse'].min()
        if cell:
            phi = C.leakage(q, ls); row[9], row[10] = float(phi.sum()), float(phi.max())
            row[11] = C.info_onesided(qP, ls); row[12] = C.info_twosided(qP, ls, ls)[0]
            phiP = C.leakage(qP, ls); row[13], row[14] = float(phiP.sum()), float(phiP.max())

    q = None
    for bi, (a, b) in enumerate(ps):
        cell = a == b
        q = ms[a] if cell else ms[b] @ q   # pairs() visits (a, a), (a, a+1), ...: chain built incrementally
        left, sv, right = C.svd_desc(q)
        res['sv'][bi] = sv[:NSV].cpu().numpy(); res['total'][bi] = C.info_total(q)
        qP = None
        if cell:
            qP = 2 * q - torch.eye(N, dtype=q.dtype, device=dev)
            res['sv_P'][bi] = torch.linalg.eigvalsh(qP).abs().sort(descending=True).values[:NSV].cpu().numpy()
            res['total_P'][bi] = C.info_total(qP)
        gr = C.gap_report(sv[:NSV]); gr.update(a=a, b=b); gaps_json.append(gr)
        res['part_start'][bi] = C.participation(right[:, 1:NV]).cpu().numpy()
        res['part_end'][bi] = C.participation(left[:, 1:NV]).cpu().numpy()
        # Extensions + agreement with stored vectors 1..8.
        av, avec = avg_modes(ms[a:b + 1], NSV)
        res['avg_vals'][bi] = av.cpu().numpy()
        for key, vec, vals, fam in (('avg', avec, av, 'layer'), ('prodR', right, sv, 'layerprodR'), ('prodL', left, sv, 'layerprodL')):
            v = sign_fix(vec[:, :NV]).cpu().numpy()
            qv, qs = quant(v[:, 9:])
            res[f'{key}_vecs'][bi], res[f'{key}_scale'][bi] = qv.T, qs
            vv = vals[:10].double().cpu().numpy()
            gaps = np.minimum(vv[0:8] - vv[1:9], vv[1:9] - vv[2:10])
            res['check'][bi, ['avg', 'prodR', 'prodL'].index(key)] = subspace_check(
                v[:, 1:9], stored[fam]['vecs'][bi], stored[fam]['scale'][bi], gaps)
        # Partitions.
        emb_s, emb_e = C.embedding(right, sv, max(K)), C.embedding(left, sv, max(K))
        seba_s = {r: C.hard_argmax(seba(right[:, :r].cpu(), r)) for r in SEBA_R}
        seba_e = seba_s if cell else {r: C.hard_argmax(seba(left[:, :r].cpu(), r)) for r in SEBA_R}
        for ki, k in enumerate(K):
            ls = C.kmeans(emb_s[:, :k - 1], k, seed=k)
            le = ls if cell else C.kmeans(emb_e[:, :k - 1], k, seed=k)
            if k in KLAB:
                res['lab_start'][bi, KLAB.index(k)] = ls.cpu().numpy(); res['lab_end'][bi, KLAB.index(k)] = le.cpu().numpy()
            score(q, qP, cell, ls.to(dev), le.to(dev), res['scores'][bi, 0, ki])
            if k in SEBA_R:
                score(q, qP, cell, seba_s[k].to(dev), seba_e[k].to(dev), res['scores'][bi, 1, ki])
            score(q, qP, cell, pca_lab[a][k], pca_lab[a][k] if cell else pca_lab[b][k], res['scores'][bi, 2, ki])
            draws = np.full((4, len(SCORES)), np.nan, np.float32)
            for d in range(4):
                rs = rand[k][d].to(dev); re = rs if cell else rand[k][4 + d].to(dev)
                score(q, qP, cell, rs, re, draws[d])
            res['scores'][bi, 3, ki] = draws.mean(0); res['scores'][bi, 3, ki, -1] = draws[:, 0].std()
            if k == 10:
                score(q, qP, cell, labels.to(dev), labels.to(dev), res['scores'][bi, 4, ki])
                score(q, qP, cell, pred.to(dev), pred.to(dev), res['scores'][bi, 5, ki])
        print(step, rule, (a, b), f'{time.time() - start:.0f}s', flush=True)
    np.savez_compressed(out, **res)
    seconds = time.time() - start
    row = dict(step=step, rule=rule, seconds=seconds, estimated_usd=seconds * (GPU_RATE + 2 * CPU_RATE + 16 * MEM_RATE),
               worst_check=float(res['check'][..., 0].min()), worst_subspace=float(res['check'][..., 1].min()))
    (outdir / f'gaps_{step}_{rule}.json').write_text(json.dumps(gaps_json))
    (wdir / f'timing_{step}_{rule}.json').write_text(json.dumps(row)); vol.commit()
    return row


@app.function(image=image, cpu=.25, memory=1024, timeout=12 * 3600, volumes=VOLS)
def coordinate(units: list, cap_usd: float = 15.0):
    rows, spent = [], 0.0
    for row in unit.starmap([tuple(u) for u in units], return_exceptions=True):
        if isinstance(row, Exception):
            print('unit failed:', repr(row), flush=True); rows.append(dict(error=repr(row))); continue
        rows.append(row); spent += row.get('estimated_usd', 0)
        print('done', row, f'spent ${spent:.2f}', flush=True)
    vol.reload(); (Path('/out/work') / TAG / 'coarse').mkdir(parents=True, exist_ok=True)
    (Path('/out/work') / TAG / 'coarse' / 'coordinator.json').write_text(json.dumps(rows))
    vol.commit()
    return dict(spent=spent, n=len(rows))

MK = list(range(2, 17))                       # partition sizes for the closure test
MMETHODS = ['spectral', 'pca', 'shared', 'random', 'labels']
SHARED_SITE = 9                               # b9: prespecified source of the shared partition
STEP_F = ['defect', 'I_fine', 'I_coarse', 'defect_chi2', 'X_fine', 'X_coarse']
TRIPLE_F = ['cmi_coarse', 'cmi_fine', 'I_next', 'H_cond']


@app.function(image=image, gpu='A100-40GB', cpu=2, memory=32768, timeout=3 * 3600, max_containers=10, retries=0, volumes=VOLS)
def markov(step: int, rule: int, tag: str = TAG):
    """Markov closure of the per-site coarse process (SPEC_COARSE.md section 6)."""
    import numpy as np
    import torch
    import coarse as C
    torch.backends.cuda.matmul.allow_tf32 = False
    vol.reload()
    root, work = Path('/out/analysis') / tag, Path('/out/work') / tag
    out = root / 'coarse' / f'markov_{step}_{rule}.npz'
    if out.exists():
        return json.loads((work / 'coarse' / f'timing_markov_{step}_{rule}.json').read_text())
    start = time.time(); dev = 'cuda'
    labels = torch.as_tensor(np.load(root / 'probe.npz')['labels']).long().to(dev)
    ms = [torch.tensor(np.load(work / f'operators_{step}_{si}.npy', mmap_mode='r')[rule], device=dev, dtype=torch.float64)
          for si in range(NSITE)]
    N = len(ms[0]); ps = pairs(NSITE)
    with np.load(root / 'coarse' / f'coarse_{step}_{rule}.npz') as z:
        lab = z['lab_start']                                   # (78, 15, N), k = 2..16
    spec = {k: [torch.as_tensor(lab[ps.index((s, s)), KLAB.index(k)]).long().to(dev) for s in range(NSITE)] for k in MK}
    pca = {k: [] for k in MK}
    for si in range(NSITE):
        g = torch.tensor(np.load(work / f'gram_{step}_{si}.npy'), device=dev, dtype=torch.float64)
        g = g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()
        lam, w = torch.linalg.eigh((g + g.T) / 2)
        sc = w[:, -50:] * lam[-50:].clamp_min(0).sqrt()
        for k in MK:
            pca[k].append(C.kmeans(sc, k, seed=1000 + k).to(dev))
        del g
    gen = torch.Generator().manual_seed(11 + step + rule)
    rnd = {k: [[(torch.randperm(N, generator=gen) % k).to(dev) for _ in range(NSITE)] for _ in range(4)] for k in MK}
    shared = {k: [spec[k][SHARED_SITE]] * NSITE for k in MK}
    cks = [(a, b) for a in range(NSITE) for b in range(a + 1, NSITE)]

    def couplings(ops):
        """Q(a -> b) = M_b ... M_{a+1} for a < b (Z_a = x, Z_b = y)."""
        qs = {}
        for a in range(NSITE - 1):
            q = ops[a + 1]; qs[a, a + 1] = q
            for b in range(a + 2, NSITE):
                q = ops[b] @ q; qs[a, b] = q
        return qs

    U = C.uniform_lazy(N, device=dev)
    worlds = [(ms, couplings(ms)), ([U] * NSITE, couplings([U] * NSITE))]

    def run(L, ops, qs):
        st = np.array([[C.closure_step(ops[s + 1], L[s], L[s + 1])[f] for f in STEP_F] for s in range(NSITE - 1)])
        tr = np.array([[C.closure_triple(ops[s + 1], ops[s + 2], L[s], L[s + 1], L[s + 2])[f] for f in TRIPLE_F]
                       for s in range(NSITE - 2)])
        one = [C.coarse_transition(ops[s + 1], L[s], L[s + 1])[0] for s in range(NSITE - 1)]
        ck = []
        for a, b in cks:
            t_true, pa = C.coarse_transition(qs[a, b], L[a], L[b])
            t = one[a]
            for s in range(a + 1, b):
                t = one[s] @ t
            ck.append(C.ck_error(t_true, t, pa))
        return st, tr, np.array(ck)

    nm, nk = len(MMETHODS), len(MK)
    res = dict(step=np.full((2, nm, nk, NSITE - 1, len(STEP_F)), np.nan),
               triple=np.full((2, nm, nk, NSITE - 2, len(TRIPLE_F)), np.nan),
               ck=np.full((2, nm, nk, len(cks), 2), np.nan), ck_pairs=np.array(cks),
               nmi=np.full((nm, nk, NSITE - 1), np.nan))
    for ki, k in enumerate(MK):
        seqs = [(0, spec[k]), (1, pca[k]), (2, shared[k])]
        if k == 10:
            seqs.append((4, [labels] * NSITE))
        for mi, L in seqs:
            res['nmi'][mi, ki] = [C.nmi(L[s].cpu(), L[s + 1].cpu()) for s in range(NSITE - 1)]
            for w, (ops, qs) in enumerate(worlds):
                res['step'][w, mi, ki], res['triple'][w, mi, ki], res['ck'][w, mi, ki] = run(L, ops, qs)
        res['nmi'][3, ki] = np.mean([[C.nmi(L[s].cpu(), L[s + 1].cpu()) for s in range(NSITE - 1)] for L in rnd[k]], 0)
        for w, (ops, qs) in enumerate(worlds):
            draws = [run(L, ops, qs) for L in rnd[k]]
            res['step'][w, 3, ki] = np.mean([d[0] for d in draws], 0)
            res['triple'][w, 3, ki] = np.mean([d[1] for d in draws], 0)
            res['ck'][w, 3, ki] = np.mean([d[2] for d in draws], 0)
        print(step, rule, 'k', k, f'{time.time() - start:.0f}s', flush=True)
    np.savez_compressed(out, **res)
    seconds = time.time() - start
    row = dict(step=step, rule=rule, seconds=seconds, estimated_usd=seconds * (GPU_RATE + 2 * CPU_RATE + 32 * MEM_RATE))
    (work / 'coarse' / f'timing_markov_{step}_{rule}.json').write_text(json.dumps(row)); vol.commit()
    return row


@app.function(image=image, cpu=.25, memory=1024, timeout=12 * 3600, volumes=VOLS)
def coordinate_markov(units: list):
    rows, spent = [], 0.0
    for row in markov.starmap([tuple(u) for u in units], return_exceptions=True):
        if isinstance(row, Exception):
            print('unit failed:', repr(row), flush=True); rows.append(dict(error=repr(row))); continue
        rows.append(row); spent += row.get('estimated_usd', 0)
        print('done', row, f'spent ${spent:.2f}', flush=True)
    vol.reload(); (Path('/out/work') / TAG / 'coarse').mkdir(parents=True, exist_ok=True)
    (Path('/out/work') / TAG / 'coarse' / 'coordinator_markov.json').write_text(json.dumps(rows))
    vol.commit()
    return dict(spent=spent, n=len(rows))


if __name__ == '__main__':
    cmd = sys.argv[1]
    rec = HERE / 'runs'
    if cmd == 'deploy':
        subprocess.run(['modal', 'deploy', __file__], check=True)
    elif cmd in ('pilot', 'launch'):
        units = [[64000, 0]] if cmd == 'pilot' else [[s, r] for s in STEPS for r in range(5) if [s, r] != [64000, 0]]
        call = modal.Function.from_name(APP, 'coordinate').spawn(units)
        r = dict(stage=f'coarse-{cmd}', call_id=call.object_id, launched=time.ctime(), units=len(units))
        (rec / f'coarse-{cmd}.json').write_text(json.dumps(r, indent=1)); print(r)
    elif cmd in ('markov-pilot', 'markov-launch'):
        units = [[64000, 0]] if cmd == 'markov-pilot' else [[s, r] for s in STEPS for r in range(5) if [s, r] != [64000, 0]]
        call = modal.Function.from_name(APP, 'coordinate_markov').spawn(units)
        r = dict(stage=cmd, call_id=call.object_id, launched=time.ctime(), units=len(units))
        (rec / f'{cmd}.json').write_text(json.dumps(r, indent=1)); print(r)
    elif cmd == 'fetch-markov':
        dest = HERE / 'data' / TAG / 'coarse'
        for s in STEPS:
            for r in range(5):
                subprocess.run(['modal', 'volume', 'get', '--force', 'cm-cifar-coherent', f'/analysis/{TAG}/coarse/markov_{s}_{r}.npz', str(dest) + '/'], check=False, capture_output=True)
        print(len(list(dest.glob('markov_*.npz'))), 'markov files')
    elif cmd == 'status':
        r = json.loads((rec / f'coarse-{sys.argv[2] if len(sys.argv) > 2 else "launch"}.json').read_text())
        try:
            print(modal.FunctionCall.from_id(r['call_id']).get(timeout=0))
        except TimeoutError:
            print('running', r)
    elif cmd == 'fetch':
        dest = HERE / 'data' / TAG / 'coarse'; dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['modal', 'volume', 'get', '--force', 'cm-cifar-coherent', f'/analysis/{TAG}/coarse/', str(dest.parent) + '/'], check=True)
