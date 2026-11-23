"""Diffusion wavelets (wavelets.py) for every single snapshot of r20s0b, kernel rule 0, on Modal CPUs.

Commands:  pilot [step site]  |  launch  |  fetch
Reads /out/work/r20s0b/operators_<step>_<site>.npy[rule 0]; writes /analysis/r20s0b/wavelets/dw_<step>_<site>.npz.
Stored per snapshot: the level table (j, n_j, n_{j+1}, wavelet count, median wavelet participation ratio), |R_kk|,
the eigenvalues of M, and dense float32 functions: the first CAP wavelets of each level in pivot order (pivoted QR ranks by residual norm in
scaling coordinates, not by localization: this is a subset, not the most localized ones), and the scaling functions Phi_j of each level j >= 1 with n_j <= CAP (always the coarsest level).
The functions are not sparse in these snapshots (pilot: median ~1,370 of 2,000 entries above 1/254 of the max),
so storing all ~2,000 per snapshot would cost ~340 MB on the site.
"""
from pathlib import Path
import json
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
TAG = 'r20s0b'
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
NSITE = 12
RULE = 0
EPS = 1e-4
CAP = 64
LOCAL = HERE / 'data' / TAG / 'wavelets'

image = (modal.Image.debian_slim(python_version='3.12').pip_install('numpy', 'scipy')
         .env({'PYTHONPATH': '/root/cc'})
         .add_local_file(str(HERE / 'wavelets.py'), remote_path='/root/cc/wavelets.py'))
app = modal.App('cm-cifar-coherent-wavelets')
vol = modal.Volume.from_name('cm-cifar-coherent')


@app.function(image=image, cpu=4, memory=8192, timeout=1800, max_containers=24, volumes={'/out': vol})
def unit(step: int, site: int, eps: float = EPS):
    import numpy as np
    from wavelets import diffusion_wavelets, sign_fix, sparsify
    t0 = time.time()
    M = np.array(np.load(f'/out/work/{TAG}/operators_{step}_{site}.npy', mmap_mode='r')[RULE], dtype=np.float64)
    L = diffusion_wavelets(M, eps=eps)
    lam = np.linalg.eigvalsh((M + M.T) / 2)[::-1]
    lev, fun, cols = [], [], []
    # fun columns: level, kind (0 phi, 1 psi), centre, rank within its level (pivot order), participation ratio
    for l in L:
        j = l['j']
        pr_psi = 1 / (l['psi'] ** 4).sum(0) if l['psi'].shape[1] else np.zeros(0)
        groups = [(1, l['psi'], l['psi_c'])]
        if j >= 1 and (l['n'] <= CAP or l is L[-1]):
            groups.insert(0, (0, l['phi'], l['phi_c']))
        for kind, F, C in groups:
            F = sign_fix(F[:, :CAP])
            for k in range(F.shape[1]):
                f = F[:, k]
                fun.append((j, kind, int(C[k]), k, float(1 / (f ** 4).sum())))
                cols.append(f.astype(np.float32))
        lev.append((j, l['n'], l['r'], l['n'] - l['r'],
                    float(np.median(pr_psi)) if len(pr_psi) else np.nan))
    rd = np.full((len(L), len(M)), np.nan, np.float32)
    for a, l in enumerate(L):
        rd[a, :len(l['rdiag'])] = l['rdiag']
    out = Path(f'/out/analysis/{TAG}/wavelets'); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'dw_{step}_{site}.npz', levels=np.array(lev, np.float64), rdiag=rd,
                        fun=np.array(fun, np.float64), F=np.stack(cols, 1), lam=lam.astype(np.float64),
                        eps=eps, cap=CAP)
    vol.commit()
    return dict(step=step, site=site, secs=time.time() - t0, levels=[[int(x) for x in r[:4]] for r in lev],
                nfun=len(fun))


@app.local_entrypoint()
def main(cmd: str = 'pilot', step: int = 64000, site: int = 6):
    if cmd == 'pilot':
        print(json.dumps(unit.remote(step, site), indent=1))
    elif cmd == 'launch':
        units = [(s, i) for s in STEPS for i in range(NSITE)]
        rows = list(unit.starmap(units))
        (HERE / 'runs').mkdir(exist_ok=True)
        (HERE / 'runs' / 'wavelets-launch.json').write_text(json.dumps(rows, indent=1))
        print(f'{len(rows)} units; {sum(r["secs"] for r in rows) / 3600:.2f} container hours; '
              f'{sum(r["nfun"] for r in rows):,} functions stored')
    elif cmd == 'fetch':
        import subprocess
        LOCAL.mkdir(parents=True, exist_ok=True)
        subprocess.run(['modal', 'volume', 'get', '--force', 'cm-cifar-coherent', f'/analysis/{TAG}/wavelets/',
                        str(LOCAL.parent)], check=True)
    else:
        sys.exit(f'unknown command {cmd}')
