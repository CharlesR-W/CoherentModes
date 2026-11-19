"""The `bench` unit (SPEC 8 item 6): synthetic data at full N = 8,000 on the production hardware, through the real
code (`mc_stats`, `mc_calib`). No model is loaded. Every operation records wall time, peak GPU memory (allocated)
and host RSS. Synthetic inputs only; nothing here is a measurement of the network.

Operations:
    buffers      the U1 snapshot buffers at full size (point 24 x 768, c 24 x 4,608, z 24 x 24,576 per item, float32;
                 sqrt p 8,000 x 50,280) and a simulated pass (129 steps of row copies from batch-1,000 state tensors)
    gram_z/gram_c one constituent's float64 cosine Gram at z's and c's width (ViewGrams.layer arithmetic)
    bc_pred      the Bhattacharyya Gram and the `pred` Gram from sqrt p
    disk_io      a 512 MB float64 array: local disk write/read, volume write+commit and read
    cell         one cell: view Gram, kNN, Sinkhorn, shares for 7 groupings x 51, kappa with 50 conditional draws
    spectral     one spectral object: eigh (33 pairs), SEBA at all r, both roundings, Upsilon_hat, eta^2, AMI, sets
    tree_8000 / tree_2000   one streaming tree-node rebuild of a 24-cell block (incl. spill to local disk)
    umap         one layout (psi_1..16 of 8,000 items)
    calib_full   the full-size calibration check (k = 8 Zipf, s = 1, and structureless, seed 100)
"""
from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

import mc_construction as K
import mc_stats as ST
import mc_calib as CB
import lm_cells as C


def _rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return float('nan')


class Timer:
    def __init__(self, log):
        self.rows, self.log = {}, log

    def run(self, name, fn, *a, **kw):
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = fn(*a, **kw)
        if cuda:
            torch.cuda.synchronize()
        row = dict(seconds=time.time() - t0, rss_gb=_rss_gb(),
                   maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                   peak_gpu_gb=(torch.cuda.max_memory_allocated() / 2 ** 30) if cuda else None,
                   gpu_now_gb=(torch.cuda.memory_allocated() / 2 ** 30) if cuda else None)
        if isinstance(out, dict) and 'extra' in out:
            row.update(out.pop('extra'))
        self.rows[name] = row
        self.log(f'{name}: {row}')
        return out


def synthetic_labels(n, rng):
    """Groupings with the probe's shape: tokens ~ Zipf over ~3,000 ids, 1,000 snippets of 8, 5 sources, 12 classes."""
    tok = (rng.zipf(1.25, n) - 1) % 3000
    snip = rng.permutation(np.repeat(np.arange(n // 8), 8))
    src = (snip % 5)
    return dict(T=tok, Nx=(rng.zipf(1.25, n) - 1) % 3000, U=snip, A=(rng.zipf(1.3, n) - 1) % 2000, W=src,
                Cc=rng.integers(0, 12, n), Cn=rng.integers(0, 12, n))


def synthetic_points(n, d, rng, k=40, s=1.0):
    lab = rng.integers(0, k, n)
    return (s * rng.normal(size=(k, d))[lab] + rng.normal(size=(n, d))).astype(np.float32)


def run_bench(device, work, theta=None, n=K.BENCH_N, log=print, quick=False, volume_commit=None, local='/tmp'):
    """All bench operations; returns {op: timing row} plus derived numbers. `quick` (tests): small sizes."""
    rng = np.random.default_rng(0)
    T = Timer(log)
    cfg = dict(d=768, e=1536, s=16, v=50280, layers=24) if not quick else dict(d=16, e=32, s=4, v=512, layers=24)
    out = dict(n=n, device=str(device), torch=torch.__version__)
    if torch.cuda.is_available():
        out['gpu'] = torch.cuda.get_device_name()
    out['cpu_count'] = os.cpu_count()

    # --- U1 buffers and a simulated pass
    def buffers():
        kw = dict(dtype=torch.float32, device=device)
        b = dict(point=torch.empty((n, cfg['layers'], cfg['d']), **kw),
                 c=torch.empty((n, cfg['layers'], cfg['e'] * 3), **kw),
                 z=torch.empty((n, cfg['layers'], cfg['e'] * cfg['s']), **kw),
                 sqrtp=torch.empty((n, cfg['v']), **kw))
        batch = min(1000, n)
        conv = torch.randn((batch, cfg['e'], 3), **kw)
        ssm = torch.randn((batch, cfg['e'], cfg['s']), **kw)
        h = torch.randn((batch, cfg['d']), **kw)
        per = max(1, n // 97)
        for u in range(129):
            dest = torch.as_tensor(rng.choice(n, per, replace=False), device=device)
            rows = torch.as_tensor(rng.choice(batch, per, replace=False), device=device)
            for li in range(cfg['layers']):
                b['point'][dest, li] = h[rows]
                b['c'][dest, li] = conv[rows].reshape(per, -1)
                b['z'][dest, li] = ssm[rows].reshape(per, -1)
        b['c'].normal_(); b['z'].normal_(); b['sqrtp'].uniform_()
        b['sqrtp'] /= b['sqrtp'].norm(dim=1, keepdim=True)
        return b
    buf = T.run('buffers', buffers)
    out['buffer_gb'] = sum(x.numel() * 4 for x in buf.values()) / 2 ** 30

    def gram(x):
        u = ST.unit_rows_(x, device)
        g = u @ u.T
        return dict(g=g, extra=dict(width=x.shape[1]))
    g = T.run('gram_z', gram, buf['z'][:, 0])['g']
    del g
    T.run('gram_c', gram, buf['c'][:, 0])

    def bc_pred():
        q = buf['sqrtp'].double()
        bc = q @ q.T
        del q
        pred = ST.view_gram(buf['sqrtp'], device)
        return dict(bc=bc, pred=pred)
    r = T.run('bc_pred', bc_pred)
    bc = r['bc']
    del r
    for k in ('c', 'z', 'sqrtp'):
        buf[k] = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- disk and volume
    def disk_io():
        a = np.random.default_rng(1).normal(size=(n, n))
        extra = {}
        for name, root in (('local', Path(local)), ('volume', Path(work))):
            root.mkdir(parents=True, exist_ok=True)
            path = root / 'bench_io.npy'
            t0 = time.time()
            np.save(path, a)
            if name == 'volume' and volume_commit:
                volume_commit()
            extra[f'{name}_write_s'] = time.time() - t0
            t0 = time.time()
            b = np.load(path)
            extra[f'{name}_read_s'] = time.time() - t0
            assert b.shape == a.shape
            path.unlink()
        extra['bytes'] = a.nbytes
        return dict(extra=extra)
    T.run('disk_io', disk_io)

    # --- one cell
    labels = synthetic_labels(n, rng)
    x = synthetic_points(n, cfg['d'], rng)
    perms = ST.permutation_bank(n)
    within, unique = ST.within_group_perms(labels['T'])

    def cell():
        g = ST.view_gram(torch.from_numpy(x), device)
        t0 = time.time()
        p, info = ST.cell_walk(g)
        t_walk = time.time() - t0
        del g
        t0 = time.time()
        st = ST.cell_statistics(p, labels, perms)
        t_shares = time.time() - t0
        t0 = time.time()
        ks = ST.kappa_stats(p, bc, labels['U'], within)
        t_kappa = time.time() - t0
        return dict(p=p, extra=dict(walk_s=t_walk, shares_s=t_shares, kappa_s=t_kappa, row_err=info['row_error'],
                                    iterations=info['iterations'], kappa=ks['kappa']))
    p = T.run('cell', cell)['p']

    def spectral():
        bm = ST.BlockMean()
        bm.add(p)
        mbar = bm.mbar()
        t0 = time.time()
        ST.spectrum(mbar)
        t_eigh = time.time() - t0
        rec, sets, _ = ST.spectral_object(mbar, labels, perms, C.permutation_bank(n, K.N_PERM_JACC, K.JACC_SEED),
                                          desc_groupings=labels)
        return dict(rec=rec, extra=dict(eigh_s=t_eigh, ups=rec['ups'].tolist()))
    rec = T.run('spectral', spectral)['rec']
    del p

    # --- tree nodes, 24-cell block
    pts = torch.from_numpy(np.stack([synthetic_points(n, cfg['d'], np.random.default_rng(10 + l))
                                     for l in range(24)], 1)).to(device)

    def node(size):
        members = np.sort(rng.choice(n, size, replace=False)) if size < n else np.arange(n)
        spill = ST.Spill(Path(local) / f'spill{size}')
        try:
            t0 = time.time()
            pbar, mask, names = ST.node_operators_streaming([ST.VectorCell(pts[:, l], device) for l in range(24)],
                                                            members, spill, device)
            t_ops = time.time() - t0
            ST.analyse_node_streaming(pbar, members, spill, names, labels['W'], mask)
        finally:
            spill.close()
        return dict(extra=dict(operators_s=t_ops))
    T.run('tree_8000', node, n)
    T.run('tree_2000', node, min(2000, n))
    del pts

    def lay():
        import umap
        v = ST.dequantize(rec['vecs'], rec['scale'])[:, :K.N_ETA]
        umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(v)
        return {}
    T.run('umap', lay)
    del bc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if theta is not None or quick:
        res = T.run('calib_full', lambda: dict(r=CB.full_size_check(theta, device, n, log)))
        out['calib_full'] = {k: v for k, v in res['r'].items() if k not in ('planted', 'structureless')}
        out['calib_full']['planted_k'] = res['r']['planted']['k']
        out['calib_full']['structureless_k'] = res['r']['structureless']['k']
    out['ops'] = T.rows
    return out
