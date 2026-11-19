"""Unit U1: streaming snapshot decode (one pass per lag), streaming Grams, predictions, raw vectors for the trees,
and the U1 controls (C1a, C1b, C3, P3, zero-norm scan). SPEC sections 2, 3, 7, 8 (items 1-3).

New module. Reused by import, unchanged: `mb_model` (the HF decode step with v1's cache handling: `Decoder`, `Hooks`,
`sequence`, `cache_position` via `mb_construction`, `assert_slow_path`, `load_model`, `parameter_sha`, and C1a/C1b:
`decode_views(c1b=True)`, `decode_residuals`, `c1a`, `_rel`), `mb_cells.explicit_concat_gram` (P3's reference).
The HF `MambaCache` facts in `../mamba_coherent/README.md` apply: the state carried forward after reading sequence
index u is `conv_states[..., 1:]` and `ssm_states`; `cache_position = max(u, 3)`.

WORK files of a checkpoint (directory work/<checkpoint>/):
    items.npz              snippet, t, order (storage order), redraws          (both checkpoints; identical)
    point_lag{tau}.npy     float32 [N, layers, D]: layers 1..24 at tau = 0, (8, 16, 24) at tau > 0
    state_lag{tau}_L{l}.npy float64 [N, N] state Grams: every l at tau = 0, l in (8, 16, 24) at tau > 0
    whole_lag{tau}.npy     float64 [N, N]
    trained only: bc.npy (float64 [N, N] Bhattacharyya coefficients of the next-token distributions), pred.npy
        (float64 [N, N] the `pred` reference Gram), top.npz (top-5 next-token ids and probabilities; top1),
        bag.npy / recent.npy (float64 [N, D]), cons_{c,z}{m}.npy (float32 [N, dim], the 48 raw constituents at lag 0,
        for T2)
    u1_<checkpoint>.json   controls, timings, de-duplication record, zero-norm scan
Lags: trained {0, 1, 2, 4, 8, 16}; init {0}.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import mc_construction as K
import mc_items as I
import mc_stats as ST
import mb_construction as MB
import mb_model as MM
import mb_cells as MC
import lm_cells as C


class SnapDecoder(MM.Decoder):
    """v1's Decoder; `step` also keeps the backbone's final hidden state (post norm_f) for the predictions."""

    def step(self, ids, u):
        self.hooks.grab.clear()
        pos = torch.tensor([MB.cache_position(u, self.k)], device=self.device)
        out = self.model.backbone(input_ids=ids[:, None].to(self.device), cache_params=self.cache, use_cache=True,
                                  cache_position=pos)
        self.last = out.last_hidden_state[:, -1]
        return dict(self.hooks.grab)


class Snapshot:
    """Preallocated float32 buffers of one pass: point [N, Lp, D], c [N, 24, 3E], z [N, 24, ES], sqrtp [N, V]."""

    def __init__(self, n, cfg, point_layers, device, constituents=True, predictions=False):
        e, s, d = cfg.intermediate_size, cfg.state_size, cfg.hidden_size
        nl = cfg.num_hidden_layers
        kw = dict(dtype=torch.float32, device=device)
        self.point_layers = list(point_layers)
        self.point = torch.empty((n, len(point_layers), d), **kw)
        self.c = torch.empty((n, nl, e * (cfg.conv_kernel - 1)), **kw) if constituents else None
        self.z = torch.empty((n, nl, e * s), **kw) if constituents else None
        self.sqrtp = torch.empty((n, cfg.vocab_size), **kw) if predictions else None
        self.top = torch.empty((n, 5), dtype=torch.long, device=device) if predictions else None
        self.topp = torch.empty((n, 5), **kw) if predictions else None
        self.filled = torch.zeros(n, dtype=torch.bool, device=device)

    def free(self):
        for k in ('point', 'c', 'z', 'sqrtp', 'top', 'topp'):
            setattr(self, k, None)


def snapshot_pass(model, tokens, items, lag, device, point_layers=K.LAYERS, constituents=True, predictions=False,
                  batch=1000, log=print):
    """One decode of every snippet (SPEC 8 item 1): at step u copy the residuals (every layer in `point_layers`),
    the carried state of every layer and (optionally) sqrt of the next-token distribution of each item with
    t + lag = u into its row of the buffers. Snippets are decoded `batch` at a time from an empty cache."""
    cfg = model.config
    seq = MM.sequence(tokens)
    snip, u_item = np.asarray(items['snippet']), np.asarray(items['t']) + lag
    assert u_item.max() <= MB.SEQ
    n = len(snip)
    snap = Snapshot(n, cfg, point_layers, device, constituents, predictions)
    dtype = next(model.parameters()).dtype
    hooks = MM.Hooks(model)
    t0 = time.time()
    try:
        with torch.no_grad():
            for b0 in range(0, len(seq), batch):
                b1 = min(b0 + batch, len(seq))
                inb = np.flatnonzero((snip >= b0) & (snip < b1))
                if not len(inb):
                    continue
                by_u = {}
                for k in inb:
                    by_u.setdefault(int(u_item[k]), []).append(k)
                dec = SnapDecoder(model, hooks, b1 - b0, dtype, device)
                for u in range(max(by_u) + 1):
                    h = dec.step(seq[b0:b1, u], u)
                    if u not in by_u:
                        continue
                    sel = np.asarray(by_u[u])
                    rows = torch.as_tensor(snip[sel] - b0, device=device)
                    dest = torch.as_tensor(sel, device=device)
                    for li, l in enumerate(snap.point_layers):
                        snap.point[dest, li] = h[l][rows].float()
                    if constituents:
                        for m in range(1, cfg.num_hidden_layers + 1):
                            cm, zm = dec.state(m)
                            snap.c[dest, m - 1] = cm[rows].reshape(len(sel), -1).float()
                            snap.z[dest, m - 1] = zm[rows].reshape(len(sel), -1).float()
                    if predictions:
                        hl = dec.last[rows]
                        logits = model.lm_head(hl.to(model.lm_head.weight.dtype)).double()
                        logp = torch.log_softmax(logits, -1)
                        snap.sqrtp[dest] = torch.exp(0.5 * logp).float()
                        tp, ti = torch.topk(logp, 5, dim=-1)
                        snap.top[dest], snap.topp[dest] = ti, tp.exp().float()
                    snap.filled[dest] = True
                del dec
    finally:
        hooks.close()
    assert bool(snap.filled.all()), 'snapshot pass left items unfilled'
    log(f'  pass lag {lag}: {time.time() - t0:.0f}s')
    return snap


def direct_snapshot(model, tokens, snippet, u, device):
    """Reference for the tests: decode one snippet alone (batch 1) up to sequence index u and read the residuals
    of every layer, the carried state of every layer and the final hidden state."""
    seq = MM.sequence(np.asarray(tokens)[[snippet]])
    hooks = MM.Hooks(model)
    try:
        dec = SnapDecoder(model, hooks, 1, next(model.parameters()).dtype, device)
        with torch.no_grad():
            for k in range(u + 1):
                h = dec.step(seq[:, k], k)
        states = [dec.state(m) for m in range(1, model.config.num_hidden_layers + 1)]
        return dict(point={l: h[l][0].clone() for l in h},
                    c=[s[0][0].reshape(-1).clone() for s in states], z=[s[1][0].reshape(-1).clone() for s in states],
                    last=dec.last[0].clone())
    finally:
        hooks.close()


# --------------------------------------------------------------------------------------------- controls
def c3_check(snap, items, v1_act):
    """C3 (SPEC 7): every lag-0 point vector at L4..L24 and t in {16, 32, 64} equals v1's stored value (v1 item =
    snippet id, act [layer (4..24), position (4..128), item, D]) to relative 1e-6. Fails closed on NaN."""
    snip, t = np.asarray(items['snippet']), np.asarray(items['t'])
    worst, count, by = 0.0, 0, {}
    for tt in K.C3_POSITIONS:
        sel = np.flatnonzero(t == tt)
        if not len(sel):
            continue
        pi = MB.POSITIONS.index(tt)
        for li, l in enumerate(MB.LAYERS):
            ref = torch.from_numpy(np.asarray(v1_act[li, pi])[snip[sel]])
            ours = snap.point[torch.as_tensor(sel, device=snap.point.device), snap.point_layers.index(l)].cpu()
            r = MM._rel(ref, ours)
            by[f'L{l}_t{tt}'] = r
            worst = K.worst_of([worst, r])
            count += len(sel)
    return dict(max_rel_err=worst, compared=count, by_cell=by, tolerance=K.C3_TOL,
                **{'pass': bool(count > 0 and worst < K.C3_TOL)},
                definition='max over items with t in {16, 32, 64} and layers 4..24 of |v1 - ours| / |v1| (point view, '
                           'lag 0); v1 = mamba_coherent work/m130/act_trained.npy')


def p3_check(snap, n_layers, device, size=K.P3_SUBSAMPLE, seed=K.P3_SEED):
    """P3 (SPEC 7): on a 500-item subsample, the streamed state Grams (every layer) and the whole-state Gram equal the
    standard pipeline on the explicit concatenation of the standard-pipeline-normalised constituents scaled by
    1/sqrt(#constituents) (`mb_cells.explicit_concat_gram`), to 1e-12."""
    size = min(size, len(snap.c))
    idx = np.sort(np.random.RandomState(seed).choice(len(snap.c), size, replace=False))
    it = torch.as_tensor(idx, device=snap.c.device)
    vg = ST.ViewGrams(size, device)
    dev, cons = {}, []
    for m in range(1, n_layers + 1):
        c, z = snap.c[it, m - 1], snap.z[it, m - 1]
        g = vg.layer(c, z)
        ref = MC.explicit_concat_gram([c, z])
        dev[f'state_L{m}'] = float((ref - g).abs().max())
        cons += [c, z]
    ref = MC.explicit_concat_gram(cons)
    dev['whole'] = float((ref - vg.whole()).abs().max())
    worst = K.worst_of(dev.values())                     # inf if any deviation is NaN or inf
    return dict(subsample=size, seed=seed, deviations=dev, max_dev=worst, tolerance=K.P3_TOL,
                **{'pass': bool(all(math.isfinite(v) and v < K.P3_TOL for v in dev.values()))})


def c1_checks(model, tokens, device, log=print, chunk32=100, chunk64=50):
    """v1's C1b (resume at t = 64 from the carried state, float32, to 1e-12) and C1a (decode vs prefill, float64
    parameters and cache, 1e-4), run unchanged on the 1,000 snippets (`mb_model`)."""
    out = MM.decode_views(model, tokens, device, c1b=True, scan=False, positions=[], log=log)
    res = MM.decode_residuals(model, tokens, device, torch.float32)
    act32 = np.stack([res[l].permute(1, 0, 2).cpu().numpy() for l in MB.LAYERS])
    del res
    c1a = MM.c1a(model, tokens, act32, device, chunk32, chunk64, log)
    # v1's c1a folds its per-layer errors with Python's max, which can drop a NaN; re-decide fail-closed here.
    vals = list(c1a['float64_by_layer'].values())
    c1a['max_rel_err'] = K.worst_of(vals)
    c1a['pass'] = bool(all(math.isfinite(v) and v < MB.C1A_TOL for v in vals))
    c1b = out['C1b']
    parts = [c1b['resid'], c1b['conv'], c1b['ssm']]
    c1b['max_rel_err'] = K.worst_of(parts)
    c1b['pass'] = bool(c1b['steps'] == MB.SEQ - MB.RESUME_T and all(math.isfinite(v) and v < MB.C1B_TOL
                                                                     for v in parts))
    return c1b, c1a


# ----------------------------------------------------------------------------------------------- U1
def _save(path, x):
    np.save(path, x.cpu().numpy() if torch.is_tensor(x) else x)


def run_u1(model, checkpoint, probe, work, device, log=print, v1_act=None, batch=1000, lags=None, c1=True,
           chunk32=100, chunk64=50):
    """U1 for one checkpoint: writes the WORK files above and u1_<checkpoint>.json; raises after writing the record
    if a control fails or a zero-norm item exists. `v1_act`: v1's act_trained.npy (array or memmap) for C3."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    final = checkpoint == K.FINAL
    model.eval().to(device)
    cfg = model.config
    tokens = np.asarray(probe.tokens)
    items = I.sample_items(tokens)
    I.check_items(items, tokens)
    np.savez(work / 'items.npz', snippet=items['snippet'], t=items['t'], order=items['order'])
    n = len(items['t'])
    lags = (K.LAGS if final else [0]) if lags is None else lags
    info = dict(checkpoint=checkpoint, construction_fingerprint=K.DECODE_FINGERPRINT, parameter_sha=MM.parameter_sha(model),
                model_config={k: getattr(cfg, k, None) for k in MM.MODEL_CONFIG_KEYS}, n=n, lags=lags,
                dedup=dict(redraws=items['redraws'], redraw_items=items['redraw_items'][:200]),
                torch_version=torch.__version__, zero_norm={}, timings={})
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    start = time.time()
    failure = None
    try:
        for lag in lags:
            t0 = time.time()
            layers = K.LAYERS if lag == 0 else K.LAG_LAYERS
            snap = snapshot_pass(model, tokens, items, lag, device, layers, predictions=final and lag == 0,
                                 batch=batch, log=log)
            info['timings'][f'pass_{lag}'] = time.time() - t0
            for li, l in enumerate(layers):
                bad = C.zero_norm_items(snap.point[:, li])
                if len(bad):
                    info['zero_norm'][f'point_L{l}_lag{lag}'] = bad.tolist()
            _save(work / f'point_lag{lag}.npy', snap.point)
            if lag == 0 and final and v1_act is not None:
                info['C3'] = c3_check(snap, items, v1_act)
                log(f"  C3 max rel err {info['C3']['max_rel_err']:.2e} over {info['C3']['compared']} item-layers")
            if lag == 0 and final:
                info['P3'] = p3_check(snap, cfg.num_hidden_layers, device)
                log(f"  P3 Gram identity {info['P3']['max_dev']:.2e}")
                for m in range(1, cfg.num_hidden_layers + 1):
                    _save(work / f'cons_c{m}.npy', snap.c[:, m - 1])
                    _save(work / f'cons_z{m}.npy', snap.z[:, m - 1])
                info['timings']['raw_vectors'] = time.time() - t0 - info['timings'][f'pass_{lag}']
            t1 = time.time()
            vg = ST.ViewGrams(n, device)
            for m in range(1, cfg.num_hidden_layers + 1):
                try:
                    g = vg.layer(snap.c[:, m - 1], snap.z[:, m - 1])
                except C.DegenerateItems as e:
                    info['zero_norm'][f'state_L{m}_lag{lag}'] = e.items
                    raise
                if lag == 0 or m in K.LAG_LAYERS:
                    _save(work / f'state_lag{lag}_L{m}.npy', g)
                del g
            _save(work / f'whole_lag{lag}.npy', vg.whole())
            del vg
            info['timings'][f'grams_{lag}'] = time.time() - t1
            if snap.sqrtp is not None:
                t1 = time.time()
                q = snap.sqrtp.double()
                _save(work / 'bc.npy', q @ q.T)
                del q
                _save(work / 'pred.npy', ST.view_gram(snap.sqrtp, device))
                np.savez(work / 'top.npz', top1=snap.top[:, 0].cpu().numpy(), top5=snap.top.cpu().numpy(),
                         top5_p=snap.topp.cpu().numpy())
                info['timings']['predictions'] = time.time() - t1
            snap.free()
            del snap
            if device != 'cpu':
                torch.cuda.empty_cache()
                info.setdefault('peak_gpu_gb', {})[f'lag{lag}'] = torch.cuda.max_memory_allocated() / 2 ** 30
            log(f'  lag {lag} done ({time.time() - start:.0f}s)')
        if final:
            emb = model.backbone.embeddings.weight.detach().float().cpu().numpy()
            for name, fn in (('bag', I.bag_vectors), ('recent', I.recent_vectors)):
                x = fn(items, tokens, emb)
                bad = C.zero_norm_items(x)
                if len(bad):
                    info['zero_norm'][name] = bad.tolist()
                np.save(work / f'{name}.npy', x)
            if c1:
                t1 = time.time()
                info['C1b'], info['C1a'] = c1_checks(model, tokens, device, log, chunk32, chunk64)
                info['timings']['c1'] = time.time() - t1
    except C.DegenerateItems as e:
        failure = f'zero-norm: {e}'
    info['seconds'] = time.time() - start
    _write(work / f'u1_{checkpoint}.json', info)
    if failure:
        raise RuntimeError(failure)
    assert not info['zero_norm'], f"zero-norm items: {info['zero_norm']}"
    if final:
        assert info['P3']['pass'], f"P3 failed: {info['P3']['max_dev']}"
        if v1_act is not None:
            assert info['C3']['pass'], f"C3 failed: {info['C3']['max_rel_err']}"
        if c1:
            assert info['C1b']['pass'], f"C1b failed: {info['C1b']}"
            assert info['C1a']['pass'], f"C1a failed: {info['C1a']['max_rel_err']}"
    return info


def _write(path, obj):
    K.write_json(path, obj)
