"""Unit U1 of the Mamba study: recurrent decoding, the recurrent state, Grams, C1a, C1b, the P3 Gram identity, C2.

New module (the Pythia U1 lived inside `lm_modal.py`; here it is a plain function so that `test_mb.py` runs the
same code on a tiny randomly initialised Mamba on CPU). Modal's `u1_forward` only loads the model and calls
`run_u1`.

HF transformers 4.57.6, torch path (`MambaMixer.slow_forward`); `assert_slow_path` refuses to run otherwise.
How the decode works, from the pinned source (checked 2026-11-12, `models/mamba/modeling_mamba.py`):
  - a call with `cache_params`, `use_cache=True` and a ONE-element `cache_position` takes the decode branch
    (`cache_position.shape[0] != conv_kernel`): `MambaCache.update_conv_state` rolls the (B, E, 4) buffer left by
    one column and writes the new conv input at column clamp(cache_position, 0, 3), then the conv output is
    sum(buffer * w). So after reading sequence index u the buffer holds [x_{u-3}, x_{u-2}, x_{u-1}, x_u]
    (zeros before the start), and the next step's roll discards column 0: the state carried forward is
    `conv_states[..., 1:]` (C1b fills column 0 with NaN on resume, which would poison every later value if it
    were read);
  - the cache_position must be >= 3 for the input to land in the last column (`MB.cache_position`); HF's own
    `generate` prefills with arange(0, 4) and then decodes with 4, 5, ...;
  - the SSM state is updated as ssm = dA * ssm + dB u and copied into `ssm_states[layer]` in place.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import mb_construction as MB
import mb_cells as MC
import lm_cells as C

L, P = len(MB.LAYERS), len(MB.POSITIONS)
MODEL_CONFIG_KEYS = ['model_type', 'num_hidden_layers', 'hidden_size', 'intermediate_size', 'state_size',
                     'conv_kernel', 'time_step_rank', 'vocab_size', 'residual_in_fp32', 'rms_norm',
                     'layer_norm_epsilon', 'use_conv_bias', 'use_bias', 'tie_word_embeddings', 'hidden_act',
                     'use_mambapy']


# ------------------------------------------------------------------------------------------------ model
def assert_slow_path():
    """The scan must run in HF's reference implementation: no mamba-ssm, no causal-conv1d, no `kernels` hub."""
    import transformers.models.mamba.modeling_mamba as mm
    from transformers.utils.import_utils import (is_causal_conv1d_available, is_kernels_available,
                                                 is_mamba_ssm_available)
    assert not is_mamba_ssm_available() and not is_causal_conv1d_available() and not is_kernels_available(), \
        'fast-path packages present: mamba-ssm / causal-conv1d / kernels must not be installed'
    assert mm.selective_state_update is None and mm.selective_scan_fn is None and mm.mamba_inner_fn is None
    assert mm._lazy_load_causal_conv1d() == (None, None)


def init_model(config, seed=MB.INIT_SEED):
    """`init`: torch.manual_seed(seed); MambaForCausalLM(config) on CPU in float32 (HF's `_init_weights`)."""
    from transformers import MambaForCausalLM
    torch.manual_seed(seed)
    with torch.device('cpu'):
        model = MambaForCausalLM(config)
    return model.float().eval()


def load_model(checkpoint, cache_dir=None):
    from transformers import MambaConfig, MambaForCausalLM
    if checkpoint == 'trained':
        model = MambaForCausalLM.from_pretrained(MB.MODEL, revision=MB.REVISION, cache_dir=cache_dir,
                                                 torch_dtype=torch.float32)
    elif checkpoint == 'init':
        model = init_model(MambaConfig.from_pretrained(MB.MODEL, revision=MB.REVISION, cache_dir=cache_dir))
    else:
        raise ValueError(checkpoint)
    return model.eval()


def parameter_sha(model):
    """sha256 over the float32 bytes of every parameter in name order (identifies the init draw)."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to('cpu', torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


class Hooks:
    """Forward hooks on backbone.layers[l-1] for every layer l: `grab[l]` is the block output (the raw residual
    after block l) at the LAST sequence index of the most recent call."""

    def __init__(self, model, layers=None, last_only=True):
        layers = layers or range(1, model.config.num_hidden_layers + 1)
        self.grab, self.last_only = {}, last_only
        blocks = model.backbone.layers
        self.handles = [blocks[l - 1].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            self.grab[l] = out[:, -1] if self.last_only else out
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


class Decoder:
    """Recurrent decoding, one token per step, with HF's MambaCache (decode branch from an empty cache)."""

    def __init__(self, model, hooks, batch, dtype, device):
        from transformers.models.mamba.modeling_mamba import MambaCache
        self.model, self.hooks, self.device = model, hooks, device
        self.k = model.config.conv_kernel
        self.cache = MambaCache(model.config, max_batch_size=batch, dtype=dtype, device=device)

    def step(self, ids, u):
        """Read the tokens `ids` (B,) at sequence index u; returns {layer: residual (B, D)}."""
        self.hooks.grab.clear()
        pos = torch.tensor([MB.cache_position(u, self.k)], device=self.device)
        self.model.backbone(input_ids=ids[:, None].to(self.device), cache_params=self.cache, use_cache=True,
                            cache_position=pos)
        return dict(self.hooks.grab)

    def state(self, m):
        """Layer m's (1-based) recurrent state after the last step: (conv buffer E x 3, SSM state E x S)."""
        return self.cache.conv_states[m - 1][..., 1:], self.cache.ssm_states[m - 1]

    def resume_from(self, other):
        """C1b: a FRESH cache holding only `other`'s carried state: conv columns 1..3 in their slots (column 0,
        which the next step discards, set to NaN), SSM state copied."""
        for m in range(len(self.cache.conv_states)):
            self.cache.conv_states[m].fill_(float('nan'))
            self.cache.conv_states[m][..., 1:] = other.cache.conv_states[m][..., 1:]
            self.cache.ssm_states[m].copy_(other.cache.ssm_states[m])


def _rel(a, b):
    """max over items of |a - b| / |a| (vectors flattened per item), float64."""
    a = a.reshape(len(a), -1).double(); b = b.reshape(len(b), -1).double()
    r = float(((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-300)).max())
    return r if math.isfinite(r) else math.inf    # fail closed: max(0.0, nan) would silently drop a NaN


def sequence(tokens):
    tokens = torch.as_tensor(np.asarray(tokens), dtype=torch.long)
    return torch.cat([torch.full((len(tokens), 1), MB.EOT, dtype=torch.long), tokens], 1)


def prefill_residuals(model, tokens, device, chunk):
    """HF's parallel full-sequence forward (use_cache=False): residuals at the sampled (layer, t), [L][n, P, D]."""
    seq = sequence(tokens)
    hooks = Hooks(model, MB.LAYERS, last_only=False)
    out = {l: [] for l in MB.LAYERS}
    try:
        with torch.no_grad():
            for at in range(0, len(seq), chunk):
                model.backbone(input_ids=seq[at:at + chunk].to(device), use_cache=False)
                for l in MB.LAYERS:
                    out[l].append(hooks.grab[l][:, MB.POSITIONS].detach())
    finally:
        hooks.close()
    return {l: torch.cat(out[l]) for l in MB.LAYERS}


def decode_residuals(model, tokens, device, dtype):
    """Recurrent decode, residuals only at the sampled (layer, t): [L][n, P, D]."""
    seq = sequence(tokens)
    hooks = Hooks(model, MB.LAYERS)
    try:
        dec = Decoder(model, hooks, len(seq), dtype, device)
        out = {l: [] for l in MB.LAYERS}
        with torch.no_grad():
            for u in range(seq.shape[1]):
                h = dec.step(seq[:, u], u)
                if u in MB.POSITIONS:
                    for l in MB.LAYERS:
                        out[l].append(h[l])
    finally:
        hooks.close()
    return {l: torch.stack(out[l], 1) for l in MB.LAYERS}


# ------------------------------------------------------------------------------------- the main decode
def decode_views(model, tokens, device, *, keep_parts=False, raw_path=None, c1b=False, scan=True, gram_id=False,
                 positions=MB.POSITIONS, log=print):
    """Decode probe items; at each position in `positions` read every layer's state and form the state and stack
    Grams (float64 on `device`). Returns a dict of numpy arrays and records (see `run_u1`)."""
    cfg = model.config
    nl = cfg.num_hidden_layers
    assert nl == MB.N_LAYERS, nl
    seq = sequence(tokens)
    n = len(seq)
    pp = len(positions)
    hooks = Hooks(model)
    dtype = next(model.parameters()).dtype
    dec = Decoder(model, hooks, n, dtype, device)
    resumed = Decoder(model, hooks, n, dtype, device) if c1b else None
    d_model = cfg.hidden_size
    act = np.empty((L, pp, n, d_model), np.float32)
    state = np.empty((L, pp, n, n))
    stack = np.empty((L, pp, n, n))
    parts = np.empty((L, pp, 2, n, n)) if keep_parts else None
    raw = (np.lib.format.open_memmap(raw_path, mode='w+', dtype=np.float64, shape=(pp, 2 * nl, n, n))
           if raw_path is not None else None)
    norm = np.zeros((L, seq.shape[1]))
    zero = set()
    c1b_rec = dict(resid=0.0, conv=0.0, ssm=0.0, steps=0) if c1b else None
    gid = None
    t0 = time.time()
    try:
        with torch.no_grad():
            for u in range(seq.shape[1]):
                h = dec.step(seq[:, u], u)
                if resumed is not None and u > MB.RESUME_T:
                    hb = resumed.step(seq[:, u], u)
                    c1b_rec['resid'] = max(c1b_rec['resid'], max(_rel(h[l], hb[l]) for l in h))
                    for m in range(1, nl + 1):
                        (ca, za), (cb, zb) = dec.state(m), resumed.state(m)
                        c1b_rec['conv'] = max(c1b_rec['conv'], _rel(ca, cb))
                        c1b_rec['ssm'] = max(c1b_rec['ssm'], _rel(za, zb))
                    c1b_rec['steps'] += 1
                for li, l in enumerate(MB.LAYERS):
                    norm[li, u] = float(h[l].double().norm(dim=-1).mean())
                    if scan and u >= 1:
                        zero.update(int(i) for i in C.zero_norm_items(h[l]))
                if resumed is not None and u == MB.RESUME_T:
                    resumed.resume_from(dec)
                if u not in positions:
                    continue
                pi = positions.index(u)
                for li, l in enumerate(MB.LAYERS):
                    act[li, pi] = h[l].float().cpu().numpy()
                if scan:
                    for m in range(1, nl + 1):
                        for x in dec.state(m):
                            zero.update(int(i) for i in C.zero_norm_items(x.reshape(n, -1)))
                if zero:
                    continue                      # no Grams once a zero-norm item exists; U1 fails after the scan
                g_state, g_stack, g_parts, g_raw = MC.position_grams(dec.state, n, device, nl, keep_parts=keep_parts,
                                                                      keep_raw=raw is not None)
                state[:, pi], stack[:, pi] = g_state, g_stack
                if keep_parts:
                    parts[:, pi] = g_parts
                if raw is not None:
                    for k, g in enumerate(g_raw):
                        raw[pi, k] = g
                    del g_raw
                if gram_id and u == MB.SEQ:
                    gid = MC.gram_identity(dec.state, g_state, g_stack, device,
                                           stack_layers=[MB.LAYERS[0], MB.LAYERS[-1]])
                log(f'  t={u}: grams done ({time.time() - t0:.0f}s)')
    finally:
        hooks.close()
        if raw is not None:
            raw.flush()
    if c1b_rec is not None:
        c1b_rec['max_rel_err'] = max(c1b_rec['resid'], c1b_rec['conv'], c1b_rec['ssm'])
        c1b_rec.update(tolerance=MB.C1B_TOL, resume_t=MB.RESUME_T, precision=str(dtype).replace('torch.', ''),
                       **{'pass': bool(c1b_rec['steps'] == MB.SEQ - MB.RESUME_T and
                                       c1b_rec['max_rel_err'] < MB.C1B_TOL)},
                       definition='max over items, t in 65..128, every layer, of |x_resumed - x_continuous| / '
                                  '|x_continuous| for residuals, conv buffers (columns 1..3) and SSM states; '
                                  'resumed = fresh MambaCache holding only the t = 64 state (conv column 0 NaN)')
    return dict(act=act, state=state, stack=stack, parts=parts, norm=norm, zero=sorted(zero), C1b=c1b_rec,
                gram_identity=gid)


# ------------------------------------------------------------------------------------------------ C1a
def c1a(model, tokens, act32, device, chunk32=100, chunk64=50, log=print):
    """C1a: decode vs prefill residuals at every sampled (layer, t). Gate: parameters and cache in float64 (stock
    HF arithmetic, which casts parts of the computation to float32), tolerance 1e-4. Hazard: the float32
    production decode (`act32`, [L, P, n, D]) against the float32 prefill. Leaves the model in float32."""
    def by_layer(dec, pre):
        return {f'L{l}': _rel_rows(dec[l], pre[l]) for l in MB.LAYERS}
    pre32 = prefill_residuals(model, tokens, device, chunk32)
    dec32 = {l: torch.from_numpy(np.array(act32[li])).permute(1, 0, 2) for li, l in enumerate(MB.LAYERS)}
    f32 = by_layer({l: dec32[l].to(pre32[l].device) for l in MB.LAYERS}, pre32)
    del pre32
    log(f'  C1a float32 by layer {f32}')
    model.double()
    try:
        pre64 = prefill_residuals(model, tokens, device, chunk64)
        dec64 = decode_residuals(model, tokens, device, torch.float64)
        f64 = by_layer(dec64, pre64)
        f64_t = {f't{t}': max(_rel_rows(dec64[l][:, [pi]], pre64[l][:, [pi]]) for l in MB.LAYERS)
                 for pi, t in enumerate(MB.POSITIONS)}
    finally:
        model.float()
    rel = max(f64.values())
    log(f'  C1a float64 by layer {f64}')
    return dict(max_rel_err=rel, tolerance=MB.C1A_TOL, precision='float64 parameters and cache (stock HF casts)',
                **{'pass': bool(rel < MB.C1A_TOL)}, float64_by_layer=f64, float64_by_t=f64_t,
                float32_max_rel_err=max(f32.values()), float32_by_layer=f32,
                definition='max over items and sampled (layer, t) of |h_decode - h_prefill| / |h_prefill|; '
                           'prefill = backbone(use_cache=False) on [EOT] + 128 tokens; decode = MambaCache, one '
                           'token per step')


def _rel_rows(a, b):
    """max over items and positions of |a - b| / |b| for (n, P, D) residuals."""
    a, b = a.double(), b.double().to(a.device)
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


# ----------------------------------------------------------------------------------------------- U1
def run_u1(model, checkpoint, probe, work, device, log=print, chunk32=100, chunk64=50, model_config=None,
           scratch=None):
    """U1 for one checkpoint. Writes WORK files (see mb_schema's docstring) and u1_<checkpoint>.json; raises after
    writing the record if a control fails or a zero-norm item exists. `scratch`: local directory for the 2.3 GB raw
    Gram memmap (copied to `work` afterwards; default: written in place)."""
    import lm_schema as S
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    final = checkpoint == MB.FINAL
    model.eval().to(device)
    tokens = np.asarray(probe.tokens)
    n = probe.n
    cfg = model.config
    info = dict(checkpoint=checkpoint, removed=probe.removed, layout='AlignedUMAP',
                construction_fingerprint=MB.FINGERPRINT, parameter_sha=parameter_sha(model),
                model_config=model_config or {k: getattr(cfg, k, None) for k in MODEL_CONFIG_KEYS})
    try:
        import transformers
        info['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    info['torch_version'] = torch.__version__
    start = time.time()
    raw_path = Path(scratch or work) / f'raw_{checkpoint}.npy'
    out = decode_views(model, tokens, device, keep_parts=final, c1b=True, gram_id=final,
                       raw_path=raw_path if final else None, log=log)
    if final and scratch is not None:
        import shutil
        shutil.copyfile(raw_path, work / raw_path.name)
        raw_path.unlink()
    info['decode_seconds'] = time.time() - start
    info['zero_norm'] = out['zero']
    if out['zero']:
        _write(work / f'u1_{checkpoint}.json', info)
        raise RuntimeError(f"zero-norm items {out['zero']}: coordinator must record them in removed.json and rerun")
    np.save(work / f'act_{checkpoint}.npy', out['act'])
    np.save(work / f'state_{checkpoint}.npy', out['state'])
    np.save(work / f'stack_{checkpoint}.npy', out['stack'])
    info['C1b'] = out['C1b']
    emb = model.backbone.embeddings.weight.detach().float().cpu().numpy()
    bag = C.bag_vectors(tokens, emb)
    np.save(work / f'bag_{checkpoint}.npy', bag)
    tokens_s = S.shuffled_tokens(probe.raw)[probe.items]
    bag_s = C.bag_vectors(tokens_s, emb, [MB.SEQ])
    info['C2'] = dict(identical=bool(np.array_equal(bag[-1], bag_s[0])),
                      max_abs_diff=float(np.abs(bag[-1] - bag_s[0]).max()))
    if final:
        np.save(work / f'parts_{checkpoint}.npy', out['parts'])
        info['norm_mean'] = out['norm'].tolist()
        info['P3_gram_identity'] = out['gram_identity']
        del out
        log('  probe S')
        outs = decode_views(model, tokens_s, device, scan=False, positions=[MB.SEQ], log=log)
        np.save(work / f'actS_{checkpoint}.npy', outs['act'][:, 0])
        np.save(work / f'stateS_{checkpoint}.npy', outs['state'][:, 0])
        np.save(work / f'stackS_{checkpoint}.npy', outs['stack'][:, 0])
        del outs
        log('  C1a')
        act = np.load(work / f'act_{checkpoint}.npy', mmap_mode='r')
        info['C1a'] = c1a(model, tokens, act, device, chunk32, chunk64, log)
    info['seconds'] = time.time() - start
    _write(work / f'u1_{checkpoint}.json', info)
    assert info['C2']['identical'], 'C2 failed'
    assert info['C1b']['pass'], f"C1b failed: {info['C1b']}"
    if final:
        assert info['P3_gram_identity']['pass'], f"P3 Gram identity failed: {info['P3_gram_identity']}"
        assert info['C1a']['pass'], f"C1a failed: {info['C1a']['max_rel_err']}"
    return info


def _write(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False, default=lambda o: o.item() if hasattr(o, 'item') else
                                     o.tolist()))


def tiny_config(**kw):
    """A tiny Mamba-1 config with the real depth (24 layers) for CPU tests."""
    from transformers import MambaConfig
    base = dict(vocab_size=512, hidden_size=16, intermediate_size=32, state_size=4, num_hidden_layers=MB.N_LAYERS,
                conv_kernel=4, time_step_rank=4, residual_in_fp32=True, use_conv_bias=True, use_bias=False,
                layer_norm_epsilon=1e-5, initializer_range=0.1, bos_token_id=0, eos_token_id=0, pad_token_id=0)
    base.update(kw)
    return MambaConfig(**base)
