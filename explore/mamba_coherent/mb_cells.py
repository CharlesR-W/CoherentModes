"""State and stack views in Gram form, the Gram-based tree cell, and the P3 Gram identity (SPEC sections 4, 6, 7).

New module (nothing here exists in `../lm_coherent`). It reuses `lm_cells` unchanged: the standard pipeline
(`unit_rows`, `standard_gram`, `renormalise`) and the Pythia slab's Gram-averaging rule (`SlabAccumulator`: mean of
standard-pipeline cosine Grams, then double-centre and renormalise), which is exactly the rule the state and stack
views use, with constituents in place of positions.

Constituents of layer m at position t: c^m_t (conv buffer, E x 3, flattened) and z^m_t (SSM state, E x S,
flattened). Each enters with equal weight after its own standard pipeline (decision D-M1).
"""
from __future__ import annotations

import math

import numpy as np
import torch

import mb_construction as MB  # noqa: F401  (sys.path bridge)
import lm_cells as C


def _t(x, device=None):
    x = x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))
    return x.to(device=device, dtype=torch.float64).reshape(len(x), -1)


def raw_gram(x, device=None):
    """Uncentred inner-product Gram X X^T, float64 (stored for T4's cells: the subset rebuild needs it)."""
    x = _t(x, device)
    return x @ x.T


class StateStack:
    """Feed layers m = 1, 2, ..., N_LAYERS in order with their constituents (c, z); read `state(m)` (layer m alone)
    and `stack()` (layers 1..m) after layer m was added. Optionally keeps each constituent's cosine Gram (`parts`)
    and raw Gram (`raw`, every layer) when asked; per-layer state Grams are kept for `keep_states` only. All Grams float64 on `device`."""

    def __init__(self, n, device='cpu', keep_states=MB.LAYERS, keep_parts=(), keep_raw=False):
        self.n, self.device = n, device
        self.stack_acc = C.SlabAccumulator(n, device=device)
        self.keep_states, self.keep_parts, self.keep_raw = set(keep_states), set(keep_parts), keep_raw
        self.parts, self.raw, self.states = {}, [], {}
        self.m = 0

    def add(self, c, z):
        self.m += 1
        state = C.SlabAccumulator(self.n, device=self.device)
        grams = []
        for x in (c, z):
            x = _t(x, self.device)
            u = C.unit_rows(x)
            g = u @ u.T
            grams.append(g)
            state.sum += g
            state.count += 1
            self.stack_acc.sum += g
            self.stack_acc.count += 1
            if self.keep_raw:
                self.raw.append(x @ x.T)
        if self.m in self.keep_states:
            self.states[self.m] = state
        if self.m in self.keep_parts:
            self.parts[self.m] = [g.clamp(-1.0, 1.0) for g in grams]     # = lm_cells.standard_gram of each part

    def state(self, m):
        return self.states[m].gram()

    def stack(self):
        return self.stack_acc.gram()


def explicit_concat_gram(constituents):
    """P3's reference: the standard pipeline on the explicit concatenation [U_1 .. U_k] / sqrt(k) of the
    standard-pipeline-normalised constituents (U_i = unit_rows(x_i))."""
    constituents = list(constituents)
    k = len(constituents)
    dims = [int(np.prod(x.shape[1:])) for x in constituents]
    dev = constituents[0].device if torch.is_tensor(constituents[0]) else None
    out = torch.empty((len(constituents[0]), sum(dims)), dtype=torch.float64, device=dev)
    at = 0
    for x, d in zip(constituents, dims):            # filled in place: one (n, sum dims) buffer, not k + 1 of them
        out[:, at:at + d] = C.unit_rows(_t(x, dev)) / math.sqrt(k)
        at += d
    return C.standard_gram(out)


def gram_from_raw(g, members=None):
    """The standard pipeline in Gram form on the items S: restrict the RAW Gram to S, double-centre over S,
    diagonal-normalise to a cosine Gram (clamped to [-1, 1]); zero-norm policy as `lm_cells.unit_rows`."""
    g = torch.as_tensor(np.asarray(g) if not torch.is_tensor(g) else g, dtype=torch.float64)
    if members is not None:
        idx = torch.as_tensor(np.asarray(members), dtype=torch.long)
        g = g[idx][:, idx]
    g = g - g.mean(0, keepdim=True) - g.mean(1, keepdim=True) + g.mean()
    d = g.diagonal().clamp_min(0).sqrt()
    bad = np.flatnonzero((d < C.ZERO_NORM_REL * d.median()).cpu().numpy())
    if bad.size:
        raise C.DegenerateItems('zero-norm', bad, '(constituent, subset rebuild)')
    return (g / d[:, None] / d[None, :]).clamp(-1.0, 1.0)


class RawGramCell:
    """A Gram-based tree cell (SPEC 6): the stack view of one (layer, position), held as the RAW Gram of every
    constituent. `subset_gram(S)` repeats the stack's construction on S: each constituent's raw Gram restricted to
    S, double-centred over S, cosine-normalised; averaged with equal weight; then the stack's final double-centring
    and renormalisation (`lm_cells.renormalise`). Restricting the parent's centred Gram is NOT what this does.

    raws: sequence of (N, N) arrays (numpy, possibly memory-mapped) or a (k, N, N) array."""

    def __init__(self, raws):
        self.raws = raws
        self.n = int(raws[0].shape[0])

    def __len__(self):
        return self.n

    def subset_gram(self, members):
        members = np.asarray(members)
        acc = None
        for g in self.raws:
            sub = np.asarray(g[np.ix_(members, members)]) if isinstance(g, np.ndarray) else g[members][:, members]
            c = gram_from_raw(sub)
            acc = c if acc is None else acc + c
        return C.renormalise(acc / len(self.raws))


def vector_subset_gram(constituents, members):
    """The tree test's independent reference: full rebuild on S from raw constituent VECTORS."""
    return explicit_concat_gram([np.asarray(x)[members] for x in constituents])


# --------------------------------------------------------------------- one position (U1 and the fixture)
def position_grams(state_of, n, device='cpu', n_layers=MB.N_LAYERS, layers=MB.LAYERS, keep_parts=False,
                   keep_raw=False):
    """Every Gram of one position from `state_of(m) -> (c, z)` for m = 1..n_layers (the shared code path of U1 and
    the synthetic fixture). Returns numpy float64 state [L, n, n], stack [L, n, n], parts [L, 2, n, n] or None
    (each constituent's standard cosine Gram), raw: list of 2 * n_layers (n, n) uncentred Grams (order c^1, z^1,
    c^2, ...) or None."""
    ss = StateStack(n, device=device, keep_states=layers, keep_parts=layers if keep_parts else (), keep_raw=keep_raw)
    state = np.empty((len(layers), n, n))
    stack = np.empty((len(layers), n, n))
    parts = np.empty((len(layers), 2, n, n)) if keep_parts else None
    for m in range(1, n_layers + 1):
        ss.add(*state_of(m))
        if m in layers:
            li = layers.index(m)
            stack[li] = ss.stack().cpu().numpy()
            state[li] = ss.state(m).cpu().numpy()
            if keep_parts:
                parts[li] = torch.stack(ss.parts[m]).cpu().numpy()
    raw = [g.cpu().numpy() for g in ss.raw] if keep_raw else None
    return state, stack, parts, raw


def gram_identity(state_of, state, stack, device='cpu', layers=MB.LAYERS, stack_layers=None, t=MB.SEQ):
    """P3's new identity at one position: the state and stack Grams equal the standard pipeline on the explicit
    concatenation of the standard-pipeline-normalised constituents, scaled by 1/sqrt(#constituents)."""
    stack_layers = layers if stack_layers is None else stack_layers
    dev = {}
    for li, l in enumerate(layers):
        ref = explicit_concat_gram(list(state_of(l)))
        dev[f'state_L{l}'] = float((ref - torch.as_tensor(np.asarray(state[li])).to(ref.device)).abs().max())
    for l in stack_layers:
        li = layers.index(l)
        ref = explicit_concat_gram([x for m in range(1, l + 1) for x in state_of(m)])
        dev[f'stack_L{l}'] = float((ref - torch.as_tensor(np.asarray(stack[li])).to(ref.device)).abs().max())
        del ref
    worst = max(dev.values())
    return dict(t=t, deviations=dev, max_dev=worst, tolerance=MB.GRAM_ID_TOL, **{'pass': bool(worst < MB.GRAM_ID_TOL)})
