"""Items (token contexts), lags, groupings and the bag/recent reference vectors (SPEC sections 2, 3, 5).

Label-free: `sample_items` uses only the snippets' token ids (the de-duplication compares token windows). Groupings
are built for scoring after the operators exist: T, Nx, U are inputs; A is the network's own output; W, Cc, Cn are
validation labels (`token_classes`, sources), and none of them enters a kernel, operator, basis or cut.
"""
from __future__ import annotations

import numpy as np

import mc_construction as K


class DedupFailed(RuntimeError):
    pass


def sample_items(tokens, seed=K.ITEM_SEED, per=K.PER_SNIPPET, lo=K.T_MIN, hi=K.T_MAX, window=K.DEDUP_WINDOW,
                 order_seed=K.ORDER_SEED, max_draws=100000):
    """SPEC 2. tokens: (n_snippets, SEQ) token ids 1..SEQ (no EOT). For each snippet in id order, draw positions
    one at a time with RandomState(seed + snippet id).randint(lo, hi + 1): a position already drawn for this snippet
    is redrawn (sampling without replacement); a position whose last `window` tokens (t-window+1..t), taken as a
    multiset, equal those of any earlier accepted item is redrawn and counted. (Multiset, not sequence: the `recent`
    reference is a function of that multiset, so two items sharing it would be exact duplicates in a view cell; a
    shared sequence is a shared multiset, so this is the stricter reading of SPEC 2. README DEVIATIONS.)
    Items are then stored in RandomState(order_seed) order.

    Returns dict(snippet (N,), t (N,), order (N,) = original index of each stored item, redraws, redraw_items)."""
    tokens = np.asarray(tokens)
    taken = {}
    snip, pos = [], []
    redraws, redraw_items = 0, []
    for i in range(len(tokens)):
        rng = np.random.RandomState(seed + i)
        chosen, draws = [], 0
        while len(chosen) < per:
            draws += 1
            if draws > max_draws:
                raise DedupFailed(f'snippet {i}: cannot find {per} positions with distinct {window}-token windows')
            t = int(rng.randint(lo, hi + 1))
            if t in chosen:
                continue
            key = window_key(tokens[i, t - window:t])
            if key in taken:
                redraws += 1
                redraw_items.append([i, t, *taken[key]])
                continue
            taken[key] = (i, t)
            chosen.append(t)
        snip += [i] * per
        pos += chosen
    snip, pos = np.array(snip), np.array(pos)
    order = np.random.RandomState(order_seed).permutation(len(snip))
    return dict(snippet=snip[order], t=pos[order], order=order, redraws=redraws, redraw_items=redraw_items)


def window_key(tokens):
    """The de-duplication key: the multiset of the window's token ids."""
    return tuple(sorted(np.asarray(tokens).tolist()))


def check_items(items, tokens, window=K.DEDUP_WINDOW):
    """Assertions on an item table: positions in range, distinct per snippet, no shared last-`window` multiset."""
    snip, t = np.asarray(items['snippet']), np.asarray(items['t'])
    assert t.min() >= K.T_MIN and t.max() <= K.T_MAX
    assert len(set(zip(snip.tolist(), t.tolist()))) == len(snip)
    keys = {window_key(np.asarray(tokens)[i, s - window:s]) for i, s in zip(snip, t)}
    assert len(keys) == len(snip), 'two items share their last tokens'
    return True


# --------------------------------------------------------------------------------------------- groupings
def token_at(tokens, snip, t):
    """Token id at position t (1-based; tokens holds positions 1..SEQ)."""
    return np.asarray(tokens)[snip, np.asarray(t) - 1]


def classes_of(token_str, snip, pos):
    """Token class (index into K.CLASSES) of the decoded token at position pos of each item (validation)."""
    return np.array([K.CLASSES.index(K.TCL.token_class(token_str[s][p - 1])) for s, p in zip(snip, pos)])


def groupings(items, tokens, token_str, sources, top1=None, lags=()):
    """Every grouping of SPEC 5 as integer arrays over the stored items; plus T_tau for each lag in `lags`.
    `top1`: the trained model's top-1 next-token prediction at t (None: A omitted)."""
    snip, t = np.asarray(items['snippet']), np.asarray(items['t'])
    g = dict(T=token_at(tokens, snip, t), Nx=token_at(tokens, snip, t + 1), U=snip.copy(),
             W=np.asarray(sources)[snip], Cc=classes_of(token_str, snip, t), Cn=classes_of(token_str, snip, t + 1))
    if top1 is not None:
        g['A'] = np.asarray(top1)
    for tau in lags:
        g[f'T{tau}'] = token_at(tokens, snip, t + tau)
    return g


def lag_groupings(g, tau):
    """Groupings of a lag cell: lag-0 T, Cc, U, W plus T_tau (SPEC 5)."""
    return dict(T=g['T'], Cc=g['Cc'], U=g['U'], W=g['W'], Ttau=g[f'T{tau}'])


def contexts(items, tokens, before=K.CONTEXT_BEFORE, after=K.CONTEXT_AFTER):
    """(N, before + after) token ids: the `before` tokens ending at t, then t+1..t+after; -1 before position 1."""
    tokens = np.asarray(tokens)
    out = np.full((len(items['t']), before + after), -1, np.int32)
    for k, (s, t) in enumerate(zip(items['snippet'], items['t'])):
        a = t - before
        seg = tokens[s, max(a, 0):t + after]
        out[k, before + after - len(seg):] = seg
    return out


# --------------------------------------------------------------------------------------- reference vectors
def mean_embedding(tokens, embedding, snip, lo, hi):
    """(N, D) float64: mean of E[x_u] over u in lo_i..hi_i (1-based, inclusive), summed in sorted token-id order
    (lm_cells.bag_vectors' arithmetic, with a per-item range)."""
    tokens = np.asarray(tokens)
    e = np.asarray(embedding, dtype=np.float64)
    out = np.empty((len(snip), e.shape[1]))
    for k, (s, a, b) in enumerate(zip(snip, lo, hi)):
        row = tokens[s, a - 1:b]
        out[k] = np.cumsum(e[np.sort(row, kind='stable')], axis=0)[-1] / len(row)
    return out


def bag_vectors(items, tokens, embedding):
    t = np.asarray(items['t'])
    return mean_embedding(tokens, embedding, items['snippet'], np.ones_like(t), t)


def recent_vectors(items, tokens, embedding, width=K.DEDUP_WINDOW):
    t = np.asarray(items['t'])
    return mean_embedding(tokens, embedding, items['snippet'], t - width + 1, t)
