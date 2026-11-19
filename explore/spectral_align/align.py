"""Spectral alignment math. No labels, fitting, or Modal dependencies.

Eigenvalue arrays include the stationary eigenvalue at index zero; U contains
only the nonconstant modes. Distances use a shared ordered node set.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64)
T_GRID = (1, 4, 16, 64, 256)


def spectrum(m, k=64, full_vectors=False):
    """Float64 constant-isolating solve; return full values and nontrivial U.

    full_vectors=True returns all N-1 nonconstant vectors, for heat kernels.
    Reuse the explorer's Householder eigensolve rather than deflate numerically.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cifar_coherent'))
    from coherent import block_modes
    values, vectors = block_modes([m.double()], len(m), shared_stationary=True)
    return values, vectors[:, 1:] if full_vectors else vectors[:, 1:k + 1]


def projector_distances(ua, ub, ks=K_GRID):
    """Every leading-cut overlap from one cross-mode Gram (supports batches)."""
    g = ua.transpose(-2, -1) @ ub
    cumulative = g.square().cumsum(-2).cumsum(-1)
    ix = torch.as_tensor(ks, device=g.device) - 1
    return (1 - cumulative[..., ix, ix] / (ix + 1)).clamp(0, 1)


def eigenvalue_distances(ea, eb, ks=K_GRID):
    ix = torch.as_tensor(ks, device=ea.device) - 1
    return ((ea[..., 1:] - eb[..., 1:]).square().cumsum(-1)[..., ix] / (ix + 1)).sqrt()


def center_gram(c):
    return c - c.mean(-1, keepdim=True) - c.mean(-2, keepdim=True) + c.mean((-2, -1), keepdim=True)


def cosine_distance(a, b):
    numerator = (a * b).sum((-2, -1))
    denominator = a.square().sum((-2, -1)).sqrt() * b.square().sum((-2, -1)).sqrt()
    if bool((denominator == 0).any()):
        raise ValueError('Cosine distance undefined for zero-norm kernel')
    return (1 - numerator / denominator).clamp(0, 2)


def cka_distance(ca, cb):
    """Baseline on H C H, not on raw-activation Grams."""
    return cosine_distance(center_gram(ca), center_gram(cb))


def heat_kernel(e, u, t):
    """Centred M**t using ALL nontrivial eigenpairs, not the top-K export."""
    if u.shape[-1] != e.shape[-1] - 1:
        raise ValueError('Heat kernel requires the full nontrivial spectrum')
    return (u * e[..., 1:].pow(t).unsqueeze(-2)) @ u.transpose(-2, -1)


def normalized_heat_kernel(e, u, t):
    """Unit-Frobenius heat kernel; log-space weights avoid float32 underflow.

    A common positive scale cancels from cosine distance. Laziness makes the
    nontrivial spectrum nonnegative, apart from numerical eigensolve roundoff.
    """
    if u.shape[-1] != e.shape[-1] - 1:
        raise ValueError('Heat kernel requires the full nontrivial spectrum')
    logs = t * e[..., 1:].double().clamp_min(0).log()
    lognorm = .5 * torch.logsumexp(2 * logs, -1, keepdim=True)
    if not bool(torch.isfinite(lognorm).all()):
        raise ValueError('Zero diffusion kernel has undefined cosine distance')
    weights = (logs - lognorm).exp().to(u.dtype)
    return (u * weights.unsqueeze(-2)) @ u.transpose(-2, -1)


def heat_distances(ea, ua, eb, ub, ts=T_GRID):
    return torch.stack([cosine_distance(normalized_heat_kernel(ea, ua, t), normalized_heat_kernel(eb, ub, t)) for t in ts], -1)


def streamed_kernel_gram(count, n, load_kernel, device='cpu', dtype=torch.float32, row_chunk=32, load_rows=None):
    """Gram of flattened kernels; only count*row_chunk*n numbers resident.

    load_kernel(i, start, stop) returns those rows. The caller may cache full
    eigendecompositions or materialize kernels for a single t, never all t.
    """
    gram = torch.zeros((count, count), dtype=dtype, device=device)
    for start in range(0, n, row_chunk):
        stop = min(n, start + row_chunk)
        if load_rows is None:
            rows = torch.stack([load_kernel(i, start, stop).to(device=device, dtype=dtype).reshape(-1) for i in range(count)])
        else:
            rows = load_rows(start, stop).to(device=device, dtype=dtype).reshape(count, -1)
        gram.addmm_(rows, rows.T)
    norms = gram.diagonal().clamp_min(0).sqrt()
    if bool((norms == 0).any()):
        raise ValueError('Zero-norm kernel in pair bank')
    distance = (1 - gram / (norms[:, None] * norms[None, :])).clamp(0, 2)
    distance.fill_diagonal_(0)
    return distance


def pairwise_spectral(e, u, ks=K_GRID, chunk=16):
    """Chunked projector/eigenvalue distances, single G for each pair."""
    count = len(e)
    proj = torch.empty((count, count, len(ks)), dtype=u.dtype, device=u.device)
    eigd = torch.empty_like(proj)
    for a in range(0, count, chunk):
        for b in range(a, count, chunk):
            pa = projector_distances(u[a:a+chunk, None], u[None, b:b+chunk], ks)
            ea = eigenvalue_distances(e[a:a+chunk, None], e[None, b:b+chunk], ks)
            proj[a:a+chunk, b:b+chunk] = pa
            eigd[a:a+chunk, b:b+chunk] = ea
            proj[b:b+chunk, a:a+chunk] = pa.transpose(0, 1)
            eigd[b:b+chunk, a:a+chunk] = ea.transpose(0, 1)
    idx = torch.arange(count, device=u.device)
    proj[idx, idx] = 0
    eigd[idx, idx] = 0
    return proj, eigd


def inversion_fraction(matches):
    """Strict order inversions among all i<j; tied matches are not inversions."""
    a = np.asarray(matches)
    if len(a) < 2:
        return 0.0
    return float(np.triu(a[:, None] > a[None, :], 1).sum() / (len(a) * (len(a)-1) / 2))


def dtw_alignment(cost):
    from dtw import dtw
    d = np.asarray(cost, dtype=np.float64)
    if d.ndim != 2 or not np.isfinite(d).all():
        raise ValueError('DTW requires a finite 2D cost matrix')
    fit = dtw(d, step_pattern='symmetric2', keep_internals=False)
    rows, cols = d.argmin(1), d.argmin(0)
    return dict(cost=float(fit.distance / (sum(d.shape) - 1)), path=np.column_stack((fit.index1, fit.index2)).tolist(),
                path_length=len(fit.index1), row_argmins=rows.tolist(), column_argmins=cols.tolist(),
                row_inversion_fraction=inversion_fraction(rows), column_inversion_fraction=inversion_fraction(cols))


def architectural_reference(depth_a, depth_b, path=None):
    """Stage-preserving ceil map and path gaps in within-stage relative depth.

    Endpoints stem/pool/logits map to themselves. Gap uses continuous stage
    coordinate s-1+j/n (stem=0,pool=4,logits=5), so cross-stage matches are penalized.
    """
    from nets import site_stage, DEPTH_N
    sa, sb = site_stage(depth_a), site_stage(depth_b)
    na, nb = DEPTH_N[depth_a], DEPTH_N[depth_b]
    mapping = []
    for i, stage in enumerate(sa):
        if 1 <= stage <= 3:
            j = i - (stage - 1) * na
            mapping.append((stage - 1) * nb + math.ceil(j * nb / na))
        else:
            mapping.append(sb.index(stage))
    ends = [(s * na, s * nb) for s in (1, 2, 3)]
    result = dict(map=mapping, stage_ends=[list(x) for x in ends])
    if path is not None:
        def coord(index, stages, n):
            stage = stages[index]
            return (index / n) if 1 <= stage <= 3 else float(stage)
        # Compare B's path coordinate to B's discrete architectural reference.
        result['mean_relative_depth_gap'] = float(np.mean([abs(coord(j, sb, nb) - coord(mapping[i], sb, nb)) for i, j in path]))
        pairs = set(map(tuple, path))
        result['passes_stage_ends'] = [end in pairs for end in ends]
    return result


def block_permutation_null(cost, repetitions=1000, seed=0):
    """Re-optimize DTW after permuting B's blocks; stem/pool/logits fixed."""
    d = np.asarray(cost)
    rng = np.random.default_rng(seed)
    out = np.empty(repetitions)
    for rep in range(repetitions):
        permutation = np.arange(d.shape[1])
        permutation[1:-2] = rng.permutation(permutation[1:-2])
        out[rep] = dtw_alignment(d[:, permutation])['cost']
    return out
