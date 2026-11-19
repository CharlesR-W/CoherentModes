"""Schema-v3 block computation for the CIFAR-100 / VGG explorers (see SPEC.md).

The construction is imported from ../cifar_coherent (coherent.py, seba.py, sweep_math.py); this module only
generalises the counts (K eigenfunctions, SEBA ranks) and adds the label-free split-core stability check.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REF = Path(__file__).resolve().parent.parent / "cifar_coherent"
if _REF.is_dir() and str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

from coherent import block_modes, centred_cosine_gram, symmetric_balance  # noqa: E402
from construction import CONSTRUCTION  # noqa: E402
from seba import seba, seba_labels  # noqa: E402

MODES = ["kNN graph, k = 15", "per-point perplexity 100"]
STAB_WINDOW = 2
STAB_THRESHOLD = 0.5
STAB_SEED = 20261113
STAB_GUARD = 16       # extra subset modes so a reproducible group crossing K is still resolved


def fingerprint(n_vecs: int, ranks: list[int], n: int) -> str:
    """Reference construction contract (rules 0 and 1 only) plus the v3 counts and stability parameters."""
    c = dict(CONSTRUCTION)
    c.update(rules=CONSTRUCTION["rules"][:2], probe=f"{n}-images", implementation_version="cifar-more-v3-2026-11-13-perplexity-ties-1e-9",
             n_vecs=n_vecs, seba_r=list(ranks),
             stability=dict(seed=STAB_SEED, window=STAB_WINDOW, threshold=STAB_THRESHOLD,
                            split="permutation->core,A,B; subsets core+A, core+B; regram+rebuild per subset"))
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------- kernels
TIE = 1e-9            # cosine distance below which two images count as coincident (well above float32 noise)


def perplexity_logkernel(c, perplexity):
    """sweep_math.perplexity_logkernel, plus one documented edge case it never met on ResNet-20.

    A row whose tied-nearest set (distance within TIE = 1e-9 of its minimum) has >= `perplexity` members cannot reach the
    target perplexity at any temperature (its entropy tends to log #ties > log perplexity). Such rows (images with
    exactly coincident activations, e.g. a collapsed early network) take the beta -> infinity limit: the conditional
    is uniform over the tied-nearest images. Every other row is calibrated exactly as the reference does (bracketing
    and bisection are per-row), so the result is identical to the reference whenever no row is unattainable."""
    d = 1 - c.double()
    d.diagonal().fill_(float("inf"))
    d -= d.min(1).values[:, None]
    moment_d = d.clone()
    moment_d.diagonal().fill_(0)
    n = len(d)
    target = math.log(perplexity)
    ties = (d <= TIE).sum(1)
    stuck = torch.log(ties.double()) >= target - 1e-9
    low = torch.zeros(n, device=d.device, dtype=d.dtype)
    high = torch.ones_like(low)

    def conditional(beta):
        logits = -beta[:, None] * d
        logz = torch.logsumexp(logits, dim=1)
        logp = logits - logz[:, None]
        entropy = logz + beta * (torch.exp(logp) * moment_d).sum(1)
        return entropy, logp
    for _ in range(60):          # reference: 40 doublings; 60 lets rows with near-ties (d ~ 1e-12) still bracket
        ent, _ = conditional(high)
        bad = (ent > target) & ~stuck
        if not bool(bad.any()):
            break
        high[bad] *= 2
    else:
        raise ValueError("Perplexity bracketing did not converge")
    for _ in range(50):
        beta = (low + high) / 2
        ent, _ = conditional(beta)
        low = torch.where(ent > target, beta, low)
        high = torch.where(ent > target, high, beta)
    beta = (low + high) / 2
    ent, logp = conditional(beta)
    if bool(stuck.any()):
        tied = d <= TIE
        limit = torch.where(tied, -torch.log(ties.double())[:, None], torch.full_like(d, float("-inf")))
        logp = torch.where(stuck[:, None], limit, logp)
        ent = torch.where(stuck, torch.log(ties.double()), ent)
        beta = torch.where(stuck, torch.full_like(beta, float("inf")), beta)
    if bool((~stuck).any()):
        assert float((ent - target)[~stuck].abs().max()) < 1e-7
    logk = torch.logaddexp(logp, logp.T) - math.log(2)
    fin = beta[~stuck]          # quantiles over calibrated rows (all rows unless some are unattainable)
    q = lambda p: float(fin.quantile(p)) if len(fin) else None
    return logk, dict(kappa_median=q(.5), kappa_q10=q(.1), kappa_q90=q(.9), perplexity_unattainable=int(stuck.sum()))
def kernel_bank(c: torch.Tensor):
    """Rules 0 and 1 of sweep_math.final_kernel_bank, without computing rules 2-4.

    Returns [(log-affinity, info)], info with kappa / kappa_q10 / kappa_q90 (None for the graph) and nn_mass
    (median share of the symmetric pre-Sinkhorn affinity on each point's 15 nearest non-self neighbours)."""
    d = 1 - c.double()
    d.diagonal().fill_(float("inf"))
    near = d.topk(15, largest=False, dim=1)
    mask = torch.zeros_like(c, dtype=torch.bool)
    mask.scatter_(1, near.indices, True)
    mask = mask | mask.T
    weights = torch.where(mask, torch.ones_like(c), torch.full_like(c, 1e-6))
    weights.diagonal().fill_(0)
    bank = [(torch.log(weights), dict(kappa=None, kappa_q10=None, kappa_q90=None))]
    logk, info = perplexity_logkernel(c, 100)
    bank.append((logk, dict(kappa=info["kappa_median"], kappa_q10=info["kappa_q10"], kappa_q90=info["kappa_q90"],
                            perplexity_unattainable=info["perplexity_unattainable"])))
    for logk, info in bank:
        probabilities = torch.exp(logk - torch.logsumexp(logk, dim=1)[:, None])
        info["nn_mass"] = float(probabilities.gather(1, near.indices).sum(1).quantile(.5))
    return bank


def lazy_operators(c: torch.Tensor):
    """[(M, info)] for both rules: zero-diagonal symmetric scaling P, M = (I + P)/2, float64."""
    out = []
    for logk, info in kernel_bank(c):
        p, diag = symmetric_balance(logk, return_info=True)
        m = p * .5
        m.diagonal().add_(.5)
        info.update(sinkhorn_iterations=diag["iterations"], row_error=diag["row_error"])
        out.append((m, info))
    return out



# ------------------------------------------------------------------------------------------- split
def split(n: int):
    """Label-free core/subset split: permutation -> core C, A', B'; S1 = C+A', S2 = C+B' (sorted positions).

    Returns (s1, s2, core_in_s1, core_in_s2): the subsets' probe positions and, for each, the row indices of the
    core inside the subset (in the same core order)."""
    perm = np.random.RandomState(STAB_SEED).permutation(n)
    k = n // 3
    core, a, b = np.sort(perm[:k]), perm[k:2 * k], perm[2 * k:]
    s1, s2 = np.sort(np.concatenate([core, a])), np.sort(np.concatenate([core, b]))
    return s1, s2, np.searchsorted(s1, core), np.searchsorted(s2, core)


def _orth(u: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(u).Q


def stability(v1: torch.Tensor, v2: torch.Tensor, c1, c2, K: int, window=STAB_WINDOW):
    """Split-core reproducibility of two subset eigenbases (SPEC section 5).

    v1, v2: (n_i, K + guard) non-trivial eigenvectors of the two subset block operators (descending eigenvalue).
    c1, c2: row indices of the shared core in each subset. Rows at the core are centred and normalised.
    Returns span_k = min cos^2 of the principal angles between the leading-k spans (k = 1..K+guard),
    cum_k = their mean cos^2, and local_k (k = 1..K) = windowed index agreement (descriptive only)."""
    def restrict(v, rows):
        u = v[torch.as_tensor(rows, device=v.device)].double()
        u = u - u.mean(0, keepdim=True)
        return u / u.norm(dim=0, keepdim=True).clamp_min(1e-300)
    u1, u2 = restrict(v1, c1), restrict(v2, c2)
    KG = u1.shape[1]
    local = torch.empty(K, dtype=torch.float64)
    for k in range(K):
        lo, hi = max(0, k - window), min(KG, k + window + 1)
        q1, q2 = _orth(u1[:, lo:hi]), _orth(u2[:, lo:hi])
        local[k] = .5 * (float((q2.T @ u1[:, k]).square().sum()) + float((q1.T @ u2[:, k]).square().sum()))
    # one QR of all columns gives nested orthonormal bases of every leading span
    q1, q2 = _orth(u1), _orth(u2)
    cross = q1.T @ q2
    span = torch.empty(KG, dtype=torch.float64)
    cum = torch.empty(KG, dtype=torch.float64)
    for k in range(KG):
        sv = torch.linalg.svdvals(cross[:k + 1, :k + 1])
        span[k] = float(sv.min()) ** 2
        cum[k] = float(sv.square().sum()) / (k + 1)
    f = lambda t: t.cpu().numpy().astype("f4")
    return f(span), f(cum), f(local)


def horizon(span, threshold=STAB_THRESHOLD) -> int:
    """Largest k whose leading-k span is reproducible (min cos^2 >= threshold); 0 if none."""
    ok = np.flatnonzero(np.asarray(span) >= threshold)
    return int(ok[-1]) + 1 if len(ok) else 0


# ------------------------------------------------------------------------------------------- records
def allocate(nm, npair, n, nc, K, ranks):
    R, rmax = len(ranks), max(ranks)
    return dict(vals=np.empty((nm, npair, K + 1), np.float32),
                vecs=np.empty((nm, npair, K, n), np.int16),
                scale=np.empty((nm, npair, K), np.float32),
                seba_label=np.empty((nm, npair, R, n), np.int8),
                seba_rot=np.zeros((nm, npair, R, rmax, rmax), np.float32),
                seba_perm=np.full((nm, npair, R, rmax), -1, np.int8),
                retention_psi=np.empty((nm, npair, K, nc), np.float32),
                retention_seba=np.full((nm, npair, R, rmax, nc), np.nan, np.float32),
                seba_val=np.zeros((nm, npair, R, n), np.uint8),
                seba_lead=np.zeros((nm, npair, R, n), np.uint8),
                stab_span=np.full((nm, npair, K + STAB_GUARD), np.nan, np.float32),
                stab_cum=np.full((nm, npair, K + STAB_GUARD), np.nan, np.float32),
                stab_local=np.full((nm, npair, K), np.nan, np.float32),
                sub_vals=np.full((nm, npair, 2, K + STAB_GUARD + 1), np.nan, np.float32))


def block_record(ms, a, b, K, ranks, eigen=None, extras=False):
    """schema_v2.block_record with K eigenfunctions and arbitrary SEBA ranks (identical at K=8, reference ranks)."""
    R, rmax = len(ranks), max(ranks)
    assert rmax <= K + 1
    values, vectors = block_modes(ms[a:b + 1], K + 1, shared_stationary=True) if eigen is None else eigen
    assert abs(float(values[0]) - 1) < 1e-6
    assert float((vectors[:, 0] - 1 / len(vectors) ** .5).abs().max()) < 1e-5
    vectors[:, 0] = 1 / len(vectors) ** .5
    v = vectors.cpu().double()
    scale = v[:, 1:].abs().max(0).values
    quant = torch.round(32767 * v[:, 1:] / scale).to(torch.int16)
    result = dict(vals=values.cpu().numpy().astype("f4"), vecs=quant.T.numpy(), scale=scale.numpy().astype("f4"),
                  seba_label=np.empty((R, len(v)), np.int8), seba_rot=np.zeros((R, rmax, rmax), np.float32),
                  seba_perm=np.full((R, rmax), -1, np.int8), retention_psi=None,
                  retention_seba=np.full((R, rmax, len(ms)), np.nan, np.float32))
    if extras:   # browser-side SEBA amplitudes, from the float64 vectors (the browser never reconstructs SEBA)
        result.update(seba_val=np.zeros((R, len(v)), np.uint8), seba_lead=np.zeros((R, len(v)), np.uint8))
    ss = []
    threads = int(__import__("os").environ.get("CM3_SEBA_THREADS", "1"))
    if threads > 1:      # independent SEBA problems; torch releases the GIL, results identical to the serial loop
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(threads) as ex:
            solved = list(ex.map(lambda r: seba(v, r, return_transform=True), ranks))
    else:
        solved = [seba(v, r, return_transform=True) for r in ranks]
    for ri, r in enumerate(ranks):
        s, rot, perm = solved[ri]
        result["seba_label"][ri] = seba_labels(s).numpy().astype("i1")
        result["seba_rot"][ri, :r, :r] = rot.numpy()
        result["seba_perm"][ri, :r] = perm.numpy()
        ss.append(s)
        if extras:
            lab = result["seba_label"][ri].astype(np.int64)
            sn = s.numpy()
            own = np.where(lab >= 0, sn[np.arange(len(sn)), np.maximum(lab, 0)], 0.)
            other = sn.copy()
            other[np.arange(len(sn)), np.maximum(lab, 0)] = -np.inf
            runner = np.maximum(other.max(1), 0) if r > 1 else np.zeros(len(sn))
            result["seba_val"][ri] = np.rint(255 * np.clip(own, 0, 1)).astype(np.uint8)
            result["seba_lead"][ri] = np.where(lab >= 0, np.rint(255 * np.clip(own - runner, 0, 1)), 0).astype(np.uint8)
    f = torch.cat([v[:, 1:]] + ss, 1).to(ms[0].device)
    ret = torch.stack([(f * (m @ f)).sum(0) / f.square().sum(0) for m in ms], 1).cpu().numpy().astype("f4")
    result["retention_psi"] = ret[:K]
    at = K
    for ri, r in enumerate(ranks):
        result["retention_seba"][ri, :r] = ret[at:at + r]
        at += r
    return result, vectors


def stability_record(ms1, ms2, a, b, K, c1, c2):
    """Subset block operators (same block, same rule): eigenvalues and the stability numbers (K + guard modes)."""
    va, wa = block_modes(ms1[a:b + 1], K + STAB_GUARD + 1, shared_stationary=True)
    vb, wb = block_modes(ms2[a:b + 1], K + STAB_GUARD + 1, shared_stationary=True)
    span, cum, local = stability(wa[:, 1:], wb[:, 1:], c1, c2, K)
    return dict(stab_span=span, stab_cum=cum, stab_local=local,
                sub_vals=np.stack([va.cpu().numpy(), vb.cpu().numpy()]).astype("f4"))


def put(arrays, mode, pair, rec):
    for key, val in rec.items():
        if key in arrays:
            arrays[key][mode, pair] = val


def pairs(n):
    return [(a, b) for a in range(n) for b in range(a, n)]


def subset_grams(acts_flat: torch.Tensor, s1, s2):
    """Centred cosine Grams of the full probe and of both subsets (centring over each set), float64."""
    full = centred_cosine_gram(acts_flat)
    dev = acts_flat.device
    g1 = centred_cosine_gram(acts_flat[torch.as_tensor(s1, device=dev)])
    g2 = centred_cosine_gram(acts_flat[torch.as_tensor(s2, device=dev)])
    return full, g1, g2


def compute_family(grams, K, ranks, log=print):
    """grams: list over the family's cells of (C_full, C_s1, C_s2) float64 tensors (one device).

    Returns (arrays, cell_infos): every block of the family for both rules, and per-cell kernel metadata."""
    nc = len(grams)
    n = grams[0][0].shape[0]
    s1, s2, c1, c2 = split(n)
    ps = pairs(nc)
    arrays = allocate(len(MODES), len(ps), n, nc, K, ranks)
    built = [[lazy_operators(g) for g in cell] for cell in grams]   # [cell][full/s1/s2][rule] -> (M, info)
    infos = [[built[ci][0][mi][1] for mi in range(len(MODES))] for ci in range(nc)]
    for mi in range(len(MODES)):
        ms = [built[ci][0][mi][0] for ci in range(nc)]
        ms1 = [built[ci][1][mi][0] for ci in range(nc)]
        ms2 = [built[ci][2][mi][0] for ci in range(nc)]
        for pi, (a, b) in enumerate(ps):
            rec, _ = block_record(ms, a, b, K, ranks, extras=True)
            rec.update(stability_record(ms1, ms2, a, b, K, c1, c2))
            put(arrays, mi, pi, rec)
        log(f"rule {mi}: {len(ps)} blocks")
    return arrays, infos
