"""Spectral coarse-graining certificates for symmetric doubly stochastic couplings (SPEC_COARSE.md).

Convention: Q[y, x] is the transition from start image x to end image y (column-vector convention, as in
product.py). Every Q here is doubly stochastic, so pi(x, y) = Q[y, x] / N has uniform marginals and the
normalised coupling is Q itself. Its SVD Q = sum_i s_i a_i b_i^T (a: end/left, b: start/right; unit vectors)
gives the start-side embedding Psi_X = sqrt(N) * (s_i b_i)_{i>=2} and end-side Psi_Y = sqrt(N) * (s_i a_i).
All information quantities are chi^2 mutual informations.
"""
from __future__ import annotations

import numpy as np
import torch


def onehot(labels: torch.Tensor, k: int | None = None) -> torch.Tensor:
    labels = torch.as_tensor(labels).long()
    k = int(labels.max()) + 1 if k is None else k
    return torch.nn.functional.one_hot(labels, k).double()


def _sizes(h: torch.Tensor) -> torch.Tensor:
    return h.sum(0)


def info_total(q: torch.Tensor) -> float:
    """I(X;Y) = ||Q||_F^2 - 1 = sum_{i>=2} s_i^2."""
    return float(q.double().square().sum()) - 1.0


def info_onesided(q: torch.Tensor, lab_x) -> float:
    """I(c(X); Y): start side coarse-grained, end side fine."""
    h = onehot(lab_x).to(q.device)
    h = h[:, _sizes(h) > 0]
    s = q.double() @ h                       # (N_y, k): S[y, j] = sum_{x in A_j} Q[y, x]
    return float((s.square().sum(0) / _sizes(h)).sum()) - 1.0


def info_onesided_end(q: torch.Tensor, lab_y) -> float:
    """I(X; c'(Y)): end side coarse-grained."""
    return info_onesided(q.T, lab_y)


def info_twosided(q: torch.Tensor, lab_x, lab_y) -> tuple[float, torch.Tensor]:
    """I(c(X); c'(Y)) and the block mass matrix T[l, j] = sum_{x in A_j, y in B_l} Q[y, x]."""
    hx = onehot(lab_x).to(q.device); hx = hx[:, _sizes(hx) > 0]
    hy = onehot(lab_y).to(q.device); hy = hy[:, _sizes(hy) > 0]
    t = hy.T @ q.double() @ hx
    val = float((t.square() / (_sizes(hy)[:, None] * _sizes(hx)[None, :])).sum()) - 1.0
    return val, t


def leakage(m: torch.Tensor, lab) -> torch.Tensor:
    """phi(A_j) = P(X_1 not in A_j | X_0 in A_j) for a symmetric doubly stochastic M."""
    h = onehot(lab).to(m.device); h = h[:, _sizes(h) > 0]
    stay = (h * (m.double() @ h)).sum(0) / _sizes(h)
    return 1.0 - stay


def floors(sv: torch.Tensor) -> torch.Tensor:
    """F_k = sum_{i=2}^k s_i^2 for k = 1..len(sv) (index k-1); sv[0] is the trivial 1."""
    sq = sv.double().square().clone(); sq[0] = 0
    return torch.cumsum(sq, 0)


def ky_fan(eigs: torch.Tensor) -> torch.Tensor:
    """sum_{i=1}^k (1 - lambda_i) for k = 1..len (index k-1); lower bound on Ncut of any k-partition."""
    return torch.cumsum(1.0 - eigs.double(), 0)


def within_variance(psi: torch.Tensor, lab) -> float:
    """(1/N) sum_x ||psi(x) - mean_{block(x)} psi||^2."""
    h = onehot(lab).to(psi.device); h = h[:, _sizes(h) > 0]
    means = (h.T @ psi) / _sizes(h)[:, None]
    return float((psi - h @ means).square().sum(1).mean())


def kmeans(x: torch.Tensor, k: int, restarts: int = 8, iters: int = 100, seed: int = 0) -> torch.Tensor:
    """Lloyd k-means, k-means++ init, uniform weights; restarts run as a batch, best within-cluster sum kept."""
    x = x.float()
    n, d = x.shape
    R = restarts
    gen = torch.Generator(device=x.device).manual_seed(seed)
    first = torch.randint(n, (R,), generator=gen, device=x.device)
    c = torch.empty(R, k, d, device=x.device)
    c[:, 0] = x[first]
    d2 = (x[None] - c[:, :1]).square().sum(-1)                    # (R, n)
    for j in range(1, k):
        p = d2.clamp_min(0) + 1e-30
        idx = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
        c[:, j] = x[idx]
        d2 = torch.minimum(d2, (x[None] - c[:, j:j + 1]).square().sum(-1))
    lab = None
    for _ in range(iters):
        dist = torch.cdist(x[None].expand(R, n, d), c).square()    # (R, n, k)
        new = dist.argmin(-1)
        if lab is not None and torch.equal(new, lab):
            break
        lab = new
        sums = torch.zeros(R, k, d, device=x.device).scatter_add_(1, lab[..., None].expand(R, n, d), x[None].expand(R, n, d))
        cnt = torch.zeros(R, k, device=x.device).scatter_add_(1, lab, torch.ones(R, n, device=x.device))
        empty = cnt == 0
        c = torch.where(empty[..., None], c, sums / cnt.clamp_min(1)[..., None])
        if empty.any():  # re-seed empty clusters at each run's worst-fit point
            far = x[dist.min(-1).values.argmax(1)]                 # (R, d)
            c = torch.where(empty[..., None], far[:, None], c)
    cost = (x[None] - torch.gather(c, 1, lab[..., None].expand(R, n, d))).square().sum((1, 2))
    return lab[int(cost.argmin())].clone()


def match_pairs(t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of end blocks (rows) to start blocks (cols) maximising matched mass."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-t.cpu().numpy())
    return r, c


def matched_agreement(t: torch.Tensor) -> dict:
    """Hungarian-matched agreement between start blocks (cols of T) and end blocks (rows of T).

    T[l, j] = sum_{x in A_j, y in B_l} Q[y, x]; with Q doubly stochastic, column sums are |A_j| and row sums |B_l|.
    Returns matched mass / N, its independence expectation for the same matching, and forward/reverse rates.
    An agreement score, not a coherence certificate.
    """
    t = t.double()
    n = float(t.sum())
    size_a, size_b = t.sum(0), t.sum(1)
    r, c = match_pairs(t)
    m = t[r, c]
    return dict(agreement=float(m.sum()) / n,
                independent=float((size_a[c] * size_b[r]).sum()) / n ** 2,
                forward=(m / size_a[c]).cpu().numpy(), reverse=(m / size_b[r]).cpu().numpy(),
                mass_a=(size_a[c] / n).cpu().numpy(), mass_b=(size_b[r] / n).cpu().numpy())


def participation(v: torch.Tensor) -> torch.Tensor:
    """Participation ratio (sum v^2)^2 / (N sum v^4) per column, in (0, 1]; small = localised."""
    v2 = v.double().square()
    return v2.sum(0).square() / (len(v) * v2.square().sum(0))


def hard_argmax(s: torch.Tensor) -> torch.Tensor:
    """Every point to its largest SEBA vector, ignoring the threshold; ties to the lower index; relabel nonempty."""
    lab = s.argmax(1)
    return torch.unique(lab, return_inverse=True)[1]


def svd_desc(q: torch.Tensor, n: int | None = None):
    """Full SVD of a doubly stochastic Q, trivial pair placed first exactly.

    The constant is a right and left singular vector with value 1; the remaining pairs come from the SVD of the
    doubly centred Q, whose top singular values are then the nontrivial ones without deflation.
    """
    q = q.double()
    N = len(q)
    c = q - q.mean(0, keepdim=True) - q.mean(1, keepdim=True) + q.mean()
    a, s, bt = torch.linalg.svd(c)
    one = torch.full((N, 1), N ** -.5, dtype=q.dtype, device=q.device)
    # The constant sits in the (numerical) null space of c; drop the smallest pair.
    a = torch.cat([one, a[:, :N - 1]], 1)
    b = torch.cat([one, bt.T[:, :N - 1]], 1)
    s = torch.cat([torch.ones(1, dtype=q.dtype, device=q.device), s[:N - 1]])
    if n is not None:
        a, s, b = a[:, :n], s[:n], b[:, :n]
    return a, s, b


def embedding(vectors: torch.Tensor, sv: torch.Tensor, k: int) -> torch.Tensor:
    """Top (k-1) nontrivial coordinates sqrt(N) s_i w_i (i = 2..k)."""
    N = len(vectors)
    return N ** .5 * vectors[:, 1:k] * sv[1:k]


def gap_report(sv: torch.Tensor, kmax: int = 32) -> dict:
    """Descriptive gap statistics over k in [2, kmax] (sv[0] is the trivial 1; sv[k-1] is sigma_k)."""
    s = sv.double().cpu().numpy()
    ks = np.arange(2, kmax + 1)
    lo, hi = s[ks - 1], s[ks]
    gap = lo - hi
    ok = hi > 1e-8
    ratio = np.where(ok, lo / np.maximum(hi, 1e-300), np.nan)
    order = np.argsort(-np.nan_to_num(ratio, nan=-1))
    top, second = ratio[order[0]], ratio[order[1]]
    clear = bool(np.isfinite(top) and top >= 1.15 and (not np.isfinite(second) or top > 1.1 * second))
    return dict(k_gap=int(ks[gap.argmax()]), gap=float(gap.max()),
                k_ratio=int(ks[order[0]]), ratio=float(top), ratio_second=float(second),
                k_ratio_second=int(ks[order[1]]), clear=clear,
                ties=[int(k) for k in ks[np.abs(gap) < 1e-10]], tiny=[int(k) for k in ks[~ok]])


# ---------------------------------------------------------------- Markov closure (SPEC_COARSE.md section 6)
# Convention: m[y, x] = P(Z_{s+1} = y | Z_s = x), columns sum to 1; Z_s uniform (every operator is doubly stochastic).

def _xlogx_mi(j: torch.Tensor) -> float:
    """Shannon mutual information (nats) of a 2-d joint distribution."""
    j = j.double()
    pa, pb = j.sum(1, keepdim=True), j.sum(0, keepdim=True)
    nz = j > 0
    return float((j[nz] * (j[nz] / (pa @ pb)[nz]).log()).sum())


def _chi2_mi(j: torch.Tensor) -> float:
    j = j.double()
    pa, pb = j.sum(1, keepdim=True), j.sum(0, keepdim=True)
    d = pa @ pb
    nz = d > 0
    return float((j[nz].square() / d[nz]).sum()) - 1.0


def _cmi(j3: torch.Tensor) -> float:
    """I(A; C | B) for a joint p(a, b, c) (Shannon, nats)."""
    j3 = j3.double()
    pb = j3.sum((0, 2))
    pab, pbc = j3.sum(2), j3.sum(0)
    num = j3 * pb[None, :, None]
    den = pab[:, :, None] * pbc[None, :, :]
    nz = (j3 > 0) & (den > 0)
    return float((j3[nz] * (num[nz] / den[nz]).log()).sum())


def closure_step(m: torch.Tensor, lab_a, lab_b) -> dict:
    """One-step closure defect for Z_s -> Z_{s+1} with partitions c_s (lab_a) and c_{s+1} (lab_b)."""
    n = len(m)
    ha, hb = onehot(lab_a).to(m.device), onehot(lab_b).to(m.device)
    ha, hb = ha[:, _sizes(ha) > 0], hb[:, _sizes(hb) > 0]
    jx = (hb.T @ m.double()) / n                # (k_b, N): P(C_{s+1} = j, Z_s = x)
    jc = jx @ ha                                # (k_b, k_a)
    out = dict(I_fine=_xlogx_mi(jx), I_coarse=_xlogx_mi(jc), X_fine=_chi2_mi(jx), X_coarse=_chi2_mi(jc))
    out['defect'] = out['I_fine'] - out['I_coarse']
    out['defect_chi2'] = out['X_fine'] - out['X_coarse']
    return out


def closure_triple(m1: torch.Tensor, m2: torch.Tensor, la, lb, lc) -> dict:
    """Two-step memory for Z_s -> Z_{s+1} (m1) -> Z_{s+2} (m2) with partitions la, lb, lc."""
    n = len(m1)
    ha, hb, hc = (onehot(l).to(m1.device) for l in (la, lb, lc))
    ha, hb, hc = (h[:, _sizes(h) > 0] for h in (ha, hb, hc))
    m1, m2 = m1.double(), m2.double()
    cy = hc.T @ m2                              # (k_c, N_y): P(C_{s+2} = l | Z_{s+1} = y)
    # fine: p(x, j, l) = (1/N) sum_{y in B_j} m1[y, x] cy[l, y]
    fine = torch.einsum('yx,yj,ly->xjl', m1, hb, cy) / n
    coarse = torch.einsum('xjl,xi->ijl', fine, ha)
    jl = coarse.sum(0)                          # p(j, l)
    pj = jl.sum(1, keepdim=True)
    nz = jl > 0
    h_cond = float(-(jl[nz] * (jl[nz] / pj.expand_as(jl)[nz]).log()).sum())   # H(C_{s+2} | C_{s+1})
    return dict(cmi_coarse=_cmi(coarse), cmi_fine=_cmi(fine), I_next=_xlogx_mi(jl), H_cond=h_cond)


def nmi(la, lb) -> float:
    """Normalised mutual information (arithmetic mean of entropies) of two labelings of the same images."""
    ha, hb = onehot(la).double(), onehot(lb).double()
    j = ha.T @ hb / len(ha)
    j = j[j.sum(1) > 0][:, j.sum(0) > 0]
    pa, pb = j.sum(1), j.sum(0)
    ent = lambda p: float(-(p[p > 0] * p[p > 0].log()).sum())
    d = (ent(pa) + ent(pb)) / 2
    return _xlogx_mi(j) / d if d > 0 else float('nan')


def uniform_lazy(n: int, dtype=torch.float64, device='cpu') -> torch.Tensor:
    """U = (I + (J - I)/(N - 1))/2: the lazy walk with no geometry (matched control)."""
    u = torch.full((n, n), 0.5 / (n - 1), dtype=dtype, device=device)
    u.fill_diagonal_(0.5)
    return u


def coarse_transition(q: torch.Tensor, lab_a, lab_b) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-stochastic P(C_b = j | C_a = i) for coupling q[y, x] (Z_a = x, Z_b = y), and P(C_a)."""
    ha, hb = onehot(lab_a).to(q.device), onehot(lab_b).to(q.device)
    ha, hb = ha[:, _sizes(ha) > 0], hb[:, _sizes(hb) > 0]
    j = hb.T @ q.double() @ ha                  # counts / ... : column sums = |A_i|
    pa = j.sum(0)
    return j / pa, pa / pa.sum()


def ck_error(t_true: torch.Tensor, t_ck: torch.Tensor, pa: torch.Tensor) -> tuple[float, float]:
    """Mass-weighted TV between true and composed coarse transitions, and TV of true from independence."""
    tv = 0.5 * (t_true - t_ck).abs().sum(0)
    pb = t_true @ pa
    tv_ind = 0.5 * (t_true - pb[:, None]).abs().sum(0)
    return float((pa * tv).sum()), float((pa * tv_ind).sum())
