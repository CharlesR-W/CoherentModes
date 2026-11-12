"""Core math for the exploratory CIFAR coherent-sets study (torch, device-agnostic).

Points are a fixed probe set of images; each site (layer) l gives a centred-cosine
Gram matrix C_l. Final affinities are union-kNN, symmetrized per-point perplexity,
or vMF K_l = exp(kappa_l (C_l - 1)); see sweep_math.final_kernel_bank.
The schema-v2 pipeline zeros the diagonal, symmetrically scales to a doubly
stochastic P with shared constant stationary mode, then uses M=(I+P)/2.
`diffusion_operator` keeps Coifman-Lafon alpha / degree normalization for the
legacy comparison; retained-diagonal Sinkhorn is a separate diagnostic.
For a block of sites [a, b] the
"dynamic" operator is the average Mbar = mean_l M_l (equivalently, the averaged
normalised graph Laplacian I - Mbar), a symmetric discretisation in the spirit
of Froyland's dynamic Laplacian / Banisch-Koltai space-time diffusion maps with
uniform mass on trajectories (each image is one trajectory through the layers).
Leading eigenvectors of Mbar are the block's coherent modes; SEBA (Froyland,
Rock & Sakellariou 2019) rotates the leading r of them into sparse,
nearly disjoint coherent sets.
"""
from __future__ import annotations

import torch
from seba import seba, seba_reconstruct, seba_labels


def centred_cosine_gram(x: torch.Tensor) -> torch.Tensor:
    """x: (N, D). Centre over the probe set, normalise rows, return C = X X^T (N, N)."""
    x = x.reshape(x.shape[0], -1).double()
    x = x - x.mean(0, keepdim=True)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    c = x @ x.T
    return c.clamp(-1.0, 1.0)


def knn_dist(c: torch.Tensor, k: int) -> torch.Tensor:
    """Per-point cosine distance (1 - cos) to the k-th nearest other point."""
    d = 1.0 - c
    d = d + torch.diag(torch.full((c.shape[0],), float("inf"), dtype=d.dtype, device=d.device))
    return d.kthvalue(k, dim=1).values


def participation_ratio(c: torch.Tensor) -> float:
    """Effective rank of the Gram matrix: (sum lam)^2 / sum lam^2."""
    lam = torch.linalg.eigvalsh(c).clamp_min(0)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def diffusion_operator(c: torch.Tensor, kappa: float, alpha: float = 1.0) -> torch.Tensor:
    """Symmetric diffusion operator of the vMF kernel exp(kappa (C - 1))."""
    k = torch.exp(kappa * (c - 1.0))
    if alpha:
        q = k.sum(1)
        k = k / (q[:, None] * q[None, :]) ** alpha
    d = k.sum(1).rsqrt()
    return d[:, None] * k * d[None, :]


def block_modes(ms: list[torch.Tensor], n_modes: int, shared_stationary=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading averaged eigenpairs; optionally resolve the exact shared constant.

    Near-identity kernels have tiny eigengaps: even float64 eigh may mix the
    known stationary function with nontrivial modes. A Householder reflection
    takes the constant to e_0; eigensolving its trailing principal submatrix
    therefore works in the exact constant-orthogonal subspace, without trusting
    the numerical leading eigenvector. The reflection costs O(N^2), not a dense
    change-of-basis multiplication. Legacy operators leave this option off.
    """
    mbar = torch.stack(ms).mean(0) if len(ms) > 1 else ms[0]
    if shared_stationary:
        n = len(mbar)
        if float((mbar.sum(1)-1).abs().max()) > 1e-10:
            raise ValueError('Shared-stationary spectrum requires a doubly stochastic operator')
        constant = torch.ones(n, dtype=mbar.dtype, device=mbar.device) / n**.5
        h = constant.clone(); h[0] -= 1; h /= h.norm()
        mh = mbar @ h
        transformed = mbar - 2*h[:,None]*mh[None,:] - 2*mh[:,None]*h[None,:] + 4*torch.dot(h,mh)*h[:,None]*h[None,:]
        lam, w = torch.linalg.eigh(transformed[1:,1:])
        lam = lam.flip(0)[:n_modes-1]
        w = w.flip(1)[:,:n_modes-1]
        lifted = torch.cat([torch.zeros((1,n_modes-1),dtype=w.dtype,device=w.device),w])
        lifted -= 2*h[:,None]*(h @ lifted)[None,:]
        vals = torch.cat([torch.ones(1,dtype=lam.dtype,device=lam.device),lam])
        vecs = torch.cat([constant[:,None],lifted],1)
    else:
        vals, vecs = torch.linalg.eigh(mbar)
        vals, vecs = vals.flip(0)[:n_modes], vecs.flip(1)[:, :n_modes]
    # deterministic sign: largest-magnitude entry positive
    idx = vecs.abs().argmax(0)
    s = torch.sign(vecs[idx, torch.arange(vecs.shape[1], device=vecs.device)])
    s[s == 0] = 1
    return vals, vecs * s


def symmetric_balance(logk, tol=2e-12, max_iter=80, return_info=False):
    """Log-domain symmetric Sinkhorn with Newton acceleration.

    Minimize .5 sum_ij exp(logK_ij+u_i+u_j) - sum_i u_i.
    Its gradient is row_sum-1 and Hessian diag(row_sum)+P. Ten
    multiplicative-equivalent warmup steps precede damped Newton steps. This
    avoids tens of thousands of fixed-point iterations near bipartite pairs;
    the returned operator must satisfy the SAME strict row-sum criterion.
    """
    logk=logk.double()
    u=-.5*torch.logsumexp(logk,dim=1)
    newton_steps=0
    for iteration in range(max_iter):
        logp=logk+u[:,None]+u[None,:]
        p=torch.exp(logp);rows=p.sum(1);error=float((rows-1).abs().max())
        if error<tol:
            info=dict(iterations=iteration,row_error=error,newton_steps=newton_steps,
                      method='10 damped log-Sinkhorn warmups then line-searched Newton')
            return (p,info) if return_info else p
        if iteration<10:
            u-=.5*torch.log(rows)
            continue
        h=p.clone();h.diagonal().add_(rows)
        direction=torch.linalg.solve(h,1-rows)
        objective=.5*p.sum()-u.sum()
        directional=(rows-1)@direction
        accepted=False
        for backtrack in range(40):
            alpha=2.**(-backtrack);candidate=u+alpha*direction
            trial=torch.exp(logk+candidate[:,None]+candidate[None,:])
            objective_new=.5*trial.sum()-candidate.sum()
            error_new=float((trial.sum(1)-1).abs().max())
            if error_new<error or bool(objective_new<=objective+1e-4*alpha*directional):
                u=candidate;accepted=True;break
        if not accepted:raise RuntimeError(f'Log-Sinkhorn Newton line search stalled at residual {error}')
        newton_steps+=1
    raise RuntimeError(f'Log-Sinkhorn did not converge: row residual {error}')


def sinkhorn_operator(c: torch.Tensor, kappa: float, tol=2e-12, max_iter=10000,
                      remove_diagonal=False):
    """Symmetric scaling K*v*v.T; all cells share counting measure.

    With remove_diagonal=True use log-domain scaling of the positive off-diagonal
    kernel. For N>=3 it has total support; its symmetric stochastic scaling exists
    and is unique, but need not be PSD. The retained-diagonal variant is kept for
    comparison and uses the faster multiplicative iteration. Both fail explicitly
    on nonconvergence, never return an unchecked scaling.
    """
    logk = float(kappa) * (c.double() - 1)
    if remove_diagonal:
        if len(c)<3:
            raise ValueError('Zero-diagonal scaling requires N>=3')
        logk.diagonal().fill_(float('-inf'))
        return symmetric_balance(logk,tol=tol,max_iter=min(max_iter,80))
    k = torch.exp(logk)
    v = k.sum(1).rsqrt()
    for _ in range(max_iter):
        kv = k @ v
        if float((v * kv - 1).abs().max()) < tol:
            return v[:, None] * k * v[None, :]
        v = (v / kv).sqrt()
    raise RuntimeError('Sinkhorn did not converge')


def fixed_lazy_operator(c: torch.Tensor, kappa: float):
    """Exploratory variant: zero-diagonal Sinkhorn P, then M=(I+P)/2.

    Fixed self-mass 1/2 removes temperature-dependent self-loop dominance while
    shifting the entire spectrum into [0,1] (no alternating singular modes).
    """
    p = sinkhorn_operator(c, kappa, remove_diagonal=True)
    m = p * .5
    m.diagonal().add_(.5)
    return m


def nn_mass(c: torch.Tensor, kappa: float, k=15):
    """Median raw-kernel mass in the k nearest NONSELF neighbours."""
    d = 1 - c.double()
    d.diagonal().fill_(float('inf'))
    # Subtract row minima: ratio unchanged, but no high-kappa underflow.
    d = d - d.min(1).values[:, None]
    weights = torch.exp(-float(kappa) * d)
    nearest = d.topk(k, largest=False, dim=1).indices
    ratios = weights.gather(1, nearest).sum(1) / weights.sum(1)
    return float(ratios.quantile(0.5))


def kappa_modes(c: torch.Tensor, k=15):
    n = len(c)
    d = 1 - c.double()
    d.diagonal().fill_(float('inf'))
    dk = float(d.kthvalue(k, dim=1).values.quantile(0.5))
    pair = float(d[torch.isfinite(d)].quantile(0.5))
    if dk <= 0:
        raise ValueError('kNN-scale is undefined for zero median neighbour distance')
    nearest = d.topk(k, largest=False, dim=1).values
    shifted = d - nearest[:, :1]
    near_shifted = nearest - nearest[:, :1]
    def mass(kap):
        return float((torch.exp(-kap * near_shifted).sum(1) /
                      torch.exp(-kap * shifted).sum(1)).quantile(.5))
    lo, hi = 0., 1.
    while mass(hi) < .5:
        hi *= 2
        if hi > 1e8:
            raise ValueError('half-mass calibration has no finite solution (distance ties)')
    for _ in range(40):
        mid = (lo + hi) / 2
        if mass(mid) < .5:
            lo = mid
        else:
            hi = mid
    ks = [1 / dk, __import__('math').log(n / k) / max(pair - dk, 1e-3), (lo + hi) / 2, 5., 30.]
    return ks, [mass(kap) for kap in ks]
