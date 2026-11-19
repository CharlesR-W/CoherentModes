"""Core operator math for the ViT smoothness-across-depth side study (PLAN.md Sections 2, 3, 6).

Conventions (all functions; torch, float64, run on the device of their inputs):
- Sites are rows. A view is an (N, D) tensor; `cosine_view` centres over rows and L2-normalises.
- d_ij^2 = 2 - 2 cos_ij on the normalised view (chord distance on the unit sphere).
- Kernels: K-st(k)  K_ij = exp(-d_ij^2 / (s_i s_j)), s_i = chord distance from i to its k-th nearest
  *other* point (self excluded); K-vmf(kappa)  K_ij = exp(kappa (cos_ij - 1)). Diagonal = 1.
- P = diag(d) K diag(d), symmetric bistochastic (P 1 = 1), from symmetric Sinkhorn.
- R = P^2 (symmetric, bistochastic, PSD); R^t = P^(2t) = U diag(lam^(2t)) U^T.
- (lam, U) are eigenpairs of P, lam descending, U[:, 0] = 1/sqrt(N) (the constant mode).
- A function on sites is a column of length N; "unit centred" = mean 0 and 2-norm 1 over the N sites,
  so smoothness s^t(f) = f^T R^t f lies in [0, 1].
- Memory: no function holds more than ~4 dense N x N float64 matrices at once (inputs included where
  noted), so N = 10,000 (800 MB per matrix) fits a 24 GB GPU.
"""
from __future__ import annotations

import math

import torch

F64 = torch.float64

# --------------------------------------------------------------------------------------------- helpers


def _f64(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == F64 else x.to(F64)


def _col(f: torch.Tensor) -> tuple[torch.Tensor, bool]:
    f = _f64(f)
    if f.dim() == 1:
        return f[:, None], True
    return f, False


def unit_centred(f: torch.Tensor) -> torch.Tensor:
    """Centre each column over sites and scale to unit 2-norm. Accepts (N,) or (N, k)."""
    f, vec = _col(f)
    f = f - f.mean(0, keepdim=True)
    n = f.norm(dim=0, keepdim=True)
    if (n < 1e-300).any():
        raise ValueError("unit_centred: a column is constant (zero after centring)")
    f = f / n
    return f[:, 0] if vec else f


def fix_signs(U: torch.Tensor) -> torch.Tensor:
    """In place: make the largest-|entry| of each column positive (first index on ties)."""
    if U.shape[1] == 0:
        return U
    idx = U.abs().argmax(0)
    s = torch.sign(U[idx, torch.arange(U.shape[1], device=U.device)])
    s[s == 0] = 1
    U.mul_(s[None, :])
    return U


def contrast_basis(labels: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (N, G-1) of the span of centred indicator functions of the G label values.
    For validation/test use only (labels never define a kernel or basis inside the construction)."""
    labels = torch.as_tensor(labels)
    vals = torch.unique(labels)
    ind = (labels[:, None] == vals[None, :].to(labels.device)).to(F64)
    ind = ind - ind.mean(0, keepdim=True)
    u, s, _ = torch.linalg.svd(ind, full_matrices=False)
    return u[:, : int((s > 1e-10 * s.max()).sum())]


def negative_part(lam: torch.Tensor) -> dict:
    """Negative part of a spectrum of P: min eigenvalue and sum(neg^2)/sum(all^2)."""
    lam = _f64(lam)
    tot = float((lam ** 2).sum())
    neg = float((lam.clamp_max(0) ** 2).sum())
    return {"min_eig": float(lam.min()), "neg_sq_frac": neg / tot if tot > 0 else 0.0,
            "n_negative": int((lam < 0).sum())}


def eff_num(psi: torch.Tensor) -> torch.Tensor:
    """Localisation: effective number of sites 1 / sum_x psi(x)^4 for unit-norm columns."""
    psi, vec = _col(psi)
    psi = psi / psi.norm(dim=0, keepdim=True)
    e = 1.0 / (psi ** 4).sum(0)
    return e[0] if vec else e


# ---------------------------------------------------------------------------------------- views


def cosine_view(v: torch.Tensor, centre: bool = True) -> torch.Tensor:
    """(N, ...) -> (N, D) float64: centre over rows (optional), then L2-normalise each row."""
    v = _f64(v.reshape(v.shape[0], -1))
    if centre:
        v = v - v.mean(0, keepdim=True)
    n = v.norm(dim=1, keepdim=True)
    if (n < 1e-12).any():
        raise ValueError("cosine_view: a row has (near) zero norm; cosine undefined")
    return v / n


# -------------------------------------------------------------------------------------- kernels


def _kth_cos(C: torch.Tensor, k: int) -> torch.Tensor:
    """k-th largest entry per row (cosine of the k-th nearest neighbour)."""
    return C.topk(k, dim=1).values[:, -1]


def _chord(ck: torch.Tensor) -> torch.Tensor:
    s = (2.0 - 2.0 * ck).clamp_min(0).sqrt()
    if (s <= 0).any():
        raise ValueError("K-st: a k-th-NN distance is 0 (>= k duplicate points); bandwidth undefined")
    return s


def knn_scale(x: torch.Tensor, k: int) -> torch.Tensor:
    """s_i = chord distance from i to its k-th nearest other point (self excluded). One N x N buffer."""
    x = _f64(x)
    C = x @ x.T
    C.fill_diagonal_(-math.inf)
    s = _chord(_kth_cos(C.clamp_(max=1.0), k))
    return s


def kernel_st(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Self-tuned kernel on a normalised view x (N, D). Returns (K (N, N), s (N,)). One N x N buffer."""
    x = _f64(x)
    C = x @ x.T
    C.clamp_(-1.0, 1.0)
    C.fill_diagonal_(-math.inf)
    s = _chord(_kth_cos(C, k))
    C.fill_diagonal_(1.0)
    C.mul_(-2.0).add_(2.0).clamp_min_(0.0)          # d^2
    C.div_(s[:, None]).div_(s[None, :])
    C.neg_().exp_()
    return C, s


def kernel_vmf(x: torch.Tensor, kappa: float) -> torch.Tensor:
    """vMF kernel exp(kappa (cos - 1)) on a normalised view x (N, D). One N x N buffer."""
    x = _f64(x)
    C = x @ x.T
    C.clamp_(-1.0, 1.0)
    C.fill_diagonal_(1.0)
    C.sub_(1.0).mul_(kappa).exp_()
    return C


def kernel_rows_st(x_new: torch.Tensor, x_ref: torch.Tensor, s_ref: torch.Tensor, k: int) -> torch.Tensor:
    """K-st rows (M, N) from new points to a reference set. The new point's own s is its chord
    distance to its k-th nearest neighbour *in the reference set*."""
    x_new, x_ref, s_ref = _f64(x_new), _f64(x_ref), _f64(s_ref)
    C = x_new @ x_ref.T
    C.clamp_(-1.0, 1.0)
    s_new = _chord(_kth_cos(C, k))
    C.mul_(-2.0).add_(2.0).clamp_min_(0.0)
    C.div_(s_new[:, None]).div_(s_ref[None, :])
    C.neg_().exp_()
    return C


def kernel_rows_vmf(x_new: torch.Tensor, x_ref: torch.Tensor, kappa: float) -> torch.Tensor:
    """vMF kernel rows (M, N) from new points to a reference set."""
    C = _f64(x_new) @ _f64(x_ref).T
    C.clamp_(-1.0, 1.0)
    C.sub_(1.0).mul_(kappa).exp_()
    return C


# -------------------------------------------------------------------------------- normalisation


def sinkhorn(K: torch.Tensor, tol: float = 1e-10, max_iter: int = 100_000,
             inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric Sinkhorn: find d > 0 with d_i (K d)_i = 1, return (P = diag(d) K diag(d), d).

    Iteration d <- sqrt(d / (K d)); near the fixed point the error e (d = d*(1+e)) maps to
    (I - P) e / 2, so it contracts at rate max (1 - lam_P)/2 <= 1 unless P has eigenvalues near -1.
    Converged when max_i |row sum of P - 1| < tol; raises RuntimeError otherwise.
    If inplace, K is overwritten by P (saves one N x N buffer)."""
    K = _f64(K)
    d = K.sum(1).rsqrt()
    err = math.inf
    for it in range(max_iter):
        Kd = K @ d
        err = float((d * Kd - 1.0).abs().max())
        if err < tol:
            break
        d = torch.sqrt(d / Kd)
    else:
        raise RuntimeError(f"sinkhorn: no convergence after {max_iter} iterations (err {err:.3e})")
    P = K if inplace else K.clone()
    P.mul_(d[:, None]).mul_(d[None, :])
    return P, d


def coifman_lafon(K: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Comparison only (PLAN §2): alpha-normalised symmetric operator M = D^-1/2 K_a D^-1/2 with
    K_a = K / (q q^T)^alpha, q = K 1, D = diag(K_a 1). Returns (M, pi) where pi = sqrt(degree) is
    M's top eigenvector (not constant, hence not usable for comparing/multiplying operators)."""
    K = _f64(K)
    q = K.sum(1)
    Ka = K / (q[:, None] ** alpha) / (q[None, :] ** alpha)
    deg = Ka.sum(1)
    Ka.mul_(deg.rsqrt()[:, None]).mul_(deg.rsqrt()[None, :])
    return Ka, deg.sqrt() / deg.sqrt().norm()


# ------------------------------------------------------------------------------------- spectra


def eig_full(P: torch.Tensor, check_tol: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Full eigendecomposition of a symmetric bistochastic P. Returns (lam (N,), U (N, N)), lam
    descending, U[:, 0] = +1/sqrt(N) exactly, other columns sign-fixed (largest |entry| positive).

    The constant eigenvalue is lifted from 1 to 2 (P + 11^T/N) before eigh so that U[:, 0] is the
    constant even when the eigenvalue 1 is (near-)degenerate (disconnected or weakly coupled sites);
    the remaining eigenvectors are then orthogonal to constants. P is restored afterwards.
    Within (near-)degenerate eigenspaces the basis is whatever LAPACK/cuSOLVER returns."""
    P = _f64(P)
    N = P.shape[0]
    P.add_(1.0 / N)
    try:
        lam, U = torch.linalg.eigh(P)
    finally:
        P.sub_(1.0 / N)
    lam = lam.flip(0)
    U = U.flip(1)
    const_dev = float((U[:, 0].abs() - 1.0 / math.sqrt(N)).abs().max())
    if abs(float(lam[0]) - 2.0) > check_tol or const_dev > check_tol:
        raise RuntimeError(f"eig_full: constant mode check failed (lam0-2={float(lam[0]) - 2:.2e}, "
                           f"dev={const_dev:.2e}); is P symmetric bistochastic?")
    if float(lam[1]) > 1.0 + check_tol:
        raise RuntimeError(f"eig_full: eigenvalue {float(lam[1])} > 1; P is not bistochastic")
    lam[0] = 1.0
    U[:, 0] = 1.0 / math.sqrt(N)
    fix_signs(U[:, 1:])
    return lam, U


def _weights(lam: torch.Tensor, t: float) -> torch.Tensor:
    """sqrt of the R^t eigenvalues: |lam|^t (so that lam^(2t) = w^2)."""
    return _f64(lam).abs() ** t


def smooth_op(lam: torch.Tensor, U: torch.Tensor, t: float) -> torch.Tensor:
    """R^t = P^(2t) = U diag(lam^(2t)) U^T (N, N). Holds U, one scaled copy and the output."""
    X = U * _weights(lam, t)[None, :]
    return X @ X.T


def smoothness(lam: torch.Tensor, U: torch.Tensor, f: torch.Tensor, t: float) -> torch.Tensor:
    """s^t(f) = f^T R^t f for each column of f, after centring and unit-normalising it.
    f (N,) -> scalar tensor; f (N, k) -> (k,)."""
    f, vec = _col(f)
    f = unit_centred(f)
    c = U.T @ f
    s = ((c ** 2) * (_weights(lam, t) ** 2)[:, None]).sum(0)
    return s[0] if vec else s


def delta_modes(lamA: torch.Tensor, UA: torch.Tensor, lamB: torch.Tensor, UB: torch.Tensor,
                t: float, m: int) -> dict:
    """Endpoint smoothness-change operator Delta = R_B^t - R_A^t and its top-m / bottom-m eigenpairs
    on the centred subspace. Delta annihilates constants; the constant direction is lifted to
    eigenvalue 2 (Delta + 2 11^T/N) before eigh and dropped, so no returned mode is (or mixes with)
    the constant even when Delta has a large (near-)null space. Returns dict with
      pos_vals (m,) descending, pos_vecs (N, m)  ("became smooth" from A to B),
      neg_vals (m,) ascending (most negative first), neg_vecs (N, m)  ("stopped being smooth"),
      n_pos, n_neg: how many eigenvalues are > 0 / < 0 (pos/neg lists are padded with the
                    next eigenpairs by value if fewer than m have the sign; check the values),
      all_vals (N-1,) the spectrum of Delta on centred functions, ascending.
    Peak memory: UA, UB (inputs), one scaled buffer, Delta; then Delta + its eigenvectors."""
    N = UA.shape[0]
    X = UB * _weights(lamB, t)[None, :]
    D = X @ X.T
    torch.mul(UA, _weights(lamA, t)[None, :], out=X)
    D.addmm_(X, X.T, alpha=-1.0)
    del X
    D.add_(2.0 / N)
    vals, V = torch.linalg.eigh(D)
    del D
    cdev = float((V[:, -1].abs() - 1.0 / math.sqrt(N)).abs().max())
    if abs(float(vals[-1]) - 2.0) > 1e-8 or cdev > 1e-6:
        raise RuntimeError(f"delta_modes: constant not annihilated (lam-2={float(vals[-1]) - 2:.2e}, dev={cdev:.2e})")
    vals, V = vals[:-1], V[:, :-1]
    pos_vecs = V[:, -m:].flip(1).clone()
    neg_vecs = V[:, :m].clone()
    del V
    return {"pos_vals": vals[-m:].flip(0).clone(), "pos_vecs": fix_signs(pos_vecs),
            "neg_vals": vals[:m].clone(), "neg_vecs": fix_signs(neg_vecs),
            "n_pos": int((vals > 0).sum()), "n_neg": int((vals < 0).sum()), "all_vals": vals}


def delta_rayleigh(lamA, UA, lamB, UB, f, t) -> torch.Tensor:
    """f^T (R_B^t - R_A^t) f for unit-centred columns of f, without forming Delta."""
    return smoothness(lamB, UB, f, t) - smoothness(lamA, UA, f, t)


def per_layer_terms(eigs: list, f: torch.Tensor, t: float) -> dict:
    """Per-layer decomposition of an endpoint smoothness change along a path of layers.
    eigs: list of L (lam, U) pairs. f: (N,) or (N, k), centred and unit-normalised internally.
    Returns dict: smooth (L[, k]) = s_l^t(f); terms (L-1[, k]) = f^T (R_{l+1}^t - R_l^t) f;
    net = s_last - s_first (= sum of terms); path = sum_l |term_l|."""
    s = torch.stack([smoothness(lam, U, f, t) for lam, U in eigs])
    terms = s[1:] - s[:-1]
    return {"smooth": s, "terms": terms, "net": s[-1] - s[0], "path": terms.abs().sum(0)}


# ------------------------------------------------------------------------------- out-of-sample


def extend(Krows: torch.Tensor, d: torch.Tensor, psi: torch.Tensor, lam: torch.Tensor,
           floor: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinkhorn-consistent Nystrom extension (PLAN §2a).
    Krows (M, N) kernel rows of new points to the reference set, d (N,) the reference Sinkhorn
    scaling, psi (N, r) eigenvectors of the reference P with eigenvalues lam (r,).
    p(x, j) = K(x, j) d_j / sum_k K(x, k) d_k;  psi(x) = sum_j p(x, j) psi_j / lam.
    (For an in-sample point p(i, .) = P[i, .], so psi is reproduced exactly.)
    Returns (psi_ext (M, r), keep (r,) bool); columns with |lam| < floor are not extended (NaN)."""
    Krows, d, lam = _f64(Krows), _f64(d), _f64(lam)
    psi, vec = _col(psi)
    rs = Krows @ d
    ext = (Krows @ (d[:, None] * psi)) / rs[:, None] / lam.reshape(-1)[None, :]
    keep = lam.reshape(-1).abs() >= floor
    ext[:, ~keep] = math.nan
    return (ext[:, 0] if vec else ext), keep


def extend_function(Krows: torch.Tensor, d: torch.Tensor, lam: torch.Tensor, U: torch.Tensor,
                    phi: torch.Tensor, floor: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoint-spectral extension (PLAN §3): expand phi (N,) or (N, k) in the eigenbasis U of the
    reference P, keep modes with |lam| >= floor, extend each by `extend` (dividing by its own
    eigenvalue) and recombine with the expansion coefficients.
    Returns (phi_ext (M[, k]), discarded_energy ([k])), discarded energy = 1 - sum_kept c^2 / ||phi||^2
    (includes any energy outside span(U) if U is not a full basis). Not centred or normalised:
    the caller centres and normalises on the new set."""
    Krows, d, lam = _f64(Krows), _f64(d), _f64(lam)
    phi, vec = _col(phi)
    c = U.T @ phi
    keep = lam.abs() >= floor
    tot = (phi ** 2).sum(0)
    disc = 1.0 - (c[keep] ** 2).sum(0) / tot
    g = U[:, keep] @ (c[keep] / lam[keep][:, None])
    ext = (Krows @ (d[:, None] * g)) / (Krows @ d)[:, None]
    return (ext[:, 0], disc[0]) if vec else (ext, disc)


# ------------------------------------------------------------------------------------------ SEBA


def seba(V: torch.Tensor, Rinit: torch.Tensor | None = None, max_iter: int = 5000,
         tol: float = 1e-12) -> dict:
    """SEBA (Froyland, Rock & Sakellariou 2019, reference MATLAB algorithm).
    V (p, r) orthonormal columns. Alternates S = soft-threshold(V R^T, mu) with unit columns,
    mu = 0.99/sqrt(p), and R = polar(S^T V). Returns dict S_unit (p, r) (before parity, scaling,
    clipping), R, iters, converged. Exactly constant columns get a fixed-seed 1e-12 perturbation as
    in the reference code."""
    V = _f64(V).clone()
    p, r = V.shape
    span = V.max(0).values - V.min(0).values
    if (span < 1e-14).any():
        g = torch.Generator(device="cpu").manual_seed(0)
        for j in torch.nonzero(span < 1e-14).flatten().tolist():
            V[:, j] += ((torch.rand(p, generator=g, dtype=F64) - 0.5) * 1e-12).to(V.device)
    mu = 0.99 / math.sqrt(p)
    R = torch.eye(r, dtype=F64, device=V.device) if Rinit is None else _f64(Rinit)

    def _S(R):
        Z = V @ R.T
        S = torch.sign(Z) * (Z.abs() - mu).clamp_min(0)
        n = S.norm(dim=0, keepdim=True)
        if (n == 0).any():
            raise RuntimeError("seba: a column was thresholded to zero")
        return S / n

    converged, it = False, 0
    for it in range(1, max_iter + 1):
        S = _S(R)
        u, _, wt = torch.linalg.svd(S.T @ V)
        Rnew = u @ wt
        delta = float((Rnew - R).norm())
        R = Rnew
        if delta < tol:
            converged = True
            break
    return {"S_unit": _S(R), "R": R, "iters": it, "converged": converged}


def seba_states(lam: torch.Tensor, U: torch.Tensor, r: int, thresh: float = 0.1) -> dict:
    """SEBA on span{constant, top r-1 nonconstant modes of P} (U[:, :r], P-eigenvalue order).
    Returns dict:
      memberships (N, r): parity-fixed (positive sum), scaled to max 1, clipped at 0, columns sorted
                          by Froyland's reliability (min value before clipping, descending);
      labels (N,): argmax membership, -1 if max membership < thresh;
      diagnostics: gap (lam[r-1] - lam[r]), subspace_dist (r,) = ||(I - V V^T) s|| of each unit SEBA
      vector before clipping (and its max), neg_mass (r,) = negative mass removed by clipping as a
      fraction of total |mass| after scaling (and its max / mean), coverage = fraction of sites with a
      hard label, overlap = mean over sites of (sum - max) of memberships, n_neg_above_cut = number of
      negative P eigenvalues with |lam| > lam[r-1] (if > 0, R-ordering would choose other modes),
      seba_iters, seba_converged."""
    lam = _f64(lam)
    V = U[:, :r]
    out = seba(V)
    S = out["S_unit"]
    sub = (S - V @ (V.T @ S)).norm(dim=0)
    sgn = torch.sign(S.sum(0))
    sgn[sgn == 0] = 1
    S = S * sgn[None, :]
    S = S / S.max(0).values[None, :]
    neg_mass = (-S).clamp_min(0).sum(0) / S.abs().sum(0)
    order = torch.argsort(S.min(0).values, descending=True)
    S, sub, neg_mass = S[:, order], sub[order], neg_mass[order]
    M = S.clamp_min(0)
    mx, lab = M.max(1)
    lab = torch.where(mx < thresh, torch.full_like(lab, -1), lab)
    cut = float(lam[r - 1])
    return {
        "memberships": M, "labels": lab,
        "gap": float(lam[r - 1] - lam[r]) if r < lam.numel() else float("nan"),
        "subspace_dist": sub, "subspace_dist_max": float(sub.max()),
        "neg_mass": neg_mass, "neg_mass_max": float(neg_mass.max()), "neg_mass_mean": float(neg_mass.mean()),
        "coverage": float((lab >= 0).to(F64).mean()),
        "overlap": float((M.sum(1) - mx).mean()),
        "n_neg_above_cut": int(((lam < 0) & (lam.abs() > cut)).sum()),
        "seba_iters": out["iters"], "seba_converged": out["converged"],
    }


# ------------------------------------------------------------------------------- flows, subspaces


def _entropy_bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def flow(labels_a: torch.Tensor, labels_b: torch.Tensor) -> dict:
    """Hard-state flow between two labelings of the same sites (unassigned = -1 is a state).
    Returns states_a, states_b (sorted label values), contingency (counts), flow (row-normalised),
    H_b_given_a, H_a_given_b, H_a, H_b, MI (bits)."""
    la, lb = torch.as_tensor(labels_a).long().cpu(), torch.as_tensor(labels_b).long().cpu()
    sa, ia = torch.unique(la, return_inverse=True)
    sb, ib = torch.unique(lb, return_inverse=True)
    C = torch.zeros(len(sa), len(sb), dtype=F64)
    C.index_put_((ia, ib), torch.ones_like(ia, dtype=F64), accumulate=True)
    n = C.sum()
    J = C / n
    Ha, Hb, Hab = _entropy_bits(J.sum(1)), _entropy_bits(J.sum(0)), _entropy_bits(J.flatten())
    return {"states_a": sa, "states_b": sb, "contingency": C,
            "flow": C / C.sum(1, keepdim=True).clamp_min(1),
            "H_b_given_a": Hab - Ha, "H_a_given_b": Hab - Hb, "H_a": Ha, "H_b": Hb,
            "MI": Ha + Hb - Hab}


def _orth(A: torch.Tensor, rtol: float = 1e-10) -> torch.Tensor:
    A = _f64(A)
    if A.dim() == 1:
        A = A[:, None]
    if A.shape[1] == 0:
        return A
    u, s, _ = torch.linalg.svd(A, full_matrices=False)
    if s.numel() == 0 or float(s.max()) == 0:
        return A[:, :0]
    return u[:, : int((s > rtol * s.max()).sum())]


def principal_cosines(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Cosines of the principal angles between span(A) and span(B) (columns), descending,
    length min(rank A, rank B); empty if either span is empty."""
    Qa, Qb = _orth(A), _orth(B)
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return torch.zeros(0, dtype=F64, device=Qa.device)
    return torch.linalg.svdvals(Qa.T @ Qb).clamp(0.0, 1.0)
