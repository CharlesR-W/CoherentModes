"""PLAN.md §6a: operator unit tests on explicitly planted geometry (asserted; no training),
plus the identical-view and random-rotation controls of §3 and implementation checks of ops.py.

Run:  uv run --with torch --with numpy python -I test_ops.py [--n-per 30]
Prints `all checks passed` only if every asserted check holds; writes data/unit_tests.json.

Planted geometry (sites shared by all views; G = 64 fine groups g of n points, y(g) = g mod 8):
  ambient R^D, D = 128; noise xi ~ N(0, I_D / D) per point (||xi|| ~ 1); every view goes through
  ops.cosine_view (centre over sites, L2-normalise).
  a_sep   (endpoint a of (ii)): point = e_g + sigma_sep xi            -> 64 separated clusters
  b       (endpoint b)       : point = e_{y(g)} + sigma_sep xi        -> 8 separated clusters; the fine
                                group is *not* encoded at b (points of a y-cluster are i.i.d.)
  a_chain (endpoint a of (iii)): point = (e_g + e_{g+1 mod 64})/sqrt2 + sigma_chain xi
                                -> 64 clusters on a ring, each touching only its two ring neighbours;
                                ring neighbours g, g+1 have different y, and y-contrasts are exactly
                                the ring frequencies {8, 16, ..., 56}.
  a_rot   : a_sep times a random orthogonal D x D matrix (cosine geometry identical)
  mid     : [a_sep, b] / sqrt2 concatenated (an intermediate view for the telescoping check)
  sigma_sep = 0.2 for both kernels. sigma_chain is the one geometry parameter chosen per kernel:
  with adjacent-cluster transition mass c of P_a, the cluster-level ring has eigenvalues
  1 - 2c(1 - cos(2 pi k/64)), so the largest y-contrast Delta at t = 1 is ~8c (< 0.05 needs
  c < 6e-3) and the smallest at t = 64 is ~1 - exp(-75c) (> 0.1 needs c > 1.4e-3). sigma_chain was
  set by measuring c from P_a alone (Delta not consulted): K-st(15) 0.3 (c = 3.7e-3), K-vmf 0.6
  (c = 2.3e-3); c is re-measured and reported below as a premise check.
Kernels: K-st(k=15) and K-vmf(kappa = 1/0.07); t in {1, 4, 16, 64}; every check runs for both.

Operationalisations (fixed before running; thresholds are PLAN §6a's):
  (ii)  "y-contrasts have |f^T Delta f| < 0.05": max |eig| of Delta compressed to the 7-dim span of
        centred y-indicators (i.e. over *all* unit y-contrasts). "within-y between-fine contrasts
        < -0.8": max eig of Delta compressed to the 56-dim span of fine contrasts orthogonal to the
        y-contrasts. "No positive eigenvalue above 0.05": max eig of Delta <= 0.05. Asserted at every t.
  (iii) "appears in the positive eigenspace": all 7 principal cosines between the y-contrast span and
        span{eigenvectors of Delta with eigenvalue > 0.05} exceed 0.9 (missing dimensions count as 0);
        0.05 is (ii)'s "positive eigenvalue" cut. Asserted true at t = 64 and false at t = 1. The
        threshold-free reading (top-7 positive eigenvectors, whatever their eigenvalues) is reported.
  (iv)  path a_sep -> b -> a_rot (restored up to rotation): endpoint Delta = 0 (max |eig| < 1e-10)
        and, for the 56 within-y contrasts, |net| < 1e-10 and path length > 0.
  (v)   telescoping: path a_sep -> mid -> b; for the top/bottom 8 Delta modes, the eigenvalue from the
        dense eigensolver vs the sum of per-layer terms (spectral path), and for 16 random unit centred
        functions the dense f^T (R_b - R_a) f vs the sum of terms: |diff| < 1e-10.
        R PSD: min eigenvalue of the explicit P @ P >= -1e-12.
        SEBA (r = 64 on a_sep, r = 8 on b): coverage (fraction of sites with a hard label) > 0.95;
        purity (sum over states of the largest planted-group count / assigned sites) > 0.95; plus
        (stricter than the plan, since "recovers" means all of them) every planted group is the
        majority group of some state. Flow (a_sep r=64 -> b r=8): for every fine group g, with s_a(g)
        the a-state holding most of g and s_b(y) the b-state holding most of y, row-normalised
        flow[s_a(g), s_b(y(g))] > 0.9.
  controls: identical view built twice -> Delta = 0; rotated view -> Delta = 0 (max |eig| < 1e-10).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ops  # noqa: E402

F64 = torch.float64
TS = [1, 4, 16, 64]
G, NY, D = 64, 8, 128
SIGMA_SEP = 0.2
SIGMA_CHAIN = {"st15": 0.3, "vmf": 0.6}
KAPPA = 1 / 0.07
KERNELS = {"st15": lambda x: ops.kernel_st(x, 15)[0], "vmf": lambda x: ops.kernel_vmf(x, KAPPA)}
TOL_ZERO = 1e-10

CHECKS: list[dict] = []
MEASURED: dict = {}


def check(name, kernel, value, thr, rel, t=None, note=""):
    ok = {"<": value < thr, "<=": value <= thr, ">": value > thr, ">=": value >= thr,
          "==": value == thr, "not>": not (value > thr)}[rel]
    CHECKS.append({"name": name, "kernel": kernel, "t": t, "value": float(value), "threshold": thr,
                   "relation": rel, "passed": bool(ok), "note": note})
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<46} {kernel:<5} t={str(t):<4} value={float(value): .3e} {rel} {thr}")


def gen(seed):
    return torch.Generator().manual_seed(seed)


def build_views(n_per, sigma_chain):
    N = G * n_per
    g = torch.arange(G).repeat_interleave(n_per)
    y = g % NY
    E = torch.eye(D, dtype=F64)
    noise = lambda s: torch.randn(N, D, generator=gen(s), dtype=F64) / math.sqrt(D)
    a_sep_raw = E[g] + SIGMA_SEP * noise(1)
    b_raw = E[y] + SIGMA_SEP * noise(2)
    chain_c = (E[:G] + E[torch.arange(1, G + 1) % G]) / math.sqrt(2)
    a_chain_raw = chain_c[g] + sigma_chain * noise(3)
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=gen(4), dtype=F64))
    v = {"a_sep": ops.cosine_view(a_sep_raw), "b": ops.cosine_view(b_raw),
         "a_chain": ops.cosine_view(a_chain_raw)}
    v["a_rot"] = ops.cosine_view(a_sep_raw @ Q)
    v["mid"] = ops.cosine_view(torch.cat([v["a_sep"], v["b"]], 1) / math.sqrt(2))
    # noise control (measured only): isotropic noise with per-row norm 0.1 x signal norm (SNR 20 dB)
    nz = noise(5)
    v["a_noisy"] = ops.cosine_view(a_sep_raw + 0.1 * nz * (a_sep_raw.norm(dim=1, keepdim=True) / nz.norm(dim=1, keepdim=True)))
    return v, g, y


def operator(x, kern):
    K = KERNELS[kern](x)
    P, d = ops.sinkhorn(K, inplace=True)
    lam, U = ops.eig_full(P)
    return {"P": P, "d": d, "lam": lam, "U": U}


def compressed(o, B, t):
    """B^T R^t B for orthonormal B (via eigenpairs)."""
    C = o["U"].T @ B
    return C.T @ (C * (o["lam"].abs() ** (2 * t))[:, None])


def compressed_delta_eigs(oa, ob, B, t):
    M = compressed(ob, B, t) - compressed(oa, B, t)
    return torch.linalg.eigvalsh((M + M.T) / 2)


def purity_and_groups(labels, truth):
    assigned = labels >= 0
    lab, tru = labels[assigned], truth[assigned]
    f = ops.flow(lab, tru)
    C = f["contingency"]
    purity = float(C.max(1).values.sum() / C.sum())
    groups_hit = len(set(f["states_b"][C.argmax(1)].tolist()))
    return purity, groups_hit


def run_kernel(kern, n_per):
    print(f"\n=== kernel {kern} ===")
    t0 = time.time()
    views, g, y = build_views(n_per, SIGMA_CHAIN[kern])
    N = G * n_per
    O = {k: operator(v, kern) for k, v in views.items()}
    O["a_sep_again"] = operator(views["a_sep"].clone(), kern)
    m = MEASURED.setdefault(kern, {})

    # --- premise check for (iii): adjacent-cluster transition mass of P_a on the chain
    ind = torch.nn.functional.one_hot(g, G).to(F64)
    Mcl = ind.T @ O["a_chain"]["P"] @ ind / n_per
    idx = torch.arange(G)
    m["chain_adjacent_mass_c"] = float(torch.cat([Mcl[idx, (idx + 1) % G], Mcl[idx, (idx - 1) % G]]).mean())
    m["chain_next_adjacent_mass"] = float(Mcl[idx, (idx + 2) % G].mean())
    m["sep_offcluster_mass_a"] = float(1 - torch.diagonal(Mcl).mean())
    indy = torch.nn.functional.one_hot(y, NY).to(F64)
    m["sep_offcluster_mass_b"] = float(1 - torch.diagonal(indy.T @ O["b"]["P"] @ indy / (N / NY)).mean())
    print(f"  premise: chain c={m['chain_adjacent_mass_c']:.2e} (next {m['chain_next_adjacent_mass']:.1e}); "
          f"separated leak a={m['sep_offcluster_mass_a']:.1e} b={m['sep_offcluster_mass_b']:.1e}")
    m["P_negative_part"] = {k: ops.negative_part(o["lam"]) for k, o in O.items()}

    # --- implementation checks
    for k in ["a_sep", "b", "a_chain"]:
        o = O[k]
        check(f"impl: P row sums = 1 ({k})", kern, float((o["P"].sum(1) - 1).abs().max()), 1e-9, "<")
        Rt = ops.smooth_op(o["lam"], o["U"], 1)
        check(f"impl: smooth_op(t=1) = P@P ({k})", kern, float((Rt - o["P"] @ o["P"]).abs().max()), 1e-10, "<")
    o = O["a_chain"]
    Kin = KERNELS[kern](views["a_chain"])
    ext, keep = ops.extend(Kin, o["d"], o["U"][:, :40], o["lam"][:40])
    check("impl: extend in-sample = psi", kern, float((ext[:, keep] - o["U"][:, :40][:, keep]).abs().max()), 1e-8, "<")
    phi = ops.unit_centred(torch.randn(N, 3, generator=gen(7), dtype=F64))
    fe, disc = ops.extend_function(Kin, o["d"], o["lam"], o["U"], phi)
    kp = o["lam"].abs() >= 0.05
    proj = o["U"][:, kp] @ (o["U"][:, kp].T @ phi)
    check("impl: extend_function in-sample = projection", kern, float((fe - proj).abs().max()), 1e-8, "<")
    check("impl: extend_function discarded energy", kern,
          float((disc - (1 - (proj ** 2).sum(0))).abs().max()), 1e-10, "<")
    # kernel rows vs brute force on fresh points
    xr, xn = views["a_chain"], views["a_noisy"][:40]
    if kern == "st15":
        s_ref = ops.knn_scale(xr, 15)
        Kr = ops.kernel_rows_st(xn, xr, s_ref, 15)
        d2 = (2 - 2 * (xn @ xr.T)).clamp_min(0)
        s_new = d2.sort(1).values[:, 14].sqrt()
        Kb = torch.exp(-d2 / (s_new[:, None] * s_ref[None, :]))
        d2in = (2 - 2 * (xr @ xr.T)).clamp_min(0)
        d2in.fill_diagonal_(math.inf)
        check("impl: knn_scale = brute force", kern, float((s_ref - d2in.sort(1).values[:, 14].sqrt()).abs().max()), 1e-10, "<")
    else:
        Kr = ops.kernel_rows_vmf(xn, xr, KAPPA)
        Kb = torch.exp(KAPPA * ((xn @ xr.T) - 1))
    check("impl: kernel rows = brute force", kern, float((Kr - Kb).abs().max()), 1e-12, "<")

    # --- controls: identical view, rotation
    for t in TS:
        dm = ops.delta_modes(O["a_sep"]["lam"], O["a_sep"]["U"], O["a_sep_again"]["lam"], O["a_sep_again"]["U"], t, 4)
        check("(i) identical view: max|eig Delta|", kern, float(dm["all_vals"].abs().max()), TOL_ZERO, "<", t)
        dm = ops.delta_modes(O["a_sep"]["lam"], O["a_sep"]["U"], O["a_rot"]["lam"], O["a_rot"]["U"], t, 4)
        check("control rotation: max|eig Delta|", kern, float(dm["all_vals"].abs().max()), TOL_ZERO, "<", t)
        dm = ops.delta_modes(O["a_sep"]["lam"], O["a_sep"]["U"], O["a_noisy"]["lam"], O["a_noisy"]["U"], t, 4)
        m.setdefault("noise_control_max_abs_eig", {})[t] = float(dm["all_vals"].abs().max())

    # --- (ii) ideal merge a_sep -> b
    Y = ops.contrast_basis(y)
    Fb = ops.contrast_basis(g)
    W = ops._orth(Fb - Y @ (Y.T @ Fb))
    assert Y.shape[1] == 7 and W.shape[1] == 56
    oa, ob = O["a_sep"], O["b"]
    for t in TS:
        ey = compressed_delta_eigs(oa, ob, Y, t)
        ew = compressed_delta_eigs(oa, ob, W, t)
        dm = ops.delta_modes(oa["lam"], oa["U"], ob["lam"], ob["U"], t, 8)
        check("(ii) y-contrasts: max|f^T Delta f|", kern, float(ey.abs().max()), 0.05, "<", t)
        check("(ii) within-y contrasts: max f^T Delta f", kern, float(ew.max()), -0.8, "<", t)
        check("(ii) max eigenvalue of Delta", kern, float(dm["all_vals"].max()), 0.05, "<=", t)

    # --- (iii) resolution dependence: a_chain -> b
    oc = O["a_chain"]
    m["iii"] = {}
    for t in TS:
        dm = ops.delta_modes(oc["lam"], oc["U"], ob["lam"], ob["U"], t, 64)
        E = dm["pos_vecs"][:, dm["pos_vals"] > 0.05]
        pc = ops.principal_cosines(Y, E)
        pc7 = torch.cat([pc, torch.zeros(7 - min(7, pc.numel()), dtype=F64)])[:7]
        pc_top7 = ops.principal_cosines(Y, dm["pos_vecs"][:, :7])
        ey = compressed_delta_eigs(oc, ob, Y, t)
        m["iii"][t] = {"dim_pos_eigspace_gt_0.05": int(E.shape[1]), "pos_vals_top10": dm["pos_vals"][:10].tolist(),
                       "pc_y_vs_pos_eigspace": pc7.tolist(), "pc_y_vs_top7_threshold_free": pc_top7.tolist(),
                       "y_span_delta_rayleigh_range": [float(ey.min()), float(ey.max())],
                       "pos_eigspace_saturated_at_64": bool(dm["pos_vals"][-1] > 0.05)}
        print(f"  (iii) t={t}: dim{{eig>0.05}}={E.shape[1]} min pc={float(pc7.min()):.3f}; y-span Delta in "
              f"[{float(ey.min()):.3f},{float(ey.max()):.3f}]; threshold-free top-7 min pc={float(pc_top7.min()):.3f}")
        if t == 64:
            check("(iii) t=64: min principal cos(y-span, eig>0.05)", kern, float(pc7.min()), 0.9, ">", t)
        if t == 1:
            check("(iii) t=1: y-span NOT in eig>0.05 (min pc)", kern, float(pc7.min()), 0.9, "not>", t)

    # --- (iv) rearrange and restore: a_sep -> b -> a_rot
    path = [(O[k]["lam"], O[k]["U"]) for k in ["a_sep", "b", "a_rot"]]
    for t in TS:
        dm = ops.delta_modes(*path[0], *path[2], t, 4)
        check("(iv) endpoint Delta(a->b->a') max|eig|", kern, float(dm["all_vals"].abs().max()), TOL_ZERO, "<", t)
        pl = ops.per_layer_terms(path, W, t)
        check("(iv) within-y contrasts: max|net|", kern, float(pl["net"].abs().max()), TOL_ZERO, "<", t)
        check("(iv) within-y contrasts: min path length", kern, float(pl["path"].min()), 0.0, ">", t)
        m.setdefault("iv_mean_path_within_y", {})[t] = float(pl["path"].mean())

    # --- (v) telescoping, PSD, SEBA, flow
    path = [(O[k]["lam"], O[k]["U"]) for k in ["a_sep", "mid", "b"]]
    frand = ops.unit_centred(torch.randn(N, 16, generator=gen(8), dtype=F64))
    for t in TS:
        dm = ops.delta_modes(*path[0], *path[2], t, 8)
        modes = torch.cat([dm["pos_vecs"], dm["neg_vecs"]], 1)
        vals = torch.cat([dm["pos_vals"], dm["neg_vals"]])
        pl = ops.per_layer_terms(path, modes, t)
        check("(v) telescoping: |eig - sum terms| (modes)", kern, float((pl["terms"].sum(0) - vals).abs().max()), TOL_ZERO, "<", t)
        check("impl: Delta modes have no constant component", kern,
              float((modes.sum(0) / math.sqrt(N)).abs().max()), 1e-8, "<", t,
              note="regression: v1 delta_modes returned the constant as a 'top positive' mode")
        Dense = ops.smooth_op(*path[2], t) - ops.smooth_op(*path[0], t)
        q = (frand * (Dense @ frand)).sum(0)
        del Dense
        pl = ops.per_layer_terms(path, frand, t)
        check("(v) telescoping: |f^T Delta f - sum terms| (rand)", kern, float((pl["terms"].sum(0) - q).abs().max()), TOL_ZERO, "<", t)
    for k in ["a_sep", "b", "a_chain", "mid"]:
        R = O[k]["P"] @ O[k]["P"]
        check(f"(v) R = P^2 PSD: min eig ({k})", kern, float(torch.linalg.eigvalsh(R).min()), -1e-12, ">=")
        del R
    sa = ops.seba_states(oa["lam"], oa["U"], 64)
    sb = ops.seba_states(ob["lam"], ob["U"], 8)
    for nm, st, truth, r in [("a_sep r=64 vs fine groups", sa, g, 64), ("b r=8 vs y", sb, y, 8)]:
        pur, hit = purity_and_groups(st["labels"], truth)
        check(f"(v) SEBA coverage ({nm})", kern, st["coverage"], 0.95, ">")
        check(f"(v) SEBA purity ({nm})", kern, pur, 0.95, ">")
        check(f"(v) SEBA groups recovered ({nm})", kern, hit, r, "==", note="stricter than plan")
        m.setdefault("seba", {})[nm] = {k: (v if not torch.is_tensor(v) else None) for k, v in st.items()
                                        if k not in ("memberships", "labels")}
    fl = ops.flow(sa["labels"], sb["labels"])
    sa_states, sb_states = fl["states_a"].tolist(), fl["states_b"].tolist()
    ca = ops.flow(g, sa["labels"])["contingency"]   # fine group x a-state
    cb = ops.flow(y, sb["labels"])["contingency"]   # y x b-state
    sa_of_g = [ops.flow(g, sa["labels"])["states_b"][int(i)].item() for i in ca.argmax(1)]
    sb_of_y = [ops.flow(y, sb["labels"])["states_b"][int(i)].item() for i in cb.argmax(1)]
    mass = [float(fl["flow"][sa_states.index(sa_of_g[gg]), sb_states.index(sb_of_y[gg % NY])]) for gg in range(G)]
    check("(v) flow a->b: min_g mass to own y state", kern, min(mass), 0.9, ">")
    m["flow_H_b_given_a"], m["flow_H_a_given_b"] = fl["H_b_given_a"], fl["H_a_given_b"]
    m["seconds"] = time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"N = {G} x {args.n_per} = {G * args.n_per}; kernels {list(KERNELS)}; t in {TS}")
    for kern in KERNELS:
        run_kernel(kern, args.n_per)
    failed = [c for c in CHECKS if not c["passed"]]
    out = {"config": {"n_groups": G, "n_per_group": args.n_per, "N": G * args.n_per, "n_y": NY, "D": D,
                      "sigma_sep": SIGMA_SEP, "sigma_chain": SIGMA_CHAIN, "kernels": {"st15": "K-st k=15",
                      "vmf": f"K-vmf kappa={KAPPA:.6g}"}, "t": TS, "zero_tol": TOL_ZERO,
                      "torch": torch.__version__},
           "checks": CHECKS, "measured": MEASURED, "n_checks": len(CHECKS), "n_failed": len(failed),
           "all_passed": not failed}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "unit_tests.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for c in failed:
            print(f"FAILED: {c['name']} [{c['kernel']}, t={c['t']}] value={c['value']:.4g} {c['relation']} {c['threshold']}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
