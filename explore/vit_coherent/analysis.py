"""PLAN.md Sections 2-5 on stored activations. All operator math is from ops.py; this file only
orchestrates it, writes the SCHEMA.md layout, and adds the few helpers ops.py does not provide
(marked HELPER). Runs inside Modal containers (pipeline.py); torch float64 on GPU.

Layout written (under <root> = /out/<tag>):
  full/<view>/<k>/...   every product for every kernel (superset of data/; the assembler copies the
                        SCHEMA.md subset into data/)
  work/*.json           per-stage summaries (selection fragments, timings)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import ops

KERNELS = {"st7": ("st", 7), "st15": ("st", 15), "st30": ("st", 30),
           "vmf14": ("vmf", 1 / 0.07), "vmf5": ("vmf", 5.0), "vmf30": ("vmf", 30.0)}
KERNEL_FORMULA = {"st": "K_ij = exp(-d_ij^2/(s_i s_j)), d^2 = 2-2cos, s_i = chord distance to k-th nearest other point",
                  "vmf": "K_ij = exp(kappa (cos_ij - 1))"}
T_LIST = (1, 4, 16, 64)
R_LIST = (8, 16, 32, 64)
R_OFF = (0, 8, 24, 56)
N_MODES, N_DELTA, N_ATTR, N_Q, FLOOR = 64, 8, 16, 32, 0.05
DECLARED = [(1, 3), (3, 5), (1, 5), (1, 6), (6, 12), (1, 12), (6, 9), (9, 12), (3, 6), (3, 9)]
TRIPLES = [(1, 3, 5), (1, 6, 12), (6, 9, 12), (3, 6, 9)]
STAB_DIMS = (7, 8, 15, 16, 31, 32, 63)
CONTROL_LAYERS = (1, 6, 12)
NOISE_SNR = 10.0          # signal power / noise power of the noise control (view rows have unit norm)
F64 = torch.float64
DEV = "cuda"


# ------------------------------------------------------------------------------------------ io

def save(path: Path, arr, dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(np.asarray(arr), dtype=dtype).tofile(path)


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=True))


def layers_of(view: str) -> list[int]:
    return list(range(1, 13)) if view == "cls" else list(range(0, 13))


def lname(l: int) -> str:
    return f"L{l:02d}"


def pname(a: int, b: int) -> str:
    return f"a{a:02d}_b{b:02d}"


def all_pairs(view: str) -> list[tuple[int, int]]:
    L = layers_of(view)
    return [(a, b) for i, a in enumerate(L) for b in L[i + 1:]]


def tolist(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).tolist()


def load_plan(root: Path) -> dict:
    return json.loads((root / "plan.json").read_text())


def load_views(root: Path, view: str, centre: bool = True):
    """Probe views (centred over the probe set, L2-normalised) and held-out views (centred with the
    probe mean, normalised), as lists over layers_of(view) of (N, D) float64 GPU tensors."""
    p = np.load(root / f"acts/probe_ln_{view}.npy", mmap_mode="r")
    h = np.load(root / f"acts/heldout_ln_{view}.npy", mmap_mode="r")
    xs, xh = [], []
    for l in layers_of(view):
        a = torch.from_numpy(np.ascontiguousarray(p[:, l])).to(DEV, F64)
        b = torch.from_numpy(np.ascontiguousarray(h[:, l])).to(DEV, F64)
        if centre:
            m = a.mean(0, keepdim=True)
            a, b = a - m, b - m
        xs.append(a / a.norm(dim=1, keepdim=True))
        xh.append(b / b.norm(dim=1, keepdim=True))
    return xs, xh


# ------------------------------------------------------------------------------------ operators

def kern_full(x, k):
    fam, p = KERNELS[k]
    if fam == "st":
        return ops.kernel_st(x, p)
    return ops.kernel_vmf(x, p), None


def kern_rows(xn, xr, s, k):
    fam, p = KERNELS[k]
    return ops.kernel_rows_st(xn, xr, s, p) if fam == "st" else ops.kernel_rows_vmf(xn, xr, p)


def build_op(x, k, alpha1: bool = False) -> dict:
    """Sinkhorn operator of view x under kernel k and its full eigendecomposition."""
    K, s = kern_full(x, k)
    out = {}
    if alpha1:
        M, _ = ops.coifman_lafon(K)
        l1, V1 = torch.linalg.eigh(M)
        del M
        l1, V1 = l1.flip(0), V1.flip(1)
        out["alpha1"] = {"lambda": tolist(l1[:N_MODES + 1]), "eff_n_modes": tolist(ops.eff_num(V1[:, 1:N_MODES + 1]))}
        del V1
    t0 = time.time()
    P, d = ops.sinkhorn(K, inplace=True)
    out["sinkhorn_s"] = time.time() - t0
    lam, U = ops.eig_full(P)
    del P, K
    out.update(lam=lam, U=U, d=d, s=s, x=x)
    return out


def eff_proj(V: torch.Tensor) -> float:
    """HELPER. THEORY §4.3: w(x) = Pi[x,x]/dim, effective number of sites = 1/sum w^2."""
    w = (V ** 2).sum(1) / V.shape[1]
    return float(1.0 / (w ** 2).sum())


def n_leading_kept(lam_sub: torch.Tensor) -> int:
    keep = (lam_sub.abs() >= FLOOR).cpu().numpy()
    return int(np.argmin(keep)) if not keep.all() else len(keep)


def centre_cols(e: torch.Tensor) -> torch.Tensor:
    return e - e.mean(0, keepdim=True)


def orth_with_const(ext: torch.Tensor) -> torch.Tensor:
    """HELPER. Orthonormal basis of span{1, columns of ext} with the constant first."""
    M = ext.shape[0]
    B = torch.cat([torch.full((M, 1), 1.0 / math.sqrt(M), dtype=F64, device=ext.device), centre_cols(ext)], 1)
    Q, _ = torch.linalg.qr(B)
    return Q


def contingency(la: torch.Tensor, lb: torch.Tensor, r: int) -> np.ndarray:
    """HELPER. (r+1)x(r+1) counts; label -1 (unassigned) mapped to index r."""
    a = torch.where(la < 0, torch.full_like(la, r), la).long().cpu()
    b = torch.where(lb < 0, torch.full_like(lb, r), lb).long().cpu()
    return torch.bincount(a * (r + 1) + b, minlength=(r + 1) ** 2).reshape(r + 1, r + 1).numpy()


# ------------------------------------------------------------------------------------- Section 2

def seba_layer(lam, U, ext, ea, eb, stab, stab_mean, N):
    """SEBA on the probe set for every r, plus the same-layer split-half baseline on held-out
    (DEVIATIONS item 12). Only run for default kernels (SEBA at r = 64 is slow)."""
    lab_l, sinfo = [], {}
    memb = np.zeros((sum(R_LIST), N), np.uint8)
    hard = np.zeros((len(R_LIST), N), np.int8)
    for ri, r in enumerate(R_LIST):
        st = ops.seba_states(lam, U, r)
        M = st["memberships"]
        memb[R_OFF[ri]:R_OFF[ri] + r] = np.round(255 * M.T.clamp(0, 1).cpu().numpy())
        hard[ri] = st["labels"].cpu().numpy()
        lab_l.append(st["labels"])
        rr = r - 1
        base = {}
        labs = {}
        for nm, H in (("A", ea), ("B", eb)):
            Q = orth_with_const(ext[nm][:, :rr])
            sth = ops.seba_states(H["lam"], Q, Q.shape[1])
            labs[nm] = sth["labels"]
        fl = ops.flow(labs["A"], labs["B"])
        base = {"H_B_given_A": fl["H_b_given_a"], "H_A_given_B": fl["H_a_given_b"],
                "H_A": fl["H_a"], "H_B": fl["H_b"], "r_used": int(min(rr, ext["A"].shape[1], ext["B"].shape[1]) + 1)}
        sinfo[str(r)] = {"gap": st["gap"], "split_half_min_cos": min(stab[str(rr)]) if stab[str(rr)] else 0.0,
                         "split_half_mean_cos": stab_mean[str(rr)],
                         "dist_from_span": st["subspace_dist_max"], "dist_from_span_each": tolist(st["subspace_dist"]),
                         "neg_mass": st["neg_mass_mean"], "neg_mass_max": st["neg_mass_max"],
                         "coverage": st["coverage"], "overlap": st["overlap"],
                         "n_neg_above_cut": st["n_neg_above_cut"],
                         "seba_converged": st["seba_converged"], "seba_iters": st["seba_iters"],
                         "sizes": np.bincount(hard[ri][hard[ri] >= 0], minlength=r).tolist(),
                         "baseline": base}
    return memb, hard, lab_l, sinfo


def section2(root: Path, view: str, k: str, plan: dict, log, write_sel: bool = True, do_seba: bool = False) -> tuple[list, list, list, dict]:
    """Per layer: full operator, halves, selection criteria, validation, SEBA. Writes full/<view>/<k>/.
    Returns (E, EA, EB, labels) where labels[l][ri] are hard SEBA labels on the probe set."""
    out_dir = root / "full" / view / k
    xs, xh = load_views(root, view)
    N = xs[0].shape[0]
    half = torch.as_tensor(plan["half"], device=DEV)
    iA, iB = torch.nonzero(half == 0).flatten(), torch.nonzero(half == 1).flatten()
    yp = torch.as_tensor(plan["probe_class"], device=DEV)
    yh = torch.as_tensor(plan["heldout_class"], device=DEV)
    onehot = torch.nn.functional.one_hot(yp, 100).to(F64)
    L = layers_of(view)
    E, EA, EB, labels, per_layer, val_knn, val_eta = [], [], [], [], [], [], []
    spectrum = np.zeros((len(L), N), np.float32)
    a1spec = np.zeros((len(L), N_MODES + 1), np.float32)
    seba_info = []
    for li, l in enumerate(L):
        t0 = time.time()
        x = xs[li]
        e = build_op(x, k, alpha1=True)
        lam, U = e["lam"], e["U"]
        psi = U[:, 1:1 + N_MODES]
        spectrum[li] = lam.cpu().numpy()
        a1spec[li] = np.asarray(e["alpha1"]["lambda"], np.float32)
        save(out_dir / "modes" / f"{lname(l)}.f16", (psi.T * math.sqrt(N)).cpu().numpy(), np.float16)
        ea, eb = build_op(x[iA], k), build_op(x[iB], k)
        # (a) split-half stability on held-out
        ext, nkeep = {}, {}
        for nm, H in (("A", ea), ("B", eb)):
            Kr = kern_rows(xh[li], H["x"], H["s"], k)
            ex, _ = ops.extend(Kr, H["d"], H["U"][:, 1:1 + N_MODES], H["lam"][1:1 + N_MODES], floor=FLOOR)
            del Kr
            nkeep[nm] = n_leading_kept(H["lam"][1:1 + N_MODES])
            ext[nm] = centre_cols(ex[:, :nkeep[nm]])
        stab, stab_mean, stab_capped = {}, {}, {}
        for r in STAB_DIMS:
            rr = min(r, nkeep["A"], nkeep["B"])
            c = ops.principal_cosines(ext["A"][:, :rr], ext["B"][:, :rr]) if rr > 0 else torch.zeros(0)
            c = tolist(c)
            stab[str(r)] = c
            stab_mean[str(r)] = float(np.mean(c)) if rr == r and len(c) else (float(np.sum(c) / r) if len(c) else 0.0)
            stab_capped[str(r)] = rr < r
        # (b) localisation, (c) spectrum
        effn = ops.eff_num(psi)
        lam_np = lam.cpu().numpy()
        neg = ops.negative_part(lam)
        rec = {
            "layer": l,
            "stability": {r: stab[r] for r in ("8", "16", "32")},
            "stability_mean": {r: stab_mean[r] for r in ("8", "16", "32")},
            "stability_capped": {r: stab_capped[r] for r in ("8", "16", "32")},
            "n_extended": nkeep,
            "eff_n_modes": tolist(effn),
            "eff_n_proj": {str(r): eff_proj(U[:, 1:1 + r]) for r in (8, 16, 32)},
            "spectrum": {"lambda": tolist(lam[:N_MODES + 1]), "min": neg["min_eig"],
                         "neg_frac": neg["neg_sq_frac"], "n_negative": neg["n_negative"],
                         "gaps": tolist(lam[1:N_MODES + 1] - lam[2:N_MODES + 2]),
                         "decay_r": {"0.5": int((lam_np[1:] > 0.5).sum()), "0.1": int((lam_np[1:] > 0.1).sum()),
                                     "0.05": int((lam_np[1:] > 0.05).sum())}},
            "alpha1": e["alpha1"],
            "sinkhorn_s": e["sinkhorn_s"],
        }
        # validation (label-based, after the fact; never used by the rule)
        Kr = kern_rows(xh[li], x, e["s"], k)
        Kr.mul_(e["d"][None, :])
        w, j = Kr.topk(20, dim=1)
        del Kr
        votes = torch.zeros(len(yh), 100, dtype=F64, device=DEV).scatter_add_(1, yp[j], w)
        val_knn.append(float((votes.argmax(1) == yh).double().mean()))
        cm = onehot.T @ psi / onehot.sum(0)[:, None]
        val_eta.append(tolist((onehot.sum(0)[:, None] * cm ** 2).sum(0) / (psi ** 2).sum(0)))
        if do_seba:
            memb, hard, lab_l, sinfo = seba_layer(lam, U, ext, ea, eb, stab, stab_mean, N)
            save(out_dir / "seba" / f"{lname(l)}.u8", memb, np.uint8)
            save(out_dir / "seba" / f"{lname(l)}_hard.i8", hard, np.int8)
            seba_info.append({"layer": l, "r": sinfo})
            labels.append(lab_l)
        for H in (ea, eb):
            H.pop("alpha1", None)
        e.pop("alpha1", None)
        E.append(e)
        EA.append(ea)
        EB.append(eb)
        rec["seconds"] = round(time.time() - t0, 1)
        per_layer.append(rec)
        log(f"{view}/{k} {lname(l)} s2 {rec['seconds']}s stab16={stab_mean['16']:.3f} "
            f"effproj16={rec['eff_n_proj']['16']:.0f} knn={val_knn[-1]:.3f}")
    save(out_dir / "spectrum.f32", spectrum, np.float32)
    save(out_dir / "spectrum_alpha1.f32", a1spec, np.float32)
    save(out_dir / "evals.f32", spectrum[:, 1:1 + N_MODES], np.float32)
    if do_seba:
        jdump(out_dir / "seba.json", {"r": list(R_LIST), "offsets": list(R_OFF), "layers": seba_info})
    if write_sel:
        jdump(root / "work" / f"sel_{view}_{k}.json",
                  {"per_layer": per_layer, "validation": {"knn_acc": val_knn, "eta2": val_eta}})
    # smoothness profiles S[a, i, b, t]
    S = np.zeros((len(L), N_MODES, len(L), len(T_LIST)), np.float32)
    for ai, ea_ in enumerate(E):
        psi = ea_["U"][:, 1:1 + N_MODES]
        for bi, eb_ in enumerate(E):
            c2 = (eb_["U"].T @ psi) ** 2
            for ti, t in enumerate(T_LIST):
                S[ai, :, bi, ti] = ((eb_["lam"].abs() ** (2 * t))[:, None] * c2).sum(0).cpu().numpy()
    save(out_dir / "smooth.f32", S, np.float32)
    if do_seba:
        flows(out_dir, view, labels, seba_info)
    return E, EA, EB, labels


def flows(out_dir: Path, view: str, labels: list, seba_info: list) -> None:
    """Coarse-state flows for every pair of layers."""
    L = layers_of(view)
    P_ = all_pairs(view)
    fj = {"pairs": P_, "r": list(R_LIST), "H_b_given_a": {}, "H_a_given_b": {}, "MI": {}}
    for ri, r in enumerate(R_LIST):
        cnt = np.zeros((len(P_), r + 1, r + 1), np.uint16)
        hba, hab, mi = [], [], []
        for pi, (a, b) in enumerate(P_):
            la, lb = labels[L.index(a)][ri], labels[L.index(b)][ri]
            cnt[pi] = contingency(la, lb, r)
            fl = ops.flow(la, lb)
            hba.append(fl["H_b_given_a"]); hab.append(fl["H_a_given_b"]); mi.append(fl["MI"])
        save(out_dir / "flows" / f"counts_r{r:02d}.u16", cnt, np.uint16)
        fj["H_b_given_a"][str(r)], fj["H_a_given_b"][str(r)], fj["MI"][str(r)] = hba, hab, mi
    fj["baseline"] = {str(r): [{"layer": s["layer"], **s["r"][str(r)]["baseline"]} for s in seba_info] for r in R_LIST}
    jdump(out_dir / "flows.json", fj)


# ------------------------------------------------------------------------------------- Section 3

class HalfRows:
    """Cache of kernel rows from half-B points to half-A points per layer (for extension A -> B)."""

    def __init__(self, EA, EB, k):
        self.EA, self.EB, self.k, self.c = EA, EB, k, {}

    def __call__(self, li):
        if li not in self.c:
            self.c[li] = kern_rows(self.EB[li]["x"], self.EA[li]["x"], self.EA[li]["s"], self.k)
        return self.c[li]


def delta_pair(E, EA, EB, rows: HalfRows, ia: int, ib: int, a: int, b: int, t: float) -> dict:
    """Delta_t^{a->b} on the probe set, its per-layer terms, and its split-half reproducibility."""
    dm = ops.delta_modes(E[ia]["lam"], E[ia]["U"], E[ib]["lam"], E[ib]["U"], t, N_DELTA)
    F = torch.cat([dm["pos_vecs"], dm["neg_vecs"]], 1)
    ev = torch.cat([dm["pos_vals"], dm["neg_vals"]])
    pl = ops.per_layer_terms([(e["lam"], e["U"]) for e in E], F, t)
    terms = pl["terms"]                                   # (L-1, 16)
    net = terms[ia:ib].sum(0)
    path = terms[ia:ib].abs().sum(0)
    # reproducibility: half A modes extended to half B
    dA = ops.delta_modes(EA[ia]["lam"], EA[ia]["U"], EA[ib]["lam"], EA[ib]["U"], t, N_DELTA)
    dB = ops.delta_modes(EB[ia]["lam"], EB[ia]["U"], EB[ib]["lam"], EB[ib]["U"], t, N_DELTA)
    exts, discs = [], []
    for vecs, li in ((dA["pos_vecs"], ib), (dA["neg_vecs"], ia)):
        H = EA[li]
        ex, disc = ops.extend_function(rows(li), H["d"], H["lam"], H["U"], vecs, floor=FLOOR)
        exts.append(ex); discs.append(disc)
    ext = ops.unit_centred(torch.cat(exts, 1))
    cv = ops.delta_rayleigh(EB[ia]["lam"], EB[ia]["U"], EB[ib]["lam"], EB[ib]["U"], ext, t)
    evA = torch.cat([dA["pos_vals"], dA["neg_vals"]])
    disc = torch.cat(discs)
    cv = torch.where(disc > 0.99, torch.full_like(cv, float("nan")), cv)      # nothing extendable
    unsup = (torch.sign(cv) != torch.sign(evA)) | (cv.abs() < 0.25 * evA.abs()) | torch.isnan(cv)
    return {"vecs": F, "evals": tolist(ev), "terms": tolist(terms.T), "net": tolist(net), "path": tolist(path),
            "smooth": tolist(pl["smooth"].T), "cv": tolist(cv), "cv_evalA": tolist(evA),
            "cv_discarded": tolist(disc), "unsupported": [bool(u) for u in unsup.cpu()],
            "cv_subspace_cos": [tolist(ops.principal_cosines(ext[:, :N_DELTA], dB["pos_vecs"])),
                                tolist(ops.principal_cosines(ext[:, N_DELTA:], dB["neg_vecs"]))],
            "all_vals_extreme": [float(dm["all_vals"][0]), float(dm["all_vals"][-1])]}


def section3(root: Path, view: str, k: str, E, EA, EB, pairs, log, keep_vecs: bool = False) -> dict:
    out_dir = root / "full" / view / k
    L = layers_of(view)
    N = E[0]["U"].shape[0]
    rows = HalfRows(EA, EB, k)
    res, vecs = {}, {}
    for (a, b) in pairs:
        ia, ib = L.index(a), L.index(b)
        t0 = time.time()
        per_t = [delta_pair(E, EA, EB, rows, ia, ib, a, b, t) for t in T_LIST]
        V = torch.stack([p.pop("vecs") for p in per_t])                 # (4, N, 16)
        save(out_dir / "delta" / f"{pname(a, b)}.f16", (V.transpose(1, 2) * math.sqrt(N)).cpu().numpy(), np.float16)
        if keep_vecs or (a, b) in DECLARED:
            vecs[(a, b)] = V
        keys = per_t[0].keys()
        res[pname(a, b)] = {kk: [p[kk] for p in per_t] for kk in keys}
        log(f"{view}/{k} delta {pname(a, b)} {time.time() - t0:.1f}s ev+={per_t[0]['evals'][0]:.3f} "
            f"cv+={per_t[0]['cv'][0]:.3f}")
    triples = []
    for (a, b, c) in TRIPLES:
        if not all(p in vecs for p in ((a, b), (b, c), (a, c))):
            continue
        for ti, t in enumerate(T_LIST):
            for sg, sl in (("+", slice(0, N_DELTA)), ("-", slice(N_DELTA, 2 * N_DELTA))):
                ac, ab, bc = vecs[(a, c)][ti][:, sl], vecs[(a, b)][ti][:, sl], vecs[(b, c)][ti][:, sl]
                triples.append({"abc": [a, b, c], "t": t, "sign": sg,
                                "cos_ac_ab": tolist(ops.principal_cosines(ac, ab)),
                                "cos_ac_bc": tolist(ops.principal_cosines(ac, bc)),
                                "cos_ac_sum": tolist(ops.principal_cosines(ac, torch.cat([ab, bc], 1)))})
    out = {"pairs": [list(p) for p in pairs], "results": res, "triples": triples, "t": list(T_LIST)}
    jdump(out_dir / "delta.json", out)
    return vecs


# ----------------------------------------------------------------------- Section 3 extras (defaults)

def controls(view, k, xs, E, delta_res: dict, log) -> dict:
    """Identical-view, random-rotation and noise controls at CONTROL_LAYERS."""
    L = layers_of(view)
    g = torch.Generator(device="cpu").manual_seed(0)
    out = {"layers": list(CONTROL_LAYERS), "noise_snr": NOISE_SNR, "t": list(T_LIST), "per_layer": []}
    for l in CONTROL_LAYERS:
        li = L.index(l)
        x = xs[li]
        D = x.shape[1]
        Qr, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=F64))
        noise = torch.randn(x.shape, generator=g, dtype=F64).to(DEV) * math.sqrt(1.0 / (NOISE_SNR * D))
        variants = {"identical": x.clone(), "rotation": x @ Qr.to(DEV), "noise": ops.cosine_view(x + noise)}
        rec = {"layer": l}
        for nm, xv in variants.items():
            ev = build_op(xv, k)
            rr = {"max_abs_eig": [], "top8": [], "bottom8": []}
            for t in T_LIST:
                dm = ops.delta_modes(E[li]["lam"], E[li]["U"], ev["lam"], ev["U"], t, N_DELTA)
                rr["max_abs_eig"].append(float(dm["all_vals"].abs().max()))
                rr["top8"].append(tolist(dm["pos_vals"]))
                rr["bottom8"].append(tolist(dm["neg_vals"]))
            rec[nm] = rr
            del ev
        dm0 = ops.delta_modes(E[li]["lam"], E[li]["U"], E[li]["lam"], E[li]["U"], 1, N_DELTA)
        rec["same_eigs_max_abs_eig"] = float(dm0["all_vals"].abs().max())
        # reference: adjacent-layer Delta at this layer
        for nb in (l - 1, l + 1):
            key = pname(min(l, nb), max(l, nb))
            if key in delta_res:
                rec[f"adjacent_{key}_max_abs_eig"] = [max(abs(v[0]), abs(v[-1])) for v in
                                                      delta_res[key]["all_vals_extreme"]]
        out["per_layer"].append(rec)
        log(f"controls {k} L{l}: ident {rec['identical']['max_abs_eig'][0]:.2e} rot "
            f"{rec['rotation']['max_abs_eig'][0]:.2e} noise {rec['noise']['max_abs_eig'][0]:.3f}")
    return out


def q_ops(view, k, E) -> dict:
    """Product operator Q^{a->b} = P_b P_a (top 32 singular values) and the predecessor's mean operator."""
    L = layers_of(view)
    out = {}

    def P_of(e):
        return (e["U"] * e["lam"][None, :]) @ e["U"].T

    for (a, b) in DECLARED:
        ia, ib = L.index(a), L.index(b)
        Pa, Pb = P_of(E[ia]), P_of(E[ib])
        Q = Pb @ Pa
        del Pb
        G = Q.T @ Q
        w, V = torch.linalg.eigh(G)
        del G
        w, V = w.flip(0)[:N_Q + 1], V.flip(1)[:, :N_Q + 1]
        sv = w.clamp_min(0).sqrt()
        Ul = (Q @ V) / sv[None, :]
        del Q
        rec = {"sv": tolist(sv),
               "cos_q_right_vs_Pa": tolist(ops.principal_cosines(V[:, 1:9], E[ia]["U"][:, 1:9])),
               "cos_q_left_vs_Pb": tolist(ops.principal_cosines(Ul[:, 1:9], E[ib]["U"][:, 1:9]))}
        Msum = Pa
        for li in range(ia + 1, ib + 1):
            Msum += P_of(E[li])
        Msum /= (ib - ia + 1)
        mw = torch.linalg.eigvalsh(Msum).flip(0)
        rec["mean_evals"] = tolist(mw[1:N_Q + 1])
        del Msum
        out[pname(a, b)] = rec
    return out


def uncentred(root, view, k, E) -> list:
    """Sensitivity: uncentred view. Top-16 principal cosines with the centred operator's modes."""
    xs, _ = load_views(root, view, centre=False)
    out = []
    for li, x in enumerate(xs):
        e = build_op(x, k)
        out.append({"layer": layers_of(view)[li],
                    "cos16_vs_centred": tolist(ops.principal_cosines(e["U"][:, 1:17], E[li]["U"][:, 1:17])),
                    "eff_n_proj16": eff_proj(e["U"][:, 1:17]),
                    "lambda": tolist(e["lam"][:17])})
        del e
    return out


# ------------------------------------------------------------------------------------- Section 4

class Components:
    """CLS-position components of PLAN §4 on the probe set (float64 on GPU)."""

    def __init__(self, root: Path):
        from acts import baseline_bias, head_outputs, neuron_dirs
        W = dict(np.load(root / "acts/weights.npz"))
        self.W = W
        h = np.load(root / "acts/probe_h_cls.npy")                       # [N, 13, D]
        ln = np.load(root / "acts/probe_ln_cls.npy")
        self.m = torch.from_numpy(ln.mean(0)).to(DEV)                     # [13, D] probe mean of LN(h)[CLS]
        self.h = torch.from_numpy(h).to(DEV)
        mu = self.h.mean(-1, keepdim=True)
        self.sigma = ((self.h - mu) ** 2).mean(-1).add(float(W["norm_eps"])).sqrt()      # [N, 13]
        lnt = torch.from_numpy(ln).to(DEV)
        self.rho = (lnt - self.m[None]).norm(dim=-1)                                     # [N, 13]
        self.V = (lnt - self.m[None]) / self.rho[..., None]                              # views [N, 13, D]
        del lnt
        heads = np.load(root / "acts/probe_heads.npy")
        self.A = torch.from_numpy(head_outputs(heads, W)).to(DEV)                        # [N, 12, 6, D]
        del heads
        self.G = torch.from_numpy(np.load(root / "acts/probe_neurons.npy")).to(DEV, F64)  # [N, 12, 1536]
        self.U = torch.from_numpy(neuron_dirs(W)).to(DEV)                                # [12, 1536, D]
        self.bias = torch.from_numpy(baseline_bias(W)).to(DEV)                           # [13, D]
        self.gamma = torch.from_numpy(W["norm_w"]).to(DEV)
        self.beta = torch.from_numpy(W["norm_b"]).to(DEV)
        self.N = self.h.shape[0]
        Gc = self.G - self.G.mean(0, keepdim=True)
        self.Gunit = Gc / Gc.norm(dim=0, keepdim=True).clamp_min(1e-300)
        # ranks for AUROC (ties broken by order; post-GELU activations are continuous)
        self.Grank = torch.empty_like(self.G)
        for l in range(12):
            self.Grank[:, l] = self.G[:, l].argsort(0).argsort(0).to(F64) + 1

    def attribute(self, l: int, F: torch.Tensor, Fh: torch.Tensor, Vh: torch.Tensor) -> dict:
        """Direct attribution of unit-centred targets F (N, T) at CLS depth l (1..12).
        Fh: target values on held-out (M, T) (extension), Vh: held-out views at l (M, D)."""
        V = self.V[:, l]
        N = self.N
        Us, s, Wt = torch.linalg.svd(V, full_matrices=False)
        z = Us.T @ F                                                        # (D, T)
        tot = (F ** 2).sum(0)
        lams = (s.max() ** 2) * torch.logspace(-8, 1, 46, dtype=F64, device=DEV)
        best = None
        for lm in lams:
            hh = s ** 2 / (s ** 2 + lm)
            rss = tot - ((2 * hh - hh ** 2)[:, None] * z ** 2).sum(0)
            gcv = N * rss / (N - hh.sum()) ** 2
            if best is None:
                best = (gcv, torch.full_like(gcv, float(lm)))
            else:
                better = gcv < best[0]
                best = (torch.where(better, gcv, best[0]), torch.where(better, lm, best[1]))
        lam_t = best[1]                                                     # (T,)
        w = Wt.T @ ((s[:, None] / (s[:, None] ** 2 + lam_t[None, :])) * z)  # (D, T)
        pred = V @ w
        r2 = 1 - ((F - pred) ** 2).sum(0) / tot
        ph = Vh @ w
        r2h = 1 - ((Fh - ph) ** 2).sum(0) / ((Fh - Fh.mean(0)) ** 2).sum(0)
        total = (pred * F).sum(0) / N                                       # cov(V w, f), F centred
        a = 1.0 / (self.sigma[:, l] * self.rho[:, l])                       # (N,)
        wg = self.gamma[:, None] * w
        wt = wg - wg.mean(0, keepdim=True)                                  # (D, T): J^T w = a(x) wt
        aF = a[:, None] * F                                                 # (N, T)
        # neurons of blocks 1..l
        neur = torch.full((12, 1536, F.shape[1]), float("nan"), dtype=F64, device=DEV)
        for lp in range(l):
            q = self.U[lp] @ wt                                             # (1536, T)
            neur[lp] = (self.G[:, lp].T @ aF) / N * q
        heads = torch.full((12, 6, F.shape[1]), float("nan"), dtype=F64, device=DEV)
        for lp in range(l):
            Aw = torch.einsum("nhd,dt->nht", self.A[:, lp], wt)
            heads[lp] = (Aw * aF[:, None, :]).sum(0) / N
        base_f = a[:, None] * (self.bias[l] @ wt)[None, :] + ((self.beta - self.m[l]) @ w)[None, :] / self.rho[:, l][:, None]
        base = (base_f * F).sum(0) / N
        ssum = base + torch.nan_to_num(heads).sum((0, 1)) + torch.nan_to_num(neur).sum((0, 1))
        wres = a.mean() * wt
        u = self.U / self.U.norm(dim=-1, keepdim=True)
        wc = torch.einsum("lnd,dt->lnt", u, wres / wres.norm(dim=0, keepdim=True))
        wc[l:] = float("nan")
        return {"r2": r2, "r2h": r2h, "lam": lam_t, "total": total, "base": base, "heads": heads, "neur": neur,
                "sum_err": float(((ssum - total).abs() / total.abs().clamp_min(1e-300)).max()),
                "wres": wres, "writecos": wc}


def section4(root: Path, k: str, E: list, delta_vecs: dict, labels: list, log) -> dict:
    """Neuron loadings / AUROC, attributions of layer modes and of Delta modes (CLS, default kernel)."""
    C = Components(root)
    out_dir = root / "full" / "cls" / k
    L = layers_of("cls")
    _, xh = load_views(root, "cls")
    N = C.N
    info = {"layers": [], "sum_rel_err_max": 0.0}
    # Delta-mode targets grouped by readout layer
    targets = {l: [] for l in L}             # l -> list of (pair, ti, row)
    for (a, b), V in delta_vecs.items():
        for ti in range(len(T_LIST)):
            for row in range(2 * N_DELTA):
                targets[b if row < N_DELTA else a].append(((a, b), ti, row))
    dres = {}
    for li, l in enumerate(L):
        t0 = time.time()
        e = E[li]
        psi = e["U"][:, 1:1 + N_MODES]
        # loadings and AUROC (block l neurons)
        save(out_dir / "load" / f"{lname(l)}.f16", (psi.T @ C.Gunit[:, l - 1]).cpu().numpy(), np.float16)
        au = np.full((sum(R_LIST), 1536), np.nan, np.float32)
        for ri, r in enumerate(R_LIST):
            lab = labels[li][ri]
            Mm = torch.nn.functional.one_hot((lab + 1).long(), r + 1)[:, 1:].to(F64)      # (N, r)
            n1 = Mm.sum(0)
            n0 = N - n1
            srank = Mm.T @ C.Grank[:, l - 1]                                              # (r, 1536)
            auc = (srank - (n1 * (n1 + 1) / 2)[:, None]) / (n1 * n0)[:, None]
            auc[(n1 == 0) | (n0 == 0)] = float("nan")
            au[R_OFF[ri]:R_OFF[ri] + r] = auc.cpu().numpy()
        save(out_dir / "auroc" / f"{lname(l)}.f16", au, np.float16)
        # held-out kernel rows to the full probe set at this layer (for extensions)
        Kr = kern_rows(xh[li], e["x"], e["s"], k)
        # (i) layer modes
        F = psi[:, :N_ATTR]
        Fh, _ = ops.extend(Kr, e["d"], F, e["lam"][1:1 + N_ATTR], floor=FLOOR)
        r = C.attribute(l, F, Fh, xh[li])
        info["sum_rel_err_max"] = max(info["sum_rel_err_max"], r["sum_err"])
        tot = r["total"]
        flat = torch.cat([r["base"][None], r["heads"].reshape(72, -1), r["neur"].reshape(12 * 1536, -1)], 0)
        save(out_dir / "attr" / f"{lname(l)}.f32", (flat / tot[None]).T.cpu().numpy(), np.float32)
        save(out_dir / "writecos" / f"{lname(l)}.f16", r["writecos"].permute(2, 0, 1).cpu().numpy(), np.float16)
        info["layers"].append({"layer": l, "r2_probe": tolist(r["r2"]), "r2_heldout": tolist(r["r2h"]),
                               "ridge_lambda": tolist(r["lam"]), "total_cov": tolist(tot),
                               "w_res": tolist(r["wres"].T)})
        # (ii) Delta modes read out at this layer, in chunks
        tg = targets[l]
        for c0 in range(0, len(tg), 256):
            chunk = tg[c0:c0 + 256]
            F = torch.stack([delta_vecs[p][ti][:, row] for p, ti, row in chunk], 1)
            F = ops.unit_centred(F)
            Fh, _ = ops.extend_function(Kr, e["d"], e["lam"], e["U"], F, floor=FLOOR)
            r = C.attribute(l, F, Fh, xh[li])
            info["sum_rel_err_max"] = max(info["sum_rel_err_max"], r["sum_err"])
            tot = r["total"]
            nf = (r["neur"] / tot).reshape(12 * 1536, -1)
            top = torch.nan_to_num(nf, nan=0.0).abs().topk(32, dim=0).indices                # (32, T)
            for j, (p, ti, row) in enumerate(chunk):
                vec = np.full(480, np.nan, np.float32)
                vec[0] = float(r["base"][j] / tot[j])
                vec[1:73] = (r["heads"][:, :, j] / tot[j]).reshape(-1).cpu().numpy()
                vec[73:85] = (r["neur"][:, :, j] / tot[j]).sum(1).cpu().numpy()
                vec[73 + l:85] = np.nan
                vec[85:469] = r["wres"][:, j].cpu().numpy()
                vec[469:474] = [float(r["r2"][j]), float(r["r2h"][j]), float(tot[j]), l, float(r["lam"][j])]
                tj = top[:, j]
                tn = np.stack([tj.cpu().numpy().astype(np.float32), nf[tj, j].cpu().numpy().astype(np.float32)], 1)
                dres.setdefault(p, [np.full((len(T_LIST), 2 * N_DELTA, 480), np.nan, np.float32),
                                    np.full((len(T_LIST), 2 * N_DELTA, 32, 2), np.nan, np.float32)])
                dres[p][0][ti, row] = vec
                dres[p][1][ti, row] = tn
        del Kr
        log(f"s4 {k} {lname(l)} {time.time() - t0:.1f}s r2(mode1)={info['layers'][-1]['r2_probe'][0]:.3f} "
            f"sum_err={info['sum_rel_err_max']:.1e} n_delta_targets={len(tg)}")
    for p, (v, tn) in dres.items():
        save(out_dir / "delta_attr" / f"{pname(*p)}.f32", v, np.float32)
        save(out_dir / "delta_attr" / f"{pname(*p)}_top.f32", tn, np.float32)
    jdump(out_dir / "attr.json", info)
    return info


# ------------------------------------------------------------------------------------- Section 5

def decodability(root: Path, plan: dict) -> dict:
    out = {"protocol": "fit on the 10,000 probe images, test on the 10,000 held-out; views centred with the probe "
                       "mean and L2-normalised; ridge = closed-form one-vs-rest least squares on centred one-hot "
                       "targets with intercept, one lambda per fit chosen by GCV summed over outputs; kNN = DINO "
                       "protocol, k = 20, weights exp(cos/0.07), class vote"}
    for view in ("cls", "mp"):
        xs, xh = load_views(root, view)
        res = {"layers": layers_of(view), "ridge": {"fine": [], "super": []}, "knn": {"fine": [], "super": []},
               "ridge_lambda": {"fine": [], "super": []}}
        for lab in ("fine", "super"):
            yp = torch.as_tensor(plan["probe_class" if lab == "fine" else "probe_super"], device=DEV)
            yh = torch.as_tensor(plan["heldout_class" if lab == "fine" else "heldout_super"], device=DEV)
            C = int(max(yp.max(), yh.max())) + 1
            Y = torch.nn.functional.one_hot(yp, C).to(F64)
            Ym = Y.mean(0, keepdim=True)
            Yc = Y - Ym
            for x, xo in zip(xs, xh):
                xm = x.mean(0, keepdim=True)
                X = x - xm
                Us, s, Wt = torch.linalg.svd(X, full_matrices=False)
                z = Us.T @ Yc
                tot = (Yc ** 2).sum()
                best = None
                for lm in (s.max() ** 2) * torch.logspace(-8, 1, 46, dtype=F64, device=DEV):
                    hh = s ** 2 / (s ** 2 + lm)
                    rss = tot - ((2 * hh - hh ** 2)[:, None] * z ** 2).sum()
                    g = X.shape[0] * rss / (X.shape[0] - hh.sum()) ** 2
                    if best is None or g < best[0]:
                        best = (g, lm)
                B = Wt.T @ ((s / (s ** 2 + best[1]))[:, None] * z)
                pred = ((xo - xm) @ B + Ym).argmax(1)
                res["ridge"][lab].append(float((pred == yh).double().mean()))
                res["ridge_lambda"][lab].append(float(best[1]))
                sim = xo @ x.T
                w, j = sim.topk(20, dim=1)
                votes = torch.zeros(len(yh), C, dtype=F64, device=DEV).scatter_add_(1, yp[j], (w / 0.07).exp())
                res["knn"][lab].append(float((votes.argmax(1) == yh).double().mean()))
                del sim
        out[view] = res
    return out


def umap_view(root: Path, view: str) -> np.ndarray:
    import umap
    p = np.load(root / f"acts/probe_ln_{view}.npy", mmap_mode="r")
    xs = []
    for l in layers_of(view):
        a = np.asarray(p[:, l], np.float64)
        a = a - a.mean(0, keepdims=True)
        xs.append((a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32))
    n = xs[0].shape[0]
    rel = [{i: i for i in range(n)} for _ in range(len(xs) - 1)]
    al = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0,
                          alignment_regularisation=0.02)
    al.fit(xs, relations=rel)
    return np.stack(al.embeddings_).astype(np.float32)


# ------------------------------------------------------------------------------------- selection

def select(fr: dict) -> dict:
    """Label-free default-kernel rule (DEVIATIONS.md item 7). fr[k] = sel_cls_<k>.json content."""
    rows = {}
    for k, s in fr.items():
        ok, st_m = [], []
        for rec in s["per_layer"]:
            stable = rec["stability_mean"]["16"] >= 0.8 and not rec["stability_capped"]["16"]
            nonloc = rec["eff_n_proj"]["16"] >= 160 and float(np.median(rec["eff_n_modes"][:16])) >= 10
            ok.append(bool(stable and nonloc))
            st_m.append(rec["stability_mean"]["16"])
        rows[k] = {"pass": ok, "score": int(sum(ok)), "mean_stab16": float(np.mean(st_m)),
                   "n_stable": int(sum(r["stability_mean"]["16"] >= 0.8 and not r["stability_capped"]["16"]
                                       for r in s["per_layer"])),
                   "n_nonlocal": int(sum(r["eff_n_proj"]["16"] >= 160 and np.median(r["eff_n_modes"][:16]) >= 10
                                         for r in s["per_layer"]))}
    order = sorted(rows, key=lambda k: (-rows[k]["score"], -rows[k]["mean_stab16"]))
    best, second = order[0], order[1]
    clear = rows[second]["score"] < rows[best]["score"] - 1
    defaults = [best] if clear else [best, second]
    n_layers = len(next(iter(fr.values()))["per_layer"])
    words = (f"Rule (pre-registered, DEVIATIONS.md item 7): count CLS layers where the top-16 split-half "
             f"stability (mean principal cosine) >= 0.8 and the top-16 block is non-localised (projector "
             f"effective number >= 160, median single-mode effective number >= 10). Scores: "
             + ", ".join(f"{k} {rows[k]['score']}/{n_layers} (stable {rows[k]['n_stable']}, non-localised "
                         f"{rows[k]['n_nonlocal']}, mean stability {rows[k]['mean_stab16']:.3f})" for k in order)
             + f". Best: {best}. ")
    if clear:
        words += f"Clear-cut: runner-up {second} is more than one layer behind; one default kept."
    else:
        words += (f"Not clear-cut: runner-up {second} is within one layer of {best}; both are kept as defaults "
                  f"({best} primary).")
    if rows[best]["score"] < n_layers / 2:
        words += (f" No candidate is stable and non-localised at most layers (best {rows[best]['score']}/"
                  f"{n_layers}); that is the finding, and the rule's winner is used downstream only so that "
                  f"downstream objects exist.")
    return {"default_kernels": defaults, "scores": rows, "order": order, "clear_cut": clear, "reasoning": words}
