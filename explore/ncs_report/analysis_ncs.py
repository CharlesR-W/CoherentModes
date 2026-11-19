"""Facts and chart data for the joint ViT / Mamba report: writes site/assets/ncs.json = {"facts": {...}, charts...}.

    uv run -q --with numpy --with scipy python -I explore/ncs_report/analysis_ncs.py

Reads the stored artifacts of the four studies in place (read-only):
  ../vit_coherent/data/vits_b2k        DINOv2 ViT-S/14, 2,000 ImageNet-100 images, CLS after blocks 1..12
  ../mamba_contexts/data/m130c         Mamba-130m, 8,000 token contexts
  ../mamba_coherent/data/m130, diagnosis/diag_v1.json    Mamba-130m over 1,000 snippets (v1)
  ../lm_coherent/data/p160t            Pythia-160m over the same 1,000 snippets, 5 checkpoints
Every number the prose quotes is a fact here (formatted string), named v.* (ViT), m.* (Mamba token contexts),
o.* (older whole-snippet studies) or s.* (the sanity table). Labels (classes, token classes, sources) only
describe and validate sets; nothing here builds a kernel, a basis or a set.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent                 # explore/ncs_report
EXP = HERE.parent                                      # explore/
VIT = EXP / "vit_coherent" / "data" / "vits_b2k"
MC = EXP / "mamba_contexts" / "data" / "m130c"
MV1 = EXP / "mamba_coherent"
PY = EXP / "lm_coherent" / "data" / "p160t"
OUT = HERE / "site" / "assets" / "ncs.json"
SEED = 20261113
N_PERM = 200

F: dict[str, str] = {}      # facts
C: dict[str, object] = {}   # chart objects


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def i_(x): return f"{int(x):,}"
def pc(x): return f"{100 * x:.1f}%" if 0 < 100 * x < 0.95 else f"{100 * x:.0f}%"   # never a non-zero share as 0%
def pc1(x): return f"{100 * x:.1f}%"


def tok_disp(s: str) -> str:
    """A token as the prose shows it: quoted, with whitespace made visible."""
    s = s.replace("\n", "↵").replace("\t", "⇥")
    if s.strip() == "" and s:
        return f"{len(s)} spaces" if set(s) == {" "} else s
    return f"“{s.replace(' ', chr(160))}”"      # no-break spaces: a token never wraps inside its quotes


# ------------------------------------------------------------------------------------------- shared helpers
def jacc_matrix(a, b, ra, rb):
    """|A_i ∩ B_j| / |A_i ∪ B_j| with each set's full membership (labels < 0 = unassigned). Same convention as the
    explorer's jaccardMatrix (site/js/explorer.js): an item unassigned in one partition still counts in the other's
    set sizes, so it enlarges the union."""
    both = (a >= 0) & (b >= 0)
    inter = np.zeros((ra, rb))
    np.add.at(inter, (a[both], b[both]), 1)
    sa, sb = np.bincount(a[a >= 0], minlength=ra), np.bincount(b[b >= 0], minlength=rb)
    union = sa[:, None] + sb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0)


def matched_jaccard(a, b, ra, rb):
    """Mean over the ra sets of a of the Jaccard with its partner in the optimal one-to-one matching with the sets of
    b (a set left unmatched scores 0), as in the explorer's matchedJaccard."""
    J = jacc_matrix(a, b, ra, rb)
    r, c = linear_sum_assignment(-J)
    return float(J[r, c].sum() / ra)


def span_agreement(U, V):
    """a_r: mean cos^2 of the principal angles between span(U) and span(V), columns orthonormalised."""
    qu, _ = np.linalg.qr(U)
    qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return float(np.mean(s ** 2))


def pn(psi):
    """participation number of each column: (sum psi^2)^2 / sum psi^4."""
    return (psi ** 2).sum(0) ** 2 / (psi ** 4).sum(0)


# =========================================================================================== ViT
def vit():
    t0 = time.time()
    meta = json.loads((VIT / "meta.json").read_text())
    sites, R = meta["sites"], meta["seba_r"]
    S = len(sites)
    lp = {tuple(p): i for i, p in enumerate(meta["layer_pairs"])}
    z = np.load(VIT / "layer_0.npz")
    LAB = z["seba_label"].astype(np.int64)                  # (M, P, R, N)
    VALS = z["vals"].astype(np.float64)                     # (M, P, 9)
    vecs, scale = z["vecs"], z["scale"]
    with np.load(VIT / "probe.npz") as p:
        grp, fine = p["labels"].astype(np.int64), p["fine_labels"].astype(np.int64)
    fine_names, groups = meta["fine_classes"], meta["classes"]
    N, G, NF = len(grp), len(groups), len(fine_names)
    ri = {r: i for i, r in enumerate(R)}
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(N) for _ in range(N_PERM)]
    pair = lambda a, b: lp[(a, b)]
    psi_of = lambda mi, pi: vecs[mi, pi].T.astype(np.float64) * scale[mi, pi] / 32767.0   # (N, 8)
    F["v.n"] = i_(N)
    F["v.n_fine"] = str(NF)
    F["v.per_class"] = str(N // NF)
    F["v.n_sites"] = str(S)
    F["v.n_groups"] = str(G)
    F["v.k"] = "15"

    def per_class(lab):
        """(NF, r+1) counts of each fine class's images per set (last column: unassigned)."""
        r = lab.max() + 1
        M = np.zeros((NF, r + 1), int)
        np.add.at(M, (fine, np.where(lab < 0, r, lab)), 1)
        return M

    def n_ge(lab, r, thr):
        M = np.zeros((NF, r), int)
        keep = lab >= 0
        np.add.at(M, (fine[keep], lab[keep]), 1)
        return int((M.max(1) >= thr).sum())

    def set_desc(lab, r, top=3):
        out = []
        for j in range(r):
            mem = np.flatnonzero(lab == j)
            fc = np.bincount(fine[mem], minlength=NF)
            gc = np.bincount(grp[mem], minlength=G)
            o = np.argsort(-fc, kind="stable")[:top]
            out.append(dict(size=len(mem), top=[(fine_names[k], int(fc[k])) for k in o if fc[k]],
                            groups=gc.tolist(), whole=[fine_names[k] for k in np.flatnonzero(fc == N // NF)]))
        return out

    # ---- n_ge16 by site, null, largest-set share (r = 8, kNN k = 15)
    r8 = ri[8]
    ge16, whole, big, ge16_null_hi, ge16_null_mean = [], [], [], [], []
    for s in range(S):
        lab = LAB[0, pair(s, s), r8]
        ge16.append(n_ge(lab, 8, 16)); whole.append(n_ge(lab, 8, 20))
        sz = np.bincount(lab[lab >= 0], minlength=8)
        big.append(sz.max() / N)
        nl = [n_ge(lab[p], 8, 16) for p in perms[:50]]
        ge16_null_mean.append(float(np.mean(nl))); ge16_null_hi.append(float(np.percentile(nl, 97.5)))
    for s, name in enumerate(sites):
        F[f"v.{name}.r8.n_ge16"] = str(ge16[s])
        F[f"v.{name}.r8.n_whole"] = str(whole[s])
        F[f"v.{name}.r8.largest_share"] = pc(big[s])
    F["v.ge16.null_max"] = f"{max(ge16_null_hi):.0f}"
    C["v_ge16"] = dict(x=list(range(S)), xticklabels=sites, xlabel="site (single-snapshot block)",
                       ylabel="fine classes with ≥ 16 of 20 images in one set", yrange=[0, 100],
                       series=[dict(name="classes with ≥ 16 of 20 images in one set (of 100)", y=ge16,
                                    band=[[0] * S, ge16_null_hi]),
                               dict(name="classes with all 20 images in one set", y=whole, dash=True)])
    C["v_bigset"] = dict(x=list(range(S)), xticklabels=sites, xlabel="site (single-snapshot block)",
                         ylabel="largest set's share of the 2,000 images", yrange=[0, 1],
                         series=[dict(name="largest of the 8 sets, share of all images", y=big)])

    # ---- named blocks
    named = {"L1": (0, 0), "L6": (5, 5), "L9": (8, 8), "L12": (11, 11), "L9_L12": (8, 11), "L1_L12": (0, 11)}
    desc = {}
    for key, (a, b) in named.items():
        pi = pair(a, b)
        v = VALS[0, pi]
        psi = psi_of(0, pi)
        P = pn(psi)
        F[f"v.{key}.lam1"] = f3(v[1]); F[f"v.{key}.lam2"] = f3(v[2]); F[f"v.{key}.lam3"] = f3(v[3])
        F[f"v.{key}.lam8"] = f3(v[8])
        F[f"v.{key}.pn_min"] = f"{P.min():.0f}"
        F[f"v.{key}.pn_max"] = f"{P.max():.0f}"
        F[f"v.{key}.pn_min_frac"] = pc1(P.min() / N)
        for r in (2, 4, 8):
            lab = LAB[0, pi, ri[r]]
            sz = np.bincount(lab[lab >= 0], minlength=r)
            F[f"v.{key}.r{r}.sizes"] = ", ".join(i_(x) for x in sz)
            F[f"v.{key}.r{r}.n_unassigned"] = i_((lab < 0).sum())
            F[f"v.{key}.r{r}.n_ge16"] = str(n_ge(lab, r, 16))
            F[f"v.{key}.r{r}.n_whole"] = str(n_ge(lab, r, 20))
        desc[key] = set_desc(LAB[0, pi, r8], 8)
    for key in ("L9_L12", "L12", "L1", "L6", "L9"):
        for j, d in enumerate(desc[key]):
            F[f"v.{key}.r8.set{j + 1}.size"] = i_(d["size"])
            F[f"v.{key}.r8.set{j + 1}.top"] = "; ".join(f"{n} {c}" for n, c in d["top"])
    # sets of L12 alone that are exactly one fine class (all 20 of it, nothing else)
    one = [d["top"][0][0] for d in desc["L12"] if d["size"] == N // NF and d["top"][0][1] == N // NF]
    F["v.L12.r8.n_single_class_sets"] = str(len(one))
    F["v.L12.r8.single_class_sets"] = ", ".join(one)
    # class-group make-up heatmaps (fraction of each group's images in each set), r = 8 and r = 2 of L9..L12
    for key, r in (("L9_L12", 8), ("L9_L12", 2), ("L1", 8), ("L6", 8), ("L9", 8), ("L12", 8)):
        a, b = named[key]
        lab = LAB[0, pair(a, b), ri[r]]
        M = np.zeros((r + 1, G))
        np.add.at(M, (np.where(lab < 0, r, lab), grp), 1)
        gsz = np.bincount(grp, minlength=G)
        C[f"v_mix_{key}_r{r}"] = dict(rows=[f"set {j + 1}" for j in range(r)] + ["unassigned"], cols=groups,
                                     z=(M / gsz[None, :]).round(4).tolist(), zrange=[0, 1], fmt="pct",
                                     rowlabel=f"r = {r} set", collabel="class group (validation only)",
                                     zlabel="share of the class group's images in the set")
    # reading the r = 8 make-up heatmap down its columns (each column sums to 1): groups that sit mostly in one set
    Z = np.array(C["v_mix_L9_L12_r8"]["z"])[:-1]           # sets only, without the unassigned row
    top_share = Z.max(0)
    one = [g for g in np.argsort(-top_share, kind="stable") if top_share[g] >= 0.9]
    spread = [g for g in range(G) if top_share[g] < 0.6]
    F["v.L9_L12.r8.mix.one_set_groups"] = ", ".join(groups[g] for g in one)
    F["v.L9_L12.r8.mix.one_set_min"] = pc(min(top_share[g] for g in one))
    F["v.L9_L12.r8.mix.spread_groups"] = ", ".join(groups[g] for g in spread)
    F["v.L9_L12.r8.mix.spread_max"] = pc(max(top_share[g] for g in spread))
    lab2 = LAB[0, pair(8, 11), ri[2]]
    rep = groups.index("reptiles")
    F["v.L9_L12.r2.reptiles_unassigned"] = i_(((grp == rep) & (lab2 < 0)).sum())
    F["v.L9_L12.r2.reptiles_n"] = i_((grp == rep).sum())
    for j in range(2):
        gc = np.bincount(grp[lab2 == j], minlength=G)
        F[f"v.L9_L12.r2.set{j + 1}.top_groups"] = ", ".join(f"{groups[g]} {gc[g]}" for g in np.argsort(-gc)[:3])
    gsz_ = np.bincount(grp, minlength=G)
    F["v.dog_classes"] = str(int(gsz_[groups.index("dogs")]) // (N // NF))
    for gname, key in (("invertebrates", "invert"), ("dogs", "dogs"), ("birds", "birds")):
        g_ = groups.index(gname)
        for j in range(2):
            F[f"v.L9_L12.r2.set{j + 1}.{key}_share"] = pc(((grp == g_) & (lab2 == j)).sum() / gsz_[g_])
    animal = np.isin(grp, [groups.index(x) for x in ("dogs", "other mammals", "birds")])
    F["v.L9_L12.r2.set1.share_mammal_bird"] = pc((animal & (lab2 == 0)).sum() / max(1, (lab2 == 0).sum()))
    F["v.L9_L12.r2.set2.share_mammal_bird"] = pc((animal & (lab2 == 1)).sum() / max(1, (lab2 == 1).sum()))

    # ---- eigenfunctions of L9..L12 and L12: fine classes at both ends (12 images each)
    ends = {}
    for key in ("L9_L12", "L1", "L6"):
        a, b = named[key]
        psi = psi_of(0, pair(a, b))
        for k in range(3):
            o = np.argsort(psi[:, k], kind="stable")
            for end, idx in (("neg", o[:12]), ("pos", o[::-1][:12])):
                fc = np.bincount(fine[idx], minlength=NF)
                gc = np.bincount(grp[idx], minlength=G)
                top = [f"{fine_names[c]} {fc[c]}" for c in np.argsort(-fc, kind="stable")[:3] if fc[c]]
                F[f"v.{key}.psi{k + 1}.{end}.classes"] = ", ".join(top)
                F[f"v.{key}.psi{k + 1}.{end}.groups"] = ", ".join(f"{groups[g]} {gc[g]}" for g in np.argsort(-gc, kind="stable")[:2] if gc[g])
                ends[f"{key}.{k + 1}.{end}"] = idx.tolist()
        # correlation ratio of psi_1..psi_8 with the class groups (between-group variance share)
        eta = []
        for k in range(8):
            x = psi[:, k] - psi[:, k].mean()
            gm = np.bincount(grp, weights=x, minlength=G) / np.bincount(grp, minlength=G)
            eta.append(float((gm[grp] ** 2).sum() / (x ** 2).sum()))
        F[f"v.{key}.eta2_group.psi1"] = f2(eta[0])
        F[f"v.{key}.eta2_group.psi1_pc"] = pc(eta[0])
        F[f"v.{key}.eta2_group.max"] = f2(max(eta))

    # ---- spectra for the measurements chart (λ1..λ8 of every single snapshot and the named blocks)
    C["v_spectra"] = dict(x=list(range(1, 9)), xticklabels=[str(k) for k in range(1, 9)], xlabel="k", ylabel="λ_k of the block operator", yrange=None,
                          series=[dict(name=nm.replace("_", "–"), y=VALS[0, pair(*named[nm]), 1:].round(4).tolist())
                                  for nm in ("L1", "L6", "L9", "L12", "L9_L12")])
    C["v_pn"] = dict(x=list(range(S)), xticklabels=sites, xlabel="site (single-snapshot block)",
                     ylabel="participation number / N", yrange=[0, 1],
                     series=[dict(name=f"PN(ψ{k}) / N", y=[float(pn(psi_of(0, pair(s, s)))[k - 1] / N) for s in range(S)])
                             for k in (1, 2, 4, 8)])
    pn_all = np.array([pn(psi_of(0, pair(s, s))) for s in range(S)])
    F["v.pn_single.min"] = f"{pn_all.min():.0f}"
    F["v.pn_single.min_site"] = sites[int(np.unravel_index(pn_all.argmin(), pn_all.shape)[0])]
    F["v.pn_single.min_frac"] = pc1(pn_all.min() / N)
    F["v.pn_single.median_frac"] = pc(np.median(pn_all) / N)

    # ---- agreement between sites (matched Jaccard of r = 8 sets) and with other kernel rules
    J = np.eye(S)
    for a in range(S):
        for b in range(a + 1, S):
            J[a, b] = J[b, a] = matched_jaccard(LAB[0, pair(a, a), r8], LAB[0, pair(b, b), r8], 8, 8)
    C["v_carry_r8"] = dict(rows=sites, cols=sites, z=J.round(3).tolist(), zrange=[0, 1], fmt="2f",
                           rowlabel="site", collabel="site", zlabel="matched Jaccard of the r = 8 sets of two single-snapshot blocks")
    adj = [J[s, s + 1] for s in range(S - 1)]
    F["v.carry.r8.adjacent_min"] = f2(min(adj)); F["v.carry.r8.adjacent_min_pair"] = f"{sites[int(np.argmin(adj))]}→{sites[int(np.argmin(adj)) + 1]}"
    F["v.carry.r8.adjacent_max"] = f2(max(adj)); F["v.carry.r8.adjacent_max_pair"] = f"{sites[int(np.argmax(adj))]}→{sites[int(np.argmax(adj)) + 1]}"
    F["v.carry.r8.L1_L12"] = f2(J[0, S - 1])
    pi = pair(8, 11)
    base = psi_of(0, pi)
    a8 = [span_agreement(base, psi_of(mi, pi)) for mi in range(1, len(meta["modes"]))]
    mj = [matched_jaccard(LAB[0, pi, r8], LAB[mi, pi, r8], 8, 8) for mi in range(1, len(meta["modes"]))]
    for mi, (x, y) in enumerate(zip(a8, mj), start=1):
        F[f"v.rules.L9_L12.a8.mode{mi}"] = f2(x); F[f"v.rules.L9_L12.mj8.mode{mi}"] = f2(y)
    F["v.rules.L9_L12.a8.pp"] = f"{min(a8[0], a8[1]):.2f}–{max(a8[0], a8[1]):.2f}"
    F["v.rules.L9_L12.mj8.pp"] = f"{min(mj[0], mj[1]):.2f}–{max(mj[0], mj[1]):.2f}"
    F["v.modes"] = "; ".join(meta["modes"])
    C["v_ends"] = ends
    # eigenfunctions against labels, by site: lambda-weighted mean over psi_1..psi_8 of eta^2 (between-group share of
    # psi_k's variance) for the 10 class groups and the 100 fine classes; chance (G - 1) / (N - 1)
    def eta2(psi, lab, G_):
        out = []
        for k in range(psi.shape[1]):
            x = psi[:, k] - psi[:, k].mean()
            gm = np.bincount(lab, weights=x, minlength=G_) / np.maximum(np.bincount(lab, minlength=G_), 1)
            out.append(float((gm[lab] ** 2).sum() / (x ** 2).sum()))
        return np.array(out)
    eg, ef = [], []
    for s_ in range(S):
        pi_ = pair(s_, s_); psi = psi_of(0, pi_); w = VALS[0, pi_, 1:9]
        eg.append(float(eta2(psi, grp, G) @ w / w.sum())); ef.append(float(eta2(psi, fine, NF) @ w / w.sum()))
    C["v_eta"] = dict(x=list(range(S)), xticklabels=sites, xlabel="site (single-snapshot block)", yrange=[0, 1],
                      ylabel="η² of ψ₁..ψ₈ (λ-weighted mean)",
                      series=[dict(name="η² against the 100 fine classes", y=ef), dict(name="η² against the 10 class groups", y=eg),
                              dict(name="chance, fine classes", y=[(NF - 1) / (N - 1)] * S, dash=True, color="#8d8c86", width=1.5)])
    F["v.eta_fine.L1"] = f2(ef[0]); F["v.eta_fine.L12"] = f2(ef[-1]); F["v.eta_fine.L6"] = f2(ef[5]); F["v.eta_fine.L9"] = f2(ef[8])
    F["v.eta_group.L1"] = f2(eg[0]); F["v.eta_group.L12"] = f2(eg[-1]); F["v.eta_fine.chance"] = f2((NF - 1) / (N - 1))
    print(f"vit: {time.time() - t0:.1f} s; n_ge16 by site {ge16}; whole {whole}")
    return desc


# =========================================================================================== Mamba contexts
def mamba():
    t0 = time.time()
    meta = json.loads((MC / "meta.json").read_text())
    vocab = json.loads((MC / "vocab.json").read_text())["tokens"]
    tok = lambda i: vocab.get(str(int(i)), f"<{int(i)}>")
    CLS, SRC = meta["classes"], meta["sources"]
    it = np.load(MC / "items.npz")
    T, Nx, Cc, W, U = (it[k].astype(np.int64) for k in ("T", "Nx", "Cc", "W", "U"))
    N = len(T)
    sets = json.loads((MC / "sets_trained.json").read_text())
    cells = json.loads((MC / "cells_trained.json").read_text())
    cells0 = json.loads((MC / "cells_init.json").read_text())
    cmp_rows = json.loads((MC / "compare.json").read_text())["rows"]
    lags = json.loads((MC / "lags.json").read_text())
    F["m.n"] = i_(N)
    F["m.n_snippets"] = i_(len(np.unique(U)))
    F["m.per_snippet"] = str(N // len(np.unique(U)))
    F["m.k"] = str(meta["k"])
    F["m.n_layers"] = str(len(meta["layers"]))
    F["m.n_classes"] = str(len(CLS))
    F["m.sources"] = ", ".join(SRC)
    F["m.ctx_before"] = str(meta["context"][0])
    tt = it["t"].astype(int)
    F["m.t_min"] = str(tt.min()); F["m.t_max"] = str(tt.max())
    F["m.lags"] = ", ".join(map(str, meta["lags"]))
    F["m.single_layers"] = ", ".join(map(str, meta["single_layers"]))
    F["m.n_layouts"] = str(len(meta["layout_objects"]))
    F["m.n_constituents"] = str(2 * len(meta["layers"]))
    F["m.n_sources"] = str(len(SRC))
    ws = np.bincount(W[np.unique(U, return_index=True)[1]], minlength=len(SRC))   # snippets per source category
    F["m.snippets_per_source"] = i_(ws[0]) if (ws == ws[0]).all() else f"{ws.min():,} to {ws.max():,}"

    def cmp(kind, a, b, key="a8", ca="trained"):
        for r in cmp_rows:
            if r["kind"] == kind and r["a"] == a and r["b"] == b and r["checkpoint_a"] == ca:
                return r[key]
        raise KeyError((kind, a, b, key))

    # ---- sets: L4 r = 8 (token sets vs broad sets), B-point r = 16, L24 r = 8, whole r = 8
    def tokens(lst, n=3):
        return ", ".join(tok_disp(tok(a)) for a, _ in lst[:n])
    for obj, r in (("point_L4", 8), ("Bpoint", 16), ("point_L24", 8), ("whole", 8), ("Bstate", 8)):
        key = {"point_L4": "L4", "Bpoint": "Bpoint", "point_L24": "L24", "whole": "whole", "Bstate": "Bstate"}[obj]
        for d in sets[obj][str(r)]:
            j = d["set"] + 1
            p = f"m.{key}.r{r}.set{j}"
            F[p + ".size"] = i_(d["size"])
            F[p + ".share"] = f2(d["top_token_share"]); F[p + ".share_pc"] = pc(d["top_token_share"])
            F[p + ".t90"] = i_(d["tokens_for_90"])
            F[p + ".tok_t"] = tokens(d["tok_t"])
            F[p + ".tok_next"] = tokens(d["tok_next"])
            src = np.array(d["source"]); F[p + ".top_source"] = SRC[int(src.argmax())]
            F[p + ".top_source_share"] = pc(src.max() / d["size"])
            ct = np.array(d["class_t"]); F[p + ".top_class"] = CLS[int(ct.argmax())].replace("_", " ")
            F[p + ".top_class_share"] = pc(ct.max() / d["size"])
    L4 = sets["point_L4"]["8"]
    tokset = [d for d in L4 if d["top_token_share"] >= 0.85]
    broad = sorted([d for d in L4 if d["top_token_share"] < 0.2], key=lambda d: -d["size"])
    F["m.L4.r8.n_token_sets"] = str(len(tokset))
    F["m.L4.r8.token_sets_size"] = i_(sum(d["size"] for d in tokset))
    F["m.L4.r8.token_sets_frac"] = pc1(sum(d["size"] for d in tokset) / N)
    F["m.L4.r8.token_share_min"] = f2(min(d["top_token_share"] for d in tokset))
    F["m.L4.r8.token_share_max"] = f2(max(d["top_token_share"] for d in tokset))
    F["m.L4.r8.token_sets_tokens"] = ", ".join(tok_disp(tok(d["tok_t"][0][0])) for d in sorted(tokset, key=lambda d: -d["size"]))
    F["m.L4.r8.n_broad"] = str(len(broad))
    F["m.L4.r8.broad_sizes"] = " and ".join(i_(d["size"]) for d in broad)
    F["m.L4.r8.broad_frac"] = pc1(sum(d["size"] for d in broad) / N)
    F["m.L4.r8.unassigned"] = i_(N - sum(d["size"] for d in L4))
    Bp = sets["Bpoint"]["16"]
    catch = max(Bp, key=lambda d: d["size"])
    F["m.Bpoint.r16.catch.set"] = str(catch["set"] + 1)
    F["m.Bpoint.r16.catch.size"] = i_(catch["size"])
    F["m.Bpoint.r16.catch.t90"] = i_(catch["tokens_for_90"])
    ct = np.array(catch["class_t"])
    F["m.Bpoint.r16.catch.word_share"] = pc((ct[CLS.index("word_start")] + ct[CLS.index("word_cont")]) / catch["size"])
    F["m.Bpoint.r16.unassigned"] = i_(N - sum(d["size"] for d in Bp))
    # rarity of word tokens (objection 3): contexts whose token at t is a word start / continuation, and how many
    # of them carry a token that occurs once among those
    word = np.isin(Cc, [CLS.index("word_start"), CLS.index("word_cont")])
    ids, cnt = np.unique(T[word], return_counts=True)
    once = np.isin(T, ids[cnt == 1]) & word
    F["m.word.n"] = i_(word.sum()); F["m.word.n_once"] = i_(once.sum()); F["m.word.once_frac"] = pc(once.sum() / word.sum())
    # how the catch-all splits under r = 24 and r = 32 of the same block
    z = np.load(MC / "spectra_trained.npz")
    lab16 = z["Bpoint__seba_label"][meta["ranks"].index(16)]
    mem = lab16 == catch["set"]
    for r in (24, 32):
        lab = z["Bpoint__seba_label"][meta["ranks"].index(r)]
        c = np.bincount(np.where(lab[mem] < 0, r, lab[mem]), minlength=r + 1)
        o = np.argsort(-c[:r])
        F[f"m.Bpoint.catch.r{r}.largest"] = i_(c[o[0]])
        F[f"m.Bpoint.catch.r{r}.largest_set"] = str(int(o[0]) + 1)
        F[f"m.Bpoint.catch.r{r}.largest_set_size"] = i_((lab == o[0]).sum())
        F[f"m.Bpoint.catch.r{r}.n_ge50"] = str(int((c[:r] >= 50).sum()))
        F[f"m.Bpoint.catch.r{r}.unassigned"] = i_(c[r])
    # ---- eigenfunctions of B-point and the whole state: token at t at both ends (12 contexts each, from `ends`)
    for obj, key in (("Bpoint", "Bpoint"), ("whole", "whole"), ("point_L4", "L4"), ("point_L24", "L24")):
        E = z[f"{obj}__ends"]                                   # (16, 2, 12): [largest 12, smallest 12]
        for k in range(4):
            for e, end in ((0, "pos"), (1, "neg")):
                idx = E[k, e]
                ids_, cnt_ = np.unique(T[idx], return_counts=True)
                o = np.argsort(-cnt_, kind="stable")[:3]
                F[f"m.{key}.psi{k + 1}.{end}.tok"] = ", ".join(f"{tok_disp(tok(ids_[q]))} {cnt_[q]}" for q in o)
                sc = np.bincount(W[idx], minlength=len(SRC))
                F[f"m.{key}.psi{k + 1}.{end}.source"] = ", ".join(f"{SRC[q]} {sc[q]}" for q in np.argsort(-sc, kind="stable")[:2] if sc[q])
        for g in ("T", "Cc", "W"):
            e2 = z[f"{obj}__eta2_{g}"]
            F[f"m.{key}.eta2_{g}.psi1"] = f2(e2[0]); F[f"m.{key}.eta2_{g}.max16"] = f2(e2.max())
            F[f"m.{key}.eta2_{g}.weighted"] = f2(float(z[f"{obj}__eta2_{g}_weighted"]))
        F[f"m.{key}.ami_W.r8"] = f3(z[f"{obj}__ami_W"][0]); F[f"m.{key}.ami_Cc.r8"] = f3(z[f"{obj}__ami_Cc"][0])
        F[f"m.{key}.ami_T.r8"] = f3(z[f"{obj}__ami_T"][0])
        vals = z[f"{obj}__vals"]
        F[f"m.{key}.lam1"] = f3(vals[1]); F[f"m.{key}.lam16"] = f3(vals[16])
    # eigenfunctions against labels by layer (stored eta^2, lambda-weighted over psi_1..psi_16)
    single = meta["single_layers"]
    names_eta = {"T": "token at t", "Nx": "token at t+1", "Cc": "token class at t", "W": "kind of source", "U": "snippet"}
    C["m_eta"] = dict(x=single, xticklabels=[f"L{l}" for l in single], xlabel="residual stream, single layer", yrange=[0, 1],
                      ylabel="η² of ψ₁..ψ₁₆ (λ-weighted mean)",
                      series=[dict(name=f"η² against the {nm}", y=[float(z[f"point_L{l}__eta2_{g}_weighted"]) for l in single],
                                   band=[[float(z[f"point_L{l}__eta2_{g}_weighted_band"][0]) for l in single], [float(z[f"point_L{l}__eta2_{g}_weighted_band"][1]) for l in single]])
                              for g, nm in names_eta.items()])
    for l in (4, 24):
        for g in ("T", "Nx", "Cc", "W"):
            F[f"m.eta_w.L{l}.{g}"] = f2(float(z[f"point_L{l}__eta2_{g}_weighted"]))
    F["m.eta_w.T.chance_max"] = f3(max(float(z[f"point_L{l}__eta2_T_weighted_chance"]) for l in single))
    # PN (participation number) of psi_1..psi_16, from the stored int16 vectors
    def pn_obj(obj, zz=z, kmax=16):
        q, sc = zz[f"{obj}__vecs"][:kmax], zz[f"{obj}__scale"][:kmax]
        return pn(q.T.astype(np.float64) * sc / 32767.0)
    pns = {}
    for obj in ("Bpoint", "Bstate", "whole", "point_L4", "point_L24", "Bearly", "Blate", "bag", "pred"):
        P = pn_obj(obj)
        pns[obj] = P
        F[f"m.pn.{obj}.min16"] = f"{P.min():.0f}"; F[f"m.pn.{obj}.min16_frac"] = pc1(P.min() / N)
        F[f"m.pn.{obj}.min8"] = f"{P[:8].min():.0f}"; F[f"m.pn.{obj}.min8_frac"] = pc1(P[:8].min() / N)
        F[f"m.pn.{obj}.median16_frac"] = pc1(np.median(P) / N)
    C["m_spectra"] = dict(x=list(range(1, 17)), xticklabels=[str(k) for k in range(1, 17)], xlabel="k", ylabel="λ_k of the block operator",
                          series=[dict(name=nm, y=z[f"{o}__vals"][1:17].round(4).tolist())
                                  for o, nm in (("Bpoint", "residual, all layers"), ("Bstate", "per-layer state, all layers"),
                                                ("whole", "whole state"), ("point_L4", "residual L4"), ("point_L24", "residual L24"))])
    # ---- walk shares per layer: excess over chance, permutation band in the same (excess) units
    def ex_band(rec):
        S0 = rec["S0"]; lo, hi = rec["band"]
        return [(lo - S0) / (1 - S0), (hi - S0) / (1 - S0)]
    layers = meta["layers"]
    series_point, series_state = [], []
    names = {"T": "token at t", "Nx": "token at t+1", "Cc": "token class at t", "Cn": "token class at t+1", "W": "kind of source", "U": "snippet"}
    for view, out in (("point", series_point), ("state", series_state)):
        for g, nm in names.items():
            y = [cells[f"{view}_L{l}"][g]["ex"] for l in layers]
            bands = [ex_band(cells[f"{view}_L{l}"][g]) for l in layers]
            out.append(dict(name=nm, y=y, band=[[b[0] for b in bands], [b[1] for b in bands]], key=g))
        y = [cells[f"{view}_L{l}"]["kappa"]["kappa_ex"] for l in layers]
        out.append(dict(name="next-token distribution (κ̃)", y=y, key="kappa", dash=True))
        y = [cells[f"{view}_L{l}"]["kappa"]["kappa_ex_T"] for l in layers]
        out.append(dict(name="next-token distribution beyond the token at t (κ̃|T)", y=y, key="kappaT", dash=True))
    for nm, ser in (("m_shares_point", series_point), ("m_shares_state", series_state)):
        C[nm] = dict(x=layers, xticklabels=[f"L{l}" for l in layers], xlabel="layer", yrange=[-0.05, 1],
                     ylabel="excess share of walk mass between linked contexts that share …", series=ser)
    pt = lambda l, g: cells[f"point_L{l}"][g]["ex"]
    ka = lambda c, k: c["kappa"][k]
    F["m.T_ex.L4"] = f3(pt(4, "T")); F["m.T_ex.L24"] = f3(pt(24, "T"))
    F["m.T_ex.L1"] = f3(pt(1, "T"))
    lmax = max(layers, key=lambda l: pt(l, "T")); F["m.T_ex.max"] = f3(pt(lmax, "T")); F["m.T_ex.max_layer"] = f"L{lmax}"
    F["m.Cc_ex.L4"] = f3(pt(4, "Cc")); F["m.Cc_ex.L24"] = f3(pt(24, "Cc"))
    F["m.W_ex.L4"] = f3(pt(4, "W")); F["m.W_ex.L24"] = f3(pt(24, "W"))
    F["m.Nx_ex.L4"] = f3(pt(4, "Nx")); F["m.Nx_ex.L24"] = f3(pt(24, "Nx"))
    F["m.kappa_ex.L4"] = f3(ka(cells["point_L4"], "kappa_ex")); F["m.kappa_ex.L24"] = f3(ka(cells["point_L24"], "kappa_ex"))
    F["m.kappaT.L4"] = f3(ka(cells["point_L4"], "kappa_ex_T")); F["m.kappaT.L24"] = f3(ka(cells["point_L24"], "kappa_ex_T"))
    F["m.kappa_ex.Bpoint"] = f3(ka(cells["Bpoint"], "kappa_ex")); F["m.kappaT.Bpoint"] = f3(ka(cells["Bpoint"], "kappa_ex_T"))
    F["m.T_band_hi.max"] = f3(max(ex_band(cells[f"point_L{l}"]["T"])[1] for l in layers))
    F["m.init.kappa_ex.L24"] = f3(ka(cells0["point_L24"], "kappa_ex"))
    # whole state against the references
    for c in ("whole", "bag", "recent", "Bpoint", "Bstate"):
        for g in ("W", "Cc", "T", "U"):
            F[f"m.ex.{c}.{g}"] = f3(cells[c][g]["ex"])
        F[f"m.ex.{c}.kappa"] = f3(ka(cells[c], "kappa_ex"))
    F["m.init.ex.whole.W"] = f3(cells0["whole"]["W"]["ex"])
    F["m.ami_W.whole_lag16.r8"] = f3(z["whole_lag16__ami_W"][0])
    # comparisons (a_r: mean cos^2 of principal angles between span(psi_1..psi_r); AMI of hard sets)
    F["m.sens.a8.k8"] = f3(cmp("sensitivity", "Bpoint", "Bk8")); F["m.sens.a8.k30"] = f3(cmp("sensitivity", "Bpoint", "Bk30"))
    F["m.sens.a16.k8"] = f3(cmp("sensitivity", "Bpoint", "Bk8", "a16")); F["m.sens.a16.k30"] = f3(cmp("sensitivity", "Bpoint", "Bk30", "a16"))
    F["m.sens.ami16.k8"] = f3(cmp("sensitivity", "Bpoint", "Bk8", "ami16")); F["m.sens.ami16.k30"] = f3(cmp("sensitivity", "Bpoint", "Bk30", "ami16"))
    F["m.init.a8.Bpoint"] = f3(cmp("init_trained", "Bpoint", "Bpoint", ca="init"))
    F["m.init.a16.Bpoint"] = f3(cmp("init_trained", "Bpoint", "Bpoint", "a16", ca="init"))
    F["m.init.a8.whole"] = f3(cmp("init_trained", "whole", "whole", ca="init"))
    F["m.init.ami8.whole"] = f3(cmp("init_trained", "whole", "whole", "ami8", ca="init"))
    for b in ("bag", "pred", "Bstate", "whole"):
        F[f"m.cmp.Bpoint.{b}.a8"] = f3(cmp("objects", "Bpoint", b)); F[f"m.cmp.Bpoint.{b}.a16"] = f3(cmp("objects", "Bpoint", b, "a16"))
    for b in ("bag", "pred"):
        F[f"m.cmp.whole.{b}.a8"] = f3(cmp("objects", "whole", b)); F[f"m.cmp.whole.{b}.ami8"] = f3(cmp("objects", "whole", b, "ami8"))
    F["m.cmp.Bstate.whole.a8"] = f3(cmp("objects", "Bstate", "whole"))
    for a, b in (("Bearly", "Bmid"), ("Bmid", "Blate"), ("Bearly", "Blate")):
        F[f"m.thirds.{a}.{b}.a8"] = f3(cmp("thirds", a, b))
    # lags: retention of the source object's r = 8 SEBA vectors in the lag cells, mean over vectors
    taus = lags["lags"]
    # lags.json "source": the vectors are the all-layer blocks' (point -> Bpoint, state -> Bstate) and the whole state's;
    # each is evaluated on the walk of one layer (or of the whole state) after tau more tokens
    assert lags["source"] == {"point": "Bpoint", "state": "Bstate", "whole": "whole"}, lags["source"]
    lag_names = {"point_L8": "residual block's sets, L8 walk", "point_L16": "residual block's sets, L16 walk",
                 "point_L24": "residual block's sets, L24 walk", "state_L8": "state block's sets, L8 walk",
                 "state_L16": "state block's sets, L16 walk", "state_L24": "state block's sets, L24 walk",
                 "whole": "whole state's sets, own walk"}
    C["m_lags"] = dict(x=taus, xticklabels=[str(t) for t in taus], xlabel="lag τ (tokens read after t)",
                       ylabel="retention f⊤P f / f⊤f of the r = 8 SEBA vectors (mean)", yrange=[0, 1],
                       refline=dict(y=lags["chance"], label="chance −1/(N−1)"),
                       series=[dict(name=lag_names[v], y=lags["views"][v]["mean"], dash=v.startswith("state")) for v in lag_names])
    lv = lags["views"]
    # m.ret.Bpoint_on_L16: the all-layer residual block's r = 8 SEBA vectors on residual layer 16's walk after tau tokens
    F["m.ret.Bpoint_on_L16.t0"] = f2(lv["point_L16"]["mean"][0]); F["m.ret.Bpoint_on_L16.t2"] = f2(lv["point_L16"]["mean"][2])
    F["m.ret.Bpoint_on_L16.t16"] = f2(lv["point_L16"]["mean"][-1])
    F["m.ret.chance"] = f"{lags['chance']:.4f}".replace("-", "−")
    F["m.lag.whole.t0"] = f2(lv["whole"]["mean"][0]); F["m.lag.whole.t16"] = f2(lv["whole"]["mean"][-1])
    F["m.lag.whole.t2"] = f2(lv["whole"]["mean"][2])
    F["m.ret.Bstate_on_L16.t16"] = f2(lv["state_L16"]["mean"][-1])
    F["m.ret.Bpoint_on_L24.t2"] = f2(lv["point_L24"]["mean"][2])
    F["m.lagami8.whole.t16"] = f3(cmp("lag", "whole", "whole_lag16", "ami8"))
    F["m.lagami8.point_L16.t2"] = f3(cmp("lag", "point_L16", "point_L16_lag2", "ami8"))
    for v in ("point_L16", "whole"):
        for t in (1, 2, 16):
            F[f"m.laga8.{v}.t{t}"] = f3(cmp("lag", v, f"{v}_lag{t}"))
    C["m_thirds"] = dict(rows=["L1–8", "L9–16", "L17–24"], cols=["L1–8", "L9–16", "L17–24"], fmt="2f", zrange=[0, 1],
                         z=[[1, cmp("thirds", "Bearly", "Bmid"), cmp("thirds", "Bearly", "Blate")],
                            [cmp("thirds", "Bearly", "Bmid"), 1, cmp("thirds", "Bmid", "Blate")],
                            [cmp("thirds", "Bearly", "Blate"), cmp("thirds", "Bmid", "Blate"), 1]],
                         ml=96, rowlabel="residual block", collabel="residual block", zlabel="a₈ (mean cos² of principal angles, span ψ₁..ψ₈)")
    objs = ["Bpoint", "Bstate", "whole", "bag", "pred"]
    onames = ["residual L1–24", "state L1–24", "whole state", "bag (ref.)", "pred (ref.)"]
    Z = np.eye(len(objs))
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i < j:
                try:
                    Z[i, j] = Z[j, i] = cmp("objects", a, b)
                except KeyError:
                    Z[i, j] = Z[j, i] = cmp("objects", b, a)
    C["m_objects"] = dict(rows=onames, cols=onames, z=Z.round(3).tolist(), fmt="2f", zrange=[0, 1], ml=120,
                          rowlabel="object", collabel="object", zlabel="a₈ (mean cos² of principal angles, span ψ₁..ψ₈)")
    print(f"mamba: {time.time() - t0:.1f} s")
    return pns


# =========================================================================================== older studies
def older():
    steps = [0, 512, 2000, 16000, 143000]
    pm = json.loads((PY / "meta.json").read_text())
    layers = pm["layers"]
    J = np.array([np.load(PY / f"blocks_{s}.npz")["reading_src_jacc"][:, 1] for s in steps])       # (5, 6), r = 4
    Jn = np.array([np.load(PY / f"blocks_{s}.npz")["reading_src_jacc_null"][:, 1] for s in steps])  # (5, 6, 3)
    Jc = np.array([np.load(PY / f"blocks_{s}.npz")["reading_src_jacc_ceiling"][:, 1] for s in steps])
    C["o_pythia"] = dict(x=list(range(len(steps))), xticklabels=[f"{s:,}" for s in steps], xlabel="training step",
                         ylabel="matched Jaccard, r = 4 sets vs the 5 sources", yrange=[0, 1],
                         series=[dict(name=f"layer {l}", y=J[:, i].round(4).tolist()) for i, l in enumerate(layers)]
                         + [dict(name="size ceiling (mean over layers)", y=Jc.mean(1).round(4).tolist(), dash=True, color="#8d8c86", width=1.5)],
                         band=[Jn[:, :, 1].min(1).round(4).tolist(), Jn[:, :, 2].max(1).round(4).tolist()],
                         band_label="permutation null, 95% band")
    F["o.py.step0.min"] = f2(J[0].min()); F["o.py.step0.max"] = f2(J[0].max())
    F["o.py.step512.min"] = f2(J[1].min()); F["o.py.step512.max"] = f2(J[1].max())
    F["o.py.final.min"] = f2(J[-1].min()); F["o.py.final.max"] = f2(J[-1].max())
    F["o.py.null_hi"] = f2(Jn[:, :, 2].max())
    F["o.py.n_layers"] = str(len(layers))
    F["o.py.positions"] = f"{min(pm['positions'])} to {max(pm['positions'])}"; F["o.py.layers"] = ", ".join(map(str, layers))
    F["o.py.n_steps"] = str(len(steps)); F["o.py.steps"] = ", ".join(f"{s:,}" for s in steps)
    F["o.n_snippets"] = f"{int(pm['n']):,}"
    F["o.n_sources"] = str(len(pm["sources"]))
    F["o.per_source"] = f"{int(pm['n']) // len(pm['sources']):,}"
    F["o.snippet_len"] = str(max(pm["positions"]))
    d1 = json.loads((MV1 / "diagnosis" / "diag_v1.json").read_text())["d1"]
    dep = d1["depth"]
    ts = sorted(dep, key=int)
    am = lambda key, r: float(np.mean([dep[t][str(r)][key]["ami"] for t in ts]))
    F["o.v1.depth.r8.cur_class"] = f2(am("cur_class", 8)); F["o.v1.depth.r8.source"] = f2(am("source", 8))
    F["o.v1.depth.cur_class.range"] = f"{min(am('cur_class', r) for r in (2, 3, 4, 6, 8)):.2f}–{max(am('cur_class', r) for r in (2, 3, 4, 6, 8)):.2f}"
    F["o.v1.depth.source.range"] = f"{min(am('source', r) for r in (2, 3, 4, 6, 8)):.2f}–{max(am('source', r) for r in (2, 3, 4, 6, 8)):.2f}"
    rd = d1["reading"]
    src = [rd[b][r]["source"]["ami"] for b in rd for r in rd[b]]
    F["o.v1.reading.source.range"] = f"{min(src):.2f}–{max(src):.2f}"
    F["o.v1.n_reading"] = str(len(rd))
    # T1 root binary cuts of the two whole-snippet studies (Pythia step 143,000 and Mamba v1), cross-tabulated
    cut = {}
    for name, p in (("py", PY / "trees.json"), ("mb", MV1 / "data" / "m130" / "trees.json")):
        nodes = json.loads(p.read_text())[0]["nodes"]
        root = nodes[0]
        side = np.full(1000, -1)
        for c, ch in enumerate(root["children"]):
            side[nodes[ch]["members"]] = c
        cut[name] = side
    agree = max(int((cut["py"] == cut["mb"]).sum()), int((cut["py"] == 1 - cut["mb"]).sum()))
    F["o.rootcut.agree"] = i_(agree)
    F["o.rootcut.n"] = "1,000"
    print("older: Pythia J", J.round(2).tolist(), "root cut agree", agree)


# =========================================================================================== sanity table
def sanity(pns):
    F["s.vit.pn"] = f"{F['v.L9_L12.pn_min']} ({F['v.L9_L12.pn_min_frac']})"
    F["s.vit.pn_l12"] = f"{F['v.L12.pn_min']} ({F['v.L12.pn_min_frac']})"
    F["s.m.pn"] = f"{F['m.pn.Bpoint.min16']} ({F['m.pn.Bpoint.min16_frac']})"
    F["s.m.pn_whole"] = f"{F['m.pn.whole.min16']} ({F['m.pn.whole.min16_frac']})"


def main():
    t0 = time.time()
    vit()
    pns = mamba()
    older()
    sanity(pns)
    bad = [k for k, v in F.items() if not isinstance(v, str) or v == "" or "nan" in v.lower()]
    if bad:
        sys.exit(f"bad facts: {bad}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    obj = {"facts": F, **C}
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(F)} facts) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
