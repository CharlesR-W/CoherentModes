"""Print the key numbers of a fetched run for RESULTS_NOTES.md (PLAN §10).  Labels are used here only
for validation numbers, printed separately.

    uv run --with numpy python -I explore/vit_coherent/summarize.py dinov2s
"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(__file__).resolve().parent / "data" / sys.argv[1]
J = lambda p: json.loads((d / p).read_text())  # noqa: E731
meta, sel = J("meta.json"), J("selection.json")
N = meta["N"]
T = meta["t"]
f2 = lambda xs: " ".join(f"{x:.2f}" if x == x else "nan" for x in xs)  # noqa: E731


def arr(rel, dt, sh):
    return np.fromfile(d / rel, dtype={"f16": np.float16, "f32": np.float32, "u8": np.uint8, "i8": np.int8,
                                       "u16": np.uint16}[dt]).reshape(sh)


print("== meta", meta["N"], meta["N_heldout"], "defaults", meta["default_kernels"])
print("reconstruction", meta["reconstruction"], "readout_vs_hub", meta["readout_vs_hub"])
print("\n== §2 selection (label-free)")
print(sel["reasoning"])
for v in ("cls", "mp"):
    print(f"-- view {v}: per layer stab16 / effproj16 / median eff_n top16 / lambda1 / #lambda>0.1 / neg_frac")
    for k in meta["kernel_ids"]:
        P = sel["views"][v][k]["per_layer"]
        print(f"{k:6s} stab16 ", f2([p["stability_mean"]["16"] for p in P]))
        print(f"{'':6s} stab8  ", f2([p["stability_mean"]["8"] for p in P]))
        print(f"{'':6s} stab32 ", f2([p["stability_mean"]["32"] for p in P]))
        print(f"{'':6s} effP16 ", " ".join(f"{p['eff_n_proj']['16']:.0f}" for p in P))
        print(f"{'':6s} effmed ", " ".join(f"{np.median(p['eff_n_modes'][:16]):.0f}" for p in P))
        print(f"{'':6s} lam1   ", f2([p["spectrum"]["lambda"][1] for p in P]))
        print(f"{'':6s} lam16  ", f2([p["spectrum"]["lambda"][16] for p in P]))
        print(f"{'':6s} n>0.1  ", " ".join(str(p["spectrum"]["decay_r"]["0.1"]) for p in P))
        print(f"{'':6s} negfr  ", " ".join(f"{p['spectrum']['neg_frac']:.1e}" for p in P))
        print(f"{'':6s} a1 l1  ", f2([p["alpha1"]["lambda"][1] for p in P]),
              " a1 effmed", " ".join(f"{np.median(p['alpha1']['eff_n_modes'][:16]):.0f}" for p in P))
print("\n== validation (labels; after the fact)")
for v in ("cls", "mp"):
    for k in meta["kernel_ids"]:
        V = sel["validation"][v][k]
        print(f"{v} {k:6s} knn ", f2(V["knn_acc"]), " | eta2 mean top8 ", f2([np.mean(e[:8]) for e in V["eta2"]]))
print("decodability")
dec = J("decodability.json")
for v in ("cls", "mp"):
    for m in ("ridge", "knn"):
        print(f"{v} {m:5s} fine ", f2(dec[v][m]["fine"]), " super ", f2(dec[v][m]["super"]))

for k in meta["default_kernels"]:
    print(f"\n== default {k}: uncentred sensitivity (mean top-16 cos with centred) per layer")
    print(f2([np.mean(u["cos16_vs_centred"]) for u in sel["uncentred_sensitivity"][k]]))
    L = 12
    S = arr(f"cls/{k}/smooth.f32", "f32", [L, 64, L, 4])
    print("== smoothness S[a, mode0..7 mean, b, t]: rows a, cols b (t=1 then t=16)")
    for ti in (0, 2):
        print(f" t={T[ti]}")
        for a in range(L):
            print(f"  a={a + 1:2d} ", f2(S[a, :8, :, ti].mean(0)))
    D = J(f"cls/{k}/delta.json")
    R = D["results"]
    print("== Delta: adjacent pairs, t=1 and t=16: top+ eval, cv(+0), top- eval, cv(-0), #unsupported of 16")
    for a in range(1, 12):
        r = R[f"a{a:02d}_b{a + 1:02d}"]
        for ti in (0, 2):
            ev, cv, un = r["evals"][ti], r["cv"][ti], r["unsupported"][ti]
            print(f"  {a:2d}->{a + 1:2d} t={T[ti]:2d} +{ev[0]:.3f} cv {cv[0]:.3f} | {ev[8]:.3f} cv {cv[8]:.3f} | "
                  f"unsup {sum(un)} disc+ {r['cv_discarded'][ti][0]:.2f}")
    print("== Delta long pairs (t=1,4,16,64): ev+0, ev-0, cv+0, cv-0, net+0, path+0, unsup")
    for a, b in meta["declared_pairs"]:
        r = R[f"a{a:02d}_b{b:02d}"]
        for ti in range(4):
            print(f"  {a:2d}->{b:2d} t={T[ti]:2d} ev {r['evals'][ti][0]:+.3f} {r['evals'][ti][8]:+.3f} cv "
                  f"{r['cv'][ti][0]:+.3f} {r['cv'][ti][8]:+.3f} net {r['net'][ti][0]:+.3f} path {r['path'][ti][0]:.3f} "
                  f"netneg {r['net'][ti][8]:+.3f} pathneg {r['path'][ti][8]:.3f} unsup {sum(r['unsupported'][ti])}")
    print("== triples: mean of top-4 principal cosines (ac vs ab, ac vs bc, ac vs span)")
    for tr in D["triples"]:
        if tr["t"] in (1, 16):
            print(f"  {tr['abc']} t={tr['t']:2d} {tr['sign']} ", f2([np.mean(tr[c][:4]) for c in ("cos_ac_ab", "cos_ac_bc", "cos_ac_sum")]))
    C = J("controls.json")[k]
    print("== controls (max |eig| at t=1,4,16,64)")
    for pl in C["per_layer"]:
        adj = {kk: vv for kk, vv in pl.items() if kk.startswith("adjacent")}
        print(f"  L{pl['layer']}: ident {pl['identical']['max_abs_eig']} rot {pl['rotation']['max_abs_eig']}\n"
              f"        noise {f2(pl['noise']['max_abs_eig'])} adjacent {[(kk[9:16], f2(vv)) for kk, vv in adj.items()]}")
    F = J(f"cls/{k}/flows.json")
    print("== flows: H(c_b|c_a) for adjacent pairs vs same-layer baseline H(B|A)  (bits)")
    pairs = [tuple(p) for p in F["pairs"]]
    for r in ("8", "16", "32", "64"):
        adj = [F["H_b_given_a"][r][pairs.index((a, a + 1))] for a in range(1, 12)]
        adjr = [F["H_a_given_b"][r][pairs.index((a, a + 1))] for a in range(1, 12)]
        base = [b["H_B_given_A"] for b in F["baseline"][r]]
        print(f"  r={r:2s} adj H(b|a) {f2(adj)}\n        adj H(a|b) {f2(adjr)}\n        base       {f2(base)}")
        print(f"        1->12 H(b|a) {F['H_b_given_a'][r][pairs.index((1, 12))]:.2f} H(a|b) {F['H_a_given_b'][r][pairs.index((1, 12))]:.2f}")
    sb = J(f"cls/{k}/seba.json")
    print("== SEBA coverage / converged / neg_mass / gap, per layer, r=16 and 64")
    for r in ("16", "64"):
        print(f"  r={r} cov ", f2([L_["r"][r]["coverage"] for L_ in sb["layers"]]),
              " conv ", "".join("y" if L_["r"][r]["seba_converged"] else "n" for L_ in sb["layers"]),
              " neg ", f2([L_["r"][r]["neg_mass"] for L_ in sb["layers"]]),
              " shcos ", f2([L_["r"][r]["split_half_mean_cos"] for L_ in sb["layers"]]))
    A = J(f"cls/{k}/attr.json")
    print("== attribution of layer modes: R2 probe / held-out (modes 1-4), and mass: baseline / heads / MLP")
    for li, Ld in enumerate(A["layers"]):
        l = Ld["layer"]
        at = arr(f"cls/{k}/attr/L{l:02d}.f32", "f32", [16, 1 + 72 + 12 * 1536])
        base, heads, mlp = at[:, 0], np.nansum(at[:, 1:73], 1), np.nansum(at[:, 73:], 1)
        habs, mabs = np.nansum(np.abs(at[:, 1:73]), 1), np.nansum(np.abs(at[:, 73:]), 1)
        nb = np.nansum(np.abs(at[:, 73:]).reshape(16, 12, 1536), 2)
        top_blk = np.argmax(nb[:, :l], 1) + 1
        print(f"  L{l:2d} R2 {f2(Ld['r2_probe'][:4])} / {f2(Ld['r2_heldout'][:4])}  base {f2(base[:4])} heads {f2(heads[:4])} "
              f"mlp {f2(mlp[:4])} |h| {f2(habs[:4])} |m| {f2(mabs[:4])} topMLPblock {top_blk[:4]}")
        ld = arr(f"cls/{k}/load/L{l:02d}.f16", "f16", [64, 1536]).astype(np.float32)
        au = arr(f"cls/{k}/auroc/L{l:02d}.f16", "f16", [120, 1536]).astype(np.float32)
        print(f"       max |loading| modes1-4 {f2(np.abs(ld[:4]).max(1))}  #neurons |load|>0.5 (any of top16) "
              f"{int((np.abs(ld[:16]) > 0.5).any(0).sum())}  max AUROC r=16 sets {np.nanmax(au[8:24]):.3f} "
              f"#neurons AUROC>0.9 (r16) {int((np.nan_to_num(au[8:24]) > 0.9).any(0).sum())}")
    Q = J(f"cls/{k}/q.json")
    print("== Q (sv 1..5) and mean-operator eigenvalues (1..5)")
    for p, q in Q.items():
        print(f"  {p} sv {f2(q['sv'][1:6])} mean {f2(q['mean_evals'][:5])} cosR {f2(q['cos_q_right_vs_Pa'][:4])}")
