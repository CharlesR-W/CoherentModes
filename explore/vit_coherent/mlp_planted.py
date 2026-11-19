"""PLAN.md §6b: trained residual MLP on a planted merge task (measured, not asserted).

Task: A, B uniform on Z_8 (independent), y = (A + B) mod 8.  x = E_A[A] + E_B[B] + nu in R^32,
E_A, E_B fixed random rows ~ N(0, I/32) (norm ~ 1), nu ~ N(0, (0.5^2/32) I) (norm ~ 0.5).
y is pairwise independent of A and of B, so the A-, B- and y-contrast spans (7 dims each) are
mutually (near) orthogonal subspaces of the 63-dim (A, B) contrast span (span "AB", also reported).
Model: h0 = W_in x (width 256); 4 pre-LN residual blocks h_{l} = h_{l-1} + W2 GELU(W1 LN(h_{l-1}));
logits = W_out LN(h4). Trained with cross-entropy on y only (AdamW, fresh samples every step).
Views (the complete state at each depth): x, h0, h1, h2, h3, h4, logits; each through
ops.cosine_view. Probe set N = 64 (A, B) combos x 30 = 1920; held-out set of the same size.
Kernels: K-st(15) and K-vmf(kappa = 1/0.07, the DINO-protocol value, declared, not tuned);
t in {1, 4, 16, 64}. Labels (A, B, y) are used only to validate, never to build an operator.
Writes data/mlp.json.  Run: uv run --with torch --with numpy python -I mlp_planted.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ops  # noqa: E402

F64 = torch.float64
D_IN, WIDTH, DEPTH, NCLS, NUIS = 32, 256, 4, 8, 0.5
N_PER = 30
TS = [1, 4, 16, 64]
KAPPA = 1 / 0.07
KERNELS = {"st15": lambda x: ops.kernel_st(x, 15)[0], "vmf": lambda x: ops.kernel_vmf(x, KAPPA)}
SEBA_R = [8, 64]
M_MODES = 8


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.ln, self.fc1, self.fc2 = nn.LayerNorm(w), nn.Linear(w, w), nn.Linear(w, w)

    def forward(self, h):
        return h + self.fc2(nn.functional.gelu(self.fc1(self.ln(h))))


class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D_IN, WIDTH)
        self.blocks = nn.ModuleList([Block(WIDTH) for _ in range(DEPTH)])
        self.ln, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, NCLS)

    def forward(self, x, return_states=False):
        h = self.inp(x)
        states = [h]
        for b in self.blocks:
            h = b(h)
            states.append(h)
        logits = self.out(self.ln(h))
        return (logits, states) if return_states else logits


def sample(n, E_A, E_B, g, balanced=False):
    if balanced:
        ab = torch.arange(64).repeat_interleave(n // 64)
        A, B = ab // 8, ab % 8
    else:
        A, B = torch.randint(0, 8, (n,), generator=g), torch.randint(0, 8, (n,), generator=g)
    x = E_A[A] + E_B[B] + NUIS * torch.randn(n, D_IN, generator=g) / math.sqrt(D_IN)
    return x, A, B, (A + B) % 8


def train(E_A, E_B, steps=4000, bs=512, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 100)
    net = ResMLP()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve = []
    for s in range(steps):
        x, _, _, y = sample(bs, E_A, E_B, g)
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if s % 100 == 0 or s == steps - 1:
            curve.append([s, float(loss.detach())])
    return net, curve


# ----------------------------------------------------------------------------------- decodability


def ridge_gcv(Xtr, ytr, Xte, yte, ncls):
    """Closed-form one-vs-rest ridge fit (a fit, not a trained model); lambda by GCV on the probe."""
    mu = Xtr.mean(0, keepdim=True)
    Xc, Xt = Xtr - mu, Xte - mu
    Y = nn.functional.one_hot(ytr, ncls).to(F64)
    ym = Y.mean(0, keepdim=True)
    Yc = Y - ym
    U, s, Vt = torch.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ Yc
    n = Xc.shape[0]
    best = None
    for lam in (s.max() ** 2) * torch.logspace(-8, 2, 41, dtype=F64):
        f = s ** 2 / (s ** 2 + lam)
        resid = ((Yc - U @ (f[:, None] * UtY)) ** 2).sum()
        gcv = float(n * resid / (n - f.sum()) ** 2)
        if best is None or gcv < best[0]:
            best = (gcv, float(lam))
    lam = best[1]
    W = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY)
    pred = (Xt @ W + ym).argmax(1)
    return float((pred == yte).to(F64).mean()), lam


def knn_dino(Xtr, ytr, Xte, yte, ncls, k=20, T=0.07):
    """DINO-protocol weighted kNN on cosine views (k = 20, T = 0.07)."""
    sim = Xte @ Xtr.T
    v, i = sim.topk(k, dim=1)
    w = torch.exp(v / T)
    votes = torch.zeros(Xte.shape[0], ncls, dtype=F64)
    votes.scatter_add_(1, ytr[i], w)
    return float((votes.argmax(1) == yte).to(F64).mean())


# -------------------------------------------------------------------------------------- analysis


def compressed_eigs(lam, U, B, t):
    C = U.T @ B
    M = C.T @ (C * (lam.abs() ** (2 * t))[:, None])
    return torch.linalg.eigvalsh((M + M.T) / 2)


def cond_entropy_bits(labels, states):
    """H(labels | states) in bits (unassigned state -1 counts as a state)."""
    return ops.flow(states, labels)["H_b_given_a"]


def main():
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 1)
    gE = torch.Generator().manual_seed(1)
    E_A = torch.randn(8, D_IN, generator=gE) / math.sqrt(D_IN)
    E_B = torch.randn(8, D_IN, generator=gE) / math.sqrt(D_IN)
    net, curve = train(E_A, E_B)
    train_s = time.time() - t0
    net.eval()
    gp, gh = torch.Generator().manual_seed(2), torch.Generator().manual_seed(3)
    probe = sample(64 * N_PER, E_A, E_B, gp, balanced=True)
    held = sample(64 * N_PER, E_A, E_B, gh, balanced=True)

    def states(x):
        with torch.no_grad():
            logits, hs = net(x, return_states=True)
        return [x] + hs + [logits]

    names = ["x", "h0", "h1", "h2", "h3", "h4", "logits"]
    raw_p, raw_h = states(probe[0]), states(held[0])
    acc_test = float((raw_h[-1].argmax(1) == held[3]).float().mean())
    print(f"trained in {train_s:.0f}s; held-out y accuracy {acc_test:.4f}; final loss {curve[-1][1]:.4f}")

    labs_p = {"A": probe[1], "B": probe[2], "y": probe[3], "AB": probe[1] * 8 + probe[2]}
    labs_h = {"A": held[1], "B": held[2], "y": held[3], "AB": held[1] * 8 + held[2]}
    views_p = [ops.cosine_view(v) for v in raw_p]
    views_h = [ops.cosine_view(v - rp.mean(0, keepdim=True), centre=False) for v, rp in zip(raw_h, raw_p)]

    # decodability (probe -> held-out), on the cosine view and on the raw state
    dec = {}
    for nm, vp, vh, rp, rh in zip(names, views_p, views_h, raw_p, raw_h):
        dec[nm] = {}
        for lab in ["A", "B", "y", "AB"]:
            nc = 64 if lab == "AB" else 8
            rc, rl = ridge_gcv(vp, labs_p[lab], vh, labs_h[lab], nc)
            rr, _ = ridge_gcv(rp.to(F64), labs_p[lab], rh.to(F64), labs_h[lab], nc)
            kn = knn_dino(vp, labs_p[lab], vh, labs_h[lab], nc)
            dec[nm][lab] = {"ridge_cos_view": rc, "ridge_lambda": rl, "ridge_raw_state": rr, "knn_dino": kn}
        print(f"  decode {nm:<6} " + "  ".join(f"{l}: ridge {dec[nm][l]['ridge_cos_view']:.3f} knn {dec[nm][l]['knn_dino']:.3f}"
                                               for l in ["A", "B", "y"]))

    spans = {lab: ops.contrast_basis(labs_p[lab]) for lab in ["A", "B", "y", "AB"]}  # AB: full 63-dim (A,B) contrast span
    out_k = {}
    for kern in KERNELS:
        print(f"=== {kern} ===")
        eigs = []
        res = {"spectrum": {}, "smoothness": {}, "delta_from_input": {}, "seba": {}, "flows": {}}
        for nm, v in zip(names, views_p):
            K = KERNELS[kern](v)
            P, d = ops.sinkhorn(K, inplace=True)
            lam, U = ops.eig_full(P)
            del P, K
            eigs.append((lam, U))
            gaps = lam[:100] - lam[1:101]
            res["spectrum"][nm] = {"top_eigs": lam[:80].tolist(), **ops.negative_part(lam),
                                   "n_eig_gt_0.5": int((lam > 0.5).sum()), "n_eig_gt_0.9": int((lam > 0.9).sum()),
                                   "largest_gap_after_index": int(gaps.argmax()) + 1}
            res["smoothness"][nm] = {}
            for t in TS:
                res["smoothness"][nm][t] = {}
                for lab, Bs in spans.items():
                    e = compressed_eigs(lam, U, Bs, t)
                    res["smoothness"][nm][t][lab] = {"mean": float(e.mean()), "min": float(e.min()), "max": float(e.max())}
        for li, nm in enumerate(names[1:], start=1):
            res["delta_from_input"][nm] = {}
            for t in TS:
                dm = ops.delta_modes(*eigs[0], *eigs[li], t, M_MODES)
                r = {"pos_vals": dm["pos_vals"].tolist(), "neg_vals": dm["neg_vals"].tolist(),
                     "n_pos_gt_0.05": int((dm["all_vals"] > 0.05).sum()), "n_neg_lt_-0.05": int((dm["all_vals"] < -0.05).sum())}
                for side, V in [("pos", dm["pos_vecs"][:, :7]), ("neg", dm["neg_vecs"][:, :7])]:
                    r[f"pc_{side}7"] = {lab: ops.principal_cosines(Bs, V).tolist() for lab, Bs in spans.items()}
                    # share of each mode's energy inside each label span (validation)
                    r[f"energy_{side}7"] = {lab: ((Bs.T @ V) ** 2).sum(0).tolist() for lab, Bs in spans.items()}
                if nm == "logits":
                    pl = ops.per_layer_terms(eigs, torch.cat([dm["pos_vecs"][:, :4], dm["neg_vecs"][:, :4]], 1), t)
                    r["per_layer_terms_top4_bottom4"] = {"terms": pl["terms"].T.tolist(), "net": pl["net"].tolist(),
                                                          "path": pl["path"].tolist()}
                res["delta_from_input"][nm][t] = r
            d1 = res["delta_from_input"][nm][1]
            print(f"  x->{nm:<6} t=1 pos {d1['pos_vals'][0]:.3f} neg {d1['neg_vals'][0]:.3f}; "
                  f"pc_pos7 min A/B/y {min(d1['pc_pos7']['A']):.2f}/{min(d1['pc_pos7']['B']):.2f}/{min(d1['pc_pos7']['y']):.2f}; "
                  f"pc_neg7 max A/B/y {max(d1['pc_neg7']['A']):.2f}/{max(d1['pc_neg7']['B']):.2f}/{max(d1['pc_neg7']['y']):.2f}")
        for r_ in SEBA_R:
            labs = []
            res["seba"][r_] = {}
            for nm, (lam, U) in zip(names, eigs):
                st = ops.seba_states(lam, U, r_)
                labs.append(st["labels"])
                res["seba"][r_][nm] = {k: st[k] for k in ["gap", "subspace_dist_max", "neg_mass_max", "neg_mass_mean",
                                                          "coverage", "overlap", "n_neg_above_cut", "seba_iters", "seba_converged"]}
                res["seba"][r_][nm]["n_states_used"] = int(torch.unique(st["labels"][st["labels"] >= 0]).numel())
                for lab in ["A", "B", "y", "AB"]:
                    res["seba"][r_][nm][f"H_{lab}_given_state"] = cond_entropy_bits(labs_p[lab], st["labels"])
            res["flows"][r_] = {}
            for li in range(1, len(names)):
                for a in [0, li - 1] if li > 1 else [0]:
                    f = ops.flow(labs[a], labs[li])
                    res["flows"][r_][f"{names[a]}->{names[li]}"] = {
                        "H_b_given_a": f["H_b_given_a"], "H_a_given_b": f["H_a_given_b"], "MI": f["MI"],
                        "flow": f["flow"].tolist(), "states_a": f["states_a"].tolist(), "states_b": f["states_b"].tolist()}
            print(f"  SEBA r={r_}: " + " ".join(
                f"{nm}[cov {res['seba'][r_][nm]['coverage']:.2f} H(y|s) {res['seba'][r_][nm]['H_y_given_state']:.2f} "
                f"H(AB|s) {res['seba'][r_][nm]['H_AB_given_state']:.2f}]" for nm in names))
            print("    H(c_l|c_x) / H(c_x|c_l): " + " ".join(
                f"{names[li]} {res['flows'][r_][f'x->{names[li]}']['H_b_given_a']:.2f}/{res['flows'][r_][f'x->{names[li]}']['H_a_given_b']:.2f}"
                for li in range(1, len(names))))
        out_k[kern] = res

    out = {"config": {"d_in": D_IN, "width": WIDTH, "depth": DEPTH, "n_classes": NCLS, "nuisance_norm": NUIS,
                      "embedding_norm": 1.0, "task": "y = (A + B) mod 8, x = E_A[A] + E_B[B] + nu",
                      "train": "AdamW lr 1e-3 wd 1e-2 cosine, 4000 steps x 512 fresh samples, pre-LN residual MLP, GELU",
                      "N_probe": 64 * N_PER, "N_heldout": 64 * N_PER, "views": names, "t": TS,
                      "kernels": {"st15": "K-st k=15", "vmf": f"K-vmf kappa={KAPPA:.6g}"}, "seba_r": SEBA_R,
                      "delta_modes_m": M_MODES, "seeds": {"embed": 1, "train": 0, "probe": 2, "heldout": 3}},
           "training": {"loss_curve": curve, "heldout_y_acc": acc_test, "train_seconds": train_s},
           "decodability": dec, "kernels": out_k, "seconds": time.time() - t0}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "mlp.json"), "w") as fh:
        json.dump(out, fh, default=float)
    print(f"wrote data/mlp.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
