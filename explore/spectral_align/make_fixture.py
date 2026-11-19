"""Synthetic stand-in for data/ (the measurement), at full shapes, so analyze.py and the site tab can be built first.

    uv run --with numpy python explore/spectral_align/make_fixture.py [--out explore/spectral_align/data_fixture]

Writes the files of the data contract (cells.json, pairs_{0,1}.npz, cka.npy, eig_{0,1}.npy, span_{0,1}.npy,
U_<net>_<step>.npy, probe.npz, iso_accuracy.json, accuracies.json, timings.json) plus fixture.json, which marks the
directory as synthetic. Nothing here is a measurement: numbers carry no meaning.

Planted structure. Every cell (network, checkpoint, site) gets a 64-dimensional subspace C of a shared D-dimensional
ambient space (embedded in R^2000 by a fixed orthonormal Q, orthogonal to the constant), C = leading columns of
QR(sum of smooth components). The components depend smoothly on a stage-aware relative depth x of the site and on
training progress tau = log(1+step)/log(1+64000), plus a stage-specific term, an "input" term that exists before
training, and a per-network noise term (the seed floor). So within a network distances grow with |dx| and with
training-step differences; across depths the closest site is at the same x, which DTW should recover.
Projector, eigenvalue-only and heat distances are computed exactly from these C and eigenvalues (heat on the top-64
modes only); U = Q C, so distances recomputed from the shipped U equal the stored ones.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nets import DEPTH_N, site_names, site_stage  # noqa: E402

K_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 32, 40, 48, 64)   # = align.K_GRID (align imports torch)
T_GRID = (1, 4, 16, 64, 256)
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
NETS = [f"resnet{d}-relu-seed{s}" for d in (20, 32, 56) for s in (0, 1)]
N, D, K = 2000, 144, 64
FINAL_ACC = {"resnet20-relu-seed0": .9152, "resnet20-relu-seed1": .9181, "resnet32-relu-seed0": .9253,
             "resnet32-relu-seed1": .9269, "resnet56-relu-seed0": .9334, "resnet56-relu-seed1": .9311}
# iso-accuracy: one off-schedule step, one that coincides with a checkpoint, one off-schedule, one unavailable
ISO = {"resnet32-relu-seed0": 19200, "resnet32-relu-seed1": 25600, "resnet56-relu-seed0": 12800, "resnet56-relu-seed1": None}


def acc_at(net, step):
    return float(.1 + (FINAL_ACC[net] - .1) * (1 - math.exp(-step / 9000)) / (1 - math.exp(-64000 / 9000)))


def rel_depth(depth):
    n = DEPTH_N[depth]
    out = []
    for i, s in enumerate(site_stage(depth)):
        if s == 0:
            out.append(0.0)
        elif s <= 3:
            j = i - (s - 1) * n
            out.append(.04 + .86 * (s - 1 + j / n) / 3)
        else:
            out.append({4: .95, 5: 1.0}[s])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data_fixture"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.resolve() == (HERE / "data").resolve():
        sys.exit("refusing to write the fixture into data/")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118)

    # probe: the explorer's probe if present (so thumbnails line up), else synthetic labels
    site_probe = HERE.parent / "cifar_coherent" / "data" / "r20s0b" / "probe.npz"
    if site_probe.exists():
        with np.load(site_probe) as p:
            idx, labels = p["idx"].astype(np.int64), p["labels"].astype(np.int64)
    else:
        idx, labels = np.arange(N), np.repeat(np.arange(10), N // 10)
    np.savez(out / "probe.npz", idx=idx, labels=labels)

    # ambient basis: 9 class contrasts first, then random, all orthogonal to the constant
    onehot = np.eye(10)[labels][:, :9]
    raw = np.concatenate([onehot, rng.standard_normal((N, D - 9))], 1)
    raw -= raw.mean(0)
    Q = np.linalg.qr(raw)[0]                                     # (N, D)

    # components (D x K each)
    M = 8
    W_bump = rng.standard_normal((M, D, K))
    W_stage = rng.standard_normal((6, D, K))
    W_in = rng.standard_normal((D, K))
    colw = 1 / (1 + .06 * np.arange(K))                          # leading columns dominate
    centres = np.linspace(0, 1, M)

    cells, C = [], []
    eigs = {0: [], 1: []}
    spans = {0: [], 1: []}
    iso_json = {}
    for net in NETS:
        depth, seed = int(net[6:8]), int(net[-1])
        names, stages, x = site_names(depth), site_stage(depth), rel_depth(depth)
        W_seed = rng.standard_normal((D, K))
        steps = sorted(set(STEPS + ([ISO[net]] if ISO.get(net) else [])))
        if depth > 20:
            target = FINAL_ACC[f"resnet20-relu-seed{seed}"]
            s_ = ISO[net]
            if s_ is None:
                iso_json[net] = dict(target=target, available=False, status="unavailable", step=3200,
                                     accuracy=target + .012, overshoot=.012)
            else:
                iso_json[net] = dict(target=target, available=True, status="available", step=s_,
                                     accuracy=target + .002, overshoot=.002)
        for step in steps:
            tau = math.log1p(step) / math.log1p(64000)
            W_step = rng.standard_normal((D, K))
            for si in range(len(names)):
                xi = x[si]
                phi = np.exp(-(xi - centres) ** 2 / (2 * .14 ** 2))
                learned = np.tensordot(phi, W_bump, 1) + .8 * W_stage[stages[si]]
                A = (.25 + 1.6 * tau) * learned + 1.2 * math.exp(-2.5 * xi) * W_in + .32 * W_seed + .12 * W_step \
                    + .05 * rng.standard_normal((D, K))
                A = A * colw
                C.append(np.linalg.qr(A)[0][:, :K])
                # spectrum: 9 separated modes (sharper with depth and training), then a continuum
                sharp = .3 + .7 * tau * (.4 + .6 * xi)
                top = .985 - np.arange(9) * (.012 + .03 * sharp)
                cont_start = top[-1] - (.02 + .08 * sharp)
                kk = np.arange(1990)
                cont = cont_start * np.exp(-kk / (40 + 200 * (1 - sharp))) * (1 + .002 * rng.standard_normal(1990))
                for r in (0, 1):
                    e = np.concatenate([[1.0], np.clip(top + .002 * r, 0, 1), np.sort(np.clip(cont * (1 - .01 * r), 0, 1))[::-1]])
                    eigs[r].append(e.astype(np.float32))
                    h = int(round(2 + 7 * sharp + 10 * tau * xi)) if step > 0 else 2
                    ks = np.arange(1, 81)
                    sp = np.where(ks <= h, .93 + .05 * rng.random(80), .45 * np.exp(-(ks - h) / 8) + .08 * rng.random(80))
                    sp[8] = max(sp[8], .7 + .2 * tau)                 # the cut after the 9 separated modes
                    spans[r].append(np.clip(sp, 0, 1).astype(np.float32))
                cells.append(dict(index=len(cells), net=net, depth=depth, seed=seed, step=step,
                                  is_iso_accuracy=bool(ISO.get(net) == step), site_index=si, site_name=names[si],
                                  stage=stages[si], val_acc=iso_json[net]["accuracy"] if ISO.get(net) == step else acc_at(net, step)))
    nc = len(cells)
    # rule 1 fails on a few collapsed early ResNet-56 snapshots (as in the real run): NaN rows for that rule
    bad1 = [c["index"] for c in cells if (c["net"], c["step"]) == ("resnet56-relu-seed0", 64) and c["site_index"] < 4] + \
           [c["index"] for c in cells if (c["net"], c["step"]) == ("resnet56-relu-seed1", 256) and c["site_index"] == 5]
    C = np.stack(C).astype(np.float64)                             # (nc, D, K)
    for r in (0, 1):
        spans[r] = np.stack(spans[r])
        failed = rng.choice(nc, 5, replace=False)
        for i in failed:                                           # a few failed subset rebuilds
            spans[r][i] = np.nan
        if r == 0:
            (out / "subset_failures.json").write_text(json.dumps([dict(net=cells[i]["net"], step=cells[i]["step"], site_index=cells[i]["site_index"], error="fixture") for i in failed]))
        np.save(out / f"eig_{r}.npy", np.stack(eigs[r]))
        np.save(out / f"span_{r}.npy", spans[r])
    print(f"{nc} cells", flush=True)

    ix = np.array(K_GRID) - 1
    for r in (0, 1):
        Cr = C if r == 0 else np.stack([np.linalg.qr(c + .04 * rng.standard_normal(c.shape))[0] for c in C])
        e = np.stack(eigs[r]).astype(np.float64)[:, 1:K + 1]
        proj = np.zeros((nc, nc, len(K_GRID)), np.float32)
        eigd = np.zeros_like(proj)
        heat = np.zeros((nc, nc, len(T_GRID)), np.float32)
        cka = np.zeros((nc, nc), np.float32) if r == 0 else None
        wts = [e ** t for t in T_GRID]
        wn = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in wts]
        wc = np.sqrt(e)
        wc = wc / np.linalg.norm(wc, axis=1, keepdims=True)
        for a_ in range(nc):
            G2 = np.matmul(Cr[a_].T, Cr[a_:]) ** 2                 # (nb, K, K)
            cum = G2.cumsum(1).cumsum(2)
            proj[a_, a_:] = 1 - cum[:, ix, ix] / (ix + 1)
            eigd[a_, a_:] = np.sqrt(((e[a_] - e[a_:]) ** 2).cumsum(1)[:, ix] / (ix + 1))
            for ti in range(len(T_GRID)):
                heat[a_, a_:, ti] = 1 - np.einsum("k,nkl,nl->n", wn[ti][a_], G2, wn[ti][a_:])
            if cka is not None:
                cka[a_, a_:] = 1 - np.einsum("k,nkl,nl->n", wc[a_], G2, wc[a_:])
            if a_ % 200 == 0:
                print(f"rule {r}: row {a_}/{nc}", flush=True)
        for arr in [proj, eigd, heat] + ([cka] if cka is not None else []):
            iu = np.triu_indices(nc, 1)
            arr[(iu[1], iu[0])] = arr[iu]
            d = np.arange(nc)
            arr[d, d] = 0
            np.clip(arr, 0, 1, out=arr)
        if r == 1:
            for arr in (proj, eigd, heat):
                arr[bad1] = np.nan
                arr[:, bad1] = np.nan
        np.savez(out / f"pairs_{r}.npz", proj=proj, eigd=eigd, heat=heat)
        if cka is not None:
            noise = (.01 * rng.random((nc, nc))).astype(np.float32)
            cka = np.clip(cka + noise + noise.T, 0, 1)
            cka[np.arange(nc), np.arange(nc)] = 0
            np.save(out / "cka.npy", cka)
        del proj, eigd, heat

    # eigenvectors (rule 0): every network at the final step, ResNet-20 seed 0 at every step
    for net in NETS:
        for step in sorted({c["step"] for c in cells if c["net"] == net}):
            if step != 64000 and net != NETS[0]:
                continue
            sel = [c["index"] for c in cells if c["net"] == net and c["step"] == step]
            np.save(out / f"U_{net}_{step}.npy", np.stack([(Q @ C[i]).astype(np.float32) for i in sel]))
    e1, s1 = np.load(out / "eig_1.npy"), np.load(out / "span_1.npy")
    e1[bad1], s1[bad1] = np.nan, np.nan
    np.save(out / "eig_1.npy", e1)
    np.save(out / "span_1.npy", s1)
    (out / "rule1_unavailable.json").write_text(json.dumps([dict(index=i, net=cells[i]["net"], step=cells[i]["step"], site=cells[i]["site_index"], error="fixture: perplexity calibration failed") for i in bad1]))
    (out / "cells.json").write_text(json.dumps(cells))
    (out / "iso_accuracy.json").write_text(json.dumps(iso_json, indent=2))
    (out / "accuracies.json").write_text(json.dumps({n: FINAL_ACC[n] for n in NETS}))
    (out / "timings.json").write_text(json.dumps({"fixture": True}))
    (out / "fixture.json").write_text(json.dumps({"fixture": True, "note": "synthetic data from make_fixture.py; numbers carry no meaning"}))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
