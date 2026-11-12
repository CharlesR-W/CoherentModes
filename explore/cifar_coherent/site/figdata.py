"""Small derived arrays for the Overview/Method figures that the stored measurement cannot provide
(true cosine-distance profiles and a few real operator matrices).  Runs on CPU in about a minute.

    uv run --no-project --with numpy --with torch --with modal \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        python site/figdata.py <tag> [--steps 64000] [--ckdir /var/tmp/cm-figs/ck]

Reads data/<tag>/probe.npz (same 2,000 probe images as the measurement) and the ResNet-20 checkpoints
(fetched from the Modal volume cm-cifar-coherent if not in --ckdir).  Writes site/figdata/<tag>_kernel.npz:

  steps, sites, k
  dist_rank   (T', S, N-1) f32  median over points of the cosine distance (1 - C) to the r-th nearest other point
  kappa       (T', S, 5)  f32   kappa of the five rules, recomputed here from the definitions in SPEC section 2
  nn_mass     (T', S, 5)  f32   median over points of the kernel mass fraction carried by the k nearest neighbours
  cum_ranks   (R,) i32          ranks at which the cumulative-mass curve is tabulated
  cum_mass    (T', S, 5, R) f32 median over points of the fraction of kernel mass in the r nearest neighbours
  rules                         rule names, in SPEC order

and the full float16 Gram matrices of that step to /var/tmp/cm-figs (not the repo), which
make_figures.py uses for the pipeline figures (it recomputes Sinkhorn there).  --grams-only computes just those
Grams (one checkpoint) and leaves figdata/ untouched.
Nothing here is part of the measurement; it re-derives the same quantities for display.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ZOO = Path.home() / "code" / "ModelZoo" / "src"
SITES = ["stem"] + [f"b{i}" for i in range(1, 10)] + ["pool", "logits"]
RULES = ["knn-scale", "half-mass (two-distance heuristic)", "half-mass (calibrated)", "global 5", "global 30"]
K = 15
MEAN = np.array([0.4914, 0.4822, 0.4465]) * 255.0
STD = np.array([0.2470, 0.2435, 0.2616]) * 255.0


def activations(ck_path: Path, images: np.ndarray):
    import torch
    sys.path.insert(0, str(ZOO))
    from zoo.models import build_model

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    x = torch.as_tensor(images).permute(0, 3, 1, 2).double()
    x = ((x - torch.as_tensor(MEAN).view(1, 3, 1, 1)) / torch.as_tensor(STD).view(1, 3, 1, 1)).float()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 250):
            h = model.act(model.bn1(model.conv1(x[s:s + 250])))
            acts = [h]
            for blk in model.blocks:
                h = blk(h)
                acts.append(h)
            p = h.mean(dim=(2, 3))
            acts += [p, model.fc(p)]
            out.append([a.reshape(len(a), -1).numpy().astype(np.float32) for a in acts])
    return [np.concatenate([o[i] for o in out]) for i in range(len(SITES))]


def gram(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    a = a - a.mean(0, keepdims=True)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return np.clip(a @ a.T, -1.0, 1.0)


def mass_frac(ds: np.ndarray, kappa: float, k: int) -> np.ndarray:
    """ds: (N, N-1) row-sorted off-diagonal distances. Fraction of off-diagonal kernel mass in the k nearest."""
    w = np.exp(-kappa * (ds - ds[:, :1]))  # shift per row for stability; ratios unchanged
    return w[:, :k].sum(1) / w.sum(1)


def calibrated(ds: np.ndarray, k: int) -> float:
    lo, hi = 1e-3, 1e4
    for _ in range(48):
        mid = np.sqrt(lo * hi)
        if np.median(mass_frac(ds, mid, k)) < 0.5:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--steps", default="0,64000")
    ap.add_argument("--ckdir", default="/var/tmp/cm-figs/ck")
    ap.add_argument("--gram-step", type=int, default=64000)
    ap.add_argument("--grams-only", action="store_true",
                    help="only write the Gram stack of --gram-step to /var/tmp/cm-figs (leave figdata/*.npz alone)")
    a = ap.parse_args()
    steps = [a.gram_step] if a.grams_only else [int(s) for s in a.steps.split(",")]
    data = ROOT / "data" / a.tag
    probe = np.load(data / "probe.npz")
    images = probe["images"]
    n = len(images)
    ckdir = Path(a.ckdir)
    ckdir.mkdir(parents=True, exist_ok=True)
    ranks = np.unique(np.round(np.logspace(0, np.log10(n - 1), 90)).astype(int))
    dist_rank = np.zeros((len(steps), len(SITES), n - 1), np.float32)
    kap = np.zeros((len(steps), len(SITES), 5), np.float32)
    nnm = np.zeros_like(kap)
    cum = np.zeros((len(steps), len(SITES), 5, len(ranks)), np.float32)
    for ti, step in enumerate(steps):
        ck = ckdir / f"step_{step}.pt"
        if not ck.exists():
            subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "cm-cifar-coherent",
                            f"runs/resnet20-relu-seed0/checkpoints/step_{step}.pt", str(ck)], check=True)
        acts = activations(ck, images)
        print(step, "activations done", flush=True)
        grams = []
        for si in range(len(acts)):
            c = gram(acts[si])
            acts[si] = None
            grams.append(c.astype(np.float16))
            d = 1.0 - c
            np.fill_diagonal(d, np.inf)
            ds = np.sort(d, axis=1)[:, :-1]
            dist_rank[ti, si] = np.median(ds, axis=0)
            dk = float(np.median(ds[:, K - 1]))
            dpair = float(np.median(d[np.isfinite(d)]))
            ks = [1.0 / dk, np.log(n / K) / max(dpair - dk, 1e-3), calibrated(ds, K), 5.0, 30.0]
            for mi, kv in enumerate(ks):
                kap[ti, si, mi] = kv
                w = np.exp(-kv * (ds - ds[:, :1]))
                cs = np.cumsum(w, axis=1)
                cum[ti, si, mi] = np.median(cs[:, ranks - 1] / cs[:, -1:], axis=0)
                nnm[ti, si, mi] = np.median(cs[:, K - 1] / cs[:, -1])
            print(step, SITES[si], " ".join(f"{v:.2f}" for v in ks), " mass", " ".join(f"{v:.2f}" for v in nnm[ti, si]),
                  flush=True)
        if step == a.gram_step:
            np.save(Path("/var/tmp/cm-figs") / f"grams_{a.tag}_{step}.npy", np.stack(grams))
            print("wrote", Path("/var/tmp/cm-figs") / f"grams_{a.tag}_{step}.npy", flush=True)
    if a.grams_only:
        return
    out = HERE / "figdata"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{a.tag}_kernel.npz", steps=np.array(steps), sites=np.array(SITES), k=K,
                        dist_rank=dist_rank, kappa=kap, nn_mass=nnm, cum_ranks=ranks.astype(np.int32),
                        cum_mass=cum, rules=np.array(RULES))
    print("wrote", out / f"{a.tag}_kernel.npz")


if __name__ == "__main__":
    main()
