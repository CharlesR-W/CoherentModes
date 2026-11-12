"""Measure Grams -> vMF kernels -> block dynamic-Laplacian modes -> SEBA sets -> UMAP,
for chosen checkpoints of the ResNet-20 run.  Exploratory side study.

Detached: a CPU coordinator on Modal fans out per checkpoint, so the VPS can die.

    uv run --with modal python explore/cifar_coherent/measure.py deploy
    uv run --with modal python explore/cifar_coherent/measure.py launch <run-name> <tag> step1,step2,...
    uv run --with modal python explore/cifar_coherent/measure.py status <tag>
    uv run --with modal python explore/cifar_coherent/measure.py fetch <tag>     # -> explore/cifar_coherent/data/<tag>

Output on Volume cm-cifar-coherent:/analysis/<tag>/
  probe.npz            idx, labels, images (N,32,32,3) uint8
  stats_<step>.json    per-site kNN scale, participation ratio, within/between class cos, val acc
  kappa.json           kappa per (mode, step, site), mode names, rule
  spec_<step>.npz      vals (M,P,K) f32, vecs (M,P,K,N) int8, seba (M,P,len(R),N) int8
  umap_<step>.npy      (S, N, 2) f32
Modes M (see _modes): two per-site rules, then global kappa values GLOBAL_KAPPAS.
Pairs P: all (a, b) with a <= b over the S sites, in row-major order.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "cm-cifar-coherent-measure"
HERE = Path(__file__).resolve().parent
ZOO_SRC = Path.home() / "code" / "ModelZoo" / "src"

N_PER_CLASS = 200
KNN_K = 15
N_MODES = 9
SEBA_R = (2, 3, 4, 6, 8)
GLOBAL_KAPPAS = (2.0, 5.0, 12.0, 30.0, 75.0)
ALPHA = 1.0
N_PCS = 50
SITES = ["stem"] + [f"b{i}" for i in range(1, 10)] + ["pool", "logits"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "umap-learn")
    .env({"PYTHONPATH": "/root/src:/root/cc"})
    .add_local_dir(str(ZOO_SRC), remote_path="/root/src")
    .add_local_file(str(HERE / "coherent.py"), remote_path="/root/cc/coherent.py")
    .add_local_file(str(HERE / "seba.py"), remote_path="/root/cc/seba.py")
)
app = modal.App(APP)
zoo_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name("cm-cifar-coherent", create_if_missing=True)
VOLS = {"/zoo": zoo_vol, "/out": out_vol}


def _pairs(s: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(s) for b in range(a, s)]


def _setup(run: str, step: int, device: str):
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model

    ck = load_checkpoint(Path(f"/out/runs/{run}/checkpoints/step_{step}.pt"))
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval().to(device)
    data = build_data(cfg, Path("/zoo/data"), "cpu")
    y = data.test_y.numpy()
    rng = np.random.RandomState(0)
    idx = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], N_PER_CLASS, replace=False)
                                  for c in range(10)]))
    xs = data.normalize(data.test_x[idx]).to(device)
    acts = []
    with torch.no_grad():
        out = model.act(model.bn1(model.conv1(xs)))
        acts.append(out)
        for blk in model.blocks:
            out = blk(out)
            acts.append(out)
        pooled = out.mean(dim=(2, 3))
        acts.append(pooled)
        acts.append(model.fc(pooled))
    assert len(acts) == len(SITES)
    return idx, y[idx], data.test_x[idx], acts


def _grams(acts):
    from coherent import centred_cosine_gram
    return [centred_cosine_gram(a).float() for a in acts]


@app.function(image=image, gpu=["L4", "A10", "L40S"], timeout=1800, volumes=VOLS)
def stats(run: str, tag: str, step: int) -> dict:
    import numpy as np
    import torch
    from coherent import knn_dist, participation_ratio

    assert torch.cuda.is_available()
    t0 = time.time()
    idx, lab, imgs, acts = _setup(run, step, "cuda")
    root = Path(f"/out/analysis/{tag}")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "probe.npz").exists():
        np.savez_compressed(root / "probe.npz", idx=idx, labels=lab,
                            images=imgs.permute(0, 2, 3, 1).numpy())
    labt = torch.as_tensor(lab, device="cuda")
    same = labt[:, None] == labt[None, :]
    eye = torch.eye(len(lab), dtype=torch.bool, device="cuda")
    rows = []
    for name, c in zip(SITES, _grams(acts)):
        kd = knn_dist(c.double(), KNN_K)
        rows.append({
            "site": name,
            "knn_median": float(kd.median()),
            "pair_median": float((1 - c[~eye]).median()),
            "participation_ratio": participation_ratio(c.double()),
            "cos_within": float(c[same & ~eye].mean()),
            "cos_between": float(c[~same].mean()),
        })
    man = json.loads(Path(f"/out/runs/{run}/manifest.json").read_text())
    ck = next((c for c in man.get("checkpoints", []) if c.get("step") == step), {})
    res = {"step": step, "val_acc": ck.get("val_acc"), "val_loss": ck.get("val_loss"),
           "train_acc": ck.get("train_acc"), "sites": rows, "seconds": round(time.time() - t0, 1)}
    (root / f"stats_{step}.json").write_text(json.dumps(res, indent=1))
    out_vol.commit()
    return res


@app.function(image=image, gpu=["L4", "A10", "L40S"], timeout=3600, volumes=VOLS)
def spectral(run: str, tag: str, step: int, modes: list) -> dict:
    import numpy as np
    import torch
    from coherent import block_modes, diffusion_operator, seba, seba_labels

    assert torch.cuda.is_available()
    t0 = time.time()
    _, _, _, acts = _setup(run, step, "cuda")
    cs = _grams(acts)
    root = Path(f"/out/analysis/{tag}")
    # PCs of the centred, normalised activations for UMAP (exact cosine geometry, truncated)
    pcs = []
    for c in cs:
        lam, u = torch.linalg.eigh(c)
        lam, u = lam.flip(0)[:N_PCS].clamp_min(0), u.flip(1)[:, :N_PCS]
        pcs.append((u * lam.sqrt()).cpu().numpy())
    np.save(root / f"pcs_{step}.npy", np.stack(pcs).astype(np.float32))
    pairs = _pairs(len(SITES))
    n = cs[0].shape[0]
    vals = np.zeros((len(modes), len(pairs), N_MODES), np.float32)
    vecs = np.zeros((len(modes), len(pairs), N_MODES, n), np.int8)
    sb = np.full((len(modes), len(pairs), len(SEBA_R), n), -1, np.int8)
    for mi, (_, ks) in enumerate(modes):
        ms = [diffusion_operator(c, k, ALPHA) for c, k in zip(cs, ks)]
        for pi, (a, b) in enumerate(pairs):
            v, f = block_modes(ms[a:b + 1], N_MODES)
            vals[mi, pi] = v.cpu().numpy()
            fc = f.cpu().double()
            vecs[mi, pi] = np.round(127 * (fc / fc.abs().max(0).values).T.numpy()).astype(np.int8)
            for ri, r in enumerate(SEBA_R):
                sb[mi, pi, ri] = seba_labels(seba(fc, r)).numpy().astype(np.int8)
    np.savez_compressed(root / f"spec_{step}.npz", vals=vals, vecs=vecs, seba=sb)
    out_vol.commit()
    return {"step": step, "seconds": round(time.time() - t0, 1)}


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes=VOLS)
def embed(tag: str, step: int) -> dict:
    import numpy as np
    import umap

    t0 = time.time()
    root = Path(f"/out/analysis/{tag}")
    out_vol.reload()
    pcs = np.load(root / f"pcs_{step}.npy")
    n = pcs.shape[1]
    rel = [{i: i for i in range(n)} for _ in range(pcs.shape[0] - 1)]
    al = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, random_state=0, alignment_regularisation=0.02)
    al.fit(list(pcs), relations=rel)
    emb = np.stack(al.embeddings_).astype(np.float32)
    np.save(root / f"umap_{step}.npy", emb)
    out_vol.commit()
    return {"step": step, "seconds": round(time.time() - t0, 1)}



def _modes(st: list[dict]) -> tuple[list[str], list[list[list[float]]], dict]:
    """Kappa per (mode, step, site).
    'layer: kNN scale'   kappa = 1 / d_k  (d_k = median over points of distance to k-th NN): kernel e^-1 at d_k.
    'layer: half-mass'   kappa = log(N/k) / (d_pair - d_k): a k-th neighbour outweighs a typical pair by N/k,
                         so the k nearest neighbours carry about as much kernel mass as all other points.
    'global kappa = c'   one value for every site and step."""
    import numpy as np
    n = 10 * N_PER_CLASS
    knn = np.array([[s["knn_median"] for s in r["sites"]] for r in st])
    pair = np.array([[s["pair_median"] for s in r["sites"]] for r in st])
    names = ["layer: kNN scale", "layer: half-mass"] + [f"global κ = {c:g}" for c in GLOBAL_KAPPAS]
    per = [1.0 / knn, np.log(n / KNN_K) / np.clip(pair - knn, 1e-3, None)]
    per += [np.full_like(knn, c) for c in GLOBAL_KAPPAS]
    info = {"knn_scale_global_median": float(1.0 / np.median(knn)),
            "half_mass_global_median": float(np.median(per[1]))}
    return names, [p.tolist() for p in per], info


def _spectral_stage(run, tag, steps, st, say):
    root = Path(f"/out/analysis/{tag}")
    names, per, info = _modes(st)
    (root / "kappa.json").write_text(json.dumps(
        {"steps": steps, "sites": SITES, "modes": names, "kappa": per, **info,
         "rule": f"k = {KNN_K}; d_k = median over points of cosine distance (1 - cos) to the k-th NN; "
                 "d_pair = median pairwise distance. kNN scale: kappa = 1/d_k. "
                 "Half-mass: kappa = log(N/k)/(d_pair - d_k). Global: fixed kappa on cosine for all cells.",
         "seba_r": list(SEBA_R), "n_modes": N_MODES, "alpha": ALPHA,
         "pairs": _pairs(len(SITES))}, indent=1))
    say(f"modes {names}; spectral")
    args = [[(nm, per[m][t]) for m, nm in enumerate(names)] for t in range(len(steps))]
    return list(spectral.map([run] * len(steps), [tag] * len(steps), steps, args))


@app.function(image=image, timeout=6 * 3600, volumes=VOLS)
def respectral(run: str, tag: str, steps: list[int]) -> dict:
    root = Path(f"/out/analysis/{tag}")
    st = [json.loads((root / f"stats_{s}.json").read_text()) for s in steps]

    def say(m):
        with open(root / "coordinator.log", "a") as f:
            f.write(time.strftime("%H:%M:%S ") + m + "\n")
        out_vol.commit()
    sp = _spectral_stage(run, tag, steps, st, say)
    say("respectral done")
    return {"spectral_s": [r["seconds"] for r in sp]}


@app.function(image=image, timeout=6 * 3600, volumes=VOLS)
def coordinate(run: str, tag: str, steps: list[int]) -> dict:
    import numpy as np

    root = Path(f"/out/analysis/{tag}")
    root.mkdir(parents=True, exist_ok=True)
    log = root / "coordinator.log"

    def say(m):
        with open(log, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + m + "\n")
        out_vol.commit()

    say(f"stats for {len(steps)} steps")
    st = list(stats.map([run] * len(steps), [tag] * len(steps), steps))
    sp = _spectral_stage(run, tag, steps, st, say)
    say("embed")
    em = list(embed.map([tag] * len(steps), steps))
    say("done")
    return {"stats_s": [r["seconds"] for r in st], "spectral_s": [r["seconds"] for r in sp],
            "embed_s": [r["seconds"] for r in em]}


if __name__ == "__main__":
    a = sys.argv[1:]
    rec_dir = HERE / "runs"
    rec_dir.mkdir(exist_ok=True)
    if not a or a[0] == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] == "launch":
        run, tag, steps = a[1], a[2], [int(s) for s in a[3].split(",")]
        call = modal.Function.from_name(APP, "coordinate").spawn(run, tag, steps)
        rec = {"tag": tag, "run": run, "steps": steps, "call_id": call.object_id, "launched": time.ctime()}
        (rec_dir / f"measure-{tag}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
    elif a[0] == "respectral":
        rec = json.loads((rec_dir / f"measure-{a[1]}.json").read_text())
        call = modal.Function.from_name(APP, "respectral").spawn(rec["run"], a[1], rec["steps"])
        rec["call_id"] = call.object_id
        (rec_dir / f"measure-{a[1]}.json").write_text(json.dumps(rec, indent=1))
        print("respectral", call.object_id)
    elif a[0] == "status":
        rec = json.loads((rec_dir / f"measure-{a[1]}.json").read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)))
        except TimeoutError:
            try:
                print("running\n" + b"".join(out_vol.read_file(f"analysis/{a[1]}/coordinator.log")).decode())
            except Exception as e:  # noqa: BLE001
                print(f"running (no log: {e!r})")
    elif a[0] == "fetch":
        dest = HERE / "data"
        dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "cm-cifar-coherent",
                        f"analysis/{a[1]}", str(dest)], check=True)
    else:
        sys.exit(__doc__)
