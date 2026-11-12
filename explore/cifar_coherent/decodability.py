"""Final-checkpoint CIFAR-10 supervised decodability (exploratory, not a construction).

Run: uv run --with modal python explore/cifar_coherent/decodability.py
A detached Modal invocation writes /analysis/r20s0/decodability.json; the local
launcher waits for that invocation and fetches its result. No existing files change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SITES = ["stem"] + [f"b{i}" for i in range(1, 10)] + ["pool", "logits"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scipy")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(str(Path.home() / "code/ModelZoo/src"), remote_path="/root/src")
)
app = modal.App("cm-cifar-decodability")
zoo_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name("cm-cifar-coherent")


def score_view(fit, test, labels, test_labels):
    """Exact ridge/GCV and neighbours; all distances exclude the diagonal."""
    import numpy as np
    import torch
    from scipy.optimize import minimize_scalar

    device = "cuda"
    x = fit.to(device)
    z = test.to(device)
    mean = x.mean(0)
    x = torch.nn.functional.normalize(x - mean, dim=1)
    z = torch.nn.functional.normalize(z - mean, dim=1)
    n, d = x.shape
    y = torch.nn.functional.one_hot(labels.to(device), 10).double()
    gram = x @ x.T
    # All ordered off-diagonal pairs duplicate the unordered-pair multiset,
    # so its usual (midpoint for even N) median equals the unordered median.
    gram.diagonal().fill_(float("-inf"))
    near15 = 1 - gram.topk(15, dim=1).values[:, -1]
    knn15_dist = float(near15.quantile(0.5))
    gram.diagonal().fill_(float("nan"))
    # NumPy partitions in-place: exact 99,990,000-pair midpoint median, no sampling.
    distances = (1 - gram).cpu().numpy().ravel()
    distances = distances[np.isfinite(distances)]
    pair_dist = float(np.median(distances, overwrite_input=True))
    del distances
    gram.diagonal().fill_(1.0)
    knn_correct = 0
    for start in range(0, len(z), 256):
        similarities = z[start:start + 256] @ x.T
        val, idx = similarities.topk(20, dim=1)
        votes = torch.zeros((len(val), 10), device=device)
        votes.scatter_add_(1, labels.to(device)[idx], (val / 0.07).exp())
        knn_correct += int((votes.argmax(1).cpu() == test_labels[start:start + 256]).sum())
    # Double-precision eigensystem, using the smaller of X X^T and X^T X.
    # Lambda convention: ||X W - Y||_F^2 + lambda ||W||_F^2, no intercept.
    if d > n:
        # Recompute in float64 rather than promoting a rounded float32 Gram.
        del gram
        a = x.double() @ x.double().T
    else:
        del gram
        a = x.double().T @ x.double()
    eig, u = torch.linalg.eigh(a)
    del a
    eig = eig.clamp_min(0)
    if d > n:
        uy = u.T @ y
        energy = uy.square().sum(1)
        null_energy = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        xty = x.double().T @ y
        projected = u.T @ xty
        positive = eig > eig.max() * 1e-12
        energy = torch.where(positive, projected.square().sum(1) / eig.clamp_min(1e-30), 0)
        null_energy = (y.square().sum() - energy.sum()).clamp_min(0)
    ev = eig.cpu().numpy()
    en = energy.cpu().numpy()
    null = float(null_energy)

    def gcv(log_lambda):
        lam = np.exp(log_lambda)
        shrink = ev / (ev + lam)
        rss = null + np.sum(en * (lam / (ev + lam)) ** 2)
        return float((rss / (n * 10)) / (1 - shrink.sum() / n) ** 2)

    # Global logarithmic scan, then refine every sampled local minimum.
    grid = np.linspace(np.log(max(float(ev.max()), 1.0)) - 24,
                       np.log(max(float(ev.max()), 1.0)) + 12, 181)
    values = np.array([gcv(v) for v in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            opt = minimize_scalar(gcv, bounds=(grid[i - 1], grid[i + 1]), method="bounded")
            candidates.append((opt.fun, opt.x))
    best_gcv, log_lam = min(candidates)
    lam = float(np.exp(log_lam))
    if d > n:
        dual = u @ (uy / (eig + lam)[:, None])
        weights = x.double().T @ dual
    else:
        weights = u @ (projected / (eig + lam)[:, None])
    correct = 0
    for start in range(0, len(z), 256):
        pred = (z[start:start + 256].double() @ weights).argmax(1).cpu()
        correct += int((pred == test_labels[start:start + 256]).sum())
    return {"ridge_acc": correct / len(z), "knn_acc": knn_correct / len(z),
            "knn15_dist": knn15_dist, "pair_dist": pair_dist,
            "lambda": lam, "feature_dim": d, "gcv": float(best_gcv),
            "ridge_form": "dual" if d > n else "primal",
            "lambda_at_search_boundary": log_lam in (grid[0], grid[-1])}


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, timeout=1250,
              retries=0, volumes={"/zoo": zoo_vol, "/out": out_vol})
def measure(step: int = 64000, tag: str = 'r20s0', run: str = 'resnet20-relu-seed0'):
    import numpy as np
    import torch
    from zoo.data import build_data
    from zoo.io import load_checkpoint
    from zoo.models import build_model

    t0 = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    ck = load_checkpoint(Path(f"/out/runs/{run}/checkpoints/step_{step}.pt"))
    model = build_model(ck["config"])
    model.load_state_dict(ck["model_state"])
    model.eval().cuda()
    data = build_data(ck["config"], Path("/zoo/data"), "cpu")
    rng = np.random.RandomState(0)
    fit_idx = np.sort(np.concatenate([
        rng.choice(np.where(data.train_y.numpy() == c)[0], 1000, replace=False)
        for c in range(10)]))
    splits = [(data.train_x[fit_idx], data.train_y[fit_idx]), (data.test_x, data.test_y)]
    # Store on host; only one view's full fit/test tensors occupy GPU at a time.
    activations = []
    for images, _ in splits:
        storage = None
        for start in range(0, len(images), 128):
            with torch.inference_mode():
                h = model.act(model.bn1(model.conv1(data.normalize(images[start:start + 128]).cuda())))
                acts = [h]
                for block in model.blocks:
                    h = block(h)
                    acts.append(h)
                pooled = h.mean((2, 3))
                acts += [pooled, model.fc(pooled)]
                if storage is None:
                    storage = [(torch.empty((len(images), a.flatten(1).shape[1])),
                                torch.empty((len(images), a.shape[1])) if a.ndim == 4 else None)
                               for a in acts]
                for (flat, gap), a in zip(storage, acts):
                    flat[start:start + len(a)] = a.flatten(1).cpu()
                    if gap is not None:
                        gap[start:start + len(a)] = a.mean((2, 3)).cpu()
        activations.append(storage)
    result = {"sites": SITES, "views": {"flat": {}, "gap": {}},
              "step": step, "fit_indices": fit_idx.tolist(),
              "protocol": "CIFAR-10: 10,000 train images, 1,000/class sampled with numpy RandomState(0), sorted indices; all 10,000 test images. Eval mode, standard model input normalization, no augmentation. Conv sites: flattened maps and spatial channel means; pool/logits appear once under flat (gap omits them). Each view subtracts fit mean then L2-normalizes rows, with the same fit mean used for test. Closed-form ridge on one-hot targets, no intercept, objective ||XW-Y||_F^2 + lambda||W||_F^2. Fit-only multi-output GCV = (RSS/(n*10))/(1-tr(H)/n)^2, float64 eigensolve, dual when d>n. Lambda: 181-point logarithmic scan from max(eigmax,1)*exp(-24) to max(eigmax,1)*exp(12), bounded scalar refinement of all sampled interior minima; boundaries checked. Weighted cosine kNN k=20, votes exp(cos/0.07). Fit diagnostics: exact median distance to 15th nonself neighbour and exact median over all nonself pairs; cosine distances in float32, midpoint medians. No test labels used in fitting or lambda selection."}
    root = Path(f"/out/analysis/{tag}")
    filename = 'decodability' if tag == 'r20s0' and step == 64000 else f'decodability_{step}'
    root.mkdir(parents=True, exist_ok=True)
    for i, site in enumerate(SITES):
        for j, view in enumerate(("flat", "gap") if i < 10 else ("flat",)):
            start = time.time()
            row = score_view(activations[0][i][j], activations[1][i][j], splits[0][1], splits[1][1])
            result["views"][view][site] = row
            print(site, view, json.dumps(row), "seconds", round(time.time() - start, 1), flush=True)
            # Durable progress, separate from final completion artifact.
            (root / f"{filename}_partial.json").write_text(json.dumps(result, indent=2))
            out_vol.commit()
        activations[0][i] = activations[1][i] = None
    result["seconds"] = time.time() - t0
    result["complete"] = True
    (root / f"{filename}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    out_vol.commit()
    return result


if __name__ == "__main__":
    with app.run(detach=True):
        call = measure.spawn()
        print("call_id", call.object_id, flush=True)
        result = call.get()
    dest = HERE / "data/r20s0/decodability.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False))
    print("Saved", dest, "runtime", result["seconds"], flush=True)
