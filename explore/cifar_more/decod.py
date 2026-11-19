"""Final-checkpoint linear decodability: ../cifar_coherent/decodability.py's score_view with the class count
parameterised (validation fit only; never part of the construction). Fit diagnostics (median distances) dropped."""
from __future__ import annotations

PROTOCOL = ("{n_fit:,} train images, balanced, numpy RandomState(0) per class, sorted; all 10,000 test images. Eval "
            "mode, model input normalization, no augmentation. Views: flat (flattened maps) and gap (spatial channel "
            "means; conv sites only). Fit-mean centring then row L2 normalization (fit mean reused for test). "
            "Closed-form ridge on one-hot targets, no intercept, ||XW-Y||^2 + lambda||W||^2; fit-only multi-output GCV "
            "(RSS/(n*C))/(1-tr(H)/n)^2; float64 eigensolve, dual when d>n; 181-point log scan of lambda plus bounded "
            "refinement of every interior local minimum. Weighted cosine kNN, k=20, votes exp(cos/0.07). No test "
            "labels in fitting or lambda selection.")


def score_view(fit, test, labels, test_labels, n_classes):
    import numpy as np
    import torch
    from scipy.optimize import minimize_scalar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = fit.to(device).float()
    z = test.to(device).float()
    mean = x.mean(0)
    x = torch.nn.functional.normalize(x - mean, dim=1)
    z = torch.nn.functional.normalize(z - mean, dim=1)
    n, d = x.shape
    lab = labels.to(device)
    y = torch.nn.functional.one_hot(lab, n_classes).double()
    knn_correct = 0
    for start in range(0, len(z), 256):
        val, idx = (z[start:start + 256] @ x.T).topk(20, dim=1)
        votes = torch.zeros((len(val), n_classes), device=device)
        votes.scatter_add_(1, lab[idx], (val / 0.07).exp())
        knn_correct += int((votes.argmax(1).cpu() == test_labels[start:start + 256]).sum())
    xd = x.double()
    a = xd @ xd.T if d > n else xd.T @ xd
    eig, u = torch.linalg.eigh(a)
    del a
    eig = eig.clamp_min(0)
    if d > n:
        uy = u.T @ y
        energy = uy.square().sum(1)
        null = 0.0
    else:
        projected = u.T @ (xd.T @ y)
        positive = eig > eig.max() * 1e-12
        energy = torch.where(positive, projected.square().sum(1) / eig.clamp_min(1e-30), 0)
        null = float((y.square().sum() - energy.sum()).clamp_min(0))
    ev, en = eig.cpu().numpy(), energy.cpu().numpy()

    def gcv(log_lambda):
        lam = np.exp(log_lambda)
        shrink = ev / (ev + lam)
        rss = null + np.sum(en * (lam / (ev + lam)) ** 2)
        return float((rss / (n * n_classes)) / (1 - shrink.sum() / n) ** 2)

    top = np.log(max(float(ev.max()), 1.0))
    grid = np.linspace(top - 24, top + 12, 181)
    values = np.array([gcv(v) for v in grid])
    candidates = [(values[0], grid[0]), (values[-1], grid[-1])]
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            opt = minimize_scalar(gcv, bounds=(grid[i - 1], grid[i + 1]), method="bounded")
            candidates.append((opt.fun, opt.x))
    best, log_lam = min(candidates)
    lam = float(np.exp(log_lam))
    weights = xd.T @ (u @ (uy / (eig + lam)[:, None])) if d > n else u @ (projected / (eig + lam)[:, None])
    correct = 0
    for start in range(0, len(z), 256):
        pred = (z[start:start + 256].double() @ weights).argmax(1).cpu()
        correct += int((pred == test_labels[start:start + 256]).sum())
    return {"ridge_acc": correct / len(z), "knn_acc": knn_correct / len(z), "lambda": lam, "feature_dim": d,
            "gcv": float(best), "lambda_at_search_boundary": bool(log_lam in (grid[0], grid[-1]))}
