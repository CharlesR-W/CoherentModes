"""Build the coherent-sets explorer from fetched measurement data.

    uv run --with numpy --with pillow python explore/cifar_coherent/build_viewer.py <tag>
    -> explore/cifar_coherent/viewer/<tag>/index.html (+ chunks/*.js, loaded lazily; works from file://)
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def procrustes_chain(emb: np.ndarray) -> np.ndarray:
    """emb (T, S, N, 2): per site, centre/scale each frame and rotate/reflect onto the previous checkpoint."""
    out = emb.astype(np.float64).copy()
    out -= out.mean(2, keepdims=True)
    out /= np.sqrt((out ** 2).sum((2, 3), keepdims=True) / out.shape[2])
    for s in range(out.shape[1]):
        for t in range(1, out.shape[0]):
            u, _, vt = np.linalg.svd(out[t, s].T @ out[t - 1, s])
            out[t, s] = out[t, s] @ (u @ vt)
    # also align sites within the first checkpoint row-chain is already aligned by AlignedUMAP
    return out


def main(tag: str) -> None:
    src = HERE / "data" / tag
    dst = HERE / "viewer" / tag
    (dst / "chunks").mkdir(parents=True, exist_ok=True)
    kap = json.loads((src / "kappa.json").read_text())
    steps, sites, modes = kap["steps"], kap["sites"], kap["modes"]
    probe = np.load(src / "probe.npz")
    labels = probe["labels"].astype(np.int8)
    n = len(labels)
    stats = [json.loads((src / f"stats_{s}.json").read_text()) for s in steps]
    emb = np.stack([np.load(src / f"umap_{s}.npy") for s in steps])  # T,S,N,2
    emb = procrustes_chain(emb)
    lo, hi = np.percentile(emb, 0.5, axis=2, keepdims=True), np.percentile(emb, 99.5, axis=2, keepdims=True)
    q = np.clip((emb - lo) / (hi - lo + 1e-12), 0, 1)
    q16 = np.round(q * 65535).astype(np.uint16)

    imgs = probe["images"]  # N,32,32,3
    cols = 50
    rows = int(np.ceil(n / cols))
    sheet = np.zeros((rows * 32, cols * 32, 3), np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = imgs[i]
    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="PNG", optimize=True)

    spec_vals = {}
    for s in steps:
        z = np.load(src / f"spec_{s}.npz")
        spec_vals[s] = z["vals"].round(5).tolist()  # M,P,K
        for mi in range(len(modes)):
            payload = {"vecs": b64(z["vecs"][mi]), "seba": b64(z["seba"][mi])}
            (dst / "chunks" / f"spec_{s}_{mi}.js").write_text(
                f"window.__chunk({json.dumps(f'{s}_{mi}')},{json.dumps(payload)});")

    meta = {
        "tag": tag, "steps": steps, "sites": sites, "modes": modes, "pairs": kap["pairs"],
        "n": n, "classes": CLASSES, "labels": b64(labels), "umap": b64(q16),
        "seba_r": kap["seba_r"], "n_modes": kap["n_modes"], "kappa": kap["kappa"],
        "rule": kap["rule"], "alpha": kap["alpha"],
        "vals": spec_vals, "stats": stats,
        "sprite": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "sprite_cols": cols,
    }
    (dst / "meta.js").write_text("window.META=" + json.dumps(meta) + ";")
    (dst / "index.html").write_text((HERE / "viewer_template.html").read_text())
    tot = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"wrote {dst} ({tot / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
