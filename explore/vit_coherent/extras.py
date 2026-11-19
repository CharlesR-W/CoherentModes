"""Post-run label-free diagnostic (added after the selection, see DEVIATIONS.md item 22): how close is
each kernel's top-r eigenspace to the view's top-r principal components (PCA, i.e. the beta -> 0 limit)?
Writes `pca_overlap` into data/<tag>/selection.json on the Volume.

    uv run --with modal python -I explore/vit_coherent/extras.py <tag>
"""
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy<2.3")
         .add_local_file(str(HERE / "analysis.py"), "/root/vc/analysis.py"))
app = modal.App("cm-vit-coherent-extras")
vol = modal.Volume.from_name("cm-vit-coherent")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/out": vol})
def pca_compare(tag: str) -> dict:
    import json

    import numpy as np
    root = Path("/out") / tag
    sel_p = root / "data/selection.json"
    sel = json.loads(sel_p.read_text())
    meta = json.loads((root / "data/meta.json").read_text())
    out = {"definition": "principal cosines between the span of a kernel's top-r nonconstant eigenvectors of P_l "
                         "and the span of the view's top-r principal-component scores (left singular vectors of "
                         "the centred, normalised probe view), r = 8 and 16; mean reported, cosines for r = 16"}
    for v, L in (("cls", range(1, 13)), ("mp", range(0, 13))):
        p = np.load(root / f"acts/probe_ln_{v}.npy", mmap_mode="r")
        out[v] = {}
        pcs = []
        for l in L:
            a = np.asarray(p[:, l], np.float64)
            a = a - a.mean(0)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            u, s, _ = np.linalg.svd(a - a.mean(0), full_matrices=False)
            pcs.append(u[:, :16])
        for k in meta["kernel_ids"]:
            rows = []
            for li, l in enumerate(L):
                f = root / "full" / v / k / "modes" / f"L{l:02d}.f16"
                m = np.fromfile(f, np.float16).astype(np.float64).reshape(64, -1).T
                rec = {"layer": l}
                for r in (8, 16):
                    q, _ = np.linalg.qr(m[:, :r])
                    c = np.linalg.svd(q.T @ pcs[li][:, :r], compute_uv=False)
                    rec[f"mean{r}"] = float(c.mean())
                    if r == 16:
                        rec["cos16"] = c.tolist()
                rows.append(rec)
            out[v][k] = rows
    sel["pca_overlap"] = out
    sel_p.write_text(json.dumps(sel, allow_nan=True))
    vol.commit()
    return {v: {k: [round(r["mean16"], 2) for r in out[v][k]] for k in meta["kernel_ids"]} for v in ("cls", "mp")}


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        res = pca_compare.remote(sys.argv[1])
    for v, d in res.items():
        for k, xs in d.items():
            print(v, k, xs)
