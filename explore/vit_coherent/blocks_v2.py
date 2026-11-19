"""ResNet-style (schema v2) layer blocks for DINOv2 ViT-S/14, for the explorer at vit_coherent/site.

Runs the ResNet study's construction *unchanged* (explore/cifar_coherent: coherent.py, sweep_math.py,
schema_v2.block_record): the same five kernel rules, zero-diagonal symmetric scaling, fixed-lazy M = (I + P)/2,
block operators = mean of the M_l over a run of layers, SEBA at r = 2, 3, 4, 6, 8, retention in every cell.

Data: the stored float64 views LN(h_l)[CLS], l = 1..12, of the ViT run (`cm-vit-coherent:dinov2s/acts/
probe_ln_cls.npy`, PLAN.md §1), restricted to N = 2,000 probe images: 20 per ImageNet-100 class, drawn with
RandomState(0) from that class's 50 ImageNet-val images and sorted. One "checkpoint" (the released weights),
twelve sites (CLS after blocks 1..12). Labels are not used anywhere in this file except to pick the stratified
subset (as in the ResNet study) and to be copied out for display.

    uv run --with modal python -I explore/vit_coherent/blocks_v2.py run      # detached; prints the call id
    uv run --with modal python -I explore/vit_coherent/blocks_v2.py status
    uv run --with modal python -I explore/vit_coherent/blocks_v2.py fetch    # -> data/<TAG>/
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
CC = HERE.parent / "cifar_coherent"
APP = "cm-vit-blocks-v2"
TAG = "vits_b2k"
SRC = "dinov2s"
N_PER_CLASS = 20
SITES = [f"L{i}" for i in range(1, 13)]
STEPS = [0]                       # one checkpoint: the released weights
MODES = ['kNN graph, k = 15', 'per-point perplexity 100', 'per-point perplexity 300', 'kNN scale', 'half-mass, calibrated']
RANKS = [2, 3, 4, 6, 8]

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "scipy", "umap-learn")
         .env({"PYTHONPATH": "/root/cc", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}))
for name in ("coherent.py", "seba.py", "schema_v2.py", "sweep_math.py", "construction.py"):
    image = image.add_local_file(str(CC / name), remote_path="/root/cc/" + name)
app = modal.App(APP)
vol = modal.Volume.from_name("cm-vit-coherent")
VOLS = {"/out": vol}


def subset(classes):
    import numpy as np
    rng = np.random.RandomState(0)
    # probe order: class-major, within a class the 50 ImageNet-val images first (SCHEMA.md)
    idx = [c * 100 + np.sort(rng.choice(50, N_PER_CLASS, replace=False)) for c in range(100)]
    idx = np.concatenate(idx)
    assert (classes[idx] == np.repeat(np.arange(100), N_PER_CLASS)).all()
    return idx


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=32768, timeout=3 * 3600, volumes=VOLS)
def compute():
    import numpy as np
    import torch
    from coherent import centred_cosine_gram, symmetric_balance
    from sweep_math import final_kernel_bank
    from schema_v2 import allocate, pairs, block_record, put_record
    from construction import FINGERPRINT
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    root = Path("/out/analysis") / TAG
    root.mkdir(parents=True, exist_ok=True)
    src = Path("/out") / SRC
    classes = np.fromfile(src / "data/images/class.u8", np.uint8) if (src / "data/images/class.u8").exists() else None
    if classes is None:
        classes = np.repeat(np.arange(100), 100).astype(np.uint8)   # SCHEMA.md: image i has class i // 100
    idx = subset(classes)
    x = np.load(src / "acts/probe_ln_cls.npy", mmap_mode="r")     # f64 [N, 13, D], depth 0..12
    views = np.ascontiguousarray(x[idx][:, 1:13])                  # CLS after blocks 1..12
    print("views", views.shape, f"{time.time() - t0:.0f}s", flush=True)

    ops = [[None] * 12 for _ in MODES]
    info_rows, pcs = [], []
    for si in range(12):
        c = centred_cosine_gram(torch.tensor(views[:, si], device="cuda"))
        val, v = torch.linalg.eigh(c)
        pcs.append((v[:, -50:] * val[-50:].clamp_min(0).sqrt()).flip(1).cpu().numpy())
        bank = final_kernel_bank(c)
        info_rows.append([info for _, info in bank])
        for mi, (logk, info) in enumerate(bank):
            p, diag = symmetric_balance(logk, return_info=True)
            m = p * .5
            m.diagonal().add_(.5)
            info.update(sinkhorn_iterations=diag["iterations"], row_error=diag["row_error"])
            ops[mi][si] = m
        print("site", SITES[si], f"{time.time() - t0:.0f}s", flush=True)

    # layer blocks: all 78 runs of consecutive sites at the one checkpoint
    ps = pairs(12)
    arrays = allocate(len(MODES), len(ps), len(idx), 12)
    for mi in range(len(MODES)):
        for pi, (a, b) in enumerate(ps):
            rec, _ = block_record(ops[mi], a, b)
            put_record(arrays, mi, pi, rec)
        print("layer blocks, rule", mi, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(root / "layer_0.npz", **arrays)
    # time blocks: with one checkpoint each is the single cell itself (kept so the schema is complete)
    for si, site in enumerate(SITES):
        t_arr = allocate(len(MODES), 1, len(idx), 1)
        for mi in range(len(MODES)):
            rec, _ = block_record([ops[mi][si]], 0, 0)
            put_record(t_arr, mi, 0, rec)
        np.savez_compressed(root / f"time_{site}.npz", **t_arr)
    print("time blocks", f"{time.time() - t0:.0f}s", flush=True)

    # aligned UMAP across sites (same call as the ResNet study's layouts)
    import umap
    model = umap.AlignedUMAP(n_neighbors=15, min_dist=.1, random_state=0, alignment_regularisation=.02)
    model.fit(pcs, relations=[{i: i for i in range(len(idx))} for _ in range(11)])
    np.save(root / "umap_0.npy", np.stack(model.embeddings_).astype("f4"))
    print("umap", f"{time.time() - t0:.0f}s", flush=True)

    field = lambda k: [[[info_rows[si][mi][k] for si in range(12)]] for mi in range(len(MODES))]  # (M, T=1, S)
    meta = dict(schema_version=2, construction_fingerprint=FINGERPRINT, model="dinov2_vits14", source_tag=SRC,
                view="LN(h_l)[CLS] (the model's own readout of the CLS token after block l), probe-centred and "
                     "row-normalised over the 2,000 images",
                steps=STEPS, step_labels=["released weights"], sites=SITES, modes=MODES,
                kappa=field("kappa"), nn_mass=field("nn_mass"), self_weight=field("self_weight"),
                kappa_q10=field("kappa_q10"), kappa_q90=field("kappa_q90"),
                sinkhorn_iterations=field("sinkhorn_iterations"), row_error=field("row_error"), default_mode=0,
                seba_r=RANKS, n_vecs=8, layer_pairs=[list(p) for p in ps], time_pairs=[[0, 0]],
                n=int(len(idx)), probe_idx=idx.tolist(),
                subset=f"{N_PER_CLASS} per class from each class's 50 ImageNet-val images (RandomState(0)), sorted",
                seconds=time.time() - t0)
    (root / "meta.json").write_text(json.dumps(meta, indent=1, allow_nan=False))
    vol.commit()
    return dict(seconds=time.time() - t0, tag=TAG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    rec = HERE / "runs" / f"{TAG}.json"
    if cmd == "run":
        with modal.enable_output():
            subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
        call = modal.Function.from_name(APP, "compute").spawn()
        rec.write_text(json.dumps(dict(call_id=call.object_id, launched=time.ctime()), indent=1))
        print(call.object_id)
    elif cmd == "status":
        cid = json.loads(rec.read_text())["call_id"]
        try:
            print(modal.FunctionCall.from_id(cid).get(timeout=0))
        except TimeoutError:
            print("running")
    elif cmd == "fetch":
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "cm-vit-coherent",
                        f"analysis/{TAG}", str(HERE / "data")], check=True)
    else:
        raise SystemExit(__doc__)
