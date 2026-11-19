"""Modal pipeline for the ViT side study (PLAN.md §§1-5, 8, 9).  Detached coordinator; results on Volume
`cm-vit-coherent` under /<tag>/; `fetch` copies /<tag>/data to explore/vit_coherent/data/<tag>/.

    uv run --with modal python -I explore/vit_coherent/pipeline.py deploy
    uv run --with modal python -I explore/vit_coherent/pipeline.py launch <tag> [smoke]
    uv run --with modal python -I explore/vit_coherent/pipeline.py status <tag>
    uv run --with modal python -I explore/vit_coherent/pipeline.py fetch <tag>
    uv run --with modal python -I explore/vit_coherent/pipeline.py verify <tag>     # local, against SCHEMA.md

Stages (coordinator): prepare (CPU: dataset to Volume, sampling plan) -> activations (GPU) ->
acts_products (CPU) -> spectral x (view, kernel) (GPU) || decode (GPU) || embed x view (CPU) ->
select (label-free rule) -> deep x default kernel (GPU: all-pair Delta, controls, Q, §4) -> assemble.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "cm-vit-coherent"
HERE = Path(__file__).resolve().parent
VOL = "cm-vit-coherent"
DATASET = "clane9/imagenet-100"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2.3", "pyarrow", "pillow", "huggingface_hub",
                 "umap-learn", "scipy", "safetensors", "transformers==4.46.3")
    .run_commands("python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)\"",
                  "python -c \"from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-small', "
                  "local_dir='/root/hf_dinov2s', allow_patterns=['*.json', '*.safetensors'])\"")
    .env({"PYTHONPATH": "/root/vc"})
    .add_local_file(str(HERE / "ops.py"), "/root/vc/ops.py")
    .add_local_file(str(HERE / "acts.py"), "/root/vc/acts.py")
    .add_local_file(str(HERE / "analysis.py"), "/root/vc/analysis.py")
    .add_local_file(str(HERE / "classes.json"), "/root/vc/classes.json")
)
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
VOLS = {"/out": vol}
GPU = "H100"


def _root(tag: str) -> Path:
    return Path("/out") / tag


def _logger(tag: str):
    root = _root(tag)
    root.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        with open(root / "coordinator.log", "a") as f:
            f.write(time.strftime("%H:%M:%S ") + m + "\n")
        vol.commit()
    return say


def _slog(tag: str, name: str):
    """Per-stage log file (committed occasionally by the stage itself)."""
    root = _root(tag)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    p = root / "logs" / f"{name}.log"

    def say(m: str) -> None:
        with open(p, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + m + "\n")
        print(m, flush=True)
    return say


# ------------------------------------------------------------------------------------- stages

@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes=VOLS)
def prepare(tag: str, smoke: bool) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    hf = Path("/out/hf")
    if not (hf / "done").exists():
        snapshot_download(DATASET, repo_type="dataset", local_dir=str(hf), allow_patterns=["data/*.parquet"])
        (hf / "done").write_text("ok")
        vol.commit()
    cls = json.loads(Path("/root/vc/classes.json").read_text())
    ds2id = {c["dataset_label"]: c["id"] for c in cls["classes"]}
    sup = {c["id"]: c["superclass"] for c in cls["classes"]}
    files = sorted(p.name for p in (hf / "data").glob("*.parquet"))
    rows = {"validation": {}, "train": {}}
    for f in files:
        split = "validation" if f.startswith("validation") else "train"
        lab = pq.read_table(hf / "data" / f, columns=["label"]).column("label").to_numpy()
        for r, y in enumerate(lab):
            rows[split].setdefault(ds2id[int(y)], []).append([f, r])
    n_val, n_tr, n_ho, n_attn = (3, 2, 5, 1) if smoke else (50, 50, 100, 15)
    rng = np.random.default_rng(0)
    probe, ho, pc, hc, src = [], [], [], [], []
    for c in range(100):
        v = rows["validation"][c][:n_val]                         # all 50 val images, dataset order
        tr = rows["train"][c]
        pick = rng.choice(len(tr), n_tr + n_ho, replace=False)
        ptr = sorted(pick[:n_tr].tolist())
        htr = sorted(pick[n_tr:].tolist())
        probe += v + [tr[i] for i in ptr]
        src += [0] * len(v) + [1] * n_tr
        pc += [c] * (len(v) + n_tr)
        ho += [tr[i] for i in htr]
        hc += [c] * n_ho
    N = len(probe)
    perm = rng.permutation(N)
    half = np.zeros(N, int)
    half[perm[N // 2:]] = 1
    attn = sorted(int(c * (n_val + n_tr) + j) for c in range(100)
                  for j in rng.choice(n_val + n_tr, n_attn, replace=False))
    plan = {"probe": probe, "heldout": ho, "probe_class": pc, "heldout_class": hc,
            "probe_super": [sup[c] for c in pc], "heldout_super": [sup[c] for c in hc],
            "source": src, "half": half.tolist(), "attn_idx": attn, "smoke": smoke,
            "n_per_class": {"val": n_val, "train": n_tr, "heldout": n_ho, "attn": n_attn}}
    root = _root(tag)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps(plan))
    vol.commit()
    return {"N": N, "N_heldout": len(ho), "files": len(files)}


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, timeout=3 * 3600, volumes=VOLS)
def activations(tag: str) -> dict:
    import torch
    from acts import run
    assert torch.cuda.is_available()
    root = _root(tag)
    (root / "acts").mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    res = run(root / "acts", Path("/out/hf/data"), plan)
    vol.commit()
    return res


@app.function(image=image, cpu=4, memory=32768, timeout=3600, volumes=VOLS)
def acts_products(tag: str) -> dict:
    """Kernel-independent data products: images/, thumbs/, cls/neurons, cls/heads, attn/, weights/."""
    import io

    import numpy as np
    from PIL import Image

    from acts import head_outputs, neuron_dirs
    from analysis import lname, save

    vol.reload()
    root = _root(tag)
    data = root / "data"
    plan = json.loads((root / "plan.json").read_text())
    N = len(plan["probe"])
    save(data / "images/class.u8", plan["probe_class"], np.uint8)
    save(data / "images/superclass.u8", plan["probe_super"], np.uint8)
    save(data / "images/source.u8", plan["source"], np.uint8)
    save(data / "images/half.u8", plan["half"], np.uint8)
    th = np.load(root / "acts/thumbs64.npy")
    (data / "thumbs").mkdir(parents=True, exist_ok=True)
    for s in range((N + 99) // 100):
        sheet = np.zeros((640, 640, 3), np.uint8)
        for j in range(min(100, N - 100 * s)):
            r, c = divmod(j, 10)
            sheet[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = th[100 * s + j]
        buf = io.BytesIO()
        Image.fromarray(sheet).save(buf, format="JPEG", quality=85)
        (data / "thumbs" / f"sheet_{s:03d}.jpg").write_bytes(buf.getvalue())
    W = dict(np.load(root / "acts/weights.npz"))
    G = np.load(root / "acts/probe_neurons.npy", mmap_mode="r")
    rng_, st_ = np.zeros((12, 1536, 2), np.float32), np.zeros((12, 1536, 2), np.float32)
    for l in range(12):
        g = np.asarray(G[:, l], np.float64)
        lo, hi = g.min(0), g.max(0)
        rng_[l, :, 0], rng_[l, :, 1] = lo, hi
        st_[l, :, 0], st_[l, :, 1] = g.mean(0), g.std(0)
        q = np.round(255 * (g - lo) / np.maximum(hi - lo, 1e-30))
        save(data / "cls/neurons" / f"{lname(l + 1)}.u8", q, np.uint8)
    save(data / "cls/neurons/range.f32", rng_, np.float32)
    save(data / "cls/neurons/stats.f32", st_, np.float32)
    A = head_outputs(np.load(root / "acts/probe_heads.npy"), W)                 # [N,12,6,D]
    nrm = np.linalg.norm(A, axis=-1)
    for l in range(12):
        save(data / "cls/heads" / f"{lname(l + 1)}.f16", nrm[:, l].T, np.float16)
    del A
    at = np.load(root / "acts/attn.npy")                                       # [n, 12, 6, 257]
    idx = np.load(root / "acts/attn_idx.npy")
    save(data / "attn/idx.i16", idx, np.int16)
    patches = at[..., 1:]
    mx = patches.max(-1)
    for l in range(12):
        save(data / "attn" / f"{lname(l + 1)}.u8", np.round(255 * patches[:, l] / mx[:, l, :, None]), np.uint8)
    save(data / "attn/max.f32", np.stack([mx, at[..., 0]], -1).transpose(1, 0, 2, 3), np.float32)
    save(data / "weights/wout_ls.f16", neuron_dirs(W), np.float16)
    vol.commit()
    return {"sheets": (N + 99) // 100, "n_attn": int(len(idx))}


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, timeout=3 * 3600, volumes=VOLS)
def spectral(tag: str, view: str, k: str, seba: bool = False) -> dict:
    import torch

    import analysis as an
    assert torch.cuda.is_available()
    vol.reload()
    t0 = time.time()
    root = _root(tag)
    say = _slog(tag, f"spectral_{view}_{k}")
    plan = an.load_plan(root)
    E, EA, EB, labels = an.section2(root, view, k, plan, say, write_sel=not seba, do_seba=seba)
    t2 = time.time() - t0
    vol.commit()
    an.section3(root, view, k, E, EA, EB, [p for p in an.DECLARED if p in an.all_pairs(view)], say)
    vol.commit()
    return {"view": view, "k": k, "s2_s": round(t2, 1), "total_s": round(time.time() - t0, 1)}


@app.function(image=image, gpu=GPU, cpu=4, memory=49152, timeout=4 * 3600, volumes=VOLS)
def deep(tag: str, k: str) -> dict:
    import torch

    import analysis as an
    assert torch.cuda.is_available()
    vol.reload()
    t0 = time.time()
    root = _root(tag)
    say = _slog(tag, f"deep_{k}")
    plan = an.load_plan(root)
    E, EA, EB, labels = an.section2(root, "cls", k, plan, say, write_sel=False, do_seba=True)
    vecs = an.section3(root, "cls", k, E, EA, EB, an.all_pairs("cls"), say, keep_vecs=True)
    t3 = time.time() - t0
    vol.commit()
    dj = json.loads((root / "full/cls" / k / "delta.json").read_text())
    xs, _ = an.load_views(root, "cls")
    ctrl = an.controls("cls", k, xs, E, dj["results"], say)
    q = an.q_ops("cls", k, E)
    unc = an.uncentred(root, "cls", k, E)
    an.jdump(root / "work" / f"deep_{k}.json", {"controls": ctrl, "q": q, "uncentred": unc})
    vol.commit()
    s4 = an.section4(root, k, E, vecs, labels, say)
    vol.commit()
    return {"k": k, "s3_s": round(t3, 1), "total_s": round(time.time() - t0, 1),
            "attr_sum_rel_err_max": s4["sum_rel_err_max"]}


@app.function(image=image, gpu=GPU, cpu=4, memory=16384, timeout=3600, volumes=VOLS)
def decode(tag: str) -> dict:
    import analysis as an
    vol.reload()
    root = _root(tag)
    res = an.decodability(root, an.load_plan(root))
    an.jdump(root / "work/decodability.json", res)
    vol.commit()
    return {v: {m: res[v][m]["fine"][-1] for m in ("ridge", "knn")} for v in ("cls", "mp")}


@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes=VOLS)
def embed(tag: str, view: str) -> dict:
    import numpy as np

    import analysis as an
    vol.reload()
    t0 = time.time()
    root = _root(tag)
    emb = an.umap_view(root, view)
    an.save(root / "data" / view / "umap.f32", emb, np.float32)
    vol.commit()
    return {"view": view, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------- assemble

def expected_files(meta: dict) -> dict:
    """Every binary file SCHEMA.md promises -> (dtype, shape). Also used by `verify`."""
    N, NA = meta["N"], meta["n_attn"]
    out = {}
    f = lambda p, dt, sh: out.__setitem__(p, (dt, list(sh)))  # noqa: E731
    f("images/class.u8", "u8", [N]); f("images/superclass.u8", "u8", [N])
    f("images/source.u8", "u8", [N]); f("images/half.u8", "u8", [N])
    f("weights/wout_ls.f16", "f16", [12, 1536, 384])
    f("attn/idx.i16", "i16", [NA]); f("attn/max.f32", "f32", [12, NA, 6, 2])
    for l in range(1, 13):
        f(f"attn/L{l:02d}.u8", "u8", [NA, 6, 256])
        f(f"cls/neurons/L{l:02d}.u8", "u8", [N, 1536])
        f(f"cls/heads/L{l:02d}.f16", "f16", [6, N])
    f("cls/neurons/range.f32", "f32", [12, 1536, 2]); f("cls/neurons/stats.f32", "f32", [12, 1536, 2])
    full = set(meta["kernels_full"])
    for v, L in (("cls", list(range(1, 13))), ("mp", list(range(0, 13)))):
        nL = len(L)
        f(f"{v}/umap.f32", "f32", [nL, N, 2])
        pairs = [(a, b) for i, a in enumerate(L) for b in L[i + 1:]]
        for kk in meta["kernel_ids"]:
            p = f"{v}/{kk}/"
            f(p + "spectrum.f32", "f32", [nL, N]); f(p + "spectrum_alpha1.f32", "f32", [nL, 65])
            f(p + "evals.f32", "f32", [nL, 64])
            if v == "cls" or kk in full:
                f(p + "smooth.f32", "f32", [nL, 64, nL, 4])
                for l in L:
                    f(p + f"modes/L{l:02d}.f16", "f16", [64, N])
            if kk in full:
                for l in L:
                    f(p + f"seba/L{l:02d}.u8", "u8", [120, N]); f(p + f"seba/L{l:02d}_hard.i8", "i8", [4, N])
                for r in (8, 16, 32, 64):
                    f(p + f"flows/counts_r{r:02d}.u16", "u16", [len(pairs), r + 1, r + 1])
            dpairs = meta["delta_pairs"][v].get(kk, [])
            for a, b in dpairs:
                f(p + f"delta/a{a:02d}_b{b:02d}.f16", "f16", [4, 16, N])
            if v == "cls" and kk in full:
                for l in L:
                    f(p + f"load/L{l:02d}.f16", "f16", [64, 1536]); f(p + f"auroc/L{l:02d}.f16", "f16", [120, 1536])
                    f(p + f"attr/L{l:02d}.f32", "f32", [16, 1 + 72 + 12 * 1536])
                    f(p + f"writecos/L{l:02d}.f16", "f16", [16, 12, 1536])
                for a, b in dpairs:
                    f(p + f"delta_attr/a{a:02d}_b{b:02d}.f32", "f32", [4, 16, 480])
                    f(p + f"delta_attr/a{a:02d}_b{b:02d}_top.f32", "f32", [4, 16, 32, 2])
    return out


@app.function(image=image, cpu=4, memory=32768, timeout=3600, volumes=VOLS)
def assemble(tag: str, defaults: list, sel: dict) -> dict:
    import shutil

    import numpy as np

    import analysis as an
    vol.reload()
    root = _root(tag)
    data, full = root / "data", root / "full"
    plan = an.load_plan(root)
    cls = json.loads(Path("/root/vc/classes.json").read_text())
    kids = list(an.KERNELS)
    dpairs = {"cls": {}, "mp": {}}

    def cp(rel: str):
        src, dst = full / rel, data / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    for v in ("cls", "mp"):
        for k in kids:
            p = f"{v}/{k}/"
            for nm in ("spectrum.f32", "spectrum_alpha1.f32", "evals.f32"):
                cp(p + nm)
            if v == "cls" or k in defaults:
                cp(p + "smooth.f32"); cp(p + "modes")
            if k in defaults:
                for nm in ("seba", "seba.json", "flows", "flows.json"):
                    cp(p + nm)
            if v == "cls" or k in defaults:
                dj = json.loads((full / p / "delta.json").read_text())
                if v == "cls" and k in defaults:
                    dj = _merge_delta_attr(dj, full / p / "attr.json")
                    for nm in ("load", "auroc", "attr", "writecos", "delta_attr", "attr.json"):
                        cp(p + nm)
                    deepj = json.loads((root / "work" / f"deep_{k}.json").read_text())
                    an.jdump(data / p / "q.json", deepj["q"])
                dpairs[v][k] = dj["pairs"]
                for a, b in dj["pairs"]:
                    cp(p + f"delta/{an.pname(a, b)}.f16")
                an.jdump(data / p / "delta.json", dj)
    # JSON summaries
    frag = {v: {k: json.loads((root / "work" / f"sel_{v}_{k}.json").read_text()) for k in kids} for v in ("cls", "mp")}
    selection = {"views": {v: {k: {"per_layer": frag[v][k]["per_layer"]} for k in kids} for v in frag},
                 "validation": {v: {k: frag[v][k]["validation"] for k in kids} for v in frag},
                 "uncentred_sensitivity": {k: json.loads((root / "work" / f"deep_{k}.json").read_text())["uncentred"]
                                           for k in defaults},
                 **sel}
    an.jdump(data / "selection.json", selection)
    an.jdump(data / "controls.json", {k: json.loads((root / "work" / f"deep_{k}.json").read_text())["controls"]
                                      for k in defaults})
    shutil.copy2(root / "work/decodability.json", data / "decodability.json")
    acts = json.loads((root / "acts/acts.json").read_text())
    N = len(plan["probe"])
    meta = {"tag": tag, "model": "dinov2_vits14", "dataset": "clane9/imagenet-100", "N": N,
            "N_heldout": len(plan["heldout"]), "n_attn": len(plan["attn_idx"]), "smoke": plan["smoke"],
            "n_per_class": plan["n_per_class"],
            "views": {"cls": {"layers": list(range(1, 13))}, "mp": {"layers": list(range(0, 13))}},
            "kernel_ids": kids,
            "kernels": [{"id": k, "family": "self-tuned" if an.KERNELS[k][0] == "st" else "vMF",
                         "param": {"k": an.KERNELS[k][1]} if an.KERNELS[k][0] == "st" else {"kappa": an.KERNELS[k][1]},
                         "formula": an.KERNEL_FORMULA[an.KERNELS[k][0]]} for k in kids],
            "kernels_full": defaults, "default_kernels": defaults,
            "t": list(an.T_LIST), "seba_r": list(an.R_LIST), "seba_offsets": list(an.R_OFF),
            "pairs": an.all_pairs("cls"), "declared_pairs": an.DECLARED, "triples": an.TRIPLES,
            "delta_pairs": dpairs,
            "n_modes": an.N_MODES, "n_delta": an.N_DELTA, "n_attr_modes": an.N_ATTR, "n_q": an.N_Q,
            "mode_scale": float(np.sqrt(N)), "noise_snr": an.NOISE_SNR, "extension_floor": an.FLOOR, "seed": 0,
            "classes": [{k: c[k] for k in ("id", "wnid", "name", "superclass")} for c in cls["classes"]],
            "superclasses": cls["superclasses"], "superclass_rule": cls["rule"],
            "reconstruction": acts.get("reconstruction"), "readout_vs_hub": acts["probe"]["readout_vs_hub"]}
    an.jdump(data / "meta.json", meta)
    exp = expected_files(meta)
    man, missing, bad = [], [], []
    for rel, (dt, sh) in sorted(exp.items()):
        pth = data / rel
        size = {"u8": 1, "i8": 1, "i16": 2, "u16": 2, "f16": 2, "f32": 4, "i32": 4}[dt] * int(np.prod(sh))
        if not pth.exists():
            missing.append(rel)
            continue
        if pth.stat().st_size != size:
            bad.append([rel, pth.stat().st_size, size])
        man.append({"path": rel, "dtype": dt, "shape": sh, "bytes": size})
    an.jdump(data / "manifest.json", man)
    tot = sum(p.stat().st_size for p in data.rglob("*") if p.is_file())
    vol.commit()
    return {"files": len(man), "missing": missing[:20], "n_missing": len(missing), "bad": bad[:20],
            "total_mb": round(tot / 1e6, 1)}


def _merge_delta_attr(dj: dict, attr_json: Path) -> dict:
    dj["attr_note"] = "Delta-mode attributions are in delta_attr/aXX_bYY.f32 and _top.f32 (see SCHEMA.md)"
    return dj


# ------------------------------------------------------------------------------------- coordinator

@app.function(image=image, cpu=2, memory=8192, timeout=12 * 3600, volumes=VOLS)
def coordinate(tag: str, smoke: bool) -> dict:
    import analysis as an
    say = _logger(tag)
    t0 = time.time()
    tim = {}
    say(f"prepare smoke={smoke}")
    tim["prepare"] = prepare.remote(tag, smoke)
    say(f"activations {tim['prepare']}")
    tim["activations"] = activations.remote(tag)
    say(f"acts done: {json.dumps(tim['activations'])[:600]}")
    prod = acts_products.spawn(tag)
    emb = [embed.spawn(tag, v) for v in ("cls", "mp")]
    dec = decode.spawn(tag)
    cells = [(v, k) for v in ("cls", "mp") for k in an.KERNELS]
    tim["spectral"] = []
    for r in spectral.starmap([(tag, v, k) for v, k in cells], return_exceptions=True):
        tim["spectral"].append(r if isinstance(r, dict) else repr(r)[:500])
        say(f"spectral {tim['spectral'][-1]}")
    vol.reload()
    fr = {k: json.loads((_root(tag) / "work" / f"sel_cls_{k}.json").read_text()) for k in an.KERNELS}
    sel = an.select(fr)
    say(f"select: {sel['reasoning']}")
    an.jdump(_root(tag) / "work/select.json", sel)
    vol.commit()
    defaults = sel["default_kernels"]
    tim["deep"] = []
    mp_calls = [spectral.spawn(tag, "mp", k, True) for k in defaults]
    for r in deep.starmap([(tag, k) for k in defaults], return_exceptions=True):
        tim["deep"].append(r if isinstance(r, dict) else repr(r)[:500])
        say(f"deep {tim['deep'][-1]}")
    tim["mp_seba"] = []
    for c in mp_calls:
        try:
            tim["mp_seba"].append(c.get())
        except Exception as e:  # noqa: BLE001
            tim["mp_seba"].append(repr(e)[:500])
        say(f"mp seba {tim['mp_seba'][-1]}")
    tim["products"] = prod.get()
    tim["decode"] = dec.get()
    say(f"decode {tim['decode']}")
    tim["embed"] = [e.get() for e in emb]
    say(f"embed {tim['embed']}")
    tim["assemble"] = assemble.remote(tag, defaults, sel)
    say(f"assemble {tim['assemble']}")
    tim["wall_s"] = round(time.time() - t0, 1)
    (_root(tag) / "work").mkdir(parents=True, exist_ok=True)
    (_root(tag) / "work/timings.json").write_text(json.dumps(tim, indent=1))
    say("done")
    vol.commit()
    return tim


# ------------------------------------------------------------------------------------- local CLI

def _verify(tag: str) -> None:
    import numpy as np
    d = HERE / "data" / tag
    meta = json.loads((d / "meta.json").read_text())
    exp = expected_files(meta)
    bad = 0
    for rel, (dt, sh) in exp.items():
        p = d / rel
        size = {"u8": 1, "i8": 1, "i16": 2, "u16": 2, "f16": 2, "f32": 4, "i32": 4}[dt] * int(np.prod(sh))
        if not p.exists():
            print("MISSING", rel); bad += 1
        elif p.stat().st_size != size:
            print("SIZE", rel, p.stat().st_size, size); bad += 1
    for j in ("meta.json", "manifest.json", "selection.json", "controls.json", "decodability.json"):
        if not (d / j).exists():
            print("MISSING", j); bad += 1
    sheets = len(list((d / "thumbs").glob("sheet_*.jpg")))
    need = (meta["N"] + 99) // 100
    if sheets != need:
        print("THUMBS", sheets, need); bad += 1
    tot = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    print(f"{len(exp)} binaries checked, {bad} problems, total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    rec_dir = HERE / "runs"
    rec_dir.mkdir(exist_ok=True)
    if not a or a[0] == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] == "launch":
        tag, smoke = a[1], len(a) > 2 and a[2] == "smoke"
        call = modal.Function.from_name(APP, "coordinate").spawn(tag, smoke)
        rec = {"tag": tag, "smoke": smoke, "call_id": call.object_id, "launched": time.ctime()}
        (rec_dir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
    elif a[0] == "status":
        rec = json.loads((rec_dir / f"{a[1]}.json").read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0))[:3000])
        except TimeoutError:
            try:
                print("running\n" + b"".join(vol.read_file(f"{a[1]}/coordinator.log")).decode()[-4000:])
            except Exception as e:  # noqa: BLE001
                print(f"running (no log: {e!r})")
    elif a[0] == "log":
        print(b"".join(vol.read_file(f"{a[1]}/logs/{a[2]}.log")).decode()[-4000:])
    elif a[0] == "fetch":
        import shutil
        dest, tmp = HERE / "data" / a[1], HERE / "data" / f".fetch_{a[1]}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOL, f"{a[1]}/data", str(tmp)],
                       check=True)
        src = tmp / "data" if (tmp / "data").is_dir() else tmp
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(src), str(dest))
        shutil.rmtree(tmp, ignore_errors=True)
    elif a[0] == "verify":
        _verify(a[1])
    else:
        sys.exit(__doc__)
