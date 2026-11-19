"""Schema-v3 detached Modal pipeline for the CIFAR-100 / VGG explorers. See SPEC.md sections 6-7.

    uv run --with modal python measure.py deploy
    uv run --with modal python measure.py launch <tag> [stage]     # stage: all (default) | pilot | finish
    uv run --with modal python measure.py status <tag> [stage]
    uv run --with modal python measure.py spend
    uv run --with modal python measure.py fetch <tag>

Tags: v16s0 (VGG-16-BN / CIFAR-10), r18c100s0 (ResNet-18 / CIFAR-100), v16s0-smoke (B at two checkpoints of the
60-step smoke run). Units: extract(tag, step), blocks(tag, family, index), embed(tag, step), decode(tag),
validation(tag). Scratch (Grams, timings, logs) under /out/work/<tag>; public schema under /out/analysis/<tag>.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "cifar_coherent"
APP = "cm-cifar-more"
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
B_RANKS = [2, 4, 6, 8, 10, 12, 16, 20, 24]
STUDIES = {
    "v16s0": dict(run="vgg16-relu-seed0", data="cifar10", steps=[0, 256, 2560, 25600, 64000], arch="vgg",
                  K=48, ranks=B_RANKS, per_class=200, n_classes=10, decode_per_class=1000,
                  model="VGG-16-BN (ModelZoo vgg16/relu-seed0 recipe)", dataset="CIFAR-10"),
    "v16s0-smoke": dict(run="vgg16-relu-seed0-smoke", data="cifar10", steps=[0, 60], arch="vgg",
                        K=48, ranks=B_RANKS, per_class=200, n_classes=10, decode_per_class=100,
                        model="VGG-16-BN SMOKE (60 steps)", dataset="CIFAR-10"),
    "r18c100s0": dict(run="resnet18-c100-seed0", data="cifar100", steps=[0, 256, 2340, 9360, 23400], arch="resnet18",
                      K=64, ranks=[2, 4, 10, 20, 30, 40], per_class=50, n_classes=100, decode_per_class=100,
                      model="ResNet-18 (CIFAR variant)", dataset="CIFAR-100"),
}
# Modal list prices (https://modal.com/pricing, as checked for the reference run on 2026-11-12), per second.
A100 = .000583
L40S = .000542
CPU = .0000131      # per physical core; requested vCPUs are charged as full cores here (conservative)
MEM = .00000222     # per GiB
STARTUP = 60        # seconds of startup charged to every unit record
CAP = 18.5          # stop submitting when completed + reserved compute would exceed this (ceiling $20)
# Spend no timing record or log captures: two CIFAR-100 smoke trainings stalled downloading from cs.toronto.edu for
# ~8 and ~20 min on L40S (8 CPU, 32 GiB charged) before being cancelled, 2026-11-13.
UNLOGGED = {"train:c100-smoke stalled downloads (no logs)": 28 * 60 * (.000542 + 8 * .0000131 + 32 * .00000222),
            "profile: aborted one-block A100 profiling run (<=15 min)": 900 * (.000583 + 4 * .0000131 + 32 * .00000222)}

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "scipy", "umap-learn", "pyarrow", "pillow")
         .env({"PYTHONPATH": "/root/src:/root/cc:/root/cm", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
               "CM3_SEBA_THREADS": "4"})
         .add_local_dir(str(Path.home() / "code/ModelZoo/src"), remote_path="/root/src"))
for name in ("coherent.py", "seba.py", "schema_v2.py", "sweep_math.py", "construction.py"):
    image = image.add_local_file(str(REF / name), remote_path="/root/cc/" + name)
for name in ("cm3.py", "nets.py", "c100.py", "decod.py", "validate3.py"):
    image = image.add_local_file(str(HERE / name), remote_path="/root/cm/" + name)
app = modal.App(APP)
vol = modal.Volume.from_name("cm-cifar-coherent")
zoo = modal.Volume.from_name("crw-modelzoo-train").read_only()
VOLS = {"/out": vol, "/zoo": zoo}


def paths(tag):
    public, work = Path("/out/analysis") / tag, Path("/out/work") / tag
    public.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    return public, work


def environment(threads=4):
    import torch
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


class Unit:
    """Timing record written in a finally, so failed attempts count against the budget."""

    def __init__(self, tag, name, rate):
        self.tag, self.name, self.rate = tag, name, rate
        self.start = time.time()

    def record(self):
        _, work = paths(self.tag)
        return work / f"timing_{self.name}.json"

    def cached(self, outputs):
        import cm3
        st = STUDIES[self.tag]
        path = self.record()
        if path.is_file() and all(Path(p).is_file() for p in outputs):
            row = json.loads(path.read_text())
            if row.get("status") == "ok" and row.get("construction_fingerprint") == cm3.fingerprint(st["K"], st["ranks"], n_of(st)):
                return row
        return None

    def finish(self, status, **extra):
        import cm3
        st = STUDIES[self.tag]
        seconds = time.time() - self.start
        path = self.record()
        previous = json.loads(path.read_text()) if path.is_file() else {}
        spent_before = previous.get("cumulative_usd", 0.0)      # money already spent is never erased by a rerun
        usd = (seconds + STARTUP) * self.rate
        row = dict(unit=self.name, status=status, seconds=seconds, estimated_usd=usd,
                   cumulative_usd=spent_before + usd, completed_at=time.time(),
                   construction_fingerprint=cm3.fingerprint(st["K"], st["ranks"], n_of(st)), **extra)
        path.write_text(json.dumps(row, allow_nan=False))
        vol.commit()
        return row


def n_of(st):
    return st["per_class"] * st["n_classes"]


def run_unit(tag, name, rate, outputs, body):
    vol.reload()
    unit = Unit(tag, name, rate)
    cached = unit.cached(outputs)
    if cached:
        return cached
    status, extra = "failed", {}
    try:
        extra = body() or {}
        status = "ok"
    finally:
        row = unit.finish(status, **extra)
    return row


# ----------------------------------------------------------------------------------------------- data and model
def load(tag, step):
    """Model at `step` (eval, cuda), the probe (positions in the test set), test images/labels, input normaliser."""
    import numpy as np
    import torch
    st = STUDIES[tag]
    ck_path = Path(f"/out/runs/{st['run']}/checkpoints/step_{step}.pt")
    if st["arch"] == "vgg":
        from zoo.data import build_data
        from zoo.io import load_checkpoint
        from zoo.models import build_model
        ck = load_checkpoint(ck_path)
        model = build_model(ck["config"])
        model.load_state_dict(ck["model_state"])
        data = build_data(ck["config"], Path("/zoo/data"), "cpu")
        d = dict(test_x=data.test_x, test_y=data.test_y, train_x=data.train_x, train_y=data.train_y,
                 norm=lambda x: data.normalize(x.cpu()).cuda(), classes=CIFAR10)
    else:
        from c100 import load_cifar100, normaliser
        from nets import ResNet18
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = ResNet18(100)
        model.load_state_dict(ck["model_state"])
        raw = load_cifar100(Path("/out/data/cifar100"))
        d = dict(test_x=raw["test_x"], test_y=raw["test_y"], train_x=raw["train_x"], train_y=raw["train_y"],
                 test_coarse=raw["test_coarse"], norm=normaliser("cuda"), classes=raw["fine_names"],
                 coarse_names=raw["coarse_names"])
    model.eval().cuda()
    y = d["test_y"].numpy()
    rng = np.random.RandomState(0)
    d["idx"] = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], st["per_class"], replace=False)
                                       for c in range(st["n_classes"])]))
    return model, d


def sites_fn(tag):
    import nets
    return nets.vgg_sites if STUDIES[tag]["arch"] == "vgg" else nets.resnet18_sites


def site_names(tag):
    import nets
    return nets.VGG_SITES if STUDIES[tag]["arch"] == "vgg" else nets.RESNET18_SITES


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=49152, timeout=1800, max_containers=5, retries=0,
              volumes=VOLS)
def extract(tag: str, step: int):
    rate = A100 + 4 * CPU + 48 * MEM

    def body():
        import numpy as np
        import torch
        import cm3
        environment()
        root, work = paths(tag)
        st = STUDIES[tag]
        model, d = load(tag, step)
        idx = d["idx"]
        n = len(idx)
        s1, s2, _, _ = cm3.split(n)
        f = sites_fn(tag)
        store = None
        with torch.inference_mode():
            for at in range(0, n, 250):
                acts = f(model, d["norm"](d["test_x"][idx[at:at + 250]]))
                if store is None:
                    store = [torch.empty((n,) + tuple(a.shape[1:])) for a in acts]
                for dst, a in zip(store, acts):
                    dst[at:at + len(a)] = a.float().cpu()
        pcs = []
        for si, a in enumerate(store):
            full, g1, g2 = cm3.subset_grams(a.flatten(1).cuda(), s1, s2)
            for key, g in (("f", full), ("1", g1), ("2", g2)):
                np.save(work / f"gram_{step}_{si}_{key}.npy", g.cpu().numpy())
            val, v = torch.linalg.eigh(full)
            pcs.append((v[:, -50:] * val[-50:].clamp_min(0).sqrt()).flip(1).cpu().numpy())
            del full, g1, g2
        np.save(work / f"pcs_{step}.npy", np.stack(pcs).astype("f4"))
        logits = store[-1].double()
        y = d["test_y"][idx]
        pred = logits.argmax(1)
        np.savez_compressed(root / f"points_{step}.npz",
                            loss=torch.nn.functional.cross_entropy(logits, y, reduction="none").numpy().astype("f4"),
                            pred=pred.numpy().astype("i1"), correct=(pred == y).numpy())
        if not (root / "probe.npz").exists():
            extra = {}
            if st["data"] == "cifar100":
                extra = dict(labels=d["test_coarse"][idx].numpy(), fine=y.numpy(),
                             coarse_names=np.array(d["coarse_names"]), fine_names=np.array(d["classes"]))
            else:
                ref = Path("/out/analysis/r20s0b/probe.npz")
                assert np.array_equal(np.load(ref)["idx"], idx), "VGG probe differs from the reference probe"
                extra = dict(labels=y.numpy())
            np.savez_compressed(root / "probe.npz", idx=idx, images=d["test_x"][idx].permute(0, 2, 3, 1).numpy(), **extra)
        return dict(n=n, sites=len(store))
    _, work = paths(tag)
    nsite = len(site_names(tag))
    return run_unit(tag, f"extract_{step}", rate, [work / f"pcs_{step}.npy"] +
                    [work / f"gram_{step}_{si}_{k}.npy" for si in range(nsite) for k in "f12"], body)


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=32768, timeout=10800, max_containers=6, retries=0,
              volumes=VOLS)
def blocks(tag: str, family: str, index: int):
    rate = A100 + 4 * CPU + 32 * MEM
    st = STUDIES[tag]
    names = site_names(tag)
    name = f"{family}_{st['steps'][index] if family == 'layer' else names[index]}"

    def body():
        import numpy as np
        import torch
        import cm3
        environment(threads=1)       # CPU work is SEBA, parallel across ranks (CM3_SEBA_THREADS); GPU does the rest
        root, work = paths(tag)
        cells = [(st["steps"][index], j) for j in range(len(names))] if family == "layer" else \
                [(s, index) for s in st["steps"]]
        grams = [tuple(torch.tensor(np.load(work / f"gram_{s}_{si}_{k}.npy"), device="cuda", dtype=torch.float64)
                       for k in "f12") for s, si in cells]
        t0 = time.time()
        arrays, infos = cm3.compute_family(grams, st["K"], st["ranks"], log=lambda m: print(name, m, round(time.time() - t0), flush=True))
        np.savez_compressed(root / f"{name}.npz", **arrays)
        if family == "layer":
            (work / f"kappa_{st['steps'][index]}.json").write_text(json.dumps(infos, allow_nan=False))
        return dict(blocks=int(arrays["vals"].shape[0] * arrays["vals"].shape[1]))
    root, work = paths(tag)
    outs = [root / f"{name}.npz"] + ([work / f"kappa_{st['steps'][index]}.json"] if family == "layer" else [])
    return run_unit(tag, name, rate, outs, body)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, max_containers=5, retries=0, volumes=VOLS)
def embed(tag: str, step: int):
    rate = 4 * CPU + 16 * MEM

    def body():
        import numpy as np
        import umap
        root, work = paths(tag)
        pcs = np.load(work / f"pcs_{step}.npy")
        n = pcs.shape[1]
        rel = [{i: i for i in range(n)} for _ in range(len(pcs) - 1)]
        model = umap.AlignedUMAP(n_neighbors=15, min_dist=.1, random_state=0, alignment_regularisation=.02)
        model.fit(list(pcs), relations=rel)
        np.save(root / f"umap_{step}.npy", np.stack(model.embeddings_).astype("f4"))
    root, _ = paths(tag)
    return run_unit(tag, f"embed_{step}", rate, [root / f"umap_{step}.npy"], body)


@app.function(image=image, gpu="A100-40GB", cpu=4, memory=65536, timeout=3600, retries=0, volumes=VOLS)
def decode(tag: str):
    rate = A100 + 4 * CPU + 64 * MEM
    st = STUDIES[tag]

    def body():
        import numpy as np
        import torch
        from decod import PROTOCOL, score_view
        environment()
        root, _ = paths(tag)
        step = st["steps"][-1]
        model, d = load(tag, step)
        ytr = d["train_y"].numpy()
        rng = np.random.RandomState(0)
        fit = np.sort(np.concatenate([rng.choice(np.where(ytr == c)[0], st["decode_per_class"], replace=False)
                                      for c in range(st["n_classes"])]))
        f = sites_fn(tag)
        names = site_names(tag)
        res = {"step": step, "sites": names, "views": ["flat", "gap"], "results": {"flat": {}, "gap": {}},
               "protocol": PROTOCOL.format(n_fit=len(fit)), "validation_only": True, "n_classes": st["n_classes"]}
        for si, site in enumerate(names):
            feats = []
            for images in (d["train_x"][fit], d["test_x"]):
                flat, gap = [], []
                with torch.inference_mode():
                    for at in range(0, len(images), 500):
                        a = f(model, d["norm"](images[at:at + 500]))[si].float()
                        flat.append(a.flatten(1).cpu())
                        if a.ndim == 4:
                            gap.append(a.mean((2, 3)).cpu())
                feats.append((torch.cat(flat), torch.cat(gap) if gap else None))
            for vi, view in enumerate(("flat", "gap")):
                if feats[0][vi] is None:
                    continue
                res["results"][view][site] = score_view(feats[0][vi], feats[1][vi], d["train_y"][fit],
                                                        d["test_y"], st["n_classes"])
                print(site, view, res["results"][view][site], flush=True)
            del feats
        (root / "decod.json").write_text(json.dumps(res, allow_nan=False))
    root, _ = paths(tag)
    return run_unit(tag, "decode", rate, [root / "decod.json"], body)


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes=VOLS)
def validation(tag: str):
    rate = 4 * CPU + 16 * MEM

    def body():
        from validate3 import validate
        root, _ = paths(tag)
        report = validate(root)
        (root / "validation.json").write_text(json.dumps(report))
        return {k: v for k, v in report.items() if not isinstance(v, (list, dict))}
    root, _ = paths(tag)
    return run_unit(tag, "validation", rate, [root / "validation.json"], body)


# ----------------------------------------------------------------------------------------------- ledger
RESERVED = Path("/out/work/_reservations.json")


def reservations():
    return json.loads(RESERVED.read_text()) if RESERVED.exists() else {}


def ledger():
    """Completed (and failed) compute across both studies, training wall time from the run logs, and every
    outstanding stage reservation (a reservation stays until its stage finishes)."""
    import re
    vol.reload()
    total, rows = 0.0, {}
    for key, usd in reservations().items():
        rows["reserved:" + key] = usd
        total += usd
    for key, usd in UNLOGGED.items():
        rows[key] = usd
        total += usd
    for tag in STUDIES:
        w = Path("/out/work") / tag
        usd = sum(json.loads(p.read_text()).get("cumulative_usd", 0.0) for p in w.glob("timing_*.json")) if w.exists() else 0.0
        rows[tag] = usd
        total += usd
    for p in Path("/out/runs").glob("*.log"):
        if not any(p.name.startswith(STUDIES[t]["run"]) for t in STUDIES):
            continue
        m = re.findall(r"wall=(\d+)", p.read_text())
        secs = float(m[-1]) if m else float(p.stat().st_mtime - p.stat().st_ctime)
        usd = (secs + STARTUP) * (L40S + 8 * CPU + 32 * MEM)
        rows["train:" + p.stem] = usd
        total += usd
    return total, rows


def meta(tag):
    """meta.json from the layer units' kernel metadata (needs every layer unit)."""
    import cm3
    st = STUDIES[tag]
    root, work = paths(tag)
    names = site_names(tag)
    infos = [json.loads((work / f"kappa_{s}.json").read_text()) for s in st["steps"]]   # [t][site][rule]
    field = lambda key: [[[infos[t][si][mi][key] for si in range(len(names))] for t in range(len(st["steps"]))]
                         for mi in range(len(cm3.MODES))]
    import numpy as np
    probe = np.load(root / "probe.npz")
    if st["data"] == "cifar100":
        classes, fine = [str(x) for x in probe["coarse_names"]], [str(x) for x in probe["fine_names"]]
    else:
        classes, fine = CIFAR10, None
    n = n_of(st)
    s1, s2, c1, _ = cm3.split(n)
    m = dict(schema_version=3, construction_fingerprint=cm3.fingerprint(st["K"], st["ranks"], n), tag=tag,
             model=st["model"], dataset=st["dataset"], run=st["run"], steps=st["steps"], sites=names, modes=cm3.MODES,
             kappa=field("kappa"), kappa_q10=field("kappa_q10"), kappa_q90=field("kappa_q90"), nn_mass=field("nn_mass"),
             sinkhorn_row_error=field("row_error"), seba_r=st["ranks"], n_vecs=st["K"],
             perplexity_unattainable=[[infos[t][si][1].get("perplexity_unattainable", 0) for si in range(len(names))]
                                      for t in range(len(st["steps"]))],
             layer_pairs=cm3.pairs(len(names)), time_pairs=cm3.pairs(len(st["steps"])), n=n, classes=classes,
             fine_classes=fine, default_mode=0,
             stability=dict(seed=cm3.STAB_SEED, core=len(c1), subset_sizes=[len(s1), len(s2)], window=cm3.STAB_WINDOW,
                            threshold=cm3.STAB_THRESHOLD, guard=cm3.STAB_GUARD,
                            definition="span_k = min cos^2 principal angle between leading-k spans of the two subset block operators' eigenvectors restricted to the shared core; horizon = largest k with span_k >= threshold"),
             rule="Cosine distance on probe-centred, row-normalised flattened activations. 0: union 15NN graph, weight 1 on edges, 1e-6 elsewhere, zero diagonal. 1: per-row conditional perplexity 100, symmetric mean affinity. Both: log-domain symmetric scaling P of the zero-diagonal affinity (row residual < 2e-12), M = (I+P)/2, block operator = mean of the cells' M.",
             quantization="int16, psi = vecs * scale / 32767")
    (root / "meta.json").write_text(json.dumps(m, indent=1, allow_nan=False))
    vol.commit()


# max_containers=1 (one input per container): Modal itself serialises coordinators across both studies, so two
# coordinators can never grant reservations concurrently; a second launch queues until the first returns.
@app.function(image=image, cpu=.25, memory=2048, timeout=12 * 3600, volumes=VOLS, max_containers=1)
def coordinate(tag: str, stage: str = "all"):
    st = STUDIES[tag]
    root, work = paths(tag)
    log = work / "coordinator.log"
    names = site_names(tag)
    T, S = len(st["steps"]), len(names)

    def say(text):
        with log.open("a") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
        print(text, flush=True)
        vol.commit()

    inv = f"{tag}:{int(time.time())}"        # this invocation's reservation keys start with inv

    def reserve(label, units, seconds_each, rate, keep=False, safety=1.5):
        if not keep:         # this invocation's previous stage succeeded, so its actual costs are recorded
            release()
        spent, _ = ledger()
        forecast = units * (safety * seconds_each + STARTUP) * rate
        say(f"{label}: {units} units, forecast ${forecast:.2f}; completed + reserved ${spent:.2f}")
        if spent + forecast > CAP:
            raise RuntimeError(f"budget: completed + reserved ${spent:.2f} + forecast ${forecast:.2f} > ${CAP}")
        r = reservations()
        r[f"{inv}:{label}"] = forecast
        RESERVED.write_text(json.dumps(r))
        vol.commit()

    def release():
        """Drop THIS invocation's reservations (their actual costs are now in the timing records)."""
        vol.reload()
        r = {k: v for k, v in reservations().items() if not k.startswith(inv + ":")}
        RESERVED.write_text(json.dumps(r))
        vol.commit()

    # Serialised by max_containers=1. On success the stage's actual costs are in the timing records and its
    # reservation is released; on failure the reservation is KEPT (workers spawned by a failed stage may still be
    # running) until checked by hand and cleared with `measure.py unreserve <tag>`.
    vol.reload()
    left = {k: v for k, v in reservations().items() if k.startswith(tag + ":")}
    if left:
        raise RuntimeError(f"outstanding reservation from an earlier failed invocation: {left}; confirm its workers "
                           f"finished, then `measure.py unreserve {tag}`")
    out = _stages(tag, stage, st, root, work, names, T, S, say, reserve, release)
    release()
    return out


def _stages(tag, stage, st, root, work, names, T, S, say, reserve, release):

    def measured(prefix):
        vol.reload()
        rows = [json.loads(p.read_text()) for p in work.glob(f"timing_{prefix}*.json")]
        rows = [r for r in rows if r.get("status") == "ok"]
        return max(r["seconds"] for r in rows) if rows else None

    if stage in ("all", "pilot"):
        reserve("extract", T, measured("extract_") or 600, A100 + 4 * CPU + 48 * MEM)
        say("extract")
        rows = list(extract.starmap([(tag, s) for s in st["steps"]]))
        assert all(r["status"] == "ok" for r in rows), rows
        vol.reload()
        man = json.loads(Path(f"/out/runs/{st['run']}/manifest.json").read_text())
        (root / "curves.json").write_text(json.dumps(man, allow_nan=False))
        reserve("pilot blocks", 2, 3600, A100 + 4 * CPU + 32 * MEM)
        say("pilot: final layer unit and the last site's time unit")
        rows = list(blocks.starmap([(tag, "layer", T - 1), (tag, "time", S - 1)]))
        say("pilot timings: " + json.dumps([(r["unit"], round(r["seconds"])) for r in rows]))
    if stage in ("all", "rest"):
        lay, tim = measured("layer_"), measured("time_")
        assert lay and tim, "run the pilot first"
        rate = A100 + 4 * CPU + 32 * MEM
        jobs = [(tag, "layer", i) for i in range(T)] + [(tag, "time", i) for i in range(S)]
        # 1.25x: these units were measured by identical pilot units (same N, K, ranks, cells per unit)
        reserve("blocks", T + S - 2, (lay * T + tim * S) / (T + S), rate, safety=1.25)
        say("all blocks (pilot units are cached)")
        list(blocks.starmap(jobs))
        vol.reload()          # the kappa files were committed by other containers
        meta(tag)
    if stage in ("all", "rest", "finish"):
        reserve("decode", 1, 1500, A100 + 4 * CPU + 64 * MEM)
        reserve("embed + validate", T + 1, 1500, 4 * CPU + 16 * MEM, keep=True)
        say("layouts, decodability, validation")
        dcall = decode.spawn(tag)
        try:
            list(embed.starmap([(tag, s) for s in st["steps"]]))
        finally:      # never leave the spawned decoder running unaccounted
            try:
                dcall.get()
            except Exception as e:  # noqa: BLE001   decodability is optional (SPEC section 3)
                say(f"decode failed: {e!r}")
        vol.reload()
        if not (root / "meta.json").exists():
            meta(tag)
        validation.remote(tag)
    spent, rows = ledger()
    say(f"{stage} complete; ledger ${spent:.3f} {json.dumps(rows)}")
    return dict(tag=tag, stage=stage, ledger=spent)


@app.function(image=image, cpu=.25, memory=1024, timeout=600, volumes=VOLS)
def unreserve(tag: str):
    """Clear a failed stage's kept reservation once its workers are confirmed finished (timing records present)."""
    vol.reload()
    r = reservations()
    gone = {k: v for k, v in r.items() if k.startswith(tag + ":")}
    RESERVED.write_text(json.dumps({k: v for k, v in r.items() if not k.startswith(tag + ":")}))
    vol.commit()
    return gone


@app.function(image=image, cpu=.25, memory=1024, timeout=600, volumes=VOLS)
def spend():
    return ledger()


@app.function(image=image, cpu=.25, memory=1024, timeout=600, volumes=VOLS)
def clean(tag: str):
    """Delete the scratch Grams of a validated tag."""
    root, work = paths(tag)
    assert json.loads((root / "validation.json").read_text()).get("complete"), "validate first"
    n = 0
    for p in work.glob("gram_*.npy"):
        p.unlink()
        n += 1
    vol.commit()
    return n


if __name__ == "__main__":
    args = sys.argv[1:]
    recdir = HERE / "runs"
    recdir.mkdir(exist_ok=True)
    cmd = args[0] if args else "deploy"
    if cmd == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif cmd == "launch":
        tag, stage = args[1], args[2] if len(args) > 2 else "all"
        call = modal.Function.from_name(APP, "coordinate").spawn(tag, stage)
        rec = dict(tag=tag, stage=stage, call_id=call.object_id, launched=time.ctime())
        (recdir / f"measure-{tag}-{stage}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
    elif cmd == "status":
        tag, stage = args[1], args[2] if len(args) > 2 else "all"
        rec = json.loads((recdir / f"measure-{tag}-{stage}.json").read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)))
        except TimeoutError:
            print("running")
            try:
                print(b"".join(vol.read_file(f"work/{tag}/coordinator.log")).decode()[-2500:])
            except Exception as e:  # noqa: BLE001
                print(type(e).__name__)
    elif cmd == "unreserve":
        print(modal.Function.from_name(APP, "unreserve").remote(args[1]))
    elif cmd == "spend":
        print(json.dumps(modal.Function.from_name(APP, "spend").remote(), indent=1))
    elif cmd == "clean":
        print(modal.Function.from_name(APP, "clean").remote(args[1]))
    elif cmd == "fetch":
        dest = HERE / "data"
        dest.mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", "cm-cifar-coherent",
                        f"analysis/{args[1]}", str(dest)], check=True)
    else:
        raise SystemExit(__doc__)
