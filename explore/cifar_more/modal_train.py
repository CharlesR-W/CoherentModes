"""Train the two networks on Modal (detached), writing to cm-cifar-coherent:/runs/<name>.

B: ModelZoo's vgg16/relu-seed0 recipe, run with ModelZoo's own zoo.train.train_run exactly as
   ../cifar_coherent/modal_train.py does for resnet20 (zoo Volume read-only for CIFAR-10). ModelZoo is not modified.
A: ResNet-18 (CIFAR variant: 3x3 stride-1 stem, no max-pool; the kuangliu/pytorch-cifar definition) on CIFAR-100,
   own loop (ModelZoo has no CIFAR-100 / ResNet-18): SGD .1, momentum .9, Nesterov, wd 5e-4, batch 128, 60 epochs,
   cosine to 0, pad-4 crop + flip, bf16 autocast in training only. Checkpoints on the zoo schedule; manifest.json in
   the zoo layout (checkpoints[step, path, train/val loss/acc]).

    uv run --with modal python modal_train.py deploy
    uv run --with modal python modal_train.py launch vgg|c100 [--smoke]
    uv run --with modal python modal_train.py status vgg|c100 [--smoke]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "cm-cifar-more-train"
GPU = "L40S"
HERE = Path(__file__).resolve().parent
ZOO_SRC = Path.home() / "code" / "ModelZoo" / "src"
EPOCHS = 60
BATCH = 128

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pyarrow", "pillow")
    .env({"PYTHONPATH": "/root/src:/root/cm", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(str(ZOO_SRC), remote_path="/root/src")
    .add_local_file(str(HERE / "nets.py"), remote_path="/root/cm/nets.py")
    .add_local_file(str(HERE / "c100.py"), remote_path="/root/cm/c100.py")
)
app = modal.App(APP)
zoo_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name("cm-cifar-coherent", create_if_missing=True)


def run_name(which: str, smoke: bool) -> str:
    return {"vgg": "vgg16-relu-seed0", "c100": "resnet18-c100-seed0"}[which] + ("-smoke" if smoke else "")


def _logger(name):
    t0 = time.time()
    log_path = Path("/out/runs") / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{time.time() - t0:7.0f}s] {msg}\n")
        out_vol.commit()
    return log, t0


@app.function(image=image, gpu=GPU, timeout=6 * 3600, volumes={"/zoo": zoo_vol, "/out": out_vol})
def train_vgg(smoke: bool) -> dict:
    import torch
    from zoo.configs import v1_runs
    from zoo.train import train_run

    assert torch.cuda.is_available(), "must run on CUDA"
    run_id = "vgg16/relu-seed0"
    cfg = v1_runs()[run_id]
    name = run_name("vgg", smoke)
    if smoke:
        cfg["total_steps"] = 60
    log, t0 = _logger(name)
    log(f"training {run_id} recipe as {name} on {torch.cuda.get_device_name()} torch {torch.__version__}")
    manifest = train_run(cfg, Path("/out/runs") / name, "cuda", Path("/zoo/data"), force=True, log=log)
    log(f"done status={manifest['status']} wall={time.time() - t0:.0f}s")
    out_vol.commit()
    return {"name": name, "status": manifest["status"], "wall_s": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name()}


def checkpoint_steps(total: int) -> list[int]:
    """The zoo schedule (zoo.schedule.checkpoint_steps): 0..16, powers of 2 to 1024, every 2%, final."""
    from zoo.schedule import checkpoint_steps as zoo_steps
    return zoo_steps(total)


@app.function(image=image, gpu=GPU, timeout=6 * 3600, volumes={"/out": out_vol})
def train_c100(smoke: bool) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from c100 import load_cifar100, normaliser
    from nets import ResNet18, augment

    assert torch.cuda.is_available()
    name = run_name("c100", smoke)
    log, t0 = _logger(name)
    run_dir = Path("/out/runs") / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    data = load_cifar100(Path("/out/data/cifar100"), commit=out_vol.commit)
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    xtr, ytr = data["train_x"].to(dev), data["train_y"].to(dev)
    xte, yte = data["test_x"].to(dev), data["test_y"].to(dev)
    norm = normaliser(dev)
    steps_per_epoch = len(xtr) // BATCH
    total = 60 if smoke else EPOCHS * steps_per_epoch
    ckpts = set(checkpoint_steps(total))
    model = ResNet18(num_classes=100).to(dev).to(memory_format=torch.channels_last)
    decay = [p for n_, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n_, p in model.named_parameters() if p.ndim <= 1]
    # weight decay on all parameters as in kuangliu/weiaicunzai recipes (both apply wd to every parameter)
    opt = torch.optim.SGD([{"params": decay}, {"params": no_decay}], lr=0.1, momentum=0.9, nesterov=True,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    gen = torch.Generator(device="cpu").manual_seed(0)
    eval_idx = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(1))[:5000].to(dev)
    manifest = dict(run_id=f"resnet18-c100/seed0{'-smoke' if smoke else ''}", family="resnet18-cifar100",
                    config=dict(arch="resnet18-cifar (kuangliu)", dataset="cifar100", epochs=EPOCHS, batch_size=BATCH,
                                optimizer="sgd lr .1 momentum .9 nesterov wd 5e-4", lr_schedule="cosine to 0 per step",
                                augment="pad-4 crop + hflip", precision="bf16 autocast (training only)", seed=0),
                    total_steps=total, checkpoint_schedule=sorted(ckpts), status="running", checkpoints=[],
                    device=torch.cuda.get_device_name())

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split, (x, y) in {"train": (xtr[eval_idx], ytr[eval_idx]), "val": (xte, yte)}.items():
            loss = correct = 0.0
            for i in range(0, len(x), 1000):
                z = model(norm(x[i:i + 1000]).contiguous(memory_format=torch.channels_last)).float()
                loss += float(F.cross_entropy(z, y[i:i + 1000], reduction="sum"))
                correct += float((z.argmax(1) == y[i:i + 1000]).sum())
            out[f"{split}_loss"], out[f"{split}_acc"] = loss / len(x), correct / len(x)
        model.train()
        return out

    def checkpoint(step):
        stats = evaluate()
        rel = f"checkpoints/step_{step}.pt"
        torch.save({"model_state": model.state_dict(), "step": step, "config": manifest["config"]}, run_dir / rel)
        manifest["checkpoints"].append({"step": step, "path": rel, **stats})
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        log(f"ckpt {step}/{total} val_acc {stats['val_acc']:.4f} val_loss {stats['val_loss']:.3f} "
            f"train_acc {stats['train_acc']:.4f}")

    model.train()
    step = 0
    checkpoint(0)
    while step < total:
        perm = torch.randperm(len(xtr), generator=gen).to(dev)
        for bi in range(steps_per_epoch):
            if step >= total:
                break
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            x, _ = augment(xtr[idx])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(norm(x).contiguous(memory_format=torch.channels_last)), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step in ckpts:
                checkpoint(step)
    manifest["status"] = "complete"
    manifest["wall_s"] = round(time.time() - t0, 1)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    out_vol.commit()
    log(f"done wall={time.time() - t0:.0f}s")
    return {"name": name, "status": "complete", "wall_s": manifest["wall_s"], "gpu": torch.cuda.get_device_name()}


def _rec_dir() -> Path:
    d = HERE / "runs"
    d.mkdir(exist_ok=True)
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] in ("launch", "status"):
        which, smoke = a[1], "--smoke" in a
        lid = f"train-{which}" + ("-smoke" if smoke else "")
        if a[0] == "launch":
            fn = {"vgg": "train_vgg", "c100": "train_c100"}[which]
            call = modal.Function.from_name(APP, fn).spawn(smoke)
            rec = {"launch_id": lid, "call_id": call.object_id, "smoke": smoke, "gpu": GPU, "launched": time.ctime()}
            (_rec_dir() / f"{lid}.json").write_text(json.dumps(rec, indent=1))
            print(json.dumps(rec, indent=1))
        else:
            rec = json.loads((_rec_dir() / f"{lid}.json").read_text())
            try:
                print(json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0), indent=1))
            except TimeoutError:
                try:
                    tail = b"".join(out_vol.read_file(f"runs/{run_name(which, smoke)}.log")).decode().splitlines()[-3:]
                except Exception as e:  # noqa: BLE001
                    tail = [f"(no log yet: {e!r})"]
                print("running\n  " + "\n  ".join(tail))
    else:
        sys.exit(__doc__)
