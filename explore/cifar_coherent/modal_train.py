"""Train ModelZoo's resnet20/relu-seed0 recipe on Modal, writing to our own Volume.

Exploratory side study (not an EXPERIMENTS.md stage).  The canonical zoo run's
weights live only on the owner's laptop, so we regenerate the same recipe with
ModelZoo's own code (``zoo.train.train_run``, same config and checkpoint
schedule: steps 0-16, powers of 2, then every 2%; 73 checkpoints).  It is a
re-run on a different GPU, not a bit-identical copy of the zoo run.

The zoo Volume is mounted read-only for CIFAR-10 data; output goes to
``cm-cifar-coherent:/runs/<name>``.

    uv run --with modal python explore/cifar_coherent/modal_train.py deploy
    uv run --with modal python explore/cifar_coherent/modal_train.py launch [--smoke]
    uv run --with modal python explore/cifar_coherent/modal_train.py status <launch-id>
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "cm-cifar-coherent-train"
GPU = "L40S"
HERE = Path(__file__).resolve().parent
ZOO_SRC = Path.home() / "code" / "ModelZoo" / "src"
RUN_ID = "resnet20/relu-seed0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .env({"PYTHONPATH": "/root/src", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(str(ZOO_SRC), remote_path="/root/src")
)
app = modal.App(APP)
zoo_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name("cm-cifar-coherent", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=6 * 3600, volumes={"/zoo": zoo_vol, "/out": out_vol})
def train(smoke: bool, seed: int = 0) -> dict:
    import torch
    from zoo.configs import v1_runs
    from zoo.train import train_run

    assert torch.cuda.is_available(), "must run on CUDA"
    run_id = f"resnet20/relu-seed{seed}"
    cfg = v1_runs()[run_id]
    name = f"resnet20-relu-seed{seed}" + ("-smoke" if smoke else "")
    if smoke:
        cfg["total_steps"] = 60
    run_dir = Path("/out/runs") / name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = run_dir.parent / f"{name}.log"
    t0 = time.time()

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{time.time() - t0:7.0f}s] {msg}\n")
        out_vol.commit()

    log(f"training {run_id} recipe as {name} on {torch.cuda.get_device_name()} torch {torch.__version__}")
    manifest = train_run(cfg, run_dir, "cuda", Path("/zoo/data"), force=True, log=log)
    log(f"done status={manifest['status']} wall={time.time() - t0:.0f}s")
    out_vol.commit()
    return {"name": name, "status": manifest["status"], "wall_s": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name()}


def _rec_dir() -> Path:
    d = HERE / "runs"
    d.mkdir(exist_ok=True)
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] == "launch":
        smoke = "--smoke" in a
        seed = int(a[a.index('--seed') + 1]) if '--seed' in a else 0
        call = modal.Function.from_name(APP, "train").spawn(smoke, seed)
        lid = time.strftime("%Y%m%d-%H%M%S") + f"-seed{seed}" + ("-smoke" if smoke else "")
        rec = {"launch_id": lid, "call_id": call.object_id, "smoke": smoke, "gpu": GPU, "seed": seed,
               "launched": time.ctime()}
        (_rec_dir() / f"{lid}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
    elif a[0] == "status":
        rec = json.loads((_rec_dir() / f"{a[1]}.json").read_text())
        try:
            print(json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0), indent=1))
        except TimeoutError:
            name = f"resnet20-relu-seed{rec.get('seed', 0)}" + ("-smoke" if rec["smoke"] else "")
            try:
                tail = b"".join(out_vol.read_file(f"runs/{name}.log")).decode().splitlines()[-3:]
            except Exception as e:  # noqa: BLE001
                tail = [f"(no log yet: {e!r})"]
            print("running\n  " + "\n  ".join(tail))
    else:
        sys.exit(__doc__)
