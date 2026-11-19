"""Train ResNet-32 / ResNet-56 on CIFAR-10 with ModelZoo's resnet20 recipe and training loop.

Same config as resnet20/relu-seed{s} except the number of blocks per stage (nets.py); zoo.train.train_run is
used unchanged with build_model swapped for nets.build_model. Output: cm-cifar-coherent:/runs/resnet{d}-relu-seed{s}.

    uv run --with modal python train_modal.py deploy
    uv run --with modal python train_modal.py launch <depth> <seed> [--smoke]
    uv run --with modal python train_modal.py status
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "cm-spectral-align-train"
GPU = "L40S"
HERE = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
    .env({"PYTHONPATH": "/root/src:/root/sa", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(str(Path.home() / "code" / "ModelZoo" / "src"), remote_path="/root/src")
    .add_local_file(str(HERE / "nets.py"), remote_path="/root/sa/nets.py")
)
app = modal.App(APP)
zoo_vol = modal.Volume.from_name("crw-modelzoo-train").read_only()
out_vol = modal.Volume.from_name("cm-cifar-coherent")


def run_name(depth, seed, smoke=False):
    return f"resnet{depth}-relu-seed{seed}" + ("-smoke" if smoke else "")


@app.function(image=image, gpu=GPU, timeout=8 * 3600, volumes={"/zoo": zoo_vol, "/out": out_vol})
def train(depth: int, seed: int, smoke: bool = False) -> dict:
    import torch
    import nets
    import zoo.train
    from zoo.configs import make_config

    assert torch.cuda.is_available()
    zoo.train.build_model = nets.build_model
    cfg = make_config("resnet20", seed, model={"arch": "resnet_cifar", "n": nets.DEPTH_N[depth]})
    cfg["family"] = f"resnet{depth}"
    if smoke:
        cfg["total_steps"] = 60
    name = run_name(depth, seed, smoke)
    run_dir = Path("/out/runs") / name
    log_path = run_dir.parent / f"{name}.log"
    t0 = time.time()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(f"[{time.time() - t0:7.0f}s] {msg}\n")
        out_vol.commit()

    log(f"training {name} on {torch.cuda.get_device_name()} torch {torch.__version__}")
    manifest = zoo.train.train_run(cfg, run_dir, "cuda", Path("/zoo/data"), force=True, log=log)
    log(f"done status={manifest['status']} wall={time.time() - t0:.0f}s")
    out_vol.commit()
    return {"name": name, "status": manifest["status"], "wall_s": round(time.time() - t0, 1),
            "final_val_acc": manifest["checkpoints"][-1]["val_acc"]}


if __name__ == "__main__":
    a = sys.argv[1:]
    rec_dir = HERE / "runs"
    if not a or a[0] == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif a[0] == "launch":
        depth, seed, smoke = int(a[1]), int(a[2]), "--smoke" in a
        call = modal.Function.from_name(APP, "train").spawn(depth, seed, smoke)
        name = run_name(depth, seed, smoke)
        rec = {"name": name, "call_id": call.object_id, "gpu": GPU, "launched": time.ctime()}
        (rec_dir / f"train-{name}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
    elif a[0] == "status":
        for p in sorted(rec_dir.glob("train-*.json")):
            rec = json.loads(p.read_text())
            try:
                print(rec["name"], json.dumps(modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)))
            except TimeoutError:
                try:
                    tail = b"".join(out_vol.read_file(f"runs/{rec['name']}.log")).decode().splitlines()[-1]
                except Exception as e:  # noqa: BLE001
                    tail = f"(no log yet: {e!r})"
                print(rec["name"], "running:", tail)
            except Exception as e:  # noqa: BLE001
                print(rec["name"], "FAILED:", repr(e)[:300])
