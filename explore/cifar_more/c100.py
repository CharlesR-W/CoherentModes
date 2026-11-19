"""CIFAR-100, shared by training and measurement.

Source: the Hugging Face mirror uoft-cs/cifar100 (the dataset authors' organisation), pinned to a revision; the
original tarball at cs.toronto.edu was too slow to download (stalled at ~0.5 kB/s from both the VPS and Modal on
2026-11-13). The parquet files keep the original order (first test labels 49, 33, 72, 51, ... as in the tarball).
Decoded uint8 arrays are cached as one .npz next to the parquet files, with their sha256s for provenance.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

HF_REV = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/{rev}/cifar100/{split}-00000-of-00001.parquet"
C100_MEAN = (0.5071, 0.4865, 0.4409)
C100_STD = (0.2673, 0.2564, 0.2762)


def _decode(path):
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path)
    feats = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in t.column("img").to_pylist()])
    return (imgs.transpose(0, 3, 1, 2).copy(), t.column("fine_label").to_numpy().astype("i8"),
            t.column("coarse_label").to_numpy().astype("i8"), feats["fine_label"]["names"], feats["coarse_label"]["names"])


def load_cifar100(root: Path, commit=None):
    """uint8 NCHW tensors, fine and coarse labels, names. Downloads and decodes once into root (then `commit()`)."""
    import numpy as np
    import torch
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"cifar100_{HF_REV[:12]}.npz"
    if not cache.exists():
        arrays, sha = {}, {}
        for split in ("train", "test"):
            part = root / f"{split}.parquet.part{os.getpid()}"
            urllib.request.urlretrieve(HF_URL.format(rev=HF_REV, split=split), part)
            sha[split] = hashlib.sha256(part.read_bytes()).hexdigest()
            x, y, c, fine_names, coarse_names = _decode(part)
            part.unlink()
            arrays.update({f"{split}_x": x, f"{split}_y": y, f"{split}_coarse": c})
        assert arrays["train_x"].shape == (50000, 3, 32, 32) and arrays["test_x"].shape == (10000, 3, 32, 32)
        assert list(arrays["test_y"][:4]) == [49, 33, 72, 51], "unexpected test order"
        tmp = root / f"cache.part{os.getpid()}.npz"
        np.savez(tmp, fine_names=np.array(fine_names), coarse_names=np.array(coarse_names),
                 sha256=np.array(json.dumps(sha)), **arrays)
        tmp.rename(cache)           # atomic: a reader never sees a partial cache
        if commit:
            commit()
    z = np.load(cache)
    out = {k: torch.from_numpy(z[k]) for k in ("train_x", "train_y", "train_coarse", "test_x", "test_y", "test_coarse")}
    out["fine_names"] = [str(s) for s in z["fine_names"]]
    out["coarse_names"] = [str(s) for s in z["coarse_names"]]
    out["sha256"] = json.loads(str(z["sha256"]))
    return out


def normaliser(device):
    import torch
    mean = torch.tensor(C100_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(C100_STD, device=device).view(1, 3, 1, 1) * 255
    return lambda x: (x.to(device).float() - mean) / std
