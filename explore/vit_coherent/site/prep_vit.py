"""Complete data/<tag>/ (from blocks_v2.py) with the display-side files the site builder reads.

    python3 site/prep_vit.py [vits_b2k]

Adds, from the local ViT run data/dinov2s/ (SCHEMA.md):
  probe.npz     idx, labels (display group 0..9), fine_labels (ImageNet-100 class 0..99), superclass (WordNet, 18),
                images (N, 64, 64, 3) uint8 thumbnails
  points_0.npz  per site L1..L12: the 6 attention-head output norms at the CLS position, ||A_{l,h}(x)||, as the
                cell's "channels" (uint8 per channel, lo/hi), for the side panel's R^2 grid and channel correlations
  meta.json     classes (display group names), fine_classes, class_groups_rule, channel_name/desc, class_option
Labels and head norms only colour and validate; none enters a kernel or a set.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "data" / "dinov2s"

# 10 display groups made from the 18 WordNet superclasses of classes.json (fixed rule, superclasses.py), so that
# the 10-colour class palette of the ResNet site applies. Display/validation only.
GROUPS = ["dogs", "other mammals", "birds", "reptiles", "invertebrates", "food and plants",
          "clothing", "devices", "containers & vehicles", "structures"]
SUPER_TO_GROUP = {"bird.n.01": 2, "commodity.n.01": 7, "container.n.01": 8, "covering.n.02": 6, "device.n.01": 7,
                  "furnishing.n.02": 9, "hunting_dog.n.01": 0, "invertebrate.n.01": 4, "mammal.n.01": 1,
                  "matter.n.03": 5, "other artifact.n.01": 8, "other dog.n.01": 0,
                  "other instrumentality.n.03": 7, "other organism.n.01": 5, "reptile.n.01": 3,
                  "structure.n.01": 9, "toy_dog.n.01": 0, "working_dog.n.01": 0}


def quantize(x):
    lo, hi = x.min(0), x.max(0)
    q = np.rint(255 * (x - lo) / np.maximum(hi - lo, 1e-30)).clip(0, 255).astype(np.uint8)
    return q, lo.astype(np.float32), hi.astype(np.float32)


def main(tag="vits_b2k"):
    dst = ROOT / "data" / tag
    meta = json.loads((dst / "meta.json").read_text())
    vm = json.loads((SRC / "meta.json").read_text())
    idx = np.asarray(meta["probe_idx"])
    fine = np.fromfile(SRC / "images/class.u8", np.uint8)[idx]
    sup = np.fromfile(SRC / "images/superclass.u8", np.uint8)[idx]
    sname = {s["id"]: s["name"] for s in vm["superclasses"]}
    group = np.array([SUPER_TO_GROUP[sname[int(s)]] for s in sup], np.int8)
    assert (fine == idx // 100).all()
    sheets = {}
    imgs = np.zeros((len(idx), 64, 64, 3), np.uint8)
    for k, i in enumerate(idx):
        c = int(i) // 100
        if c not in sheets:
            sheets[c] = np.asarray(Image.open(SRC / f"thumbs/sheet_{c:03d}.jpg").convert("RGB"))
        r, q = divmod(int(i) % 100, 10)
        imgs[k] = sheets[c][r * 64:(r + 1) * 64, q * 64:(q + 1) * 64]
    np.savez_compressed(dst / "probe.npz", idx=idx, labels=group, fine_labels=fine, superclass=sup, images=imgs)

    pts = {}
    for si, site in enumerate(meta["sites"]):
        heads = np.fromfile(SRC / f"cls/heads/L{si + 1:02d}.f16", np.float16).reshape(6, -1)[:, idx].T.astype(np.float32)
        q, lo, hi = quantize(heads)
        pts[f"chan_{site}"], pts[f"chan_{site}_lo"], pts[f"chan_{site}_hi"] = q, lo, hi
    np.savez_compressed(dst / "points_0.npz", **pts)

    names = [c["name"].split(",")[0] for c in vm["classes"]]
    meta.update(classes=GROUPS, fine_classes=names, class_option="class group (10 groups of the 100 classes)",
                class_noun="the 10 class groups",
                class_groups_rule="10 display groups of the 100 ImageNet classes, merged from the 18 WordNet "
                                  "superclasses of classes.json: " + "; ".join(
                                      f"{g}: " + ", ".join(sorted(k for k, v in SUPER_TO_GROUP.items() if v == gi))
                                      for gi, g in enumerate(GROUPS)),
                channel_name="head", channel_desc="attention-head output norms at the CLS position",
                group_sizes=np.bincount(group, minlength=10).tolist())
    (dst / "meta.json").write_text(json.dumps(meta, indent=1, allow_nan=False))
    print("groups (images):", dict(zip(GROUPS, np.bincount(group, minlength=10).tolist())))


if __name__ == "__main__":
    main(*sys.argv[1:])
