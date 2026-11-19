"""Instantiate site/content/<tag>/ from the shared _*.tmpl.html templates (per-tag text below).
Hand-written per-tag files (findings.html, tours.json) are kept if present.   python3 make_content.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGS = {
    "v16s0": dict(
        title="Coherent sets in VGG-16 on CIFAR-10", short="Coherent sets: VGG-16 / CIFAR-10",
        subtitle="an explorer with 48 eigenfunctions per block and a stability horizon; companion to the ResNet-20 report",
        PROBE_DESC="200 per class, exactly the probe of the ResNet-20 report",
        SITES_DESC="ReLU outputs of convolutions c1, c2, c4, c6, c7, c9, c10, c12 and c13 (all thirteen except the first of stages 2 to 5), the 512-d classifier input after the fifth max-pool, and the logits",
        CLASS_DESC="the 10 CIFAR-10 classes",
        TRAIN_DESC="ModelZoo's vgg16/relu-seed0 recipe (the ResNet-20 recipe: SGD 0.1, momentum 0.9, weight decay 1e-4, batch 128, 64,000 steps, learning rate divided by 10 at 32,000 and 48,000), retrained on Modal.",
        DECOD_FIT="10,000 training images (1,000 per class)"),
    "r18c100s0": dict(
        title="Coherent sets in ResNet-18 on CIFAR-100", short="Coherent sets: ResNet-18 / CIFAR-100",
        subtitle="an explorer with 64 eigenfunctions per block and a stability horizon; companion to the ResNet-20 report",
        PROBE_DESC="50 per fine class",
        SITES_DESC="the stem (3×3 convolution, batch norm, ReLU), the outputs of the eight basic blocks b1 to b8 (two per stage; b3, b5, b7 halve the resolution), the 512-d global average pool and the 100 logits",
        CLASS_DESC="the 20 CIFAR-100 superclasses by default; the 100 fine classes as a second colouring (shades of their superclass's hue)",
        TRAIN_DESC="ResNet-18, CIFAR variant (3×3 stride-1 stem, no max-pool); SGD 0.1, Nesterov momentum 0.9, weight decay 5e-4, batch 128, 60 epochs (23,400 steps) with a cosine schedule, crop and flip augmentation.",
        DECOD_FIT="10,000 training images (100 per fine class), 100-way"),
}
TAGS["v16s0-smoke"] = dict(TAGS["v16s0"], title="SMOKE TEST: VGG-16 after 60 steps")

for tag, d in TAGS.items():
    out = HERE / tag
    out.mkdir(exist_ok=True)
    for name in ("overview", "measurements", "method", "findings"):
        if name == "findings" and (out / "findings.html").exists():
            continue
        text = (HERE / f"_{name}.tmpl.html").read_text()
        for key in ("PROBE_DESC", "SITES_DESC", "CLASS_DESC", "TRAIN_DESC", "DECOD_FIT"):
            text = text.replace(key, d[key])
        (out / f"{name}.html").write_text(text)
    (out / "header.json").write_text(json.dumps(dict(title=d["title"], subtitle=d["subtitle"], short_title=d["short"],
                                                     date="2026-11-13"), indent=1))
    print("wrote", out)
