"""Network definitions and site extractors for the two studies.

ResNet18: the CIFAR variant from kuangliu/pytorch-cifar (models/resnet.py; MIT licence): 3x3 stride-1 stem, no
max-pool, BasicBlocks [2, 2, 2, 2], 4x4 average pool, linear head.
VGG: ModelZoo's zoo.models.VGG (imported at extraction time, not redefined here).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.avg_pool2d(out, 4).flatten(1)
        return self.linear(out)


RESNET18_SITES = ["stem"] + [f"b{i}" for i in range(1, 9)] + ["pool", "logits"]
VGG_CONVS = [1, 2, 4, 6, 7, 9, 10, 12, 13]          # all convs except the first of stages 2-5 (c3, c5, c8, c11)
VGG_SITES = [f"c{i}" for i in VGG_CONVS] + ["pool", "logits"]


def resnet18_sites(model: ResNet18, x):
    h = F.relu(model.bn1(model.conv1(x)))
    acts = [h]
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            h = block(h)
            acts.append(h)
    pool = F.avg_pool2d(h, 4).flatten(1)
    return acts + [pool, model.linear(pool)]


def vgg_sites(model, x):
    """ReLU outputs of the listed convs, the flattened classifier input (after the fifth max-pool), logits."""
    acts, conv = [], 0
    pending = False
    h = x
    for layer in model.features:
        h = layer(h)
        if isinstance(layer, nn.Conv2d):
            conv += 1
            pending = conv in VGG_CONVS
        elif pending and isinstance(layer, (nn.ReLU, nn.GELU)):
            acts.append(h)
            pending = False
    assert conv == 13 and len(acts) == len(VGG_CONVS)
    pool = h.flatten(1)
    assert pool.shape[1] == 512
    return acts + [pool, model.classifier(pool)]


def augment(x):
    """Pad-4 zero padding, random 32x32 crop, random horizontal flip, on the GPU (uint8 NCHW in and out)."""
    b = len(x)
    dev = x.device
    x = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=dev)
    ox = torch.randint(0, 9, (b,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (oy[:, None] + ar[None]).view(b, 1, 32, 1).expand(b, x.shape[1], 32, 40)
    cols = (ox[:, None] + ar[None]).view(b, 1, 1, 32).expand(b, x.shape[1], 32, 32)
    x = x.gather(2, rows).gather(3, cols)
    flip = torch.rand(b, device=dev) < .5
    return torch.where(flip[:, None, None, None], x.flip(-1), x), (oy, ox, flip)
