"""CIFAR ResNets of depth 6n+2 built from ModelZoo's own BasicBlock (ModelZoo is not modified).

n=3 reproduces zoo.models.ResNet20 exactly (same modules, same init order). Sites, in order:
stem, b1..b{3n} (BasicBlock outputs), pool, logits. Stage s (0,1,2) holds blocks s*n+1 .. (s+1)*n.
"""
import torch.nn as nn

DEPTH_N = {20: 3, 32: 5, 56: 9}


def site_names(depth):
    return ["stem"] + [f"b{i}" for i in range(1, 3 * DEPTH_N[depth] + 1)] + ["pool", "logits"]


def site_stage(depth):
    """Stage of each site: 0 = stem, 1..3 = residual stages, 4 = pool, 5 = logits."""
    n = DEPTH_N[depth]
    return [0] + [1 + i // n for i in range(3 * n)] + [4, 5]


def build_resnet(n, activation="relu", num_classes=10):
    from zoo.models import BasicBlock, _activation

    class ResNetCifar(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.act = _activation(activation)
            blocks, in_planes = [], 16
            for planes, stride in ((16, 1), (32, 2), (64, 2)):
                for i in range(n):
                    blocks.append(BasicBlock(in_planes, planes, stride if i == 0 else 1, activation))
                    in_planes = planes
            self.blocks = nn.Sequential(*blocks)
            self.fc = nn.Linear(64, num_classes)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            out = self.act(self.bn1(self.conv1(x)))
            out = self.blocks(out)
            return self.fc(out.mean(dim=(2, 3)))

    return ResNetCifar()


def build_model(cfg, vocab_size=None):
    """Drop-in for zoo.models.build_model that also knows arch 'resnet_cifar' (model.n blocks per stage)."""
    from zoo import models
    m = cfg["model"]
    if m["arch"] == "resnet_cifar":
        return build_resnet(m["n"], m["activation"], m["num_classes"])
    return models.build_model(cfg, vocab_size)


def activations(model, x):
    h = model.act(model.bn1(model.conv1(x)))
    acts = [h]
    for block in model.blocks:
        h = block(h)
        acts.append(h)
    pool = h.mean((2, 3))
    return acts + [pool, model.fc(pool)]
