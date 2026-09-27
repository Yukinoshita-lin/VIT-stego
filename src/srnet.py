"""SRNet baseline for steganalysis (Boroumand et al. 2019), adapted for VIT-stego.

A deep residual CNN with an optional fixed SRM-30 residual stem (matching
LightViT's `--srm` flag). Output is a single logit so it shares train.py /
evaluate.py with LightViT and is directly comparable (same data, split, seed, and
BCEWithLogits objective).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vit_stego import SRMStem
except ImportError:  # fallback when run as a plain script
    from vit_stego import SRMStem


class Bottleneck(nn.Module):
    """ResNet-style bottleneck: 1x1 reduce -> 3x3 -> 1x1 expand + residual."""

    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        mid = out_ch // self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=True)


class SRNet(nn.Module):
    """SRNet steganalyzer. Input (N, 1, H, W) -> (N,) logit.

    Channel plan follows the paper's 64->512 growth; three strided bottlenecks
    downsample by 8x (128x128 -> 16x16) before global average pooling.
    """

    def __init__(
        self,
        img_size: int = 128,
        in_ch: int = 1,
        base_width: int = 64,
        num_classes: int = 1,
        use_srm: bool = False,
        truncate: float = 0.0,
    ):
        super().__init__()
        self.use_srm = use_srm
        if use_srm:
            # Same truncation knob as the ViT path, so a LightViT-vs-SRNet
            # comparison is not decided by one model seeing clipped residuals
            # and the other seeing content edges at full magnitude.
            self.stem = SRMStem(truncate=truncate)
            in_ch = 30
        else:
            self.stem = nn.Identity()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_ch, base_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        c = [
            base_width,
            base_width * 2,
            base_width * 4,
            base_width * 8,
            base_width * 8,
            base_width * 8,
            base_width * 8,
        ]
        self.layer1 = Bottleneck(c[0], c[0], stride=1)
        self.layer2 = Bottleneck(c[0], c[1], stride=2)
        self.layer3 = Bottleneck(c[1], c[2], stride=2)
        self.layer4 = Bottleneck(c[2], c[3], stride=2)
        self.layer5 = Bottleneck(c[3], c[4], stride=1)
        self.layer6 = Bottleneck(c[4], c[5], stride=1)
        self.layer7 = Bottleneck(c[5], c[6], stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c[6], num_classes)

        for name, m in self.named_modules():
            # Never re-initialise the SRM stem: kaiming_normal_ writes in place and
            # ignores requires_grad=False, so the frozen kernels would be silently
            # replaced by random ones and the model would train on noise.
            if name == "stem" or name.startswith("stem."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.gap(x).flatten(1)
        return self.fc(x).squeeze(-1)


if __name__ == "__main__":
    for use_srm in (False, True):
        m = SRNet(img_size=128, use_srm=use_srm)
        m.eval()
        with torch.no_grad():
            x = torch.randn(2, 1, 128, 128)
            y = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"SRNet use_srm={use_srm}: {tuple(x.shape)} -> {tuple(y.shape)}, params={n:,}")