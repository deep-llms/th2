"""Scratch-only prototype: separate nonlinear processing of all four Haar bands.

One stem and one ResNet body, not independent classifiers. The full 512px
image is transformed before any lossy spatial reduction. This is a research
candidate, not a reproduced paper or a claimed new wavelet transform.
"""
import torch
from torch import nn
from torchvision.models import resnet18


def haar_bands(x):
    if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
        raise ValueError('Haar input must be NCHW with even spatial dimensions')
    # Float32 before BF16 learned convolutions; retain signed weak differences.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x if x.dtype == torch.float64 else x.float()
        a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return torch.cat(((a+b+c+d)/2, (a-b+c-d)/2,
                          (a+b-c-d)/2, (a-b-c+d)/2), dim=1)


class BandStemResNet(nn.Module):
    def __init__(self, transform='haar'):
        super().__init__()
        if transform not in ('haar', 'phases'):
            raise ValueError('Expected haar or phases')
        self.transform = transform
        # Four groups correspond to LL/LH/HL/HH, each including all RGB channels.
        # Nonlinearity BEFORE fusion gives a distinct inductive bias; Haar plus
        # an unrestricted linear projection alone would just change the basis.
        self.band_stem = nn.Sequential(
            nn.Conv2d(12, 128, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=4, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.local = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2)
        base = resnet18(weights=None, num_classes=3)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.fc = nn.Sequential(nn.Dropout(.5), nn.Linear(512, 3))
        for module in (self.band_stem, self.local):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.fc[1].weight, std=.02)
        nn.init.zeros_(self.fc[1].bias)

    def forward(self, x):
        if self.transform == 'haar':
            x = haar_bands(x)
        else:
            if x.ndim != 4 or x.shape[-1] % 2 or x.shape[-2] % 2:
                raise ValueError('Phase input must be NCHW with even spatial dimensions')
            # Matched information-preserving control: same shape/parameters,
            # groups receive spatial phases rather than frequency bands.
            x = torch.cat([x[..., i::2, j::2].float() for i in range(2) for j in range(2)], dim=1)
        x = self.band_stem(x)
        x = self.pool(self.relu(x + self.local(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(x.float().mean((-2, -1)))
