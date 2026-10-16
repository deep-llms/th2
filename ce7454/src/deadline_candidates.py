"""Small, scratch-trained architecture candidates; no pretrained assets.

SRResNet adapts the learned, unpooled front-end principle of SRNet, rather
than reproducing SRNet. MultiDepthStats retains first/second-order feature
statistics from several depths of one backbone, with one classifier.
"""
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


def spatial_stats(x):
    # FP32 accumulation under AMP, population variance, finite flat-input gradient.
    var, mean = torch.var_mean(x.float(), dim=(-2, -1), correction=0)
    return torch.cat((mean, (var + 1e-6).sqrt()), dim=1)


def conv_bn(i, o, activation=True):
    layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualFrontBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(conv_bn(channels, channels),
                                    conv_bn(channels, channels, activation=False))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.branch(x))


class IntegratingDownsample(nn.Module):
    def __init__(self, i=32, o=64):
        super().__init__()
        # Both paths integrate on the same grid before stride-2 output.
        self.branch = nn.Sequential(conv_bn(i, o), conv_bn(o, o, activation=False),
                                    nn.AvgPool2d(3, 2, 1, count_include_pad=False))
        self.shortcut = nn.Sequential(nn.AvgPool2d(3, 2, 1, count_include_pad=False),
                                      nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


class SRResNet(nn.Module):
    def __init__(self, stats=False, num_classes=3, final_width=512):
        super().__init__()
        self.stats = stats
        self.front = nn.Sequential(conv_bn(3, 64), conv_bn(64, 32),
                                   *[ResidualFrontBlock(32) for _ in range(4)],
                                   IntegratingDownsample())
        body = resnet18(weights=None, num_classes=num_classes)
        self.layer1, self.layer2 = body.layer1, body.layer2
        self.layer3, self.layer4 = body.layer3, body.layer4
        if final_width != 512:
            downsample = nn.Sequential(nn.Conv2d(256, final_width, 1, stride=2, bias=False),
                                       nn.BatchNorm2d(final_width))
            self.layer4 = nn.Sequential(BasicBlock(256, final_width, stride=2, downsample=downsample),
                                        BasicBlock(final_width, final_width))
            for m in self.layer4.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.front.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(final_width * (2 if stats else 1), num_classes))
        nn.init.normal_(self.fc[1].weight, std=0.02)
        nn.init.zeros_(self.fc[1].bias)

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.layer1(self.front(x)))))
        return self.fc(spatial_stats(x) if self.stats else x.float().mean((-2, -1)))


class MultiDepthStats(nn.Module):
    def __init__(self, backbone, num_classes=3):
        super().__init__()
        self.backbone = backbone
        # Original stage-4 mean, then stage-1/2/3 mean and std; no patch labels,
        # fixed texture weighting, multiple predictions, or input views at inference.
        old_head = backbone.fc[1]
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512 + 2 * (64 + 128 + 256), num_classes))
        # Function-preserving extension in eval mode when wrapping a trained backbone.
        with torch.no_grad():
            self.fc[1].weight.zero_()
            self.fc[1].weight[:, :512].copy_(old_head.weight)
            self.fc[1].bias.copy_(old_head.bias)

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        descriptors = []
        for layer in (b.layer1, b.layer2, b.layer3):
            x = layer(x)
            descriptors.append(spatial_stats(x))
        deep = b.layer4(x).float().mean((-2, -1))
        return self.fc(torch.cat([deep, *descriptors], dim=1))
